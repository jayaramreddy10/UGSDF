"""
Human (and vehicle) poses on held-out frames: compare the SMPLNodes held-out pose modes (ctrl.test_interp) on a trained
run, then fit the poses of the held-out frames to the held-out images with everything else frozen
("test-time pose fitting"; uses the held-out images, so report it as such).

  modes: linear (original), cubic, own (the frame's initial estimate), init_residual
  fit:   start from the best mode, optimise each held-out frame's SMPL root rotation, joint rotations and
         translation for --fit_iters steps against that frame's images (L1 + D-SSIM), then render

usage: python tools/eval_human_poses.py <run_dir> [--fit_iters 100] [--modes linear cubic own init_residual]
writes <run_dir>/metrics_eval/human_poses.json
"""
import os, sys, json, argparse, logging, time
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("human_poses")


def to_dev(d, dev):
    return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--modes", nargs="+", default=["linear", "cubic", "own", "init_residual"])
    ap.add_argument("--fit_iters", type=int, default=100)
    ap.add_argument("--fit_from", default=None, help="mode to start the fit from (default: best mode)")
    ap.add_argument("--fit_lr_rot", type=float, default=2e-3)
    ap.add_argument("--fit_lr_trans", type=float, default=2e-3)
    ap.add_argument("--rigid_modes", nargs="*", default=["linear", "cubic"], help="held-out vehicle pose modes to compare")
    ap.add_argument("--fit_rigid", action="store_true", help="also fit the held-out vehicle poses")
    args, opts = ap.parse_known_args()
    run = args.run_dir
    cfg = OmegaConf.load(os.path.join(run, "config.yaml")); cfg.data.preload_device = "cpu"
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))
    dev = torch.device("cuda")
    ds = DrivingDataset(data_cfg=cfg.data)
    tr = import_str(cfg.trainer.type)(**cfg.trainer, num_timesteps=ds.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(ds.train_image_set), num_full_images=len(ds.full_image_set),
        test_set_indices=ds.test_timesteps, scene_aabb=ds.get_aabb().reshape(2, 3), device=dev)
    smpl = tr.models["SMPLNodes"]
    tracked = type(smpl).__name__ == "TrackNodes"       # no pose parameters: compare the interpolations only
    if tracked:
        args.fit_iters = 0
        args.modes = [m for m in args.modes if m in ("linear", "cubic", "own")]
        tr.resume_from_checkpoint(ckpt_path=os.path.join(run, args.ckpt), load_only_model=True)
        tr.set_eval()
    else:
        # initial (estimator) poses, for init_residual: build the node from the dataset, keep its init, then resume
        smpl.create_from_pcd(ds.get_init_smpl_objects(**OmegaConf.to_container(cfg.model.SMPLNodes.init)))
        init_pose = tuple(x.detach().clone() for x in smpl.init_pose)
        tr.resume_from_checkpoint(ckpt_path=os.path.join(run, args.ckpt), load_only_model=True)
        tr.set_eval()
        smpl.init_pose = init_pose
        assert init_pose[0].shape[1] == smpl.num_instances, "dataset humans differ from the checkpoint's"
    OmegaConf.set_struct(smpl.ctrl_cfg, False)
    ps = ds.pixel_source
    test_idx = list(ds.test_image_set.split_indices)
    images = [(i, *[to_dev(d, dev) for d in ps.get_image(i)]) for i in test_idx]
    test_frames = sorted(set(i // ps.num_cams for i in test_idx))

    def evaluate():
        acc = {k: [0, 0.0] for k in ("all", "human", "vehicle")}
        psnrs = []
        with torch.no_grad():
            for _, ii, ci in images:
                out = tr(ii, ci)
                gt = ii["pixels"]; H, W = gt.shape[:2]
                se = ((out["rgb"].reshape(H, W, 3).clamp(0, 1) - gt) ** 2).mean(-1)
                psnrs.append(-10 * np.log10(float(se.mean())))
                for k, m in (("all", torch.ones_like(se, dtype=torch.bool)), ("human", ii["human_masks"] > 0.5), ("vehicle", ii["vehicle_masks"] > 0.5)):
                    acc[k][0] += int(m.sum()); acc[k][1] += float(se[m].sum())
        r = {f"{k}_psnr": -10 * np.log10(s / max(n, 1)) for k, (n, s) in acc.items()}
        r["psnr_mean_per_image"] = float(np.mean(psnrs))
        return r

    res = {"modes": {}}
    for mode in args.modes:
        smpl.ctrl_cfg.test_interp = mode
        res["modes"][mode] = evaluate()
        log.info(f"{mode:14s} " + " ".join(f"{k}={v:.3f}" for k, v in res["modes"][mode].items()))

    rigid = tr.models.get("RigidNodes")
    best_h = max(res["modes"], key=lambda m: res["modes"][m]["human_psnr"])
    if rigid is not None and args.rigid_modes:
        OmegaConf.set_struct(rigid.ctrl_cfg, False)
        smpl.ctrl_cfg.test_interp = best_h
        res["rigid_modes"] = {}
        for mode in args.rigid_modes:
            rigid.ctrl_cfg.test_interp = mode
            res["rigid_modes"][mode] = evaluate()
            log.info(f"vehicles {mode:8s} (humans {best_h}) " + " ".join(f"{k}={v:.3f}" for k, v in res["rigid_modes"][mode].items()))
        best_r = max(res["rigid_modes"], key=lambda m: res["rigid_modes"][m]["vehicle_psnr"])
        rigid.ctrl_cfg.test_interp = best_r

    if args.fit_iters > 0:
        start = args.fit_from or best_h
        smpl.ctrl_cfg.test_interp = start
        # write the start poses into the held-out frames' own parameters, then render them as they are
        with torch.no_grad():
            smpl.in_test_set = True
            for f in test_frames:
                smpl.cur_frame = torch.tensor(f)
                theta, trans = smpl.current_pose()
                smpl.instances_quats.data[f] = theta[:, :1].clone(); smpl.smpl_qauts.data[f] = theta[:, 1:].clone()
                smpl.instances_trans.data[f] = trans.clone()
        smpl.ctrl_cfg.test_interp = "own"
        if args.fit_rigid and rigid is not None:
            with torch.no_grad():
                rigid.in_test_set = True
                for f in test_frames:
                    rigid.cur_frame = torch.tensor(f)
                    q, t = rigid.current_instance_pose()
                    rigid.instances_quats.data[f] = q.clone(); rigid.instances_trans.data[f] = t.clone()
            rigid.ctrl_cfg.test_interp = "own"
        res["fit_start"] = dict(mode=start, **evaluate())
        for m in tr.models.values():
            for p in m.parameters():
                p.requires_grad_(False)
        pose_params = [smpl.instances_quats, smpl.smpl_qauts, smpl.instances_trans]
        if args.fit_rigid and rigid is not None:
            pose_params += [rigid.instances_quats, rigid.instances_trans]
        for p in pose_params:
            p.requires_grad_(True)
        from pytorch_msssim import ssim
        by_frame = {}
        for i, ii, ci in images:
            by_frame.setdefault(i // ps.num_cams, []).append((ii, ci))
        t0 = time.time()
        for n, (f, views) in enumerate(sorted(by_frame.items())):
            groups = [dict(params=[smpl.instances_quats, smpl.smpl_qauts], lr=args.fit_lr_rot),
                      dict(params=[smpl.instances_trans], lr=args.fit_lr_trans)]
            if args.fit_rigid and rigid is not None:
                groups += [dict(params=[rigid.instances_quats], lr=args.fit_lr_rot * 0.1), dict(params=[rigid.instances_trans], lr=args.fit_lr_trans)]
            opt = torch.optim.Adam(groups)
            for it in range(args.fit_iters):
                opt.zero_grad()
                loss = 0.0
                for ii, ci in views:
                    out = tr(ii, ci)
                    gt = ii["pixels"]; H, W = gt.shape[:2]
                    pred = out["rgb"].reshape(H, W, 3)
                    l1 = (pred - gt).abs().mean()
                    ds_ = 1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1.0, size_average=True)
                    loss = loss + 0.8 * l1 + 0.2 * ds_
                loss.backward()
                for p in pose_params:   # only this frame's poses move
                    if p.grad is not None:
                        keep = p.grad[f].clone(); p.grad.zero_(); p.grad[f] = keep
                opt.step()
            log.info(f"fit frame {f} ({n + 1}/{len(by_frame)}): loss {float(loss) / len(views):.4f}, {time.time() - t0:.0f} s")
        for p in pose_params:
            p.requires_grad_(False)
        res["fit"] = dict(iters=args.fit_iters, start=start, fit_rigid=bool(args.fit_rigid), **evaluate())
        log.info("fit           " + " ".join(f"{k}={v:.3f}" for k, v in res["fit"].items() if isinstance(v, float)))
    os.makedirs(os.path.join(run, "metrics_eval"), exist_ok=True)
    json.dump(res, open(os.path.join(run, "metrics_eval", "human_poses.json"), "w"), indent=1)
    log.info("RESULT " + json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
