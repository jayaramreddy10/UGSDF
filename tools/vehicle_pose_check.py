"""
How good are the vehicle poses of a trained SDFGS run, and what kind of error do they have?

Per vehicle and frame (training and held-out frames):
  - the mask-verified LiDAR returns, placed with the annotated pose and with the refined pose, and their
    mean |SDF| (how well each pose puts that frame's returns on the shared shape)
  - the rigid correction that registers the frame's returns to the SDF (registration against the
    learned shape), its size and the residual after it
  - speed and heading from the annotated trajectory, and the refined-minus-annotated offset and the
    registration correction split into along-track / cross-track / vertical (world frame)
Then, over all frames: the regression of the along-track offsets on speed. A slope (seconds) with a
good fit means a timing offset between the boxes (LiDAR time) and what the poses are fitted to
(refined: camera images; registration: LiDAR); scatter without a slope means per-frame noise.

usage: python tools/vehicle_pose_check.py <run_dir> [--min_returns 30]
writes <run_dir>/metrics_eval/vehicle_pose_check.json
"""
import os, sys, json, argparse, logging
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.gaussians.basics import quat_to_rotmat
from models.nodes.sdf_supervision import (AnnotatorMasks, gather_instance_lidar, recover_instance_ds_ids,
                                          register_to_sdf, rotvec_to_rotmat)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vehicle_pose_check")
FPS = 10.0


def fit_slope(x, y):
    """Least-squares slope through the origin and R^2 of y ~ k x."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 5 or (x * x).sum() == 0:
        return None, None
    k = float((x * y).sum() / (x * x).sum())
    ss_res = float(((y - k * x) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return k, (1 - ss_res / ss_tot) if ss_tot > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--min_returns", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    args, opts = ap.parse_known_args()
    run = args.run_dir
    cfg = OmegaConf.load(os.path.join(run, "config.yaml")); cfg.data.preload_device = "cpu"
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))
    dev = torch.device("cuda")
    ds = DrivingDataset(data_cfg=cfg.data)
    tr = import_str(cfg.trainer.type)(**cfg.trainer, num_timesteps=ds.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(ds.train_image_set), num_full_images=len(ds.full_image_set),
        test_set_indices=ds.test_timesteps, scene_aabb=ds.get_aabb().reshape(2, 3), device=dev)
    tr.resume_from_checkpoint(ckpt_path=os.path.join(run, args.ckpt), load_only_model=True); tr.set_eval()
    rig = tr.models["RigidNodes"]
    for p in rig.parameters():
        p.requires_grad_(False)
    ps = ds.pixel_source
    ds_ids = recover_instance_ds_ids(rig, ps)
    test = set(int(t) for t in ds.test_timesteps)
    frames = list(range(rig.num_frames))
    mroot = ds.annotator_masks_root(cfg.model.RigidNodes.init.get("annotator_masks"))
    masks = AnnotatorMasks(mroot, start_timestep=next(iter(ps.camera_data.values())).start_timestep) if mroot else None
    true_ids = [int(ps.instances_true_id[j]) for j in ds_ids]
    sup = cfg.model.RigidNodes.sdf.get("supervision", {})
    # annotated and refined poses (object -> world)
    q_ref = rig.quat_act(rig.instances_quats.data); t_ref = rig.instances_trans.data
    ann = torch.stack([ps.instances_pose[:, j] for j in ds_ids], dim=1).to(dev)          # (F, N, 4, 4)
    R_ann, t_ann = ann[..., :3, :3], ann[..., :3, 3]
    from models.gaussians.basics import matrix_to_quaternion
    q_ann = matrix_to_quaternion(R_ann)
    kw = dict(box_pad=sup.get("box_pad", 0.3), ground_margin=sup.get("ground_margin", 0.12),
              surface_tol=sup.get("surface_tol", 0.02), masks=masks, true_ids=true_ids, pixel_source=ps)
    per_ref = gather_instance_lidar(ds.lidar_source, frames, q_ref, t_ref, rig.instances_fv, rig.instances_size, dev, **kw)
    per_ann = gather_instance_lidar(ds.lidar_source, frames, q_ann, t_ann, rig.instances_fv, rig.instances_size, dev, **kw)

    rows = []
    for i in range(rig.num_instances):
        s = per_ref[i]
        if not s["surf_p"]:
            continue
        p = torch.cat(s["surf_p"]); fr = torch.cat(s["surf_frame"]); w = torch.cat(s["surf_w"])
        keep = w >= 1.0
        p, fr = p[keep], fr[keep]
        uf, pair = torch.unique(fr, return_inverse=True)
        cnt = torch.bincount(pair)
        ok = cnt[pair] >= args.min_returns
        p, fr = p[ok], fr[ok]
        if p.shape[0] == 0:
            continue
        uf, pair = torch.unique(fr, return_inverse=True)
        ids = torch.full((p.shape[0],), i, dtype=torch.long, device=dev)
        wv, uv, before, after = register_to_sdf(rig.query_sdf, p, ids, pair, len(uf), iters=args.iters)
        # residual with the annotated pose
        sa = per_ann[i]
        pa = torch.cat(sa["surf_p"]); fa = torch.cat(sa["surf_frame"]); wa = torch.cat(sa["surf_w"])
        pa, fa = pa[wa >= 1.0], fa[wa >= 1.0]
        with torch.no_grad():
            ra = rig.query_sdf(pa, torch.full((pa.shape[0],), i, dtype=torch.long, device=dev)).abs().clamp(max=0.1)
        for k, f in enumerate(uf.tolist()):
            m = fa == f
            R = quat_to_rotmat(q_ref[f, i][None])[0]
            # registration correction in world: new pose R' = R R(w)^T, t' = t - R' u
            Rn = R @ rotvec_to_rotmat(wv[k]).T
            dt_reg = -(Rn @ uv[k])
            # speed and heading from the annotated trajectory (central difference)
            f0, f1 = max(f - 1, 0), min(f + 1, rig.num_frames - 1)
            v = (t_ann[f1, i] - t_ann[f0, i]) * FPS / max(f1 - f0, 1)
            speed = float(v[:2].norm())
            along = v / v.norm().clamp_min(1e-6) if speed > 0.3 else R_ann[f, i][:, 0]
            along = torch.nn.functional.normalize(torch.stack([along[0], along[1], torch.zeros_like(along[0])]), dim=0)
            up = torch.tensor([0.0, 0.0, 1.0], device=dev); cross = torch.linalg.cross(up, along)
            d_ref = t_ref[f, i] - t_ann[f, i]
            yaw = lambda Rm: float(torch.atan2(Rm[1, 0], Rm[0, 0]))
            dyaw = (yaw(R) - yaw(R_ann[f, i]) + np.pi) % (2 * np.pi) - np.pi
            rows.append(dict(instance=i, frame=f, test=f in test, returns=int((pair == k).sum()), speed=speed,
                             sdf_ann=float(ra[m].mean()) if m.any() else None, sdf_ref=float(before[k]), sdf_reg=float(after[k]),
                             ref_along=float(d_ref @ along), ref_cross=float(d_ref @ cross), ref_up=float(d_ref[2]), ref_dyaw_deg=float(np.degrees(dyaw)),
                             reg_along=float(dt_reg @ along), reg_cross=float(dt_reg @ cross), reg_up=float(dt_reg[2]),
                             reg_rot_deg=float(np.degrees(float(wv[k].norm()))), reg_trans=float(uv[k].norm())))
        log.info(f"instance {i}: {len(uf)} frames with >= {args.min_returns} returns")

    def stats(sel, key):
        x = np.array([r[key] for r in sel if r[key] is not None])
        return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90))) if len(x) else None
    out = dict(run=run, frames=len(rows))
    for name, sel in (("train", [r for r in rows if not r["test"]]), ("test", [r for r in rows if r["test"]])):
        if not sel:
            continue
        out[name] = dict(frames=len(sel),
                         sdf_abs_annotated=stats(sel, "sdf_ann"), sdf_abs_refined=stats(sel, "sdf_ref"), sdf_abs_registered=stats(sel, "sdf_reg"),
                         reg_trans_m=stats(sel, "reg_trans"), reg_rot_deg=stats(sel, "reg_rot_deg"),
                         abs_ref_along=stats([dict(v=abs(r["ref_along"])) for r in sel], "v"),
                         abs_ref_dyaw_deg=stats([dict(v=abs(r["ref_dyaw_deg"])) for r in sel], "v"))
        moving = [r for r in sel if r["speed"] > 0.5]
        for key in ("ref_along", "reg_along"):
            k, r2 = fit_slope([r["speed"] for r in moving], [r[key] for r in moving])
            out[name][f"{key}_vs_speed"] = dict(slope_s=k, r2=r2, frames=len(moving))
    per_ins = {}
    for r in rows:
        per_ins.setdefault(r["instance"], []).append(r)
    out["per_instance"] = []
    for i, sel in per_ins.items():
        moving = [r for r in sel if r["speed"] > 0.5]
        k, r2 = fit_slope([r["speed"] for r in moving], [r["ref_along"] for r in moving])
        out["per_instance"].append(dict(instance=i, frames=len(sel), speed_median=float(np.median([r["speed"] for r in sel])),
                                        sdf_ann=float(np.mean([r["sdf_ann"] for r in sel if r["sdf_ann"] is not None])),
                                        sdf_ref=float(np.mean([r["sdf_ref"] for r in sel])), sdf_reg=float(np.mean([r["sdf_reg"] for r in sel])),
                                        reg_trans_median=float(np.median([r["reg_trans"] for r in sel])),
                                        reg_rot_median_deg=float(np.median([r["reg_rot_deg"] for r in sel])),
                                        ref_along_vs_speed=dict(slope_s=k, r2=r2)))
    out["rows"] = rows
    os.makedirs(os.path.join(run, "metrics_eval"), exist_ok=True)
    json.dump(out, open(os.path.join(run, "metrics_eval", "vehicle_pose_check.json"), "w"), indent=1)
    log.info("RESULT " + json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
