"""
Export each modelled person as a subsampled set of SMPL mesh points and their trajectories, instead of
SMPL pose parameters and the full mesh.

  points:      `--n_points` vertices of the SMPL mesh, chosen by farthest point sampling on the canonical
               (rest-pose) body; the same vertex ids for every person (shared topology)
  trajectory:  world position of every point at every frame, posed with the run's SMPL (`--source trained`:
               the refined poses of the checkpoint, held-out frames by `--test_interp`; `--source init`: the
               per-frame pose estimates the run started from)
  per camera:  pixel position of every point, whether it falls in the image, and whether it is visible
               (in front of the rendered depth of the whole scene, within `--vis_tol` metres)

usage: python tools/export_point_tracks.py <run_dir> [--n_points 1200] [--source trained|init] [--test_interp cubic]
writes <run_dir>/exports/point_tracks_<source>.npz and a preview PNG next to it

npz contents (P points, H persons, F frames, C cameras):
  vertex_ids (P,)                 SMPL vertex index of each point
  canonical (H, P, 3)             rest-pose positions (person frame, metres)
  canonical_full (H, 6890, 3)     rest-pose positions of the whole mesh
  tracks_world (F, H, P, 3)       world positions (NaN where the person is not present)
  present (F, H) bool             the person is active at the frame
  frame_is_test (F,) bool         held-out frame
  person_dataset_id, person_true_id (H,)
  cam_names (C,), uv (F, C, H, P, 2), in_image (F, C, H, P), visible (F, C, H, P), depth (F, C, H, P)
  intrinsics (F, C, 3, 3), cam_to_world (F, C, 4, 4), image_hw (C, 2)
  faces (13776, 3)                full SMPL faces (for the full mesh, if wanted)
"""
import os, sys, json, argparse, logging
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.nodes.sdf_supervision import project_points

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("point_tracks")


def farthest_point_sampling(x: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.zeros(n, dtype=torch.long, device=x.device)
    d = torch.full((x.shape[0],), float("inf"), device=x.device)
    idx[0] = torch.argmax((x - x.mean(0)).norm(dim=-1))
    for k in range(1, n):
        d = torch.minimum(d, (x - x[idx[k - 1]]).norm(dim=-1))
        idx[k] = torch.argmax(d)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--n_points", type=int, default=1200)
    ap.add_argument("--source", choices=["trained", "init"], default="trained")
    ap.add_argument("--test_interp", default="cubic", help="held-out frame poses for --source trained")
    ap.add_argument("--vis_tol", type=float, default=0.05)
    ap.add_argument("--no_visibility", action="store_true", help="skip rendering (visible = in_image)")
    ap.add_argument("--suffix", default="", help="appended to the output name, e.g. _3445")
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
    smpl_dict = ds.get_init_smpl_objects(**OmegaConf.to_container(cfg.model.SMPLNodes.init))
    person_ids = [int(k) for k in smpl_dict.keys()]
    if args.source == "init":
        smpl.create_from_pcd(smpl_dict)
        init = [p.detach().clone() for p in (smpl.instances_quats, smpl.smpl_qauts, smpl.instances_trans)]
    tr.resume_from_checkpoint(ckpt_path=os.path.join(run, args.ckpt), load_only_model=True)
    tr.set_eval()
    assert smpl.num_instances == len(person_ids), "dataset persons differ from the checkpoint's"
    OmegaConf.set_struct(smpl.ctrl_cfg, False)
    if args.source == "init":
        with torch.no_grad():
            smpl.instances_quats.data, smpl.smpl_qauts.data, smpl.instances_trans.data = init
        smpl.ctrl_cfg.test_interp = "own"       # the estimate of every frame, held-out ones included
    else:
        smpl.ctrl_cfg.test_interp = args.test_interp

    ps = ds.pixel_source
    test = set(int(t) for t in ds.test_timesteps)
    F_, H_ = smpl.num_frames, smpl.num_instances
    with torch.no_grad():
        v, faces = smpl.template.get_init_vf()                      # (H, 6890, 3) rest pose, with each person's shape
        vids = farthest_point_sampling(v.mean(0), args.n_points)    # shared vertex ids
        canonical = v[:, vids]
        tracks = torch.full((F_, H_, args.n_points, 3), float("nan"), device=dev)
        present = smpl.instances_fv.clone()
        for f in range(F_):
            if not present[f].any():
                continue
            smpl.cur_frame = torch.tensor(f); smpl.in_test_set = f in test
            world = smpl.transform_means(v.reshape(-1, 3)).reshape(H_, -1, 3)
            tracks[f, present[f]] = world[present[f]][:, vids]
    log.info(f"{H_} persons, {args.n_points} points each, {F_} frames ({len(test)} held out)")

    cams = sorted(ps.camera_data.values(), key=lambda c: c.unique_cam_idx)
    C_ = len(cams)
    uv = np.full((F_, C_, H_, args.n_points, 2), np.nan, np.float32)
    in_img = np.zeros((F_, C_, H_, args.n_points), bool); vis = np.zeros_like(in_img)
    depth = np.full((F_, C_, H_, args.n_points), np.nan, np.float32)
    K_all = np.zeros((F_, C_, 3, 3), np.float32); c2w_all = np.zeros((F_, C_, 4, 4), np.float32)
    for f in range(F_):
        for c, cam in enumerate(cams):
            K_all[f, c] = cam.intrinsics[f].cpu().numpy(); c2w_all[f, c] = cam.cam_to_worlds[f].cpu().numpy()
        if not present[f].any():
            continue
        p = tracks[f][present[f]].reshape(-1, 3)
        for c, cam in enumerate(cams):
            uv_c, z_c, inside = project_points(p, cam, f, dev)
            hm = present[f].nonzero()[:, 0].cpu().numpy()
            uv[f, c, hm] = uv_c.reshape(len(hm), -1, 2).cpu().numpy()
            depth[f, c, hm] = z_c.reshape(len(hm), -1).cpu().numpy()
            in_img[f, c, hm] = inside.reshape(len(hm), -1).cpu().numpy()
            if args.no_visibility:
                vis[f, c, hm] = in_img[f, c, hm]
                continue
            if not inside.any():
                continue
            ii, ci = ps.get_image(f * ps.num_cams + cam.unique_cam_idx)
            ii = {k: (x.to(dev) if isinstance(x, torch.Tensor) else x) for k, x in ii.items()}
            ci = {k: (x.to(dev) if isinstance(x, torch.Tensor) else x) for k, x in ci.items()}
            with torch.no_grad():
                out = tr(ii, ci)
            Hh, Ww = ii["pixels"].shape[:2]
            dmap = out["depth"].reshape(Hh, Ww)
            q = uv_c[inside].long()
            ok = torch.zeros_like(inside)
            ok[inside] = z_c[inside] <= dmap[q[:, 1].clamp(0, Hh - 1), q[:, 0].clamp(0, Ww - 1)] + args.vis_tol
            vis[f, c, hm] = ok.reshape(len(hm), -1).cpu().numpy()
        if f % 20 == 0:
            log.info(f"frame {f}/{F_}")

    out_dir = os.path.join(run, "exports"); os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"point_tracks_{args.source}{args.suffix}.npz")
    np.savez_compressed(fn, vertex_ids=vids.cpu().numpy(), canonical=canonical.cpu().numpy(), canonical_full=v.cpu().numpy(),
                        tracks_world=tracks.cpu().numpy(), present=present.cpu().numpy(),
                        frame_is_test=np.array([f in test for f in range(F_)]),
                        person_dataset_id=np.array(person_ids), person_true_id=np.array([int(ps.instances_true_id[k]) for k in person_ids]),
                        cam_names=np.array([c.cam_name for c in cams]), uv=uv, in_image=in_img, visible=vis, depth=depth,
                        intrinsics=K_all, cam_to_world=c2w_all, image_hw=np.array([[c.HEIGHT, c.WIDTH] for c in cams]),
                        faces=faces.cpu().numpy(), source=args.source, test_interp=smpl.ctrl_cfg.test_interp)
    log.info(f"wrote {fn}")

    # preview: the camera and frame with the most visible points; tracks over the previous 10 frames
    counts = vis.sum(axis=(2, 3))
    f0, c0 = np.unravel_index(np.argmax(counts), counts.shape)
    img = (ps.get_image(f0 * ps.num_cams + cams[c0].unique_cam_idx)[0]["pixels"].cpu().numpy() * 255).astype(np.uint8).copy()
    rng = np.random.default_rng(0); colors = rng.integers(60, 255, (H_, 3))
    for h in range(H_):
        for pi in range(args.n_points):
            pts = [uv[f, c0, h, pi] for f in range(max(f0 - 10, 0), f0 + 1) if vis[f, c0, h, pi]]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, tuple(int(x) for x in a), tuple(int(x) for x in b), tuple(int(x) for x in colors[h]), 1, cv2.LINE_AA)
            if vis[f0, c0, h, pi]:
                cv2.circle(img, tuple(int(x) for x in uv[f0, c0, h, pi]), 2, tuple(int(x) for x in colors[h]), -1)
    cv2.imwrite(os.path.join(out_dir, f"point_tracks_{args.source}{args.suffix}_preview_f{f0:03d}_cam{c0}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    summary = dict(persons=H_, points=args.n_points, frames=F_, visible_point_frames=int(vis.sum()),
                   in_image_point_frames=int(in_img.sum()), preview=dict(frame=int(f0), cam=int(c0)))
    json.dump(summary, open(os.path.join(out_dir, f"point_tracks_{args.source}{args.suffix}.json"), "w"), indent=1)
    log.info("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
