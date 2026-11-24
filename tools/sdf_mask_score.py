"""
Per-instance score of the SDFs of a trained SDFGS run: does the shape fit the masks, the LiDAR, and is it smooth?

  silhouette  camera rays through a crop around the projected box, marched through the instance SDF with the
              pose of that frame; compared with the annotator mask. Pixels of other objects are ignored.
              iou; spill = silhouette farther than 2 px outside the mask; miss = mask pixels the SDF does not reach.
              Reported for held-out frames (poses by ctrl.test_interp) and every `--train_stride`-th training frame.
  lidar       mask-verified returns of the held-out frames (never used by the SDF): mean |sdf|, share within 5 cm
  roughness   surface points; angle between the normal at a point and the normal 3 cm along the surface

usage: python tools/sdf_mask_score.py <run_dir> [--test_interp cubic]
writes <run_dir>/metrics_eval/sdf_mask_score.json
"""
import os, sys, json, argparse, logging
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.nodes.sdf_supervision import (AnnotatorMasks, gather_instance_lidar, recover_instance_ds_ids, ray_box_intersect,
                                          world_to_object, project_points, pixel_rays, dilate)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sdf_score")


@torch.no_grad()
def min_sdf(rig, i, o, d, a, b, samples=64, chunk=8192):
    out = torch.zeros(o.shape[0], device=o.device)
    u = torch.linspace(0, 1, samples, device=o.device)[None]
    for k in range(0, o.shape[0], chunk):
        tt = a[k:k + chunk, None] + u * (b - a)[k:k + chunk, None]
        p = o[k:k + chunk, None] + d[k:k + chunk, None] * tt[..., None]
        ids = torch.full((p.shape[0] * samples,), i, dtype=torch.long, device=o.device)
        out[k:k + chunk] = rig.query_sdf(p.reshape(-1, 3), ids).reshape(-1, samples).min(1).values
    return out


def roughness(rig, i, half, dev, n=20000, step=0.03):
    def project(x, ids, iters):
        for _ in range(iters):
            s, g = rig.query_sdf_normal(x, ids)
            x = x - s[:, None].clamp(-0.5, 0.5) * g
        return x
    x = (torch.rand(n, 3, device=dev) * 2 - 1) * (half + 0.05)
    ids = torch.full((n,), i, dtype=torch.long, device=dev)
    x = project(x, ids, 6)
    s, g = rig.query_sdf_normal(x, ids)
    ok = (s.abs() < 3e-3) & (x.abs() <= half + 0.06).all(-1)
    if int(ok.sum()) < 100:
        return None
    x, g, ids = x[ok], g[ok], ids[ok]
    t = torch.nn.functional.normalize(torch.linalg.cross(g, torch.randn_like(g), dim=-1), dim=-1)
    y = project(x + step * t, ids, 3)
    _, h = rig.query_sdf_normal(y, ids)
    ang = torch.rad2deg(torch.arccos((g * h).sum(-1).clamp(-1, 1)))
    return float(ang.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--test_interp", default="cubic")
    ap.add_argument("--train_stride", type=int, default=5)
    ap.add_argument("--pixel_stride", type=int, default=2)
    ap.add_argument("--min_area", type=int, default=400, help="mask pixels a view needs")
    ap.add_argument("--min_returns", type=int, default=30)
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
    OmegaConf.set_struct(rig.ctrl_cfg, False); rig.ctrl_cfg.test_interp = args.test_interp
    ps = ds.pixel_source
    ds_ids = recover_instance_ds_ids(rig, ps)
    true_ids = [int(ps.instances_true_id[j]) for j in ds_ids]
    test = sorted(int(t) for t in ds.test_timesteps)
    train = [f for f in range(rig.num_frames) if f not in set(test)][:: args.train_stride]
    start = next(iter(ps.camera_data.values())).start_timestep
    masks = AnnotatorMasks(ds.annotator_masks_root(cfg.model.RigidNodes.init.get("annotator_masks")), start_timestep=start)
    N = rig.num_instances
    # the pose used for every frame, held-out ones by interpolation
    q_all, t_all = [], []
    for f in range(rig.num_frames):
        rig.cur_frame = torch.tensor(f); rig.in_test_set = f in set(test)
        q, t = rig.current_instance_pose()
        q_all.append(rig.quat_act(q.detach())); t_all.append(t.detach())
    q_all, t_all = torch.stack(q_all), torch.stack(t_all)
    corners = torch.tensor([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], device=dev, dtype=torch.float32)

    def silhouettes(frames):
        c = np.zeros((N, 6), dtype=np.int64)      # tp, union, spill, sil, miss, mask   (and views separately)
        views = np.zeros(N, dtype=np.int64)
        for fi in frames:
            for i in torch.where(rig.instances_fv[fi])[0].tolist():
                tid = true_ids[i]
                for cid in masks.cameras(tid, fi):
                    if cid not in ps.camera_data:
                        continue
                    cam = ps.camera_data[cid]
                    m = masks.get(tid, cid, fi, cam, dev)
                    if m is None or int(m.sum()) < args.min_area:
                        continue
                    R, t = world_to_object(q_all[fi, i], t_all[fi, i])
                    half = rig.instances_size[i] / 2
                    uv_c, z_c, _ = project_points((corners * half * 1.2) @ R.T + t, cam, fi, dev)
                    if (z_c <= 0.1).any():
                        continue
                    x0, y0 = [int(v) for v in uv_c.min(0).values.clamp_min(0)]
                    x1, y1 = int(uv_c[:, 0].max().clamp_max(cam.WIDTH - 1)), int(uv_c[:, 1].max().clamp_max(cam.HEIGHT - 1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    ys, xs = torch.meshgrid(torch.arange(y0, y1 + 1, args.pixel_stride, device=dev), torch.arange(x0, x1 + 1, args.pixel_stride, device=dev), indexing="ij")
                    v, u = ys.reshape(-1), xs.reshape(-1)
                    near = dilate(m, 3)
                    other = torch.zeros_like(m)
                    for name in ("dynamic_masks", "vehicle_masks", "human_masks"):
                        mm = getattr(cam, name, None)
                        if mm is not None:
                            other = other | (mm[fi].to(dev) > 0.5)
                    keep = ~(other & ~near)[v, u]
                    u, v = u[keep], v[keep]
                    o_w, d_w, _ = pixel_rays(torch.stack([u, v], -1).float(), cam, fi, dev)
                    o = ((o_w - t) @ R).expand(d_w.shape[0], 3); d = d_w @ R
                    a, b = ray_box_intersect(o, d, half * 1.2)
                    a = a.clamp_min(0.0)
                    sil = torch.zeros_like(u, dtype=torch.bool)
                    hit = b > a
                    if hit.any():
                        sil[hit] = min_sdf(rig, i, o[hit], d[hit], a[hit], b[hit]) <= 0
                    mm_, wide = m[v, u], dilate(m, 2)[v, u]
                    c[i] += [int((sil & mm_).sum()), int((sil | mm_).sum()), int((sil & ~wide).sum()), int(sil.sum()), int((mm_ & ~sil).sum()), int(mm_.sum())]
                    views[i] += 1
        return c, views

    res = dict(run=run, test_interp=args.test_interp, per_instance={})
    sc = {}
    for name, frames in (("test", test), ("train", train)):
        sc[name] = silhouettes(frames)
        log.info(f"{name}: {int(sc[name][1].sum())} views")
    kw = dict(box_pad=0.3, ground_margin=0.12, surface_tol=0.02, masks=masks, true_ids=true_ids, pixel_source=ps)
    per = gather_instance_lidar(ds.lidar_source, test, q_all, t_all, rig.instances_fv, rig.instances_size, dev, **kw)
    for i in range(N):
        row = dict(true_id=true_ids[i], size=[round(float(x), 2) for x in rig.instances_size[i]])
        for name, (c, views) in sc.items():
            if views[i] > 0:
                row[name] = dict(views=int(views[i]), iou=c[i, 0] / max(c[i, 1], 1), spill=c[i, 2] / max(c[i, 3], 1), miss=c[i, 4] / max(c[i, 5], 1))
        s = per[i]
        if s["surf_p"]:
            p, w = torch.cat(s["surf_p"]), torch.cat(s["surf_w"])
            p = p[w >= 1.0]
            if p.shape[0] >= args.min_returns:
                with torch.no_grad():
                    f = rig.query_sdf(p, torch.full((p.shape[0],), i, dtype=torch.long, device=dev)).abs().clamp(max=0.1)
                row["lidar"] = dict(returns=int(p.shape[0]), sdf_abs=float(f.mean()), within_5cm=float((f < 0.05).float().mean()))
        row["roughness_deg"] = roughness(rig, i, rig.instances_size[i] / 2, dev)
        res["per_instance"][i] = row
    def agg(get):
        x = [get(r) for r in res["per_instance"].values()]
        x = [v for v in x if v is not None]
        return dict(mean=float(np.mean(x)), median=float(np.median(x)), n=len(x)) if x else None
    res["summary"] = {f"{n}_{k}": agg(lambda r: r[n][k] if n in r else None) for n in ("test", "train") for k in ("iou", "spill", "miss")}
    res["summary"]["lidar_sdf_abs"] = agg(lambda r: r["lidar"]["sdf_abs"] if "lidar" in r else None)
    res["summary"]["lidar_within_5cm"] = agg(lambda r: r["lidar"]["within_5cm"] if "lidar" in r else None)
    res["summary"]["roughness_deg"] = agg(lambda r: r["roughness_deg"])
    os.makedirs(os.path.join(run, "metrics_eval"), exist_ok=True)
    json.dump(res, open(os.path.join(run, "metrics_eval", "sdf_mask_score.json"), "w"), indent=1)
    log.info("RESULT " + json.dumps(res["summary"], indent=1))
    for i in (1, 3, 4):
        if i in res["per_instance"]:
            log.info(f"instance {i}: " + json.dumps(res["per_instance"][i]))


if __name__ == "__main__":
    main()
