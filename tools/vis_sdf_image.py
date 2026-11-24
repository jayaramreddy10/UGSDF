"""
Visualise the instance SDFs of a trained SDFGS run in the camera images.

For each chosen (frame, camera) view of an instance, camera rays through a crop around the
projected box are marched through the instance SDF (object frame, pose of that frame).
One PNG per view, six panels:
  1  image, annotator mask outline (green), projected box (yellow)
  2  SDF surface: first zero crossing, shaded by its normal, over the image;
     red = mask pixels whose ray never reaches sdf <= 0 (the SDF misses the visible object)
  3  min SDF along each ray (blue inside, white at 0, red outside), mask outline on top
  4  SDF depth minus LiDAR depth at LiDAR returns inside the box crossing (blue: SDF in front of
     the return, red: behind); grey: returns in front of the box (occluders); yellow: returns in
     the box where the SDF has no surface
  5  rendered image (whole model)
  6  RigidNodes Gaussian depth minus SDF depth where both exist (blue: Gaussians in front)

usage:
  python tools/vis_sdf_image.py <run_dir> --instances 4 [--frames 10 40 ...] [--cams 0 1 2]
         [--num_views 8] [--out <dir>]
Frames default to the views where the instance's mask is largest, spread over the sequence.
Instance numbers are RigidNodes indices, as in the eval per-instance tables.
"""
import os, sys, argparse, logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.nodes.sdf_supervision import (AnnotatorMasks, ray_box_intersect, world_to_object, project_points,
                                          pixel_rays, recover_instance_ds_ids)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vis_sdf")


def diverging(x, lim):
    """(H, W) values -> (H, W, 3) uint8 RGB, blue negative, white zero, red positive."""
    a = np.clip(x / lim, -1, 1)
    r = np.where(a > 0, 1.0, 1.0 + a); b = np.where(a < 0, 1.0, 1.0 - a); g = 1.0 - np.abs(a)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def march(rigid, i, o, d, t0, t1, samples=256, refine=8, chunk=4096):
    """First zero crossing along each ray in [t0, t1] (object frame). Returns t_hit (nan if none), min sdf."""
    n = o.shape[0]
    t_hit = torch.full((n,), float("nan"), device=o.device); smin = torch.full((n,), float("inf"), device=o.device)
    ids_c = None
    for k in range(0, n, chunk):
        oo, dd, a, b = o[k:k + chunk], d[k:k + chunk], t0[k:k + chunk], t1[k:k + chunk]
        m = oo.shape[0]
        u = torch.linspace(0, 1, samples, device=o.device)[None]
        tt = a[:, None] + u * (b - a)[:, None]
        p = oo[:, None] + dd[:, None] * tt[..., None]
        ids = torch.full((m * samples,), i, dtype=torch.long, device=o.device)
        s = rigid.query_sdf(p.reshape(-1, 3), ids).reshape(m, samples)
        smin[k:k + m] = s.min(1).values
        neg = s <= 0
        has = neg.any(1)
        j = neg.float().argmax(1)                                  # first sample inside
        jp = (j - 1).clamp_min(0)
        lo = torch.gather(tt, 1, jp[:, None])[:, 0]; hi = torch.gather(tt, 1, j[:, None])[:, 0]
        first_inside = has & (j == 0)                              # ray starts inside the solid
        for _ in range(refine):
            mid = (lo + hi) / 2
            sm = rigid.query_sdf(oo + dd * mid[:, None], torch.full((m,), i, dtype=torch.long, device=o.device))
            inside = sm <= 0
            hi = torch.where(inside, mid, hi); lo = torch.where(inside, lo, mid)
        th = torch.where(first_inside, a, hi)
        t_hit[k:k + m] = torch.where(has, th, torch.full_like(th, float("nan")))
    return t_hit, smin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--instances", type=int, nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--cams", type=int, nargs="*", default=None, help="unique camera indices (default: all)")
    ap.add_argument("--num_views", type=int, default=8)
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--margin", type=float, default=0.25, help="crop margin around the projected box, as a fraction")
    ap.add_argument("--min_crop", type=int, default=160)
    ap.add_argument("--no_render", action="store_true", help="skip the Gaussian render panels")
    args, opts = ap.parse_known_args()   # leftover key=value pairs override the run config

    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    cfg.data.preload_device = "cpu"
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))
    device = torch.device("cuda")
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(**cfg.trainer, num_timesteps=dataset.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(dataset.train_image_set), num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps, scene_aabb=dataset.get_aabb().reshape(2, 3), device=device)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(args.run_dir, args.ckpt), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    ps = dataset.pixel_source
    out_dir = args.out or os.path.join(args.run_dir, "vis_sdf")
    os.makedirs(out_dir, exist_ok=True)

    ds_ids = recover_instance_ds_ids(rigid, ps)
    masks_root = dataset.annotator_masks_root(cfg.model.RigidNodes.get("init", {}).get("annotator_masks"))
    masks = AnnotatorMasks(masks_root, start_timestep=next(iter(ps.camera_data.values())).start_timestep) if masks_root else None
    quats = rigid.quat_act(rigid.instances_quats.data); trans = rigid.instances_trans.data
    test_frames = set(int(f) for f in (dataset.test_timesteps if dataset.test_timesteps is not None else []))
    cams = {c.unique_cam_idx: (cid, c) for cid, c in ps.camera_data.items()}
    cam_sel = args.cams if args.cams else sorted(cams)
    corners_unit = torch.tensor([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], device=device, dtype=torch.float32)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]

    for i in args.instances:
        tid = int(ps.instances_true_id[ds_ids[i]].item()) if ds_ids[i] >= 0 else None
        half = rigid.instances_size[i] / 2
        # candidate views: instance visible, mask present; score by mask area
        cands = []
        frames = args.frames if args.frames else [f for f in range(rigid.instances_fv.shape[0]) if bool(rigid.instances_fv[f, i])]
        for fi in frames:
            for u in cam_sel:
                cid, cam = cams[u]
                m = masks.get(tid, cid, fi, cam, device) if (masks is not None and tid is not None) else None
                area = int(m.sum()) if m is not None else 0
                if args.frames or area > 0:
                    cands.append((area, fi, u))
        if not args.frames:
            cands.sort(reverse=True)
            # spread over the sequence: greedily take large views at least 5 frames apart
            chosen = []
            for c in cands:
                if all(abs(c[1] - x[1]) >= 5 for x in chosen):
                    chosen.append(c)
                if len(chosen) >= args.num_views:
                    break
            cands = sorted(chosen, key=lambda c: c[1])
        logger.info(f"instance {i} (true id {tid}, size {[round(float(v), 2) for v in rigid.instances_size[i]]}): "
                    f"{len(cands)} views {[(f, u) for _, f, u in cands]}")
        sheet = []
        for area, fi, u in cands:
            cid, cam = cams[u]
            H, W = cam.HEIGHT, cam.WIDTH
            R, t = world_to_object(quats[fi, i], trans[fi, i])
            cw = (corners_unit * half) @ R.T + t
            uv_c, z_c, _ = project_points(cw, cam, fi, device)
            if (z_c <= 0.1).any():
                logger.info(f"  frame {fi} cam {u}: box behind the camera, skipped"); continue
            x0, y0 = uv_c.min(0).values.tolist(); x1, y1 = uv_c.max(0).values.tolist()
            mx, my = (x1 - x0) * args.margin, (y1 - y0) * args.margin
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hw, hh = max((x1 - x0) / 2 + mx, args.min_crop / 2), max((y1 - y0) / 2 + my, args.min_crop / 2)
            X0, X1 = int(max(cx - hw, 0)), int(min(cx + hw, W)); Y0, Y1 = int(max(cy - hh, 0)), int(min(cy + hh, H))
            if X1 - X0 < 8 or Y1 - Y0 < 8:
                continue
            ys, xs = torch.meshgrid(torch.arange(Y0, Y1, device=device), torch.arange(X0, X1, device=device), indexing="ij")
            uv = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1).float()
            o_w, d_w, z_per_t = pixel_rays(uv, cam, fi, device)
            o = ((o_w - t) @ R).expand(d_w.shape[0], 3); d = d_w @ R
            t_in, t_out = ray_box_intersect(o, d, half * 1.2)
            t_in = t_in.clamp_min(0.0); hit_box = t_out > t_in
            hh_, ww_ = Y1 - Y0, X1 - X0
            t_hit = torch.full((uv.shape[0],), float("nan"), device=device); smin = torch.full((uv.shape[0],), float("nan"), device=device)
            if hit_box.any():
                th, sm = march(rigid, i, o[hit_box], d[hit_box], t_in[hit_box], t_out[hit_box])
                t_hit[hit_box] = th; smin[hit_box] = sm
            hit = ~torch.isnan(t_hit)
            normal = torch.zeros(uv.shape[0], 3, device=device)
            if hit.any():
                p_hit = o[hit] + d[hit] * t_hit[hit][:, None]
                normal[hit] = rigid.query_sdf_normal(p_hit, torch.full((p_hit.shape[0],), i, dtype=torch.long, device=device))[1]
            shade = (-(normal * d).sum(-1)).clamp(0, 1)            # headlight shading; back-facing (reversed) normals go black
            backfacing = hit & ((normal * d).sum(-1) > 0)
            z_sdf = (t_hit * z_per_t).reshape(hh_, ww_)

            img_infos, cam_infos = ps.get_image(fi * ps.num_cams + u)
            rgb = img_infos["pixels"].cpu().numpy()
            if rgb.shape[0] != H:
                rgb = cv2.resize(rgb, (W, H))
            crop = (np.clip(rgb[Y0:Y1, X0:X1], 0, 1) * 255).astype(np.uint8)
            m_full = masks.get(tid, cid, fi, cam, device) if (masks is not None and tid is not None) else None
            m = m_full[Y0:Y1, X0:X1].cpu().numpy() if m_full is not None else np.zeros((hh_, ww_), bool)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            p1 = crop.copy()
            cv2.drawContours(p1, contours, -1, (0, 255, 0), 1)
            uvc = (uv_c.cpu().numpy() - [X0, Y0]).astype(int)
            for a, b in edges:
                cv2.line(p1, tuple(uvc[a]), tuple(uvc[b]), (255, 220, 0), 1, cv2.LINE_AA)
            split = "test" if fi in test_frames else "train"

            hm = hit.reshape(hh_, ww_).cpu().numpy(); sh = shade.reshape(hh_, ww_).cpu().numpy()
            bf = backfacing.reshape(hh_, ww_).cpu().numpy()
            p2 = (crop * 0.35).astype(np.uint8)
            col = np.stack([sh * 0.85 + 0.15] * 3, -1) * np.array([0.75, 0.85, 1.0])
            p2[hm] = (col[hm] * 255).astype(np.uint8)
            p2[bf] = (np.array([255, 0, 255]) * 0.6 + p2[bf] * 0.4).astype(np.uint8)
            miss = m & ~hm
            p2[miss] = np.array([255, 0, 0], np.uint8)

            smn = smin.reshape(hh_, ww_).cpu().numpy()
            p3 = diverging(np.nan_to_num(smn, nan=0.3), 0.1)
            p3[np.isnan(smn)] = (crop[np.isnan(smn)] * 0.3).astype(np.uint8)
            cv2.drawContours(p3, contours, -1, (0, 160, 0), 1)

            p4 = (crop * 0.3).astype(np.uint8)
            ld = img_infos.get("lidar_depth_map")
            n_lidar = 0
            dbg = {}
            if ld is not None:
                dbg["lidar_map_shape"] = tuple(ld.shape); dbg["image_shape"] = (H, W)
                l_c = ld.cpu().numpy()[Y0:Y1, X0:X1]; zs_c = z_sdf.cpu().numpy()
                on = m & (l_c > 1e-3)
                dbg["lidar_on_mask_median"] = float(np.median(l_c[on])) if on.any() else None
                dbg["sdf_on_mask_median"] = float(np.nanmedian(zs_c[m])) if m.any() else None
                dbg["t_in_median"] = float(t_in[hit_box].median()) if hit_box.any() else None
            if ld is not None:
                ld = ld.cpu().numpy()[Y0:Y1, X0:X1]
                zs = z_sdf.cpu().numpy()
                zin = torch.where(hit_box, t_in * z_per_t, torch.full_like(t_in, float("nan"))).reshape(hh_, ww_).cpu().numpy()
                zout = torch.where(hit_box, t_out * z_per_t, torch.full_like(t_out, float("nan"))).reshape(hh_, ww_).cpu().numpy()
                has = ld > 1e-3
                occl = has & (ld < zin - 0.1)                       # return in front of the box: an occluder
                inbox = has & (ld >= zin - 0.1) & (ld <= zout + 0.1)  # return on the object (or inside its box)
                lv = inbox & ~np.isnan(zs)
                n_lidar = int(lv.sum())
                err = np.where(lv, zs - ld, 0)
                k3 = np.ones((3, 3), np.uint8)
                errd = cv2.dilate(np.where(lv, err + 10, 0).astype(np.float32), k3)
                k = cv2.dilate(lv.astype(np.uint8), k3) > 0
                c = diverging(np.where(k, errd - 10, 0), 0.3)
                p4[k] = c[k]
                ko = (cv2.dilate(occl.astype(np.uint8), k3) > 0) & ~k
                p4[ko] = (110, 110, 110)
                lo_ = cv2.dilate((inbox & np.isnan(zs)).astype(np.uint8), k3) > 0   # returns in the box the SDF never reaches
                p4[lo_ & ~k] = (255, 255, 0)
                cv2.drawContours(p4, contours, -1, (0, 160, 0), 1)
                med = float(np.median(err[lv])) if n_lidar else float("nan")
                dbg["lidar_on_mask_occluder_frac"] = float((occl & m).sum() / max((has & m).sum(), 1))
            else:
                med = float("nan")

            panels = [label(p1, f"inst {i} frame {fi} ({split}) cam {u}"), label(p2, "SDF surface (red: mask missed, magenta: reversed)"),
                      label(p3, "min SDF on ray (+-10 cm)"), label(p4, f"SDF-LiDAR (+-30cm) med {med:+.2f}m, grey: occluder")]
            if not args.no_render:
                for dct in (img_infos, cam_infos):
                    for kk, v in dct.items():
                        if isinstance(v, torch.Tensor):
                            dct[kk] = v.to(device)
                with torch.no_grad():
                    out = trainer(img_infos, cam_infos)
                r = (out["rgb"].reshape(H, W, 3).clamp(0, 1).cpu().numpy()[Y0:Y1, X0:X1] * 255).astype(np.uint8)
                gd = out["RigidNodes_depth"].reshape(H, W).cpu().numpy()[Y0:Y1, X0:X1]
                ga = out["RigidNodes_opacity"].reshape(H, W).cpu().numpy()[Y0:Y1, X0:X1]
                zs = z_sdf.cpu().numpy()
                both = (ga > 0.5) & ~np.isnan(zs)
                p6 = (crop * 0.3).astype(np.uint8)
                c = diverging(np.where(both, gd / np.maximum(ga, 1e-6) - zs, 0), 0.3)
                p6[both] = c[both]
                gonly = (ga > 0.5) & np.isnan(zs)
                p6[gonly] = (255, 255, 0)
                cv2.drawContours(p6, contours, -1, (0, 160, 0), 1)
                panels += [label(r, "render"), label(p6, "Gaussian-SDF depth (+-30 cm), yellow: no SDF")]
            row = np.concatenate([np.pad(p, ((0, 0), (0, 4), (0, 0))) for p in panels], 1)
            scale = 260 / row.shape[0] if row.shape[0] < 260 else 1.0
            if scale != 1.0:
                row = cv2.resize(row, (int(row.shape[1] * scale), 260), interpolation=cv2.INTER_NEAREST)
            fn = os.path.join(out_dir, f"inst{i:02d}_f{fi:03d}_cam{u}.png")
            cv2.imwrite(fn, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
            m8 = cv2.dilate(m.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
            stats = dict(mask_px=int(m.sum()), mask_missed=float(miss.sum() / max(m.sum(), 1)),
                         # SDF solid seen outside the mask, relative to the mask area (and outside the mask dilated by 8 px)
                         solid_outside_mask=float((hm & ~m).sum() / max(m.sum(), 1)),
                         solid_outside_mask8=float((hm & ~m8).sum() / max(m.sum(), 1)),
                         reversed=float(bf[hm].mean()) if hm.any() else float("nan"), lidar_px=n_lidar, sdf_minus_lidar_median=med, **dbg)
            logger.info(f"  {fn}: {stats}")
            sheet.append(row)
        if sheet:
            w = max(r.shape[1] for r in sheet)
            sheet = np.concatenate([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in sheet], 0)
            fn = os.path.join(out_dir, f"inst{i:02d}_sheet.png")
            cv2.imwrite(fn, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            logger.info(f"instance {i}: contact sheet {fn}")


if __name__ == "__main__":
    main()
