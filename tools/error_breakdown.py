"""
Where is the held-out image error?  Renders the test images of a trained run once and splits
the squared error three ways, then checks which moving objects are modelled and how well the
SMPL humans fit their masks.

  1. regions: sky, static, vehicles, humans, other dynamic; object boundary band vs interior;
     static pixels by rendered distance; which node renders each dynamic pixel. For each region:
     pixel share, error share, PSNR inside it, and the gain in (per-image mean) PSNR if its error
     were zero.
  2. moving objects: every dataset instance with its class, whether it moves, has annotator masks,
     an SMPL fit, and whether a dynamic node models it; the error inside the projected box of each
     moving object on the test images, and dynamic-mask pixels outside every box.
  3. SMPL humans: per modelled human, the IoU of the rendered SMPLNodes silhouette with its
     annotator mask on test frames and on training frames, mask coverage, spill outside the mask
     and PSNR inside the mask.

usage: python tools/error_breakdown.py <run_dir> [--train_frames_every 10] [key=value overrides]
writes <run_dir>/metrics_eval/error_breakdown.json and PNGs under <run_dir>/metrics_eval/error_breakdown/
"""
import os, sys, json, argparse, logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset, load_annotator_instance_ids
from utils.misc import import_str
from models.nodes.sdf_supervision import AnnotatorMasks, project_points

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("error_breakdown")
CLASS_NAMES = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
NODES = ["Background", "RigidNodes", "SMPLNodes", "DeformableNodes"]


def dilate(m, r):
    return F.max_pool2d(m[None, None].float(), 2 * r + 1, 1, r)[0, 0] > 0.5 if r > 0 else m


def erode(m, r):
    return ~dilate(~m, r)


def box_mask(pose, size, cam, fi, device, H, W):
    corners = torch.tensor([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], device=device, dtype=torch.float32)
    cw = (corners * size / 2) @ pose[:3, :3].T + pose[:3, 3]
    uv, z, _ = project_points(cw, cam, fi, device)
    if (z <= 0.1).any():
        return None
    hull = cv2.convexHull(uv.cpu().numpy().astype(np.float32)).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    return torch.from_numpy(m > 0).to(device) if m.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--boundary_px", type=int, default=3)
    ap.add_argument("--train_frames_every", type=int, default=10, help="training frames for the SMPL fit check")
    ap.add_argument("--num_vis", type=int, default=6)
    args, opts = ap.parse_known_args()
    run = args.run_dir
    cfg = OmegaConf.load(os.path.join(run, "config.yaml"))
    cfg.data.preload_device = "cpu"
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))
    dev = torch.device("cuda")
    ds = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(**cfg.trainer, num_timesteps=ds.num_img_timesteps, model_config=cfg.model,
        num_train_images=len(ds.train_image_set), num_full_images=len(ds.full_image_set),
        test_set_indices=ds.test_timesteps, scene_aabb=ds.get_aabb().reshape(2, 3), device=dev)
    trainer.resume_from_checkpoint(ckpt_path=os.path.join(run, args.ckpt), load_only_model=True)
    trainer.set_eval()
    ps = ds.pixel_source
    out_dir = os.path.join(run, "metrics_eval", "error_breakdown"); os.makedirs(out_dir, exist_ok=True)
    cams = {c.unique_cam_idx: (cid, c) for cid, c in ps.camera_data.items()}
    nodes = [n for n in NODES if n in trainer.models]

    # ---------------------------------------------------------------- which instances are modelled
    modelled = {}
    for node in ("RigidNodes", "DeformableNodes"):
        if node in cfg.model and node in trainer.models:
            init = OmegaConf.to_container(cfg.model[node].init)
            init.pop("mono_points", None)   # lifted points do not change which instances are selected
            kw = dict(exclude_smpl="SMPLNodes" in cfg.model) if node == "DeformableNodes" else {}
            for k in ds.get_init_objects(cur_node_type=node, **init, **kw):
                modelled[int(k)] = node
    if "SMPLNodes" in cfg.model and "SMPLNodes" in trainer.models:
        for k in ds.get_init_smpl_objects(**OmegaConf.to_container(cfg.model.SMPLNodes.init)):
            modelled[int(k)] = "SMPLNodes"
    mask_path = cfg.model.get("RigidNodes", {}).get("init", {}).get("annotator_masks")
    masked_ids = load_annotator_instance_ids(mask_path, ds.scene_idx) if mask_path else set()
    masks_root = ds.annotator_masks_root(mask_path) if mask_path else None
    amasks = AnnotatorMasks(masks_root, start_timestep=next(iter(ps.camera_data.values())).start_timestep) if masks_root else None
    smpl_ids = set(int(k) for k in getattr(ps, "smpl_human_all", {}).keys())
    fv = ps.per_frame_instance_mask.bool()
    poses = ps.instances_pose
    instances = []
    for k in range(ps.instances_pose.shape[1]):
        frames = torch.where(fv[:, k])[0]
        tr = poses[frames, k, :3, 3]
        traj = float((tr[1:] - tr[:-1]).norm(dim=-1).sum()) if len(frames) > 1 else 0.0
        tid = int(ps.instances_true_id[k])
        instances.append(dict(ds_id=k, true_id=tid, cls=CLASS_NAMES.get(int(ps.instances_model_types[k]), "other"),
                              frames=int(len(frames)), traj_m=round(traj, 2), moving=traj > 0.5,
                              masked=tid in masked_ids, smpl=tid in smpl_ids, node=modelled.get(k)))
    log.info(f"{len(instances)} instances, {sum(i['moving'] for i in instances)} moving, "
             f"{sum(i['node'] is not None for i in instances)} modelled")

    # ---------------------------------------------------------------- render the test images
    test_idx = list(ds.test_image_set.split_indices)
    acc = {}                       # region -> [pixels, sse]
    per_img = []                   # (mse_total, {region: sse}, npix)
    inst_err = {i["ds_id"]: [0, 0.0] for i in instances if i["moving"]}
    nobox = [0, 0.0]
    vis = []
    def add(name, m, se, store):
        n = int(m.sum()); s = float(se[m].sum())
        a = acc.setdefault(name, [0, 0.0]); a[0] += n; a[1] += s
        store[name] = s
    depth_bins = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 1e9)]
    test_frames = sorted(set(i // ps.num_cams for i in test_idx))
    for n_img, img_idx in enumerate(test_idx):
        fi, u = img_idx // ps.num_cams, img_idx % ps.num_cams
        cid, cam = cams[u]
        ii, ci = ps.get_image(img_idx)
        for d in (ii, ci):
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.to(dev)
        with torch.no_grad():
            out = trainer(ii, ci)
        gt = ii["pixels"]; H, W = gt.shape[:2]
        pred = out["rgb"].reshape(H, W, 3).clamp(0, 1)
        se = ((pred - gt) ** 2).mean(-1)
        npix = H * W
        sky = ii["sky_masks"] > 0.5 if "sky_masks" in ii else torch.zeros_like(se, dtype=torch.bool)
        veh = ii["vehicle_masks"] > 0.5; hum = ii["human_masks"] > 0.5
        dyn = (ii["dynamic_masks"] > 0.5) | veh | hum
        oth = dyn & ~veh & ~hum
        stat = ~dyn & ~sky
        store = {}
        add("sky", sky & ~dyn, se, store); add("static", stat, se, store)
        add("vehicle", veh, se, store); add("human", hum & ~veh, se, store); add("other_dynamic", oth, se, store)
        b = args.boundary_px
        band = dilate(dyn, b) & ~erode(dyn, b)
        add("object_boundary_band", band, se, store); add("object_interior", erode(dyn, b), se, store)
        skyband = (dilate(sky, b) & ~erode(sky, b)) & ~band
        add("sky_edge_band", skyband, se, store)
        depth = out["depth"].reshape(H, W)
        for lo, hi in depth_bins:
            add(f"static_depth_{lo}-{hi if hi < 1e9 else 'inf'}m", stat & ~skyband & (depth >= lo) & (depth < hi), se, store)
        # which node renders the dynamic pixels
        owner = torch.full((H, W), -1, device=dev, dtype=torch.long)
        best = torch.zeros(H, W, device=dev)
        for j, nd in enumerate(nodes):
            if f"{nd}_opacity" in out:
                op = out[f"{nd}_opacity"].reshape(H, W)
                take = op > best
                owner[take] = j; best[take] = op[take]
        for j, nd in enumerate(nodes):
            add(f"vehicle_px_rendered_by_{nd}", veh & (owner == j), se, store)
            add(f"human_px_rendered_by_{nd}", hum & ~veh & (owner == j), se, store)
        # error inside the projected box of each moving instance
        covered = torch.zeros_like(dyn)
        for k in inst_err:
            if not bool(fv[fi, k]):
                continue
            bm = box_mask(poses[fi, k].to(dev), ps.instances_size[k].to(dev), cam, fi, dev, H, W)
            if bm is None:
                continue
            covered |= bm
            m = bm & dyn
            inst_err[k][0] += int(m.sum()); inst_err[k][1] += float(se[m].sum())
        m = dyn & ~covered
        nobox[0] += int(m.sum()); nobox[1] += float(se[m].sum())
        per_img.append((float(se.mean()), store, npix))
        vis.append((float(se.mean()), fi, u, gt.cpu().numpy(), pred.cpu().numpy(), se.cpu().numpy(), dyn.cpu().numpy()))
        vis = sorted(vis, key=lambda x: -x[0])[: args.num_vis]
        if n_img % 10 == 0:
            log.info(f"rendered {n_img + 1}/{len(test_idx)}")

    tot_pix = sum(p[2] for p in per_img); tot_sse = sum(p[0] * p[2] for p in per_img)
    psnr_mean = float(np.mean([-10 * np.log10(p[0]) for p in per_img]))
    regions = {}
    for name, (n, s) in acc.items():
        gain = float(np.mean([-10 * np.log10(max(p[0] - p[1].get(name, 0.0) / p[2], 1e-10)) for p in per_img])) - psnr_mean
        regions[name] = dict(pixel_share=n / tot_pix, error_share=s / tot_sse,
                             psnr=float(-10 * np.log10(s / max(n, 1))) if n else None, psnr_gain_if_zero=gain)
    res = dict(run=run, images=len(per_img), psnr_mean=psnr_mean, regions=regions)

    # ---------------------------------------------------------------- moving objects
    rows = []
    for i in instances:
        if not i["moving"]:
            continue
        n, s = inst_err.get(i["ds_id"], (0, 0.0))
        rows.append(dict(i, test_px=n, test_px_per_image=n / len(per_img), error_share=s / tot_sse,
                         psnr=float(-10 * np.log10(s / n)) if n else None))
    rows.sort(key=lambda r: -r["error_share"])
    res["moving_instances"] = rows
    res["unmodelled_moving"] = dict(
        count=sum(r["node"] is None for r in rows),
        count_visible_in_test=sum(r["node"] is None and r["test_px"] > 0 for r in rows),
        error_share=sum(r["error_share"] for r in rows if r["node"] is None),
        by_reason=dict(no_mask=sum(r["node"] is None and not r["masked"] and r["test_px"] > 0 for r in rows),
                       masked_but_not_modelled=sum(r["node"] is None and r["masked"] and r["test_px"] > 0 for r in rows)))
    res["dynamic_px_outside_every_moving_box"] = dict(pixel_share=nobox[0] / tot_pix, error_share=nobox[1] / tot_sse)
    res["instance_counts"] = {c: dict(total=sum(i["cls"] == c for i in instances), moving=sum(i["cls"] == c and i["moving"] for i in instances),
                                      modelled=sum(i["cls"] == c and i["node"] is not None for i in instances))
                              for c in sorted(set(i["cls"] for i in instances))}

    # ---------------------------------------------------------------- SMPL fit
    humans = [i for i in instances if i["node"] == "SMPLNodes"]
    train_frames = [f for f in range(args.train_frames_every // 2, fv.shape[0], args.train_frames_every) if f not in set(test_frames)]
    res["smpl"] = []
    if humans and amasks is not None and "SMPLNodes_opacity" not in out:
        log.info("no SMPLNodes_opacity in the render outputs; skipping the SMPL check")
    elif humans and amasks is not None:
        hum_vis = []
        stats = {h["ds_id"]: dict(test=[], train=[]) for h in humans}
        for split, frames in (("test", test_frames), ("train", train_frames)):
            for fi in frames:
                active = [h for h in humans if bool(fv[fi, h["ds_id"]])]
                if not active:
                    continue
                for u, (cid, cam) in cams.items():
                    have = [h for h in active if cid in amasks.cameras(h["true_id"], fi)]
                    if not have:
                        continue
                    ii, ci = ps.get_image(fi * ps.num_cams + u)
                    for d in (ii, ci):
                        for k, v in d.items():
                            if isinstance(v, torch.Tensor):
                                d[k] = v.to(dev)
                    with torch.no_grad():
                        out = trainer(ii, ci)
                    gt = ii["pixels"]; H, W = gt.shape[:2]
                    pred = out["rgb"].reshape(H, W, 3).clamp(0, 1)
                    sil = out["SMPLNodes_opacity"].reshape(H, W) > 0.5
                    for h in have:
                        m = amasks.get(h["true_id"], cid, fi, cam, dev)
                        bm = box_mask(poses[fi, h["ds_id"]].to(dev), ps.instances_size[h["ds_id"]].to(dev) * 1.3, cam, fi, dev, H, W)
                        if m is None or bm is None or m.sum() < 50:
                            continue
                        region = dilate(bm | m, 10)
                        s = sil & region
                        inter = int((s & m).sum()); uni = int((s | m).sum())
                        se = ((pred - gt) ** 2).mean(-1)[m]
                        stats[h["ds_id"]][split].append(dict(iou=inter / max(uni, 1), coverage=inter / int(m.sum()),
                                                             spill=int((s & ~m).sum()) / int(m.sum()), mask_px=int(m.sum()),
                                                             mse=float(se.mean())))
                        if split == "test" and len(hum_vis) < 12 and m.sum() > 400:
                            ys, xs = torch.where(region)
                            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                            hum_vis.append((h["true_id"], fi, u, gt.cpu().numpy()[y0:y1, x0:x1], pred.cpu().numpy()[y0:y1, x0:x1],
                                            m.cpu().numpy()[y0:y1, x0:x1], s.cpu().numpy()[y0:y1, x0:x1]))
        for h in humans:
            row = dict(true_id=h["true_id"], ds_id=h["ds_id"], frames=h["frames"], traj_m=h["traj_m"])
            for split in ("test", "train"):
                v = stats[h["ds_id"]][split]
                if v:
                    row[split] = dict(views=len(v), iou=float(np.mean([x["iou"] for x in v])),
                                      coverage=float(np.mean([x["coverage"] for x in v])), spill=float(np.mean([x["spill"] for x in v])),
                                      mask_px=float(np.mean([x["mask_px"] for x in v])),
                                      psnr_in_mask=float(-10 * np.log10(np.mean([x["mse"] for x in v]))))
            res["smpl"].append(row)
        # contact sheet of human crops: image | render | mask (green) vs SMPL silhouette (red)
        tiles = []
        for tid, fi, u, g, p, m, s in hum_vis:
            g8 = (g * 255).astype(np.uint8); p8 = (p * 255).astype(np.uint8)
            ov = (g8 * 0.5).astype(np.uint8)
            ov[m & s] = (255, 255, 0); ov[m & ~s] = (0, 255, 0); ov[s & ~m] = (255, 0, 0)
            t = np.concatenate([g8, p8, ov], 1)
            t = cv2.resize(t, (int(t.shape[1] * 200 / t.shape[0]), 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(t, f"id {tid} f{fi} c{u}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(t)
        if tiles:
            w = max(t.shape[1] for t in tiles)
            sheet = np.concatenate([np.pad(t, ((0, 4), (0, w - t.shape[1]), (0, 0))) for t in tiles], 0)
            cv2.imwrite(os.path.join(out_dir, "smpl_fit_test.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    # worst test images: image | render | squared error (dynamic mask outlined)
    for k, (mse, fi, u, g, p, se, dyn) in enumerate(vis):
        e = np.clip(se / 0.02, 0, 1)
        e8 = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
        cs, _ = cv2.findContours(dyn.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(e8, cs, -1, (0, 255, 0), 1)
        row = np.concatenate([(g * 255).astype(np.uint8), (p * 255).astype(np.uint8), e8], 1)
        cv2.putText(row, f"frame {fi} cam {u} PSNR {-10 * np.log10(mse):.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"worst_{k}_f{fi:03d}_cam{u}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    with open(os.path.join(run, "metrics_eval", "error_breakdown.json"), "w") as f:
        json.dump(res, f, indent=1)
    log.info("RESULT " + json.dumps(dict(psnr_mean=psnr_mean, regions=regions, unmodelled_moving=res["unmodelled_moving"],
                                         instance_counts=res["instance_counts"]), indent=1))


if __name__ == "__main__":
    main()
