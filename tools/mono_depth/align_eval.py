"""
How accurate is MoGe-2 depth on vehicles once it is aligned to LiDAR with a scale and a shift?

For every training view and every vehicle with enough LiDAR returns in it, the returns are split
into a fit half and a test half; the alignment is fitted on one and the aligned depth is scored
on the other (z-depth error in metres), by range. Two splits: `random` (interpolation between
returns) and `spatial` (left / right halves of the vehicle in the image: extrapolation to a part
with no returns, which is what lifting needs).

    python tools/mono_depth/align_eval.py <dump dir> [--min_anchors 40]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import fit_scale, fit_scale_shift, backproject, in_box

parser = argparse.ArgumentParser()
parser.add_argument("dump_dir")
parser.add_argument("--min_anchors", type=int, default=40)
parser.add_argument("--max_range", type=float, default=80.0)
args = parser.parse_args()

meta = np.load(os.path.join(args.dump_dir, "meta.npz"))
poses, active = meta["instances_pose"], meta["instance_active"]
sizes = meta["instances_size"]
if sizes.ndim == 3:
    sizes = sizes.max(axis=0)
rigid = np.where(meta["instance_model_type"] == 0)[0]
test_frames = set(meta["test_frames"].tolist())
F = poses.shape[0]
# a vehicle counts as moving in a frame if its box centre moves more than 5 cm to the next frame
step = np.zeros(active.shape)
d = np.linalg.norm(poses[1:, :, :3, 3] - poses[:-1, :, :3, 3], axis=-1)
step[:-1] = d; step[-1] = d[-1]
moving = (step > 0.05) & active

METHODS = ["raw", "image_scale_shift", "object_scale", "object_shift", "object_scale_shift"]
BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80)]
rows = []          # (split, method, range, abs error, moving)
image_fits, n_groups, n_shift_used, n_shift_tried = [], 0, 0, 0
rng = np.random.default_rng(0)

for cam_id in meta["cam_ids"].tolist():
    cam = np.load(os.path.join(args.dump_dir, f"cam_{cam_id}.npz"))
    lidar_all, Ks, c2ws = cam["lidar_depth"], cam["intrinsics"], cam["cam_to_worlds"]   # decompress once
    for fi in range(F):
        if fi in test_frames:
            continue
        mg = np.load(os.path.join(args.dump_dir, "moge2", f"{cam_id}_{fi:03d}.npz"))
        L, M = lidar_all[fi], mg["depth"]
        v, u = np.nonzero((L > 1e-3) & (L < args.max_range) & mg["mask"] & (M > 0))
        if u.size < 200:
            continue
        l, m = L[v, u].astype(np.float64), M[v, u].astype(np.float64)
        p_world = backproject(u, v, l, Ks[fi], c2ws[fi])
        owner = np.full(u.size, -1)
        for i in rigid[active[fi, rigid]]:
            _, inside = in_box(p_world, poses[fi, i], sizes[i])
            owner[inside] = i
        static = owner < 0
        # whole-image fit on the returns that are on no vehicle
        gs, gt, _ = fit_scale_shift(m[static], l[static], delta=0.5)
        image_fits.append((gs, gt, fit_scale(m[static], l[static])))
        for i in np.unique(owner[owner >= 0]):
            idx = np.nonzero(owner == i)[0]
            if idx.size < args.min_anchors:
                continue
            n_groups += 1
            order = idx[np.argsort(u[idx])]
            perm = rng.permutation(idx)
            h = idx.size // 2
            splits = [("random", perm[:h], perm[h:]), ("spatial", order[:h], order[h:]), ("spatial", order[h:], order[:h])]
            for name, fit, test in splits:
                s1 = fit_scale(m[fit], l[fit])
                s2, t2, used = fit_scale_shift(m[fit], l[fit])
                n_shift_tried += 1; n_shift_used += int(used)
                t3 = float(np.median(l[fit] - gs * m[fit]))
                pred = {"raw": m[test], "image_scale_shift": gs * m[test] + gt, "object_scale": s1 * m[test],
                        "object_shift": gs * m[test] + t3, "object_scale_shift": s2 * m[test] + t2}
                for k, p in pred.items():
                    e = np.abs(p - l[test])
                    rows.append((name, k, l[test], e, np.full(e.size, bool(moving[fi, i]))))

def summarise(e):
    if e.size == 0:
        return None
    return dict(n=int(e.size), median_cm=round(float(np.median(e)) * 100, 1), mean_cm=round(float(e.mean()) * 100, 1),
                within_5cm=round(float((e < 0.05).mean()), 3), within_10cm=round(float((e < 0.10).mean()), 3),
                within_20cm=round(float((e < 0.20).mean()), 3))

out = dict(groups=n_groups, shift_used_frac=round(n_shift_used / max(n_shift_tried, 1), 3), results={})
fits = np.array(image_fits)
out["image_fit"] = dict(views=int(len(fits)), scale_q10_50_90=np.percentile(fits[:, 0], [10, 50, 90]).round(3).tolist(),
                        shift_m_q10_50_90=np.percentile(fits[:, 1], [10, 50, 90]).round(3).tolist(),
                        scale_only_q10_50_90=np.percentile(fits[:, 2], [10, 50, 90]).round(3).tolist())
for split in ("random", "spatial"):
    for k in METHODS:
        sel = [r for r in rows if r[0] == split and r[1] == k]
        rg, e, mv = (np.concatenate([r[j] for r in sel]) for j in (2, 3, 4))
        res = dict(all=summarise(e), parked=summarise(e[~mv]), moving=summarise(e[mv]))
        for lo, hi in BINS:
            res[f"{lo}-{hi}m"] = summarise(e[(rg >= lo) & (rg < hi)])
        out["results"][f"{split}/{k}"] = res
json.dump(out, open(os.path.join(args.dump_dir, "align_eval.json"), "w"), indent=1)

print(f"{n_groups} vehicle-view groups; shift used in {out['shift_used_frac']:.0%} of fits; image fit {out['image_fit']}")
cols = ["all", "parked", "moving"] + [f"{lo}-{hi}m" for lo, hi in BINS]
for split in ("random", "spatial"):
    print(f"\n== {split} split: median error cm | within 5 cm | within 10 cm   (n = {out['results'][split + '/raw']['all']['n']})")
    print(f"{'method':22s}" + "".join(f"{c:>20s}" for c in cols))
    for k in METHODS:
        r = out["results"][f"{split}/{k}"]
        print(f"{k:22s}" + "".join(
            f"{'-':>20s}" if r[c] is None else f"{r[c]['median_cm']:8.1f} {r[c]['within_5cm']:5.2f} {r[c]['within_10cm']:5.2f}" for c in cols))
