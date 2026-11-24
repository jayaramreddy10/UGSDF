"""
Lift MoGe-2 depth to 3D per vehicle and fuse it across views in the object frame.

For every training view and every vehicle in it, the monocular depth is aligned to the
vehicle's own LiDAR returns in that view (scale and shift, see align.py) and integrated into a
truncated signed distance volume over the vehicle's box, in the object frame. Pixels whose
aligned depth lies beyond the box carve the volume, pixels in front of it are occluders and are
ignored. The fused surface (zero crossing seen by enough views) is written as points with
colours and normals, each flagged with its distance to the nearest LiDAR return, so the points
that fill regions LiDAR never measured can be picked out.

    python tools/mono_depth/lift.py <dump dir> [--cell 0.05 --trunc 0.15 --max_range 40]

Writes <dump dir>/lifted.npz (pts, colors, normals pointing out of the object, views that saw the
point, `facing`: integrated views its normal faces, lidar_dist, instance; object frame;
`normals_outward` marks files with that normal convention)
and <dump dir>/lifted_stats.json.
"""
import argparse, json, os, sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import fit_scale, fit_scale_shift, backproject, in_box

parser = argparse.ArgumentParser()
parser.add_argument("dump_dir")
parser.add_argument("--cell", type=float, default=0.05, help="voxel size (m)")
parser.add_argument("--trunc", type=float, default=0.15, help="truncation distance (m)")
parser.add_argument("--trunc_behind", type=float, default=0.5,
                    help="a view votes 'inside' this far behind its surface (m). The per-view depth error is about as large as "
                         "the truncation, so with the usual symmetric band (set this to --trunc) a view that puts the surface "
                         "deeper outvotes nothing behind a shallower view's band, and the volume crosses zero three times: the "
                         "fused surface then has an inner copy and, between the two, a sheet with inverted normals")
parser.add_argument("--max_range", type=float, default=40.0, help="views farther than this from the vehicle are not used")
parser.add_argument("--min_anchors", type=int, default=30, help="LiDAR returns on the vehicle needed for its own scale+shift")
parser.add_argument("--min_anchors_shift", type=int, default=5, help="... for the image scale with a per-vehicle shift")
parser.add_argument("--edge_rel", type=float, default=0.03, help="relative depth jump to a neighbour that marks an edge pixel")
parser.add_argument("--min_views", type=int, default=3, help="views that must see a surface voxel")
parser.add_argument("--ground_margin", type=float, default=0.12)
parser.add_argument("--instances", type=int, nargs="*", default=None, help="dataset instance indices (default: all vehicles)")
parser.add_argument("--align", default="scale_shift", choices=["scale_shift", "scale"],
                    help="per-vehicle alignment to its LiDAR returns: scale and shift (scale only where the shift is not identifiable), or scale only")
parser.add_argument("--out", default="lifted")
args = parser.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta = np.load(os.path.join(args.dump_dir, "meta.npz"))
poses, active = meta["instances_pose"], meta["instance_active"]
sizes = meta["instances_size"]
if sizes.ndim == 3:
    sizes = sizes.max(axis=0)
rigid = np.where(meta["instance_model_type"] == 0)[0] if args.instances is None else np.asarray(args.instances)
test_frames = set(meta["test_frames"].tolist())
F = poses.shape[0]
corners_unit = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)


class Volume:
    """Truncated signed distance volume over one instance's box (object frame)."""

    def __init__(self, size):
        self.half = size / 2
        self.dims = np.maximum(np.ceil(size / args.cell).astype(int), 2)
        ax = [(-self.half[k] + (np.arange(self.dims[k]) + 0.5) * size[k] / self.dims[k]) for k in range(3)]
        self.spacing = [size[k] / self.dims[k] for k in range(3)]
        g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
        self.centres = torch.from_numpy(g).float().to(dev)
        n = self.centres.shape[0]
        self.D = torch.zeros(n, device=dev); self.W = torch.zeros(n, device=dev)
        self.C = torch.zeros(n, 3, device=dev); self.Wc = torch.zeros(n, device=dev)
        self.near = torch.zeros(n, device=dev)            # views that saw a surface within the truncation
        self.views, self.own_fit, self.lidar = 0, 0, []
        self.cams = []                                    # camera centres of the integrated views, object frame

    def integrate(self, depth, rgb, K, c2w, pose):
        """depth: (H, W) aligned z-depth, 0 where unusable. rgb: (H, W, 3) in [0, 1]."""
        H, W = depth.shape
        w2c = torch.linalg.inv(c2w)
        p_cam = (self.centres @ pose[:3, :3].T + pose[:3, 3]) @ w2c[:3, :3].T + w2c[:3, 3]
        z = p_cam[:, 2]
        u = torch.floor(K[0, 0] * p_cam[:, 0] / z.clamp_min(1e-6) + K[0, 2]).long()
        v = torch.floor(K[1, 1] * p_cam[:, 1] / z.clamp_min(1e-6) + K[1, 2]).long()
        ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = torch.nonzero(ok)[:, 0]
        d = depth[v[idx], u[idx]]
        sdf = d - z[idx]
        use = (d > 0) & (sdf > -max(args.trunc_behind, args.trunc))
        idx, sdf, d = idx[use], sdf[use].clamp(-args.trunc, args.trunc), d[use]
        w = (10.0 / d.clamp_min(10.0)) ** 2                # nearer views count more
        self.D[idx] += w * sdf; self.W[idx] += w
        near = sdf.abs() < args.trunc
        self.near[idx[near]] += 1
        self.C[idx[near]] += w[near, None] * rgb[v[idx[near]], u[idx[near]]]
        self.Wc[idx[near]] += w[near]
        self.views += 1
        self.cams.append(torch.linalg.inv(pose)[:3, :3] @ c2w[:3, 3] + torch.linalg.inv(pose)[:3, 3])

    def surface(self):
        """Zero crossing of the fused volume where at least min_views views saw a surface."""
        from skimage.measure import marching_cubes
        dims = tuple(self.dims.tolist())
        tsdf = (self.D / self.W.clamp_min(1e-9)).reshape(dims).cpu().numpy()
        seen = ((self.W > 0) & (self.near >= args.min_views)).reshape(dims).cpu().numpy()
        free = ((self.W > 0) & (self.near < args.min_views)).reshape(dims).cpu().numpy()
        if seen.sum() < 8 or tsdf[seen].min() >= 0 or tsdf[seen].max() <= 0:
            return None
        # observed-free voxels keep their (positive) value so a surface next to carved space closes;
        # unobserved voxels are masked out so no surface appears at the edge of what was seen
        mask = seen | (free & (tsdf > 0))
        try:
            verts, _, normals, _ = marching_cubes(tsdf, level=0.0, spacing=self.spacing, mask=mask)
        except (ValueError, RuntimeError):
            return None
        verts = verts + np.array(self.spacing) / 2 - self.half
        ijk = np.clip(np.floor((verts + self.half) / np.array(self.spacing)).astype(int), 0, np.array(dims) - 1)
        flat = np.ravel_multi_index(ijk.T, dims)
        keep = seen.reshape(-1)[flat] & (verts[:, 2] > -self.half[2] + args.ground_margin)
        col = (self.C / self.Wc.clamp_min(1e-9)[:, None]).cpu().numpy()[flat]
        views = self.near.cpu().numpy()[flat]
        # marching cubes returns normals along decreasing values, i.e. into the object: flip them
        verts, normals = verts[keep].astype(np.float32), -normals[keep].astype(np.float32)
        # integrated views the normal faces (no visibility test): a correctly oriented point faces several
        cams = torch.stack(self.cams)
        v_t, n_t = torch.from_numpy(verts).to(dev), torch.from_numpy(normals).to(dev)
        to_cam = torch.nn.functional.normalize(cams[None] - v_t[:, None], dim=-1)
        facing = ((to_cam * n_t[:, None]).sum(-1) > 0.2).sum(1).cpu().numpy()
        return verts, col[keep].astype(np.float32), normals, views[keep].astype(np.int32), facing.astype(np.int32)


def edge_mask(depth):
    """Pixels whose depth jumps by more than edge_rel relative to a 4-neighbour (flying pixels)."""
    e = np.zeros_like(depth, dtype=bool)
    for ax in (0, 1):
        d = np.abs(np.diff(depth, axis=ax)) > args.edge_rel * np.minimum(depth.take(range(1, depth.shape[ax]), ax),
                                                                         depth.take(range(0, depth.shape[ax] - 1), ax))
        if ax == 0:
            e[1:] |= d; e[:-1] |= d
        else:
            e[:, 1:] |= d; e[:, :-1] |= d
    return e


volumes = {int(i): Volume(sizes[i]) for i in rigid}
used = dict(own=0, shift=0, skipped_no_lidar=0, skipped_far=0)
for cam_id in meta["cam_ids"].tolist():
    cam = np.load(os.path.join(args.dump_dir, f"cam_{cam_id}.npz"))
    lidar_all, Ks, c2ws = cam["lidar_depth"], cam["intrinsics"], cam["cam_to_worlds"]
    H, W = lidar_all.shape[1:]
    for fi in range(F):
        if fi in test_frames or not active[fi, rigid].any():
            continue
        K, c2w = Ks[fi], c2ws[fi]
        w2c = np.linalg.inv(c2w)
        todo = []
        for i in rigid[active[fi, rigid]]:
            c_cam = w2c[:3, :3] @ poses[fi, i, :3, 3] + w2c[:3, 3]
            if c_cam[2] < 0.5:
                continue
            if c_cam[2] > args.max_range:
                used["skipped_far"] += 1; continue
            pc = ((corners_unit * sizes[i] / 2) @ poses[fi, i, :3, :3].T + poses[fi, i, :3, 3]) @ w2c[:3, :3].T + w2c[:3, 3]
            if (pc[:, 2] < 0.1).any():
                continue
            uv = pc[:, :2] / pc[:, 2:] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
            if uv[:, 0].max() < 0 or uv[:, 0].min() >= W or uv[:, 1].max() < 0 or uv[:, 1].min() >= H:
                continue
            todo.append(int(i))
        if not todo:
            continue
        mg = np.load(os.path.join(args.dump_dir, "moge2", f"{cam_id}_{fi:03d}.npz"))
        M = np.where(mg["mask"], mg["depth"], 0).astype(np.float32)
        usable = (M > 0) & ~edge_mask(M)
        L = lidar_all[fi]
        v, u = np.nonzero((L > 1e-3) & (L < 80) & (M > 0))
        l, m = L[v, u].astype(np.float64), M[v, u].astype(np.float64)
        p_world = backproject(u, v, l, K, c2w)
        owner = np.full(u.size, -1)
        objs = {}
        for i in rigid[active[fi, rigid]]:
            p_obj, inside = in_box(p_world, poses[fi, i], sizes[i], ground_margin=args.ground_margin)
            owner[inside] = i
            objs[int(i)] = p_obj[inside]
        static = owner < 0
        if static.sum() < 200:
            continue
        gs, gt, _ = fit_scale_shift(m[static], l[static], delta=0.5)
        rgb = torch.from_numpy(cv2.imread(os.path.join(args.dump_dir, "views", f"{cam_id}_{fi:03d}.png"))[..., ::-1].copy()).float().to(dev) / 255
        M_t, usable_t = torch.from_numpy(M).to(dev), torch.from_numpy(usable).to(dev)
        K_t, c2w_t = torch.from_numpy(K).float().to(dev), torch.from_numpy(c2w).float().to(dev)
        for i in todo:
            a = owner == i
            if a.sum() >= args.min_anchors:
                s, t = fit_scale_shift(m[a], l[a])[:2] if args.align == "scale_shift" else (fit_scale(m[a], l[a]), 0.0)
                used["own"] += 1; volumes[i].own_fit += 1
            elif a.sum() >= args.min_anchors_shift:
                s, t = gs, float(np.median(l[a] - gs * m[a])); used["shift"] += 1
            else:
                used["skipped_no_lidar"] += 1; continue
            depth = torch.where(usable_t, s * M_t + t, torch.zeros_like(M_t))
            volumes[i].integrate(depth, rgb, K_t, c2w_t, torch.from_numpy(poses[fi, i]).float().to(dev))
            volumes[i].lidar.append(objs[i].astype(np.float32))
    print(f"camera {cam_id} done: {used}", flush=True)

from scipy.spatial import cKDTree
out = dict(pts=[], colors=[], normals=[], views=[], facing=[], lidar_dist=[], instance=[])
stats = []
for i, vol in volumes.items():
    if vol.views == 0:
        continue
    surf = vol.surface()
    row = dict(instance=i, views=vol.views, own_fit_views=vol.own_fit, size=[round(float(x), 2) for x in sizes[i]], surface_points=0)
    if surf is not None and surf[0].shape[0] > 0:
        pts, col, nrm, nv, facing = surf
        lid = np.concatenate(vol.lidar) if vol.lidar else np.zeros((0, 3), np.float32)
        dist = cKDTree(lid).query(pts)[0].astype(np.float32) if lid.shape[0] else np.full(pts.shape[0], np.inf, np.float32)
        row.update(surface_points=int(pts.shape[0]), lidar_points=int(lid.shape[0]), normal_faces_no_view=round(float((facing == 0).mean()), 3),
                   beyond_5cm_of_lidar=int((dist > 0.05).sum()), beyond_10cm_of_lidar=int((dist > 0.10).sum()))
        if lid.shape[0]:   # how close the fused surface is to the LiDAR returns it can be compared with
            d_l2s = cKDTree(pts).query(lid)[0]
            row.update(lidar_to_surface_median_cm=round(float(np.median(d_l2s)) * 100, 1),
                       lidar_within_5cm=round(float((d_l2s < 0.05).mean()), 3), lidar_within_10cm=round(float((d_l2s < 0.10).mean()), 3))
        out["pts"].append(pts); out["colors"].append(col); out["normals"].append(nrm); out["views"].append(nv)
        out["lidar_dist"].append(dist); out["instance"].append(np.full(pts.shape[0], i, np.int32)); out["facing"].append(facing)
    stats.append(row)
np.savez_compressed(os.path.join(args.dump_dir, args.out + ".npz"), **{k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()},
                    cell=args.cell, trunc=args.trunc, trunc_behind=args.trunc_behind, normals_outward=True)
json.dump(dict(args=vars(args), fits=used, instances=stats), open(os.path.join(args.dump_dir, args.out + "_stats.json"), "w"), indent=1)
print(f"fits: {used}")
print(f"{'ins':>4s} {'views':>6s} {'own':>5s} {'surface':>8s} {'>5cm':>7s} {'>10cm':>7s} {'lidar':>8s} {'l2s cm':>7s} {'<5cm':>6s} {'<10cm':>6s}  size")
for r in sorted(stats, key=lambda r: -r["surface_points"]):
    if r["surface_points"] == 0:
        continue
    print(f"{r['instance']:4d} {r['views']:6d} {r['own_fit_views']:5d} {r['surface_points']:8d} {r['beyond_5cm_of_lidar']:7d} {r['beyond_10cm_of_lidar']:7d} "
          f"{r.get('lidar_points', 0):8d} {r.get('lidar_to_surface_median_cm', float('nan')):7.1f} {r.get('lidar_within_5cm', float('nan')):6.2f} {r.get('lidar_within_10cm', float('nan')):6.2f}  {r['size']}")
