"""
Helpers shared by SDFGSNodes (training supervision) and the evaluation: per-instance
LiDAR gathering in the object frame, annotator-mask loading, and camera projection.
All geometry is in metres; object frames follow the Waymo box convention.
"""
from typing import Dict, List, Optional, Tuple
import json
import os

import cv2
import numpy as np
import torch

from models.gaussians.basics import quat_to_rotmat


def ray_box_intersect(o: torch.Tensor, d: torch.Tensor, half: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slab test of rays (o + t d) against the axis-aligned box [-half, half]. Returns (t_in, t_out)."""
    d_safe = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t1 = (-half - o) / d_safe
    t2 = (half - o) / d_safe
    t_in = torch.minimum(t1, t2).max(dim=-1).values
    t_out = torch.maximum(t1, t2).min(dim=-1).values
    return t_in, t_out


def world_to_object(quat: torch.Tensor, trans: torch.Tensor):
    """Returns (R, t) with p_obj = (p_world - t) @ R, for one instance pose (wxyz quat, translation)."""
    R = quat_to_rotmat(quat[None])[0]  # object -> world
    return R, trans


class AnnotatorMasks:
    """
    MaskPtAnnotator per-instance masks, keyed by the dataset's instance true id, camera id
    and dataset frame index. Masks are returned at the camera's loaded resolution and
    undistorted exactly like DriveStudio's own masks, so they align with the images.
    """

    def __init__(self, masks_root: str, start_timestep: int = 0):
        index = os.path.join(masks_root, "masks_index.json")
        with open(index, "r") as f:
            records = json.load(f)["records"]
        self.records: Dict[int, Dict[int, Dict[int, str]]] = {
            int(k): {int(c): {int(fr): p for fr, p in frames.items()} for c, frames in v["cameras"].items()}
            for k, v in records.items()
        }
        self.start_timestep = start_timestep
        self._cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def has_instance(self, true_id: int) -> bool:
        return true_id in self.records

    def cameras(self, true_id: int, frame_idx: int) -> List[int]:
        return [c for c, frames in self.records.get(true_id, {}).items() if frame_idx + self.start_timestep in frames]

    def get(self, true_id: int, cam_id: int, frame_idx: int, cam, device) -> Optional[torch.Tensor]:
        """Bool (H, W) mask at the camera's loaded size, or None."""
        path = self.records.get(true_id, {}).get(cam_id, {}).get(frame_idx + self.start_timestep)
        if path is None:
            return None
        key = (true_id, cam_id, frame_idx)
        if key not in self._cache:
            m = np.load(path)["mask"]
            m = cv2.resize(m, (cam.load_size[1], cam.load_size[0]), interpolation=cv2.INTER_NEAREST)
            if cam.undistort:
                m = cv2.undistort(m, cam.intrinsics[frame_idx].cpu().numpy(), cam.distortions[frame_idx].cpu().numpy())
            self._cache[key] = torch.from_numpy(m > 127).to(device)
        return self._cache[key]


def repose(x: torch.Tensor, frame: torch.Tensor, ids: torch.Tensor, R0: torch.Tensor, t0: torch.Tensor,
           R1: torch.Tensor, t1: torch.Tensor, is_dir: bool = False) -> torch.Tensor:
    """Object-frame points (or directions) recorded under pose (R0, t0) of their frame, expressed in the
    object frame of pose (R1, t1). R*: (F, N, 3, 3) object -> world, t*: (F, N, 3); frame, ids: (M,)."""
    M = R0[frame, ids].transpose(-1, -2) @ R1[frame, ids]          # p1 = p0 @ R0^T @ R1 + (t0 - t1) @ R1
    out = torch.einsum("nj,njk->nk", x, M)
    if not is_dir:
        out = out + torch.einsum("nj,njk->nk", t0[frame, ids] - t1[frame, ids], R1[frame, ids])
    return out


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    return torch.nn.functional.max_pool2d(mask[None, None].float(), 2 * r + 1, stride=1, padding=r)[0, 0] > 0.5


def project_points(points_world: torch.Tensor, cam, frame_idx: int, device):
    """Pixel coordinates (N, 2), z-depth (N,) and in-image mask (N,) for world points."""
    K = cam.intrinsics[frame_idx].to(device)
    c2w = cam.cam_to_worlds[frame_idx].to(device)
    w2c = torch.linalg.inv(c2w)
    p_cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = p_cam[:, 2]
    uv = (p_cam @ K.T)[:, :2] / z.clamp_min(1e-6)[:, None]
    inside = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.HEIGHT)
    return uv, z, inside


def pixel_rays(uv: torch.Tensor, cam, frame_idx: int, device):
    """World-space origin (3,) and unit directions (N, 3) of the rays through pixel centres."""
    K = cam.intrinsics[frame_idx].to(device)
    c2w = cam.cam_to_worlds[frame_idx].to(device)
    d_cam = torch.cat([uv + 0.5, torch.ones_like(uv[:, :1])], dim=-1) @ torch.linalg.inv(K).T
    d_world = d_cam @ c2w[:3, :3].T
    d_world = torch.nn.functional.normalize(d_world, dim=-1)
    return c2w[:3, 3], d_world, d_cam[:, 2] / d_cam.norm(dim=-1)  # last: z-depth per unit ray length


@torch.no_grad()
def gather_instance_lidar(
    lidar_source,
    frames: List[int],
    quats: torch.Tensor,      # (F, N, 4) wxyz, activated
    trans: torch.Tensor,      # (F, N, 3)
    fv: torch.Tensor,         # (F, N) bool, instance active
    sizes: torch.Tensor,      # (N, 3)
    device,
    box_pad: float = 0.3,
    ground_margin: float = 0.12,
    surface_tol: float = 0.02,
    free_margin: float = 0.05,
    masks: Optional[AnnotatorMasks] = None,
    true_ids: Optional[List[int]] = None,
    pixel_source=None,
    mask_tol_px: int = 2,
    unverified_w: float = 0.35,
) -> List[Dict[str, List[torch.Tensor]]]:
    """
    Per instance (model order), the LiDAR rays crossing its padded box and the returns on
    it, in the object frame of each frame. With annotator masks, a return counts as surface
    only if it projects inside the instance's mask (weight 1); returns the masks cannot
    check keep the box/ground-margin test with weight `unverified_w`; returns that project
    outside the mask are dropped (ground, neighbours).

    Returns a list of dicts with lists of tensors: ray_o, ray_d, ray_t (M,2), surf_p,
    surf_d, surf_w, surf_oy (object-frame y of the origin of the ray behind each return) and
    surf_frame (frame index per surface return).
    """
    num_instances = sizes.shape[0]
    per_ins = [dict(ray_o=[], ray_d=[], ray_t=[], ray_frame=[], surf_p=[], surf_d=[], surf_w=[], surf_frame=[], surf_oy=[])
               for _ in range(num_instances)]
    for fi in frames:
        active = torch.where(fv[fi])[0]
        if len(active) == 0:
            continue
        rays = lidar_source.get_lidar_rays(fi)
        o_w = rays["lidar_origins"].to(device)
        d_w = rays["lidar_viewdirs"].to(device)
        r = rays["lidar_ranges"].reshape(-1).to(device)
        for i in active.tolist():
            R, t = world_to_object(quats[fi, i], trans[fi, i])
            o = (o_w - t) @ R
            d = d_w @ R
            half = sizes[i] / 2
            t_in, t_out = ray_box_intersect(o, d, half + box_pad)
            t_in = t_in.clamp_min(0.0)
            crosses = (t_out > t_in) & (t_in < r)
            if crosses.sum() == 0:
                continue
            o, d, r_i, t_in, t_out = o[crosses], d[crosses], r[crosses], t_in[crosses], t_out[crosses]
            t1 = torch.minimum(t_out, r_i - free_margin)
            has_free = t1 > t_in
            per_ins[i]["ray_o"].append(o[has_free])
            per_ins[i]["ray_d"].append(d[has_free])
            per_ins[i]["ray_t"].append(torch.stack([t_in[has_free], t1[has_free]], dim=-1))
            per_ins[i]["ray_frame"].append(torch.full((int(has_free.sum()),), fi, device=device, dtype=torch.long))

            p = o + d * r_i[:, None]
            in_box = (p.abs() <= half + surface_tol).all(dim=-1)
            if in_box.sum() == 0:
                continue
            p_box, d_box, oy_box = p[in_box], d[in_box], o[in_box, 1]
            w = torch.full((p_box.shape[0],), unverified_w, device=device)
            keep = p_box[:, 2] > -half[2] + ground_margin
            if masks is not None and true_ids is not None and pixel_source is not None:
                tid = true_ids[i]
                seen = torch.zeros(p_box.shape[0], dtype=torch.bool, device=device)
                verified = torch.zeros_like(seen)
                p_world = p_box @ R.T + t
                for cam_id in masks.cameras(tid, fi):
                    if cam_id not in pixel_source.camera_data:
                        continue  # masks exist for cameras this run does not load
                    cam = pixel_source.camera_data[cam_id]
                    m = masks.get(tid, cam_id, fi, cam, device)
                    if m is None:
                        continue
                    uv, z, inside = project_points(p_world, cam, fi, device)
                    m_tol = dilate(m, mask_tol_px)
                    uvi = uv[inside].long()
                    hit = m_tol[uvi[:, 1], uvi[:, 0]]
                    seen[inside] = True
                    verified[inside] |= hit
                # seen by a camera with a mask: the mask decides; otherwise fall back to the box test
                keep = torch.where(seen, verified, keep)
                w = torch.where(verified, torch.ones_like(w), w)
            per_ins[i]["surf_p"].append(p_box[keep])
            per_ins[i]["surf_d"].append(d_box[keep])
            per_ins[i]["surf_w"].append(w[keep])
            per_ins[i]["surf_oy"].append(oy_box[keep])  # y of the ray origin: which side saw it
            per_ins[i]["surf_frame"].append(torch.full((int(keep.sum()),), fi, device=device, dtype=torch.long))
    return per_ins


@torch.no_grad()
def gather_carving_rays(
    frames: List[int],
    quats: torch.Tensor,
    trans: torch.Tensor,
    fv: torch.Tensor,
    sizes: torch.Tensor,
    device,
    masks: AnnotatorMasks,
    true_ids: List[int],
    pixel_source,
    box_pad: float = 0.3,
    mask_dilate_px: int = 8,
    pixels_per_view: int = 1500,
    seed: int = 0,
    inside: bool = False,
    mask_erode_px: int = 4,
    inside_tol: float = 0.02,
):
    """
    Camera rays that see past an instance: pixels inside the projected (padded) box but
    outside the dilated instance mask, not on another dynamic object, and whose LiDAR
    return (if any) lies beyond the box. Their crossing of the padded box is free space.
    Returns per instance lists of ray_o, ray_d (object frame) and ray_t (M, 2).

    With `inside`, also returns the opposite evidence as a second list of the same layout:
    rays through pixels of the instance mask eroded by `mask_erode_px`. The mask is the visible
    part of the instance, so such a ray meets the object somewhere on its crossing of the box
    (grown by `inside_tol`), which is the interval stored in ray_t.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    per_ins = [dict(ray_o=[], ray_d=[], ray_t=[], ray_frame=[]) for _ in range(sizes.shape[0])]
    per_ins_in = [dict(ray_o=[], ray_d=[], ray_t=[], ray_frame=[]) for _ in range(sizes.shape[0])]
    corners_unit = torch.tensor([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], device=device, dtype=torch.float32)
    for fi in frames:
        for i in torch.where(fv[fi])[0].tolist():
            tid = true_ids[i]
            cam_ids = masks.cameras(tid, fi)
            if len(cam_ids) == 0:
                continue
            R, t = world_to_object(quats[fi, i], trans[fi, i])
            half = sizes[i] / 2 + box_pad
            corners_world = (corners_unit * half) @ R.T + t
            for cam_id in cam_ids:
                if cam_id not in pixel_source.camera_data:
                    continue
                cam = pixel_source.camera_data[cam_id]
                m = masks.get(tid, cam_id, fi, cam, device)
                if m is None:
                    continue
                uv_c, z_c, inside_c = project_points(corners_world, cam, fi, device)
                if (z_c <= 0.1).any():
                    continue  # box partly behind the camera; skip this view
                x0, y0 = uv_c.min(0).values.clamp_min(0).long()
                x1 = uv_c[:, 0].max().clamp_max(cam.WIDTH - 1).long()
                y1 = uv_c[:, 1].max().clamp_max(cam.HEIGHT - 1).long()
                if x1 <= x0 or y1 <= y0:
                    continue
                n = pixels_per_view
                u = torch.randint(int(x0), int(x1) + 1, (n,), generator=gen, device=device)
                v = torch.randint(int(y0), int(y1) + 1, (n,), generator=gen, device=device)
                outside = ~dilate(m, mask_dilate_px)[v, u]
                if hasattr(cam, "dynamic_masks") and cam.dynamic_masks is not None:
                    outside &= cam.dynamic_masks[fi].to(device)[v, u] < 0.5
                if inside:
                    on_obj = ~dilate(~m, mask_erode_px)[v, u]
                    if on_obj.any():
                        o_w, d_w, _ = pixel_rays(torch.stack([u[on_obj], v[on_obj]], dim=-1).float(), cam, fi, device)
                        o = ((o_w - t) @ R).expand(d_w.shape[0], 3)
                        d = d_w @ R
                        t_in, t_out = ray_box_intersect(o, d, sizes[i] / 2 + inside_tol)
                        t_in = t_in.clamp_min(0.0)
                        ok = t_out > t_in
                        per_ins_in[i]["ray_o"].append(o[ok]); per_ins_in[i]["ray_d"].append(d[ok])
                        per_ins_in[i]["ray_t"].append(torch.stack([t_in[ok], t_out[ok]], dim=-1))
                        per_ins_in[i]["ray_frame"].append(torch.full((int(ok.sum()),), fi, device=device, dtype=torch.long))
                if outside.sum() == 0:
                    continue
                u, v = u[outside], v[outside]
                uv = torch.stack([u, v], dim=-1).float()
                o_w, d_w, z_per_t = pixel_rays(uv, cam, fi, device)
                o = ((o_w - t) @ R).expand(d_w.shape[0], 3)
                d = d_w @ R
                t_in, t_out = ray_box_intersect(o, d, half)
                t_in = t_in.clamp_min(0.0)
                crosses = t_out > t_in
                if hasattr(cam, "lidar_depth_maps") and cam.lidar_depth_maps is not None:
                    ld = cam.lidar_depth_maps[fi].to(device)[v, u]
                    # a return closer than the box exit means an occluder (or the object): not free
                    crosses &= (ld <= 1e-3) | (ld > t_out * z_per_t + 0.5)
                if crosses.sum() == 0:
                    continue
                per_ins[i]["ray_o"].append(o[crosses])
                per_ins[i]["ray_d"].append(d[crosses])
                per_ins[i]["ray_t"].append(torch.stack([t_in[crosses], t_out[crosses]], dim=-1))
                per_ins[i]["ray_frame"].append(torch.full((int(crosses.sum()),), fi, device=device, dtype=torch.long))
    return (per_ins, per_ins_in) if inside else per_ins


@torch.no_grad()
def gather_camera_views(
    frames: List[int],
    quats: torch.Tensor,
    trans: torch.Tensor,
    fv: torch.Tensor,
    sizes: torch.Tensor,
    device,
    pixel_source,
    max_dist: float = 60.0,
) -> List[Dict[str, torch.Tensor]]:
    """
    Per instance, the camera views (one per frame and camera) in which its box is in front of
    the camera, closer than `max_dist` and at least partly inside the image, expressed in the
    object frame: pixel_h = A @ p_obj + b (z-depth is the third component), camera centre c,
    image size wh. Used to tell which parts of an instance's surface a training camera faces.
    """
    per_ins = [dict(A=[], b=[], c=[], wh=[]) for _ in range(sizes.shape[0])]
    corners_unit = torch.tensor([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], device=device, dtype=torch.float32)
    for fi in frames:
        active = torch.where(fv[fi])[0].tolist()
        if not active:
            continue
        for cam in pixel_source.camera_data.values():
            K = cam.intrinsics[fi].to(device).float()
            c2w = cam.cam_to_worlds[fi].to(device).float()
            w2c = torch.linalg.inv(c2w)
            Rc, tc = w2c[:3, :3], w2c[:3, 3]
            wh = torch.tensor([cam.WIDTH, cam.HEIGHT], device=device, dtype=torch.float32)
            for i in active:
                R, t = world_to_object(quats[fi, i], trans[fi, i])
                A = K @ Rc @ R
                b = K @ (Rc @ t + tc)
                if not (0.1 < b[2] < max_dist):
                    continue  # box centre behind the camera or too far to matter
                h = (corners_unit * sizes[i] / 2) @ A.T + b
                uv = h[:, :2] / h[:, 2:].clamp_min(1e-6)
                inside = (h[:, 2] > 0.1) & (uv >= 0).all(-1) & (uv < wh).all(-1)
                if not inside.any():
                    continue
                v = per_ins[i]
                v["A"].append(A); v["b"].append(b); v["c"].append((c2w[:3, 3] - t) @ R); v["wh"].append(wh)
    out = []
    for v in per_ins:
        n = len(v["A"])
        out.append(dict(
            A=torch.stack(v["A"]) if n else torch.zeros(0, 3, 3, device=device),
            b=torch.stack(v["b"]) if n else torch.zeros(0, 3, device=device),
            c=torch.stack(v["c"]) if n else torch.zeros(0, 3, device=device),
            wh=torch.stack(v["wh"]) if n else torch.zeros(0, 2, device=device),
        ))
    return out


@torch.no_grad()
def count_facing_views(p: torch.Tensor, n: torch.Tensor, views: Dict[str, torch.Tensor],
                       min_cos: float = 0.2, max_dist: float = 60.0, chunk: int = 64) -> torch.Tensor:
    """
    (M,) number of views in which the object-frame surface point p (outward unit normal n)
    projects inside the image, is closer than max_dist and faces the camera (cosine between
    the normal and the direction to the camera above min_cos). Occlusion by other objects is
    not tested.
    """
    count = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)
    V = views["A"].shape[0]
    for s in range(0, V, chunk):
        A, b, c, wh = (views[k][s:s + chunk] for k in ("A", "b", "c", "wh"))
        h = torch.einsum("vij,mj->vmi", A, p) + b[:, None]
        z = h[..., 2]
        uv = h[..., :2] / z.clamp_min(1e-6)[..., None]
        inside = (z > 0.1) & (z < max_dist) & (uv >= 0).all(-1) & (uv < wh[:, None]).all(-1)
        to_cam = c[:, None] - p[None]
        cos = (to_cam * n[None]).sum(-1) / to_cam.norm(dim=-1).clamp_min(1e-6)
        count += (inside & (cos > min_cos)).sum(0)
    return count


@torch.no_grad()
def nearest_distance(a: torch.Tensor, b: torch.Tensor, max_elems: int = 2 ** 25,
                     extent: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each row of a (Na, 3): distance to and index of the nearest row of b (Nb, 3). inf / 0 if
    b is empty. With `extent` (Nb,), each b is a ball of that radius and the distance is to its
    surface (negative inside).
    """
    dist = torch.full((a.shape[0],), float("inf"), device=a.device)
    idx = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    if b.shape[0] == 0 or a.shape[0] == 0:
        return dist, idx
    chunk = max(1, max_elems // b.shape[0])
    for s in range(0, a.shape[0], chunk):
        d = torch.cdist(a[s:s + chunk], b)
        if extent is not None:
            d = d - extent[None]
        d, j = d.min(dim=1)
        dist[s:s + chunk], idx[s:s + chunk] = d, j
    return dist, idx


@torch.no_grad()
def near_ray_segments(p: torch.Tensor, o: torch.Tensor, d: torch.Tensor, t: torch.Tensor, tol: float,
                      max_elems: int = 2 ** 25) -> torch.Tensor:
    """
    (Np,) bool: the point lies within `tol` of one of the segments o + d*s, s in [t[:, 0], t[:, 1]]
    (d unit). Used to find points that sit in space a ray is known to have crossed freely.
    """
    hit = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
    if p.shape[0] == 0 or o is None or o.shape[0] == 0:
        return hit
    # start every segment at its first point: origins are tens of metres away, and the
    # squared distances below would lose the centimetres in float32
    a = o + d * t[:, :1]
    length = (t[:, 1] - t[:, 0]).clamp_min(0)
    chunk = max(1, max_elems // p.shape[0])
    p2 = p.square().sum(-1)
    for k in range(0, a.shape[0], chunk):
        ak, dk = a[k:k + chunk], d[k:k + chunk]
        along = p @ dk.T - (ak * dk).sum(-1)[None]                       # (Np, R)
        dist2 = p2[:, None] + ak.square().sum(-1)[None] - 2 * p @ ak.T   # squared distance to the start
        perp2 = dist2 - along.square()
        hit |= ((along >= 0) & (along <= length[k:k + chunk][None]) & (perp2 <= tol * tol)).any(dim=1)
    return hit


def flank_counts(s: Dict[str, List[torch.Tensor]], half_y: float, band: float) -> Tuple[int, int]:
    """Returns on the +y and the -y flank of an instance (within `band` of the box side),
    each counted only from rays whose origin is on that same side of the object."""
    if len(s["surf_p"]) == 0:
        return 0, 0
    p, oy = torch.cat(s["surf_p"]), torch.cat(s["surf_oy"])
    return (int(((p[:, 1] > half_y - band) & (oy > 0)).sum()), int(((p[:, 1] < -half_y + band) & (oy < 0)).sum()))


def choose_withheld_sides(per_ins, sizes: torch.Tensor, mode: str, min_returns: int, band: float):
    """
    Which side of each instance to withhold (+1: rays from origins at y > 0, -1: y < 0,
    0: none). Only instances whose two flanks both have `min_returns` returns take part, so
    the withheld side has a reference and the kept side has data to mirror. 'auto' withholds
    the flank with fewer returns; 'pos' / 'neg' fix the side. Returns (sides (N,), counts).
    """
    sides = torch.zeros(len(per_ins), dtype=torch.long, device=sizes.device)
    counts = []
    for i, s in enumerate(per_ins):
        n_pos, n_neg = flank_counts(s, float(sizes[i, 1]) / 2, band)
        counts.append((n_pos, n_neg))
        if min(n_pos, n_neg) < min_returns:
            continue
        sides[i] = {"pos": 1, "neg": -1}.get(mode, 1 if n_pos <= n_neg else -1)
    return sides, counts


def split_by_side(s: Dict[str, List[torch.Tensor]], side: int, keep_withheld: bool) -> Dict[str, List[torch.Tensor]]:
    """Copy of an instance's ray/return lists with only the rays whose origin is on the
    withheld side (`keep_withheld`) or only the others. Works for LiDAR and carving dicts."""
    out = {k: [] for k in s}
    def pick(keys, origin_y_parts):
        for j, oy in enumerate(origin_y_parts):
            m = (oy * side > 0) == keep_withheld
            for k in keys:
                out[k].append(s[k][j][m])
    pick([k for k in ("ray_o", "ray_d", "ray_t", "ray_frame") if k in s], [o[:, 1] for o in s["ray_o"]])
    if "surf_oy" in s:
        pick(["surf_p", "surf_d", "surf_w", "surf_frame", "surf_oy"], s["surf_oy"])
    return out


def cat_cap(parts: List[torch.Tensor], cap: int, gen: torch.Generator) -> Optional[torch.Tensor]:
    """Concatenate and randomly subsample to `cap` rows (same generator state -> aligned subsets)."""
    if len(parts) == 0:
        return None
    x = torch.cat(parts, dim=0)
    if x.shape[0] > cap:
        keep = torch.randperm(x.shape[0], generator=gen, device=x.device)[:cap]
        x = x[keep]
    return x


def recover_instance_ds_ids(model, pixel_source) -> List[int]:
    """Dataset instance index of each instance of a node model (-1 if ambiguous). Uses the
    stored ids if the model has them, else matches the box size and the per-frame activity
    pattern, which come from the dataset at init."""
    ids = getattr(model, "instance_ds_ids", None)
    if ids:
        return list(ids)
    sizes = pixel_source.instances_size.to(model.device)
    fv = pixel_source.per_frame_instance_mask.to(model.device)
    out = []
    for i in range(model.instances_size.shape[0]):
        cand = torch.where((sizes - model.instances_size[i]).abs().sum(-1) < 1e-4)[0].tolist()
        if len(cand) > 1:
            # same box size: the activity pattern decides (SMPL nodes are active on a subset
            # of the box's frames, so take the best agreement rather than equality)
            agree = [int((fv[:, j] == model.instances_fv[:, i]).sum()) for j in cand]
            best = max(agree)
            cand = [j for j, a in zip(cand, agree) if a == best]
        out.append(int(cand[0]) if len(cand) == 1 else -1)
    return out


def modelled_instance_ids(models: Dict, pixel_source) -> List[int]:
    """Dataset instance indices of every instance that has a node in `models`."""
    out = []
    for model in models.values():
        if hasattr(model, "instances_size") and hasattr(model, "instances_fv"):
            out.extend(j for j in recover_instance_ds_ids(model, pixel_source) if j >= 0)
    return sorted(set(out))


def train_frames(dataset) -> List[int]:
    """Frame indices that are not held out for testing."""
    test = set(int(t) for t in dataset.test_timesteps) if dataset.test_timesteps is not None else set()
    return [f for f in range(dataset.frame_num) if f not in test]


def rotvec_to_rotmat(r: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vectors -> (..., 3, 3) rotation matrices (Rodrigues)."""
    a = r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = r / a
    K = torch.zeros(*r.shape[:-1], 3, 3, device=r.device, dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    s, c = torch.sin(a)[..., None], torch.cos(a)[..., None]
    eye = torch.eye(3, device=r.device, dtype=r.dtype).expand_as(K)
    return eye + s * K + (1 - c) * (K @ K)


def register_to_sdf(query_sdf, p: torch.Tensor, ids: torch.Tensor, pair: torch.Tensor, n_pairs: int,
                    iters: int = 60, lr: float = 2e-3, damping: float = 1e-3, robust: float = 0.1,
                    max_trans: float = 0.2, max_rot: float = 0.05):
    """
    Per group of points (`pair`: index in [0, n_pairs), e.g. one frame of one instance), the small rigid
    correction (rotation vector w, translation u, both (n_pairs, 3), object frame) that moves the group's
    points p (object frame) onto the zero level of the SDF: minimise mean(min(|f(R(w) p + u)|, robust))
    + damping (|u|^2 + |w|^2). The SDF is not changed (query_sdf must not need its gradients).
    Returns w, u and the mean |f| per group before and after (n_pairs,), clamped to max_trans / max_rot.
    To apply to a pose (R, t) with p = R^T (x - t):  R' = R R(w)^T,  t' = t - R' u.
    """
    w = torch.zeros(n_pairs, 3, device=p.device, requires_grad=True)
    u = torch.zeros(n_pairs, 3, device=p.device, requires_grad=True)
    cnt = torch.zeros(n_pairs, device=p.device).index_add_(0, pair, torch.ones_like(pair, dtype=torch.float)).clamp_min(1)
    def residual(w_, u_):
        q = (rotvec_to_rotmat(w_)[pair] @ p[..., None])[..., 0] + u_[pair]
        r = query_sdf(q, ids).abs().clamp(max=robust)
        return torch.zeros(n_pairs, device=p.device).index_add_(0, pair, r) / cnt
    with torch.no_grad():
        before = residual(w, u)
    opt = torch.optim.Adam([w, u], lr=lr)
    with torch.enable_grad():
        for _ in range(iters):
            opt.zero_grad()
            loss = residual(w, u).sum() + damping * (u.square().sum() + w.square().sum())
            loss.backward()
            opt.step()
            with torch.no_grad():
                u.data = u.data * (max_trans / u.data.norm(dim=-1, keepdim=True).clamp_min(max_trans))
                w.data = w.data * (max_rot / w.data.norm(dim=-1, keepdim=True).clamp_min(max_rot))
    with torch.no_grad():
        after = residual(w, u)
    return w.detach(), u.detach(), before, after
