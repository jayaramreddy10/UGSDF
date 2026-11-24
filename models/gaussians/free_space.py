"""
Ray-cast free-space evidence for the static background.

Every LiDAR ray says two things: the segment before its return is empty, and there is a
surface at the return. Casting all rays of the (training) frames through a world-frame
voxel grid gives, per voxel, the ray length that crossed it (`traversal`, in voxel
units) and the number of static returns in it (`hits`). A voxel is

  * solid   if it or a neighbour within `dilate` voxels holds a static return,
  * free    if rays crossed it for at least `min_traversal` voxel lengths and it is not solid,
  * unknown otherwise (never observed, or too little evidence).

Free voxels are free of *static* matter at all times, so background gaussians centred
in them are floaters: their opacity is penalised and they are culled. Returns on the
boxes in `exclude_boxes` (objects that have their own node) are not static and do not
make a voxel solid; the ray segments in front of them still count as free.
"""
from typing import Dict, List, Optional, Sequence
import logging

import torch

logger = logging.getLogger()


class FreeSpaceGrid:
    def __init__(self, origin: torch.Tensor, voxel: float, free: torch.Tensor, solid: torch.Tensor):
        self.origin = origin          # (3,) world position of the grid's min corner
        self.voxel = float(voxel)     # metres
        self.free = free              # (X, Y, Z) bool
        self.solid = solid            # (X, Y, Z) bool (already dilated)
        self.dims = torch.tensor(free.shape, device=free.device)

    # ------------------------------------------------------------------ queries
    def _index(self, xyz: torch.Tensor):
        ijk = torch.floor((xyz - self.origin) / self.voxel).long()
        inside = ((ijk >= 0) & (ijk < self.dims)).all(dim=-1)
        ijk = ijk.clamp_min(0).minimum(self.dims - 1)
        return ijk, inside

    @torch.no_grad()
    def is_free(self, xyz: torch.Tensor) -> torch.Tensor:
        """(N,) bool: the point lies in a voxel the LiDAR saw through and never hit."""
        ijk, inside = self._index(xyz)
        return self.free[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & inside

    @torch.no_grad()
    def is_observed(self, xyz: torch.Tensor) -> torch.Tensor:
        """(N,) bool: the point lies in a free or solid voxel (anything but unknown)."""
        ijk, inside = self._index(xyz)
        return (self.free | self.solid)[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & inside

    def stats(self) -> Dict[str, float]:
        n = float(self.free.numel())
        return {"voxels": n, "free_frac": self.free.sum().item() / n, "solid_frac": self.solid.sum().item() / n}

    # ------------------------------------------------------------------ construction
    @classmethod
    @torch.no_grad()
    def build(
        cls,
        lidar_source,
        frames: Sequence[int],
        device,
        voxel: float = 0.25,
        max_range: float = 75.0,
        min_traversal: float = 5.0,
        min_hits: int = 1,
        dilate: int = 1,
        end_margin: float = 0.25,
        exclude_boxes: Optional[Dict[str, torch.Tensor]] = None,
        exclude_pad: float = 0.2,
        max_voxels: int = 200_000_000,
        ray_chunk: int = 60_000,
    ) -> "FreeSpaceGrid":
        """
        frames: lidar timesteps to cast. exclude_boxes: dict with 'pose' (F, N, 4, 4) object
        to world, 'size' (N, 3), 'active' (F, N) bool and 'ids' (list of instance indices):
        returns inside these boxes (padded) are not static.
        """
        frames = [int(f) for f in frames]
        # bounds from the returns of the frames (clipped to max_range by construction below)
        lo = torch.full((3,), float("inf"), device=device)
        hi = torch.full((3,), float("-inf"), device=device)
        per_frame = []
        for fi in frames:
            rays = lidar_source.get_lidar_rays(fi)
            o = rays["lidar_origins"].to(device).float()
            d = rays["lidar_viewdirs"].to(device).float()
            r = rays["lidar_ranges"].reshape(-1).to(device).float()
            if o.shape[0] == 0:
                continue
            static = torch.ones_like(r, dtype=torch.bool)
            if exclude_boxes is not None and len(exclude_boxes["ids"]) > 0:
                p = o + d * r[:, None]
                for j in exclude_boxes["ids"]:
                    if not bool(exclude_boxes["active"][fi, j]):
                        continue
                    o2w = exclude_boxes["pose"][fi, j].to(device).float()
                    half = exclude_boxes["size"][j].to(device).float() / 2 + exclude_pad
                    p_obj = (p - o2w[:3, 3]) @ o2w[:3, :3]
                    static &= ~(p_obj.abs() <= half).all(dim=-1)
            end = r.clamp_max(max_range)
            pts = o + d * end[:, None]
            lo = torch.minimum(lo, torch.minimum(pts.min(0).values, o.min(0).values))
            hi = torch.maximum(hi, torch.maximum(pts.max(0).values, o.max(0).values))
            per_frame.append((o, d, r, static))
        if not per_frame:
            raise ValueError("FreeSpaceGrid: no lidar rays in the given frames")

        lo, hi = lo - voxel, hi + voxel
        while True:
            dims = torch.ceil((hi - lo) / voxel).long()
            if int(dims.prod()) <= max_voxels:
                break
            voxel *= 1.25
        X, Y, Z = [int(v) for v in dims]
        traversal = torch.zeros(X * Y * Z, device=device)
        hits = torch.zeros(X * Y * Z, device=device)

        def flat(p):
            ijk = torch.floor((p - lo) / voxel).long()
            ok = ((ijk >= 0) & (ijk < dims)).all(dim=-1)
            return (ijk[:, 0] * Y + ijk[:, 1]) * Z + ijk[:, 2], ok

        n_steps = int(max_range / voxel) + 1
        gen = torch.Generator(device=device).manual_seed(0)
        n_rays = 0
        for o, d, r, static in per_frame:
            n_rays += o.shape[0]
            # static returns inside the range make their voxel solid
            ret = static & (r <= max_range)
            idx, ok = flat(o[ret] + d[ret] * r[ret, None])
            hits.index_add_(0, idx[ok], torch.ones(int(ok.sum()), device=device))
            # march each ray in voxel-sized steps (jittered) up to just before its return;
            # each sample adds one voxel length of traversal to the voxel it falls in
            t_end = (r - end_margin).clamp_max(max_range)
            for s in range(0, o.shape[0], ray_chunk):
                oc, dc, tc = o[s:s + ray_chunk], d[s:s + ray_chunk], t_end[s:s + ray_chunk]
                jitter = torch.rand(oc.shape[0], 1, generator=gen, device=device)
                t = (torch.arange(n_steps, device=device)[None] + jitter) * voxel
                valid = t < tc[:, None]
                p = oc[:, None] + dc[:, None] * t[..., None]
                idx, ok = flat(p[valid])
                traversal.index_add_(0, idx[ok], torch.ones(int(ok.sum()), device=device))

        solid = (hits >= min_hits).reshape(X, Y, Z)
        if dilate > 0:
            k = 2 * dilate + 1
            solid = torch.nn.functional.max_pool3d(solid[None, None].float(), k, stride=1, padding=dilate)[0, 0] > 0.5
        free = (traversal >= min_traversal).reshape(X, Y, Z) & ~solid
        grid = cls(lo, voxel, free, solid)
        st = grid.stats()
        logger.info(
            f"FreeSpaceGrid: {len(per_frame)} frames, {n_rays} rays, {X}x{Y}x{Z} voxels of {voxel:.2f} m, "
            f"free {st['free_frac']*100:.1f}%, solid {st['solid_frac']*100:.1f}%"
        )
        return grid
