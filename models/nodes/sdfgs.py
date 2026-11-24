"""
Rigid instance nodes with a per-instance SDF in the object frame.

The Gaussians are the ones from RigidNodes (object-frame means, per-frame box
poses). On top of them each instance owns a small Fourier-feature MLP that
predicts a signed distance in metres in the same object frame. The two are
coupled both ways:

  * LiDAR -> SDF: returns inside the box are zero-level samples, the ray
    segment before a return is free space (hinge), a thin band behind a
    return is interior (hinge), plus an Eikonal term.
  * Gaussians -> SDF: Gaussian centres act as extra zero-level samples,
    weighted by opacity.
  * SDF -> Gaussians: centres are pulled onto the zero level (|sdf| or sdf^2), Gaussians
    farther than `prune.tau` from the surface are culled, and (`loss.align_w`) the
    thinnest axis of a Gaussian near the surface is turned along the SDF normal.
  * LiDAR -> SDF orientation (`loss.facing_w`): the SDF normal at a return faces the sensor.
  * Masks -> SDF: camera rays outside the annotator mask see past the object (carving);
    with `loss.inside_w`, rays through the mask must meet the object, i.e. reach a
    negative distance somewhere on their crossing of the box. This is the inside evidence
    LiDAR does not give: a return proves the space before it empty, not what is behind it.
  * SDF -> new Gaussians (`sdf.seed`): at intervals, Gaussians are added on the zero
    level where the surface has evidence (a LiDAR return nearby), a training camera
    faces it, and no Gaussian of the instance covers it.
  * Monocular depth -> SDF (`sdf.mono`): a surface lifted from a monocular depth model
    (tools/mono_depth/lift.py) supervises the SDF where LiDAR is lacking: its points are
    zero-level samples, its normals orient the gradient, and a thin band along the normal
    is free in front of the surface and interior behind it. LiDAR wins where it has data:
    lifted points near a return, or in space a LiDAR or carving ray crossed, are dropped.
  * Optional left-right symmetry (`sdf.symmetry`): the supervision mirrored about the
    object's x-z plane is applied at a reduced weight, so a side that was never
    observed inherits the surface and the free space of the side that was.

Supervision rays are gathered once at init from the lidar source
(`init_sdf_supervision`) and kept on the GPU in the object frame.
"""
from typing import Dict, List, Tuple
import logging
import os
import struct

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.func import functional_call
from omegaconf import OmegaConf

from models.gaussians.basics import *
from models.nodes.rigid import RigidNodes
from models.nodes.sdf_supervision import (
    AnnotatorMasks, gather_instance_lidar, gather_carving_rays, cat_cap, ray_box_intersect as _ray_box_intersect,
    choose_withheld_sides, split_by_side, gather_camera_views, count_facing_views, nearest_distance,
    near_ray_segments, repose, register_to_sdf, rotvec_to_rotmat,
)

logger = logging.getLogger()


def box_sdf(p: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Exact signed distance to the axis-aligned box [-half, half]."""
    q = p.abs() - half
    outside = q.clamp_min(0.0).norm(dim=-1)
    inside = q.max(dim=-1).values.clamp_max(0.0)
    return outside + inside


class InstanceSDF(nn.Module):
    """
    Object-frame xyz (metres) -> signed distance (metres), as a Fourier-feature MLP
    residual over the analytic SDF of the (slightly shrunk) annotated box. The box is
    the prior: unobserved regions default to solid, and the LiDAR carves the residual.

    With `box_clamp` (metres), the field is intersected with the annotated box grown by
    that tolerance: sdf = max(prior + residual, box_sdf(box + box_clamp)). No surface can
    then exist outside the box, exactly, instead of only being discouraged by a loss.

    With `c2f_end` > `c2f_start` (steps) the Fourier bands fade in one after the other between those
    steps (BARF-style), so the field is fitted coarse to fine. Outside training all bands are on.
    """

    def __init__(
        self,
        half_size: torch.Tensor,
        hidden_dim: int = 128,
        layers: int = 4,
        frequencies: int = 4,
        pad: float = 1.3,
        box_shrink: float = 0.05,
        box_clamp: float = None,
        c2f_start: int = 0,
        c2f_end: int = 0,
    ):
        super().__init__()
        self.box_clamp = box_clamp
        self.c2f_start, self.c2f_end = c2f_start, c2f_end
        self.register_buffer("band_weights", torch.ones(frequencies), persistent=False)
        # not persistent: checkpoints written before the option keep loading
        self.register_buffer("clamp_half", half_size.float() + (box_clamp or 0.0), persistent=False)
        # the box maps to roughly [-1/pad, 1/pad]^3 so the encoding sees the margin around it
        self.register_buffer("scale", half_size.float().clamp_min(0.1) * pad)
        self.register_buffer("prior_half", (half_size.float() - box_shrink).clamp_min(0.05))
        self.register_buffer("frequency_bands", 2 ** torch.arange(frequencies, dtype=torch.float32))
        width = 3 * (1 + 2 * frequencies)
        blocks = []
        for _ in range(layers - 1):
            blocks.extend((nn.Linear(width, hidden_dim), nn.Softplus(beta=20)))
            width = hidden_dim
        blocks.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*blocks)
        # the residual starts at ~0 so the field begins as the box prior
        nn.init.normal_(self.network[-1].weight, mean=0, std=1e-4)
        nn.init.zeros_(self.network[-1].bias)

    @torch.no_grad()
    def set_step(self, step: int) -> None:
        """Weights of the Fourier bands at this training step (coarse-to-fine)."""
        if self.c2f_end <= self.c2f_start:
            return
        L = self.band_weights.shape[0]
        alpha = L * min(max((step - self.c2f_start) / (self.c2f_end - self.c2f_start), 0.0), 1.0)
        k = torch.arange(L, device=self.band_weights.device, dtype=torch.float32)
        self.band_weights.copy_((1 - torch.cos(torch.pi * (alpha - k).clamp(0, 1))) / 2)

    def forward(self, xyz_obj: torch.Tensor) -> torch.Tensor:
        x = xyz_obj / self.scale
        angle = x.unsqueeze(-1) * self.frequency_bands * torch.pi
        encoded = torch.cat((x, (angle.sin() * self.band_weights).flatten(-2), (angle.cos() * self.band_weights).flatten(-2)), dim=-1)
        sdf = box_sdf(xyz_obj, self.prior_half) + self.network(encoded).squeeze(-1)
        if self.box_clamp is not None:
            sdf = torch.maximum(sdf, box_sdf(xyz_obj, self.clamp_half))
        return sdf


def _write_ply(path: str, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray = None) -> None:
    with open(path, "wb") as f:
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}",
                  "property float x", "property float y", "property float z"]
        if normals is not None:
            header += ["property float nx", "property float ny", "property float nz"]
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header"]
        f.write(("\n".join(header) + "\n").encode())
        v = vertices if normals is None else np.concatenate([vertices, normals], axis=1)
        f.write(v.astype("<f4").tobytes())
        face_dtype = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
        packed = np.empty(len(faces), dtype=face_dtype)
        packed["n"] = 3
        packed["idx"] = faces
        f.write(packed.tobytes())


class SDFGSNodes(RigidNodes):
    def __init__(self, sdf: OmegaConf = None, **kwargs):
        super().__init__(**kwargs)
        self.sdf_cfg = sdf if sdf is not None else OmegaConf.create({})
        self.sdf_nets: nn.ModuleList = nn.ModuleList()
        # raw per-instance offset (metres) of the symmetry plane along the object's y axis
        self.sdf_sym_offset = Parameter(torch.zeros(0, device=self.device))
        # supervision, filled by init_sdf_supervision (all in the object frame)
        self.sup_ray_o = None     # (M, 3) ray origins
        self.sup_ray_d = None     # (M, 3) unit directions
        self.sup_ray_t = None     # (M, 2) free interval [t0, t1] along the ray
        self.sup_ray_id = None    # (M,) instance id in model
        self.sup_surf_p = None    # (K, 3) returns inside the box
        self.sup_surf_d = None    # (K, 3) direction of the ray that produced them
        self.sup_surf_w = None    # (K,) 1 if mask-verified, else the unverified weight
        self.sup_surf_id = None   # (K,)
        self.sup_carve_o = None   # camera rays that see past the object (object frame)
        self.sup_carve_d = None
        self.sup_carve_t = None   # (M, 2) crossing of the padded box
        self.sup_carve_id = None
        self.sup_inside_o = None  # camera rays through the instance mask (object frame)
        self.sup_inside_d = None
        self.sup_inside_t = None  # (M, 2) crossing of the box, on which the ray must meet the object
        self.sup_inside_id = None
        self.sup_mono_p = None    # (L, 3) surface lifted from monocular depth, where LiDAR is lacking
        self.sup_mono_n = None    # (L, 3) its unit normals, pointing out of the object
        self.sup_mono_w = None    # (L,) weight in (0, 1], rising with the distance to the nearest return
        self.sup_mono_id = None   # (L,)
        self.instance_ds_ids: List[int] = []  # dataset instance index per model instance
        # per instance the training camera views in the object frame (sdf.seed.require_camera)
        self.sdf_cam_views: List[Dict[str, torch.Tensor]] = None
        self.sdf_seed_log: List[Dict] = []    # one entry per seeding event
        # experiment (sdf.supervision.withhold_side): per instance the side whose rays are kept
        # out of the supervision (+1: origins at object y > 0, -1: y < 0, 0: nothing withheld)
        self.sdf_withheld_side = torch.zeros(0, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ config helpers
    def _net_cfg(self) -> dict:
        net = self.sdf_cfg.get("net", {})
        return dict(
            hidden_dim=net.get("hidden_dim", 128),
            layers=net.get("layers", 4),
            frequencies=net.get("frequencies", 4),
            pad=net.get("pad", 1.3),
            box_shrink=net.get("box_shrink", 0.05),
            box_clamp=net.get("box_clamp", None),
            c2f_start=net.get("c2f_start", 0),
            c2f_end=net.get("c2f_end", 0),
        )

    @property
    def mesh_resolution(self) -> int:
        """Marching-cubes cells along the longest side of the padded box."""
        return int(self.sdf_cfg.get("mesh", {}).get("resolution", 192))

    def _loss_cfg(self, key: str, default):
        return self.sdf_cfg.get("loss", {}).get(key, default)

    def _seed_cfg(self, key: str, default):
        return self.sdf_cfg.get("seed", {}).get(key, default)

    def _mono_cfg(self, key: str, default):
        return self.sdf_cfg.get("mono", {}).get(key, default)

    def _sym_cfg(self, key: str, default):
        return self.sdf_cfg.get("symmetry", {}).get(key, default)

    @property
    def symmetry_offset(self) -> torch.Tensor:
        """(num_instances,) y of each instance's symmetry plane, bounded by symmetry.max_offset."""
        bound = self._sym_cfg("max_offset", 0.1)
        if not self._sym_cfg("learn_plane", False) or bound <= 0:
            return torch.zeros_like(self.sdf_sym_offset)
        return bound * torch.tanh(self.sdf_sym_offset / bound)

    def mirror_points(self, xyz: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Reflect object-frame points about their instance's symmetry plane (y = offset)."""
        y = 2 * self.symmetry_offset[ids] - xyz[:, 1]
        return torch.stack([xyz[:, 0], y, xyz[:, 2]], dim=-1)

    @staticmethod
    def mirror_dirs(d: torch.Tensor) -> torch.Tensor:
        return d * d.new_tensor([1.0, -1.0, 1.0])

    # ------------------------------------------------------------------ construction
    def _build_sdf_nets(self) -> None:
        cfg = self._net_cfg()
        self.sdf_nets = nn.ModuleList(
            [InstanceSDF(self.instances_size[i] / 2, **cfg) for i in range(self.num_instances)]
        ).to(self.device)
        self.sdf_sym_offset = Parameter(torch.zeros(self.num_instances, device=self.device))
        self.sdf_withheld_side = torch.zeros(self.num_instances, dtype=torch.long, device=self.device)

    def create_from_pcd(self, instance_pts_dict: Dict[str, torch.Tensor]) -> None:
        super().create_from_pcd(instance_pts_dict)
        self.instance_ds_ids = [int(k) for k in instance_pts_dict.keys()]
        self._build_sdf_nets()
        logger.info(f"{self.class_prefix} built {self.num_instances} instance SDFs")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups[self.class_prefix + "sdf"] = list(self.sdf_nets.parameters())
        if self._sym_cfg("enabled", False) and self._sym_cfg("learn_plane", False):
            param_groups[self.class_prefix + "sdf"].append(self.sdf_sym_offset)
        return param_groups

    def state_dict(self) -> Dict:
        sd = super().state_dict()
        sd["instance_ds_ids"] = torch.tensor(self.instance_ds_ids, dtype=torch.long)
        sd["sdf_withheld_side"] = self.sdf_withheld_side
        return sd

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        # nets must exist with the right count before the parameters are loaded
        self.instances_size = state_dict["instances_size"]
        self.instances_fv = state_dict["instances_fv"]
        ids = state_dict.pop("instance_ds_ids", None)
        self.instance_ds_ids = ids.tolist() if ids is not None else []
        withheld = state_dict.pop("sdf_withheld_side", None)
        self._build_sdf_nets()
        if withheld is not None:
            self.sdf_withheld_side = withheld.to(self.device).long()
        if "sdf_sym_offset" not in state_dict:  # checkpoints from before the symmetry option
            state_dict["sdf_sym_offset"] = torch.zeros(self.instances_size.shape[0], device=self.device)
        return super().load_state_dict(state_dict, **kwargs)

    # ------------------------------------------------------------------ supervision
    @torch.no_grad()
    def init_sdf_supervision(self, lidar_source, pixel_source=None, masks_root: str = None, frames: List[int] = None,
                             mono_surface: Dict[int, Dict[str, torch.Tensor]] = None) -> None:
        """
        Collect, per instance, the LiDAR rays that cross its padded box and the returns on it
        (object frame of each frame). With `pixel_source` and `masks_root`, returns are
        verified against the annotator masks and camera rays that see past the object are
        added as carving rays. `frames` defaults to every frame (training uses all of them).
        With `supervision.withhold_side` (auto | pos | neg), every ray whose origin is on one
        side of a two-sided instance is dropped; tools/eval.py scores the SDF against them.
        `mono_surface` ({dataset instance index: dict(pts, normals)}, object frame, normals
        pointing out) is a surface lifted from monocular depth; see `_init_mono_supervision`.
        """
        sup = self.sdf_cfg.get("supervision", {})
        max_rays = sup.get("max_rays_per_instance", 200_000)
        max_surf = sup.get("max_surface_per_instance", 100_000)
        max_carve = sup.get("max_carve_rays_per_instance", 200_000)
        frames = list(range(self.num_frames)) if frames is None else frames
        quats = self.quat_act(self.instances_quats.data)
        trans = self.instances_trans.data

        masks, true_ids = None, None
        if sup.get("use_masks", True) and masks_root is not None and pixel_source is not None and self.instance_ds_ids:
            cam0 = next(iter(pixel_source.camera_data.values()))
            masks = AnnotatorMasks(masks_root, start_timestep=cam0.start_timestep)
            true_ids = [int(pixel_source.instances_true_id[i].item()) for i in self.instance_ds_ids]
            logger.info(f"{self.class_prefix} mask-verified SDF supervision from {masks_root}")

        per_ins = gather_instance_lidar(
            lidar_source, frames, quats, trans, self.instances_fv, self.instances_size, self.device,
            box_pad=sup.get("box_pad", 0.3), ground_margin=sup.get("ground_margin", 0.12),
            surface_tol=sup.get("surface_tol", 0.02), free_margin=self._loss_cfg("free_margin", 0.05),
            masks=masks, true_ids=true_ids, pixel_source=pixel_source,
            mask_tol_px=sup.get("mask_tol_px", 2), unverified_w=sup.get("unverified_w", 0.35),
        )
        carve, inside = None, None
        want_inside = self._loss_cfg("inside_w", 0.0) > 0
        if masks is not None and (sup.get("carve", True) or want_inside):
            carve = gather_carving_rays(
                frames, quats, trans, self.instances_fv, self.instances_size, self.device, masks, true_ids, pixel_source,
                box_pad=sup.get("box_pad", 0.3), mask_dilate_px=sup.get("carve_dilate_px", 8),
                pixels_per_view=sup.get("carve_pixels_per_view", 1500),
                inside=want_inside, mask_erode_px=sup.get("inside_erode_px", 4), inside_tol=sup.get("surface_tol", 0.02),
            )
            if want_inside:
                carve, inside = carve
            if not sup.get("carve", True):
                carve = None

        self.sdf_cam_views = None
        if pixel_source is not None and self._seed_cfg("enabled", False):
            self.sdf_cam_views = gather_camera_views(
                frames, quats, trans, self.instances_fv, self.instances_size, self.device, pixel_source,
                max_dist=self._seed_cfg("camera_max_dist", 60.0))
            logger.info(f"{self.class_prefix} camera views per instance for SDF seeding: "
                        f"{[int(v['A'].shape[0]) for v in self.sdf_cam_views]}")

        # experiment: leave one side of each two-sided instance out of the supervision, so the
        # returns from that side can score what the SDF does where it has no data
        mode = sup.get("withhold_side", None)
        self.sdf_withheld_side = torch.zeros(self.num_instances, dtype=torch.long, device=self.device)
        if mode:
            self.sdf_withheld_side, counts = choose_withheld_sides(
                per_ins, self.instances_size, str(mode), sup.get("withhold_min_returns", 300), sup.get("withhold_band", 0.25))
            for i, (side, (n_pos, n_neg)) in enumerate(zip(self.sdf_withheld_side.tolist(), counts)):
                logger.info(f"{self.class_prefix} instance {i}: flank returns +y {n_pos}, -y {n_neg}; "
                            + ("nothing withheld" if side == 0 else f"withholding rays from the {'+y' if side > 0 else '-y'} side"))
                if side != 0:
                    per_ins[i] = split_by_side(per_ins[i], side, keep_withheld=False)
                    if carve is not None:
                        carve[i] = split_by_side(carve[i], side, keep_withheld=False)
                    if inside is not None:
                        inside[i] = split_by_side(inside[i], side, keep_withheld=False)

        gen = torch.Generator(device=self.device)
        pools = dict(ray=[[], [], [], [], []], surf=[[], [], [], [], []], carve=[[], [], [], [], []], inside=[[], [], [], [], []])
        for i, s in enumerate(per_ins):
            def take(parts, cap, seed):
                gen.manual_seed(seed)
                return cat_cap(parts, cap, gen)
            o, d, t = take(s["ray_o"], max_rays, 0), take(s["ray_d"], max_rays, 0), take(s["ray_t"], max_rays, 0)
            p, pd, pw = take(s["surf_p"], max_surf, 1), take(s["surf_d"], max_surf, 1), take(s["surf_w"], max_surf, 1)
            rf, pf = take(s["ray_frame"], max_rays, 0), take(s["surf_frame"], max_surf, 1)
            n_rays = 0 if o is None else o.shape[0]
            n_surf = 0 if p is None else p.shape[0]
            n_ver = 0 if p is None else int((pw >= 1.0).sum())
            msg = f"{self.class_prefix} instance {i}: {n_rays} free-space rays, {n_surf} surface returns"
            if masks is not None:
                msg += f" ({n_ver} mask-verified)"
            if n_rays:
                pools["ray"][0].append(o); pools["ray"][1].append(d); pools["ray"][2].append(t)
                pools["ray"][3].append(torch.full((n_rays,), i, dtype=torch.long, device=self.device)); pools["ray"][4].append(rf)
            if n_surf:
                pools["surf"][0].append(p); pools["surf"][1].append(pd); pools["surf"][2].append(pw)
                pools["surf"][3].append(torch.full((n_surf,), i, dtype=torch.long, device=self.device)); pools["surf"][4].append(pf)
            if carve is not None:
                c = carve[i]
                co, cd, ct = take(c["ray_o"], max_carve, 2), take(c["ray_d"], max_carve, 2), take(c["ray_t"], max_carve, 2)
                cf = take(c["ray_frame"], max_carve, 2)
                n_carve = 0 if co is None else co.shape[0]
                msg += f", {n_carve} carving rays"
                if n_carve:
                    pools["carve"][0].append(co); pools["carve"][1].append(cd); pools["carve"][2].append(ct)
                    pools["carve"][3].append(torch.full((n_carve,), i, dtype=torch.long, device=self.device)); pools["carve"][4].append(cf)
            if inside is not None:
                c = inside[i]
                io, idr, it = take(c["ray_o"], max_carve, 4), take(c["ray_d"], max_carve, 4), take(c["ray_t"], max_carve, 4)
                inf_ = take(c["ray_frame"], max_carve, 4)
                n_in = 0 if io is None else io.shape[0]
                msg += f", {n_in} mask rays"
                if n_in:
                    pools["inside"][0].append(io); pools["inside"][1].append(idr); pools["inside"][2].append(it)
                    pools["inside"][3].append(torch.full((n_in,), i, dtype=torch.long, device=self.device)); pools["inside"][4].append(inf_)
            logger.info(msg)
        self.sup_inside_o = self.sup_inside_d = self.sup_inside_t = self.sup_inside_id = None
        frames_of = {}
        if pools["inside"][0]:
            self.sup_inside_o, self.sup_inside_d, self.sup_inside_t, self.sup_inside_id, frames_of["inside"] = [torch.cat(x) for x in pools["inside"]]
        if pools["ray"][0]:
            self.sup_ray_o, self.sup_ray_d, self.sup_ray_t, self.sup_ray_id, frames_of["ray"] = [torch.cat(x) for x in pools["ray"]]
        if pools["surf"][0]:
            self.sup_surf_p, self.sup_surf_d, self.sup_surf_w, self.sup_surf_id, frames_of["surf"] = [torch.cat(x) for x in pools["surf"]]
        if pools["carve"][0]:
            self.sup_carve_o, self.sup_carve_d, self.sup_carve_t, self.sup_carve_id, frames_of["carve"] = [torch.cat(x) for x in pools["carve"]]
        # the object-frame data above was placed with the poses of this moment; with
        # supervision.follow_pose it is re-placed with the refined poses as training updates them
        self._sup_follow = None
        if sup.get("follow_pose", False):
            R0 = quat_to_rotmat(quats.reshape(-1, 4)).reshape(*quats.shape[:2], 3, 3).clone()
            self._sup_follow = dict(R0=R0, t0=trans.clone(), frames=frames_of, last_step=-1,
                                    orig={k: (getattr(self, f"sup_{k}_{a}").clone(), getattr(self, f"sup_{k}_{b}").clone())
                                          for k, (a, b) in dict(ray=("o", "d"), surf=("p", "d"), carve=("o", "d"), inside=("o", "d")).items()
                                          if k in frames_of})
            logger.info(f"{self.class_prefix} SDF supervision follows the refined instance poses "
                        f"(every {sup.get('follow_pose_every', 100)} steps)")
        self._init_mono_supervision(mono_surface)

    def register_poses_to_sdf(self, force: bool = False) -> Dict[str, float]:
        """
        sdf.register: every `every` steps from `start`, register each training frame's LiDAR returns of each
        instance to the current SDF and move that frame's pose by `step` times the correction (clamped to
        max_trans metres / max_rot_deg degrees). The SDF is the shape of all frames together, so this is
        tracking against the learned shape; the Gaussians move with the pose. Needs supervision.follow_pose
        (the returns are re-placed with the new poses right after). Returns summary numbers.
        """
        R_cfg = self.sdf_cfg.get("register", {})
        f = getattr(self, "_sup_follow", None)
        if not R_cfg.get("enabled", False) or f is None or self.sup_surf_p is None:
            return {}
        start, every = R_cfg.get("start", 5000), R_cfg.get("every", 1000)
        if not force and (self.step < start or (self.step - start) % every != 0 or f.get("last_reg") == self.step):
            return {}
        f["last_reg"] = self.step
        self.follow_refined_poses(force=True)
        fr, ids = f["frames"]["surf"], self.sup_surf_id
        keep = self.sup_surf_w >= 1.0                                     # mask-verified returns only
        if not keep.any():                                                # no masks: all returns
            keep = torch.ones_like(keep)
        key = fr * self.num_instances + ids
        uk, pair = torch.unique(key[keep], return_inverse=True)
        cnt = torch.bincount(pair)
        good = cnt >= R_cfg.get("min_returns", 30)
        sel = good[pair]
        if not sel.any():
            return {}
        uk2, pair2 = torch.unique(key[keep][sel], return_inverse=True)
        p, pid = self.sup_surf_p[keep][sel], ids[keep][sel]
        with torch.no_grad():
            max_rot = float(np.radians(R_cfg.get("max_rot_deg", 2.0)))
            w, u, before, after = register_to_sdf(
                lambda x, i: self.query_sdf(x, i, detach_params=True), p, pid, pair2, len(uk2),
                iters=R_cfg.get("iters", 50), max_trans=R_cfg.get("max_trans", 0.1), max_rot=max_rot)
            frames, inst = uk2 // self.num_instances, uk2 % self.num_instances
            drift = torch.zeros(1, device=self.device)
            if R_cfg.get("anchor", False):      # shape and poses can shift together: keep each instance's mean correction at zero
                n = torch.bincount(inst, minlength=self.num_instances).clamp_min(1)[:, None]
                mean_u = torch.zeros(self.num_instances, 3, device=self.device).index_add_(0, inst, u) / n
                mean_w = torch.zeros(self.num_instances, 3, device=self.device).index_add_(0, inst, w) / n
                u, w = u - mean_u[inst], w - mean_w[inst]
                drift = mean_u[inst.unique()].norm(dim=-1)
            a = R_cfg.get("step", 0.5)
            w, u = w * a, u * a
            q = self.quat_act(self.instances_quats.data[frames, inst])
            R = quat_to_rotmat(q)
            Rn = R @ rotvec_to_rotmat(w).transpose(-1, -2)
            self.instances_quats.data[frames, inst] = matrix_to_quaternion(Rn)
            self.instances_trans.data[frames, inst] = self.instances_trans.data[frames, inst] - (Rn @ u[..., None])[..., 0]
        self.follow_refined_poses(force=True)
        out = dict(pairs=int(len(uk2)), sdf_before=float(before.mean()), sdf_after=float(after.mean()),
                   trans_median=float(u.norm(dim=-1).median()), rot_median_deg=float(np.degrees(w.norm(dim=-1).median().item())),
                   anchor_removed=float(drift.mean()))
        logger.info(f"{self.class_prefix} step {self.step}: registered {out['pairs']} frame poses to the SDF, mean |sdf| "
                    f"{out['sdf_before']:.4f} -> {out['sdf_after']:.4f} m, applied median {out['trans_median'] * 100:.2f} cm / "
                    f"{out['rot_median_deg']:.3f} deg, anchor removed {out['anchor_removed'] * 100:.2f} cm")
        return out

    @torch.no_grad()
    def follow_refined_poses(self, force: bool = False) -> None:
        """Re-express the stored object-frame rays and returns with the current instance poses (see
        supervision.follow_pose). The poses are detached: the SDF terms do not move the poses."""
        f = getattr(self, "_sup_follow", None)
        if f is None:
            return
        every = self.sdf_cfg.get("supervision", {}).get("follow_pose_every", 100)
        if not force and f["last_step"] >= 0 and self.step - f["last_step"] < every:
            return
        f["last_step"] = self.step
        quats = self.quat_act(self.instances_quats.data)
        R1 = quat_to_rotmat(quats.reshape(-1, 4)).reshape(*quats.shape[:2], 3, 3)
        t1 = self.instances_trans.data
        names = dict(ray=("o", "d"), surf=("p", "d"), carve=("o", "d"), inside=("o", "d"))
        for k, (x0, d0) in f["orig"].items():
            fr, ids = f["frames"][k], getattr(self, f"sup_{k}_id")
            a, b = names[k]
            setattr(self, f"sup_{k}_{a}", repose(x0, fr, ids, f["R0"], f["t0"], R1, t1))
            setattr(self, f"sup_{k}_{b}", repose(d0, fr, ids, f["R0"], f["t0"], R1, t1, is_dir=True))

    @torch.no_grad()
    def _init_mono_supervision(self, mono_surface: Dict[int, Dict[str, torch.Tensor]] = None) -> None:
        """
        Keep the lifted surface where LiDAR is lacking. Per instance, a lifted point is dropped
        if it is within mono.min_lidar_dist of a training return (LiDAR measured that surface
        better), or within mono.free_tol of a free-space or carving ray (LiDAR or a mask says
        nothing is there). The weight of the rest rises linearly to 1 at mono.full_lidar_dist
        from the nearest return, so the lifted surface fades in next to the LiDAR one.
        Note for `supervision.withhold_side`: lifted points on the withheld side are kept (they
        are what would fill it), but the depth they come from was aligned with all the LiDAR.
        """
        self.sup_mono_p = self.sup_mono_n = self.sup_mono_w = self.sup_mono_id = None
        if not mono_surface:
            return
        M = self._mono_cfg
        d0 = M("min_lidar_dist", 0.05)
        d1 = max(M("full_lidar_dist", 0.15), d0)
        tol, cap = M("free_tol", 0.03), M("max_per_instance", 50000)
        gen = torch.Generator(device=self.device)
        pools = [[], [], [], []]
        for i, ds_id in enumerate(self.instance_ds_ids):
            if ds_id not in mono_surface:
                continue
            p = mono_surface[ds_id]["pts"].to(self.device).float()
            n = torch.nn.functional.normalize(mono_surface[ds_id]["normals"].to(self.device).float(), dim=-1)
            n_all = p.shape[0]
            ret = self.sup_surf_p[self.sup_surf_id == i] if self.sup_surf_p is not None else p.new_zeros(0, 3)
            dist = nearest_distance(p, ret)[0]
            keep = dist > d0
            n_far = int(keep.sum())
            if tol > 0:
                for o_all, d_all, t_all, id_all in ((self.sup_ray_o, self.sup_ray_d, self.sup_ray_t, self.sup_ray_id),
                                                    (self.sup_carve_o, self.sup_carve_d, self.sup_carve_t, self.sup_carve_id)):
                    if o_all is None or not keep.any():
                        continue
                    m = id_all == i
                    sel = torch.where(keep)[0]
                    keep[sel[near_ray_segments(p[sel], o_all[m], d_all[m], t_all[m], tol)]] = False
            n_kept = int(keep.sum())
            p, n, dist = p[keep], n[keep], dist[keep]
            if n_kept > cap:
                gen.manual_seed(3)
                sub = torch.randperm(n_kept, generator=gen, device=self.device)[:cap]
                p, n, dist = p[sub], n[sub], dist[sub]
            w = ((dist - d0) / (d1 - d0)).clamp(0, 1) if d1 > d0 else torch.ones_like(dist)
            w = torch.where(torch.isfinite(dist), w, torch.ones_like(w)).clamp_min(1e-3)
            logger.info(f"{self.class_prefix} instance {i}: {n_all} lifted surface points, {n_far} farther than {d0} m "
                        f"from a return, {n_far - n_kept} of those in observed free space, {p.shape[0]} supervise the SDF "
                        f"({int((w >= 1).sum())} at full weight)")
            if p.shape[0]:
                pools[0].append(p); pools[1].append(n); pools[2].append(w)
                pools[3].append(torch.full((p.shape[0],), i, dtype=torch.long, device=self.device))
        if pools[0]:
            self.sup_mono_p, self.sup_mono_n, self.sup_mono_w, self.sup_mono_id = [torch.cat(x) for x in pools]

    # ------------------------------------------------------------------ SDF queries
    def query_sdf(self, xyz: torch.Tensor, ids: torch.Tensor, detach_params: bool = False) -> torch.Tensor:
        """Evaluate each point with the SDF of its instance. xyz: (N, 3) object frame, ids: (N,)."""
        out = xyz.new_zeros(xyz.shape[0])
        for i in ids.unique().tolist():
            m = ids == i
            net = self.sdf_nets[i]
            if detach_params:
                params = {k: v.detach() for k, v in net.named_parameters()}
                buffers = dict(net.named_buffers())
                out[m] = functional_call(net, {**params, **buffers}, (xyz[m],))
            else:
                out[m] = net(xyz[m])
        return out

    def query_sdf_with_grad(self, xyz: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = xyz.detach().requires_grad_(True)
        sdf = self.query_sdf(xyz, ids)
        grad = torch.autograd.grad(sdf.sum(), xyz, create_graph=True)[0]
        return sdf, grad

    def query_sdf_normal(self, xyz: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signed distance and unit gradient at xyz, both detached from the graph."""
        with torch.enable_grad():
            x = xyz.detach().requires_grad_(True)
            sdf = self.query_sdf(x, ids)
            grad = torch.autograd.grad(sdf.sum(), x)[0]
        return sdf.detach(), torch.nn.functional.normalize(grad, dim=-1)

    def mask_ray_min_sdf(self, o: torch.Tensor, d: torch.Tensor, t: torch.Tensor, ids: torch.Tensor, samples: int) -> torch.Tensor:
        """Smallest signed distance on each ray's interval [t[:, 0], t[:, 1]], from stratified samples."""
        n = o.shape[0]
        u = (torch.arange(samples, device=o.device)[None] + torch.rand(n, samples, device=o.device)) / samples
        tt = t[:, :1] + u * (t[:, 1:] - t[:, :1])
        p = o[:, None] + d[:, None] * tt[..., None]
        return self.query_sdf(p.reshape(-1, 3), ids.repeat_interleave(samples)).reshape(n, samples).min(dim=1).values

    def _sample(self, n: int, total: int) -> torch.Tensor:
        return torch.randint(0, total, (min(n, total),), device=self.device)

    # ------------------------------------------------------------------ losses
    def compute_reg_loss(self) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_reg_loss()
        if len(self.sdf_nets) == 0:
            return loss_dict
        L = self._loss_cfg
        for net in self.sdf_nets:
            net.set_step(self.step)
        self.register_poses_to_sdf()
        self.follow_refined_poses()

        # LiDAR returns are on the surface
        w = L("surface_w", 1.0)
        if w > 0 and self.sup_surf_p is not None:
            idx = self._sample(L("n_surface", 2048), self.sup_surf_p.shape[0])
            p_surf, d_surf, id_surf = self.sup_surf_p[idx], self.sup_surf_d[idx], self.sup_surf_id[idx]
            w_surf = self.sup_surf_w[idx]
            sdf, grad = self.query_sdf_with_grad(p_surf, id_surf)
            loss_dict["sdf_surface"] = w * (w_surf * sdf.abs()).sum() / w_surf.sum().clamp_min(1e-6)

            # a ray meets the surface from outside, so the SDF normal at a return faces the sensor.
            # Without this the sign of the field next to a return is fixed only by the box prior.
            wf = L("facing_w", 0.0)
            if wf > 0:
                cos_f = (torch.nn.functional.normalize(grad, dim=-1) * d_surf).sum(-1)
                loss_dict["sdf_facing"] = wf * (w_surf * torch.relu(cos_f + L("facing_margin", 0.0))).sum() / w_surf.sum().clamp_min(1e-6)

            # a band behind the return is inside the object (TSDF-style). The distance below
            # the surface at s along the ray is s*|cos| with the surface normal, taken from
            # the current (detached) SDF gradient; the target shrinks to 0 for grazing rays.
            wb = L("behind_w", 0.5)
            if wb > 0:
                depth = L("behind_depth", 0.5)
                normal = torch.nn.functional.normalize(grad.detach(), dim=-1)
                cos = (normal * d_surf).sum(-1).abs()
                s = torch.rand(idx.shape[0], device=self.device) * depth
                p_behind = p_surf + d_surf * s[:, None]
                sdf_b = self.query_sdf(p_behind, id_surf)
                loss_dict["sdf_behind"] = wb * (w_surf * torch.relu(sdf_b + s * cos)).sum() / w_surf.sum().clamp_min(1e-6)

        # the segment before a return is free: sdf >= margin
        w = L("free_w", 0.5)
        if w > 0 and self.sup_ray_o is not None:
            idx = self._sample(L("n_free", 4096), self.sup_ray_o.shape[0])
            t0, t1 = self.sup_ray_t[idx, 0], self.sup_ray_t[idx, 1]
            t = t0 + torch.rand_like(t0) * (t1 - t0)
            p_free = self.sup_ray_o[idx] + self.sup_ray_d[idx] * t[:, None]
            sdf_f = self.query_sdf(p_free, self.sup_ray_id[idx])
            loss_dict["sdf_free"] = w * torch.relu(L("free_margin", 0.05) - sdf_f).mean()

        # camera rays that see past the object: their crossing of the box is free
        w = L("carve_w", 0.5)
        if w > 0 and self.sup_carve_o is not None:
            idx = self._sample(L("n_free", 4096), self.sup_carve_o.shape[0])
            t0, t1 = self.sup_carve_t[idx, 0], self.sup_carve_t[idx, 1]
            t = t0 + torch.rand_like(t0) * (t1 - t0)
            p_c = self.sup_carve_o[idx] + self.sup_carve_d[idx] * t[:, None]
            sdf_c = self.query_sdf(p_c, self.sup_carve_id[idx])
            loss_dict["sdf_carve"] = w * torch.relu(L("free_margin", 0.05) - sdf_c).mean()

        # camera rays through the instance mask meet the object: somewhere on their crossing of
        # the box the distance is negative. The inside evidence that LiDAR rays do not carry.
        w = L("inside_w", 0.0)
        if w > 0 and self.sup_inside_o is not None and self.step >= L("inside_start", 0):
            idx = self._sample(L("n_inside", 1024), self.sup_inside_o.shape[0])
            lowest = self.mask_ray_min_sdf(self.sup_inside_o[idx], self.sup_inside_d[idx], self.sup_inside_t[idx],
                                           self.sup_inside_id[idx], L("inside_samples", 16))
            loss_dict["sdf_inside"] = w * torch.relu(lowest + L("inside_margin", 0.02)).mean()

        # surface lifted from monocular depth, where LiDAR is lacking
        M = self._mono_cfg
        if self.sup_mono_p is not None and self.step >= M("start", 0):
            idx = self._sample(M("n", 1024), self.sup_mono_p.shape[0])
            p_m, n_m, w_m, id_m = self.sup_mono_p[idx], self.sup_mono_n[idx], self.sup_mono_w[idx], self.sup_mono_id[idx]
            w_sum = w_m.sum().clamp_min(1e-6)
            ws, wn, wb = M("surface_w", 0.3), M("normal_w", 0.05), M("band_w", 0.15)
            if ws > 0 or wn > 0:
                sdf, grad = self.query_sdf_with_grad(p_m, id_m)
                if ws > 0:    # its points are on the zero level
                    loss_dict["sdf_mono_surface"] = ws * (w_m * sdf.abs()).sum() / w_sum
                if wn > 0:    # its normals are the direction in which the distance grows
                    cos = (torch.nn.functional.normalize(grad, dim=-1) * n_m).sum(-1)
                    loss_dict["sdf_mono_normal"] = wn * (w_m * (1 - cos)).sum() / w_sum
            if wb > 0:
                # a thin band along the normal: s in front of the surface the distance is at
                # least s (the views that lifted the point saw through it), s behind it at most -s
                s = torch.rand(idx.shape[0], device=self.device) * M("band_depth", 0.1)
                side = torch.where(torch.rand(idx.shape[0], device=self.device) < 0.5, 1.0, -1.0)
                sdf_b = self.query_sdf(p_m + n_m * (side * s)[:, None], id_m)
                loss_dict["sdf_mono_band"] = wb * (w_m * torch.relu(s - side * sdf_b)).sum() / w_sum

        # left-right symmetry: the same supervision, reflected about the object's x-z plane.
        # Targets come from data only (never from the other side's prediction), so an
        # unobserved side is filled in and an observed side is not dragged to the box prior.
        ws = self._sym_cfg("w", 0.3)
        if self._sym_cfg("enabled", False) and ws > 0 and self.step >= self._sym_cfg("start", 0):
            margin = L("free_margin", 0.05)
            w = L("surface_w", 1.0)
            if w > 0 and self.sup_surf_p is not None:
                idx = self._sample(L("n_surface", 2048) // 2, self.sup_surf_p.shape[0])
                ids, w_surf = self.sup_surf_id[idx], self.sup_surf_w[idx]
                sdf_m = self.query_sdf(self.mirror_points(self.sup_surf_p[idx], ids), ids)
                loss_dict["sdf_sym_surface"] = ws * w * (w_surf * sdf_m.abs()).sum() / w_surf.sum().clamp_min(1e-6)
            for name, w, o_all, d_all, t_all, id_all in (
                ("sdf_sym_free", L("free_w", 0.5), self.sup_ray_o, self.sup_ray_d, self.sup_ray_t, self.sup_ray_id),
                ("sdf_sym_carve", L("carve_w", 0.5), self.sup_carve_o, self.sup_carve_d, self.sup_carve_t, self.sup_carve_id),
            ):
                if w <= 0 or o_all is None:
                    continue
                idx = self._sample(L("n_free", 4096) // 2, o_all.shape[0])
                t0, t1 = t_all[idx, 0], t_all[idx, 1]
                t = t0 + torch.rand_like(t0) * (t1 - t0)
                ids = id_all[idx]
                p_m = self.mirror_points(o_all[idx] + d_all[idx] * t[:, None], ids)
                loss_dict[name] = ws * w * torch.relu(margin - self.query_sdf(p_m, ids)).mean()
            # a learned plane is only identified where both sides are observed; elsewhere it
            # would drift to make the mirrored data agree with the box prior, so keep it near 0
            wp = self._sym_cfg("plane_reg_w", 1.0)
            if self._sym_cfg("learn_plane", False) and wp > 0:
                loss_dict["sdf_sym_plane"] = wp * self.symmetry_offset.abs().mean()

        # the shell between the box and the padded box is outside the object
        w = L("outside_w", 0.5)
        if w > 0:
            n = L("n_free", 4096) // 2
            ids = torch.randint(0, self.num_instances, (n,), device=self.device)
            half = self.instances_size[ids] / 2
            pad = self.sdf_cfg.get("supervision", {}).get("box_pad", 0.3)
            p = (torch.rand(n, 3, device=self.device) * 2 - 1) * (half + pad)
            in_shell = (p.abs() > half).any(dim=-1)
            sdf_o = self.query_sdf(p[in_shell], ids[in_shell])
            loss_dict["sdf_outside"] = w * torch.relu(L("free_margin", 0.05) - sdf_o).mean()

        # Eikonal: near-surface jitter plus uniform points in the padded box
        w = L("eikonal_w", 0.1)
        if w > 0:
            n = L("n_eikonal", 1024)
            pts, ids = [], []
            if self.sup_surf_p is not None:
                idx = self._sample(n // 2, self.sup_surf_p.shape[0])
                pts.append(self.sup_surf_p[idx] + torch.randn_like(self.sup_surf_p[idx]) * 0.1)
                ids.append(self.sup_surf_id[idx])
            if self.sup_mono_p is not None and self.step >= self._mono_cfg("start", 0):
                idx = self._sample(n // 4, self.sup_mono_p.shape[0])
                pts.append(self.sup_mono_p[idx] + torch.randn_like(self.sup_mono_p[idx]) * 0.1)
                ids.append(self.sup_mono_id[idx])
            u_ids = torch.randint(0, self.num_instances, (n // 2,), device=self.device)
            half = self.instances_size[u_ids] / 2 * 1.2
            pts.append((torch.rand(n // 2, 3, device=self.device) * 2 - 1) * half)
            ids.append(u_ids)
            _, grad = self.query_sdf_with_grad(torch.cat(pts), torch.cat(ids))
            loss_dict["sdf_eikonal"] = w * (grad.norm(dim=-1) - 1).square().mean()


        # Gaussians -> SDF: opaque centres are surface samples
        # (only confident gaussians, and truncated so floaters far from the surface don't bend it)
        w = L("gs2sdf_w", 0.1)
        if w > 0 and self.step >= L("gs2sdf_start", 1000):
            idx = self._sample(L("n_gs", 2048), self.num_points)
            opac = self.get_opacity.detach()[idx, 0]
            idx = idx[opac >= L("gs2sdf_min_opacity", 0.3)]
            if self.sdf_withheld_side.any():  # gaussians on a withheld side are evidence the test must not see
                side = self.sdf_withheld_side[self.point_ids[idx, 0]]
                idx = idx[self._means.data[idx, 1] * side <= 0]
            if idx.shape[0] > 0:
                sdf_g = self.query_sdf(self._means[idx].detach(), self.point_ids[idx, 0])
                trunc = self.sdf_cfg.get("prune", {}).get("tau", 0.2)
                loss_dict["sdf_gs2sdf"] = w * sdf_g.abs().clamp_max(trunc).mean()

        # SDF -> Gaussians: pull centres onto the zero level (gradient only into the means)
        w = L("sdf2gs_w", 0.01)
        if w > 0 and self.step >= L("sdf2gs_start", 3000):
            idx = self._sample(L("n_gs", 2048), self.num_points)
            opac = self.get_opacity.detach()[idx, 0]
            sdf_m = self.query_sdf(self._means[idx], self.point_ids[idx, 0], detach_params=True)
            # power 1: constant pull; power 2: sdf^2, a pull that grows with the distance
            dist = sdf_m.square() if int(L("sdf2gs_power", 1)) == 2 else sdf_m.abs()
            loss_dict["sdf_sdf2gs"] = w * (opac * dist).sum() / opac.sum().clamp_min(1e-6)

        # SDF -> Gaussians: near the surface, a Gaussian's thinnest axis lies along the SDF normal
        # (gradient only into the rotations; which axis is thinnest is read from the scales)
        w = L("align_w", 0.0)
        if w > 0 and self.step >= L("align_start", 3000):
            idx = self._sample(L("n_gs", 2048), self.num_points)
            sdf_a, normal = self.query_sdf_normal(self._means[idx], self.point_ids[idx, 0])
            weight = self.get_opacity.detach()[idx, 0] * (sdf_a.abs() <= L("align_band", 0.1))
            if weight.sum() > 0:
                loss_dict["sdf_align"] = w * (weight * (1 - self.short_axis_cos(idx, normal))).sum() / weight.sum()

        return loss_dict

    def short_axis_cos(self, idx: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """|cos| between the thinnest axis of the Gaussians `idx` and `normal` (object frame)."""
        rot = quat_to_rotmat(self.get_quats[idx])                      # columns are the Gaussian's axes
        thinnest = self._scales.detach()[idx].argmin(dim=-1)
        axis = rot[torch.arange(idx.shape[0], device=idx.device), :, thinnest]
        return (axis * normal).sum(-1).abs()

    # ------------------------------------------------------------------ pruning
    @torch.no_grad()
    def sdf_prune_mask(self) -> torch.Tensor:
        """Gaussians farther than prune.tau (metres) from their instance's zero level."""
        prune = self.sdf_cfg.get("prune", {})
        mask = torch.zeros(self.num_points, dtype=torch.bool, device=self.device)
        if len(self.sdf_nets) == 0 or self.step < prune.get("start", 3000):
            return mask
        tau = prune.get("tau", 0.2)
        for s in range(0, self.num_points, 262144):
            e = min(s + 262144, self.num_points)
            sdf = self.query_sdf(self._means[s:e], self.point_ids[s:e, 0])
            mask[s:e] = sdf.abs() > tau
        return mask

    def cull_gaussians(self):
        """RigidNodes.cull_gaussians plus the SDF distance test."""
        n_bef = self.num_points
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.ctrl_cfg.cull_out_of_bound:
            culls = culls | self.get_out_of_bound_mask()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values >
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        sdf_culls = self.sdf_prune_mask()
        culls = culls | sdf_culls
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self.point_ids = self.point_ids[~culls]
        print(f"     Cull: {n_bef - self.num_points} (sdf: {int(sdf_culls.sum())})")
        return culls

    # ------------------------------------------------------------------ SDF -> new Gaussians
    @torch.no_grad()
    def sdf_surface_cells(self, i: int, n_samples: int, cell: float, iters: int = 6, tol: float = 0.005):
        """
        Points on instance i's zero level inside its box, at most one per `cell`-sized voxel,
        with their outward unit normals. Random points in the box are projected onto the zero
        level with Newton steps along the SDF gradient.
        """
        net = self.sdf_nets[i]
        half = self.instances_size[i] / 2

        def value_grad(x):
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                sdf = net(x)
                grad = torch.autograd.grad(sdf.sum(), x)[0]
            return sdf.detach(), grad.detach()

        p = (torch.rand(n_samples, 3, device=self.device) * 2 - 1) * half
        for _ in range(iters):
            sdf, grad = value_grad(p)
            p = p - sdf[:, None] * grad / grad.pow(2).sum(-1, keepdim=True).clamp_min(1e-4)
            p = torch.maximum(torch.minimum(p, half), -half)
        sdf, grad = value_grad(p)
        norm = grad.norm(dim=-1)
        keep = (sdf.abs() < tol) & (norm > 0.2)
        p, n = p[keep], grad[keep] / norm[keep, None]
        if p.shape[0] == 0:
            return p, n
        key = torch.floor((p + half) / cell).long()
        key = (key[:, 0] * 4096 + key[:, 1]) * 4096 + key[:, 2]
        uniq, inv = torch.unique(key, return_inverse=True)
        first = torch.full((uniq.shape[0],), p.shape[0], dtype=torch.long, device=self.device)
        first = first.scatter_reduce(0, inv, torch.arange(p.shape[0], device=self.device), reduce="amin")
        return p[first], n[first]

    @torch.no_grad()
    def sdf_seed_candidates(self, i: int) -> Dict[str, torch.Tensor]:
        """
        Surface cells of instance i with, per cell: `covered` (within seed.radius of a Gaussian of
        the instance, counting its extent of seed.cover_sigma standard deviations), `supported` (the SDF has evidence there: a training LiDAR return within
        seed.support_radius; seed.support = lidar | lidar_sym | any), `views` (training camera
        views facing it) and `source` (index of the nearest Gaussian of the instance, -1 if none).
        """
        radius = self._seed_cfg("radius", 0.05)
        p, n = self.sdf_surface_cells(i, self._seed_cfg("n_samples", 40000), radius)
        M = p.shape[0]
        own = torch.where(self.point_ids[:, 0] == i)[0]
        min_op = self._seed_cfg("min_opacity", 0.0)
        cover = own[self.get_opacity.data[own, 0] >= min_op] if min_op > 0 else own
        # a Gaussian covers what lies within `radius` of its centre, or of its cover_sigma-sigma ball
        extent = self._seed_cfg("cover_sigma", 1.0) * self.get_scaling.data[cover].max(dim=-1).values
        covered = nearest_distance(p, self._means.data[cover], extent=extent)[0] <= radius
        source = own[nearest_distance(p, self._means.data[own])[1]] if own.numel() else torch.full((M,), -1, dtype=torch.long, device=self.device)

        mode = str(self._seed_cfg("support", "lidar"))
        if mode == "any":
            supported = torch.ones(M, dtype=torch.bool, device=self.device)
        else:
            if self.sup_surf_p is not None:
                ret = self.sup_surf_p[self.sup_surf_id == i]
            else:
                ret = torch.zeros(0, 3, device=self.device)
            if mode == "lidar_sym" and ret.shape[0]:
                ids = torch.full((ret.shape[0],), i, dtype=torch.long, device=self.device)
                ret = torch.cat([ret, self.mirror_points(ret, ids)])
            supported = nearest_distance(p, ret)[0] <= self._seed_cfg("support_radius", 0.05)

        views = torch.zeros(M, dtype=torch.long, device=self.device)
        if self.sdf_cam_views is not None and M:
            views = count_facing_views(p, n, self.sdf_cam_views[i], min_cos=self._seed_cfg("camera_min_cos", 0.2),
                                       max_dist=self._seed_cfg("camera_max_dist", 60.0))
        return dict(p=p, n=n, covered=covered, supported=supported, views=views, source=source)

    @torch.no_grad()
    def seed_gaussians_from_sdf(self, optimizer: torch.optim.Optimizer = None) -> int:
        """
        Add Gaussians on each instance's zero level where the surface is supported, faced by a
        training camera (seed.require_camera) and not covered by a Gaussian of the instance.
        New Gaussians are flat discs in the tangent plane, coloured like the nearest Gaussian
        of the instance, at low opacity so the photometric loss decides whether they stay.
        """
        radius = self._seed_cfg("radius", 0.05)
        cap = self._seed_cfg("max_per_instance", 2000)
        need_cam = self._seed_cfg("require_camera", True)
        min_views = self._seed_cfg("min_views", 3)
        new_p, new_n, new_src, new_id, stats = [], [], [], [], []
        for i in range(self.num_instances):
            c = self.sdf_seed_candidates(i)
            seen = c["views"] >= min_views
            ok = ~c["covered"] & c["supported"] & (c["source"] >= 0)
            if need_cam:
                ok &= seen
            idx = torch.where(ok)[0]
            if idx.numel() > cap:
                idx = idx[torch.randperm(idx.numel(), device=self.device)[:cap]]
            stats.append(dict(
                instance=i, cells=int(c["p"].shape[0]), supported=int(c["supported"].sum()), seen=int(seen.sum()),
                supported_seen=int((c["supported"] & seen).sum()), covered=int(c["covered"].sum()),
                supported_seen_covered=int((c["supported"] & seen & c["covered"]).sum()),
                eligible=int(ok.sum()), added=int(idx.numel())))
            if idx.numel():
                new_p.append(c["p"][idx]); new_n.append(c["n"][idx]); new_src.append(c["source"][idx])
                new_id.append(torch.full((idx.numel(), 1), i, dtype=torch.long, device=self.device))
        total = sum(s["added"] for s in stats)
        self.sdf_seed_log.append(dict(step=int(self.step), added=total, instances=stats))
        tot = {k: sum(s[k] for s in stats) for k in ("cells", "supported", "seen", "supported_seen", "supported_seen_covered", "eligible")}
        logger.info(
            f"{self.class_prefix} SDF seeding @ step {self.step}: {tot['cells']} surface cells, "
            f"{tot['supported']} supported (support={self._seed_cfg('support', 'lidar')}), {tot['seen']} camera-facing, {tot['supported_seen']} both "
            f"({tot['supported_seen_covered']} of those covered), {tot['eligible']} eligible, {total} added; "
            f"per instance {[s['added'] for s in stats]}")
        if total == 0:
            return 0
        p, n, src, ids = torch.cat(new_p), torch.cat(new_n), torch.cat(new_src), torch.cat(new_id)
        # rotation taking the local z axis onto the normal (wxyz); the disc is thin along z
        quats = torch.stack([1 + n[:, 2], -n[:, 1], n[:, 0], torch.zeros_like(n[:, 0])], dim=-1)
        flip = quats.norm(dim=-1) < 1e-6
        quats[flip] = quats.new_tensor([0.0, 1.0, 0.0, 0.0])
        quats = quats / quats.norm(dim=-1, keepdim=True)
        s = radius * self._seed_cfg("scale", 0.5)
        scales = torch.log(torch.tensor([s, s, s * self._seed_cfg("flatten", 0.3)], device=self.device)).expand(total, 3)
        opac = torch.logit(torch.full((total, 1), float(self._seed_cfg("opacity", 0.1)), device=self.device))

        self._means = Parameter(torch.cat([self._means.detach(), p], dim=0))
        self._features_dc = Parameter(torch.cat([self._features_dc.detach(), self._features_dc.detach()[src]], dim=0))
        self._features_rest = Parameter(torch.cat([self._features_rest.detach(), self._features_rest.detach()[src]], dim=0))
        self._opacities = Parameter(torch.cat([self._opacities.detach(), opac], dim=0))
        self._scales = Parameter(torch.cat([self._scales.detach(), scales], dim=0))
        self._quats = Parameter(torch.cat([self._quats.detach(), quats], dim=0))
        self.point_ids = torch.cat([self.point_ids, ids], dim=0)
        # densification statistics gathered since the last refinement (None right after one)
        if self.xys_grad_norm is not None:
            self.xys_grad_norm = torch.cat([self.xys_grad_norm, self.xys_grad_norm.new_zeros(total)])
            self.vis_counts = torch.cat([self.vis_counts, self.vis_counts.new_ones(total)])
        if self.max_2Dsize is not None:
            self.max_2Dsize = torch.cat([self.max_2Dsize, self.max_2Dsize.new_zeros(total)])
        if optimizer is not None:
            groups = self.get_gaussian_param_groups()
            for group in optimizer.param_groups:
                if group["name"] not in groups:
                    continue
                state = optimizer.state.pop(group["params"][0], None)
                group["params"] = groups[group["name"]]
                if state is not None:
                    for k in ("exp_avg", "exp_avg_sq"):
                        if k in state:
                            state[k] = torch.cat([state[k], state[k].new_zeros((total,) + tuple(state[k].shape[1:]))], dim=0)
                    optimizer.state[group["params"][0]] = state
        return total

    def refinement_after(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        super().refinement_after(step, optimizer)
        if not self._seed_cfg("enabled", False) or len(self.sdf_nets) == 0:
            return
        if step < self._seed_cfg("start", 5000) or step > self._seed_cfg("stop", 20000):
            return
        if step % self._seed_cfg("interval", 1000) != 0:
            return
        self.seed_gaussians_from_sdf(optimizer)

    # ------------------------------------------------------------------ export
    @torch.no_grad()
    def export_meshes(self, out_dir: str, step: int, resolution: int = None) -> None:
        """Marching cubes of each instance SDF over its padded box, written as PLY."""
        from skimage.measure import marching_cubes
        os.makedirs(out_dir, exist_ok=True)
        resolution = self.mesh_resolution if resolution is None else resolution
        for i, net in enumerate(self.sdf_nets):
            half = (self.instances_size[i] / 2 * 1.2).cpu().numpy()
            res = np.maximum((resolution * half / half.max()).astype(int), 16)
            axes = [torch.linspace(-h, h, int(r), device=self.device) for h, r in zip(half, res)]
            grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
            vals = torch.cat([net(grid[s:s + 262144]) for s in range(0, grid.shape[0], 262144)])
            vol = vals.reshape(*[int(r) for r in res]).cpu().numpy()
            if vol.min() >= 0 or vol.max() <= 0:
                logger.info(f"{self.class_prefix} instance {i}: no zero crossing, mesh skipped")
                continue
            spacing = tuple(float(2 * h / (r - 1)) for h, r in zip(half, res))
            verts, faces, normals, _ = marching_cubes(vol, level=0.0, spacing=spacing)
            verts = verts - half
            _write_ply(os.path.join(out_dir, f"step_{step:05d}_instance_{i:02d}.ply"), verts, faces, normals)
        logger.info(f"{self.class_prefix} exported SDF meshes to {out_dir}")
