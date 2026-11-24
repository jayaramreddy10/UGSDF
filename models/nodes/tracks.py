from typing import Dict, List, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

from models.gaussians.basics import *
from models.gaussians.vanilla import VanillaGaussians
from models.human_body import get_on_mesh_init_geo_values
from models.nodes.smpl import SMPLNodes
from utils.geometry import knn_points

logger = logging.getLogger()


class _MeshOnly:
    """The one thing get_on_mesh_init_geo_values reads from an SMPL template."""
    def __init__(self, v: torch.Tensor, f: torch.Tensor):
        self.v, self.f = v, f

    def get_init_vf(self):
        return self.v, self.f


def positional(x: torch.Tensor, n: int) -> torch.Tensor:
    """x, sin(2^k x), cos(2^k x) for k < n, over the last dimension."""
    out = [x]
    for k in range(n):
        out += [torch.sin(2.0 ** k * x), torch.cos(2.0 ** k * x)]
    return torch.cat(out, -1)


def local_rotations(c: torch.Tensor, nn_idx: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """(A, P, 3, 3) best rigid rotation of each point's rest neighbours onto their posed positions.
    c, p (A, P, 3) rest and posed positions; nn_idx (A, P, K) neighbours."""
    a = torch.arange(p.shape[0], device=p.device)[:, None, None]
    src = c[a, nn_idx] - c[:, :, None]
    dst = p[a, nn_idx] - p[:, :, None]
    U, _, Vh = torch.linalg.svd((src.transpose(-1, -2) @ dst).reshape(-1, 3, 3))
    V = Vh.transpose(-1, -2)
    sign = torch.ones_like(U[:, :, 0])
    sign[:, 2] = torch.sign(torch.linalg.det(V @ U.transpose(-1, -2)))
    sign[sign == 0] = 1.0
    return ((V * sign[:, None, :]) @ U.transpose(-1, -2)).reshape(p.shape[0], p.shape[1], 3, 3)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, zero_last: bool = False):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        if zero_last:
            nn.init.zeros_(layers[-1].weight); nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TrackNodes(SMPLNodes):
    """
    People driven by 3D point tracks instead of SMPL poses. Takes the place of SMPLNodes in the scene graph.

    Input is the npz of tools/export_point_tracks.py (`ctrl.tracks_path`): P control points per person with a
    rest position and a world position per frame. Each Gaussian starts on the rest mesh and is bound to its
    `ctrl.bind_k` nearest control points. At a frame each control point gets a rotation, the best rigid fit
    (Kabsch) of its `ctrl.n_kabsch` rest neighbours onto their tracked positions, and a Gaussian moves as
        x_world = sum_k w_k [ R_k (x - c_k) + p_k ]
    Tracks are fixed: no pose parameter is optimised and no SMPL pose or joint is read at any point.

    Held-out frames never use their own track unless the neighbouring frames are missing. `ctrl.test_interp`:
    linear (mean of f-1, f+1), cubic (Lagrange through f-2, f-1, f+1, f+2, falling back to linear), own.

    `ctrl.n_tracks` keeps the first n control points, `ctrl.n_gaussians` the first n vertices of the npz's ordering as
    Gaussians (0: every mesh vertex).

    Motion options (`ctrl.motion`), all acting on the tracked points and trained by the image loss:
      mode none    the tracks as they are
      mode time    a per-scene field: offset = MLP(rest position, time, person)
      mode window  a prior shared by all people: offset = MLP(rest position, the point's positions at f-2, f-1,
                   f+1, f+2 relative to the cubic prediction, in the point's local frame). Never sees frame f.
      p_hide       on training steps each person with probability p_hide is posed from the neighbouring
                   frames (cubic) instead of its own track, the same conditions as a held-out frame.
                   In window mode the MLP only acts in those cases and on held-out frames.
    Rotations come from the positions before the offset. `reg.body_sdf` adds a canonical SDF per person, supervised
    by the rest positions of the tracked points, and holds the Gaussians on it.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state_cache = {}
        self.motion_net = None
        self.person_embed = None
        self.body_sdf = None
        self.sdf_embed = None
        self._last_delta = None

    # ------------------------------------------------------------------ data
    def _load_tracks(self, person_ids: torch.Tensor) -> None:
        d = np.load(self.ctrl_cfg.tracks_path)
        ids = person_ids.cpu().numpy().tolist()
        where = {int(p): i for i, p in enumerate(d["person_dataset_id"])}
        missing = [p for p in ids if p not in where]
        assert not missing, f"persons {missing} are not in {self.ctrl_cfg.tracks_path}"
        idx = [where[p] for p in ids]
        dev = self.device
        tracks = torch.from_numpy(d["tracks_world"][:, idx]).float().to(dev)      # (F, H, P, 3)
        n_tracks = int(self.ctrl_cfg.get("n_tracks", 0) or 0)          # the first n of a farthest-point ordering stay well spread
        if n_tracks > 0:
            tracks = tracks[:, :, :n_tracks]
        self.tracks = torch.nan_to_num(tracks, nan=0.0)
        self.tracks_present = torch.from_numpy(d["present"][:, idx]).bool().to(dev)
        self.ctrl_canon = torch.from_numpy(d["canonical"][idx]).float().to(dev)   # (H, P, 3)
        if n_tracks > 0:
            self.ctrl_canon = self.ctrl_canon[:, :n_tracks]
        n_frames = self.tracks.shape[0]
        self.frame_is_test = torch.from_numpy(d["frame_is_test"]).bool().to(dev) if "frame_is_test" in d.files else torch.zeros(n_frames, dtype=torch.bool, device=dev)
        self.mesh_full = torch.from_numpy(d["canonical_full"][idx]).float().to(dev)   # (H, V, 3)
        n_gs = int(self.ctrl_cfg.get("n_gaussians", 0) or 0)           # 0: one Gaussian per mesh vertex
        self.gs_vertices = None
        if n_gs > 0:
            assert n_gs <= len(d["vertex_ids"]), f"n_gaussians {n_gs} exceeds the {len(d['vertex_ids'])} ordered vertices of the npz"
            self.gs_vertices = torch.from_numpy(d["vertex_ids"][:n_gs]).long().to(dev)
        self.mesh_canon = self.mesh_full if self.gs_vertices is None else self.mesh_full[:, self.gs_vertices]    # (H, G, 3)
        self.mesh_faces = torch.from_numpy(d["faces"]).long().to(dev)
        self.smpl_points_num = self.mesh_canon.shape[1]
        K_n, K_b = int(self.ctrl_cfg.get("n_kabsch", 12)), int(self.ctrl_cfg.get("bind_k", 4))
        _, self.ctrl_nn, _ = knn_points(self.ctrl_canon, self.ctrl_canon, K=K_n, return_nn=False)    # (H, P, K_n)
        d2, self.bind_idx, _ = knn_points(self.mesh_canon, self.ctrl_canon, K=K_b, return_nn=False)  # (H, G, K_b)
        w = torch.exp(-d2 / (2 * d2[..., -1:].clamp_min(1e-6)))
        self.bind_w = (w / w.sum(-1, keepdim=True).clamp_min(1e-12))[..., None]                      # (H, G, K_b, 1)
        self._state_cache = {}
        self._build_networks(len(ids))
        logger.info(f"TrackNodes: {len(ids)} persons, {self.ctrl_canon.shape[1]} tracks each, "
                    f"{self.smpl_points_num} Gaussians each, from {self.ctrl_cfg.tracks_path}")

    def create_from_pcd(self, instance_pts_dict: Dict[str, Dict]) -> None:
        """Only the person ids and box sizes are read from the dict; its SMPL poses are not used."""
        keys = list(instance_pts_dict.keys())
        self.person_ids = torch.tensor([int(k) for k in keys], dtype=torch.long, device=self.device)
        self._load_tracks(self.person_ids)
        self.instances_fv = self.tracks_present
        self.instances_size = torch.stack([instance_pts_dict[k]["size"] for k in keys]).to(self.device)
        H, G = self.num_instances, self.smpl_points_num
        self.point_ids = torch.arange(H, device=self.device).repeat_interleave(G)[:, None]
        o0 = torch.tensor(self.ctrl_cfg.opacity_init_value)
        x, q, s, o = get_on_mesh_init_geo_values(_MeshOnly(self.mesh_full.cpu(), self.mesh_faces.cpu()), opacity_init_logit=torch.logit(o0))
        if self.gs_vertices is not None:        # a subset of the vertices: normals from the mesh, disc size from the subset's own spacing
            sub = self.gs_vertices.cpu()
            x, q, s, o = [v.reshape(H, -1, v.shape[-1])[:, sub].reshape(-1, v.shape[-1]) for v in (x, q, s, o)]
            d2, _, _ = knn_points(x.reshape(H, G, 3), x.reshape(H, G, 3), K=4, return_nn=False)
            radius = (0.5 * d2[..., 1:].sqrt().mean(-1)).reshape(-1, 1).clamp(1e-4, 1 - 1e-4)
            s = torch.logit(torch.cat([radius, radius, (0.5 * radius).clamp(1e-4, 1 - 1e-4)], -1))
        if self.ball_gaussians:
            s = s.mean(-1, keepdim=True)
        x, q, s, o = [v.to(dtype=torch.float32, device=self.device) for v in (x, q, s, o)]
        self.update_knn(x)
        self._means = Parameter(x, requires_grad=not self.ctrl_cfg.freeze_x)
        self._scales = Parameter(s, requires_grad=not self.ctrl_cfg.freeze_s)
        self._quats = Parameter(q, requires_grad=not self.ctrl_cfg.freeze_q)
        self._opacities = Parameter(o, requires_grad=not self.ctrl_cfg.freeze_o)
        dim_sh = num_sh_bases(self.sh_degree)
        fused = RGB2SH(torch.rand((self.num_points, 3), device=self.device))
        shs = torch.zeros((fused.shape[0], dim_sh, 3), device=self.device)
        shs[:, 0, :3] = fused
        self._features_dc = Parameter(shs[:, 0, :], requires_grad=not self.ctrl_cfg.freeze_shs_dc)
        self._features_rest = Parameter(shs[:, 1:, :], requires_grad=not self.ctrl_cfg.freeze_shs_rest)

    def _motion_cfg(self, key, default):
        m = self.ctrl_cfg.get("motion", None)
        return m.get(key, default) if m is not None else default

    def _sdf_cfg(self):
        cfg = self.reg_cfg.get("body_sdf", None) if self.reg_cfg is not None else None
        return cfg if cfg is not None and cfg.get("enabled", False) else None

    def _build_networks(self, n_persons: int) -> None:
        mode = self._motion_cfg("mode", "none")
        hidden, depth = int(self._motion_cfg("hidden", 128)), int(self._motion_cfg("depth", 4))
        L = 4
        if mode == "time":
            self.person_embed = nn.Embedding(n_persons, 16).to(self.device)
            self.motion_net = MLP(3 * (1 + 2 * L) + (1 + 2 * L) + 16, 3, hidden, depth, zero_last=True).to(self.device)
        elif mode == "window":
            self.motion_net = MLP(3 * (1 + 2 * L) + 12, 3, hidden, depth, zero_last=True).to(self.device)
        else:
            assert mode == "none", f"unknown motion mode {mode}"
        if self._sdf_cfg() is not None:
            self.sdf_embed = nn.Embedding(n_persons, 16).to(self.device)
            self.body_sdf = MLP(3 * (1 + 2 * 6) + 16, 1, 128, 4).to(self.device)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = self.get_gaussian_param_groups()
        if self.motion_net is not None:
            params = list(self.motion_net.parameters())
            if self.person_embed is not None:
                params += list(self.person_embed.parameters())
            groups[self.class_prefix + "motion"] = params
        if self.body_sdf is not None:
            groups[self.class_prefix + "body_sdf"] = list(self.body_sdf.parameters()) + list(self.sdf_embed.parameters())
        return groups

    # ------------------------------------------------------------------ motion
    def tracks_at(self, f: int, test: bool) -> torch.Tensor:
        """(H, P, 3) control point positions at frame f."""
        own = self.tracks[f]
        mode = self.ctrl_cfg.get("test_interp", "cubic")
        if not test or mode == "own" or f < 1 or f + 1 >= self.num_frames:
            return own
        fv, T = self.instances_fv, self.tracks
        ok1 = fv[f - 1] & fv[f + 1]
        out = torch.where(ok1[:, None, None], 0.5 * (T[f - 1] + T[f + 1]), own)
        if mode == "cubic" and f >= 2 and f + 2 < self.num_frames:
            ok2 = ok1 & fv[f - 2] & fv[f + 2]
            cubic = (2 * (T[f - 1] + T[f + 1]) / 3) - ((T[f - 2] + T[f + 2]) / 6)
            out = torch.where(ok2[:, None, None], cubic, out)
        return out

    def window_valid(self, f: int) -> torch.Tensor:
        """(H,) persons with all of f-2, f-1, f+1, f+2 present, none of them a held-out frame."""
        if f < 2 or f + 2 >= self.num_frames:
            return torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
        ks = (f - 2, f - 1, f + 1, f + 2)
        if self.frame_is_test[list(ks)].any():
            return torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)
        out = self.instances_fv[f].clone()
        for k in ks:
            out &= self.instances_fv[k]
        return out

    def _cubic(self, f: int) -> torch.Tensor:
        T = self.tracks
        return 2 * (T[f - 1] + T[f + 1]) / 3 - (T[f - 2] + T[f + 2]) / 6

    @torch.no_grad()
    def _base_state(self, f: int, test: bool, hide: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Positions (A, P, 3) and rotation quaternions (A, P, 4) of the control points of the A active persons."""
        key = (f, test, hide, self.ctrl_cfg.get("test_interp", "cubic"))
        if key not in self._state_cache:
            mask = self.instances_fv[f]
            if hide:
                ok = self.window_valid(f)
                p = torch.where(ok[:, None, None], self._cubic(f), self.tracks[f]) if ok.any() else self.tracks[f]
            else:
                p = self.tracks_at(f, test)
            p = p[mask]
            R = local_rotations(self.ctrl_canon[mask], self.ctrl_nn[mask], p)
            self._state_cache[key] = (p, matrix_to_quaternion(R.reshape(-1, 3, 3)).reshape(p.shape[0], p.shape[1], 4))
        return self._state_cache[key]

    def _frame_state(self):
        """Control point positions and rotations at the current frame, with the motion option applied."""
        f, test = int(self.cur_frame), bool(self.in_test_set)
        mask = self.instances_fv[f]
        p, q = self._base_state(f, test, False)
        A = p.shape[0]
        hidden = torch.zeros(A, dtype=torch.bool, device=p.device)
        p_hide = float(self._motion_cfg("p_hide", 0.0))
        if self.training and p_hide > 0 and not test:
            valid = self.window_valid(f)[mask]
            hidden = (torch.rand(A, device=p.device) < p_hide) & valid
            if hidden.any():
                ph, qh = self._base_state(f, False, True)
                p = torch.where(hidden[:, None, None], ph, p)
                q = torch.where(hidden[:, None, None], qh, q)
        self._last_delta = None
        if self.motion_net is None:
            return p, q
        c = self.ctrl_canon[mask]
        P = c.shape[1]
        if self._motion_cfg("mode", "none") == "time":
            t = torch.full((A, P, 1), f / max(self.num_frames - 1, 1), device=p.device)
            e = self.person_embed(mask.nonzero()[:, 0])[:, None].expand(-1, P, -1)
            delta = torch.tanh(self.motion_net(torch.cat((positional(c, 4), positional(t, 4), e), -1)))
        else:
            use = (self.window_valid(f)[mask] if test else hidden)
            delta = torch.zeros(A, P, 3, device=p.device)
            if use.any():
                Rw = quat_to_rotmat(q[use].reshape(-1, 4)).reshape(int(use.sum()), P, 3, 3)
                ks = (f - 2, f - 1, f + 1, f + 2)
                rel = torch.cat([(Rw.transpose(-1, -2) @ (self.tracks[k][mask][use] - p[use])[..., None])[..., 0] for k in ks], -1) / 0.05
                local = torch.tanh(self.motion_net(torch.cat((positional(c[use], 4), rel), -1)))
                delta = delta.clone()
                delta[use] = (Rw @ local[..., None])[..., 0]
        delta = delta * float(self._motion_cfg("max_offset", 0.1))
        self._last_delta = delta
        return p + delta, q

    def _deform(self, means: torch.Tensor):
        """World positions (A, G, 3) and rotation quaternions (A, G, 4) of the Gaussians of the active persons."""
        assert means.shape[0] == self.point_ids.shape[0], "means must hold every Gaussian"
        mask = self.instances_fv[self.cur_frame]
        p, q = self._frame_state()
        x = means.reshape(self.num_instances, self.smpl_points_num, 3)[mask]
        idx, w = self.bind_idx[mask], self.bind_w[mask]
        a = torch.arange(p.shape[0], device=p.device)[:, None, None]
        qk = q[a, idx]                                                   # (A, G, K, 4)
        Rk = quat_to_rotmat(qk.reshape(-1, 4)).reshape(*qk.shape[:-1], 3, 3)
        moved = (Rk @ (x[:, :, None] - self.ctrl_canon[mask][a, idx])[..., None])[..., 0] + p[a, idx]
        world = (w * moved).sum(2)
        aligned = qk * torch.sign((qk * qk[:, :, :1]).sum(-1, keepdim=True)).clamp_min(0).mul(2).sub(1)
        blend = (w * aligned).sum(2)
        return world, blend / blend.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _sdf(self, x: torch.Tensor, person: torch.Tensor) -> torch.Tensor:
        return self.body_sdf(torch.cat((positional(x, 6), self.sdf_embed(person)), -1))[..., 0]

    def compute_reg_loss(self):
        loss_dict = super().compute_reg_loss()
        mask = self.instances_fv[self.cur_frame]
        if mask.sum() == 0:
            return loss_dict
        reg = self.reg_cfg.get("motion_reg", None)
        if reg is not None and self._last_delta is not None:
            loss_dict["motion_offset"] = self._last_delta.abs().mean() * reg.w
        cfg = self._sdf_cfg()
        if cfg is not None:
            n = int(cfg.get("n", 2048)); dev = self.device
            active = mask.nonzero()[:, 0]
            person = active[torch.randint(active.shape[0], (n,), device=dev)]
            on = self.ctrl_canon[person, torch.randint(self.ctrl_canon.shape[1], (n,), device=dev)]
            loss_dict["body_sdf_surface"] = self._sdf(on, person).abs().mean() * cfg.get("surface_w", 1.0)
            lo, hi = self.mesh_canon.reshape(-1, 3).min(0).values - 0.1, self.mesh_canon.reshape(-1, 3).max(0).values + 0.1
            free = (torch.rand(n, 3, device=dev) * (hi - lo) + lo).requires_grad_(True)
            f = self._sdf(free, person)
            g = torch.autograd.grad(f.sum(), free, create_graph=True)[0]
            loss_dict["body_sdf_eikonal"] = ((g.norm(dim=-1) - 1) ** 2).mean() * cfg.get("eikonal_w", 0.01)
            loss_dict["body_sdf_off"] = torch.exp(-50 * f.abs()).mean() * cfg.get("off_w", 0.1)
            if self.step >= cfg.get("gs_start", 1000):
                G = self.smpl_points_num
                xg = self._means.reshape(self.num_instances, G, 3)[person, torch.randint(G, (n,), device=dev)]
                loss_dict["body_sdf_gaussians"] = self._sdf(xg, person).abs().mean() * cfg.get("gs_w", 0.1)
        return loss_dict

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        mask = self.instances_fv[self.cur_frame]
        world, _ = self._deform(means)
        out = torch.zeros(self.num_instances, self.smpl_points_num, 3, device=self.device, dtype=world.dtype)
        out[mask] = world
        return out.reshape(-1, 3)

    def transform_means_and_quats(self, means: torch.Tensor, quats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.instances_fv[self.cur_frame]
        world, q = self._deform(means)
        local = quats.reshape(self.num_instances, self.smpl_points_num, 4)[mask]
        out_x = torch.zeros(self.num_instances, self.smpl_points_num, 3, device=self.device, dtype=world.dtype)
        out_q = torch.zeros(self.num_instances, self.smpl_points_num, 4, device=self.device, dtype=world.dtype)
        out_q[..., 0] = 1.0
        out_x[mask] = world
        out_q[mask] = quat_mult(self.quat_act(q), self.quat_act(local))
        return out_x.reshape(-1, 3), out_q.reshape(-1, 4)

    # ------------------------------------------------------------------ checkpoint (the tracks stay in the npz)
    def state_dict(self) -> Dict:
        sd = super().state_dict()
        sd["person_ids"] = self.person_ids
        return sd

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.person_ids = state_dict.pop("person_ids")
        self.point_ids = state_dict.pop("points_ids")
        self.instances_size = state_dict.pop("instances_size")
        self.instances_fv = state_dict.pop("instances_fv")
        self._load_tracks(self.person_ids)
        return VanillaGaussians.load_state_dict(self, state_dict, **kwargs)
