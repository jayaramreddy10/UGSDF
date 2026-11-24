"""Synthetic checks of (1) the left-right symmetry option of SDFGSNodes, (1b) withholding one
side of an instance from the SDF supervision, (1c) seeding Gaussians from the SDF, (1d) a surface
lifted from monocular depth supervising the SDF where LiDAR is lacking, (1e) the coupling terms
(mask rays as inside evidence, thinnest axis along the SDF normal, squared pull), and (2) the
ray-cast free-space grid with the background loss/prune.

Run from the repo root:  PYTHONPATH=$(pwd) python tests/test_symmetry_freespace.py $(pwd)   (GPU; ~1 min)
"""
import os, sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, sys.argv[1])
from models.nodes import SDFGSNodes
from models.gaussians import VanillaGaussians
from models.gaussians.free_space import FreeSpaceGrid

torch.manual_seed(0)
dev = torch.device(os.environ.get("SDFGS_DEVICE", "cuda"))
ctrl = OmegaConf.create(dict(
    warmup_steps=0, reset_alpha_interval=3000, refine_interval=100, sh_degree_interval=1000,
    n_split_samples=2, reset_alpha_value=0.01, densify_grad_thresh=5e-4, densify_size_thresh=3e-3,
    cull_alpha_thresh=0.005, cull_scale_thresh=0.1, cull_screen_size=0.15, split_screen_size=0.05,
    stop_screen_size_at=30000, stop_split_at=20000, sh_degree=3, cull_out_of_bound=True))

# ---------------------------------------------------------------- 0. box clamp
from models.nodes.sdfgs import InstanceSDF, box_sdf
half = torch.tensor([2.0, 1.0, 0.75])
free_net, clamped = InstanceSDF(half), InstanceSDF(half, box_clamp=0.03)
for net in (free_net, clamped):          # a residual that wants everything to be deep inside
    torch.nn.init.zeros_(net.network[-1].weight); torch.nn.init.constant_(net.network[-1].bias, -1.0)
p_out = (torch.rand(4000, 3) * 2 - 1) * (half + 0.4)
p_out = p_out[((p_out.abs() - half).max(-1).values > 0.03)]
p_in = (torch.rand(4000, 3) * 2 - 1) * (half - 0.05)
with torch.no_grad():
    assert (free_net(p_out) < 0).all(), "without the clamp this residual puts matter outside the box"
    assert (clamped(p_out) > 0).all(), "the clamp must keep everything beyond box + tolerance outside"
    assert torch.allclose(clamped(p_out), box_sdf(p_out, half + 0.03))
    # inside the box the clamp is a lower bound that is itself negative: the sign, and so
    # the zero level, cannot change there
    assert ((clamped(p_in) < 0) == (free_net(p_in) < 0)).all() and (clamped(p_in) >= free_net(p_in)).all()
assert set(clamped.state_dict()) == set(free_net.state_dict()), "the clamp adds no checkpoint keys"
print("box clamp OK")

# ---------------------------------------------------------------- 1. symmetry
SIZE = torch.tensor([4.0, 2.0, 1.5]); HALF = SIZE / 2
OBJ_HALF = HALF - torch.tensor([0.0, 0.3, 0.0])   # the object is 0.3 m narrower than its box on each side
F = 6


def first_hit(o, d, centre, half):
    d_safe = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    t1 = (centre - half - o) / d_safe; t2 = (centre + half - o) / d_safe
    t_in = torch.minimum(t1, t2).max(-1).values; t_out = torch.maximum(t1, t2).min(-1).values
    return (t_out > t_in) & (t_in > 0), t_in


class OneSideLidar:
    """Sensor at the origin, the box always to its left (y > 0): only the -x and -y faces are seen."""
    def __init__(self):
        self.frames, self.poses = [], []
        for fi in range(F):
            t = torch.tensor([10.0, 2.5 + 0.5 * fi, 0.0])
            targets = torch.rand(15000, 3) * torch.tensor([8, 6, 3.0]) + torch.tensor([6, t[1] - 3, -HALF[2]])
            o = torch.tensor([0.0, 0.0, 1.0]).expand_as(targets)
            d = targets - o; d = d / d.norm(dim=-1, keepdim=True)
            hit_box, t_box = first_hit(o, d, t, OBJ_HALF)
            t_g = (-HALF[2] - o[:, 2]) / torch.where(d[:, 2].abs() < 1e-9, torch.full_like(d[:, 2], -1e-9), d[:, 2])
            r = torch.full((len(d),), 30.0)
            r = torch.where((d[:, 2] < 0) & (t_g > 0), t_g, r)
            r = torch.where(hit_box, t_box, r)
            self.frames.append(dict(lidar_origins=o.to(dev), lidar_viewdirs=d.to(dev), lidar_ranges=r[:, None].to(dev)))
            self.poses.append(torch.cat([torch.cat([torch.eye(3), t[:, None]], 1), torch.tensor([[0, 0, 0, 1.0]])]))

    def get_lidar_rays(self, fi):
        return self.frames[fi]


def fit(symmetry: bool, learn_plane: bool = False):
    torch.manual_seed(0)
    lidar = OneSideLidar()
    sdf_cfg = OmegaConf.create(dict(
        prune=dict(start=0, tau=0.2), loss=dict(gs2sdf_w=0.0, sdf2gs_w=0.0),
        symmetry=dict(enabled=symmetry, w=0.3, learn_plane=learn_plane, max_offset=0.1)))
    node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
    pts = (torch.rand(500, 3) * 2 - 1) * OBJ_HALF
    node.create_from_pcd({3: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                                  size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
    node.init_sdf_supervision(lidar)
    sp = node.sup_surf_p
    on_far_face = (sp[:, 1] > OBJ_HALF[1] - 0.02) & (sp[:, 2] < OBJ_HALF[2] - 0.05) & (sp[:, 0].abs() < OBJ_HALF[0] - 0.05)
    assert on_far_face.sum() == 0, "the +y face must be unobserved in this test"
    group = node.get_param_groups()[node.class_prefix + "sdf"]
    assert any(p is node.sdf_sym_offset for p in group) == (symmetry and learn_plane)
    opt = torch.optim.Adam(group, lr=1e-3)
    node.cur_frame = 0
    for step in range(800):
        node.step = step
        loss_dict = node.compute_reg_loss()
        loss = sum(v for k, v in loss_dict.items() if k.startswith("sdf"))
        opt.zero_grad(); loss.backward(); opt.step()
    assert ("sdf_sym_surface" in loss_dict) == symmetry
    with torch.no_grad():
        n = 2000
        ids = torch.zeros(n, dtype=torch.long, device=dev)
        xz = (torch.rand(n, 2) * 2 - 1) * torch.tensor([OBJ_HALF[0] * 0.8, OBJ_HALF[2] * 0.6])
        def at_y(y_lo, y_hi):
            y = y_lo + torch.rand(n) * (y_hi - y_lo)
            return node.query_sdf(torch.stack([xz[:, 0], y, xz[:, 1]], -1).to(dev), ids)
        gap_seen = at_y(-0.92, -0.78)      # between object and box on the observed side: free
        gap_unseen = at_y(0.78, 0.92)      # same gap on the side no ray reached
        face_unseen = node.query_sdf(torch.stack([xz[:, 0], torch.full((n,), float(OBJ_HALF[1])), xz[:, 1]], -1).to(dev), ids)
    return node, gap_seen, gap_unseen, face_unseen


for sym, learn in ((False, False), (True, False), (True, True)):
    node, gap_seen, gap_unseen, face_unseen = fit(sym, learn)
    print(f"symmetry={sym} learn_plane={learn}: observed-side gap free {(gap_seen > 0).float().mean():.2f}, "
          f"unobserved-side gap free {(gap_unseen > 0).float().mean():.2f} (mean sdf {gap_unseen.mean():+.3f}), "
          f"|sdf| on the unobserved face {face_unseen.abs().mean():.3f}, plane offset {node.symmetry_offset.item():+.3f}")
    assert (gap_seen > 0).float().mean() > 0.9
    if sym:
        assert (gap_unseen > 0).float().mean() > 0.9, "mirrored free space did not carve the unobserved side"
        assert face_unseen.abs().mean() < 0.08, "mirrored returns did not place the unobserved face"
        assert abs(node.symmetry_offset.item()) <= 0.1
    else:
        assert (gap_unseen > 0).float().mean() < 0.3, "without symmetry the unobserved side should stay at the box prior"

# checkpoints written before the option existed still load
sd = node.state_dict(); sd.pop("sdf_sym_offset")
node2 = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=node.sdf_cfg, scene_scale=30.0, device=dev)
node2.load_state_dict(sd)
assert node2.sdf_sym_offset.abs().sum() == 0

# ---------------------------------------------------------------- 1b. withheld side
# The box passes the sensor on its left in even frames and on its right in odd frames, so both
# flanks are observed. Withholding one side must remove every ray from that side, and the
# withheld returns then measure what symmetry does for a side without data.
from models.nodes.sdf_supervision import gather_instance_lidar, split_by_side


class TwoSideLidar(OneSideLidar):
    def __init__(self):
        super().__init__()
        flip = torch.tensor([1.0, -1.0, 1.0])
        for fi in range(1, F, 2):
            fr = self.frames[fi]
            fr["lidar_origins"] = fr["lidar_origins"] * flip.to(dev); fr["lidar_viewdirs"] = fr["lidar_viewdirs"] * flip.to(dev)
            self.poses[fi] = self.poses[fi].clone(); self.poses[fi][1, 3] *= -1


def withheld_node(symmetry: bool, mode="auto", min_returns=100, gs2sdf_w=0.0):
    torch.manual_seed(0)
    lidar = TwoSideLidar()
    sdf_cfg = OmegaConf.create(dict(
        prune=dict(start=0, tau=0.2), loss=dict(gs2sdf_w=gs2sdf_w, sdf2gs_w=0.0),
        supervision=dict(withhold_side=mode, withhold_min_returns=min_returns, withhold_band=0.4),
        symmetry=dict(enabled=symmetry, w=0.3)))
    node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
    pts = (torch.rand(500, 3) * 2 - 1) * OBJ_HALF
    node.create_from_pcd({3: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                                  size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
    node.init_sdf_supervision(lidar)
    return node, lidar


def flank(p, side):   # interior of the object's face on that side
    return (p[:, 1] * side > OBJ_HALF[1] - 0.02) & (p[:, 2] < OBJ_HALF[2] - 0.05) & (p[:, 0].abs() < OBJ_HALF[0] - 0.05)


node, lidar = withheld_node(False, mode=None)
assert node.sdf_withheld_side.abs().sum() == 0
assert flank(node.sup_surf_p, 1).sum() > 100 and flank(node.sup_surf_p, -1).sum() > 100, "both flanks must be observed"
node, _ = withheld_node(False, min_returns=10 ** 9)
assert node.sdf_withheld_side.abs().sum() == 0 and flank(node.sup_surf_p, 1).sum() > 100, "too few returns: nothing withheld"
for mode, expect in (("pos", 1), ("neg", -1), ("auto", None)):
    node, _ = withheld_node(False, mode=mode)
    side = int(node.sdf_withheld_side[0])
    assert side != 0 and (expect is None or side == expect)
    assert flank(node.sup_surf_p, side).sum() == 0 and flank(node.sup_surf_p, -side).sum() > 100
    assert (node.sup_ray_o[:, 1] * side > 0).sum() == 0, "a ray from the withheld side is still in the supervision"

res = {}
for sym in (False, True):
    node, lidar = withheld_node(sym, mode="pos")
    opt = torch.optim.Adam(node.get_param_groups()[node.class_prefix + "sdf"], lr=1e-3)
    node.cur_frame = 0
    for step in range(800):
        node.step = step
        loss = sum(v for k, v in node.compute_reg_loss().items() if k.startswith("sdf"))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        per_ins = gather_instance_lidar(lidar, list(range(F)), node.quat_act(node.instances_quats.data), node.instances_trans.data,
                                        node.instances_fv, node.instances_size, dev)
        held = torch.cat(split_by_side(per_ins[0], 1, keep_withheld=True)["surf_p"])
        held = held[flank(held, 1)]
        res[sym] = node.query_sdf(held, torch.zeros(len(held), dtype=torch.long, device=dev)).abs().mean().item()
print(f"withheld side: {len(held)} withheld flank returns, |sdf| there without symmetry {res[False]:.3f}, with symmetry {res[True]:.3f}")
assert res[False] > 0.15, "without symmetry the withheld flank should stay at the box prior"
assert res[True] < 0.08, "symmetry did not place the withheld flank"

# gaussians on the withheld side do not feed the SDF; the choice survives a checkpoint round trip
node, _ = withheld_node(False, mode="pos", gs2sdf_w=0.1)
node.step = 2000; node.cur_frame = 0
node._opacities.data.fill_(5.0)
node._means.data[:, 1] = 0.5
assert "sdf_gs2sdf" not in node.compute_reg_loss()
node._means.data[:, 1] = -0.5
assert "sdf_gs2sdf" in node.compute_reg_loss()
sd = node.state_dict()
node2 = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=node.sdf_cfg, scene_scale=30.0, device=dev)
node2.load_state_dict(dict(sd))
assert int(node2.sdf_withheld_side[0]) == 1
sd.pop("sdf_withheld_side")
node2 = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=node.sdf_cfg, scene_scale=30.0, device=dev)
node2.load_state_dict(sd)
assert node2.sdf_withheld_side.abs().sum() == 0
print("withheld side OK")

# ---------------------------------------------------------------- 1c. SDF seeds Gaussians
def seeded_node(**seed):
    torch.manual_seed(0)
    lidar = TwoSideLidar()
    sdf_cfg = OmegaConf.create(dict(
        prune=dict(start=0, tau=0.2), loss=dict(gs2sdf_w=0.0, sdf2gs_w=0.0), symmetry=dict(enabled=False),
        seed=dict(dict(enabled=True, start=0, stop=10 ** 6, interval=100, radius=0.05, support="lidar", support_radius=0.05,
                       require_camera=False, min_views=1, n_samples=60000, max_per_instance=10 ** 6), **seed)))
    node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
    pts = (torch.rand(300, 3) * 2 - 1) * OBJ_HALF * 0.3      # deep inside: they cover no surface
    node.create_from_pcd({3: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                                  size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
    node.init_sdf_supervision(lidar)
    opt = torch.optim.Adam(node.get_param_groups()[node.class_prefix + "sdf"], lr=1e-3)
    node.cur_frame = 0
    for step in range(800):
        node.step = step
        loss = sum(v for k, v in node.compute_reg_loss().items() if k.startswith("sdf"))
        opt.zero_grad(); loss.backward(); opt.step()
    return node


def gs_optimizer(node):
    opt = torch.optim.Adam([dict(params=v, name=k, lr=1e-3) for k, v in node.get_gaussian_param_groups().items()])
    sum(v[0].sum() for v in node.get_gaussian_param_groups().values()).backward(); opt.step(); opt.zero_grad()
    return opt


def gs_consistent(node, opt):
    n = node.num_points
    for name, (param,) in node.get_gaussian_param_groups().items():
        assert param.shape[0] == n and param.requires_grad, name
        group = [g for g in opt.param_groups if g["name"] == name][0]
        assert group["params"][0] is param, f"{name}: the optimizer still holds the old tensor"
        assert opt.state[param]["exp_avg"].shape == param.shape and opt.state[param]["exp_avg_sq"].shape == param.shape
    assert node.point_ids.shape == (n, 1)
    sum(v[0].sum() for v in node.get_gaussian_param_groups().values()).backward(); opt.step(); opt.zero_grad()


node = seeded_node()
opt = gs_optimizer(node)
n0, means0 = node.num_points, node._means.data.clone()
dc0 = node._features_dc.data.clone()
added = node.seed_gaussians_from_sdf(opt)
assert added > 500 and node.num_points == n0 + added, added
assert torch.equal(node._means.data[:n0], means0), "seeding must not move existing gaussians"
new_p = node._means.data[n0:]
zero_ids = torch.zeros(added, dtype=torch.long, device=dev)
assert node.query_sdf(new_p, zero_ids).abs().max() < 0.006, "seeds must sit on the zero level"
assert (new_p.abs() <= HALF.to(dev)).all() and not node.get_out_of_bound_mask()[n0:].any()
assert torch.cdist(new_p, node.sup_surf_p).min(1).values.max() <= 0.05 + 1e-4, "a seed without a LiDAR return nearby"
assert torch.cdist(new_p, means0).min(1).values.min() > 0.05, "a seed on a cell that was already covered"
def far_end(p):   # interior of the +x face, which no ray reaches
    return (p[:, 0] > OBJ_HALF[0] - 0.1) & (p[:, 1].abs() < OBJ_HALF[1] - 0.15) & (p[:, 2].abs() < OBJ_HALF[2] - 0.15)
assert far_end(node.sup_surf_p).sum() == 0 and far_end(new_p).sum() == 0, "the +x face has no returns: nothing may be seeded on it"
assert (torch.sigmoid(node._opacities.data[n0:]) - 0.1).abs().max() < 1e-5
assert (node._features_dc.data[n0:][:, None] == dc0[None]).all(-1).any(1).all(), "seed colours come from existing gaussians"
# the discs are thin along the surface normal
from models.gaussians.basics import quat_to_rotmat
_, normal = node.query_sdf_with_grad(new_p, zero_ids)
thin_axis = quat_to_rotmat(node._quats.data[n0:])[:, :, 2]
assert ((thin_axis * torch.nn.functional.normalize(normal.detach(), dim=-1)).sum(-1) > 0.99).all()
assert (node._scales.data[n0:, 2] < node._scales.data[n0:, 0]).all()
log = node.sdf_seed_log[-1]["instances"][0]
assert log["added"] == added == log["eligible"] and log["supported"] < log["cells"]
gs_consistent(node, opt)      # sizes, optimizer state, and an optimizer step on the grown model
# a second pass only fills what the first left: the surface is now mostly covered
again = node.seed_gaussians_from_sdf(opt)
gs_consistent(node, opt)
assert again < 0.5 * added, (added, again)
print(f"sdf seeding: {log['cells']} surface cells, {log['supported']} with LiDAR support, {added} seeded, {again} on a second pass")

# support = any also seeds the faces no ray reached; the per-event cap holds
node = seeded_node(support="any", max_per_instance=200)
n0 = node.num_points
assert node.seed_gaussians_from_sdf() == 200
node.sdf_cfg.seed.max_per_instance = 10 ** 6
node.seed_gaussians_from_sdf()
assert far_end(node._means.data[n0:]).sum() > 50, "support=any must reach the unobserved +x face"

# require_camera: one camera on the -y side, looking along +y. Only surface facing it is seeded.
node = seeded_node(require_camera=True)
n0 = node.num_points
assert node.seed_gaussians_from_sdf() == 0, "no camera views recorded: nothing may be seeded"
K = torch.tensor([[500.0, 0, 500], [0, 500, 500], [0, 0, 1]])
Rc = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])       # camera x = object x, y = -z (down), z = +y (forward)
c = torch.tensor([0.0, -10.0, 0.0])
node.sdf_cam_views = [dict(A=(K @ Rc)[None].to(dev), b=(-K @ Rc @ c)[None].to(dev), c=c[None].to(dev),
                           wh=torch.tensor([[1000.0, 1000.0]]).to(dev))]
added = node.seed_gaussians_from_sdf()
new_p = node._means.data[n0:]
assert added > 100 and (new_p[:, 1] < -OBJ_HALF[1] + 0.1).float().mean() > 0.95, "seeds must be on the face the camera sees"
node.sdf_cam_views[0]["wh"] = torch.tensor([[10.0, 10.0]]).to(dev)      # the object now projects outside the image
assert node.seed_gaussians_from_sdf() == 0
# the hook: seeding runs from refinement_after only inside [start, stop] and on the interval
node = seeded_node(start=500, stop=900, interval=200)
opt = gs_optimizer(node)
node.xys_grad_norm = node.vis_counts = node.max_2Dsize = None
calls = []
node.seed_gaussians_from_sdf = lambda optimizer=None: calls.append(node.step)
for step in range(100, 1300, 100):
    node.step = step
    node.xys_grad_norm = torch.zeros(node.num_points, device=dev); node.vis_counts = torch.ones(node.num_points, device=dev)
    node.max_2Dsize = torch.zeros(node.num_points, device=dev)
    node.refinement_after(step, opt)
assert calls == [600, 800], calls
print("sdf seeding OK")

# ---------------------------------------------------------------- 1d. lifted surface -> SDF
# The +y face is never hit by LiDAR. A surface lifted from monocular depth covers it (with
# noise), and also claims two things LiDAR contradicts: the observed -y face 4 cm too far in,
# and a sheet floating in the gap that LiDAR rays crossed. Only the first may reach the losses.
import tempfile
import numpy as np
from models.nodes.sdf_supervision import near_ray_segments
from datasets.driving_dataset import load_lifted_surface

o = torch.tensor([[-50.0, 30.0, 1.0], [-50.0, 30.0, 1.0]]); d = torch.nn.functional.normalize(torch.tensor([[5.0, -3.0, 0.0]] * 2), dim=-1)
t = torch.tensor([[55.0, 60.0], [0.0, 1.0]])      # the first segment passes the origin, the second ends far before it
q = torch.tensor([[0.0, 0.0, 1.02], [0.0, 0.0, 1.05], [0.0, 0.0, 1.0]]) + torch.tensor([[0, 0, 0], [0, 0, 0], [5.0, -3.0, 0.0]])
assert near_ray_segments(q, o, d, t, 0.03).tolist() == [True, False, False], "2 cm off the segment, 5 cm off, beyond its end"


def lifted_surface(n=4000):
    g = torch.Generator().manual_seed(1)
    xz = (torch.rand(3 * n, 2, generator=g) * 2 - 1) * torch.tensor([1.8, 0.6])
    y_far = OBJ_HALF[1] + 0.02 * torch.randn(n, generator=g)                  # the unobserved face, 2 cm noise
    y_near = torch.full((n,), -OBJ_HALF[1] + 0.04)                            # the observed face, biased
    y_gap = -OBJ_HALF[1] - 0.08 - 0.12 * torch.rand(n, generator=g)           # in the gap LiDAR saw through
    pts = torch.stack([xz[:, 0], torch.cat([y_far, y_near, y_gap]), xz[:, 1]], -1)
    normals = torch.tensor([[0.0, 1.0, 0.0]]).repeat(3 * n, 1); normals[n:, 1] = -1
    return pts, normals, n


def mono_node(use_mono: bool, **mono):
    torch.manual_seed(0)
    lidar = OneSideLidar()
    sdf_cfg = OmegaConf.create(dict(
        prune=dict(start=0, tau=0.2), loss=dict(gs2sdf_w=0.0, sdf2gs_w=0.0), symmetry=dict(enabled=False), mono=mono))
    node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
    gs = (torch.rand(500, 3) * 2 - 1) * OBJ_HALF
    node.create_from_pcd({3: dict(pts=gs, colors=torch.rand(len(gs), 3), poses=torch.stack(lidar.poses),
                                  size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(gs))})
    pts, normals, n = lifted_surface()
    # instance 7 is not modelled and must be ignored
    surface = {3: dict(pts=pts, normals=normals), 7: dict(pts=pts, normals=normals)} if use_mono else None
    node.init_sdf_supervision(lidar, mono_surface=surface)
    return node, n


def train_sdf(node, steps=800):
    opt = torch.optim.Adam(node.get_param_groups()[node.class_prefix + "sdf"], lr=1e-3)
    node.cur_frame = 0
    for step in range(steps):
        node.step = step
        loss_dict = node.compute_reg_loss()
        loss = sum(v for k, v in loss_dict.items() if k.startswith("sdf"))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss_dict


def unseen_side(node, n=2000):
    ids = torch.zeros(n, dtype=torch.long, device=dev)
    xz = (torch.rand(n, 2) * 2 - 1) * torch.tensor([OBJ_HALF[0] * 0.8, OBJ_HALF[2] * 0.6])
    face = torch.stack([xz[:, 0], torch.full((n,), float(OBJ_HALF[1])), xz[:, 1]], -1).to(dev)
    gap = torch.stack([xz[:, 0], 0.78 + 0.14 * torch.rand(n), xz[:, 1]], -1).to(dev)
    seen = torch.stack([xz[:, 0], torch.full((n,), -float(OBJ_HALF[1])), xz[:, 1]], -1).to(dev)
    sdf_face, grad = node.query_sdf_with_grad(face, ids)
    cos = torch.nn.functional.normalize(grad, dim=-1)[:, 1]
    with torch.no_grad():
        return sdf_face.detach().abs().mean().item(), cos.mean().item(), (node.query_sdf(gap, ids) > 0).float().mean().item(), \
            node.query_sdf(seen, ids).abs().mean().item()


node, _ = mono_node(False)
assert node.sup_mono_p is None
ld = train_sdf(node)
assert not any(k.startswith("sdf_mono") for k in ld)
face0, cos0, gap0, seen0 = unseen_side(node)

node, n = mono_node(True)
mp, mw = node.sup_mono_p, node.sup_mono_w
far, near, gap = mp[:, 1] > 0.5, (mp[:, 1] + OBJ_HALF[1] - 0.04).abs() < 1e-4, mp[:, 1] < -OBJ_HALF[1] - 0.05
assert (far | near | gap).all() and (node.sup_mono_id == 0).all()
print(f"lifted surface: kept {int(far.sum())}/{n} on the unobserved face, {int(near.sum())}/{n} next to LiDAR returns, "
      f"{int(gap.sum())}/{n} in observed free space; weights {mw.min():.3f}..{mw.max():.3f}")
assert far.sum() > 0.99 * n and (mw[far] == 1).all(), "the unobserved face must be kept at full weight"
assert near.sum() < 0.2 * n and (mw[near] < 0.2).all(), "lifted points next to LiDAR returns must be dropped or faded out"
assert gap.sum() < 0.3 * n, "lifted points in space LiDAR rays crossed must be dropped"
assert torch.allclose(node.sup_mono_n.norm(dim=-1), torch.ones_like(mw))
ld = train_sdf(node)
assert all(k in ld for k in ("sdf_mono_surface", "sdf_mono_normal", "sdf_mono_band"))
face1, cos1, gap1, seen1 = unseen_side(node)
print(f"unobserved face without / with the lifted surface: |sdf| {face0:.3f} / {face1:.3f}, gradient along the outward "
      f"normal {cos0:+.2f} / {cos1:+.2f}, gap free {gap0:.2f} / {gap1:.2f}; observed face |sdf| {seen0:.3f} / {seen1:.3f}")
assert face0 > 0.15 and gap0 < 0.3, "without lifted points the unobserved side stays at the box prior"
assert face1 < 0.04, "the lifted points did not place the unobserved face"
assert cos1 > 0.95, "the SDF gradient does not follow the lifted normals"
assert gap1 > 0.9, "the space in front of the lifted surface is not free"
assert seen1 < seen0 + 0.01, "the biased lifted points next to LiDAR moved the observed face"

node, _ = mono_node(True, free_tol=0.0, min_lidar_dist=0.0, full_lidar_dist=0.0)
assert node.sup_mono_p.shape[0] == 3 * n and (node.sup_mono_w == 1).all(), "both filters off: everything is kept"
node, _ = mono_node(True, surface_w=0.0, normal_w=0.0, band_w=0.0)
assert not any(k.startswith("sdf_mono") for k in train_sdf(node, steps=1))
node, _ = mono_node(True, start=5)
assert not any(k.startswith("sdf_mono") for k in train_sdf(node, steps=5)) and "sdf_mono_surface" in train_sdf(node, steps=6)

# the loader hands out normals that point out of the object, whichever convention the file has
with tempfile.TemporaryDirectory() as tmp:
    base = dict(pts=np.zeros((4, 3), np.float32), colors=np.full((4, 3), 0.5, np.float32), views=np.array([5, 5, 1, 5]),
                lidar_dist=np.ones(4, np.float32), instance=np.array([2, 2, 2, 9]))
    inward = np.tile(np.array([[0, 0, -1.0]], np.float32), (4, 1))
    np.savez(os.path.join(tmp, "old.npz"), normals=inward, **base)
    np.savez(os.path.join(tmp, "new_{scene_idx:03d}.npz".format(scene_idx=23)), normals=-inward, normals_outward=True, **base)
    old = load_lifted_surface(os.path.join(tmp, "old.npz"))
    new = load_lifted_surface(os.path.join(tmp, "new_{scene_idx:03d}.npz"), scene_idx=23)
    for surf in (old, new):
        assert sorted(surf) == [2, 9] and surf[2]["pts"].shape[0] == 2, "points seen by too few views are left out"
        assert (surf[2]["normals"][:, 2] == 1).all()
    np.savez(os.path.join(tmp, "facing.npz"), normals=-inward, normals_outward=True, facing=np.array([4, 0, 4, 2]), **base)
    assert load_lifted_surface(os.path.join(tmp, "facing.npz"))[2]["facing"].tolist() == [4, 0]
    surf = load_lifted_surface(os.path.join(tmp, "facing.npz"), min_facing=3)
    assert sorted(surf) == [2] and surf[2]["pts"].shape[0] == 1, "points whose normal faces too few views are left out"
    try:
        load_lifted_surface(os.path.join(tmp, "old.npz"), min_facing=3); raise RuntimeError("a file without 'facing' must be refused")
    except AssertionError:
        pass
print("lifted surface -> SDF OK")

# ---------------------------------------------------------------- 1e. coupling
# (i) Masks -> SDF. With an empty prior and no LiDAR term, the silhouettes alone must build a
# solid: rays through the mask have to meet it, rays outside the mask keep the space free.
import json


class MaskCamera:
    """Pinhole at the LiDAR origin looking along +x (x right = -y world, y down = -z world)."""
    WIDTH = HEIGHT = 400
    load_size = (400, 400)
    undistort = False
    start_timestep = 0

    def __init__(self):
        K = torch.tensor([[200.0, 0, 200], [0, 200, 200], [0, 0, 1]])
        c2w = torch.eye(4)
        c2w[:3, :3] = torch.tensor([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        c2w[:3, 3] = torch.tensor([0.0, 0.0, 1.0])
        self.intrinsics, self.cam_to_worlds = K[None].repeat(F, 1, 1), c2w[None].repeat(F, 1, 1)

    def render_mask(self, centre):
        v, u = torch.meshgrid(torch.arange(self.HEIGHT), torch.arange(self.WIDTH), indexing="ij")
        d_cam = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)], -1).reshape(-1, 3).float() @ torch.linalg.inv(self.intrinsics[0]).T
        d = torch.nn.functional.normalize(d_cam @ self.cam_to_worlds[0, :3, :3].T, dim=-1)
        hit, _ = first_hit(self.cam_to_worlds[0, :3, 3].expand_as(d), d, centre, OBJ_HALF)
        return hit.reshape(self.HEIGHT, self.WIDTH)


class MaskPixels:
    def __init__(self, cam):
        self.camera_data = {0: cam}
        self.instances_true_id = torch.arange(10) + 100      # dataset instance 3 has true id 103


def silhouette_node(tmp, **loss):
    torch.manual_seed(0)
    lidar, cam = OneSideLidar(), MaskCamera()
    frames = {}
    for fi in range(F):
        frames[str(fi)] = os.path.join(tmp, f"mask_{fi}.npz")
        np.savez(frames[str(fi)], mask=(cam.render_mask(lidar.poses[fi][:3, 3]).numpy() * 255).astype(np.uint8))
    json.dump({"records": {"103": {"cameras": {"0": frames}}}}, open(os.path.join(tmp, "masks_index.json"), "w"))
    sdf_cfg = OmegaConf.create(dict(
        net=dict(box_shrink=10.0, box_clamp=0.03),            # the prior is a 10 cm speck: the field starts empty
        prune=dict(start=0, tau=0.2), symmetry=dict(enabled=False),
        loss=dict(surface_w=0.0, behind_w=0.0, free_w=0.0, gs2sdf_w=0.0, sdf2gs_w=0.0, **loss)))
    node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
    gs = (torch.rand(500, 3) * 2 - 1) * OBJ_HALF
    node.create_from_pcd({3: dict(pts=gs, colors=torch.rand(len(gs), 3), poses=torch.stack(lidar.poses),
                                  size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(gs))})
    node.init_sdf_supervision(lidar, pixel_source=MaskPixels(cam), masks_root=tmp)
    return node


def mask_ray_stats(node, rays):
    o, d, t, ids = rays
    with torch.no_grad():
        miss = (node.mask_ray_min_sdf(o, d, t, ids, 32) > 0).float().mean().item()
        co, cd, ct = node.sup_carve_o, node.sup_carve_d, node.sup_carve_t
        tt = ct[:, :1] + torch.rand(ct.shape[0], 8, device=dev) * (ct[:, 1:] - ct[:, :1])
        solid = (node.query_sdf((co[:, None] + cd[:, None] * tt[..., None]).reshape(-1, 3), node.sup_carve_id.repeat_interleave(8)) < 0).float().mean().item()
        centre = node.query_sdf(torch.zeros(1, 3, device=dev), torch.zeros(1, dtype=torch.long, device=dev)).item()
    return miss, solid, centre


with tempfile.TemporaryDirectory() as tmp:
    node = silhouette_node(tmp, inside_w=0.5)
    obj = OBJ_HALF.to(dev)
    io, idr, it = node.sup_inside_o, node.sup_inside_d, node.sup_inside_t
    assert io.shape[0] > 1000 and node.sup_carve_o.shape[0] > 1000
    hit_in, t_obj = first_hit(io, idr, torch.zeros(3, device=dev), obj)
    assert hit_in.all(), "every mask ray must cross the object"
    assert ((t_obj >= it[:, 0] - 1e-3) & (t_obj <= it[:, 1] + 1e-3)).all(), "... inside the interval it is given"
    assert not first_hit(node.sup_carve_o, node.sup_carve_d, torch.zeros(3, device=dev), obj)[0].any(), "no carving ray may cross the object"
    rays = (io, idr, it, node.sup_inside_id)
    miss0, solid0, centre0 = mask_ray_stats(node, rays)
    ld = train_sdf(node)
    assert "sdf_inside" in ld and "sdf_carve" in ld and "sdf_surface" not in ld
    miss1, solid1, centre1 = mask_ray_stats(node, rays)
    print(f"silhouettes only, before / after: mask rays that miss the solid {miss0:.2f} / {miss1:.2f}, carving-ray samples "
          f"in the solid {solid0:.3f} / {solid1:.3f}, sdf at the object centre {centre0:+.2f} / {centre1:+.2f}")
    assert miss0 > 0.9 and miss1 < 0.03, "mask rays did not build the solid"
    assert solid1 < 0.03, "the solid spilled into space the carving rays saw through"
    assert centre1 < 0, "the object centre must be inside"
    held = silhouette_node(tmp)                                # inside_w 0: no mask rays are gathered, the field stays empty
    assert held.sup_inside_o is None and held.sup_carve_o is not None
    assert "sdf_inside" not in train_sdf(held, steps=200) and mask_ray_stats(held, rays)[0] > 0.9

# (ii) SDF -> Gaussians: thinnest axis along the SDF normal, |sdf| or sdf^2 pull. The untrained
# field is the box shrunk by 5 cm, so 8 cm below the top face the normal is +z and sdf = -0.03.
def flat_node(**loss):
    torch.manual_seed(0)
    lidar = OneSideLidar()
    sdf_cfg = OmegaConf.create(dict(prune=dict(start=0, tau=0.2), symmetry=dict(enabled=False),
                                    loss={**dict(gs2sdf_w=0.0, sdf2gs_w=0.0), **loss}))
    node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
    xy = (torch.rand(400, 2) * 2 - 1) * (HALF[:2] - 0.3)
    near_top = torch.cat([xy[:300], torch.full((300, 1), float(HALF[2]) - 0.08)], -1)
    deep = torch.cat([xy[300:] * 0.3, torch.zeros(100, 1)], -1)                # 0.7 m from the surface: outside the band
    pts = torch.cat([near_top, deep])
    node.create_from_pcd({3: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                                  size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
    node._scales.data[:] = torch.log(torch.tensor([0.10, 0.02, 0.10], device=dev))    # thin along the local y axis
    node._opacities.data[:] = 2.0
    node.cur_frame, node.step = 0, 10
    return node


node = flat_node(align_w=1.0, align_start=0)
idx_top, idx_deep = torch.arange(300, device=dev), torch.arange(300, 400, device=dev)
up = torch.tensor([[0.0, 0.0, 1.0]], device=dev)
with torch.no_grad():
    sdf_top, n_top = node.query_sdf_normal(node._means[idx_top], torch.zeros(300, dtype=torch.long, device=dev))
    assert torch.allclose(sdf_top, torch.full_like(sdf_top, -0.03), atol=1e-3) and torch.allclose(n_top, up.expand(300, 3), atol=1e-3)
    cos0, deep0 = node.short_axis_cos(idx_top, up).mean().item(), node.short_axis_cos(idx_deep, up).clone()
means0, scales0, quats_deep0 = node._means.data.clone(), node._scales.data.clone(), node._quats.data[idx_deep].clone()
opt = torch.optim.Adam([node._quats, node._means, node._scales] + list(node.sdf_nets.parameters()), lr=2e-2)
for _ in range(400):
    loss = node.compute_reg_loss()["sdf_align"]
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    cos1 = node.short_axis_cos(idx_top, up).mean().item()
print(f"alignment of the thinnest axis with the SDF normal: |cos| {cos0:.2f} -> {cos1:.2f}")
assert cos0 < 0.65 and cos1 > 0.98, "the thinnest axis did not turn along the SDF normal"
assert torch.equal(node._means.data, means0) and torch.equal(node._scales.data, scales0), "alignment may only turn gaussians"
assert torch.equal(node._quats.data[idx_deep], quats_deep0), "gaussians away from the surface must be left alone"
assert all(p.grad is None or p.grad.abs().sum() == 0 for p in node.sdf_nets.parameters()), "alignment must not train the SDF"
assert "sdf_align" not in flat_node().compute_reg_loss(), "align_w defaults to 0"

for power, expect in ((1, 0.03), (2, 0.03 ** 2)):
    node = flat_node(sdf2gs_w=1.0, sdf2gs_start=0, sdf2gs_power=power)
    node._opacities.data[300:] = -20.0                      # only the gaussians 3 cm inside the surface count
    got = node.compute_reg_loss()["sdf_sdf2gs"].item()
    assert abs(got - expect) < 0.05 * expect, f"sdf2gs power {power}: {got} vs {expect}"

# (iii) LiDAR -> SDF orientation: the normal at a return faces the sensor
def facing_node(**loss):
    torch.manual_seed(0)
    lidar = OneSideLidar()
    sdf_cfg = OmegaConf.create(dict(prune=dict(start=0, tau=0.2), symmetry=dict(enabled=False),
                                    loss={**dict(gs2sdf_w=0.0, sdf2gs_w=0.0), **loss}))
    node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
    gs = (torch.rand(500, 3) * 2 - 1) * OBJ_HALF
    node.create_from_pcd({3: dict(pts=gs, colors=torch.rand(len(gs), 3), poses=torch.stack(lidar.poses),
                                  size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(gs))})
    node.init_sdf_supervision(lidar)
    return node


def faces_sensor(node):
    _, normal = node.query_sdf_normal(node.sup_surf_p, node.sup_surf_id)
    return ((normal * node.sup_surf_d).sum(-1) < 0).float().mean().item()


assert "sdf_facing" not in train_sdf(facing_node(), steps=1), "facing_w defaults to 0"
node = facing_node(facing_w=0.1)
ld = train_sdf(node)
good = ld["sdf_facing"].item()
face = faces_sensor(node)
node.sup_surf_d = -node.sup_surf_d                          # pretend every ray came from inside the object
node.cur_frame, node.step = 0, 800
flipped = node.compute_reg_loss()["sdf_facing"].item()
print(f"normals at returns facing the sensor {face:.3f}; facing loss {good:.4f}, with reversed rays {flipped:.4f}")
assert face > 0.97 and good < 0.005 and flipped > 0.04
print("coupling OK")

# ---------------------------------------------------------------- 2. free-space grid
class WallLidar:
    """Sensor at (0,0,1.5). A wall at x=10, the ground at z=0, and in frame 1 a 'moving' box
    centred at (5, 0, 0.75) that is listed as an excluded instance."""
    def __init__(self):
        self.frames = []
        for fi in range(4):
            targets = torch.rand(40000, 3) * torch.tensor([0.0, 16, 5.0]) + torch.tensor([10.0, -8, 0])
            targets = torch.cat([targets, torch.rand(40000, 3) * torch.tensor([9.0, 16, 0.0]) + torch.tensor([1.0, -8, 0])])
            o = torch.tensor([0.0, 0.0, 1.5]).expand_as(targets)
            d = targets - o; r = d.norm(dim=-1); d = d / r[:, None]
            if fi == 1:
                hit, t_box = first_hit(o, d, torch.tensor([5.0, 0.0, 0.75]), torch.tensor([1.0, 0.5, 0.75]))
                r = torch.where(hit & (t_box < r), t_box, r)
            self.frames.append(dict(lidar_origins=o, lidar_viewdirs=d, lidar_ranges=r[:, None]))

    def get_lidar_rays(self, fi):
        return self.frames[fi]


pose = torch.eye(4).repeat(4, 1, 1, 1); pose[:, 0, :3, 3] = torch.tensor([5.0, 0.0, 0.75])
active = torch.zeros(4, 1, dtype=torch.bool); active[1, 0] = True
boxes = dict(pose=pose, size=torch.tensor([[2.0, 1.0, 1.5]]), active=active, ids=[0])
wall = WallLidar()
grid = FreeSpaceGrid.build(wall, range(4), dev, voxel=0.25, exclude_boxes=boxes)
grid_all_static = FreeSpaceGrid.build(wall, range(4), dev, voxel=0.25, exclude_boxes=None)

mid_air = torch.rand(2000, 3) * torch.tensor([6.0, 6, 0.6]) + torch.tensor([2.0, -3, 0.8])   # between sensor and wall
on_wall = torch.rand(2000, 3) * torch.tensor([0.0, 8, 3.0]) + torch.tensor([10.0, -4, 0.5])
on_ground = torch.rand(2000, 3) * torch.tensor([6.0, 8, 0.0]) + torch.tensor([2.0, -4, 0.02])
behind = torch.rand(2000, 3) * torch.tensor([3.0, 8, 3.0]) + torch.tensor([11.0, -4, 0.5])
in_box = torch.rand(2000, 3) * torch.tensor([0.15, 0.6, 0.5]) + torch.tensor([4.02, -0.3, 0.8])  # the mover's front face in frame 1
f = lambda g, p: g.is_free(p.to(dev)).float().mean().item()
print(f"free grid: mid-air {f(grid, mid_air):.2f}, wall {f(grid, on_wall):.2f}, ground {f(grid, on_ground):.2f}, "
      f"behind wall {f(grid, behind):.2f}, mover's box {f(grid, in_box):.2f} (as static: {f(grid_all_static, in_box):.2f})")
assert f(grid, mid_air) > 0.95 and f(grid, on_wall) == 0 and f(grid, on_ground) == 0 and f(grid, behind) == 0
assert f(grid, in_box) > 0.9, "space a modelled mover passed through should be free of static matter"
assert f(grid_all_static, in_box) < 0.2, "without the exclusion its returns must keep the space solid"
assert grid.is_observed(on_wall.to(dev)).all() and not grid.is_observed(behind.to(dev)).any()

# background gaussians: floaters in mid-air get an opacity loss and are culled, the rest stay
reg = OmegaConf.create(dict(free_space=dict(w=1.0, start=0, prune=True, prune_start=0)))
bg = VanillaGaussians(class_name="Background", ctrl=ctrl, reg=reg, scene_scale=30.0, device=dev)
floaters = mid_air[grid.is_free(mid_air.to(dev)).cpu()][:300]
pts = torch.cat([on_wall, on_ground, floaters])
bg.create_from_pcd(init_means=pts.to(dev), init_colors=torch.rand(len(pts), 3).to(dev))
bg.step = 10
assert "free_space" not in bg.compute_reg_loss(), "no grid, no loss"
bg.set_free_space_grid(grid)
loss = bg.compute_reg_loss()["free_space"]
loss.backward()
g = bg._opacities.grad.squeeze(-1)
assert (g[:4000] == 0).all() and (g[4000:] > 0).all(), "only the floaters' opacity is pushed down"
mask = bg.free_space_prune_mask()
print(f"background: loss {loss.item():.3f}, pruned {int(mask[:4000].sum())}/4000 surface gaussians, {int(mask[4000:].sum())}/300 floaters")
assert mask[:4000].sum() == 0 and mask[4000:].all()
n_before = bg.num_points
bg.cull_gaussians()
assert bg.num_points == n_before - 300
print("ALL OK")
