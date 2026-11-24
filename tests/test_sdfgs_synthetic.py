"""Synthetic check of SDFGSNodes: one box object, fake lidar, SDF-only optimisation.

Run from the repo root:  PYTHONPATH=$(pwd) SDFGS_DEVICE=cpu python tests/test_sdfgs_synthetic.py $(pwd)
"""
import os, sys, tempfile
import torch
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, sys.argv[1])
from models.nodes import SDFGSNodes

torch.manual_seed(0)
dev = torch.device(os.environ.get("SDFGS_DEVICE", "cuda"))
SIZE = torch.tensor([4.0, 2.0, 1.5])            # box length, width, height
HALF = SIZE / 2
OBJ_HALF = HALF - torch.tensor([0.5, 0.0, 0.0])  # the solid object is shorter than its annotated box
F = 6                                            # frames


def box_surface_points(n, half=OBJ_HALF):
    """Uniform points on the surface of the box [-half, half]."""
    face = torch.randint(0, 6, (n,))
    p = (torch.rand(n, 3) * 2 - 1) * half
    axis, sign = face // 2, (face % 2) * 2 - 1
    p[torch.arange(n), axis] = sign.float() * half[axis]
    # the sensor never sees the back face (+x); LiDAR-seeded gaussians would not be there either
    return p[p[:, 0] < half[0] - 1e-3]


class FakeLidar:
    """Sensor at the origin; a solid box 10 m ahead slides sideways with the frame.
    Each ray returns its first hit: the box, else the ground plane z=-HALF.z, else 30 m."""
    def __init__(self):
        self.frames, self.poses = [], []
        for fi in range(F):
            R = torch.eye(3); t = torch.tensor([10.0, 0.5 * fi, 0.0])
            targets = torch.rand(12000, 3) * torch.tensor([20, 8, 3.0]) + torch.tensor([4, -4, -HALF[2]])
            o = torch.tensor([0.0, 0.0, 1.0]).expand_as(targets)
            d = targets - o
            d = d / d.norm(dim=-1, keepdim=True)
            # solid box (axis aligned here)
            d_safe = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
            t1 = (t - OBJ_HALF - o) / d_safe; t2 = (t + OBJ_HALF - o) / d_safe
            t_in = torch.minimum(t1, t2).max(-1).values; t_out = torch.maximum(t1, t2).min(-1).values
            hit_box = (t_out > t_in) & (t_in > 0)
            # ground plane
            t_g = (-HALF[2] - o[:, 2]) / d_safe[:, 2]
            hit_ground = (d[:, 2] < 0) & (t_g > 0)
            r = torch.full((len(d),), 30.0)
            r = torch.where(hit_ground, t_g, r)
            r = torch.where(hit_box, t_in, r)
            self.frames.append(dict(lidar_origins=o.to(dev), lidar_viewdirs=d.to(dev), lidar_ranges=r[:, None].to(dev)))
            self.poses.append(torch.cat([torch.cat([R, t[:, None]], 1), torch.tensor([[0, 0, 0, 1.0]])]))

    def get_lidar_rays(self, fi):
        return self.frames[fi]


lidar = FakeLidar()
ctrl = OmegaConf.create(dict(
    warmup_steps=0, reset_alpha_interval=3000, refine_interval=100, sh_degree_interval=1000,
    n_split_samples=2, reset_alpha_value=0.01, densify_grad_thresh=5e-4, densify_size_thresh=3e-3,
    cull_alpha_thresh=0.005, cull_scale_thresh=0.1, cull_screen_size=0.15, split_screen_size=0.05,
    stop_screen_size_at=30000, stop_split_at=20000, sh_degree=3, cull_out_of_bound=True))
sdf_cfg = OmegaConf.create(dict(prune=dict(start=0, tau=0.2),
                                loss=dict(gs2sdf_start=0, sdf2gs_start=0, sdf2gs_w=0.0, gs2sdf_w=float(os.environ.get("GS2SDF_W", "0.1")))))
node = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg,
                  scene_scale=30.0, device=dev)

# gaussians: box surface points plus a cloud of far-off junk that the SDF should prune
good = box_surface_points(3000)
junk = (torch.rand(500, 3) * 2 - 1) * HALF * 0.6   # deep inside the box, >0.2 m from any face
pts = torch.cat([good, junk])
node.create_from_pcd({7: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                              size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
node.init_sdf_supervision(lidar)
assert node.sup_surf_p is not None and node.sup_ray_o is not None
print(f"supervision: {node.sup_ray_o.shape[0]} rays, {node.sup_surf_p.shape[0]} surface returns")
assert (node.sup_surf_p.abs() <= HALF.to(dev) + 1e-4).all()
assert (node.sup_surf_p[:, 2] > -HALF[2] + 0.12 - 1e-4).all(), "ground band leaked into surface set"

opt = torch.optim.Adam(node.get_param_groups()[node.class_prefix + "sdf"], lr=1e-3)
node.cur_frame = 0
for step in range(800):
    node.step = step
    loss_dict = node.compute_reg_loss()
    loss = sum(v for k, v in loss_dict.items() if k.startswith("sdf"))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 150 == 0 or step == 799:
        print(f"step {step:3d} " + " ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items()))

with torch.no_grad():
    ids = torch.zeros(2000, dtype=torch.long, device=dev)
    surf_pts = box_surface_points(2000).to(dev)
    s_surf = node.query_sdf(surf_pts, ids[: len(surf_pts)])
    inside = ((torch.rand(2000, 3) * 2 - 1) * OBJ_HALF * 0.5).to(dev)   # at least 0.37 m inside
    # the slab between the object's front face and the annotated box must be carved out
    carve = torch.rand(2000, 3) * torch.tensor([0.35, 1.2, 0.9]) + torch.tensor([-1.95, -0.6, -0.45])
    s_carve = node.query_sdf(carve.to(dev), ids)
    s_in = node.query_sdf(inside, ids)
    outside = box_surface_points(2000) * 1.25                            # 0.19-0.5 m outside, in front
    outside = outside[outside[:, 0] < 0].to(dev)                         # visible (sensor) side
    s_out = node.query_sdf(outside, torch.zeros(len(outside), dtype=torch.long, device=dev))
print(f"carved slab: pos frac {(s_carve>0).float().mean():.2f} mean {s_carve.mean():.3f}")
assert (s_carve > 0).float().mean() > 0.9
print(f"surface |sdf| mean {s_surf.abs().mean():.3f}  inside mean {s_in.mean():.3f} (neg frac {(s_in<0).float().mean():.2f})  "
      f"outside mean {s_out.mean():.3f} (pos frac {(s_out>0).float().mean():.2f})")
assert s_surf.abs().mean() < 0.08
assert (s_out > 0).float().mean() > 0.9

# pruning: junk gaussians deep inside the box should go, surface ones stay
node.step = 100
mask = node.sdf_prune_mask()
n_good = len(good)
print(f"prune: {int(mask[:n_good].sum())}/{n_good} surface gaussians, {int(mask[n_good:].sum())}/500 interior junk")
assert mask[n_good:].float().mean() > 0.6 and mask[:n_good].float().mean() < 0.1

# mesh export and checkpoint round-trip
out = tempfile.mkdtemp()
node.export_meshes(out, 100, resolution=48)
files = os.listdir(out); print("meshes:", files); assert files
sd = node.state_dict()
node2 = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_cfg, scene_scale=30.0, device=dev)
node2.load_state_dict(sd)
with torch.no_grad():
    assert torch.allclose(node2.query_sdf(inside, ids), s_in)
# supervision.follow_pose: the stored rays/returns follow the refined poses (world points stay put)
from models.gaussians.basics import quat_to_rotmat
sdf_f = OmegaConf.merge(sdf_cfg, dict(supervision=dict(follow_pose=True, follow_pose_every=1)))
node3 = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_f, scene_scale=30.0, device=dev)
node3.create_from_pcd({7: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                               size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
node3.init_sdf_supervision(lidar)
fr = node3._sup_follow["frames"]["surf"]
def to_world(nd, p, fr, is_dir=False):
    q = nd.quat_act(nd.instances_quats.data)[fr, 0]; R = quat_to_rotmat(q)
    out = torch.einsum("nk,njk->nj", p, R)
    return out if is_dir else out + nd.instances_trans.data[fr, 0]
w_before = to_world(node3, node3.sup_surf_p, fr); dw_before = to_world(node3, node3.sup_surf_d, fr, True)
c_before = to_world(node3, node3.sup_carve_o, node3._sup_follow["frames"]["carve"]) if node3.sup_carve_o is not None else None
r_before = to_world(node3, node3.sup_ray_o, node3._sup_follow["frames"]["ray"])
with torch.no_grad():   # "refinement": shift frame 2 by 0.3 m and yaw it by 0.1 rad
    node3.instances_trans.data[2, 0] += torch.tensor([0.3, -0.1, 0.05], device=dev)
    yaw = torch.tensor([0.99875, 0.0, 0.0, 0.04998], device=dev)   # wxyz, ~0.1 rad about z
    from models.gaussians.basics import quat_mult
    node3.instances_quats.data[2, 0] = quat_mult(yaw[None], node3.instances_quats.data[2, 0][None])[0]
p_old = node3.sup_surf_p.clone()
node3.step = 5; node3.follow_refined_poses()
moved = fr == 2
assert (node3.sup_surf_p[moved] - p_old[moved]).norm(dim=-1).min() > 0.05, "frame-2 returns did not move"
assert torch.allclose(node3.sup_surf_p[~moved], p_old[~moved], atol=1e-5), "other frames moved"
assert torch.allclose(to_world(node3, node3.sup_surf_p, fr), w_before, atol=1e-4), "returns left their world position"
assert torch.allclose(to_world(node3, node3.sup_surf_d, fr, True), dw_before, atol=1e-5)
assert torch.allclose(to_world(node3, node3.sup_ray_o, node3._sup_follow["frames"]["ray"]), r_before, atol=1e-4)
node3.cur_frame = 0; node3.compute_reg_loss()   # runs with the re-placed data
print(f"follow_pose OK ({int(moved.sum())} frame-2 returns re-placed)")
# sdf.register: a frame whose pose is off by 6 cm / 1.5 deg is pulled back by registration to the trained SDF
sdf_r = OmegaConf.merge(sdf_cfg, dict(supervision=dict(follow_pose=True, follow_pose_every=1),
                                      register=dict(enabled=True, start=0, every=1, iters=150, step=1.0, max_trans=0.2, max_rot_deg=5.0)))
node5 = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_r, scene_scale=30.0, device=dev)
node5.create_from_pcd({7: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                               size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
node5.sdf_nets.load_state_dict(node.sdf_nets.state_dict())
node5.init_sdf_supervision(lidar)
t_true = node5.instances_trans.data[2, 0].clone(); q_true = node5.instances_quats.data[2, 0].clone()
with torch.no_grad():
    node5.instances_trans.data[2, 0] += torch.tensor([0.0, 0.06, -0.03], device=dev)   # sideways and down
    yaw = torch.tensor([0.99991, 0.0, 0.0, 0.01309], device=dev)                      # 1.5 deg about z
    node5.instances_quats.data[2, 0] = quat_mult(yaw[None], node5.instances_quats.data[2, 0][None])[0]
err0 = (node5.instances_trans.data[2, 0] - t_true).norm().item()
node5.step = 0
info = node5.register_poses_to_sdf(force=True)
err1 = (node5.instances_trans.data[2, 0] - t_true).norm().item()
qa = node5.quat_act(node5.instances_quats.data[2, 0]); ang = 2 * torch.acos((qa * q_true / q_true.norm()).sum().abs().clamp(max=1)).item()
other = (node5.instances_trans.data[0, 0] - torch.tensor([10.0, 0.0, 0.0], device=dev)).norm().item()
print(f"register: frame-2 offset {err0 * 100:.1f} cm -> {err1 * 100:.1f} cm, rotation error {np.degrees(ang):.2f} deg, frame 0 moved {other * 100:.2f} cm; {info}")
assert err1 < 0.4 * err0 and np.degrees(ang) < 0.8 and other < 0.02
# register.anchor: same offset, but the corrections of the instance average to zero
sdf_a = OmegaConf.merge(sdf_r, dict(register=dict(anchor=True)))
node6 = SDFGSNodes(class_name="RigidNodes", ctrl=ctrl, reg=OmegaConf.create({}), sdf=sdf_a, scene_scale=30.0, device=dev)
node6.create_from_pcd({7: dict(pts=pts, colors=torch.rand(len(pts), 3), poses=torch.stack(lidar.poses),
                               size=SIZE, frame_info=torch.ones(F, dtype=torch.bool), num_pts=len(pts))})
node6.sdf_nets.load_state_dict(node.sdf_nets.state_dict())
node6.init_sdf_supervision(lidar)
with torch.no_grad():
    node6.instances_trans.data[2, 0] += torch.tensor([0.0, 0.06, -0.03], device=dev)
t0_a = node6.instances_trans.data.clone()
node6.step = 0
info = node6.register_poses_to_sdf(force=True)
d_a = node6.instances_trans.data - t0_a
print(f"anchor: {info}, mean translation change {d_a.mean(0).norm() * 100:.3f} cm, frame 2 moved {d_a[2, 0].norm() * 100:.2f} cm")
assert info["anchor_removed"] > 0 and d_a[:, 0].mean(0).norm() < 0.01 and d_a[2, 0].norm() > 0.5 * 0.067
# net.c2f: bands fade in over the training steps, an evaluated network has them all
from models.nodes.sdfgs import InstanceSDF
c2f = InstanceSDF(torch.tensor([2.0, 1.0, 0.8]), frequencies=4, c2f_start=0, c2f_end=100).to(dev)
c2f.set_step(0); assert c2f.band_weights.abs().sum() == 0
c2f.set_step(50); w_mid = c2f.band_weights.tolist(); assert w_mid[0] == 1.0 and w_mid[1] == 1.0 and w_mid[2] == 0.0 and w_mid[3] == 0.0, w_mid
c2f.set_step(100); assert (c2f.band_weights == 1).all()
x_c = torch.rand(64, 3, device=dev)
full_net = InstanceSDF(torch.tensor([2.0, 1.0, 0.8]), frequencies=4).to(dev); full_net.load_state_dict(c2f.state_dict())
assert torch.allclose(c2f(x_c), full_net(x_c))
c2f.set_step(0); low = c2f(x_c); c2f.set_step(100); assert not torch.allclose(low, c2f(x_c))
print("ALL OK")
