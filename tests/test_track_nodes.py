"""TrackNodes follow 3D tracks.  Run from the repo root:  PYTHONPATH=$(pwd) python tests/test_track_nodes.py $(pwd)"""
import os, sys, tempfile
import numpy as np, torch
from omegaconf import OmegaConf
sys.path.insert(0, sys.argv[1])
from models.nodes import TrackNodes
from models.gaussians.basics import quat_to_rotmat

torch.manual_seed(0); np.random.seed(0)
dev = torch.device(os.environ.get("SDFGS_DEVICE", "cuda"))
F_, H_, G, P = 16, 2, 600, 300

def sphere(n, r=0.5):
    x = np.random.randn(n, 3); return r * x / np.linalg.norm(x, axis=1, keepdims=True)
def yaw(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

full = np.stack([sphere(G), sphere(G) * 1.2])
ctrl_idx = np.arange(P)
canon = full[:, ctrl_idx]
angles = np.arange(F_) * 0.1
offset = np.array([[1.0, 0.5, 0.2], [-2.0, 1.0, 0.0]])
vel = np.array([[0.3, 0.0, 0.1], [0.0, -0.2, 0.0]])
tracks = np.zeros((F_, H_, P, 3), np.float32)
for f in range(F_):
    R = yaw(angles[f])
    for h in range(H_):
        tracks[f, h] = canon[h] @ R.T + offset[h] + f * vel[h]
present = np.ones((F_, H_), bool); present[0, 1] = False
tracks[4] = 99.0                                     # the held-out frame's own track must not be used
tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "t.npz")
from scipy.spatial import ConvexHull
faces = ConvexHull(full[0]).simplices
np.savez(path, tracks_world=tracks, present=present, canonical=canon, canonical_full=full, faces=faces, vertex_ids=ctrl_idx, frame_is_test=np.arange(F_) == 4, person_dataset_id=np.array([3, 8]))

ctrl = OmegaConf.create(dict(warmup_steps=0, reset_alpha_interval=3000, refine_interval=100, sh_degree_interval=1000, n_split_samples=2, reset_alpha_value=0.1,
    densify_grad_thresh=5e-4, densify_size_thresh=3e-3, cull_alpha_thresh=0.005, cull_scale_thresh=0.1, cull_screen_size=0.15, split_screen_size=0.05,
    stop_screen_size_at=30000, stop_split_at=20000, sh_degree=1, opacity_init_value=0.99, ball_gaussians=False, constrain_xyz_offset=False, knn_update_interval=100,
    knn_neighbors=3, use_voxel_deformer=False, freeze_x=False, freeze_o=False, freeze_q=False, freeze_s=False, freeze_shs_dc=False, freeze_shs_rest=False,
    tracks_path=path, n_kabsch=12, bind_k=4, test_interp="cubic"))
make = lambda: TrackNodes(class_name="SMPLNodes", ctrl=ctrl, reg=OmegaConf.create({}), scene_scale=30.0, device=dev)
node = make()
d = {k: dict(size=torch.tensor([1.0, 1.0, 1.0])) for k in (3, 8)}
node.create_from_pcd(d)

def world(f, test=False, n=None):
    n = n or node
    n.cur_frame = torch.tensor(f); n.in_test_set = test
    return n.transform_means(n._means.detach()).detach().reshape(H_, G, 3)
def truth(f):
    return torch.from_numpy(np.stack([full[h] @ yaw(angles[f]).T + offset[h] + f * vel[h] for h in range(H_)])).float().to(dev)

# 1. a rigidly moving surface: every Gaussian lands on the true position, also those between control points
e = (world(3) - truth(3)).norm(dim=-1).max()
assert e < 2e-3, f"rigid motion not reproduced: {e}"
# 2. orientations rotate with the body
node.cur_frame = torch.tensor(3); node.in_test_set = False
_, q = node.transform_means_and_quats(node._means.detach(), node._quats.detach())
R0 = quat_to_rotmat(node.quat_act(node._quats.detach()))
R1 = quat_to_rotmat(q.detach())
Ry = torch.from_numpy(yaw(angles[3])).float().to(dev)
assert (R1 - Ry @ R0).abs().max() < 2e-3, f"orientations do not follow: {(R1 - Ry @ R0).abs().max()}"
# 3. a held-out frame is rebuilt from its neighbours and ignores its own track; cubic is exact on a straight line
#    (the yaw is linear in time, so the points move on arcs: compare the translation-only person with a small tolerance)
w = world(4, test=True)
c = w[0].mean(0) - truth(4)[0].mean(0)
assert c.norm() < 5e-3, f"held-out frame not interpolated: {c.norm()}"
ctrl.test_interp = "linear"
wl = world(4, test=True)
assert (wl - w).abs().max() > 1e-5, "linear and cubic should differ on arcs"
ctrl.test_interp = "cubic"
# 4. an absent person does not move and does not crash
w0 = world(0)
assert (w0[1] == 0).all() and not node.instances_fv[0, 1]
# 5. checkpoint round trip
sd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in node.state_dict().items()}
other = make(); other.load_state_dict(sd, strict=True)
assert (world(3, n=other) - world(3)).abs().max() < 1e-5
# 6. gradients reach the Gaussians, not the tracks
node.cur_frame = torch.tensor(3); node.in_test_set = False
x = node.transform_means(node._means); x.sum().backward()
assert node._means.grad is not None and node._means.grad.abs().sum() > 0
assert not node.tracks.requires_grad
# 7. the window only uses valid frames: not next to a held-out frame, not at the sequence ends
assert not node.window_valid(3).any() and not node.window_valid(6).any() and not node.window_valid(1).any() and node.window_valid(10).all()
# 8. hidden on a training step: posed from the neighbours; own track otherwise
OmegaConf.set_struct(ctrl, False)
ctrl.motion = dict(mode="none", p_hide=1.0)
node.train(); hid = world(10)
node.eval(); own = world(10)
assert (hid - own).abs().max() > 1e-5, "hide had no effect"
assert (hid - truth(10)).norm(dim=-1).max() < 3e-2, "hidden pose far from the truth"
# 9. both MLPs start at zero, train, and survive a checkpoint
for mode in ("time", "window"):
    ctrl.motion = dict(mode=mode, p_hide=1.0, hidden=32, depth=2, max_offset=0.1)
    m = make(); m.create_from_pcd(d); m.step = 2000
    g = m.get_param_groups(); assert "SMPLNodes#motion" in g
    m.train(); m.cur_frame = torch.tensor(10); m.in_test_set = False
    x0 = m.transform_means(m._means)
    assert m._last_delta is not None and m._last_delta.abs().max() < 1e-7, "must start at zero"
    assert (x0.detach().reshape(H_, G, 3) - truth(10)).norm(dim=-1).max() < 3e-2
    # a nonzero output moves the Gaussians and the output is bounded
    for pm in m.motion_net.parameters():
        torch.nn.init.normal_(pm, std=0.3)
    x1 = m.transform_means(m._means); x1.sum().backward()
    assert m._last_delta.norm(dim=-1).max() <= 0.1 * 3 ** 0.5 + 1e-5 and (x1 - x0).abs().max() > 1e-4, (mode, float(m._last_delta.norm(dim=-1).max()), float((x1 - x0).abs().max()))
    assert any(pm.grad is not None and pm.grad.abs().sum() > 0 for pm in m.motion_net.parameters()), "no gradient to the MLP"
    m.eval(); m.in_test_set = True; m.cur_frame = torch.tensor(4)
    y = m.transform_means(m._means).detach()
    sd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in m.state_dict().items()}
    o = make(); o.load_state_dict(sd, strict=True); o.eval(); o.in_test_set = True; o.cur_frame = torch.tensor(4)
    assert (o.transform_means(o._means).detach() - y).abs().max() < 1e-5, f"checkpoint round trip failed for {mode}"
# 10. the body SDF trains and reaches the Gaussians
ctrl.motion = dict(mode="none")
make2 = lambda: TrackNodes(class_name="SMPLNodes", ctrl=ctrl, reg=OmegaConf.create(dict(body_sdf=dict(enabled=True, n=256, gs_start=0))), scene_scale=30.0, device=dev)
s_ = make2(); s_.create_from_pcd(d); s_.step = 10; s_.cur_frame = torch.tensor(10); s_.in_test_set = False
l = s_.compute_reg_loss()
assert {"body_sdf_surface", "body_sdf_eikonal", "body_sdf_off", "body_sdf_gaussians"} <= set(l), l.keys()
sum(l.values()).backward()
assert s_._means.grad is not None and s_._means.grad.abs().sum() > 0
# 11. fewer tracks
ctrl.n_tracks = 100
s3 = make(); s3.create_from_pcd(d); s3.cur_frame = torch.tensor(3); s3.in_test_set = False
assert s3.ctrl_canon.shape[1] == 100
e = (world(3, n=s3) - truth(3)).norm(dim=-1).max()
assert e < 5e-2, f"100 tracks: {e}"
ctrl.n_tracks = 0
# 12. fewer Gaussians: sizes follow the sparser spacing, motion is still right, a checkpoint round-trips
ctrl.n_gaussians = 100
s4 = make(); s4.create_from_pcd(d)
assert s4.smpl_points_num == 100 and s4._means.shape[0] == 2 * 100 and s4.point_ids.shape[0] == 2 * 100
assert (s4._means.detach().reshape(2, 100, 3).cpu().numpy() - full[:, :100]).max() < 1e-5
assert s4.get_scaling[:100, 0].mean() > node.get_scaling[:G, 0].mean() * 1.2, "subset discs should be larger"     # person 0: the faces fit its mesh
s4.cur_frame = torch.tensor(3); s4.in_test_set = False
w = s4.transform_means(s4._means.detach()).detach().reshape(2, 100, 3)
e = (w - truth(3)[:, :100]).norm(dim=-1).max()
assert e < 2e-3, f"subset of Gaussians: {e}"
s5 = make(); print(s5.load_state_dict(s4.state_dict(), strict=False))
assert s5.smpl_points_num == 100 and (s5._means - s4._means).abs().max() == 0
ctrl.n_gaussians = 0
print("ALL OK")
