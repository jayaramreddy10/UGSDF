"""CarriedNodes follow the SMPL root.  Run from the repo root:  PYTHONPATH=$(pwd) python tests/test_carried_nodes.py $(pwd)"""
import os, sys, types
import torch
from omegaconf import OmegaConf
sys.path.insert(0, sys.argv[1])
from models.nodes import CarriedNodes
from models.gaussians.basics import quat_to_rotmat, matrix_to_quaternion

torch.manual_seed(0)
dev = torch.device(os.environ.get("SDFGS_DEVICE", "cuda"))
F, N = 6, 2

class FakeTemplate:
    def __call__(self, masked_theta=None, instances_mask=None, xyz_canonical=None):
        A = torch.eye(4, device=dev).repeat(masked_theta.shape[0], 24, 1, 1)
        A[:, 0, :3, :3] = quat_to_rotmat(masked_theta[:, 0])
        A[:, 0, :3, 3] = torch.tensor([0.0, -0.2, 0.0], device=dev)   # root joint offset of the posed body
        return None, A

class FakeBody:
    """What CarriedNodes reads from SMPLNodes: current_pose, quat_act, template, instances_fv, num_instances."""
    def __init__(self):
        self.num_instances = N; self.template = FakeTemplate()
        self.instances_fv = torch.ones(F, N, dtype=torch.bool, device=dev); self.instances_fv[0, 1] = False
        self.q = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(F, N, 1); self.t = torch.rand(F, N, 3, device=dev) * 5
        self.cur_frame, self.in_test_set = torch.tensor(0), False; self.test_override = None
    def quat_act(self, q): return q / q.norm(dim=-1, keepdim=True)
    def current_pose(self):
        f = int(self.cur_frame)
        if self.in_test_set and self.test_override is not None:
            return self.test_override[0], self.test_override[1]
        return self.q[f][:, None].repeat(1, 24, 1), self.t[f]

ctrl = OmegaConf.create(dict(warmup_steps=0, reset_alpha_interval=3000, refine_interval=100, sh_degree_interval=1000, n_split_samples=2,
    reset_alpha_value=0.1, densify_grad_thresh=5e-4, densify_size_thresh=3e-3, cull_alpha_thresh=0.005, cull_scale_thresh=0.1, cull_screen_size=0.15,
    split_screen_size=0.05, stop_screen_size_at=30000, stop_split_at=20000, sh_degree=3, cull_out_of_bound=False, use_deformgs_for_nonrigid=True,
    use_deformgs_after=100000, stop_optimizing_canonical_xyz=True))
nets = OmegaConf.create(dict(D=4, W=128, embed_dim=16, x_multires=10, t_multires=10, deform_quat=True, deform_scale=False))
def make():
    n = CarriedNodes(class_name="CarriedNodes", ctrl=ctrl, reg=OmegaConf.create({}), networks=nets, scene_scale=30.0, device=dev)
    return n
node = make()
poses = torch.eye(4).repeat(F, 1, 1); poses[:, :3, 3] = torch.tensor([3.0, 1.0, 0.5])
d = {k: dict(pts=torch.rand(200, 3) - 0.5, colors=torch.rand(200, 3), poses=poses.clone(), size=torch.tensor([1.5, 1.5, 2.0]),
             frame_info=torch.ones(F, dtype=torch.bool), num_pts=200) for k in (3, 8)}
node.create_from_pcd(d)
body = FakeBody(); node.link_body(body); node.attach_to_body()
assert not node.instances_fv[0, 1] and node.instances_fv[0, 0], "items must render only where the body exists"

def world(frame, test=False):
    node.cur_frame = torch.tensor(frame); node.in_test_set = test
    return node.transform_means(node._means.detach()).detach()
# 1. at the reference frame the box pose is reproduced (middle frame of the overlap)
fr = 3
w = world(fr)
p = node._means.detach() @ torch.eye(3, device=dev) + torch.tensor([3.0, 1.0, 0.5], device=dev)
assert (w - p).abs().max() < 1e-4, f"reference frame box pose not reproduced: {(w - p).abs().max()}"
# 2. moving and turning the body moves the items rigidly with the root
with torch.no_grad():
    yaw = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834], device=dev)  # 45 deg about z
    body.q[4] = yaw.repeat(N, 1); body.t[4] = body.t[3] + torch.tensor([0.5, -0.3, 0.1], device=dev)
w4 = world(4)
R_y = quat_to_rotmat(yaw[None])[0]
root3 = body.t[3] + torch.tensor([0.0, -0.2, 0.0], device=dev)
root4 = body.t[4] + torch.tensor([0.0, -0.2, 0.0], device=dev)
ids = node.point_ids[:, 0]
expected = (R_y @ (w - root3[ids])[..., None])[..., 0] + root4[ids]
assert (w4 - expected).abs().max() < 1e-3, f"items do not follow the root: {(w4 - expected).abs().max()}"
# 3. orientations rotate with the root as well
node.cur_frame = torch.tensor(4); node.in_test_set = False
q4 = node.transform_quats(node._quats.detach()); node.cur_frame = torch.tensor(3)
q3 = node.transform_quats(node._quats.detach())
Rr = quat_to_rotmat(q4.detach()) @ quat_to_rotmat(q3.detach()).transpose(-1, -2)
assert (Rr - R_y).abs().max() < 1e-3, "orientations do not follow the root"
# 4. held-out frames use the body's held-out pose
ov_t = body.t[4] + torch.tensor([1.0, 0.0, 0.0], device=dev)
body.test_override = (body.q[4][:, None].repeat(1, 24, 1), ov_t)
wt = world(4, test=True)
assert ((wt - w4) - torch.tensor([1.0, 0.0, 0.0], device=dev)).abs().max() < 1e-3, "held-out frames ignore the body's held-out pose"
body.test_override = None
# 5. checkpoint round trip
sd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in node.state_dict().items()}
node2 = make(); node2.create_from_pcd(d); node2.link_body(body)
node2.load_state_dict(sd)
node2.cur_frame = torch.tensor(4); node2.in_test_set = False
assert (node2.transform_means(node2._means.detach()).detach() - w4).abs().max() < 1e-4, "checkpoint round trip changed the poses"
print("ALL OK")
