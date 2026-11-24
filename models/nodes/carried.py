from typing import Dict, Tuple
import logging

import torch
from omegaconf import OmegaConf

from models.gaussians.basics import *
from models.nodes.deformable import DeformableNodes

logger = logging.getLogger()


class CarriedNodes(DeformableNodes):
    """
    Free Gaussians for what the SMPL body does not cover (bags, suitcases, loose clothing), one instance per
    SMPL person, in the same order as the SMPLNodes instances. The Gaussians live in the person's annotated
    box frame, as in DeformableNodes, but the box follows the SMPL root joint: at every frame the pose is
    T_root(frame) * C, where C is fixed per person and places the annotated box on the root at a reference
    frame. On held-out frames the root comes from SMPLNodes.current_pose, so items and body use the same
    interpolated (or fitted) pose. The body pose is detached: items do not move the body.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._body = None            # (SMPLNodes,); a tuple so the body is not registered as a submodule
        self.attach_R = None         # (num_instances, 3, 3) box frame -> root frame
        self.attach_t = None         # (num_instances, 3)

    def link_body(self, body) -> None:
        self._body = (body,)

    @torch.no_grad()
    def _root_pose(self, frame: int, test: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Root joint rotation (num_instances, 3, 3) and translation (num_instances, 3) in world, identity where absent."""
        body = self._body[0]
        body.cur_frame = frame if isinstance(frame, torch.Tensor) else torch.tensor(frame, device=self.device)
        body.in_test_set = test
        theta, trans = body.current_pose()
        mask = body.instances_fv[int(frame)]
        N = theta.shape[0]
        R = torch.eye(3, device=self.device).repeat(N, 1, 1)
        t = torch.zeros(N, 3, device=self.device)
        if mask.any():
            _, A = body.template(masked_theta=body.quat_act(theta[mask]), instances_mask=mask)
            R[mask] = A[:, 0, :3, :3].to(R.dtype)
            t[mask] = A[:, 0, :3, 3].to(t.dtype) + trans[mask]
        return R, t

    @torch.no_grad()
    def attach_to_body(self) -> None:
        """Fix C per person at the middle frame where both the box and the body exist; render only where the body exists."""
        body = self._body[0]
        assert body.num_instances == self.num_instances, "CarriedNodes needs one instance per SMPL person"
        q = self.quat_act(self.instances_quats.data); t = self.instances_trans.data
        self.attach_R = torch.eye(3, device=self.device).repeat(self.num_instances, 1, 1)
        self.attach_t = torch.zeros(self.num_instances, 3, device=self.device)
        both = self.instances_fv & body.instances_fv
        for i in range(self.num_instances):
            frames = torch.where(both[:, i])[0]
            if len(frames) == 0:
                continue
            fr = int(frames[len(frames) // 2])
            R_root, t_root = self._root_pose(fr, False)
            R_box = quat_to_rotmat(q[fr, i][None])[0]
            self.attach_R[i] = R_root[i].T @ R_box
            self.attach_t[i] = R_root[i].T @ (t[fr, i] - t_root[i])
        self.instances_fv = both
        logger.info(f"{self.class_prefix} attached {self.num_instances} instances to the SMPL roots")

    def current_instance_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        R_root, t_root = self._root_pose(self.cur_frame, bool(self.in_test_set))
        R = R_root @ self.attach_R
        t = (R_root @ self.attach_t[..., None])[..., 0] + t_root
        return matrix_to_quaternion(R), t

    def transform_quats(self, quats: torch.Tensor) -> torch.Tensor:
        q, _ = self.current_instance_pose()
        return quat_mult(self.quat_act(q[self.point_ids[..., 0]]), self.quat_act(quats))

    def state_dict(self) -> Dict:
        sd = super().state_dict()
        sd.update({"attach_R": self.attach_R, "attach_t": self.attach_t})
        return sd

    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        self.attach_R = state_dict.pop("attach_R").to(self.device)
        self.attach_t = state_dict.pop("attach_t").to(self.device)
        return super().load_state_dict(state_dict, **kwargs)
