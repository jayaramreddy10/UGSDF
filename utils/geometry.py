# Utility functions for geometric transformations and projections.
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

def transform_points(points, transform_matrix):
    """
    Apply a 4x4 transformation matrix to 3D points.

    Args:
        points: (N, 3) tensor of 3D points
        transform_matrix: (4, 4) transformation matrix

    Returns:
        (N, 3) tensor of transformed 3D points
    """
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    homo_points = torch.cat([points, ones], dim=1)  # N x 4
    transformed_points = torch.matmul(homo_points, transform_matrix.T)
    return transformed_points[:, :3]

def get_corners(l: float, w: float, h: float):
    """
    Get 8 corners of a 3D bounding box centered at origin.

    Args:
        l, w, h: length, width, height of the box

    Returns:
        (3, 8) array of corner coordinates
    """
    return np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2],
    ])
    
def project_camera_points_to_image(points_cam, cam_intrinsics):
    """
    Project 3D points from camera space to 2D image space.

    Args:
        points_cam (np.ndarray): Shape (N, 3), points in camera space.
        cam_intrinsics (np.ndarray): Shape (3, 3), intrinsic matrix of the camera.

    Returns:
        tuple: (projected_points, depths)
            - projected_points (np.ndarray): Shape (N, 2), projected 2D points in image space.
            - depths (np.ndarray): Shape (N,), depth values of the projected points.
    """
    points_img = cam_intrinsics @ points_cam.T
    depths = points_img[2, :]
    projected_points = (points_img[:2, :] / (depths + 1e-6)).T
    
    return projected_points, depths

def cube_root(x):
    return torch.sign(x) * torch.abs(x) ** (1. / 3)

def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=1)

def uniform_sample_sphere(num_samples, device, inverse=False):
    """
    refer to https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    sample points uniformly inside a sphere
    """
    if not inverse:
        dist = torch.rand((num_samples,)).to(device)
        dist = cube_root(dist)
    else:
        dist = torch.rand((num_samples,)).to(device)
        dist = 1 / dist.clamp_min(0.02)
    thetas = torch.arccos(2 * torch.rand((num_samples,)) - 1).to(device)
    phis = 2 * torch.pi * torch.rand((num_samples,)).to(device)
    pts = spherical_to_cartesian(dist, thetas, phis)
    return pts

def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

# ---------------------------------------------------------------------------
# Rotation conversions and KNN, replacing the few pytorch3d functions used here.
# Quaternions are real-first (w, x, y, z), as in pytorch3d.
# ---------------------------------------------------------------------------

def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """(*, 4) real-first quaternions -> (*, 3, 3) rotation matrices. Adapted from pytorch3d."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: Tensor) -> Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """(*, 3, 3) rotation matrices -> (*, 4) real-first quaternions with w >= 0. Adapted from pytorch3d."""
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    # each row is the quaternion multiplied by one of its components
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    # pick the best-conditioned candidate (largest |component|)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return torch.where(out[..., 0:1] < 0, -out, out)

def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """(*, 3) axis-angle -> (*, 3, 3) rotation matrices (Rodrigues)."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    small = angle < 1e-6
    # first-order sin(t)/t and (1-cos t)/t^2 near zero
    safe = torch.where(small, torch.ones_like(angle), angle)
    a = torch.where(small, 1.0 - angle ** 2 / 6, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - angle ** 2 / 24, (1 - torch.cos(safe)) / safe ** 2)
    x, y, z = torch.unbind(axis_angle, -1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(K)
    return eye + a[..., None] * K + b[..., None] * (K @ K)

def knn_points(p1: Tensor, p2: Tensor, K: int = 1, return_nn: bool = False, max_chunk_elems: int = 2 ** 23):
    """
    Brute-force KNN with the pytorch3d.ops.knn_points interface.
    p1: [B, N, D], p2: [B, M, D] -> (squared dists [B, N, K], idx [B, N, K], nn [B, N, K, D] or None)
    Queries are processed in chunks so each [B, chunk, M] distance matrix has at most
    `max_chunk_elems` entries (32 MB in float32; it runs beside a ~21 GB training state).
    """
    chunk_size = max(1, max_chunk_elems // (p1.shape[0] * p2.shape[1]))
    dists, idxs = [], []
    for start in range(0, p1.shape[1], chunk_size):
        d = torch.cdist(p1[:, start:start + chunk_size], p2).square()
        dist, idx = d.topk(K, dim=-1, largest=False)
        dists.append(dist)
        idxs.append(idx)
    dists, idxs = torch.cat(dists, dim=1), torch.cat(idxs, dim=1)
    nn = None
    if return_nn:
        nn = torch.gather(
            p2[:, None].expand(-1, p1.shape[1], -1, -1), 2,
            idxs[..., None].expand(-1, -1, -1, p2.shape[-1])
        )
    return dists, idxs, nn
