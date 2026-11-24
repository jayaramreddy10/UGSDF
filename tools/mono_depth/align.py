"""Scale-and-shift alignment of monocular depth to LiDAR, and the per-view geometry it needs."""
from typing import Tuple
import numpy as np


def fit_scale(mono: np.ndarray, lidar: np.ndarray) -> float:
    """Robust scale: median ratio."""
    return float(np.median(lidar / mono))


def fit_scale_shift(mono: np.ndarray, lidar: np.ndarray, delta: float = 0.1, iters: int = 15,
                    min_spread: float = 0.3, scale_bounds: Tuple[float, float] = (0.5, 2.0)) -> Tuple[float, float, bool]:
    """
    Robust fit of lidar ~ s * mono + t (Huber, threshold `delta` metres, iteratively reweighted).
    Falls back to scale only (t = 0) when the anchors span less than `min_spread` metres of
    depth, where the shift is not identifiable, or when the fitted scale leaves `scale_bounds`.
    Returns (s, t, used_shift).
    """
    s, t = fit_scale(mono, lidar), 0.0
    if mono.shape[0] < 3 or np.percentile(mono, 90) - np.percentile(mono, 10) < min_spread:
        return s, t, False
    s0 = s
    A = np.stack([mono, np.ones_like(mono)], axis=1)
    for _ in range(iters):
        r = np.abs(lidar - (s * mono + t))
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
        Aw = A * w[:, None]
        try:
            s, t = np.linalg.solve(A.T @ Aw, Aw.T @ lidar)
        except np.linalg.LinAlgError:
            return s0, 0.0, False
    if not (scale_bounds[0] <= s <= scale_bounds[1]):
        return s0, 0.0, False
    return float(s), float(t), True


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World points of pixels (u, v) (integer indices, centres at +0.5) with z-depth z."""
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1) @ c2w[:3, :3].T + c2w[:3, 3]


def in_box(p_world: np.ndarray, pose: np.ndarray, size: np.ndarray, tol: float = 0.02, ground_margin: float = 0.12):
    """Object-frame coordinates and the mask of points inside the box (above its floor margin)."""
    p_obj = (p_world - pose[:3, 3]) @ pose[:3, :3]
    half = size / 2
    inside = (np.abs(p_obj) <= half + tol).all(-1) & (p_obj[:, 2] > -half[2] + ground_margin)
    return p_obj, inside
