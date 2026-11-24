"""
Dump what the monocular-depth pipeline needs from a DriveStudio dataset, exactly as the trainer
sees it (undistorted, at the loaded resolution): every camera image, its sparse LiDAR depth map,
intrinsics and pose, plus the instance boxes.

The LiDAR depth maps are projected here with the intrinsics of the undistorted images, keeping
the nearest return per pixel, independently of the loader's `lidar_depth_intrinsics` setting.

    PYTHONPATH=$(pwd) python tools/mono_depth/dump_views.py <run config.yaml> <out_dir> [data overrides]

Writes  <out_dir>/views/<cam>_<frame>.png, <out_dir>/cam_<cam>.npz (lidar_depth, intrinsics,
cam_to_worlds) and <out_dir>/meta.npz (instances_pose, instances_size, instance_active,
instance_true_id, instance_model_type, test_frames).
"""
import os, sys
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.driving_dataset import DrivingDataset

cfg = OmegaConf.load(sys.argv[1])
out_dir = sys.argv[2]
write_images = "--no_images" not in sys.argv
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli([a for a in sys.argv[3:] if not a.startswith("--")]))
cfg.data.preload_device = "cpu"
dataset = DrivingDataset(data_cfg=cfg.data)
ps = dataset.pixel_source
os.makedirs(os.path.join(out_dir, "views"), exist_ok=True)

def lidar_depth_map(cam, fi):
    """z-depth of the LiDAR sweep nearest in time, projected with the image intrinsics (nearest return per pixel)."""
    lidar = dataset.lidar_source
    info = lidar.get_lidar_rays(lidar.find_closest_timestep(ps.normalized_time[fi]))
    pts = (info["lidar_origins"] + info["lidar_viewdirs"] * info["lidar_ranges"]).cpu().double()
    w2c = torch.linalg.inv(cam.cam_to_worlds[fi].cpu().double())
    K = cam.intrinsics[fi].cpu().double()
    p = (pts @ w2c[:3, :3].T + w2c[:3, 3]) @ K.T
    z = p[:, 2]
    uv = p[:, :2] / z.clamp_min(1e-6)[:, None]
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.WIDTH) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.HEIGHT)
    u, v, z = uv[ok, 0].long(), uv[ok, 1].long(), z[ok]
    order = torch.argsort(z, descending=True)          # write far to near: the nearest return stays
    depth = torch.zeros(cam.HEIGHT, cam.WIDTH, dtype=torch.float64)
    depth[v[order], u[order]] = z[order]
    return depth.numpy().astype(np.float32)


for cam_id, cam in ps.camera_data.items():
    images = (cam.images.cpu().numpy() * 255).round().astype(np.uint8)
    if write_images:
        for fi in range(images.shape[0]):
            cv2.imwrite(os.path.join(out_dir, "views", f"{cam_id}_{fi:03d}.png"), images[fi][..., ::-1])
    np.savez_compressed(
        os.path.join(out_dir, f"cam_{cam_id}.npz"),
        lidar_depth=np.stack([lidar_depth_map(cam, fi) for fi in range(images.shape[0])]),
        intrinsics=cam.intrinsics.cpu().numpy(), cam_to_worlds=cam.cam_to_worlds.cpu().numpy(),
        width=cam.WIDTH, height=cam.HEIGHT)
    print(f"camera {cam_id}: {images.shape[0]} views of {images.shape[2]}x{images.shape[1]}", flush=True)

test = [] if dataset.test_timesteps is None else [int(t) for t in dataset.test_timesteps]
np.savez_compressed(
    os.path.join(out_dir, "meta.npz"),
    instances_pose=ps.instances_pose.cpu().numpy(), instances_size=ps.instances_size.cpu().numpy(),
    instance_active=ps.per_frame_instance_mask.cpu().numpy().astype(bool),
    instance_true_id=np.asarray(ps.instances_true_id), instance_model_type=np.asarray(ps.instances_model_types),
    test_frames=np.asarray(test, dtype=np.int64), cam_ids=np.asarray(list(ps.camera_data.keys())))
print("done", flush=True)
