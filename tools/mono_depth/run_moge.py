"""
Run MoGe-2 on the views written by dump_views.py and cache metric depth, validity mask and
normals per view. Runs in an environment that has MoGe (not the DriveStudio one):

    PYTHONPATH=<MoGe repo> python tools/mono_depth/run_moge.py <dump dir> [--model Ruicheng/moge-2-vitl-normal]

The calibrated horizontal field of view of each view is passed to the model. Writes
<dump dir>/moge2/<cam>_<frame>.npz with depth (float32, z-depth in metres, 0 where invalid),
mask (bool) and normal (int8, camera frame x right / y down / z forward, scaled by 127).
"""
import argparse, glob, os, math
import numpy as np
import cv2
import torch
from moge.model.v2 import MoGeModel

parser = argparse.ArgumentParser()
parser.add_argument("dump_dir")
parser.add_argument("--model", default="Ruicheng/moge-2-vitl-normal")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

device = torch.device("cuda")
model = MoGeModel.from_pretrained(args.model).to(device).eval()
out_dir = os.path.join(args.dump_dir, "moge2")
os.makedirs(out_dir, exist_ok=True)
cams = {}
paths = sorted(glob.glob(os.path.join(args.dump_dir, "views", "*.png")))
for n, path in enumerate(paths):
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, name + ".npz")
    if os.path.exists(out_path) and not args.overwrite:
        continue
    cam_id, fi = name.split("_"); fi = int(fi)
    if cam_id not in cams:
        cams[cam_id] = np.load(os.path.join(args.dump_dir, f"cam_{cam_id}.npz"))
    K = cams[cam_id]["intrinsics"][fi]
    image = cv2.imread(path)[..., ::-1].copy()
    h, w = image.shape[:2]
    fov_x = math.degrees(2 * math.atan(w / (2 * float(K[0, 0]))))
    x = torch.from_numpy(image).to(device).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        out = model.infer(x, fov_x=fov_x)
    mask = out["mask"].cpu().numpy().astype(bool)
    depth = out["depth"].float().cpu().numpy()
    depth = np.where(mask & np.isfinite(depth), depth, 0).astype(np.float32)
    save = dict(depth=depth, mask=mask)
    if "normal" in out:
        save["normal"] = np.clip(np.nan_to_num(out["normal"].float().cpu().numpy()) * 127, -127, 127).round().astype(np.int8)
    np.savez_compressed(out_path, **save)
    if n % 50 == 0:
        print(f"{n}/{len(paths)} {name} fov_x {fov_x:.1f} valid {mask.mean():.2f}", flush=True)
print("done", flush=True)
