from typing import List, Optional
from omegaconf import OmegaConf
import os
import gc
import time
import json
import wandb
import logging
from utils.logging import setup_logging
import argparse
import numpy as np

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

@torch.no_grad()
def gaussian_sdf_stats(trainer, ref_ckpt: str = None, taus=(0.2, 0.5)) -> dict:
    """
    |sdf| of every RigidNodes gaussian in its object frame: the model's own SDFs if it has
    them (SDFGSNodes), else the SDFs of a reference SDFGS checkpoint of the same scene
    (same instances in the same order), so a plain RigidNodes model can be scored too.
    """
    model = trainer.models.get("RigidNodes", None)
    if model is None or not hasattr(model, "point_ids"):
        return {}
    if hasattr(model, "sdf_nets") and len(model.sdf_nets) > 0:
        nets = model.sdf_nets
    elif ref_ckpt:
        from models.nodes.sdfgs import InstanceSDF
        state = torch.load(ref_ckpt, map_location=model.device, weights_only=False)["models"]["RigidNodes"]
        ref_cfg = OmegaConf.load(os.path.join(os.path.dirname(ref_ckpt), "config.yaml"))
        net_cfg = ref_cfg.model.RigidNodes.sdf.get("net", {})
        sizes = state["instances_size"].to(model.device)
        if sizes.shape[0] != model.num_instances or not torch.allclose(sizes, model.instances_size):
            logger.warning("gaussian_sdf_stats: reference checkpoint instances differ, skipping")
            return {}
        nets = torch.nn.ModuleList([
            InstanceSDF(sizes[i] / 2, hidden_dim=net_cfg.get("hidden_dim", 128), layers=net_cfg.get("layers", 4),
                        frequencies=net_cfg.get("frequencies", 4), pad=net_cfg.get("pad", 1.3),
                        box_shrink=net_cfg.get("box_shrink", 0.05), box_clamp=net_cfg.get("box_clamp", None))
            for i in range(sizes.shape[0])
        ]).to(model.device)
        nets.load_state_dict({k[len("sdf_nets."):]: v for k, v in state.items() if k.startswith("sdf_nets.")})
    else:
        return {}
    ids = model.point_ids[:, 0]
    sdf = torch.zeros(model.num_points, device=model.device)
    for i in range(len(nets)):
        m = ids == i
        if m.any():
            sdf[m] = nets[i](model._means[m])
    opaque = model.get_opacity.squeeze(-1) > 0.3
    out = {"gs_num": float(model.num_points), "gs_abs_sdf_mean": sdf.abs().mean().item(),
           "gs_opaque_abs_sdf_mean": sdf[opaque].abs().mean().item() if opaque.any() else -1.0}
    for tau in taus:
        out[f"gs_frac_sdf_gt_{tau}"] = (sdf.abs() > tau).float().mean().item()
        out[f"gs_opaque_frac_sdf_gt_{tau}"] = (sdf[opaque].abs() > tau).float().mean().item() if opaque.any() else -1.0
    # how well the thinnest axis of opaque gaussians near the surface follows the SDF normal
    # (|cos|: 1 aligned, 0.5 is the mean for random orientations), and how flat they are
    if hasattr(model, "short_axis_cos") and nets is getattr(model, "sdf_nets", None):
        sel = torch.where(opaque & (sdf.abs() <= 0.1))[0]
        if sel.numel() > 0:
            _, normal = model.query_sdf_normal(model._means[sel], ids[sel])
            cos = model.short_axis_cos(sel, normal)
            scale = model.get_scaling[sel]
            out["gs_short_axis_cos"] = cos.mean().item()
            out["gs_short_axis_cos_gt_0.9"] = (cos > 0.9).float().mean().item()
            out["gs_thin_ratio_median"] = (scale.min(-1).values / scale.max(-1).values).median().item()
    return out


@torch.no_grad()
def background_free_space_stats(trainer, dataset, cfg=None) -> dict:
    """
    Background gaussians in free space, judged by LiDAR rays of the HELD-OUT frames cast
    through a voxel grid (models.gaussians.free_space): the share of gaussians, and of their
    opacity, centred in voxels those rays saw through and never hit.
    """
    from models.gaussians.free_space import FreeSpaceGrid
    from models.nodes.sdf_supervision import modelled_instance_ids
    bg = trainer.models.get("Background", None)
    frames = list(dataset.test_timesteps) if dataset.test_timesteps is not None else []
    if bg is None or len(frames) == 0 or dataset.lidar_source is None:
        return {}
    fs = {}
    if cfg is not None:
        fs = cfg.model.get("Background", {}).get("reg", {}).get("free_space", None) or {}
    ps = dataset.pixel_source
    boxes = dict(pose=ps.instances_pose, size=ps.instances_size, active=ps.per_frame_instance_mask,
                 ids=modelled_instance_ids(trainer.models, ps))
    grid = FreeSpaceGrid.build(
        dataset.lidar_source, frames, bg.device, voxel=fs.get("voxel", 0.25), max_range=fs.get("max_range", 75.0),
        min_traversal=fs.get("min_traversal", 5.0), min_hits=fs.get("min_hits", 1), dilate=fs.get("dilate", 1),
        exclude_boxes=boxes,
    )
    xyz, opacity = bg._means.data, bg.get_opacity.squeeze(-1)
    free, observed = grid.is_free(xyz), grid.is_observed(xyz)
    n_obs = max(int(observed.sum()), 1)
    return {
        "bg_num": float(xyz.shape[0]),
        "bg_observed_frac": observed.float().mean().item(),
        "bg_free_num": float(free.sum()),
        "bg_free_frac": float(free.sum()) / n_obs,
        "bg_free_opaque_num": float((free & (opacity > 0.3)).sum()),
        "bg_free_opacity_frac": (opacity[free].sum() / opacity[observed].sum().clamp_min(1e-6)).item(),
    }


def _observed_by_rays(pts, o, d, t, radius: float, chunk: int = 256):
    """(N,) bool: the point is within `radius` of one of the ray segments o + s d, s in [t0, t1]."""
    seen = torch.zeros(pts.shape[0], dtype=torch.bool, device=pts.device)
    for k in range(0, pts.shape[0], chunk):
        v = pts[k:k + chunk]
        rel = v[:, None] - o[None]                                   # (c, R, 3)
        s = (rel * d[None]).sum(-1).clamp(min=t[None, :, 0]).clamp(max=t[None, :, 1])
        dist = (rel - s[..., None] * d[None]).norm(dim=-1)
        seen[k:k + chunk] = (dist < radius).any(dim=1)
    return seen


@torch.no_grad()
def _sdf_mesh_vertices(model, i: int) -> Optional[torch.Tensor]:
    """Vertices (object frame) of the zero level of instance i's SDF, by marching cubes."""
    net = model.sdf_nets[i]
    half = (model.instances_size[i] / 2 * 1.2).cpu().numpy()
    res = np.maximum((model.mesh_resolution * half / half.max()).astype(int), 16)
    axes = [torch.linspace(-h, h, int(r), device=model.device) for h, r in zip(half, res)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    vol = torch.cat([net(grid[k:k + 262144]) for k in range(0, grid.shape[0], 262144)])
    vol = vol.reshape(*[int(r) for r in res]).cpu().numpy()
    if not (vol.min() < 0 < vol.max()):
        return None
    from skimage.measure import marching_cubes
    spacing = tuple(float(2 * h / (r - 1)) for h, r in zip(half, res))
    verts = marching_cubes(vol, level=0.0, spacing=spacing)[0] - half
    return torch.from_numpy(verts.astype(np.float32)).to(model.device)


@torch.no_grad()
def withheld_side_stats(trainer, dataset, masks_root: str = None, tau: float = 0.05, obs_radius: float = 0.05,
                        max_rays: int = 30000) -> dict:
    """
    Score of the SDF on the side it was trained without (sdf.supervision.withhold_side).
    The reference is every LiDAR ray, from any frame, whose origin is on the withheld side
    of an instance; none of them was in the supervision. On the flank of that side (returns
    within `withhold_band` of the box side) it reports |sdf| at the returns, completeness
    (return to nearest mesh vertex), accuracy of the mesh vertices those rays observed, the
    free-space violation along the rays, and what the bare box face would score
    (`box_dist`: distance from the returns to the box side). Weighted by flank returns.
    """
    from models.nodes.sdf_supervision import AnnotatorMasks, gather_instance_lidar, recover_instance_ds_ids, split_by_side
    from utils.geometry import knn_points
    model = trainer.models.get("RigidNodes", None)
    sides = getattr(model, "sdf_withheld_side", None)
    if model is None or sides is None or not bool(sides.any()) or dataset.lidar_source is None:
        return {}
    ps = dataset.pixel_source
    ds_ids = recover_instance_ds_ids(model, ps)
    masks, true_ids = None, None
    if masks_root is not None and all(j >= 0 for j in ds_ids):
        masks = AnnotatorMasks(masks_root, start_timestep=next(iter(ps.camera_data.values())).start_timestep)
        true_ids = [int(ps.instances_true_id[j].item()) for j in ds_ids]
    per_ins = gather_instance_lidar(
        dataset.lidar_source, list(range(model.num_frames)), model.quat_act(model.instances_quats.data),
        model.instances_trans.data, model.instances_fv, model.instances_size, model.device,
        masks=masks, true_ids=true_ids, pixel_source=ps,
    )
    band = float(model.sdf_cfg.get("supervision", {}).get("withhold_band", 0.25))
    rows, weights = [], []
    for i, side in enumerate(sides.tolist()):
        if side == 0:
            continue
        s = split_by_side(per_ins[i], side, keep_withheld=True)
        if len(s["surf_p"]) == 0:
            continue
        ref, w = torch.cat(s["surf_p"]), torch.cat(s["surf_w"])
        if masks is not None:
            ref = ref[w >= 1.0]
        half_y = float(model.instances_size[i, 1]) / 2
        flank = ref[ref[:, 1] * side > half_y - band]
        if flank.shape[0] < 50:
            continue
        net = model.sdf_nets[i]
        sdf_f = net(flank).abs()
        row = {"instance": i, "side": side, "ref_points": float(flank.shape[0]),
               "sdf_abs": sdf_f.mean().item(), f"sdf_{int(tau*100)}cm": (sdf_f < tau).float().mean().item(),
               "box_dist": (half_y - flank[:, 1] * side).abs().mean().item()}
        o, d, t = torch.cat(s["ray_o"]), torch.cat(s["ray_d"]), torch.cat(s["ray_t"])
        tt = t[:, :1] + torch.rand(t.shape[0], 3, device=t.device) * (t[:, 1:] - t[:, :1])
        row["free_violation"] = (net((o[:, None] + d[:, None] * tt[..., None]).reshape(-1, 3)) < 0).float().mean().item()
        verts = _sdf_mesh_vertices(model, i)
        if verts is not None:
            d_rp, _, _ = knn_points(flank[None], verts[None], K=1); d_rp = d_rp[0, :, 0].sqrt()
            row["comp_mean"] = d_rp.mean().item(); row[f"comp_{int(tau*100)}cm"] = (d_rp < tau).float().mean().item()
            v = verts[verts[:, 1] * side > half_y - band]
            if v.shape[0] > 0:
                t_obs = t.clone(); t_obs[:, 1] += 0.05   # the stored interval stops free_margin short of the return
                if o.shape[0] > max_rays:
                    keep = torch.randperm(o.shape[0], generator=torch.Generator().manual_seed(0))[:max_rays].to(o.device)
                    o, d, t_obs = o[keep], d[keep], t_obs[keep]
                seen = _observed_by_rays(v, o, d, t_obs, radius=obs_radius)
                row["obs_frac"] = seen.float().mean().item()
                if seen.any():
                    d_pr, _, _ = knn_points(v[seen][None], ref[None], K=1); d_pr = d_pr[0, :, 0].sqrt()
                    row["acc_mean"] = d_pr.mean().item(); row[f"acc_{int(tau*100)}cm"] = (d_pr < tau).float().mean().item()
        rows.append(row); weights.append(flank.shape[0])
    if not rows:
        return {}
    out = {}
    for k in sorted({k for r in rows for k in r if k not in ("instance", "side")}):
        vals = [(r[k], wt) for r, wt in zip(rows, weights) if k in r]
        out["wh_" + k] = sum(v * wt for v, wt in vals) / sum(wt for _, wt in vals)
    out["wh_num_instances"] = float(len(rows))
    out["wh_per_instance"] = rows
    return out


@torch.no_grad()
def object_geometry_stats(trainer, dataset, masks_root: str = None, taus=(0.05,), observed_cfg=None) -> dict:
    """
    Object-frame geometry of each RigidNodes instance against LiDAR from the HELD-OUT frames
    (timing offsets cancel in the box frame): accuracy/completeness of opaque gaussian
    centres and, for SDFGSNodes, of the SDF zero level, plus |sdf| at held-out returns and
    the SDF free-space violation rate along held-out rays. Weighted by returns per instance.

    With `observed_cfg.observed_surface`, accuracy is also reported on the observed surface
    only (`*_obs_acc_*`): predicted points within `observed_radius` of a held-out LiDAR ray
    (up to its return). Surface no held-out ray came near (the far side of the vehicle, the
    underside, interior sheets) cannot be judged by LiDAR and is left out of that number;
    `*_obs_frac` says how much of the prediction was judged.
    """
    observed_cfg = observed_cfg or {}
    use_observed = bool(observed_cfg.get("observed_surface", False))
    obs_radius = float(observed_cfg.get("observed_radius", 0.05))
    obs_max_rays = int(observed_cfg.get("max_rays", 30000))
    from models.nodes.sdf_supervision import AnnotatorMasks, gather_instance_lidar, recover_instance_ds_ids
    from utils.geometry import knn_points
    model = trainer.models.get("RigidNodes", None)
    frames = list(dataset.test_timesteps) if dataset.test_timesteps is not None else []
    if model is None or not hasattr(model, "instances_fv") or len(frames) == 0:
        return {}
    ps = dataset.pixel_source
    ds_ids = recover_instance_ds_ids(model, ps)
    masks, true_ids = None, None
    if masks_root is not None and all(j >= 0 for j in ds_ids):
        masks = AnnotatorMasks(masks_root, start_timestep=next(iter(ps.camera_data.values())).start_timestep)
        true_ids = [int(ps.instances_true_id[j].item()) for j in ds_ids]
    quats = model.quat_act(model.instances_quats.data)
    per_ins = gather_instance_lidar(
        dataset.lidar_source, frames, quats, model.instances_trans.data, model.instances_fv, model.instances_size,
        model.device, masks=masks, true_ids=true_ids, pixel_source=ps,
    )
    has_sdf = hasattr(model, "sdf_nets") and len(model.sdf_nets) > 0
    opacity = model.get_opacity.squeeze(-1)
    mask_rays = None   # held-out camera rays through each instance's mask: they must meet the object
    if has_sdf and masks is not None and hasattr(model, "mask_ray_min_sdf"):
        from models.nodes.sdf_supervision import gather_carving_rays
        mask_rays = gather_carving_rays(frames, quats, model.instances_trans.data, model.instances_fv, model.instances_size,
                                        model.device, masks, true_ids, ps, inside=True)[1]
    rows, weights = [], []
    for i, s in enumerate(per_ins):
        if len(s["surf_p"]) == 0:
            continue
        ref = torch.cat(s["surf_p"]); w = torch.cat(s["surf_w"]); ref_d = torch.cat(s["surf_d"])
        if masks is not None:
            ref, ref_d = ref[w >= 1.0], ref_d[w >= 1.0]  # mask-verified returns only
        if ref.shape[0] < 50:
            continue
        row = {"instance": i, "ref_points": float(ref.shape[0])}
        obs_rays = None
        if use_observed and len(s["ray_o"]):
            o, d, t = torch.cat(s["ray_o"]), torch.cat(s["ray_d"]), torch.cat(s["ray_t"]).clone()
            t[:, 1] += 0.05  # the stored interval stops free_margin short of the return
            if o.shape[0] > obs_max_rays:
                keep = torch.randperm(o.shape[0], generator=torch.Generator().manual_seed(0))[:obs_max_rays].to(o.device)
                o, d, t = o[keep], d[keep], t[keep]
            obs_rays = (o, d, t)
        def acc_comp(pred, prefix):
            if pred is None or pred.shape[0] == 0:
                return
            d_pr, _, _ = knn_points(pred[None], ref[None], K=1); d_pr = d_pr[0, :, 0].sqrt()
            d_rp, _, _ = knn_points(ref[None], pred[None], K=1); d_rp = d_rp[0, :, 0].sqrt()
            row[prefix + "acc_mean"] = d_pr.mean().item(); row[prefix + "comp_mean"] = d_rp.mean().item()
            for tau in taus:
                row[prefix + f"acc_{int(tau*100)}cm"] = (d_pr < tau).float().mean().item()
                row[prefix + f"comp_{int(tau*100)}cm"] = (d_rp < tau).float().mean().item()
            if obs_rays is not None:
                seen = _observed_by_rays(pred, *obs_rays, radius=obs_radius)
                row[prefix + "obs_frac"] = seen.float().mean().item()
                if seen.any():
                    row[prefix + "obs_acc_mean"] = d_pr[seen].mean().item()
                    for tau in taus:
                        row[prefix + f"obs_acc_{int(tau*100)}cm"] = (d_pr[seen] < tau).float().mean().item()
        m = (model.point_ids[:, 0] == i) & (opacity > 0.3)
        acc_comp(model._means[m].detach(), "gs_")
        if has_sdf:
            net = model.sdf_nets[i]
            row["sdf_abs_at_ref"] = net(ref).abs().mean().item()
            if hasattr(model, "query_sdf_normal"):
                # a ray meets the surface from outside: the SDF normal at a return must face the sensor
                _, normal = model.query_sdf_normal(ref, torch.full((ref.shape[0],), i, dtype=torch.long, device=ref.device))
                cos = (normal * ref_d).sum(-1)
                row["sdf_normal_faces_sensor"] = (cos < 0).float().mean().item()
                row["sdf_normal_reversed"] = (cos > 0.5).float().mean().item()
            if mask_rays is not None and len(mask_rays[i]["ray_o"]):
                o, d, t = torch.cat(mask_rays[i]["ray_o"]), torch.cat(mask_rays[i]["ray_d"]), torch.cat(mask_rays[i]["ray_t"])
                lowest = model.mask_ray_min_sdf(o, d, t, torch.full((o.shape[0],), i, dtype=torch.long, device=o.device), 32)
                row["sdf_mask_ray_miss"] = (lowest > 0).float().mean().item()
            if len(s["ray_o"]):
                o, d, t = torch.cat(s["ray_o"]), torch.cat(s["ray_d"]), torch.cat(s["ray_t"])
                tt = t[:, :1] + torch.rand(t.shape[0], 3, device=t.device) * (t[:, 1:] - t[:, :1])
                pts = (o[:, None] + d[:, None] * tt[..., None]).reshape(-1, 3)
                row["sdf_free_violation"] = (net(pts) < 0).float().mean().item()
            acc_comp(_sdf_mesh_vertices(model, i), "sdf_")
        rows.append(row); weights.append(ref.shape[0])
    if not rows:
        return {}
    out = {}
    keys = sorted({k for r in rows for k in r if k != "instance"})
    for k in keys:
        vals = [(r[k], wt) for r, wt in zip(rows, weights) if k in r]
        out["obj_" + k] = sum(v * wt for v, wt in vals) / sum(wt for _, wt in vals)
    out["obj_num_instances"] = float(len(rows))
    out["obj_per_instance"] = rows
    return out


@torch.no_grad()
def do_evaluation(
    step: int = 0,                              #   
    cfg: OmegaConf = None,                      #   
    trainer: BasicTrainer = None,               #   
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True
):
    trainer.set_eval()

    # dataset.test_image_set = dataset.train_image_set
    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        # dataset.test_image_set = dataset.train_image_set
        #   Render Results for the images.
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
            # no vis_indices: render_images indexes the split, not the full dataset
        )
        if log_metrics:
            eval_dict = {k: float(v) for k, v in render_results.items()
                         if isinstance(v, (int, float, np.floating))}
            eval_dict["num_images"] = len(dataset.test_image_set.split_indices)
            eval_dict.update(gaussian_sdf_stats(trainer, getattr(args, "sdf_ref_ckpt", None)))
            masks_root = None
            if "RigidNodes" in cfg.model and hasattr(dataset, "annotator_masks_root"):
                masks_root = dataset.annotator_masks_root(cfg.model.RigidNodes.get("init", {}).get("annotator_masks"))
            eval_dict.update(object_geometry_stats(
                trainer, dataset, masks_root, observed_cfg=cfg.render.get("obj_geometry", None)))
            eval_dict.update(withheld_side_stats(trainer, dataset, masks_root))
            eval_dict.update(background_free_space_stats(trainer, dataset, cfg))
            logger.info(f"Test set ({eval_dict['num_images']} images) metrics: "
                        + ", ".join(f"{k}={v:.4f}" for k, v in eval_dict.items() if isinstance(v, (int, float))))
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f, indent=2)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/iccv{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/iccv{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
            )

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            #num_timestamps=dataset.num_test_timesteps,
            num_timestamps=50,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=True,
            start_idx=0
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    

    # car_id_mask = trainer.models['RigidNodes'].point_ids[..., 0] == 3
    # trainer.models['RigidNodes']._means[car_id_mask,1]-=5       #   -=x is closer to the screen.
    # trainer.models['RigidNodes']._means[car_id_mask,0]+=5       #    
    #   Remove rigid instances
    # trainer.models['RigidNodes'].remove_instances([3])

    if cfg.render.render_full:

        logger.info("Evaluating Full Set...")        
        # full-res frames of every output key are held in host RAM per chunk (~0.2 GB per image)
        chunk_len = cfg.render.get("eval_chunk_len", 25)
        full_metrics = []  # (num images, metrics) per chunk
        num_chunks = int(np.ceil(len(dataset.full_image_set.split_indices)/chunk_len))
        num_chunks = min(num_chunks, cfg.render.get("eval_max_chunks", num_chunks))

        for chunk_id in range(num_chunks):
            start_id = chunk_id*chunk_len
            end_id = (chunk_id+1)*chunk_len
            if end_id > len(dataset.full_image_set.split_indices):
                end_id = len(dataset.full_image_set.split_indices)
            vis_indices = dataset.full_image_set.split_indices[start_id:end_id]

            render_results = render_images(
                trainer=trainer,
                dataset=dataset.full_image_set,
                compute_metrics=True,
                compute_error_map=cfg.render.vis_error,
            vis_indices=vis_indices
            )

            full_metrics.append((len(vis_indices), {
                k: float(v) for k, v in render_results.items() if isinstance(v, (int, float, np.floating))
            }))
            if args.render_video_postfix is None:
                video_output_pth = f"{cfg.log_dir}/iccv{post_fix}/full_set_{step}.mp4"
            else:
                video_output_pth = (
                    f"{cfg.log_dir}/iccv{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
                )

            print(video_output_pth)
            print(f"START ID: {start_id}")

            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                layout=dataset.layout,
                num_timestamps = len(vis_indices),
                #num_timestamps=dataset.num_img_timesteps,
                keys=render_keys,
                num_cams=dataset.pixel_source.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
                start_idx = start_id
            )

            # free the chunk before rendering the next one; keeping all chunks exhausts host RAM
            del render_results, vis_frame_dict
            gc.collect()
            torch.cuda.empty_cache()
        total = sum(n for n, _ in full_metrics)
        avg = {k: sum(n * m[k] for n, m in full_metrics) / total for k in full_metrics[0][1]}
        avg["num_images"] = total
        avg.update(gaussian_sdf_stats(trainer, getattr(args, "sdf_ref_ckpt", None)))
        logger.info(f"Full set ({total} images) metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        with open(f"{cfg.log_dir}/metrics_eval/full_set_{step}{post_fix}.json", "w") as f:
            json.dump(avg, f, indent=2)
        
    # render_novel_cfg = cfg.render.get("render_novel", None)
    # render_novel_cfg = None
    # if render_novel_cfg is not None:
    #     logger.info("Rendering novel views...")
    #     render_traj = dataset.get_novel_render_traj(
    #         traj_types=render_novel_cfg.traj_types,
    #         target_frames=render_novel_cfg.get("frames", dataset.frame_num),
    #     )
    #     video_output_dir = f"{cfg.log_dir}/iccv{post_fix}/novel_{step}"
    #     if not os.path.exists(video_output_dir):
    #         os.makedirs(video_output_dir)
        
    #     for traj_type, traj in render_traj.items():
    #         # Prepare rendering data
    #         render_data = dataset.prepare_novel_view_render_data(traj)
            
    #         # Render and save video
    #         save_path = os.path.join(video_output_dir, f"{traj_type}.mp4")
    #         render_novel_views(
    #             trainer, render_data, save_path,
    #             fps=render_novel_cfg.get("fps", cfg.render.fps)
    #         )
    #         logger.info(f"Saved novel view video for trajectory type: {traj_type} to {save_path}")
            
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )
    
    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

    print(f"args.resume_from: {args.resume_from}")
    print(f"Done loading checkpoints.")

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        # "Background_rgbs",
        # "RigidNodes_rgbs",
        # "DeformableNodes_rgbs",
        # "SMPLNodes_rgbs",
        "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths"
        # "mask",
        # "rgb_sky"
    ]
    # if cfg.render.vis_lidar:
    #     render_keys.insert(0, "lidar_on_images")
    # if cfg.render.vis_sky:
    #     render_keys += ["rgb_sky_blend", "rgb_sky"]
    # if cfg.render.vis_error:
    #     render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = True
    cfg.logging.save_seperate_video = True

    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix="_eval"
    )
    
    # if args.enable_viewer:
    #     print("Viewer running... Ctrl+C to exit.")
    #     time.sleep(1000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str, required=True)
    parser.add_argument("--sdf_ref_ckpt", type=str, default=None, help="SDFGS checkpoint whose instance SDFs score a plain RigidNodes model's gaussians")
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)