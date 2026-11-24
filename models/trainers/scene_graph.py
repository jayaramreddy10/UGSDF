from typing import Dict
import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere

logger = logging.getLogger()

class MultiTrainer(BasicTrainer):
    
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
            # self.gaussian_classes["RigidNodes"] = GSModelType.SDFNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
            # self.gaussian_classes["SMPLNodes"] = GSModelType.SDFNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
        if "CarriedNodes" in self.model_config:
            self.gaussian_classes["CarriedNodes"] = GSModelType.CarriedNodes
            # self.gaussian_classes["DeformableNodes"] = GSModelType.SDFNodes

           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys:
                #   Calls the constructors for different types of nodes.
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        if "CarriedNodes" in self.models and hasattr(self.models["CarriedNodes"], "link_body") and "SMPLNodes" in self.models:
            self.models["CarriedNodes"].link_body(self.models["SMPLNodes"])
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',        
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict}
        carriednode_pts_dict = {}
        if "CarriedNodes" in self.model_config:
            # one node per modelled person, for what the SMPL body does not cover
            persons = list(smplnode_pts_dict.keys())
            if not hasattr(self.models["CarriedNodes"], "link_body"):   # plain DeformableNodes: box-following items
                persons += [k for k in deformnode_pts_dict.keys() if int(dataset.pixel_source.instances_model_types[k]) == 1]
            carriednode_pts_dict = dataset.get_init_carried_objects(
                person_ids=persons, **self.model_config["CarriedNodes"].get("init", {}))
        
        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = [] 
        
        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            empty = False
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sub-sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                
                #   Background points are filtered out of the boxes of modelled instances.
                #   With `filter_moving_thres`, also out of every moving box, so moving
                #   objects that no node models (e.g. no annotator mask) don't become background.
                filter_ids = set(allnode_pts_dict.keys())
                moving_thres = init_cfg.get("filter_moving_thres", None)
                if moving_thres is not None:
                    unmodelled = dataset.get_moving_instance_ids(moving_thres) - filter_ids
                    logger.info(f"Background: also filtering {len(unmodelled)} moving instances without nodes")
                    filter_ids |= unmodelled
                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict={**dict.fromkeys(filter_ids), **allnode_pts_dict}
                )
                
                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"]
                )
                
            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )
                
            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )
            
            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )

            if class_name == 'CarriedNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=carriednode_pts_dict
                )
                
            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")
        
        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")
                
        if "CarriedNodes" in self.models and hasattr(self.models["CarriedNodes"], "attach_to_body"):
            self.models["CarriedNodes"].attach_to_body()
        logger.info(f"Initialized gaussians from pcd")
        self.init_ray_supervision(dataset)

    def init_ray_supervision(self, dataset: DrivingDataset) -> None:
        """
        LiDAR ray supervision that lives outside the checkpoint: the per-instance SDF rays of
        SDFGS nodes and the background's free-space grid. Called after the gaussians exist
        (fresh init or resume). Held-out frames are left out unless configured otherwise.
        """
        from models.nodes.sdf_supervision import train_frames, modelled_instance_ids
        frames_train = train_frames(dataset)
        frames_all = list(range(dataset.frame_num))

        rigid = self.models.get("RigidNodes", None)
        if rigid is not None and hasattr(rigid, "init_sdf_supervision"):
            sup = self.model_config["RigidNodes"].get("sdf", {}).get("supervision", {})
            mono = self.model_config["RigidNodes"].get("sdf", {}).get("mono", {})
            mono_surface = None
            if mono.get("points", None):
                from datasets.driving_dataset import load_lifted_surface
                mono_surface = load_lifted_surface(mono["points"], dataset.scene_idx, mono.get("min_views", 3), mono.get("min_facing", 3))
            rigid.init_sdf_supervision(
                dataset.lidar_source, pixel_source=dataset.pixel_source,
                masks_root=dataset.annotator_masks_root(self.model_config["RigidNodes"]["init"].get("annotator_masks")),
                frames=frames_train if sup.get("exclude_test_frames", True) else frames_all,
                mono_surface=mono_surface,
            )

        bg = self.models.get("Background", None)
        fs = self.model_config["Background"].get("reg", {}).get("free_space", None) if bg is not None else None
        if fs is not None and dataset.lidar_source is not None:
            from models.gaussians.free_space import FreeSpaceGrid
            ps = dataset.pixel_source
            mode = fs.get("exclude_returns", "modelled")
            if mode == "modelled":      # objects that have their own node
                ids = modelled_instance_ids(self.models, ps)
            elif mode == "moving":      # every annotated moving box, modelled or not
                ids = sorted(dataset.get_moving_instance_ids(fs.get("moving_thres", 0.5)))
            else:
                ids = []
            boxes = dict(pose=ps.instances_pose, size=ps.instances_size, active=ps.per_frame_instance_mask, ids=ids)
            logger.info(f"Background free space: returns on {len(ids)} instance boxes ({mode}) are not static")
            bg.set_free_space_grid(FreeSpaceGrid.build(
                dataset.lidar_source,
                frames_train if fs.get("exclude_test_frames", True) else frames_all,
                self.device,
                voxel=fs.get("voxel", 0.25), max_range=fs.get("max_range", 75.0),
                min_traversal=fs.get("min_traversal", 5.0), min_hits=fs.get("min_hits", 1),
                dilate=fs.get("dilate", 1), exclude_boxes=boxes,
            ))
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                    outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
        
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict