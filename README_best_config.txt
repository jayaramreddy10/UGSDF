Best configuration so far: people as 3445 tracked mesh points with the `window` motion MLP.
Waymo sequence 023, every 10th frame held out (test_image_stride=10), held-out poses by cubic interpolation.

Held-out results (no held-out images used), run humans_motion_window_023_ho10:
  PSNR per image 32.12, human box 28.29, vehicle box 28.99
  SMPL baseline (baseline_masked_fixdepth_023_ho10): 31.89 / 28.05 / 28.87
  One seed, one sequence.

Contents (unzip next to the data; paths are relative to the drivestudio repo root):
  code                     the repo at commit 48036f8 (branch sdfgs)
  configs/omnire_tracks_window.yaml   the configuration, with the training recipe in its header
  work_dirs/sdfgs/humans_motion_window_023_ho10/   saved config, final checkpoint, metrics, logs
  work_dirs/sdfgs/baseline_masked_fixdepth_023_ho10/exports/point_tracks_trained_3445.npz
                           the point tracks the config reads (ctrl.tracks_path)
  work_dirs/sdfgs/logs/    training, held-out pose evaluation and error breakdown logs of this run

Evaluate:  python tools/eval_human_poses.py work_dirs/sdfgs/humans_motion_window_023_ho10 --modes cubic --rigid_modes
Needs the Waymo 023 data under data/waymo/processed/training/023 and the annotator masks (see annotations_waymo_023.zip).
