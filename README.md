# WSIs Rigid Registration

Rigid registration pipeline for Whole Slide Images (WSIs), including DISK keypoint detector finetuning, LightGlue matcher finetuning, and benchmark evaluation on multiple datasets (HyReCo, ANHIR, ACROBAT).

---

## 1. finetuning_DISK

Finetune the **DISK** keypoint detector on pathology/WSI data using reinforcement learning with identity-grid and affine rewards.

**Code base.** This module is based on [cvlab-epfl/disk `train.py`](https://github.com/cvlab-epfl/disk/blob/master/train.py). This repository contains **only the modified or added code**; the full DISK training codebase is required (clone and merge or run from the original repo with these files overlaid).

### Structure

| Path | Description |
|------|-------------|
| `train_patch.py` | Main training script (REINFORCE, patch-based). |
| `disk/data/` | Datasets (`Patch_Dataset`, `HyReCoDataset`) and WSI augmentations (`wsi_augments.py`: stain jitter, channel mixing, affine/elastic/piecewise transforms). |
| `disk/loss/` | Reward and loss: `rewards.py` (IdentityGridReward, AffineReward, etc.), `reinforce.py`, `discrete_metric.py`, `pose_metric.py`, `ransac.py`. |
| `disk/model/benchmark_hyreco.py` | DISK + CycleMatcher registration and TRE evaluation on HyReCo-style HE/IHC pairs. |

### Usage

```bash
cd finetuning_DISK
python train_patch.py --experiment <exp_name> \
  --data_dir <path_to_training_patches> \
  --benchmark-data-dir <path_to_hyreco_benchmark> \
  --n-epochs 30 --batch-size 16 --lr 1e-5 \
  --height 512 --width 512
```

- Set `data_dir` and `benchmark-data-dir` in the script or CLI.
- Optional: `--threshold`, `--lm-fp`, `--lm-roi-fp`, `--entropy-loss-alpha`, `--window-size`, `--desc-dim`, etc.
- Checkpoints are saved under the experiment directory; benchmark (TRE 90th percentile, mean, num_matches, num_inliers) runs during training when benchmark data is provided.

---

## 2. finetuning_LightGlue

Finetune the **LightGlue** matcher (with a frozen DISK extractor) on affine-augmented patch pairs for WSI-style matching.

**Code base.** This module is based on [cvg/glue-factory](https://github.com/cvg/glue-factory/tree/main/gluefactory). This repository contains **only the modified or added code**; the full glue-factory codebase is required (clone and merge or run from the original repo with these files overlaid).

### Structure

| Path | Description |
|------|-------------|
| `finetune.py` | Main training entry (config-driven, supports distributed). Saves LightGlue matcher `.pth` each epoch. |
| `configs/disk+lightglue_affine.yaml` | Default config: dataset `affines_patch`, DISK extractor (trainable=False), affine matcher GT, LightGlue matcher, lr schedule, etc. |
| `datasets/affines_patch.py` | Patch dataset with affine augmentation. |
| `datasets/augmentations_wsi.py` | WSI-oriented augmentations (stain jitter, channel mixing, affine/elastic/piecewise). |
| `geometry/gt_generation.py` | GT generation (affine warping, identity grid, `warp_points_to_orig`). |

### Usage

1. Set in `configs/disk+lightglue_affine.yaml` (or override via CLI):
   - `data.data_dir`, `data.train_size`, `data.val_size`
   - `model.extractor.weights` (DISK weights)
   - `model.matcher.weights` (LightGlue weights, or `None` to train from scratch)

2. Run from the **finetuning_LightGlue** directory (so that the config path and submodules resolve correctly):

```bash
cd finetuning_LightGlue
python -m finetune <experiment_name> --conf configs/disk+lightglue_affine.yaml [dotlist overrides]
```

Example overrides:

```bash
python -m finetune my_run --conf configs/disk+lightglue_affine.yaml data.data_dir=/path/to/data model.extractor.weights=/path/to/disk.pth
```

- Use `--restore` to resume from the experiment’s `config.yaml` and checkpoints.
- Use `--distributed` for multi-GPU training.
- Outputs (checkpoints, config, logs) are written under the experiment path (e.g. `TRAINING_PATH/<experiment_name>`).

---

## 3. registration

Run **benchmark registration** with DISK + (LightGlue or CycleMatcher) on three datasets: **HyReCo** (HE/IHC), **ANHIR**, and **ACROBAT**.

**Code base.** The model implementations in `registration/models/` (e.g. LightGlue, matchers) are from [cvg/LightGlue](https://github.com/cvg/LightGlue/tree/main/lightglue). Integrate or copy the LightGlue model code from that repository as needed to run the benchmarks.

### Structure

| Path | Description |
|------|-------------|
| `benchmark_hyreco.py` | HyReCo benchmark: `WSI_Registration` with DISK + LightGlue or CycleMatcher; HE/IHC reference; optional tiling and local refinement. |
| `benchmark_anhir.py` | ANHIR benchmark: same pipeline, `ANHIRDataset`, configurable MPP/resize. |
| `benchmark_acrobat.py` | ACROBAT benchmark: rotation search + downscaled matching; `ACROBATDataset`. |
| `models/` | DISK, LightGlue, CycleMatcher wrappers. |
| `datasets/` | `HyReCoDataset`, `ANHIRDataset`, `ACROBATDataset`, slide/utils (e.g. MPP from TIFF). |
| `utils/` | Affine/decompose (`calculator.py`), visualization/save (`save.py`, `viz2d`), logger, preprocessing. |
| `configs/` | `config_disk_hyreco.yaml`, `config_disk_anhir.yaml`, `config_disk_acrobat.yaml`. |

### Usage

From the **registration** directory:

**HyReCo**

```bash
cd registration
python benchmark_hyreco.py --config ./configs/config_disk_hyreco.yaml
```

**ANHIR**

```bash
python benchmark_anhir.py --config ./configs/config_disk_anhir.yaml
```

**ACROBAT**

```bash
python benchmark_acrobat.py --config ./configs/config_disk_acrobat.yaml
```

In each config YAML set:

- `data_dir` (dataset root)
- `extractor.weights` (DISK)
- `matcher.weights` (LightGlue; or use CycleMatcher via matcher name)
- Optionally `output_dir`, `tile_size`, `max_num_keypoints`, `tiling`, `local_refinement_top_k`, etc.

Results (and optional visualizations) are written to the configured `output_dir`.

---

## Summary

| Module | Purpose |
|--------|---------|
| **finetuning_DISK** | Train DISK on WSI patches with RL (IdentityGrid/Affine rewards); optional HyReCo benchmark during training. |
| **finetuning_LightGlue** | Train LightGlue on affine patch pairs with frozen DISK; config in `configs/disk+lightglue_affine.yaml`. |
| **registration** | Run DISK + matcher on HyReCo, ANHIR, and ACROBAT via `benchmark_*.py` and per-dataset configs. |

Use finetuned DISK and LightGlue weights from the first two modules in the registration configs to evaluate on the three benchmarks.
