# Appearance-Free Cross-Domain Action Recognition

This directory contains the local motion-recognition code for training, finetuning, evaluation, and experiment orchestration. The repo now uses a thin-root layout:

- Stable public Python entrypoints stay at the root: `train.py`, `finetune.py`, `eval.py`
- Canonical bash launchers live under `scripts/`
- Shared implementation is grouped into `cli/`, `data/`, `models/`, and `utils/`
- `privacy/` and `tc-clip/` remain separate subprojects

## Installation

The recommended environment is the project Apptainer image. In the local setup this repo is commonly run inside a shared Apptainer container built from a project-specific definition file.

That definition captures the intended software stack for this project, including:

- Python 3.10 via Conda
- `torch` and `torchvision`
- `transformers`, `accelerate`, `pytorchvideo`, `zstandard`
- `ultralytics`
- `opencv-python-headless`
- `tensorboard`
- the local OpenAI CLIP checkout installed from `/opt/CLIP`

If you already use the provided Apptainer image, that is enough for the full environment and is the preferred option for reproducing the training and sbatch runs in this repo.

Typical usage:

```bash
apptainer exec --nv {container}.sif bash
```

Then inside the container:

```bash
cd path/to/appearance_free_cross_domain_action_recognition
python train.py --help
```

Conda is also possible if you want a local editable environment outside Apptainer. A practical Conda-based setup is:

```bash
conda create -n afcdar python=3.10
conda activate afcdar
conda install -c conda-forge ffmpeg
pip install --upgrade pip wheel "setuptools<82"
pip install "torch>=2.4" torchvision
pip install "transformers>=4.53" accelerate scipy pytorchvideo zstandard
pip install torchao ffmpeg-python imageio timm bitsandbytes==0.45.4
pip install --no-deps ultralytics==8.4.21
pip install matplotlib polars ultralytics-thop psutil py-cpuinfo pyyaml requests
pip install omegaconf tqdm h5py tensorboard
pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless || true
pip install --force-reinstall --no-deps "opencv-python-headless==4.10.0.84"
pip install --no-build-isolation <path-to-your-local-CLIP-checkout>
```

The Apptainer definition remains the best source of truth for exact package choices and should be preferred over this abbreviated list.

## Structure

```text
.
├── train.py / finetune.py / eval.py
├── config.py / dataset.py / util.py / augment.py / model.py / e2s_x3d.py
├── cli/
├── data/
├── models/
├── utils/
├── configs/
├── scripts/
├── dataset/
├── privacy/
└── tc-clip/
```

What each area is for:

- Root wrappers:
  Backward-compatible imports and CLI entrypoints.
- `cli/`:
  Main train/finetune/eval implementations and parser shims.
- `data/`:
  Datasets, collate functions, samplers, RGB helpers, and motion augmentations.
- `models/`:
  Canonical local backbone implementations for I3D and X3D.
- `utils/`:
  Checkpoint, scheduler, manifest, parsing, and text-bank helpers.
- `configs/`:
  TOML configs for training, finetuning, and evaluation.
- `dataset/`:
  Auxiliary preprocessing and manifest scripts.
- `scripts/`:
  Canonical experiment launchers and local analysis scripts.
- `privacy/`:
  Privacy experiments and notes.
- `tc-clip/`:
  Embedded TC-CLIP subproject. See `tc-clip/README.md`.

## Working Directory

Run commands from:

```bash
cd path/to/appearance_free_cross_domain_action_recognition
```

Direct Python entrypoints:

```bash
python train.py --help
python finetune.py --help
python eval.py --help
```

Canonical bash entrypoints:

```bash
bash scripts/<launcher>.sh
```

The old root-level `run_*.sh` files still exist as thin compatibility wrappers.

## Dataset Setup

Dataset roots are expected to be class-folder trees, with train/val/test membership defined by manifest files.

The two common variants are:

- original video or RGB roots:
  `root/ClassName/sample.avi`
- precomputed motion roots:
  `root/ClassName/sample.zst`

This matches the current local setup, for example:

- RGB/video:
  `path/to/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi`
- precomputed motion:
  `path/to/UCF101_motion/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.zst`

For the expected layout, manifest format, and label CSV conventions, see `appearance_free_cross_domain_action_recognition/data/README.md`.

## Runtime Artifacts

These directories are runtime outputs, not source layout:

- `out/`
- `eval_out/`
- `workspace/`
- model/cache directories such as `HF_HOME`, `TORCH_HOME`, and `XDG_CACHE_HOME`

## Main Launchers

| Launcher | Purpose | Slurm job |
| --- | --- | --- |
| `scripts/run_few_shot_back_to_back.sh` | Few-shot motion and RGB sweeps | `jobs/run_few_shot_back_to_back.sbatch` |
| `scripts/run_domain_adaptation.sh` | Domain adaptation and privacy benchmark | `jobs/run_domain_adaptation.sbatch` |
| `scripts/run_pa_hmdb51_attack_all.sh` | PA-HMDB51 attacker sweep | `jobs/run_pa_hmdb51_vit_attack.sbatch` |
| `scripts/run_surveillance_transfer_back_to_back.sh` | Surveillance transfer experiments | `jobs/run_surveillance_transfer_motion.sbatch` |
| `scripts/run_motion_i3d_full_ablation_local.sh` | Local ablation orchestrator | `jobs/run_motion_i3d_full_ablation.sbatch` |

## Common Calls

Few-shot:

```bash
bash scripts/run_few_shot_back_to_back.sh
MODEL=i3d_mhi_of FEWSHOT_SHOTS="8 16" bash scripts/run_few_shot_back_to_back.sh
```

Domain adaptation:

```bash
bash scripts/run_domain_adaptation.sh
MODELS="i3d_of_only" bash scripts/run_domain_adaptation.sh
```

Surveillance transfer:

```bash
bash scripts/run_surveillance_transfer_back_to_back.sh
MODELS="motion_mhi_of,tc_clip" TRAIN_DATASETS="rwf2000,ucf_crime" bash scripts/run_surveillance_transfer_back_to_back.sh
```

## Notes

- `scripts/train_torchvision_rgb_probe.py` is the canonical helper for torchvision RGB baselines.
- `privacy/README.md` contains privacy-specific details.
- Unused transfer, skin-tone, and local STPrivacy launcher wrappers were retired from this checkout. Keep local copies outside git if you want to reintroduce them later.
