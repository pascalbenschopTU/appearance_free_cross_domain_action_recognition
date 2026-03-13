# Privacy Evaluation

This folder now contains two separate privacy benchmark paths:

- `PA-HMDB51` with category-style privacy labels
- `STPrivacy` with binary identifiability labels for `VP-HMDB51` and `VP-UCF101`

What it does:

- Uses the official STPrivacy pickles in `data/stprivacy/annotations`.
- Reuses the same two-stream encoder as the main pipeline for either:
  - `motion` (`MHI + optical flow`)
  - `rgb`
- Trains one privacy attacker per attribute and per official split.
- Uses the official action-recognition split protocol:
  - HMDB51: `train{1,2,3}.txt` / `test{1,2,3}.txt`
  - UCF101: `train{1,2,3}.txt` / `test{1,2,3}.txt`
- Writes CSV/JSON metrics, saved predictions, checkpoints, and SVG/PDF plots.

STPrivacy labels are binary identifiability labels, not category labels like `male/female` or `white/black`.
Each attribute is predicted as:

- `not_identifiable`
- `identifiable`

Attributes:

- `face`
- `skin_color`
- `gender`
- `nudity`
- `relationship`

Default motion settings match the existing local motion runs:

- `img_size=224`
- `mhi_frames=32`
- `flow_frames=128`
- `flow_hw=112`
- `diff_threshold=15`
- `mhi_windows=25`
- `flow_backend=farneback`
- `fuse=avg_then_proj`

RGB comparison is supported with:

- `--input_modality rgb`
- `--rgb_frames 64`
- `--rgb_sampling uniform`
- `--rgb_norm i3d`

Joint action+privacy mode is also available in the STPrivacy trainer:

- `--joint_action_privacy`
- trains one action head plus one binary privacy head per selected attribute
- reports action `Top-1` and mean privacy `F1/cMAP`

Freezing:

- `--freeze_strategy auto` is intended for these small train splits.
- If `--pretrained_ckpt` is set, `auto` freezes most of the encoder and only fine-tunes:
  - `mixed_5b`
  - `mixed_5c`
  - projection layers
  - privacy head
- If no pretrained checkpoint is provided, `auto` falls back to full fine-tuning.

Preparation:

- `--prepare_only` on the main trainer only builds the privacy manifests, dataset metadata, and plots for a selected run.

Example commands:

```powershell
python models/appearance_free_cross_domain_action_recognition/privacy/train_stprivacy_privacy_cv.py --dataset_name hmdb51 --prepare_only
```

```powershell
python models/appearance_free_cross_domain_action_recognition/privacy/train_stprivacy_privacy_cv.py --dataset_name hmdb51 --attributes all
```

```powershell
python models/appearance_free_cross_domain_action_recognition/privacy/train_stprivacy_privacy_cv.py --dataset_name ucf101 --attributes all
```

```bash
python models/appearance_free_cross_domain_action_recognition/privacy/train_stprivacy_privacy_cv.py --dataset_name hmdb51 --input_modality rgb --attributes all
```

```bash
bash models/appearance_free_cross_domain_action_recognition/scripts/run_privacy_stprivacy_local.sh
```

```bash
PRIVACY_DATASET=ucf101 bash models/appearance_free_cross_domain_action_recognition/scripts/run_privacy_stprivacy_local.sh
```

```bash
bash models/appearance_free_cross_domain_action_recognition/scripts/run_privacy_pa_hmdb51_local.sh
```

Compatibility aliases:

- `train_hmdb51_privacy_cv.py` forwards to the dedicated `PA-HMDB51` trainer
- `run_privacy_pa_hmdb51_local.sh` is again the actual `PA-HMDB51` runner

Outputs are written under `privacy/out/...` by default, or under the runner-specific `scripts/out/privacy_stprivacy_<dataset>_<timestamp>` and `scripts/out/privacy_pa_hmdb51_<timestamp>` folders.
