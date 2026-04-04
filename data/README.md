# Dataset Layout

This project supports two main dataset storage styles:

- original video or RGB roots, where each class is a directory and each sample is a video file
- precomputed motion roots, where each class is a directory and each sample is a `.zst` motion tensor file

The loaders do not expect a train/val/test split to be encoded in the directory tree itself. Splits are usually defined by manifest files such as the ones under `tc-clip/datasets_splits/`.

## Expected Structure

Original video or RGB roots:

```text
dataset_root/
├── ClassA/
│   ├── sample_0001.avi
│   ├── sample_0002.mp4
│   └── ...
├── ClassB/
│   ├── sample_0101.avi
│   └── ...
└── ...
```

Precomputed motion roots:

```text
motion_root/
├── ClassA/
│   ├── sample_0001.zst
│   ├── sample_0002.zst
│   └── ...
├── ClassB/
│   ├── sample_0101.zst
│   └── ...
└── ...
```

Examples:

- UCF-101 RGB/video:
  `path/to/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi`
- UCF-101 motion:
  `path/to/UCF101_motion/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.zst`
- Kinetics RGB/video:
  `path/to/Kinetics/k400/train/<class>/<video>`
- Kinetics motion:
  `path/to/Kinetics/k400_mhi_of/train/<class>/<sample>.zst`

## Manifests

Most experiments rely on manifest files instead of scanning the whole dataset blindly.

Typical manifest format:

```text
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 0
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02.avi 0
Archery/v_Archery_g01_c01.avi 1
```

Important details:

- The path is relative to the dataset root passed as `root_dir`.
- The label id must match the CSV used via `class_id_to_label_csv`.
- For motion datasets, the manifest path should usually point to the original sample stem or the `.zst` file name used in the motion root.
- The code tries to resolve some filename mismatches, but matching stems and class folders is still the safest setup.

## Label CSV

Class labels are normally provided through a CSV:

```text
id,name
0,ApplyEyeMakeup
1,Archery
```

The numeric ids in the manifest must align with this CSV.

## Recommendations For New Users

- Keep one root per modality or representation:
  `.../UCF-101`, `.../UCF101_motion`, `.../hmdb51`, `.../hmdb51_motion`, `.../Kinetics/k400`, `.../Kinetics/k400_mhi_of`
- Keep class folder names stable between RGB and motion variants when they refer to the same dataset.
- Keep sample stems stable across variants when possible:
  `video.avi` and `video.zst`
- Put train/val/test membership in manifest files, not in separate copied directory trees.
- For motion finetuning with `motion_data_source=zstd`, use the precomputed `.zst` root.
- For motion evaluation with `motion_data_source=video`, use the original video root.

## Notes

- `VideoMotionDataset` computes motion features on the fly from original videos.
- `MotionTwoStreamZstdDataset` loads precomputed motion tensors from `.zst`.
- `RGBVideoClipDataset` reads RGB clips directly from the original dataset root.
- If you adopt a different extension or naming convention, update manifests first and only change loader code if manifest resolution is no longer enough.
