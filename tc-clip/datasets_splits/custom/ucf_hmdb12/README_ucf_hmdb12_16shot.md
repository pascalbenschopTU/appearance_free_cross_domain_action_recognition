# UCF-HMDB 12-class 16-shot splits

- UCF source: `tc-clip/datasets_splits/ucf_splits/train1_few_shot_16.txt`
- HMDB source: `tc-clip/datasets_splits/hmdb_splits/train1_few_shot_16.txt` (+ top-up from `val1.txt` if needed)
- Output class ids are remapped to common taxonomy `0..11` for cross-dataset training/eval.
- UCF `climb` merges `RockClimbingIndoor` and `RopeClimbing` as 8+8 samples.
- Filenames starting with `#` are excluded because `dataset_split_txt` parser treats those lines as comments.

## Mapping

| new_id | common_name | hmdb_id | ucf_id(s) |
|---:|---|---:|---|
| 0 | climb | 5 | 73,74 |
| 1 | fencing | 13 | 27 |
| 2 | golf | 15 | 32 |
| 3 | kick_ball | 20 | 84 |
| 4 | pullup | 26 | 69 |
| 5 | punch | 27 | 70 |
| 6 | pushup | 29 | 71 |
| 7 | ride_bike | 30 | 10 |
| 8 | ride_horse | 31 | 41 |
| 9 | shoot_ball | 34 | 7 |
| 10 | shoot_bow | 35 | 2 |
| 11 | walk | 49 | 97 |
