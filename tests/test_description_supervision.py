import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from util import aggregate_description_logits_to_classes, build_description_match_resolver


CSV_COLUMNS = [
    "video_relpath",
    "class_dir_label",
    "class_description_key",
    "description_index",
    "description_text",
    "similarity",
    "probability",
    "margin",
    "num_sampled_frames",
    "status",
    "error",
]


class DescriptionSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.tmpdir.name)
        self.classnames = ["abseiling", "archery"]
        self.class_to_desc_indices = torch.tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            dtype=torch.long,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_csv(self, rows):
        csv_path = self.root_dir / "matches.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_exact_and_extension_insensitive_resolution(self):
        csv_path = self._write_csv(
            [
                {
                    "video_relpath": "abseiling/video_a.mp4",
                    "class_dir_label": "abseiling",
                    "class_description_key": "abseiling",
                    "description_index": "1",
                    "description_text": "desc",
                    "similarity": "0.2",
                    "probability": "0.8",
                    "margin": "0.02",
                    "num_sampled_frames": "8",
                    "status": "ok",
                    "error": "",
                },
                {
                    "video_relpath": "archery/video_b.mp4",
                    "class_dir_label": "archery",
                    "class_description_key": "archery",
                    "description_index": "2",
                    "description_text": "desc",
                    "similarity": "0.3",
                    "probability": "0.7",
                    "margin": "0.04",
                    "num_sampled_frames": "8",
                    "status": "ok",
                    "error": "",
                },
            ]
        )
        resolver = build_description_match_resolver(
            csv_path=csv_path,
            root_dir=self.root_dir,
            classnames=self.classnames,
            class_to_desc_indices=self.class_to_desc_indices,
        )

        exact = resolver.resolve(self.root_dir / "abseiling" / "video_a.mp4")
        self.assertEqual(exact.class_index, 0)
        self.assertEqual(exact.description_abs_index, 1)

        stem_match = resolver.resolve(self.root_dir / "abseiling" / "video_a.zst")
        self.assertEqual(stem_match.class_index, 0)
        self.assertEqual(stem_match.description_abs_index, 1)

    def test_ambiguous_stem_only_resolution_raises(self):
        csv_path = self._write_csv(
            [
                {
                    "video_relpath": "abseiling/shared.mp4",
                    "class_dir_label": "abseiling",
                    "class_description_key": "abseiling",
                    "description_index": "0",
                    "description_text": "desc",
                    "similarity": "0.2",
                    "probability": "0.8",
                    "margin": "0.02",
                    "num_sampled_frames": "8",
                    "status": "ok",
                    "error": "",
                },
                {
                    "video_relpath": "archery/shared.mp4",
                    "class_dir_label": "archery",
                    "class_description_key": "archery",
                    "description_index": "0",
                    "description_text": "desc",
                    "similarity": "0.3",
                    "probability": "0.7",
                    "margin": "0.04",
                    "num_sampled_frames": "8",
                    "status": "ok",
                    "error": "",
                },
            ]
        )
        resolver = build_description_match_resolver(
            csv_path=csv_path,
            root_dir=self.root_dir,
            classnames=self.classnames,
            class_to_desc_indices=self.class_to_desc_indices,
        )

        with self.assertRaises(KeyError):
            resolver.resolve(self.root_dir / "other_parent" / "shared.zst")

    def test_soft_targets_are_normalized_and_ranked(self):
        csv_path = self._write_csv(
            [
                {
                    "video_relpath": "abseiling/video_a.mp4",
                    "class_dir_label": "abseiling",
                    "class_description_key": "abseiling",
                    "description_index": "1",
                    "description_text": "desc",
                    "similarity": "0.2",
                    "probability": "0.8",
                    "margin": "0.02",
                    "num_sampled_frames": "8",
                    "status": "ok",
                    "error": "",
                },
                {
                    "video_relpath": "archery/video_b.mp4",
                    "class_dir_label": "archery",
                    "class_description_key": "archery",
                    "description_index": "3",
                    "description_text": "desc",
                    "similarity": "0.3",
                    "probability": "0.7",
                    "margin": "0.08",
                    "num_sampled_frames": "8",
                    "status": "ok",
                    "error": "",
                },
            ]
        )
        resolver = build_description_match_resolver(
            csv_path=csv_path,
            root_dir=self.root_dir,
            classnames=self.classnames,
            class_to_desc_indices=self.class_to_desc_indices,
        )
        targets, matched = resolver.build_targets(
            [
                self.root_dir / "abseiling" / "video_a.zst",
                self.root_dir / "archery" / "video_b.mp4",
            ],
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(targets.shape), (2, 10))
        self.assertTrue(torch.allclose(targets.sum(dim=1), torch.ones(2), atol=1e-6))

        for row_idx, matched_idx in enumerate(matched.tolist()):
            matched_value = targets[row_idx, matched_idx].item()
            class_idx = 0 if row_idx == 0 else 1
            same_class = self.class_to_desc_indices[class_idx]
            other_class = self.class_to_desc_indices[1 - class_idx]
            other_same = [idx for idx in same_class.tolist() if idx != matched_idx]
            self.assertTrue(all(matched_value > targets[row_idx, idx].item() for idx in other_same))
            self.assertTrue(all(targets[row_idx, other_same[0]].item() > targets[row_idx, idx].item() for idx in other_class.tolist()))

    def test_description_logits_aggregate_to_class_logits(self):
        logits = torch.tensor([[1.0, 2.0, 0.5, -0.5, 0.0, 3.0, 2.5, 1.5, 1.0, 0.5]])
        aggregated = aggregate_description_logits_to_classes(logits, self.class_to_desc_indices)
        expected = torch.stack(
            [
                torch.logsumexp(logits[:, :5], dim=-1) - torch.log(torch.tensor(5.0)),
                torch.logsumexp(logits[:, 5:], dim=-1) - torch.log(torch.tensor(5.0)),
            ],
            dim=-1,
        )
        self.assertTrue(torch.allclose(aggregated, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
