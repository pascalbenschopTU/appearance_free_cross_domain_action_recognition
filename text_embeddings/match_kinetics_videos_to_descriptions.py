import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# The on-disk Kinetics folder names are not perfectly aligned with the
# normalized keys used in Kinetics_descriptions.json.
CLASS_KEY_ALIASES = {
    "barbequing": "barbecuing",
    "driving car": "driving a car",
    "driving tractor": "driving a tractor",
    "eating burger": "eating a burger",
    "giving or receiving award": "giving or receiving an award",
    "passing American football (in game)": "passing american football (in game)",
    "passing American football (not in game)": "passing american football (not in game)",
    "skiing (not slalom or crosscountry)": "skiing (not slalom or cross-country)",
    "skiing crosscountry": "skiing cross-country",
    "swimming breast stroke": "swimming breaststroke",
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign a single in-class description to each Kinetics video by "
            "matching CLIP video embeddings against CLIP text embeddings."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../../datasets/Kinetics/k400/train"),
        help="Root folder with class subdirectories of Kinetics videos.",
    )
    parser.add_argument(
        "--descriptions-json",
        type=Path,
        default=Path("tc-clip/labels/custom/Kinetics_descriptions.json"),
        help="JSON mapping class labels to 5 descriptions.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("text_embeddings/kinetics400_train_description_matches_clip.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("text_embeddings/kinetics400_train_description_matches_clip_summary.json"),
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="OpenAI CLIP model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Uniformly sampled frames per video.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Frame batch size for CLIP image encoding.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional cap for debugging. Use 0 for all videos.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Progress print interval.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV instead of resuming.",
    )
    return parser.parse_args()


def load_clip_model(device: torch.device, model_name: str):
    try:
        import clip
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import 'clip'. Install OpenAI CLIP with "
            "'pip install git+https://github.com/openai/CLIP.git'."
        ) from exc

    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess, clip.tokenize


def resolve_class_key(class_dir_label: str, descriptions: Dict[str, List[str]]) -> str:
    if class_dir_label in descriptions:
        return class_dir_label
    mapped = CLASS_KEY_ALIASES.get(class_dir_label)
    if mapped and mapped in descriptions:
        return mapped
    raise KeyError(f"No description key found for class folder '{class_dir_label}'.")


def load_descriptions(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    cleaned: Dict[str, List[str]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            raise ValueError(f"Expected a list of descriptions for key '{key}'.")
        texts = [str(item).strip() for item in value if str(item).strip()]
        if not texts:
            raise ValueError(f"Key '{key}' does not contain any usable descriptions.")
        cleaned[str(key)] = texts
    return cleaned


@torch.inference_mode()
def encode_text_bank(
    model,
    tokenize_fn,
    descriptions: Dict[str, List[str]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    text_bank: Dict[str, torch.Tensor] = {}
    for class_key, texts in descriptions.items():
        tokens = tokenize_fn(texts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        text_bank[class_key] = text_features
    return text_bank


def iter_videos(dataset_root: Path) -> Iterable[Path]:
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for video_path in sorted(class_dir.iterdir()):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_SUFFIXES:
                yield video_path


def _sample_indices(frame_count: int, num_frames: int) -> np.ndarray:
    if frame_count <= 1:
        return np.zeros((1,), dtype=np.int64)
    return np.unique(np.linspace(0, frame_count - 1, num=min(num_frames, frame_count), dtype=np.int64))


def read_uniform_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_indices = _sample_indices(frame_count, num_frames)
        frames: List[Image.Image] = []

        for frame_index in target_indices.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        if not frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while len(frames) < num_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

        if not frames:
            raise RuntimeError("No decodable frames were found.")

        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        return frames
    finally:
        cap.release()


@torch.inference_mode()
def encode_video_frames(
    model,
    preprocess,
    frames: Sequence[Image.Image],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    tensors = [preprocess(frame) for frame in frames]
    batches: List[torch.Tensor] = []
    for start in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[start : start + batch_size], dim=0).to(device)
        feats = model.encode_image(batch)
        feats = F.normalize(feats, dim=-1)
        batches.append(feats)
    frame_features = torch.cat(batches, dim=0)
    video_feature = F.normalize(frame_features.mean(dim=0, keepdim=True), dim=-1)
    return video_feature.squeeze(0)


def load_processed_relpaths(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["video_relpath"] for row in reader if row.get("video_relpath")}


def write_header_if_needed(output_csv: Path, overwrite: bool) -> None:
    needs_header = overwrite or (not output_csv.exists()) or output_csv.stat().st_size == 0
    if not needs_header:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_rows(output_csv: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


def build_summary(output_csv: Path) -> Dict[str, object]:
    class_counts: Dict[str, Counter] = defaultdict(Counter)
    class_prob_sums: Dict[str, float] = defaultdict(float)
    class_margin_sums: Dict[str, float] = defaultdict(float)
    class_valid_counts: Counter = Counter()
    status_counts: Counter = Counter()
    total_rows = 0

    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            class_dir_label = row["class_dir_label"]
            status = row["status"]
            status_counts[status] += 1
            if status != "ok":
                continue

            class_counts[class_dir_label][row["description_index"]] += 1
            class_valid_counts[class_dir_label] += 1
            class_prob_sums[class_dir_label] += float(row["probability"])
            class_margin_sums[class_dir_label] += float(row["margin"])

    per_class = {}
    for class_dir_label in sorted(class_counts):
        valid = class_valid_counts[class_dir_label]
        per_class[class_dir_label] = {
            "num_videos": int(valid),
            "description_counts": dict(sorted(class_counts[class_dir_label].items(), key=lambda item: int(item[0]))),
            "mean_probability": class_prob_sums[class_dir_label] / max(valid, 1),
            "mean_margin": class_margin_sums[class_dir_label] / max(valid, 1),
        }

    return {
        "total_rows": total_rows,
        "status_counts": dict(status_counts),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    descriptions_json = args.descriptions_json.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()

    descriptions = load_descriptions(descriptions_json)

    device = torch.device(args.device)
    model, preprocess, tokenize_fn = load_clip_model(device=device, model_name=args.clip_model)
    text_bank = encode_text_bank(model, tokenize_fn, descriptions, device)
    logit_scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 1.0

    processed_relpaths = set()
    if not args.overwrite:
        processed_relpaths = load_processed_relpaths(output_csv)

    write_header_if_needed(output_csv, overwrite=args.overwrite)

    videos = list(iter_videos(dataset_root))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    pending_videos = [path for path in videos if str(path.relative_to(dataset_root)) not in processed_relpaths]
    print(
        f"Found {len(videos)} videos, {len(processed_relpaths)} already in CSV, "
        f"{len(pending_videos)} remaining."
    )

    buffer: List[Dict[str, object]] = []
    for idx, video_path in enumerate(tqdm(pending_videos, desc="Matching videos"), start=1):
        video_relpath = str(video_path.relative_to(dataset_root))
        class_dir_label = video_path.parent.name.replace("_", " ")

        try:
            class_description_key = resolve_class_key(class_dir_label, descriptions)
            frames = read_uniform_frames(video_path, num_frames=args.num_frames)
            video_feature = encode_video_frames(
                model=model,
                preprocess=preprocess,
                frames=frames,
                device=device,
                batch_size=args.batch_size,
            )
            scores = torch.matmul(text_bank[class_description_key], video_feature)
            scaled_scores = scores * logit_scale
            probabilities = torch.softmax(scaled_scores, dim=0)
            topk = torch.topk(scores, k=min(2, scores.numel()))

            best_index = int(topk.indices[0].item())
            best_score = float(scores[best_index].item())
            best_probability = float(probabilities[best_index].item())
            second_score = float(topk.values[1].item()) if topk.values.numel() > 1 else best_score
            margin = best_score - second_score

            row = {
                "video_relpath": video_relpath,
                "class_dir_label": class_dir_label,
                "class_description_key": class_description_key,
                "description_index": best_index,
                "description_text": descriptions[class_description_key][best_index],
                "similarity": f"{best_score:.6f}",
                "probability": f"{best_probability:.6f}",
                "margin": f"{margin:.6f}",
                "num_sampled_frames": len(frames),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            row = {
                "video_relpath": video_relpath,
                "class_dir_label": class_dir_label,
                "class_description_key": "",
                "description_index": "",
                "description_text": "",
                "similarity": "",
                "probability": "",
                "margin": "",
                "num_sampled_frames": 0,
                "status": "error",
                "error": str(exc),
            }

        buffer.append(row)
        if len(buffer) >= args.save_every:
            append_rows(output_csv, buffer)
            buffer.clear()
            print(f"Wrote {idx}/{len(pending_videos)} new rows to {output_csv}.")

    append_rows(output_csv, buffer)

    summary = build_summary(output_csv)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "descriptions_json": str(descriptions_json),
            "output_csv": str(output_csv),
            "clip_model": args.clip_model,
            "device": str(device),
            "num_frames": args.num_frames,
            "batch_size": args.batch_size,
        }
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"CSV written to: {output_csv}")
    print(f"Summary written to: {summary_json}")


if __name__ == "__main__":
    main()
