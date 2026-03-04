import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute text embeddings and t-SNE plots from label files. "
            "This script intentionally writes only 4 artifacts."
        )
    )
    parser.add_argument(
        "label_files",
        nargs="+",
        help="Input label files (.json or .csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <repo>/artifacts/text_embeddings.",
    )
    parser.add_argument(
        "--st-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name.",
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
        help="Device for embedding models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for text encoding.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=10.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for t-SNE.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable CLIP embedding + CLIP t-SNE PDF.",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(output_dir_arg) if output_dir_arg else (repo_root / "artifacts" / "text_embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _first_non_empty_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    return text
    return None


def load_json_texts(json_path: Path) -> List[Tuple[str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object.")

    items: List[Tuple[str, str]] = []
    for key, value in data.items():
        text = _first_non_empty_text(value)
        if text is not None:
            items.append((str(key), text))

    def sort_key(item: Tuple[str, str]):
        key = item[0]
        try:
            return (0, int(key))
        except Exception:
            return (1, key)

    items.sort(key=sort_key)
    return items


def load_csv_texts(csv_path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "name" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a 'name' column.")

        for i, row in enumerate(reader):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            key = (row.get("id") or str(i)).strip()
            items.append((key, name))
    return items


def load_texts_from_file(path: Path) -> List[Tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_texts(path)
    if suffix == ".csv":
        return load_csv_texts(path)
    raise ValueError(f"Unsupported file type for {path}. Use .json or .csv.")


def collect_samples(label_files: List[str]) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    for file_str in label_files:
        path = Path(file_str)
        dataset = path.stem
        items = load_texts_from_file(path)
        for key, text in items:
            samples.append({"dataset": dataset, "key": str(key), "text": text})
    if not samples:
        raise ValueError("No valid class text entries were found.")
    return samples


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def encode_sentence_transformers(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype(np.float32)


def load_clip_text_encoder(device: torch.device, model_name: str):
    try:
        import clip
    except Exception as exc:
        raise RuntimeError(
            "Could not import 'clip'. Install OpenAI CLIP with: "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    clip_model, _ = clip.load(model_name, device=device, jit=False)
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)
    return clip_model, clip.tokenize


@torch.no_grad()
def encode_clip_texts(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    dev = torch.device(device)
    clip_model, tokenize_fn = load_clip_text_encoder(dev, model_name)
    all_feats = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tok = tokenize_fn(batch).to(dev)
        feats = clip_model.encode_text(tok)
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())

    emb = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
    return emb


def tsne_2d(x: np.ndarray, random_state: int, perplexity: float) -> np.ndarray:
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute t-SNE.")
    max_valid = max(1.0, (x.shape[0] - 1) / 3.0)
    effective_perplexity = min(perplexity, max_valid)
    return TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(x).astype(np.float32)


def save_tsne_pdf(proj: np.ndarray, samples: List[Dict[str, str]], title: str, out_path: Path):
    datasets = [s["dataset"] for s in samples]
    unique_datasets = list(dict.fromkeys(datasets))
    plt.figure(figsize=(10, 8))

    for dataset in unique_datasets:
        idx = [i for i, d in enumerate(datasets) if d == dataset]
        pts = proj[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.8, label=dataset)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def main():
    args = parse_args()
    out_dir = resolve_output_dir(args.output_dir)

    samples = collect_samples(args.label_files)
    texts = [s["text"] for s in samples]

    print(f"Loaded {len(samples)} class texts from {len(args.label_files)} file(s).")
    print(f"Output directory: {out_dir}")

    st_emb = encode_sentence_transformers(
        texts=texts,
        model_name=args.st_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    st_emb = l2_normalize_np(st_emb)
    np.save(out_dir / "sentence_transformer_embeddings.npy", st_emb)
    st_proj = tsne_2d(st_emb, random_state=args.random_state, perplexity=args.perplexity)
    save_tsne_pdf(
        st_proj,
        samples,
        title=f"SentenceTransformer ({args.st_model}) t-SNE",
        out_path=out_dir / "sentence_transformer_tsne.pdf",
    )

    if not args.no_clip:
        clip_emb = encode_clip_texts(
            texts=texts,
            model_name=args.clip_model,
            device=args.device,
        )
        clip_emb = l2_normalize_np(clip_emb)
        np.save(out_dir / "clip_embeddings.npy", clip_emb)
        clip_proj = tsne_2d(clip_emb, random_state=args.random_state, perplexity=args.perplexity)
        save_tsne_pdf(
            clip_proj,
            samples,
            title=f"CLIP Text ({args.clip_model}) t-SNE",
            out_path=out_dir / "clip_tsne.pdf",
        )

    print("Wrote artifacts:")
    print(f"  {out_dir / 'sentence_transformer_embeddings.npy'}")
    print(f"  {out_dir / 'sentence_transformer_tsne.pdf'}")
    if not args.no_clip:
        print(f"  {out_dir / 'clip_embeddings.npy'}")
        print(f"  {out_dir / 'clip_tsne.pdf'}")


if __name__ == "__main__":
    main()
