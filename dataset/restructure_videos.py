#!/usr/bin/env python3
"""
Restructure raw CI3D videos into class folders that the generic loader can read.

Expected raw input layout:
  <root>/<subject_id>/<ClassName> <idx>.mp4

Example:
  .../datasets/CI3D/s04/videos/60457274/Grab 7.mp4

Output layout:
  <out_root>/<ClassName>/<subject_id>_<ClassName>_<idx>.mp4

Example:
  .../datasets/CI3D/s03/videos/Grab/60457274_Grab_7.mp4

The script can copy or move files and is safe to re-run because it skips files
already inside the output class folders.
"""

import re
import shutil
from pathlib import Path

CLASSES = ["Grab", "Handshake", "Hit", "HoldingHands", "Hug", "Kick", "Posing", "Push"]
VIDEO_EXTS = {".mp4"}  # extend if needed

# Detect class token anywhere in filename
CLASS_RE = re.compile(r"(" + "|".join(map(re.escape, CLASSES)) + r")")


def ensure_unique_path(path: Path) -> Path:
    """If path exists, append _dupN before suffix."""
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    k = 1
    while True:
        cand = path.with_name(f"{stem}_dup{k}{suf}")
        if not cand.exists():
            return cand
        k += 1


def find_class(name: str) -> str | None:
    m = CLASS_RE.search(name)
    return m.group(1) if m else None


def extract_id_from_path(src: Path) -> str | None:
    """
    Prefer numeric ID from parent folder name: videos/<id>/<file>.mp4
    Fallback: first 6+ digit number found anywhere in the path.
    """
    parent = src.parent.name
    if parent.isdigit():
        return parent

    # fallback: any long-ish number anywhere in the path
    m = re.search(r"(\d{6,})", str(src))
    return m.group(1) if m else None


def extract_idx_from_name(stem: str) -> str | None:
    """
    Get the clip index from filename stem.
    Examples:
      "Grab 1" -> 1
      "Hit 25" -> 25
      "Grab_1" -> 1
      "Something_Else_12" -> 12
    Heuristic: last number in the stem.
    """
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else None


def should_skip(src: Path, out_root: Path) -> bool:
    """
    Prevent processing files that are already in the output class folders.
    This avoids recursion / re-processing on subsequent runs.
    """
    try:
        rel = src.resolve().relative_to(out_root.resolve())
    except Exception:
        return False

    # If file is under out_root/<Class>/..., skip
    if rel.parts and rel.parts[0] in CLASSES:
        return True
    return False


def restructure(root: Path, out_root: Path, dry_run: bool, move: bool):
    # Create output class folders
    for cls in CLASSES:
        (out_root / cls).mkdir(parents=True, exist_ok=True)

    unknown_dir = out_root / "_UNKNOWN"
    unknown_dir.mkdir(parents=True, exist_ok=True)

    # Walk everything under root, but skip output folders
    # (Path.rglob is fine as long as we exclude paths under out_root)
    for src in root.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in VIDEO_EXTS:
            continue
        if should_skip(src, out_root):
            continue

        cls = find_class(src.stem)
        vid_id = extract_id_from_path(src)
        idx = extract_idx_from_name(src.stem)

        if cls and vid_id and idx:
            dst_dir = out_root / cls
            new_name = f"{vid_id}_{cls}_{idx}{src.suffix}"
            dst = ensure_unique_path(dst_dir / new_name)
            tag = "MOVE" if move else "COPY"
            print(f"[{tag}] {src} -> {dst}")
        else:
            # Anything we can't confidently parse goes to _UNKNOWN (keep original name)
            dst = ensure_unique_path(unknown_dir / src.name)
            print(f"[UNKNOWN] {src} -> {dst}")

        if dry_run:
            continue

        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Restructure raw CI3D videos from subject folders into class folders. "
            "For example: s04/videos/<subject>/<Class> <idx>.mp4 -> "
            "s03/videos/<Class>/<subject>_<Class>_<idx>.mp4"
        )
    )
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Raw CI3D videos root to scan, e.g. path/to/datasets/CI3D/s04/videos",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output root for class folders, e.g. path/to/datasets/CI3D/s03/videos",
    )
    ap.add_argument("--copy", action="store_true", help="Copy files instead of moving")
    ap.add_argument("--dry_run", action="store_true", help="Print actions only")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else root

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    move = not args.copy

    print(f"Scan root : {root}")
    print(f"Out root  : {out_root}")
    print(f"Mode      : {'MOVE' if move else 'COPY'}")
    print(f"Dry-run   : {args.dry_run}")
    print()

    restructure(root=root, out_root=out_root, dry_run=args.dry_run, move=move)

    print("\nDone.")
    print("Classes:", ", ".join(CLASSES))
    print("Unknowns:", out_root / "_UNKNOWN")


if __name__ == "__main__":
    main()
