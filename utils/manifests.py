"""Dataset manifest and video-discovery helpers."""

import csv
import glob
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

HASHED_VIDEO_SUFFIX_RE = re.compile(r"^(?P<base>.+)_[0-9a-f]{8,}$", re.IGNORECASE)

def _strip_hashed_video_suffix(stem: str) -> str:
    match = HASHED_VIDEO_SUFFIX_RE.match(stem)
    return match.group("base") if match else stem


def _normalize_manifest_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_only.lower() if ch.isalnum())
def _read_id_name_csv(csv_path: str) -> Dict[int, str]:
    id2name: Dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(f"CSV must have columns id,name. Got: {reader.fieldnames}")
        for row in reader:
            if row is None:
                continue
            sid = str(row.get("id", "")).strip()
            sname = str(row.get("name", "")).strip()
            if sid == "":
                continue
            cid = int(sid)
            id2name[cid] = sname
    return id2name
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".zst"}


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered_values: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered_values.append(value)
    return ordered_values

def _scan_videos_under_root(root: Path) -> List[Path]:
    """
    Faster than Path.rglob for large trees.
    video_exts should be lowercase, including dot: {".mp4", ".avi", ...}
    """
    root = root.resolve()
    out: List[Path] = []
    stack = [os.fspath(root)]

    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    # Skip symlinks to avoid loops unless you want them
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                            continue
                        if not e.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        # permissions / broken links / transient IO issues
                        continue

                    # Extension check without Path construction
                    name = e.name
                    dot = name.rfind(".")
                    if dot == -1:
                        continue
                    ext = name[dot:].lower()
                    if ext in VIDEO_EXTS:
                        out.append(Path(e.path))
        except OSError:
            continue

    return out

def _parse_dataset_split_txt(txt_path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            try:
                fname, label_str = ln.rsplit(maxsplit=1)
            except ValueError:
                raise ValueError(f"Bad line in {txt_path!r}: {ln!r} (expected: <filename> <label>)")
            label = int(label_str)
            items.append((fname, label))
    if not items:
        raise ValueError(f"No entries found in dataset_split_txt: {txt_path}")
    return items

def _build_video_lookup_tables(root: Path) -> Dict[str, Dict[Any, Any]]:
    relative_path_map: Dict[str, str] = {}
    stem_map: Dict[str, List[str]] = defaultdict(list)
    dir_stem_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    stripped_stem_map: Dict[str, List[str]] = defaultdict(list)
    dir_stripped_stem_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    normalized_stem_map: Dict[str, List[str]] = defaultdict(list)
    dir_normalized_stem_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for video_path in _scan_videos_under_root(root):
        absolute_path = str(video_path)
        relative_path = video_path.relative_to(root).as_posix()
        relative_path_map[relative_path] = absolute_path

        relative_parent = video_path.parent.relative_to(root).as_posix().lower()
        parent = "" if relative_parent == "." else relative_parent
        stem = video_path.stem.lower()
        stem_map[stem].append(absolute_path)
        dir_stem_map[(parent, stem)].append(absolute_path)

        stripped_stem = _strip_hashed_video_suffix(stem)
        if stripped_stem != stem:
            stripped_stem_map[stripped_stem].append(absolute_path)
            dir_stripped_stem_map[(parent, stripped_stem)].append(absolute_path)

        normalized_stem = _normalize_manifest_lookup_text(stripped_stem)
        if normalized_stem:
            normalized_stem_map[normalized_stem].append(absolute_path)
            dir_normalized_stem_map[(parent, normalized_stem)].append(absolute_path)

    return {
        "relative_path": relative_path_map,
        "stem": stem_map,
        "dir_stem": dir_stem_map,
        "stripped_stem": stripped_stem_map,
        "dir_stripped_stem": dir_stripped_stem_map,
        "normalized_stem": normalized_stem_map,
        "dir_normalized_stem": dir_normalized_stem_map,
    }


def _pick_unique_video_match(
    candidates: Sequence[str],
    *,
    match_type: str,
    source_name: str,
    root_dir: str,
) -> Optional[str]:
    unique_candidates = _dedupe_keep_order(candidates)
    if not unique_candidates:
        return None
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    raise ValueError(
        f"Ambiguous {match_type} match for {source_name!r}: "
        f"found {len(unique_candidates)} files under {root_dir}."
    )


def _resolve_manifest_video_path(
    root: Path,
    entry_name: str,
    *,
    root_dir: str,
    video_lookup: Dict[str, Dict[Any, Any]],
) -> Optional[str]:
    absolute_candidate: Optional[Path] = None
    entry_path = Path(entry_name)

    if entry_path.is_absolute():
        absolute_candidate = entry_path.resolve()
        try:
            absolute_candidate.relative_to(root)
            if absolute_candidate.exists() and absolute_candidate.is_file():
                return str(absolute_candidate)
        except ValueError:
            pass
    else:
        joined_candidate = (root / entry_path).resolve()
        if joined_candidate.exists() and joined_candidate.is_file():
            return str(joined_candidate)

    normalized_entry = entry_name.replace("\\", "/").lstrip("./")
    direct_relative_match = video_lookup["relative_path"].get(normalized_entry)
    if direct_relative_match is not None:
        return direct_relative_match

    normalized_path = Path(normalized_entry)
    parent = normalized_path.parent.as_posix().lower()
    if parent == ".":
        parent = ""
    stem = normalized_path.stem.lower()
    stripped_stem = _strip_hashed_video_suffix(stem)
    normalized_stem = _normalize_manifest_lookup_text(stripped_stem)
    candidate_specs: List[Tuple[str, Any, Dict[Any, List[str]]]] = [
        ("dir+stem", (parent, stem), video_lookup["dir_stem"]),
        ("stem", stem, video_lookup["stem"]),
    ]
    if stripped_stem != stem:
        candidate_specs.insert(1, ("dir+stem-without-hash", (parent, stripped_stem), video_lookup["dir_stripped_stem"]))
        candidate_specs.append(("stem-without-hash", stripped_stem, video_lookup["stripped_stem"]))
    if normalized_stem and normalized_stem not in {stem, stripped_stem}:
        candidate_specs.append(
            ("dir+normalized-stem", (parent, normalized_stem), video_lookup["dir_normalized_stem"])
        )
        candidate_specs.append(("normalized-stem", normalized_stem, video_lookup["normalized_stem"]))

    for match_type, lookup_key, lookup_table in candidate_specs:
        match = _pick_unique_video_match(
            lookup_table.get(lookup_key, []),
            match_type=match_type,
            source_name=entry_name,
            root_dir=root_dir,
        )
        if match is not None:
            return match

    if absolute_candidate is not None and absolute_candidate.exists() and absolute_candidate.is_file():
        return str(absolute_candidate)
    return None

def list_videos(
    root_dir: str,
    dataset_split_txt: Optional[str] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Returns:
      paths:      list of absolute paths
      labels:     list of int labels
      classnames: list of class names (best-effort). For dataset_split_txt, defaults to strings of label ids.
    """
    root = Path(root_dir).resolve()
    if not root.exists():
        raise ValueError(f"root_dir does not exist: {root}")

    # ---- Mode A: dataset_split_txt (tc_clip style) ----
    if dataset_split_txt:
        txt_items = _parse_dataset_split_txt(dataset_split_txt)

        # Fast path: manifests generated by our probe scripts already contain exact
        # relative paths under root, so we can avoid scanning the whole tree.
        direct_paths: List[str] = []
        direct_labels: List[int] = []
        direct_ok = True
        for fname, y in txt_items:
            fname_path = Path(fname)
            if fname_path.is_absolute():
                direct_ok = False
                break
            candidate = (root / fname).resolve()
            if not (candidate.exists() and candidate.is_file()):
                direct_ok = False
                break
            direct_paths.append(str(candidate))
            direct_labels.append(int(y))
        if direct_ok:
            max_label = max(int(label) for label in direct_labels)
            classnames = [str(label) for label in range(max_label + 1)]
            return direct_paths, direct_labels, classnames

        video_lookup = _build_video_lookup_tables(root)
        paths: List[str] = []
        labels: List[int] = []
        missing_entries: List[str] = []
        for fname, label in txt_items:
            resolved_path = _resolve_manifest_video_path(
                root,
                fname,
                root_dir=root_dir,
                video_lookup=video_lookup,
            )
            if resolved_path is None:
                missing_entries.append(fname)
                continue
            paths.append(resolved_path)
            labels.append(int(label))

        if missing_entries:
            preview = ", ".join(repr(entry) for entry in missing_entries[:5])
            remainder = len(missing_entries) - min(len(missing_entries), 5)
            if remainder > 0:
                preview = f"{preview}, and {remainder} more"
            print(
                f"[WARN] Skipped {len(missing_entries)} entries from split {dataset_split_txt!r} "
                f"because they could not be resolved under root_dir={root_dir!r}: {preview}",
                file=sys.stderr,
                flush=True,
            )

        # Build classnames (best-effort)
        uniq = sorted(set(labels))
        if not uniq:
            raise ValueError(
                f"No usable videos were resolved from split {dataset_split_txt!r} under root_dir={root_dir!r}."
            )
        # Keep it simple/robust: label ids as strings in index order
        # (If you want, you can later replace with real names from json, etc.)
        max_id = max(uniq)
        classnames = [str(i) for i in range(max_id + 1)]
        return paths, labels, classnames

    # ---- Mode B: no split txt; support either flat-root files or class folders ----
    root_files = sorted([
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ])
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if root_files:
        paths = [str(p) for p in root_files]
        labels = [0 for _ in paths]
        classnames = [root.name]
        return paths, labels, classnames

    if class_dirs:
        classnames = [p.name for p in class_dirs]
        paths: List[str] = []
        labels: List[int] = []
        for ci, cdir in enumerate(class_dirs):
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    paths.append(str(p))
                    labels.append(ci)
        if paths:
            return paths, labels, classnames

    raise ValueError(
        f"No videos found under: {root_dir}. "
        f"Expected either files directly under root or class subdirectories containing video files "
        f"(extensions: {sorted(VIDEO_EXTS)})."
    )

def classnames_from_id_csv(
    csv_path: str,
    class_ids: List[int],
    *,
    unknown_fmt: str = "class_{id}",
) -> List[str]:
    """
    Parse id->name from a CSV with header: id,name
    Then build a classnames list aligned with your label space.

    Args:
      csv_path: path to CSV (id,name)
      class_ids: labels returned by list_videos when using dataset_split_txt
                 (typically one label per sample)
      unknown_fmt: fallback for ids missing in csv

    Returns:
      classnames: list where classnames[i] is the classname for id i
                 for i in [0..max(class_ids)].
    """
    id_to_name = _read_id_name_csv(csv_path)
    if not class_ids:
        return []

    max_id = int(max(class_ids))
    classnames = [unknown_fmt.format(id=i) for i in range(max_id + 1)]
    for i in range(max_id + 1):
        name = id_to_name.get(i, "").strip()
        if name:
            classnames[i] = name
    return classnames
def expand_manifest_args(manifest_args: Optional[Sequence[str]]) -> List[str]:
    """Accept explicit files and/or globs; return sorted unique absolute file paths."""
    if not manifest_args:
        return []
    out: List[str] = []
    for s in manifest_args:
        matches = glob.glob(s)
        if matches:
            out.extend(matches)
        else:
            out.append(s)
    return sorted({os.path.abspath(p) for p in out})


def resolve_single_manifest(
    manifest_arg: Optional[str],
    *,
    label: str = "Manifest",
) -> Optional[str]:
    if manifest_arg is None:
        return None
    value = str(manifest_arg).strip()
    if not value:
        return None
    matches = expand_manifest_args([value])
    if not matches:
        raise FileNotFoundError(f"{label} not found / glob matched nothing: {manifest_arg}")
    if len(matches) > 1:
        print(f"[WARN] multiple matches for {label.lower()}; using first: {matches[0]}", flush=True)
    return matches[0]


def split_name_from_manifest(manifest_path: Optional[str]) -> str:
    if manifest_path is None:
        return "all"
    return os.path.splitext(os.path.basename(manifest_path))[0]

__all__ = [
    "_build_video_lookup_tables",
    "_read_id_name_csv",
    "_resolve_manifest_video_path",
    "classnames_from_id_csv",
    "expand_manifest_args",
    "list_videos",
    "resolve_single_manifest",
    "split_name_from_manifest",
]
