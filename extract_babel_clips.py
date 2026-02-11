#!/usr/bin/env python3
"""
Extract class-filtered clips from BABEL JSON annotations.

This script is designed for cross-domain transfer setup (e.g., BABEL -> NTU / CI3D):
1) stream over a BABEL-format JSON file (without json.load on the full file),
2) match frame-level (and optional sequence-level) annotations to target classes,
3) download source videos once,
4) cut the matched time segments into clip files,
5) write manifests.

Default manifest format:
  <filename> <class_name>
Example:
  jab_9804_0.mp4 jab

Optional ID manifest format:
  <filename> <class_id>
plus class_id_to_label.csv

Example:
  python3 extract_babel_clips.py \
    --babel_json ../../../datasets/babel_v1.0_release/train.json \
    --out_dir ../../../datasets/babel_surveillance \
    --target_set combined \
    --max_per_class 200

python extract_babel_clips.py --babel_json ../../datasets/babel_v1.0_release/train.json --out_dir ../../datasets/babel_surveillance --target_set combined --max_per_class 200
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

# Handshake is shit, hit is shit, push is shit, holding hands is shit
PRESET_TARGETS: Dict[str, List[str]] = {
    # Practical default for surveillance-style positives + common negatives.
    "combined": [
        "punch",
        "jab",
        "kick",
        "walk",
        "hug",
        "grab",
        "dancing",
    ],
    # CI3D classes as present in local dataset structure.
    "ci3d_all": [
        "grab",
        "handshake",
        "hit",
        "holding_hands",
        "hug",
        "kick",
        "posing",
        "push",
    ],
    # Small NTU interaction/surveillance-focused subset.
    "ntu_interactions_small": [
        "punch",
        "kick",
        "push",
        "hug",
        "handshake",
        "walk",
        "grab",
        "hit",
        "follow",
        "exchange",
        "support",
        "whisper",
    ],
}


TARGET_ALIASES: Dict[str, List[str]] = {
    "punch": ["punch", "punching", "boxing", "box", "slap", "strike"],
    "jab": ["jab", "jabs", "jabbing", "hook", "left hook", "right hook", "uppercut"],
    "kick": ["kick", "kicking", "side kick", "butt kicks"],
    "walk": ["walk", "walking", "walking towards", "walking apart", "stroll"],
    "hug": ["hug", "hugging", "embrace"],
    "handshake": ["handshake", "shake hands", "shaking hands"],
    "push": ["push", "pushing", "shove"],
    "grab": ["grab", "grabbing", "grasp"],
    "hit": ["hit", "hitting", "strike", "striking"],
    "holding_hands": ["holding hands", "hold hands"],
    "posing": ["pose", "posing"],
    "follow": ["follow", "following"],
    "exchange": ["exchange", "exchange things", "swap"],
    "support": ["support", "support somebody"],
    "whisper": ["whisper", "whispering"],
    "dancing": ["dance"]
}


_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_MULTISPACE = re.compile(r"\s+")


@dataclass
class SegmentJob:
    sid: str
    url: str
    label: str
    ann_idx: int
    ann_source: str  # "frame" or "seq"
    start_t: float
    end_t: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract class-filtered clips from BABEL JSON.")
    ap.add_argument("--babel_json", default=None, help="Path to BABEL JSON (e.g., train.json).")
    ap.add_argument("--out_dir", default=None, help="Output root directory.")

    ap.add_argument(
        "--target_set",
        action="append",
        default=[],
        choices=sorted(PRESET_TARGETS.keys()),
        help=f"Built-in class set. Can be repeated. Available: {', '.join(sorted(PRESET_TARGETS.keys()))}",
    )
    ap.add_argument(
        "--targets",
        default="",
        help="Comma-separated target classes (e.g., 'jab,punch,kick,walk,hug,handshake').",
    )
    ap.add_argument(
        "--targets_file",
        default=None,
        help="Optional text file with one target class per line (comments with # allowed).",
    )
    ap.add_argument(
        "--extra_aliases_json",
        default=None,
        help="Optional JSON file with extra aliases: {\"class_name\": [\"alias1\", ...]}",
    )
    ap.add_argument("--list_target_sets", action="store_true", help="Print available target sets and exit.")

    ap.add_argument("--include_frame_ann", action="store_true", default=True, help="Use frame_ann labels.")
    ap.add_argument("--no_frame_ann", dest="include_frame_ann", action="store_false")
    ap.add_argument("--include_seq_ann", action="store_true", help="Also use seq_ann labels (full video duration).")

    ap.add_argument("--min_duration", type=float, default=0.20, help="Minimum clip duration in seconds.")
    ap.add_argument("--max_records", type=int, default=None, help="Stop after scanning this many BABEL records.")
    ap.add_argument("--max_total", type=int, default=None, help="Maximum number of clips to produce.")
    ap.add_argument("--max_per_class", type=int, default=None, help="Maximum clips per class.")

    ap.add_argument("--clips_subdir", default="clips", help="Subdirectory under out_dir for extracted clips.")
    ap.add_argument(
        "--downloads_subdir",
        default="_downloads",
        help="Subdirectory under out_dir for cached source videos.",
    )
    ap.add_argument("--video_ext", default=".mp4", help="Output video extension (e.g., .mp4 or .avi).")
    ap.add_argument(
        "--manifest_name",
        default="clips_manifest.txt",
        help="Name of class-name manifest file.",
    )
    ap.add_argument(
        "--manifest_path_mode",
        choices=["name", "relative", "absolute"],
        default="name",
        help="Path style in manifests.",
    )

    ap.add_argument("--write_id_manifest", action="store_true", help="Also write filename class_id manifest.")
    ap.add_argument("--id_manifest_name", default="clips_manifest_ids.txt")
    ap.add_argument("--class_map_csv", default="class_id_to_label.csv")
    ap.add_argument(
        "--append_manifests",
        action="store_true",
        default=True,
        help="Append to manifest files instead of rewriting (default: enabled).",
    )
    ap.add_argument(
        "--rewrite_manifests",
        dest="append_manifests",
        action="store_false",
        help="Rewrite manifest files from scratch.",
    )
    ap.add_argument(
        "--dedupe_manifest_entries",
        action="store_true",
        default=True,
        help="When appending, avoid writing duplicate manifest lines.",
    )
    ap.add_argument(
        "--no_dedupe_manifest_entries",
        dest="dedupe_manifest_entries",
        action="store_false",
        help="Disable manifest de-duplication while appending.",
    )

    ap.add_argument("--ffmpeg_bin", default="ffmpeg", help="ffmpeg executable.")
    ap.add_argument("--crf", type=int, default=23, help="CRF for x264 encoding.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing downloaded/clipped files.")
    ap.add_argument("--dry_run", action="store_true", help="Parse/match/write manifests, but do not download/clip.")
    ap.add_argument("--download_retries", type=int, default=3)
    ap.add_argument("--download_timeout", type=int, default=120)
    ap.add_argument(
        "--prefer_label_fields",
        action="store_true",
        default=True,
        help="Prefer proc/raw labels over act_cat when matching.",
    )
    ap.add_argument(
        "--no_prefer_label_fields",
        dest="prefer_label_fields",
        action="store_false",
        help="Disable proc/raw preference.",
    )
    ap.add_argument(
        "--label_field_bonus",
        type=int,
        default=200,
        help="Bonus score applied to proc/raw matches when --prefer_label_fields is enabled.",
    )
    ap.add_argument("--num_workers", type=int, default=1, help="Parallel workers for URL processing.")
    ap.add_argument(
        "--remove_downloads_after_extract",
        dest="remove_downloads_after_extract",
        action="store_true",
        default=True,
        help="Delete cached source videos after all clips from that source are extracted.",
    )
    ap.add_argument(
        "--keep_downloads",
        dest="remove_downloads_after_extract",
        action="store_false",
        help="Keep downloaded source videos in cache.",
    )
    ap.add_argument("--progress_every", type=int, default=1000, help="Log progress every N scanned records.")
    return ap.parse_args()


def _norm_text(text: str) -> str:
    t = text.lower()
    t = t.replace("_", " ").replace("-", " ").replace("/", " ")
    t = _NON_ALNUM.sub(" ", t)
    t = _MULTISPACE.sub(" ", t).strip()
    return t


def _slug(text: str) -> str:
    t = _norm_text(text)
    return t.replace(" ", "_")


def _read_targets_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def resolve_targets(args: argparse.Namespace) -> List[str]:
    targets: List[str] = []
    for s in args.target_set:
        targets.extend(PRESET_TARGETS[s])
    if args.targets.strip():
        targets.extend([x.strip() for x in args.targets.split(",") if x.strip()])
    targets.extend(_read_targets_file(args.targets_file))

    if not targets:
        targets = PRESET_TARGETS["combined"][:]

    out: List[str] = []
    seen = set()
    for t in targets:
        st = _slug(t)
        if not st:
            continue
        if st not in seen:
            out.append(st)
            seen.add(st)
    return out


def _load_extra_aliases(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("--extra_aliases_json must contain an object: {class_name: [aliases...]}")
    out: Dict[str, List[str]] = {}
    for k, v in obj.items():
        kk = _slug(str(k))
        if not kk:
            continue
        if isinstance(v, str):
            out[kk] = [v]
        elif isinstance(v, list):
            out[kk] = [str(x) for x in v]
        else:
            raise ValueError(f"Invalid aliases for class {k!r}: expected list or string")
    return out


def build_target_patterns(targets: Sequence[str], extra_aliases: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    patt: Dict[str, List[str]] = {}
    extra_aliases = extra_aliases or {}
    for t in targets:
        parts = set()
        parts.add(_norm_text(t.replace("_", " ")))
        for alias in TARGET_ALIASES.get(t, []):
            na = _norm_text(alias)
            if na:
                parts.add(na)
        for alias in extra_aliases.get(t, []):
            na = _norm_text(alias)
            if na:
                parts.add(na)
        patt[t] = sorted(parts)
    return patt


def _match_target(
    candidates: Sequence[Tuple[str, str]],
    target_patterns: Dict[str, List[str]],
    *,
    prefer_label_fields: bool,
    label_field_bonus: int,
) -> Optional[str]:
    best_target: Optional[str] = None
    best_score = 0
    for cand, source in candidates:
        nc = _norm_text(cand)
        if not nc:
            continue
        src_bonus = 0
        if prefer_label_fields and source in ("proc_label", "raw_label"):
            src_bonus = int(label_field_bonus)
        for target, patterns in target_patterns.items():
            for p in patterns:
                score = 0
                if nc == p:
                    score = 300 + len(p)
                elif re.search(rf"\b{re.escape(p)}\b", nc):
                    score = 200 + len(p)
                elif p in nc:
                    score = 100 + len(p)
                if score > 0:
                    score += src_bonus
                if score > best_score:
                    best_score = score
                    best_target = target
    return best_target


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _label_candidates(label_obj: Dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for key in ("proc_label", "raw_label"):
        v = label_obj.get(key)
        if isinstance(v, str) and v.strip():
            out.append((v, key))
    act_cat = label_obj.get("act_cat")
    if isinstance(act_cat, list):
        for x in act_cat:
            if isinstance(x, str) and x.strip():
                out.append((x, "act_cat"))
    elif isinstance(act_cat, str) and act_cat.strip():
        out.append((act_cat, "act_cat"))
    return out


def _resolve_sid(key: str, entry: Dict) -> str:
    sid = entry.get("babel_sid", key)
    try:
        return str(int(sid))
    except (TypeError, ValueError):
        cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(sid))
        return cleaned or str(key)


def iter_segment_jobs(
    key: str,
    entry: Dict,
    *,
    target_patterns: Dict[str, List[str]],
    include_frame_ann: bool,
    include_seq_ann: bool,
    min_duration: float,
    prefer_label_fields: bool,
    label_field_bonus: int,
) -> Iterator[SegmentJob]:
    url = entry.get("url")
    if not isinstance(url, str) or not url.strip():
        return

    sid = _resolve_sid(key, entry)
    dur = _to_float(entry.get("dur"))

    if include_frame_ann:
        frame_ann = entry.get("frame_ann") or {}
        labels = frame_ann.get("labels") if isinstance(frame_ann, dict) else None
        if isinstance(labels, list):
            for idx, lab in enumerate(labels):
                if not isinstance(lab, dict):
                    continue
                target = _match_target(
                    _label_candidates(lab),
                    target_patterns,
                    prefer_label_fields=prefer_label_fields,
                    label_field_bonus=label_field_bonus,
                )
                if target is None:
                    continue
                st = _to_float(lab.get("start_t"))
                en = _to_float(lab.get("end_t"))
                if st is None or en is None:
                    continue
                if dur is not None:
                    st = max(0.0, min(st, dur))
                    en = max(0.0, min(en, dur))
                if en - st < min_duration:
                    continue
                yield SegmentJob(
                    sid=sid,
                    url=url,
                    label=target,
                    ann_idx=idx,
                    ann_source="frame",
                    start_t=st,
                    end_t=en,
                )

    if include_seq_ann:
        seq_ann = entry.get("seq_ann") or {}
        labels = seq_ann.get("labels") if isinstance(seq_ann, dict) else None
        if isinstance(labels, list):
            for idx, lab in enumerate(labels):
                if not isinstance(lab, dict):
                    continue
                target = _match_target(
                    _label_candidates(lab),
                    target_patterns,
                    prefer_label_fields=prefer_label_fields,
                    label_field_bonus=label_field_bonus,
                )
                if target is None:
                    continue
                st = 0.0
                en = dur if dur is not None else None
                if en is None or (en - st) < min_duration:
                    continue
                yield SegmentJob(
                    sid=sid,
                    url=url,
                    label=target,
                    ann_idx=idx,
                    ann_source="seq",
                    start_t=st,
                    end_t=en,
                )


def iter_babel_records_stream(json_path: Path, chunk_size: int = 1 << 20) -> Iterator[Tuple[str, Dict]]:
    """
    Stream top-level object entries from a JSON file:
      {"9804": {...}, "9805": {...}, ...}
    """
    decoder = json.JSONDecoder()
    with json_path.open("r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        eof = False

        def read_more() -> bool:
            nonlocal buf, eof
            chunk = f.read(chunk_size)
            if chunk == "":
                eof = True
                return False
            buf += chunk
            return True

        def compact():
            nonlocal buf, pos
            if pos > 0 and (pos > (1 << 20) or pos > len(buf) // 2):
                buf = buf[pos:]
                pos = 0

        def ensure(n: int = 1) -> bool:
            while len(buf) - pos < n and not eof:
                if not read_more():
                    break
            return (len(buf) - pos) >= n

        def skip_ws():
            nonlocal pos
            while True:
                if not ensure(1):
                    return
                c = buf[pos]
                if c in " \t\r\n":
                    pos += 1
                    compact()
                    continue
                return

        def expect(ch: str):
            nonlocal pos
            skip_ws()
            if not ensure(1):
                raise ValueError(f"Unexpected EOF while expecting {ch!r}")
            got = buf[pos]
            if got != ch:
                raise ValueError(f"Expected {ch!r}, got {got!r}")
            pos += 1
            compact()

        def decode_one():
            nonlocal pos
            while True:
                skip_ws()
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    pos = end
                    compact()
                    return obj
                except json.JSONDecodeError:
                    if eof:
                        near = buf[max(0, pos - 80) : pos + 80]
                        raise ValueError(f"Invalid JSON near: {near!r}") from None
                    read_more()

        read_more()
        skip_ws()
        expect("{")
        skip_ws()
        if ensure(1) and buf[pos] == "}":
            return

        while True:
            key = decode_one()
            if not isinstance(key, str):
                raise ValueError(f"Top-level key must be str, got {type(key)}")
            expect(":")
            value = decode_one()
            if isinstance(value, dict):
                yield key, value

            skip_ws()
            if not ensure(1):
                raise ValueError("Unexpected EOF after top-level entry")
            ch = buf[pos]
            pos += 1
            compact()
            if ch == ",":
                continue
            if ch == "}":
                break
            raise ValueError(f"Expected ',' or '}}', got {ch!r}")


def _download_src_video(
    url: str,
    sid: str,
    download_dir: Path,
    *,
    retries: int,
    timeout_sec: int,
    overwrite: bool,
) -> Path:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix or ".mp4"
    dst = download_dir / f"{sid}{ext}"
    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return dst

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_err: Optional[Exception] = None
    for i in range(1, max(1, retries) + 1):
        try:
            with urlopen(req, timeout=timeout_sec) as r, tmp.open("wb") as w:
                shutil.copyfileobj(r, w, length=1024 * 1024)
            tmp.replace(dst)
            return dst
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            last_err = e
            if tmp.exists():
                tmp.unlink()
            if i < retries:
                time.sleep(min(2 ** (i - 1), 5))
                continue
            break
    raise RuntimeError(f"Download failed for {url!r}: {last_err}")


def _clip_with_ffmpeg(
    ffmpeg_bin: str,
    src: Path,
    dst: Path,
    start_t: float,
    end_t: float,
    *,
    crf: int,
    overwrite: bool,
):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    if end_t <= start_t:
        raise RuntimeError(f"Invalid segment [{start_t}, {end_t}] for {src}")

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_t:.3f}",
        "-to",
        f"{end_t:.3f}",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(err)


def _clip_name(job: SegmentJob, video_ext: str) -> str:
    ext = video_ext if video_ext.startswith(".") else f".{video_ext}"
    base = f"{job.label}_{job.sid}_{job.ann_idx}"
    if job.ann_source == "seq":
        base = f"{base}_seq"
    return f"{base}{ext}"


def _manifest_path_str(path: Path, out_dir: Path, mode: str) -> str:
    if mode == "name":
        return path.name
    if mode == "relative":
        return path.relative_to(out_dir).as_posix()
    return str(path.resolve())


def _write_class_map_csv(path: Path, targets: Sequence[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name"])
        for i, t in enumerate(targets):
            w.writerow([i, t])


def _read_class_map_csv(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(_slug(str(row.get("name", ""))))
    return out


def _tqdm_wrap(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


@dataclass
class UrlTask:
    url: str
    jobs: List[SegmentJob]
    clips_dir: str
    downloads_dir: str
    out_dir: str
    manifest_path_mode: str
    video_ext: str
    ffmpeg_bin: str
    crf: int
    overwrite: bool
    dry_run: bool
    download_retries: int
    download_timeout: int
    remove_downloads_after_extract: bool


def _process_url_task(task: UrlTask) -> Dict:
    out_entries: List[Tuple[str, str]] = []
    class_counts: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    errors: List[str] = []

    stats = Counter()

    clips_dir = Path(task.clips_dir)
    downloads_dir = Path(task.downloads_dir)
    out_dir = Path(task.out_dir)

    src_path: Optional[Path] = None
    pending_jobs: List[SegmentJob] = []
    if task.overwrite:
        pending_jobs = list(task.jobs)
    else:
        # If clip already exists and overwrite is off, skip work for this segment.
        for job in task.jobs:
            clip_name = _clip_name(job, task.video_ext)
            clip_path = clips_dir / clip_name
            if clip_path.exists():
                stats["clips_already_exist"] += 1
                continue
            pending_jobs.append(job)

    if not pending_jobs:
        stats["urls_skipped_existing"] += 1
        return {
            "entries": out_entries,
            "class_counts": dict(class_counts),
            "failure_types": dict(failure_types),
            "errors": errors,
            "stats": dict(stats),
        }

    try:
        if not task.dry_run:
            sid = pending_jobs[0].sid
            src_path = _download_src_video(
                task.url,
                sid,
                downloads_dir,
                retries=task.download_retries,
                timeout_sec=task.download_timeout,
                overwrite=task.overwrite,
            )
            stats["downloads"] += 1

        for job in pending_jobs:
            clip_name = _clip_name(job, task.video_ext)
            clip_path = clips_dir / clip_name
            try:
                if not task.dry_run:
                    if src_path is None:
                        raise RuntimeError("source path missing after download")
                    _clip_with_ffmpeg(
                        task.ffmpeg_bin,
                        src_path,
                        clip_path,
                        job.start_t,
                        job.end_t,
                        crf=task.crf,
                        overwrite=task.overwrite,
                    )
                else:
                    clip_path.parent.mkdir(parents=True, exist_ok=True)

                mpth = _manifest_path_str(clip_path, out_dir, task.manifest_path_mode)
                out_entries.append((mpth, job.label))
                class_counts[job.label] += 1
                stats["clips_saved"] += 1
            except Exception as e:
                stats["failed"] += 1
                failure_types[type(e).__name__] += 1
                errors.append(
                    f"sid={job.sid} label={job.label} idx={job.ann_idx} source={job.ann_source}: {e}"
                )
    finally:
        if (
            task.remove_downloads_after_extract
            and (not task.dry_run)
            and src_path is not None
            and src_path.exists()
        ):
            try:
                src_path.unlink()
                stats["downloads_removed"] += 1
            except OSError:
                pass

    return {
        "entries": out_entries,
        "class_counts": dict(class_counts),
        "failure_types": dict(failure_types),
        "errors": errors,
        "stats": dict(stats),
    }


def main():
    args = parse_args()

    if args.list_target_sets:
        print("Available target sets:")
        for k in sorted(PRESET_TARGETS.keys()):
            print(f"  {k}: {', '.join(PRESET_TARGETS[k])}")
        return

    if not args.babel_json or not args.out_dir:
        raise SystemExit("Both --babel_json and --out_dir are required (unless --list_target_sets is used).")
    if not args.include_frame_ann and not args.include_seq_ann:
        raise SystemExit("Nothing to extract: enable --include_frame_ann and/or --include_seq_ann.")

    targets = resolve_targets(args)
    if not targets:
        raise SystemExit("No targets resolved. Provide --targets or --target_set.")
    extra_aliases = _load_extra_aliases(args.extra_aliases_json)
    target_patterns = build_target_patterns(targets, extra_aliases=extra_aliases)

    out_dir = Path(args.out_dir).resolve()
    clips_dir = out_dir / args.clips_subdir
    downloads_dir = out_dir / args.downloads_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        downloads_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and shutil.which(args.ffmpeg_bin) is None:
        raise SystemExit(f"ffmpeg not found: {args.ffmpeg_bin!r}")

    manifest_path = out_dir / args.manifest_name
    id_manifest_path = out_dir / args.id_manifest_name
    class_map_path = out_dir / args.class_map_csv
    target_to_id = {t: i for i, t in enumerate(targets)}

    print("[CONFIG]")
    print(f"  babel_json       : {Path(args.babel_json).resolve()}")
    print(f"  out_dir          : {out_dir}")
    print(f"  clips_dir        : {clips_dir}")
    print(f"  download_cache   : {downloads_dir}")
    print(f"  targets ({len(targets)}): {', '.join(targets)}")
    print(f"  frame_ann        : {args.include_frame_ann}")
    print(f"  seq_ann          : {args.include_seq_ann}")
    print(f"  prefer_labels    : {args.prefer_label_fields}")
    print(f"  label_bonus      : {args.label_field_bonus}")
    print(f"  dry_run          : {args.dry_run}")
    print(f"  append_manifests : {args.append_manifests}")
    print(f"  dedupe_manifests : {args.dedupe_manifest_entries}")
    print(f"  num_workers      : {max(1, int(args.num_workers))}")
    print(f"  remove_downloads : {args.remove_downloads_after_extract}")
    if tqdm is None:
        print("  tqdm             : not installed (falling back to plain logs)")

    stats = Counter()
    class_counts_selected: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    failures = defaultdict(int)
    selected_jobs: List[SegmentJob] = []

    babel_json = Path(args.babel_json).resolve()
    if not babel_json.exists():
        raise SystemExit(f"BABEL JSON not found: {babel_json}")

    scan_iter = iter_babel_records_stream(babel_json)
    if tqdm is None:
        scan_iter = iter_babel_records_stream(babel_json)
    else:
        scan_iter = _tqdm_wrap(scan_iter, desc="Scanning BABEL", unit="rec", dynamic_ncols=True)

    stop_all = False
    for key, entry in scan_iter:
        stats["records_scanned"] += 1
        if args.max_records is not None and stats["records_scanned"] > args.max_records:
            break

        for job in iter_segment_jobs(
            key,
            entry,
            target_patterns=target_patterns,
            include_frame_ann=args.include_frame_ann,
            include_seq_ann=args.include_seq_ann,
            min_duration=float(args.min_duration),
            prefer_label_fields=bool(args.prefer_label_fields),
            label_field_bonus=int(args.label_field_bonus),
        ):
            stats["matched_segments"] += 1

            if args.max_per_class is not None and class_counts_selected[job.label] >= args.max_per_class:
                stats["skipped_max_per_class"] += 1
                continue
            if args.max_total is not None and len(selected_jobs) >= args.max_total:
                stop_all = True
                break

            selected_jobs.append(job)
            class_counts_selected[job.label] += 1
        if stop_all:
            break

        if tqdm is None and stats["records_scanned"] % max(1, args.progress_every) == 0:
            print(
                f"[SCAN] records={stats['records_scanned']} matched={stats['matched_segments']} "
                f"selected={len(selected_jobs)}"
            )

    jobs_by_url: Dict[str, List[SegmentJob]] = defaultdict(list)
    for j in selected_jobs:
        jobs_by_url[j.url].append(j)
    url_tasks: List[UrlTask] = [
        UrlTask(
            url=url,
            jobs=jobs,
            clips_dir=str(clips_dir),
            downloads_dir=str(downloads_dir),
            out_dir=str(out_dir),
            manifest_path_mode=args.manifest_path_mode,
            video_ext=args.video_ext,
            ffmpeg_bin=args.ffmpeg_bin,
            crf=int(args.crf),
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            download_retries=int(args.download_retries),
            download_timeout=int(args.download_timeout),
            remove_downloads_after_extract=bool(args.remove_downloads_after_extract),
        )
        for url, jobs in jobs_by_url.items()
    ]

    print(f"\n[PLAN] selected_segments={len(selected_jobs)} unique_urls={len(url_tasks)}")

    all_entries: List[Tuple[str, str]] = []
    workers = max(1, int(args.num_workers))
    if workers == 1:
        proc_iter = (_process_url_task(t) for t in url_tasks)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        proc_iter = pool.imap_unordered(_process_url_task, url_tasks, chunksize=1)

    try:
        proc_iter = _tqdm_wrap(proc_iter, total=len(url_tasks), desc="Processing URLs", unit="url", dynamic_ncols=True)
        for res in proc_iter:
            for k, v in res["stats"].items():
                stats[k] += v
            class_counts.update(res["class_counts"])
            all_entries.extend(res["entries"])
            for k, v in res["failure_types"].items():
                failures[k] += v
            for e in res["errors"]:
                print(f"[WARN] {e}", file=sys.stderr, flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    all_entries.sort(key=lambda x: x[0])
    manifest_mode = "a" if args.append_manifests else "w"
    id_manifest_mode = "a" if args.append_manifests else "w"

    if args.write_id_manifest:
        if class_map_path.exists() and args.append_manifests:
            existing = _read_class_map_csv(class_map_path)
            if existing != list(targets):
                raise SystemExit(
                    "Existing class_id_to_label.csv does not match current targets while appending. "
                    "Use --rewrite_manifests or a different out_dir/class_map."
                )
        else:
            _write_class_map_csv(class_map_path, targets)

    existing_manifest_lines = set()
    existing_id_lines = set()
    if args.append_manifests and args.dedupe_manifest_entries:
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        existing_manifest_lines.add(s)
        if args.write_id_manifest and id_manifest_path.exists():
            with id_manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        existing_id_lines.add(s)

    with manifest_path.open(manifest_mode, encoding="utf-8") as mf, (
        id_manifest_path.open(id_manifest_mode, encoding="utf-8") if args.write_id_manifest else open(os.devnull, "w")
    ) as mif:
        for mpth, label in all_entries:
            line = f"{mpth} {label}"
            if args.append_manifests and args.dedupe_manifest_entries and line in existing_manifest_lines:
                stats["manifest_dupes_skipped"] += 1
                continue
            mf.write(line + "\n")
            if args.append_manifests and args.dedupe_manifest_entries:
                existing_manifest_lines.add(line)
            if args.write_id_manifest:
                id_line = f"{mpth} {target_to_id[label]}"
                if args.append_manifests and args.dedupe_manifest_entries and id_line in existing_id_lines:
                    stats["id_manifest_dupes_skipped"] += 1
                    continue
                mif.write(id_line + "\n")
                if args.append_manifests and args.dedupe_manifest_entries:
                    existing_id_lines.add(id_line)

    print("\n[SUMMARY]")
    print(f"  records_scanned     : {stats['records_scanned']}")
    print(f"  matched_segments    : {stats['matched_segments']}")
    print(f"  clips_saved         : {stats['clips_saved']}")
    print(f"  downloads           : {stats['downloads']}")
    print(f"  downloads_removed   : {stats['downloads_removed']}")
    print(f"  urls_skipped_exist  : {stats['urls_skipped_existing']}")
    print(f"  clips_already_exist : {stats['clips_already_exist']}")
    print(f"  manifest_dupes_skip : {stats['manifest_dupes_skipped']}")
    print(f"  id_manifest_dupes   : {stats['id_manifest_dupes_skipped']}")
    print(f"  skipped_max_per_cls : {stats['skipped_max_per_class']}")
    print(f"  failed              : {stats['failed']}")
    if failures:
        print("  failure_types       :", dict(failures))

    print("\n[OUTPUT]")
    print(f"  manifest            : {manifest_path}")
    if args.write_id_manifest:
        print(f"  id_manifest         : {id_manifest_path}")
        print(f"  class_map_csv       : {class_map_path}")
    print(f"  clips_dir           : {clips_dir}")
    if not args.dry_run:
        print(f"  download_cache      : {downloads_dir}")

    print("\n[CLASS COUNTS]")
    for k in targets:
        print(f"  {k:16s} {class_counts.get(k, 0)}")


if __name__ == "__main__":
    main()
