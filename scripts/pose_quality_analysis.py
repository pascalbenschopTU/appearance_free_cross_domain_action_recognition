#!/usr/bin/env python3
"""
Pose estimation quality analysis for surveillance videos.

Runs YOLO-pose + ByteTrack on a directory of videos and computes heuristics
that quantify how poorly pose estimation performs in difficult surveillance
scenarios (occlusion, low resolution, crowds, motion blur).

For each tracked person the script records:
  - total_frames   : how many frames the tracker kept this person alive
  - skeleton_frames: subset where mean keypoint confidence > --skeleton_conf
  - detection_rate : skeleton_frames / total_frames  (the "Y%" metric)
  - kp_rate_<T>    : fraction of (frame x joint) pairs with conf >= T (the "Z%" metric)
  - joint_<name>   : per-joint mean confidence (weighted average over skeleton frames)

Outputs (all in --output_dir):
  per_track_stats.csv   – one row per tracked person per video
  summary.json          – aggregate statistics across all tracks
  plots/                – four figures
  annotated/            – (optional) annotated MP4s for the first N videos

Usage example:
  python pose_quality_analysis.py \\
      --video_dir  /path/to/videos \\
      --output_dir /path/to/results \\
      --model      yolo11n-pose.pt \\
      --device     mps \\
      --save_annotated 5
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COCO_KP_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
NUM_KP = len(COCO_KP_NAMES)

FACE_IDX  = [0, 1, 2, 3, 4]
UPPER_IDX = [5, 6, 7, 8, 9, 10]
LOWER_IDX = [11, 12, 13, 14, 15, 16]

GROUP_COLORS = {
    "face":  "#e05c5c",
    "upper": "#e09a5c",
    "lower": "#5c8ee0",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Pose quality analysis for surveillance videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video_dir", required=True,
                   help="Directory containing .mp4 videos")
    p.add_argument("--output_dir", required=True,
                   help="Root directory for all outputs")
    p.add_argument("--model", default="yolo11n-pose.pt",
                   help="YOLO pose model path or name (auto-downloaded if name only)")
    p.add_argument("--device", default="mps",
                   help="Inference device: cpu | cuda | mps")
    p.add_argument("--det_conf", type=float, default=0.25,
                   help="YOLO detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--skeleton_conf", type=float, default=0.30,
                   help="Mean keypoint confidence threshold to consider a skeleton 'detected'")
    p.add_argument("--conf_thresholds", nargs="+", type=float, default=[0.5, 0.75],
                   help="Joint confidence thresholds used for kp_rate_* columns and plots")
    p.add_argument("--min_track_frames", type=int, default=5,
                   help="Minimum track length (frames) to include in aggregate stats and plots")
    p.add_argument("--save_annotated", type=int, default=3,
                   help="Number of videos to save as annotated MP4 (0 = none)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(model, video_path, device, det_conf, iou,
                  skeleton_conf, save_annotated_path=None):
    """
    Run YOLO pose + ByteTrack on one video.

    Returns
    -------
    tracks : dict[int -> dict]
        track_id -> {
            "total_frames"   : int,
            "skeleton_frames": int,
            "kp_confs"       : list[np.ndarray shape (17,)]
        }
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if save_annotated_path is not None:
        Path(save_annotated_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_annotated_path), fourcc, fps, (w, h))

    tracks = defaultdict(lambda: {
        "total_frames":    0,
        "skeleton_frames": 0,
        "kp_confs":        [],
    })

    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=det_conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    for result in results:
        boxes = result.boxes
        kps   = result.keypoints  # None if no detections

        if boxes is None or len(boxes) == 0:
            if writer is not None and result.orig_img is not None:
                writer.write(result.orig_img)
            continue

        ids = boxes.id  # Tensor or None (None when tracker has no IDs yet)
        n   = len(boxes)

        for i in range(n):
            tid = int(ids[i].item()) if ids is not None else -(i + 1)
            tracks[tid]["total_frames"] += 1

            if kps is not None and kps.conf is not None:
                kp_conf = kps.conf[i].cpu().numpy()   # (17,)
                if float(np.mean(kp_conf)) >= skeleton_conf:
                    tracks[tid]["skeleton_frames"] += 1
                    tracks[tid]["kp_confs"].append(kp_conf)

        if writer is not None:
            annotated = result.plot(kpt_line=True, kpt_radius=4, boxes=True)

            # Overlay: track-id + mean keypoint confidence per person
            if kps is not None and kps.conf is not None and ids is not None:
                for i in range(n):
                    tid     = int(ids[i].item())
                    box     = boxes.xyxy[i].cpu().numpy().astype(int)
                    mean_kp = float(np.mean(kps.conf[i].cpu().numpy()))
                    label   = f"ID:{tid}  kp:{mean_kp:.2f}"
                    cv2.putText(
                        annotated, label,
                        (box[0], max(box[1] - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                    )

            # Top-left HUD: frame-level stats
            n_skel = sum(
                1 for i in range(n)
                if kps is not None and kps.conf is not None
                and float(np.mean(kps.conf[i].cpu().numpy())) >= skeleton_conf
            )
            hud = f"persons:{n}  skeletons:{n_skel}"
            cv2.putText(annotated, hud, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(annotated)

    if writer is not None:
        writer.release()

    return dict(tracks)


# ---------------------------------------------------------------------------
# Per-track statistics
# ---------------------------------------------------------------------------

def compute_track_stats(video_name, tracks, conf_thresholds):
    """Return a list of dicts, one per track."""
    rows = []
    for tid, data in tracks.items():
        total = data["total_frames"]
        skel  = data["skeleton_frames"]
        det_rate = skel / total if total > 0 else 0.0

        kp_list = data["kp_confs"]  # list of (17,) arrays
        if kp_list:
            kp_mat       = np.stack(kp_list, axis=0)  # (S, 17)
            kp_conf_mean = float(np.mean(kp_mat))
            per_joint    = kp_mat.mean(axis=0)         # (17,)
            kp_rates     = {
                f"kp_rate_{int(t * 100)}": float(np.mean(kp_mat >= t))
                for t in conf_thresholds
            }
        else:
            kp_conf_mean = 0.0
            per_joint    = np.zeros(NUM_KP)
            kp_rates     = {f"kp_rate_{int(t * 100)}": 0.0 for t in conf_thresholds}

        row = {
            "video":         video_name,
            "track_id":      tid,
            "total_frames":  total,
            "skeleton_frames": skel,
            "detection_rate":  round(det_rate, 4),
            "kp_conf_mean":    round(kp_conf_mean, 4),
            **{k: round(v, 4) for k, v in kp_rates.items()},
        }
        for j, jname in enumerate(COCO_KP_NAMES):
            row[f"joint_{jname}"] = round(float(per_joint[j]), 4)

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def _bar_colors():
    colors = []
    for i in range(NUM_KP):
        if i in FACE_IDX:
            colors.append(GROUP_COLORS["face"])
        elif i in UPPER_IDX:
            colors.append(GROUP_COLORS["upper"])
        else:
            colors.append(GROUP_COLORS["lower"])
    return colors


def plot_results(all_rows, output_dir, conf_thresholds, min_track_frames):
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in all_rows if r["total_frames"] >= min_track_frames]
    if not rows:
        print(f"  Warning: no tracks with >= {min_track_frames} frames; using all.")
        rows = all_rows
    if not rows:
        print("  No data to plot.")
        return

    det_rates = np.array([r["detection_rate"] for r in rows])

    # ------------------------------------------------------------------
    # Figure 1: Distribution of skeleton detection rates
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(det_rates, bins=25, range=(0, 1),
            color="#e05c5c", edgecolor="white", linewidth=0.5)
    med = np.median(det_rates)
    ax.axvline(med, color="black", linestyle="--",
               label=f"Median: {med:.2f}")
    ax.set_xlabel("Detection rate  (skeleton frames / visible frames)")
    ax.set_ylabel("Number of tracked persons")
    ax.set_title(
        "Pose Skeleton Detection Rate per Tracked Person\n"
        "(UCF-Crime Fighting — surveillance footage)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "1_detection_rate_distribution.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Distribution of joint-confidence rates per threshold
    # ------------------------------------------------------------------
    palette = ["#5c8ee0", "#e09a5c", "#5ce07a", "#c25ce0"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, color in zip(conf_thresholds, palette):
        col  = f"kp_rate_{int(t * 100)}"
        vals = np.array([r[col] for r in rows])
        ax.hist(vals, bins=25, range=(0, 1), alpha=0.65, color=color,
                edgecolor="white", linewidth=0.5,
                label=f"conf ≥ {t:.0%}  (median {np.median(vals):.2f})")
    ax.set_xlabel("Fraction of (frame × joint) pairs above threshold")
    ax.set_ylabel("Number of tracked persons")
    ax.set_title(
        "Keypoint Confidence Rate per Tracked Person\n"
        "(UCF-Crime Fighting)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "2_joint_confidence_rate_distribution.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Per-joint mean confidence (bar chart)
    # ------------------------------------------------------------------
    weights     = np.array([max(r["skeleton_frames"], 1) for r in rows], dtype=float)
    joint_mat   = np.array([[r[f"joint_{jn}"] for jn in COCO_KP_NAMES] for r in rows])
    weighted_mu = np.average(joint_mat, axis=0, weights=weights)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(NUM_KP), weighted_mu, color=_bar_colors(),
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(NUM_KP))
    ax.set_xticklabels([n.replace("_", "\n") for n in COCO_KP_NAMES], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean keypoint confidence")
    ax.set_title(
        "Per-Joint Mean Confidence in Surveillance Footage\n"
        "(weighted by skeleton-frame count)"
    )
    for t, ls in zip([0.5, 0.75], ["--", ":"]):
        ax.axhline(t, color="black", linestyle=ls, linewidth=0.8, alpha=0.6,
                   label=f"{t} threshold")
    legend_patches = [
        mpatches.Patch(color=GROUP_COLORS["face"],  label="Face"),
        mpatches.Patch(color=GROUP_COLORS["upper"], label="Upper body"),
        mpatches.Patch(color=GROUP_COLORS["lower"], label="Lower body"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="0.5 threshold"),
        plt.Line2D([0], [0], color="black", linestyle=":",  label="0.75 threshold"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "3_per_joint_confidence.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Per-video summary (median detection rate + kp_rate_50)
    # ------------------------------------------------------------------
    first_conf_col = f"kp_rate_{int(conf_thresholds[0] * 100)}"
    videos = sorted(set(r["video"] for r in rows))
    vid_det, vid_kp = [], []
    for v in videos:
        vr = [r for r in rows if r["video"] == v]
        vid_det.append(np.median([r["detection_rate"] for r in vr]))
        vid_kp.append(np.median([r[first_conf_col] for r in vr]))

    x = np.arange(len(videos))
    fig, ax = plt.subplots(figsize=(max(8, len(videos) * 0.45), 4))
    ax.bar(x - 0.2, vid_det, 0.38, label="Skeleton detection rate", color="#e05c5c")
    ax.bar(x + 0.2, vid_kp,  0.38,
           label=f"Joint conf ≥ {conf_thresholds[0]:.0%} rate", color="#5c8ee0")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [v.replace("Fighting", "F").replace("_x264", "") for v in videos],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Median rate (over tracks)")
    ax.set_title("Per-Video Pose Quality — UCF-Crime Fighting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "4_per_video_summary.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# CSV + JSON output
# ---------------------------------------------------------------------------

def save_csv(all_rows, output_dir):
    if not all_rows:
        return
    out = Path(output_dir) / "per_track_stats.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  CSV  → {out}")


def save_summary(all_rows, output_dir, conf_thresholds, min_track_frames):
    if not all_rows:
        return
    rows = [r for r in all_rows if r["total_frames"] >= min_track_frames] or all_rows

    det_rates = np.array([r["detection_rate"] for r in rows])
    summary = {
        "n_videos":               len(set(r["video"] for r in all_rows)),
        "n_tracks_total":         len(all_rows),
        f"n_tracks_min{min_track_frames}frames": len(rows),
        "detection_rate": {
            "mean":          round(float(np.mean(det_rates)),   4),
            "median":        round(float(np.median(det_rates)), 4),
            "std":           round(float(np.std(det_rates)),    4),
            "pct_below_0.5": round(float(np.mean(det_rates < 0.5)), 4),
        },
    }
    for t in conf_thresholds:
        col  = f"kp_rate_{int(t * 100)}"
        vals = np.array([r[col] for r in rows])
        summary[col] = {
            "mean":   round(float(np.mean(vals)),   4),
            "median": round(float(np.median(vals)), 4),
            "std":    round(float(np.std(vals)),    4),
        }

    out = Path(output_dir) / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  JSON → {out}")
    print("\n" + json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model : {args.model}")
    model = YOLO(args.model)

    video_paths = sorted(Path(args.video_dir).glob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in {args.video_dir}")
    print(f"Found {len(video_paths)} videos in {args.video_dir}\n")

    all_rows = []

    for i, vp in enumerate(video_paths):
        print(f"[{i + 1:3d}/{len(video_paths)}] {vp.name}")

        ann_path = (out_dir / "annotated" / vp.name) if i < args.save_annotated else None

        tracks = process_video(
            model=model,
            video_path=vp,
            device=args.device,
            det_conf=args.det_conf,
            iou=args.iou,
            skeleton_conf=args.skeleton_conf,
            save_annotated_path=ann_path,
        )

        rows = compute_track_stats(vp.name, tracks, args.conf_thresholds)
        all_rows.extend(rows)

        if rows:
            med_det = np.median([r["detection_rate"] for r in rows])
            med_kp  = np.median([r[f"kp_rate_{int(args.conf_thresholds[0]*100)}"] for r in rows])
            print(f"         tracks={len(rows)}  "
                  f"med_det_rate={med_det:.2f}  "
                  f"med_kp_rate@{args.conf_thresholds[0]:.0%}={med_kp:.2f}")
        else:
            print("         no tracks found")

    print("\n--- Saving outputs ---")
    save_csv(all_rows, out_dir)
    save_summary(all_rows, out_dir, args.conf_thresholds, args.min_track_frames)
    plot_results(all_rows, out_dir, args.conf_thresholds, args.min_track_frames)
    print("\nDone.")


if __name__ == "__main__":
    main()
