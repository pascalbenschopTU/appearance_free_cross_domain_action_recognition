import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple

try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_IMPORT_ERR = None
except Exception as _e:
    YOLO = None
    _YOLO_IMPORT_ERR = _e

_YOLO_MODEL_CACHE: Dict[str, Any] = {}


def yolo_is_available() -> bool:
    return YOLO is not None


def yolo_import_error_repr() -> str:
    return repr(_YOLO_IMPORT_ERR)


def _square_bbox_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    """Return the smallest in-frame square bbox that fully contains (x1,y1,x2,y2)."""
    x1 = int(np.clip(x1, 0, max(0, frame_w - 1)))
    y1 = int(np.clip(y1, 0, max(0, frame_h - 1)))
    x2 = int(np.clip(x2, x1 + 1, frame_w))
    y2 = int(np.clip(y2, y1 + 1, frame_h))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = max(bw, bh)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx2 = sx1 + side
    sy2 = sy1 + side

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > frame_w:
        shift = sx2 - frame_w
        sx1 -= shift
        sx2 = frame_w
    if sy2 > frame_h:
        shift = sy2 - frame_h
        sy1 -= shift
        sy2 = frame_h

    sx1 = int(max(0, sx1))
    sy1 = int(max(0, sy1))
    sx2 = int(min(frame_w, sx2))
    sy2 = int(min(frame_h, sy2))
    if sx2 <= sx1 or sy2 <= sy1:
        return 0, 0, frame_w, frame_h
    return sx1, sy1, sx2, sy2


def _get_yolo_model(model_name: str):
    if YOLO is None:
        raise RuntimeError(
            f"YOLO not available. Install ultralytics. Import error: {_YOLO_IMPORT_ERR!r}"
        )
    model = _YOLO_MODEL_CACHE.get(model_name)
    if model is None:
        model = YOLO(model_name)
        _YOLO_MODEL_CACHE[model_name] = model
    return model


def detect_square_roi_yolo_person(
    path: str,
    model_name: str = "yolo11n.pt",
    stride: int = 3,
    conf: float = 0.25,
    device: Optional[str] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Detect person boxes through the video and return a square union ROI."""
    model = _get_yolo_model(model_name)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for ROI detection: {path}")

    stride = max(1, int(stride))
    union_x1 = union_y1 = None
    union_x2 = union_y2 = None
    frame_h = frame_w = None

    t = -1
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t += 1
        if t % stride != 0:
            continue

        if frame_h is None or frame_w is None:
            frame_h, frame_w = frame_bgr.shape[:2]

        results = model.predict(
            source=frame_bgr,
            classes=[0],  # COCO person class
            conf=float(conf),
            device=device,
            verbose=False,
        )
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            continue

        for b in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b[:4]
            x1i = int(max(0, np.floor(x1)))
            y1i = int(max(0, np.floor(y1)))
            x2i = int(np.ceil(x2))
            y2i = int(np.ceil(y2))
            if x2i <= x1i or y2i <= y1i:
                continue
            if union_x1 is None:
                union_x1, union_y1, union_x2, union_y2 = x1i, y1i, x2i, y2i
            else:
                union_x1 = min(union_x1, x1i)
                union_y1 = min(union_y1, y1i)
                union_x2 = max(union_x2, x2i)
                union_y2 = max(union_y2, y2i)

    cap.release()

    if union_x1 is None or frame_h is None or frame_w is None:
        return None

    return _square_bbox_xyxy(union_x1, union_y1, union_x2, union_y2, frame_w, frame_h)


def detect_square_roi_largest_motion(
    path: str,
    threshold: float,
    stride: int = 1,
    min_area: int = 64,
) -> Optional[Tuple[int, int, int, int]]:
    """Find a square ROI around the dominant motion region from frame differences."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for ROI detection: {path}")

    stride = max(1, int(stride))
    prev_gray = None
    acc = None
    frame_h = frame_w = None
    t = -1

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t += 1
        if t % stride != 0:
            continue

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if frame_h is None or frame_w is None:
            frame_h, frame_w = gray.shape[:2]
            acc = np.zeros((frame_h, frame_w), dtype=np.float32)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mask = (diff > float(threshold)).astype(np.uint8)
            acc += mask
        prev_gray = gray

    cap.release()

    if acc is None or float(acc.max()) <= 0.0 or frame_h is None or frame_w is None:
        return None

    motion_any = (acc > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(motion_any, connectivity=8)
    if n_labels <= 1:
        return None

    label_ids = labels.reshape(-1)
    weights = acc.reshape(-1)
    scores = np.bincount(label_ids, weights=weights, minlength=n_labels)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    scores[areas < float(max(1, int(min_area)))] = 0.0
    scores[0] = 0.0  # background
    best = int(np.argmax(scores))
    if best <= 0 or float(scores[best]) <= 0.0:
        return None

    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    return _square_bbox_xyxy(x, y, x + w, y + h, frame_w, frame_h)


def crop_frame_by_roi(
    frame_bgr: np.ndarray,
    roi_xyxy: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Safely crop by ROI; if invalid/missing, returns original frame."""
    if roi_xyxy is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = roi_xyxy
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, x1 + 1, w))
    y2 = int(np.clip(y2, y1 + 1, h))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    cropped = frame_bgr[y1:y2, x1:x2]
    if cropped.shape[0] <= 1 or cropped.shape[1] <= 1:
        return frame_bgr
    return cropped
