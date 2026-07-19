import sys
from os import listdir, path
import subprocess
import logging
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

from musetalk.utils.scrfd_detector import SCRFDDetector

logger = logging.getLogger("musetalk_service")

# PyTorch 2.6+ defaults to weights_only=True; allow trusted local checkpoints.
_original_torch_load = torch.load

def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load

# Face detector: SCRFD (replaces S3FD). Landmark: DWPose RTMPose-M (replaces L).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = os.environ.get(
    "MUSETALK_DWPOSE_CONFIG",
    "./musetalk/utils/dwpose/rtmpose-m_8xb64-270e_coco-ubody-wholebody-256x192.py",
)
checkpoint_file = os.environ.get(
    "MUSETALK_DWPOSE_CKPT",
    "./models/dwpose/dw-mm_ucoco.pth",
)
model = init_model(config_file, checkpoint_file, device=device)

_scrfd_model_path = os.environ.get(
    "MUSETALK_SCRFD_MODEL",
    "./models/scrfd/det_10g.onnx",
)
_scrfd = SCRFDDetector(
    _scrfd_model_path,
    conf_threshold=float(os.environ.get("MUSETALK_SCRFD_CONF", "0.5")),
)
logger.info("SCRFD ONNX providers: %s", getattr(_scrfd, "providers", None))

# maker if the bbox is not sufficient
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for frame in tqdm(frames):
        faces = _collect_scrfd_faces(frame)
        if not faces:
            continue
        fh, fw = frame.shape[:2]
        best = max(faces, key=lambda f: _seed_face_score(f, fw, fh))
        results = inference_topdown(
            model,
            frame,
            bboxes=np.asarray([_expand_face_bbox_for_pose(best, fw, fh)], dtype=np.float32),
            bbox_format="xyxy",
        )
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        if keypoints is None or len(keypoints) == 0:
            continue
        face_land_mark = keypoints[0][23:91].astype(np.int32)
        range_minus = (face_land_mark[30] - face_land_mark[29])[1]
        range_plus = (face_land_mark[29] - face_land_mark[28])[1]
        average_range_minus.append(range_minus)
        average_range_plus.append(range_plus)
    if not average_range_minus:
        return f'Total frame:「{len(frames)}」 No face detected'
    text_range = (
        f"Total frame:「{len(frames)}」 Manually adjust range : "
        f"[ -{int(sum(average_range_minus) / len(average_range_minus))}"
        f"~{int(sum(average_range_plus) / len(average_range_plus))} ] "
        f", the current value: {upperbondrange}"
    )
    return text_range
    

def _resize_for_detection(frame, detect_short_side=720):
    """
    Downscale frame for detection when short side exceeds detect_short_side.
    Small/equal resolution frames are returned unchanged (scale 1.0).
    Returns (detect_frame, scale_x, scale_y) to map detect coords back to original.
    """
    h, w = frame.shape[:2]
    short = min(h, w)
    if detect_short_side is None or detect_short_side <= 0 or short <= detect_short_side:
        return frame, 1.0, 1.0

    scale = float(detect_short_side) / float(short)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / float(new_w), h / float(new_h)


def _scale_bbox_to_original(bbox, scale_x, scale_y):
    if bbox == coord_placeholder or (scale_x == 1.0 and scale_y == 1.0):
        return bbox
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def _bbox_area(bbox):
    if bbox is None or bbox == coord_placeholder:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _face_meets_min_area(bbox, frame_w, frame_h, min_area_ratio: float) -> bool:
    """True if face box area is at least ``min_area_ratio`` of the frame area."""
    if min_area_ratio is None or min_area_ratio <= 0:
        return True
    frame_area = float(max(1, frame_w) * max(1, frame_h))
    return _bbox_area(bbox) >= min_area_ratio * frame_area


def _filter_candidates_by_min_area(candidates, frame_w, frame_h, min_area_ratio: float):
    if min_area_ratio is None or min_area_ratio <= 0:
        return candidates
    return [
        c
        for c in candidates
        if _face_meets_min_area(c["bbox"], frame_w, frame_h, min_area_ratio)
    ]


def _bbox_iou(a, b):
    if a is None or b is None or a == coord_placeholder or b == coord_placeholder:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_center_dist(a, b):
    if a is None or b is None or a == coord_placeholder or b == coord_placeholder:
        return float("inf")
    ax = 0.5 * (a[0] + a[2])
    ay = 0.5 * (a[1] + a[3])
    bx = 0.5 * (b[0] + b[2])
    by = 0.5 * (b[1] + b[3])
    return float(np.hypot(ax - bx, ay - by))


def _face_bbox_from_landmarks(face_land_mark, upperbondrange=0, scale_y=1.0):
    """Build MuseTalk half-face crop bbox from DWPose face landmarks (detect space)."""
    face_land_mark = np.asarray(face_land_mark, dtype=np.float64)
    if face_land_mark.shape[0] < 31:
        return None, None, None

    half_face_coord = face_land_mark[29].copy()
    range_minus = float((face_land_mark[30] - face_land_mark[29])[1])
    range_plus = float((face_land_mark[29] - face_land_mark[28])[1])
    if upperbondrange != 0:
        half_face_coord[1] = half_face_coord[1] + float(upperbondrange) / float(scale_y)

    half_face_dist = float(np.max(face_land_mark[:, 1]) - half_face_coord[1])
    upper_bond = max(0.0, half_face_coord[1] - half_face_dist)
    f_landmark = (
        float(np.min(face_land_mark[:, 0])),
        float(upper_bond),
        float(np.max(face_land_mark[:, 0])),
        float(np.max(face_land_mark[:, 1])),
    )
    x1, y1, x2, y2 = f_landmark
    if y2 - y1 <= 1 or x2 - x1 <= 1:
        return None, range_minus, range_plus
    return f_landmark, range_minus, range_plus


def _mouth_bbox_from_face_landmarks(face_land_mark, pad_ratio: float = 0.35):
    """Mouth box centered on DWPose lip landmarks (pts 48-67).

    VSDLM expects a mouth-region crop roughly like a detector box — not an
    ultra-tight lip outline (which, when resized to 30x48, falsely looks open).
    Center on the lips, size ≈ 55%×35% of the face landmark span (matches the
    geometric mouth band that VSDLM handles reliably).

    Returns ``(bbox, mouth_aspect_ratio)`` where MAR = inner-lip height / width
    (low ≈ closed mouth).
    """
    lm = np.asarray(face_land_mark, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[0] < 68 or lm.shape[1] < 2:
        return None, None
    lips = lm[48:68]
    if not np.isfinite(lips).all() or np.allclose(lips, 0):
        return None, None
    x1 = float(np.min(lips[:, 0]))
    y1 = float(np.min(lips[:, 1]))
    x2 = float(np.max(lips[:, 0]))
    y2 = float(np.max(lips[:, 1]))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    lip_w = max(1.0, x2 - x1)
    lip_h = max(1.0, y2 - y1)

    face_w = float(np.max(lm[:, 0]) - np.min(lm[:, 0]))
    face_h = float(np.max(lm[:, 1]) - np.min(lm[:, 1]))
    min_w = max(lip_w * (1.0 + 2.0 * pad_ratio), 0.55 * max(face_w, 1.0))
    min_h = max(lip_h * (1.0 + 2.0 * pad_ratio), 0.35 * max(face_h, 1.0))
    # Slight downward bias so lower lip stays inside.
    cy = cy + 0.08 * min_h
    bbox = (
        cx - 0.5 * min_w,
        cy - 0.5 * min_h,
        cx + 0.5 * min_w,
        cy + 0.5 * min_h,
    )
    # Geometric openness: inner lip vertical gap / outer mouth width.
    width = float(np.linalg.norm(lm[54] - lm[48]))
    height = float(np.linalg.norm(lm[66] - lm[62]))
    mar = height / max(width, 1e-6)
    return bbox, float(mar)


def _clamp_bbox(bbox, frame_w: int, frame_h: int):
    if bbox is None or bbox == coord_placeholder:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = float(np.clip(x1, 0, frame_w - 1))
    y1 = float(np.clip(y1, 0, frame_h - 1))
    x2 = float(np.clip(x2, 0, frame_w - 1))
    y2 = float(np.clip(y2, 0, frame_h - 1))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return (x1, y1, x2, y2)


def _collect_scrfd_faces(detect_frame):
    """Return SCRFD face boxes in detect-frame space as (x1,y1,x2,y2) tuples."""
    faces = []
    for x1, y1, x2, y2, _score in _scrfd.detect(detect_frame):
        faces.append((float(x1), float(y1), float(x2), float(y2)))
    return faces


def _expand_face_bbox_for_pose(bbox, frame_w, frame_h):
    """Expand a face box so wholebody RTMPose sees enough head/shoulder context."""
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    nx1 = x1 - 0.35 * w
    nx2 = x2 + 0.35 * w
    ny1 = y1 - 0.55 * h
    ny2 = y2 + 0.45 * h
    return (
        float(np.clip(nx1, 0, frame_w - 1)),
        float(np.clip(ny1, 0, frame_h - 1)),
        float(np.clip(nx2, 0, frame_w - 1)),
        float(np.clip(ny2, 0, frame_h - 1)),
    )


def _seed_face_score(bbox, frame_w, frame_h):
    """Prefer large faces near the frame center (better for multi-person shots)."""
    area = _bbox_area(bbox)
    if area <= 0 or frame_w <= 0 or frame_h <= 0:
        return 0.0
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])
    dx = (cx - 0.5 * frame_w) / float(frame_w)
    dy = (cy - 0.5 * frame_h) / float(frame_h)
    dist = float(np.hypot(dx, dy))
    center_weight = max(0.2, 1.0 - dist)
    return area * center_weight


def _select_tracked_candidate(
    candidates,
    prev_bbox,
    min_iou=0.15,
    max_center_ratio=1.5,
    frame_shape=None,
):
    """
    Pick the face to track.
    - With prev_bbox: prefer max IoU if >= min_iou; else nearest center within
      max_center_ratio * prev face diagonal; else None (caller may re-seed).
    - Without prev_bbox: largest face near frame center.
    """
    if not candidates:
        return None

    if prev_bbox is None or prev_bbox == coord_placeholder:
        if frame_shape is not None:
            fh, fw = frame_shape[:2]
            return max(
                candidates,
                key=lambda c: _seed_face_score(c["bbox"], fw, fh),
            )
        return max(candidates, key=lambda c: _bbox_area(c["bbox"]))

    best_iou = None
    best_iou_val = -1.0
    for c in candidates:
        iou = _bbox_iou(prev_bbox, c["bbox"])
        if iou > best_iou_val:
            best_iou_val = iou
            best_iou = c
    if best_iou is not None and best_iou_val >= min_iou:
        return best_iou

    prev_diag = max(
        1.0,
        float(np.hypot(prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1])),
    )
    best_dist = None
    best_dist_val = float("inf")
    for c in candidates:
        dist = _bbox_center_dist(prev_bbox, c["bbox"])
        if dist < best_dist_val:
            best_dist_val = dist
            best_dist = c
    if best_dist is not None and best_dist_val <= max_center_ratio * prev_diag:
        return best_dist
    return None


def _detect_bbox_for_frame(
    frame,
    upperbondrange=0,
    detect_short_side=720,
    prev_bbox=None,
    track_min_iou=0.15,
    min_face_area_ratio: float = 1.0 / 50.0,
):
    """Run SCRFD (+ optional DWPose-M landmarks) on a single frame.

    Pipeline:
      1. SCRFD detects faces (fast).
      2. Track / seed one face (largest near center).
      3. If no face → placeholder (skip pose).
      4. DWPose-M topdown on an expanded face box → MuseTalk half-face crop.

    Returns (bbox, range_minus, range_plus, held, mouth_bbox, mouth_mar).
    ``held`` is always False now. ``mouth_bbox`` / ``mouth_mar`` come from lip
    landmarks when available.
    """
    h_orig, w_orig = frame.shape[:2]
    detect_frame, scale_x, scale_y = _resize_for_detection(frame, detect_short_side)
    h_det, w_det = detect_frame.shape[:2]

    prev_det = None
    if prev_bbox is not None and prev_bbox != coord_placeholder:
        prev_det = (
            prev_bbox[0] / scale_x,
            prev_bbox[1] / scale_y,
            prev_bbox[2] / scale_x,
            prev_bbox[3] / scale_y,
        )

    scrfd_faces = [
        f
        for f in _collect_scrfd_faces(detect_frame)
        if _face_meets_min_area(f, w_det, h_det, min_face_area_ratio)
    ]
    frame_shape = (h_det, w_det)
    if not scrfd_faces:
        return coord_placeholder, None, None, False, None, None

    scrfd_cands = [
        {"bbox": f, "range_minus": None, "range_plus": None} for f in scrfd_faces
    ]
    selected_face = _select_tracked_candidate(
        scrfd_cands, prev_det, min_iou=track_min_iou, frame_shape=frame_shape
    )
    if selected_face is None:
        if prev_det is not None:
            return coord_placeholder, None, None, False, None, None
        selected_face = max(
            scrfd_cands, key=lambda c: _seed_face_score(c["bbox"], w_det, h_det)
        )

    face_bbox = selected_face["bbox"]
    pose_bbox = _expand_face_bbox_for_pose(face_bbox, w_det, h_det)
    if pose_bbox[2] <= pose_bbox[0] or pose_bbox[3] <= pose_bbox[1]:
        out_bbox = _scale_bbox_to_original(face_bbox, scale_x, scale_y)
        if not _face_meets_min_area(out_bbox, w_orig, h_orig, min_face_area_ratio):
            return coord_placeholder, None, None, False, None, None
        return out_bbox, None, None, False, None, None

    results = inference_topdown(
        model,
        detect_frame,
        bboxes=np.asarray([pose_bbox], dtype=np.float32),
        bbox_format="xyxy",
    )
    results = merge_data_samples(results)
    keypoints = results.pred_instances.keypoints
    selected = None
    mouth_det = None
    mouth_mar = None
    if keypoints is not None and len(keypoints) > 0:
        face_land_mark = np.asarray(keypoints[0][23:91], dtype=np.float64)
        if (
            np.isfinite(face_land_mark).all()
            and not np.allclose(face_land_mark, 0)
            and (face_land_mark[:, 0].max() - face_land_mark[:, 0].min()) >= 5
            and (face_land_mark[:, 1].max() - face_land_mark[:, 1].min()) >= 5
        ):
            f_landmark, range_minus, range_plus = _face_bbox_from_landmarks(
                face_land_mark, upperbondrange=upperbondrange, scale_y=scale_y
            )
            if f_landmark is not None:
                selected = {
                    "bbox": f_landmark,
                    "range_minus": range_minus,
                    "range_plus": range_plus,
                }
            mouth_det, mouth_mar = _mouth_bbox_from_face_landmarks(face_land_mark)

    if selected is None:
        out_bbox = _scale_bbox_to_original(face_bbox, scale_x, scale_y)
        if not _face_meets_min_area(out_bbox, w_orig, h_orig, min_face_area_ratio):
            return coord_placeholder, None, None, False, None, None
        return out_bbox, None, None, False, None, None

    f_landmark = selected["bbox"]
    x1, y1, x2, y2 = f_landmark
    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        out_bbox = _scale_bbox_to_original(face_bbox, scale_x, scale_y)
        range_minus = selected.get("range_minus")
        range_plus = selected.get("range_plus")
    else:
        out_bbox = _scale_bbox_to_original(
            (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            scale_x,
            scale_y,
        )
        range_minus = selected.get("range_minus")
        range_plus = selected.get("range_plus")

    if not _face_meets_min_area(out_bbox, w_orig, h_orig, min_face_area_ratio):
        return coord_placeholder, None, None, False, None, None

    if range_minus is not None:
        range_minus = float(range_minus) * scale_y
        range_plus = float(range_plus) * scale_y

    mouth_bbox = None
    if mouth_det is not None:
        mouth_bbox = _clamp_bbox(
            _scale_bbox_to_original(mouth_det, scale_x, scale_y),
            w_orig,
            h_orig,
        )
    return out_bbox, range_minus, range_plus, False, mouth_bbox, mouth_mar




def _lerp_bbox(b0, b1, t, min_iou_for_lerp=0.15):
    if b0 == coord_placeholder or b1 == coord_placeholder:
        return b0 if t < 0.5 else b1
    # Avoid morphing a bbox between two different people / shot cuts.
    if _bbox_iou(b0, b1) < min_iou_for_lerp:
        return b0 if t < 0.5 else b1
    return tuple(int(round(b0[j] + (b1[j] - b0[j]) * t)) for j in range(4))


def _interpolate_sparse_coords(sparse_coords, n_frames):
    """Fill None slots by linear interpolation between detected keyframes."""
    result = [coord_placeholder] * n_frames
    key_idxs = [i for i, c in enumerate(sparse_coords) if c is not None]
    if not key_idxs:
        return result

    first = key_idxs[0]
    for i in range(first):
        result[i] = sparse_coords[first]

    for k, i0 in enumerate(key_idxs):
        result[i0] = sparse_coords[i0]
        if k + 1 >= len(key_idxs):
            for i in range(i0 + 1, n_frames):
                result[i] = sparse_coords[i0]
            break
        i1 = key_idxs[k + 1]
        b0, b1 = sparse_coords[i0], sparse_coords[i1]
        span = i1 - i0
        for i in range(i0 + 1, i1):
            result[i] = _lerp_bbox(b0, b1, (i - i0) / span)
    return result


def _interpolate_sparse_mouth_coords(sparse_mouth, sparse_face, n_frames):
    """Interpolate mouth boxes; hard-cut when face identity jumps (low IoU)."""
    result = [None] * n_frames
    key_idxs = [i for i, c in enumerate(sparse_mouth) if c is not None]
    if not key_idxs:
        return result

    first = key_idxs[0]
    for i in range(first):
        result[i] = sparse_mouth[first]

    for k, i0 in enumerate(key_idxs):
        result[i0] = sparse_mouth[i0]
        if k + 1 >= len(key_idxs):
            for i in range(i0 + 1, n_frames):
                result[i] = sparse_mouth[i0]
            break
        i1 = key_idxs[k + 1]
        m0, m1 = sparse_mouth[i0], sparse_mouth[i1]
        f0 = sparse_face[i0] if sparse_face is not None else None
        f1 = sparse_face[i1] if sparse_face is not None else None
        span = i1 - i0
        # Do not morph mouth boxes across different faces / cuts.
        if (
            f0 is not None
            and f1 is not None
            and f0 != coord_placeholder
            and f1 != coord_placeholder
            and _bbox_iou(f0, f1) < 0.15
        ):
            mid = i0 + span // 2
            for i in range(i0 + 1, i1):
                result[i] = m0 if i <= mid else m1
            continue
        for i in range(i0 + 1, i1):
            t = (i - i0) / span
            result[i] = tuple(
                float(m0[j] + (m1[j] - m0[j]) * t) for j in range(4)
            )
    return result


def get_landmark_and_bbox(
    img_list=None,
    upperbondrange=0,
    detect_stride=3,
    frames=None,
    detect_short_side=720,
    min_face_area_ratio: float = 1.0 / 50.0,
    return_mouth_coords: bool = False,
):
    """
    Detect face bboxes. When detect_stride > 1, only every N-th frame (and the
    last frame) is detected; intermediate frames use linear bbox interpolation.

    Multi-person shots track one face across keyframes via IoU/center matching
    so detector person order flips do not switch identity mid-clip.

    Faces with area < ``min_face_area_ratio`` of the frame are ignored.

    Pass either img_list (paths) or frames (already decoded BGR arrays).
    When frame short side > detect_short_side, detection runs on a downscaled
    copy and bboxes are mapped back to original coordinates. Frames already
    at or below detect_short_side are unchanged.

    When ``return_mouth_coords`` is True, also return per-frame tight mouth
    boxes from DWPose lip landmarks (or None when unavailable).
    """
    if frames is None:
        if not img_list:
            raise ValueError("Either img_list or frames must be provided")
        frames = read_imgs(img_list)
    n_frames = len(frames)
    detect_stride = max(1, int(detect_stride))

    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    if detect_stride > 1:
        print(f'bbox detect_stride={detect_stride} (intermediate frames interpolated)')
    print(
        f'face tracking: IoU/center match (no hold); '
        f'min_face_area_ratio={min_face_area_ratio:.4f}'
    )

    if n_frames > 0 and detect_short_side and detect_short_side > 0:
        h0, w0 = frames[0].shape[:2]
        short0 = min(h0, w0)
        if short0 > detect_short_side:
            print(
                f'detect_short_side={detect_short_side}: '
                f'downscale {w0}x{h0} (short={short0}) for detection'
            )
        else:
            print(
                f'detect_short_side={detect_short_side}: '
                f'keep original {w0}x{h0} (short={short0} <= threshold)'
            )

    sparse_coords = [None] * n_frames
    sparse_mouth = [None] * n_frames
    sparse_mar = [None] * n_frames
    average_range_minus = []
    average_range_plus = []

    detect_indices = list(range(0, n_frames, detect_stride))
    if n_frames > 0 and (n_frames - 1) not in detect_indices:
        detect_indices.append(n_frames - 1)

    prev_bbox = None
    for idx in tqdm(detect_indices):
        bbox, range_minus, range_plus, _held, mouth_bbox, mouth_mar = _detect_bbox_for_frame(
            frames[idx],
            upperbondrange,
            detect_short_side=detect_short_side,
            prev_bbox=prev_bbox,
            min_face_area_ratio=min_face_area_ratio,
        )
        sparse_coords[idx] = bbox
        sparse_mouth[idx] = mouth_bbox
        sparse_mar[idx] = mouth_mar
        # Clear track on lost face so we never keep pasting on an empty shot.
        if bbox == coord_placeholder:
            prev_bbox = None
        else:
            prev_bbox = bbox
        if range_minus is not None:
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)

    coords_list = (
        sparse_coords
        if detect_stride == 1
        else _interpolate_sparse_coords(sparse_coords, n_frames)
    )
    mouth_coords_list = (
        sparse_mouth
        if detect_stride == 1
        else _interpolate_sparse_mouth_coords(sparse_mouth, sparse_coords, n_frames)
    )
    # MAR: hold last keyframe value (do not lerp across gaps).
    mouth_mar_list = [None] * n_frames
    last_mar = None
    for i in range(n_frames):
        if sparse_mar[i] is not None:
            last_mar = sparse_mar[i]
        mouth_mar_list[i] = last_mar
    # fill leading Nones from first key
    first_mar = next((m for m in sparse_mar if m is not None), None)
    if first_mar is not None:
        for i in range(n_frames):
            if mouth_mar_list[i] is None:
                mouth_mar_list[i] = first_mar
            else:
                break

    print("********************************************bbox_shift parameter adjustment**********************************************************")
    if average_range_minus:
        print(
            f"Total frame:「{n_frames}」 Manually adjust range : "
            f"[ -{int(sum(average_range_minus) / len(average_range_minus))}"
            f"~{int(sum(average_range_plus) / len(average_range_plus))} ] , "
            f"the current value: {upperbondrange}"
        )
    else:
        print(f"Total frame:「{n_frames}」 No valid face ranges, current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    if return_mouth_coords:
        return coords_list, frames, mouth_coords_list, mouth_mar_list
    return coords_list, frames


def mouth_bbox_from_face(
    frame: np.ndarray,
    face_bbox,
    *,
    detect_short_side: int = 720,
) -> tuple | None:
    """Run DWPose on one face box and return a tight lip bbox in original coords."""
    if frame is None or face_bbox is None or face_bbox == coord_placeholder:
        return None
    h_orig, w_orig = frame.shape[:2]
    detect_frame, scale_x, scale_y = _resize_for_detection(frame, detect_short_side)
    h_det, w_det = detect_frame.shape[:2]
    face_det = (
        face_bbox[0] / scale_x,
        face_bbox[1] / scale_y,
        face_bbox[2] / scale_x,
        face_bbox[3] / scale_y,
    )
    pose_bbox = _expand_face_bbox_for_pose(face_det, w_det, h_det)
    if pose_bbox[2] <= pose_bbox[0] or pose_bbox[3] <= pose_bbox[1]:
        return None
    results = inference_topdown(
        model,
        detect_frame,
        bboxes=np.asarray([pose_bbox], dtype=np.float32),
        bbox_format="xyxy",
    )
    results = merge_data_samples(results)
    keypoints = results.pred_instances.keypoints
    if keypoints is None or len(keypoints) == 0:
        return None
    face_land_mark = np.asarray(keypoints[0][23:91], dtype=np.float64)
    mouth_det, _mar = _mouth_bbox_from_face_landmarks(face_land_mark)
    if mouth_det is None:
        return None
    return _clamp_bbox(
        _scale_bbox_to_original(mouth_det, scale_x, scale_y),
        w_orig,
        h_orig,
    )
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
