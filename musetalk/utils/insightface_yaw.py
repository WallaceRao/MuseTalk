"""InsightFace 1k3d68 yaw estimator (degrees)."""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger("musetalk_service")


class InsightFaceYawEstimator:
    """Run buffalo_l ``1k3d68.onnx`` and return head yaw in degrees."""

    def __init__(
        self,
        model_path: str,
        *,
        providers: Optional[Sequence[str]] = None,
    ):
        model_path = os.path.abspath(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"1k3d68 model not found: {model_path}")

        from insightface.model_zoo import get_model

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.model = get_model(model_path, providers=list(providers))
        if self.model is None:
            raise RuntimeError(f"Failed to load InsightFace landmark model: {model_path}")
        # Landmark.prepare only flips to CPU when ctx_id < 0.
        self.model.prepare(ctx_id=0)
        if not getattr(self.model, "require_pose", False):
            raise RuntimeError(
                f"Model does not expose 3D pose (expected 1k3d68): {model_path}"
            )
        self.model_path = model_path
        logger.info(
            "InsightFace yaw estimator ready (%s, providers=%s)",
            model_path,
            getattr(self.model.session, "_providers", providers),
        )

    def yaw_deg(self, frame_bgr: np.ndarray, bbox) -> Optional[float]:
        """Return signed yaw degrees for one face box, or None on failure."""
        if frame_bgr is None or bbox is None:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError, IndexError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        h, w = frame_bgr.shape[:2]
        x1 = max(0.0, min(float(w - 1), x1))
        x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h - 1), y1))
        y2 = max(0.0, min(float(h), y2))
        if x2 <= x1 or y2 <= y1:
            return None

        from insightface.app.common import Face

        face = Face(bbox=np.asarray([x1, y1, x2, y2], dtype=np.float32))
        try:
            self.model.get(frame_bgr, face)
        except Exception:
            logger.debug("1k3d68 inference failed", exc_info=True)
            return None

        pose = face.get("pose") if hasattr(face, "get") else getattr(face, "pose", None)
        if pose is None:
            return None
        try:
            # InsightFace pose = [pitch, yaw, roll] in degrees.
            yaw = float(pose[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(yaw):
            return None
        return yaw

    def estimate_sequence(
        self,
        frames: Sequence[np.ndarray],
        coord_list: Sequence,
        *,
        coord_placeholder,
        stride: int = 3,
        indices: Optional[Sequence[int]] = None,
    ) -> List[Optional[float]]:
        """Estimate yaw (degrees) for a frame sequence.

        Runs every ``stride`` frame (plus last) among ``indices`` or all frames
        with a valid bbox, then linearly interpolates gaps.
        """
        from musetalk.utils.yaw_gate import interpolate_sparse_yaw

        n = len(frames)
        out_sparse: List[Optional[float]] = [None] * n
        if n == 0:
            return out_sparse

        stride = max(1, int(stride))
        if indices is None:
            candidate = list(range(n))
        else:
            candidate = sorted({int(i) for i in indices if 0 <= int(i) < n})
        if not candidate:
            return out_sparse

        detect_indices = candidate[::stride]
        if candidate[-1] not in detect_indices:
            detect_indices.append(candidate[-1])

        for idx in detect_indices:
            bbox = coord_list[idx] if idx < len(coord_list) else None
            if bbox is None or bbox == coord_placeholder:
                continue
            out_sparse[idx] = self.yaw_deg(frames[idx], bbox)

        if stride == 1 and indices is None:
            return out_sparse
        return interpolate_sparse_yaw(out_sparse, n)
