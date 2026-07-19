"""Lightweight SCRFD face detector (ONNXRuntime)."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _nms(dets: np.ndarray, thresh: float) -> List[int]:
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class SCRFDDetector:
    """SCRFD ONNX face detector (e.g. InsightFace ``det_10g.onnx``)."""

    def __init__(
        self,
        model_path: str,
        *,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        providers: Sequence[str] | None = None,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        import onnxruntime as ort

        available = ort.get_available_providers()
        if providers is None:
            providers = [
                p
                for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in available
            ] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=list(providers))
        self.providers = list(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = (int(input_size[0]), int(input_size[1]))  # (w, h)
        self._feat_stride_fpn = (8, 16, 32)
        self._num_anchors = 2
        self._center_cache: dict = {}

    def _preprocess(self, image_bgr: np.ndarray):
        h0, w0 = image_bgr.shape[:2]
        input_w, input_h = self.input_size
        scale = min(input_w / float(w0), input_h / float(h0))
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        blob[:nh, :nw] = resized
        blob = (blob.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None, ...]
        return blob, scale

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Return list of (x1, y1, x2, y2, score) in original image coords."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        blob, scale = self._preprocess(image_bgr)
        net_outs = self.session.run(None, {self.input_name: blob})

        input_w, input_h = self.input_size
        scores_list = []
        bboxes_list = []
        # InsightFace det_10g outputs: score/bbox(/kps) per FPN level.
        fmc = len(self._feat_stride_fpn)
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            if scores.ndim == 3:
                scores = scores.reshape(-1)
            else:
                scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4) * float(stride)

            height = input_h // stride
            width = input_w // stride
            key = (height, width, stride)
            if key not in self._center_cache:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                self._center_cache[key] = anchor_centers
            anchor_centers = self._center_cache[key]

            pos_inds = np.where(scores >= self.conf_threshold)[0]
            if pos_inds.size == 0:
                continue
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

        if not scores_list:
            return []

        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0)
        # Map from letterboxed input space back to original.
        bboxes = bboxes / max(scale, 1e-6)
        dets = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, self.nms_threshold)
        out = []
        h0, w0 = image_bgr.shape[:2]
        for i in keep:
            x1, y1, x2, y2, sc = dets[i].tolist()
            x1 = float(np.clip(x1, 0, w0 - 1))
            y1 = float(np.clip(y1, 0, h0 - 1))
            x2 = float(np.clip(x2, 0, w0 - 1))
            y2 = float(np.clip(y2, 0, h0 - 1))
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, float(sc)))
        return out
