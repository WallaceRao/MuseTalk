"""FairFace ONNX gender detector for the speaking-mask gender gate."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from musetalk.utils.scrfd_detector import SCRFDDetector

logger = logging.getLogger("musetalk_service")

GENDER_LABELS = ("Male", "Female")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.maximum(e.sum(), 1e-12)


class FairFaceGenderDetector:
    """SCRFD face crop + FairFace ONNX gender (with confidence).

    ``detect_gender`` returns ``None`` when no face is found or the gender
    softmax confidence is below ``min_confidence`` (treated as unclear by the
    gender gate so the speaking run is kept).
    """

    def __init__(
        self,
        model_path: str,
        *,
        scrfd_model_path: str,
        min_confidence: float = 0.75,
        input_size: Tuple[int, int] = (224, 224),
        bbox_pad: float = 0.25,
        providers: Sequence[str] | None = None,
        det_threshold: float = 0.5,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"FairFace ONNX not found: {model_path}")
        if not os.path.isfile(scrfd_model_path):
            raise FileNotFoundError(f"SCRFD model not found: {scrfd_model_path}")

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
        self.output_names = [o.name for o in self.session.get_outputs()]
        if len(self.output_names) < 2:
            raise RuntimeError(
                f"FairFace ONNX unexpected outputs: {self.output_names}"
            )

        self.input_size = (int(input_size[0]), int(input_size[1]))  # (h, w)
        self.bbox_pad = float(bbox_pad)
        self.min_confidence = float(min_confidence)
        self.det_threshold = float(det_threshold)
        self.detector = SCRFDDetector(
            scrfd_model_path,
            conf_threshold=self.det_threshold,
            providers=list(providers),
        )
        self.model_path = model_path
        logger.info(
            "FairFace gender detector ready (min_conf=%.2f, providers=%s)",
            self.min_confidence,
            ",".join(self.providers),
        )

    def _select_primary_bbox(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[float, float, float, float]]:
        faces = self.detector.detect(frame_bgr)
        best = None
        best_area = 0.0
        for x1, y1, x2, y2, score in faces:
            if score < self.det_threshold:
                continue
            w, h = x2 - x1, y2 - y1
            if w < 40 or h < 40:
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        return best

    def _preprocess(
        self, frame_bgr: np.ndarray, bbox: Sequence[float]
    ) -> np.ndarray:
        h0, w0 = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * self.bbox_pad
        pad_y = bh * self.bbox_pad
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w0, int(x2 + pad_x))
        y2 = min(h0, int(y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid face crop")

        crop = frame_bgr[y1:y2, x1:x2]
        ih, iw = self.input_size
        image = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))[None, ...]
        return image

    def _gender_from_outputs(
        self, outputs: List[np.ndarray]
    ) -> Tuple[Optional[str], Optional[float]]:
        # yakhyo/fairface-onnx: [race, gender, age]
        if len(outputs) >= 2:
            gender_logits = np.asarray(outputs[1]).reshape(-1)
        else:
            gender_logits = np.asarray(outputs[0]).reshape(-1)
        if gender_logits.size < 2:
            return None, None
        probs = _softmax(gender_logits[:2])
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = "male" if GENDER_LABELS[idx] == "Male" else "female"
        return label, conf

    def detect_gender_with_confidence(
        self, frame, threshold: float = 0.5
    ) -> Tuple[Optional[str], Optional[float]]:
        """Return ``(gender, confidence)`` for the largest face (BGR frame)."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return None, None
        frame_bgr = np.asarray(frame)
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None, None

        prev = self.detector.conf_threshold
        try:
            self.detector.conf_threshold = float(threshold)
            bbox = self._select_primary_bbox(frame_bgr)
            if bbox is None:
                return None, None
            blob = self._preprocess(frame_bgr, bbox)
            outputs = self.session.run(
                self.output_names, {self.input_name: blob}
            )
            return self._gender_from_outputs(outputs)
        except Exception as exc:
            logger.debug("FairFace gender failed: %s", exc)
            return None, None
        finally:
            self.detector.conf_threshold = prev

    def detect_gender(self, frame, threshold: float = 0.5) -> Optional[str]:
        """Return ``male`` / ``female``, or ``None`` when unclear/low-conf."""
        label, conf = self.detect_gender_with_confidence(frame, threshold=threshold)
        if label is None or conf is None:
            return None
        if conf < self.min_confidence:
            return None
        return label
