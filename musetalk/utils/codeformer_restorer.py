"""CodeFormer face restoration for MuseTalk speaking-face crops."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

logger = logging.getLogger("musetalk_service")

CODEFORMER_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../third_party/CodeFormer")
)


def _ensure_codeformer_path() -> None:
    version_py = os.path.join(CODEFORMER_ROOT, "basicsr", "version.py")
    if not os.path.isfile(version_py):
        with open(version_py, "w", encoding="utf-8") as f:
            f.write(
                "__version__ = '1.4.2'\n"
                "__gitsha__ = 'unknown'\n"
                "version_info = (1, 4, 2)\n"
            )
    if CODEFORMER_ROOT not in sys.path:
        sys.path.insert(0, CODEFORMER_ROOT)


class CodeFormerRestorer:
    """Restore already-cropped BGR face images (no face detection)."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        fidelity_weight: float = 0.7,
        input_size: int = 512,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CodeFormer weights not found: {model_path}")
        if not os.path.isdir(CODEFORMER_ROOT):
            raise FileNotFoundError(f"CodeFormer repo not found: {CODEFORMER_ROOT}")

        _ensure_codeformer_path()
        # Import registers CodeFormer into ARCH_REGISTRY via codeformer_arch.
        from basicsr.archs.codeformer_arch import CodeFormer  # noqa: WPS410

        self.device = device
        self.fidelity_weight = float(np.clip(fidelity_weight, 0.0, 1.0))
        self.input_size = int(input_size)

        net = CodeFormer(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state = checkpoint.get("params_ema", checkpoint)
        net.load_state_dict(state)
        net.eval()
        self.net = net
        logger.info(
            "CodeFormer loaded from %s (fidelity=%.2f, input_size=%d)",
            model_path,
            self.fidelity_weight,
            self.input_size,
        )

    @torch.no_grad()
    def restore_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Restore a cropped face image.

        Args:
            face_bgr: HxWx3 uint8 BGR crop (e.g. MuseTalk 256x256 output).

        Returns:
            Restored face as uint8 BGR, same spatial size as input.
            On failure, returns the original crop unchanged.
        """
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            return face_bgr

        from basicsr.utils import img2tensor, tensor2img  # noqa: WPS410

        orig_h, orig_w = face_bgr.shape[:2]
        try:
            face = face_bgr.astype(np.uint8)
            if face.shape[0] != self.input_size or face.shape[1] != self.input_size:
                face = cv2.resize(
                    face,
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            face_t = img2tensor(face / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(self.device)

            output = self.net(face_t, w=self.fidelity_weight, adain=True)[0]
            restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            restored = restored.astype(np.uint8)

            if restored.shape[0] != orig_h or restored.shape[1] != orig_w:
                restored = cv2.resize(
                    restored, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
                )
            return restored
        except Exception as exc:
            logger.warning("CodeFormer restore failed, using original face: %s", exc)
            return face_bgr.astype(np.uint8)
