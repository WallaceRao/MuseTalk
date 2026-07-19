"""CodeFormer face restoration for MuseTalk speaking-face crops."""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Sequence

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
        use_fp16: bool = True,
        batch_size: int = 2,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"CodeFormer weights not found: {model_path}")
        if not os.path.isdir(CODEFORMER_ROOT):
            raise FileNotFoundError(f"CodeFormer repo not found: {CODEFORMER_ROOT}")

        _ensure_codeformer_path()
        from basicsr.archs.codeformer_arch import CodeFormer  # noqa: WPS410

        self.device = device
        self.fidelity_weight = float(np.clip(fidelity_weight, 0.0, 1.0))
        self.input_size = int(input_size)
        self.use_fp16 = bool(use_fp16) and device.type == "cuda"
        self.batch_size = max(1, int(batch_size))

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
            "CodeFormer loaded from %s (fidelity=%.2f, input_size=%d, fp16=%s, batch_size=%d)",
            model_path,
            self.fidelity_weight,
            self.input_size,
            self.use_fp16,
            self.batch_size,
        )

    def _preprocess(self, face_bgr: np.ndarray) -> tuple[torch.Tensor, int, int]:
        from basicsr.utils import img2tensor  # noqa: WPS410

        orig_h, orig_w = face_bgr.shape[:2]
        face = face_bgr.astype(np.uint8)
        if face.shape[0] != self.input_size or face.shape[1] != self.input_size:
            face = cv2.resize(
                face,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )
        face_t = img2tensor(face / 255.0, bgr2rgb=True, float32=True)
        normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        return face_t, orig_h, orig_w

    def _postprocess(
        self, output: torch.Tensor, orig_h: int, orig_w: int
    ) -> np.ndarray:
        from basicsr.utils import tensor2img  # noqa: WPS410

        restored = tensor2img(output.float(), rgb2bgr=True, min_max=(-1, 1))
        restored = restored.astype(np.uint8)
        if restored.shape[0] != orig_h or restored.shape[1] != orig_w:
            restored = cv2.resize(
                restored, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        return restored

    @torch.no_grad()
    def restore_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Restore a single cropped face image."""
        results = self.restore_faces([face_bgr])
        return results[0]

    @torch.no_grad()
    def restore_faces(self, faces_bgr: Sequence[np.ndarray]) -> List[np.ndarray]:
        """
        Restore a batch of cropped BGR face images.

        Invalid entries are returned unchanged. Processing uses ``batch_size``.
        """
        if not faces_bgr:
            return []

        outputs: List[np.ndarray | None] = [None] * len(faces_bgr)
        valid_indices: List[int] = []
        tensors: List[torch.Tensor] = []
        sizes: List[tuple[int, int]] = []

        for i, face_bgr in enumerate(faces_bgr):
            if (
                face_bgr is None
                or getattr(face_bgr, "size", 0) == 0
                or face_bgr.ndim != 3
                or face_bgr.shape[2] != 3
            ):
                outputs[i] = face_bgr
                continue
            try:
                face_t, orig_h, orig_w = self._preprocess(face_bgr)
                valid_indices.append(i)
                tensors.append(face_t)
                sizes.append((orig_h, orig_w))
            except Exception as exc:
                logger.warning("CodeFormer preprocess failed, using original: %s", exc)
                outputs[i] = face_bgr.astype(np.uint8)

        for start in range(0, len(valid_indices), self.batch_size):
            end = min(start + self.batch_size, len(valid_indices))
            batch_idxs = valid_indices[start:end]
            batch_tensors = tensors[start:end]
            batch_sizes = sizes[start:end]
            try:
                batch = torch.stack(batch_tensors, dim=0).to(
                    device=self.device, dtype=torch.float32
                )
                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred = self.net(batch, w=self.fidelity_weight, adain=True)[0]
                else:
                    pred = self.net(batch, w=self.fidelity_weight, adain=True)[0]
                for j, (idx, (oh, ow)) in enumerate(zip(batch_idxs, batch_sizes)):
                    outputs[idx] = self._postprocess(pred[j], oh, ow)
            except Exception as exc:
                logger.warning(
                    "CodeFormer batch restore failed, falling back per-face: %s",
                    exc,
                )
                for j, idx in enumerate(batch_idxs):
                    try:
                        single = (
                            batch_tensors[j]
                            .unsqueeze(0)
                            .to(device=self.device, dtype=torch.float32)
                        )
                        if self.use_fp16:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                pred = self.net(
                                    single, w=self.fidelity_weight, adain=True
                                )[0]
                        else:
                            pred = self.net(
                                single, w=self.fidelity_weight, adain=True
                            )[0]
                        oh, ow = batch_sizes[j]
                        outputs[idx] = self._postprocess(pred[0], oh, ow)
                    except Exception as inner_exc:
                        logger.warning(
                            "CodeFormer restore failed, using original face: %s",
                            inner_exc,
                        )
                        outputs[idx] = faces_bgr[idx].astype(np.uint8)

        return [
            out if out is not None else faces_bgr[i].astype(np.uint8)
            for i, out in enumerate(outputs)
        ]
