import math
import os
import sys
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

from musetalk.utils.preprocessing import coord_placeholder

LR_ASD_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../third_party/LR-ASD")
)
if LR_ASD_ROOT not in sys.path:
    sys.path.insert(0, LR_ASD_ROOT)

from loss import lossAV  # noqa: E402
from model.Model import ASD_Model  # noqa: E402

# LR-ASD was trained with 25fps video + 100Hz MFCC.
LR_ASD_FPS = 25.0
MFCC_RATE = 100.0


class LRASDDetector:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        threshold: float = 0.0,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LR-ASD model not found: {model_path}")

        self.device = device
        self.threshold = threshold
        self.model = ASD_Model().to(device).eval()
        self.loss_av = lossAV().to(device).eval()
        self._load_weights(model_path)

    def _load_weights(self, model_path: str) -> None:
        loaded_state = torch.load(model_path, map_location=self.device)
        model_state = self.model.state_dict()
        loss_state = self.loss_av.state_dict()

        for name, param in loaded_state.items():
            if name.startswith("model."):
                key = name[len("model.") :]
                if key in model_state and model_state[key].shape == param.shape:
                    model_state[key].copy_(param)
            elif name.startswith("lossAV."):
                key = name[len("lossAV.") :]
                if key in loss_state and loss_state[key].shape == param.shape:
                    loss_state[key].copy_(param)

        self.model.load_state_dict(model_state, strict=False)
        self.loss_av.load_state_dict(loss_state, strict=False)

    def _load_audio_16k(self, audio_path: str) -> np.ndarray:
        sr, audio = wavfile.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if audio.dtype == np.int16 or np.max(np.abs(audio)) > 1.5:
            audio = audio / 32768.0

        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio

    def _extract_visual_features(
        self,
        frame_list: Sequence[np.ndarray],
        coord_list: Sequence[Tuple[float, float, float, float]],
    ) -> np.ndarray:
        features = []
        for frame, bbox in zip(frame_list, coord_list):
            if bbox == coord_placeholder:
                features.append(np.zeros((112, 112), dtype=np.float32))
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros((112, 112), dtype=np.float32))
                continue

            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            face = face[56:168, 56:168]
            features.append(face.astype(np.float32))
        return np.array(features)

    @staticmethod
    def _resample_visual_to_target_fps(
        video_feature: np.ndarray,
        source_fps: float,
        target_fps: float = LR_ASD_FPS,
    ) -> np.ndarray:
        """Uniformly sample visual features from source_fps to target_fps."""
        if video_feature.shape[0] == 0:
            return video_feature

        source_fps = float(source_fps)
        target_fps = float(target_fps)
        if source_fps <= 0:
            raise ValueError(f"Invalid source_fps: {source_fps}")

        if abs(source_fps - target_fps) < 1e-6:
            return video_feature

        duration = video_feature.shape[0] / source_fps
        target_len = max(1, int(round(duration * target_fps)))
        # Map each target frame time to nearest source frame.
        src_indices = np.clip(
            np.round(np.arange(target_len) * source_fps / target_fps).astype(np.int64),
            0,
            video_feature.shape[0] - 1,
        )
        return video_feature[src_indices]

    @staticmethod
    def _map_scores_to_source_fps(
        scores_25fps: Sequence[float],
        source_frame_count: int,
        source_fps: float,
        target_fps: float = LR_ASD_FPS,
    ) -> List[float]:
        """Map per-frame scores at target_fps back to original frame indices."""
        if source_frame_count <= 0:
            return []
        if not scores_25fps:
            return [float("-inf")] * source_frame_count

        source_fps = float(source_fps)
        target_fps = float(target_fps)
        if abs(source_fps - target_fps) < 1e-6:
            mapped = list(scores_25fps[:source_frame_count])
            if len(mapped) < source_frame_count:
                mapped.extend([float("-inf")] * (source_frame_count - len(mapped)))
            return mapped

        mapped: List[float] = []
        last_idx = len(scores_25fps) - 1
        for frame_idx in range(source_frame_count):
            target_idx = int(round(frame_idx * target_fps / source_fps))
            target_idx = max(0, min(target_idx, last_idx))
            mapped.append(float(scores_25fps[target_idx]))
        return mapped

    def _score_av_clip(
        self,
        audio_feature: np.ndarray,
        video_feature_25fps: np.ndarray,
        audio_start: int,
        audio_end: int,
        video_start: int,
        video_end: int,
    ) -> List[float]:
        """Score one AV clip; audio/video lengths must stay at 4 MFCC frames per video frame."""
        if audio_end <= audio_start or video_end <= video_start:
            return []
        video_len = video_end - video_start
        audio_len = audio_end - audio_start
        # Keep LR-ASD's 100Hz MFCC : 25fps video = 4:1 alignment.
        usable_video = min(video_len, audio_len // 4)
        if usable_video <= 0:
            return []
        audio_end = audio_start + usable_video * 4
        video_end = video_start + usable_video

        input_a = torch.FloatTensor(
            audio_feature[audio_start:audio_end, :]
        ).unsqueeze(0).to(self.device)
        input_v = torch.FloatTensor(
            video_feature_25fps[video_start:video_end, :, :]
        ).unsqueeze(0).to(self.device)

        embed_a = self.model.forward_audio_frontend(input_a)
        embed_v = self.model.forward_visual_frontend(input_v)
        out = self.model.forward_audio_visual_backend(embed_a, embed_v)
        return list(self.loss_av.forward(out, labels=None))

    def _score_sequence_25fps(
        self,
        audio_feature: np.ndarray,
        video_feature_25fps: np.ndarray,
    ) -> List[float]:
        """Score AV sequence assuming video_feature is already at 25fps.

        Full windows are scored as usual. The remaining tail that is shorter than
        one window is scored as its own clip so the end of the video is covered.
        Multi-duration results are averaged with NaN-ignored means so incomplete
        scales do not poison the tail with -inf.
        """
        fps = LR_ASD_FPS
        audio_seconds = (audio_feature.shape[0] - audio_feature.shape[0] % 4) / MFCC_RATE
        video_seconds = video_feature_25fps.shape[0] / fps
        length = min(audio_seconds, video_seconds)
        full_video_len = video_feature_25fps.shape[0]
        if length <= 0:
            return [float("-inf")] * full_video_len

        audio_len = int(round(length * MFCC_RATE))
        video_len = int(round(length * fps))
        # Snap to 4:1 so clip scoring stays aligned.
        video_len = min(video_len, audio_len // 4, full_video_len)
        audio_len = video_len * 4
        audio_feature = audio_feature[:audio_len]
        video_feature_25fps = video_feature_25fps[:video_len]

        duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
        all_scores: List[List[float]] = []

        with torch.no_grad():
            for duration in duration_set:
                scores: List[float] = []
                batch_size = int(math.ceil(length / duration))
                for batch_idx in range(batch_size):
                    audio_start = int(batch_idx * duration * MFCC_RATE)
                    audio_end = int((batch_idx + 1) * duration * MFCC_RATE)
                    video_start = int(batch_idx * duration * fps)
                    video_end = int((batch_idx + 1) * duration * fps)

                    expected_video = int(duration * fps)
                    expected_audio = int(duration * MFCC_RATE)
                    full_window = (
                        audio_end <= audio_len
                        and video_end <= video_len
                        and (video_end - video_start) == expected_video
                        and (audio_end - audio_start) == expected_audio
                    )
                    if full_window:
                        batch_scores = self._score_av_clip(
                            audio_feature,
                            video_feature_25fps,
                            audio_start,
                            audio_end,
                            video_start,
                            video_end,
                        )
                        scores.extend(batch_scores)
                        continue

                    # Last incomplete window: score only the remaining tail once.
                    if video_start >= video_len or audio_start >= audio_len:
                        break
                    tail_scores = self._score_av_clip(
                        audio_feature,
                        video_feature_25fps,
                        audio_start,
                        audio_len,
                        video_start,
                        video_len,
                    )
                    scores.extend(tail_scores)
                    break

                if scores:
                    # Pad missing frames as NaN so they are ignored in nanmean.
                    if len(scores) < video_len:
                        scores = scores + [float("nan")] * (video_len - len(scores))
                    else:
                        scores = scores[:video_len]
                    all_scores.append(scores)

        if not all_scores:
            return [float("-inf")] * full_video_len

        score_arr = np.asarray(all_scores, dtype=np.float64)
        with np.errstate(all="ignore"):
            mean_scores = np.nanmean(score_arr, axis=0)
        mean_scores = np.where(np.isfinite(mean_scores), np.round(mean_scores, 1), np.NINF)
        mean_list = mean_scores.astype(float).tolist()

        if len(mean_list) < full_video_len:
            mean_list.extend([float("-inf")] * (full_video_len - len(mean_list)))
        return mean_list[:full_video_len]

    def compute_speaking_mask(
        self,
        audio_path: str,
        frame_list: Sequence[np.ndarray],
        coord_list: Sequence[Tuple[float, float, float, float]],
        fps: float,
    ) -> Tuple[List[bool], List[float]]:
        audio = self._load_audio_16k(audio_path)
        audio_feature = python_speech_features.mfcc(
            audio,
            16000,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )
        video_feature = self._extract_visual_features(frame_list, coord_list)
        source_fps = float(fps) if fps and fps > 0 else LR_ASD_FPS

        video_feature_25fps = self._resample_visual_to_target_fps(
            video_feature, source_fps, LR_ASD_FPS
        )
        scores_25fps = self._score_sequence_25fps(audio_feature, video_feature_25fps)
        raw_scores = self._map_scores_to_source_fps(
            scores_25fps,
            source_frame_count=len(coord_list),
            source_fps=source_fps,
            target_fps=LR_ASD_FPS,
        )

        speaking_mask = []
        smoothed_scores = []
        for frame_idx, bbox in enumerate(coord_list):
            if bbox == coord_placeholder:
                speaking_mask.append(False)
                smoothed_scores.append(float("-inf"))
                continue

            if frame_idx >= len(raw_scores):
                speaking_mask.append(False)
                smoothed_scores.append(float("-inf"))
                continue

            window = raw_scores[
                max(frame_idx - 2, 0) : min(frame_idx + 3, len(raw_scores))
            ]
            finite = [s for s in window if np.isfinite(s)]
            score = float(np.mean(finite)) if finite else float("-inf")
            smoothed_scores.append(score)
            speaking_mask.append(score >= self.threshold)

        return speaking_mask, smoothed_scores


class VSDLMDetector:
    """Visual-only mouth activity detector (lip-motion gate).

    Uses VSDLM ONNX on the lower face crop of the tracked face. A frame is
    marked speaking only when lip motion persists for a contiguous stretch
    (default 0.5s). A statically open mouth alone does **not** count. No audio
    is required.
    """

    def __init__(
        self,
        model_path: str,
        *,
        open_threshold: float = 0.15,
        activity_threshold: float = 0.12,
        activity_window: int = 4,
        min_speak_duration_sec: float = 0.5,
        # When landmark MAR says the mouth is open enough, also score a
        # face-band mouth crop and take max(open). Recovers tight landmark
        # crops that under-score slightly-open speech without a pure-MAR gate.
        mar_open_threshold: float = 0.12,
        # Kept for config compatibility; dual-crop rescue no longer uses it.
        mar_activity_threshold: float = 0.04,
        # Spread open peaks across ±radius frames (same shot). 0 = off.
        # Keep a small radius so sparse real peaks (e.g. ~35s / 1:43) survive
        # window mean; FP whole-shot fill is controlled by expand min-run.
        temporal_max_radius: int = 2,
        # Soft closed-mouth assist: when landmark MAR < this, linearly attenuate
        # open (mar=0 → open=0, mar=thr → unchanged). Kills VSDLM FPs on clearly
        # closed lips without the old hard veto at higher MAR (~0.10).
        soft_closed_mar: float = 0.06,
        batch_size: int = 64,
        providers: Sequence[str] | None = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VSDLM model not found: {model_path}")

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
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # NCHW: [N, 3, H, W] — H/W may be dynamic strings.
        self.input_h = int(shape[2]) if isinstance(shape[2], int) else 30
        self.input_w = int(shape[3]) if isinstance(shape[3], int) else 48
        # mean(open) in the activity window must also exceed this.
        self.open_threshold = float(open_threshold)
        self.activity_threshold = float(activity_threshold)
        self.activity_window = max(1, int(activity_window))
        self.min_speak_duration_sec = max(0.0, float(min_speak_duration_sec))
        self.mar_open_threshold = float(mar_open_threshold)
        self.mar_activity_threshold = float(mar_activity_threshold)
        self.temporal_max_radius = max(0, int(temporal_max_radius))
        self.soft_closed_mar = max(0.0, float(soft_closed_mar))
        self.batch_size = max(1, int(batch_size))

    @staticmethod
    def _mouth_crop_from_box(
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _mouth_crop_from_face(
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        mouth_bbox: Tuple[float, float, float, float] | None = None,
    ) -> np.ndarray | None:
        """Crop mouth for VSDLM.

        Prefer a tight lip box from landmarks when provided. Fallback uses the
        lower-central face band (not chin/collar), which avoids fur-collar
        false opens from the old 55%–100% crop.
        """
        if mouth_bbox is not None and mouth_bbox != coord_placeholder:
            crop = VSDLMDetector._mouth_crop_from_box(frame, mouth_bbox)
            if crop is not None and crop.size > 0:
                return crop

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        fh, fw = y2 - y1, x2 - x1
        # Typical mouth band inside a face box; excludes most collar/chin.
        my1 = y1 + int(fh * 0.62)
        my2 = y1 + int(fh * 0.92)
        mx1 = x1 + int(fw * 0.18)
        mx2 = x1 + int(fw * 0.82)
        my1, mx1 = max(0, my1), max(0, mx1)
        my2, mx2 = min(frame.shape[0], my2), min(frame.shape[1], mx2)
        if my2 <= my1 or mx2 <= mx1:
            return None
        return frame[my1:my2, mx1:mx2]

    @staticmethod
    def _invalidate_unstable_open_probs(
        open_probs: Sequence[float],
        coord_list: Sequence[Tuple[float, float, float, float]],
        shot_ids: Sequence[int] | None = None,
        *,
        cut_radius: int = 1,
        jump_iou: float = 0.35,
    ) -> List[float]:
        """Set open probs to NaN at shot cuts and hard face-box jumps."""
        from musetalk.utils.preprocessing import _bbox_iou

        n = len(open_probs)
        out = [float(v) for v in open_probs]
        cut_flags = [False] * n
        if shot_ids is not None:
            for i in range(1, n):
                if int(shot_ids[i]) != int(shot_ids[i - 1]):
                    for k in range(max(0, i - cut_radius), min(n, i + cut_radius + 1)):
                        cut_flags[k] = True
        for i in range(1, n):
            a, b = coord_list[i - 1], coord_list[i]
            if a == coord_placeholder or b == coord_placeholder:
                continue
            if _bbox_iou(a, b) < jump_iou:
                cut_flags[i - 1] = True
                cut_flags[i] = True
        for i, bad in enumerate(cut_flags):
            if bad:
                out[i] = float("nan")
        return out

    def _preprocess_crop(self, bgr_crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0
        return np.ascontiguousarray(resized.transpose(2, 0, 1))

    def _predict_open_batch(self, bgr_crops: Sequence[np.ndarray]) -> List[float]:
        """Score many mouth crops; model input batch dim is dynamic."""
        if not bgr_crops:
            return []
        probs: List[float] = []
        bs = self.batch_size
        for start in range(0, len(bgr_crops), bs):
            chunk = bgr_crops[start : start + bs]
            batch = np.stack([self._preprocess_crop(c) for c in chunk], axis=0)
            out = self.session.run(
                [self.output_name], {self.input_name: batch}
            )[0]
            flat = np.asarray(out).reshape(-1)
            for v in flat:
                probs.append(float(np.clip(float(v), 0.0, 1.0)))
        return probs

    def _predict_open(self, bgr_crop: np.ndarray) -> float:
        return self._predict_open_batch([bgr_crop])[0]

    @staticmethod
    def _keep_runs_at_least(
        mask: Sequence[bool],
        min_len: int,
        shot_ids: Sequence[int] | None = None,
    ) -> List[bool]:
        """Keep only contiguous True runs whose length is >= min_len frames.

        When ``shot_ids`` is provided, runs never cross shot boundaries: each
        shot is filtered independently so cross-cut True streaks cannot merge.
        """
        n = len(mask)
        if shot_ids is not None and len(shot_ids) != n:
            raise ValueError("shot_ids length must match mask")
        if min_len <= 1 and shot_ids is None:
            return list(mask)

        out = [False] * n
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i + 1
            while j < n and mask[j]:
                if shot_ids is not None and int(shot_ids[j]) != int(shot_ids[i]):
                    break
                j += 1
            if j - i >= min_len:
                out[i:j] = [True] * (j - i)
            i = j
        return out

    def score_open_probs(
        self,
        frame_list: Sequence[np.ndarray],
        coord_list: Sequence[Tuple[float, float, float, float]],
        *,
        mouth_coord_list: Sequence[Tuple[float, float, float, float] | None] | None = None,
        mouth_mar_list: Sequence[float | None] | None = None,
        closed_mouth_mar: float = 0.0,
        mar_open_threshold: float | None = None,
        shot_ids: Sequence[int] | None = None,
    ) -> List[float]:
        """Batched VSDLM open scores with dual-crop assist.

        MAR is a soft assist only (not a hard veto):
        - Always score landmark mouth crop when available, plus face-band crop,
          and take ``max`` (recovers tight-crop under-scores).
        - ``mar_open_threshold`` is kept for API compat; dual-crop no longer
          requires MAR >= threshold.
        - Near-zero MAR linearly attenuates open via ``soft_closed_mar``
          (applied after temporal max so closed frames stay suppressed).
        - Legacy hard clamp is **off by default** (``closed_mouth_mar <= 0``).
        """
        n = len(coord_list)
        if len(frame_list) != n:
            raise ValueError("frame_list length must match coord_list")
        if shot_ids is not None and len(shot_ids) != n:
            raise ValueError("shot_ids length must match coord_list")
        if mouth_coord_list is not None and len(mouth_coord_list) != n:
            raise ValueError("mouth_coord_list length must match coord_list")
        if mouth_mar_list is not None and len(mouth_mar_list) != n:
            raise ValueError("mouth_mar_list length must match coord_list")

        del mar_open_threshold  # dual-crop always on; retained for API compat

        open_probs: List[float] = [float("nan")] * n
        primary_idxs: List[int] = []
        primary_crops: List[np.ndarray] = []
        band_idxs: List[int] = []
        band_crops: List[np.ndarray] = []

        for i, (frame, bbox) in enumerate(zip(frame_list, coord_list)):
            if bbox == coord_placeholder:
                continue
            mouth_bb = mouth_coord_list[i] if mouth_coord_list is not None else None
            crop = self._mouth_crop_from_face(frame, bbox, mouth_bbox=mouth_bb)
            if crop is None or crop.size == 0:
                continue
            primary_idxs.append(i)
            primary_crops.append(crop)
            # Always also score face-band; take max later (MAR no longer gates this).
            band = self._mouth_crop_from_face(frame, bbox, mouth_bbox=None)
            if band is not None and band.size > 0:
                # Skip duplicate work when landmark crop is already the face-band.
                if mouth_bb is not None and mouth_bb != coord_placeholder:
                    band_idxs.append(i)
                    band_crops.append(band)

        for i, prob in zip(primary_idxs, self._predict_open_batch(primary_crops)):
            open_probs[i] = prob
        if band_idxs:
            for i, prob in zip(band_idxs, self._predict_open_batch(band_crops)):
                cur = open_probs[i]
                open_probs[i] = prob if not np.isfinite(cur) else max(cur, prob)

        # Optional legacy hard veto (disabled when closed_mouth_mar <= 0).
        if (
            mouth_mar_list is not None
            and closed_mouth_mar is not None
            and float(closed_mouth_mar) > 0
        ):
            thr = float(closed_mouth_mar)
            for i, mar in enumerate(mouth_mar_list):
                if mar is not None and mar < thr and np.isfinite(open_probs[i]):
                    open_probs[i] = min(open_probs[i], 0.05)

        open_probs = self._invalidate_unstable_open_probs(
            open_probs, coord_list, shot_ids=shot_ids
        )
        if self.temporal_max_radius > 0:
            open_probs = self._temporal_max_filter(
                open_probs, radius=self.temporal_max_radius, shot_ids=shot_ids
            )
        # Soft closed-mouth: attenuate after temporal max so mar≈0 frames
        # are not refilled by neighboring high-open spikes.
        return self._apply_soft_closed_mar(open_probs, mouth_mar_list)

    def _apply_soft_closed_mar(
        self,
        open_probs: Sequence[float],
        mouth_mar_list: Sequence[float | None] | None,
    ) -> List[float]:
        thr = float(self.soft_closed_mar)
        out = [float(v) for v in open_probs]
        if mouth_mar_list is None or thr <= 0:
            return out
        for i, mar in enumerate(mouth_mar_list):
            if mar is None or not np.isfinite(mar) or not np.isfinite(out[i]):
                continue
            if mar >= thr:
                continue
            scale = max(0.0, float(mar) / thr)
            out[i] = out[i] * scale
        return out

    @staticmethod
    def _temporal_max_filter(
        open_probs: Sequence[float],
        *,
        radius: int = 2,
        shot_ids: Sequence[int] | None = None,
    ) -> List[float]:
        n = len(open_probs)
        if radius <= 0:
            return list(open_probs)
        out = [float(v) for v in open_probs]
        for i in range(n):
            if not np.isfinite(open_probs[i]):
                continue
            cur_shot = int(shot_ids[i]) if shot_ids is not None else None
            best = float(open_probs[i])
            for j in range(max(0, i - radius), min(n, i + radius + 1)):
                if shot_ids is not None and int(shot_ids[j]) != cur_shot:
                    continue
                if np.isfinite(open_probs[j]):
                    best = max(best, float(open_probs[j]))
            out[i] = best
        return out

    def speaking_mask_from_open_probs(
        self,
        open_probs: Sequence[float],
        coord_list: Sequence[Tuple[float, float, float, float]],
        *,
        fps: float = LR_ASD_FPS,
        min_speak_duration_sec: float | None = None,
        identity_min_iou: float = 0.5,
        shot_ids: Sequence[int] | None = None,
    ) -> Tuple[List[bool], List[float]]:
        """Convert open probs → activity gate + min-run speaking mask."""
        from musetalk.utils.preprocessing import _bbox_iou

        n = len(coord_list)
        if len(open_probs) != n:
            raise ValueError("open_probs length must match coord_list")
        if shot_ids is not None and len(shot_ids) != n:
            raise ValueError("shot_ids length must match coord_list")

        win = self.activity_window
        # Mean-open uses the activity threshold (slightly softer than open_thr)
        # so a sharp open peak is not rejected solely because neighbors are low.
        mean_thr = self.activity_threshold
        motion_mask: List[bool] = []
        activity_scores: List[float] = []
        for i in range(n):
            if coord_list[i] == coord_placeholder or not np.isfinite(open_probs[i]):
                motion_mask.append(False)
                activity_scores.append(float("-inf"))
                continue
            cur = coord_list[i]
            cur_shot = int(shot_ids[i]) if shot_ids is not None else None
            finite = []
            for j in range(max(0, i - win), min(n, i + win + 1)):
                if shot_ids is not None and int(shot_ids[j]) != cur_shot:
                    continue
                if not np.isfinite(open_probs[j]):
                    continue
                if coord_list[j] == coord_placeholder:
                    continue
                if j != i and _bbox_iou(cur, coord_list[j]) < identity_min_iou:
                    continue
                finite.append(open_probs[j])
            if len(finite) < 3:
                activity = 0.0
                mean_open = 0.0
            else:
                activity = float(np.std(finite))
                mean_open = float(np.mean(finite))
            motion_mask.append(
                bool(
                    open_probs[i] >= self.open_threshold
                    and activity >= self.activity_threshold
                    and mean_open >= mean_thr
                )
            )
            activity_scores.append(activity)

        fps = float(fps) if fps and fps > 0 else LR_ASD_FPS
        duration = (
            self.min_speak_duration_sec
            if min_speak_duration_sec is None
            else max(0.0, float(min_speak_duration_sec))
        )
        min_frames = max(1, int(round(duration * fps))) if duration > 0 else 1
        speaking_mask = self._keep_runs_at_least(
            motion_mask, min_frames, shot_ids=shot_ids
        )
        return speaking_mask, activity_scores

    def compute_speaking_mask(
        self,
        frame_list: Sequence[np.ndarray],
        coord_list: Sequence[Tuple[float, float, float, float]],
        fps: float = LR_ASD_FPS,
        audio_path: str | None = None,
        min_speak_duration_sec: float | None = None,
        identity_min_iou: float = 0.5,
        shot_ids: Sequence[int] | None = None,
        mouth_coord_list: Sequence[Tuple[float, float, float, float] | None] | None = None,
        mouth_mar_list: Sequence[float | None] | None = None,
        closed_mouth_mar: float = 0.0,
        mar_open_threshold: float | None = None,
        mar_activity_threshold: float | None = None,
    ) -> Tuple[List[bool], List[float]]:
        """Per-frame lip-motion gate on a single bbox stream.

        A frame is motion-positive when, inside a short temporal window (same
        shot + identity IoU), all of:
          - current open >= ``open_threshold``
          - std(open) >= ``activity_threshold``
          - mean(open) >= ``open_threshold``

        Open scores use landmark mouth crop and face-band crop (take max).
        Landmark MAR is not used as a hard veto by default
        (``closed_mouth_mar <= 0``). Scores at shot cuts / hard bbox jumps
        are discarded.
        """
        del audio_path  # visual-only; kept for call-site compatibility
        del mar_activity_threshold  # retained for API compat

        open_probs = self.score_open_probs(
            frame_list,
            coord_list,
            mouth_coord_list=mouth_coord_list,
            mouth_mar_list=mouth_mar_list,
            closed_mouth_mar=closed_mouth_mar,
            mar_open_threshold=mar_open_threshold,
            shot_ids=shot_ids,
        )
        return self.speaking_mask_from_open_probs(
            open_probs,
            coord_list,
            fps=fps,
            min_speak_duration_sec=min_speak_duration_sec,
            identity_min_iou=identity_min_iou,
            shot_ids=shot_ids,
        )


def detect_shot_ids(
    frames: Sequence[np.ndarray],
    *,
    hist_threshold: float = 0.45,
    hist_soft_threshold: float = 0.30,
    gray_mae_threshold: float = 25.0,
    min_shot_len: int = 3,
    sample_short_side: int = 160,
) -> List[int]:
    """Assign a shot id to each frame via adjacent-frame cut detection.

    A cut is declared when either:
      1. HSV histogram Bhattacharyya distance >= ``hist_threshold``, or
      2. distance >= ``hist_soft_threshold`` **and** grayscale mean-abs-diff
         >= ``gray_mae_threshold`` (catches hard cuts that keep a similar
         color palette, e.g. palace interiors / same costume tones).

    Cuts closer than ``min_shot_len`` frames are ignored to reduce flicker
    false positives.
    """
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return [0]

    def _resize(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        short = min(h, w)
        if sample_short_side > 0 and short > sample_short_side:
            scale = float(sample_short_side) / float(short)
            return cv2.resize(
                frame,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _hist(frame_small: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    shot_ids = [0] * n
    current = 0
    last_cut = 0
    prev_small = _resize(frames[0])
    prev_hist = _hist(prev_small)
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    soft_thr = float(hist_soft_threshold) if hist_soft_threshold is not None else 0.0
    mae_thr = float(gray_mae_threshold) if gray_mae_threshold is not None else 0.0
    for i in range(1, n):
        small = _resize(frames[i])
        hist = _hist(small)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        mae = float(np.mean(np.abs(prev_gray - gray)))
        prev_hist = hist
        prev_gray = gray
        hard = dist >= float(hist_threshold)
        soft = (
            soft_thr > 0
            and mae_thr > 0
            and dist >= soft_thr
            and mae >= mae_thr
        )
        if (hard or soft) and (i - last_cut) >= min_shot_len:
            current += 1
            last_cut = i
        shot_ids[i] = current
    return shot_ids


def prune_short_speaking_runs(
    speaking_mask: Sequence[bool],
    *,
    min_duration_sec: float,
    fps: float,
    shot_ids: Sequence[int] | None = None,
) -> List[bool]:
    """Drop contiguous True runs shorter than ``min_duration_sec``.

    Intended to run on the raw speaking mask *before* dilation so brief
    false-positive blips are not expanded into longer false segments.
    """
    fps = float(fps) if fps and fps > 0 else LR_ASD_FPS
    min_len = (
        max(1, int(round(float(min_duration_sec) * fps)))
        if min_duration_sec and min_duration_sec > 0
        else 1
    )
    return VSDLMDetector._keep_runs_at_least(
        speaking_mask, min_len, shot_ids=shot_ids
    )


def dilate_speaking_mask(
    speaking_mask: Sequence[bool],
    radius: int,
    *,
    valid_face: Sequence[bool] | None = None,
    shot_ids: Sequence[int] | None = None,
) -> List[bool]:
    """Expand True regions by ``radius`` frames on both sides.

    Frames without a valid face stay False when ``valid_face`` is provided.
    When ``shot_ids`` is provided, dilation never crosses shot boundaries.
    """
    n = len(speaking_mask)
    if n == 0 or radius <= 0:
        return list(speaking_mask)

    # Precompute inclusive [start, end) range for each shot id for fast clamp.
    shot_range: dict[int, Tuple[int, int]] = {}
    if shot_ids is not None:
        if len(shot_ids) != n:
            raise ValueError("shot_ids length must match speaking_mask")
        i = 0
        while i < n:
            sid = int(shot_ids[i])
            j = i + 1
            while j < n and int(shot_ids[j]) == sid:
                j += 1
            shot_range[sid] = (i, j)
            i = j

    dilated = [False] * n
    for i, is_speaking in enumerate(speaking_mask):
        if not is_speaking:
            continue
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        if shot_ids is not None:
            s0, s1 = shot_range[int(shot_ids[i])]
            lo = max(lo, s0)
            hi = min(hi, s1)
        for j in range(lo, hi):
            if valid_face is not None and not valid_face[j]:
                continue
            dilated[j] = True
    return dilated


def expand_speaking_mask_to_shots(
    speaking_mask: Sequence[bool],
    shot_ids: Sequence[int],
    *,
    min_speak_duration_sec: float = 0.0,
    short_shot_sec: float = 0.0,
    short_shot_min_speak_sec: float | None = None,
    keep_partial_min_sec: float = 0.5,
    fps: float = LR_ASD_FPS,
) -> List[bool]:
    """Expand speaking to whole shots when a run is long enough.

    - Shots whose longest contiguous speaking run meets the expand bar are
      filled entirely (shot-level lipsync).
    - Shots that do **not** qualify keep their original speaking runs only if
      each run is at least ``keep_partial_min_sec`` (e.g. ~0.84s at 1:43 still
      gets lipsync without filling the whole cut). Shorter blips stay off.
    """
    n = len(speaking_mask)
    if n == 0:
        return []
    if len(shot_ids) != n:
        raise ValueError("shot_ids length must match speaking_mask")

    fps = float(fps) if fps and fps > 0 else LR_ASD_FPS
    default_min_len = (
        max(1, int(round(float(min_speak_duration_sec) * fps)))
        if min_speak_duration_sec and min_speak_duration_sec > 0
        else 1
    )
    use_short = (
        short_shot_sec is not None
        and float(short_shot_sec) > 0
        and short_shot_min_speak_sec is not None
        and float(short_shot_min_speak_sec) >= 0
    )
    short_max_frames = (
        max(1, int(round(float(short_shot_sec) * fps))) if use_short else 0
    )
    short_min_len = (
        max(1, int(round(float(short_shot_min_speak_sec) * fps)))
        if use_short and float(short_shot_min_speak_sec) > 0
        else 1
    )
    keep_partial_len = (
        max(1, int(round(float(keep_partial_min_sec) * fps)))
        if keep_partial_min_sec and keep_partial_min_sec > 0
        else 0
    )

    # Shot length (frames) and longest True run per shot.
    shot_len: dict[int, int] = {}
    longest: dict[int, int] = {}
    for i, sid_raw in enumerate(shot_ids):
        sid = int(sid_raw)
        shot_len[sid] = shot_len.get(sid, 0) + 1

    i = 0
    while i < n:
        if not speaking_mask[i]:
            i += 1
            continue
        sid = int(shot_ids[i])
        j = i + 1
        while (
            j < n
            and speaking_mask[j]
            and int(shot_ids[j]) == sid
        ):
            j += 1
        run = j - i
        if run > longest.get(sid, 0):
            longest[sid] = run
        i = j

    speak_shots: set[int] = set()
    for sid, run in longest.items():
        if use_short and shot_len.get(sid, 0) < short_max_frames:
            need = short_min_len
        else:
            need = default_min_len
        if run >= need:
            speak_shots.add(sid)

    out = [int(shot_ids[i]) in speak_shots for i in range(n)]

    # Non-expanded shots: keep original runs that are long enough for partial
    # lipsync (avoids wiping real speech that failed the whole-shot bar).
    if keep_partial_len > 0:
        i = 0
        while i < n:
            if not speaking_mask[i]:
                i += 1
                continue
            sid = int(shot_ids[i])
            j = i + 1
            while (
                j < n
                and speaking_mask[j]
                and int(shot_ids[j]) == sid
            ):
                j += 1
            if sid not in speak_shots and (j - i) >= keep_partial_len:
                for k in range(i, j):
                    out[k] = True
            i = j

    return out
