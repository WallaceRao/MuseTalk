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


def dilate_speaking_mask(
    speaking_mask: Sequence[bool],
    radius: int,
    *,
    valid_face: Sequence[bool] | None = None,
) -> List[bool]:
    """Expand True regions by ``radius`` frames on both sides.

    Frames without a valid face stay False when ``valid_face`` is provided.
    """
    n = len(speaking_mask)
    if n == 0 or radius <= 0:
        return list(speaking_mask)

    dilated = [False] * n
    for i, is_speaking in enumerate(speaking_mask):
        if not is_speaking:
            continue
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        for j in range(lo, hi):
            if valid_face is not None and not valid_face[j]:
                continue
            dilated[j] = True
    return dilated
