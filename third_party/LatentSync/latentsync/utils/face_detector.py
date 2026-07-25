from insightface.app import FaceAnalysis
import cv2
import numpy as np
import torch
from insightface.utils import face_align

INSIGHTFACE_DETECT_SIZE = 512


def _softmax2(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.maximum(e.sum(), 1e-12)


class FaceDetector:
    def __init__(self, device="cuda"):
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106", "genderage"],
            root="checkpoints/auxiliary",
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))
        self._patch_genderage_scores()

    def _patch_genderage_scores(self):
        """Attach gender softmax scores onto each Face during genderage inference.

        Upstream InsightFace only stores ``argmax(pred[:2])``; we keep that
        behavior and additionally set ``gender_scores`` / ``gender_confidence``.
        """
        models = getattr(self.app, "models", None)
        if not models or "genderage" not in models:
            return
        ga = models["genderage"]
        if getattr(ga, "_musetalk_scores_patched", False):
            return

        def get_with_scores(img, face):
            bbox = face.bbox
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            _scale = ga.input_size[0] / (max(w, h) * 1.5)
            aimg, _M = face_align.transform(img, center, ga.input_size[0], _scale, 0)
            input_size = tuple(aimg.shape[0:2][::-1])
            blob = cv2.dnn.blobFromImage(
                aimg,
                1.0 / ga.input_std,
                input_size,
                (ga.input_mean, ga.input_mean, ga.input_mean),
                swapRB=True,
            )
            pred = ga.session.run(ga.output_names, {ga.input_name: blob})[0][0]
            if ga.taskname == "genderage":
                assert len(pred) == 3
                scores = _softmax2(pred[:2])
                gender = int(np.argmax(scores))
                age = int(np.round(pred[2] * 100))
                face["gender"] = gender
                face["age"] = age
                face["gender_scores"] = scores.astype(np.float32)  # [female, male]
                face["gender_confidence"] = float(scores[gender])
                return gender, age
            return pred

        ga.get = get_with_scores
        ga._musetalk_scores_patched = True

    def _select_primary_face(self, frame, threshold=0.5):
        """Return the largest eligible InsightFace face, or None."""
        faces = self.app.get(frame)
        get_face_store = None
        max_size = 0
        for face in faces:
            bbox = face.bbox.astype(np.int_).tolist()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < 50 or h < 80:
                continue
            if w / h > 1.5 or w / h < 0.2:
                continue
            if face.det_score < threshold:
                continue
            size_now = w * h
            if size_now > max_size:
                max_size = size_now
                get_face_store = face
        return get_face_store

    def detect_primary_det_bbox(self, frame, threshold=0.5):
        """Raw InsightFace detection bbox (xyxy) of the largest eligible face."""
        face = self._select_primary_face(frame, threshold=threshold)
        if face is None:
            return None
        return face.bbox.astype(np.int_).tolist()

    @staticmethod
    def _gender_label_from_face(face):
        """Map InsightFace face attrs to ``male`` / ``female`` / None."""
        if face is None:
            return None
        # InsightFace genderage: 0 = female, 1 = male.
        gender = getattr(face, "gender", None)
        if gender is None:
            sex = getattr(face, "sex", None)
            if sex in ("M", "m"):
                return "male"
            if sex in ("F", "f"):
                return "female"
            return None
        try:
            g = int(gender)
        except (TypeError, ValueError):
            return None
        if g == 1:
            return "male"
        if g == 0:
            return "female"
        return None

    def detect_gender(self, frame, threshold=0.5):
        """Return ``\"male\"`` / ``\"female\"`` for the primary face, or None."""
        face = self._select_primary_face(frame, threshold=threshold)
        return self._gender_label_from_face(face)

    def detect_gender_with_confidence(self, frame, threshold=0.5):
        """Return ``(gender, confidence)`` for the primary face, or ``(None, None)``.

        ``confidence`` is softmax(pred[:2]) of the predicted class in ``[0, 1]``.
        """
        face = self._select_primary_face(frame, threshold=threshold)
        label = self._gender_label_from_face(face)
        if label is None:
            return None, None
        conf = getattr(face, "gender_confidence", None)
        if conf is None:
            scores = getattr(face, "gender_scores", None)
            if scores is not None and len(scores) >= 2:
                conf = float(scores[1] if label == "male" else scores[0])
        if conf is None:
            return label, None
        return label, float(conf)

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape

        get_face_store = self._select_primary_face(frame, threshold=threshold)

        if get_face_store is None:
            return None, None
        else:
            face = get_face_store
            lmk = np.round(face.landmark_2d_106).astype(np.int_)

            halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)  # lmk[73]

            sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
            halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
            upper_bond = halk_face_coord[1] - halk_face_dist  # *0.94

            x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

            y2 += int((x2 - x1) * 0.1)
            x1 -= int((x2 - x1) * 0.05)
            x2 += int((x2 - x1) * 0.05)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)

            return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
