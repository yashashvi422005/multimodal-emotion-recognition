"""Phase 2 pilot: face detection (MediaPipe Face Mesh / FaceLandmarker,
RetinaFace fallback, no Haar cascades) and landmark-based synthetic
occlusion. Pilot-quality approximations — documented inline — not final
Phase 2 production code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import mediapipe as mp

# Standard MediaPipe Face Mesh (478-landmark topology) index groups.
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
MOUTH_OUTLINE = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
NOSE_BRIDGE_TOP = 168
CHIN = 152
JAW_LEFT = 234
JAW_RIGHT = 454
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
# Lower-face oval: right cheek (near ear) -> jaw -> chin -> jaw -> left cheek,
# a contiguous slice of the canonical face-oval contour, used as the mask's
# outer (cheek/jaw/chin) boundary.
LOWER_FACE_OVAL = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
SEVERITIES = [0.3, 0.6, 0.9]


def _angle_sorted_polygon(points: np.ndarray) -> list[tuple[float, float]]:
    """Order a point cloud into a simple (non-self-intersecting) polygon by
    sorting around its centroid by angle. Correct for star-shaped regions
    like a face mask outline (concave at the chin, but every boundary point
    is still visible from the centroid)."""
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    order = np.argsort(angles)
    return [tuple(p) for p in points[order]]


def clip_seed(clip_id: str, seed: int) -> int:
    """Deterministic per-clip RNG seed: same clip_id+seed always gives the
    same occlusion placement/parameters, but different clips don't share
    a trivially-correlated sequence."""
    h = hashlib.sha256(f"{clip_id}_{seed}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class DetectionResult:
    success: bool
    method: str | None  # "mediapipe" | "retinaface" | None
    bbox: tuple[int, int, int, int] | None  # x1,y1,x2,y2 in frame pixel coords
    landmarks_478: np.ndarray | None  # (478, 2) pixel coords, or None
    landmarks_5: dict | None  # RetinaFace 5-point landmarks, pixel coords, or None


class FaceDetector:
    def __init__(self, mediapipe_model_path: Path):
        base_options = mp.tasks.BaseOptions(model_asset_path=str(mediapipe_model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        self._mp_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._retinaface = None  # lazy import/init — heavy (torch) and only needed on fallback

    def detect_mediapipe_only(self, frame_rgb: np.ndarray) -> DetectionResult:
        """No RetinaFace fallback — used for the occlusion-robustness
        measurement, which specifically asks whether MediaPipe (the primary
        detector) still finds a face on an occluded crop."""
        result = self._detect_mediapipe(frame_rgb)
        if result is not None:
            return result
        return DetectionResult(success=False, method=None, bbox=None, landmarks_478=None, landmarks_5=None)

    def _detect_mediapipe(self, frame_rgb: np.ndarray) -> DetectionResult | None:
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._mp_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return DetectionResult(
            success=True, method="mediapipe",
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            landmarks_478=pts, landmarks_5=None,
        )

    def _detect_retinaface(self, frame_rgb: np.ndarray) -> DetectionResult | None:
        if self._retinaface is None:
            from retinaface import RetinaFace
            self._retinaface = RetinaFace
        faces = self._retinaface.detect_faces(frame_rgb)
        if not isinstance(faces, dict) or not faces:
            return None
        best = max(faces.values(), key=lambda f: f.get("score", 0))
        x1, y1, x2, y2 = best["facial_area"]
        lm5 = {k: tuple(v) for k, v in best["landmarks"].items()}
        return DetectionResult(success=True, method="retinaface", bbox=(x1, y1, x2, y2), landmarks_478=None, landmarks_5=lm5)

    def detect(self, frame_rgb: np.ndarray) -> DetectionResult:
        result = self._detect_mediapipe(frame_rgb)
        if result is not None:
            return result
        result = self._detect_retinaface(frame_rgb)
        if result is not None:
            return result
        return DetectionResult(success=False, method=None, bbox=None, landmarks_478=None, landmarks_5=None)


def crop_face(frame_rgb: np.ndarray, bbox: tuple[int, int, int, int], out_size: int = 224, margin: float = 0.3):
    """Crop with a margin around bbox, resize to out_size x out_size, and
    return (crop, transform) where transform maps original-frame pixel
    coords -> crop-local pixel coords, for remapping landmarks."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    crop = frame_rgb[cy1:cy2, cx1:cx2]
    scale_x = out_size / max(1, (cx2 - cx1))
    scale_y = out_size / max(1, (cy2 - cy1))
    crop_resized = np.array(Image.fromarray(crop).resize((out_size, out_size)))

    def transform(pts: np.ndarray) -> np.ndarray:
        out = pts.copy()
        out[:, 0] = (out[:, 0] - cx1) * scale_x
        out[:, 1] = (out[:, 1] - cy1) * scale_y
        return out

    return crop_resized, transform


def _avg_color(img: np.ndarray, region_frac=(0.15, 0.5, 0.35, 0.65)) -> tuple[int, int, int]:
    h, w = img.shape[:2]
    y1, y2 = int(h * region_frac[0]), int(h * region_frac[1])
    x1, x2 = int(w * region_frac[2]), int(w * region_frac[3])
    patch = img[y1:y2, x1:x2].reshape(-1, 3)
    return tuple(int(v) for v in patch.mean(axis=0))


# Mask severity is defined by VERTICAL EXTENT from real landmark groups,
# not a continuous scale factor (see fix in the Phase 4 diagnostics round):
#   0.3 -> mouth and chin only (top edge at the upper lip)
#   0.6 -> up to the nose bridge midpoint (top edge at landmark 168)
#   0.9 -> up to just below the eyes (top edge at each eye's lowest point)
# The bottom/side boundary (cheek -> jaw -> chin) is always the same
# LOWER_FACE_OVAL contour; severity only moves the top edge up the face.
_MASK_SEVERITY_LEVELS = {0.3: "mouth_chin", 0.6: "nose_bridge", 0.9: "below_eyes"}
# Upper-lip subset of MOUTH_OUTLINE (the list interleaves upper then lower lip).
_MOUTH_TOP = MOUTH_OUTLINE[:10]


def _mask_top_points(landmarks_478: np.ndarray, level: str) -> np.ndarray:
    if level == "mouth_chin":
        return landmarks_478[_MOUTH_TOP]
    if level == "nose_bridge":
        return landmarks_478[[LEFT_EYE_INNER, NOSE_BRIDGE_TOP, RIGHT_EYE_INNER]]
    if level == "below_eyes":
        left_eye, right_eye = landmarks_478[LEFT_EYE], landmarks_478[RIGHT_EYE]
        return np.array([left_eye[np.argmax(left_eye[:, 1])], right_eye[np.argmax(right_eye[:, 1])]])
    raise ValueError(f"unknown mask severity level {level!r}")


def apply_mask(crop_rgb: np.ndarray, landmarks_478: np.ndarray | None, severity: float) -> np.ndarray:
    """Surgical-mask-style occlusion, fully opaque: a landmark polygon whose
    outer boundary is always the lower-face oval (cheek -> jaw -> chin ->
    jaw -> cheek), and whose top edge moves up the face with severity (see
    `_MASK_SEVERITY_LEVELS`). Falls back to a generic lower-face rectangle,
    with the same three top-edge heights, if no dense landmarks are
    available (RetinaFace-only fallback)."""
    img = Image.fromarray(crop_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = crop_rgb.shape[:2]
    if severity not in _MASK_SEVERITY_LEVELS:
        raise ValueError(f"mask severity must be one of {sorted(_MASK_SEVERITY_LEVELS)}, got {severity}")
    level = _MASK_SEVERITY_LEVELS[severity]
    if landmarks_478 is not None:
        oval_pts = landmarks_478[LOWER_FACE_OVAL]
        top_pts = _mask_top_points(landmarks_478, level)
        all_pts = np.vstack([oval_pts, top_pts])
        polygon = _angle_sorted_polygon(all_pts)
    else:
        top_y = {"mouth_chin": h * 0.62, "nose_bridge": h * 0.45, "below_eyes": h * 0.35}[level]
        bottom_y = h * 0.92
        left_x, right_x = w * 0.15, w * 0.85
        polygon = [(left_x, top_y), (right_x, top_y), (right_x * 1.02, bottom_y), (left_x * 0.98, bottom_y)]
    draw.polygon(polygon, fill=(180, 200, 215, 255))
    return np.array(img)


def apply_sunglasses(crop_rgb: np.ndarray, landmarks_478: np.ndarray | None, severity: float) -> np.ndarray:
    img = Image.fromarray(crop_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = crop_rgb.shape[:2]
    if landmarks_478 is not None:
        eye_pts = landmarks_478[LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW]
        x1, y1 = eye_pts.min(axis=0)
        x2, y2 = eye_pts.max(axis=0)
    else:
        x1, y1, x2, y2 = w * 0.15, h * 0.25, w * 0.85, h * 0.5
    pad_x, pad_y = (x2 - x1) * 0.15 * severity, (y2 - y1) * 0.6 * severity
    box = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    draw.ellipse(box, fill=(20, 20, 20, 255))
    return np.array(img)


def apply_patch(crop_rgb: np.ndarray, landmarks_478: np.ndarray | None, severity: float, rng: np.random.Generator) -> np.ndarray:
    img = Image.fromarray(crop_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = crop_rgb.shape[:2]
    size = int(min(h, w) * (0.25 + 0.4 * severity))
    cx = rng.integers(size // 2, w - size // 2) if w > size else w // 2
    cy = rng.integers(size // 2, h - size // 2) if h > size else h // 2
    gray = int(rng.integers(60, 160))
    draw.rectangle([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2], fill=(gray, gray, gray, 255))
    return np.array(img)


def apply_hand_over_mouth(crop_rgb: np.ndarray, landmarks_478: np.ndarray | None, severity: float, rng: np.random.Generator) -> np.ndarray:
    img = Image.fromarray(crop_rgb.copy())
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = crop_rgb.shape[:2]
    skin_color = _avg_color(crop_rgb)
    if landmarks_478 is not None:
        mouth_pts = landmarks_478[MOUTH_OUTLINE]
        cx, cy = mouth_pts.mean(axis=0)
    else:
        cx, cy = w * 0.5, h * 0.7
    jitter = rng.uniform(-0.05, 0.05, size=2) * [w, h]
    cx, cy = cx + jitter[0], cy + jitter[1]
    rx, ry = w * (0.22 + 0.15 * severity), h * (0.18 + 0.15 * severity)
    draw.ellipse([cx - rx, cy - ry * 0.6, cx + rx, cy + ry], fill=skin_color + (255,))
    return np.array(img)


OCCLUSION_FNS = {
    "mask": apply_mask,
    "sunglasses": apply_sunglasses,
    "patch": apply_patch,
    "hand_over_mouth": apply_hand_over_mouth,
}


def apply_occlusion(name: str, crop_rgb: np.ndarray, landmarks_478: np.ndarray | None, severity: float, clip_id: str, seed: int) -> np.ndarray:
    fn = OCCLUSION_FNS[name]
    if name in ("patch", "hand_over_mouth"):
        rng = np.random.default_rng(clip_seed(f"{clip_id}_{name}", seed))
        return fn(crop_rgb, landmarks_478, severity, rng)
    return fn(crop_rgb, landmarks_478, severity)
