import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (100, 100)
HAAR_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"
HAAR_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"
LBF_MODEL_PATH = "lbfmodel.yaml"
_FACEMARK_LBF = None


def _get_facemark():
    global _FACEMARK_LBF
    if _FACEMARK_LBF is None:
        if hasattr(cv2, 'face') and os.path.exists(LBF_MODEL_PATH):
            fm = cv2.face.createFacemarkLBF()
            fm.loadModel(LBF_MODEL_PATH)
            _FACEMARK_LBF = fm
    return _FACEMARK_LBF


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)


def load_image_from_pil(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def detect_face(gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    for cascade_path in [HAAR_FRONTAL, HAAR_PROFILE]:
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) > 0:
            return tuple(sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0])
    return None


def align_face_lbf(gray_image: np.ndarray, bbox: Tuple[int, int, int, int], target_size: Tuple[int, int] = TARGET_SIZE) -> Tuple[np.ndarray, bool]:
    """
    SDM (Supervised Descent Method) LBF Alignment.
    Memutar, menyesuaikan skala, dan mentranslasi wajah sehingga mata selalu jatuh pada koordinat piksel yang absolut.
    """
    facemark = _get_facemark()
    if facemark is None:
        return gray_image, False

    x, y, w, h = bbox
    ok, landmarks = facemark.fit(gray_image, np.array([[x, y, w, h]]))
    if not ok or len(landmarks) == 0:
        return gray_image, False
        
    pts = landmarks[0][0]
    left_eye = np.mean(pts[36:42], axis=0)
    right_eye = np.mean(pts[42:48], axis=0)
    
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    dist = np.sqrt(dx**2 + dy**2)
    # Jarak antar mata diset konstan 40% dari lebar gambar akhir
    desired_dist = target_size[0] * 0.40
    scale = desired_dist / max(dist, 1.0)
    
    eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    
    # Translasi: geser pusat mata ke koordinat X=50%, Y=35% dari gambar akhir
    t_x = target_size[0] * 0.50
    t_y = target_size[1] * 0.35
    M[0, 2] += (t_x - eye_center[0])
    M[1, 2] += (t_y - eye_center[1])
    
    aligned_face = cv2.warpAffine(gray_image, M, target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned_face, True


def detect_eyes_and_angle(gray_crop: np.ndarray) -> Tuple[float, bool]:
    eye_cascade = cv2.CascadeClassifier(HAAR_EYE)
    eyes = eye_cascade.detectMultiScale(
        gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
    )

    if len(eyes) == 2:
        # Sort by x coordinate (left eye first)
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]

        # Calculate centers
        cx1, cy1 = ex1 + ew1 // 2, ey1 + eh1 // 2
        cx2, cy2 = ex2 + ew2 // 2, ey2 + eh2 // 2

        dy = cy2 - cy1
        dx = cx2 - cx1
        angle = np.degrees(np.arctan2(dy, dx))
        # Valid angles shouldn't be upside down, clamp it reasonably (-30 to 30)
        if -30.0 <= angle <= 30.0:
            return angle, True
    return 0.0, False


def crop_face(
    gray_image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = -0.08
) -> np.ndarray:
    x, y, w, h = bbox
    H, W = gray_image.shape
    # padding negatif itu maksudnya kita memotong ke dalam (inner crop)
    # Ini disebut "Inner-Face Cropping" untuk membuang background dan rambut, jadi yang nampak hanyalah alis mata, mata, hidung, dan mulut
    x1 = max(0, x - int(w * padding))
    y1 = max(0, y - int(h * padding))
    x2 = min(W, x + w + int(w * padding))
    y2 = min(H, y + h + int(h * padding))
    return gray_image[y1:y2, x1:x2]


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_elliptical_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.35), int(h * 0.45))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess_face(
    gray_image: np.ndarray,
    detect: bool = True,
    target_size: Tuple[int, int] = TARGET_SIZE,
    blur: bool = False,
    force_angle: Optional[float] = None,
    pre_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, dict]:
    info = {
        "face_detected": False,
        "bbox": None,
        "eye_aligned": False,
        "angle_used": 0.0,
        "steps": {"original_gray": gray_image.copy()},
    }
    face_crop = gray_image

    if detect:
        bbox = pre_bbox if pre_bbox is not None else detect_face(gray_image)
        if bbox is not None:
            info["face_detected"] = True
            info["bbox"] = bbox
            
            aligned_face, success = align_face_lbf(gray_image, bbox, target_size)
            if success:
                face_crop = aligned_face
                info["steps"]["crop"] = face_crop.copy()
                info["steps"]["aligned"] = face_crop.copy()
                info["eye_aligned"] = True
            else:
                face_crop = crop_face(gray_image, bbox)
                info["steps"]["crop"] = face_crop.copy()
                info["steps"]["aligned"] = face_crop.copy()

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    face_crop = clahe.apply(face_crop)
    info["steps"]["equalized"] = face_crop.copy()

    if blur:
        face_crop = apply_gaussian_blur(face_crop)

    if face_crop.shape[:2] != target_size:
        resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized = face_crop

    info["steps"]["final"] = resized.copy()

    normalized = resized.astype(np.float64) / 255.0
    return normalized, info


def draw_face_box(
    original_image: np.ndarray, bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    display = (
        cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        if original_image.ndim == 2
        else original_image.copy()
    )
    x, y, w, h = bbox
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 100), 2)
    return display
