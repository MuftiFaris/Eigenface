import os
import cv2
from PIL import Image
import numpy as np

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

FIXED_SIZE = (100, 100)

def detect_and_crop_pil(pil_img):
    """Detect wajah terbesar di image, lalu crop."""

    cv_img = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return pil_img
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    return pil_img.crop((x, y, x+w, y+h))

def load_dataset(folder):
    """
    Load gambar dari folder:
    - Detect & crop wajah
    - Resize ke FIXED_SIZE
    - Convert ke grayscale → flatten
    Menghasilkan A (M×N), list path, dan FIXED_SIZE.
    """
    vectors, paths = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, fname)
            try:
                pil = Image.open(path)
            except Exception:
                continue
            pil = detect_and_crop_pil(pil)
            gray = pil.convert('L').resize(FIXED_SIZE)
            vec = np.asarray(gray, np.float32).flatten()
            vectors.append(vec)
            paths.append(path)

    if not vectors:
        raise ValueError(f"Tidak ada gambar valid di folder: {folder}")

    A = np.stack(vectors, axis=1)
    return A, paths, FIXED_SIZE

def process_input_image(img_file, mean_face, eigenfaces, img_size):
    """
    Proses 1 citra input:
    - Detect & crop wajah
    - Resize ke FIXED_SIZE
    - Convert ke grayscale, flatten
    - Center dengan mean_face
    - Projection ke eigenfaces → weights
    """
    pil = Image.open(img_file)
    pil = detect_and_crop_pil(pil)
    gray = pil.convert('L').resize(FIXED_SIZE)
    vec = np.asarray(gray, np.float32).flatten().reshape(-1,1)
    centered = vec - mean_face
    weights = eigenfaces.T @ centered
    return vec, weights
