"""
compute_eigen_aging.py — Extract per-person aging deltas from FG-NET.

Output: eigen_aging_vectors.npz  (unwhitened PCA space)
  - fgnet_child_scores_{pix,lbp,hog}: (n_pairs, 100) child face scores
  - fgnet_deltas_{pix,lbp,hog}:       (n_pairs, 100) adult - child delta
  - eigen_aging_{pix,lbp,hog}:        (k, 100) PCA components on deltas
  - eigen_aging_mean_{pix,lbp,hog}:   (100,) mean delta
  - n_pairs: number of valid pairs
"""

import os
import re
import sys
import warnings
import zipfile

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from streamlit_app.core.pca_svd import project_to_eigenspace

NPZ_EIGEN = "pretrained_eigenspace.npz"
NPZ_OUT = "eigen_aging_vectors.npz"
FGNET_ZIP = "FGNET.zip"
N_COMPONENTS = 100


def load_pretrained():
    data = np.load(NPZ_EIGEN)
    return {
        "pix": {"mean": data["mean_face"], "eigenfaces": data["eigenfaces"]},
        "lbp": {"mean": data["mean_lbp"], "eigenfaces": data["eigenfaces_lbp"]},
        "hog": {"mean": data["mean_hog"], "eigenfaces": data["eigenfaces_hog"]},
        "n_samples": int(data["n_samples"].item() if data["n_samples"].ndim == 0 else data["n_samples"][0]),
    }


def extract_features(img_2d):
    img_uint8 = (np.clip(img_2d, 0, 1) * 255).astype(np.uint8)
    pix = img_2d.flatten().astype(float)
    lbp = local_binary_pattern(img_uint8, P=8, R=1, method="uniform")
    lbp_flat = lbp.flatten().astype(np.float64) / (lbp.max() + 1e-8)
    h = hog(
        img_uint8,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    ).astype(np.float32)
    return pix, lbp_flat, h


def align_face(img, facemark, face_cascade, target_size=(100, 100)):
    faces = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(30, 30))
    if len(faces) > 0:
        bbox = faces[0]
        ok, landmarks = facemark.fit(img, np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]]))
        if ok and len(landmarks) > 0:
            pts = landmarks[0][0]
            left_eye = np.mean(pts[36:42], axis=0)
            right_eye = np.mean(pts[42:48], axis=0)
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            dist = np.sqrt(dx**2 + dy**2)
            desired_dist = target_size[0] * 0.40
            scale = desired_dist / max(dist, 1.0)
            eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
            M = cv2.getRotationMatrix2D(eye_center, angle, scale)
            t_x = target_size[0] * 0.50
            t_y = target_size[1] * 0.35
            M[0, 2] += (t_x - eye_center[0])
            M[1, 2] += (t_y - eye_center[1])
            return cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x, y, w, h_ = bbox
        pad_x, pad_y = int(w * 0.1), int(h_ * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h_ + pad_y)
        img = img[y1:y2, x1:x2]
    return cv2.resize(img, target_size)


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  Eigen-Aging Reference Coding (FG-NET)")
    print("=" * 60)

    if not os.path.exists(NPZ_EIGEN):
        print(f"  ERROR: {NPZ_EIGEN} not found. Run colab_pca_evaluation.py first.")
        sys.exit(1)
    if not os.path.exists(FGNET_ZIP):
        print(f"  ERROR: {FGNET_ZIP} not found.")
        sys.exit(1)

    pc = load_pretrained()

    # ---- Init face detection + alignment ----
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    facemark = cv2.face.createFacemarkLBF()
    lbf_path = "lbfmodel.yaml"
    if not os.path.exists(lbf_path):
        print("  Downloading lbfmodel.yaml ...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
            lbf_path,
        )
    facemark.loadModel(lbf_path)

    # ---- Process FG-NET ----
    print("\n  1. Processing FG-NET images...")
    person_photos = {}  # pid -> [(age, pix_scores, lbp_scores, hog_scores)]
    total_images = 0

    with zipfile.ZipFile(FGNET_ZIP, "r") as zf:
        jpg_files = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
        print(f"     Found {len(jpg_files)} images in FG-NET")

        for f in jpg_files:
            basename = os.path.basename(f)
            m = re.match(r"(\d+)[aA](\d+)[a-zA-Z]?\.JPG", basename)
            if not m:
                continue
            pid = int(m.group(1))
            age = int(m.group(2))
            if age <= 0:
                continue

            file_data = zf.read(f)
            img_array = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            aligned = align_face(img, facemark, face_cascade)
            aligned_float = aligned.astype(np.float32) / 255.0
            pix, lbp_f, hog_f = extract_features(aligned_float)

            # Project to PCA spaces
            w_pix = project_to_eigenspace(pix, pc["pix"]["eigenfaces"], pc["pix"]["mean"])
            w_lbp = project_to_eigenspace(lbp_f, pc["lbp"]["eigenfaces"], pc["lbp"]["mean"])
            w_hog = project_to_eigenspace(hog_f, pc["hog"]["eigenfaces"], pc["hog"]["mean"])

            if pid not in person_photos:
                person_photos[pid] = []
            person_photos[pid].append((age, w_pix, w_lbp, w_hog))
            total_images += 1

    print(f"     Processed {total_images} images from {len(person_photos)} people")

    # ---- Compute per-person deltas ----
    print("\n  2. Computing aging deltas per person...")
    child_scores_pix, child_scores_lbp, child_scores_hog = [], [], []
    deltas_pix, deltas_lbp, deltas_hog = [], [], []
    valid_pairs = 0

    for pid, photos in person_photos.items():
        ages = np.array([p[0] for p in photos])
        child_mask = (ages > 0) & (ages <= 12)
        adult_mask = ages >= 18
        if not np.any(child_mask) or not np.any(adult_mask):
            continue

        child_idx = np.where(child_mask)[0]
        adult_idx = np.where(adult_mask)[0]

        # mean child score
        c_pix = np.mean([photos[i][1] for i in child_idx], axis=0)
        c_lbp = np.mean([photos[i][2] for i in child_idx], axis=0)
        c_hog = np.mean([photos[i][3] for i in child_idx], axis=0)

        # mean adult score
        a_pix = np.mean([photos[i][1] for i in adult_idx], axis=0)
        a_lbp = np.mean([photos[i][2] for i in adult_idx], axis=0)
        a_hog = np.mean([photos[i][3] for i in adult_idx], axis=0)

        child_scores_pix.append(c_pix)
        child_scores_lbp.append(c_lbp)
        child_scores_hog.append(c_hog)
        deltas_pix.append(a_pix - c_pix)
        deltas_lbp.append(a_lbp - c_lbp)
        deltas_hog.append(a_hog - c_hog)
        valid_pairs += 1

    child_scores_pix = np.array(child_scores_pix)
    child_scores_lbp = np.array(child_scores_lbp)
    child_scores_hog = np.array(child_scores_hog)
    deltas_pix = np.array(deltas_pix)
    deltas_lbp = np.array(deltas_lbp)
    deltas_hog = np.array(deltas_hog)

    print(f"     Valid pairs (child ≤12, adult ≥18): {valid_pairs}")

    # ---- Verify delta quality ----
    print("\n  3. Delta statistics:")
    for name, d in [("pix", deltas_pix), ("lbp", deltas_lbp), ("hog", deltas_hog)]:
        norms = np.linalg.norm(d, axis=1)
        print(f"     {name}: mean norm={np.mean(norms):.3f}, std={np.std(norms):.3f}, "
              f"min={np.min(norms):.3f}, max={np.max(norms):.3f}")

    # ---- PCA on deltas (eigen-aging) ----
    print("\n  4. Computing eigen-aging PCA on deltas...")
    from sklearn.decomposition import PCA

    n_aging = min(valid_pairs, N_COMPONENTS)
    ea_results = {}
    for name, d in [("pix", deltas_pix), ("lbp", deltas_lbp), ("hog", deltas_hog)]:
        mean_d = np.mean(d, axis=0)
        d_centered = d - mean_d
        if valid_pairs >= 2:
            pca_a = PCA(n_components=min(n_aging, valid_pairs - 1))
            pca_a.fit(d)
            ea_results[name] = {
                "components": pca_a.components_,
                "mean": mean_d,
                "ev_ratio": pca_a.explained_variance_ratio_,
                "singular_values": pca_a.singular_values_,
            }
            var_pct = np.sum(pca_a.explained_variance_ratio_) * 100
            print(f"     {name}: {len(pca_a.components_)} components, "
                  f"variance={var_pct:.1f}%")
        else:
            ea_results[name] = {
                "components": mean_d.reshape(1, -1),
                "mean": mean_d,
                "ev_ratio": np.array([1.0]),
                "singular_values": np.array([np.linalg.norm(d)]),
            }
            print(f"     {name}: too few pairs, using single delta (fallback)")

    # ---- Save ----
    print(f"\n  5. Saving to {NPZ_OUT}...")
    np.savez_compressed(
        NPZ_OUT,
        # NN-weighted deltas
        fgnet_child_scores_pix=child_scores_pix,
        fgnet_child_scores_lbp=child_scores_lbp,
        fgnet_child_scores_hog=child_scores_hog,
        fgnet_deltas_pix=deltas_pix,
        fgnet_deltas_lbp=deltas_lbp,
        fgnet_deltas_hog=deltas_hog,
        # Eigen-aging PCA
        eigen_aging_pix=ea_results["pix"]["components"],
        eigen_aging_lbp=ea_results["lbp"]["components"],
        eigen_aging_hog=ea_results["hog"]["components"],
        eigen_aging_mean_pix=ea_results["pix"]["mean"],
        eigen_aging_mean_lbp=ea_results["lbp"]["mean"],
        eigen_aging_mean_hog=ea_results["hog"]["mean"],
        eigen_aging_ev_pix=ea_results["pix"]["ev_ratio"],
        eigen_aging_ev_lbp=ea_results["lbp"]["ev_ratio"],
        eigen_aging_ev_hog=ea_results["hog"]["ev_ratio"],
        eigen_aging_sv_pix=ea_results["pix"]["singular_values"],
        eigen_aging_sv_lbp=ea_results["lbp"]["singular_values"],
        eigen_aging_sv_hog=ea_results["hog"]["singular_values"],
        n_pairs=np.array([valid_pairs]),
    )
    print(f"     ✅ Saved {NPZ_OUT}")

    # ---- Print config for tune_cross_age.py ----
    print("\n" + "=" * 60)
    print("  Next: run tune_cross_age.py with eigen-aging mode")
    print("=" * 60)


if __name__ == "__main__":
    main()
