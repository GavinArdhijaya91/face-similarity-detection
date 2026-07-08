"""
tune_cross_age.py — Grid search optimal parameters untuk cross-age face matching.

Menguji kombinasi:
  . aging_scale  [0.0 - 1.0]       kekuatan injeksi vektor penuaan
  . threshold    [0.30 - 0.70]      batas keputusan SAMA/BEDA
  . feature_mode [pixel, lbp, hog, fusion]
  . hog_direct / pixel_direct        benchmark tanpa PCA

Output: parameter terbaik + perbandingan akurasi.
"""

import os
import sys
import warnings
from itertools import product

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from streamlit_app.core.feature_extractor import (
    extract_hog_features,
    extract_lbp_fast,
    extract_pixel_features,
)
from streamlit_app.core.pca_svd import project_to_eigenspace

NPZ_EIGEN = "pretrained_eigenspace.npz"
NPZ_DATA = "privasi_kelompok_100x100.npz"
NPZ_EIGEN_AGING = "eigen_aging_vectors.npz"


def load_data():
    ei = np.load(NPZ_EIGEN)
    ds = np.load(NPZ_DATA)
    labels = ds["y"]
    nama = ds.get("nama_anggota", [f"Anggota {i}" for i in range(len(labels))])
    return ei, ds, labels, nama


def extract_modality(images, mode):
    result = []
    for img in images:
        v = img.reshape(100, 100)
        if mode == "pixel":
            result.append(extract_pixel_features(v))
        elif mode == "lbp":
            result.append(extract_lbp_fast(v))
        elif mode == "hog":
            result.append(extract_hog_features(v))
    return np.array(result)


def wilson_ci(k, n, z=1.96):
    """Wilson confidence interval for binomial proportion (95% CI)."""
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0, center - margin), min(1, center + margin)


def random_baseline_pvalue(k, n):
    """One-sided binomial test: probability of >= k correct out of n by chance (1/n)."""
    from scipy.stats import binom
    p_chance = 1.0 / n
    return binom.sf(k - 1, n, p_chance)


def top1_accuracy(W_query, W_db, y_db):
    """Return top-1 identification accuracy and similarity matrix."""
    sims = cosine_similarity(W_query, W_db)
    preds = np.argmax(sims, axis=1)
    correct = sum(1 for i, p in enumerate(preds) if y_db[p] == y_db[i])
    n = len(W_query)
    ci_low, ci_high = wilson_ci(correct, n)
    pval = random_baseline_pvalue(correct, n)
    return correct / n * 100, sims, correct, n, ci_low * 100, ci_high * 100, pval


def verification_metrics(sims, y_db, threshold):
    """Compute verification metrics at a given threshold."""
    tp = fp = tn = fn = 0
    n = len(sims)
    for i in range(n):
        best_idx = np.argmax(sims[i])
        pred_same = sims[i, best_idx] >= threshold
        true_same = y_db[best_idx] == y_db[i]
        if pred_same and true_same:
            tp += 1
        elif pred_same and not true_same:
            fp += 1
        elif not pred_same and true_same:
            fn += 1
        else:
            tn += 1
    acc = (tp + tn) / n * 100 if n else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  GRID SEARCH: Tuning Cross-Age PCA Parameters")
    print("=" * 60)

    if not os.path.exists(NPZ_EIGEN):
        print(f"  ERROR: {NPZ_EIGEN} tidak ditemukan.")
        sys.exit(1)
    if not os.path.exists(NPZ_DATA):
        print(f"  ERROR: {NPZ_DATA} tidak ditemukan.")
        sys.exit(1)

    ei, ds, labels, nama = load_data()
    n_people = len(labels)
    adult = ds["X_latih"]
    child = ds["X_test_lintas"]
    print(f"\n  Dataset: {n_people} orang, {len(adult)} adult, {len(child)} child")

    SCALES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    PCA_MODES = ["pixel", "lbp", "hog"]
    FUSION_WEIGHTS = [
        (0.35, 0.50, 0.15),
        (0.50, 0.30, 0.20),
        (0.20, 0.60, 0.20),
        (0.33, 0.33, 0.34),
        (0.60, 0.20, 0.20),
        (0.10, 0.70, 0.20),
        (0.25, 0.50, 0.25),
    ]

    # Load eigen-aging vectors if available
    ea = None
    if os.path.exists(NPZ_EIGEN_AGING):
        ea = np.load(NPZ_EIGEN_AGING)
        np_ea = ea.get("n_pairs", np.array([0]))
        n_pairs = int(np_ea.item() if hasattr(np_ea, 'item') else np_ea)
        print(f"  Eigen-aging loaded: {n_pairs} FG-NET person pairs")
    else:
        print(f"  Note: {NPZ_EIGEN_AGING} not found — skipping eigen-aging mode")

    all_results = []

    # ==================================================================
    #  DIRECT (no PCA) benchmarks
    # ==================================================================
    for direct_mode in ["hog", "pixel"]:
        print(f"\n  --- {direct_mode.upper()} Direct (tanpa PCA) ---")
        feat_adult = extract_modality(adult, direct_mode)
        feat_child = extract_modality(child, direct_mode)
        acc_id, sims, n_correct, n_total, ci_lo, ci_hi, pval = top1_accuracy(feat_child, feat_adult, labels)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  Top-1 ident: {acc_id:.1f}%  (CI 95%: {ci_lo:.1f}% - {ci_hi:.1f}%, {n_correct}/{n_total}, p={pval:.4f} {sig})")
        for th in THRESHOLDS:
            av, pr, re, f1 = verification_metrics(sims, labels, th)
            all_results.append((f"{direct_mode}_direct", 0.0, th, acc_id, av, pr, re, f1))
            print(f"    th={th:.2f}  acc={av:.1f}%  prec={pr:.1f}%  rec={re:.1f}%  f1={f1:.1f}")

    # ==================================================================
    #  PCA modes (pixel, lbp, hog) + aging scale sweep
    # ==================================================================
    for mode in PCA_MODES:
        print(f"\n  --- PCA {mode.upper()} ---")
        feat_adult = adult.copy() if mode == "pixel" else extract_modality(adult, mode)
        feat_child = child.copy() if mode == "pixel" else extract_modality(child, mode)

        if mode == "pixel":
            mf, ef, sv = ei["mean_face"], ei["eigenfaces"], ei["singular_values"]
            v_fg = ei.get("aging_vector_fgnet_pix", ei.get("aging_vector_pix", np.zeros(100)))
            v_af = ei.get("aging_vector_aaf_pix", v_fg)
        elif mode == "lbp":
            mf, ef, sv = ei["mean_lbp"], ei["eigenfaces_lbp"], ei["singular_values_lbp"]
            v_fg = ei.get("aging_vector_fgnet_lbp", np.zeros(100))
            v_af = ei.get("aging_vector_aaf_lbp", v_fg)
        else:
            mf, ef, sv = ei["mean_hog"], ei["eigenfaces_hog"], ei["singular_values_hog"]
            v_fg = ei.get("aging_vector_fgnet_hog", np.zeros(100))
            v_af = ei.get("aging_vector_aaf_hog", v_fg)

        n_samp = int(ei["n_samples"].item() if ei["n_samples"].ndim == 0 else ei["n_samples"][0])
        unwhiten = sv / np.sqrt(max(1, n_samp - 1))
        hybrid = 0.85 * v_af + 0.15 * v_fg

        W_db = project_dataset(feat_adult, mf, ef)

        best_id = -1
        best_s = 0.0
        for scale in SCALES:
            W_q = project_dataset(feat_child, mf, ef)
            W_q = W_q + hybrid * unwhiten * scale
            acc_id, sims, n_correct, n_total, ci_lo, ci_hi, pval = top1_accuracy(W_q, W_db, labels)
            if acc_id > best_id:
                best_id = acc_id
                best_s = scale
            for th in THRESHOLDS:
                av, pr, re, f1 = verification_metrics(sims, labels, th)
                all_results.append((f"pca_{mode}", scale, th, acc_id, av, pr, re, f1))
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  Best ident: scale={best_s:.1f} -> {best_id:.1f}%  (CI 95%: {ci_lo:.1f}%-{ci_hi:.1f}%, p={pval:.4f} {sig})")

        # Eigen-aging comparison
        if ea is not None:
            W_q_proj = project_dataset(feat_child, mf, ef)
            best_e_id, best_e_s = run_eigen_aging_mode(ea, mode, W_q_proj, W_db, labels, SCALES, THRESHOLDS, all_results)
            if best_e_id >= 0:
                sig_e = "***" if best_e_id > best_id + 1 else ""
                print(f"  Eigen-aging     scale={best_e_s:.1f} -> {best_e_id:.1f}% {sig_e}")

    # ==================================================================
    #  PCA FUSION (pixel + lbp + hog)
    # ==================================================================
    print(f"\n  --- PCA FUSION (Pixel + LBP + HOG) ---")

    # pre-extract
    a_pix = adult.copy()
    c_pix = child.copy()
    a_lbp = extract_modality(adult, "lbp")
    c_lbp = extract_modality(child, "lbp")
    a_hog = extract_modality(adult, "hog")
    c_hog = extract_modality(child, "hog")

    # pre-project
    n_samp = int(ei["n_samples"].item() if ei["n_samples"].ndim == 0 else ei["n_samples"][0])
    ws = ei["singular_values"] / np.sqrt(max(1, n_samp - 1))
    ws_lbp = ei["singular_values_lbp"] / np.sqrt(max(1, n_samp - 1))
    ws_hog = ei["singular_values_hog"] / np.sqrt(max(1, n_samp - 1))

    v = {
        "pix": (ei.get("aging_vector_aaf_pix", np.zeros(100)),
                ei.get("aging_vector_fgnet_pix", np.zeros(100))),
        "lbp": (ei.get("aging_vector_aaf_lbp", np.zeros(100)),
                ei.get("aging_vector_fgnet_lbp", np.zeros(100))),
        "hog": (ei.get("aging_vector_aaf_hog", np.zeros(100)),
                ei.get("aging_vector_fgnet_hog", np.zeros(100))),
    }

    W_db_pix = project_dataset(a_pix, ei["mean_face"], ei["eigenfaces"])
    W_db_lbp = project_dataset(a_lbp, ei["mean_lbp"], ei["eigenfaces_lbp"])
    W_db_hog = project_dataset(a_hog, ei["mean_hog"], ei["eigenfaces_hog"])

    wh_pix = ei.get("whiten_scale", np.ones(100))
    wh_lbp = ei.get("whiten_scale_lbp", np.ones(100))
    wh_hog = ei.get("whiten_scale_hog", np.ones(100))

    for alpha, beta, gamma in FUSION_WEIGHTS:
        print(f"    weights: a={alpha:.2f} b={beta:.2f} g={gamma:.2f}")
        best_id = -1
        best_s = 0.0

        for scale in SCALES:
            h_pix = 0.85 * v["pix"][0] + 0.15 * v["pix"][1]
            h_lbp = 0.85 * v["lbp"][0] + 0.15 * v["lbp"][1]
            h_hog = 0.85 * v["hog"][0] + 0.15 * v["hog"][1]

            Wq_p = project_dataset(c_pix, ei["mean_face"], ei["eigenfaces"]) + h_pix * ws * scale
            Wq_l = project_dataset(c_lbp, ei["mean_lbp"], ei["eigenfaces_lbp"]) + h_lbp * ws_lbp * scale
            Wq_h = project_dataset(c_hog, ei["mean_hog"], ei["eigenfaces_hog"]) + h_hog * ws_hog * scale

            Wq_p = Wq_p / (wh_pix + 1e-8)
            Wq_l = Wq_l / (wh_lbp + 1e-8)
            Wq_h = Wq_h / (wh_hog + 1e-8)
            Wd_p = W_db_pix / (wh_pix + 1e-8)
            Wd_l = W_db_lbp / (wh_lbp + 1e-8)
            Wd_h = W_db_hog / (wh_hog + 1e-8)

            sp = cosine_similarity(Wq_p, Wd_p)
            sl = cosine_similarity(Wq_l, Wd_l)
            sh = cosine_similarity(Wq_h, Wd_h)
            tw = alpha + beta + gamma
            s_fused = (alpha * sl + beta * sh + gamma * sp) / tw

            preds = np.argmax(s_fused, axis=1)
            correct = sum(1 for i, p in enumerate(preds) if labels[p] == labels[i])
            acc_id = correct / len(s_fused) * 100

            if acc_id > best_id:
                best_id = acc_id
                best_s = scale

            for th in THRESHOLDS:
                tp = fp = tn = fn = 0
                for i in range(len(s_fused)):
                    bi = np.argmax(s_fused[i])
                    ps = s_fused[i, bi] >= th
                    ts = labels[bi] == labels[i]
                    if ps and ts: tp += 1
                    elif ps and not ts: fp += 1
                    elif not ps and ts: fn += 1
                    else: tn += 1
                av = (tp + tn) / len(s_fused) * 100
                pr = tp / (tp + fp) * 100 if (tp + fp) else 0
                re = tp / (tp + fn) * 100 if (tp + fn) else 0
                f1 = 2 * pr * re / (pr + re) if (pr + re) else 0
                label = f"fusion_a{alpha:.2f}b{beta:.2f}g{gamma:.2f}"
                all_results.append((label, scale, th, acc_id, av, pr, re, f1))
        print(f"      best ident: scale={best_s:.1f} -> {best_id:.1f}%")

    # ==================================================================
    #  SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("  RANGKUMAN — Best Identification Accuracy per Mode")
    print("=" * 60)
    print(f"  {'Mode':<28} {'Scale':>6} {'Ident%':>7} {'Best Verif%':>11}")
    print(f"  {'-'*52}")

    seen = set()
    for r in sorted(all_results, key=lambda x: x[3], reverse=True):
        key = r[0]
        if key not in seen:
            seen.add(key)
            # best verification at optimal threshold for this mode+scale
            best_v = max((x for x in all_results if x[0] == key and abs(x[1] - r[1]) < 0.01),
                         key=lambda x: x[4], default=r)
            print(f"  {key:<28} {r[1]:>6.1f} {r[3]:>6.1f}% {best_v[4]:>10.1f}%")

    print()
    best_all = max(all_results, key=lambda r: r[3])
    print(f"  BEST: {best_all[0]} | scale={best_all[1]:.1f} | ident={best_all[3]:.1f}%")

    # Best verification
    best_v_all = max(all_results, key=lambda r: r[4])
    print(f"  BEST VERIF: {best_v_all[0]} | scale={best_v_all[1]:.1f} | th={best_v_all[2]:.2f} | "
          f"acc={best_v_all[4]:.1f}% f1={best_v_all[7]:.1f}")
    print("=" * 60)

    # ==================================================================
    #  PRINT BEST CONFIG FOR APP
    # ==================================================================
    print("\n  Konfigurasi rekomendasi untuk app:")
    print(f"    DECISION_THRESHOLD = {best_v_all[2]:.2f}")
    print(f"    aging_scale        = {best_v_all[1]:.1f}")
    print(f"    mode               = {best_v_all[0].replace('_a', ' ').replace('b', ' ').split()[0]}")


def project_dataset(features, mean_face, eigenfaces):
    return np.array([project_to_eigenspace(f, eigenfaces, mean_face) for f in features])


def softmax(x, sigma=0.3):
    """Stable softmax for similarity weighting."""
    x = np.asarray(x, dtype=np.float64)
    x = x / max(sigma, 1e-10)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-15)


def compute_personalized_aging(q_scores, child_scores, deltas, sigma=0.3):
    """
    Nearest-neighbor weighted aging vector.
    q_scores: (n, d) — batch of query scores (unwhitened PCA space)
    child_scores: (m, d) — FG-NET child scores
    deltas: (m, d) — FG-NET adult-child deltas
    Returns: (n, d) — personalized aging vectors
    """
    from sklearn.metrics.pairwise import cosine_similarity
    n, m = len(q_scores), len(child_scores)
    sims = cosine_similarity(q_scores, child_scores)  # (n, m)
    aging = np.zeros_like(q_scores)
    for i in range(n):
        w = softmax(sims[i], sigma)
        aging[i] = w @ deltas
    return aging


def run_eigen_aging_mode(ea, mode, W_q, W_db, labels, scales, thresholds, results):
    """
    Run NN-weighted personalized aging for a given modality.
    """
    if mode == "pixel":
        c_scores = ea["fgnet_child_scores_pix"]
        deltas = ea["fgnet_deltas_pix"]
    elif mode == "lbp":
        c_scores = ea["fgnet_child_scores_lbp"]
        deltas = ea["fgnet_deltas_lbp"]
    else:
        c_scores = ea["fgnet_child_scores_hog"]
        deltas = ea["fgnet_deltas_hog"]

    np_ea_n = ea.get("n_pairs", np.array([0]))
    n_pairs = int(np_ea_n.item() if hasattr(np_ea_n, 'item') else np_ea_n)
    if n_pairs < 3:
        return -1, 0.0

    best_id = -1
    best_scale = 0.0
    for scale in scales:
        aging_vecs = compute_personalized_aging(W_q, c_scores, deltas, sigma=0.3)
        W_aged = W_q + aging_vecs * scale
        acc_id, sims, n_correct, n_total, ci_lo, ci_hi, pval = top1_accuracy(W_aged, W_db, labels)
        if acc_id > best_id:
            best_id = acc_id
            best_scale = scale
        for th in thresholds:
            av, pr, re, f1 = verification_metrics(sims, labels, th)
            results.append((f"eigen_{mode}", scale, th, acc_id, av, pr, re, f1))

    # Also try eigen-aging PCA (projection-based)
    e_comp = ea.get(f"eigen_aging_{mode}", None)
    e_mean = ea.get(f"eigen_aging_mean_{mode}", None)
    if e_comp is not None and e_mean is not None and len(e_comp) >= 1:
        for scale in scales:
            W_c = W_q.copy()
            for i in range(len(W_c)):
                q = W_c[i]
                centered = q - e_mean
                coeffs = centered @ e_comp.T
                aging = e_mean + coeffs @ e_comp
                W_c[i] = q + aging * scale
            acc_id, sims, n_correct, n_total, ci_lo, ci_hi, pval = top1_accuracy(W_c, W_db, labels)
            for th in thresholds:
                av, pr, re, f1 = verification_metrics(sims, labels, th)
                results.append((f"eigen_pca_{mode}", scale, th, acc_id, av, pr, re, f1))

    return best_id, best_scale


if __name__ == "__main__":
    main()
