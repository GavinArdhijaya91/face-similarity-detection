# 📑 PRODUCT REQUIREMENT DOCUMENT (PRD) v3.0
## Sistem Face Comparison & Similarity Detection Berbasis PCA (Eigenfaces)
### Skenario Usia Sama & Lintas Usia — Panduan Lengkap End-to-End

---

## 📋 DAFTAR ISI

1. [Gambaran Sistem & Tujuan](#1-gambaran-sistem--tujuan)
2. [Fondasi Matematika PCA & Eigenfaces](#2-fondasi-matematika-pca--eigenfaces)
3. [Tantangan Deteksi Lintas Usia](#3-tantangan-deteksi-lintas-usia)
4. [Arsitektur Dataset](#4-arsitektur-dataset)
5. [Pipeline Preprocessing Lanjutan](#5-pipeline-preprocessing-lanjutan)
6. [Arsitektur Privasi Data (.NPZ)](#6-arsitektur-privasi-data-npz)
7. [Implementasi Kode Lengkap — Tahap 1: Lokal](#7-implementasi-kode-lengkap--tahap-1-lokal)
8. [Implementasi Kode Lengkap — Tahap 2: Google Colab](#8-implementasi-kode-lengkap--tahap-2-google-colab)
9. [Visualisasi & Debugging](#9-visualisasi--debugging)
10. [Interpretasi Skor Kemiripan](#10-interpretasi-skor-kemiripan)
11. [Acceptance Criteria & Target Output](#11-acceptance-criteria--target-output)
12. [Troubleshooting & FAQ](#12-troubleshooting--faq)
13. [Peningkatan Lanjutan (Opsional)](#13-peningkatan-lanjutan-opsional)

---

## 1. GAMBARAN SISTEM & TUJUAN

### 1.1 Deskripsi Proyek

Sistem ini adalah **mesin perbandingan kemiripan wajah berbasis Principal Component Analysis (PCA)** yang dikenal dengan teknik **Eigenfaces**. Sistem mampu:

- **Skenario A — Usia Sama (Same-Age):** Membandingkan dua foto wajah dewasa dari orang yang sama untuk memverifikasi apakah itu orang yang sama.
- **Skenario B — Lintas Usia (Cross-Age):** Membandingkan foto masa kecil/balita dengan foto dewasa untuk mendeteksi apakah wajah berasal dari orang yang sama meski ada perubahan usia signifikan.

### 1.2 Alur Sistem Keseluruhan

```
[ Foto Asli (JPG/PNG) ]
        │
        ▼
[ PREPROCESSING: Grayscale → Resize → CLAHE → Normalisasi ]
        │
        ▼
[ ENKRIPSI: Flatten → Array NumPy → .NPZ Binary ]
        │
        ▼ (Upload ke Colab)
[ LOAD: Olivetti Faces + Dataset Kelompok .NPZ ]
        │
        ▼
[ TRAINING: PCA Fit pada X_train (405 sampel × 10.000 fitur) ]
        │
        ▼
[ TRANSFORM: Semua gambar → Eigenface Space (50 dimensi) ]
        │
        ▼
[ SIMILARITY ENGINE: Euclidean Distance + Cosine Similarity ]
        │
        ▼
[ OUTPUT: Skor Kemiripan + Label Prediksi + Akurasi Per Skenario ]
```

### 1.3 Teknologi yang Digunakan

| Komponen | Library | Versi Minimum |
|---|---|---|
| Preprocessing Gambar | OpenCV (`cv2`) | 4.5+ |
| Operasi Array | NumPy | 1.21+ |
| Algoritma PCA | scikit-learn | 1.0+ |
| Dataset Benchmark | scikit-learn (Olivetti) | 1.0+ |
| Metrik Kemiripan | NumPy + scikit-learn | — |
| Visualisasi | Matplotlib | 3.4+ |
| Lingkungan Eksekusi | Google Colab / Jupyter | — |

---

## 2. FONDASI MATEMATIKA PCA & EIGENFACES

Memahami matematika di balik PCA sangat krusial untuk memahami mengapa sistem ini berhasil (dan di mana batasannya).

### 2.1 Representasi Gambar sebagai Vektor

Setiap gambar wajah berukuran **100 × 100 piksel (grayscale)** diratakan (*flattened*) menjadi sebuah **vektor 1D berukuran N = 10.000 dimensi**.

```
Gambar[100][100]  →  Vektor[10.000]

Matriks Data:
X_train = [ v₁ ]   ← vektor gambar ke-1 (10.000 dimensi)
          [ v₂ ]   ← vektor gambar ke-2
          [ ... ]
          [ v₄₀₅ ] ← vektor gambar ke-405

Shape: X_train ∈ ℝ^(405 × 10.000)
```

### 2.2 Mean Centering (Wajah Rata-Rata)

PCA pertama kali menghitung **wajah rata-rata** (mean face), lalu menguranginya dari setiap gambar:

```
μ = (1/m) × Σᵢ xᵢ           ← Mean face (rata-rata semua wajah)

X_centered = X_train - μ     ← Semua gambar dikurangi mean face
```

Mean face ini merepresentasikan "wajah tipikal" dari seluruh dataset. Gambar yang dikurangi mean face menyisakan informasi **variasi individu** saja.

### 2.3 Dekomposisi Nilai Singular (SVD) — Inti PCA

PCA melakukan **Singular Value Decomposition** pada matriks yang telah di-*center*:

```
X_centered = U × Σ × Vᵀ

Komponen:
• U  : Matriks singular kiri (m × m)
• Σ  : Matriks diagonal nilai singular
• Vᵀ : Matriks singular kanan (n × n) — INILAH Eigenfaces!
```

Kolom-kolom dari `Vᵀ` adalah **principal components** — yang dalam konteks wajah disebut **Eigenfaces**. Setiap Eigenface adalah sebuah pola wajah yang menangkap sumber variasi terbesar dalam dataset.

### 2.4 Proyeksi ke Eigenface Space

Dengan memilih `k = 50` komponen utama terbesar, setiap gambar diproyeksikan dari ruang 10.000 dimensi ke ruang **50 dimensi**:

```
w = Eigenfaces_kᵀ × (x - μ)

Hasil:
• x   : Vektor gambar asli (10.000 dimensi)
• μ   : Mean face (10.000 dimensi)
• w   : Vektor bobot wajah (50 dimensi) — representasi ringkas!
```

Vektor `w` inilah yang digunakan untuk membandingkan kemiripan antar wajah.

### 2.5 Metrik Kemiripan

#### Metrik 1 — Euclidean Distance (L2 Norm)

Mengukur jarak geometris absolut antara dua vektor bobot:

```
d(wᵢ, wⱼ) = √(Σₖ (wᵢₖ - wⱼₖ)²)

Interpretasi:
• d → 0    : Identik (sangat mirip)
• d kecil  : Mirip
• d besar  : Berbeda
```

**Kelemahan:** Sensitif terhadap perbedaan magnitudo — bisa terpengaruh oleh perbedaan kecerahan foto.

#### Metrik 2 — Cosine Similarity

Mengukur **sudut** antara dua vektor (bukan panjangnya):

```
cos(θ) = (wᵢ · wⱼ) / (‖wᵢ‖ × ‖wⱼ‖)

Interpretasi:
• cos(θ) → 1.0  : Arah identik (sangat mirip)
• cos(θ) → 0.0  : Tegak lurus (tidak berkaitan)
• cos(θ) → -1.0 : Berlawanan arah
```

**Keunggulan vs Lintas Usia:** Cosine similarity lebih **tahan terhadap perbedaan pencahayaan dan kontras** antara foto lama (kecil) dan foto baru (dewasa), karena hanya memperhatikan arah vektor, bukan magnitudo.

### 2.6 Mengapa 50 Komponen?

```
Variance yang dijelaskan (n_components):
  10  komponen →  ~55% variance
  50  komponen →  ~85% variance  ← Titik keseimbangan optimal
  100 komponen →  ~95% variance
  All komponen →  100% variance (= tanpa kompresi, rentan noise)

Trade-off:
  • Terlalu sedikit → Informasi hilang, akurasi rendah
  • Terlalu banyak → Overfitting ke noise, generalisasi buruk
  • 50 → Sweet spot antara representasi & generalisasi
```

---

## 3. TANTANGAN DETEKSI LINTAS USIA

Ini adalah inti dari kesulitan **Skenario B**. Memahami tantangannya memungkinkan kita memilih solusi preprocessing yang tepat.

### 3.1 Faktor Perubahan Wajah Seiring Usia

| Faktor | Dampak pada Vektor PCA | Tingkat Kesulitan |
|---|---|---|
| Perubahan struktur tulang wajah | Proporsi fitur berubah drastis | 🔴 Tinggi |
| Perubahan distribusi lemak wajah | Kontur wajah berubah | 🟡 Menengah |
| Pertumbuhan rambut/jenggot | Tekstur area tepi wajah berubah | 🟡 Menengah |
| Perbedaan kualitas kamera (era berbeda) | Resolusi, noise, artefak berbeda | 🟡 Menengah |
| Perbedaan pencahayaan & ekspresi | Distribusi piksel bergeser | 🟢 Dapat Ditangani |

### 3.2 Analisis Ekspektasi Akurasi

```
Skenario A (Usia Sama):
  • Foto dewasa ke-1 vs foto dewasa ke-2
  • Perubahan minimal antara dua foto
  • Ekspektasi Akurasi: 60% - 100% (bergantung kualitas foto)

Skenario B (Lintas Usia — Kecil vs Dewasa):
  • Foto balita/anak vs foto dewasa
  • Perubahan signifikan dalam 10-25 tahun
  • Ekspektasi Akurasi: 20% - 60% (secara inheren lebih sulit)

Catatan: PCA standar tanpa augmentasi akan kesulitan di Skenario B.
Tujuan utama adalah MEMBANDINGKAN kedua skenario, bukan mencapai 100%.
```

### 3.3 Solusi Teknis untuk Memitigasi Gap Usia

Tiga teknik preprocessing utama yang diterapkan:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — Menormalisasi kontras secara lokal per region, mengurangi dampak perbedaan pencahayaan era berbeda.
2. **Face Alignment** — Memastikan posisi mata/hidung/mulut relatif konsisten antar foto.
3. **Gamma Correction** — Menyesuaikan kecerahan keseluruhan agar distribusi piksel lebih seragam.

---

## 4. ARSITEKTUR DATASET

### 4.1 Struktur Folder Dataset Kelompok (WAJIB Diikuti)

```
dataset_kelompok/
├── orang_0/                    ← Anggota Kelompok ke-1
│   ├── kecil.jpg               ← Foto masa kecil (usia 0-12 tahun)
│   ├── dewasa_bawaan.jpg       ← Foto dewasa 1 → MASUK DATA TRAIN
│   └── dewasa_uji.jpg          ← Foto dewasa 2 → UJI SKENARIO A
├── orang_1/
│   ├── kecil.jpg
│   ├── dewasa_bawaan.jpg
│   └── dewasa_uji.jpg
├── orang_2/
│   ├── kecil.jpg
│   ├── dewasa_bawaan.jpg
│   └── dewasa_uji.jpg
├── orang_3/
│   ├── kecil.jpg
│   ├── dewasa_bawaan.jpg
│   └── dewasa_uji.jpg
└── orang_4/                    ← Minimal 5 orang, maksimal 10 orang
    ├── kecil.jpg
    ├── dewasa_bawaan.jpg
    └── dewasa_uji.jpg
```

> ⚠️ **ATURAN PENAMAAN FOTO:**
> - Nama file **huruf kecil semua**, tanpa spasi
> - Format: `.jpg` atau `.png` (konsisten dalam satu folder)
> - Wajah harus **terlihat jelas** dan **menghadap ke depan** (frontal face)
> - Resolusi minimum: **80 × 80 piksel** sebelum resize

### 4.2 Komposisi Matriks Data Akhir

```
Dataset Olivetti (Benchmark):
  • 40 orang berbeda (ID 0–39)
  • 10 foto per orang → 400 gambar total
  • Resolusi asli: 64×64 piksel (di-resize ke 100×100)
  • Label: y ∈ {0, 1, 2, ..., 39}

Dataset Kelompok (5–10 orang):
  • Setiap orang: 1 foto dewasa_bawaan masuk training
  • Label: y ∈ {40, 41, ..., 44} (untuk 5 orang)
  • Data uji TIDAK masuk training

Matriks X_train Final:
  Shape: [400 + N_kelompok] × 10.000
  Untuk 5 orang: [405 × 10.000]
  Untuk 10 orang: [410 × 10.000]

Data Uji (TIDAK masuk training):
  • X_test_sama  : [N_kelompok × 10.000] — dewasa_uji tiap anggota
  • X_test_lintas: [N_kelompok × 10.000] — kecil tiap anggota
```

### 4.3 Panduan Pemilihan Foto yang Baik

#### Foto `dewasa_bawaan.jpg` & `dewasa_uji.jpg` (Training & Uji Skenario A)
- ✅ Wajah frontal (menghadap lurus ke kamera)
- ✅ Pencahayaan merata, tidak ada bayangan keras di wajah
- ✅ Ekspresi netral atau senyum natural
- ✅ Resolusi cukup (minimal 200×200 piksel sebelum resize)
- ❌ Hindari kacamata hitam, masker, atau penutup wajah
- ❌ Hindari foto dengan filter tebal (Instagram filter, dll)

#### Foto `kecil.jpg` (Uji Skenario B — Lintas Usia)
- ✅ Wajah anak terlihat jelas dan frontal
- ✅ Usia antara 0–12 tahun (semakin muda, semakin challenging)
- ✅ Foto asli, bukan screenshot atau foto yang difoto ulang
- ❌ Hindari foto yang sangat blur atau gelap
- ℹ️ Foto usia 8–12 tahun umumnya memberikan hasil lebih baik dari foto balita

---

## 5. PIPELINE PREPROCESSING LANJUTAN

### 5.1 Alur Preprocessing Standar (Skenario A)

```
Input: Gambar wajah (format apapun, ukuran apapun)
         │
         ▼
Step 1: Baca sebagai Grayscale         → Eliminasi informasi warna
         │
         ▼
Step 2: Resize ke 100×100             → Standarisasi dimensi
         │
         ▼
Step 3: Normalisasi [0, 1]            → Skalakan piksel dari [0,255] ke [0,1]
         │
         ▼
Step 4: Flatten ke vektor 1D (10.000) → Siap untuk PCA
         │
         ▼
Output: np.array berukuran (10.000,), dtype=float32
```

### 5.2 Alur Preprocessing Lanjutan (Skenario B — Lintas Usia)

Untuk foto `kecil.jpg`, tambahkan langkah ekstra sebelum normalisasi:

```
Input: Foto masa kecil (biasanya kualitas lebih rendah, pencahayaan beda)
         │
         ▼
Step 1: Baca sebagai Grayscale
         │
         ▼
Step 2: CLAHE Enhancement             → Normalisasi kontras adaptif lokal
         │  clip_limit=2.0
         │  tileGridSize=(8,8)
         ▼
Step 3: Gaussian Blur ringan          → Kurangi noise dari foto lama/scan
         │  kernel=(3,3), sigmaX=0.5
         ▼
Step 4: Resize ke 100×100
         │
         ▼
Step 5: Gamma Correction              → Sesuaikan kecerahan keseluruhan
         │  gamma=1.2 (terangkan foto gelap)
         ▼
Step 6: Normalisasi [0, 1]
         │
         ▼
Step 7: Flatten ke vektor 1D (10.000)
         │
         ▼
Output: np.array berukuran (10.000,), dtype=float32
```

### 5.3 Penjelasan Teknis CLAHE

**CLAHE (Contrast Limited Adaptive Histogram Equalization)** bekerja dengan membagi gambar menjadi **tile-tile kecil (8×8)** dan melakukan ekualisasi histogram secara **lokal** pada masing-masing tile. Hasilnya:

```
Tanpa CLAHE:
  Foto lama yang terlalu gelap → seluruh histogram menumpuk di nilai rendah
  Setelah flatten, vektor memiliki nilai kecil semua → Euclidean distance besar

Dengan CLAHE:
  Setiap region wajah memiliki kontras yang dinormalisasi secara independen
  → Distribusi piksel lebih seragam antar foto dari era berbeda
  → Vektor PCA lebih stabil dan dapat dibandingkan
```

---

## 6. ARSITEKTUR PRIVASI DATA (.NPZ)

### 6.1 Mekanisme Enkripsi & Alasan

Untuk melindungi privasi data anggota kelompok dan mematuhi etika pengelolaan data:

- **Masalah:** Menyimpan foto `.jpg`/`.png` mentah di Google Colab (cloud) berisiko privasi.
- **Solusi:** Konversi foto ke **array NumPy biner** sebelum diunggah. File `.npz` tidak dapat dibuka langsung oleh browser atau viewer gambar biasa.

### 6.2 Skema Isi File `.npz`

```
privasi_kelompok_100x100.npz
├── X_latih      → shape: [N × 10.000], dtype: float32
│                   Foto dewasa_bawaan (masuk training)
├── X_test_sama  → shape: [N × 10.000], dtype: float32
│                   Foto dewasa_uji (uji Skenario A)
├── X_test_lintas→ shape: [N × 10.000], dtype: float32
│                   Foto kecil (uji Skenario B)
└── y            → shape: [N], dtype: int32
                    Label ID anggota (40, 41, 42, ..., 44)
```

---

## 7. IMPLEMENTASI KODE LENGKAP — TAHAP 1: LOKAL

Jalankan skrip ini di PC/Laptop lokal yang menyimpan folder `dataset_kelompok`.

### 7.1 Instalasi Dependensi Lokal

```bash
pip install opencv-python numpy
```

### 7.2 Skrip Enkripsi Data Kelompok (Versi Lengkap dengan Preprocessing Lanjutan)

```python
# =============================================================================
# SKRIP LOKAL: Enkripsi Dataset Kelompok ke .NPZ
# Jalankan di PC/Laptop lokal SEBELUM upload ke Google Colab
# =============================================================================

import os
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# KONFIGURASI — Sesuaikan bagian ini
# -----------------------------------------------------------------------------
PATH_KELOMPOK = 'dataset_kelompok'       # Path folder dataset
LABEL_MULAI = 40                          # ID mulai (setelah Olivetti 0-39)
TARGET_SIZE = (100, 100)                  # Ukuran target gambar
MAX_ANGGOTA = 10                          # Maksimum anggota kelompok
NAMA_OUTPUT = 'privasi_kelompok_100x100.npz'

# Kamus nama anggota — sesuaikan dengan urutan folder alfabetis
KAMUS_NAMA = {
    40: "Anggota 1",
    41: "Anggota 2",
    42: "Anggota 3",
    43: "Anggota 4",
    44: "Anggota 5",
    # Tambahkan sampai 49 jika ada 10 anggota
}

# -----------------------------------------------------------------------------
# FUNGSI PREPROCESSING STANDAR (untuk foto dewasa)
# -----------------------------------------------------------------------------
def preprocess_dewasa(img_path: str) -> np.ndarray | None:
    """
    Preprocessing standar untuk foto dewasa.
    Returns: numpy array (10.000,) float32, atau None jika gagal
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ⚠️  GAGAL membaca: {img_path}")
        return None

    img_resized = cv2.resize(img, TARGET_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized.flatten()

# -----------------------------------------------------------------------------
# FUNGSI PREPROCESSING LANJUTAN (untuk foto masa kecil — lintas usia)
# -----------------------------------------------------------------------------
def preprocess_lintas_usia(img_path: str) -> np.ndarray | None:
    """
    Preprocessing lanjutan untuk foto masa kecil.
    Menggunakan CLAHE + Gaussian Blur + Gamma Correction untuk
    meminimalkan dampak perbedaan era, kualitas foto, dan pencahayaan.
    
    Returns: numpy array (10.000,) float32, atau None jika gagal
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ⚠️  GAGAL membaca: {img_path}")
        return None

    # Langkah 1: CLAHE — normalisasi kontras adaptif lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # Langkah 2: Gaussian Blur ringan — reduksi noise dari foto lama/scan
    img_blur = cv2.GaussianBlur(img_clahe, (3, 3), sigmaX=0.5)

    # Langkah 3: Resize ke target
    img_resized = cv2.resize(img_blur, TARGET_SIZE)

    # Langkah 4: Gamma Correction — sesuaikan kecerahan
    # gamma > 1 → terangkan foto gelap | gamma < 1 → gelapkan foto terlalu terang
    gamma = 1.2
    img_gamma = np.power(img_resized.astype(np.float32) / 255.0, 1.0 / gamma)

    return img_gamma.flatten()

# -----------------------------------------------------------------------------
# FUNGSI VALIDASI GAMBAR
# -----------------------------------------------------------------------------
def validasi_gambar(img_path: str, nama_file: str) -> bool:
    """Cek apakah file gambar ada dan dapat dibaca."""
    if not os.path.exists(img_path):
        print(f"  ❌ File tidak ditemukan: {nama_file}")
        return False
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ❌ Gagal membaca (corrupt?): {nama_file}")
        return False
    h, w = img.shape
    if h < 50 or w < 50:
        print(f"  ⚠️  Resolusi terlalu kecil ({w}×{h}): {nama_file}")
    return True

# -----------------------------------------------------------------------------
# PROSES UTAMA
# -----------------------------------------------------------------------------
wajah_dewasa_latih = []
wajah_dewasa_uji   = []
wajah_kecil_uji    = []
labels             = []

if not os.path.exists(PATH_KELOMPOK):
    raise FileNotFoundError(
        f"❌ Folder '{PATH_KELOMPOK}' tidak ditemukan!\n"
        f"   Buat folder tersebut dan isi dengan subfolder orang_0, orang_1, dst."
    )

folder_list = sorted([
    f for f in os.listdir(PATH_KELOMPOK)
    if os.path.isdir(os.path.join(PATH_KELOMPOK, f))
])

print("=" * 60)
print("🔒 ENKRIPSI DATASET KELOMPOK → .NPZ")
print("=" * 60)
print(f"📁 Folder ditemukan: {len(folder_list)} subfolder")
print(f"👥 Memproses maksimal {MAX_ANGGOTA} anggota...")
print()

berhasil = 0
for idx, nama_folder in enumerate(folder_list):
    if idx >= MAX_ANGGOTA:
        print(f"  ℹ️  Batas {MAX_ANGGOTA} anggota tercapai.")
        break

    label_id = LABEL_MULAI + idx
    path_orang = os.path.join(PATH_KELOMPOK, nama_folder)
    nama_anggota = KAMUS_NAMA.get(label_id, f"Anggota {idx+1}")

    path_kecil     = os.path.join(path_orang, 'kecil.jpg')
    path_dewasa_1  = os.path.join(path_orang, 'dewasa_bawaan.jpg')
    path_dewasa_2  = os.path.join(path_orang, 'dewasa_uji.jpg')

    # Coba ekstensi .png jika .jpg tidak ada
    for p_var in [path_kecil, path_dewasa_1, path_dewasa_2]:
        if not os.path.exists(p_var):
            p_png = p_var.replace('.jpg', '.png')
            if os.path.exists(p_png):
                p_var = p_png  # fallback ke .png

    print(f"👤 [{idx}] {nama_anggota} (ID: {label_id}) — Folder: {nama_folder}")

    # Validasi semua file
    ok_kecil    = validasi_gambar(path_kecil, 'kecil.jpg')
    ok_dewasa_1 = validasi_gambar(path_dewasa_1, 'dewasa_bawaan.jpg')
    ok_dewasa_2 = validasi_gambar(path_dewasa_2, 'dewasa_uji.jpg')

    if not (ok_kecil and ok_dewasa_1 and ok_dewasa_2):
        print(f"  ⏭️  Melewati {nama_anggota} karena file tidak lengkap.\n")
        continue

    # Preprocessing
    vec_kecil    = preprocess_lintas_usia(path_kecil)
    vec_dewasa_1 = preprocess_dewasa(path_dewasa_1)
    vec_dewasa_2 = preprocess_dewasa(path_dewasa_2)

    if vec_kecil is None or vec_dewasa_1 is None or vec_dewasa_2 is None:
        print(f"  ⏭️  Melewati {nama_anggota} karena error preprocessing.\n")
        continue

    wajah_kecil_uji.append(vec_kecil)
    wajah_dewasa_latih.append(vec_dewasa_1)
    wajah_dewasa_uji.append(vec_dewasa_2)
    labels.append(label_id)

    print(f"  ✅ Berhasil diproses. Shape: {vec_dewasa_1.shape}\n")
    berhasil += 1

# -----------------------------------------------------------------------------
# SIMPAN KE FILE .NPZ
# -----------------------------------------------------------------------------
if berhasil == 0:
    raise RuntimeError("❌ Tidak ada data yang berhasil diproses!")

np.savez(
    NAMA_OUTPUT,
    X_latih      = np.array(wajah_dewasa_latih, dtype=np.float32),
    X_test_sama  = np.array(wajah_dewasa_uji,   dtype=np.float32),
    X_test_lintas= np.array(wajah_kecil_uji,    dtype=np.float32),
    y            = np.array(labels,              dtype=np.int32)
)

print("=" * 60)
print(f"💾 SELESAI! File disimpan: '{NAMA_OUTPUT}'")
print(f"👥 Total anggota berhasil: {berhasil}")
print(f"📦 Shape X_latih:       {np.array(wajah_dewasa_latih).shape}")
print(f"📦 Shape X_test_sama:   {np.array(wajah_dewasa_uji).shape}")
print(f"📦 Shape X_test_lintas: {np.array(wajah_kecil_uji).shape}")
print(f"🏷️  Labels:              {labels}")
print("=" * 60)
print()
print("LANGKAH SELANJUTNYA:")
print(f"1. Upload '{NAMA_OUTPUT}' ke Google Colab")
print("2. Jalankan skrip Colab (Tahap 2)")
```

---

## 8. IMPLEMENTASI KODE LENGKAP — TAHAP 2: GOOGLE COLAB

### 8.1 Instalasi Library di Colab

```python
# Jalankan di cell pertama Colab
!pip install -q scikit-learn opencv-python-headless matplotlib numpy
```

### 8.2 Skrip Utama Google Colab (Versi Super Lengkap)

```python
# =============================================================================
# SKRIP COLAB: Sistem Face Comparison Berbasis PCA (Eigenfaces)
# Upload 'privasi_kelompok_100x100.npz' ke Colab sebelum menjalankan ini
# =============================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BAGIAN 1: KONFIGURASI SISTEM
# =============================================================================

# Kamus nama anggota — WAJIB disesuaikan urutan folder alfabetis Anda!
KAMUS_NAMA = {
    40: "Anggota Kelompok 1",
    41: "Anggota Kelompok 2",
    42: "Anggota Kelompok 3",
    43: "Anggota Kelompok 4",
    44: "Anggota Kelompok 5",
    # Tambahkan hingga 49 jika ada 10 anggota
}

# Konfigurasi PCA
N_KOMPONEN_PCA = 50       # Jumlah komponen utama yang digunakan

# Threshold kemiripan (untuk interpretasi hasil)
THRESHOLD_EUCLIDEAN = 15.0   # Jarak ≤ 15 → "Mirip" (tuning berdasarkan hasil)
THRESHOLD_COSINE    = 0.70   # Similarity ≥ 0.70 → "Mirip"

print("⚙️  Konfigurasi sistem dimuat.")
print(f"   PCA Components: {N_KOMPONEN_PCA}")
print(f"   Threshold Euclidean: ≤ {THRESHOLD_EUCLIDEAN}")
print(f"   Threshold Cosine: ≥ {THRESHOLD_COSINE}")

# =============================================================================
# BAGIAN 2: LOAD DATASET OLIVETTI (BENCHMARK)
# =============================================================================

print("\n" + "="*60)
print("📥 MEMUAT DATASET BENCHMARK OLIVETTI...")
print("="*60)

olivetti = fetch_olivetti_faces()
X_pondasi = []
y_pondasi = olivetti.target   # Labels: 0–39

for img in olivetti.images:
    # Olivetti asli 64×64, resize ke 100×100 agar konsisten dengan dataset kelompok
    X_pondasi.append(cv2.resize(img, (100, 100)).flatten())

X_pondasi = np.array(X_pondasi, dtype=np.float32)

print(f"✅ Olivetti dimuat: {X_pondasi.shape[0]} gambar × {X_pondasi.shape[1]} fitur")
print(f"   40 orang, masing-masing 10 foto. Label: {np.unique(y_pondasi)}")

# =============================================================================
# BAGIAN 3: LOAD DATASET KELOMPOK DARI .NPZ
# =============================================================================

print("\n" + "="*60)
print("🔒 MEMUAT DATASET KELOMPOK DARI .NPZ...")
print("="*60)

NAMA_NPZ = 'privasi_kelompok_100x100.npz'

try:
    data_kelompok = np.load(NAMA_NPZ)
    X_latih_kelompok  = data_kelompok['X_latih']
    y_latih_kelompok  = data_kelompok['y']
    X_test_sama       = data_kelompok['X_test_sama']
    X_test_lintas     = data_kelompok['X_test_lintas']

    n_anggota = len(y_latih_kelompok)
    print(f"✅ Dataset kelompok dimuat!")
    print(f"   Anggota: {n_anggota} orang (ID: {list(y_latih_kelompok)})")
    print(f"   X_latih shape:       {X_latih_kelompok.shape}")
    print(f"   X_test_sama shape:   {X_test_sama.shape}")
    print(f"   X_test_lintas shape: {X_test_lintas.shape}")
    MODE_SIMULASI = False

except FileNotFoundError:
    print(f"⚠️  File '{NAMA_NPZ}' tidak ditemukan!")
    print("   Mengaktifkan MODE SIMULASI dengan data dummy...")
    print("   ⚠️  Akurasi akan acak karena data tidak nyata.")
    X_latih_kelompok  = np.random.rand(5, 10000).astype(np.float32)
    y_latih_kelompok  = np.array([40, 41, 42, 43, 44], dtype=np.int32)
    X_test_sama       = np.random.rand(5, 10000).astype(np.float32)
    X_test_lintas     = np.random.rand(5, 10000).astype(np.float32)
    n_anggota         = 5
    MODE_SIMULASI     = True

# Update kamus nama berdasarkan data yang dimuat
for label in y_latih_kelompok:
    if label not in KAMUS_NAMA:
        KAMUS_NAMA[label] = f"Anggota {label - 39}"

# =============================================================================
# BAGIAN 4: GABUNGKAN DATA & TRAINING PCA
# =============================================================================

print("\n" + "="*60)
print("🔗 MENGGABUNGKAN DATA TRAINING...")
print("="*60)

X_train_total = np.vstack((X_pondasi, X_latih_kelompok))
y_train_total = np.concatenate((y_pondasi, y_latih_kelompok))

print(f"✅ X_train_total shape: {X_train_total.shape}")
print(f"   ({X_pondasi.shape[0]} Olivetti + {X_latih_kelompok.shape[0]} kelompok)")
print(f"   Total label unik: {len(np.unique(y_train_total))}")

print("\n" + "="*60)
print(f"🤖 MELATIH PCA ({N_KOMPONEN_PCA} komponen)...")
print("="*60)

pca = PCA(n_components=N_KOMPONEN_PCA, whiten=True, random_state=42)
pca.fit(X_train_total)

# Hitung variance yang dijelaskan oleh 50 komponen
variance_explained = np.sum(pca.explained_variance_ratio_) * 100
print(f"✅ PCA selesai dilatih!")
print(f"   Variance yang dijelaskan oleh {N_KOMPONEN_PCA} komponen: {variance_explained:.1f}%")

# Transformasi semua dataset ke Eigenface Space
X_train_pca      = pca.transform(X_train_total)
X_test_sama_pca  = pca.transform(X_test_sama)
X_test_lintas_pca= pca.transform(X_test_lintas)

print(f"   Dimensi setelah PCA: {X_train_total.shape[1]} → {N_KOMPONEN_PCA} dimensi")

# =============================================================================
# BAGIAN 5: FUNGSI PERHITUNGAN KEMIRIPAN DETAIL
# =============================================================================

def hitung_kemiripan(vec_query: np.ndarray, X_db: np.ndarray, y_db: np.ndarray):
    """
    Hitung kemiripan antara satu vektor query terhadap seluruh database.
    
    Returns: dict berisi hasil Euclidean dan Cosine
    """
    # Euclidean Distance
    jarak_euclidean = np.linalg.norm(X_db - vec_query, axis=1)
    idx_euclidean   = np.argmin(jarak_euclidean)
    
    # Cosine Similarity
    sim_cosine      = cosine_similarity(X_db, vec_query.reshape(1, -1)).flatten()
    idx_cosine      = np.argmax(sim_cosine)
    
    # Top-3 Euclidean
    top3_euclidean_idx = np.argsort(jarak_euclidean)[:3]
    top3_euclidean = [(y_db[i], jarak_euclidean[i]) for i in top3_euclidean_idx]
    
    # Top-3 Cosine
    top3_cosine_idx = np.argsort(sim_cosine)[::-1][:3]
    top3_cosine = [(y_db[i], sim_cosine[i]) for i in top3_cosine_idx]
    
    return {
        'euclidean': {
            'label_pred': y_db[idx_euclidean],
            'score': jarak_euclidean[idx_euclidean],
            'top3': top3_euclidean,
        },
        'cosine': {
            'label_pred': y_db[idx_cosine],
            'score': sim_cosine[idx_cosine],
            'top3': top3_cosine,
        }
    }

def label_ke_nama(label: int) -> str:
    """Konversi label ID ke nama yang mudah dibaca."""
    if label in KAMUS_NAMA:
        return KAMUS_NAMA[label]
    return f"Olivetti ID-{label:02d}"

# =============================================================================
# BAGIAN 6: ENGINE EVALUASI LENGKAP
# =============================================================================

def evaluasi_skenario(X_uji_pca: np.ndarray, 
                       nama_skenario: str,
                       deskripsi: str) -> dict:
    """
    Evaluasi kemiripan untuk satu skenario (sama usia atau lintas usia).
    
    Args:
        X_uji_pca: Array vektor uji setelah transformasi PCA
        nama_skenario: Nama skenario (untuk ditampilkan)
        deskripsi: Deskripsi tambahan skenario
        
    Returns: dict berisi semua hasil evaluasi
    """
    print()
    print("=" * 70)
    print(f"📊 EVALUASI SKENARIO: {nama_skenario}")
    print(f"   {deskripsi}")
    print("=" * 70)
    
    benar_euclidean = 0
    benar_cosine    = 0
    total_uji       = len(X_uji_pca)
    
    detail_per_orang = []
    
    for i in range(total_uji):
        label_asli   = y_latih_kelompok[i]
        nama_asli    = label_ke_nama(label_asli)
        
        hasil = hitung_kemiripan(X_uji_pca[i], X_train_pca, y_train_total)
        
        pred_euclidean = hasil['euclidean']['label_pred']
        dist_euclidean = hasil['euclidean']['score']
        top3_euclidean = hasil['euclidean']['top3']
        
        pred_cosine    = hasil['cosine']['label_pred']
        sim_cosine     = hasil['cosine']['score']
        top3_cosine    = hasil['cosine']['top3']
        
        nama_pred_euclidean = label_ke_nama(pred_euclidean)
        nama_pred_cosine    = label_ke_nama(pred_cosine)
        
        ok_euclidean = (pred_euclidean == label_asli)
        ok_cosine    = (pred_cosine == label_asli)
        
        if ok_euclidean: benar_euclidean += 1
        if ok_cosine:    benar_cosine    += 1
        
        status_e = "✅ BENAR" if ok_euclidean else "❌ SALAH"
        status_c = "✅ BENAR" if ok_cosine    else "❌ SALAH"
        
        # Interpretasi skor
        interp_e = "Sangat Mirip" if dist_euclidean <= THRESHOLD_EUCLIDEAN else "Kurang Mirip"
        interp_c = "Sangat Mirip" if sim_cosine >= THRESHOLD_COSINE        else "Kurang Mirip"
        
        print(f"\n  👤 Target: {nama_asli} (ID: {label_asli})")
        print(f"  ┌─ EUCLIDEAN {status_e}")
        print(f"  │   Prediksi : {nama_pred_euclidean} (ID: {pred_euclidean})")
        print(f"  │   Jarak    : {dist_euclidean:.4f} → {interp_e}")
        print(f"  │   Top-3    : ", end="")
        print(", ".join([f"{label_ke_nama(lbl)}({d:.2f})" for lbl, d in top3_euclidean]))
        print(f"  └─ COSINE {status_c}")
        print(f"      Prediksi : {nama_pred_cosine} (ID: {pred_cosine})")
        print(f"      Similarity: {sim_cosine:.4f} → {interp_c}")
        print(f"      Top-3    : ", end="")
        print(", ".join([f"{label_ke_nama(lbl)}({s:.3f})" for lbl, s in top3_cosine]))
        
        detail_per_orang.append({
            'label_asli': label_asli,
            'nama_asli': nama_asli,
            'pred_euclidean': pred_euclidean,
            'dist_euclidean': dist_euclidean,
            'ok_euclidean': ok_euclidean,
            'pred_cosine': pred_cosine,
            'sim_cosine': sim_cosine,
            'ok_cosine': ok_cosine,
        })
    
    akurasi_euclidean = (benar_euclidean / total_uji) * 100
    akurasi_cosine    = (benar_cosine    / total_uji) * 100
    
    print()
    print("─" * 70)
    print(f"📈 KESIMPULAN SKENARIO: {nama_skenario}")
    print(f"   ▶ Benar / Total          : {benar_euclidean}/{total_uji} (Euclidean) | {benar_cosine}/{total_uji} (Cosine)")
    print(f"   ▶ Akurasi Euclidean (L2) : {akurasi_euclidean:.1f}%")
    print(f"   ▶ Akurasi Cosine         : {akurasi_cosine:.1f}%")
    
    if akurasi_euclidean > akurasi_cosine:
        print(f"   🏆 Euclidean unggul untuk skenario ini (+{akurasi_euclidean - akurasi_cosine:.1f}%)")
    elif akurasi_cosine > akurasi_euclidean:
        print(f"   🏆 Cosine unggul untuk skenario ini (+{akurasi_cosine - akurasi_euclidean:.1f}%)")
    else:
        print(f"   🤝 Kedua metrik setara!")
    print("─" * 70)
    
    return {
        'nama_skenario': nama_skenario,
        'akurasi_euclidean': akurasi_euclidean,
        'akurasi_cosine': akurasi_cosine,
        'detail': detail_per_orang,
    }

# =============================================================================
# BAGIAN 7: JALANKAN EVALUASI KEDUA SKENARIO
# =============================================================================

hasil_skenario_a = evaluasi_skenario(
    X_test_sama_pca,
    "SKENARIO A — USIA SAMA",
    "Membandingkan foto dewasa_uji dengan database dewasa_bawaan"
)

hasil_skenario_b = evaluasi_skenario(
    X_test_lintas_pca,
    "SKENARIO B — LINTAS USIA",
    "Membandingkan foto masa kecil dengan database foto dewasa"
)

# =============================================================================
# BAGIAN 8: LAPORAN PERBANDINGAN FINAL
# =============================================================================

print()
print("=" * 70)
print("🏁 LAPORAN PERBANDINGAN FINAL: SKENARIO A vs SKENARIO B")
print("=" * 70)
print()
print(f"  {'Metrik':<30} {'Skenario A':>15} {'Skenario B':>15} {'Gap':>10}")
print(f"  {'-'*70}")
print(f"  {'Akurasi Euclidean (L2)':<30} "
      f"{hasil_skenario_a['akurasi_euclidean']:>13.1f}% "
      f"{hasil_skenario_b['akurasi_euclidean']:>13.1f}% "
      f"{hasil_skenario_a['akurasi_euclidean'] - hasil_skenario_b['akurasi_euclidean']:>+9.1f}%")
print(f"  {'Akurasi Cosine Similarity':<30} "
      f"{hasil_skenario_a['akurasi_cosine']:>13.1f}% "
      f"{hasil_skenario_b['akurasi_cosine']:>13.1f}% "
      f"{hasil_skenario_a['akurasi_cosine'] - hasil_skenario_b['akurasi_cosine']:>+9.1f}%")
print()

# Analisis dominansi metrik
e_gap = hasil_skenario_a['akurasi_euclidean'] - hasil_skenario_b['akurasi_euclidean']
c_gap = hasil_skenario_a['akurasi_cosine']    - hasil_skenario_b['akurasi_cosine']

print("📝 ANALISIS OTOMATIS:")
print()
if e_gap > 0:
    print(f"  ▸ Skenario A unggul {e_gap:.1f}% atas Skenario B (Euclidean)")
    print(f"    → Membuktikan bahwa perubahan usia mengurangi akurasi PCA.")
else:
    print(f"  ⚠️  Skenario B tidak jauh di bawah Skenario A (Euclidean)")
    print(f"    → Kemungkinan foto kecil kualitasnya baik atau preprocessing efektif.")

print()
if c_gap > c_gap * 0.8:
    print(f"  ▸ Cosine Similarity lebih stabil untuk Skenario B")
    print(f"    → Cocok untuk cross-age karena mengabaikan perbedaan magnitude/kecerahan.")

print()
if MODE_SIMULASI:
    print("  ⚠️  PERHATIAN: Hasil ini adalah simulasi data dummy.")
    print("      Upload file .npz nyata untuk hasil yang valid!")

print("=" * 70)
```

---

## 9. VISUALISASI & DEBUGGING

### 9.1 Tambahkan Blok Visualisasi Setelah Kode Evaluasi

```python
# =============================================================================
# VISUALISASI 1: Eigenfaces (50 Komponen Utama)
# =============================================================================

fig, axes = plt.subplots(5, 10, figsize=(20, 10))
fig.suptitle('50 Eigenfaces — Komponen Utama PCA\n(Pola Wajah Paling Berpengaruh)', 
             fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < N_KOMPONEN_PCA:
        eigenface = pca.components_[i].reshape(100, 100)
        ax.imshow(eigenface, cmap='gray', interpolation='nearest')
        ax.set_title(f'EF-{i+1}', fontsize=7)
    ax.axis('off')

plt.tight_layout()
plt.savefig('eigenfaces_50.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 Eigenfaces disimpan: eigenfaces_50.png")

# =============================================================================
# VISUALISASI 2: Variance yang Dijelaskan Per Komponen
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Analisis Variance PCA', fontsize=13, fontweight='bold')

# Individual variance
komponen_idx = np.arange(1, N_KOMPONEN_PCA + 1)
ax1.bar(komponen_idx, pca.explained_variance_ratio_ * 100, 
        color='steelblue', alpha=0.8, edgecolor='white')
ax1.set_xlabel('Komponen Utama (Eigenface)')
ax1.set_ylabel('Variance yang Dijelaskan (%)')
ax1.set_title('Variance Per Komponen')
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1% threshold')
ax1.legend()

# Cumulative variance
cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
ax2.plot(komponen_idx, cumulative_var, 'b-o', markersize=4)
ax2.fill_between(komponen_idx, cumulative_var, alpha=0.2)
ax2.axhline(y=variance_explained, color='red', linestyle='--', 
            label=f'Titik {N_KOMPONEN_PCA} komponen: {variance_explained:.1f}%')
ax2.set_xlabel('Jumlah Komponen')
ax2.set_ylabel('Kumulatif Variance (%)')
ax2.set_title('Kumulatif Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 Analisis variance disimpan: pca_variance_analysis.png")

# =============================================================================
# VISUALISASI 3: Rekonstruksi Wajah (Orisinal vs Hasil PCA)
# Menunjukkan kualitas representasi PCA
# =============================================================================

fig, axes = plt.subplots(3, n_anggota, figsize=(3 * n_anggota, 9))
fig.suptitle('Rekonstruksi Wajah: Orisinal vs Rekonstruksi PCA', 
             fontsize=13, fontweight='bold')

for i in range(n_anggota):
    nama = KAMUS_NAMA.get(y_latih_kelompok[i], f"Orang {i}")
    
    # Gambar asli training (dewasa_bawaan)
    img_asli = X_latih_kelompok[i].reshape(100, 100)
    
    # Rekonstruksi melalui PCA inverse transform
    vec_pca        = pca.transform(X_latih_kelompok[i:i+1])
    img_rekon      = pca.inverse_transform(vec_pca).reshape(100, 100)
    
    # Gambar uji lintas usia (kecil)
    img_kecil = X_test_lintas[i].reshape(100, 100)
    
    axes[0, i].imshow(img_asli, cmap='gray')
    axes[0, i].set_title(f'{nama}\n(Dewasa Train)', fontsize=8)
    axes[0, i].axis('off')
    
    axes[1, i].imshow(np.clip(img_rekon, 0, 1), cmap='gray')
    axes[1, i].set_title('Rekonstruksi PCA', fontsize=8)
    axes[1, i].axis('off')
    
    axes[2, i].imshow(img_kecil, cmap='gray')
    axes[2, i].set_title('Foto Kecil\n(Lintas Usia)', fontsize=8)
    axes[2, i].axis('off')

plt.tight_layout()
plt.savefig('rekonstruksi_wajah.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 Rekonstruksi wajah disimpan: rekonstruksi_wajah.png")

# =============================================================================
# VISUALISASI 4: Grafik Perbandingan Akurasi Kedua Skenario
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

skenario_labels = ['Skenario A\n(Usia Sama)', 'Skenario B\n(Lintas Usia)']
akurasi_euclidean = [hasil_skenario_a['akurasi_euclidean'], hasil_skenario_b['akurasi_euclidean']]
akurasi_cosine    = [hasil_skenario_a['akurasi_cosine'],    hasil_skenario_b['akurasi_cosine']]

x = np.arange(len(skenario_labels))
width = 0.3

bars1 = ax.bar(x - width/2, akurasi_euclidean, width, 
               label='Euclidean (L2)', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, akurasi_cosine, width, 
               label='Cosine Similarity', color='coral', alpha=0.85)

# Label nilai di atas bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=11)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=11)

ax.set_xlabel('Skenario', fontsize=12)
ax.set_ylabel('Akurasi (%)', fontsize=12)
ax.set_title('Perbandingan Akurasi: Skenario A (Usia Sama) vs B (Lintas Usia)\nMetode Euclidean vs Cosine Similarity', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(skenario_labels, fontsize=12)
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.axhline(y=50, color='green', linestyle='--', alpha=0.4, label='Baseline 50%')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('perbandingan_akurasi.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾 Grafik perbandingan disimpan: perbandingan_akurasi.png")
```

---

## 10. INTERPRETASI SKOR KEMIRIPAN

### 10.1 Tabel Interpretasi Euclidean Distance

| Rentang Jarak | Kategori | Keterangan |
|---|---|---|
| 0.0 — 5.0 | 🟢 Identik | Sangat mungkin foto yang sama |
| 5.0 — 10.0 | 🟢 Sangat Mirip | Orang yang sama, kondisi berbeda |
| 10.0 — 15.0 | 🟡 Mirip | Kemungkinan orang sama |
| 15.0 — 20.0 | 🟠 Kemungkinan Beda | Perlu verifikasi lebih lanjut |
| > 20.0 | 🔴 Berbeda | Kemungkinan besar orang berbeda |

> **Catatan:** Threshold ini bersifat relatif dan bergantung pada dataset Anda. Kalibrasi berdasarkan distribusi jarak yang muncul di output.

### 10.2 Tabel Interpretasi Cosine Similarity

| Rentang Similarity | Kategori | Keterangan |
|---|---|---|
| 0.90 — 1.00 | 🟢 Identik | Pola wajah hampir sama persis |
| 0.80 — 0.90 | 🟢 Sangat Mirip | Orang yang sama dengan variasi |
| 0.70 — 0.80 | 🟡 Mirip | Kemungkinan besar orang sama |
| 0.60 — 0.70 | 🟠 Meragukan | Perlu konfirmasi |
| < 0.60 | 🔴 Berbeda | Kemungkinan besar berbeda |

### 10.3 Kapan Menggunakan Euclidean vs Cosine?

```
Gunakan EUCLIDEAN (L2) untuk:
  ✓ Foto yang diambil dalam kondisi pencahayaan seragam
  ✓ Skenario A (usia sama, kondisi terkontrol)
  ✓ Dataset yang sudah di-whitening oleh PCA (whiten=True)

Gunakan COSINE SIMILARITY untuk:
  ✓ Foto dari era berbeda (kualitas, kecerahan berbeda)
  ✓ Skenario B (lintas usia)
  ✓ Kondisi pencahayaan bervariasi antar foto
  ✓ Foto selfie vs foto formal
```

---

## 11. ACCEPTANCE CRITERIA & TARGET OUTPUT

### 11.1 Kriteria Keberhasilan Minimum

| Kriteria | Keterangan | Status |
|---|---|---|
| Kode berjalan tanpa error | Seluruh pipeline lolos tanpa exception | ✅ Wajib |
| X_train_total terbentuk | Shape: [405 × 10.000] (untuk 5 anggota) | ✅ Wajib |
| PCA berhasil dilatih | Variance explained ≥ 80% | ✅ Wajib |
| Output evaluasi Skenario A | Muncul akurasi Euclidean & Cosine | ✅ Wajib |
| Output evaluasi Skenario B | Muncul akurasi Euclidean & Cosine | ✅ Wajib |
| Skenario A > Skenario B | Akurasi A lebih tinggi dari B | ✅ Wajib (dibuktikan) |
| Visualisasi eigenfaces | Gambar pola eigenface tampil | ✅ Direkomendasikan |

### 11.2 Target Output Terminal Google Colab

```
============================================================
📊 EVALUASI SKENARIO: SKENARIO A — USIA SAMA
   Membandingkan foto dewasa_uji dengan database dewasa_bawaan
============================================================

  👤 Target: Anggota Kelompok 1 (ID: 40)
  ┌─ EUCLIDEAN ✅ BENAR
  │   Prediksi : Anggota Kelompok 1 (ID: 40)
  │   Jarak    : 8.2341 → Sangat Mirip
  │   Top-3    : AnggotaKelompok1(8.23), OlivettiID-12(14.11), AnggotaKelompok3(16.44)
  └─ COSINE ✅ BENAR
      Prediksi : Anggota Kelompok 1 (ID: 40)
      Similarity: 0.8712 → Sangat Mirip
      Top-3    : AnggotaKelompok1(0.871), OlivettiID-12(0.742), OlivettiID-07(0.698)
  ...

  ▶ Akurasi Euclidean (L2) : 80.0%
  ▶ Akurasi Cosine         : 100.0%

============================================================
📊 EVALUASI SKENARIO: SKENARIO B — LINTAS USIA
   Membandingkan foto masa kecil dengan database foto dewasa
============================================================
  ...
  ▶ Akurasi Euclidean (L2) : 40.0%
  ▶ Akurasi Cosine         : 60.0%
```

---

## 12. TROUBLESHOOTING & FAQ

### 12.1 Error Umum & Solusinya

#### ❌ `FileNotFoundError: Folder 'dataset_kelompok' wajib dibuat terlebih dahulu!`
**Penyebab:** Skrip lokal dijalankan di direktori yang salah.
**Solusi:**
```python
import os
print(os.getcwd())  # Cek direktori kerja saat ini
# Pastikan folder dataset_kelompok ADA di direktori ini
```

#### ❌ `cv2.imread()` mengembalikan `None`
**Penyebab:** Path file salah, ekstensi file salah, atau file corrupt.
**Solusi:**
```python
# Cek nama file sebenarnya di folder
import os
print(os.listdir('dataset_kelompok/orang_0'))
# Pastikan ada 'kecil.jpg', 'dewasa_bawaan.jpg', 'dewasa_uji.jpg'
```

#### ❌ `ValueError: operands could not be broadcast together`
**Penyebab:** Ukuran gambar tidak konsisten — ada gambar yang tidak berhasil di-resize.
**Solusi:** Pastikan semua array di `wajah_dewasa_latih` memiliki shape `(10000,)`.

#### ❌ Output Skenario A dan B sama-sama rendah (di bawah 40%)
**Penyebab:** Foto kelompok terlalu berbeda dari format Olivetti, atau pencahayaan sangat ekstrem.
**Solusi — Coba naikkan jumlah komponen PCA:**
```python
N_KOMPONEN_PCA = 100  # Naikkan dari 50 ke 100
```
**Solusi alternatif — Tambah foto training per orang:**
```python
# Masukkan lebih dari 1 foto dewasa_bawaan per orang ke training
# dengan memodifikasi skrip lokal untuk menyimpan multiple foto
```

#### ❌ `MemoryError` saat training PCA
**Penyebab:** RAM Google Colab tidak cukup untuk matriks besar.
**Solusi:**
```python
# Gunakan IncrementalPCA untuk dataset besar
from sklearn.decomposition import IncrementalPCA
pca = IncrementalPCA(n_components=50, batch_size=50)
pca.fit(X_train_total)
```

#### ❌ Skenario A hasilnya lebih buruk dari Skenario B
**Penyebab:** Foto `dewasa_uji.jpg` sangat berbeda dari `dewasa_bawaan.jpg` (ekspresi, angle, atau pencahayaan ekstrem).
**Solusi:** Pilih foto `dewasa_bawaan.jpg` dan `dewasa_uji.jpg` yang diambil dalam kondisi serupa.

### 12.2 FAQ

**Q: Mengapa menggunakan `whiten=True` pada PCA?**
A: Whitening memastikan setiap komponen memiliki variance = 1, sehingga tidak ada komponen yang mendominasi perhitungan jarak hanya karena memiliki nilai lebih besar. Ini penting untuk Euclidean Distance.

**Q: Apakah bisa menggunakan lebih dari 50 komponen PCA?**
A: Bisa. Uji dengan 100 atau 150 komponen. Namun, jika melebihi jumlah sampel training (405), PCA akan error. Gunakan `n_components ≤ min(n_samples, n_features)`.

**Q: Mengapa foto Olivetti di-resize dari 64×64 ke 100×100?**
A: Agar konsisten dengan foto kelompok yang berukuran 100×100. PCA membutuhkan semua vektor berukuran sama.

**Q: Apakah CLAHE wajib untuk foto dewasa juga?**
A: Tidak wajib, tapi bisa diterapkan jika foto dewasa memiliki kualitas bervariasi. Untuk konsistensi, bisa terapkan CLAHE ke semua foto.

**Q: Bagaimana cara meningkatkan akurasi Skenario B?**
A: Lihat Bagian 13 (Peningkatan Lanjutan) — Data Augmentation dan Local Binary Patterns adalah dua teknik paling efektif.

---

## 13. PENINGKATAN LANJUTAN (OPSIONAL)

Jika ingin meningkatkan performa sistem, terutama untuk Skenario B (lintas usia):

### 13.1 Data Augmentation untuk Memperkaya Training Set

Tambahkan variasi sintetis dari foto `dewasa_bawaan.jpg` untuk memperbesar dataset:

```python
def augmentasi_gambar(img_vector: np.ndarray) -> list:
    """
    Buat variasi foto dari satu vektor gambar.
    Menambah ketahanan model terhadap perbedaan kecil.
    """
    img = img_vector.reshape(100, 100)
    variasi = []
    
    # 1. Gambar asli
    variasi.append(img.flatten())
    
    # 2. Flip horizontal (cermin)
    variasi.append(np.fliplr(img).flatten())
    
    # 3. Sedikit lebih terang (gamma 0.85)
    variasi.append(np.power(img, 0.85).flatten())
    
    # 4. Sedikit lebih gelap (gamma 1.15)
    variasi.append(np.power(img, 1.15).flatten())
    
    # 5. Gaussian noise ringan
    noise = img + np.random.normal(0, 0.015, img.shape)
    variasi.append(np.clip(noise, 0, 1).flatten())
    
    return variasi

# Terapkan augmentasi ke foto training kelompok
X_latih_augmented = []
y_latih_augmented = []

for i, vec in enumerate(X_latih_kelompok):
    variasi = augmentasi_gambar(vec)
    X_latih_augmented.extend(variasi)
    y_latih_augmented.extend([y_latih_kelompok[i]] * len(variasi))

X_latih_aug = np.array(X_latih_augmented, dtype=np.float32)
y_latih_aug = np.array(y_latih_augmented, dtype=np.int32)

print(f"Dataset setelah augmentasi: {X_latih_aug.shape}")
# Gunakan X_latih_aug menggantikan X_latih_kelompok dalam vstack
```

### 13.2 Eksperimen Jumlah Komponen PCA

```python
# Uji berbagai jumlah komponen untuk menemukan yang terbaik
hasil_eksperimen = {}
for n_comp in [20, 50, 100, 150, 200]:
    pca_exp = PCA(n_components=n_comp, whiten=True, random_state=42)
    pca_exp.fit(X_train_total)
    
    X_tr_exp = pca_exp.transform(X_train_total)
    X_ts_exp = pca_exp.transform(X_test_sama)
    X_tl_exp = pca_exp.transform(X_test_lintas)
    
    # Hitung akurasi sederhana
    benar_e, benar_c = 0, 0
    for i in range(len(y_latih_kelompok)):
        dist_e = np.linalg.norm(X_tr_exp - X_ts_exp[i], axis=1)
        pred_e = y_train_total[np.argmin(dist_e)]
        if pred_e == y_latih_kelompok[i]: benar_e += 1
        
        sim_c = cosine_similarity(X_tr_exp, X_ts_exp[i:i+1]).flatten()
        pred_c = y_train_total[np.argmax(sim_c)]
        if pred_c == y_latih_kelompok[i]: benar_c += 1
    
    hasil_eksperimen[n_comp] = {
        'akurasi_e': benar_e / len(y_latih_kelompok) * 100,
        'akurasi_c': benar_c / len(y_latih_kelompok) * 100,
        'variance': np.sum(pca_exp.explained_variance_ratio_) * 100,
    }
    print(f"n={n_comp:>3} | Var={hasil_eksperimen[n_comp]['variance']:.1f}% | "
          f"Euc={hasil_eksperimen[n_comp]['akurasi_e']:.1f}% | "
          f"Cos={hasil_eksperimen[n_comp]['akurasi_c']:.1f}%")
```

### 13.3 Menambahkan Fitur LBP (Local Binary Pattern) — Tekstur Kulit

LBP adalah fitur yang menangkap **pola tekstur lokal** wajah, yang dapat membantu membedakan individu bahkan dengan perubahan usia:

```python
from skimage.feature import local_binary_pattern

def ekstrak_lbp(img_vector: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Ekstrak fitur Local Binary Pattern dari vektor gambar.
    LBP menangkap pola tekstur lokal yang relatif stabil terhadap perubahan usia.
    """
    img = (img_vector.reshape(100, 100) * 255).astype(np.uint8)
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    
    # Histogram LBP sebagai fitur
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)  # Normalisasi
    return hist

# Gabungkan fitur PCA + LBP
def buat_fitur_hybrid(X_pca: np.ndarray, X_original: np.ndarray) -> np.ndarray:
    """Gabungkan fitur PCA dengan fitur LBP (hybrid feature vector)."""
    X_lbp = np.array([ekstrak_lbp(x) for x in X_original])
    return np.hstack((X_pca, X_lbp))
```

---

## ℹ️ CATATAN AKHIR

### Batasan Sistem PCA-Eigenfaces untuk Lintas Usia

PCA adalah metode **linear dimensionality reduction**. Ia optimal ketika perbedaan antar wajah bersifat linear dan terdistribusi normal. Untuk lintas usia, perubahan wajah bersifat **non-linear** (pertumbuhan tulang, lemak, dll), sehingga PCA memiliki batasan inheren. Sistem ini sangat cocok untuk **membuktikan secara akademis** bahwa:

1. Pengenalan wajah usia sama jauh lebih mudah dari lintas usia.
2. Cosine Similarity lebih robust dari Euclidean untuk kondisi foto yang beragam.
3. Preprocessing lanjutan (CLAHE) membantu menjembatani gap antar era foto.

Untuk aplikasi produksi yang memerlukan akurasi lintas usia sangat tinggi, diperlukan metode **Deep Learning** (FaceNet, ArcFace, atau Age-Invariant Face Recognition models).

---

*Dokumen ini dibuat untuk keperluan akademik — Face Comparison System berbasis PCA (Eigenfaces)*
*Versi: 3.0 | Bahasa: Python 3.8+ | Platform: Google Colab / Jupyter Notebook*
