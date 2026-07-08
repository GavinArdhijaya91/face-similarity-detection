# 📑 PRODUCT REQUIREMENT DOCUMENT (PRD) v2.0
## Sistem Face Comparison Lintas Usia & Usia Sama Berbasis Principal Component Analysis (PCA)

---

## 1. SISTEM DOKUMENTASI & STRUKTUR DATASET

Untuk menghindari kesalahan pembacaan data (*I/O Error*) saat pelatihan di Google Colab, struktur direktori lokal/Drive kelompok **wajib** mengikuti bagan di bawah ini secara presisi.

### 1.1 Struktur Folder Lokal Sebelum Kompresi
Setiap folder anggota kelompok diisi tepat **3 foto** dengan format nama file huruf kecil:
```text
dataset_kelompok/
├── orang_0/
│   ├── kecil.jpg            <-- Foto masa kecil/balita (Uji Skenario B)
│   ├── dewasa_bawaan.jpg    <-- Foto dewasa 1 (Masuk ke Data Train)
│   └── dewasa_uji.jpg       <-- Foto dewasa 2 (Uji Skenario A)
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
└── orang_4/
    ├── kecil.jpg
    ├── dewasa_bawaan.jpg
    └── ...
```

---

## 2. ARSITEKTUR MATRIKS & MATEMATIKA DATA (SPECIFICATION)

*   **Ukuran Citra Input:** 100 × 100 piksel (Greyscale, 1 Channel).
*   **Dimensi Vektor flattened (N):** 100 x 100 = 10.000 dimensi per foto.
*   **Ukuran Matriks Database Gabungan (X_train):** [405 sampel x 10.000 fitur].
    *   *400 sampel berasal dari Olivetti Faces (ID 0-39) dan 5 sampel berasal dari foto dewasa_bawaan kelompok (ID 40-44).*
*   **Metrik Kemiripan (Similarity Metrics):**
    1.  **Euclidean Distance (L2 Norm):** Mengukur jarak geometris absolut antar-vektor bobot PCA. Semakin kecil nilainya (d mendekati 0), semakin mirip wajah tersebut.
    2.  **Cosine Similarity:** Mengukur sudut arah antar-vektor bobot PCA tanpa memedulikan panjang magnitudonya. Berguna untuk menekan bias kontras pencahayaan ekstrem. Semakin mendekati nilai 1.0, semakin mirip.

---

## 3. ARSITEKTUR PRIVASI DATA (DATA MASKING VIA .NPZ)

Untuk mematuhi etika privasi data kelompok, proyek tidak diperbolehkan menyimpan atau membaca file gambar mentah (`.jpg`/`.png`) saat *runtime* produksi di Google Colab.

*   **Mekanisme Enkripsi:** Seluruh gambar kelompok diekstrak ke dalam representasi larik (*array*) NumPy, diratakan menjadi vektor 1D berukuran `10.000` elemen.
*   **Output File:** Seluruh array dibungkus ke dalam format kompresi biner tunggal: `privasi_kelompok_100x100.npz`.
*   **Manfaat:** Setelah file `.npz` terbentuk, folder foto asli kelompok dapat dihapus secara permanen demi keamanan data privasi.

---

## 4. CODE IMPLEMENTATION PIPELINE

### 4.1 Tahap 1: Skrip Lokal Python (Penyiapan & Enkripsi .npz)
Jalankan skrip ini di PC/Laptop lokal Anda yang menyimpan folder `dataset_kelompok`. Skrip ini akan menghasilkan file biner `privasi_kelompok_100x100.npz`.

```python
import os
import cv2
import numpy as np

wajah_dewasa_latih = []
wajah_dewasa_uji = []
wajah_kecil_uji = []
labels = []

path_kelompok = 'dataset_kelompok'
label_mulai = 40  # ID Kelompok dimulai setelah ID Olivetti (0-39)

if not os.path.exists(path_kelompok):
    raise FileNotFoundError(f"Folder '{path_kelompok}' wajib dibuat terlebih dahulu!")

print("🔒 Memulai enkripsi privasi wajah kelompok ke biner...")
for idx, nama_folder in enumerate(sorted(os.listdir(path_kelompok))):
    path_orang = os.path.join(path_kelompok, nama_folder)
    if os.path.isdir(path_orang) and idx < 5:
        img_kecil = cv2.imread(os.path.join(path_orang, 'kecil.jpg'), cv2.IMREAD_GRAYSCALE)
        img_dewasa_1 = cv2.imread(os.path.join(path_orang, 'dewasa_bawaan.jpg'), cv2.IMREAD_GRAYSCALE)
        img_dewasa_2 = cv2.imread(os.path.join(path_orang, 'dewasa_uji.jpg'), cv2.IMREAD_GRAYSCALE)
        
        if img_kecil is not None and img_dewasa_1 is not None and img_dewasa_2 is not None:
            # Resize paksa ke dimensi target matriks 100x100 dan normalisasi 0-1
            wajah_kecil_uji.append(cv2.resize(img_kecil, (100, 100)).flatten() / 255.0)
            wajah_dewasa_latih.append(cv2.resize(img_dewasa_1, (100, 100)).flatten() / 255.0)
            wajah_dewasa_uji.append(cv2.resize(img_dewasa_2, (100, 100)).flatten() / 255.0)
            labels.append(label_mulai + idx)

# Simpan ke file .npz tunggal
np.savez('privasi_kelompok_100x100.npz', 
         X_latih=np.array(wajah_dewasa_latih, dtype=np.float32),
         X_test_sama=np.array(wajah_dewasa_uji, dtype=np.float32),
         X_test_lintas=np.array(wajah_kecil_uji, dtype=np.float32),
         y=np.array(labels, dtype=np.int32))

print("💾 Selesai! File 'privasi_kelompok_100x100.npz' siap diunggah ke Google Colab.")
```

---

### 4.2 Tahap 2: Skrip Utama Google Colab (Pelatihan & Perbandingan Model)
Unggah file `privasi_kelompok_100x100.npz` ke direktori Google Colab, lalu jalankan blok kode di bawah ini:

```python
import numpy as np
import cv2
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 1. MENDEFINISIKAN KAMUS NAMA ANGGOTA (Sesuaikan urutan folder alfabetis Anda)
kamus_nama = {
    40: "Anggota Kelompok 1",
    41: "Anggota Kelompok 2",
    42: "Anggota Kelompok 3",
    43: "Anggota Kelompok 4",
    44: "Anggota Kelompok 5"
}

# 2. LOAD DATASET BENCHMARK OLIVETTI
print("📥 Memuat database benchmark Olivetti...")
olivetti = fetch_olivetti_faces()
X_pondasi = []
y_pondasi = olivetti.target  # Label ID 0-39

for img in olivetti.images:
    # Manipulasi dimensi citra Olivetti dari 64x64 menjadi 100x100 matriks
    X_pondasi.append(cv2.resize(img, (100, 100)).flatten())
X_pondasi = np.array(X_pondasi, dtype=np.float32)

# 3. LOAD DATASET PRIVASI KELOMPOK DARI .NPZ
print("🔒 Memuat file privasi_kelompok_100x100.npz...")
try:
    data_kelompok = np.load('privasi_kelompok_100x100.npz')
    X_latih_kelompok = data_kelompok['X_latih']
    y_latih_kelompok = data_kelompok['y']
    X_test_sama = data_kelompok['X_test_sama']
    X_test_lintas = data_kelompok['X_test_lintas']
except FileNotFoundError:
    print("⚠️ File .npz tidak ditemukan! Mengaktifkan mode simulasi dummy untuk testing.")
    X_latih_kelompok = np.random.rand(5, 10000).astype(np.float32)
    y_latih_kelompok = np.array([40, 41, 42, 43, 44], dtype=np.int32)
    X_test_sama = np.random.rand(5, 10000).astype(np.float32)
    X_test_lintas = np.random.rand(5, 10000).astype(np.float32)

# Gabungkan data benchmark dan data internal untuk membentuk data latih total
X_train_total = np.vstack((X_pondasi, X_latih_kelompok))
y_train_total = np.concatenate((y_pondasi, y_latih_kelompok))

print(f"Dimensi akhir matriks latih: {X_train_total.shape} (100x100 pixels)")

# 4. EKSTRAKSI FITUR EIGENFACES MENGGUNAKAN PCA
print("\n🤖 Melatih Algoritma PCA untuk Ekstraksi Fitur...")
n_komponen = 50  # Mengambil 50 komponen utama wajah paling berpengaruh
pca = PCA(n_components=n_komponen, whiten=True).fit(X_train_total)

# Transformasi seluruh gambar ke ruang bobot fitur rendah (Eigenface Space)
X_train_pca = pca.transform(X_train_total)
X_test_sama_pca = pca.transform(X_test_sama)
X_test_lintas_pca = pca.transform(X_test_lintas)

# 5. ENGINE EVALUASI (EUCLIDEAN VS COSINE)
def evaluasi_sistem(X_uji_pca, nama_skenario):
    print(f"\n======================================================================")
    print(f"📊 EVALUASI SKENARIO: {nama_skenario}")
    print(f"======================================================================")
    
    benar_euclidean = 0
    benar_cosine = 0
    total_uji = len(X_uji_pca)
    
    for i in range(total_uji):
        label_asli = y_latih_kelompok[i]
        nama_asli = kamus_nama.get(label_asli, "Unknown")
        
        # --- METRIK 1: Euclidean Distance ---
        jarak_euclidean = np.linalg.norm(X_train_pca - X_uji_pca[i], axis=1)
        idx_euclidean = np.argmin(jarak_euclidean)
        pred_euclidean = y_train_total[idx_euclidean]
        nama_pred_euclidean = kamus_nama.get(pred_euclidean, f"Orang Olivetti ID {pred_euclidean}")
        
        status_euclidean = "✅" if pred_euclidean == label_asli else "❌"
        if pred_euclidean == label_asli: benar_euclidean += 1
        
        # --- METRIK 2: Cosine Similarity ---
        sim_cosine = cosine_similarity(X_train_pca, X_uji_pca[i].reshape(1, -1)).flatten()
        idx_cosine = np.argmax(sim_cosine)
        pred_cosine = y_train_total[idx_cosine]
        nama_pred_cosine = kamus_nama.get(pred_cosine, f"Orang Olivetti ID {pred_cosine}")
        
        status_cosine = "✅" if pred_cosine == label_asli else "❌"
        if pred_cosine == label_asli: benar_cosine += 1
        
        print(f"👤 Target: {nama_asli}")
        print(f"   └─ Metrik Euclidean -> Prediksi: {nama_pred_euclidean} {status_euclidean} (Dist: {jarak_euclidean[idx_euclidean]:.3f})")
        print(f"   └─ Metrik Cosine    -> Prediksi: {nama_pred_cosine} {status_cosine} (Sim: {sim_cosine[idx_cosine]:.3f})")
        
    print(f"\n📈 KESIMPULAN METRIK [{nama_skenario}]:")
    print(f"   ▶ Akurasi Akhir Metode L2/Euclidean : {(benar_euclidean / total_uji)*100:.1f}%")
    print(f"   ▶ Akurasi Akhir Metode Cosine       : {(benar_cosine / total_uji)*100:.1f}%")

# JALANKAN PROSES PERBANDINGAN KOMPARATIF
evaluasi_sistem(X_test_sama_pca, "USIA YANG SAMA (KONTROL)")
evaluasi_sistem(X_test_lintas_pca, "LINTAS USIA (TANTANGAN)")
```

---

## 5. ACCEPTANCE CRITERIA & OUTPUT TARGET

Tugas dinyatakan berhasil memuaskan jika terminal Google Colab berhasil mencetak kesimpulan metrik komparatif yang membuktikan:
1. Akurasi Skenario **Usia yang Sama** jauh lebih tinggi dibandingkan Skenario **Lintas Usia**.
2. Evaluasi performa ketahanan antara rumus **Euclidean** vs **Cosine Similarity** dalam menangani pergeseran piksel wajah akibat penuaan alami.
