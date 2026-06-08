[![Vercel](https://img.shields.io/badge/vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white)](https://detection-similarity-between-old-ph.vercel.app/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://face-verification-v1.streamlit.app/)

# FaceMatch: Deteksi Kemiripan Wajah Lintas Usia

> **Proyek Mata Kuliah Aljabar Linear Semester 2**  
> Sistem pendeteksi kemiripan wajah menggunakan Principal Component Analysis (PCA) dan Singular Value Decomposition (SVD). Fokus utama sistem ini adalah memverifikasi identitas antara foto masa kecil dan foto saat ini (Cross-Age Face Verification).

## 🚀 Fitur Utama

- **Pendeteksian Wajah Otomatis (Face Detection):** Menggunakan OpenCV Haar Cascades untuk memotong dan memusatkan wajah secara presisi.
- **Ekstraksi Fitur Geometris:** Mengonversi piksel wajah menjadi vektor matematika menggunakan *Eigenfaces* (PCA).
- **Penilaian Kemiripan (Similarity Metrics):** Menggunakan *Cosine Similarity* untuk mengukur kemiripan fitur wajah yang lebih tahan terhadap perubahan usia, serta membandingkannya dengan *Euclidean Distance*.
- **Visualisasi Model (Eigenspace):** Sistem otomatis merender komponen utama pembentuk wajah dalam bentuk grafik dan eigenfaces visual.

## 📁 Struktur Utama

Proyek ini dibangun menjadi dua jenis aplikasi berbasis Python:
1. **`streamlit_app/`**: Aplikasi antarmuka web interaktif menggunakan Streamlit untuk demo lokal.
2. **`fastapi_app/`**: Backend API yang tangguh menggunakan FastAPI, siap diintegrasikan dengan frontend Vercel/Next.js.

## 🛠️ Cara Menjalankan (Lokal)

### 1. Menjalankan Versi Streamlit
Buka terminal dan jalankan perintah berikut:
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
Aplikasi dapat diakses melalui browser di: `http://localhost:8501`

### 2. Menjalankan Versi FastAPI Backend
Buka terminal dan jalankan perintah berikut:
```bash
cd fastapi_app
pip install -r requirements.txt
uvicorn main:app --reload
```
Aplikasi dapat diakses melalui browser di: `http://localhost:8000`

## 🧠 Algoritma & Konsep Pembelajaran

Sistem ini mengeksplorasi tantangan **Semantic Gap** dalam Computer Vision:
- PCA dilatih menggunakan lebih dari 3.000 foto gabungan dari dataset Olivetti, LFW (Labeled Faces in the Wild), FG-NET Aging Database, dan sampel wajah pribadi.
- Hasil pengujian menunjukkan bahwa PCA murni memiliki akurasi 100% untuk mendeteksi wajah di usia yang sama (Skenario A).
- Untuk perbandingan lintas usia dari masa kecil ke dewasa (Skenario B), metrik *Cosine Similarity* terbukti jauh lebih stabil daripada metrik jarak Euclidean biasa.

## 📦 Teknologi yang Digunakan
- **Python 3**
- **NumPy & Scikit-Learn** (Komputasi Matriks, SVD, dan PCA)
- **OpenCV** (Computer Vision & Haar Cascade)
- **FastAPI** (Backend Web Server)
- **Streamlit** (UI/UX Interaktif)
