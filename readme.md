# 🎯 Deteksi & Klasifikasi Multi-Kelas Ikan Hias
### Menggunakan Swin Transformer + RetinaNet

Proyek skripsi ini bertujuan untuk mengembangkan sistem deteksi dan klasifikasi otomatis untuk 10 jenis ikan hias menggunakan arsitektur Deep Learning modern.

---

## 📋 Daftar Isi
- [Deskripsi Proyek](#-deskripsi-proyek)
- [Struktur Folder](#-struktur-folder)
- [Persiapan Lingkungan (Setup)](#-persiapan-lingkungan-setup)
- [Cara Penggunaan](#-cara-penggunaan)
  - [1. Persiapan Data](#1-persiapan-data)
  - [2. Training Model](#2-training-model)
  - [3. Analisis & Evaluasi](#3-analisis--evaluasi)
- [Log Eksperimen](#-log-eksperimen)
- [Referensi](#-referensi)

---

## 📖 Deskripsi Proyek

Sistem ini dibangun untuk mengatasi tantangan dalam pemantauan ikan hias di akuarium, seperti variasi pencahayaan, refleksi kaca, dan oklusi (ikan saling menutupi).

**Fitur Utama:**
- **Deteksi Lokasi:** Menggunakan Bounding Box untuk melokalisir ikan.
- **Klasifikasi:** Mengenali 10 jenis ikan hias (Manfish, Betta, Goldfish, Discuss, Pleco, Guppy, Tiger Barb, Gurami, Molly, Swordtail).
- **Counting:** Menghitung jumlah ikan per jenis secara otomatis.

**Teknologi:**
- **Framework:** PyTorch & MMDetection
- **Backbone:** Swin Transformer (Swin-T/Swin-S)
- **Detector:** RetinaNet (One-stage detector)

---

## 📂 Struktur Folder

Berikut adalah struktur utama direktori proyek ini untuk membantu Anda menavigasi kode:

```
Deteksi_Jenis_Ikan/
├── dataset/                    # Folder penyimpanan dataset gambar & anotasi
├── mmdetection/                # Library MMDetection (submodule)
├── outputs_.../                # Folder output hasil training (log, checkpoint model)
├── Fish_Detection_With_Swin_Transformer.ipynb  # Notebook UTAMA untuk training
├── analisis_aja.py             # Script untuk analisis hasil & plotting grafik
├── test.py                     # Script utilitas untuk merapikan nama file dataset
├── EXPERIMENTS.md              # Catatan detail log eksperimen & hasil
└── readme.md                   # Dokumentasi proyek ini
```

---

## ⚙️ Persiapan Lingkungan (Setup)

Ikuti langkah-langkah ini untuk menyiapkan environment agar kode dapat berjalan dengan baik.

### Prasyarat
- OS: Windows / Linux
- GPU: NVIDIA (Disarankan untuk training)
- Anaconda / Miniconda

### Langkah Instalasi

1. **Buat Environment Conda**
   ```bash
   conda create -n ikan_env python=3.10
   conda activate ikan_env
   ```

2. **Install PyTorch** (Sesuaikan versi CUDA dengan perangkat Anda, contoh CUDA 12.1)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install MMDetection & Dependencies**
   ```bash
   # Install library pendukung
   conda install numpy scikit-learn jupyter cmake ninja -y
   pip install future tensorboard

   # Install MMCV (Computer Vision foundation for MMDetection)
   pip install -U openmim
   mim install mmengine
   mim install "mmcv>=2.0.0"

   # Install MMDetection
   mim install mmdet
   ```

4. **Verifikasi Instalasi**
   Jalankan python dan coba import:
   ```python
   import mmdet
   print(mmdet.__version__)
   ```

---

## 🚀 Cara Penggunaan

### 1. Persiapan Data
Jika Anda memiliki dataset baru, pastikan format penamaan file rapi. Gunakan script `test.py` untuk menstandarisasi nama file gambar dalam folder dataset.

```bash
# Edit variabel 'dataset_root' di dalam file test.py sesuai lokasi data Anda
python test.py
```

### 2. Training Model
Proses training dilakukan menggunakan Jupyter Notebook.
1. Buka **`Fish_Detection_With_Swin_Transformer.ipynb`**.
2. Sesuaikan path dataset dan konfigurasi di bagian awal notebook.
3. Jalankan sel secara berurutan (Run All).
4. Hasil training (model `.pth` dan log) akan tersimpan di folder `outputs_...`.

### 3. Analisis & Evaluasi
Setelah training selesai, gunakan script `analisis_aja.py` untuk membuat grafik evaluasi (Confusion Matrix, PR Curve, Training Stability).

1. Buka `analisis_aja.py`.
2. Edit variabel `WORK_DIR` agar mengarah ke folder output training Anda (misal: `./outputs_swin_retinanet_deepfish`).
3. Jalankan script:
   ```bash
   python analisis_aja.py
   ```
4. Hasil analisis akan disimpan di folder `reports_final_analysis` di dalam direktori output tersebut.

---

## 🧪 Log Eksperimen

Detail lengkap mengenai skenario pengujian, parameter yang digunakan, dan hasil analisis per eksperimen telah dipindahkan ke dokumen terpisah agar lebih rapi.

👉 **[Lihat EXPERIMENTS.md](./EXPERIMENTS.md)**

---

## 📚 Referensi

- **Swin Transformer:** Liu, Z. et al. (ICCV 2021)
- **Focal Loss:** Lin, T.Y. et al. (ICCV 2017)
- **MMDetection:** OpenMMLab Detection Toolbox
