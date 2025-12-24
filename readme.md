# 🎓 Deteksi & Klasifikasi Multi-Kelas Ikan Hias

> **Proyek Skripsi**
> Sistem deteksi dan klasifikasi otomatis untuk 10 jenis ikan hias menggunakan Deep Learning.

---

## 📖 Tentang Proyek

Proyek ini merupakan bagian dari penelitian skripsi yang bertujuan untuk mengembangkan solusi cerdas dalam pemantauan ikan hias. Sistem ini mampu mendeteksi lokasi ikan dalam akuarium dan mengklasifikasikannya ke dalam 10 spesies berbeda secara real-time atau berbasis gambar.

**Spesies yang dideteksi:**
Manfish, Betta, Goldfish, Discuss, Pleco, Guppy, Tiger Barb, Gurami, Molly, Swordtail.

### 🏗️ Ekosistem Proyek
Sistem ini terdiri dari tiga komponen utama yang saling terhubung (Repository Terpisah):

1.  **[FishDetectionTrain](https://github.com/Blauzaa/FishDetectionTrain)** (Repo ini)
    *   **Fokus:** Eksperimen, Training Model, Evaluasi.
    *   **Tech:** PyTorch, MMDetection, Swin Transformer.
    *   **Output:** File model (`.pth`) yang siap dideploy.

2.  **[FishDetectionAPI](https://github.com/Blauzaa/FishDetectionAPI)**
    *   **Fokus:** Backend Service / API.
    *   **Fungsi:** Menjembatani komunikasi antara Aplikasi Mobile dengan Model Deep Learning. Menerima gambar dari app, memprosesnya dengan model, dan mengirimkan hasil deteksi kembali.

3.  **[FishDetectionApp](https://github.com/Blauzaa/FishDetectionApp)**
    *   **Fokus:** User Interface (Mobile App).
    *   **Fungsi:** Antarmuka bagi pengguna untuk mengambil gambar atau video ikan dan melihat hasil deteksi.

---

## 📂 Struktur Folder (FishDetectionTrain)

Berikut adalah struktur direktori penting dalam repository ini:

```
FishDetectionTrain/
├── dataset/                    # Direktori dataset (Gambar & Anotasi)
├── mmdetection/                # Submodule: Framework MMDetection
├── outputs_.../                # Direktori Output: Log training & Checkpoint Model (.pth)
├── Fish_Detection_With_Swin_Transformer.ipynb  # ⭐️ Notebook UTAMA untuk Training
├── analisis_aja.py             # Script untuk analisis hasil & visualisasi grafik
├── test.py                     # Utilitas: Standarisasi nama file dataset
├── EXPERIMENTS.md              # Log detail eksperimen & hasil
└── readme.md                   # Dokumentasi ini
```

---

## ⚙️ Persiapan Lingkungan (Setup)

Untuk menjalankan kode training atau eksperimen di repository ini, ikuti langkah berikut:

### Prasyarat
*   **OS:** Windows / Linux
*   **GPU:** NVIDIA (Sangat disarankan untuk training)
*   **Manager:** Anaconda / Miniconda

### Langkah Instalasi

1.  **Clone Repository**
    ```bash
    git clone https://github.com/Blauzaa/FishDetectionTrain.git
    cd FishDetectionTrain
    ```

2.  **Buat Environment**
    ```bash
    conda create -n ikan_env python=3.10
    conda activate ikan_env
    ```

3.  **Install PyTorch**
    Sesuaikan dengan versi CUDA Anda (contoh di bawah untuk CUDA 12.1):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install MMDetection & Library Pendukung**
    ```bash
    # Install library dasar
    conda install numpy scikit-learn jupyter cmake ninja -y
    pip install future tensorboard

    # Install MMCV (Computer Vision foundation)
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.0"

    # Install MMDetection
    mim install mmdet
    ```

5.  **Verifikasi Instalasi**
    ```python
    import mmdet
    print(mmdet.__version__)
    ```

---

## 🚀 Cara Penggunaan

### 1. Training Model Baru
Gunakan Jupyter Notebook untuk proses training yang interaktif dan mudah dipantau.
1.  Buka file **`Fish_Detection_With_Swin_Transformer.ipynb`**.
2.  Atur path dataset Anda di bagian konfigurasi awal notebook.
3.  Jalankan sel secara berurutan (Run All).
4.  Model terbaik akan tersimpan otomatis di folder `outputs_...` (biasanya dengan nama `best_coco_bbox_mAP_epoch_XX.pth`).

### 2. Evaluasi & Analisis
Untuk melihat performa model secara mendalam (Confusion Matrix, PR Curve, Grafik Loss):
1.  Pastikan proses training sudah menghasilkan folder output.
2.  Buka file **`analisis_aja.py`**.
3.  Ubah variabel `WORK_DIR` agar mengarah ke folder output yang ingin dianalisis (contoh: `./outputs_swin_retinanet_deepfish`).
4.  Jalankan script:
    ```bash
    python analisis_aja.py
    ```
5.  Hasil analisis akan muncul di folder `reports_final_analysis` di dalam direktori output tersebut.

### 3. Menggunakan Model di API
Setelah mendapatkan file model (`.pth`) dengan akurasi terbaik dari repository ini:
1.  Copy file `.pth` tersebut.
2.  Paste ke dalam folder model di repository **[FishDetectionAPI](https://github.com/Blauzaa/FishDetectionAPI)**.
3.  Ikuti petunjuk di repository API untuk memuat model tersebut.

---

## 🧪 Log Eksperimen
Detail lengkap mengenai skenario pengujian, parameter yang digunakan, dan hasil analisis per eksperimen didokumentasikan secara terpisah.

👉 **[Lihat EXPERIMENTS.md](./EXPERIMENTS.md)**

---

## 📚 Referensi
*   **Swin Transformer:** Liu, Z. et al. (ICCV 2021)
*   **Focal Loss:** Lin, T.Y. et al. (ICCV 2017)
*   **MMDetection:** OpenMMLab Detection Toolbox
