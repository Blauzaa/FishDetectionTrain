# 🎯 Deteksi & Klasifikasi Multi-Kelas Ikan Hias
### Menggunakan Swin Transformer + RetinaNet

---

## 1. Deskripsi Proyek
Proyek ini mengimplementasikan arsitektur **Swin Transformer** sebagai *backbone* pada detektor **RetinaNet** untuk tugas deteksi multi-kelas ikan hias di akuarium.  

**Tujuan utama:**
- Mendeteksi lokasi ikan dengan bounding box.  
- Mengklasifikasikan jenis ikan (10 kelas target).  
- Menghitung jumlah ikan tiap jenis secara otomatis.  

**Tantangan:**
- Variasi pencahayaan dalam akuarium.  
- Refleksi air & kaca.  
- Pergerakan ikan cepat & *occlusion*.  

---

## 2. Metodologi

### 2.1 Arsitektur
- **Backbone:** Swin Transformer (Swin-T / Swin-B)  
- **Detector:** RetinaNet (single-stage)  
- **Feature Extractor:** Feature Pyramid Network (FPN)  
- **Loss Function:**  
  - Focal Loss (klasifikasi)  
  - Smooth L1 Loss (lokalisasi bounding box)  

### 2.2 Dataset
Dataset pribadi dengan 10 jenis ikan hias:  

Manfish, Betta, Goldfish, Discuss, Pleco, Guppy, Tiger Barb, Gurami, Molly, Swordtail


**Skenario pengambilan data:**  
1. Akuarium berisi 1 ikan (deteksi dasar).  
2. Akuarium berisi 3–5 ikan jenis sama (multi-objek).  
3. Akuarium berisi 3–5 ikan dari 2–3 jenis (multi-kelas).  
4. Akuarium padat >5 ikan (occlusion tinggi).  
5. Variasi pencahayaan & dekorasi akuarium.  

---

## 3. Alur Kerja
1. **Persiapan dataset** → anotasi dengan bounding box & label kelas.  
2. **Preprocessing & augmentasi** → resize 640x640, rotasi, flipping, adjust brightness.  
3. **Training** → RetinaNet + Swin Transformer dengan *transfer learning*.  
4. **Evaluasi** → metrik: Precision, Recall, F1-Score, mAP.  
5. **Pengujian sistem** → deteksi ikan pada citra akuarium uji.  

---

## 4. Setup Lingkungan
```bash
# 1. Buat environment
conda create -n ikan_env python=3.10
conda activate ikan_env

# 2. Install PyTorch (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
conda install numpy scikit-learn jupyter cmake ninja -y
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
pip install -U openmim
mim install mmdet
pip install future tensorboard

# 4. Jalankan Jupyter Notebook
jupyter notebook



```

---

## 5. Log Pengujian

Tabel ini digunakan untuk mencatat setiap eksperimen (konfigurasi + hasil).  

---

### 🧪 Eksperimen #1: Baseline dengan Dataset Offline Roboflow

**Tujuan/Hipotesis:**  
Menguji performa dasar model dengan dataset yang diproses sepenuhnya di Roboflow.

🔧 **Pengaturan:**
- **Backbone:** Swin-S  
- **Epoch Dijalankan:** 15 / 15 (Selesai)  
- **Learning Rate:** Statis `1e-4`  
- **Augmentasi:** `RandomFlip`  
- **Dataset:** Versi Roboflow (augmentasi offline & resize stretch)  

📊 **Observasi & Hasil:**
- mAP naik sangat lambat, stagnan di epoch 4–5.  
- **Hasil Final Terbaik (epoch ~5):**  
  - **mAP@.50:.95:** 3.7%  
  - **mAP@.50:** 10.2%  

🔬 **Analisis:**  
- Performa rendah karena **data poisoning** dari augmentasi offline.  
- `Resize stretch` merusak bentuk ikan.  
- LR statis tidak efisien.  

💡 **Rencana:**  
1. Gunakan dataset mentah.  
2. Terapkan augmentasi online.  
3. Gunakan LR scheduler.  

---

### 🧪 Eksperimen #2: Augmentasi Online Penuh dengan LR Standar

**Tujuan/Hipotesis:**  
Menguji apakah augmentasi online + LR scheduler bisa mengatasi stagnasi.

🔧 **Pengaturan:**
- **Dataset:** mentah.  
- **Augmentasi:** `RandomFlip`, `PhotoMetricDistortion`, `RandomAffine`.  
- **Base LR:** `1e-4` (Scheduler).  
- **Backbone:** Swin-S  
- **Epoch Dijalankan:** dihentikan di epoch 7 (target 50).  

📊 **Progress mAP per Epoch:**

| Epoch | mAP  | mAP@50 | Catatan            |
|-------|------|--------|--------------------|
| 3     | 2.8% | 7.7%   | Puncak sementara   |
| 4     | 0.4% | 1.2%   | Kolaps total       |
| 5–7   | 0.4% | 1.3%   | Stagnan, dihentikan |

🔬 **Analisis:**  
- LR `1e-4` terlalu tinggi → training tidak stabil.  
- Beberapa kelas (`Betta`, `Manfish`, `Tiger Barb`) selalu mAP 0.0.  

💡 **Rencana:**  
1. Turunkan LR → `2e-5`.  
2. Nonaktifkan augmentasi kompleks.  
3. Audit dataset manual.  

---

### 🧪 Eksperimen #3: Stabilisasi Training (Baseline Baru)

**Tujuan/Hipotesis:**  
Menurunkan learning rate + menyederhanakan augmentasi untuk mendapatkan baseline stabil.

🔧 **Pengaturan:**
- **Base LR:** `2e-5`  
- **Augmentasi:** hanya `RandomFlip`  
- **Backbone:** Swin-S  
- **Epoch Dijalankan:** 50 / 50 (hasil terbaik di epoch 35)  

📊 **Hasil Terbaik (Epoch 35):**
- **mAP@.50:.95:** 58.6%  
- **mAP@.50:** 92.9%  

🔬 **Analisis:**  
- Training berhasil distabilkan.  
- Gap besar mAP (58.6%) vs mAP@50 (92.9%) → bounding box presisi masih kurang.  
- Kelas lemah: `Manfish` (42.3%) & `Janitor` (55.3%).  

💡 **Rencana:**  
- Tambah augmentasi & resolusi untuk meningkatkan presisi bbox + confidence score.  

---
### 🧪 Eksperimen #4: Peningkatan Performa dengan Augmentasi & Resolusi

**Tujuan/Hipotesis:**
Augmentasi kompleks + resolusi input lebih tinggi akan meningkatkan presisi bounding box & confidence score.

🔧 **Pengaturan:**
- **Base LR:** `2e-5` (tetap untuk stabilitas)
- **Augmentasi:** `PhotoMetricDistortion` & `RandomAffine` diaktifkan
- **Resolusi (Resize scale):** `(800, 800)`
- **Training:** Melanjutkan dari checkpoint Eksperimen #3 (epoch\_35.pth)
- **Epoch Dijalankan:** 75 (hasil terbaik di epoch 50)

📊 **Hasil Terbaik (Epoch 50):**
- **mAP@.50:.95:** 57.8% (🔻 Turun dari 58.6%)
- **mAP@.50:** 95.3% (🔼 Naik dari 92.9%)
- **mAP@.75:** 62.0% (🔻 Turun dari 63.9%)

🔬 **Analisis:**
- Hipotesis gagal. Augmentasi yang lebih kompleks dan resolusi yang lebih tinggi ternyata kontra-produktif.
- Meskipun kemampuan deteksi dasar (mAP@.50) sedikit meningkat, presisi bounding box (mAP@.75) dan performa keseluruhan (mAP) justru menurun.
- Model menjadi kurang percaya diri pada objek yang sulit, menyebabkan beberapa deteksi valid hilang.

💡 **Rencana:**
- Kembali ke konfigurasi stabil dari Eksperimen #3 sebagai dasar.
- Lakukan tuning yang lebih halus dan terarah, fokus pada peningkatan kelas Manfish dan presisi BBox.

---

### 🧪 Eksperimen #5: Fine-Tuning dengan Augmentasi Warna

**Tujuan/Hipotesis:**
Menambahkan augmentasi warna (PhotoMetricDistortion) pada baseline terbaik (Eksperimen #3) akan meningkatkan kemampuan generalisasi dan presisi bounding box.

🔧 **Pengaturan:**
- **Base LR:** `2e-5`
- **Augmentasi:** `PhotoMetricDistortion` diaktifkan
- **Resolusi (Resize scale):** `(640, 640)`
- **Training:** Melanjutkan dari checkpoint Eksperimen #3 (epoch\_35.pth)
- **Epoch Dijalankan:** 60 (hasil terbaik di epoch 44)

📊 **Hasil Terbaik (Epoch 44):**
- **mAP@.50:.95:** 59.2% (🔼 Naik dari 58.6%)
- **mAP@.50:** 94.0% (🔼 Naik dari 92.9%)
- **mAP@.75:** 67.6% (🔼 Naik signifikan dari 63.9%)

🔬 **Analisis:**
- Hipotesis berhasil. Augmentasi warna terbukti efektif meningkatkan presisi bounding box (mAP@.75) dan performa keseluruhan.
- Ini adalah model terbaik sejauh ini.
- Kelas `Manfish` (mAP 42.3%) tetap menjadi yang terlemah dan tidak menunjukkan peningkatan, mengindikasikan masalah intrinsik pada data.

💡 **Rencana:**
- Tuning augmentasi lebih lanjut kemungkinan tidak akan memberikan banyak hasil.
- Langkah berikutnya adalah mencoba teknik augmentasi yang secara fundamental lebih kuat untuk menangani variasi dan presisi.

---

### 🧪 Eksperimen #6: Peningkatan Presisi dengan Mosaic Augmentation

**Tujuan/Hipotesis:**
Menambahkan augmentasi Mosaic akan membuat model lebih tangguh dalam mengenali objek dalam berbagai skala dan konteks, sehingga dapat meningkatkan performa pada kelas sulit (Manfish) dan mAP keseluruhan.

🔧 **Pengaturan:**
- **Base LR:** `2e-5`
- **Augmentasi:** Menambahkan `Mosaic` ke pipeline dari Eksperimen #5
- **Training:** Melanjutkan dari checkpoint Eksperimen #5 (epoch\_44.pth)
- **Epoch Dijalankan:** 75 (hasil terbaik di epoch 31 dari sesi fine-tuning)

📊 **Hasil Terbaik (Epoch 31 Sesi Fine-Tuning):**
- **mAP@.50:.95:** 62.9% (🔼 Naik signifikan dari 59.2%)
- **mAP@.50:** 95.2% (🔼 Naik dari 94.0%)
- **mAP@.75:** 70.8% (🔼 Naik signifikan dari 67.6%)
---

### 📌 Ringkasan Hasil Eksperimen

| No | Backbone | Epoch | Batch | Learning Rate | Augmentasi              | mAP@.50:.95 | mAP@.50 | Precision | Recall | F1-Score | Catatan          |
|----|----------|-------|-------|---------------|-------------------------|-------------|---------|-----------|--------|----------|-----------------|
| 1  | Swin-T   | 24    | 4     | 1e-4          | flip, brightness        | XX.X        | XX.X    | XX.X      | XX.X   | XX.X     | Run awal        |
| 2  | Swin-T   | 36    | 4     | 1e-4          | flip, rotation          | XX.X        | XX.X    | XX.X      | XX.X   | XX.X     | Tuning epoch    |
| 3  | Swin-B   | 24    | 2     | 5e-5          | flip, crop              | XX.X        | XX.X    | XX.X      | XX.X   | XX.X     | Coba backbone   |
| 4  | Swin-S   | 50    | 4     | 2e-5          | flip (baseline stabil)  | 58.6        | 92.9    | XX.X      | XX.X   | XX.X     | Stabilisasi     |
| 5  | Swin-S   | 75    | 4     | 2e-5          | flip, distortion, affine| XX.X        | XX.X    | XX.X      | XX.X   | XX.X     | Peningkatan bbox|

---

## 6. Referensi

- Liu, Z. et al. **Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows**, ICCV 2021.  
- Lin, T.Y. et al. **Focal Loss for Dense Object Detection**, ICCV 2017.  
- Padilla, R. et al. **A Survey on Performance Metrics for Object Detection Algorithms**, ICSSIP 2020.  
- Hu, X. et al. **Demystify Transformers & Convolutions in Modern Image Deep Networks**, IEEE TPAMI 2025.  

---
