import json
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# ==================== PENGATURAN ====================
# PENTING: Arahkan kembali ke file .json di dalam vis_data
LOG_FILE_PATH = './outputs_bs8_lr0001_workers4/20251023_112231/vis_data/20251023_112231.json'

# Arahkan ke folder reports yang Anda inginkan
REPORT_DIR = 'outputs_bs8_lr0001_workers4/reports_125_8'

# Nama file output untuk grafik
OUTPUT_GRAPH_NAME = 'training_stability_curve_FIXED.png'
# ======================================================

def create_curve_from_json_log(log_path, output_dir):
    """
    Versi FINAL yang membaca file log .json dengan benar, di mana baris 
    validasi (mAP) tidak memiliki key 'epoch'.
    """
    if not os.path.exists(log_path):
        print(f"❌ ERROR: File log tidak ditemukan di '{log_path}'")
        return

    print(f"✅ Membaca data dari file JSON: {log_path}")

    # Struktur untuk menyimpan data: epoch -> {'losses': [...], 'mAP': ...}
    epoch_data = defaultdict(lambda: {'losses': [], 'mAP': None})
    last_seen_epoch = 0

    with open(log_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())

                # Cek apakah ini baris training (ada 'loss' dan 'epoch')
                if 'loss' in log and 'epoch' in log:
                    epoch = log['epoch']
                    loss = log['loss']
                    epoch_data[epoch]['losses'].append(loss)
                    last_seen_epoch = epoch # Perbarui epoch terakhir yang terlihat

                # Cek apakah ini baris validasi (hanya ada 'coco/bbox_mAP')
                elif 'coco/bbox_mAP' in log:
                    mAP = log['coco/bbox_mAP']
                    # Kaitkan mAP ini dengan epoch terakhir yang kita lihat
                    if last_seen_epoch > 0:
                        epoch_data[last_seen_epoch]['mAP'] = mAP
                        print(f"  [+] Ditemukan mAP {mAP} untuk Epoch {last_seen_epoch}")

            except (json.JSONDecodeError, KeyError):
                continue
    
    # Proses penggabungan data
    final_epochs, final_loss, final_map = [], [], []

    # Urutkan berdasarkan nomor epoch
    for epoch, data in sorted(epoch_data.items()):
        # Hanya proses epoch yang memiliki KEDUA data (loss dan mAP)
        if data['losses'] and data['mAP'] is not None:
            avg_loss = sum(data['losses']) / len(data['losses'])
            
            final_epochs.append(epoch)
            final_loss.append(avg_loss)
            final_map.append(data['mAP'])

    if not final_epochs:
        print("\n❌ GAGAL: Tidak dapat menggabungkan data loss dan mAP.")
        print("Pastikan file log berisi kedua jenis data tersebut.")
        return

    print(f"\n📊 Berhasil memproses data lengkap untuk {len(final_epochs)} epoch. Membuat plot...")

    # Membuat Plot
    fig, ax1 = plt.subplots(figsize=(14, 7))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Rata-rata Training Loss', color=color, fontsize=12)
    ax1.plot(final_epochs, final_loss, 'o-', color=color, label='Training Loss', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(left=0) # Pastikan sumbu X mulai dari 0

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation mAP (Akurasi)', color=color, fontsize=12)
    ax2.plot(final_epochs, final_map, 's-', color=color, label='Validation mAP', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0) # Pastikan sumbu Y akurasi mulai dari 0

    plt.title('Kurva Stabilitas Training: Loss vs. Akurasi (mAP)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_GRAPH_NAME)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"🎉 Grafik berhasil disimpan di: {output_path}")

if __name__ == '__main__':
    create_curve_from_json_log(LOG_FILE_PATH, REPORT_DIR)