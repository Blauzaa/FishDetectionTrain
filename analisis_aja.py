import os
import torch
import numpy as np
import mmcv
import json
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.registry import DATASETS
from mmdet.utils import register_all_modules

# ===================================================================
# BAGIAN 1: KONFIGURASI YANG HARUS SAMA DENGAN SKRIP TRAINING
# ===================================================================

print("--- Memulai skrip analisis dan plotting ---")

# Definisikan kelas Anda
CLASSES = (
    'Betta', 'Discuss', 'Glofish', 'Goldfish', 'Guppy', 'Gurami',
    'Manfish', 'Molly', 'Swordtail', 'Tiger Barb'
)
PALETTE = [
    (220, 20, 60), (0, 128, 255), (60, 179, 113), (255, 140, 0), (138, 43, 226),
    (255, 215, 0), (199, 21, 133), (127, 255, 212), (70, 130, 180), (255, 99, 71)
]

# Load file konfigurasi yang sama
cfg_path = 'mmdetection/configs/swin/retinanet_swin-s-p4-w7_fpn_1x_coco_custom.py'
cfg = Config.fromfile(cfg_path)

# ===================================================================
# PENTING: Atur jumlah kelas di konfigurasi sebelum memuat model
# ===================================================================
print("--- Mengatur jumlah kelas di head model menjadi 10 ---")
if hasattr(cfg.model, 'bbox_head'):
    cfg.model.bbox_head.num_classes = len(CLASSES)
elif hasattr(cfg.model, 'roi_head'):
    if isinstance(cfg.model.roi_head.bbox_head, list):
        for head in cfg.model.roi_head.bbox_head:
            head.num_classes = len(CLASSES)
    else:
        cfg.model.roi_head.bbox_head.num_classes = len(CLASSES)
# ===================================================================

# --- Atur variabel-variabel penting ---
WORK_DIR = './outputs_model_dengan_grafik' 
REPORT_DIR = os.path.join(WORK_DIR, 'reports_final_analysis')
os.makedirs(REPORT_DIR, exist_ok=True)

print(f"--- Folder Kerja diatur ke: {WORK_DIR} ---")
print(f"--- Laporan akan disimpan di: {REPORT_DIR} ---")


# ===================================================================
# BAGIAN 2: PLOTTING KURVA STABILITAS TRAINING (BERDASARKAN EPOCH)
# ===================================================================
print("\n>>> Membuat grafik analisis training (berdasarkan Epoch)...")

def create_stability_curve_by_epoch(log_path, output_dir, output_filename='training_stability_curve.png'):
    if not os.path.exists(log_path):
        print(f"❌ ERROR: File log tidak ditemukan di '{log_path}'")
        return

    print(f"✅ Membaca data dari file JSON: {log_path}")
    epoch_data = defaultdict(lambda: {'losses': [], 'mAP': None})
    last_seen_epoch = 0

    with open(log_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                if 'loss' in log and 'epoch' in log:
                    epoch = log['epoch']
                    epoch_data[epoch]['losses'].append(log['loss'])
                    last_seen_epoch = epoch
                elif 'coco/bbox_mAP' in log:
                    if last_seen_epoch > 0:
                        epoch_data[last_seen_epoch]['mAP'] = log['coco/bbox_mAP']
            except (json.JSONDecodeError, KeyError):
                continue
    
    final_epochs, final_loss, final_map = [], [], []
    for epoch, data in sorted(epoch_data.items()):
        if data['losses'] and data['mAP'] is not None:
            final_epochs.append(epoch)
            final_loss.append(sum(data['losses']) / len(data['losses']))
            final_map.append(data['mAP'])

    if not final_epochs:
        print("\n❌ GAGAL: Tidak dapat menggabungkan data loss dan mAP per epoch. Periksa isi file log.")
        return

    print(f"\n📊 Berhasil memproses data untuk {len(final_epochs)} epoch. Membuat plot...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    color = 'tab:red'
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Rata-rata Training Loss', color=color, fontsize=12)
    ax1.plot(final_epochs, final_loss, 'o-', color=color, label='Training Loss (rata-rata per epoch)')
    ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', alpha=0.6); ax1.set_xlim(left=0)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation mAP', color=color, fontsize=12)
    ax2.plot(final_epochs, final_map, 's-', color=color, label='Validation mAP (per epoch)', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color); ax2.set_ylim(bottom=0)

    plt.title('Kurva Stabilitas Training: Loss vs. Validation mAP', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"🎉 Grafik berhasil disimpan di: {output_path}")

# --- Logika untuk menemukan file log dan memanggil fungsi plotting ---
log_file_path = './outputs_model_dengan_grafik/20251028_112001/vis_data/20251028_112001.json'
print(f"--- Mencari file log .json di dalam '{WORK_DIR}'... ---")
for root, dirs, files in os.walk(WORK_DIR):
    for file in files:
        if file.endswith('.json'):
            try:
                base_name = file.replace('.json', '')
                if any(char.isdigit() for char in base_name):
                    log_file_path = os.path.join(root, file)
                    print(f"--- DITEMUKAN: File log kandidat: {log_file_path} ---")
                    break
            except ValueError:
                continue
    if log_file_path:
        break

if log_file_path and os.path.exists(log_file_path):
    create_stability_curve_by_epoch(log_file_path, REPORT_DIR)
else:
    print(f"--- FATAL ERROR: File log .json tidak ditemukan. Grafik stabilitas tidak dapat dibuat. ---")


# ===================================================================
# BAGIAN 3: PLOTTING CONFUSION MATRIX & PR CURVE
# ===================================================================

def compute_iou_matrix(boxes1, boxes2):
    if boxes1.size == 0 or boxes2.size == 0: return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    x11, y11, x12, y12 = boxes1[:,0], boxes1[:,1], boxes1[:,2], boxes1[:,3]
    x21, y21, x22, y22 = boxes2[:,0], boxes2[:,1], boxes2[:,2], boxes2[:,3]
    inter_x1 = np.maximum(x11[:, None], x21[None, :]); inter_y1 = np.maximum(y11[:, None], y21[None, :])
    inter_x2 = np.minimum(x12[:, None], x22[None, :]); inter_y2 = np.minimum(y12[:, None], y22[None, :])
    inter_w = np.maximum(0, inter_x2 - inter_x1); inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = (x12 - x11) * (y12 - y11); area2 = (x22 - x21) * (y22 - y21)
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)

def greedy_match(iou_mat, iou_thr=0.5):
    matches = []; gt_used, pred_used = set(), set()
    if iou_mat.size == 0: return matches
    pairs = sorted([(i, j, iou_mat[i, j]) for i in range(iou_mat.shape[0]) for j in range(iou_mat.shape[1])], key=lambda x: x[2], reverse=True)
    for i, j, iou in pairs:
        if iou < iou_thr: break
        if i in gt_used or j in pred_used: continue
        gt_used.add(i); pred_used.add(j); matches.append((i, j))
    return matches

def evaluate_confusion_pr(model, dataset, score_thr=0.3, iou_thr=0.5):
    print("\n>>> Memulai evaluasi untuk Confusion Matrix dan PR Curves...")
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    n_classes = len(CLASSES)
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int32)
    per_class_counts = {'TP': np.zeros(n_classes, dtype=np.int32), 'FP': np.zeros(n_classes, dtype=np.int32), 'FN': np.zeros(n_classes, dtype=np.int32)}
    pr_store = {c: {'scores': [], 'match': []} for c in range(n_classes)}

    # === PERBAIKAN UTAMA ADA DI SINI ===
    # Kembali ke metode perulangan yang andal
    for idx in range(len(base_dataset)):
        if idx % 50 == 0:
            print(f"  Mengevaluasi gambar {idx+1}/{len(base_dataset)}...")
        
        # Ambil info mentah dari dataset, ini cara paling aman
        info = base_dataset.get_data_info(idx)
        img_path = info['img_path']
        
        # Ambil ground truth dari info mentah
        gt_instances = info.get('instances', [])
        gt_bboxes = np.array([ann['bbox'] for ann in gt_instances], dtype=np.float32).reshape(-1, 4)
        gt_labels = np.array([ann['bbox_label'] for ann in gt_instances], dtype=np.int64)
        
        # Lakukan inferensi pada path gambar
        result = inference_detector(model, img_path)
        pred = result.pred_instances
        keep = pred.scores >= score_thr
        pred_bboxes, pred_scores, pred_labels = pred.bboxes[keep].cpu().numpy(), pred.scores[keep].cpu().numpy(), pred.labels[keep].cpu().numpy()

        # Sisa logika matching dan counting tidak berubah
        iou_mat = compute_iou_matrix(gt_bboxes, pred_bboxes)
        matches = greedy_match(iou_mat, iou_thr=iou_thr)
        matched_gt, matched_pred = {m[0] for m in matches}, {m[1] for m in matches}

        for gi, pj in matches:
            gt_c, pd_c = int(gt_labels[gi]), int(pred_labels[pj])
            conf_mat[gt_c, pd_c] += 1
            per_class_counts['TP'][pd_c] += 1
            pr_store[pd_c]['scores'].append(float(pred_scores[pj]))
            pr_store[pd_c]['match'].append(1)

        for j, pd_c in enumerate(pred_labels):
            if j not in matched_pred:
                per_class_counts['FP'][int(pd_c)] += 1
                pr_store[int(pd_c)]['scores'].append(float(pred_scores[j]))
                pr_store[int(pd_c)]['match'].append(0)

        for i, gt_c in enumerate(gt_labels):
            if i not in matched_gt:
                per_class_counts['FN'][int(gt_c)] += 1
    
    # Plotting (tidak berubah)
    fig_cm = plt.figure(figsize=(12, 10)); ax = fig_cm.add_subplot(111)
    im = ax.imshow(conf_mat, interpolation='nearest', cmap='viridis')
    ax.set_title(f"Confusion Matrix (IoU>{iou_thr}, Score>{score_thr})"); plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(CLASSES)); ax.set_xticks(tick_marks, labels=CLASSES, rotation=45, ha='right'); ax.set_yticks(tick_marks, labels=CLASSES)
    thresh = conf_mat.max() / 2.;
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        ax.text(j, i, format(conf_mat[i, j], 'd'), ha="center", va="center", color="white" if conf_mat[i, j] > thresh else "black")
    plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png"); fig_cm.savefig(cm_path, dpi=200); plt.close(fig_cm)

    for c in range(len(CLASSES)):
        scores, match = np.array(pr_store[c]['scores']), np.array(pr_store[c]['match'])
        if scores.size == 0: continue
        order = np.argsort(-scores); scores, match = scores[order], match[order]
        tp_cum, fp_cum = np.cumsum(match), np.cumsum(1 - match)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        total_pos = per_class_counts['TP'][c] + per_class_counts['FN'][c]
        recall = tp_cum / max(total_pos, 1) if total_pos > 0 else 0
        fig_pr = plt.figure(); plt.plot(recall, precision, '-o', markersize=4); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve - {CLASSES[c]}"); plt.grid(); plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05)
        pr_path = os.path.join(REPORT_DIR, f"pr_curve_{c:02d}_{CLASSES[c].replace(' ','_')}.png"); fig_pr.savefig(pr_path, dpi=200, bbox_inches='tight'); plt.close(fig_pr)

    print(f">>> BERHASIL: Confusion matrix disimpan di: {cm_path}")
    print(f">>> BERHASIL: PR curves disimpan di: {REPORT_DIR}")

# --- Load Model untuk Evaluasi ---
checkpoint_path = None
if os.path.exists(WORK_DIR):
    best_ckpts = [f for f in os.listdir(WORK_DIR) if f.startswith('best_') and f.endswith('.pth')]
    if best_ckpts:
        checkpoint_path = os.path.join(WORK_DIR, sorted(best_ckpts)[-1])
if not checkpoint_path or not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(WORK_DIR, 'latest.pth')

if os.path.exists(checkpoint_path):
    print(f"\n--- INFO: Menggunakan checkpoint: {checkpoint_path} ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_detector(cfg, checkpoint_path, device=device)
    
    register_all_modules()
    
    val_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(416, 416), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
        dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]
    
    cat_id_map = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8, 12: 9}
    metainfo = {'classes': CLASSES, 'palette': PALETTE, 'cat_id_map': cat_id_map}

    val_dataset = DATASETS.build(dict(
        type='CocoDataset',
        data_root='dataset/10_jenis_ikan/',
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
        pipeline=val_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=32)))
    
    _ = evaluate_confusion_pr(model, val_dataset, score_thr=0.3, iou_thr=0.5)
else:
    print(f"--- FATAL ERROR: Checkpoint tidak ditemukan di {checkpoint_path}. ---")
    print("--- Pastikan path WORK_DIR sudah benar dan ada file .pth di dalamnya. ---")

print("\n--- Analisis selesai. Periksa folder laporan Anda. ---")