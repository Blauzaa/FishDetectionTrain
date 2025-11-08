import os
import re
from collections import defaultdict

# Lokasi dataset
dataset_root = "dataset/dataset"   # ganti sesuai folder dataset kamu

# Ekstensi gambar yang didukung
valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

summary = defaultdict(int)

print("=== SUMMARY DATASET ===")

for class_name in sorted(os.listdir(dataset_root)):
    class_dir = os.path.join(dataset_root, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = [f for f in os.listdir(class_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    images.sort()  # biar urutan stabil

    # Regex cek format <class_name>_<angka>
    pattern = re.compile(rf"^{class_name}_(\d+)\.[a-z]+$", re.IGNORECASE)

    for i, img in enumerate(images, 1):
        ext = os.path.splitext(img)[1].lower()
        src = os.path.join(class_dir, img)

        # Kalau namanya sudah sesuai, skip
        if pattern.match(img):
            continue

        # Kalau tidak sesuai, rename
        new_name = f"{class_name}_{i}{ext}"
        dst = os.path.join(class_dir, new_name)

        # Hindari overwrite
        counter = 1
        while os.path.exists(dst):
            new_name = f"{class_name}_{i}_{counter}{ext}"
            dst = os.path.join(class_dir, new_name)
            counter += 1

        os.rename(src, dst)

    summary[class_name] = len(images)
    print(f"- {class_name}: {len(images)} gambar")

print("\n=== TOTAL GAMBAR ===")
print(sum(summary.values()))
