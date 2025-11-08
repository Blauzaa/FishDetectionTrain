import json
from pathlib import Path
from collections import Counter, defaultdict

# Path ke folder dataset final
DATASET_ROOT = Path("dataset_final")
SPLITS = ["train", "val", "test"]

def count_per_class(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    anns = coco["annotations"]
    imgs = coco["images"]

    bbox_count = Counter()
    img_per_class = defaultdict(set)

    for ann in anns:
        cls_name = cat_id_to_name[ann["category_id"]]
        bbox_count[cls_name] += 1
        img_per_class[cls_name].add(ann["image_id"])

    results = {}
    for cls in cat_id_to_name.values():
        results[cls] = {
            "images": len(img_per_class[cls]),
            "bboxes": bbox_count[cls]
        }
    return results, len(imgs), len(anns)

print("\n=== Statistik Dataset Per Split ===\n")
for split in SPLITS:
    coco_path = DATASET_ROOT / split / "coco.json"
    if not coco_path.exists():
        print(f"⚠️ Tidak ada file {coco_path}, lewati split '{split}'")
        continue

    stats, total_imgs, total_anns = count_per_class(coco_path)
    print(f"📂 Split: {split.upper()}")
    print(f"   Total Gambar : {total_imgs}")
    print(f"   Total BBoxes : {total_anns}")
    for cls, vals in stats.items():
        print(f"   - {cls:12s}: {vals['images']:5d} images, {vals['bboxes']:5d} bboxes")
    print("-" * 50)
