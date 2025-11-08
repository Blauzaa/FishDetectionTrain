import os, re, json, shutil, random, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
# Coba import yaml, jika tidak ada, beri tahu pengguna
try:
    import yaml
except ImportError:
    print("Paket PyYAML tidak terinstal. Jalankan: pip install pyyaml")
    yaml = None


# ========= CONFIG =========
SOURCE_ROOT = "dataset_sementara"   # folder yang berisi dataset1..dataset4
OUTPUT_ROOT = "dataset_final"       # folder hasil gabungan
SPLIT_NAMES = ["train", "valid", "val", "test"]  # skrip auto-normalize 'valid' -> 'val'
RANDOM_SEED = 42

# Target classes (English, tanpa "ikan")
TARGET_CLASSES = [
    "Manfish", "Betta", "Goldfish", "Discuss", "Janitor",
    "Guppy", "Tiger Barb", "Gurami", "Corydoras","Swordtail"
]
CLASS2ID = {c: i+1 for i, c in enumerate(TARGET_CLASSES)}  # COCO IDs start at 1

# Label mapping (kunci akan dinormalisasi; tambahkan varian jika perlu)
RAW_TO_TARGET = {
    
    #Manfish
    "Manfish": "Manfish",

    # betta (cupang)
    "betta": "Betta",
    "Betta Candy Koi Fighting Fish": "Betta",
    "Candy Koi Betta": "Betta",
    "galaxy betta": "Betta",
    "Galaxy Betta Fish": "Betta",
    "copper betta": "Betta",
    "Galaxy betta": "Betta",

    # goldfish (ikan koki)
    "goldfish": "Goldfish",
    "gold fish": "Goldfish",
    "koki mutiara": "Goldfish",
    "koki ranchu": "Goldfish",
    "ranchu": "Goldfish",
    "rancho": "Goldfish",
    "koki panda": "Goldfish",
    "koki black ranchu": "Goldfish",
    "koki oranda": "Goldfish",
    "komet": "Goldfish",
    "Goldfish": "Goldfish",

    # discus
    "discus": "Discus",
    "discuss fish": "Discus",
    "Discus": "Discus",
    "Discuss Fish": "Discus",

    #Janitor
    "pleco": "Janitor",
    "janitor fish": "Janitor",
    "suckerfish": "Janitor",
    "sucker fish": "Janitor",
    "botia": "Janitor",

    # guppy
    "guppy": "Guppy",
    "Guppy Dumbo": "Guppy",
    "Guppy Dumbo Fish": "Guppy",


    # barb
    "tiger barb": "Tiger Barb",
    "tiger barb": "Tiger Barb",

    # gourami (gurami)
    "Gourami": "Gurami",
    "gourami": "Gurami",

    #Corydoras

    #Swordtail
    "swordtail": "Swordtail",

}



# ========= HELPERS =========

def generate_safe_filename(src_path: Path, prefix: str) -> str:
    """
    Generates a shorter, safe filename using a hash to avoid path length issues on Windows.
    """
    original_stem = src_path.stem
    original_ext = src_path.suffix
    # Create a hash of the original name to ensure uniqueness and shortness
    name_hash = hashlib.md5(original_stem.encode()).hexdigest()
    # Gabungkan prefix, hash, dan ekstensi asli
    return f"{prefix}__{name_hash}{original_ext}"

def norm(s: str) -> str:
    """Normalisasi nama kelas agar mapping lebih tahan banting."""
    s = s.strip()
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = s.lower()
    s = re.sub(r'[_/:-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

MAPPING = {norm(k): v for k, v in RAW_TO_TARGET.items()}

def map_class(name: str):
    n = norm(name)
    return MAPPING.get(n, None)

def ensure_dirs():
    for split in ["train", "val", "test"]:
        Path(OUTPUT_ROOT, split, "images").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_ROOT, "dump").mkdir(exist_ok=True)

def detect_splits(ds_root: Path):
    found = {}
    for child in ds_root.iterdir():
        if not child.is_dir(): continue
        nm = child.name.lower()
        if nm in ("train", "training"): found["train"] = child
        elif nm in ("valid", "validation", "val"): found["val"] = child
        elif nm == "test": found["test"] = child
    return found

def is_coco_split(split_path: Path):
    return (split_path / "_annotations.coco.json").exists()

def yolo_class_names(ds_root: Path):
    if not yaml: return None
    yaml_paths = list(ds_root.glob("**/data.yaml")) + list(ds_root.glob("data.yaml"))
    names = None
    for yp in yaml_paths:
        try:
            with open(yp, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "names" in data:
                names = data["names"]; break
        except Exception: pass
    if names is None:
        for cand in ["classes.txt", "obj.names"]:
            p = ds_root / cand
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    names = [ln.strip() for ln in f if ln.strip()]; break
    return names

def copy_with_unique_name(src_path: Path, prefix: str, split: str):
    # dst_name = f"{prefix}__{src_path.name}" # <--- BARIS LAMA
    dst_name = generate_safe_filename(src_path, prefix) # <--- BARIS BARU
    dst_path = Path(OUTPUT_ROOT, split, "images", dst_name)
    shutil.copy2(src_path, dst_path)
    return dst_name

# ========= ACCUMULATORS =========
per_split_images = defaultdict(list)
per_split_annotations = defaultdict(list)
unknown_labels = Counter()
kept_labels = Counter()

dump_data = defaultdict(lambda: {"images": [], "annotations": [], "categories": []})
dump_img_counter = defaultdict(int)
dump_ann_counter = defaultdict(int)

image_id_counter = {"train": 1, "val": 1, "test": 1}
ann_id_counter = {"train": 1, "val": 1, "test": 1}

def save_to_dump(src_img: Path, orig_name: str, bbox=None, wh=None):
    cls_name = norm(orig_name).replace(" ", "_")
    dump_dir = Path(OUTPUT_ROOT, "dump", cls_name)
    img_dir = dump_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # new_fname = f"{cls_name}__{src_img.name}" # <--- BARIS LAMA
    new_fname = generate_safe_filename(src_img, cls_name) # <--- BARIS BARU
    dst_path = img_dir / new_fname
    
    if not dst_path.exists(): # Hindari duplikasi copy
        shutil.copy2(src_img, dst_path)

    # ... sisa fungsi tidak perlu diubah ...
    dump_img_counter[cls_name] += 1
    img_id = dump_img_counter[cls_name]
    W, H = wh if wh else (0, 0)
    
    if not any(d['file_name'] == new_fname for d in dump_data[cls_name]["images"]):
        dump_data[cls_name]["images"].append({
            "id": img_id, "file_name": new_fname, "width": W, "height": H
        })
    else: 
        img_id = next(d['id'] for d in dump_data[cls_name]["images"] if d['file_name'] == new_fname)

    if bbox:
        dump_ann_counter[cls_name] += 1
        ann_id = dump_ann_counter[cls_name]
        dump_data[cls_name]["annotations"].append({
            "id": ann_id, "image_id": img_id, "category_id": 1,
            "bbox": bbox, "area": bbox[2]*bbox[3], "iscrowd": 0, "segmentation": []
        })
    dump_data[cls_name]["categories"] = [{"id": 1, "name": cls_name, "supercategory": "fish"}]

# ========= MAIN MERGE =========
random.seed(RANDOM_SEED)
ensure_dirs()

dataset_roots = sorted([p for p in Path(SOURCE_ROOT).iterdir() if p.is_dir()])

for ds_root in dataset_roots:
    splits = detect_splits(ds_root)
    if not splits:
        print(f"⚠️  Lewati {ds_root.name}: tidak ditemukan subfolder train/val/test.")
        continue

    ds_type = "coco" if any(is_coco_split(p) for p in splits.values()) else "yolo"
    print(f"📦 {ds_root.name} terdeteksi sebagai {ds_type.upper()}.")

    yolo_names = None
    if ds_type == "yolo":
        yolo_names = yolo_class_names(ds_root)
        if not yolo_names:
            print(f"   ⚠️  {ds_root.name}: tidak menemukan data.yaml/classes.txt → lewati dataset ini.")
            continue

    for split_key, split_path in splits.items():
        target_split = "val" if split_key in ("val", "valid") else split_key

        if ds_type == "coco" and is_coco_split(split_path):
            ann_file = split_path / "_annotations.coco.json"
            with open(ann_file, "r", encoding="utf-8") as f: coco = json.load(f)

            cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
            
            # --- Perubahan Logika Dimulai Di Sini ---
            anns_by_img = defaultdict(list)
            dump_anns_by_img = defaultdict(list) # BARU: Untuk menyimpan anotasi yg akan di-dump

            for ann in coco.get("annotations", []):
                orig_name = cat_id_to_name.get(ann["category_id"])
                if not orig_name: continue
                
                mapped = map_class(orig_name)
                if not mapped or mapped not in CLASS2ID:
                    unknown_labels[norm(orig_name)] += 1
                    # DIUBAH: Jangan panggil save_to_dump di sini, simpan dulu informasinya
                    dump_anns_by_img[ann["image_id"]].append({
                        "orig_name": orig_name,
                        "bbox": ann["bbox"]
                    })
                    continue
                
                kept_labels[mapped] += 1
                new_ann = {
                    "id": ann_id_counter[target_split], "image_id": None, 
                    "category_id": CLASS2ID[mapped], "bbox": ann["bbox"],
                    "area": ann.get("area", ann["bbox"][2]*ann["bbox"][3]),
                    "iscrowd": ann.get("iscrowd", 0), "segmentation": ann.get("segmentation", [])
                }
                anns_by_img[ann["image_id"]].append(new_ann)
                ann_id_counter[target_split] += 1

            img_folder1 = split_path / "images"; img_folder2 = split_path
            for img in coco.get("images", []):
                img_rel = img.get("file_name")
                cand1 = img_folder1 / Path(img_rel).name; cand2 = img_folder2 / Path(img_rel).name
                src = cand1 if cand1.exists() else (cand2 if cand2.exists() else (split_path / img_rel if (split_path / img_rel).exists() else None))
                if not src: continue

                has_kept_anns = img["id"] in anns_by_img
                has_dump_anns = img["id"] in dump_anns_by_img

                # BARU: Panggil save_to_dump di sini jika ada anotasi untuk di-dump
                if has_dump_anns:
                    W = img.get("width"); H = img.get("height")
                    for dump_ann in dump_anns_by_img[img["id"]]:
                        save_to_dump(src, dump_ann["orig_name"], bbox=dump_ann["bbox"], wh=(W, H))

                if not has_kept_anns: continue

                new_fname = copy_with_unique_name(src, prefix=f"{ds_root.name}_{target_split}", split=target_split)
                new_img_id = image_id_counter[target_split]; image_id_counter[target_split] += 1

                per_split_images[target_split].append({
                    "id": new_img_id, "file_name": new_fname,
                    "width": img.get("width"), "height": img.get("height")
                })

                for a in anns_by_img[img["id"]]:
                    a["image_id"] = new_img_id
                    per_split_annotations[target_split].append(a)

        elif ds_type == "yolo":
            img_dir = split_path / "images" if (split_path / "images").exists() else split_path
            lbl_dir = split_path / "labels"
            if not lbl_dir.exists(): continue

            for lbl_path in lbl_dir.glob("*.txt"):
                img_path = None
                for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                    if (img_dir / (lbl_path.stem + ext)).exists():
                        img_path = img_dir / (lbl_path.stem + ext); break
                if not img_path: continue

                try:
                    with Image.open(img_path) as im: W, H = im.size
                except Exception: continue

                anns_for_this_image = []
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue # Minimal harus class_id + 2 titik (x,y)
                        
                        try:
                            cid = int(float(parts[0]))
                            coords = list(map(float, parts[1:]))
                        except ValueError:
                            continue # Baris korup

                        if not (0 <= cid < len(yolo_names)): continue
                        
                        orig_name = yolo_names[cid]
                        
                        # --- PERUBAHAN LOGIKA DIMULAI DI SINI ---
                        bbox = None
                        if len(coords) == 4:
                            # Kasus 1: Bounding Box [x_center, y_center, width, height]
                            x_c, y_c, w, h = coords
                            x = (x_c - w/2.0) * W
                            y = (y_c - h/2.0) * H
                            bw = w * W
                            bh = h * H
                            bbox = [x, y, bw, bh]
                        
                        elif len(coords) > 4 and len(coords) % 2 == 0:
                            # Kasus 2: Segmentasi [x1, y1, x2, y2, ...]
                            # Buat bounding box dari titik-titik poligon
                            x_points = [c * W for c in coords[0::2]] # Ambil semua x, kalikan dengan Width
                            y_points = [c * H for c in coords[1::2]] # Ambil semua y, kalikan dengan Height
                            
                            x = min(x_points)
                            y = min(y_points)
                            bw = max(x_points) - x
                            bh = max(y_points) - y
                            bbox = [x, y, bw, bh]

                        if bbox is None:
                            continue # Format tidak dikenali
                        # --- PERUBAHAN LOGIKA SELESAI ---

                        mapped = map_class(orig_name)
                        if not mapped or mapped not in CLASS2ID:
                            unknown_labels[norm(orig_name)] += 1
                            save_to_dump(img_path, orig_name, bbox=bbox, wh=(W, H))
                            continue
                        
                        kept_labels[mapped] += 1
                        anns_for_this_image.append({
                            "id": ann_id_counter[target_split], "image_id": None,
                            "category_id": CLASS2ID[mapped], "bbox": bbox,
                            "area": bbox[2] * bbox[3], "iscrowd": 0, "segmentation": []
                        })
                        ann_id_counter[target_split] += 1

                if not anns_for_this_image: continue

                new_fname = copy_with_unique_name(img_path, prefix=f"{ds_root.name}_{target_split}", split=target_split)
                new_img_id = image_id_counter[target_split]; image_id_counter[target_split] += 1

                per_split_images[target_split].append({
                    "id": new_img_id, "file_name": new_fname, "width": W, "height": H
                })
                for a in anns_for_this_image:
                    a["image_id"] = new_img_id
                    per_split_annotations[target_split].append(a)

# ========= WRITE COCO JSON PER SPLIT =========
categories = [{"id": cid, "name": cname, "supercategory": "fish"} for cname, cid in CLASS2ID.items()]

for split in ["train", "val", "test"]:
    if not per_split_images[split]: continue
    out_json = {"images": per_split_images[split], "annotations": per_split_annotations[split], "categories": categories}
    out_path = Path(OUTPUT_ROOT, split, "coco.json")
    with open(out_path, "w", encoding="utf-8") as f: json.dump(out_json, f, indent=2)
    print(f"✅ COCO json ditulis: {out_path} (images={len(per_split_images[split])}, anns={len(per_split_annotations[split])})")

# ========= WRITE COCO JSON PER DUMP CLASS =========
for cls, data in dump_data.items():
    out_path = Path(OUTPUT_ROOT, "dump", cls, "coco.json")
    with open(out_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    print(f"🗑️  Dump json ditulis: {out_path} (images={len(data['images'])}, anns={len(data['annotations'])})")

# ========= SUMMARY =========
print("\n" + "=== Rangkuman " * 5 + "===")

# --- Bagian Kelas Tersimpan ---
print("\n📊 Kelas yang tersimpan (setelah mapping & filter):")
total_kept = sum(kept_labels.values())
if not kept_labels:
    print("  Tidak ada anotasi yang disimpan.")
else:
    # Tampilkan kelas yang ditemukan
    for k, v in kept_labels.most_common():
        print(f"  ✅ {k:15s}: {v:5d} bbox")
    
    # Periksa kelas target yang tidak ditemukan
    found_classes = set(kept_labels.keys())
    all_target_classes = set(TARGET_CLASSES)
    missing_classes = all_target_classes - found_classes

    if missing_classes:
        print("\n  ⚠️  Kelas target berikut TIDAK DITEMUKAN dalam dataset:")
        for cls in sorted(list(missing_classes)):
            print(f"     - {cls}")

    print("-" * 45)
    print(f"  {'TOTAL TERSIMPAN':18s}: {total_kept:5d} bbox")


# --- Bagian Kelas Dibuang (Dump) ---
if unknown_labels:
    print("\n🗑️  Label yang tidak terpetakan (masuk dump/):")
    total_dumped = sum(unknown_labels.values())
    for k, v in unknown_labels.most_common():
        print(f"  - '{k:28s}': {v:5d}x")
    print("-" * 45)
    print(f"  {'TOTAL DI-DUMP':31s}: {total_dumped:5d}x")
else:
    print("\n✅ Semua label terbaca & terpetakan dengan baik!")

# --- Pesan Akhir ---
print("\n" + "="*55)
print(f"🎉 Proses selesai! Dataset akhir ada di folder '{OUTPUT_ROOT}'.")
print("="*55)