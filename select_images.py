import json
import random
from pathlib import Path

# ======= CONFIG =======
COCO_JSON_PATH = "data/sets/nuimages/nuimages_coco_train.json"   # ไฟล์ coco.json ต้นฉบับ
OUT_JSON_PATH  = "data/sets/nuimages/nuimages_200.json"    # ไฟล์ coco.json ชุดย่อย
N_IMAGES       = 200                                     # จำนวนรูปที่ต้องการ
STRATEGY       = "random"  # "random" หรือ "first"
SEED           = 42
KEEP_ALL_CATEGORIES = True  # False = เก็บเฉพาะ categories ที่ถูกใช้งานจริง
# ======================

def subset_coco_no_copy(
    coco_json_path,
    out_json_path,
    n_images=20000,
    strategy="random",
    seed=42,
    keep_all_categories=True
):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])

    total = len(images)
    k = min(n_images, total)

    # เลือกรูป
    if strategy == "random":
        random.seed(seed)
        selected_images = random.sample(images, k)
    elif strategy == "first":
        selected_images = images[:k]
    else:
        raise ValueError("strategy ต้องเป็น 'random' หรือ 'first'")

    selected_image_ids = {im["id"] for im in selected_images}

    # คัดกรอง annotations ให้ตรงกับ images
    filtered_anns = [ann for ann in annotations if ann.get("image_id") in selected_image_ids]

    # categories
    if keep_all_categories:
        filtered_cats = categories
    else:
        used_cat_ids = {ann["category_id"] for ann in filtered_anns}
        filtered_cats = [c for c in categories if c["id"] in used_cat_ids]

    subset_coco = {
        "info": info,
        "licenses": licenses,
        "images": selected_images,
        "annotations": filtered_anns,
        "categories": filtered_cats,
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(subset_coco, f, ensure_ascii=False)

    print(f"Done: {len(selected_images)}/{total} images kept")
    print(f"Annotations: {len(filtered_anns)}")
    print(f"Categories : {len(filtered_cats)} (keep_all_categories={keep_all_categories})")
    print(f"Output JSON: {out_json_path}")

if __name__ == "__main__":
    subset_coco_no_copy(
        coco_json_path=COCO_JSON_PATH,
        out_json_path=OUT_JSON_PATH,
        n_images=N_IMAGES,
        strategy=STRATEGY,
        seed=SEED,
        keep_all_categories=KEEP_ALL_CATEGORIES
    )
