import numpy as np
import json
from collections import defaultdict

def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0

def manual_eval(gt_json_path, dt_json_path, iou_threshold=0.5):
    """
    Evaluate detection results against ground truth using IoU and category match.
    Returns TP, FP, FN, precision, recall.
    """

    with open(gt_json_path) as f:
        gt = json.load(f)
    with open(dt_json_path) as f:
        dt = json.load(f)

    gt_dict = {}
    for ann in gt['annotations']:
        gt_dict.setdefault(ann['image_id'], []).append((ann['category_id'], ann['bbox']))

    TP, FP, FN = 0, 0, 0
    iou_list = []
    matched_gts = set()

    for det in dt:
        image_id = det['image_id']
        cat_id = det['category_id']
        bbox = det['bbox']
        matched = False
        for idx, (gt_cat, gt_bbox) in enumerate(gt_dict.get(image_id, [])):
            iou_ = iou(bbox, gt_bbox)
            if cat_id == gt_cat and iou_ >= iou_threshold and (image_id, idx) not in matched_gts:
                iou_list.append(iou_)
                TP += 1
                matched_gts.add((image_id, idx))
                matched = True
                break
        if not matched:
            FP += 1

    # Count FN (ground truths not detected)
    total_gt = sum(len(anns) for anns in gt_dict.values())
    FN = total_gt - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return TP, FP, FN, precision, recall, np.mean(iou_list)

def bbox_xywh_to_xyxy(b):
    x,y,w,h = b
    return (x, y, x+w, y+h)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_name(name: str) -> str:
    # ให้ Car/Truck/Person/Traffic_Light, traffic light, TRAFFIC-LIGHT, etc. มาเจอกันได้
    return name.lower().replace("-", " ").replace("_", " ").strip()

def compute_per_class_iou(
    gt_json_path: str,
    dt_json_path: str,
    iou_thresh: float = 0.50,
    target_classes=None,   # ตัวอย่าง: ["Car","Truck","Person","Traffic_Light"]
):
    with open(gt_json_path) as f:
        gt = json.load(f)
    with open(dt_json_path) as f:
        dt = json.load(f)

    # 1) เตรียม mapping พื้นฐาน
    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    cat_name_norm_to_ids = defaultdict(list)
    for cid, cname in cat_id_to_name.items():
        cat_name_norm_to_ids[normalize_name(cname)].append(cid)

    if target_classes:
        target_norms = set(normalize_name(c) for c in target_classes)
        keep_cat_ids = set()
        for n in target_norms:
            keep_cat_ids.update(cat_name_norm_to_ids.get(n, []))
    else:
        # ถ้าไม่ระบุ จะคำนวณทุกคลาส
        keep_cat_ids = set(cat_id_to_name.keys())

    # 2) index GT/DT ต่อรูป ต่อคลาส
    gt_by_img_cat = defaultdict(lambda: defaultdict(list))
    for ann in gt["annotations"]:
        cid = ann["category_id"]
        if cid in keep_cat_ids:
            gt_by_img_cat[ann["image_id"]][cid].append(ann)

    # บาง JSON ของ DT เป็น list ของ dict แบบ COCO detections
    # คาดว่ามี fields: image_id, category_id, bbox=[x,y,w,h], score
    dt_by_img_cat = defaultdict(lambda: defaultdict(list))
    for ann in dt:
        cid = ann.get("category_id")
        if cid in keep_cat_ids:
            dt_by_img_cat[ann["image_id"]][cid].append(ann)

    # 3) คำนวณแบบ greedy matching ต่อรูป ต่อคลาส
    results = {}
    per_class_iou_list = defaultdict(list)
    per_class_counts = defaultdict(lambda: {"TP":0,"FP":0,"FN":0})

    # loop เฉพาะรูปที่มี GT/DT ก็ได้ แต่เพื่อความครบ แยกรูปจาก GT
    image_ids = set([img["id"] for img in gt["images"]])
    for img_id in image_ids:
        gt_cats = gt_by_img_cat.get(img_id, {})
        dt_cats = dt_by_img_cat.get(img_id, {})
        for cid in keep_cat_ids:
            gt_list = gt_cats.get(cid, [])
            dt_list = dt_cats.get(cid, [])

            if not gt_list and not dt_list:
                continue

            # เตรียม bboxes
            gt_boxes = [bbox_xywh_to_xyxy(a["bbox"]) for a in gt_list]
            # จัด dt ตาม score สูง→ต่ำ
            dt_list_sorted = sorted(dt_list, key=lambda x: x.get("score", 0.0), reverse=True)
            dt_boxes = [bbox_xywh_to_xyxy(a["bbox"]) for a in dt_list_sorted]

            gt_used = [False]*len(gt_boxes)
            dt_used = [False]*len(dt_boxes)

            # greedy: วน dt ทีละกล่อง หา gt ที่ IoU สูงสุดที่ยังไม่ใช้
            for j, db in enumerate(dt_boxes):
                best_iou = 0.0
                best_i = -1
                for i, gb in enumerate(gt_boxes):
                    if gt_used[i]:
                        continue
                    iou = iou_xyxy(gb, db)
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                if best_i >= 0 and best_iou >= iou_thresh:
                    # TP
                    gt_used[best_i] = True
                    dt_used[j] = True
                    per_class_iou_list[cid].append(best_iou)

            TP = sum(gt_used)
            FP = sum(1 for u in dt_used if not u)
            FN = sum(1 for u in gt_used if not u)

            per_class_counts[cid]["TP"] += TP
            per_class_counts[cid]["FP"] += FP
            per_class_counts[cid]["FN"] += FN

    # 4) สรุปผลรายคลาส
    table = []
    for cid in sorted(keep_cat_ids):
        cname = cat_id_to_name[cid]
        TP = per_class_counts[cid]["TP"]
        FP = per_class_counts[cid]["FP"]
        FN = per_class_counts[cid]["FN"]
        ious = per_class_iou_list[cid]
        mean_iou = (sum(ious)/len(ious)) if ious else 0.0
        precision = TP / (TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP / (TP+FN) if (TP+FN) > 0 else 0.0

        # ถ้ามี target_classes ให้โชว์เฉพาะที่เกี่ยวข้อง
        if target_classes:
            if normalize_name(cname) not in set(normalize_name(t) for t in target_classes):
                continue

        table.append({
            "category_id": cid,
            "category_name": cname,
            "TP": TP, "FP": FP, "FN": FN,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            f"Mean IoU@{iou_thresh:.2f} (TP only)": round(mean_iou, 4),
            "TP matched count": len(ious)
        })

    # เรียงตามชื่อคลาสให้อ่านง่าย
    table.sort(key=lambda r: r["category_name"].lower())
    return table

# Example usage:
# TP, FP, FN, precision, recall = manual_eval('data/sets/nuimages/nuimages_coco.json', 'outputs/FasterRCNN_results_clean.json')
# print(f"TP: {TP}, FP: {FP}, FN: {FN}, Precision: {precision:.3f}, Recall: {recall:.3f}")

def get_mean_confident_score(dt_json_path: str):
    """
    Calculate the average confidence score from detection results JSON.
    Returns mean score (float). If no detections, returns 0.
    """
    import json
    with open(dt_json_path) as f:
        dt = json.load(f)
    if not dt:
        return 0.0
    scores = [det['score'] for det in dt if 'score' in det]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
