import json
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from torchvision import transforms
import torch
import os
from tqdm import tqdm
from NuImagesCOCODataset import NuImagesCOCODataset
import concurrent.futures


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


with open('data/sets/nuimages/nuimages_coco.json') as f:
    coco_data = json.load(f)

# Build name → ID mapping from your annotation file
name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}


# Lock to ensure single-threaded execution
_show_predictions_lock = threading.Lock()

def to_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return torch.tensor(x).cpu()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    
def show_predictions(image, predictions, label_map=None, save_path=None, correct_boxes=None, score_threshold = 0.5):
    """
    Display predictions and optionally ground truth boxes.
    - predictions: dict with 'boxes', 'labels', 'scores'
    - correct_boxes: list of [x, y, w, h] ground truth boxes (optional)
    """
    with _show_predictions_lock:
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(F.to_pil_image(image.squeeze(0)))

        # Show predicted boxes in red
        boxes = to_cpu_tensor(predictions['boxes'])
        labels = to_cpu_tensor(predictions['labels'])
        scores = to_cpu_tensor(predictions['scores'])

        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            x, y, w, h = box.tolist()
            rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            label_text = f"{label_map.get(label.item(), label.item())}: {score:.2f}" if label_map else f"{label.item()}: {score:.2f}"
            ax.text(x, y, label_text, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

        # Show correct (ground truth) boxes in green
        if correct_boxes is not None:
            for gt_box in correct_boxes:
                x, y, w, h = gt_box
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
                ax.add_patch(rect)

        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)




def start_prediction(model, dataset: NuImagesCOCODataset, output_dir: str = "data/results", threshold: float = 0.5):

    if output_dir != None:
        print(f"Saving prediction results to {output_dir}/{dataset.hazard if dataset.hazard != None else 'clean'}")
        os.makedirs(f"{output_dir}/{dataset.hazard if dataset.hazard != None else 'clean'}", exist_ok=True)
        

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

    results = []

    # Helper: convert tensor to float list
    def to_list(tensor):
        return tensor.detach().cpu().numpy().tolist()

    # Run inference
    for idx, (images, _) in enumerate(loader):
        correct_boxes = [i['bbox'] for i in dataset.coco.loadAnns(dataset.coco.getAnnIds(dataset.coco.getImgIds()[idx]))]
        img_id = dataset.ids[idx]
        with torch.no_grad():
            outputs = model(images)



        label_map = {i: name for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}

        # Only proceed if output_dir is not None or idx == 0
        if output_dir is not None or idx <= 5:
            if output_dir is not None:
                if dataset.hazard is None:
                    save_path = f"{output_dir}/clean/{idx}.jpg"
                else:
                    save_path = f"{output_dir}/{dataset.hazard}/{idx}_{dataset.severeLevel}.jpg"
            else:
                save_path = None

            show_predictions(images, outputs[0], label_map=label_map, save_path=save_path, correct_boxes=correct_boxes)


        output = outputs[0]
        boxes = output['boxes']
        scores = output['scores']
        labels = output['labels']

        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue

            label_idx = label.item()
            if label_idx >= len(COCO_INSTANCE_CATEGORY_NAMES):
                print(f"⚠️ Skipping unknown label index: {label_idx}")
                continue  # skip invalid label

            coco_label_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            nuimages_cat_id = name_to_id.get(coco_label_name)
            if nuimages_cat_id is None:
                continue  # skip if not in my annotation categories

            # Convert [xmin, ymin, xmax, ymax] to [x, y, width, height]
            box = to_list(box)
            x = box[0]
            y = box[1]
            width = box[2] - box[0]
            height = box[3] - box[1]

            results.append({
                "image_id": img_id,
                "category_id": nuimages_cat_id,
                "bbox": [round(x, 2), round(y, 2), round(width, 2), round(height, 2)],
                "score": round(float(score), 3)
            })

    # Save to result.json
    os.makedirs(f"outputs", exist_ok=True)
    if (dataset.hazard == None):
        with open(f"outputs/{type(model).__name__}_results_clean.json", "w") as f:
            json.dump(results, f)
    else:
        with open(f"outputs/{type(model).__name__}_results_{dataset.hazard}_{dataset.severeLevel}.json", "w") as f:
            json.dump(results, f)


def start_prediction_2(
    model,
    dataset: NuImagesCOCODataset,
    threshold: float = 0.5,
    showProgress: bool = False,
    save_every_n: int = 1,
    result_path: str = "outputs"
):
    # 1) Prepare directories
    tmp_folder = f"./temp_{type(model).__name__}_{dataset.hazard + '_' + str(dataset.severeLevel) if dataset.hazard != None else "clean"}_output"
    os.makedirs(tmp_folder, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)

    def to_list(t):
        return t.detach().cpu().numpy().tolist()

    # 2) Inference + per-image dump (skipping existing)
    per_image = []
    old_tmp_path = ""
    for idx, (images, _) in (
        tqdm(enumerate(loader), total=len(loader))
        if showProgress
        else enumerate(loader)
    ):
        tmp_path = os.path.join(tmp_folder, f"{idx // save_every_n}.json")

        
        # ← Skip if already processed
        if old_tmp_path == tmp_path or os.path.exists(tmp_path):
            old_tmp_path = tmp_path
            continue

        img_id = dataset.ids[idx]
        with torch.no_grad():
            out = model(images)[0]

        for box, score, label in zip(
            out["boxes"], out["scores"], out["labels"]
        ):
            if score < threshold:
                continue

            label_idx = label.item()
            if label_idx >= len(COCO_INSTANCE_CATEGORY_NAMES):
                continue

            name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            nu_id = name_to_id.get(name)
            if nu_id is None:
                continue

            x1, y1, x2, y2 = to_list(box)
            per_image.append(
                {
                    "image_id":    img_id,
                    "category_id": nu_id,
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                    "score": round(float(score), 3),
                }
            )

        # only write the per-image JSON every N images
        # idx is 0-based, so (idx+1) % save_every_n == 0 means “every Nth image”
        if (idx + 1) % save_every_n == 0 or idx == len(loader) - 1:
            with open(tmp_path, "w") as f:
                json.dump(per_image, f)
            per_image = []

    # 3) Merge all into one final JSON
    print("Merging all detections…")
    all_results = []
    for fname in sorted(os.listdir(tmp_folder), key=lambda x: int(x.split(".")[0])):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(tmp_folder, fname)) as f:
            all_results.extend(json.load(f))

    os.makedirs(result_path, exist_ok=True)
    suffix = f"{dataset.hazard}_{dataset.severeLevel}" if dataset.hazard else "clean"
    final_path = os.path.join(result_path, f"{type(model).__name__}_results_{suffix}.json")

    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Merged {len(all_results)} detections into {final_path}")

def start_prediction_3(
    model,
    dataset: NuImagesCOCODataset,
    threshold: float = 0.5,
    showProgress: bool = False,
    save_every_n: int = 1,
    batch_size: int = 1,
):
    # 1) Prepare directories
    tmp_folder = f"./temp_{type(model).__name__}_{dataset.hazard + '_' + str(dataset.severeLevel) if dataset.hazard != None else 'clean'}_output"
    os.makedirs(tmp_folder, exist_ok=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    def to_list(t):
        return t.detach().cpu().numpy().tolist()

    # 2) Inference + per-image dump (skipping existing)
    per_image = []
    old_tmp_path = ""
    for idx, (images, _) in (
        tqdm(enumerate(loader), total=len(loader))
        if showProgress
        else enumerate(loader)
    ):
        tmp_path = os.path.join(tmp_folder, f"{idx // save_every_n}.json")

        # ← Skip if already processed
        if old_tmp_path == tmp_path or os.path.exists(tmp_path):
            old_tmp_path = tmp_path
            continue

        batch_start = idx * batch_size
        batch_end = batch_start + images.shape[0]
        batch_img_ids = dataset.ids[batch_start:batch_end]

        with torch.no_grad():
            outs = model(images)

        for b, out in enumerate(outs):
            img_id = batch_img_ids[b]
            for box, score, label in zip(
                out["boxes"], out["scores"], out["labels"]
            ):
                if score < threshold:
                    continue

                label_idx = label.item()
                if label_idx >= len(COCO_INSTANCE_CATEGORY_NAMES):
                    continue

                name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                nu_id = name_to_id.get(name)
                if nu_id is None:
                    continue

                x1, y1, x2, y2 = to_list(box)
                per_image.append(
                    {
                        "image_id":    img_id,
                        "category_id": nu_id,
                        "bbox": [
                            round(x1, 2),
                            round(y1, 2),
                            round(x2 - x1, 2),
                            round(y2 - y1, 2),
                        ],
                        "score": round(float(score), 3),
                    }
                )

        # only write the per-image JSON every N images
        # idx is 0-based, so (idx+1) % save_every_n == 0 means “every Nth image”
        if (idx + 1) % save_every_n == 0 or idx == len(loader) - 1:
            with open(tmp_path, "w") as f:
                json.dump(per_image, f)
            per_image = []

    # 3) Merge all into one final JSON
    print("Merging all detections…")
    all_results = []
    for fname in sorted(os.listdir(tmp_folder), key=lambda x: int(x.split(".")[0])):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(tmp_folder, fname)) as f:
            all_results.extend(json.load(f))

    os.makedirs("outputs", exist_ok=True)
    suffix = f"{dataset.hazard}_{dataset.severeLevel}" if dataset.hazard else "clean"
    final_path = f"outputs/{type(model).__name__}_results_{suffix}.json"

    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Merged {len(all_results)} detections into {final_path}")

def run_predictions_for_hazard(model,
                               hazard_type,
                               severity_levels,
                               parallel=False,
                               saveLocation:str = None,
                               ann_file:str = 'data/sets/nuimages/nuimages_coco.json',
                               img_dir:str = 'data/sets/generated',
                               batch_size: int = 1,
                               save_every_n: int = 100,
                               result_path: str = "outputs"
                               ):
    """
    Runs predictions for a given hazard type across specified severity levels.

    Parameters:
    - model: The model to use for prediction.
    - hazard_type (str): Type of hazard (e.g., "fog", "dark").
    - severity_levels (list of int): List of severity levels to evaluate.
    - parallel (bool): Whether to run predictions in parallel.
    - save (bool): Location to save prediction result image to file if exist.
    """
    def predict_for_level(level):
        dataset = NuImagesCOCODataset(
            img_dir=img_dir,
            ann_file=ann_file,
            transform=transforms.ToTensor(),
            hazard=hazard_type,
            severeLevel=level
        )

        if batch_size > 1 :
            start_prediction_3(model=model, dataset=dataset, save_every_n=save_every_n, showProgress=True, batch_size=batch_size)
        else :
            start_prediction_2(model=model, dataset=dataset, save_every_n=save_every_n, showProgress=True, result_path=result_path)


    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(predict_for_level, severity_levels), total=len(severity_levels)))
    else:
        for level in severity_levels:
            predict_for_level(level)
