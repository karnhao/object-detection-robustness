import json
from nuimages import NuImages
from tqdm import tqdm

# NuImages → COCO label mapping
NUIMAGES_TO_COCO = {
    'human.pedestrian.adult': 'person',
    'human.pedestrian.child': 'person',
    'human.pedestrian.construction_worker': 'person',
    'human.pedestrian.police_officer': 'person',
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.rigid': 'bus',
    'vehicle.bus.bendy': 'bus',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.trailer': 'truck',
    'vehicle.construction': 'truck',
    'movable_object.barrier': None,
    'movable_object.trafficcone': None,
    'animal': None,
    'object.unknown': None,
    'flat.driveable_surface': None,
    'vehicle.ego': None,
    'static.manmade': None,
    'static.other': None,
    'static.vegetation': None,
    'flat.sidewalk': None,
    'flat.terrain': None
}




VERSION = 'v1.0-train'
SAVE_COCO_FILE_NAME = 'nuimages_coco_train'




# Initialize NuImages
nuim = NuImages(dataroot='data/sets/nuimages', version=VERSION, verbose=True, lazy=True)

# COCO format containers
coco = {
    "info": {
        "description": "NuImages converted to COCO format",
        "version": nuim.version,
        "year": 2020
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

# Build category mapping: COCO-style name → ID
category_name_to_id = {}
category_token_to_id = {}
category_id = 1

for cat in nuim.category:
    nu_name = cat['name']
    coco_name = NUIMAGES_TO_COCO.get(nu_name)
    if coco_name is None:
        continue
    if coco_name not in category_name_to_id:
        category_name_to_id[coco_name] = category_id
        coco["categories"].append({
            "id": category_id,
            "name": coco_name,
            "supercategory": coco_name
        })
        category_id += 1
    category_token_to_id[cat['token']] = category_name_to_id[coco_name]

# Add images from sample_data
image_tokens = set()
for sd in nuim.sample_data:
    if sd['is_key_frame'] and sd['filename'].endswith('.jpg'):
        image_id = int(sd['token'][:16], 16)
        image_tokens.add(sd['token'])
        coco["images"].append({
            "id": image_id,
            "file_name": sd['filename'],
            "width": sd['width'],
            "height": sd['height']
        })

# Add annotations
ann_id = 0
for ann in tqdm(nuim.object_ann):
    if ann['sample_data_token'] not in image_tokens:
        continue
    cat_id = category_token_to_id.get(ann['category_token'])
    if not cat_id:
        continue
    bbox = ann['bbox']  # [xmin, ymin, xmax, ymax]
    x = bbox[0]
    y = bbox[1]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image_id = int(ann['sample_data_token'][:16], 16)
    coco["annotations"].append({
        "id": ann_id,
        "image_id": image_id,
        "category_id": cat_id,
        "bbox": [x, y, width, height],
        "area": width * height,
        "iscrowd": 0,
        "segmentation": []
    })
    ann_id += 1

# Save to JSON
with open(f'data/sets/nuimages/{SAVE_COCO_FILE_NAME}.json', 'w') as f:
    json.dump(coco, f)
