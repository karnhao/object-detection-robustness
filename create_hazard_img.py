import cv2
import numpy as np
import random
import os
import json
from multiprocessing import Pool
from tqdm import tqdm

PROCESS_COUNT = 4

def add_rain(image, level):
    rain_layer = np.zeros_like(image)
    drops = level * 500  # more drops for higher severity
    for _ in range(drops):
        x = random.randint(0, image.shape[1] - 1)
        y = random.randint(0, image.shape[0] - 1)
        length = random.randint(5, 10 + level * 5)
        cv2.line(rain_layer, (x, y), (x, y + length), (200, 200, 200), 1)
    return cv2.addWeighted(image, 1 - 0.1 * level, rain_layer, 0.1 * level, 0)

def add_snow(image, level):
    snow_layer = np.zeros_like(image)
    flakes = level * 500
    for _ in range(flakes):
        x = random.randint(0, image.shape[1] - 1)
        y = random.randint(0, image.shape[0] - 1)
        radius = random.randint(1, level + 2)
        cv2.circle(snow_layer, (x, y), radius, (255, 255, 255), -1)
    return cv2.addWeighted(image, 1 - 0.1 * level, snow_layer, 0.1 * level, 0)

def add_fog(image, level):
    fog_layer = np.full_like(image, 255)
    blur_strength = 51 + level * 50
    mask = cv2.GaussianBlur(fog_layer, (blur_strength, blur_strength), 0)
    return cv2.addWeighted(image, 1 - 0.2 * level, mask, 0.2 * level, 0)

def add_darkness(image, level):
    alpha = 1 - 0.2 * level
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def add_blur(image, level):
    # Kernel size increases with level
    ksize = 5 + level * 4
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def add_motion_blur(image, level):
    # Create a horizontal motion blur kernel
    ksize = 5 + level * 5
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1)/2), :] = np.ones(ksize)
    kernel /= ksize
    return cv2.filter2D(image, -1, kernel)

def add_sandstorm(image, level):
    # haze = np.full_like(image, (180, 160, 130))  # sandy tint
    haze = np.full_like(image, (130, 160, 180))  # sandy tint
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    sand_layer = cv2.add(haze, noise)
    ksize = 15 + level * 5
    if ksize % 2 == 0:
        ksize += 1  # make it odd
    sand_layer = cv2.GaussianBlur(sand_layer, (ksize, ksize), 0)
    return cv2.addWeighted(image, 1 - 0.3 * level, sand_layer, 0.3 * level, 0)


def add_hail(image, level):
    hail_layer = np.zeros_like(image)
    hail_count = level * 300
    for _ in range(hail_count):
        x = random.randint(0, image.shape[1] - 1)
        y = random.randint(0, image.shape[0] - 1)
        radius = random.randint(2, 4 + level)
        cv2.circle(hail_layer, (x, y), radius, (230, 230, 230), -1)
    return cv2.addWeighted(image, 1 - 0.2 * level, hail_layer, 0.2 * level, 0)

def add_lightning(image, level):
    lightning_layer = image.copy()
    if random.random() < 0.5:  # occasional flash
        x = random.randint(0, image.shape[1] - 1)
        y = random.randint(0, image.shape[0] - 1)
        for i in range(5 + level):
            end_x = x + random.randint(-30, 30)
            end_y = y + random.randint(50, 100)
            cv2.line(lightning_layer, (x, y), (end_x, end_y), (255, 255, 255), 2)
            x, y = end_x, end_y
        lightning_layer = cv2.convertScaleAbs(lightning_layer, alpha=1.5, beta=30)
    return lightning_layer

def add_heatwave(image, level):
    distorted = image.copy()
    rows, cols = image.shape[:2]
    for i in range(0, rows, 1):
        offset = int(4 * level * np.sin(i / 20.0))
        distorted[i:i+1, :] = np.roll(distorted[i:i+1, :], offset, axis=1)
    tinted = cv2.addWeighted(distorted, 1, np.full_like(image, (20, 20, 0)), 0.2 * level, 0)
    return tinted

def add_fog_then_rain(image, level):
    return add_rain(add_fog(image, level), level)

def process_image_chunk(images_info_chunk, image_root_dir, output_dir, position):
    effects = {
        "fog_rain": add_fog_then_rain
        # "heatwave": add_heatwave,
        # "sandstorm": add_sandstorm
        # "motion_blur": add_motion_blur
        # Add other effects as needed
    }

    # Initialize tqdm progress bar for this chunk
    for img_info in tqdm(images_info_chunk, desc=f"Process {os.getpid()}", position=position):
        file_name = img_info.get("file_name")
        if not file_name:
            continue

        image_path = os.path.join(image_root_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}.")
            continue

        base_name, ext = os.path.splitext(os.path.basename(file_name))

        # Remove 'samples/' prefix if present
        relative_path = file_name
        if relative_path.startswith("samples/"):
            relative_path = relative_path[len("samples/"):]

        subdir = os.path.dirname(relative_path)  # e.g., CAM_FRONT

        for effect_name, effect_func in effects.items():
            for level in range(1, 4):

                output_subdir = os.path.join(output_dir, effect_name, subdir)
                os.makedirs(output_subdir, exist_ok=True)

                output_filename = f"{base_name}_{effect_name}_{level}{ext}"
                output_path = os.path.join(output_subdir, output_filename)

                if os.path.exists(output_path):
                    continue

                output_img = effect_func(image, level)
                
                cv2.imwrite(output_path, output_img)
                # tqdm.write(f"Saved: {output_path}")

def run_parallel_from_coco(coco_json_path, image_root_dir, output_dir, process_count):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images_info = coco_data.get("images", [])
    chunk_size = len(images_info) // process_count + 1
    chunks = [images_info[i:i + chunk_size] for i in range(0, len(images_info), chunk_size)]

    args = [(chunk, image_root_dir, output_dir, idx) for idx, chunk in enumerate(chunks)]
    os.makedirs(output_dir, exist_ok=True)

    with Pool(process_count) as pool:
        pool.starmap(process_image_chunk, args)

if __name__ == "__main__":
    coco_json_path = "data/sets/nuimages/nuimages_200.json"
    image_root_dir = "data/sets/nuimages"
    output_dir = "data/sets/generated/nuimages-200"
    process_count = PROCESS_COUNT

    run_parallel_from_coco(coco_json_path, image_root_dir, output_dir, process_count)
