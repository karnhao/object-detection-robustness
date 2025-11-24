import os
import json
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Resizing, Conv2DTranspose
from keras.optimizers import Adam
from tqdm import tqdm

# === Autoencoder definition ===
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    h, w = input_shape[:2]

    # Encoder
    x = Conv2D(64, (3, 3), activation=None, padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation=None, padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    # x = Conv2DTranspose(32, (3, 3), strides=2, activation=None, padding='same')(encoded)
    # x = Conv2DTranspose(64, (3, 3), strides=2, activation=None, padding='same')(x)
    # x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation=None, padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


    # Force output to match input shape
    x = Resizing(h, w)(x)

    autoencoder = Model(input_img, x)
    autoencoder.compile(optimizer=Adam(), loss="mse")
    return autoencoder

# === Image utilities ===
def split_image(image, chunk_size):
    h, w = image.shape[:2]
    cw, ch = chunk_size
    if h % ch != 0 or w % cw != 0:
        raise ValueError(f"Image size {w}x{h} not divisible by chunk size {cw}x{ch}")
    chunks = []
    for y in range(0, h, ch):
        for x in range(0, w, cw):
            chunk = image[y:y+ch, x:x+cw]
            chunks.append(chunk / 255.0)
    return np.array(chunks)


def merge_chunks(chunks, image_size, chunk_size):
    w, h = image_size
    cw, ch = chunk_size
    full_image = np.zeros((h, w, 3))
    idx = 0
    for y in range(0, h, ch):
        for x in range(0, w, cw):
            full_image[y:y+ch, x:x+cw] = chunks[idx]
            idx += 1

    return (full_image * 255).astype(np.uint8)

# === Batch denoising from COCO JSON ===
def batch_denoise_from_coco(
    coco_json_path,
    corruption_root,
    weight_paths,            # ← เปลี่ยนเป็น list เช่น ["fog", "rain"]
    output_dir,
    corruption_type="fog_rain",
    severity_level=2,
    chunk_size=(200, 150),
    full_size=(1600, 900)
):

    print(os.path.abspath(coco_json_path))
    print(os.path.exists(coco_json_path))


    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # เตรียมโมเดล autoencoder หนึ่งตัวใช้ร่วมทุก weight
    autoencoder = build_autoencoder(chunk_size[::-1] + (3,))

    # corruption_type สำหรับชื่อไฟล์
    save_type = corruption_type

    for img_info in tqdm(coco_data['images'], desc=f"Denoising {corruption_type}..."):
        relative_path = img_info['file_name']
        base_name = os.path.splitext(os.path.basename(relative_path))[0]
        camera_dir = os.path.basename(os.path.dirname(relative_path))

        corrupted_fname = f"{base_name}_{corruption_type}_{severity_level}.jpg"
        corrupted_path = os.path.join(corruption_root, camera_dir, corrupted_fname)

        if not os.path.exists(corrupted_path):
            tqdm.write(f"Skipping missing file: {corrupted_path}")
            continue

        # โหลดรูป corrupted
        img = cv2.imread(corrupted_path)
        img = cv2.resize(img, full_size)

        # แปลงเป็น chunks
        chunks = split_image(img, chunk_size)

        # === apply models sequentially ===
        for wt in weight_paths:
            weight_file = f"weights/autoencoder_{wt}_{severity_level}.weights.h5"
            if not os.path.exists(weight_file):
                tqdm.write(f"Missing weight: {weight_file}")
                continue

            autoencoder.load_weights(weight_file)
            chunks = autoencoder.predict(chunks, verbose=0)

        # merge กลับ
        final_img = merge_chunks(chunks, full_size, chunk_size)

        # save
        out_dir = os.path.join(output_dir, save_type, camera_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, corrupted_fname)
        cv2.imwrite(out_path, final_img)


# === Example usage ===
if __name__ == "__main__":

    corruption_type = "fog_rain"
    severity_level = 3

    # ใช้ fog ก่อน rain
    weight_order = ["rain","fog"]

    batch_denoise_from_coco(
        coco_json_path="../data/sets/nuimages/nuimages_200.json",
        corruption_root=f"../data/sets/generated/nuimages-200/fog_rain",
        weight_paths=weight_order,                     # ← list ของ weight
        output_dir="../data/sets/denoised/test",
        corruption_type=corruption_type,
        severity_level=severity_level,
        chunk_size=(200, 150),
        full_size=(1600, 900)
    )

