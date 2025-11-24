import os
import json
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Resizing, Concatenate
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


# === CrossEncoder definition ===
def build_crossencoder(input_shape, ae1, ae2):
    """
    Build a CrossEncoder that combines outputs of two pretrained AutoEncoders.
    ae1, ae2: pretrained single-hazard autoencoders (frozen).
    """
    input_img = Input(shape=input_shape)

    # Pass input through both pretrained AEs
    out1 = ae1(input_img)
    out2 = ae2(input_img)

    # Concatenate their outputs
    merged = Concatenate(axis=-1)([out1, out2])

    h, w = input_shape[:2]

    # CrossEncoder layers
    x = Conv2D(64, (3, 3), padding='same')(merged)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    x = Resizing(h, w)(x)

    crossencoder = Model(input_img, x)
    crossencoder.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
    return crossencoder

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
    weight_path1, weight_path2, cross_weight_path,
    output_dir,
    corruption_types=("fog", "rain"),
    severity_level=2,
    chunk_size=(200, 150),
    full_size=(1600, 900)
):
    # Load model
    autoencoder1 = build_autoencoder(chunk_size[::-1] + (3,))
    autoencoder1.load_weights(weight_path1)

    # Load model
    autoencoder2 = build_autoencoder(chunk_size[::-1] + (3,))
    autoencoder2.load_weights(weight_path2)

    crossencoder = build_crossencoder(chunk_size[::-1] + (3,), autoencoder1, autoencoder2)
    crossencoder.load_weights(cross_weight_path)

    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for img_info in tqdm(coco_data['images'], desc=f"Denoising {corruption_types}..."):
        relative_path = img_info['file_name']
        base_name = os.path.splitext(os.path.basename(relative_path))[0]
        camera_dir = os.path.basename(os.path.dirname(relative_path))

        corrupted_fname = f"{base_name}_{'_'.join(corruption_types)}_{severity_level}.jpg"
        corrupted_path = os.path.join(corruption_root, camera_dir, corrupted_fname)

        if not os.path.exists(corrupted_path):
            tqdm.write(f"Skipping missing file: {corrupted_path}")
            continue

        corrupted_img = cv2.imread(corrupted_path)
        corrupted_img = cv2.resize(corrupted_img, full_size)

        corrupted_chunks = split_image(corrupted_img, chunk_size)
        denoised_chunks = crossencoder.predict(corrupted_chunks, verbose=0)
        denoised_img = merge_chunks(denoised_chunks, full_size, chunk_size)

        os.makedirs(os.path.join(output_dir, '_'.join(corruption_types), camera_dir), exist_ok=True)
        output_path = os.path.join(output_dir, '_'.join(corruption_types), camera_dir, corrupted_fname)
        cv2.imwrite(output_path, denoised_img)
        # tqdm.write(f"Saved: {output_path}")

# === Example usage ===
if __name__ == "__main__":

    corruption_typess = [("fog", "rain")]
    severity_level = 3

    for corruption_types in corruption_typess:

        batch_denoise_from_coco(
            coco_json_path="../data/sets/nuimages/nuimages_200.json",
            corruption_root=f"../data/sets/generated/nuimages-200/{'_'.join(corruption_types)}",
            weight_path1=f"weights/autoencoder_{corruption_types[0]}_{severity_level}.weights.h5",
            weight_path2=f"weights/autoencoder_{corruption_types[1]}_{severity_level}.weights.h5",
            cross_weight_path=f"weights/crossencoder_{'_'.join(corruption_types)}_{severity_level}.weights.h5",
            output_dir="../data/sets/denoised",
            corruption_types=corruption_types,
            severity_level=severity_level,
            chunk_size=(1600, 900),
            full_size=(1600, 900)
        )
