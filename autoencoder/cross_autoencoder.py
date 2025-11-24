# %%
import os
import json
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Resizing, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
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
    h, w = image_size
    cw, ch = chunk_size
    full_image = np.zeros((h, w, 3))
    idx = 0
    for y in range(0, h, ch):
        for x in range(0, w, cw):
            full_image[y:y+ch, x:x+cw] = chunks[idx]
            idx += 1
    return full_image


# %%
# === Base Autoencoder definition (same as before) ===
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
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
    return autoencoder

# %%
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

# %%
# === Dataset generator for multi-hazard images ===
def image_chunk_generator_multi(
    coco_json_path,
    clean_root_prefix,
    corruption_root,
    corruption_types,
    severity_level=2,
    chunk_size=(200, 150),
    full_size=(1600, 900)
):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    for img_info in coco_data['images']:
        relative_path = img_info['file_name']
        clean_path = os.path.join(clean_root_prefix, relative_path)

        base_name = os.path.splitext(os.path.basename(relative_path))[0]
        camera_dir = os.path.basename(os.path.dirname(relative_path))

        # Build multi-hazard corrupted filename (e.g., rain+snow)
        corrupted_fname = f"{base_name}_{corruption_types[0]}_{corruption_types[1]}_{severity_level}.jpg"
        corrupted_path = os.path.join(corruption_root, camera_dir, corrupted_fname)

        clean = cv2.imread(clean_path)
        corrupted = cv2.imread(corrupted_path)

        if clean is None or corrupted is None:
            continue

        clean = cv2.resize(clean, full_size)
        corrupted = cv2.resize(corrupted, full_size)

        clean_chunks = split_image(clean, chunk_size)
        corrupted_chunks = split_image(corrupted, chunk_size)

        for c_chunk, cl_chunk in zip(corrupted_chunks, clean_chunks):
            yield c_chunk.astype(np.float32), cl_chunk.astype(np.float32)

# %%
def get_datasets_multi(coco_json_path, clean_root_prefix, corruption_root,
                       corruption_types, severity_level=2,
                       chunk_size=(200, 150), full_size=(1600, 900),
                       batch_size=32, val_split=0.1):

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    n_total = len(images)
    n_val = int(n_total * val_split)

    train_images = images[:-n_val]
    val_images = images[-n_val:]

    output_signature = (
        tf.TensorSpec(shape=(chunk_size[1], chunk_size[0], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(chunk_size[1], chunk_size[0], 3), dtype=tf.float32)
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: image_chunk_generator_multi(coco_json_path, clean_root_prefix, corruption_root,
                                            corruption_types, severity_level, chunk_size, full_size),
        output_signature=output_signature
    ).shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: image_chunk_generator_multi(coco_json_path, clean_root_prefix, corruption_root,
                                            corruption_types, severity_level, chunk_size, full_size),
        output_signature=output_signature
    ).skip(len(train_images)).take(n_val).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# %%

def show_full_image_result(model, corrupted_img, clean_img, chunk_size=(200, 150)):
    corrupted_chunks = split_image(corrupted_img, chunk_size)
    predicted_chunks = model.predict(corrupted_chunks)
    
    reconstructed_img = merge_chunks(predicted_chunks, corrupted_img.shape[:2], chunk_size)


    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Corrupted")
    plt.imshow(corrupted_img[..., ::-1] / 255.0)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Clean")
    plt.imshow(clean_img[..., ::-1] / 255.0)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Denoised")
    plt.imshow(reconstructed_img[..., ::-1])
    plt.axis('off')
    plt.show()


# %%
# === Training function for CrossEncoder ===
def train_crossencoder(
    coco_json_path,
    clean_root_prefix,
    corruption_root,
    corruption_types,
    severity_level=2,
    chunk_size=(200, 150),
    full_size=(1600, 900),
    epochs=50,
    batch_size=32,
    weights_dir="weights",
):
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, f"crossencoder_{corruption_types[0]}_{corruption_types[1]}_{severity_level}.weights.h5")

    train_ds, val_ds = get_datasets_multi(
        coco_json_path, clean_root_prefix, corruption_root,
        corruption_types, severity_level, chunk_size, full_size,
        batch_size, val_split=0.1
    )

    # Peek one batch to build model
    sample_batch = next(iter(train_ds))

    # Load pretrained single-hazard AEs
    ae1 = build_autoencoder(sample_batch[0].shape[1:])
    ae1.load_weights(os.path.join(weights_dir, f"autoencoder_{corruption_types[0]}_{severity_level}.weights.h5"))
    ae2 = build_autoencoder(sample_batch[0].shape[1:])
    ae2.load_weights(os.path.join(weights_dir, f"autoencoder_{corruption_types[1]}_{severity_level}.weights.h5"))

    # Freeze them
    ae1.trainable = False
    ae2.trainable = False

    # Build CrossEncoder
    crossencoder = build_crossencoder(sample_batch[0].shape[1:], ae1, ae2)

    if os.path.exists(weight_path):
        print(f"Loading model weights from {weight_path}")
        crossencoder.load_weights(weight_path)
    else:
        print("Training new CrossEncoder...")
        crossencoder.fit(train_ds, validation_data=val_ds, epochs=epochs)
        crossencoder.save_weights(weight_path)
        print(f"CrossEncoder weights saved to {weight_path}")

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    sample_clean = cv2.imread(os.path.join(clean_root_prefix, coco_data['images'][0]['file_name']))
    sample_corrupted = cv2.imread(os.path.join(corruption_root, os.path.basename(os.path.dirname(coco_data['images'][0]['file_name'])), f"{os.path.splitext(os.path.basename(coco_data['images'][0]['file_name']))[0]}_{'_'.join(corruption_types)}_{severity_level}.jpg"))
    sample_clean = cv2.resize(sample_clean, full_size)
    sample_corrupted = cv2.resize(sample_corrupted, full_size)
    show_full_image_result(crossencoder, sample_corrupted, sample_clean, chunk_size)

    return crossencoder



# %%
coco_json_path = "../data/sets/nuimages/nuimages_1k.json"
corruption_types = ("fog", "rain")
clean_root_prefix = "../data/sets/nuimages"
corruption_root = f"../data/sets/generated/nuimages-1k/{'_'.join(corruption_types)}"
severity_level = 3

# %%
train_crossencoder(
    coco_json_path=coco_json_path,
    clean_root_prefix=clean_root_prefix,
    corruption_root=corruption_root,
    corruption_types=corruption_types,
    severity_level=severity_level,
    chunk_size=(200, 150),
    full_size=(1600, 900),
    epochs=8,
    batch_size=32,
    weights_dir="weights"
)


