# Object Detection Robustness

Analyzing the impact of image corruption on object detection model performance.


## Introduction
This project evaluates the robustness of six object detection models under various weather conditions and investigates how Autoencoder-based denoising can improve detection performance. The work is divided into **three main tasks**:

---

## Baseline Evaluation on NuImages
The first task uses raw images from the **NuImages dataset** to compute the baseline performance of six object detection models:

- Fast R-CNN
- Faster R-CNN
- Mask R-CNN
- SSD
- YOLO
- DETR

### **Metrics Computed**
- Precision
- Recall
- F1-Score

These scores serve as the benchmark before applying any denoising techniques.

---

## Autoencoder for Weather Denoising
The second task involves building an **Autoencoder** to remove noise caused by different weather conditions.  
The denoising model was designed to handle **8 types of weather**:

1. Fog  
2. Rain  
3. Snow  
4. Dark / Low Light  
5. Sandstorm / Dust  
6. Blur  
7. Motion Blur  
8. Heatwave Distortion  

After denoising, the cleaned images were tested again on all six detection models:

- Fast R-CNN  
- Faster R-CNN  
- Mask R-CNN  
- SSD  
- YOLO  
- DETR  

This allows comparison between **original vs. denoised** image performance.

---

## Cross-Weather Autoencoder
The third task introduces a **Cross Autoencoder** capable of restoring images affected by **multiple overlapping weather conditions**.  
Examples include:

- Fog + Rain  

The goal is to determine whether the Autoencoder can reconstruct multi-weather images effectively, and how this influences detection accuracy of the 6 models.


### Dataset
 - [NuImages](https://www.nuscenes.org/nuimages)
 - Locate in `data/sets/nuimages`

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NuImages

Download the NuImages dataset from [nuScenes.org](https://www.nuscenes.org/nuimages) and extract it to:
```
data/sets/nuimages/
```

## Usage

### Convert NuImages to COCO Format

```bash
python nuimages_to_coco.py
```

This will generate:
```
data/sets/nuimages/nuimages_coco.json
```

### Run Inference

- **Clean Images**: `python inference_clean.py`
- **Corrupted/Hazard Images**: `python inference_hazard.py`

### Object Detection Models

- **DETR**: `detr.py`
- **Faster R-CNN**: `fast_rcnn.py`
- **YOLO**: `yolo.py`

### Image Processing & Corruption

- **Create Hazard Images**: `create_hazard_img.py`
- **Select Images**: `select_images.py`
- **Denoise with Autoencoder**: `denoise_hazard_autoencoder.py`

### Autoencoder Denoising

The `autoencoder/` directory contains models for removing various types of image corruption:

- **Fog Denoising**: `denoise_fog_rain.py`
- **Multi-corruption Denoising**: `create_multi_denoise_img.py`
- **Single Image Denoising**: `create_denoise_img.py`

Includes pretrained weights for:
- `autoencoder_dark_3.weights.h5`
- `autoencoder_fog_3.weights.h5`
- `autoencoder_rain_3.weights.h5`
- `autoencoder_snow_3.weights.h5`
- `crossencoder_fog_rain_3.weights.h5`

## Evaluation



## File Structure

```
object-detection-robustness/
├── create_hazard_img.py              # Generate corrupted images
├── denoise_hazard_autoencoder.py     # Denoise corrupted images
├── inference_clean.py                 # Inference on clean images
├── inference_hazard.py                # Inference on corrupted images
├── nuimages_to_coco.py                # Convert NuImages to COCO format
├── detr.py                            # DETR model implementation
├── fast_rcnn.py                       # Faster R-CNN model implementation
├── yolo.py                            # YOLO model implementation
├── select_images.py                   # Image selection utility
├── tool.py                            # Utility functions
├── requirements.txt                   # Project dependencies
├── README.md                          # This file
└── autoencoder/                       # Denoising autoencoder models
    ├── cross_autoencoder.py           # Cross-corruption autoencoder
    ├── denoise_fog_rain.py            # Fog/rain denoising
    ├── create_denoise_img.py          # Single image denoising
    ├── create_multi_denoise_img.py    # Multi-image denoising
    ├── *.ipynb                        # Jupyter notebooks for model development
    └── weights/                       # Pretrained model weights
        ├── autoencoder_dark_3.weights.h5
        ├── autoencoder_fog_3.weights.h5
        ├── autoencoder_rain_3.weights.h5
        ├── autoencoder_snow_3.weights.h5
        └── crossencoder_fog_rain_3.weights.h5
```

## License

This project uses NuImages under the nuScenes license. All code is MIT licensed unless otherwise noted.
