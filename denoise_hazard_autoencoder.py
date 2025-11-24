import os
import torch
from tool import run_predictions_for_hazard

import platform
import sys
import time
import json
from datetime import datetime

def shutdown_computer():
    try:
        # Detect the operating system
        current_os = platform.system().lower()

        if "windows" in current_os:
            # Windows shutdown command
            os.system("shutdown /s /t 1")
        elif "linux" in current_os or "darwin" in current_os:  # Darwin is macOS
            # Linux and macOS shutdown command
            os.system("sudo shutdown now")
        else:
            print("Unsupported operating system.")
            sys.exit(1)

        print("Shutdown command executed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

generated_hazards = os.listdir("./data/sets/denoised")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")

gt_path = "data/sets/nuimages/nuimages_200.json"
output_path = "outputs/autoencoder_no_denoised"

def printParams(model):
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024**2)  # float32 = 4 bytes

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Approx. size in MB: {size_mb:.2f} MB")

    stats = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "size_mb": float(size_mb)
    }
    return stats

def getFasterRCNN():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    model.to(device)
    model.eval()
    stats = printParams(model)
    return model

def getMaskRCNN():
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)
    model.eval()
    stats = printParams(model)
    return model

def getSSD():
    from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.to(device)
    model.eval()
    stats = printParams(model)
    return model

def getFastRCNNWrapper():
    from fast_rcnn import FastRCNNWrapper
    model = FastRCNNWrapper()
    stats = printParams(model)
    return model

def getWrappedDETR():
    from detr import WrappedDETR
    model = WrappedDETR()
    model.to(device)
    model.eval()
    stats = printParams(model)
    return model

def getWrappedYOLO():
    from yolo import WrappedYOLO
    model = WrappedYOLO()
    model.to(device)
    stats = printParams(model)
    return model

model_loaders = {
    "FasterRCNN": getFasterRCNN,
    "WrappedYOLO": getWrappedYOLO,
    "SSD": getSSD,
    "MaskRCNN": getMaskRCNN,
    "WrappedDETR": getWrappedDETR,
    "FastRCNNWrapper": getFastRCNNWrapper,
}

parallel_settings = {
    "FasterRCNN": True,
    "WrappedYOLO": False,
    "SSD": True,
    "MaskRCNN": True,
    "WrappedDETR": True,
    "FastRCNNWrapper": True,
}

models = {name: None for name in model_loaders}

def _ensure_outputs_dir():
    out_dir = os.path.join(".", output_path)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save_model_stats(model_name, hazard, params_stats, timing_stats):
    out_dir = _ensure_outputs_dir()
    fname = f"{model_name}_stats_{hazard}.json"
    path = os.path.join(out_dir, fname)
    payload = {
        "model": model_name,
        "hazard": hazard,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "params": params_stats,
        "timing": timing_stats
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved stats to {path}")
    except Exception as e:
        print(f"Failed to save stats to {path}: {e}")

def hazard_predictions(model_name, hazard):
    if not all(os.path.exists(os.path.join(output_path,f"{model_name}_results_{hazard}_{i}.json")) for i in [3]):
        if models[model_name] is None:
            models[model_name] = model_loaders[model_name]()
        model = models[model_name]

        # prepare a counter by wrapping model.__call__ when possible
        call_count = {"n": 0}
        original_call = None
        wrapped = False
        if hasattr(model, "__call__") and callable(model.__call__):
            try:
                original_call = model.__call__
                def _wrapped_call(*args, **kwargs):
                    call_count["n"] += 1
                    return original_call(*args, **kwargs)
                model.__call__ = _wrapped_call
                wrapped = True
            except Exception as e:
                # If wrapping fails, continue without counting
                print(f"Could not wrap model call for counting: {e}")
                wrapped = False

        # Run and time predictions
        start = time.time()
        run_predictions_for_hazard(
            model,
            hazard_type=hazard,
            severity_levels=[3],
            parallel=parallel_settings[model_name],
            ann_file=gt_path,
            batch_size=1,
            save_every_n=100,
            img_dir='data/sets/generated/nuimages-200',
            result_path=output_path
        )
        elapsed = time.time() - start

        # restore original call
        if wrapped and original_call is not None:
            try:
                model.__call__ = original_call
            except Exception:
                pass

        # gather stats
        params_stats = {}
        try:
            params_stats = printParams(model)
        except Exception:
            # if printParams fails, try to fill minimal info
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                params_stats = {
                    "total_params": int(total_params),
                    "trainable_params": int(trainable_params),
                    "size_mb": float(total_params * 4 / (1024**2))
                }
            except Exception:
                params_stats = {}

        inference_calls = int(call_count["n"])
        ips = inference_calls / elapsed if elapsed > 0 else None

        timing_stats = {
            "total_time_seconds": float(elapsed),
            "inference_calls": inference_calls,
            "iterations_per_second": float(ips) if ips is not None else None
        }

        _save_model_stats(model_name, hazard, params_stats, timing_stats)

    else:
        print(f"{model_name} {hazard} results already exists.")

for hazard in generated_hazards:
    for model_name in model_loaders:
        hazard_predictions(model_name, hazard)


# shutdown_computer()
