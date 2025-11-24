import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO  # Requires `pip install ultralytics`

class WrappedYOLO(torch.nn.Module):
    def __init__(self, model_name="yolov8n.pt", confidence_threshold=0.5):
        super().__init__()
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def forward(self, images):
        # Accepts list of PIL images or tensors in [0, 1] range
        if isinstance(images, torch.Tensor):
            images = [F.to_pil_image(img) for img in images]

        results = self.model(images, verbose=False)

        formatted_results = []
        for result in results:
            boxes = result.boxes.xyxy  # (x1, y1, x2, y2)
            scores = result.boxes.conf
            labels = result.boxes.cls.to(torch.int64)

            # Filter by confidence threshold
            keep = scores > self.confidence_threshold
            formatted_results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": [i+1 for i in labels[keep]]
            })

        return formatted_results