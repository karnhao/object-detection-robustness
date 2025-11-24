import torch
import torchvision.transforms.functional as F
from torchvision.ops import box_convert
from transformers import DetrForObjectDetection, DetrImageProcessor

class WrappedDETR(torch.nn.Module):
    def __init__(self, model_name="facebook/detr-resnet-50", confidence_threshold=0.5):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name).to(self.device)
        self.confidence_threshold = confidence_threshold

    def forward(self, images):
        # Expecting a list of PIL images or tensors in [0, 1] range
        if isinstance(images, torch.Tensor):
            images = [F.to_pil_image(img) for img in images]

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Post-process to match torchvision format
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.confidence_threshold, target_sizes=[img.size[::-1] for img in images]
        )

        formatted_results = []
        for result in results:
            boxes = result["boxes"].to("cpu")  # Move back to CPU if needed
            scores = result["scores"].to("cpu")
            labels = result["labels"].to("cpu")

            formatted_results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })

        return formatted_results
