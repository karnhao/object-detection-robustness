import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ProposalGenerator:
    """
    Generates region proposals using sliding windows at multiple scales and aspect ratios.
    This is a simple heuristic proposal generator for Fast R-CNN.
    """

    def __init__(self, sizes=[64, 128, 256], aspect_ratios=[0.5, 1.0, 2.0], stride=32):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride

    def generate(self, image_tensor):
        """
        Args:
            image_tensor: Tensor[C,H,W] in [0,1]
        Returns:
            Tensor[N,4] of proposals in XYXY format
        """
        _, H, W = image_tensor.shape
        proposals = []

        for size in self.sizes:
            for ar in self.aspect_ratios:
                w = int(size * (ar ** 0.5))
                h = int(size / (ar ** 0.5))
                for y in range(0, H - h, self.stride):
                    for x in range(0, W - w, self.stride):
                        x1 = x
                        y1 = y
                        x2 = x + w
                        y2 = y + h
                        if x2 <= W and y2 <= H:
                            proposals.append([x1, y1, x2, y2])

        return torch.tensor(proposals, dtype=torch.float32)

class FastRCNNWrapper(torch.nn.Module):
    """
    A torchvision-style detection model that mimics Fast R-CNN.
    Accepts list[Tensor] images and optional list[Tensor] proposals.
    If proposals are not provided, uses ProposalGenerator to generate them.
    """

    def __init__(self, num_classes=91, proposal_generator=None):
        super().__init__()
        base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.backbone = base_model.backbone
        self.roi_heads = base_model.roi_heads
        self.transform = base_model.transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proposal_generator = proposal_generator or ProposalGenerator()
        self.to(self.device)
        self.eval()

    def forward(self, images, proposals_list=None):
        original_sizes = [img.shape[-2:] for img in images]

        # Apply transform manually
        transformed_images, _ = self.transform(images, None)
        images_tensor = transformed_images.tensors
        image_sizes = transformed_images.image_sizes

        # Extract features
        features = self.backbone(images_tensor)

        # Generate proposals if not provided
        if proposals_list is None:
            proposals_list = [self.proposal_generator.generate(img).to(self.device) for img in images]

        # Run ROI heads
        detections, _ = self.roi_heads(features, proposals_list, image_sizes)

        # Postprocess to original image sizes
        detections = self.transform.postprocess(detections, image_sizes, original_sizes)
        return detections
