# src/models/baseline.py

import os
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class FasterRCNNBaseline:
    """
    Wrapper around torchvision Faster R-CNN ResNet50-FPN for object detection.
    Reference: https://pytorch.org/vision/stable/models.html#faster-r-cnn
    """
    def __init__(self, device=None):
        # pick GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load pretrained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # standard transform to convert PIL→Tensor
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def predict(self, image_paths):
        """
        image_paths: list of file paths
        returns: list of dicts with keys 'boxes', 'labels', 'scores'
        """
        images = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            images.append(self.transform(img).to(self.device))

        with torch.no_grad():
            outputs = self.model(images)
        return outputs

if __name__ == "__main__":
    # -------------------------
    # Smoke‑test the baseline
    # -------------------------
    this_dir    = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

    img_dir = os.path.join(project_root, "data", "raw", "images")
    all_imgs = sorted(os.listdir(img_dir))[:2]
    sample_imgs = [os.path.join(img_dir, f) for f in all_imgs]

    detector = FasterRCNNBaseline()
    results = detector.predict(sample_imgs)

    # 1) Check number of outputs
    print("Number of images processed:", len(results))
    # 2) For first image
    first = results[0]
    num_dets = first["boxes"].shape[0]
    print("Detections in first image:", num_dets)
    # 3) Show first 3 boxes + scores + labels
    print("Example detections [x1, y1, x2, y2, score, label]:")
    for i in range(min(3, num_dets)):
        box = first["boxes"][i].cpu().tolist()
        score = first["scores"][i].item()
        label = first["labels"][i].item()
        print(f"  {box + [score, label]}")
