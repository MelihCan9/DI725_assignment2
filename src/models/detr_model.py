# src/models/detr_model.py

import os
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import box_convert

class DETRModel:
    """
    PyTorch‐hub DETR ResNet‑50 wrapper for object detection.
    Paper: Carion et al., “End‐to‐End Object Detection with Transformers (DETR)”, ECCV 2020.
    Code: https://github.com/facebookresearch/detr
    """
    def __init__(self, device=None):
        # choose GPU if available
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load pretrained DETR ResNet‑50
        self.model = torch.hub.load(
            'facebookresearch/detr', 
            'detr_resnet50', 
            pretrained=True
        ).to(self.device).eval()

        # transforms: to Tensor + normalize as DETR expects
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_paths, threshold=0.5):
        """
        image_paths: list of file paths
        returns: list of dicts {boxes, scores, labels}
        """
        imgs = []
        orig_sizes = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            orig_sizes.append((img.height, img.width))
            imgs.append(self.transform(img).unsqueeze(0))
        batch = torch.cat(imgs, dim=0).to(self.device)

        outputs = self.model(batch)
        probs = outputs['pred_logits'].softmax(-1)[..., :-1]
        boxes = outputs['pred_boxes']

        results = []
        for i in range(batch.size(0)):
            scores, labels = probs[i].max(-1)
            keep = scores > threshold

            b = boxes[i][keep]
            l = labels[keep]
            s = scores[keep]

            b_xyxy = box_convert(b, in_fmt='cxcywh', out_fmt='xyxy')
            h, w = orig_sizes[i]
            scale = torch.tensor([w, h, w, h], device=self.device)
            b_xyxy = b_xyxy * scale

            results.append({
                "boxes": b_xyxy,
                "scores": s,
                "labels": l
            })

        # ← this must be here, **outside** the for‑loop
        return results



if __name__ == "__main__":
    # -------------------------
    # Smoke‑test the DETR wrapper
    # -------------------------
    this_dir    = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

    img_dir = os.path.join(project_root, "data", "raw", "images")
    sample = sorted(os.listdir(img_dir))[:2]
    sample_imgs = [os.path.join(img_dir, f) for f in sample]

    detr = DETRModel()
    dets = detr.predict(sample_imgs)

    # 1) number of images
    print("Number of images processed:", len(dets))
    # 2) number of detections in first image
    first = dets[0]
    print("Detections in first image:", first["boxes"].shape[0])
    # 3) example detections
    print("Example [x1, y1, x2, y2, score, label]:")
    for bbox, score, label in zip(
            first["boxes"][:3],
            first["scores"][:3],
            first["labels"][:3]
        ):
        coords = [round(x.item(), 2) for x in bbox]
        print(coords + [round(score.item(), 2), int(label.item())])
