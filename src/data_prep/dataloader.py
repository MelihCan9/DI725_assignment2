# src/data_prep/dataloader.py
import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class AUAIRDataset(Dataset):
    def __init__(self, images_dir, ann_path, transforms=None):
        """
        images_dir: path to data/raw/images
        ann_path:   path to data/raw/annotations.json
        transforms: torchvision transforms to apply to each image
        """
        self.images_dir = images_dir
        self.transforms = transforms or T.Compose([T.ToTensor()])

        # Load annotation file
        with open(ann_path, 'r') as f:
            data = json.load(f)

        # Each entry in data['annotations'] corresponds to one image
        self.entries = data['annotations']

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        file_name = entry['image_name']
        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        # Extract all bounding boxes and labels for this image
        boxes = []
        labels = []
        for obj in entry['bbox']:
            top = obj['top']
            left = obj['left']
            width = obj['width']
            height = obj['height']
            # convert to [x_min, y_min, x_max, y_max]
            boxes.append([left, top, left + width, top + height])
            labels.append(obj['class'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

def get_dataloader(images_dir, ann_path, batch_size=4, shuffle=True, num_workers=4):
    dataset = AUAIRDataset(images_dir, ann_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

if __name__ == "__main__":
    import os

    # 1) Compute project root (two levels up from this file)
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))

    # 2) Build correct absolute paths
    images_dir = os.path.join(project_root, "data", "raw", "images")
    ann_path    = os.path.join(project_root, "data", "raw", "annotations.json")

    # 3) Instantiate loader
    loader = get_dataloader(
        images_dir=images_dir,
        ann_path=ann_path,
        batch_size=2
    )

    # 4) Run a single batch and print
    for images, targets in loader:
        print("Batch size:", len(images))
        print("First image tensor shape:", images[0].shape)
        print("Number of boxes in first sample:", targets[0]["boxes"].shape[0])
        break

