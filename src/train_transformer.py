# # src/train_transformer.py

# import os, json, torch
# from torch.utils.data import DataLoader
# from transformers import (
#     DetrConfig,
#     DetrForObjectDetection,
#     DetrImageProcessor,
#     get_scheduler
# )
# from torch.optim import AdamW
# from datasets import Dataset
# import evaluate

# def load_auair_dataset(images_dir, ann_file):
#     """
#     Load AU‑AIR JSON where each annotation entry corresponds to one image
#     and categories may be a list of strings.
#     """
#     import os, json
#     from datasets import Dataset

#     data = json.load(open(ann_file))
#     # Build category map, handling both dicts and plain strings:
#     cats = data.get('categories', [])
#     if cats and isinstance(cats[0], dict):
#         id2name = {c['id']: c['name'] for c in cats}
#     else:
#         # categories is list of names; index = class id
#         id2name = {i: name for i, name in enumerate(cats)}

#     records = []
#     for idx, entry in enumerate(data['annotations']):
#         # pick the correct field for file name
#         fname = entry.get('image_name', entry.get('file_name'))
#         img_path = os.path.join(images_dir, fname)

#         # some JSONs wrap all bboxes for a single image in entry['bbox']
#         bboxes = []
#         labels = []
#         for obj in entry['bbox']:
#             x, y, w, h = obj['left'], obj['top'], obj['width'], obj['height']
#             bboxes.append([x, y, w, h])
#             # obj['class'] holds the integer ID matching our id2name map
#             labels.append(obj.get('class', obj.get('category_id')))

#         records.append({
#             'image_id': idx,
#             'file_name': img_path,
#             'bboxes': bboxes,
#             'labels': labels
#         })

#     return Dataset.from_list(records), id2name


# def preprocess_fn(example, processor):
#     # processor normalizes and converts boxes → expected format
#     enc = processor(
#         images=example['file_name'],
#         annotations={
#             'image_id': example['image_id'],
#             'labels': example['labels'],
#             'boxes': example['bboxes']
#         },
#         return_tensors="pt"
#     )
#     example['pixel_values'] = enc.pixel_values.squeeze(0)
#     example['labels'] = enc.labels[0]
#     return example

# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
#         'labels': [b['labels'] for b in batch]
#     }

# def main():
#     # ---- Setup ----
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     images_dir = "data/raw/images"
#     ann_file   = "data/raw/annotations.json"

#     # ---- Load dataset + map ----
#     ds, id2name = load_auair_dataset(images_dir, ann_file)
#     num_labels = len(id2name)
#     print(f"Loaded {len(ds)} images with {num_labels} categories.")

#     # ---- Build random‑init DETR ----
#     config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
#     config.num_labels = num_labels
#     config.id2label   = { str(k): v for k,v in id2name.items() }
#     config.label2id   = { v: str(k) for k,v in id2name.items() }

#     model = DetrForObjectDetection(config).to(device)
#     processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

#     # ---- Preprocess dataset ----
#     ds = ds.map(lambda ex: preprocess_fn(ex, processor),
#                 remove_columns=ds.column_names)

#     # ---- DataLoader ----
#     loader = DataLoader(ds,
#                         batch_size=2,
#                         shuffle=True,
#                         collate_fn=collate_fn)

#     # ---- Optimizer & LR scheduler ----
#     optimizer = AdamW(model.parameters(), lr=1e-4)
#     epochs = 20
#     total_steps = len(loader) * epochs
#     scheduler = get_scheduler(
#         "linear", optimizer,
#         num_warmup_steps=100,
#         num_training_steps=total_steps
#     )

#     # ---- COCO metric ----
#     #metric = load_metric("coco")
#     metric = evaluate.load("coco")

#     # ---- Training loop ----
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for batch in loader:
#             inputs = {k:v.to(device) for k,v in batch.items()}
#             outputs = model(**inputs)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#             running_loss += loss.item()
#         print(f"[Epoch {epoch+1:02d}/{epochs}] Loss: {running_loss/len(loader):.4f}")

#     # ---- Evaluation loop ----
#     model.eval()
#     for batch in loader:
#         inputs = {k:v.to(device) for k,v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**inputs)
#         results = processor.post_process_object_detection(
#             outputs,
#             threshold=0.5,
#             target_sizes=[(int(x.shape[1]), int(x.shape[2]))
#                           for x in inputs['pixel_values']]
#         )
#         # accumulate for metric
#         for i, res in enumerate(results):
#             metric.add(
#                 predictions=[{
#                     'image_id': batch['labels'][i]['image_id'][0].item(),
#                     'boxes':   res['boxes'].cpu().tolist(),
#                     'scores':  res['scores'].cpu().tolist(),
#                     'labels':  res['labels'].cpu().tolist()
#                 }],
#                 references=[{
#                     'image_id': batch['labels'][i]['image_id'][0].item(),
#                     'boxes':   batch['labels'][i]['boxes'].cpu().tolist(),
#                     'labels':  batch['labels'][i]['labels'].cpu().tolist()
#                 }]
#             )
#     coco_res = metric.compute()
#     print("⏹  Final evaluation results:", coco_res)

# if __name__ == "__main__":
#     ds, id2name = load_auair_dataset("data/raw/images", "data/raw/annotations.json")
#     print("Total images:", len(ds))
#     print("Sample categories:", id2name)
#     print("Example record:", ds[0])



# src/train_transformer.py

import os, json, torch
from torch.utils.data import DataLoader
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    get_scheduler
)
from torch.optim import AdamW
from datasets import Dataset
import evaluate

def load_auair_dataset(images_dir, ann_file):
    """
    Load AU-AIR JSON where each annotation entry corresponds to one image.
    """
    data = json.load(open(ann_file))
    # Build category map (handle dict or plain string entries)
    cats = data.get('categories', [])
    if cats and isinstance(cats[0], dict):
        id2name = {c['id']: c['name'] for c in cats}
    else:
        id2name = {i: name for i, name in enumerate(cats)}

    records = []
    for idx, entry in enumerate(data['annotations']):
        # determine file name field
        fname = entry.get('image_name', entry.get('file_name'))
        img_path = os.path.join(images_dir, fname)

        bboxes, labels = [], []
        for obj in entry['bbox']:
            x, y, w, h = obj['left'], obj['top'], obj['width'], obj['height']
            bboxes.append([x, y, w, h])
            labels.append(obj.get('class', obj.get('category_id')))

        records.append({
            'image_id': idx,
            'file_name': img_path,
            'bboxes': bboxes,
            'labels': labels
        })

    return Dataset.from_list(records), id2name

from PIL import Image   # add this at the top of your file

from PIL import Image     # ensure this import is at the top

from PIL import Image    # ensure at top of file

def preprocess_fn(example, processor):
    # 1) Load image
    image = Image.open(example['file_name']).convert("RGB")

    # 2) Build COCO‐style annotation dict
    anns = []
    for ann_id, (bbox, label) in enumerate(zip(example['bboxes'], example['labels'])):
        x, y, w, h = bbox
        anns.append({
            "bbox": [x, y, w, h],           # x_min, y_min, width, height
            "category_id": int(label),      # class ID
            "area": float(w * h),           # required area field
            "iscrowd": 0,                   # assume no crowd annotations
            "id": ann_id                    # unique annotation ID
        })
    annotation = {
        "image_id": int(example['image_id']),
        "annotations": anns
    }

    # 3) Process
    enc = processor(
        images=image,
        annotations=annotation,
        return_tensors="pt"
    )

    # 4) Squeeze out batch dimension
    example['pixel_values'] = enc.pixel_values.squeeze(0)
    example['labels']       = enc.labels[0]
    return example




def collate_fn(batch):
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'labels': [b['labels'] for b in batch]
    }

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = "data/raw/images"
    ann_file   = "data/raw/annotations.json"

    # Load dataset
    ds, id2name = load_auair_dataset(images_dir, ann_file)
    num_labels = len(id2name)
    print(f"Loaded {len(ds)} images and {num_labels} categories.")

    # Build random-init DETR
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_labels
    config.id2label   = {str(k): v for k,v in id2name.items()}
    config.label2id   = {v: str(k) for k,v in id2name.items()}
    model = DetrForObjectDetection(config).to(device)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Preprocess dataset
    ds = ds.map(lambda ex: preprocess_fn(ex, processor), remove_columns=ds.column_names)

    # DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 20
    total_steps = len(loader) * num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    # COCO metric
    metric = evaluate.load("coco")

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in loader:
            inputs = {'pixel_values': batch['pixel_values'].to(device),
                      'labels': batch['labels']}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"[Train] Epoch {epoch+1}/{num_epochs} loss: {avg_loss:.4f}")

    # Evaluation loop
    model.eval()
    for batch in loader:
        inputs = {'pixel_values': batch['pixel_values'].to(device),
                  'labels': batch['labels']}
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=[(int(im.shape[1]), int(im.shape[2])) for im in batch['pixel_values']]
        )
        for i, res in enumerate(results):
            metric.add(
                predictions=[{
                    'image_id': batch['labels'][i]['image_id'][0].item(),
                    'boxes': res['boxes'].cpu().tolist(),
                    'scores': res['scores'].cpu().tolist(),
                    'labels': res['labels'].cpu().tolist()
                }],
                references=[{
                    'image_id': batch['labels'][i]['image_id'][0].item(),
                    'boxes': batch['labels'][i]['boxes'].cpu().tolist(),
                    'labels': batch['labels'][i]['labels'].cpu().tolist()
                }]
            )
    coco_res = metric.compute()
    print("[Eval] COCO results:", coco_res)

if __name__ == "__main__":
    main()
    # from transformers import DetrImageProcessor
    # ds, _ = load_auair_dataset("data/raw/images", "data/raw/annotations.json")
    # processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    # ex = ds[0]
    # ex2 = preprocess_fn(ex, processor)
    # print("pixel_values shape:", ex2['pixel_values'].shape)
    # print("labels:", ex2['labels'])
    # import sys; sys.exit(0)

