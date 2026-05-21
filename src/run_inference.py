#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

SAISDATA_DIR = Path(os.environ.get('SAISDATA_DIR', '/saisdata'))
OUTPUT_FILE = Path(os.environ.get('OUTPUT_FILE', '/saisresult/prediction.json'))
DET_MODEL_PATH = Path(os.environ.get('DET_MODEL_PATH', '/app/models/det/best.pt'))
REC_MODEL_PATH = Path(os.environ.get('REC_MODEL_PATH', '/app/models/rec/best.pth'))
MAPPING_PATH = Path(os.environ.get('MAPPING_PATH', '/app/models/rec/char_to_id.json'))
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = int(os.environ.get('IMG_SIZE', '224'))
CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', '0.15'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '64'))


class Recognizer(nn.Module):
    def __init__(self, num_classes, backbone='swin_small', pretrained=False, dropout=0.5):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swin_tiny':
            self.backbone = models.swin_t(weights=None)
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'swin_small':
            self.backbone = models.swin_s(weights=None)
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'swin_base':
            self.backbone = models.swin_b(weights=None)
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'convnext_small':
            self.backbone = models.convnext_small(weights=None)
            feat_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(feat_dim, eps=1e-6),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'convnext_base':
            self.backbone = models.convnext_base(weights=None)
            feat_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(feat_dim, eps=1e-6),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


def load_recognition_model(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    backbone = config.get('backbone', 'swin_small')

    num_classes = 0
    for key in ckpt['model_state_dict']:
        if 'head.1' in key or 'classifier.3' in key:
            shape = ckpt['model_state_dict'][key].shape
            if len(shape) >= 1:
                num_classes = max(num_classes, shape[-1])

    if num_classes == 0:
        for key in ckpt['model_state_dict']:
            if 'weight' in key:
                shape = ckpt['model_state_dict'][key].shape
                if len(shape) == 2 and shape[0] > num_classes:
                    num_classes = shape[0]

    model = Recognizer(num_classes, backbone=backbone, pretrained=False, dropout=0.5)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, num_classes


def load_char_mapping(mapping_path):
    id_to_char = {}
    with open(str(mapping_path), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        if isinstance(data, dict):
            first_val = next(iter(data.values()), None)
            if isinstance(first_val, int):
                id_to_char = {v: k for k, v in data.items()}
            elif 'char_to_id' in data:
                id_to_char = {v: k for k, v in data['char_to_id'].items()}
            elif 'id_to_char' in data:
                id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
    return id_to_char


def find_input_dir():
    candidates = [
        SAISDATA_DIR / '13' / 'eval' / 'images',
        SAISDATA_DIR / 'eval' / 'images',
        SAISDATA_DIR / 'images',
        SAISDATA_DIR,
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            pngs = list(d.glob('*.png'))
            if len(pngs) > 0:
                print(f'Found input directory: {d} ({len(pngs)} images)')
                return d
    print(f'WARNING: No images found, using {SAISDATA_DIR}')
    return SAISDATA_DIR


def batch_recognize(crops, model, transform, device, id_to_char, batch_size=64):
    if len(crops) == 0:
        return []
    
    tensors = []
    for crop in crops:
        if crop.mode != 'RGB':
            crop = crop.convert('RGB')
        tensors.append(transform(crop))
    
    results = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            output = model(batch)
            probs = torch.softmax(output, dim=1)
            confs, preds = probs.max(dim=1)
            for j in range(len(preds)):
                pred_id = preds[j].item()
                char = id_to_char.get(pred_id, '?')
                results.append(char)
    
    return results


def main():
    device = torch.device(DEVICE)

    print(f'Loading recognition model from {REC_MODEL_PATH}...')
    rec_model, num_classes = load_recognition_model(REC_MODEL_PATH, device)
    print(f'  Recognition model loaded: {num_classes} classes')

    id_to_char = load_char_mapping(MAPPING_PATH)
    print(f'  Char mapping loaded: {len(id_to_char)} chars')

    det_model = None
    if DET_MODEL_PATH.exists():
        print(f'Loading detection model from {DET_MODEL_PATH}...')
        from ultralytics import YOLO
        det_model = YOLO(str(DET_MODEL_PATH))
        print(f'  Detection model loaded')
    else:
        print(f'ERROR: Detection model not found at {DET_MODEL_PATH}')
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_dir = find_input_dir()
    image_files = sorted(list(input_dir.glob('*.png')))
    print(f'Found {len(image_files)} images')

    predictions = {}
    total_chars = 0
    start_time = time.time()

    for i, img_path in enumerate(image_files):
        image_id = img_path.stem

        det_results = det_model(str(img_path), conf=CONF_THRESHOLD, iou=0.45, verbose=False)
        boxes = []
        if len(det_results) > 0 and det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

        if len(boxes) == 0:
            predictions[image_id] = []
            continue

        try:
            full_image = Image.open(str(img_path))
        except Exception as e:
            print(f'  Error opening {img_path.name}: {e}')
            predictions[image_id] = []
            continue

        crops = []
        for x1, y1, x2, y2 in boxes:
            cx1 = max(0, x1)
            cy1 = max(0, y1)
            cx2 = min(full_image.width, x2)
            cy2 = min(full_image.height, y2)
            try:
                crop = full_image.crop((cx1, cy1, cx2, cy2))
                crops.append(crop)
            except Exception:
                crops.append(None)

        chars = batch_recognize(
            [c for c in crops if c is not None],
            rec_model, transform, device, id_to_char, BATCH_SIZE
        )

        char_idx = 0
        char_results = []
        for j, (x1, y1, x2, y2) in enumerate(boxes):
            if crops[j] is not None and char_idx < len(chars):
                w = x2 - x1
                h = y2 - y1
                char_results.append({
                    'bbox': [x1, y1, w, h],
                    'text': chars[char_idx]
                })
                char_idx += 1

        predictions[image_id] = char_results
        total_chars += len(char_results)

        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (len(image_files) - i - 1) / speed if speed > 0 else 0
            print(f'  [{i+1}/{len(image_files)}] {img_path.name}: {len(char_results)} chars | '
                  f'Speed: {speed:.1f} img/s | ETA: {eta:.0f}s | Total: {total_chars} chars')

    elapsed = time.time() - start_time
    print(f'\nPipeline complete: {len(image_files)} images, {total_chars} chars in {elapsed:.1f}s')

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUTPUT_FILE), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f'Predictions saved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
