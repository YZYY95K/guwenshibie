#!/usr/bin/env python3
import os
import sys
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


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
        elif backbone == 'convnext_small':
            self.backbone = models.convnext_small(weights=None)
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
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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
    with open(mapping_path, 'r', encoding='utf-8') as fp:
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


def parse_xml_chars(xml_path):
    chars = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for elem in root.iter():
        if elem.tag == 'char':
            text = elem.text.strip() if elem.text else ''
            if not text:
                continue
            pos = elem.attrib.get('position', '')
            if not pos:
                continue
            if ';' in pos:
                points = []
                for pair in pos.split(';'):
                    try:
                        coords = [int(x) for x in pair.split(',')]
                        if len(coords) >= 2:
                            points.append((coords[0], coords[1]))
                    except:
                        continue
                if len(points) < 3:
                    continue
                x1 = min(p[0] for p in points)
                y1 = min(p[1] for p in points)
                x2 = max(p[0] for p in points)
                y2 = max(p[1] for p in points)
            else:
                try:
                    coords = [int(x) for x in pos.split(',')]
                    if len(coords) != 4:
                        continue
                    x1, y1, x2, y2 = coords
                except:
                    continue
            if x2 <= x1 or y2 <= y1:
                continue
            w = x2 - x1
            h = y2 - y1
            chars.append({'bbox': [x1, y1, w, h], 'text': text})
    return chars


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate(predictions, ground_truth, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_det_tp = 0
    total_det_fp = 0
    total_det_fn = 0

    for image_id in ground_truth:
        gt_chars = ground_truth[image_id]
        pred_chars = predictions.get(image_id, [])

        matched_gt = set()
        matched_pred = set()
        det_tp = 0
        char_tp = 0

        for pi, pred in enumerate(pred_chars):
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_chars):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold:
                det_tp += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                if pred['text'] == gt_chars[best_gi]['text']:
                    char_tp += 1

        det_fp = len(pred_chars) - det_tp
        det_fn = len(gt_chars) - det_tp
        char_fp = det_tp - char_tp

        total_tp += char_tp
        total_fp += det_fp + char_fp
        total_fn += det_fn
        total_det_tp += det_tp
        total_det_fp += det_fp
        total_det_fn += det_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    det_precision = total_det_tp / (total_det_tp + total_det_fp) if (total_det_tp + total_det_fp) > 0 else 0
    det_recall = total_det_tp / (total_det_tp + total_det_fn) if (total_det_tp + total_det_fn) > 0 else 0
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0

    char_accuracy = total_tp / total_det_tp if total_det_tp > 0 else 0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'det_f1': det_f1,
        'det_precision': det_precision,
        'det_recall': det_recall,
        'char_accuracy': char_accuracy,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_det_tp': total_det_tp,
        'total_det_fp': total_det_fp,
        'total_det_fn': total_det_fn,
        'total_gt_chars': sum(len(v) for v in ground_truth.values()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_model', type=str, default='/root/project/models/detection/yolov8n_char_det_v2/weights/best.pt')
    parser.add_argument('--rec_model', type=str, default='/root/project/models/recognizer/exp037_swin_small_hybrid_2stage_ema/best.pth')
    parser.add_argument('--mapping', type=str, default='/root/project/data/processed/crops_full/char_to_id.json')
    parser.add_argument('--ood_dir', type=str, default='/root/project/data/raw/train/out_of_domain')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--conf_threshold', type=float, default=0.25)
    parser.add_argument('--max_images', type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f'Loading recognition model from {args.rec_model}...')
    rec_model, num_classes = load_recognition_model(args.rec_model, device)
    print(f'  Recognition model loaded: {num_classes} classes')

    id_to_char = load_char_mapping(args.mapping)
    print(f'  Char mapping loaded: {len(id_to_char)} chars')

    print(f'Loading detection model from {args.det_model}...')
    from ultralytics import YOLO
    det_model = YOLO(args.det_model)
    print(f'  Detection model loaded')

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    xml_files = sorted([f for f in os.listdir(args.ood_dir) if f.endswith('.xml')])
    if args.max_images > 0:
        xml_files = xml_files[:args.max_images]
    print(f'Evaluating on {len(xml_files)} OOD images')

    ground_truth = {}
    predictions = {}
    start_time = time.time()

    for i, xf in enumerate(xml_files):
        image_id = xf.rsplit('.', 1)[0]
        xml_path = os.path.join(args.ood_dir, xf)

        img_path = os.path.join(args.ood_dir, image_id + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(args.ood_dir, image_id + '.jpg')
        if not os.path.exists(img_path):
            continue

        gt_chars = parse_xml_chars(xml_path)
        if len(gt_chars) == 0:
            continue
        ground_truth[image_id] = gt_chars

        det_results = det_model(img_path, conf=args.conf_threshold, iou=0.45, verbose=False)
        boxes = []
        if len(det_results) > 0 and det_results[0].boxes is not None:
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))

        try:
            full_image = Image.open(img_path)
        except:
            continue

        char_results = []
        for x1, y1, x2, y2 in boxes:
            cx1 = max(0, x1)
            cy1 = max(0, y1)
            cx2 = min(full_image.width, x2)
            cy2 = min(full_image.height, y2)
            try:
                crop = full_image.crop((cx1, cy1, cx2, cy2))
                if crop.mode != 'RGB':
                    crop = crop.convert('RGB')
                img_tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = rec_model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = probs.max(dim=1)
                    pred_id = pred.item()
                    char = id_to_char.get(pred_id, '?')
                w = x2 - x1
                h = y2 - y1
                char_results.append({'bbox': [x1, y1, w, h], 'text': char})
            except:
                continue

        predictions[image_id] = char_results

        if (i + 1) % 50 == 0 or (i + 1) == len(xml_files):
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            print(f'  [{i+1}/{len(xml_files)}] Speed: {speed:.1f} img/s')

    results = evaluate(predictions, ground_truth)
    elapsed = time.time() - start_time

    print(f'\n{"="*60}')
    print(f'End-to-End OCR Evaluation Results')
    print(f'{"="*60}')
    print(f'Images evaluated: {len(ground_truth)}')
    print(f'Total GT chars:   {results["total_gt_chars"]}')
    print(f'Time:             {elapsed:.1f}s')
    print(f'')
    print(f'--- Detection ---')
    print(f'  Det Precision:  {results["det_precision"]:.4f}')
    print(f'  Det Recall:     {results["det_recall"]:.4f}')
    print(f'  Det F1:         {results["det_f1"]:.4f}')
    print(f'  Det TP:         {results["total_det_tp"]}')
    print(f'  Det FP:         {results["total_det_fp"]}')
    print(f'  Det FN:         {results["total_det_fn"]}')
    print(f'')
    print(f'--- Character Recognition ---')
    print(f'  Char Accuracy:  {results["char_accuracy"]:.4f}')
    print(f'  Char TP:        {results["total_tp"]}')
    print(f'  Char FP:        {results["total_fp"] - results["total_det_fp"]}')
    print(f'')
    print(f'--- Overall (Competition Metric) ---')
    print(f'  Precision:      {results["precision"]:.4f}')
    print(f'  Recall:         {results["recall"]:.4f}')
    print(f'  F1 Score:       {results["f1"]:.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
