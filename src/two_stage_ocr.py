#!/usr/bin/env python3
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional
import argparse
import math

from models.encoder_decoder_recognizer import AncientCharEncoderDecoder

class TwoStageAncientOCR:
    def __init__(
        self,
        detector_path: str,
        recognizer_path: str = None,
        num_classes: int = 15000,
        device: str = 'cuda'
    ):
        self.device = device

        from ultralytics import YOLO
        print(f"Loading YOLOv8-OBB detector from {detector_path}...")
        self.detector = YOLO(detector_path)

        print(f"Loading Encoder-Decoder recognizer...")
        self.recognizer = AncientCharEncoderDecoder(
            num_classes=num_classes
        ).to(device)

        if recognizer_path and Path(recognizer_path).exists():
            checkpoint = torch.load(recognizer_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.recognizer.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.recognizer.load_state_dict(checkpoint)
            print(f"Loaded recognizer from {recognizer_path}")

        self.recognizer.eval()

    def compute_perspective_transform(self, points: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        points = points.astype(np.float32)

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.sort(box, axis=0)

        top_left = box[0] if box[0][1] <= box[1][1] else box[1]
        top_right = box[1] if box[0][1] <= box[1][1] else box[0]
        bottom_right = box[2] if box[2][1] >= box[3][1] else box[3]
        bottom_left = box[3] if box[2][1] >= box[3][1] else box[2]

        ordered_box = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

        width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
        height = int(max(np.linalg.norm(top_left - bottom_left), np.linalg.norm(top_right - bottom_right)))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_box, dst)
        return M, (width, height)

    def warp_and_crop(self, img: np.ndarray, points: np.ndarray, margin: int = 5) -> np.ndarray:
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(img.shape[1], x_max + margin)
        y_max = min(img.shape[0], y_max + margin)

        cropped = img[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return None

        shifted_points = points.copy()
        shifted_points[:, 0] -= x_min
        shifted_points[:, 1] -= y_min

        M, dst_size = self.compute_perspective_transform(shifted_points)
        warped = cv2.warpPerspective(cropped, M, dst_size)

        return warped

    def detect_with_obb(self, image_path: Path) -> List[Dict]:
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        results = self.detector(img, verbose=False)

        detections = []
        for result in results:
            if result.obb is None:
                continue

            boxes = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            clss = result.obb.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                points = box.reshape(-1, 2)

                x_coords = points[:, 0]
                y_coords = points[:, 1]
                x, y = x_coords.min(), y_coords.min()
                w = x_coords.max() - x_coords.min()
                h = y_coords.max() - y_coords.min()

                detections.append({
                    'points': points,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': float(conf),
                    'class': int(cls)
                })

        return detections

    def recognize_char(self, char_img: np.ndarray) -> Tuple[str, float]:
        if char_img is None or char_img.size == 0:
            return '', 0.0

        if char_img.shape[0] < 10 or char_img.shape[1] < 10:
            return '', 0.0

        char_pil = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)).convert('RGB')
        char_pil = char_pil.resize((64, 64), Image.BILINEAR)
        char_array = np.array(char_pil).astype(np.float32) / 255.0
        char_tensor = torch.from_numpy(char_array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.recognizer(char_tensor)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred = probs.max(dim=-1)

        char_code = pred.item()
        conf = confidence.item()

        try:
            char_text = chr(char_code)
        except:
            char_text = f"U+{char_code:04X}"

        return char_text, conf

    def process_image(self, image_path: Path) -> List[Dict]:
        detections = self.detect_with_obb(image_path)

        if not detections:
            return []

        img = cv2.imread(str(image_path))
        if img is None:
            return []

        results = []
        for det in detections:
            points = det['points']

            warped = self.warp_and_crop(img, points)

            char_text, conf = self.recognize_char(warped)

            results.append({
                'bbox': det['bbox'],
                'text': char_text,
                'confidence': conf
            })

        return results

    def process_directory(self, input_dir: Path, output_file: Path):
        image_files = sorted(input_dir.glob('*.png'))
        print(f"Found {len(image_files)} images in {input_dir}")

        all_results = {}

        for idx, img_path in enumerate(image_files):
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(image_files)}...")

            image_id = img_path.stem
            results = self.process_image(img_path)

            formatted_results = []
            for r in results:
                formatted_results.append({
                    'bbox': r['bbox'],
                    'text': r.get('text', '')
                })

            all_results[image_id] = formatted_results

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {output_file}")
        print(f"Total images processed: {len(all_results)}")

def main():
    parser = argparse.ArgumentParser(description='Two-Stage Ancient Character OCR')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing evaluation images')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--detector', type=str,
                        default='/root/project/models/detector/yolo_ancient_obbs/weights/best.pt',
                        help='Path to YOLO-OBB detector model')
    parser.add_argument('--recognizer', type=str, default=None,
                        help='Path to recognizer model')
    parser.add_argument('--num_classes', type=int, default=15000,
                        help='Number of character classes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    ocr = TwoStageAncientOCR(
        detector_path=args.detector,
        recognizer_path=args.recognizer,
        num_classes=args.num_classes,
        device=args.device
    )

    ocr.process_directory(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output_file)
    )

if __name__ == '__main__':
    main()