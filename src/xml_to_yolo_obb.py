#!/usr/bin/env python3
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import math
import cv2
import numpy as np

class OBBDataConverter:
    def __init__(self, raw_data_dir: Path, output_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.char_stats = {}

    def is_valid_unicode(self, char: str) -> bool:
        try:
            if not char or len(char) == 0:
                return False
            for c in char:
                unicodedata.name(c)
            return True
        except ValueError:
            return False

    def filter_dirty_data(self, char: str) -> bool:
        if not char:
            return False
        if char.startswith('ZH-') and '(' in char:
            return False
        if char == 'None' or char == 'none':
            return False
        if not self.is_valid_unicode(char):
            return False
        return True

    def parse_polygon(self, polygon_str: str) -> List[Tuple[float, float]]:
        points = polygon_str.split(';')
        result = []
        for point in points:
            x, y = map(float, point.split(','))
            result.append((x, y))
        return result

    def parse_rectangle(self, rect_str: str) -> List[Tuple[float, float]]:
        x1, y1, x2, y2 = map(float, rect_str.split(','))
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def get_min_area_rect(self, points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        pts = np.array(points, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        center, size, angle = rect
        return center, size, angle

    def rect_to_rotated_box(self, center: Tuple[float, float], size: Tuple[float, float], angle: float, width: int, height: int) -> Tuple[float, float, float, float, float]:
        x_c, y_c = center[0] / width, center[1] / height
        w, h = size[0] / width, size[1] / height
        angle_rad = math.radians(angle)
        return x_c, y_c, w, h, angle_rad

    def compute_perspective_transform(self, points: List[Tuple[float, float]]) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
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

    def warp_perspective(self, img: np.ndarray, M: np.ndarray, dst_size: Tuple[int, int]) -> np.ndarray:
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped

    def parse_xml(self, xml_path: Path) -> Optional[Dict]:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            page = root.find('page')
            if page is None:
                return None

            page_id = page.get('id', '')
            width = int(page.get('width', 0))
            height = int(page.get('height', 0))

            image_path = self.raw_data_dir / f"{page_id}.png"
            if not image_path.exists():
                return None

            annotations = []
            for char_elem in root.findall('.//char'):
                text = char_elem.get('text', '').strip()
                position = char_elem.get('position', '')

                if not self.filter_dirty_data(text):
                    continue

                if not position:
                    continue

                try:
                    if ';' in position:
                        points = self.parse_polygon(position)
                    else:
                        points = self.parse_rectangle(position)

                    if len(points) < 4:
                        continue

                    center, size, angle = self.get_min_area_rect(points)
                    x_c_norm, y_c_norm, w_norm, h_norm, angle_rad = self.rect_to_rotated_box(
                        center, size, angle, width, height
                    )

                    char_unicode = ord(text[0]) if len(text) == 1 else 0

                    annotations.append({
                        'text': text,
                        'char_unicode': char_unicode,
                        'points': points,
                        'center': center,
                        'size': size,
                        'angle': angle,
                        'obb': [x_c_norm, y_c_norm, w_norm, h_norm, angle_rad]
                    })

                    if text not in self.char_stats:
                        self.char_stats[text] = 0
                    self.char_stats[text] += 1

                except Exception as e:
                    continue

            return {
                'image_id': page_id,
                'width': width,
                'height': height,
                'image_path': str(image_path),
                'annotations': annotations
            }

        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None

    def convert_dataset(self, split: str = 'train'):
        xml_dir = self.raw_data_dir / 'out_of_domain'
        if not xml_dir.exists():
            print(f"XML directory not found: {xml_dir}")
            return

        xml_files = list(xml_dir.glob('*.xml'))
        print(f"Found {len(xml_files)} XML files for {split}")

        output_images_dir = self.images_dir / split
        output_labels_dir = self.labels_dir / split
        output_crops_dir = self.output_dir / "crops" / split
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        output_crops_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0
        total_chars = 0
        skipped_chars = 0

        for idx, xml_file in enumerate(xml_files):
            if idx % 500 == 0:
                print(f"Processing {idx}/{len(xml_files)}...")

            data = self.parse_xml(xml_file)
            if data is None or len(data['annotations']) == 0:
                skipped_chars += 1
                continue

            image_id = data['image_id']
            annotations = data['annotations']

            import shutil
            src_image = Path(data['image_path'])
            if src_image.exists():
                dst_image = output_images_dir / f"{image_id}.png"
                if not dst_image.exists():
                    shutil.copy(src_image, dst_image)

                img = cv2.imread(str(src_image))
                if img is not None:
                    for ann_idx, ann in enumerate(annotations):
                        M, dst_size = self.compute_perspective_transform(ann['points'])
                        warped = self.warp_perspective(img, M, dst_size)

                        crop_path = output_crops_dir / f"{image_id}_{ann_idx}.png"
                        cv2.imwrite(str(crop_path), warped)

            label_file = output_labels_dir / f"{image_id}.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                for ann in annotations:
                    char_unicode = ann['char_unicode']
                    obb = ann['obb']
                    f.write(f"{char_unicode} {obb[0]:.6f} {obb[1]:.6f} {obb[2]:.6f} {obb[3]:.6f} {obb[4]:.6f}\n")

            processed_count += 1
            total_chars += len(annotations)

        print(f"\n=== OBB Conversion Summary ===")
        print(f"Processed images: {processed_count}")
        print(f"Total characters: {total_chars}")
        print(f"Skipped (dirty/invalid): {skipped_chars}")
        print(f"Unique characters: {len(self.char_stats)}")

        chars_file = self.output_dir / f"{split}_char_stats.json"
        with open(chars_file, 'w', encoding='utf-8') as f:
            json.dump(self.char_stats, f, ensure_ascii=False, indent=2)
        print(f"Character stats saved to {chars_file}")

    def create_class_mapping(self):
        sorted_chars = sorted(self.char_stats.keys(), key=lambda x: self.char_stats[x], reverse=True)
        class_mapping = {char: idx for idx, char in enumerate(sorted_chars)}

        mapping_file = self.output_dir / "class_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)
        print(f"Class mapping saved to {mapping_file}")
        print(f"Total classes: {len(class_mapping)}")
        return class_mapping

def main():
    parser = argparse.ArgumentParser(description='Convert ancient character dataset to YOLO OBB format')
    parser.add_argument('--raw_dir', type=str, default='/root/project/data/raw',
                        help='Raw data directory containing XML annotations')
    parser.add_argument('--output_dir', type=str, default='/root/project/data/yolo_obbs_format',
                        help='Output directory for YOLO OBB format data')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split to process')

    args = parser.parse_args()

    converter = OBBDataConverter(
        raw_data_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir)
    )

    print(f"Starting OBB conversion...")
    print(f"Raw data: {args.raw_dir}")
    print(f"Output: {args.output_dir}")

    converter.convert_dataset(split=args.split)
    converter.create_class_mapping()

    print("Conversion complete!")

if __name__ == '__main__':
    main()