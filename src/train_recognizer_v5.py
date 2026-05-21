#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageChops
import random
import math
from pathlib import Path
from collections import Counter
import copy


class AncientTextAugmentationV2:
    def __init__(self, img_size, augment_level='heavy'):
        self.img_size = img_size
        self.augment_level = augment_level

    def _add_ink_bleed(self, img, severity=None):
        if severity is None:
            severity = random.uniform(0.3, 2.0)
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
        kernel_size = max(3, int(severity * 3) * 2 + 1)
        if random.random() < 0.5:
            result = maximum_filter(img_arr, size=kernel_size)
        else:
            result = minimum_filter(img_arr, size=kernel_size)
        alpha = random.uniform(0.3, 0.8)
        blended = (img_arr * (1 - alpha) + result * alpha).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def _add_paper_texture(self, img):
        w, h = img.size
        noise_type = random.choice(['gaussian', 'perlin_like', 'speckle'])
        if noise_type == 'gaussian':
            noise = np.random.randint(0, 40, (h, w, 3), dtype=np.uint8)
        elif noise_type == 'perlin_like':
            sh, sw = max(1, h // 4), max(1, w // 4)
            noise = np.random.randint(0, 20, (sh, sw, 3), dtype=np.uint8)
            noise = np.repeat(np.repeat(noise, 4, axis=0), 4, axis=1)
            noise = noise[:h, :w]
            if noise.shape[0] < h or noise.shape[1] < w:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:noise.shape[0], :noise.shape[1]] = noise
                noise = padded
        else:
            noise = np.zeros((h, w, 3), dtype=np.uint8)
            speckle_mask = np.random.random((h, w)) < 0.05
            noise[speckle_mask] = np.random.randint(100, 200, (speckle_mask.sum(), 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise, 'RGB')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_arr = np.array(img).astype(np.float32)
        noise_arr = np.array(noise_img).astype(np.float32)
        alpha = random.uniform(0.02, 0.15)
        blended = (img_arr * (1 - alpha) + noise_arr * alpha).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def _add_erosion_dilation(self, img):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        from scipy.ndimage import binary_erosion, binary_dilation
        gray = np.mean(img_arr, axis=-1)
        threshold = random.randint(80, 160)
        mask = gray < threshold
        iters = random.randint(1, 3)
        if random.random() < 0.5:
            mask = binary_erosion(mask, iterations=iters)
        else:
            mask = binary_dilation(mask, iterations=iters)
        result = img_arr.copy()
        result[mask] = np.clip(result[mask].astype(np.float32) * random.uniform(0.0, 0.3), 0, 255).astype(np.uint8)
        result[~mask] = np.clip(result[~mask].astype(np.float32) * random.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def _add_crack_noise(self, img):
        if random.random() > 0.4:
            return img
        draw = ImageDraw.Draw(img)
        w, h = img.size
        num_cracks = random.randint(1, 4)
        for _ in range(num_cracks):
            x = random.randint(0, w)
            y = random.randint(0, h)
            crack_color = random.choice([
                (180, 160, 140), (160, 140, 120), (200, 180, 160),
                (140, 120, 100), (120, 100, 80)
            ])
            length = random.randint(5, max(10, w // 2))
            for _ in range(length):
                dx = random.randint(-4, 4)
                dy = random.randint(-2, 2)
                draw.line([(x, y), (x + dx, y + dy)], fill=crack_color, width=random.randint(1, 2))
                x += dx
                y += dy
        return img

    def _add_stain(self, img):
        if random.random() > 0.5:
            return img
        draw = ImageDraw.Draw(img)
        w, h = img.size
        num_stains = random.randint(1, 4)
        for _ in range(num_stains):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rx = random.randint(3, max(8, w // 3))
            ry = random.randint(3, max(8, h // 3))
            stain_color = random.choice([
                (160, 140, 100), (140, 120, 80), (180, 160, 120),
                (120, 100, 70), (100, 80, 60), (90, 70, 50)
            ])
            alpha = random.randint(20, 100)
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=stain_color + (alpha,))
            img_rgba = img.convert('RGBA')
            img_rgba = Image.alpha_composite(img_rgba, overlay)
            img = img_rgba.convert('RGB')
        return img

    def _simulate_weathering(self, img):
        if random.random() > 0.4:
            return img
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.4, 0.95))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.6, 1.2))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return img

    def _random_perspective(self, img):
        if random.random() > 0.4:
            return img
        w, h = img.size
        distortion_scale = random.uniform(0.1, 0.35)
        half_w = w // 2
        half_h = h // 2
        topleft = [random.randint(0, int(distortion_scale * half_w)),
                    random.randint(0, int(distortion_scale * half_h))]
        topright = [w - random.randint(0, int(distortion_scale * half_w)),
                     random.randint(0, int(distortion_scale * half_h))]
        botright = [w - random.randint(0, int(distortion_scale * half_w)),
                     h - random.randint(0, int(distortion_scale * half_h))]
        botleft = [random.randint(0, int(distortion_scale * half_w)),
                    h - random.randint(0, int(distortion_scale * half_h))]
        coeffs = self._find_perspective_coefficients(
            [(0, 0), (w, 0), (w, h), (0, h)],
            [topleft, topright, botright, botleft]
        )
        return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(255, 255, 255))

    @staticmethod
    def _find_perspective_coefficients(pts_start, pts_end):
        matrix = []
        for p1, p2 in zip(pts_start, pts_end):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(pts_end).reshape(8)
        res = np.linalg.solve(A, B)
        return np.array(res).reshape(8)

    def _elastic_deformation(self, img):
        if random.random() > 0.4:
            return img
        w, h = img.size
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        from scipy.ndimage import map_coordinates, gaussian_filter
        alpha = random.uniform(20, 60)
        sigma = random.uniform(3, 6)
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        indices_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
        result = np.zeros_like(img_arr)
        for c in range(3):
            result[:, :, c] = map_coordinates(img_arr[:, :, c], [indices_y, indices_x], order=1, mode='reflect')
        return Image.fromarray(result)

    def _random_invert(self, img):
        if random.random() > 0.15:
            return img
        img_arr = np.array(img)
        inverted = 255 - img_arr
        return Image.fromarray(inverted)

    def _add_lighting_variation(self, img):
        if random.random() > 0.5:
            return img
        w, h = img.size
        img_arr = np.array(img).astype(np.float32)
        light_type = random.choice(['gradient', 'spot', 'shadow'])
        if light_type == 'gradient':
            direction = random.choice(['h', 'v', 'd'])
            if direction == 'h':
                gradient = np.linspace(0.7, 1.3, w).reshape(1, -1, 1)
            elif direction == 'v':
                gradient = np.linspace(0.7, 1.3, h).reshape(-1, 1, 1)
            else:
                gradient = np.linspace(0.7, 1.3, max(w, h))[:w].reshape(1, -1, 1)
                gradient = gradient * np.linspace(0.8, 1.2, h).reshape(-1, 1, 1)
            img_arr = (img_arr * gradient).clip(0, 255)
        elif light_type == 'spot':
            cx, cy = random.randint(0, w), random.randint(0, h)
            radius = random.randint(min(w, h) // 3, max(w, h))
            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            brightness = 1.0 + 0.3 * np.exp(-dist ** 2 / (2 * radius ** 2))
            img_arr = (img_arr * brightness[:, :, np.newaxis]).clip(0, 255)
        else:
            sx, sy = random.randint(0, w), random.randint(0, h)
            shadow_alpha = random.uniform(0.5, 0.8)
            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((x_coords - sx) ** 2 + (y_coords - sy) ** 2)
            shadow = 1.0 - (1 - shadow_alpha) * np.exp(-dist ** 2 / (2 * (max(w, h) * 0.8) ** 2))
            img_arr = (img_arr * shadow[:, :, np.newaxis]).clip(0, 255)
        return Image.fromarray(img_arr.astype(np.uint8))

    def _random_cutout(self, img):
        if random.random() > 0.3:
            return img
        w, h = img.size
        num_cuts = random.randint(1, 3)
        img_arr = np.array(img)
        for _ in range(num_cuts):
            cut_w = random.randint(max(1, w // 10), max(2, w // 5))
            cut_h = random.randint(max(1, h // 10), max(2, h // 5))
            cx = random.randint(0, w - cut_w)
            cy = random.randint(0, h - cut_h)
            fill_val = random.choice([0, 128, 200, 255])
            img_arr[cy:cy + cut_h, cx:cx + cut_w] = fill_val
        return Image.fromarray(img_arr)

    def _add_gaussian_noise(self, img):
        if random.random() > 0.4:
            return img
        img_arr = np.array(img).astype(np.float32)
        sigma = random.uniform(5, 25)
        noise = np.random.normal(0, sigma, img_arr.shape)
        img_arr = (img_arr + noise).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    def _simulate_low_quality(self, img):
        if random.random() > 0.3:
            return img
        scale = random.uniform(0.3, 0.7)
        w, h = img.size
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((small_w, small_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img

    def _stroke_width_variation(self, img):
        if random.random() > 0.3:
            return img
        from scipy.ndimage import maximum_filter, minimum_filter
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        gray = np.mean(img_arr, axis=-1)
        threshold = 128
        mask = gray < threshold
        if random.random() < 0.5:
            dilated = maximum_filter(img_arr, size=3)
            result = np.where(mask[:, :, np.newaxis], dilated, img_arr)
        else:
            eroded = minimum_filter(img_arr, size=3)
            result = np.where(mask[:, :, np.newaxis], eroded, img_arr)
        return Image.fromarray(result.astype(np.uint8))

    def _morphological_close(self, img):
        if random.random() > 0.3:
            return img
        from scipy.ndimage import binary_closing, binary_opening
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        gray = np.mean(img_arr, axis=-1)
        threshold = random.randint(80, 160)
        mask = gray < threshold
        if random.random() < 0.5:
            mask = binary_closing(mask, iterations=random.randint(1, 2))
        else:
            mask = binary_opening(mask, iterations=random.randint(1, 2))
        result = img_arr.copy()
        result[mask] = np.clip(result[mask].astype(np.float32) * 0.2, 0, 255).astype(np.uint8)
        result[~mask] = np.clip(result[~mask].astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def __call__(self, img):
        if self.augment_level == 'heavy':
            img = self._add_paper_texture(img)
            if random.random() < 0.5:
                img = self._add_ink_bleed(img)
            if random.random() < 0.4:
                img = self._add_erosion_dilation(img)
            if random.random() < 0.4:
                img = self._morphological_close(img)
            if random.random() < 0.4:
                img = self._stroke_width_variation(img)
            img = self._add_stain(img)
            img = self._add_crack_noise(img)
            img = self._simulate_weathering(img)
            img = self._random_perspective(img)
            img = self._elastic_deformation(img)
            img = self._add_lighting_variation(img)
            img = self._random_cutout(img)
            img = self._add_gaussian_noise(img)
            img = self._random_invert(img)
            if random.random() < 0.3:
                img = self._simulate_low_quality(img)
            if random.random() < 0.3:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(random.uniform(0.3, 2.5))
        elif self.augment_level == 'medium':
            img = self._add_paper_texture(img)
            if random.random() < 0.3:
                img = self._add_ink_bleed(img)
            if random.random() < 0.2:
                img = self._add_erosion_dilation(img)
            img = self._add_stain(img)
            img = self._add_crack_noise(img)
            img = self._simulate_weathering(img)
            img = self._random_perspective(img)
            if random.random() < 0.3:
                img = self._elastic_deformation(img)
            if random.random() < 0.3:
                img = self._add_lighting_variation(img)
            if random.random() < 0.2:
                img = self._random_cutout(img)
        return img


class CharDataset(Dataset):
    def __init__(self, samples, transform=None, ancient_aug=None):
        self.samples = samples
        self.transform = transform
        self.ancient_aug = ancient_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (64, 64), (255, 255, 255))
        if self.ancient_aug:
            img = self.ancient_aug(img)
        if self.transform:
            img = self.transform(img)
        return img, label


class HybridSampler:
    def __init__(self, labels, num_classes, alpha=0.5):
        self.labels = np.array(labels)
        self.num_classes = num_classes
        self.alpha = alpha
        class_counts = np.bincount(self.labels, minlength=num_classes).astype(float)
        cb_weights = 1.0 / (class_counts + 1)
        cb_weights = cb_weights / cb_weights.sum()
        uniform_weights = np.ones(num_classes) / num_classes
        per_class_weights = alpha * cb_weights + (1 - alpha) * uniform_weights
        self.sample_weights = per_class_weights[self.labels]
        self.sample_weights = self.sample_weights / self.sample_weights.sum() * len(self.labels)

    def get_sampler(self):
        return torch.utils.data.WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.labels),
            replacement=True
        )


class Recognizer(nn.Module):
    def __init__(self, num_classes, backbone='swin_small', pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swin_tiny':
            self.backbone = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'swin_small':
            self.backbone = models.swin_s(weights=models.Swin_S_Weights.DEFAULT if pretrained else None)
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'swin_base':
            self.backbone = models.swin_b(weights=models.Swin_B_Weights.DEFAULT if pretrained else None)
            feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'convnext_small':
            self.backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
            feat_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.LayerNorm(feat_dim, eps=1e-6),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes)
            )
        elif backbone == 'convnext_base':
            self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
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


def get_train_transform(img_size, augment_level='heavy'):
    base_transforms = [
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop((img_size, img_size)),
    ]
    if augment_level in ('heavy', 'medium'):
        base_transforms.extend([
            transforms.RandomAffine(
                degrees=20, translate=(0.15, 0.15),
                scale=(0.8, 1.2), shear=15,
                fill=(255, 255, 255)
            ),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.08),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5)),
        ])
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])
    return transforms.Compose(base_transforms)


def get_val_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing,
                                  weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, label_smoothing=0.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list + 1))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        index = torch.zeros_like(inputs, dtype=torch.uint8)
        index.scatter_(1, targets.view(-1, 1), 1)
        batch_m = self.m_list.to(inputs.device)[targets]
        x_m = inputs - batch_m.unsqueeze(1) * index.float()
        output = torch.where(index, x_m, inputs)
        return F.cross_entropy(self.s * output, targets, label_smoothing=self.label_smoothing)


class CutMixCollator:
    def __init__(self, num_classes, alpha=1.0, prob=0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        if random.random() > self.prob:
            return images, labels
        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size)
        shuffled_labels = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        shuffled_onehot = F.one_hot(shuffled_labels, self.num_classes).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * shuffled_onehot
        return images, mixed_labels

    @staticmethod
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2


def evaluate_per_frequency(model, val_loader, device, label_counts, num_classes):
    model.eval()
    freq_buckets = {'high(>100)': [], 'medium(10-100)': [], 'low(3-10)': [], 'rare(1-2)': []}
    for label, count in label_counts.items():
        if count > 100:
            freq_buckets['high(>100)'].append(label)
        elif count >= 10:
            freq_buckets['medium(10-100)'].append(label)
        elif count >= 3:
            freq_buckets['low(3-10)'].append(label)
        else:
            freq_buckets['rare(1-2)'].append(label)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = (all_preds == all_labels).mean()
    freq_accs = {}
    for bucket_name, labels_in_bucket in freq_buckets.items():
        mask = np.isin(all_labels, labels_in_bucket)
        if mask.sum() > 0:
            bucket_acc = (all_preds[mask] == all_labels[mask]).mean()
            freq_accs[bucket_name] = bucket_acc
    return overall_acc, freq_accs


def train_recognizer(samples, char_to_id, exp_id, config, device):
    num_classes = len(char_to_id)
    img_size = config.get('img_size', 224)
    epochs = config.get('epochs', 200)
    batch_size = config.get('batch_size', 32)
    lr = config.get('lr', 3e-4)
    backbone = config.get('backbone', 'swin_small')
    dropout = config.get('dropout', 0.5)
    label_smoothing = config.get('label_smoothing', 0.15)
    augment_level = config.get('augment_level', 'heavy')
    pretrained = config.get('pretrained', True)
    weight_decay = config.get('weight_decay', 1e-4)
    scheduler_type = config.get('scheduler', 'cosine_warmup')
    warmup_epochs = config.get('warmup_epochs', 10)
    loss_type = config.get('loss_type', 'focal')
    focal_gamma = config.get('focal_gamma', 2.0)
    sampler_type = config.get('sampler', 'hybrid')
    sampler_alpha = config.get('sampler_alpha', 0.5)
    resume = config.get('resume', False)
    two_stage = config.get('two_stage', False)
    stage2_epoch = config.get('stage2_epoch', epochs // 2)
    use_ema = config.get('ema', False)
    ema_decay = config.get('ema_decay', 0.999)
    use_cutmix = config.get('cutmix', False)
    cutmix_alpha = config.get('cutmix_alpha', 1.0)
    cutmix_prob = config.get('cutmix_prob', 0.5)

    print(f"\n=== Training Recognizer V5: {exp_id} ===")
    print(f"  Num classes: {num_classes}")
    print(f"  Num samples: {len(samples)}")
    print(f"  Config: {config}")
    print(f"  Device: {device}")

    save_dir = f'/root/project/models/recognizer/{exp_id}'
    os.makedirs(save_dir, exist_ok=True)

    labels = [s[1] for s in samples]
    label_counts = Counter(labels)

    np.random.seed(42)
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    split_idx = int(len(samples) * 0.85)
    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]

    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    ancient_aug = AncientTextAugmentationV2(img_size, augment_level) if augment_level in ('heavy', 'medium') else None
    train_transform = get_train_transform(img_size, augment_level)
    val_transform = get_val_transform(img_size)

    train_dataset = CharDataset(train_samples, transform=train_transform, ancient_aug=ancient_aug)
    val_dataset = CharDataset(val_samples, transform=val_transform)

    train_labels = [s[1] for s in train_samples]

    if sampler_type == 'hybrid':
        hybrid_sampler = HybridSampler(train_labels, num_classes, alpha=sampler_alpha)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=hybrid_sampler.get_sampler(), num_workers=4,
                                  pin_memory=True, drop_last=True)
        print(f"  Using Hybrid Sampler (alpha={sampler_alpha})")
    elif sampler_type == 'class_balanced':
        hybrid_sampler = HybridSampler(train_labels, num_classes, alpha=1.0)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=hybrid_sampler.get_sampler(), num_workers=4,
                                  pin_memory=True, drop_last=True)
        print("  Using Class-Balanced Sampler")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        print("  Using Random Sampler")

    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    cutmix_collator = None
    if use_cutmix:
        cutmix_collator = CutMixCollator(num_classes, alpha=cutmix_alpha, prob=cutmix_prob)
        print(f"  Using CutMix (alpha={cutmix_alpha}, prob={cutmix_prob})")

    model = Recognizer(num_classes, backbone=backbone, pretrained=pretrained, dropout=dropout).to(device)

    start_epoch = 0
    best_val_acc = 0.0
    best_rare_acc = 0.0
    metrics_history = []

    if resume:
        ckpt_path = os.path.join(save_dir, 'best.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0)
            best_rare_acc = checkpoint.get('rare_acc', 0)
            print(f"  Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    cls_num_list = np.zeros(num_classes, dtype=np.float64)
    for label, count in label_counts.items():
        cls_num_list[label] = count

    if loss_type == 'focal':
        criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        print(f"  Using Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    elif loss_type == 'ldam':
        criterion = LDAMLoss(cls_num_list, max_m=0.5, s=30, label_smoothing=label_smoothing)
        print(f"  Using LDAM Loss (s=30, max_m=0.5)")
    elif loss_type == 'focal_cb':
        cb_weights = (cls_num_list.sum() / (cls_num_list + 1))
        cb_weights = cb_weights / cb_weights.sum() * num_classes
        cb_weights = torch.tensor(cb_weights, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing, class_weights=cb_weights)
        print(f"  Using Focal Loss with CB weights (gamma={focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"  Using CE Loss (label_smoothing={label_smoothing})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'cosine_warmup':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'cosine_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"  Using EMA (decay={ema_decay})")

    val_label_counts = Counter(s[1] for s in val_samples)

    for epoch in range(start_epoch, epochs):
        if two_stage and epoch == stage2_epoch:
            print(f"\n  === STAGE 2: Switching to Class-Balanced sampler ===")
            cb_sampler = HybridSampler(train_labels, num_classes, alpha=0.8)
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      sampler=cb_sampler.get_sampler(), num_workers=4,
                                      pin_memory=True, drop_last=True)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * 0.1
            print(f"  LR reduced to {lr * 0.1}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels_batch) in enumerate(train_loader):
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            if cutmix_collator and random.random() < cutmix_prob:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                rand_index = torch.randperm(images.size(0), device=device)
                bbx1, bby1, bbx2, bby2 = CutMixCollator._rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
                optimizer.zero_grad()
                outputs = model(images)
                loss = lam * criterion(outputs, labels_batch) + (1 - lam) * criterion(outputs, labels_batch[rand_index])
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if ema_model:
                with torch.no_grad():
                    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1 - ema_decay)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            if (batch_idx + 1) % 500 == 0:
                print(f"  Epoch {epoch+1}/{epochs} Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        scheduler.step()

        train_acc = correct / max(1, total) if total > 0 else 0
        train_loss = running_loss / max(1, len(train_loader.dataset))

        eval_model = ema_model if ema_model else model
        val_acc, freq_accs = evaluate_per_frequency(eval_model, val_loader, device, val_label_counts, num_classes)

        current_lr = optimizer.param_groups[0]['lr']
        freq_str = ' | '.join([f'{k}={v:.4f}' for k, v in freq_accs.items()])
        print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f}, train_loss={train_loss:.4f}, lr={current_lr:.8f} | {freq_str}")

        metrics_history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'lr': current_lr,
            'freq_accs': {k: float(v) for k, v in freq_accs.items()},
        })

        rare_acc = freq_accs.get('rare(1-2)', 0)
        combined_score = val_acc * 0.7 + rare_acc * 0.3

        if combined_score > best_val_acc * 0.7 + best_rare_acc * 0.3:
            best_val_acc = val_acc
            best_rare_acc = rare_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': (ema_model if ema_model else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'rare_acc': rare_acc,
                'freq_accs': {k: float(v) for k, v in freq_accs.items()},
                'config': config,
            }, os.path.join(save_dir, 'best.pth'))
            print(f"  *** New best: val_acc={val_acc:.4f}, rare_acc={rare_acc:.4f} ***")

        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': (ema_model if ema_model else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'rare_acc': rare_acc,
                'freq_accs': {k: float(v) for k, v in freq_accs.items()},
                'config': config,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth'))

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train character recognizer v5')
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--crop_dir', type=str, default='/root/project/data/processed/crops_full')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--backbone', type=str, default='swin_small',
                        choices=['swin_tiny', 'swin_small', 'swin_base',
                                 'convnext_small', 'convnext_base'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--label_smoothing', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--augment_level', type=str, default='heavy',
                        choices=['none', 'medium', 'heavy'])
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['cosine', 'cosine_warmup', 'cosine_restart'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'focal', 'ldam', 'focal_cb'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--sampler', type=str, default='hybrid',
                        choices=['random', 'class_balanced', 'hybrid'])
    parser.add_argument('--sampler_alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--stage2_epoch', type=int, default=150)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--cutmix_prob', type=float, default=0.5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    config = {
        'backbone': args.backbone,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'img_size': args.img_size,
        'label_smoothing': args.label_smoothing,
        'dropout': args.dropout,
        'augment_level': args.augment_level,
        'pretrained': True,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup_epochs,
        'loss_type': args.loss_type,
        'focal_gamma': args.focal_gamma,
        'sampler': args.sampler,
        'sampler_alpha': args.sampler_alpha,
        'two_stage': args.two_stage,
        'stage2_epoch': args.stage2_epoch,
        'ema': args.ema,
        'ema_decay': args.ema_decay,
        'cutmix': args.cutmix,
        'cutmix_alpha': args.cutmix_alpha,
        'cutmix_prob': args.cutmix_prob,
        'resume': args.resume,
    }

    mapping_path = os.path.join(args.crop_dir, 'char_to_id.json')
    samples_path = os.path.join(args.crop_dir, 'samples.json')

    if os.path.exists(mapping_path) and os.path.exists(samples_path):
        print("Loading existing crops...")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            char_to_id = json.load(f)
        with open(samples_path, 'r') as f:
            samples_raw = json.load(f)
        samples = [(s[0], s[1]) for s in samples_raw]
        print(f"Loaded {len(samples)} samples, {len(char_to_id)} classes")
    else:
        print(f"ERROR: No pre-cropped data found at {args.crop_dir}")
        sys.exit(1)

    best_acc = train_recognizer(samples, char_to_id, args.exp_id, config, args.device)
    print(f"\nTraining complete! Best val_acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()
