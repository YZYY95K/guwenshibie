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
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random
import math
from pathlib import Path
from collections import Counter

try:
    from timm.data.mixup import Mixup
    from timm.models import create_model as timm_create_model
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class AncientTextAugmentation:
    def __init__(self, img_size, augment_level='natural'):
        self.img_size = img_size
        self.augment_level = augment_level

    def _add_ink_bleed(self, img, severity=None):
        if severity is None:
            severity = random.uniform(0.3, 1.5)
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        kernel_size = int(severity * 3) * 2 + 1
        from scipy.ndimage import maximum_filter, minimum_filter
        if random.random() < 0.5:
            img_arr = maximum_filter(img_arr, size=kernel_size)
        else:
            img_arr = minimum_filter(img_arr, size=kernel_size)
        return Image.fromarray(img_arr.astype(np.uint8))

    def _add_paper_texture(self, img):
        w, h = img.size
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise, 'RGB')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_arr = np.array(img).astype(np.float32)
        noise_arr = np.array(noise_img).astype(np.float32)
        alpha = random.uniform(0.02, 0.1)
        blended = (img_arr * (1 - alpha) + noise_arr * alpha).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)

    def _add_erosion_dilation(self, img):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        from scipy.ndimage import binary_erosion, binary_dilation
        gray = np.mean(img_arr, axis=-1)
        mask = gray < 128
        if random.random() < 0.5:
            mask = binary_erosion(mask, iterations=random.randint(1, 2))
        else:
            mask = binary_dilation(mask, iterations=random.randint(1, 2))
        result = img_arr.copy()
        result[mask] = [0, 0, 0]
        result[~mask] = [255, 255, 255]
        return Image.fromarray(result)

    def _add_crack_noise(self, img):
        if random.random() > 0.3:
            return img
        draw = ImageDraw.Draw(img)
        w, h = img.size
        num_cracks = random.randint(1, 3)
        for _ in range(num_cracks):
            x = random.randint(0, w)
            y = random.randint(0, h)
            crack_color = random.choice([(180, 160, 140), (160, 140, 120), (200, 180, 160)])
            for _ in range(random.randint(3, 10)):
                dx = random.randint(-5, 5)
                dy = random.randint(-3, 3)
                draw.line([(x, y), (x+dx, y+dy)], fill=crack_color, width=random.randint(1, 2))
                x += dx
                y += dy
        return img

    def _add_stain(self, img):
        if random.random() > 0.4:
            return img
        draw = ImageDraw.Draw(img)
        w, h = img.size
        num_stains = random.randint(1, 3)
        for _ in range(num_stains):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            rx = random.randint(5, max(10, w // 4))
            ry = random.randint(5, max(10, h // 4))
            stain_color = random.choice([
                (160, 140, 100), (140, 120, 80), (180, 160, 120),
                (120, 100, 70), (100, 80, 60)
            ])
            alpha = random.randint(30, 80)
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=stain_color + (alpha,))
            img_rgba = img.convert('RGBA')
            img_rgba = Image.alpha_composite(img_rgba, overlay)
            img = img_rgba.convert('RGB')
        return img

    def _simulate_weathering(self, img):
        if random.random() > 0.3:
            return img
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.5, 0.9))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.1))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return img

    def _random_perspective(self, img):
        if random.random() > 0.3:
            return img
        w, h = img.size
        distortion_scale = random.uniform(0.1, 0.3)
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
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(pts_end).reshape(8)
        res = np.linalg.solve(A, B)
        return np.array(res).reshape(8)

    def __call__(self, img):
        if self.augment_level == 'natural':
            img = self._add_paper_texture(img)
            if random.random() < 0.3:
                img = self._add_ink_bleed(img)
            if random.random() < 0.2:
                img = self._add_erosion_dilation(img)
            img = self._add_stain(img)
            img = self._add_crack_noise(img)
            img = self._simulate_weathering(img)
            img = self._random_perspective(img)
        elif self.augment_level == 'extreme_natural':
            img = self._add_paper_texture(img)
            img = self._add_ink_bleed(img)
            if random.random() < 0.4:
                img = self._add_erosion_dilation(img)
            img = self._add_stain(img)
            img = self._add_crack_noise(img)
            img = self._simulate_weathering(img)
            img = self._random_perspective(img)
            if random.random() < 0.3:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(random.uniform(0.3, 2.0))
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


def get_train_transform(img_size, augment_level='natural'):
    base_transforms = [
        transforms.Resize((img_size, img_size)),
    ]
    if augment_level in ('natural', 'extreme_natural'):
        base_transforms.extend([
            transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    elif augment_level == 'extreme':
        base_transforms.extend([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
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
    augment_level = config.get('augment_level', 'natural')
    pretrained = config.get('pretrained', True)
    weight_decay = config.get('weight_decay', 1e-4)
    scheduler_type = config.get('scheduler', 'cosine_warmup')
    warmup_epochs = config.get('warmup_epochs', 10)
    use_mixup = config.get('mixup', False)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    loss_type = config.get('loss_type', 'focal')
    focal_gamma = config.get('focal_gamma', 2.0)
    sampler_type = config.get('sampler', 'hybrid')
    sampler_alpha = config.get('sampler_alpha', 0.5)
    resume = config.get('resume', False)
    two_stage = config.get('two_stage', False)
    stage2_epoch = config.get('stage2_epoch', epochs // 2)
    use_ema = config.get('ema', False)
    ema_decay = config.get('ema_decay', 0.999)

    print(f"\n=== Training Recognizer V4: {exp_id} ===")
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

    ancient_aug = AncientTextAugmentation(img_size, augment_level) if augment_level in ('natural', 'extreme_natural') else None
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

    mixup_fn = None
    if use_mixup and HAS_TIMM:
        mixup_fn = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=0.0, prob=0.5,
                         switch_prob=0.0, mode='batch', label_smoothing=label_smoothing, num_classes=num_classes)
        print(f"  Using Mixup (alpha={mixup_alpha})")

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

            if mixup_fn:
                images, labels_batch = mixup_fn(images, labels_batch)

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
            if not use_mixup:
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


import copy


def main():
    parser = argparse.ArgumentParser(description='Train character recognizer v4')
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
    parser.add_argument('--augment_level', type=str, default='natural',
                        choices=['none', 'medium', 'extreme', 'natural', 'extreme_natural'])
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['cosine', 'cosine_warmup', 'cosine_restart'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
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
        'mixup': args.mixup,
        'mixup_alpha': args.mixup_alpha,
        'loss_type': args.loss_type,
        'focal_gamma': args.focal_gamma,
        'sampler': args.sampler,
        'sampler_alpha': args.sampler_alpha,
        'two_stage': args.two_stage,
        'stage2_epoch': args.stage2_epoch,
        'ema': args.ema,
        'ema_decay': args.ema_decay,
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
