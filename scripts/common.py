"""Shared pieces: class list, lesion-level split, dataset, transforms, model builders."""
import os
import platform
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import models, transforms

# Fixed, sorted class order so every split and every script uses the same label indices.
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEFAULT_VIT = "models/vit-local" if os.path.isdir("models/vit-local") else "google/vit-base-patch16-224"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hardware_name():
    """e.g. 'Apple M5, 16 GB' on macOS; falls back to the platform string elsewhere."""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
        return f"{chip}, {mem / 2**30:.0f} GB"
    except Exception:
        return platform.platform()


def make_splits(df, val_frac=0.15, test_frac=0.15, seed=42):
    """Stratified train/val/test split grouped by lesion_id.

    All images of one lesion land in the same split, so the model is never
    evaluated on a lesion it has already seen. Stratification is on dx.
    """
    per_lesion = df.groupby("lesion_id")["dx"].nunique()
    if (per_lesion > 1).any():
        raise ValueError("Some lesion_ids have more than one dx label; cannot group-split safely.")

    def take_fold(sub, frac, seed):
        # StratifiedGroupKFold gives folds of ~1/n_splits; pick the first fold as the held-out part.
        n_splits = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        keep_idx, out_idx = next(sgkf.split(sub, sub["dx"], groups=sub["lesion_id"]))
        return sub.iloc[keep_idx], sub.iloc[out_idx]

    rest, test = take_fold(df, test_frac, seed)
    train, val = take_fold(rest, val_frac / (1 - test_frac), seed + 1)

    out = df.copy()
    out["split"] = ""
    out.loc[train.index, "split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test.index, "split"] = "test"

    # Sanity checks: no lesion in two splits, every image assigned.
    assert (out["split"] != "").all()
    assert out.groupby("lesion_id")["split"].nunique().max() == 1
    return out[["image_id", "lesion_id", "dx", "split"]]


def load_or_make_splits(metadata_csv, split_csv, seed=42):
    if os.path.exists(split_csv):
        print(f"Using existing split file: {split_csv}")
        return pd.read_csv(split_csv)
    df = pd.read_csv(metadata_csv)
    splits = make_splits(df, seed=seed)
    os.makedirs(os.path.dirname(split_csv) or ".", exist_ok=True)
    splits.to_csv(split_csv, index=False)
    print(f"Wrote new split file: {split_csv} (commit it so the split is reproducible)")
    return splits


class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.image_ids = df["image_id"].tolist()
        self.labels = [CLASS_TO_IDX[d] for d in df["dx"]]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, f"{self.image_ids[idx]}.jpg")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def build_transforms(mean, std, train, augment=True):
    if train and augment:
        # Dermoscopy images have no canonical orientation, so flips/rotations are label-preserving.
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_resnet(pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model, IMAGENET_MEAN, IMAGENET_STD


def get_vit(path=DEFAULT_VIT, pretrained=True, attentions=False):
    from transformers import ViTConfig, ViTForImageClassification

    kwargs = {"attn_implementation": "eager"} if attentions else {}
    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            path, num_labels=len(CLASSES), ignore_mismatched_sizes=True, **kwargs
        )
        mean, std = [0.5] * 3, [0.5] * 3
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(path)
            mean, std = list(proc.image_mean), list(proc.image_std)
        except Exception:
            pass  # google/vit-base-patch16-224 uses 0.5/0.5
    else:  # random init, only for offline smoke tests
        model = ViTForImageClassification(ViTConfig(num_labels=len(CLASSES), **kwargs))
        mean, std = [0.5] * 3, [0.5] * 3
    model.config.id2label = dict(enumerate(CLASSES))
    model.config.label2id = CLASS_TO_IDX
    return model, mean, std


def build_model(name, vit_path=DEFAULT_VIT, pretrained=True, attentions=False):
    if name == "resnet50":
        return get_resnet(pretrained)
    if name == "vit":
        return get_vit(vit_path, pretrained, attentions)
    raise ValueError(name)


def logits_of(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs


def class_weights(labels, mode="inv", n_classes=len(CLASSES)):
    """Per-class loss weights from training-set frequencies.

    inv:  w_c proportional to 1/n_c  (nv vs df is roughly 60x)
    sqrt: w_c proportional to 1/sqrt(n_c)  (milder)
    Normalised so the average weight over training samples is 1.
    """
    counts = np.maximum(np.bincount(labels, minlength=n_classes).astype(float), 1)
    w = 1.0 / counts if mode == "inv" else 1.0 / np.sqrt(counts)
    w = w * counts.sum() / (w * counts).sum()
    return torch.tensor(w, dtype=torch.float32)
