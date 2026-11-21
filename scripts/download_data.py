"""Check that the dataset and pretrained weights are in place before training.

HAM10000 is downloaded by hand (Harvard Dataverse or Kaggle); this script only checks it.
Run from the repo root:  python3 scripts/download_data.py
"""
import os
import sys

import pandas as pd
import torch
from PIL import Image

from common import CLASSES, hardware_name

ok = True


def report(passed, msg):
    global ok
    ok &= passed
    print(("  OK   " if passed else "  FAIL ") + msg)


print(f"Machine: {hardware_name()} | MPS available: {torch.backends.mps.is_available()}")

print("\nDataset")
meta = "data/HAM10000_metadata.csv"
img_dir = "data/all_images"
if os.path.exists(meta):
    df = pd.read_csv(meta)
    report({"lesion_id", "image_id", "dx"} <= set(df.columns), f"{meta} has lesion_id, image_id, dx columns")
    report(len(df) == 10015, f"{len(df)} rows (expected 10015), {df.lesion_id.nunique()} lesions")
    report(set(df.dx) <= set(CLASSES), "classes: " + str({k: int(v) for k, v in df.dx.value_counts().items()}))
    if os.path.isdir(img_dir):
        files = {f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg")}
        missing = set(df.image_id) - files
        report(not missing, f"{len(files)} .jpg files in {img_dir}, {len(missing)} listed in metadata but missing")
        if files - missing:
            size = Image.open(os.path.join(img_dir, sorted(files)[0] + ".jpg")).size
            report(size == (600, 450), f"image size {size} (original HAM10000 is (600, 450))")
    else:
        report(False, f"{img_dir}/ not found")
else:
    report(False, f"{meta} not found")

print("\nViT weights (models/vit-local)")
for name, min_mb in (("config.json", 0), ("preprocessor_config.json", 0), ("model.safetensors", 300)):
    p = os.path.join("models/vit-local", name)
    mb = os.path.getsize(p) / 2**20 if os.path.exists(p) else -1
    report(mb >= min_mb, f"{name}: " + (f"{mb:.1f} MB" if mb >= 0 else "missing"))

print("\nResNet-50 weights")
p = os.path.join(torch.hub.get_dir(), "checkpoints", "resnet50-11ad3fa6.pth")
mb = os.path.getsize(p) / 2**20 if os.path.exists(p) else -1
report(mb > 90, f"{p}: " + (f"{mb:.1f} MB" if mb >= 0 else "missing (will be downloaded on first run if the network allows)"))

print("\nAll checks passed." if ok else "\nFix the FAIL lines above before training.")
sys.exit(0 if ok else 1)
