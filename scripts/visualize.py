"""Explanation maps from the trained checkpoints.

ResNet-50: Grad-CAM on the last convolutional block (Selvaraju et al., 2017).
ViT-Base:  attention rollout over all layers, CLS-token row (Abnar & Zuidema, 2020).

Both maps are computed for the class each model predicts. They show which pixels the
prediction is sensitive to; they are qualitative and do not prove the model uses
clinically meaningful features.

Usage (after train_models.py):
    python3 scripts/visualize.py                         # one test image each for mel, bcc, nv
    python3 scripts/visualize.py --images ISIC_0024306 ISIC_0025030
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import CLASSES, DEFAULT_VIT, build_model, build_transforms, get_device


def load(name, ckpt_path, device, vit_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Rebuild the same architecture used in training, then overwrite with the fine-tuned weights.
    model, _, _ = build_model(name, ckpt.get("vit_path") or vit_path,
                              pretrained=ckpt.get("pretrained", True), attentions=True)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), build_transforms(ckpt["mean"], ckpt["std"], train=False)


def grad_cam(model, x):
    acts, grads = {}, {}
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    logits = model(x)
    pred = logits.argmax(1).item()
    model.zero_grad()
    logits[0, pred].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), pred, logits.softmax(1)[0, pred].item()


@torch.no_grad()
def attention_rollout(model, x):
    out = model(x, output_attentions=True)
    pred = out.logits.argmax(1).item()
    rollout = None
    for att in out.attentions:                      # each: (1, heads, tokens, tokens)
        a = att.mean(1)[0]                          # average heads
        a = a + torch.eye(a.size(-1), device=a.device)  # residual connection
        a = a / a.sum(-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout
    cls = rollout[0, 1:]                            # CLS attention to patch tokens
    side = int(cls.numel() ** 0.5)
    m = cls.reshape(1, 1, side, side)
    m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.cpu().numpy(), pred, out.logits.softmax(1)[0, pred].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", help="image_ids; default: first test image of mel, bcc, nv")
    ap.add_argument("--split-csv", default="splits/ham10000_split.csv")
    ap.add_argument("--img-dir", default="data/all_images")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--vit-path", default=DEFAULT_VIT)
    ap.add_argument("--out", default="supplementary/attention_map.png")
    args = ap.parse_args()

    device = get_device()
    splits = pd.read_csv(args.split_csv)
    if args.images:
        rows = splits.set_index("image_id").loc[args.images].reset_index()
    else:
        test = splits[splits.split == "test"]
        rows = pd.concat([test[test.dx == c].head(1) for c in ("mel", "bcc", "nv")])

    resnet, tf_r = load("resnet50", os.path.join(args.model_dir, "resnet50_best.pth"), device, args.vit_path)
    vit, tf_v = load("vit", os.path.join(args.model_dir, "vit_best.pth"), device, args.vit_path)

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.1 * len(rows)), squeeze=False)
    for i, (_, r) in enumerate(rows.iterrows()):
        img = Image.open(os.path.join(args.img_dir, f"{r.image_id}.jpg")).convert("RGB")
        shown = np.asarray(img.resize((224, 224))) / 255.0
        cam, pr, cr = grad_cam(resnet, tf_r(img).unsqueeze(0).to(device))
        rol, pv, cv = attention_rollout(vit, tf_v(img).unsqueeze(0).to(device))

        axes[i, 0].imshow(shown)
        axes[i, 0].set_title(f"{r.image_id}\ntrue: {r.dx}", fontsize=9)
        for ax, m, name, p, c in ((axes[i, 1], cam, "ResNet-50 Grad-CAM", pr, cr),
                                  (axes[i, 2], rol, "ViT attention rollout", pv, cv)):
            ax.imshow(shown)
            ax.imshow(m, cmap="jet", alpha=0.45)
            ax.set_title(f"{name}\npred: {CLASSES[p]} ({c:.2f})", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
