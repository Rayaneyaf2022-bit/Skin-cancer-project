"""Fine-tune ResNet-50 and ViT-Base on HAM10000 with a lesion-level train/val/test split.

Usage (from the repo root):
    python3 scripts/train_models.py                       # both models, 5 epochs
    python3 scripts/train_models.py --models resnet50 --epochs 3
    python3 scripts/train_models.py --limit 200 --epochs 1  # quick check that everything runs

Outputs:
    splits/ham10000_split.csv          lesion-grouped split (commit this)
    models/<model>_best.pth            checkpoint with the best validation balanced accuracy
    results/<model>_history.csv        per-epoch train loss / val metrics / timing
    results/<model>_metrics.json       final val + test metrics of the best checkpoint
    results/<model>_confusion_test.png
    results/summary.md, results/*.tex  tables built from the JSON files (used by README / report)
"""
import argparse
import gc
import glob
import json
import os
import platform
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from common import (CLASSES, DEFAULT_VIT, SkinDataset, build_model, build_transforms, class_weights,
                    get_device, hardware_name, load_or_make_splits, logits_of)

PRETTY = {"resnet50": "ResNet-50", "vit": "ViT-Base/16"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(logits_of(model(images)), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    ys, preds, loss_sum = [], [], 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = logits_of(model(images))
        loss_sum += ce(logits, labels).item()
        ys.append(labels.cpu())
        preds.append(logits.argmax(1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    return y, p, loss_sum / len(y)


def metrics(y, p):
    labels = list(range(len(CLASSES)))
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "per_class": {c: {"recall": float(rec[i]), "precision": float(prec[i]),
                          "f1": float(f1[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
    }


def bootstrap_ci(y, p, seed, n_boot=2000):
    """95% percentile bootstrap CI over test images for accuracy and balanced accuracy."""
    rng = np.random.default_rng(seed)
    acc, bacc = [], []
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        acc.append(np.mean(y[i] == p[i]))
        bacc.append(balanced_accuracy_score(y[i], p[i]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"accuracy": q(acc), "balanced_accuracy": q(bacc)}


def paired_bootstrap_diff(y, p_a, p_b, seed, n_boot=2000):
    """CI for balanced_accuracy(a) - balanced_accuracy(b) on the same test images."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d.append(balanced_accuracy_score(y[i], p_a[i]) - balanced_accuracy_score(y[i], p_b[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def save_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(name, args, splits, device):
    set_seed(args.seed)
    model, mean, std = build_model(name, args.vit_path, pretrained=not args.no_pretrained)
    model = model.to(device)

    parts = {s: splits[splits.split == s] for s in ("train", "val", "test")}
    if args.limit:
        parts = {s: d.sample(min(args.limit, len(d)), random_state=args.seed) for s, d in parts.items()}

    img_dir = os.path.join(args.data_dir, "all_images")
    ds = {
        "train": SkinDataset(parts["train"], img_dir, build_transforms(mean, std, True, not args.no_augment)),
        "val": SkinDataset(parts["val"], img_dir, build_transforms(mean, std, False)),
        "test": SkinDataset(parts["test"], img_dir, build_transforms(mean, std, False)),
    }
    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(ds["train"], shuffle=True, generator=g,
                          persistent_workers=args.num_workers > 0, **dl_kw)
    val_dl = DataLoader(ds["val"], shuffle=False, **dl_kw)
    test_dl = DataLoader(ds["test"], shuffle=False, **dl_kw)

    weights = None
    if args.class_weights != "none":
        weights = class_weights(ds["train"].labels, args.class_weights)
        print("Class weights:", {c: round(float(w), 2) for c, w in zip(CLASSES, weights)})
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    lr = args.lr_resnet if name == "resnet50" else args.lr_vit
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = os.path.join(args.model_dir, f"{name}_best.pth")
    history, best = [], -1.0
    print(f"\n=== {PRETTY[name]}: train {len(ds['train'])}, val {len(ds['val'])}, "
          f"test {len(ds['test'])} images, lr {lr}, {args.epochs} epochs ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
        train_time = time.time() - t0
        scheduler.step()
        y, p, val_loss = predict(model, val_dl, device)
        acc, bacc = accuracy_score(y, p), balanced_accuracy_score(y, p)
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=acc,
                   val_balanced_accuracy=bacc, train_time_s=train_time,
                   train_img_per_s=len(ds["train"]) / train_time)
        history.append(row)
        flag = ""
        if bacc > best:
            best = bacc
            torch.save({"model": name, "state_dict": model.state_dict(), "epoch": epoch,
                        "vit_path": args.vit_path if name == "vit" else None,
                        "pretrained": not args.no_pretrained, "mean": mean, "std": std}, ckpt)
            flag = "  <- best, saved"
        print(f"epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {acc:.4f} | "
              f"val bal acc {bacc:.4f} | {train_time / 60:.1f} min{flag}")
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, f"{name}_history.csv"), index=False)

    # Evaluate the best-on-validation checkpoint; the test set is used exactly once, here.
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    yv, pv, _ = predict(model, val_dl, device)
    yt, pt, _ = predict(model, test_dl, device)
    result = {
        "model": name,
        "best_epoch": state["epoch"],
        "val": metrics(yv, pv),
        "test": metrics(yt, pt) | {"bootstrap_ci95": bootstrap_ci(yt, pt, args.seed)},
        "test_predictions": {"y_true": yt.tolist(), "y_pred": pt.tolist()},
        "majority_baseline_test": {
            "accuracy": float(np.mean(yt == np.bincount(ds["train"].labels).argmax())),
            "balanced_accuracy": 1.0 / len(CLASSES),
        },
        "mean_train_min_per_epoch": float(np.mean([h["train_time_s"] for h in history]) / 60),
        "config": {k: v for k, v in vars(args).items() if k != "models"} | {
            "lr": lr, "device": str(device), "torch": torch.__version__,
            "machine": platform.platform(), "hardware": hardware_name(), "smoke_test": bool(args.limit or args.no_pretrained),
        },
    }
    with open(os.path.join(args.out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    save_confusion(result["test"]["confusion_matrix"], os.path.join(args.out_dir, f"{name}_confusion_test.png"),
                   f"{PRETTY[name]} - test set")
    print(f"{PRETTY[name]} best epoch {state['epoch']} | TEST acc {result['test']['accuracy']:.4f} | "
          f"bal acc {result['test']['balanced_accuracy']:.4f} | macro F1 {result['test']['macro_f1']:.4f}")

    # Release workers and memory before the next model starts.
    del train_dl, val_dl, test_dl, model, optimizer, state
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def write_summaries(out_dir, split_csv="splits/ham10000_split.csv"):
    """Build Markdown and LaTeX tables from every <model>_metrics.json in out_dir."""
    results = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_metrics.json"))):
        with open(path) as f:
            results.append(json.load(f))
    if not results:
        return
    smoke = any(r["config"].get("smoke_test") for r in results)
    base = results[0]["majority_baseline_test"]
    n_test = results[0]["test"]["n"]

    md = []
    if smoke:
        md.append("> **Smoke-test run (subsampled data or no pretrained weights). Not real results.**\n")
    md.append(f"Test set: {n_test} images (lesion-level split, never used for model selection).\n")
    md.append("| Model | Best epoch | Val bal. acc | Test acc | Test bal. acc [95% CI] | Test macro F1 | Train min/epoch |")
    md.append("|---|---|---|---|---|---|---|")
    md.append(f"| Majority class (nv) | - | - | {base['accuracy']:.3f} | {base['balanced_accuracy']:.3f} | - | - |")
    for r in results:
        lo, hi = r["test"]["bootstrap_ci95"]["balanced_accuracy"]
        md.append(f"| {PRETTY[r['model']]} | {r['best_epoch']}/{r['config']['epochs']} | "
                  f"{r['val']['balanced_accuracy']:.3f} | {r['test']['accuracy']:.3f} | "
                  f"{r['test']['balanced_accuracy']:.3f} [{lo:.3f}, {hi:.3f}] | {r['test']['macro_f1']:.3f} | "
                  f"{r['mean_train_min_per_epoch']:.1f} |")
    md.append("\nPer-class test recall (support in brackets):\n")
    md.append("| Class | " + " | ".join(PRETTY[r["model"]] for r in results) + " |")
    md.append("|---|" + "---|" * len(results))
    for c in CLASSES:
        sup = results[0]["test"]["per_class"][c]["support"]
        md.append(f"| {c} ({sup}) | " + " | ".join(f"{r['test']['per_class'][c]['recall']:.3f}" for r in results) + " |")
    by = {r["model"]: r for r in results}
    diff = None
    if {"resnet50", "vit"} <= by.keys():
        ya = np.array(by["resnet50"]["test_predictions"]["y_true"])
        yb = np.array(by["vit"]["test_predictions"]["y_true"])
        if np.array_equal(ya, yb):
            diff = paired_bootstrap_diff(ya, np.array(by["resnet50"]["test_predictions"]["y_pred"]),
                                         np.array(by["vit"]["test_predictions"]["y_pred"]),
                                         results[0]["config"]["seed"])
            md.append(f"\nTest balanced accuracy, ResNet-50 minus ViT: {diff[0]:+.3f} "
                      f"(95% paired bootstrap CI {diff[1]:+.3f} to {diff[2]:+.3f}). "
                      "If the interval contains 0, this run does not show a reliable difference.")
    md.append("\nSingle run, single seed, single split. 95% CIs are bootstrap over test images "
              "and do not include seed-to-seed training variance.")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    tex = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
           r"Model & Test acc. & Test bal. acc. [95\% CI] & Macro F1 & min/ep. \\ \midrule",
           f"Majority (nv) & {base['accuracy']:.3f} & {base['balanced_accuracy']:.3f} & -- & -- \\\\"]
    for r in results:
        lo, hi = r["test"]["bootstrap_ci95"]["balanced_accuracy"]
        tex.append(f"{PRETTY[r['model']]} & {r['test']['accuracy']:.3f} & "
                   f"{r['test']['balanced_accuracy']:.3f} [{lo:.3f}, {hi:.3f}] & {r['test']['macro_f1']:.3f} & "
                   f"{r['mean_train_min_per_epoch']:.1f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "results_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    tex = [r"\begin{tabular}{@{}lc" + "c" * len(results) + "@{}}", r"\toprule",
           "Class & $n$ & " + " & ".join(PRETTY[r["model"]] for r in results) + r" \\ \midrule"]
    for c in CLASSES:
        sup = results[0]["test"]["per_class"][c]["support"]
        tex.append(f"{c} & {sup} & " + " & ".join(f"{r['test']['per_class'][c]['recall']:.3f}" for r in results) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "per_class_recall.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    counts = pd.read_csv(split_csv) if os.path.exists(split_csv) else None
    if counts is not None:
        t = counts.groupby(["dx", "split"]).size().unstack(fill_value=0).reindex(CLASSES)[["train", "val", "test"]]
        tex = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Class & Train & Val & Test \\ \midrule"]
        tex += [f"{c} & {t.loc[c, 'train']} & {t.loc[c, 'val']} & {t.loc[c, 'test']} \\\\" for c in CLASSES]
        tex += [r"\midrule", f"Total & {t['train'].sum()} & {t['val'].sum()} & {t['test'].sum()} \\\\",
                r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(out_dir, "split_table.tex"), "w") as f:
            f.write("\n".join(tex) + "\n")
    def fmt(x):
        return f"{x:.3f}"
    hw = results[0]["config"].get("hardware", "??").replace("_", r"\_")
    mac = [f"\\newcommand{{\\nTest}}{{{n_test}}}",
           f"\\newcommand{{\\majorityAcc}}{{{fmt(base['accuracy'])}}}",
           f"\\newcommand{{\\smokeTest}}{{{'1' if smoke else '0'}}}",
           f"\\newcommand{{\\hardware}}{{{hw}}}"]
    for key, r in (("resnet", by.get("resnet50")), ("vit", by.get("vit"))):
        if r is None:
            continue
        lo, hi = r["test"]["bootstrap_ci95"]["balanced_accuracy"]
        vals = {"TestAcc": fmt(r["test"]["accuracy"]), "TestBacc": fmt(r["test"]["balanced_accuracy"]),
                "TestBaccCI": f"[{fmt(lo)}, {fmt(hi)}]", "TestFone": fmt(r["test"]["macro_f1"]),
                "MelRecall": fmt(r["test"]["per_class"]["mel"]["recall"]),
                "BestEpoch": str(r["best_epoch"]), "Epochs": str(r["config"]["epochs"]),
                "MinPerEpoch": f"{r['mean_train_min_per_epoch']:.1f}"}
        mac += [f"\\newcommand{{\\{key}{k}}}{{{v}}}" for k, v in vals.items()]
    if counts is not None:
        mac += [f"\\newcommand{{\\nTrain}}{{{int((counts.split == 'train').sum())}}}",
                f"\\newcommand{{\\nVal}}{{{int((counts.split == 'val').sum())}}}",
                f"\\newcommand{{\\nImages}}{{{len(counts)}}}",
                f"\\newcommand{{\\nLesions}}{{{counts.lesion_id.nunique()}}}"]
    if diff:
        mac.append(f"\\newcommand{{\\baccDiff}}{{{diff[0]:+.3f}}}")
        mac.append(f"\\newcommand{{\\baccDiffCI}}{{[{diff[1]:+.3f}, {diff[2]:+.3f}]}}")
    with open(os.path.join(out_dir, "macros.tex"), "w") as f:
        f.write("\n".join(mac) + "\n")
    print(f"\nSummary tables written to {out_dir}/ (summary.md, *.tex)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit"], choices=["resnet50", "vit"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-resnet", type=float, default=1e-4)
    ap.add_argument("--lr-vit", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--class-weights", choices=["inv", "sqrt", "none"], default="inv")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split-csv", default="splits/ham10000_split.csv")
    ap.add_argument("--vit-path", default=DEFAULT_VIT, help="local folder or Hugging Face id")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap images per split")
    ap.add_argument("--no-pretrained", action="store_true", help="debug: random init (offline smoke test)")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device} ({hardware_name()})")

    splits = load_or_make_splits(os.path.join(args.data_dir, "HAM10000_metadata.csv"), args.split_csv, args.seed)
    print(splits.groupby(["split", "dx"]).size().unstack(fill_value=0).reindex(columns=CLASSES))

    for name in args.models:
        run(name, args, splits, device)
    write_summaries(args.out_dir, args.split_csv)


if __name__ == "__main__":
    main()
