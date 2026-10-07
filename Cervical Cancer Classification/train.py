"""Quick grouped-holdout experiment; use cross_validate.py for final results."""
from __future__ import annotations
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from experiment_utils import (MODEL_NAMES, CellDataset, build_model, device_from, evaluate,
                              loader, make_transforms, save_confusion_matrix, set_seed,
                              train_model, write_csv)

def group(path):
    match = re.match(r"(.+?)_\d+$", Path(path).stem)
    return match.group(1) if match else Path(path).stem

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", type=Path, default=Path("data/SIPaKMeD_grouped")); p.add_argument("--output-dir", type=Path, default=Path("runs")); p.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--patience", type=int, default=10); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--workers", type=int, default=2); p.add_argument("--seed", type=int, default=42); p.add_argument("--image-size", type=int, default=224); p.add_argument("--pretrained", action="store_true"); p.add_argument("--class-weights", action="store_true"); p.add_argument("--device")
    args = p.parse_args(); set_seed(args.seed); device = device_from(args.device)
    train_tf, eval_tf = make_transforms(args.image_size)
    base = ImageFolder(args.data_dir / "train"); classes = base.classes
    data = {}
    for split, transform in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
        split_base = ImageFolder(args.data_dir / split)
        if split_base.classes != classes: raise ValueError("Class mappings differ between splits")
        samples = [(Path(path), label, group(path)) for path, label in split_base.samples]
        data[split] = loader(CellDataset(samples, transform), args.batch_size, split == "train", args.workers)
    run = args.output_dir / f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; run.mkdir(parents=True)
    model = build_model(args.model, len(classes), args.pretrained).to(device)
    train_labels = np.array([label for _, label, _ in data["train"].dataset.samples]); counts = np.bincount(train_labels, minlength=len(classes))
    weights = torch.tensor(len(train_labels) / (len(classes) * counts), dtype=torch.float32, device=device) if args.class_weights else None
    history, stopped_epoch = train_model(model, data["train"], data["val"], device, classes, args.epochs, args.learning_rate, args.weight_decay, args.patience, weights)
    metrics, report, matrix, predictions = evaluate(model, data["test"], torch.nn.CrossEntropyLoss(weight=weights), device, classes)
    config = {**vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir), "classes": classes, "device": str(device), "train_class_counts": counts.tolist(), "stopped_epoch": stopped_epoch}
    (run / "config.json").write_text(json.dumps(config, default=str, indent=2)); (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"model": args.model, "classes": classes, "state_dict": model.state_dict()}, run / "best_checkpoint.pt")
    write_csv(run / "history.csv", history); write_csv(run / "classification_report.csv", report); write_csv(run / "predictions.csv", predictions)
    write_csv(run / "confusion_matrix.csv", [{"true_class": classes[i], **{classes[j]: int(matrix[i,j]) for j in range(len(classes))}} for i in range(len(classes))]); save_confusion_matrix(matrix, classes, run / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2)); print(f"Saved results to {run}")

if __name__ == "__main__": main()
