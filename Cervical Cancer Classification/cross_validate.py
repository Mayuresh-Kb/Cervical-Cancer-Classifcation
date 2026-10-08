"""Final paper evaluation: repeated group-aware cross-validation.

Example: python cross_validate.py --model convnext_tiny --pretrained --folds 5
"""
from __future__ import annotations
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from experiment_utils import (MODEL_NAMES, CellDataset, build_model, device_from, evaluate,
                              loader, make_transforms, save_confusion_matrix, set_seed,
                              train_model, write_csv)

def source_group(path):
    match = re.match(r"(.+?)_\d+$", path.stem)
    return match.group(1) if match else path.stem

def collect_samples(root):
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    class_to_index = {name: i for i, name in enumerate(classes)}; samples = []
    for label in classes:
        for path in sorted((root / label / label / "CROPPED").glob("*.bmp")):
            samples.append((path, class_to_index[label], source_group(path)))
    if not samples: raise FileNotFoundError(f"No archive images found in {root}")
    return samples, classes

def confidence_interval(values):
    values = np.asarray(values, dtype=float); mean = values.mean()
    half_width = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.
    return float(mean), float(mean - half_width), float(mean + half_width)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--archive-dir", type=Path, default=Path("archive")); p.add_argument("--output-dir", type=Path, default=Path("cv_runs")); p.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    p.add_argument("--folds", type=int, default=5); p.add_argument("--epochs", type=int, default=50); p.add_argument("--patience", type=int, default=10); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--workers", type=int, default=2); p.add_argument("--seed", type=int, default=42); p.add_argument("--image-size", type=int, default=224); p.add_argument("--pretrained", action="store_true"); p.add_argument("--class-weights", action="store_true"); p.add_argument("--device")
    args = p.parse_args()
    if args.folds < 3: raise ValueError("Use at least 3 group-aware folds.")
    samples, classes = collect_samples(args.archive_dir); labels = np.array([x[1] for x in samples]); groups = np.array([x[2] for x in samples]); set_seed(args.seed); device = device_from(args.device)
    run = args.output_dir / f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; run.mkdir(parents=True)
    train_tf, eval_tf = make_transforms(args.image_size); outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows, all_predictions = [], []
    for fold, (train_index, test_index) in enumerate(outer.split(np.zeros(len(labels)), labels, groups), 1):
        # Validation is drawn only from outer-training source groups.
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + fold)
        inner_train, val = next(inner.split(np.zeros(len(train_index)), labels[train_index], groups[train_index]))
        actual_train, actual_val = train_index[inner_train], train_index[val]
        train_set, val_set, test_set = CellDataset([samples[i] for i in actual_train], train_tf), CellDataset([samples[i] for i in actual_val], eval_tf), CellDataset([samples[i] for i in test_index], eval_tf)
        train_loader, val_loader, test_loader = loader(train_set, args.batch_size, True, args.workers), loader(val_set, args.batch_size, False, args.workers), loader(test_set, args.batch_size, False, args.workers)
        model = build_model(args.model, len(classes), args.pretrained).to(device)
        train_y = labels[actual_train]; counts = np.bincount(train_y, minlength=len(classes)); weights = torch.tensor(len(train_y) / (len(classes) * counts), dtype=torch.float32, device=device) if args.class_weights else None
        print(f"Starting fold {fold}/{args.folds}: {len(train_set)} train, {len(val_set)} validation, {len(test_set)} test images", flush=True)
        history, stopped = train_model(model, train_loader, val_loader, device, classes, args.epochs, args.learning_rate, args.weight_decay, args.patience, weights, label=f"fold {fold}")
        metrics, report, matrix, predictions = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(weight=weights), device, classes)
        fold_dir = run / f"fold_{fold}"; fold_dir.mkdir(); torch.save({"model": args.model, "classes": classes, "state_dict": model.state_dict()}, fold_dir / "best_checkpoint.pt")
        write_csv(fold_dir / "history.csv", history); write_csv(fold_dir / "classification_report.csv", report); write_csv(fold_dir / "predictions.csv", predictions); save_confusion_matrix(matrix, classes, fold_dir / "confusion_matrix.png")
        fold_rows.append({"fold": fold, "stopped_epoch": stopped, **metrics}); all_predictions.extend([{**row, "fold": fold} for row in predictions])
        print(f"fold {fold}/{args.folds}: macro-F1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
    write_csv(run / "fold_metrics.csv", fold_rows); write_csv(run / "oof_predictions.csv", all_predictions)
    numeric = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_specificity", "macro_ovr_auc"]
    summary = [{"metric": name, "mean": confidence_interval([row[name] for row in fold_rows if row[name] is not None])[0], "ci95_low": confidence_interval([row[name] for row in fold_rows if row[name] is not None])[1], "ci95_high": confidence_interval([row[name] for row in fold_rows if row[name] is not None])[2], "folds": sum(row[name] is not None for row in fold_rows)} for name in numeric]
    write_csv(run / "cross_validation_summary.csv", summary)
    (run / "config.json").write_text(json.dumps({**vars(args), "archive_dir": str(args.archive_dir), "output_dir": str(args.output_dir), "classes": classes, "source_groups": int(len(set(groups))), "device": str(device)}, default=str, indent=2))
    print(f"Saved final cross-validation results to {run}")

if __name__ == "__main__": main()
