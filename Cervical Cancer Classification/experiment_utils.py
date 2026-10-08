"""Shared training, evaluation, and reporting utilities for experiments."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

MODEL_NAMES = ("resnet18", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v3_large", "swin_t", "vgg16")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class CellDataset(Dataset):
    def __init__(self, samples, transform): self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        path, label, group = self.samples[index]
        with Image.open(path) as image: image = image.convert("RGB")
        return self.transform(image), label, str(path), group


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def device_from(value=None):
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))


def make_transforms(size=224):
    train = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    evaluate = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, evaluate


def build_model(name, n_classes, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    if name == "resnet18": model = models.resnet18(weights=weights); model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif name == "densenet121": model = models.densenet121(weights=weights); model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    elif name == "efficientnet_b0": model = models.efficientnet_b0(weights=weights); model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    elif name == "convnext_tiny": model = models.convnext_tiny(weights=weights); model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_classes)
    elif name == "mobilenet_v3_large": model = models.mobilenet_v3_large(weights=weights); model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    elif name == "swin_t": model = models.swin_t(weights=weights); model.head = nn.Linear(model.head.in_features, n_classes)
    elif name == "vgg16": model = models.vgg16(weights=weights); model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    else: raise ValueError(f"Unknown model: {name}")
    return model


def loader(dataset, batch_size, shuffle, workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def metric_report(y_true, y_pred, probabilities, classes):
    labels = np.arange(len(classes)); matrix = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    total = matrix.sum(); specificity = []
    for index in labels:
        tp = matrix[index, index]; fn = matrix[index].sum() - tp; fp = matrix[:, index].sum() - tp; tn = total - tp - fn - fp
        specificity.append(tn / (tn + fp) if tn + fp else 0.)
    try: macro_auc = float(roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro"))
    except ValueError: macro_auc = None
    summary = {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "weighted_f1": float(np.average(f1, weights=support)), "macro_specificity": float(np.mean(specificity)), "macro_ovr_auc": macro_auc, "samples": int(len(y_true))}
    rows = [{"class": label, "precision": float(precision[i]), "recall_sensitivity": float(recall[i]), "specificity": float(specificity[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(classes)]
    return summary, rows, matrix


@torch.inference_mode()
def evaluate(model, data_loader, loss_fn, device, classes):
    model.eval(); loss_total = 0.; truth, pred, probs, rows = [], [], [], []
    for x, y, paths, groups in data_loader:
        logits = model(x.to(device, non_blocking=True)); loss_total += loss_fn(logits, y.to(device, non_blocking=True)).item() * len(y)
        probability = torch.softmax(logits, 1).cpu().numpy(); guessed = probability.argmax(1)
        for path, group, actual, guess, values in zip(paths, groups, y.numpy(), guessed, probability):
            rows.append({"image": path, "source_group": group, "true_class": classes[actual], "predicted_class": classes[guess], "confidence": float(values[guess]), **{f"prob_{label}": float(values[i]) for i, label in enumerate(classes)}})
        truth.extend(y.numpy()); pred.extend(guessed); probs.extend(probability)
    summary, report, matrix = metric_report(np.asarray(truth), np.asarray(pred), np.asarray(probs), classes)
    summary["loss"] = loss_total / len(data_loader.dataset)
    return summary, report, matrix, rows


def train_model(model, train_loader, val_loader, device, classes, epochs, learning_rate, weight_decay, patience, class_weights=None, label=""):
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=.5)
    history, best, stale, checkpoint = [], -1., 0, None
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.
        for x, y, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True)); loss.backward(); optimizer.step(); total += loss.item() * len(y)
        validation, _, _, _ = evaluate(model, val_loader, loss_fn, device, classes); scheduler.step(validation["macro_f1"])
        history.append({"epoch": epoch, "train_loss": total / len(train_loader.dataset), "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{k}": v for k, v in validation.items()}})
        prefix = f"{label} " if label else ""
        print(f"{prefix}epoch {epoch:03d}/{epochs}: train_loss={total / len(train_loader.dataset):.4f}, val_macro_f1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best:
            best, stale, checkpoint = validation["macro_f1"], 0, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: stale += 1
        if stale >= patience:
            print(f"{prefix}early stopping at epoch {epoch} (patience={patience})", flush=True)
            break
    model.load_state_dict(checkpoint)
    return history, epoch


def write_csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_confusion_matrix(matrix, classes, path):
    figure, axis = plt.subplots(figsize=(8, 6)); image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted class", ylabel="True class", title="Confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    for i in range(len(classes)):
        for j in range(len(classes)): axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(path, dpi=200); plt.close(figure)
