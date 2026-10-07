"""Create a source-group-aware SIPaKMeD split to prevent crop leakage."""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def source_group(path: Path) -> str:
    """The prefix (e.g. 023) identifies the multi-cell source image."""
    match = re.match(r"(.+?)_\d+$", path.stem)
    return match.group(1) if match else path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("archive"))
    parser.add_argument("--destination", type=Path, default=Path("data/SIPaKMeD_grouped"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": 1 - args.train_ratio - args.val_ratio}
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("Ratios must be positive and leave room for test.")
    if args.destination.exists() and any(args.destination.iterdir()):
        raise FileExistsError(f"{args.destination} is not empty; choose a new destination.")

    groups, all_classes = defaultdict(list), set()
    for class_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        images = sorted((class_dir / class_dir.name / "CROPPED").glob("*.bmp"))
        if not images:
            raise FileNotFoundError(f"No BMP files for {class_dir.name}")
        all_classes.add(class_dir.name)
        for image in images:
            groups[source_group(image)].append((class_dir.name, image))

    # Search reproducible group-level partitions and retain the one with the
    # closest class distribution. Source groups are indivisible, so exact class
    # percentages are not always mathematically possible.
    totals = Counter(label for group in groups.values() for label, _ in group)
    target = {split: {label: totals[label] * ratios[split] for label in all_classes} for split in ratios}
    group_ids = list(groups)
    cut_train, cut_val = round(len(group_ids) * ratios["train"]), round(len(group_ids) * (ratios["train"] + ratios["val"]))
    best_score, assignment, current = float("inf"), None, None
    for trial in range(2000):
        shuffled = group_ids[:]
        random.Random(args.seed + trial).shuffle(shuffled)
        candidates = {"train": shuffled[:cut_train], "val": shuffled[cut_train:cut_val], "test": shuffled[cut_val:]}
        trial_counts = {split: Counter(label for group in candidate for label, _ in groups[group]) for split, candidate in candidates.items()}
        score = sum((trial_counts[split][label] - target[split][label]) ** 2 / max(target[split][label], 1) for split in ratios for label in all_classes)
        if score < best_score:
            best_score, current = score, trial_counts
            assignment = {group: split for split, candidate in candidates.items() for group in candidate}

    manifest = []
    for group, items in groups.items():
        split = assignment[group]
        for label, image in items:
            target_path = args.destination / split / label
            target_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target_path / image.name)
            manifest.append({"split": split, "class": label, "source_group": group, "image": image.name})
    with (args.destination / "split_manifest.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["split", "class", "source_group", "image"])
        writer.writeheader(); writer.writerows(manifest)
    print(f"Wrote {len(manifest)} images from {len(groups)} source groups to {args.destination}")
    for split in ratios:
        print(split, dict(sorted(current[split].items())))


if __name__ == "__main__":
    main()
