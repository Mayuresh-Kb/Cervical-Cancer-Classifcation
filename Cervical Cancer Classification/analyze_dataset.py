"""Audit class balance, integrity, and filename-family leakage in a split."""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

def family(path):
    match = re.match(r"(.+?)_\d+$", path.stem)
    return match.group(1) if match else path.stem

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", type=Path, default=Path("data/SIPaKMeD")); p.add_argument("--output", type=Path, default=Path("dataset_report.json")); args = p.parse_args()
    records, hashes, families, dimensions, corrupt = [], defaultdict(set), defaultdict(set), Counter(), []
    for path in sorted(args.data_dir.glob("*/*/*")):
        if path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg"}: continue
        split, label = path.relative_to(args.data_dir).parts[:2]
        try:
            with Image.open(path) as image: dimensions[f"{image.width}x{image.height}"] += 1; image.verify()
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()].add(split); families[family(path)].add(split); records.append({"split": split, "class": label, "source_group": family(path), "image": path.name})
        except Exception as error: corrupt.append({"image": str(path), "error": str(error)})
    counts = Counter((r["split"], r["class"]) for r in records)
    report = {"total_images": len(records), "source_groups": len(families), "classes": sorted({r["class"] for r in records}), "split_class_counts": {f"{a}/{b}": n for (a,b),n in sorted(counts.items())}, "unique_dimensions": len(dimensions), "exact_duplicate_groups_across_splits": sum(len(x)>1 for x in hashes.values()), "source_groups_across_splits": sum(len(x)>1 for x in families.values()), "corrupt_images": corrupt}
    args.output.write_text(json.dumps(report, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as out: writer = csv.DictWriter(out, fieldnames=["split", "class", "source_group", "image"]); writer.writeheader(); writer.writerows(records)
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
