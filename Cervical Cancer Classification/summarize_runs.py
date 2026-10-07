"""Build a paper-ready model-comparison table from completed experiment runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("model_comparison.csv"))
    args = parser.parse_args()
    rows = []
    for run in sorted(args.runs_dir.iterdir() if args.runs_dir.exists() else []):
        metrics_file, config_file = run / "test_metrics.json", run / "config.json"
        cv_file = run / "cross_validation_summary.csv"
        if cv_file.exists() and config_file.exists():
            config = json.loads(config_file.read_text())
            for metric in csv.DictReader(cv_file.open()):
                rows.append({"run": run.name, "protocol": "grouped_cross_validation", "model": config["model"], "metric": metric["metric"], "mean": metric["mean"], "ci95_low": metric["ci95_low"], "ci95_high": metric["ci95_high"]})
            continue
        if not metrics_file.exists() or not config_file.exists():
            continue
        metrics, config = json.loads(metrics_file.read_text()), json.loads(config_file.read_text())
        rows.append({"run": run.name, "protocol": "grouped_holdout", "model": config["model"], "metric": "macro_f1", "mean": metrics["macro_f1"], "ci95_low": "", "ci95_high": ""})
    if not rows:
        raise FileNotFoundError(f"No completed runs with metrics found in {args.runs_dir}")
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} completed runs to {args.output}")


if __name__ == "__main__": main()
