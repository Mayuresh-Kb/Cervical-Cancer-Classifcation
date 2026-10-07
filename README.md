# Cervical cytology classification research pipeline

This project supports reproducible five-class SIPaKMeD cytology experiments. It is a cell-morphology classification benchmark, not a validated patient-level cervical-cancer diagnostic system.

## Data integrity

The archive contains 4,049 RGB cell crops from 271 filename source groups in this copy of the dataset. The checked-in image-level split leaks 227 of those groups across partitions. Crops from the same source image can have different class labels, so all labels sharing an image prefix must remain in the same partition. The supplied splitter and cross-validation workflow enforce that rule.

```bash
cd "Cervical Cancer Classification"
python split_dataset.py
python analyze_dataset.py --data-dir data/SIPaKMeD_grouped --output grouped_dataset_report.json
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Development experiments

Use the grouped holdout only for quick iteration and model selection. It has early stopping on validation macro-F1 and writes checkpoints, per-image probabilities, learning curves, per-class metrics, and a confusion-matrix figure.

```bash
python train.py --model resnet18 --pretrained --epochs 50 --patience 10
python train.py --model densenet121 --pretrained --epochs 50 --patience 10
```

Supported architectures are `resnet18`, `densenet121`, `efficientnet_b0`, `convnext_tiny`, `mobilenet_v3_large`, `swin_t`, and `vgg16`. For this data size, use pretrained weights; VGG16 is a historical baseline rather than a preferred final model.

## Final paper results

Run the same protocol for the selected comparison models. This performs nested validation within each outer, source-group-aware test fold and reports mean with 95% confidence intervals across folds.

```bash
python cross_validate.py --model resnet18 --pretrained --folds 5 --epochs 50 --patience 10
python cross_validate.py --model densenet121 --pretrained --folds 5 --epochs 50 --patience 10
python cross_validate.py --model efficientnet_b0 --pretrained --folds 5 --epochs 50 --patience 10
python cross_validate.py --model convnext_tiny --pretrained --folds 5 --epochs 50 --patience 10
python summarize_runs.py --runs-dir cv_runs --output paper_model_comparison.csv
```

Do not pick the winning model from its outer-fold test scores. Fix models and hyperparameters using development work first, then use the cross-validation results solely for final comparison. Repeat the final protocol with additional random seeds when compute permits.

Metrics include accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, macro specificity, and macro one-vs-rest ROC-AUC, plus per-class sensitivity, specificity, precision, F1, confusion matrices, and out-of-fold predictions.

## Explainability

For a representative correctly and incorrectly classified image, generate Grad-CAM explanations for a CNN checkpoint:

```bash
python gradcam.py --checkpoint runs/<run>/best_checkpoint.pt --image path/to/cell.bmp --output gradcam.png
```

## Paper scope and limitations

Document the dataset source, source-group split protocol, all hyperparameters, seed, preprocessing, augmentation, pretrained weights, selected epoch, and complete metrics. Report results as cross-validation means and confidence intervals, not only the best fold. Augmentation does not create independent clinical cases. Before clinical-performance claims, validate unchanged models on an independent external dataset or cohort.
