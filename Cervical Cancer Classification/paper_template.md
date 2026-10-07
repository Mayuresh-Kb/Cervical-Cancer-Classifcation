# Source-group-aware cervical cytology classification with pretrained deep networks

## Abstract

**Background:** Automated cervical cytology classification may support screening workflows. **Methods:** We evaluated pretrained convolutional and transformer-based models on the five-class SIPaKMeD cell-morphology benchmark using source-group-aware cross-validation. **Results:** Insert the cross-validation mean and 95% confidence interval for the selected metrics from `cross_validation_summary.csv`. **Conclusion:** State only benchmark-level conclusions; do not claim clinical utility without independent external validation.

## Methods

### Dataset and labels

Describe the five SIPaKMeD classes, the original dataset citation, class counts, and that this local copy comprises 4,049 cropped cells from 271 filename source groups. Explain that a source group is the filename prefix and that all its crops, including crops with differing labels, were kept in one fold.

### Models and training

List every compared architecture, ImageNet-pretrained status, input size, normalization, augmentations, optimizer, learning rate, batch size, weight decay, maximum epochs, early-stopping patience, seed, device, and software versions. Copy exact values from each run's `config.json`.

### Evaluation protocol

State the number of outer group-aware folds, the inner validation protocol, and that the outer test fold was not used for epoch selection. Report mean and 95% confidence intervals across outer folds. Define all metrics.

## Results

### Model comparison

Insert the rows from `paper_model_comparison.csv` or `cross_validation_summary.csv`. Include accuracy, balanced accuracy, macro-F1, weighted F1, macro recall/sensitivity, macro specificity, and macro one-vs-rest AUC. Include per-class metrics and a combined out-of-fold confusion matrix.

### Error and explainability analysis

Describe recurrent confusions using out-of-fold predictions. Include representative Grad-CAM figures for correct and incorrect predictions; present them as qualitative explanations, not proof of causal reasoning.

## Limitations

The unit of analysis is a cropped cell rather than a patient. This dataset is small, comes from a limited source, and does not establish prospective or patient-level diagnostic performance. Augmentation does not add independent cases. State whether an external, untouched dataset was evaluated; if not, explicitly label it as future work.

## Reproducibility statement

Provide the repository revision, commands, requirements file, random seeds, and all experiment artifacts. Do not report only a best fold or a test result chosen after model selection.
