# AML Detection from Blood Smear Images
 
> A deep-learning pipeline for binary classification of Acute Myeloid Leukemia (AML) from peripheral blood smear images, comparing **ResNet-50** and **MobileNet** architectures for the accuracy–efficiency trade-off in resource-constrained medical-imaging settings.

---
 
## Overview
 
Acute Myeloid Leukemia is a fast-progressing cancer whose traditional diagnosis relies on manual microscopic review of blood smears — a workflow that is slow, subjective, and error-prone. This project investigates whether two widely-used CNN architectures can automate that classification, and which one offers the better trade-off for point-of-care deployment.

**Takeaway:** ResNet-50 gives a small accuracy lead via deeper residual feature learning, but MobileNet's depthwise-separable convolutions make it **~53 % faster per epoch at comparable diagnostic performance** — a strong case for low-resource clinical deployment.
 
---
 
## Dataset
 
- **Source:** [Blood Cancer Image Dataset](https://www.kaggle.com/datasets/akhiljethwa/blood-cancer-image-dataset) on Kaggle, originally from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- **Size:** 10,000 single-cell images, 64×64 px (upsampled to 224×224 for the CNNs)
- **Task:** Binary classification — AML positive vs. non-AML
- **Split:** 70 % train / 15 % validation / 15 % test (stratified)
The dataset is **not** included in this repo — download it from Kaggle and point `DATA_ROOT` in the notebook config cell at the unzipped folder.
 
---
 
## Preprocessing Pipeline
 
All images pass through the same preprocessing chain before hitting the CNN:
 
```
RGBA/RGB  →  resize 224×224  →  grayscale  →  median filter (k=5)
       →  stack to 3-channel  →  normalize [0, 1]  →  CNN
```
 
Two additional feature-extraction techniques from the manuscript are computed for descriptive dataset analysis (shown in notebooks, not fed to the model):
 
- **Otsu thresholding** — automatic foreground/background segmentation for cell isolation visualizations.
- **GLCM texture features** — contrast, energy, homogeneity, correlation, computed on the filtered grayscale channel and averaged across the dataset to characterize image texture.
---
 
## Model Architecture
 
Both models share the same custom classification head on top of the base convolutional backbone, trained **from scratch** (no ImageNet pretrained weights, matching the manuscript):
 
```
Base CNN (MobileNet  |  ResNet-50, include_top=False, weights=None)
   │
   ├─ GlobalAveragePooling2D
   ├─ Dense(1024, ReLU)
   ├─ Dropout(0.5)
   └─ Dense(1, sigmoid)      ← binary classifier
```
 
**Training setup**
 
| Hyperparameter    | Value                        |
|-------------------|------------------------------|
| Optimizer         | Adam                         |
| Learning rate     | 1e-3 (reduced on plateau)    |
| Loss              | Binary cross-entropy         |
| Batch size        | 32                           |
| Epochs            | 20 (with early stopping)     |
| Augmentation      | Rotation ±20°, shift 20%, h-flip |
| Regularization    | Dropout 0.5                  |
| Metrics tracked   | Accuracy, AUC                |
 
