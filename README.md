# Aesthetic-SHAP-Interpreter

> Interpretable Aesthetic Image Scoring on AVA Dataset: 8-Category Semantic Region Decomposition + Kernel SHAP Attribution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [1. Overview & Motivation](#1-overview--motivation)
- [2. Method Pipeline](#2-method-pipeline)
  - [2.1 Semantic Segmentation: ADE20K → 8 Categories](#21-semantic-segmentation-ade20k--8-categories)
  - [2.2 Feature Extraction: Global + ROI-per-Region](#22-feature-extraction-global--roi-per-region)
  - [2.3 Scoring Model](#23-scoring-model)
  - [2.4 SHAP Interpretability](#24-shap-interpretability)
- [3. Project Structure](#3-project-structure)
- [4. Environment & Dependencies](#4-environment--dependencies)
- [5. Data Preparation](#5-data-preparation)
  - [5.1 AVA Dataset](#51-ava-dataset)
  - [5.2 Four-Step Pipeline](#52-four-step-pipeline)
- [6. Training](#6-training)
  - [6.1 SegFormer Fine-Tuning (Optional)](#61-segformer-fine-tuning-optional)
  - [6.2 Full Model End-to-End Training](#62-full-model-end-to-end-training)
- [7. SHAP Analysis & Faithfulness Evaluation](#7-shap-analysis--faithfulness-evaluation)
- [8. Expected Directory Layout on Your Machine](#8-expected-directory-layout-on-your-machine)
- [9. Results Summary](#9-results-summary)
- [10. FAQ / Common Pitfalls](#10-faq--common-pitfalls)
- [11. Citation](#11-citation)
- [12. License](#12-license)

---

## 1. Overview & Motivation

**Goal.** Explain *why* a deep aesthetic model assigns a certain score to an image, by decomposing the image into 8 human-interpretable semantic regions and using SHAP to attribute each region's contribution.

**Why not vanilla SHAP on pixels?** Pixel-level SHAP is noisy and hard to read for an artist or reviewer. Instead, we first run a segmentation model to obtain 8 semantic masks (background / face / sky / building / vegetation / water / object / noise), then treat each region as a single "feature unit" for SHAP. The result is an 8-bar attribution per image — directly interpretable.

**Contributions of this repository:**

1. An **8-category semantic segmentation** pipeline that adapts ADE20K-pretrained SegFormer-B0 (150 classes) down to 8 in-domain categories by *keeping the encoder and replacing the classification head*, then fine-tuning on AVA for 30 epochs.
2. A **dual-stream feature extractor** that computes 12 hand-crafted features both globally and per-region (ROI-masked), yielding a compact region-aware feature vector per image.
3. An **aesthetic scoring model** (MLP on top of global + regional features) with SROCC / PLCC evaluation.
4. A **Kernel SHAP explainability suite** with stratified-sampling background sets, per-image waterfall plots, dataset-level beeswarm summaries, and faithfulness evaluation via region-masking perturbation.

---

## 2. Method Pipeline

### 2.1 Semantic Segmentation: ADE20K → 8 Categories

| Index | Name        | Typical ADE20K classes remapped to this category        |
|------:|:------------|:--------------------------------------------------------|
| 0     | background  | wall, floor, ceiling, stairs, rug, and unclassified pixels |
| 1     | face        | *not present in ADE20K; learned from human-face crops during fine-tuning* |
| 2     | sky         | sky                                                     |
| 3     | building    | building, house, skyscraper, bridge, door, window, stairs, column, arch |
| 4     | vegetation  | tree, plant, flower, grass, bush, field, lawn           |
| 5     | water       | sea, river, lake, water, fountain, swimming pool        |
| 6     | object      | *everything else* — furniture, car, chair, desk, lamp, food, clothing, book, screen, box, painting, … |
| 7     | noise       | low-confidence / isolated pixels (post-filtered by median blur) |

**How the 150→8 mapping is actually implemented:**

- **It is NOT a post-hoc lookup table.** We load SegFormer-B0 pretrained on ADE20K (150 outputs), replace the 1×1 classification conv with a new one that has 8 output channels, then fine-tune the whole network on AVA images with 8-class pseudo-ground-truth masks.
- The first round of pseudo-ground-truth was produced by running the original 150-class SegFormer and heuristically merging classes into the 8 above. Once the first fine-tuned 8-class model is available, we use it to regenerate final masks (`step3__semantic_annotation.py`), apply a 5×5 median blur to remove salt-and-pepper artefacts, and save as single-channel uint8 PNGs with pixel values in `{0..7}`.
- Class frequency weights used for training the segmentation branch (`models/segformer_module.py:70`):

  | background | face  | sky   | building | vegetation | water | object | noise |
  |:----------:|:-----:|:-----:|:--------:|:----------:|:-----:|:------:|:-----:|
  | 15.56%     | 0.48% | 5.48% | 40.91%   | 15.39%     | 7.42% | 14.76% | 1.00% |

This is a realistic frequency distribution for AVA-style photography (buildings dominate, faces are relatively rare).

### 2.2 Feature Extraction: Global + ROI-per-Region

For every image we extract **12 hand-crafted features** in two modes:

| Feature              | Description                                    |
|----------------------|------------------------------------------------|
| Brightness (mean)    | Mean luminance in YCbCr                         |
| Brightness (std)     | Std of luminance (textures)                     |
| Contrast             | RMS contrast                                    |
| Saturation (mean)    | Mean saturation in HSV                          |
| Saturation (std)     | Std of saturation                               |
| Hue (mean)           | Mean hue                                        |
| Sharpness            | Variance of Laplacian (MLV focus / clarity)     |
| Colorfulness        | Hasler & Süsstrunk colorfulness metric         |
| Symmetry             | Horizontal + vertical symmetry score            |
| Composition (rule-of-thirds) | Distance of mass centroid to rule-of-thirds grid points |
| Warm / cool color ratio | Warm pixels vs. cool pixels                    |
| Clear / blurry ratio (MLV) | High-variance vs. low-variance pixel mass     |

These 12 features are computed:

- **Once globally** over the full image.
- **Once per region** for each of the 8 semantic masks (ROI-masked computation inside the bounding box of that region).

Final feature vector per image → `12 (global) + 8 × 12 (region) = 108-dim`, flattened and passed to the scoring MLP.

### 2.3 Scoring Model
