# Targeted-Training-Data-Augmentation
Targeted Training Data Augmentation for Under-sampled Subspaces in Machine Learning Models



> A data-driven, automated augmentation framework that identifies and preferentially fills intra-class sparse regions in feature space — no manual tuning required.

---

## Overview

Many real-world datasets contain classes with **regions of feature space sparsity** — areas where data samples are rare, sparse, or poorly represented. These regions limit a model's ability to learn reliable decision boundaries, impairing generalization.

Conventional augmentation approaches generate synthetic samples **uniformly** across the dataset, without explicitly addressing these sparse regions.

---

## Method

We propose a novel **targeted augmentation** approach that:

- Operates on a **class-by-class basis**
- Uses **L₁-norm Principal Component Analysis (PCA)** to characterize intra-class feature space geometry
- Identifies **non-core (sparse) regions** and allocates synthetic samples preferentially to these areas

The method automatically:

1. Computes a **class-specific augmentation budget**
2. Allocates synthetic samples — generated via multiple augmentation techniques — **in proportion to the degree of intra-class data scarcity**

---

## Key Properties

| Property | Description |
|---|---|
|  **Targeted** | Focuses augmentation on sparse, underrepresented regions |
|  **Automated** | No manual parameter tuning required |
| **Modular** | Compatible with multiple synthetic data generation techniques |
|  **Data-driven** | Budget and allocation derived entirely from data geometry |

---

## Results

Extensive experiments on real-world datasets demonstrate that the proposed method **consistently improves classification performance** relative to:

- Baseline (no augmentation)
- Generic (uniform) augmentation strategies

Gains are most pronounced in **data-sparse regions**.

---

## Citation

> Coming soon.
