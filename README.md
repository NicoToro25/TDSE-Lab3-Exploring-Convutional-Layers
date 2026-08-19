# Exploring Convolutional Layers through Data

## Digital Transformation in Enterprise Systems (TDSE) - Lab 3

## Nicolás Toro Criollo

---

## Introduction

This project compares two neural network architectures — a fully connected (dense) baseline and a convolutional neural network (CNN) — on an image classification task, in order to analyze how architectural choices (specifically, the use of convolutional and pooling layers) affect learning on image data. Beyond just training a model, the goal is to understand why convolution works well for images, through a controlled experiment and a written architectural interpretation.

---


## Project Structure

```
.
├── exploring-convolutional-layers.ipynb
├── README.md
```

---

## How to Run

1. Clone or download the repository.
2. Place `heart.csv` in the project directory.
3. Open `exploring-convolutional-layers.ipynb`.
4. Execute all notebook cells in order.
5. Review the generated figures, tables, and evaluation metrics.

---

## Dataset Description

The dataset used is **CIFAR-10**, restricted to a 4-class **vehicle subset**: `airplane`, `automobile`, `ship`, and `truck`. Each image is 32x32 pixels, RGB (3 channels).

This subset was chosen because these classes have enough visual diversity (different shapes, backgrounds, and viewpoints) to make the comparison between architectures meaningful, while still sharing local visual patterns (edges, textures, metallic surfaces) that convolutional layers are specifically designed to exploit through local receptive fields and weight sharing. This is in contrast to datasets like MNIST, where uniform backgrounds make the advantage of convolution less clear.

Before training, pixel values were normalized to the [0, 1] range, and a stratified validation split (15% of the training data) was created to monitor training without touching the test set.

---

## Architecture Diagrams

**Baseline (Dense) — 820,100 parameters**

Input (32x32x3)
-> Flatten (3072)
-> Dense (256, ReLU) -> Dropout (0.3)
-> Dense (128, ReLU) -> Dropout (0.3)
-> Dense (4, Softmax)


**CNN — 101,444 parameters**

Input (32x32x3)
-> Conv2D (32 filters, 3x3, same) -> MaxPooling (2x2) [32x32 -> 16x16]
-> Conv2D (64 filters, 3x3, same) -> MaxPooling (2x2) [16x16 -> 8x8]
-> Conv2D (128 filters, 3x3, same) -> MaxPooling (2x2) [8x8 -> 4x4]
-> Flatten (2048) -> Dropout (0.3)
-> Dense (4, Softmax)


The key structural difference: the baseline applies `Flatten` immediately, discarding spatial relationships between pixels. The CNN preserves spatial structure through 3 convolutional blocks before flattening only once, right before classification.

---

## Experimental Results

**Baseline vs. CNN**

| Model | Test Accuracy | Test Loss | Trainable Params |
|---|---|---|---|
| Baseline (Dense) | 60.83% | 0.9718 | 820,100 |
| CNN | 87.30% | 0.4076 | 101,444 |

The CNN improved test accuracy by ~26 percentage points while using ~8x fewer parameters than the baseline.

**Controlled experiment: effect of pooling**

| Variant | Test Accuracy | Test Loss | Trainable Params |
|---|---|---|---|
| CNN with pooling | 87.30% | 0.4076 | 101,444 |
| CNN without pooling | 80.18% | 1.1345 | 617,540 |

Removing pooling reduced test accuracy by ~7 points, more than doubled test loss, and caused clear overfitting despite the no-pooling variant having ~6x more parameters.

---

## Interpretation

The CNN works better than the dense model mainly because its architecture is better suited for images. It can look at small parts of the image and use the same filters in different positions. The baseline does not do this because it flattens the image from the beginning, so it has to learn similar patterns separately in different parts of the image.

The pooling experiment also showed that pooling is important. When we removed it, the model overfitted a lot and the results became worse. Pooling helps reduce unnecessary information from the image and makes the model focus more on the important features.

This type of architecture is useful for images and other data where nearby information is related. For normal tabular data, where the position of the columns does not really matter, convolution would not be as useful.


## Deployment Notes (SageMaker)

Training and deploying the model through Amazon SageMaker (Task 6) was attempted but not fully completed, due to a series of network restrictions in the university-provided AWS environment: no direct internet access for downloading data, a newer SDK version (v3) requiring a different API than most available tutorials, and connectivity issues reaching STS and S3 from within the sandboxed VPC. These issues, along with the workarounds attempted for each, are documented in detail inside the notebook (Section 7.5).
