# EEG-Based Emotion Classification using SEED Dataset

## Overview
This repository contains a machine learning pipeline for classifying human emotions (Positive, Neutral, Negative) using Electroencephalogram (EEG) data from the [SEED (SJTU Emotion EEG Dataset)](https://bcmi.sjtu.edu.cn/home/seed/index.html). 

The project uses a Support Vector Machine (SVM) and leverages Linear Dynamic System (LDS) smoothed features to achieve high-accuracy, subject-independent emotion recognition.

## Key Features
* **Robust Data Pipeline:** Utilizes `scikit-learn`'s `Pipeline` API to ensure sequential data scaling and dimensionality reduction without data leakage.
* **Leakage Prevention:** Implements `GroupShuffleSplit` at the file-level to ensure that adjacent time-series data from the same trial do not bleed across training and testing sets.
* **Dimensionality Reduction:** Uses Principal Component Analysis (PCA) retaining 95% variance to filter EEG noise and optimize training time.
* **Temporal Granularity:** Processes EEG data second-by-second (310 features per second) rather than averaging whole trials, capturing micro-fluctuations in emotional states.

## Dataset
This project uses the **ExtractedFeatures_1s** subset of the SEED dataset, specifically utilizing the `de_LDS` (Differential Entropy smoothed by Linear Dynamic System) features.

> **Note:** Due to licensing and file size constraints, the dataset is NOT included in this repository. You must request access from the [official BCMI website](https://bcmi.sjtu.edu.cn/home/seed/index.html).

## Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/sumer7568/seed-emotion-classification.git](https://github.com/sumer7568/seed-emotion-classification.git)
   cd seed-emotion-classification
