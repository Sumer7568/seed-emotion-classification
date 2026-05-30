# EEG-Based Emotion Classification using SEED Dataset

## Overview
This repository contains a machine learning pipeline for classifying human emotions (Positive, Neutral, Negative) using Electroencephalogram (EEG) data from the [SEED (SJTU Emotion EEG Dataset)](https://bcmi.sjtu.edu.cn/home/seed/index.html). 

The project uses a Support Vector Machine (SVM) and leverages Linear Dynamic System (LDS) smoothed features to achieve high-accuracy, subject-independent emotion recognition.

## Key Features
* **Leakage Prevention:** Implements a custom file-level train/test split. By dividing the dataset by sessions rather than individual seconds, it ensures that adjacent time-series data from the same trial do not bleed across the training and testing sets.
* **Temporal Granularity:** Processes EEG data second-by-second (310 features per second) rather than averaging whole trials, capturing micro-fluctuations in emotional states.
* **Dimensionality Reduction:** Uses Principal Component Analysis (PCA) retaining 95% variance to filter EEG noise, reduce data complexity, and optimize model training time.
* **Class Balancing:** Utilizes a balanced class-weight Support Vector Machine (SVM) to fairly classify emotional states across complex physiological data.

## Dataset
This project uses the **ExtractedFeatures_1s** subset of the SEED dataset, specifically utilizing the `de_LDS` (Differential Entropy smoothed by Linear Dynamic System) features.

> **Note:** Due to licensing and file size constraints, the dataset is NOT included in this repository. You must request access from the [official BCMI website](https://bcmi.sjtu.edu.cn/home/seed/index.html) and place the downloaded `.mat` files into the appropriate directory before running the code.

## Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/sumer7568/seed-emotion-classification.git](https://github.com/sumer7568/seed-emotion-classification.git)
   cd seed-emotion-classification
Install the required Python packages:

Bash
pip install -r requirements.txt
Update the path variable inside the script to point to your local or mounted dataset directory.

Usage
To train the model and generate the evaluation metrics, run the main script:

Bash
python src/train_svm.py
(If you are running this in Google Colab, you can upload the notebook file and run the cells directly).

Results
The model evaluates performance across three emotional states:

0: Negative (Frustrated/Sad)

1: Neutral (Calm/Processing)

2: Positive (Happy/Engaged)

Upon completion, the script outputs the overall Accuracy, Precision, Recall, and F1-score to the console, and generates a visual Confusion Matrix (confusion_matrix.png) to help analyze misclassifications.

Technologies Used
Python 3.x

Scikit-Learn (Machine Learning, PCA, Data Scaling)

SciPy (MATLAB .mat file parsing)

NumPy (Data manipulation and matrix operations)

Matplotlib / Seaborn (Data visualization)
