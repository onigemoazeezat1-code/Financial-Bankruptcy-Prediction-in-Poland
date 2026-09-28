# Financial-Bankruptcy-Prediction-in-Poland
This repository delivers an end-to-end data processing and machine learning workflow designed to predict corporate bankruptcy using financial ratio datasets from companies in Poland. It demonstrates full pipeline development—from handling gzip-compressed JSON ingestion and schema normalization to baseline evaluation, class imbalance handling, and model fitting.

# Project Overview
Financial bankruptcy prediction is a classic high-imbalance classification problem where minority cases (bankrupt firms) carry significant financial and operational risk.

This project establishes:

* A robust data wrangling pipeline to parse nested, compressed financial records into unified tabular formats.

* Data quality checks and schema validation across multiple regional datasets.

* Class imbalance analysis to measure majority-to-minority distribution ratios.

* A Logistic Regression baseline evaluated against a dummy classifier to benchmark predictive performance under extreme class imbalance.
# Dataset Structure
The raw datasets are stored as gzipped JSON files (.json.gz) containing company financial ratios:
* Poland Dataset: 64 financial features (feat_1 to feat_64), labeled with bankrupt (0 or 1).
* Taiwan Dataset: 95 financial features (feat_1 to feat_95), labeled with bankrupt (0 or 1).
# Key Dataset Metrics & Class ImbalanceDataset
Dataset         Total Records    Feature Count      Imbalance Ratio (Majority : Minority)
Poland(Full)    9,977            64                 20.36 : 1 (~4.68% bankrupt)

# Data Wrangling & Engineering Pipeline
* The custom wrangle() function automates data transformation through the following steps:

* Decompression & Parsing: Safely unpacks .json.gz or .json payloads regardless of structural key differences (data vs. observations).

* Tabular Conversion: Extracts records into a pandas DataFrame.

* Column Normalization: Standardizes field names into snake_case (stripping special characters, hyphens, and whitespace).

* Numeric Coercion: Coerces financial string metrics into float values, converting non-numeric artifacts into NaN.

# Machine Learning & Baseline Modeling
Baseline Comparison (Poland Dataset)
A DummyClassifier using the most_frequent strategy was evaluated on a 75/25 train-test split to illustrate the accuracy paradox inherent to highly imbalanced data:

* Dummy Classifier Accuracy: ~97.60%

* Dummy Classifier Precision / Recall / F1: 0.00 / 0.00 / 0.00

Key Takeaway: While raw accuracy appears high, a naïve baseline fails completely at detecting any actual bankrupt firms. F1-Score, Precision-Recall curves, and ROC-AUC are required to evaluate model effectiveness.

# Tech Stack & Dependencies
* Language: Python 3.10+

* Data Manipulation: pandas, numpy

* Machine Learning: scikit-learn, imbalanced-learn

* Visualization: matplotlib, ipywidgets

* File Operations: gzip, pathlib, json, pickle
