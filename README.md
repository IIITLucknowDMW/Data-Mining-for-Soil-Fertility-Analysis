# Data Mining Project: Soil Fertility Analysis

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Algorithms Implemented](#algorithms-implemented)
- [Results](#results)
- [Module Documentation](#module-documentation)

---

## Overview

A comprehensive data mining project implementing **supervised** and **unsupervised learning algorithms** from scratch for soil fertility analysis. This project demonstrates complete data mining pipelines including preprocessing, classification, clustering, and evaluation with a modular, production-ready architecture.

**Key Highlights:**
- All core ML algorithms implemented from scratch (Decision Tree, K-Means)
- Modular, reusable codebase with clean architecture
- Complete data preprocessing pipeline
- Comprehensive evaluation metrics and visualization

---

## Project Structure

```
DMW-project/
├── src/                          # Modular source code
│   ├── __init__.py
│   ├── preprocessing.py          # Data preprocessing & cleaning
│   ├── classifiers.py            # Decision Tree classifier
│   ├── clustering.py             # K-Means clustering
│   ├── metrics.py                # Evaluation metrics
│   └── utils.py                  # Utility functions
├── datasets/                     # Dataset files
│   ├── Dataset1.csv              # Classification dataset
│   └── Dataset2.csv              # Clustering dataset
├── notebooks/                    # Original Jupyter notebook (reference)
│   └── SoilFertility.ipynb
├── main.py                       # Main pipeline entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## Features

### 1. Data Preprocessing Pipeline
- Missing value handling (mode/mean imputation)
- Outlier detection & treatment (linear regression, discretization)
- Correlation-based dimension reduction
- Normalization (Min-Max, Z-score)
- Duplicate removal

### 2. Classification (Decision Tree)
- Gini index and Entropy-based splitting
- Pre-pruning and post-pruning techniques
- Configurable hyperparameters
- Verbose prediction paths

### 3. Clustering (K-Means)
- K-means++ initialization
- Multiple distance metrics (Cosine, Manhattan, Euclidean)
- Convergence optimization
- Cluster prediction for new samples

### 4. Evaluation Metrics
- **Classification**: Confusion matrix, Accuracy, Precision, Recall, F1-Score, Specificity
- **Clustering**: Silhouette score with distance analysis
- **Distance**: Cosine and Minkowski implementations

---

## 📊 Dataset

The project uses two soil fertility datasets:

### Dataset1.csv (Classification)
Used for supervised learning tasks with the following features:
- **N**: Nitrogen content
- **P**: Phosphorus content
- **K**: Potassium content
- **pH**: Soil pH level
- **EC**: Electrical conductivity
- **OC**: Organic carbon
- **S**: Sulphur content
- **Zn**: Zinc content
- **Fe**: Iron content
- **Cu**: Copper content
- **Mn**: Manganese content
- **B**: Boron content
- **OM**: Organic matter
- **Fertility**: Target variable (fertility class: 0, 1, 2)

### Dataset2.csv (Clustering)
Used for unsupervised learning tasks. Contains similar soil features for clustering analysis.

---

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip

### Quick Setup

```bash
# Navigate to project directory
cd DMW-project

# Create and activate virtual environment
python -m venv dmw_env
source dmw_env/bin/activate  # macOS/Linux
# OR
dmw_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

### Dependencies
- numpy, pandas - Data manipulation
- matplotlib, seaborn - Visualization
- scikit-learn - Preprocessing utilities only
- scipy - Scientific computing

---

## Results

### Classification Performance
- **Algorithm**: Decision Tree (Gini index, max_depth=5)
- **Accuracy**: ~90% on test data
- **Precision/Recall**: >0.90 for major fertility classes
- **F1-Score**: 0.68 (global average)

### Clustering Performance
- **Algorithm**: K-Means (k=3, Manhattan distance, K-means++ init)
- **Silhouette Score**: 0.33 (acceptable cluster separation)
- **Convergence**: Achieved within 10,000 iterations

### Key Insights
- Preprocessing reduced dimensions from 14 to 13 features
- Decision Tree effectively classifies soil fertility with high accuracy
- K-Means successfully identifies 3 distinct soil fertility clusters

---

## Module Documentation

### Core Modules

**`src/preprocessing.py`** - Data preprocessing pipeline
- Missing value handling, outlier treatment, normalization, dimension reduction

**`src/classifiers.py`** - Classification algorithms
- `DtClassifier`: Decision Tree with Gini/Entropy splitting

**`src/clustering.py`** - Clustering algorithms
- `K_MEANS`: K-Means with multiple distance metrics
- `silhouette_score_custom()`: Clustering evaluation

**`src/metrics.py`** - Evaluation metrics
- Classification metrics (accuracy, precision, recall, F1)
- Distance metrics (Cosine, Minkowski)
- Confusion matrix visualization

**`src/utils.py`** - Utility functions
- Dataset loading and train-test splitting

---

## Algorithms Implemented

### Implemented from Scratch

✅ **Decision Tree Classifier**
- Gini index and Entropy splitting criteria
- Pre-pruning and post-pruning
- Recursive tree building
- Custom prediction traversal

✅ **K-Means Clustering**
- K-means++ initialization
- Multiple distance metrics (Cosine, Manhattan, Euclidean)
- Iterative centroid updates
- Convergence detection

✅ **Evaluation Metrics**
- Confusion matrix, accuracy, precision, recall, F1-score, specificity
- Silhouette score for clustering
- Custom distance implementations

✅ **Data Preprocessing**
- Missing value imputation
- Outlier detection and treatment
- Feature normalization and standardization
- Correlation-based dimension reduction

---

## Technologies Used

- **Python 3.7+**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **scikit-learn**: Preprocessing utilities only (LinearRegression for outliers)

**Note**: Core ML algorithms (Decision Tree, K-Means, evaluation metrics) implemented from scratch without sklearn.
