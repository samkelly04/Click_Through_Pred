# CTR Prediction Project

**UCLA Stats C161 - Final Project**  
**Topic**: Click-Through Rate (CTR) Prediction using Machine Learning

---

## Project Overview

This project develops machine learning models to predict the probability that a user will click on an advertisement, given features of both the user and the advertisement.

**Core Question**: P(click | advertisement features, user features)

**Target Variable**: `label` - binary click indicator (0 = no click, 1 = click)

### Key Results
- **Best Model**: LightGBM with CVAE-augmented data
- **AUC-ROC**: ~0.75-0.79
- **Challenge**: Severe class imbalance (98.5% negative class)

---

## Project Structure

```
f_project/
├── README.md                    # This file
├── CLAUDE.md                    # AI agent instructions
│
├── data/                        # Data directory
│   ├── raw/                     # Original CSV files
│   │   ├── train_data_ads.csv
│   │   ├── train_data_feeds.csv
│   │   ├── test_data_ads.csv
│   │   └── test_data_feeds.csv
│   ├── processed/               # Preprocessed pickle files
│   │   ├── train_encoded.pkl
│   │   └── test_encoded.pkl
│   └── synthetic/               # Generated synthetic data
│       ├── ctgan_synthetic_data.pkl
│       ├── conditional_tvae_synthetic.pkl
│       └── conditional_tvae_synthetic.csv
│
├── src/                         # Python source code
│   ├── data_preprocessing.py    # Data preprocessing pipeline
│   ├── logistic_regression_with_synthetic.py
│   ├── logistic_regression_with_synthetic_optimized.py
│   ├── lightgbm_analysis.py
│   ├── lightgbm_with_synthetic.py
│   ├── train_cvae_models.py
│   ├── conditional_vae_demo.py
│   ├── conditional_vae_production.py
│   ├── generate_ctgan_data.py
│   ├── analyze_cvae_data_quality.py
│   ├── compare_models_visualizations.py
│   ├── lr_comparison_visualizations.py
│   ├── visualize_cvae_results.py
│   ├── test_evaluation.py
│   └── test_visualizations.py
│
├── notebooks/                   # Jupyter notebooks
│   ├── logistic_regression_analysis.ipynb  # Primary analysis
│   ├── ctr_pred_pre-processing.ipynb       # Preprocessing exploration
│   └── lasted-version 2.ipynb              # Legacy notebook
│
├── models/                      # Trained model files
│   ├── lr_baseline_model.pkl
│   ├── lr_augmented_model.pkl
│   ├── lr_cvae_baseline_model.pkl
│   ├── lr_cvae_augmented_model.pkl
│   ├── lightgbm_augmented_model.pkl
│   ├── lightgbm_ctgan_model.pkl
│   ├── lgbm_cvae_baseline_model.pkl
│   ├── lgbm_cvae_augmented_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── conditional_tvae_model.pkl
│
├── figures/                     # Generated visualizations
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── comparison/              # Model comparison plots
│   ├── lr_comparison/           # Logistic regression comparisons
│   ├── ctgan_comparison/        # CTGAN model comparisons
│   ├── ctgan_lgbm/              # CTGAN + LightGBM results
│   ├── cvae_comparison/         # CVAE model comparisons
│   ├── cvae_models/             # CVAE-specific plots
│   ├── cvae_quality/            # CVAE data quality analysis
│   └── test/                    # Test set analysis
│
├── results/                     # Model outputs
│   ├── metrics/                 # Performance metrics CSVs
│   └── predictions/             # Model predictions
│
└── docs/                        # Documentation
    ├── CVAE_ANALYSIS.md
    ├── CVAE_DATA_QUALITY_ANALYSIS.md
    ├── GRADIENT_BOOSTING_RESULTS.md
    ├── LIGHTGBM_DETAILED_ANALYSIS.md
    ├── LIGHTGBM_TEST_PREDICTIONS.md
    ├── PREPROCESSING_FIXES.md
    ├── SYNTHETIC_DATA_COMPARISON.md
    └── logistic_regression_analysis_summary.txt
```

---

## Dataset

### Source Data
- **Training**: 7.6M rows, 38 columns
- **Test**: 976K rows, 38 columns
- Each row represents a user-ad interaction

### Data Files
| File | Description |
|------|-------------|
| `train_data_ads.csv` / `test_data_ads.csv` | User demographic data |
| `train_data_feeds.csv` / `test_data_feeds.csv` | Advertisement interaction data |
| `train_encoded.pkl` / `test_encoded.pkl` | Preprocessed, encoded data |

---

## Models Implemented

### 1. Logistic Regression
- Baseline model with class weighting
- Augmented with synthetic data (CTGAN, CVAE)
- Threshold optimization for F1 score

### 2. LightGBM (Gradient Boosting)
- Best performing model
- Trained on original and synthetic-augmented data
- Feature importance analysis

### 3. Synthetic Data Generation
- **CTGAN**: Conditional Tabular GAN
- **CVAE**: Conditional Variational Autoencoder
- Used to address class imbalance

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Overall ranking quality (primary metric) |
| **Accuracy** | Proportion of correct predictions |
| **Precision** | P(true click \| predicted click) |
| **Recall** | P(predicted click \| true click) |
| **F1 Score** | Harmonic mean of precision/recall |
| **Log-Loss** | Probabilistic calibration |

---

## Quick Start

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm jupyter
```

### Data Preprocessing
```bash
# Generate preprocessed pickle files
python src/data_preprocessing.py

# Output: data/processed/train_encoded.pkl, data/processed/test_encoded.pkl
```

### Run Analysis
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/logistic_regression_analysis.ipynb
```

---

## Key Findings

1. **Class Imbalance**: 98.5% negative class requires special handling
2. **Synthetic Data**: CVAE augmentation improves recall
3. **Best Model**: LightGBM achieves AUC ~0.79
4. **Threshold Tuning**: Optimal threshold << 0.5 due to imbalance

---

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- jupyter

---

## Authors

UCLA Stats C161 - Fall 2024

---

**Last Updated**: December 2024  
