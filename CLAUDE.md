# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Click-Through Rate (CTR) prediction project for UCLA Stats C161. Predicts whether users will click on advertisements using Logistic Regression and Decision Trees.

**Dataset**: 7.6M training rows, 976K test rows merged from user and ad interaction data.
**Target**: `label` (binary: -1/1 or 0/1 for click prediction)
**Performance**: Models achieve AUC ~0.75-0.79 with severe class imbalance (98.5% negative class)

## Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux

# Install dependencies (if requirements.txt exists)
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Main analysis notebooks
# - logistic_regression_analysis.ipynb: Primary modeling notebook
# - ctr_pred_pre-processing.ipynb: Data preprocessing exploration
```

### Data Preprocessing
```bash
# Generate preprocessed pickle files
python data_preprocessing.py

# Output: train_encoded.pkl, test_encoded.pkl
```

## Architecture

### Data Pipeline

1. **Raw Data** → 4 CSV files:
   - `train_data_ads.csv` / `test_data_ads.csv`: User demographics
   - `train_data_feeds.csv` / `test_data_feeds.csv`: Ad interactions

2. **Preprocessing** (`data_preprocessing.py`):
   - Merges user and ad data on `user_id`
   - Creates aggregated features: `feeds_imps`, `feeds_clicks`, `feeds_ctr`
   - Handles label conversion: -1/1 → 0/1
   - Drops `adv_id` to avoid ad-specific random effects
   - **Output**: `train_encoded.pkl`, `test_encoded.pkl`

3. **Feature Engineering**:
   - **One-hot encoding** (low cardinality ≤10): `gender`, `net_type`, `creat_type_cd`, `inter_type_cd`, `series_group`
   - **Target encoding** (high cardinality >50): `task_id`, `device_name`, `slot_id`, `city`, `adv_prim_id`, `device_size`
   - **Target encoding** (medium 11-50): `residence`, `series_dev`, `emui_dev`, `hispace_app_tags`, `app_second_class`, `spread_app_id`
   - **Ordinal (keep numeric)**: `age`, `city_rank`
   - **Interaction features**: `engagement_by_slot`, `bandwidth_by_creative` (or `bandwidth_by_slot`)

4. **Modeling Workflow**:
   - Load pickles → Split train/validation → Scale features → Train models → Evaluate

### Critical Design Patterns

#### Safe Column Selection
```python
def safe_keep(df, cols, name):
    """Safely keep columns, reporting missing ones"""
    keep = [c for c in cols if c in df.columns]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"[{name}] missing in the dataset: {miss}")
    return keep
```
Use this pattern to avoid KeyErrors when column sets differ between train/test.

#### Target Encoding (Out-of-Fold)
```python
def target_encode_oof(train_df, test_df, col, y_col='label', n_splits=5):
    """
    OOF target encoding to prevent leakage:
    - Train: encode each fold using means from other folds
    - Test: encode using full training mean
    """
```
**Critical**: Always use OOF encoding for high-cardinality features to prevent data leakage.

#### Train/Test Alignment for One-Hot Encoding
```python
# Ensure test has same columns as train (fill missing with 0)
dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)
```
**Critical**: Test set may have missing categories. Always reindex to match training columns.

### Class Imbalance Handling

**Two approaches implemented**:

1. **Downsampling**: 590K balanced samples (84.7% / 15.3%)
   - Higher precision (12%), lower recall (8%)
   - Faster training

2. **Class Weighting** (`class_weight='balanced'`): Full 6.14M samples
   - Lower precision (3-4%), higher recall (67-74%)
   - Better for catching rare clicks
   - 63x weight on positive class

**Threshold tuning**: Default 0.5 is suboptimal. Search range [0.01, 0.5] to maximize F1.

### Feature Scaling

**Always standardize before logistic regression**:
```python
scaler = StandardScaler(with_mean=False)  # Safe for sparse-like inputs
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_val_scaled = scaler.transform(X_val)          # Transform using train stats
```

### Model Evaluation

**Required metrics** (always report all):
- **AUC-ROC**: Primary metric (~0.75-0.79 typical)
- **Accuracy**: Can be misleading with imbalance
- **Precision**: P(true click | predicted click)
- **Recall**: P(predicted click | true click) - prioritize for CTR
- **F1**: Harmonic mean of precision/recall
- **Log-Loss**: Probabilistic calibration

### Memory Management

**Large datasets require careful handling**:
```python
import gc
del train_user, test_user
gc.collect()
```
Use after dropping large DataFrames (7.6M rows).

## Key Constraints

### Data Leakage Prevention
- **NEVER** use test set for computing statistics (target encoding, scaling, imputation)
- **ALWAYS** fit transformations on training data only
- Use OOF encoding for high-cardinality features
- Train/validation split **before** any feature engineering

### Train/Test Consistency
- All transformations applied **identically** to train and test
- Use `.reindex()` to align one-hot encoded columns
- Fill missing test categories with 0 or global train mean

### Features to Ignore
**Object columns** (initially skip, contain list-like data):
- `ad_click_list_v001/002/003`
- `ad_close_list_v001/002/003`
- `u_newsCatInterestsST`

**Constant features** (drop entirely):
- `site_id` (single unique value)

### Code Style (from .cursorrules)
- **Snake_case** for variables: `train_merged` not `df1`
- **One task at a time**: Complete each step fully before moving on
- **Simple and readable**: Avoid clever one-liners that sacrifice clarity
- **Explicit over implicit**: Self-documenting code preferred
- **Educational focus**: Code should teach the user, not just work

### Notebook Documentation
- **Markdown before code**: Explain "what" and "why" before each section
- **Concise in notebook**: Educational explanations in chat, not in markdown cells
- **Track decisions**: Document feature engineering rationale
- **Interpret results**: Explain coefficients, metrics, comparisons

## Hyperparameter Tuning

**Regularization (C parameter)**:
```python
# Grid search over C values [0.1, 0.5, 1.0, 2.0, 5.0]
# Minimize log-loss, tiebreaker: maximize AUC
# Use 3-fold CV (not 5) for speed
# For class-weighted models: subsample to 1M during grid search, refit on full 6M
```

**Threshold tuning**:
```python
# Search range: [0.05, 0.51] with 0.01 steps
# Optimize for F1 score (balances precision/recall)
# Or target specific recall (e.g., 0.70) for click-focused applications
```

## File Organization

**Pickle files** (load these, don't regenerate):
- `train_encoded.pkl` (6.14M rows, ~2GB)
- `test_encoded.pkl` (1.54M rows, ~262MB)

**Notebooks**:
- `logistic_regression_analysis.ipynb`: Primary analysis (refitted models, threshold tuning, comparison)
- `ctr_pred_pre-processing.ipynb`: Original preprocessing exploration
- `lasted-version 2.ipynb`: Older iteration (likely deprecated)

**Python scripts**:
- `data_preprocessing.py`: Canonical preprocessing pipeline (mirrors notebook)

**Click_Through_Pred/** subdirectory appears to be a duplicate/fork of the main project.

## Common Pitfalls (Now Fixed in data_preprocessing.py)

✅ **All pitfalls below have been fixed** in the updated `data_preprocessing.py` script:

1. ✅ **Label conversion**: Labels are now automatically converted to 0/1 before any encoding (previously -1/1)
2. ✅ **Target encoding statistics**: Now uses 0/1 labels (previously used -1/1, giving wrong CTR estimates)
3. ✅ **One-hot encoding dtype**: Explicitly cast to int to avoid mixed types
4. ✅ **Train/test column alignment**: Uses `.reindex()` with `fill_value=0` for all one-hot encodings
5. ✅ **Non-numeric column detection**: Final validation step warns about object columns
6. ✅ **Test label handling**: Safely handles NaN labels in unlabeled test sets
7. ✅ **Validation checks**: Automated checks for label range, column alignment, and data types

**When loading pickles**: Labels are already 0/1, no manual conversion needed (unlike old workflow).

**Remaining considerations**:
- Drop non-numeric columns before modeling: `X.select_dtypes(include=[np.number])`
- Use `gc.collect()` after large operations for memory management
- Scale **after** loading pickles but **before** modeling
