# CTR Prediction Project

**UCLA Stats C161 - Midterm Project**  
**Topic**: Click-Through Rate (CTR) Prediction

---

## Project Overview

### Goal
Develop machine learning models (primarily Logistic Regression and Decision Trees) to predict the probability that a user will click on an advertisement given the features of both the user and the advertisement.

**Core Question**: P(click | advertisement features, user features)

**Target Variable**: `label` - binary click indicator (0 = no click, 1 = click)

---

## Data Architecture

### Source Datasets
- **`train_data_ads.csv` / `test_data_ads.csv`**: User-level data
- **`train_data_feeds.csv` / `test_data_feeds.csv`**: Advertisement interaction data

### Merged Dataset Strategy
The datasets are merged so that each row represents a user-ad interaction, combining:
- User demographic features (age, gender, residence, device info)
- Advertisement features (ad campaign ID, creative type, placement info)
- User behavior features (aggregated from feeds data: impressions, clicks, CTR)

**Final Merged Datasets**:
- **Training**: `train_merged` (7.6M rows, 38 columns)
- **Test**: `test_merged` (976K rows, 38 columns)

---

## File Structure

```
Project Root/
├── README.md                       # This file - project overview
├── .cursorrules                    # AI agent behavior instructions
├── ctr-prediction.ipynb           # Main notebook: all analysis, outputs, documentation
├── data_preprocessing.py           # Clean preprocessing script
├── train_merged.pkl               # Saved processed training data
├── test_merged.pkl                # Saved processed test data
├── train_data_ads.csv
├── train_data_feeds.csv
├── test_data_ads.csv
├── test_data_feeds.csv
└── [future modeling notebooks]   # Will load .pkl files for modeling
```

---

## Workflow

### 1. Data Preprocessing (`data_preprocessing.py`)
- Loads and merges the ads and feeds datasets
- Performs data cleaning and merging operations
- Saves processed data as `.pkl` files for efficient loading
- Mirrors the preprocessing code from `ctr-prediction.ipynb`

### 2. Main Notebook (`ctr-prediction.ipynb`)
- Contains all analysis, visualizations, and documentation
- Loads `.pkl` files when needed (avoids re-running expensive operations)
- Houses feature engineering and modeling workflows
- Includes markdown cells explaining the statistical and technical approaches

### 3. Future Modeling Notebooks
- Cleanly load `train_merged.pkl` and `test_merged.pkl`
- Focus on modeling without re-processing data

---

## Statistical Approach

### Evaluation Strategy
1. **Cross-validation** on training data for hyperparameter tuning
2. **Final evaluation** on held-out test set
3. Report performance for both CV and test

### Required Metrics
Always report:
- **AUC-ROC**: Overall ranking quality
- **Accuracy**: Proportion of correct predictions
- **Precision**: P(predicted click | actual click)
- **Recall**: P(click predicted | actual click)
- **Log-Loss**: Probabilistic accuracy

### Modeling Philosophy
- Start with baseline models before complex approaches
- Use Logistic Regression and Decision Trees (primary models)
- Explore additional methods if time permits
- Document feature importance and statistical interpretation

---

## Feature Engineering

### Categorical Encoding Strategy

Based on cardinality (number of unique values):

| Cardinality | Strategy | Examples |
|-------------|----------|----------|
| ≤ 10 | One-hot encoding | `gender`, `net_type`, `inter_type_cd` |
| 11-50 | Target encoding or grouping | `residence`, `series_dev` |
| > 50 | Target encoding or frequency encoding | `device_name`, `adv_id`, `task_id` |

### Feature Categories

**Low Cardinality (One-Hot Encode)**:
- `gender` (3 unique)
- `age` (8 unique)
- `city_rank` (4 unique)
- `net_type` (6 unique)
- `series_group` (7 unique)
- `creat_type_cd` (9 unique)
- `inter_type_cd` (4 unique)

**High Cardinality (Handle Case-by-Case)**:
- `city` (341 unique)
- `device_name` (256 unique)
- `task_id` (11,209 unique)
- `adv_id` (12,615 unique)

### Object Columns
Initially ignored for modeling:
- `ad_click_list_v001`, `ad_click_list_v002`, `ad_click_list_v003`
- `ad_close_list_v001`, `ad_close_list_v002`, `ad_close_list_v003`
- `u_newsCatInterestsST`

These contain list-like data and will be revisited later if needed.

---

## Libraries

### Core Dependencies
- **pandas**: Data manipulation and merging
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models and evaluation
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **gc**: Garbage collection for memory management

Additional libraries added as needed with clear justification.

---

## Notebook Organization

### Recommended Section Order
1. **Data Loading & Merging**
2. **Exploratory Data Analysis**
3. **Feature Engineering**
4. **Model Development** (Logistic Regression, Decision Trees, etc.)
5. **Model Evaluation & Comparison**
6. **Conclusions**

### Cell Pattern
```python
# 1. Markdown cell: Explain what we're doing and why
# 2. Code cell: Execute the operation
# 3. Markdown cell (if needed): Interpret results
```

---

## Key Principles

- **Simplicity**: Code should be readable and self-explanatory
- **One task at a time**: Complete each step fully before moving to the next
- **Documentation**: Markdown cells explain both technical and statistical perspectives
- **No data leakage**: Never use test set information in training transformations
- **Train/test consistency**: All transformations applied identically to both sets

---

**Last Updated**: December 2024  
**Status**: Preprocessing complete, feature engineering in progress

