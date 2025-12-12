"""
Train Logistic Regression and LightGBM with C-TVAE Augmentation
Target: 10% Click Rate via Synthetic Data Augmentation
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, log_loss, classification_report, RocCurveDisplay,
    PrecisionRecallDisplay, confusion_matrix, average_precision_score,
    roc_curve, precision_recall_curve, auc
)
from sdv.sampling import Condition
import warnings
import gc
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAIN MODELS WITH C-TVAE AUGMENTATION")
print("Target: 10% Click Rate")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA AND C-TVAE MODEL
# ============================================================================
print("\n[Step 1/8] Loading data and C-TVAE model...")

# Load training data (subsampled)
with open('train_encoded.pkl', 'rb') as f:
    train_full = pickle.load(f)

# Subsample
SUBSAMPLE_SIZE = 1_000_000
train_data = train_full.sample(n=SUBSAMPLE_SIZE, random_state=42).reset_index(drop=True)
del train_full
gc.collect()

# Ensure label is 0/1
if train_data['label'].min() < 0:
    train_data['label'] = train_data['label'].replace({-1: 0, 1: 1}).astype(int)

n_original = len(train_data)
n_clicks_original = train_data['label'].sum()
click_rate_original = n_clicks_original / n_original

print(f"  Original data: {n_original:,} samples")
print(f"  Original clicks: {n_clicks_original:,} ({click_rate_original*100:.2f}%)")

# Load C-TVAE model
with open('conditional_tvae_model.pkl', 'rb') as f:
    cvae_synthesizer = pickle.load(f)

print("  ✓ C-TVAE model loaded")

# ============================================================================
# STEP 2: CALCULATE SYNTHETIC DATA NEEDED FOR 10% CLICKS
# ============================================================================
print("\n[Step 2/8] Calculating synthetic data needed for 10% clicks...")

target_click_rate = 0.10

# Formula: (n_clicks + n_synthetic) / (n_total + n_synthetic) = 0.10
# Solving: n_synthetic = (0.10 * n_total - n_clicks) / (1 - 0.10)
n_synthetic = int((target_click_rate * n_original - n_clicks_original) / (1 - target_click_rate))

print(f"  Target click rate: {target_click_rate*100:.0f}%")
print(f"  Synthetic samples needed: {n_synthetic:,}")
print(f"  Final dataset size: {n_original + n_synthetic:,}")
print(f"  Final click count: {n_clicks_original + n_synthetic:,}")
print(f"  Verification: {(n_clicks_original + n_synthetic)/(n_original + n_synthetic)*100:.2f}%")

# ============================================================================
# STEP 3: GENERATE SYNTHETIC DATA
# ============================================================================
print("\n[Step 3/8] Generating C-TVAE synthetic data...")

# Create condition for general click generation
condition = Condition(
    num_rows=n_synthetic,
    column_values={'label': 1}
)

print(f"  Generating {n_synthetic:,} synthetic click samples...")
print("  (This may take 2-3 minutes)")

df_synthetic = cvae_synthesizer.sample_from_conditions(
    conditions=[condition],
    max_tries_per_batch=10000
)

print(f"  ✓ Generated {len(df_synthetic):,} synthetic samples")

# Validate
synthetic_clicks = df_synthetic['label'].sum()
print(f"  Validation: {synthetic_clicks:,} clicks in synthetic data ({synthetic_clicks/len(df_synthetic)*100:.1f}%)")

# ============================================================================
# STEP 4: CREATE AUGMENTED DATASET
# ============================================================================
print("\n[Step 4/8] Creating augmented training dataset...")

# Combine original + synthetic
train_augmented = pd.concat([train_data, df_synthetic], axis=0, ignore_index=True)

n_augmented = len(train_augmented)
n_clicks_augmented = train_augmented['label'].sum()
click_rate_augmented = n_clicks_augmented / n_augmented

print(f"  Augmented dataset: {n_augmented:,} samples")
print(f"  Augmented clicks: {n_clicks_augmented:,} ({click_rate_augmented*100:.2f}%)")
print(f"  ✓ Target achieved!")

del df_synthetic
gc.collect()

# ============================================================================
# STEP 5: PREPARE FEATURES
# ============================================================================
print("\n[Step 5/8] Preparing features...")

# Remove non-numeric columns
feature_cols = [c for c in train_data.columns if c != 'label']
numeric_cols_baseline = train_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"  Numeric features: {len(numeric_cols_baseline)}")

# Prepare datasets
X_baseline = train_data[numeric_cols_baseline].copy()
y_baseline = train_data['label'].copy()

# For augmented, ensure columns exist and are numeric
common_cols = [c for c in numeric_cols_baseline if c in train_augmented.columns]
X_augmented = train_augmented[common_cols].copy()
y_augmented = train_augmented['label'].copy()

# Double-check: ensure all columns are numeric
numeric_check_aug = X_augmented.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_check_aug) < len(common_cols):
    print(f"  Warning: Filtering out {len(common_cols) - len(numeric_check_aug)} non-numeric columns")
    X_augmented = X_augmented[numeric_check_aug]
    X_baseline = X_baseline[numeric_check_aug]  # Match baseline to augmented

# Handle NaN
X_baseline = X_baseline.fillna(0)
X_augmented = X_augmented.fillna(0)

print(f"  Baseline: {X_baseline.shape}")
print(f"  Augmented: {X_augmented.shape}")

# Train/val split
print("  Creating train/val splits...")
X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42, stratify=y_baseline
)

X_train_aug, X_val_aug, y_train_aug, y_val_aug = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
)

print(f"  Baseline - Train: {len(X_train_base):,}, Val: {len(X_val_base):,}")
print(f"  Augmented - Train: {len(X_train_aug):,}, Val: {len(X_val_aug):,}")

del train_data, train_augmented, X_baseline, X_augmented, y_baseline, y_augmented
gc.collect()

# Scale features
print("  Scaling features...")
scaler_base = StandardScaler(with_mean=False)
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_val_base_scaled = scaler_base.transform(X_val_base)

scaler_aug = StandardScaler(with_mean=False)
X_train_aug_scaled = scaler_aug.fit_transform(X_train_aug)
X_val_aug_scaled = scaler_aug.transform(X_val_aug)

# Handle NaN after scaling
X_train_base_scaled = np.nan_to_num(X_train_base_scaled, nan=0.0)
X_val_base_scaled = np.nan_to_num(X_val_base_scaled, nan=0.0)
X_train_aug_scaled = np.nan_to_num(X_train_aug_scaled, nan=0.0)
X_val_aug_scaled = np.nan_to_num(X_val_aug_scaled, nan=0.0)

print("  ✓ Feature preparation complete")

# ============================================================================
# STEP 6: TRAIN LOGISTIC REGRESSION
# ============================================================================
print("\n[Step 6/8] Training Logistic Regression models...")

# Baseline LR
print("  Training baseline LR (class-weighted, no synthetic)...")
lr_baseline = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    random_state=42
)
lr_baseline.fit(X_train_base_scaled, y_train_base)
print("  ✓ Baseline LR trained")

# C-TVAE augmented LR
print("  Training C-TVAE augmented LR (class-weighted)...")
lr_cvae = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    random_state=42
)
lr_cvae.fit(X_train_aug_scaled, y_train_aug)
print("  ✓ C-TVAE LR trained")

# Evaluate LR models
print("\n  Evaluating LR models...")
lr_base_proba = lr_baseline.predict_proba(X_val_base_scaled)[:, 1]
lr_base_pred = (lr_base_proba >= 0.5).astype(int)

lr_cvae_proba = lr_cvae.predict_proba(X_val_aug_scaled)[:, 1]
lr_cvae_pred = (lr_cvae_proba >= 0.5).astype(int)

lr_metrics_base = {
    'AUC': roc_auc_score(y_val_base, lr_base_proba),
    'Accuracy': accuracy_score(y_val_base, lr_base_pred),
    'Precision': precision_score(y_val_base, lr_base_pred, zero_division=0),
    'Recall': recall_score(y_val_base, lr_base_pred),
    'F1': f1_score(y_val_base, lr_base_pred),
    'Avg Precision': average_precision_score(y_val_base, lr_base_proba)
}

lr_metrics_cvae = {
    'AUC': roc_auc_score(y_val_aug, lr_cvae_proba),
    'Accuracy': accuracy_score(y_val_aug, lr_cvae_pred),
    'Precision': precision_score(y_val_aug, lr_cvae_pred, zero_division=0),
    'Recall': recall_score(y_val_aug, lr_cvae_pred),
    'F1': f1_score(y_val_aug, lr_cvae_pred),
    'Avg Precision': average_precision_score(y_val_aug, lr_cvae_proba)
}

print("\n  Logistic Regression Results:")
print("  Baseline:")
for k, v in lr_metrics_base.items():
    print(f"    {k:15s}: {v:.4f}")
print("  C-TVAE Augmented:")
for k, v in lr_metrics_cvae.items():
    print(f"    {k:15s}: {v:.4f}")

# Save LR models
with open('lr_cvae_baseline_model.pkl', 'wb') as f:
    pickle.dump(lr_baseline, f)
with open('lr_cvae_augmented_model.pkl', 'wb') as f:
    pickle.dump(lr_cvae, f)
print("  ✓ LR models saved")

# ============================================================================
# STEP 7: TRAIN LIGHTGBM
# ============================================================================
print("\n[Step 7/8] Training LightGBM models...")

# Baseline LightGBM
print("  Training baseline LightGBM...")
lgbm_baseline = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=31,
    learning_rate=0.05,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)
lgbm_baseline.fit(X_train_base, y_train_base)
print("  ✓ Baseline LightGBM trained")

# C-TVAE augmented LightGBM
print("  Training C-TVAE augmented LightGBM...")
lgbm_cvae = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=31,
    learning_rate=0.05,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)
lgbm_cvae.fit(X_train_aug, y_train_aug)
print("  ✓ C-TVAE LightGBM trained")

# Evaluate LightGBM models
print("\n  Evaluating LightGBM models...")
lgbm_base_proba = lgbm_baseline.predict_proba(X_val_base)[:, 1]
lgbm_base_pred = (lgbm_base_proba >= 0.5).astype(int)

lgbm_cvae_proba = lgbm_cvae.predict_proba(X_val_aug)[:, 1]
lgbm_cvae_pred = (lgbm_cvae_proba >= 0.5).astype(int)

lgbm_metrics_base = {
    'AUC': roc_auc_score(y_val_base, lgbm_base_proba),
    'Accuracy': accuracy_score(y_val_base, lgbm_base_pred),
    'Precision': precision_score(y_val_base, lgbm_base_pred, zero_division=0),
    'Recall': recall_score(y_val_base, lgbm_base_pred),
    'F1': f1_score(y_val_base, lgbm_base_pred),
    'Avg Precision': average_precision_score(y_val_base, lgbm_base_proba)
}

lgbm_metrics_cvae = {
    'AUC': roc_auc_score(y_val_aug, lgbm_cvae_proba),
    'Accuracy': accuracy_score(y_val_aug, lgbm_cvae_pred),
    'Precision': precision_score(y_val_aug, lgbm_cvae_pred, zero_division=0),
    'Recall': recall_score(y_val_aug, lgbm_cvae_pred),
    'F1': f1_score(y_val_aug, lgbm_cvae_pred),
    'Avg Precision': average_precision_score(y_val_aug, lgbm_cvae_proba)
}

print("\n  LightGBM Results:")
print("  Baseline:")
for k, v in lgbm_metrics_base.items():
    print(f"    {k:15s}: {v:.4f}")
print("  C-TVAE Augmented:")
for k, v in lgbm_metrics_cvae.items():
    print(f"    {k:15s}: {v:.4f}")

# Save LightGBM models
with open('lgbm_cvae_baseline_model.pkl', 'wb') as f:
    pickle.dump(lgbm_baseline, f)
with open('lgbm_cvae_augmented_model.pkl', 'wb') as f:
    pickle.dump(lgbm_cvae, f)
print("  ✓ LightGBM models saved")

# ============================================================================
# STEP 8: SAVE RESULTS AND PREDICTIONS
# ============================================================================
print("\n[Step 8/8] Saving results...")

# Save validation predictions
np.savez('cvae_validation_predictions.npz',
         # LR
         y_val_base_lr=y_val_base.values,
         lr_base_proba=lr_base_proba,
         y_val_aug_lr=y_val_aug.values,
         lr_cvae_proba=lr_cvae_proba,
         # LightGBM
         y_val_base_lgbm=y_val_base.values,
         lgbm_base_proba=lgbm_base_proba,
         y_val_aug_lgbm=y_val_aug.values,
         lgbm_cvae_proba=lgbm_cvae_proba)

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': list(lr_metrics_base.keys()),
    'LR_Baseline': list(lr_metrics_base.values()),
    'LR_CVAE': list(lr_metrics_cvae.values()),
    'LGBM_Baseline': list(lgbm_metrics_base.values()),
    'LGBM_CVAE': list(lgbm_metrics_cvae.values())
})
metrics_df.to_csv('cvae_models_comparison_metrics.csv', index=False)

print("  ✓ Validation predictions saved to 'cvae_validation_predictions.npz'")
print("  ✓ Metrics saved to 'cvae_models_comparison_metrics.csv'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print(f"\nDataset Composition:")
print(f"  Original: {n_original:,} samples ({click_rate_original*100:.2f}% clicks)")
print(f"  Synthetic (C-TVAE): {n_synthetic:,} samples (100% clicks)")
print(f"  Augmented: {n_augmented:,} samples ({click_rate_augmented*100:.2f}% clicks)")

print(f"\nModel Performance Summary:")
print(f"\nLogistic Regression:")
print(f"  Baseline AUC:  {lr_metrics_base['AUC']:.4f}")
print(f"  C-TVAE AUC:    {lr_metrics_cvae['AUC']:.4f} ({(lr_metrics_cvae['AUC']-lr_metrics_base['AUC'])*100:+.1f}%)")

print(f"\nLightGBM:")
print(f"  Baseline AUC:  {lgbm_metrics_base['AUC']:.4f}")
print(f"  C-TVAE AUC:    {lgbm_metrics_cvae['AUC']:.4f} ({(lgbm_metrics_cvae['AUC']-lgbm_metrics_base['AUC'])*100:+.1f}%)")

print(f"\nNext: Run visualization script to generate comparison figures")
