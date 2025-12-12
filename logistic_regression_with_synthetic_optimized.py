"""
Logistic Regression with CTGAN Synthetic Data Augmentation (Memory Optimized)
Goal: Augment training data to achieve 15% total clicks using CTGAN synthetic data
Strategy: Use 2M subsample of training data to manage memory
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, log_loss, classification_report, RocCurveDisplay,
    PrecisionRecallDisplay, confusion_matrix, average_precision_score
)
import warnings
import gc
warnings.filterwarnings('ignore')

print("=" * 80)
print("LOGISTIC REGRESSION WITH CTGAN SYNTHETIC DATA (OPTIMIZED)")
print("Target: 15% Total Clicks")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA (WITH SUBSAMPLING FOR MEMORY EFFICIENCY)
# ============================================================================
print("\n[1/7] Loading data...")

# Load full datasets
with open('train_encoded.pkl', 'rb') as f:
    train_full = pickle.load(f)
with open('test_encoded.pkl', 'rb') as f:
    test_encoded = pickle.load(f)
with open('ctgan_synthetic_data.pkl', 'rb') as f:
    synthetic_data = pickle.load(f)

print(f"  Full training data: {len(train_full):,} samples")
print(f"  Synthetic data available: {len(synthetic_data):,} samples")
print(f"  Test data: {len(test_encoded):,} samples")

# Ensure label is 0/1
if train_full['label'].min() < 0:
    train_full['label'] = train_full['label'].replace({-1: 0, 1: 1}).astype(int)

# Get class distribution
original_click_rate = train_full['label'].mean()
n_clicks_full = train_full['label'].sum()
n_noclicks_full = len(train_full) - n_clicks_full

print(f"\n  Full training class distribution:")
print(f"    No-click: {n_noclicks_full:,} ({(1-original_click_rate)*100:.2f}%)")
print(f"    Click: {n_clicks_full:,} ({original_click_rate*100:.2f}%)")

# ============================================================================
# 2. SUBSAMPLE TRAINING DATA (STRATIFIED)
# ============================================================================
print("\n[2/7] Subsampling training data for memory efficiency...")

# Subsample to 2M samples (stratified to maintain class balance)
SUBSAMPLE_SIZE = 2_000_000
if len(train_full) > SUBSAMPLE_SIZE:
    train_encoded = train_full.sample(n=SUBSAMPLE_SIZE, random_state=42,
                                      weights=train_full['label'].map({0: 1, 1: 1})).reset_index(drop=True)
    print(f"  Subsampled training data: {len(train_encoded):,} samples")
else:
    train_encoded = train_full.copy()

del train_full
gc.collect()

# Current class distribution in subsample
n_total_original = len(train_encoded)
n_clicks_original = train_encoded['label'].sum()
n_noclicks_original = n_total_original - n_clicks_original
click_rate_original = n_clicks_original / n_total_original

print(f"  Subsample class distribution:")
print(f"    No-click: {n_noclicks_original:,} ({(1-click_rate_original)*100:.2f}%)")
print(f"    Click: {n_clicks_original:,} ({click_rate_original*100:.2f}%)")

# ============================================================================
# 3. CALCULATE SYNTHETIC DATA NEEDED FOR 15% CLICKS
# ============================================================================
print("\n[3/7] Calculating synthetic data augmentation...")

target_click_rate = 0.15
n_synthetic_needed = int((target_click_rate * n_total_original - n_clicks_original) / (1 - target_click_rate))

print(f"  Synthetic samples needed for {target_click_rate*100:.0f}% clicks: {n_synthetic_needed:,}")
print(f"  Synthetic samples available: {len(synthetic_data):,}")

# If we need more synthetic samples than available, resample with replacement
if n_synthetic_needed > len(synthetic_data):
    print(f"  WARNING: Not enough unique synthetic samples. Will resample with replacement.")
    n_resamples_needed = int(np.ceil(n_synthetic_needed / len(synthetic_data)))
    print(f"  Resampling synthetic data ~{n_resamples_needed}x to reach target")

n_total_augmented = n_total_original + n_synthetic_needed
n_clicks_augmented = n_clicks_original + n_synthetic_needed
click_rate_augmented = n_clicks_augmented / n_total_augmented

print(f"\n  Augmented class distribution:")
print(f"    No-click: {n_noclicks_original:,} ({(1-click_rate_augmented)*100:.2f}%)")
print(f"    Click: {n_clicks_augmented:,} ({click_rate_augmented*100:.2f}%)")
print(f"  Achieved click rate: {click_rate_augmented*100:.2f}%")

# ============================================================================
# 4. CREATE AUGMENTED TRAINING SET
# ============================================================================
print("\n[4/7] Creating augmented training set...")

# Sample synthetic data (with replacement if needed)
replace = n_synthetic_needed > len(synthetic_data)
synthetic_sample = synthetic_data.sample(n=n_synthetic_needed, random_state=42, replace=replace).reset_index(drop=True)

print(f"  Sampled {len(synthetic_sample):,} synthetic samples (replace={replace})")

# Combine original + synthetic
train_augmented = pd.concat([train_encoded, synthetic_sample], axis=0, ignore_index=True)

print(f"  Combined training data: {len(train_augmented):,} samples")
print(f"  Verification - Click rate: {train_augmented['label'].mean()*100:.2f}%")

del synthetic_data, synthetic_sample
gc.collect()

# ============================================================================
# 5. PREPARE FEATURES FOR MODELING
# ============================================================================
print("\n[5/7] Preparing features...")

# Separate features/target and drop non-numeric columns
feature_cols = [c for c in train_encoded.columns if c != 'label']

# Baseline model (no synthetic data)
X_full_baseline = train_encoded[feature_cols].copy()
non_numeric_cols = X_full_baseline.select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric_cols) > 0:
    print(f"  Dropping {len(non_numeric_cols)} non-numeric columns")
    X_baseline = X_full_baseline.drop(columns=non_numeric_cols)
else:
    X_baseline = X_full_baseline
y_baseline = train_encoded['label'].astype(int).copy()

del X_full_baseline
gc.collect()

# Augmented model (with synthetic data)
X_augmented = train_augmented[feature_cols].copy()
if len(non_numeric_cols) > 0:
    X_augmented = X_augmented.drop(columns=non_numeric_cols)
y_augmented = train_augmented['label'].astype(int).copy()

# Test set (no labels available)
test_feature_cols = [c for c in test_encoded.columns if c != 'label']
X_test = test_encoded[test_feature_cols].copy()
if len(non_numeric_cols) > 0:
    X_test = X_test.drop(columns=[c for c in non_numeric_cols if c in X_test.columns])

print(f"  Baseline features: {X_baseline.shape}")
print(f"  Augmented features: {X_augmented.shape}")
print(f"  Test features: {X_test.shape}")

# Ensure same features across all sets
common_features = list(set(X_baseline.columns) & set(X_augmented.columns) & set(X_test.columns))
common_features = sorted(common_features)  # Sort for consistency
X_baseline = X_baseline[common_features]
X_augmented = X_augmented[common_features]
X_test = X_test[common_features]

print(f"  Common features: {len(common_features)}")

# Train/validation split for both datasets
print("  Creating train/validation splits...")
X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42, stratify=y_baseline
)

X_train_aug, X_val_aug, y_train_aug, y_val_aug = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
)

print(f"  Baseline - Train: {X_train_base.shape[0]:,}, Val: {X_val_base.shape[0]:,}")
print(f"  Augmented - Train: {X_train_aug.shape[0]:,}, Val: {X_val_aug.shape[0]:,}")

del X_baseline, X_augmented, y_baseline, y_augmented, train_encoded, train_augmented
gc.collect()

# ============================================================================
# 6. SCALE FEATURES
# ============================================================================
print("\n[6/7] Scaling features...")

# Baseline scaler
scaler_baseline = StandardScaler(with_mean=False)
X_train_base_scaled = scaler_baseline.fit_transform(X_train_base)
X_val_base_scaled = scaler_baseline.transform(X_val_base)
X_test_base_scaled = scaler_baseline.transform(X_test)

print("  Baseline scaling complete")

# Augmented scaler
scaler_augmented = StandardScaler(with_mean=False)
X_train_aug_scaled = scaler_augmented.fit_transform(X_train_aug)
X_val_aug_scaled = scaler_augmented.transform(X_val_aug)
X_test_aug_scaled = scaler_augmented.transform(X_test)

print("  Augmented scaling complete")

del X_train_base, X_val_base, X_train_aug, X_val_aug, X_test
gc.collect()

# ============================================================================
# 6.5. HANDLE NaN VALUES
# ============================================================================
print("\nHandling NaN values...")

# Check for NaN values
nan_count_train_base = np.isnan(X_train_base_scaled).sum()
nan_count_val_base = np.isnan(X_val_base_scaled).sum()
nan_count_train_aug = np.isnan(X_train_aug_scaled).sum()
nan_count_val_aug = np.isnan(X_val_aug_scaled).sum()
nan_count_test_base = np.isnan(X_test_base_scaled).sum()
nan_count_test_aug = np.isnan(X_test_aug_scaled).sum()

print(f"  NaN counts - Train baseline: {nan_count_train_base}, Val baseline: {nan_count_val_base}")
print(f"  NaN counts - Train augmented: {nan_count_train_aug}, Val augmented: {nan_count_val_aug}")
print(f"  NaN counts - Test baseline: {nan_count_test_base}, Test augmented: {nan_count_test_aug}")

# Replace NaN with 0 (after scaling, NaN → 0 is reasonable for sparse features)
if nan_count_train_base > 0 or nan_count_val_base > 0 or nan_count_test_base > 0:
    print("  Replacing NaN with 0 in baseline data...")
    X_train_base_scaled = np.nan_to_num(X_train_base_scaled, nan=0.0)
    X_val_base_scaled = np.nan_to_num(X_val_base_scaled, nan=0.0)
    X_test_base_scaled = np.nan_to_num(X_test_base_scaled, nan=0.0)

if nan_count_train_aug > 0 or nan_count_val_aug > 0 or nan_count_test_aug > 0:
    print("  Replacing NaN with 0 in augmented data...")
    X_train_aug_scaled = np.nan_to_num(X_train_aug_scaled, nan=0.0)
    X_val_aug_scaled = np.nan_to_num(X_val_aug_scaled, nan=0.0)
    X_test_aug_scaled = np.nan_to_num(X_test_aug_scaled, nan=0.0)

print("  ✓ NaN handling complete")

# ============================================================================
# 7. TRAIN MODELS
# ============================================================================
print("\n[7/7] Training logistic regression models...")

# Baseline model (class-weighted, no synthetic data)
print("  Training baseline model (class-weighted, no synthetic)...")
lr_baseline = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    random_state=42,
    verbose=0
)
lr_baseline.fit(X_train_base_scaled, y_train_base)
print("  ✓ Baseline model trained")

# Augmented model (class-weighted, with synthetic data)
print("  Training augmented model (class-weighted, with synthetic to 15%)...")
lr_augmented = LogisticRegression(
    solver='liblinear',
    class_weight='balanced',
    max_iter=1000,
    C=1.0,
    random_state=42,
    verbose=0
)
lr_augmented.fit(X_train_aug_scaled, y_train_aug)
print("  ✓ Augmented model trained")

del X_train_base_scaled, X_train_aug_scaled
gc.collect()

# ============================================================================
# 8. EVALUATE ON VALIDATION SET
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SET RESULTS")
print("=" * 80)

# Baseline predictions
val_pred_proba_base = lr_baseline.predict_proba(X_val_base_scaled)[:, 1]
val_pred_base = (val_pred_proba_base >= 0.5).astype(int)

# Augmented predictions
val_pred_proba_aug = lr_augmented.predict_proba(X_val_aug_scaled)[:, 1]
val_pred_aug = (val_pred_proba_aug >= 0.5).astype(int)

# Calculate metrics
metrics_baseline = {
    'AUC': roc_auc_score(y_val_base, val_pred_proba_base),
    'Accuracy': accuracy_score(y_val_base, val_pred_base),
    'Precision': precision_score(y_val_base, val_pred_base, zero_division=0),
    'Recall': recall_score(y_val_base, val_pred_base),
    'F1': f1_score(y_val_base, val_pred_base),
    'Log Loss': log_loss(y_val_base, val_pred_proba_base),
    'Avg Precision': average_precision_score(y_val_base, val_pred_proba_base)
}

metrics_augmented = {
    'AUC': roc_auc_score(y_val_aug, val_pred_proba_aug),
    'Accuracy': accuracy_score(y_val_aug, val_pred_aug),
    'Precision': precision_score(y_val_aug, val_pred_aug, zero_division=0),
    'Recall': recall_score(y_val_aug, val_pred_aug),
    'F1': f1_score(y_val_aug, val_pred_aug),
    'Log Loss': log_loss(y_val_aug, val_pred_proba_aug),
    'Avg Precision': average_precision_score(y_val_aug, val_pred_proba_aug)
}

# Print comparison
print("\nBaseline Model (No Synthetic Data):")
for k, v in metrics_baseline.items():
    print(f"  {k:15s}: {v:.4f}")

print("\nAugmented Model (With Synthetic Data to 15%):")
for k, v in metrics_augmented.items():
    print(f"  {k:15s}: {v:.4f}")

print("\nImprovement (Augmented - Baseline):")
for k in metrics_baseline.keys():
    diff = metrics_augmented[k] - metrics_baseline[k]
    if k == 'Log Loss':
        rel_change = (diff / metrics_baseline[k] * 100) if metrics_baseline[k] != 0 else 0
        print(f"  {k:15s}: {diff:+.4f} ({rel_change:+.1f}%) {'↓ better' if diff < 0 else '↑ worse'}")
    else:
        rel_change = (diff / metrics_baseline[k] * 100) if metrics_baseline[k] != 0 else 0
        print(f"  {k:15s}: {diff:+.4f} ({rel_change:+.1f}%)")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': list(metrics_baseline.keys()),
    'Baseline': list(metrics_baseline.values()),
    'Augmented': list(metrics_augmented.values())
})
metrics_df['Improvement'] = metrics_df['Augmented'] - metrics_df['Baseline']
metrics_df['Relative_Change_%'] = (metrics_df['Improvement'] / metrics_df['Baseline'] * 100).round(2)
metrics_df.to_csv('logistic_regression_comparison_metrics.csv', index=False)
print("\n✓ Metrics saved to 'logistic_regression_comparison_metrics.csv'")

# ============================================================================
# 9. GENERATE TEST SET PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("TEST SET PREDICTIONS")
print("=" * 80)

# Baseline predictions on test
test_pred_proba_base = lr_baseline.predict_proba(X_test_base_scaled)[:, 1]
test_pred_base = (test_pred_proba_base >= 0.5).astype(int)

# Augmented predictions on test
test_pred_proba_aug = lr_augmented.predict_proba(X_test_aug_scaled)[:, 1]
test_pred_aug = (test_pred_proba_aug >= 0.5).astype(int)

# Save predictions
test_predictions_base = pd.DataFrame({
    'predicted_probability': test_pred_proba_base,
    'predicted_class': test_pred_base
})
test_predictions_base.to_csv('test_predictions_lr_baseline.csv', index=False)

test_predictions_aug = pd.DataFrame({
    'predicted_probability': test_pred_proba_aug,
    'predicted_class': test_pred_aug
})
test_predictions_aug.to_csv('test_predictions_lr_augmented.csv', index=False)

print(f"✓ Baseline test predictions saved to 'test_predictions_lr_baseline.csv'")
print(f"✓ Augmented test predictions saved to 'test_predictions_lr_augmented.csv'")

# Print test prediction statistics
print("\nTest Prediction Statistics:")
print(f"\nBaseline Model:")
print(f"  Mean probability: {test_pred_proba_base.mean():.4f}")
print(f"  Median probability: {np.median(test_pred_proba_base):.4f}")
print(f"  Max probability: {test_pred_proba_base.max():.4f}")
print(f"  Predictions > 0.5: {(test_pred_proba_base > 0.5).sum():,}")
print(f"  Predictions > 0.1: {(test_pred_proba_base > 0.1).sum():,}")
print(f"  Predictions > 0.05: {(test_pred_proba_base > 0.05).sum():,}")

print(f"\nAugmented Model:")
print(f"  Mean probability: {test_pred_proba_aug.mean():.4f}")
print(f"  Median probability: {np.median(test_pred_proba_aug):.4f}")
print(f"  Max probability: {test_pred_proba_aug.max():.4f}")
print(f"  Predictions > 0.5: {(test_pred_proba_aug > 0.5).sum():,}")
print(f"  Predictions > 0.1: {(test_pred_proba_aug > 0.1).sum():,}")
print(f"  Predictions > 0.05: {(test_pred_proba_aug > 0.05).sum():,}")

# ============================================================================
# 10. SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

with open('lr_baseline_model.pkl', 'wb') as f:
    pickle.dump(lr_baseline, f)
print("✓ Baseline model saved to 'lr_baseline_model.pkl'")

with open('lr_augmented_model.pkl', 'wb') as f:
    pickle.dump(lr_augmented, f)
print("✓ Augmented model saved to 'lr_augmented_model.pkl'")

# Save validation predictions for visualization script
np.savez('validation_predictions.npz',
         y_val_base=y_val_base.values,
         val_pred_proba_base=val_pred_proba_base,
         val_pred_base=val_pred_base,
         y_val_aug=y_val_aug.values,
         val_pred_proba_aug=val_pred_proba_aug,
         val_pred_aug=val_pred_aug)
print("✓ Validation predictions saved to 'validation_predictions.npz'")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nNext: Run visualization script to generate comparison figures")
