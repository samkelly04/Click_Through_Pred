"""
Train LightGBM with CTGAN synthetic data augmentation
Compare results with baseline model
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

print("="*70)
print("LIGHTGBM WITH CTGAN SYNTHETIC DATA AUGMENTATION")
print("="*70)

# =================== LOAD OR GENERATE SYNTHETIC DATA ===================
SYNTHETIC_FILE = 'ctgan_synthetic_data.pkl'

if os.path.exists(SYNTHETIC_FILE):
    print(f"\n✓ Loading existing synthetic data from {SYNTHETIC_FILE}...")
    with open(SYNTHETIC_FILE, 'rb') as f:
        synthetic_data = pickle.load(f)
    print(f"  Loaded {len(synthetic_data):,} synthetic samples")
else:
    print(f"\n✗ Synthetic data not found. Generating using simplified approach...")
    print("  (For production CTGAN, run generate_ctgan_data.py separately)")

    # Load training data
    with open('train_encoded.pkl', 'rb') as f:
        train_encoded = pickle.load(f)

    # Simple oversampling approach (faster than CTGAN for initial testing)
    # Generate synthetic data by adding noise to minority class
    drop_cols = ['user_id', 'log_id']
    non_numeric_cols = train_encoded.select_dtypes(exclude=[np.number]).columns.tolist()

    minority_data = train_encoded[train_encoded['label'] == 1].copy()
    minority_numeric = minority_data.drop(columns=drop_cols + non_numeric_cols, errors='ignore')

    print(f"  Creating {len(minority_numeric):,} synthetic samples via noise augmentation...")

    # Add Gaussian noise to create synthetic samples
    noise_scale = 0.1
    synthetic_data = minority_numeric.copy()
    for col in synthetic_data.columns:
        if col != 'label':
            std = synthetic_data[col].std()
            noise = np.random.normal(0, std * noise_scale, len(synthetic_data))
            synthetic_data[col] = synthetic_data[col] + noise

    # Ensure label is 1
    synthetic_data['label'] = 1

    # Clip to reasonable ranges
    for col in synthetic_data.columns:
        if col != 'label':
            synthetic_data[col] = synthetic_data[col].clip(
                lower=minority_numeric[col].min(),
                upper=minority_numeric[col].max()
            )

    # Save for future use
    with open(SYNTHETIC_FILE, 'wb') as f:
        pickle.dump(synthetic_data, f)
    print(f"  Saved {len(synthetic_data):,} synthetic samples to {SYNTHETIC_FILE}")

print(f"\nSynthetic data summary:")
print(f"  Shape: {synthetic_data.shape}")
print(f"  Label distribution: {synthetic_data['label'].value_counts().to_dict()}")

# =================== LOAD ORIGINAL DATA ===================
print("\n[1/6] Loading original training data...")
with open('train_encoded.pkl', 'rb') as f:
    train_original = pickle.load(f)

print(f"  Original data: {train_original.shape}")
print(f"  Original label distribution:\n{train_original['label'].value_counts()}")

# Prepare original data
drop_cols = ['user_id', 'log_id']
non_numeric_cols = train_original.select_dtypes(exclude=[np.number]).columns.tolist()

train_original_clean = train_original.drop(columns=drop_cols + non_numeric_cols, errors='ignore')

# Align columns between original and synthetic
common_cols = list(set(train_original_clean.columns) & set(synthetic_data.columns))
train_original_aligned = train_original_clean[common_cols]
synthetic_aligned = synthetic_data[common_cols]

print(f"  Original (numeric): {train_original_aligned.shape}")
print(f"  Synthetic: {synthetic_aligned.shape}")
print(f"  Common columns: {len(common_cols)}")

# =================== TRAIN/VALIDATION SPLIT (ON ORIGINAL DATA FIRST) ===================
print("\n[2/6] Splitting original data into train/validation...")

X_original = train_original_aligned.drop(columns=['label'])
y_original = train_original_aligned['label']

X_train_orig, X_val, y_train_orig, y_val = train_test_split(
    X_original, y_original,
    test_size=0.2,
    random_state=42,
    stratify=y_original
)

print(f"  Original train: {X_train_orig.shape}, Validation: {X_val.shape}")
print(f"  Original train class balance: {y_train_orig.value_counts(normalize=True).to_dict()}")
print(f"  Validation class balance: {y_val.value_counts(normalize=True).to_dict()}")

# =================== COMBINE TRAINING DATA + SYNTHETIC ===================
print("\n[3/6] Combining training data with synthetic data...")

# Combine training data with synthetic (validation stays real-only)
train_augmented = pd.concat([
    pd.DataFrame(X_train_orig, columns=X_train_orig.columns).assign(label=y_train_orig.values),
    synthetic_aligned
], axis=0, ignore_index=True)

X_train = train_augmented.drop(columns=['label'])
y_train = train_augmented['label']

print(f"\n  Augmented training dataset: {X_train.shape}")
print(f"  Augmented training label distribution:\n{y_train.value_counts()}")
print(f"  Augmented training class balance: {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Validation set (real-only): {X_val.shape}")

# =================== TRAIN MODEL ===================
print("\n[4/6] Training LightGBM on augmented data...")
print("  Note: Training on augmented data (real + synthetic), evaluating on real-only validation")

model_augmented = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=31,
    learning_rate=0.05,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42,
    verbose=1
)

print("  Training model...")
model_augmented.fit(X_train, y_train)

print(f"\n  Training completed")
print(f"  Number of iterations: {model_augmented.n_iter_}")

# =================== PREDICTIONS ===================
print("\n[5/6] Generating predictions...")

y_pred_proba_train = model_augmented.predict_proba(X_train)[:, 1]
y_pred_proba_val = model_augmented.predict_proba(X_val)[:, 1]

y_pred_train = (y_pred_proba_train >= 0.5).astype(int)
y_pred_val = (y_pred_proba_val >= 0.5).astype(int)

# Compute metrics
def compute_metrics(y_true, y_pred, y_proba, set_name):
    metrics = {
        'AUC': roc_auc_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Log Loss': log_loss(y_true, y_proba),
        'AP': average_precision_score(y_true, y_proba)
    }
    print(f"\n  {set_name} Metrics:")
    for k, v in metrics.items():
        print(f"    {k:<12s}: {v:.4f}")
    return metrics

train_metrics_aug = compute_metrics(y_train, y_pred_train, y_pred_proba_train, "Training (Augmented - Real + Synthetic)")
val_metrics_aug = compute_metrics(y_val, y_pred_val, y_pred_proba_val, "Validation (Real-Only)")

# =================== TEST SET PREDICTIONS ===================
print("\n[6/6] Evaluating on test set...")

with open('test_encoded.pkl', 'rb') as f:
    test_encoded = pickle.load(f)

# Prepare test features (align with augmented training data)
X_test = test_encoded.drop(columns=['user_id', 'log_id', 'label'] + non_numeric_cols, errors='ignore')
X_test = X_test[[col for col in common_cols if col != 'label']]

print(f"  Test set shape: {X_test.shape}")

y_pred_proba_test_aug = model_augmented.predict_proba(X_test)[:, 1]
y_pred_test_aug = (y_pred_proba_test_aug >= 0.5).astype(int)

print(f"  Predictions generated for {len(X_test):,} test samples")
print(f"  Test predictions - min: {y_pred_proba_test_aug.min():.4f}, max: {y_pred_proba_test_aug.max():.4f}, mean: {y_pred_proba_test_aug.mean():.4f}")
print(f"  Predicted labels (threshold=0.5): {pd.Series(y_pred_test_aug).value_counts().to_dict()}")

# =================== SAVE RESULTS ===================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save model
joblib.dump(model_augmented, 'lightgbm_ctgan_model.pkl')
print("\n  Model saved: lightgbm_ctgan_model.pkl")

# Save metrics
metrics_aug_df = pd.DataFrame({
    'Metric': list(train_metrics_aug.keys()),
    'Training': list(train_metrics_aug.values()),
    'Validation': list(val_metrics_aug.values())
})
metrics_aug_df.to_csv('lightgbm_ctgan_metrics.csv', index=False)
print("  Metrics saved: lightgbm_ctgan_metrics.csv")

# Save test predictions
test_predictions_aug = test_encoded[['user_id', 'log_id']].copy()
test_predictions_aug['predicted_probability'] = y_pred_proba_test_aug
test_predictions_aug['predicted_label'] = y_pred_test_aug
test_predictions_aug.to_csv('test_predictions_ctgan_lgbm.csv', index=False)
print("  Test predictions saved: test_predictions_ctgan_lgbm.csv")

# =================== COMPARE WITH BASELINE ===================
print("\n" + "="*70)
print("COMPARISON: BASELINE vs AUGMENTED")
print("="*70)

# Load baseline metrics
baseline_metrics = pd.read_csv('gradient_boosting_metrics.csv')

# Load baseline test predictions
baseline_test_preds = pd.read_csv('test_predictions.csv')

print("\nValidation Performance:")
print("-" * 60)
print(f"{'Metric':<15} {'Baseline':>12} {'Augmented':>12} {'Change':>12}")
print("-" * 60)

for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss', 'AP']:
    baseline_val = baseline_metrics[baseline_metrics['Metric']==metric]['Validation'].values[0]
    augmented_val = val_metrics_aug[metric]
    change = augmented_val - baseline_val
    change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
    print(f"{metric:<15} {baseline_val:>12.4f} {augmented_val:>12.4f} {change_str:>12}")

print("-" * 60)

print("\nTest Set Predictions Comparison:")
print("-" * 60)
print(f"{'Metric':<30} {'Baseline':>15} {'Augmented':>15}")
print("-" * 60)
print(f"{'Mean predicted probability':<30} {baseline_test_preds['predicted_probability'].mean():>15.4f} {y_pred_proba_test_aug.mean():>15.4f}")
print(f"{'Median predicted probability':<30} {baseline_test_preds['predicted_probability'].median():>15.4f} {np.median(y_pred_proba_test_aug):>15.4f}")
print(f"{'Max predicted probability':<30} {baseline_test_preds['predicted_probability'].max():>15.4f} {y_pred_proba_test_aug.max():>15.4f}")
print(f"{'Predictions > 0.5':<30} {(baseline_test_preds['predicted_label']==1).sum():>15,} {(y_pred_test_aug==1).sum():>15,}")
print(f"{'Predictions > 0.1':<30} {(baseline_test_preds['predicted_probability']>0.1).sum():>15,} {(y_pred_proba_test_aug>0.1).sum():>15,}")
print("-" * 60)

print("\n✓ Training and comparison complete!")
print("\nNext: Run 'python compare_models_visualizations.py' to generate comparison plots")
