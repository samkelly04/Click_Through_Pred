"""
Gradient Boosting Analysis for CTR Prediction
Trains a gradient boosting model and generates publication-quality visualizations.
Uses HistGradientBoostingClassifier (similar performance to LightGBM, no dependencies)
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

print("="*70)
print("GRADIENT BOOSTING ANALYSIS FOR CTR PREDICTION")
print("="*70)

# =================== LOAD DATA ===================
print("\n[1/5] Loading preprocessed data...")
with open('train_encoded.pkl', 'rb') as f:
    train_encoded = pickle.load(f)

print(f"  Loaded training data: {train_encoded.shape}")
print(f"  Label distribution:\n{train_encoded['label'].value_counts()}")

# =================== PREPARE FEATURES ===================
print("\n[2/5] Preparing features...")

# Drop non-feature columns
drop_cols = ['user_id', 'log_id', 'label']
non_numeric_cols = train_encoded.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"  Dropping identifier columns: {drop_cols}")
print(f"  Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")

# Prepare X and y
X_full = train_encoded.drop(columns=drop_cols, errors='ignore')
X_full = X_full.select_dtypes(include=[np.number])  # Keep only numeric
y_full = train_encoded['label'].copy()

print(f"  Final feature matrix: {X_full.shape}")
print(f"  Features: {list(X_full.columns[:10])}... (showing first 10)")

# Train/validation split (80/20)
print("\n  Splitting into train/validation (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

print(f"  Train: {X_train.shape}, Validation: {X_val.shape}")
print(f"  Train class balance: {y_train.value_counts(normalize=True).to_dict()}")
print(f"  Val class balance: {y_val.value_counts(normalize=True).to_dict()}")

# =================== TRAIN GRADIENT BOOSTING ===================
print("\n[3/5] Training Gradient Boosting model...")

# Basic hyperparameters (minimal tuning)
# HistGradientBoosting is similar to LightGBM
model = HistGradientBoostingClassifier(
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

print("  Hyperparameters:")
print(f"    max_iter: {model.max_iter}")
print(f"    max_leaf_nodes: {model.max_leaf_nodes}")
print(f"    learning_rate: {model.learning_rate}")
print(f"    early_stopping: {model.early_stopping}")
print(f"    validation_fraction: {model.validation_fraction}")

# Train model
print("\n  Training model...")
model.fit(X_train, y_train)

print(f"\n  Training completed")
print(f"  Number of iterations: {model.n_iter_}")

# =================== PREDICTIONS ===================
print("\n[4/5] Generating predictions...")

y_pred_proba_train = model.predict_proba(X_train)[:, 1]
y_pred_proba_val = model.predict_proba(X_val)[:, 1]

# Default threshold 0.5
y_pred_train = (y_pred_proba_train >= 0.5).astype(int)
y_pred_val = (y_pred_proba_val >= 0.5).astype(int)

# =================== METRICS ===================
print("\n  Computing metrics...")

def compute_metrics(y_true, y_pred, y_proba, set_name):
    """Compute comprehensive metrics"""
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
        print(f"    {k:12s}: {v:.4f}")

    return metrics

train_metrics = compute_metrics(y_train, y_pred_train, y_pred_proba_train, "Training")
val_metrics = compute_metrics(y_val, y_pred_val, y_pred_proba_val, "Validation")

# =================== VISUALIZATIONS ===================
print("\n[5/5] Creating publication-quality visualizations...")

# Create figure directory
import os
os.makedirs('figures', exist_ok=True)

# 1. ROC Curve
print("  [1/7] ROC curve...")
fig, ax = plt.subplots(figsize=(6, 5))

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)

ax.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_metrics["AUC"]:.3f})',
        linewidth=2, color='#2E86AB')
ax.plot(fpr_val, tpr_val, label=f'Validation (AUC = {val_metrics["AUC"]:.3f})',
        linewidth=2, color='#A23B72')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Gradient Boosting CTR Prediction', fontweight='bold')
ax.legend(loc='lower right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figures/roc_curve.png', bbox_inches='tight')
print("    Saved: figures/roc_curve.png")
plt.close()

# 2. Precision-Recall Curve
print("  [2/7] Precision-Recall curve...")
fig, ax = plt.subplots(figsize=(6, 5))

precision_train, recall_train, _ = precision_recall_curve(y_train, y_pred_proba_train)
precision_val, recall_val, _ = precision_recall_curve(y_val, y_pred_proba_val)

baseline_train = y_train.mean()
baseline_val = y_val.mean()

ax.plot(recall_train, precision_train,
        label=f'Train (AP = {train_metrics["AP"]:.3f})',
        linewidth=2, color='#2E86AB')
ax.plot(recall_val, precision_val,
        label=f'Validation (AP = {val_metrics["AP"]:.3f})',
        linewidth=2, color='#A23B72')
ax.axhline(baseline_val, color='k', linestyle='--', linewidth=1,
           label=f'Baseline (pos rate = {baseline_val:.3f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - Gradient Boosting CTR Prediction', fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figures/precision_recall_curve.png', bbox_inches='tight')
print("    Saved: figures/precision_recall_curve.png")
plt.close()

# 3. Feature Importance (Top 20) - using permutation importance
print("  [3/7] Feature importance (computing permutation importance)...")
from sklearn.inspection import permutation_importance

# Compute permutation importance on a sample for speed
sample_size = min(50000, len(X_val))
sample_idx = np.random.choice(len(X_val), sample_size, replace=False)
X_val_sample = X_val.iloc[sample_idx]
y_val_sample = y_val.iloc[sample_idx]

perm_importance = permutation_importance(
    model, X_val_sample, y_val_sample,
    n_repeats=5, random_state=42, n_jobs=-1
)

fig, ax = plt.subplots(figsize=(8, 6))

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False).head(20)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
ax.barh(range(len(feature_importance)), feature_importance['importance'][::-1], color=colors[::-1])
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['feature'][::-1])
ax.set_xlabel('Importance (Gain)')
ax.set_title('Top 20 Feature Importance - Gradient Boosting', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', bbox_inches='tight')
print("    Saved: figures/feature_importance.png")
plt.close()

# 4. Confusion Matrix (Validation)
print("  [4/7] Confusion matrix...")
fig, ax = plt.subplots(figsize=(6, 5))

cm = confusion_matrix(y_val, y_pred_val)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=np.array([[f'{val:.2%}\n(n={cm[i,j]:,})'
            for j, val in enumerate(row)] for i, row in enumerate(cm_normalized)]),
            fmt='', cmap='Blues', cbar_kws={'label': 'Proportion'},
            xticklabels=['No Click (0)', 'Click (1)'],
            yticklabels=['No Click (0)', 'Click (1)'],
            ax=ax)

ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix (Validation) - Normalized', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', bbox_inches='tight')
print("    Saved: figures/confusion_matrix.png")
plt.close()

# 5. Score Distribution
print("  [5/7] Prediction score distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, y_true, y_proba, title in zip(
    axes,
    [y_train, y_val],
    [y_pred_proba_train, y_pred_proba_val],
    ['Training Set', 'Validation Set']
):
    ax.hist(y_proba[y_true == 0], bins=50, alpha=0.6, label='No Click (0)',
            color='#2E86AB', density=True)
    ax.hist(y_proba[y_true == 1], bins=50, alpha=0.6, label='Click (1)',
            color='#F18F01', density=True)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/score_distribution.png', bbox_inches='tight')
print("    Saved: figures/score_distribution.png")
plt.close()

# 6. Threshold Analysis
print("  [6/7] Threshold analysis...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

thresholds = np.linspace(0.01, 0.99, 100)
precisions, recalls, f1s, accuracies = [], [], [], []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba_val >= thresh).astype(int)
    precisions.append(precision_score(y_val, y_pred_thresh, zero_division=0))
    recalls.append(recall_score(y_val, y_pred_thresh))
    f1s.append(f1_score(y_val, y_pred_thresh))
    accuracies.append(accuracy_score(y_val, y_pred_thresh))

# Plot 1: Precision/Recall vs Threshold
ax = axes[0]
ax.plot(thresholds, precisions, label='Precision', linewidth=2, color='#2E86AB')
ax.plot(thresholds, recalls, label='Recall', linewidth=2, color='#A23B72')
ax.plot(thresholds, f1s, label='F1 Score', linewidth=2, color='#F18F01')
ax.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Default (0.5)')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Metrics vs Classification Threshold', fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

# Plot 2: F1 and Accuracy vs Threshold
ax = axes[1]
ax.plot(thresholds, f1s, label='F1 Score', linewidth=2, color='#F18F01')
ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2, color='#06A77D')
ax.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Default (0.5)')
best_f1_idx = np.argmax(f1s)
ax.axvline(thresholds[best_f1_idx], color='green', linestyle=':', linewidth=2,
           label=f'Best F1 ({thresholds[best_f1_idx]:.3f})')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('F1 & Accuracy vs Threshold', fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/threshold_analysis.png', bbox_inches='tight')
print("    Saved: figures/threshold_analysis.png")
plt.close()

# 7. Calibration Curve
print("  [7/7] Calibration curve...")
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(6, 5))

prob_true, prob_pred = calibration_curve(y_val, y_pred_proba_val, n_bins=10, strategy='uniform')

ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
        label='Gradient Boosting', color='#2E86AB')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')

ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve - Gradient Boosting', fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figures/calibration_curve.png', bbox_inches='tight')
print("    Saved: figures/calibration_curve.png")
plt.close()

# =================== SUMMARY REPORT ===================
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

print("\nModel: Histogram-based Gradient Boosting")
print(f"Training samples: {len(y_train):,}")
print(f"Validation samples: {len(y_val):,}")
print(f"Features: {X_train.shape[1]}")
print(f"Iterations: {model.n_iter_}")

print("\nPerformance Metrics:")
print("-" * 50)
print(f"{'Metric':<15} {'Training':>12} {'Validation':>12}")
print("-" * 50)
for metric in ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss', 'AP']:
    print(f"{metric:<15} {train_metrics[metric]:>12.4f} {val_metrics[metric]:>12.4f}")
print("-" * 50)

print("\nTop 10 Most Important Features:")
print("-" * 50)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:<30} {row['importance']:>10.1f}")
print("-" * 50)

print("\nClassification Report (Validation):")
print(classification_report(y_val, y_pred_val, target_names=['No Click', 'Click'], digits=4))

print("\n" + "="*70)
print("All visualizations saved to 'figures/' directory")
print("="*70)

# Save model
print("\nSaving model...")
import joblib
joblib.dump(model, 'gradient_boosting_model.pkl')
print("  Model saved: gradient_boosting_model.pkl")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': list(train_metrics.keys()),
    'Training': list(train_metrics.values()),
    'Validation': list(val_metrics.values())
})
metrics_df.to_csv('gradient_boosting_metrics.csv', index=False)
print("  Metrics saved: gradient_boosting_metrics.csv")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("  Feature importance saved: feature_importance.csv")

print("\nAnalysis complete!")
