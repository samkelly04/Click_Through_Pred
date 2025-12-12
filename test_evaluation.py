"""
Evaluate LightGBM model on held-out test set
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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

print("="*70)
print("LIGHTGBM - TEST SET EVALUATION")
print("="*70)

# Load model
print("\n[1/4] Loading trained LightGBM model...")
model = joblib.load('gradient_boosting_model.pkl')
print("  Model loaded successfully")

# Load test data
print("\n[2/4] Loading test data...")
with open('test_encoded.pkl', 'rb') as f:
    test_encoded = pickle.load(f)

print(f"  Loaded test data: {test_encoded.shape}")

# Check if test set has labels
has_labels = False
if 'label' in test_encoded.columns:
    label_count = test_encoded['label'].notna().sum()
    if label_count > 0:
        has_labels = True
        print(f"  Test set has {label_count:,} labeled samples")
        print(f"  Label distribution:\n{test_encoded['label'].value_counts()}")
    else:
        print("  Test set is unlabeled (all NaN) - will generate predictions only")
else:
    print("  Test set has no label column - will generate predictions only")

# Prepare features
drop_cols = ['user_id', 'log_id', 'label']
non_numeric_cols = test_encoded.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\n  Dropping identifier columns: {drop_cols}")
print(f"  Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")

X_test = test_encoded.drop(columns=drop_cols, errors='ignore')
X_test = X_test.select_dtypes(include=[np.number])

if has_labels:
    y_test = test_encoded['label'].copy()
    # Remove rows with NaN labels
    mask = y_test.notna()
    X_test = X_test[mask]
    y_test = y_test[mask].astype(int)
    print(f"  Final test set: {X_test.shape}")
    print(f"  Class balance: {y_test.value_counts(normalize=True).to_dict()}")

# Generate predictions
print("\n[3/4] Generating predictions on test set...")
y_pred_proba_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba_test >= 0.5).astype(int)

print(f"  Predictions generated for {len(X_test):,} samples")
print(f"  Predicted probabilities - min: {y_pred_proba_test.min():.4f}, max: {y_pred_proba_test.max():.4f}, mean: {y_pred_proba_test.mean():.4f}")
print(f"  Predicted labels (threshold=0.5): {pd.Series(y_pred_test).value_counts().to_dict()}")

# Save predictions
predictions_df = test_encoded[['user_id', 'log_id']].copy()
predictions_df['predicted_probability'] = y_pred_proba_test
predictions_df['predicted_label'] = y_pred_test
if has_labels:
    predictions_df['true_label'] = test_encoded['label'].values

predictions_df.to_csv('test_predictions.csv', index=False)
print("\n  Predictions saved to: test_predictions.csv")

# If test set has labels, compute metrics and visualizations
if has_labels:
    print("\n[4/4] Computing test set metrics and visualizations...")

    # Compute metrics
    test_metrics = {
        'AUC': roc_auc_score(y_test, y_pred_proba_test),
        'Accuracy': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test, zero_division=0),
        'Recall': recall_score(y_test, y_pred_test),
        'F1': f1_score(y_test, y_pred_test),
        'Log Loss': log_loss(y_test, y_pred_proba_test),
        'AP': average_precision_score(y_test, y_pred_proba_test)
    }

    print("\n  Test Set Metrics:")
    print("  " + "-" * 40)
    for k, v in test_metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("  " + "-" * 40)

    # Create comparison with validation
    val_metrics = pd.read_csv('gradient_boosting_metrics.csv')
    comparison_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Validation': [val_metrics[val_metrics['Metric']==k]['Validation'].values[0] for k in test_metrics.keys()],
        'Test': list(test_metrics.values())
    })
    comparison_df.to_csv('validation_vs_test_metrics.csv', index=False)
    print("\n  Comparison saved to: validation_vs_test_metrics.csv")

    # Create visualizations
    import os
    os.makedirs('figures/test', exist_ok=True)

    # 1. ROC Curve Comparison
    print("\n  Creating visualizations...")
    print("    [1/5] ROC curve comparison...")

    # Load validation predictions for comparison
    with open('train_encoded.pkl', 'rb') as f:
        train_data = pickle.load(f)

    from sklearn.model_selection import train_test_split
    X_full = train_data.drop(columns=['user_id', 'log_id', 'label'], errors='ignore')
    X_full = X_full.select_dtypes(include=[np.number])
    y_full = train_data['label'].copy()
    _, X_val, _, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))

    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)

    ax.plot(fpr_val, tpr_val, label=f'Validation (AUC = {roc_auc_score(y_val, y_pred_proba_val):.3f})',
            linewidth=2, color='#2E86AB')
    ax.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_metrics["AUC"]:.3f})',
            linewidth=2, color='#A23B72')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - LightGBM (Validation vs Test)', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/test/roc_comparison.png', bbox_inches='tight')
    print("      Saved: figures/test/roc_comparison.png")
    plt.close()

    # 2. Precision-Recall Curve
    print("    [2/5] Precision-recall comparison...")
    fig, ax = plt.subplots(figsize=(7, 6))

    precision_val, recall_val, _ = precision_recall_curve(y_val, y_pred_proba_val)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_pred_proba_test)

    ax.plot(recall_val, precision_val,
            label=f'Validation (AP = {average_precision_score(y_val, y_pred_proba_val):.3f})',
            linewidth=2, color='#2E86AB')
    ax.plot(recall_test, precision_test,
            label=f'Test (AP = {test_metrics["AP"]:.3f})',
            linewidth=2, color='#A23B72')
    ax.axhline(y_test.mean(), color='k', linestyle='--', linewidth=1,
               label=f'Baseline (pos rate = {y_test.mean():.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - LightGBM (Validation vs Test)', fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/test/pr_comparison.png', bbox_inches='tight')
    print("      Saved: figures/test/pr_comparison.png")
    plt.close()

    # 3. Metrics Comparison Bar Chart
    print("    [3/5] Metrics comparison chart...")
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_to_plot = ['AUC', 'Precision', 'Recall', 'F1', 'AP']
    val_values = [comparison_df[comparison_df['Metric']==m]['Validation'].values[0] for m in metrics_to_plot]
    test_values = [comparison_df[comparison_df['Metric']==m]['Test'].values[0] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    ax.bar(x - width/2, val_values, width, label='Validation', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, test_values, width, label='Test', color='#A23B72', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('LightGBM Performance: Validation vs Test', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for i, (v, t) in enumerate(zip(val_values, test_values)):
        ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/test/metrics_comparison.png', bbox_inches='tight')
    print("      Saved: figures/test/metrics_comparison.png")
    plt.close()

    # 4. Test Set Confusion Matrix
    print("    [4/5] Test confusion matrix...")
    fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_test, y_pred_test)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=np.array([[f'{val:.2%}\n(n={cm[i,j]:,})'
                for j, val in enumerate(row)] for i, row in enumerate(cm_normalized)]),
                fmt='', cmap='Blues', cbar_kws={'label': 'Proportion'},
                xticklabels=['No Click (0)', 'Click (1)'],
                yticklabels=['No Click (0)', 'Click (1)'],
                ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Test Set Confusion Matrix - LightGBM', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/test/test_confusion_matrix.png', bbox_inches='tight')
    print("      Saved: figures/test/test_confusion_matrix.png")
    plt.close()

    # 5. Score Distribution on Test
    print("    [5/5] Test score distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(y_pred_proba_test[y_test == 0], bins=50, alpha=0.6, label='No Click (0)',
            color='#2E86AB', density=True)
    ax.hist(y_pred_proba_test[y_test == 1], bins=50, alpha=0.6, label='Click (1)',
            color='#F18F01', density=True)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Test Set Score Distribution - LightGBM', fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/test/test_score_distribution.png', bbox_inches='tight')
    print("      Saved: figures/test/test_score_distribution.png")
    plt.close()

    # Classification Report
    print("\n  Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['No Click', 'Click'], digits=4))

else:
    print("\n[4/4] Test set has no labels - skipping metrics computation")
    print("  Predictions have been saved to test_predictions.csv")

print("\n" + "="*70)
print("LIGHTGBM TEST EVALUATION COMPLETE")
print("="*70)

if has_labels:
    print("\nSummary:")
    print(f"  Test AUC: {test_metrics['AUC']:.4f}")
    print(f"  Test Accuracy: {test_metrics['Accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['Precision']:.4f}")
    print(f"  Test Recall: {test_metrics['Recall']:.4f}")
    print(f"  Test F1: {test_metrics['F1']:.4f}")
    print("\nAll test visualizations saved to 'figures/test/' directory")
else:
    print("\nPredictions saved to 'test_predictions.csv'")
    print("  Columns: user_id, log_id, predicted_probability, predicted_label")

print("\nDone!")
