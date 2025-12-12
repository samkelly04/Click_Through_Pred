"""
Create visualizations for LightGBM test predictions (unlabeled test set)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

print("Creating test set visualizations...")

# Load predictions
pred = pd.read_csv('test_predictions.csv')

import os
os.makedirs('figures/test', exist_ok=True)

# 1. Prediction Probability Distribution
print("  [1/4] Prediction probability distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(pred['predicted_probability'], bins=100, color='#2E86AB', alpha=0.7, edgecolor='black')
ax.axvline(pred['predicted_probability'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean = {pred["predicted_probability"].mean():.4f}')
ax.axvline(pred['predicted_probability'].median(), color='orange', linestyle='--',
           linewidth=2, label=f'Median = {pred["predicted_probability"].median():.4f}')
ax.axvline(0.5, color='green', linestyle=':', linewidth=2, label='Threshold = 0.5')

ax.set_xlabel('Predicted Click Probability')
ax.set_ylabel('Frequency')
ax.set_title('LightGBM - Test Set Prediction Distribution (N=976,058)', fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figures/test/test_probability_distribution.png', bbox_inches='tight')
print("    Saved: figures/test/test_probability_distribution.png")
plt.close()

# 2. Probability Distribution (Log Scale)
print("  [2/4] Probability distribution (log scale)...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(pred['predicted_probability'], bins=100, color='#A23B72', alpha=0.7, edgecolor='black')
ax.set_yscale('log')
ax.axvline(pred['predicted_probability'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean = {pred["predicted_probability"].mean():.4f}')
ax.axvline(pred['predicted_probability'].median(), color='orange', linestyle='--',
           linewidth=2, label=f'Median = {pred["predicted_probability"].median():.4f}')

ax.set_xlabel('Predicted Click Probability')
ax.set_ylabel('Frequency (log scale)')
ax.set_title('LightGBM - Test Set Prediction Distribution (Log Scale)', fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figures/test/test_probability_distribution_log.png', bbox_inches='tight')
print("    Saved: figures/test/test_probability_distribution_log.png")
plt.close()

# 3. Binned Probability Distribution
print("  [3/4] Binned probability distribution...")
bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
labels = ['<1%', '1-5%', '5-10%', '10-20%', '20-50%', '>50%']
pred['prob_bin'] = pd.cut(pred['predicted_probability'], bins=bins, labels=labels)
bin_counts = pred['prob_bin'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(bin_counts)))
bars = ax.bar(range(len(bin_counts)), bin_counts.values, color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(bin_counts)))
ax.set_xticklabels(bin_counts.index)
ax.set_ylabel('Number of Predictions')
ax.set_xlabel('Predicted Probability Range')
ax.set_title('LightGBM - Test Predictions by Probability Range', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, bin_counts.values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}\n({count/len(pred)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/test/test_probability_bins.png', bbox_inches='tight')
print("    Saved: figures/test/test_probability_bins.png")
plt.close()

# 4. Top Predictions Analysis
print("  [4/4] Top predictions analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Box plot of probabilities
ax = axes[0, 0]
ax.boxplot([pred['predicted_probability']], vert=True, labels=['All Predictions'])
ax.set_ylabel('Predicted Probability')
ax.set_title('Distribution Overview (Box Plot)', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# Top right: Top 1% predictions
ax = axes[0, 1]
top_1pct = pred.nlargest(int(len(pred) * 0.01), 'predicted_probability')
ax.hist(top_1pct['predicted_probability'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Frequency')
ax.set_title(f'Top 1% Predictions (N={len(top_1pct):,})', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# Bottom left: Cumulative distribution
ax = axes[1, 0]
sorted_probs = np.sort(pred['predicted_probability'])
cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
ax.plot(sorted_probs, cumulative, linewidth=2, color='#2E86AB')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
ax.axhline(0.99, color='green', linestyle='--', alpha=0.5, label='99th percentile')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Cumulative Proportion')
ax.set_title('Cumulative Distribution Function', fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

# Bottom right: Statistics table
ax = axes[1, 1]
ax.axis('off')
stats_text = f"""
LightGBM Test Predictions Summary

Total Samples: {len(pred):,}

Probability Statistics:
  Mean:     {pred['predicted_probability'].mean():.4f}
  Median:   {pred['predicted_probability'].median():.4f}
  Std Dev:  {pred['predicted_probability'].std():.4f}
  Min:      {pred['predicted_probability'].min():.4f}
  Max:      {pred['predicted_probability'].max():.4f}

Percentiles:
  25th:     {pred['predicted_probability'].quantile(0.25):.4f}
  50th:     {pred['predicted_probability'].quantile(0.50):.4f}
  75th:     {pred['predicted_probability'].quantile(0.75):.4f}
  90th:     {pred['predicted_probability'].quantile(0.90):.4f}
  95th:     {pred['predicted_probability'].quantile(0.95):.4f}
  99th:     {pred['predicted_probability'].quantile(0.99):.4f}

Predictions (threshold=0.5):
  No Click (0): {(pred['predicted_label']==0).sum():,}
  Click (1):    {(pred['predicted_label']==1).sum():,}
"""
ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('figures/test/test_analysis_summary.png', bbox_inches='tight')
print("    Saved: figures/test/test_analysis_summary.png")
plt.close()

print("\nAll test visualizations created successfully!")
print("\nGenerated files:")
print("  - figures/test/test_probability_distribution.png")
print("  - figures/test/test_probability_distribution_log.png")
print("  - figures/test/test_probability_bins.png")
print("  - figures/test/test_analysis_summary.png")
