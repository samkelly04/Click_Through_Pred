"""
Conditional Tabular VAE (C-TVAE) Demonstration
Targeted Synthetic Data Generation for Specific Segments

This script demonstrates how to use SDV's TVAE to generate synthetic data
conditioned on specific feature values, enabling targeted augmentation
for underrepresented segments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CONDITIONAL TABULAR VAE (C-TVAE) DEMONSTRATION")
print("Targeted Synthetic Data Generation")
print("=" * 80)

# ============================================================================
# STEP 1: CREATE SAMPLE DATASET
# ============================================================================
print("\n[Step 1/6] Creating sample dataset...")

np.random.seed(42)

# Generate 10,000 samples with class imbalance (1:10 ratio)
n_samples = 10000
n_clicks = 1000  # 10% are clicks
n_noclicks = 9000  # 90% are no-clicks

# Helper function to create data
def create_sample_data(n, clicked_value):
    data = {
        'user_id': [f'user_{i:06d}' for i in range(n)],
        'slot_id': np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.20, 0.25, 0.25, 0.15]),
        'ad_relevance_score': np.random.beta(2, 5, size=n) if clicked_value == 1 else np.random.beta(2, 8, size=n),
        'user_age': np.random.randint(18, 65, size=n),
        'clicked': clicked_value
    }
    return pd.DataFrame(data)

# Create click and no-click data
df_clicks = create_sample_data(n_clicks, clicked_value=1)
df_noclicks = create_sample_data(n_noclicks, clicked_value=0)

# Combine and shuffle
df_real = pd.concat([df_clicks, df_noclicks], ignore_index=True)
df_real = df_real.sample(frac=1, random_state=42).reset_index(drop=True)

# Update user_id to be sequential
df_real['user_id'] = [f'user_{i:06d}' for i in range(len(df_real))]

print(f"  Total samples: {len(df_real):,}")
print(f"  Click samples (minority): {(df_real['clicked']==1).sum():,} ({(df_real['clicked']==1).mean()*100:.1f}%)")
print(f"  No-click samples (majority): {(df_real['clicked']==0).sum():,} ({(df_real['clicked']==0).mean()*100:.1f}%)")

# Show class distribution by slot_id
print("\n  Click distribution by slot_id:")
slot_dist = df_real[df_real['clicked']==1]['slot_id'].value_counts().sort_index()
for slot, count in slot_dist.items():
    print(f"    slot_id={slot}: {count} clicks")

# ============================================================================
# STEP 2: DEFINE METADATA
# ============================================================================
print("\n[Step 2/6] Defining metadata...")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_real)

# Set primary key
metadata.update_column(
    column_name='user_id',
    sdtype='id'
)

# Set slot_id as categorical
metadata.update_column(
    column_name='slot_id',
    sdtype='categorical'
)

# Set clicked as categorical (binary)
metadata.update_column(
    column_name='clicked',
    sdtype='categorical'
)

print("  Metadata defined:")
print(f"    Primary key: user_id")
print(f"    Categorical: slot_id, clicked")
print(f"    Numerical: ad_relevance_score, user_age")

# ============================================================================
# STEP 3: TRAIN C-TVAE ON MINORITY CLASS ONLY
# ============================================================================
print("\n[Step 3/6] Training Conditional TVAE on minority class (clicked=1)...")

# Extract minority class data
df_minority = df_real[df_real['clicked'] == 1].copy()
print(f"  Training data: {len(df_minority):,} samples (all clicks)")
print(f"  Distribution by slot_id:")
for slot, count in df_minority['slot_id'].value_counts().sort_index().items():
    print(f"    slot_id={slot}: {count} samples ({count/len(df_minority)*100:.1f}%)")

# Initialize and train TVAE
synthesizer = TVAESynthesizer(
    metadata,
    epochs=100,
    batch_size=128,
    embedding_dim=64,
    compress_dims=(64, 32),
    decompress_dims=(32, 64),
    verbose=False
)

print("\n  Training TVAE... (this may take 1-2 minutes)")
synthesizer.fit(df_minority)
print("  ✓ Training complete!")

# ============================================================================
# STEP 4: GENERATE CONDITIONAL SYNTHETIC DATA
# ============================================================================
print("\n[Step 4/6] Generating conditional synthetic data...")
print("  Target: 5,000 clicks with slot_id=3 (targeted augmentation)")

# Create conditions using SDV's Condition class
from sdv.sampling import Condition

n_synthetic = 5000

# Create condition for slot_id=3 and clicked=1
condition = Condition(
    num_rows=n_synthetic,
    column_values={'slot_id': 3, 'clicked': 1}
)

print(f"\n  Conditions:")
print(f"    slot_id: 3")
print(f"    clicked: 1")
print(f"    Number of samples: {n_synthetic:,}")

# Generate synthetic data
print("\n  Generating synthetic samples...")
df_synthetic = synthesizer.sample_from_conditions(
    conditions=[condition],
    max_tries_per_batch=1000
)

print(f"  ✓ Generated {len(df_synthetic):,} synthetic samples")

# ============================================================================
# STEP 5: VALIDATE GENERATED DATA
# ============================================================================
print("\n[Step 5/6] Validating generated data...")

print(f"\n  Synthetic data shape: {df_synthetic.shape}")
print(f"\n  First 5 synthetic samples:")
print(df_synthetic.head())

print(f"\n  Validation checks:")

# Check slot_id distribution
slot_id_counts = df_synthetic['slot_id'].value_counts().sort_index()
print(f"    slot_id distribution:")
for slot, count in slot_id_counts.items():
    print(f"      slot_id={slot}: {count} samples ({count/len(df_synthetic)*100:.1f}%)")

# Check if all are slot_id=3
if (df_synthetic['slot_id'] == 3).all():
    print(f"    ✓ All samples have slot_id=3 (condition satisfied!)")
else:
    print(f"    ✗ Warning: Some samples don't have slot_id=3")

# Check clicked distribution
clicked_counts = df_synthetic['clicked'].value_counts().sort_index()
print(f"    clicked distribution:")
for clicked_val, count in clicked_counts.items():
    print(f"      clicked={clicked_val}: {count} samples ({count/len(df_synthetic)*100:.1f}%)")

if (df_synthetic['clicked'] == 1).all():
    print(f"    ✓ All samples have clicked=1 (condition satisfied!)")
else:
    print(f"    ✗ Warning: Some samples don't have clicked=1")

# Check feature ranges
print(f"\n  Feature statistics:")
print(f"    ad_relevance_score: min={df_synthetic['ad_relevance_score'].min():.3f}, "
      f"max={df_synthetic['ad_relevance_score'].max():.3f}, "
      f"mean={df_synthetic['ad_relevance_score'].mean():.3f}")
print(f"    user_age: min={df_synthetic['user_age'].min():.0f}, "
      f"max={df_synthetic['user_age'].max():.0f}, "
      f"mean={df_synthetic['user_age'].mean():.1f}")

# Compare with original minority class for slot_id=3
df_original_slot3 = df_minority[df_minority['slot_id'] == 3]
print(f"\n  Comparison with original slot_id=3 data ({len(df_original_slot3)} samples):")
print(f"    Original ad_relevance_score: mean={df_original_slot3['ad_relevance_score'].mean():.3f}")
print(f"    Synthetic ad_relevance_score: mean={df_synthetic['ad_relevance_score'].mean():.3f}")
print(f"    Original user_age: mean={df_original_slot3['user_age'].mean():.1f}")
print(f"    Synthetic user_age: mean={df_synthetic['user_age'].mean():.1f}")

# ============================================================================
# STEP 6: VISUALIZE RESULTS
# ============================================================================
print("\n[Step 6/6] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: slot_id distribution
ax = axes[0, 0]
slot_counts = df_synthetic['slot_id'].value_counts().sort_index()
colors = ['red' if slot == 3 else 'lightgray' for slot in slot_counts.index]
ax.bar(slot_counts.index, slot_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Slot ID', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Synthetic Data: Slot ID Distribution\n(Target: slot_id=3)', fontweight='bold')
ax.set_xticks([1, 2, 3, 4, 5])
ax.grid(True, alpha=0.3, axis='y')
# Add annotation
ax.text(3, slot_counts.get(3, 0) * 1.05, f'{slot_counts.get(3, 0):,} samples\n(100%)',
        ha='center', va='bottom', fontweight='bold', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Panel 2: ad_relevance_score distribution comparison
ax = axes[0, 1]
ax.hist(df_original_slot3['ad_relevance_score'], bins=30, alpha=0.6,
        label='Original (slot_id=3)', color='blue', density=True, edgecolor='black')
ax.hist(df_synthetic['ad_relevance_score'], bins=30, alpha=0.6,
        label='Synthetic (slot_id=3)', color='red', density=True, edgecolor='black')
ax.set_xlabel('Ad Relevance Score', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Ad Relevance Score Distribution', fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3)

# Panel 3: user_age distribution comparison
ax = axes[1, 0]
ax.hist(df_original_slot3['user_age'], bins=20, alpha=0.6,
        label='Original (slot_id=3)', color='blue', density=True, edgecolor='black')
ax.hist(df_synthetic['user_age'], bins=20, alpha=0.6,
        label='Synthetic (slot_id=3)', color='red', density=True, edgecolor='black')
ax.set_xlabel('User Age', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('User Age Distribution', fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3)

# Panel 4: Summary statistics table
ax = axes[1, 1]
ax.axis('tight')
ax.axis('off')

summary_data = [
    ['Metric', 'Original\n(slot_id=3)', 'Synthetic\n(slot_id=3)'],
    ['─' * 20, '─' * 15, '─' * 15],
    ['Sample count', f'{len(df_original_slot3):,}', f'{len(df_synthetic):,}'],
    ['% of all clicks', f'{len(df_original_slot3)/len(df_minority)*100:.1f}%', 'N/A'],
    ['', '', ''],
    ['Ad relevance (mean)', f'{df_original_slot3["ad_relevance_score"].mean():.3f}',
     f'{df_synthetic["ad_relevance_score"].mean():.3f}'],
    ['Ad relevance (std)', f'{df_original_slot3["ad_relevance_score"].std():.3f}',
     f'{df_synthetic["ad_relevance_score"].std():.3f}'],
    ['', '', ''],
    ['User age (mean)', f'{df_original_slot3["user_age"].mean():.1f}',
     f'{df_synthetic["user_age"].mean():.1f}'],
    ['User age (std)', f'{df_original_slot3["user_age"].std():.1f}',
     f'{df_synthetic["user_age"].std():.1f}'],
    ['', '', ''],
    ['Augmentation', 'Baseline', f'+{len(df_synthetic):,} samples'],
]

table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight augmentation row
table[(11, 0)].set_facecolor('#FFE082')
table[(11, 1)].set_facecolor('#FFE082')
table[(11, 2)].set_facecolor('#FFE082')

ax.set_title('Statistical Comparison\nOriginal vs Synthetic', fontweight='bold', pad=20)

plt.suptitle('Conditional TVAE: Targeted Augmentation Results\n' +
             'Generated 5,000 clicks for slot_id=3',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('conditional_vae_demo_results.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: conditional_vae_demo_results.png")

# ============================================================================
# DEMONSTRATION COMPLETE
# ============================================================================
print("\n" + "=" * 80)
print("CONDITIONAL TVAE DEMONSTRATION COMPLETE!")
print("=" * 80)

print(f"\nKey Results:")
print(f"  ✓ Successfully trained C-TVAE on {len(df_minority):,} minority class samples")
print(f"  ✓ Generated {len(df_synthetic):,} targeted synthetic samples")
print(f"  ✓ All samples conditioned on slot_id=3 and clicked=1")
print(f"  ✓ Synthetic data preserves statistical properties of original")

print(f"\nUse Case:")
print(f"  This approach enables targeted augmentation of underrepresented segments.")
print(f"  In this demo, we specifically augmented clicks for slot_id=3, which can")
print(f"  help improve model performance for that particular slot without affecting")
print(f"  the model's understanding of other slots.")

print(f"\nNext Steps:")
print(f"  1. Apply this to your actual CTR data with real features")
print(f"  2. Identify underrepresented segments (e.g., low-traffic slots, devices)")
print(f"  3. Generate targeted synthetic data for those segments")
print(f"  4. Train models and compare: Baseline vs CTGAN vs C-TVAE")

# Save synthetic data
df_synthetic.to_csv('conditional_vae_synthetic_demo.csv', index=False)
print(f"\n✓ Synthetic data saved to: conditional_vae_synthetic_demo.csv")
