"""
Conditional TVAE for CTR Prediction - Production Implementation
Targeted Synthetic Data Generation for Underrepresented Segments

This script identifies underrepresented segments in the CTR data and
generates targeted synthetic click data to balance them.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
import warnings
import gc
warnings.filterwarnings('ignore')

print("=" * 80)
print("CONDITIONAL TVAE FOR CTR PREDICTION - PRODUCTION")
print("Targeted Augmentation for Underrepresented Segments")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[Step 1/7] Loading training data...")

with open('train_encoded.pkl', 'rb') as f:
    train_full = pickle.load(f)

print(f"  Full training data: {len(train_full):,} samples")

# Ensure label is 0/1
if train_full['label'].min() < 0:
    train_full['label'] = train_full['label'].replace({-1: 0, 1: 1}).astype(int)

# Get class distribution
click_rate = train_full['label'].mean()
n_clicks = train_full['label'].sum()
print(f"  Click rate: {click_rate*100:.2f}%")
print(f"  Clicks: {n_clicks:,}")
print(f"  No-clicks: {len(train_full) - n_clicks:,}")

# ============================================================================
# STEP 2: SUBSAMPLE FOR MEMORY EFFICIENCY
# ============================================================================
print("\n[Step 2/7] Subsampling for memory efficiency...")

# Subsample to 1M for faster processing
SUBSAMPLE_SIZE = 1_000_000
if len(train_full) > SUBSAMPLE_SIZE:
    train_data = train_full.sample(n=SUBSAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"  Subsampled to {len(train_data):,} samples")
else:
    train_data = train_full.copy()

del train_full
gc.collect()

# Update counts
n_clicks_sub = train_data['label'].sum()
click_rate_sub = n_clicks_sub / len(train_data)
print(f"  Subsample click rate: {click_rate_sub*100:.2f}%")
print(f"  Subsample clicks: {n_clicks_sub:,}")

# ============================================================================
# STEP 3: IDENTIFY UNDERREPRESENTED SEGMENTS
# ============================================================================
print("\n[Step 3/7] Analyzing segment distribution...")

# Extract minority class
df_clicks = train_data[train_data['label'] == 1].copy()
print(f"  Minority class size: {len(df_clicks):,}")

# Analyze distribution by key categorical features
# We'll focus on features that are likely to have underrepresented segments

# Check which categorical columns exist
categorical_features = []
potential_cats = ['slot_id', 'net_type', 'creat_type_cd', 'gender', 'city_rank',
                  'age', 'residence', 'series_group']

for col in potential_cats:
    if col in df_clicks.columns:
        n_unique = df_clicks[col].nunique()
        if 2 <= n_unique <= 20:  # Reasonable range for segmentation
            categorical_features.append(col)
            print(f"    Found categorical feature: {col} ({n_unique} unique values)")

# Choose primary segmentation feature (slot_id if available, otherwise first categorical)
if 'slot_id' in categorical_features:
    segment_feature = 'slot_id'
elif len(categorical_features) > 0:
    segment_feature = categorical_features[0]
else:
    print("  WARNING: No suitable categorical features found for segmentation")
    print("  Defaulting to generating general synthetic data")
    segment_feature = None

if segment_feature:
    print(f"\n  Primary segmentation feature: {segment_feature}")

    # Analyze distribution
    segment_dist = df_clicks[segment_feature].value_counts().sort_index()
    print(f"\n  Click distribution by {segment_feature}:")
    for seg, count in segment_dist.items():
        pct = count / len(df_clicks) * 100
        print(f"    {segment_feature}={seg}: {count:,} clicks ({pct:.1f}%)")

    # Identify underrepresented segments (< 10% of minority class)
    threshold = len(df_clicks) * 0.10
    underrep_segments = segment_dist[segment_dist < threshold].index.tolist()

    if len(underrep_segments) > 0:
        print(f"\n  Underrepresented segments (< 10% of clicks):")
        for seg in underrep_segments:
            count = segment_dist[seg]
            print(f"    {segment_feature}={seg}: {count:,} clicks ({count/len(df_clicks)*100:.1f}%)")

        # Focus on most underrepresented
        target_segment = underrep_segments[0]
        target_count = segment_dist[target_segment]
        print(f"\n  → Target segment for augmentation: {segment_feature}={target_segment}")
        print(f"    Current count: {target_count:,}")
    else:
        # Use least represented segment
        target_segment = segment_dist.idxmin()
        target_count = segment_dist[target_segment]
        print(f"\n  → Target segment (least represented): {segment_feature}={target_segment}")
        print(f"    Current count: {target_count:,}")

# ============================================================================
# STEP 4: PREPARE DATA FOR TVAE
# ============================================================================
print("\n[Step 4/7] Preparing data for TVAE...")

# Select numeric features only (TVAE works best with numeric data)
# Drop non-numeric columns
numeric_cols = df_clicks.select_dtypes(include=[np.number]).columns.tolist()

# Ensure label is included
if 'label' not in numeric_cols:
    numeric_cols.append('label')

# Ensure segment feature is included (if it's numeric or categorical with low cardinality)
if segment_feature and segment_feature not in numeric_cols:
    if df_clicks[segment_feature].dtype in [int, np.int64, np.int32]:
        numeric_cols.append(segment_feature)

# Remove identifier columns
id_cols = ['user_id', 'log_id', 'adv_id', 'task_id', 'device_name']
numeric_cols = [c for c in numeric_cols if c not in id_cols]

print(f"  Selected {len(numeric_cols)} numeric features")
print(f"  Features: {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")

# Prepare training data (minority class only)
df_train_vae = df_clicks[numeric_cols].copy()

# Handle NaN values
n_nan = df_train_vae.isna().sum().sum()
if n_nan > 0:
    print(f"  Found {n_nan:,} NaN values, filling with 0")
    df_train_vae = df_train_vae.fillna(0)

print(f"  Training data shape: {df_train_vae.shape}")
print(f"  Memory usage: {df_train_vae.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================================
# STEP 5: DEFINE METADATA AND TRAIN TVAE
# ============================================================================
print("\n[Step 5/7] Training Conditional TVAE...")

# Create metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_train_vae)

# Set categorical types for specific features
if segment_feature in df_train_vae.columns:
    metadata.update_column(column_name=segment_feature, sdtype='categorical')
metadata.update_column(column_name='label', sdtype='categorical')

# Additional categorical features
for col in ['gender', 'net_type', 'creat_type_cd', 'series_group']:
    if col in df_train_vae.columns:
        metadata.update_column(column_name=col, sdtype='categorical')

print(f"  Metadata configured")
print(f"  Training on {len(df_train_vae):,} click samples...")

# Initialize TVAE with reasonable hyperparameters
synthesizer = TVAESynthesizer(
    metadata,
    epochs=50,  # Reduced for faster training
    batch_size=256,
    embedding_dim=64,
    compress_dims=(64, 32),
    decompress_dims=(32, 64),
    verbose=False
)

print(f"  Training TVAE (epochs=50, this may take 2-5 minutes)...")
synthesizer.fit(df_train_vae)
print(f"  ✓ Training complete!")

# Save model
with open('conditional_tvae_model.pkl', 'wb') as f:
    pickle.dump(synthesizer, f)
print(f"  ✓ Model saved to 'conditional_tvae_model.pkl'")

# ============================================================================
# STEP 6: GENERATE CONDITIONAL SYNTHETIC DATA
# ============================================================================
print("\n[Step 6/7] Generating conditional synthetic data...")

if segment_feature:
    # Calculate how many synthetic samples to generate for the target segment
    # Goal: Bring underrepresented segment up to median representation
    median_count = int(segment_dist.median())
    n_to_generate = max(median_count - target_count, 1000)  # At least 1000

    # Ensure positive number
    if n_to_generate <= 0:
        n_to_generate = min(5000, target_count * 2)  # Generate 2x the current size, up to 5000

    print(f"  Target segment has {target_count:,} samples, median is {median_count:,}")
    print(f"  Generating {n_to_generate:,} samples for {segment_feature}={target_segment}")
    print(f"  This will bring segment from {target_count:,} to {target_count + n_to_generate:,} samples")

    # Create condition
    n_to_generate = int(n_to_generate)  # Ensure it's an integer
    assert n_to_generate > 0, f"n_to_generate must be positive, got {n_to_generate}"

    print(f"  Creating condition with num_rows={n_to_generate} (type: {type(n_to_generate)})")
    print(f"  Condition values: {segment_feature}={int(target_segment)}, label=1")

    # Ensure values are primitive Python types (not numpy)
    condition_values = {
        segment_feature: int(target_segment) if isinstance(target_segment, (int, np.integer)) else target_segment,
        'label': 1
    }

    condition = Condition(
        num_rows=n_to_generate,
        column_values=condition_values
    )

    # Generate
    print(f"  Generating...")
    df_synthetic = synthesizer.sample_from_conditions(
        conditions=[condition],
        max_tries_per_batch=5000
    )
else:
    # Generate general synthetic data
    n_to_generate = 5000
    print(f"  Generating {n_to_generate:,} general synthetic click samples...")

    condition = Condition(
        num_rows=n_to_generate,
        column_values={'label': 1}
    )

    df_synthetic = synthesizer.sample_from_conditions(
        conditions=[condition],
        max_tries_per_batch=5000
    )

print(f"  ✓ Generated {len(df_synthetic):,} synthetic samples")

# ============================================================================
# STEP 7: VALIDATE AND SAVE
# ============================================================================
print("\n[Step 7/7] Validating and saving synthetic data...")

print(f"\n  Synthetic data shape: {df_synthetic.shape}")
print(f"\n  First 5 samples:")
print(df_synthetic.head())

# Validate conditions
if segment_feature:
    segment_check = df_synthetic[segment_feature].value_counts()
    print(f"\n  {segment_feature} distribution in synthetic data:")
    for seg, count in segment_check.items():
        print(f"    {segment_feature}={seg}: {count:,} ({count/len(df_synthetic)*100:.1f}%)")

    if target_segment in segment_check.index:
        target_pct = segment_check[target_segment] / len(df_synthetic) * 100
        if target_pct > 80:
            print(f"  ✓ Target segment {target_segment} represents {target_pct:.1f}% (condition satisfied)")
        else:
            print(f"  ⚠ Target segment {target_segment} only represents {target_pct:.1f}%")

# Check label
label_check = df_synthetic['label'].value_counts()
print(f"\n  Label distribution:")
for label, count in label_check.items():
    print(f"    label={label}: {count:,} ({count/len(df_synthetic)*100:.1f}%)")

# Save synthetic data
with open('conditional_tvae_synthetic.pkl', 'wb') as f:
    pickle.dump(df_synthetic, f)
print(f"\n✓ Synthetic data saved to 'conditional_tvae_synthetic.pkl'")

df_synthetic.to_csv('conditional_tvae_synthetic.csv', index=False)
print(f"✓ Synthetic data saved to 'conditional_tvae_synthetic.csv'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("CONDITIONAL TVAE PRODUCTION - COMPLETE!")
print("=" * 80)

print(f"\nResults:")
print(f"  ✓ Trained C-TVAE on {len(df_train_vae):,} click samples")
print(f"  ✓ Generated {len(df_synthetic):,} targeted synthetic samples")
if segment_feature:
    print(f"  ✓ Focused on underrepresented segment: {segment_feature}={target_segment}")
    print(f"  ✓ Increased segment from {target_count:,} to {target_count + len(df_synthetic):,} samples")

print(f"\nNext Steps:")
print(f"  1. Combine synthetic data with original training data")
print(f"  2. Train logistic regression model with augmented data")
print(f"  3. Compare performance: Baseline vs CTGAN vs C-TVAE")
print(f"  4. Analyze improvement for target segment specifically")

print(f"\nFiles generated:")
print(f"  - conditional_tvae_model.pkl (trained model)")
print(f"  - conditional_tvae_synthetic.pkl (synthetic data)")
print(f"  - conditional_tvae_synthetic.csv (synthetic data CSV)")
