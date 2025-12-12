"""
Generate CTGAN synthetic data for CTR prediction
Saves synthetic data as .pkl for efficient reuse
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os

print("="*70)
print("CTGAN SYNTHETIC DATA GENERATION")
print("="*70)

# Check if synthetic data already exists
SYNTHETIC_FILE = 'ctgan_synthetic_data.pkl'

if os.path.exists(SYNTHETIC_FILE):
    print(f"\n✓ Synthetic data already exists: {SYNTHETIC_FILE}")
    print("  Loading existing synthetic data...")
    with open(SYNTHETIC_FILE, 'rb') as f:
        synthetic_data = pickle.load(f)
    print(f"  Loaded {len(synthetic_data):,} synthetic samples")
else:
    print(f"\n✗ Synthetic data not found: {SYNTHETIC_FILE}")
    print("  Generating new synthetic data using CTGAN...")

    # Install CTGAN if needed
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        print("\n  Installing CTGAN (sdv library)...")
        import subprocess
        subprocess.check_call(['pip', 'install', '-q', 'sdv'])
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata

    # Load training data
    print("\n[1/4] Loading training data...")
    with open('train_encoded.pkl', 'rb') as f:
        train_encoded = pickle.load(f)

    print(f"  Original training data: {train_encoded.shape}")
    print(f"  Label distribution:\n{train_encoded['label'].value_counts()}")

    # Prepare data for CTGAN (focus on minority class)
    print("\n[2/4] Preparing data for CTGAN...")

    # Keep only numeric columns for CTGAN
    drop_cols = ['user_id', 'log_id']
    non_numeric_cols = train_encoded.select_dtypes(exclude=[np.number]).columns.tolist()

    ctgan_data = train_encoded.drop(columns=drop_cols + non_numeric_cols, errors='ignore')
    print(f"  CTGAN input shape: {ctgan_data.shape}")

    # Focus on minority class (clicks) for oversampling
    minority_data = ctgan_data[ctgan_data['label'] == 1].copy()
    print(f"  Minority class samples: {len(minority_data):,}")

    # Use a sample for faster training (CTGAN is slow on full data)
    if len(minority_data) > 50000:
        print(f"  Sampling 50,000 minority samples for CTGAN training...")
        minority_sample = minority_data.sample(n=50000, random_state=42)
    else:
        minority_sample = minority_data

    print(f"  Using {len(minority_sample):,} samples for CTGAN training")

    # Create metadata
    print("\n[3/4] Training CTGAN model...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(minority_sample)

    # Configure CTGAN
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=100,  # Reduced for speed
        batch_size=500,
        verbose=True
    )

    # Train CTGAN
    print("  Training CTGAN (this may take several minutes)...")
    synthesizer.fit(minority_sample)

    # Generate synthetic samples
    print("\n[4/4] Generating synthetic samples...")
    # Generate same number as minority class in original data
    n_synthetic = len(minority_data)
    print(f"  Generating {n_synthetic:,} synthetic samples...")

    synthetic_data = synthesizer.sample(num_rows=n_synthetic)

    # Ensure label is 1 (these are synthetic clicks)
    synthetic_data['label'] = 1

    # Save synthetic data
    print(f"\n  Saving synthetic data to {SYNTHETIC_FILE}...")
    with open(SYNTHETIC_FILE, 'wb') as f:
        pickle.dump(synthetic_data, f)

    # Save CTGAN model for future use
    print("  Saving CTGAN model to ctgan_model_trained.pkl...")
    with open('ctgan_model_trained.pkl', 'wb') as f:
        pickle.dump(synthesizer, f)

    print(f"\n✓ Generated and saved {len(synthetic_data):,} synthetic samples")

# Print synthetic data info
print("\n" + "="*70)
print("SYNTHETIC DATA SUMMARY")
print("="*70)
print(f"Shape: {synthetic_data.shape}")
print(f"Label distribution:\n{synthetic_data['label'].value_counts()}")
print(f"\nFirst few rows:")
print(synthetic_data.head())
print(f"\nColumn types:")
print(synthetic_data.dtypes.value_counts())

print("\n✓ Synthetic data ready for training!")
print(f"  File: {SYNTHETIC_FILE}")
print(f"  Size: {os.path.getsize(SYNTHETIC_FILE) / 1024 / 1024:.1f} MB")
