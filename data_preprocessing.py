"""
Data Preprocessing Script for CTR Prediction Project
Loads, merges, and encodes features from ads and feeds datasets.
Saves preprocessed data as pickles for model training.
"""

import numpy as np
import pandas as pd
import gc
import pickle


def to01(s):
    """Convert label from -1/1 to 0/1"""
    s = s.copy()
    s = s.replace({-1: 0, 1: 1})
    return s.astype(int)


def safe_keep(df, cols, name):
    """Safely keep columns, reporting missing ones"""
    keep = [c for c in cols if c in df.columns]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"[{name}] missing in the dataset: {miss}")
    return keep


def main():
    """Main preprocessing pipeline"""
    
    # Load data
    print("Loading data...")
    train_user = pd.read_csv('train_data_ads.csv')
    train_adv = pd.read_csv('train_data_feeds.csv')
    test_user = pd.read_csv('test_data_ads.csv')
    test_adv = pd.read_csv('test_data_feeds.csv')
    
    # Rename user_id column
    train_adv.rename(columns={'u_userId': 'user_id'}, inplace=True)
    test_adv.rename(columns={'u_userId': 'user_id'}, inplace=True)
    
    # Define variables to keep
    user_var = [
        'user_id', 'log_id', 'age', 'gender', 'residence', 'device_name',
        'device_size', 'net_type', 'task_id', 'creat_type_cd'
    ]
    adv_var = ['user_id', 'label']
    
    # Safe keep columns
    train_user_cols = safe_keep(train_user, user_var, "train_user")
    test_user_cols = safe_keep(test_user, user_var, "test_user")
    train_adv_cols = safe_keep(train_adv, adv_var, "train_adv")
    test_adv_cols = safe_keep(test_adv, adv_var, "test_adv")
    
    # Merge train and test
    train_user['istest'] = 0
    test_user['istest'] = 1
    data_user = pd.concat([train_user, test_user], axis=0, ignore_index=True)
    del train_user, test_user
    gc.collect()
    
    train_adv['istest'] = 0
    test_adv['istest'] = 1
    data_adv = pd.concat([train_adv, test_adv], axis=0, ignore_index=True)
    del train_adv, test_adv
    gc.collect()
    
    # Create user aggregation features from feeds data
    adv_train_only = data_adv[data_adv['istest'] == 0].copy()
    adv_train_only['label01'] = to01(adv_train_only['label'])
    
    user_agg = (
        adv_train_only
        .groupby('user_id', as_index=False)
        .agg(
            feeds_imps=('label01', 'count'),
            feeds_clicks=('label01', 'sum'),
            feeds_ctr=('label01', 'mean')
        )
    )
    
    # Merge user aggregates
    data_user = data_user.merge(user_agg, on='user_id', how='left')
    
    # Fill missing values
    for c in ['feeds_imps', 'feeds_clicks', 'feeds_ctr']:
        if c in data_user.columns:
            data_user[c] = data_user[c].fillna(0 if c != 'feeds_ctr' else data_user[c].mean())
    
    # Create final train and test sets
    train_merged = data_user[data_user['istest'] == 0].drop(columns=['istest']).reset_index(drop=True)
    test_merged = data_user[data_user['istest'] == 1].drop(columns=['istest']).reset_index(drop=True)

    # Drop adv_id to avoid ad-specific random effects (do not encode)
    for df in (train_merged, test_merged):
        if 'adv_id' in df.columns:
            df.drop(columns=['adv_id'], inplace=True)
    
    print(f"Train rows: {len(train_merged)} | Test rows: {len(test_merged)}")
    print(f"Train columns: {len(train_merged.columns)} | Test columns: {len(test_merged.columns)}")
    
    # =================== FEATURE ENCODING ===================
    
    # One-hot encode gender
    gender_dummies_train = pd.get_dummies(train_merged['gender'], prefix='gender', drop_first=True)
    gender_dummies_test = pd.get_dummies(test_merged['gender'], prefix='gender', drop_first=True)
    # Ensure test has same columns as train (fill missing with 0)
    gender_dummies_test = gender_dummies_test.reindex(columns=gender_dummies_train.columns, fill_value=0)
    train_encoded = train_merged.drop(columns=['gender']).reset_index(drop=True)
    test_encoded = test_merged.drop(columns=['gender']).reset_index(drop=True)
    train_encoded = pd.concat([train_encoded, gender_dummies_train], axis=1)
    test_encoded = pd.concat([test_encoded, gender_dummies_test], axis=1)
    
    # One-hot encode net_type
    net_type_dummies_train = pd.get_dummies(train_encoded['net_type'], prefix='net_type', drop_first=True)
    net_type_dummies_test = pd.get_dummies(test_encoded['net_type'], prefix='net_type', drop_first=True)
    # Ensure test has same columns as train (fill missing with 0)
    net_type_dummies_test = net_type_dummies_test.reindex(columns=net_type_dummies_train.columns, fill_value=0)
    train_encoded = train_encoded.drop(columns=['net_type']).reset_index(drop=True)
    test_encoded = test_encoded.drop(columns=['net_type']).reset_index(drop=True)
    train_encoded = pd.concat([train_encoded, net_type_dummies_train], axis=1)
    test_encoded = pd.concat([test_encoded, net_type_dummies_test], axis=1)
    
    # Target encoding for high cardinality features (adv_id removed to avoid random effects)
    target_encode_features = ['slot_id', 'device_name', 'task_id', 'city', 'adv_prim_id']
    
    for feat in target_encode_features:
        if feat in train_encoded.columns:
            encoding = train_merged.groupby(feat)['label'].mean()
            train_encoded[f'{feat}_encoded'] = train_encoded[feat].map(encoding).fillna(train_merged['label'].mean())
            test_encoded[f'{feat}_encoded'] = test_encoded[feat].map(encoding).fillna(train_merged['label'].mean())
            train_encoded = train_encoded.drop(columns=[feat])
            test_encoded = test_encoded.drop(columns=[feat])
    
    # Drop site_id (constant feature)
    train_encoded = train_encoded.drop(columns=['site_id'])
    test_encoded = test_encoded.drop(columns=['site_id'])
    
    # One-hot encode low cardinality features
    onehot_features = ['creat_type_cd', 'inter_type_cd', 'series_group']
    
    for feat in onehot_features:
        if feat in train_encoded.columns:
            dummies_train = pd.get_dummies(train_encoded[feat], prefix=feat, drop_first=True)
            dummies_test = pd.get_dummies(test_encoded[feat], prefix=feat, drop_first=True)
            # Ensure test has same columns as train (fill missing with 0)
            dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)
            train_encoded = train_encoded.drop(columns=[feat])
            test_encoded = test_encoded.drop(columns=[feat])
            train_encoded = pd.concat([train_encoded, dummies_train], axis=1)
            test_encoded = pd.concat([test_encoded, dummies_test], axis=1)
    
    # Target encode device_size
    if 'device_size' in train_encoded.columns:
        device_size_encoding = train_merged.groupby('device_size')['label'].mean()
        train_encoded['device_size_encoded'] = train_encoded['device_size'].map(device_size_encoding).fillna(train_merged['label'].mean())
        test_encoded['device_size_encoded'] = test_encoded['device_size'].map(device_size_encoding).fillna(train_merged['label'].mean())
        train_encoded = train_encoded.drop(columns=['device_size'])
        test_encoded = test_encoded.drop(columns=['device_size'])
    
    # Target encode remaining medium cardinality features
    medium_card_features = ['residence', 'series_dev', 'emui_dev', 'hispace_app_tags', 'app_second_class', 'spread_app_id']
    
    for feat in medium_card_features:
        if feat in train_encoded.columns:
            encoding = train_merged.groupby(feat)['label'].mean()
            train_encoded[f'{feat}_encoded'] = train_encoded[feat].map(encoding).fillna(train_merged['label'].mean())
            test_encoded[f'{feat}_encoded'] = test_encoded[feat].map(encoding).fillna(train_merged['label'].mean())
            train_encoded = train_encoded.drop(columns=[feat])
            test_encoded = test_encoded.drop(columns=[feat])
    
    # =================== INTERACTION FEATURES ===================
    # 1) engagement_by_slot = feeds_ctr * slot_id_encoded
    if 'feeds_ctr' in train_encoded.columns and 'slot_id_encoded' in train_encoded.columns:
        train_encoded['engagement_by_slot'] = train_encoded['feeds_ctr'] * train_encoded['slot_id_encoded']
        test_encoded['engagement_by_slot'] = test_encoded['feeds_ctr'] * test_encoded['slot_id_encoded']

    # 2) bandwidth_by_creative (fallback: bandwidth_by_slot)
    # Define high_bandwidth using available net_type one-hot columns (4/5/6/7 observed in data)
    def _compute_high_bandwidth(df):
        candidates = [c for c in ['net_type_4', 'net_type_5', 'net_type_6', 'net_type_7'] if c in df.columns]
        if not candidates:
            return None
        return df[candidates].sum(axis=1)

    hb_train = _compute_high_bandwidth(train_encoded)
    hb_test = _compute_high_bandwidth(test_encoded)

    if hb_train is not None and hb_test is not None:
        # Try to find a video creative dummy; if not present, fallback to bandwidth_by_slot
        video_cols = [c for c in train_encoded.columns if c.startswith('creat_type_') and 'video' in c.lower()]
        if video_cols:
            video_col = video_cols[0]
            train_encoded['bandwidth_by_creative'] = hb_train * train_encoded[video_col]
            test_encoded['bandwidth_by_creative'] = hb_test * test_encoded[video_col]
        elif 'slot_id_encoded' in train_encoded.columns:
            # Fallback per plan: bandwidth_by_slot
            train_encoded['bandwidth_by_slot'] = hb_train * train_encoded['slot_id_encoded']
            test_encoded['bandwidth_by_slot'] = hb_test * test_encoded['slot_id_encoded']

    print(f"\nFinal train shape: {train_encoded.shape}")
    print(f"Final test shape: {test_encoded.shape}")
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    with open('train_encoded.pkl', 'wb') as f:
        pickle.dump(train_encoded, f)
    
    with open('test_encoded.pkl', 'wb') as f:
        pickle.dump(test_encoded, f)
    
    print("Preprocessing complete! Files saved:")
    print("  - train_encoded.pkl")
    print("  - test_encoded.pkl")
    
    return train_encoded, test_encoded


if __name__ == '__main__':
    train_encoded, test_encoded = main()

