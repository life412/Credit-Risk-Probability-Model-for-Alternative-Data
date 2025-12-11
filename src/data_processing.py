# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Load raw CSV data"""
    df = pd.read_csv(file_path)
    return df

def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate numeric features per CustomerId"""
    agg_df = df.groupby('CustomerId').agg(
        total_transaction_amount=('Amount', 'sum'),
        avg_transaction_amount=('Amount', 'mean'),
        transaction_count=('Amount', 'count'),
        std_transaction_amount=('Amount', 'std')
    ).reset_index()
    return agg_df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, month, year from TransactionStartTime"""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    return df

def preprocess_and_save(df: pd.DataFrame, save_path: str):
    """Full preprocessing pipeline and save processed dataset"""
    # Aggregate features
    agg_df = create_aggregate_features(df)
    
    # Merge aggregate features back to original df
    df = df.merge(agg_df, on='CustomerId', how='left')
    
    # Time features
    df = create_time_features(df)
    
    # Save processed dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")

if __name__ == "__main__":
    raw_file = "../data/raw/data.csv"
    processed_file = "../data/processed/processed_data.csv"
    df = load_data(raw_file)
    preprocess_and_save(df, processed_file)
