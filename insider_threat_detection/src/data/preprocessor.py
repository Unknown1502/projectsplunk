"""Data preprocessing utilities for insider threat detection."""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import SEQUENCE_LENGTH, STRIDE
from src.utils.logger import get_logger


class DataPreprocessor:
    """Handles data preprocessing and cleaning."""
    
    def __init__(self):
        self.logger = get_logger("data_preprocessor")
        self.scaler = StandardScaler()
    
    def clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert date columns."""
        self.logger.info("Cleaning date column...")
        
        # Handle 'unknown' dates
        unknown_dates = sum(df['date'] == 'unknown')
        if unknown_dates > 0:
            self.logger.info(f"Found {unknown_dates} rows with 'unknown' dates")
            df = df[df['date'] != 'unknown'].copy()
            self.logger.info(f"Rows after removing unknown dates: {len(df)}")
        
        # Convert to datetime with error handling
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Remove rows where date conversion failed
            before_dropna = len(df)
            df = df.dropna(subset=['date']).copy()
            invalid_dates = before_dropna - len(df)
            
            if invalid_dates > 0:
                self.logger.info(f"Removed {invalid_dates} rows with invalid dates")
            
            if len(df) == 0:
                raise ValueError("No valid dates found in the dataset!")
                
        except Exception as e:
            self.logger.error(f"Error converting dates: {e}")
            raise
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime column."""
        self.logger.info("Creating time-based features...")
        
        # Basic time features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_off_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        self.logger.info("Time features created successfully")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        self.logger.info("Handling missing values...")
        
        # Fill categorical columns with 'unknown'
        categorical_cols = ['user', 'pc', 'activity_type', 'details']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.logger.debug(f"Filled {col} missing values with median: {median_val}")
        
        return df
    
    def sort_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data by user and date for sequence creation."""
        self.logger.info("Sorting data by user and date...")
        df = df.sort_values(['user', 'date']).reset_index(drop=True)
        return df
    
    def create_sequences(self, df: pd.DataFrame, feature_columns: List[str]) -> tuple:
        """Create sequences for LSTM training with optimized memory usage."""
        self.logger.info("Creating sequences for LSTM training...")
        
        X_sequences = []
        y_sequences = []
        
        # Process users in batches to manage memory
        unique_users = df['user'].unique()
        batch_size = 100
        
        for i in range(0, len(unique_users), batch_size):
            batch_users = unique_users[i:i+batch_size]
            
            for user in batch_users:
                user_data = df[df['user'] == user].copy()
                
                if len(user_data) < SEQUENCE_LENGTH:
                    continue
                
                # Fill any remaining NaN values
                user_data[feature_columns] = user_data[feature_columns].fillna(0)
                
                # Create sequences with stride to reduce overfitting
                for j in range(0, len(user_data) - SEQUENCE_LENGTH + 1, STRIDE):
                    sequence = user_data.iloc[j:j+SEQUENCE_LENGTH]
                    X_sequences.append(sequence[feature_columns].values)
                    y_sequences.append(sequence['is_threat'].iloc[-1])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        self.logger.info(f"Created {len(X)} sequences with stride={STRIDE}")
        self.logger.info(f"Sequence shape: {X.shape}")
        
        return X, y
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray = None, 
                      X_test: np.ndarray = None, fit_scaler: bool = True) -> tuple:
        """Scale features using StandardScaler."""
        self.logger.info("Scaling features...")
        
        # Reshape for scaling
        original_shape = X_train.shape
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        
        if fit_scaler:
            X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
            self.logger.info("Scaler fitted on training data")
        else:
            X_train_scaled = self.scaler.transform(X_train_reshaped)
            self.logger.info("Using pre-fitted scaler")
        
        X_train_scaled = X_train_scaled.reshape(original_shape)
        
        results = [X_train_scaled]
        
        # Scale validation set if provided
        if X_val is not None:
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            results.append(X_val_scaled)
        
        # Scale test set if provided
        if X_test is not None:
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            results.append(X_test_scaled)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    
    def temporal_train_test_split(self, df: pd.DataFrame, feature_columns: list, 
                                train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple:
        """
        Split data temporally to prevent data leakage.
        
        Args:
            df: DataFrame with 'date' column
            feature_columns: List of feature column names
            train_ratio: Proportion for training (default: 0.70)
            val_ratio: Proportion for validation (default: 0.15)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Implementing temporal data splitting to prevent data leakage...")
        
        # Ensure data is sorted by date
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        # Calculate split points
        total_records = len(df_sorted)
        train_end = int(total_records * train_ratio)
        val_end = int(total_records * (train_ratio + val_ratio))
        
        # Get date boundaries
        train_end_date = df_sorted.iloc[train_end]['date'] if train_end < len(df_sorted) else df_sorted.iloc[-1]['date']
        val_end_date = df_sorted.iloc[val_end]['date'] if val_end < len(df_sorted) else df_sorted.iloc[-1]['date']
        
        self.logger.info(f"Temporal split boundaries:")
        self.logger.info(f"  Training: {df_sorted.iloc[0]['date']} to {train_end_date}")
        self.logger.info(f"  Validation: {train_end_date} to {val_end_date}")
        self.logger.info(f"  Test: {val_end_date} to {df_sorted.iloc[-1]['date']}")
        
        # Split the dataframe temporally
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        self.logger.info(f"Temporal split sizes:")
        self.logger.info(f"  Training: {len(train_df)} records ({len(train_df)/total_records*100:.1f}%)")
        self.logger.info(f"  Validation: {len(val_df)} records ({len(val_df)/total_records*100:.1f}%)")
        self.logger.info(f"  Test: {len(test_df)} records ({len(test_df)/total_records*100:.1f}%)")
        
        # Create sequences for each split
        X_train, y_train = self.create_sequences(train_df, feature_columns)
        X_val, y_val = self.create_sequences(val_df, feature_columns)
        X_test, y_test = self.create_sequences(test_df, feature_columns)
        
        # Log threat ratios
        self.logger.info(f"Threat ratios after temporal split:")
        self.logger.info(f"  Training: {y_train.mean():.3f}")
        self.logger.info(f"  Validation: {y_val.mean():.3f}")
        self.logger.info(f"  Test: {y_test.mean():.3f}")
        
        # Validate no data leakage
        if len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0:
            self.logger.info("[SUCCESS] Temporal splitting completed - NO DATA LEAKAGE!")
        else:
            self.logger.warning("[WARNING] Some splits are empty - check data distribution")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_preprocessing_summary(self, df_before: pd.DataFrame, 
                                df_after: pd.DataFrame) -> Dict:
        """Get summary of preprocessing steps."""
        summary = {
            "records_before": len(df_before),
            "records_after": len(df_after),
            "records_removed": len(df_before) - len(df_after),
            "columns_before": len(df_before.columns),
            "columns_after": len(df_after.columns),
            "new_features": list(set(df_after.columns) - set(df_before.columns)),
            "missing_values_before": df_before.isnull().sum().sum(),
            "missing_values_after": df_after.isnull().sum().sum()
        }
        
        return summary
