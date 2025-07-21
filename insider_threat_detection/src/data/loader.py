"""Data loading utilities for insider threat detection."""

import pandas as pd
import os
import sys
from typing import List, Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import DATA_PATH, DATA_FILES, COMMON_COLUMNS
from src.utils.logger import get_logger


class DataLoader:
    """Handles loading and merging of CSV data files."""
    
    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.logger = get_logger("data_loader")
        self.merged_df = None
    
    def load_single_file(self, log_type: str, possible_filenames: List[str]) -> Optional[pd.DataFrame]:
        """Load a single CSV file with multiple possible names."""
        for filename in possible_filenames:
            file_path = os.path.join(self.data_path, filename)
            try:
                df = pd.read_csv(file_path)
                df['activity_type'] = log_type.upper()
                
                # Standardize column names
                if 'url' in df.columns:
                    df = df.rename(columns={'url': 'details'})
                elif 'activity' in df.columns:
                    df = df.rename(columns={'activity': 'details'})
                
                self.logger.info(f"{log_type.upper()} logs loaded from {filename}: {len(df)} records")
                return df
                
            except FileNotFoundError:
                self.logger.debug(f"File not found: {filename}")
                continue
            except Exception as e:
                self.logger.error(f"Error loading {filename}: {e}")
                continue
        
        self.logger.warning(f"No {log_type} logs found. Tried: {possible_filenames}")
        return None
    
    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe has consistent structure."""
        # Ensure all common columns exist
        for col in COMMON_COLUMNS:
            if col not in df.columns:
                df[col] = 'unknown'
        
        # Return only common columns in consistent order
        return df[COMMON_COLUMNS].copy()
    
    def load_and_merge_data(self) -> pd.DataFrame:
        """Load and merge all CSV files."""
        self.logger.info("Starting data loading process...")
        
        dataframes = []
        
        # Load each type of log file
        for log_type, possible_names in DATA_FILES.items():
            df = self.load_single_file(log_type, possible_names)
            if df is not None:
                # Standardize the dataframe structure
                df_standardized = self.standardize_dataframe(df)
                dataframes.append(df_standardized)
        
        if not dataframes:
            raise ValueError("No data files found. Please check your file paths and names.")
        
        # Merge all dataframes
        self.merged_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Total merged records: {len(self.merged_df)}")
        
        # Remove duplicates
        initial_len = len(self.merged_df)
        self.merged_df = self.merged_df.drop_duplicates()
        duplicates_removed = initial_len - len(self.merged_df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate records")
        
        self.logger.info("Data loading completed successfully")
        return self.merged_df
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the loaded data."""
        if self.merged_df is None:
            return {"error": "No data loaded"}
        
        summary = {
            "total_records": len(self.merged_df),
            "unique_users": self.merged_df['user'].nunique(),
            "unique_pcs": self.merged_df['pc'].nunique(),
            "activity_types": self.merged_df['activity_type'].value_counts().to_dict(),
            "date_range": {
                "start": str(self.merged_df['date'].min()) if 'date' in self.merged_df.columns else "N/A",
                "end": str(self.merged_df['date'].max()) if 'date' in self.merged_df.columns else "N/A"
            },
            "missing_values": self.merged_df.isnull().sum().to_dict()
        }
        
        return summary
    
    def validate_data_quality(self) -> Dict:
        """Validate the quality of loaded data."""
        if self.merged_df is None:
            return {"error": "No data loaded"}
        
        validation_results = {
            "total_records": len(self.merged_df),
            "issues": []
        }
        
        # Check for missing critical columns
        for col in COMMON_COLUMNS:
            if col not in self.merged_df.columns:
                validation_results["issues"].append(f"Missing column: {col}")
        
        # Check for high percentage of missing values
        for col in self.merged_df.columns:
            missing_pct = (self.merged_df[col].isnull().sum() / len(self.merged_df)) * 100
            if missing_pct > 50:
                validation_results["issues"].append(f"High missing values in {col}: {missing_pct:.1f}%")
        
        # Check for unknown dates
        if 'date' in self.merged_df.columns:
            unknown_dates = sum(self.merged_df['date'] == 'unknown')
            if unknown_dates > 0:
                validation_results["issues"].append(f"Unknown dates found: {unknown_dates} records")
        
        # Check data types
        if 'date' in self.merged_df.columns:
            try:
                pd.to_datetime(self.merged_df['date'], errors='raise')
            except:
                validation_results["issues"].append("Date column contains invalid date formats")
        
        validation_results["is_valid"] = len(validation_results["issues"]) == 0
        
        return validation_results
