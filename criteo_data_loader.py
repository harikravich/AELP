"""
Criteo Display Advertising Challenge Dataset Loader for GAELP

This module provides comprehensive data loading, preprocessing, and analysis
capabilities for the Criteo dataset to support CTR prediction tasks in the
reinforcement learning advertising environment.

Key Features:
- Load and parse Criteo dataset with proper type handling
- Feature engineering and preprocessing
- Train/validation/test splits
- Statistical analysis for simulator calibration
- Integration with GAELP BigQuery storage architecture
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CriteoDatasetStats:
    """Statistics and metadata for Criteo dataset"""
    total_samples: int
    num_features: int
    cat_features: int
    click_rate: float
    numerical_stats: Dict[str, Dict[str, float]]
    categorical_stats: Dict[str, Dict[str, Any]]
    missing_values: Dict[str, int]
    feature_correlations: Dict[str, float]

@dataclass
class DataSplits:
    """Container for train/validation/test splits"""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_size: int
    val_size: int
    test_size: int

class CriteoDataLoader:
    """
    Comprehensive data loader for Criteo Display Advertising Challenge dataset
    
    Supports:
    - Data loading and preprocessing
    - Feature engineering
    - Statistical analysis
    - Train/validation/test splits
    - BigQuery integration preparation
    """
    
    def __init__(self, data_path: str = "/home/hariravichandran/AELP/data"):
        self.data_path = Path(data_path)
        self.criteo_file = self.data_path / "criteo_sample_data.csv"
        self.metadata_file = self.data_path / "metadata.json"
        
        # Criteo dataset schema
        self.numerical_features = [f"num_{i}" for i in range(13)]
        self.categorical_features = [f"cat_{i}" for i in range(26)]
        self.target_column = "click"
        
        # Loaded data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.stats: Optional[CriteoDatasetStats] = None
        self.splits: Optional[DataSplits] = None
        
        # Preprocessing objects
        self.numerical_scaler = StandardScaler()
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw Criteo dataset from CSV"""
        try:
            logger.info(f"Loading Criteo data from {self.criteo_file}")
            self.raw_data = pd.read_csv(self.criteo_file)
            logger.info(f"Loaded {len(self.raw_data)} samples with {len(self.raw_data.columns)} features")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality and missing values"""
        if self.raw_data is None:
            self.load_raw_data()
        
        quality_report = {
            'total_samples': len(self.raw_data),
            'total_features': len(self.raw_data.columns),
            'missing_values': {},
            'data_types': {},
            'value_ranges': {}
        }
        
        # Check for missing values
        for col in self.raw_data.columns:
            missing_count = self.raw_data[col].isnull().sum()
            quality_report['missing_values'][col] = missing_count
            quality_report['data_types'][col] = str(self.raw_data[col].dtype)
            
            if col in self.numerical_features:
                quality_report['value_ranges'][col] = {
                    'min': float(self.raw_data[col].min()),
                    'max': float(self.raw_data[col].max()),
                    'mean': float(self.raw_data[col].mean()),
                    'std': float(self.raw_data[col].std())
                }
            elif col in self.categorical_features:
                quality_report['value_ranges'][col] = {
                    'unique_values': int(self.raw_data[col].nunique()),
                    'most_frequent': str(self.raw_data[col].mode().iloc[0]),
                    'top_5_values': list(self.raw_data[col].value_counts().head().index.astype(str))
                }
        
        return quality_report
    
    def preprocess_features(self) -> pd.DataFrame:
        """Preprocess numerical and categorical features"""
        if self.raw_data is None:
            self.load_raw_data()
        
        logger.info("Starting feature preprocessing...")
        processed_data = self.raw_data.copy()
        
        # Handle missing values
        # For numerical features, fill with median
        for col in self.numerical_features:
            if col in processed_data.columns:
                median_val = processed_data[col].median()
                processed_data[col].fillna(median_val, inplace=True)
        
        # For categorical features, fill with mode or create 'unknown' category
        for col in self.categorical_features:
            if col in processed_data.columns:
                processed_data[col].fillna('unknown', inplace=True)
                processed_data[col] = processed_data[col].astype(str)
        
        # Scale numerical features
        if self.numerical_features:
            available_num_features = [col for col in self.numerical_features if col in processed_data.columns]
            if available_num_features:
                processed_data[available_num_features] = self.numerical_scaler.fit_transform(
                    processed_data[available_num_features]
                )
        
        # Encode categorical features
        for col in self.categorical_features:
            if col in processed_data.columns:
                encoder = LabelEncoder()
                processed_data[col] = encoder.fit_transform(processed_data[col])
                self.categorical_encoders[col] = encoder
        
        self.processed_data = processed_data
        logger.info("Feature preprocessing completed")
        return processed_data
    
    def compute_statistics(self) -> CriteoDatasetStats:
        """Compute comprehensive dataset statistics"""
        if self.processed_data is None:
            self.preprocess_features()
        
        logger.info("Computing dataset statistics...")
        
        # Basic stats
        total_samples = len(self.processed_data)
        num_features = len(self.numerical_features)
        cat_features = len(self.categorical_features)
        click_rate = self.processed_data[self.target_column].mean()
        
        # Numerical feature statistics
        numerical_stats = {}
        for col in self.numerical_features:
            if col in self.processed_data.columns:
                numerical_stats[col] = {
                    'mean': float(self.processed_data[col].mean()),
                    'std': float(self.processed_data[col].std()),
                    'min': float(self.processed_data[col].min()),
                    'max': float(self.processed_data[col].max()),
                    'median': float(self.processed_data[col].median()),
                    'q25': float(self.processed_data[col].quantile(0.25)),
                    'q75': float(self.processed_data[col].quantile(0.75))
                }
        
        # Categorical feature statistics
        categorical_stats = {}
        for col in self.categorical_features:
            if col in self.processed_data.columns:
                value_counts = self.processed_data[col].value_counts()
                categorical_stats[col] = {
                    'unique_count': int(self.processed_data[col].nunique()),
                    'most_frequent': int(value_counts.index[0]),
                    'most_frequent_count': int(value_counts.iloc[0]),
                    'entropy': float(-np.sum((value_counts / len(self.processed_data)) * 
                                           np.log2(value_counts / len(self.processed_data) + 1e-10)))
                }
        
        # Missing values
        missing_values = {col: int(self.raw_data[col].isnull().sum()) 
                         for col in self.raw_data.columns}
        
        # Feature correlations with target
        feature_correlations = {}
        for col in self.numerical_features + self.categorical_features:
            if col in self.processed_data.columns:
                correlation = float(self.processed_data[col].corr(self.processed_data[self.target_column]))
                feature_correlations[col] = correlation if not np.isnan(correlation) else 0.0
        
        self.stats = CriteoDatasetStats(
            total_samples=total_samples,
            num_features=num_features,
            cat_features=cat_features,
            click_rate=click_rate,
            numerical_stats=numerical_stats,
            categorical_stats=categorical_stats,
            missing_values=missing_values,
            feature_correlations=feature_correlations
        )
        
        logger.info(f"Statistics computed - CTR: {click_rate:.4f}, Samples: {total_samples}")
        return self.stats
    
    def create_data_splits(self, train_size: float = 0.7, val_size: float = 0.15, 
                          test_size: float = 0.15, random_state: int = 42) -> DataSplits:
        """Create train/validation/test splits"""
        if self.processed_data is None:
            self.preprocess_features()
        
        # Validate split sizes
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"
        
        logger.info(f"Creating data splits: train={train_size}, val={val_size}, test={test_size}")
        
        # Prepare features and target
        feature_columns = [col for col in self.processed_data.columns if col != self.target_column]
        X = self.processed_data[feature_columns]
        y = self.processed_data[self.target_column]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        relative_val_size = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, 
            random_state=random_state, stratify=y_temp
        )
        
        self.splits = DataSplits(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            train_size=len(X_train), val_size=len(X_val), test_size=len(X_test)
        )
        
        logger.info(f"Data splits created - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return self.splits
    
    def generate_simulator_calibration_data(self) -> Dict[str, Any]:
        """Generate statistics for calibrating the advertising simulator"""
        if self.stats is None:
            self.compute_statistics()
        
        calibration_data = {
            'ctr_statistics': {
                'baseline_ctr': self.stats.click_rate,
                'ctr_std': np.sqrt(self.stats.click_rate * (1 - self.stats.click_rate)),
                'high_ctr_threshold': self.stats.click_rate + 2 * np.sqrt(self.stats.click_rate * (1 - self.stats.click_rate)),
                'low_ctr_threshold': max(0, self.stats.click_rate - 2 * np.sqrt(self.stats.click_rate * (1 - self.stats.click_rate)))
            },
            'feature_importance': {
                'top_numerical_features': sorted(
                    [(k, abs(v)) for k, v in self.stats.feature_correlations.items() 
                     if k in self.numerical_features],
                    key=lambda x: x[1], reverse=True
                )[:5],
                'top_categorical_features': sorted(
                    [(k, abs(v)) for k, v in self.stats.feature_correlations.items() 
                     if k in self.categorical_features],
                    key=lambda x: x[1], reverse=True
                )[:5]
            },
            'data_distribution': {
                'numerical_ranges': {k: {'min': v['min'], 'max': v['max'], 'mean': v['mean']} 
                                   for k, v in self.stats.numerical_stats.items()},
                'categorical_diversity': {k: v['unique_count'] 
                                        for k, v in self.stats.categorical_stats.items()}
            },
            'simulation_parameters': {
                'noise_level': 0.1,  # Based on feature variance
                'correlation_strength': np.mean([abs(v) for v in self.stats.feature_correlations.values()]),
                'recommended_episode_length': 1000,  # Based on data size
                'recommended_batch_size': 64
            }
        }
        
        return calibration_data
    
    def save_processed_data(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Save processed data and statistics to files"""
        if output_dir is None:
            output_dir = self.data_path
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save processed data
        if self.processed_data is not None:
            processed_file = output_dir / "criteo_processed.csv"
            self.processed_data.to_csv(processed_file, index=False)
            saved_files['processed_data'] = str(processed_file)
        
        # Save data splits
        if self.splits is not None:
            splits_dir = output_dir / "splits"
            splits_dir.mkdir(exist_ok=True)
            
            self.splits.X_train.to_csv(splits_dir / "X_train.csv", index=False)
            self.splits.X_val.to_csv(splits_dir / "X_val.csv", index=False)
            self.splits.X_test.to_csv(splits_dir / "X_test.csv", index=False)
            self.splits.y_train.to_csv(splits_dir / "y_train.csv", index=False)
            self.splits.y_val.to_csv(splits_dir / "y_val.csv", index=False)
            self.splits.y_test.to_csv(splits_dir / "y_test.csv", index=False)
            
            saved_files['splits'] = str(splits_dir)
        
        # Save statistics
        if self.stats is not None:
            stats_file = output_dir / "criteo_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(asdict(self.stats), f, indent=2)
            saved_files['statistics'] = str(stats_file)
        
        # Save calibration data
        calibration_data = self.generate_simulator_calibration_data()
        calibration_file = output_dir / "simulator_calibration.json"
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        saved_files['calibration'] = str(calibration_file)
        
        logger.info(f"Saved processed data to {output_dir}")
        return saved_files
    
    def get_bigquery_schema(self) -> List[Dict[str, str]]:
        """Generate BigQuery schema for the processed data"""
        schema = []
        
        # Target column
        schema.append({
            'name': self.target_column,
            'type': 'INTEGER',
            'mode': 'REQUIRED',
            'description': 'Click indicator (0 or 1)'
        })
        
        # Numerical features
        for col in self.numerical_features:
            schema.append({
                'name': col,
                'type': 'FLOAT',
                'mode': 'NULLABLE',
                'description': f'Normalized numerical feature {col}'
            })
        
        # Categorical features
        for col in self.categorical_features:
            schema.append({
                'name': col,
                'type': 'INTEGER',
                'mode': 'NULLABLE',
                'description': f'Encoded categorical feature {col}'
            })
        
        return schema
    
    def export_for_bigquery(self, table_name: str = "criteo_training_data") -> str:
        """Export data in BigQuery-compatible format"""
        if self.processed_data is None:
            self.preprocess_features()
        
        export_file = self.data_path / f"{table_name}.json"
        
        # Convert to JSONL format for BigQuery import
        with open(export_file, 'w') as f:
            for _, row in self.processed_data.iterrows():
                json.dump(row.to_dict(), f)
                f.write('\n')
        
        logger.info(f"Exported data for BigQuery import: {export_file}")
        return str(export_file)


def main():
    """Main function to demonstrate the data loader capabilities"""
    print("=== GAELP Criteo Data Loader Demo ===\n")
    
    # Initialize data loader
    loader = CriteoDataLoader()
    
    # Load and analyze raw data
    print("1. Loading raw data...")
    raw_data = loader.load_raw_data()
    print(f"   Loaded {len(raw_data)} samples with {len(raw_data.columns)} features")
    
    # Analyze data quality
    print("\n2. Analyzing data quality...")
    quality_report = loader.analyze_data_quality()
    print(f"   Total samples: {quality_report['total_samples']}")
    print(f"   Total features: {quality_report['total_features']}")
    print(f"   Missing values: {sum(quality_report['missing_values'].values())} total")
    
    # Preprocess features
    print("\n3. Preprocessing features...")
    processed_data = loader.preprocess_features()
    print(f"   Processed data shape: {processed_data.shape}")
    
    # Compute statistics
    print("\n4. Computing statistics...")
    stats = loader.compute_statistics()
    print(f"   Click-through rate: {stats.click_rate:.4f}")
    print(f"   Numerical features: {stats.num_features}")
    print(f"   Categorical features: {stats.cat_features}")
    
    # Create data splits
    print("\n5. Creating data splits...")
    splits = loader.create_data_splits()
    print(f"   Train: {splits.train_size} samples")
    print(f"   Validation: {splits.val_size} samples")
    print(f"   Test: {splits.test_size} samples")
    
    # Generate calibration data
    print("\n6. Generating simulator calibration data...")
    calibration_data = loader.generate_simulator_calibration_data()
    print(f"   Baseline CTR: {calibration_data['ctr_statistics']['baseline_ctr']:.4f}")
    print(f"   Top numerical feature: {calibration_data['feature_importance']['top_numerical_features'][0]}")
    
    # Save processed data
    print("\n7. Saving processed data...")
    saved_files = loader.save_processed_data()
    print("   Saved files:")
    for key, path in saved_files.items():
        print(f"     {key}: {path}")
    
    # Generate BigQuery schema
    print("\n8. Generating BigQuery schema...")
    schema = loader.get_bigquery_schema()
    print(f"   Generated schema with {len(schema)} fields")
    
    print("\n=== Data Loading Complete ===")
    return loader


if __name__ == "__main__":
    loader = main()