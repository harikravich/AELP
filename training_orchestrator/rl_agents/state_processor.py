"""
State Processing and Feature Engineering

Advanced state preprocessing for ad campaign optimization including
feature normalization, dimensionality reduction, and domain-specific
feature engineering.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StateProcessorConfig:
    """Configuration for state processor"""
    
    # Feature engineering
    enable_feature_engineering: bool = True
    include_interaction_features: bool = True
    include_temporal_features: bool = True
    
    # Normalization
    normalization_method: str = "standard"  # "standard", "robust", "minmax", "none"
    normalize_per_feature: bool = True
    
    # Dimensionality reduction
    enable_pca: bool = False
    pca_components: Optional[int] = None
    pca_variance_threshold: float = 0.95
    
    # Feature selection
    enable_feature_selection: bool = False
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01
    
    # Missing value handling
    fill_missing_values: bool = True
    missing_value_strategy: str = "median"  # "mean", "median", "mode", "zero"
    
    # Outlier handling
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0


class StateProcessor:
    """
    Advanced state processor for ad campaign environments.
    
    Handles feature engineering, normalization, dimensionality reduction,
    and other preprocessing steps to improve RL agent performance.
    """
    
    def __init__(self, config: StateProcessorConfig):
        self.config = config
        self.is_fitted = False
        
        # Preprocessing components
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        
        # Feature statistics
        self.feature_names = []
        self.feature_stats = {}
        self.feature_importance = {}
        
        self.logger = logging.getLogger(__name__)
        
        self._setup_preprocessors()
    
    def _setup_preprocessors(self):
        """Setup preprocessing components"""
        
        # Normalization
        if self.config.normalization_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.normalization_method == "robust":
            self.scaler = RobustScaler()
        elif self.config.normalization_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        # Dimensionality reduction
        if self.config.enable_pca:
            self.pca = PCA(
                n_components=self.config.pca_components,
                random_state=42
            )
    
    def fit(self, raw_states: List[Dict[str, Any]]):
        """
        Fit preprocessors on training data.
        
        Args:
            raw_states: List of raw state dictionaries from environment
        """
        self.logger.info("Fitting state processor on training data")
        
        # Extract features from raw states
        feature_matrix = []
        for state in raw_states:
            features = self._extract_features(state)
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        
        # Handle missing values
        if self.config.fill_missing_values:
            feature_matrix = self._handle_missing_values(feature_matrix, fit=True)
        
        # Handle outliers
        if self.config.handle_outliers:
            feature_matrix = self._handle_outliers(feature_matrix, fit=True)
        
        # Fit normalizer
        if self.scaler is not None:
            self.scaler.fit(feature_matrix)
            feature_matrix = self.scaler.transform(feature_matrix)
        
        # Fit PCA
        if self.config.enable_pca and self.pca is not None:
            self.pca.fit(feature_matrix)
            
            # Determine number of components based on variance threshold
            if self.config.pca_components is None:
                cumsum_var = np.cumsum(self.pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_var >= self.config.pca_variance_threshold) + 1
                
                # Refit with optimal number of components
                self.pca = PCA(n_components=n_components, random_state=42)
                self.pca.fit(feature_matrix)
        
        # Compute feature statistics
        self._compute_feature_statistics(feature_matrix)
        
        self.is_fitted = True
        self.logger.info(f"State processor fitted with {feature_matrix.shape[1]} features")
    
    def transform(self, raw_state: Dict[str, Any]) -> torch.Tensor:
        """
        Transform raw state to processed feature tensor.
        
        Args:
            raw_state: Raw state dictionary from environment
            
        Returns:
            Processed state tensor
        """
        if not self.is_fitted:
            raise ValueError("State processor must be fitted before transform")
        
        # Extract features
        features = self._extract_features(raw_state)
        features = features.reshape(1, -1)
        
        # Handle missing values
        if self.config.fill_missing_values:
            features = self._handle_missing_values(features, fit=False)
        
        # Handle outliers
        if self.config.handle_outliers:
            features = self._handle_outliers(features, fit=False)
        
        # Apply normalization
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Apply PCA
        if self.config.enable_pca and self.pca is not None:
            features = self.pca.transform(features)
        
        # Apply feature selection
        if self.config.enable_feature_selection and self.feature_selector is not None:
            features = features[:, self.selected_features]
        
        return torch.tensor(features.flatten(), dtype=torch.float32)
    
    def _extract_features(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from raw state"""
        
        features = []
        
        # Market context features
        market_context = raw_state.get("market_context", {})
        market_features = [
            market_context.get("competition_level", 0.5),
            market_context.get("seasonality_factor", 1.0),
            market_context.get("trend_momentum", 0.0),
            market_context.get("market_volatility", 0.1),
            market_context.get("economic_indicator", 1.0),
            market_context.get("consumer_confidence", 0.5)
        ]
        features.extend(market_features)
        
        # Historical performance features
        performance_history = raw_state.get("performance_history", {})
        perf_features = [
            performance_history.get("avg_roas", 1.0),
            performance_history.get("avg_ctr", 0.02),
            performance_history.get("avg_conversion_rate", 0.05),
            performance_history.get("total_spend", 0.0),
            performance_history.get("total_revenue", 0.0),
            performance_history.get("avg_cpc", 1.0),
            performance_history.get("avg_cpm", 10.0),
            performance_history.get("frequency", 2.0),
            performance_history.get("reach", 1000.0)
        ]
        features.extend(perf_features)
        
        # Budget and financial features
        budget_info = raw_state.get("budget_constraints", {})
        budget_features = [
            budget_info.get("daily_budget", 100.0),
            budget_info.get("remaining_budget", 100.0),
            budget_info.get("budget_utilization", 0.0),
            budget_info.get("cost_per_acquisition", 20.0),
            budget_info.get("lifetime_value", 100.0)
        ]
        features.extend(budget_features)
        
        # Demographic and persona features
        persona_data = raw_state.get("persona", {})
        demographics = persona_data.get("demographics", {})
        
        # Age group (ordinal encoding)
        age_group = demographics.get("age_group", "25-35")
        age_mapping = {"18-25": 0, "25-35": 1, "35-45": 2, "45-55": 3, "55-65": 4, "65+": 5}
        features.append(age_mapping.get(age_group, 1))
        
        # Income level (ordinal encoding)
        income = demographics.get("income", "medium")
        income_mapping = {"low": 0, "medium": 1, "high": 2}
        features.append(income_mapping.get(income, 1))
        
        # Gender (binary encoding)
        gender = demographics.get("gender", "unknown")
        features.extend([
            1.0 if gender == "male" else 0.0,
            1.0 if gender == "female" else 0.0
        ])
        
        # Interest categories (multi-hot encoding)
        interests = persona_data.get("interests", [])
        interest_categories = [
            "technology", "entertainment", "health", "finance", "travel",
            "food", "sports", "fashion", "education", "home", "automotive",
            "beauty", "books", "music", "gaming", "fitness"
        ]
        interest_features = [1.0 if cat in interests else 0.0 for cat in interest_categories]
        features.extend(interest_features)
        
        # Temporal features
        time_info = raw_state.get("time_context", {})
        temporal_features = [
            time_info.get("hour_of_day", 12),
            time_info.get("day_of_week", 3),
            time_info.get("day_of_month", 15),
            time_info.get("month", 6),
            time_info.get("quarter", 2),
            time_info.get("is_weekend", 0),
            time_info.get("is_holiday", 0)
        ]
        features.extend(temporal_features)
        
        # Previous action features
        prev_action = raw_state.get("previous_action", {})
        prev_features = [
            prev_action.get("budget", 50.0),
            prev_action.get("bid_amount", 5.0),
            prev_action.get("audience_size", 0.5),
            1.0 if prev_action.get("creative_type") == "video" else 0.0,
            1.0 if prev_action.get("creative_type") == "carousel" else 0.0,
            1.0 if prev_action.get("bid_strategy") == "cpm" else 0.0,
            1.0 if prev_action.get("bid_strategy") == "cpa" else 0.0
        ]
        features.extend(prev_features)
        
        # Campaign history aggregates
        campaign_history = raw_state.get("campaign_history", [])
        if campaign_history:
            recent_campaigns = campaign_history[-5:]  # Last 5 campaigns
            
            roas_values = [c.get("roas", 1.0) for c in recent_campaigns]
            ctr_values = [c.get("ctr", 0.02) for c in recent_campaigns]
            
            history_features = [
                np.mean(roas_values),
                np.std(roas_values),
                np.max(roas_values),
                np.min(roas_values),
                np.mean(ctr_values),
                np.std(ctr_values),
                len(recent_campaigns)
            ]
        else:
            history_features = [1.0, 0.0, 1.0, 1.0, 0.02, 0.0, 0.0]
        
        features.extend(history_features)
        
        # Feature engineering
        if self.config.enable_feature_engineering:
            engineered_features = self._engineer_features(raw_state, features)
            features.extend(engineered_features)
        
        return np.array(features, dtype=np.float32)
    
    def _engineer_features(self, raw_state: Dict[str, Any], 
                          base_features: List[float]) -> List[float]:
        """Engineer additional features from raw state and base features"""
        
        engineered = []
        
        # Interaction features
        if self.config.include_interaction_features:
            
            # Budget * Performance interactions
            budget = raw_state.get("budget_constraints", {}).get("daily_budget", 100.0)
            roas = raw_state.get("performance_history", {}).get("avg_roas", 1.0)
            ctr = raw_state.get("performance_history", {}).get("avg_ctr", 0.02)
            
            engineered.extend([
                budget * roas,
                budget * ctr,
                roas * ctr,
                budget / max(roas, 0.1),  # Budget efficiency
                roas / max(ctr, 0.001)   # Conversion efficiency
            ])
            
            # Demographic * Market interactions
            age_encoded = base_features[14]  # Age group encoding
            market_comp = raw_state.get("market_context", {}).get("competition_level", 0.5)
            
            engineered.extend([
                age_encoded * market_comp,
                age_encoded * budget,
                market_comp * budget
            ])
        
        # Temporal features
        if self.config.include_temporal_features:
            time_info = raw_state.get("time_context", {})
            
            # Cyclical encoding for temporal features
            hour = time_info.get("hour_of_day", 12)
            day_of_week = time_info.get("day_of_week", 3)
            month = time_info.get("month", 6)
            
            engineered.extend([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7),
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12)
            ])
        
        # Performance trend features
        performance_history = raw_state.get("performance_history", {})
        if "roas_trend" in performance_history:
            trend = performance_history["roas_trend"]
            engineered.extend([
                trend,
                trend ** 2,  # Quadratic trend
                max(0, trend),  # Positive trend only
                max(0, -trend)  # Negative trend only
            ])
        
        # Risk features
        volatility = raw_state.get("market_context", {}).get("market_volatility", 0.1)
        budget_util = raw_state.get("budget_constraints", {}).get("budget_utilization", 0.0)
        
        engineered.extend([
            volatility * budget_util,  # Risk exposure
            1.0 / (1.0 + volatility),  # Risk-adjusted score
            budget_util ** 2,  # Budget pressure
        ])
        
        return engineered
    
    def _handle_missing_values(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Handle missing values in feature matrix"""
        
        if fit:
            # Compute statistics for filling missing values
            if self.config.missing_value_strategy == "mean":
                self.fill_values = np.nanmean(features, axis=0)
            elif self.config.missing_value_strategy == "median":
                self.fill_values = np.nanmedian(features, axis=0)
            elif self.config.missing_value_strategy == "zero":
                self.fill_values = np.zeros(features.shape[1])
            else:
                self.fill_values = np.nanmedian(features, axis=0)  # Default to median
        
        # Fill missing values
        filled_features = features.copy()
        for i in range(features.shape[1]):
            mask = np.isnan(features[:, i])
            filled_features[mask, i] = self.fill_values[i]
        
        return filled_features
    
    def _handle_outliers(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Handle outliers in feature matrix"""
        
        if fit:
            # Compute outlier detection parameters
            if self.config.outlier_method == "iqr":
                self.q1 = np.percentile(features, 25, axis=0)
                self.q3 = np.percentile(features, 75, axis=0)
                self.iqr = self.q3 - self.q1
                self.outlier_lower = self.q1 - 1.5 * self.iqr
                self.outlier_upper = self.q3 + 1.5 * self.iqr
            
            elif self.config.outlier_method == "zscore":
                self.outlier_mean = np.mean(features, axis=0)
                self.outlier_std = np.std(features, axis=0)
                self.outlier_threshold = self.config.outlier_threshold
        
        # Clip outliers
        clipped_features = features.copy()
        
        if self.config.outlier_method == "iqr":
            clipped_features = np.clip(
                clipped_features, self.outlier_lower, self.outlier_upper
            )
        elif self.config.outlier_method == "zscore":
            z_scores = np.abs((clipped_features - self.outlier_mean) / (self.outlier_std + 1e-8))
            outlier_mask = z_scores > self.outlier_threshold
            
            # Replace outliers with clipped values
            for i in range(features.shape[1]):
                col_outliers = outlier_mask[:, i]
                if np.any(col_outliers):
                    clipped_features[col_outliers, i] = np.clip(
                        clipped_features[col_outliers, i],
                        self.outlier_mean[i] - self.outlier_threshold * self.outlier_std[i],
                        self.outlier_mean[i] + self.outlier_threshold * self.outlier_std[i]
                    )
        
        return clipped_features
    
    def _compute_feature_statistics(self, features: np.ndarray):
        """Compute and store feature statistics"""
        
        self.feature_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'num_features': features.shape[1],
            'num_samples': features.shape[0]
        }
        
        # Compute feature correlation matrix
        self.feature_correlation = np.corrcoef(features.T)
    
    def get_feature_importance(self, model: Optional[Any] = None) -> Dict[str, float]:
        """Compute feature importance scores"""
        
        if model is not None and hasattr(model, 'feature_importances_'):
            # Use model-based feature importance
            importance_scores = model.feature_importances_
        else:
            # Use statistical measures as proxy for importance
            variance_scores = self.feature_stats['std'] ** 2
            importance_scores = variance_scores / np.sum(variance_scores)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, score in enumerate(importance_scores):
            feature_name = f"feature_{i}"
            feature_importance[feature_name] = float(score)
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def get_state(self) -> Dict[str, Any]:
        """Get processor state for checkpointing"""
        state = {
            'config': self.config.__dict__,
            'is_fitted': self.is_fitted,
            'feature_stats': self.feature_stats,
            'feature_importance': self.feature_importance
        }
        
        # Add preprocessor states
        if self.scaler is not None:
            state['scaler_state'] = {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'var': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None
            }
        
        if self.pca is not None:
            state['pca_state'] = {
                'components': self.pca.components_.tolist(),
                'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
                'n_components': self.pca.n_components_
            }
        
        # Add missing value and outlier handling parameters
        if hasattr(self, 'fill_values'):
            state['fill_values'] = self.fill_values.tolist()
        
        if hasattr(self, 'outlier_lower'):
            state['outlier_params'] = {
                'outlier_lower': self.outlier_lower.tolist(),
                'outlier_upper': self.outlier_upper.tolist(),
                'q1': self.q1.tolist(),
                'q3': self.q3.tolist(),
                'iqr': self.iqr.tolist()
            }
        
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """Load processor state from checkpoint"""
        self.is_fitted = state['is_fitted']
        self.feature_stats = state['feature_stats']
        self.feature_importance = state['feature_importance']
        
        # Restore preprocessor states
        if 'scaler_state' in state and self.scaler is not None:
            scaler_state = state['scaler_state']
            if scaler_state['mean'] is not None:
                self.scaler.mean_ = np.array(scaler_state['mean'])
                self.scaler.scale_ = np.array(scaler_state['scale'])
                if scaler_state['var'] is not None:
                    self.scaler.var_ = np.array(scaler_state['var'])
        
        if 'pca_state' in state and self.pca is not None:
            pca_state = state['pca_state']
            self.pca.components_ = np.array(pca_state['components'])
            self.pca.explained_variance_ratio_ = np.array(pca_state['explained_variance_ratio'])
            self.pca.n_components_ = pca_state['n_components']
        
        # Restore missing value and outlier parameters
        if 'fill_values' in state:
            self.fill_values = np.array(state['fill_values'])
        
        if 'outlier_params' in state:
            params = state['outlier_params']
            self.outlier_lower = np.array(params['outlier_lower'])
            self.outlier_upper = np.array(params['outlier_upper'])
            self.q1 = np.array(params['q1'])
            self.q3 = np.array(params['q3'])
            self.iqr = np.array(params['iqr'])