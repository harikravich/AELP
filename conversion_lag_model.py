"""
Conversion Lag Model using Survival Analysis

Handles user conversion journeys with extended lag periods (30+ days), 
right-censored data for ongoing journeys, and timeout logic for abandoned journeys.
Uses survival analysis techniques to predict conversion probability over time.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

# NO FALLBACKS - Must use lifelines for survival analysis
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')
from NO_FALLBACKS import StrictModeEnforcer

# Survival analysis libraries - REQUIRED
from lifelines import WeibullFitter, KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
LIFELINES_AVAILABLE = True  # MUST be true, no fallbacks!

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class ConversionJourney:
    """Represents a user's conversion journey with temporal data."""
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    converted: bool = False
    duration_days: Optional[float] = None
    touchpoints: List[Dict] = None
    features: Dict = None
    is_censored: bool = False
    timeout_reason: Optional[str] = None


class SurvivalModel(ABC):
    """Abstract base class for survival models."""
    
    @abstractmethod
    def fit(self, durations: np.ndarray, events: np.ndarray, X: Optional[np.ndarray] = None):
        pass
    
    @abstractmethod
    def predict_survival(self, times: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_hazard(self, times: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        pass


class WeibullSurvivalModel(SurvivalModel):
    """Weibull distribution survival model for conversion prediction."""
    
    def __init__(self):
        self.fitter = None
        self.is_fitted = False
        
    def fit(self, durations: np.ndarray, events: np.ndarray, X: Optional[np.ndarray] = None):
        """Fit Weibull survival model."""
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines required for WeibullSurvivalModel")
            
        self.fitter = WeibullFitter()
        self.fitter.fit(durations, events)
        self.is_fitted = True
        return self
        
    def predict_survival(self, times: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict survival probability at given times."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.fitter.survival_function_at_times(times).values
        
    def predict_hazard(self, times: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict hazard rate at given times."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.fitter.hazard_at_times(times).values


class CoxRegressionModel(SurvivalModel):
    """Cox Proportional Hazards model with covariates."""
    
    def __init__(self, penalizer: float = 0.1):
        self.fitter = None
        self.penalizer = penalizer
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    def fit(self, durations: np.ndarray, events: np.ndarray, X: np.ndarray):
        """Fit Cox regression model."""
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines required for CoxRegressionModel")
            
        if X is None:
            raise ValueError("Covariates X required for Cox regression")
            
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        df = pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['duration'] = durations
        df['event'] = events
        
        # Fit model
        self.fitter = CoxPHFitter(penalizer=self.penalizer)
        self.fitter.fit(df, duration_col='duration', event_col='event')
        self.is_fitted = True
        return self
        
    def predict_survival(self, times: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict survival probability at given times."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        df = pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        survival_functions = self.fitter.predict_survival_function(df)
        # Interpolate survival probabilities at specified times
        predictions = np.zeros((len(X), len(times)))
        
        for i, sf in enumerate(survival_functions.T):
            for j, t in enumerate(times):
                if t in sf.index:
                    predictions[i, j] = sf.loc[t]
                else:
                    # Linear interpolation
                    idx_before = sf.index[sf.index <= t]
                    idx_after = sf.index[sf.index > t]
                    
                    if len(idx_before) == 0:
                        predictions[i, j] = 1.0  # Before any events
                    elif len(idx_after) == 0:
                        predictions[i, j] = sf.iloc[-1]  # After all events
                    else:
                        t_before, t_after = idx_before[-1], idx_after[0]
                        s_before, s_after = sf.loc[t_before], sf.loc[t_after]
                        # Linear interpolation
                        predictions[i, j] = s_before + (s_after - s_before) * (t - t_before) / (t_after - t_before)
        
        return predictions
        
    def predict_hazard(self, times: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict hazard rate at given times."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        df = pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Get partial hazard (relative to baseline)
        partial_hazards = self.fitter.predict_partial_hazard(df)
        baseline_hazard = self.fitter.baseline_hazard_
        
        predictions = np.zeros((len(X), len(times)))
        for i, ph in enumerate(partial_hazards):
            for j, t in enumerate(times):
                if t in baseline_hazard.index:
                    predictions[i, j] = ph * baseline_hazard.loc[t].values[0]
                else:
                    # Find closest baseline hazard
                    closest_idx = np.argmin(np.abs(baseline_hazard.index - t))
                    predictions[i, j] = ph * baseline_hazard.iloc[closest_idx].values[0]
        
        return predictions


class ConversionLagModel:
    """
    Main conversion lag model handling extended conversion windows,
    right-censored data, and timeout logic.
    """
    
    def __init__(self, 
                 attribution_window_days: int = 30,
                 timeout_threshold_days: int = 45,
                 model_type: str = 'weibull'):
        """
        Initialize conversion lag model.
        
        Args:
            attribution_window_days: Days to attribute conversions to touchpoints
            timeout_threshold_days: Days after which to consider journey abandoned
            model_type: Type of survival model ('weibull', 'cox', 'kaplan_meier')
        """
        self.attribution_window_days = attribution_window_days
        self.timeout_threshold_days = timeout_threshold_days
        self.model_type = model_type
        
        # Initialize model
        if model_type == 'weibull':
            self.survival_model = WeibullSurvivalModel()
        elif model_type == 'cox':
            self.survival_model = CoxRegressionModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.is_fitted = False
        self.feature_columns = None
        
    def prepare_journey_data(self, journeys: List[ConversionJourney]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare journey data for survival analysis.
        
        Returns:
            durations: Time to event (conversion or censoring)
            events: Whether event occurred (1) or was censored (0)
            features: Feature matrix for Cox regression
        """
        durations = []
        events = []
        features_list = []
        
        current_time = datetime.now()
        
        for journey in journeys:
            # Calculate duration
            if journey.end_time is not None:
                duration = (journey.end_time - journey.start_time).days
            else:
                # Ongoing journey - use current time
                duration = (current_time - journey.start_time).days
                
            # Determine if censored
            is_censored = False
            if journey.is_censored:
                is_censored = True
            elif journey.end_time is None:
                # Check if should be considered timeout
                if duration > self.timeout_threshold_days:
                    is_censored = True
                    journey.timeout_reason = 'timeout'
                else:
                    is_censored = True  # Ongoing journey
                    
            durations.append(max(duration, 0.1))  # Avoid zero durations
            events.append(1 if journey.converted and not is_censored else 0)
            
            # Extract features
            features = self._extract_features(journey)
            features_list.append(features)
            
        return np.array(durations), np.array(events), np.array(features_list)
    
    def _extract_features(self, journey: ConversionJourney) -> List[float]:
        """Extract numerical features from journey."""
        features = []
        
        # Basic features
        features.append(len(journey.touchpoints) if journey.touchpoints else 0)  # Touchpoint count
        features.append((datetime.now() - journey.start_time).days)  # Journey age
        
        # Touchpoint diversity (unique channels)
        if journey.touchpoints:
            channels = set(tp.get('channel', 'unknown') for tp in journey.touchpoints)
            features.append(len(channels))
        else:
            features.append(0)
            
        # Custom features from journey.features
        if journey.features:
            for key in ['user_engagement_score', 'product_interest_score', 'demographic_score']:
                features.append(journey.features.get(key, 0.0))
        else:
            features.extend([0.0, 0.0, 0.0])
            
        return features
    
    def handle_censored_data(self, journeys: List[ConversionJourney]) -> List[ConversionJourney]:
        """
        Handle right-censored data by identifying ongoing and abandoned journeys.
        """
        current_time = datetime.now()
        processed_journeys = []
        
        for journey in journeys:
            processed_journey = journey
            
            if journey.end_time is None:
                duration = (current_time - journey.start_time).days
                
                if duration > self.timeout_threshold_days:
                    # Mark as abandoned/timeout
                    processed_journey.is_censored = True
                    processed_journey.timeout_reason = 'abandoned'
                    processed_journey.end_time = journey.start_time + timedelta(days=self.timeout_threshold_days)
                else:
                    # Mark as ongoing (right-censored)
                    processed_journey.is_censored = True
                    processed_journey.end_time = current_time
                    
            processed_journeys.append(processed_journey)
            
        return processed_journeys
    
    def fit(self, journeys: List[ConversionJourney]):
        """Fit the survival model on journey data."""
        # Handle censored data
        processed_journeys = self.handle_censored_data(journeys)
        
        # Prepare data
        durations, events, features = self.prepare_journey_data(processed_journeys)
        
        # Store feature information
        self.feature_columns = [
            'touchpoint_count', 'journey_age', 'channel_diversity',
            'engagement_score', 'interest_score', 'demographic_score'
        ]
        
        print(f"Training on {len(durations)} journeys:")
        print(f"  - Conversions: {events.sum()}")
        print(f"  - Censored: {len(events) - events.sum()}")
        print(f"  - Mean duration: {durations.mean():.1f} days")
        
        # Fit model
        if self.model_type == 'cox':
            self.survival_model.fit(durations, events, features)
        else:
            self.survival_model.fit(durations, events)
            
        self.is_fitted = True
        return self
    
    def predict_conversion_time(self, 
                              journeys: List[ConversionJourney],
                              time_points: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Predict conversion probability over time for given journeys.
        
        Args:
            journeys: List of journeys to predict for
            time_points: Time points (days) to predict at
            
        Returns:
            Dictionary with user_ids as keys and survival curves as values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if time_points is None:
            time_points = list(range(1, self.attribution_window_days + 1))
            
        # Prepare data
        durations, events, features = self.prepare_journey_data(journeys)
        
        # Get predictions
        if self.model_type == 'cox':
            survival_probs = self.survival_model.predict_survival(np.array(time_points), features)
        else:
            survival_probs = self.survival_model.predict_survival(np.array(time_points))
            
        # Convert survival to conversion probability
        if survival_probs.ndim == 1:
            # Single journey case
            conversion_probs = 1 - survival_probs
            results = {journeys[0].user_id: conversion_probs}
        else:
            # Multiple journeys
            results = {}
            for i, journey in enumerate(journeys):
                conversion_probs = 1 - survival_probs[i]
                results[journey.user_id] = conversion_probs
                
        return results
    
    def calculate_hazard_rate(self, 
                            journeys: List[ConversionJourney],
                            time_points: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Calculate hazard rate (instantaneous conversion probability) over time.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if time_points is None:
            time_points = list(range(1, self.attribution_window_days + 1))
            
        # Prepare data
        durations, events, features = self.prepare_journey_data(journeys)
        
        # Get hazard predictions
        if self.model_type == 'cox':
            hazard_rates = self.survival_model.predict_hazard(np.array(time_points), features)
        else:
            hazard_rates = self.survival_model.predict_hazard(np.array(time_points))
            
        # Format results
        if hazard_rates.ndim == 1:
            results = {journeys[0].user_id: hazard_rates}
        else:
            results = {}
            for i, journey in enumerate(journeys):
                results[journey.user_id] = hazard_rates[i]
                
        return results
    
    def get_conversion_insights(self, journeys: List[ConversionJourney]) -> Dict:
        """Generate insights about conversion patterns."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating insights")
            
        # Analyze conversion timing patterns
        converted_journeys = [j for j in journeys if j.converted]
        conversion_times = [j.duration_days for j in converted_journeys if j.duration_days is not None]
        
        insights = {
            'total_journeys': len(journeys),
            'conversions': len(converted_journeys),
            'conversion_rate': len(converted_journeys) / len(journeys) if journeys else 0,
            'median_time_to_conversion': np.median(conversion_times) if conversion_times else None,
            'mean_time_to_conversion': np.mean(conversion_times) if conversion_times else None,
            'conversion_time_percentiles': {
                'p25': np.percentile(conversion_times, 25) if conversion_times else None,
                'p50': np.percentile(conversion_times, 50) if conversion_times else None,
                'p75': np.percentile(conversion_times, 75) if conversion_times else None,
                'p90': np.percentile(conversion_times, 90) if conversion_times else None,
            }
        }
        
        # Analyze censoring patterns
        censored_journeys = [j for j in journeys if j.is_censored]
        insights['censored_journeys'] = len(censored_journeys)
        insights['censoring_rate'] = len(censored_journeys) / len(journeys) if journeys else 0
        
        # Timeout analysis
        timeout_journeys = [j for j in journeys if j.timeout_reason == 'abandoned']
        insights['abandoned_journeys'] = len(timeout_journeys)
        insights['abandonment_rate'] = len(timeout_journeys) / len(journeys) if journeys else 0
        
        return insights
    
    def predict_attribution_window_impact(self, 
                                        journeys: List[ConversionJourney],
                                        window_days: List[int]) -> Dict[int, float]:
        """
        Analyze impact of different attribution windows on conversion attribution.
        """
        results = {}
        
        for window in window_days:
            # Count conversions within this window
            conversions_in_window = 0
            total_journeys = 0
            
            for journey in journeys:
                if journey.converted and journey.duration_days is not None:
                    if journey.duration_days <= window:
                        conversions_in_window += 1
                total_journeys += 1
                        
            attribution_rate = conversions_in_window / total_journeys if total_journeys > 0 else 0
            results[window] = attribution_rate
            
        return results


# Example usage and testing
def create_sample_journeys() -> List[ConversionJourney]:
    """Create sample conversion journeys for testing."""
    import random
    
    journeys = []
    current_time = datetime.now()
    
    # Create various journey patterns
    for i in range(1000):
        start_time = current_time - timedelta(days=random.randint(1, 60))
        
        # Simulate different conversion patterns
        if random.random() < 0.3:  # 30% conversion rate
            # Converted journey
            duration = np.random.exponential(scale=7)  # Most convert quickly
            if random.random() < 0.1:  # 10% take longer
                duration += np.random.exponential(scale=20)
            
            end_time = start_time + timedelta(days=duration)
            converted = True
            is_censored = False
        else:
            # Non-converted journey
            if random.random() < 0.6:  # 60% of non-converted are ongoing
                end_time = None
                converted = False
                is_censored = True
                duration = None
            else:  # 40% abandoned
                duration = random.randint(45, 90)
                end_time = start_time + timedelta(days=duration)
                converted = False
                is_censored = True
        
        # Create touchpoints
        num_touchpoints = np.random.poisson(3) + 1
        touchpoints = []
        for j in range(num_touchpoints):
            touchpoints.append({
                'channel': random.choice(['email', 'web', 'social', 'paid', 'organic']),
                'timestamp': start_time + timedelta(days=random.randint(0, int(duration or 30)))
            })
        
        # Create features
        features = {
            'user_engagement_score': random.uniform(0, 1),
            'product_interest_score': random.uniform(0, 1),
            'demographic_score': random.uniform(0, 1)
        }
        
        journey = ConversionJourney(
            user_id=f'user_{i}',
            start_time=start_time,
            end_time=end_time,
            converted=converted,
            duration_days=duration,
            touchpoints=touchpoints,
            features=features,
            is_censored=is_censored
        )
        
        journeys.append(journey)
    
    return journeys


if __name__ == "__main__":
    # Example usage
    print("Creating sample conversion journeys...")
    sample_journeys = create_sample_journeys()
    
    print(f"Created {len(sample_journeys)} sample journeys")
    
    # Initialize and fit model
    print("\nFitting Weibull survival model...")
    model = ConversionLagModel(
        attribution_window_days=30,
        timeout_threshold_days=45,
        model_type='weibull'
    )
    
    model.fit(sample_journeys)
    
    # Generate insights
    print("\nConversion insights:")
    insights = model.get_conversion_insights(sample_journeys)
    for key, value in insights.items():
        print(f"  {key}: {value}")
    
    # Test prediction
    test_journeys = sample_journeys[:5]
    print(f"\nPredicting conversion probabilities for {len(test_journeys)} journeys...")
    
    predictions = model.predict_conversion_time(test_journeys, time_points=[1, 7, 14, 30])
    
    for user_id, probs in predictions.items():
        print(f"  {user_id}: Day 1: {probs[0]:.3f}, Day 7: {probs[1]:.3f}, Day 14: {probs[2]:.3f}, Day 30: {probs[3]:.3f}")
    
    # Test attribution window analysis
    print("\nAttribution window impact analysis:")
    window_impact = model.predict_attribution_window_impact(sample_journeys, [7, 14, 21, 30, 45])
    for window, rate in window_impact.items():
        print(f"  {window} days: {rate:.3f} attribution rate")
    
    print("\nConversion lag model implementation complete!")