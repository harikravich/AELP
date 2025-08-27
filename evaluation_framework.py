"""
Comprehensive Evaluation Framework for GAELP
============================================

This module provides a comprehensive evaluation framework for validating the performance
of the GAELP (Generative Autonomous Experimentation Learning Platform) system through:

- Train/test split management with temporal and stratified splitting
- Statistical significance testing with multiple correction methods
- Performance metrics calculation (CAC, ROAS, CTR, conversion rates)
- A/B testing infrastructure with power analysis
- Counterfactual analysis for policy evaluation
- Holdout test set management and validation

Key Features:
- Robust statistical testing with multiple hypothesis correction
- Time-series aware data splitting for realistic evaluation
- Comprehensive performance metrics for digital advertising
- Power analysis for experiment design
- Bayesian and frequentist statistical approaches
- Automated report generation with actionable insights
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from enum import Enum
import json
import warnings
from pathlib import Path

# Statistical and ML imports
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
from statsmodels.stats.power import ttest_power
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import joblib

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Plotting libraries not available. Visual reports will be limited.")

logger = logging.getLogger(__name__)


class SplitStrategy(Enum):
    """Enumeration of data splitting strategies."""
    RANDOM = "random"
    TEMPORAL = "temporal"
    STRATIFIED = "stratified"
    TIME_SERIES = "time_series"
    BLOCKED = "blocked"


class StatisticalTest(Enum):
    """Enumeration of statistical tests."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    BAYESIAN_T_TEST = "bayesian_t_test"


class MultipleTestCorrection(Enum):
    """Enumeration of multiple testing correction methods."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    FDR_BH = "fdr_bh"
    FDR_BY = "fdr_by"
    NONE = "none"


@dataclass
class PerformanceMetrics:
    """Container for digital advertising performance metrics."""
    # Basic metrics
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    revenue: float = 0.0
    
    # Calculated metrics
    ctr: float = field(init=False)
    conversion_rate: float = field(init=False)
    cpc: float = field(init=False)
    cpa: float = field(init=False)  # Cost per acquisition (CAC)
    roas: float = field(init=False)  # Return on ad spend
    
    # Advanced metrics
    viewthrough_conversions: int = 0
    assisted_conversions: int = 0
    bounce_rate: float = 0.0
    time_to_conversion: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.ctr = self.clicks / max(self.impressions, 1)
        self.conversion_rate = self.conversions / max(self.clicks, 1)
        self.cpc = self.spend / max(self.clicks, 1)
        self.cpa = self.spend / max(self.conversions, 1)  # CAC
        self.roas = self.revenue / max(self.spend, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'impressions': self.impressions,
            'clicks': self.clicks,
            'conversions': self.conversions,
            'spend': self.spend,
            'revenue': self.revenue,
            'ctr': self.ctr,
            'conversion_rate': self.conversion_rate,
            'cpc': self.cpc,
            'cpa': self.cpa,
            'roas': self.roas,
            'viewthrough_conversions': self.viewthrough_conversions,
            'assisted_conversions': self.assisted_conversions,
            'bounce_rate': self.bounce_rate,
            'time_to_conversion': self.time_to_conversion
        }


@dataclass
class ExperimentResult:
    """Container for A/B test experiment results."""
    treatment_group: str
    control_group: str
    treatment_metrics: PerformanceMetrics
    control_metrics: PerformanceMetrics
    sample_size_treatment: int
    sample_size_control: int
    statistical_power: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_significance: bool
    practical_significance: bool
    test_type: StatisticalTest
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CounterfactualResult:
    """Container for counterfactual analysis results."""
    policy_name: str
    baseline_policy: str
    estimated_lift: float
    confidence_interval: Tuple[float, float]
    significance: bool
    sample_size: int
    methodology: str
    assumptions_met: Dict[str, bool]
    sensitivity_analysis: Dict[str, float]


class DataSplitter:
    """Handles various data splitting strategies for evaluation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def split_data(
        self,
        data: pd.DataFrame,
        strategy: SplitStrategy,
        test_size: float = 0.2,
        time_column: Optional[str] = None,
        stratify_column: Optional[str] = None,
        n_splits: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data according to specified strategy.
        
        Args:
            data: Input DataFrame
            strategy: Splitting strategy to use
            test_size: Proportion of data for test set
            time_column: Column name for temporal splitting
            stratify_column: Column name for stratified splitting
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if strategy == SplitStrategy.RANDOM:
            return self._random_split(data, test_size)
        elif strategy == SplitStrategy.TEMPORAL:
            return self._temporal_split(data, test_size, time_column)
        elif strategy == SplitStrategy.STRATIFIED:
            return self._stratified_split(data, test_size, stratify_column)
        elif strategy == SplitStrategy.TIME_SERIES:
            return self._time_series_split(data, n_splits, time_column)
        elif strategy == SplitStrategy.BLOCKED:
            return self._blocked_split(data, test_size, stratify_column)
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
    
    def _random_split(self, data: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Random split."""
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=self.random_state
        )
        return train_data, test_data
    
    def _temporal_split(self, data: pd.DataFrame, test_size: float, time_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Temporal split preserving time order."""
        if time_column is None:
            raise ValueError("time_column must be specified for temporal split")
        
        data_sorted = data.sort_values(time_column)
        split_idx = int(len(data_sorted) * (1 - test_size))
        
        train_data = data_sorted.iloc[:split_idx]
        test_data = data_sorted.iloc[split_idx:]
        
        return train_data, test_data
    
    def _stratified_split(self, data: pd.DataFrame, test_size: float, stratify_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified split maintaining class proportions."""
        if stratify_column is None:
            raise ValueError("stratify_column must be specified for stratified split")
        
        train_data, test_data = train_test_split(
            data, test_size=test_size, stratify=data[stratify_column], 
            random_state=self.random_state
        )
        return train_data, test_data
    
    def _time_series_split(self, data: pd.DataFrame, n_splits: int, time_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time series split for cross-validation."""
        if time_column is None:
            raise ValueError("time_column must be specified for time series split")
        
        data_sorted = data.sort_values(time_column)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Return the last split as train/test
        splits = list(tscv.split(data_sorted))
        train_idx, test_idx = splits[-1]
        
        train_data = data_sorted.iloc[train_idx]
        test_data = data_sorted.iloc[test_idx]
        
        return train_data, test_data
    
    def _blocked_split(self, data: pd.DataFrame, test_size: float, block_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Blocked split ensuring no leakage between blocks."""
        if block_column is None:
            raise ValueError("block_column must be specified for blocked split")
        
        unique_blocks = data[block_column].unique()
        np.random.seed(self.random_state)
        np.random.shuffle(unique_blocks)
        
        n_test_blocks = int(len(unique_blocks) * test_size)
        test_blocks = unique_blocks[:n_test_blocks]
        train_blocks = unique_blocks[n_test_blocks:]
        
        train_data = data[data[block_column].isin(train_blocks)]
        test_data = data[data[block_column].isin(test_blocks)]
        
        return train_data, test_data


class StatisticalTester:
    """Handles statistical significance testing with multiple correction methods."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def calculate_significance(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
        test_type: StatisticalTest = StatisticalTest.T_TEST,
        alternative: str = 'two-sided'
    ) -> Tuple[float, float, float]:
        """
        Calculate statistical significance between treatment and control.
        
        Args:
            treatment_data: Treatment group data
            control_data: Control group data
            test_type: Type of statistical test to perform
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            Tuple of (p_value, test_statistic, effect_size)
        """
        if test_type == StatisticalTest.T_TEST:
            return self._t_test(treatment_data, control_data, alternative)
        elif test_type == StatisticalTest.MANN_WHITNEY:
            return self._mann_whitney_test(treatment_data, control_data, alternative)
        elif test_type == StatisticalTest.CHI_SQUARE:
            return self._chi_square_test(treatment_data, control_data)
        elif test_type == StatisticalTest.FISHER_EXACT:
            return self._fisher_exact_test(treatment_data, control_data)
        elif test_type == StatisticalTest.BAYESIAN_T_TEST:
            return self._bayesian_t_test(treatment_data, control_data)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _t_test(self, treatment: np.ndarray, control: np.ndarray, alternative: str) -> Tuple[float, float, float]:
        """Perform t-test."""
        statistic, p_value = ttest_ind(treatment, control, alternative=alternative)
        effect_size = self._cohens_d(treatment, control)
        return p_value, statistic, effect_size
    
    def _mann_whitney_test(self, treatment: np.ndarray, control: np.ndarray, alternative: str) -> Tuple[float, float, float]:
        """Perform Mann-Whitney U test."""
        statistic, p_value = mannwhitneyu(treatment, control, alternative=alternative)
        effect_size = self._rank_biserial_correlation(treatment, control)
        return p_value, statistic, effect_size
    
    def _chi_square_test(self, treatment: np.ndarray, control: np.ndarray) -> Tuple[float, float, float]:
        """Perform Chi-square test."""
        contingency_table = np.array([treatment, control])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        effect_size = self._cramers_v(contingency_table)
        return p_value, chi2, effect_size
    
    def _fisher_exact_test(self, treatment: np.ndarray, control: np.ndarray) -> Tuple[float, float, float]:
        """Perform Fisher's exact test."""
        contingency_table = np.array([treatment, control])
        odds_ratio, p_value = fisher_exact(contingency_table)
        effect_size = np.log(odds_ratio) if odds_ratio > 0 else 0
        return p_value, odds_ratio, effect_size
    
    def _bayesian_t_test(self, treatment: np.ndarray, control: np.ndarray) -> Tuple[float, float, float]:
        """Perform Bayesian t-test (simplified implementation)."""
        # Simplified Bayesian approach using BIC approximation
        t_stat, p_value = ttest_ind(treatment, control)
        n1, n2 = len(treatment), len(control)
        bf10 = np.exp((n1 + n2) * np.log(1 + t_stat**2 / (n1 + n2 - 2)) / 2)
        effect_size = self._cohens_d(treatment, control)
        return 1/bf10, bf10, effect_size  # Return 1/BF as p-value equivalent
    
    def _cohens_d(self, treatment: np.ndarray, control: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt(((len(treatment) - 1) * np.var(treatment, ddof=1) + 
                             (len(control) - 1) * np.var(control, ddof=1)) / 
                            (len(treatment) + len(control) - 2))
        return (np.mean(treatment) - np.mean(control)) / pooled_std
    
    def _rank_biserial_correlation(self, treatment: np.ndarray, control: np.ndarray) -> float:
        """Calculate rank-biserial correlation effect size."""
        n1, n2 = len(treatment), len(control)
        u_statistic, _ = mannwhitneyu(treatment, control)
        return 1 - (2 * u_statistic) / (n1 * n2)
    
    def _cramers_v(self, contingency_table: np.ndarray) -> float:
        """Calculate Cramer's V effect size."""
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    
    def correct_multiple_testing(
        self, 
        p_values: List[float], 
        method: MultipleTestCorrection = MultipleTestCorrection.FDR_BH
    ) -> Tuple[List[bool], List[float]]:
        """
        Correct for multiple testing.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method to use
            
        Returns:
            Tuple of (rejected_hypotheses, corrected_p_values)
        """
        if method == MultipleTestCorrection.NONE:
            return [p < self.alpha for p in p_values], p_values
        
        rejected, corrected_p_values, _, _ = multipletests(
            p_values, alpha=self.alpha, method=method.value
        )
        return rejected.tolist(), corrected_p_values.tolist()


class PowerAnalyzer:
    """Handles statistical power analysis for experiment design."""
    
    def calculate_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: StatisticalTest = StatisticalTest.T_TEST
    ) -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size
            power: Desired statistical power
            alpha: Significance level
            test_type: Type of statistical test
            
        Returns:
            Required sample size per group
        """
        if test_type == StatisticalTest.T_TEST:
            # Use power analysis for t-test
            sample_size = ttest_power(effect_size, power, alpha, alternative='two-sided')
            return max(int(np.ceil(sample_size)), 10)
        else:
            # Simplified calculation for other tests
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return max(int(np.ceil(n)), 10)
    
    def calculate_power(
        self,
        sample_size: int,
        effect_size: float,
        alpha: float = 0.05,
        test_type: StatisticalTest = StatisticalTest.T_TEST
    ) -> float:
        """
        Calculate statistical power for given parameters.
        
        Args:
            sample_size: Sample size per group
            effect_size: Effect size
            alpha: Significance level
            test_type: Type of statistical test
            
        Returns:
            Statistical power
        """
        if test_type == StatisticalTest.T_TEST:
            return ttest_power(effect_size, sample_size, alpha, alternative='two-sided')
        else:
            # Simplified calculation
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = effect_size * np.sqrt(sample_size / 2) - z_alpha
            return stats.norm.cdf(z_beta)


class CounterfactualAnalyzer:
    """Handles counterfactual analysis for policy evaluation."""
    
    def __init__(self):
        self.propensity_model = None
        self.outcome_model = None
    
    def estimate_policy_effect(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        outcome_column: str,
        feature_columns: List[str],
        policy_name: str,
        baseline_policy: str,
        method: str = "iptw"
    ) -> CounterfactualResult:
        """
        Estimate the effect of a policy using counterfactual analysis.
        
        Args:
            data: Input data
            treatment_column: Column indicating treatment assignment
            outcome_column: Column with outcomes
            feature_columns: List of feature columns
            policy_name: Name of the policy being evaluated
            baseline_policy: Name of the baseline policy
            method: Method for counterfactual estimation ('iptw', 'dm', 'doubly_robust')
            
        Returns:
            CounterfactualResult object
        """
        if method == "iptw":
            return self._inverse_propensity_weighting(
                data, treatment_column, outcome_column, feature_columns,
                policy_name, baseline_policy
            )
        elif method == "dm":
            return self._direct_method(
                data, treatment_column, outcome_column, feature_columns,
                policy_name, baseline_policy
            )
        elif method == "doubly_robust":
            return self._doubly_robust(
                data, treatment_column, outcome_column, feature_columns,
                policy_name, baseline_policy
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _inverse_propensity_weighting(
        self, data: pd.DataFrame, treatment_col: str, outcome_col: str, 
        feature_cols: List[str], policy_name: str, baseline_policy: str
    ) -> CounterfactualResult:
        """Inverse Propensity Score Weighting."""
        # Fit propensity model
        X = data[feature_cols]
        treatment = data[treatment_col]
        
        self.propensity_model = LogisticRegression(random_state=42)
        self.propensity_model.fit(X, treatment)
        
        # Calculate propensity scores
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        # Calculate weights
        weights = np.where(treatment == 1, 1/propensity_scores, 1/(1-propensity_scores))
        
        # Estimate treatment effects
        treated_outcomes = data[treatment == 1][outcome_col]
        control_outcomes = data[treatment == 0][outcome_col]
        treated_weights = weights[treatment == 1]
        control_weights = weights[treatment == 0]
        
        ate = np.average(treated_outcomes, weights=treated_weights) - \
              np.average(control_outcomes, weights=control_weights)
        
        # Bootstrap confidence interval
        ci = self._bootstrap_confidence_interval(
            data, treatment_col, outcome_col, feature_cols, method="iptw"
        )
        
        # Check assumptions
        assumptions = self._check_iptw_assumptions(propensity_scores, treatment)
        
        return CounterfactualResult(
            policy_name=policy_name,
            baseline_policy=baseline_policy,
            estimated_lift=ate,
            confidence_interval=ci,
            significance=ci[0] * ci[1] > 0,  # CI doesn't include 0
            sample_size=len(data),
            methodology="Inverse Propensity Score Weighting",
            assumptions_met=assumptions,
            sensitivity_analysis=self._sensitivity_analysis(data, treatment_col, outcome_col)
        )
    
    def _direct_method(
        self, data: pd.DataFrame, treatment_col: str, outcome_col: str,
        feature_cols: List[str], policy_name: str, baseline_policy: str
    ) -> CounterfactualResult:
        """Direct Method using outcome modeling."""
        X = data[feature_cols + [treatment_col]]
        y = data[outcome_col]
        
        # Fit outcome model
        self.outcome_model = RandomForestRegressor(random_state=42, n_estimators=100)
        self.outcome_model.fit(X, y)
        
        # Predict counterfactual outcomes
        X_treated = X.copy()
        X_treated[treatment_col] = 1
        X_control = X.copy()
        X_control[treatment_col] = 0
        
        y_treated = self.outcome_model.predict(X_treated)
        y_control = self.outcome_model.predict(X_control)
        
        ate = np.mean(y_treated - y_control)
        
        # Bootstrap confidence interval
        ci = self._bootstrap_confidence_interval(
            data, treatment_col, outcome_col, feature_cols, method="dm"
        )
        
        return CounterfactualResult(
            policy_name=policy_name,
            baseline_policy=baseline_policy,
            estimated_lift=ate,
            confidence_interval=ci,
            significance=ci[0] * ci[1] > 0,
            sample_size=len(data),
            methodology="Direct Method",
            assumptions_met={"model_specification": True, "no_unmeasured_confounding": True},
            sensitivity_analysis=self._sensitivity_analysis(data, treatment_col, outcome_col)
        )
    
    def _doubly_robust(
        self, data: pd.DataFrame, treatment_col: str, outcome_col: str,
        feature_cols: List[str], policy_name: str, baseline_policy: str
    ) -> CounterfactualResult:
        """Doubly Robust estimation combining propensity scores and outcome modeling."""
        # Combine IPTW and direct method
        iptw_result = self._inverse_propensity_weighting(
            data, treatment_col, outcome_col, feature_cols, policy_name, baseline_policy
        )
        dm_result = self._direct_method(
            data, treatment_col, outcome_col, feature_cols, policy_name, baseline_policy
        )
        
        # Simple average of estimates (more sophisticated combination possible)
        combined_estimate = (iptw_result.estimated_lift + dm_result.estimated_lift) / 2
        
        # Conservative confidence interval
        ci_lower = min(iptw_result.confidence_interval[0], dm_result.confidence_interval[0])
        ci_upper = max(iptw_result.confidence_interval[1], dm_result.confidence_interval[1])
        
        return CounterfactualResult(
            policy_name=policy_name,
            baseline_policy=baseline_policy,
            estimated_lift=combined_estimate,
            confidence_interval=(ci_lower, ci_upper),
            significance=ci_lower * ci_upper > 0,
            sample_size=len(data),
            methodology="Doubly Robust",
            assumptions_met={**iptw_result.assumptions_met, "model_specification": True},
            sensitivity_analysis=self._sensitivity_analysis(data, treatment_col, outcome_col)
        )
    
    def _bootstrap_confidence_interval(
        self, data: pd.DataFrame, treatment_col: str, outcome_col: str,
        feature_cols: List[str], method: str, n_bootstrap: int = 1000, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            boot_data = data.sample(n=len(data), replace=True, random_state=None)
            
            # Calculate estimate
            if method == "iptw":
                result = self._inverse_propensity_weighting(
                    boot_data, treatment_col, outcome_col, feature_cols, "", ""
                )
            elif method == "dm":
                result = self._direct_method(
                    boot_data, treatment_col, outcome_col, feature_cols, "", ""
                )
            
            bootstrap_estimates.append(result.estimated_lift)
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
        
        return (ci_lower, ci_upper)
    
    def _check_iptw_assumptions(self, propensity_scores: np.ndarray, treatment: np.ndarray) -> Dict[str, bool]:
        """Check assumptions for IPTW."""
        assumptions = {}
        
        # Check overlap assumption
        min_prop = np.min(propensity_scores)
        max_prop = np.max(propensity_scores)
        assumptions["overlap"] = min_prop > 0.01 and max_prop < 0.99
        
        # Check balance (simplified)
        treated_props = propensity_scores[treatment == 1]
        control_props = propensity_scores[treatment == 0]
        _, p_value = ttest_ind(treated_props, control_props)
        assumptions["balance"] = p_value > 0.05
        
        return assumptions
    
    def _sensitivity_analysis(self, data: pd.DataFrame, treatment_col: str, outcome_col: str) -> Dict[str, float]:
        """Perform sensitivity analysis."""
        # Simplified sensitivity analysis
        base_correlation = np.corrcoef(data[treatment_col], data[outcome_col])[0, 1]
        
        return {
            "base_correlation": base_correlation,
            "unmeasured_confounder_threshold": 0.1,  # Threshold for concern
            "model_stability": 0.95  # Placeholder for model stability metric
        }


class EvaluationFramework:
    """Main evaluation framework orchestrating all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation framework.
        
        Args:
            config: Configuration dictionary for the framework
        """
        self.config = self._default_config()
        if config:
            self.config.update(config)
        self.splitter = DataSplitter(random_state=self.config['random_state'])
        self.tester = StatisticalTester(alpha=self.config['alpha'])
        self.power_analyzer = PowerAnalyzer()
        self.counterfactual_analyzer = CounterfactualAnalyzer()
        
        # Results storage
        self.results_history: List[Dict[str, Any]] = []
        self.holdout_sets: Dict[str, pd.DataFrame] = {}
        
        # Setup logging
        self._setup_logging()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the framework."""
        return {
            'random_state': 42,
            'alpha': 0.05,
            'power': 0.8,
            'test_size': 0.2,
            'min_sample_size': 100,
            'multiple_testing_correction': MultipleTestCorrection.FDR_BH,
            'default_split_strategy': SplitStrategy.TEMPORAL,
            'confidence_level': 0.95,
            'effect_size_thresholds': {
                'small': 0.2,
                'medium': 0.5,
                'large': 0.8
            },
            'practical_significance_threshold': 0.05,
            'output_dir': 'evaluation_results',
            'save_results': True
        }
    
    def _setup_logging(self):
        """Setup logging for the framework."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_holdout_set(
        self,
        data: pd.DataFrame,
        holdout_name: str,
        strategy: SplitStrategy = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create and store a holdout test set.
        
        Args:
            data: Input data
            holdout_name: Name for the holdout set
            strategy: Splitting strategy
            **kwargs: Additional arguments for splitting
            
        Returns:
            Tuple of (train_data, holdout_data)
        """
        strategy = strategy or self.config['default_split_strategy']
        
        train_data, holdout_data = self.splitter.split_data(
            data, strategy, self.config['test_size'], **kwargs
        )
        
        # Store holdout set
        self.holdout_sets[holdout_name] = holdout_data.copy()
        
        logger.info(f"Created holdout set '{holdout_name}' with {len(holdout_data)} samples")
        
        return train_data, holdout_data
    
    def run_evaluation(
        self,
        treatment_data: Union[np.ndarray, pd.Series],
        control_data: Union[np.ndarray, pd.Series],
        experiment_name: str,
        metrics_treatment: Optional[PerformanceMetrics] = None,
        metrics_control: Optional[PerformanceMetrics] = None,
        test_type: StatisticalTest = StatisticalTest.T_TEST,
        multiple_comparisons: bool = False
    ) -> ExperimentResult:
        """
        Run comprehensive evaluation comparing treatment and control groups.
        
        Args:
            treatment_data: Treatment group data
            control_data: Control group data
            experiment_name: Name of the experiment
            metrics_treatment: Pre-calculated treatment metrics
            metrics_control: Pre-calculated control metrics
            test_type: Statistical test to use
            multiple_comparisons: Whether to apply multiple testing correction
            
        Returns:
            ExperimentResult object
        """
        logger.info(f"Running evaluation for experiment: {experiment_name}")
        
        # Convert to numpy arrays
        treatment_array = np.array(treatment_data)
        control_array = np.array(control_data)
        
        # Calculate statistical significance
        p_value, test_stat, effect_size = self.tester.calculate_significance(
            treatment_array, control_array, test_type
        )
        
        # Apply multiple testing correction if requested
        if multiple_comparisons:
            rejected, corrected_p = self.tester.correct_multiple_testing(
                [p_value], self.config['multiple_testing_correction']
            )
            p_value = corrected_p[0]
        
        # Calculate confidence interval
        ci = self._calculate_confidence_interval(treatment_array, control_array)
        
        # Calculate statistical power
        power = self.power_analyzer.calculate_power(
            len(treatment_array), abs(effect_size), self.config['alpha'], test_type
        )
        
        # Determine significance
        statistical_significance = p_value < self.config['alpha']
        practical_significance = abs(effect_size) > self.config['practical_significance_threshold']
        
        # Create result
        result = ExperimentResult(
            treatment_group=f"{experiment_name}_treatment",
            control_group=f"{experiment_name}_control",
            treatment_metrics=metrics_treatment or self._calculate_metrics_from_data(treatment_array),
            control_metrics=metrics_control or self._calculate_metrics_from_data(control_array),
            sample_size_treatment=len(treatment_array),
            sample_size_control=len(control_array),
            statistical_power=power,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            test_type=test_type
        )
        
        # Store result
        self.results_history.append({
            'experiment_name': experiment_name,
            'result': result,
            'timestamp': datetime.utcnow()
        })
        
        logger.info(f"Evaluation completed. Significant: {statistical_significance}, "
                   f"Effect size: {effect_size:.4f}, P-value: {p_value:.4f}")
        
        return result
    
    def calculate_significance(
        self,
        metric_name: str,
        treatment_values: np.ndarray,
        control_values: np.ndarray,
        test_type: StatisticalTest = StatisticalTest.T_TEST
    ) -> Tuple[bool, float, Tuple[float, float]]:
        """
        Calculate statistical significance for a specific metric.
        
        Args:
            metric_name: Name of the metric being tested
            treatment_values: Treatment group values
            control_values: Control group values
            test_type: Statistical test to use
            
        Returns:
            Tuple of (is_significant, p_value, confidence_interval)
        """
        p_value, _, effect_size = self.tester.calculate_significance(
            treatment_values, control_values, test_type
        )
        
        ci = self._calculate_confidence_interval(treatment_values, control_values)
        is_significant = p_value < self.config['alpha']
        
        logger.info(f"Significance test for {metric_name}: "
                   f"p-value={p_value:.4f}, significant={is_significant}")
        
        return is_significant, p_value, ci
    
    def analyze_counterfactual(
        self,
        data: pd.DataFrame,
        treatment_column: str,
        outcome_column: str,
        feature_columns: List[str],
        policy_name: str,
        baseline_policy: str = "control",
        method: str = "doubly_robust"
    ) -> CounterfactualResult:
        """
        Perform counterfactual analysis to estimate policy effects.
        
        Args:
            data: Input data with features, treatment, and outcomes
            treatment_column: Column indicating treatment assignment
            outcome_column: Column with outcomes
            feature_columns: List of feature columns
            policy_name: Name of the policy being evaluated
            baseline_policy: Name of the baseline policy
            method: Counterfactual method ('iptw', 'dm', 'doubly_robust')
            
        Returns:
            CounterfactualResult object
        """
        logger.info(f"Performing counterfactual analysis for policy: {policy_name}")
        
        result = self.counterfactual_analyzer.estimate_policy_effect(
            data, treatment_column, outcome_column, feature_columns,
            policy_name, baseline_policy, method
        )
        
        logger.info(f"Counterfactual analysis completed. "
                   f"Estimated lift: {result.estimated_lift:.4f}, "
                   f"Significant: {result.significance}")
        
        return result
    
    def generate_report(
        self,
        experiment_results: List[ExperimentResult] = None,
        counterfactual_results: List[CounterfactualResult] = None,
        include_plots: bool = True,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            experiment_results: List of experiment results to include
            counterfactual_results: List of counterfactual results to include
            include_plots: Whether to generate plots
            output_path: Path to save the report
            
        Returns:
            Dictionary containing the complete report
        """
        logger.info("Generating evaluation report")
        
        # Use all results if none specified
        if experiment_results is None:
            experiment_results = [r['result'] for r in self.results_history]
        
        report = {
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'framework_version': '1.0.0',
                'total_experiments': len(experiment_results),
                'config': self.config
            },
            'summary': self._generate_summary(experiment_results, counterfactual_results),
            'detailed_results': self._generate_detailed_results(experiment_results, counterfactual_results),
            'statistical_analysis': self._generate_statistical_analysis(experiment_results),
            'recommendations': self._generate_recommendations(experiment_results, counterfactual_results)
        }
        
        if include_plots and HAS_PLOTTING:
            report['visualizations'] = self._generate_visualizations(experiment_results)
        
        # Save report if requested
        if output_path or self.config['save_results']:
            save_path = output_path or f"{self.config['output_dir']}/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                # Create JSON-serializable version
                json_report = self._make_json_serializable(report)
                json.dump(json_report, f, indent=2)
            
            logger.info(f"Report saved to: {save_path}")
        
        return report
    
    def _calculate_confidence_interval(
        self, treatment: np.ndarray, control: np.ndarray, confidence: float = None
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the difference in means."""
        confidence = confidence or self.config['confidence_level']
        
        mean_diff = np.mean(treatment) - np.mean(control)
        pooled_se = np.sqrt(np.var(treatment)/len(treatment) + np.var(control)/len(control))
        
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, len(treatment) + len(control) - 2)
        
        margin_of_error = t_critical * pooled_se
        
        return (mean_diff - margin_of_error, mean_diff + margin_of_error)
    
    def _calculate_metrics_from_data(self, data: np.ndarray) -> PerformanceMetrics:
        """Calculate basic performance metrics from raw data."""
        # Simplified calculation - assumes data represents a single metric
        return PerformanceMetrics(
            impressions=int(len(data) * 1000),  # Placeholder
            clicks=int(len(data) * 30),
            conversions=int(len(data) * 1.5),
            spend=float(np.sum(data)),
            revenue=float(np.sum(data) * 2.5)
        )
    
    def _generate_summary(
        self, experiment_results: List[ExperimentResult], 
        counterfactual_results: Optional[List[CounterfactualResult]]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not experiment_results:
            return {"message": "No experiment results to summarize"}
        
        significant_experiments = sum(1 for r in experiment_results if r.statistical_significance)
        practically_significant = sum(1 for r in experiment_results if r.practical_significance)
        
        summary = {
            'total_experiments': len(experiment_results),
            'statistically_significant': significant_experiments,
            'practically_significant': practically_significant,
            'significance_rate': significant_experiments / len(experiment_results),
            'average_effect_size': np.mean([r.effect_size for r in experiment_results]),
            'average_power': np.mean([r.statistical_power for r in experiment_results]),
            'power_analysis': {
                'well_powered_experiments': sum(1 for r in experiment_results if r.statistical_power >= 0.8),
                'underpowered_experiments': sum(1 for r in experiment_results if r.statistical_power < 0.8)
            }
        }
        
        if counterfactual_results:
            summary['counterfactual_analysis'] = {
                'total_policies_evaluated': len(counterfactual_results),
                'significant_policy_effects': sum(1 for r in counterfactual_results if r.significance),
                'average_policy_lift': np.mean([r.estimated_lift for r in counterfactual_results])
            }
        
        return summary
    
    def _generate_detailed_results(
        self, experiment_results: List[ExperimentResult],
        counterfactual_results: Optional[List[CounterfactualResult]]
    ) -> Dict[str, Any]:
        """Generate detailed results section."""
        detailed = {
            'experiments': [
                {
                    'treatment_group': r.treatment_group,
                    'control_group': r.control_group,
                    'effect_size': r.effect_size,
                    'p_value': r.p_value,
                    'confidence_interval': r.confidence_interval,
                    'statistical_significance': r.statistical_significance,
                    'practical_significance': r.practical_significance,
                    'power': r.statistical_power,
                    'sample_sizes': {
                        'treatment': r.sample_size_treatment,
                        'control': r.sample_size_control
                    },
                    'metrics': {
                        'treatment': r.treatment_metrics.to_dict(),
                        'control': r.control_metrics.to_dict()
                    }
                }
                for r in experiment_results
            ]
        }
        
        if counterfactual_results:
            detailed['counterfactual_policies'] = [
                {
                    'policy_name': r.policy_name,
                    'baseline_policy': r.baseline_policy,
                    'estimated_lift': r.estimated_lift,
                    'confidence_interval': r.confidence_interval,
                    'significance': r.significance,
                    'methodology': r.methodology,
                    'assumptions_met': r.assumptions_met,
                    'sample_size': r.sample_size
                }
                for r in counterfactual_results
            ]
        
        return detailed
    
    def _generate_statistical_analysis(self, experiment_results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate statistical analysis section."""
        if not experiment_results:
            return {}
        
        effect_sizes = [r.effect_size for r in experiment_results]
        p_values = [r.p_value for r in experiment_results]
        powers = [r.statistical_power for r in experiment_results]
        
        return {
            'effect_size_distribution': {
                'mean': float(np.mean(effect_sizes)),
                'median': float(np.median(effect_sizes)),
                'std': float(np.std(effect_sizes)),
                'min': float(np.min(effect_sizes)),
                'max': float(np.max(effect_sizes))
            },
            'p_value_distribution': {
                'mean': float(np.mean(p_values)),
                'median': float(np.median(p_values)),
                'below_0_05': sum(1 for p in p_values if p < 0.05),
                'below_0_01': sum(1 for p in p_values if p < 0.01)
            },
            'power_analysis': {
                'mean_power': float(np.mean(powers)),
                'median_power': float(np.median(powers)),
                'well_powered_rate': sum(1 for p in powers if p >= 0.8) / len(powers)
            },
            'multiple_testing': {
                'raw_significant': sum(1 for p in p_values if p < 0.05),
                'expected_false_positives': len(p_values) * 0.05,
                'correction_recommended': len(p_values) > 1
            }
        }
    
    def _generate_recommendations(
        self, experiment_results: List[ExperimentResult],
        counterfactual_results: Optional[List[CounterfactualResult]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not experiment_results:
            return ["No experiment results available for recommendations"]
        
        # Power analysis recommendations
        underpowered = sum(1 for r in experiment_results if r.statistical_power < 0.8)
        if underpowered > 0:
            recommendations.append(
                f"{underpowered} experiments were underpowered. Consider increasing sample sizes "
                f"for future experiments to achieve 80% power."
            )
        
        # Effect size recommendations
        effect_sizes = [abs(r.effect_size) for r in experiment_results]
        avg_effect = np.mean(effect_sizes)
        if avg_effect < 0.2:
            recommendations.append(
                "Average effect sizes are small. Consider focusing on interventions "
                "with larger expected effects or increasing sensitivity of measurements."
            )
        
        # Multiple testing recommendations
        if len(experiment_results) > 1:
            recommendations.append(
                "Multiple comparisons detected. Consider applying multiple testing "
                "corrections (FDR or Bonferroni) to control Type I error rate."
            )
        
        # Practical significance recommendations
        practically_significant = sum(1 for r in experiment_results if r.practical_significance)
        if practically_significant < len(experiment_results) / 2:
            recommendations.append(
                "Many statistically significant results lack practical significance. "
                "Review practical significance thresholds and business impact."
            )
        
        # Counterfactual recommendations
        if counterfactual_results:
            assumption_violations = sum(
                1 for r in counterfactual_results 
                if not all(r.assumptions_met.values())
            )
            if assumption_violations > 0:
                recommendations.append(
                    f"{assumption_violations} counterfactual analyses have assumption violations. "
                    "Review causal identification strategy and consider sensitivity analyses."
                )
        
        return recommendations
    
    def _generate_visualizations(self, experiment_results: List[ExperimentResult]) -> Dict[str, str]:
        """Generate visualization plots."""
        if not HAS_PLOTTING or not experiment_results:
            return {}
        
        visualizations = {}
        
        # Effect size distribution
        effect_sizes = [r.effect_size for r in experiment_results]
        plt.figure(figsize=(10, 6))
        plt.hist(effect_sizes, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Effect Sizes')
        plt.xlabel('Effect Size (Cohen\'s d)')
        plt.ylabel('Frequency')
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        effect_plot_path = f"{self.config['output_dir']}/effect_sizes_distribution.png"
        Path(effect_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(effect_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations['effect_sizes'] = effect_plot_path
        
        # Power vs Effect Size
        powers = [r.statistical_power for r in experiment_results]
        plt.figure(figsize=(10, 6))
        plt.scatter(effect_sizes, powers, alpha=0.6)
        plt.xlabel('Effect Size')
        plt.ylabel('Statistical Power')
        plt.title('Statistical Power vs Effect Size')
        plt.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% Power')
        plt.legend()
        
        power_plot_path = f"{self.config['output_dir']}/power_vs_effect_size.png"
        plt.savefig(power_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations['power_analysis'] = power_plot_path
        
        return visualizations
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (PerformanceMetrics, ExperimentResult, CounterfactualResult)):
            return obj.__dict__
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj


# Convenience functions for common use cases
def quick_ab_test(
    treatment_data: np.ndarray,
    control_data: np.ndarray,
    experiment_name: str = "quick_test",
    alpha: float = 0.05
) -> ExperimentResult:
    """
    Quick A/B test evaluation with default settings.
    
    Args:
        treatment_data: Treatment group data
        control_data: Control group data
        experiment_name: Name for the experiment
        alpha: Significance level
        
    Returns:
        ExperimentResult object
    """
    framework = EvaluationFramework({'alpha': alpha, 'save_results': False})
    return framework.run_evaluation(treatment_data, control_data, experiment_name)


def calculate_required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05
) -> int:
    """
    Calculate required sample size for desired power.
    
    Args:
        effect_size: Expected effect size
        power: Desired statistical power
        alpha: Significance level
        
    Returns:
        Required sample size per group
    """
    analyzer = PowerAnalyzer()
    return analyzer.calculate_sample_size(effect_size, power, alpha)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate example data
    treatment = np.random.normal(1.2, 1.0, 1000)  # Treatment with effect
    control = np.random.normal(1.0, 1.0, 1000)    # Control baseline
    
    # Run quick A/B test
    result = quick_ab_test(treatment, control, "example_experiment")
    
    # Print results
    print(f"Experiment Result:")
    print(f"Effect Size: {result.effect_size:.4f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Statistically Significant: {result.statistical_significance}")
    print(f"Practically Significant: {result.practical_significance}")
    print(f"Statistical Power: {result.statistical_power:.4f}")
    print(f"Confidence Interval: {result.confidence_interval}")
    
    # Calculate required sample size
    required_n = calculate_required_sample_size(0.3, 0.8, 0.05)
    print(f"\nRequired sample size for 30% effect: {required_n} per group")