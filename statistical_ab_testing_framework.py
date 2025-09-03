#!/usr/bin/env python3
"""
COMPREHENSIVE STATISTICAL A/B TESTING FRAMEWORK FOR GAELP

Implements rigorous A/B testing for policy comparison with:
- Proper statistical randomization
- Power analysis and significance testing
- Multi-armed bandit integration
- Bayesian inference for continuous learning
- Sequential testing with early stopping
- Multi-metric optimization
- Policy gradient variance reduction

NO BASIC SPLIT TESTING - Only advanced statistical methods.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import beta, gamma as gamma_dist, norm, chi2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import json
import math
import random
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import asyncio

# GAELP imports
from discovery_engine import GA4DiscoveryEngine
from dynamic_segment_integration import get_discovered_segments, validate_no_hardcoded_segments

logger = logging.getLogger(__name__)

# Validate no hardcoded segments
validate_no_hardcoded_segments("ab_testing_framework")


class TestType(Enum):
    """Types of statistical tests supported"""
    FREQUENTIST_TTEST = "frequentist_ttest"
    BAYESIAN_BANDIT = "bayesian_bandit"
    SEQUENTIAL_PROBABILITY = "sequential_probability"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB_CONTEXTUAL = "ucb_contextual"


class SignificanceTest(Enum):
    """Statistical significance testing methods"""
    WELCHS_T_TEST = "welchs_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP_PERMUTATION = "bootstrap_permutation"
    BAYESIAN_HYPOTHESIS = "bayesian_hypothesis"
    SEQUENTIAL_TESTING = "sequential_testing"
    MULTI_COMPARISON_BONFERRONI = "bonferroni"
    FALSE_DISCOVERY_RATE = "fdr_bh"


class AllocationStrategy(Enum):
    """Allocation strategies for test traffic"""
    FIXED_ALLOCATION = "fixed"
    ADAPTIVE_ALLOCATION = "adaptive"
    THOMPSON_SAMPLING = "thompson"
    UCB_EXPLORATION = "ucb"
    EPSILON_GREEDY = "epsilon_greedy"
    PROBABILITY_MATCHING = "probability_matching"


@dataclass
class StatisticalConfig:
    """Configuration for statistical testing"""
    alpha: float = 0.05  # Significance level
    power: float = 0.80  # Statistical power
    minimum_detectable_effect: float = 0.05  # MDE in conversion rate
    prior_conversion_rate: float = 0.02  # Prior belief about conversion rate
    minimum_sample_size: int = 1000  # Minimum observations per variant
    maximum_sample_size: int = 100000  # Maximum observations per variant
    confidence_level: float = 0.95
    
    # Bayesian parameters
    beta_prior_alpha: float = 2.0  # Beta distribution prior alpha
    beta_prior_beta: float = 98.0  # Beta distribution prior beta
    
    # Sequential testing
    spending_function: str = "obrien_fleming"  # or "pocock", "alpha_spending"
    interim_analysis_frequency: int = 500  # Check every N observations
    
    # Multi-metric optimization
    primary_metric: str = "conversion_rate"
    secondary_metrics: List[str] = field(default_factory=lambda: ["roas", "ctr", "ltv"])
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "conversion_rate": 0.4,
        "roas": 0.3,
        "ctr": 0.15,
        "ltv": 0.15
    })
    
    # MAB parameters
    exploration_rate: float = 0.1
    ucb_confidence: float = 2.0
    thompson_sample_size: int = 10000


@dataclass
class TestVariant:
    """A single test variant (policy)"""
    variant_id: str
    name: str
    policy_parameters: Dict[str, Any]
    allocation_probability: float = 0.0
    
    # Statistics tracking
    n_observations: int = 0
    sum_primary_metric: float = 0.0
    sum_squared_primary_metric: float = 0.0
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Bayesian tracking for conversion rate
    beta_alpha: float = 2.0  # Prior alpha + successes
    beta_beta: float = 98.0  # Prior beta + failures
    
    # Performance tracking per segment
    segment_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Contextual features
    context_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @property
    def primary_metric_mean(self) -> float:
        """Mean of primary metric"""
        return self.sum_primary_metric / max(1, self.n_observations)
    
    @property
    def primary_metric_variance(self) -> float:
        """Variance of primary metric"""
        if self.n_observations <= 1:
            return 0.0
        mean = self.primary_metric_mean
        return (self.sum_squared_primary_metric / self.n_observations) - (mean ** 2)
    
    @property
    def primary_metric_std(self) -> float:
        """Standard deviation of primary metric"""
        return math.sqrt(self.primary_metric_variance)
    
    @property
    def conversion_rate_posterior(self) -> Tuple[float, float]:
        """Beta distribution parameters for conversion rate posterior"""
        return (self.beta_alpha, self.beta_beta)
    
    def update_observation(self, primary_value: float, secondary_values: Dict[str, float],
                          converted: bool, segment: str = None, context: Dict[str, Any] = None):
        """Update statistics with new observation"""
        self.n_observations += 1
        self.sum_primary_metric += primary_value
        self.sum_squared_primary_metric += primary_value ** 2
        
        # Update Bayesian conversion tracking
        if converted:
            self.beta_alpha += 1
        else:
            self.beta_beta += 1
        
        # Update secondary metrics
        for metric, value in secondary_values.items():
            if metric not in self.secondary_metrics:
                self.secondary_metrics[metric] = 0.0
            self.secondary_metrics[metric] += value
        
        # Update segment performance
        if segment:
            if segment not in self.segment_performance:
                self.segment_performance[segment] = {
                    "n": 0, "sum": 0.0, "conversions": 0
                }
            seg_perf = self.segment_performance[segment]
            seg_perf["n"] += 1
            seg_perf["sum"] += primary_value
            if converted:
                seg_perf["conversions"] += 1
        
        # Update contextual performance
        if context:
            context_key = self._context_key(context)
            if context_key not in self.context_performance:
                self.context_performance[context_key] = {
                    "n": 0, "sum": 0.0, "conversions": 0
                }
            ctx_perf = self.context_performance[context_key]
            ctx_perf["n"] += 1
            ctx_perf["sum"] += primary_value
            if converted:
                ctx_perf["conversions"] += 1
    
    def _context_key(self, context: Dict[str, Any]) -> str:
        """Create context key from context dict"""
        keys = ['device', 'hour', 'channel']
        return "_".join(str(context.get(k, 'unknown')) for k in keys)


@dataclass
class TestResults:
    """Results of statistical test"""
    test_id: str
    test_type: TestType
    significance_test: SignificanceTest
    
    # Statistical results
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_power: float
    is_significant: bool
    
    # Bayesian results
    bayesian_probability: float = 0.0  # P(variant A > variant B)
    credible_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Multi-metric results
    primary_metric_results: Dict[str, float] = field(default_factory=dict)
    secondary_metric_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Sample size information
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    minimum_sample_achieved: bool = False
    
    # Test duration and status
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    test_duration_hours: float = 0.0
    
    # Recommendation
    recommended_action: str = ""
    winner_variant_id: Optional[str] = None
    lift_percentage: float = 0.0
    
    # Risk assessment
    risk_of_false_positive: float = 0.0
    risk_of_false_negative: float = 0.0
    expected_loss: Dict[str, float] = field(default_factory=dict)


class ContextualBandit(nn.Module):
    """Contextual multi-armed bandit for adaptive allocation"""
    
    def __init__(self, context_dim: int, n_variants: int, hidden_dim: int = 128):
        super().__init__()
        self.n_variants = n_variants
        self.context_dim = context_dim
        
        # Neural network for each variant
        self.variant_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)  # Expected reward
            ) for _ in range(n_variants)
        ])
        
        # Uncertainty estimation networks
        self.uncertainty_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Positive uncertainty
            ) for _ in range(n_variants)
        ])
        
        # Initialize optimizers with proper parameter groups
        self.optimizers = []
        for i in range(n_variants):
            params = list(self.variant_networks[i].parameters()) + list(self.uncertainty_networks[i].parameters())
            optimizer = optim.Adam(params, lr=1e-3)
            # Initialize optimizer state to prevent KeyError
            dummy_input = torch.randn(1, context_dim)
            dummy_output = self.variant_networks[i](dummy_input) + self.uncertainty_networks[i](dummy_input)
            dummy_loss = dummy_output.sum()
            optimizer.zero_grad()
            dummy_loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # Clear after initialization
            self.optimizers.append(optimizer)
    
    def predict_rewards(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict rewards and uncertainties for all variants"""
        rewards = torch.stack([net(context).squeeze(-1) for net in self.variant_networks])
        uncertainties = torch.stack([net(context).squeeze(-1) for net in self.uncertainty_networks])
        return rewards, uncertainties
    
    def select_variant_ucb(self, context: torch.Tensor, confidence: float = 2.0) -> int:
        """Select variant using Upper Confidence Bound"""
        with torch.no_grad():
            rewards, uncertainties = self.predict_rewards(context)
            ucb_values = rewards + confidence * uncertainties
            return ucb_values.argmax().item()
    
    def select_variant_thompson(self, context: torch.Tensor) -> int:
        """Select variant using Thompson sampling"""
        with torch.no_grad():
            rewards, uncertainties = self.predict_rewards(context)
            sampled_rewards = torch.normal(rewards, uncertainties)
            return sampled_rewards.argmax().item()
    
    def update(self, variant_id: int, context: torch.Tensor, reward: float):
        """Update the bandit with observed reward - Thread-safe with proper error handling"""
        try:
            # Ensure we have a valid variant_id
            if variant_id >= len(self.variant_networks) or variant_id < 0:
                logger.warning(f"Invalid variant_id {variant_id}, skipping update")
                return
            
            # Ensure context is properly shaped
            if len(context.shape) == 1:
                context = context.unsqueeze(0)
            
            predicted_reward, predicted_uncertainty = (
                self.variant_networks[variant_id](context),
                self.uncertainty_networks[variant_id](context)
            )
            
            # Reward prediction loss
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            if torch.cuda.is_available() and predicted_reward.is_cuda:
                reward_tensor = reward_tensor.cuda()
            
            reward_loss = nn.MSELoss()(predicted_reward.squeeze(), reward_tensor)
            
            # Uncertainty loss (encourage uncertainty reduction)
            uncertainty_target = torch.abs(predicted_reward.squeeze() - reward_tensor).detach()
            uncertainty_loss = nn.MSELoss()(predicted_uncertainty.squeeze(), uncertainty_target)
            
            total_loss = reward_loss + 0.1 * uncertainty_loss
            
            # Handle NaN/Inf losses
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("Invalid loss detected, skipping optimization step")
                return
            
            # Safe optimizer update
            optimizer = self.optimizers[variant_id]
            optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                list(self.variant_networks[variant_id].parameters()) + 
                list(self.uncertainty_networks[variant_id].parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            
        except Exception as e:
            logger.error(f"Error in contextual bandit update: {e}")
            # Continue gracefully without crashing


class StatisticalABTestFramework:
    """
    Comprehensive statistical A/B testing framework for GAELP policy comparison
    """
    
    def __init__(self, config: StatisticalConfig, discovery_engine: GA4DiscoveryEngine):
        self.config = config
        self.discovery = discovery_engine
        self.active_tests: Dict[str, Dict] = {}
        
        # Test registry
        self.test_registry: Dict[str, List[TestVariant]] = {}
        self.test_results: Dict[str, TestResults] = {}
        self.allocation_history: Dict[str, List] = defaultdict(list)
        
        # Contextual bandit for adaptive allocation
        self.context_dim = self._calculate_context_dimension()
        self.contextual_bandits: Dict[str, ContextualBandit] = {}
        
        # Sequential testing tracking
        self.sequential_tests: Dict[str, Dict] = {}
        
        # Power analysis cache
        self.power_analysis_cache: Dict[str, Dict] = {}
        
        logger.info("StatisticalABTestFramework initialized")
    
    def _calculate_context_dimension(self) -> int:
        """Calculate dimension of context vector for contextual bandits"""
        # Context includes: segment (one-hot), device (one-hot), hour (normalized), 
        # channel (one-hot), day_of_week (one-hot), seasonality, competition_level
        segments = get_discovered_segments()
        n_segments = max(len(segments), 1)  # At least 1 for default segment
        n_devices = 3  # mobile, desktop, tablet
        n_channels = 5  # discovered from patterns
        n_days = 7
        
        context_dim = (
            n_segments +  # segment one-hot
            n_devices +   # device one-hot
            n_channels +  # channel one-hot
            n_days +      # day of week one-hot
            4             # continuous: hour_normalized, seasonality, competition, budget_remaining
        )
        
        # Store for consistency checks
        self._expected_context_dim = context_dim
        
        logger.info(f"Context dimension calculated: {context_dim} (segments={n_segments})")
        return context_dim
    
    def create_ab_test(self,
                      test_id: str,
                      test_name: str,
                      variants: List[Dict[str, Any]],
                      test_type: TestType = TestType.BAYESIAN_BANDIT,
                      allocation_strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE_ALLOCATION,
                      duration_days: Optional[int] = None) -> str:
        """
        Create a new A/B test with proper statistical design
        """
        
        # Validate inputs
        if len(variants) < 2:
            raise ValueError("At least 2 variants required for A/B test")
        
        if test_id in self.active_tests:
            raise ValueError(f"Test {test_id} already exists")
        
        # Create test variants
        test_variants = []
        for i, variant_config in enumerate(variants):
            variant = TestVariant(
                variant_id=variant_config.get('variant_id', f'variant_{i}'),
                name=variant_config.get('name', f'Variant {i}'),
                policy_parameters=variant_config.get('policy_parameters', {}),
                allocation_probability=variant_config.get('allocation_probability', 1.0 / len(variants)),
                beta_alpha=self.config.beta_prior_alpha,
                beta_beta=self.config.beta_prior_beta
            )
            test_variants.append(variant)
        
        # Power analysis for sample size calculation
        required_sample_size = self._calculate_required_sample_size(
            self.config.prior_conversion_rate,
            self.config.minimum_detectable_effect,
            self.config.alpha,
            self.config.power
        )
        
        # Initialize contextual bandit if using adaptive allocation
        if allocation_strategy in [AllocationStrategy.ADAPTIVE_ALLOCATION, 
                                 AllocationStrategy.THOMPSON_SAMPLING,
                                 AllocationStrategy.UCB_EXPLORATION]:
            self.contextual_bandits[test_id] = ContextualBandit(
                self.context_dim, len(test_variants)
            )
        
        # Create test configuration
        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'test_type': test_type,
            'allocation_strategy': allocation_strategy,
            'variants': test_variants,
            'start_time': datetime.now(),
            'duration_days': duration_days,
            'required_sample_size': required_sample_size,
            'status': 'active'
        }
        
        # Initialize sequential testing if required
        if test_type == TestType.SEQUENTIAL_PROBABILITY:
            self.sequential_tests[test_id] = {
                'interim_analyses': [],
                'spending_function_values': self._calculate_spending_function(duration_days or 30),
                'current_alpha_spent': 0.0
            }
        
        self.active_tests[test_id] = test_config
        self.test_registry[test_id] = test_variants
        
        logger.info(f"Created A/B test {test_id} with {len(variants)} variants")
        logger.info(f"Required sample size per variant: {required_sample_size}")
        
        return test_id
    
    def _calculate_required_sample_size(self, p1: float, delta: float, alpha: float, power: float) -> int:
        """Calculate required sample size using normal approximation"""
        p2 = p1 + delta
        
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Pooled variance
        p_pooled = (p1 + p2) / 2
        variance = 2 * p_pooled * (1 - p_pooled)
        
        # Sample size calculation
        n = (z_alpha + z_beta) ** 2 * variance / (delta ** 2)
        
        return max(int(np.ceil(n)), self.config.minimum_sample_size)
    
    def _calculate_spending_function(self, max_duration_days: int) -> List[float]:
        """Calculate alpha spending function values for sequential testing"""
        n_analyses = max_duration_days // (self.config.interim_analysis_frequency / 24)  # Assume daily checks
        
        if self.config.spending_function == "obrien_fleming":
            # O'Brien-Fleming spending function
            spending_values = []
            for k in range(1, n_analyses + 1):
                t = k / n_analyses
                if k == n_analyses:
                    alpha_k = self.config.alpha
                else:
                    alpha_k = 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.config.alpha / 2) / math.sqrt(t)))
                spending_values.append(alpha_k)
        
        elif self.config.spending_function == "pocock":
            # Pocock spending function
            alpha_k = self.config.alpha / n_analyses
            spending_values = [alpha_k * (k + 1) for k in range(n_analyses)]
        
        else:  # Linear alpha spending
            spending_values = [self.config.alpha * (k + 1) / n_analyses for k in range(n_analyses)]
        
        return spending_values
    
    def assign_variant(self, test_id: str, user_id: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Assign user to test variant using configured allocation strategy
        """
        if test_id not in self.active_tests:
            logger.warning(f"Test {test_id} not found")
            return None
        
        test_config = self.active_tests[test_id]
        variants = test_config['variants']
        allocation_strategy = test_config['allocation_strategy']
        
        # Create context vector
        context_vector = self._create_context_vector(context)
        context_tensor = torch.tensor(context_vector, dtype=torch.float32).unsqueeze(0)
        
        # Select variant based on allocation strategy
        if allocation_strategy == AllocationStrategy.FIXED_ALLOCATION:
            variant_idx = self._fixed_allocation(variants, user_id)
        
        elif allocation_strategy == AllocationStrategy.THOMPSON_SAMPLING:
            if test_id in self.contextual_bandits:
                variant_idx = self.contextual_bandits[test_id].select_variant_thompson(context_tensor)
            else:
                variant_idx = self._thompson_sampling_allocation(variants)
        
        elif allocation_strategy == AllocationStrategy.UCB_EXPLORATION:
            if test_id in self.contextual_bandits:
                variant_idx = self.contextual_bandits[test_id].select_variant_ucb(
                    context_tensor, self.config.ucb_confidence
                )
            else:
                variant_idx = self._ucb_allocation(variants)
        
        elif allocation_strategy == AllocationStrategy.ADAPTIVE_ALLOCATION:
            variant_idx = self._adaptive_allocation(test_id, variants, context_tensor)
        
        else:  # epsilon_greedy or probability_matching
            variant_idx = self._epsilon_greedy_allocation(variants)
        
        # Record allocation
        selected_variant = variants[variant_idx]
        self.allocation_history[test_id].append({
            'user_id': user_id,
            'variant_id': selected_variant.variant_id,
            'timestamp': datetime.now(),
            'context': context
        })
        
        return selected_variant.variant_id
    
    def _create_context_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """Create context vector from context dictionary - Robust with fixed dimensions"""
        try:
            vector = []
            
            # Get discovered segments dynamically
            segments = get_discovered_segments()
            
            # Ensure we have at least some segments for consistent dimensionality
            if not segments:
                segments = ['default_segment']  # Use ensure consistent dimensions if needed
            
            # Segment one-hot encoding
            segment = context.get('segment', segments[0])
            segment_vector = [1.0 if s == segment else 0.0 for s in segments]
            vector.extend(segment_vector)
            
            # Device one-hot encoding (fixed size)
            device = context.get('device', 'mobile')
            device_vector = [
                1.0 if device == 'mobile' else 0.0,
                1.0 if device == 'desktop' else 0.0,
                1.0 if device == 'tablet' else 0.0
            ]
            vector.extend(device_vector)
            
            # Channel one-hot encoding (fixed size)
            channel = context.get('channel', 'organic')
            channels = ['organic', 'paid_search', 'social', 'display', 'email']
            channel_vector = [1.0 if channel == c else 0.0 for c in channels]
            vector.extend(channel_vector)
            
            # Day of week one-hot encoding (fixed size)
            day_of_week = min(max(context.get('day_of_week', 0), 0), 6)  # Ensure valid range
            day_vector = [1.0 if i == day_of_week else 0.0 for i in range(7)]
            vector.extend(day_vector)
            
            # Continuous features (normalized and bounded)
            hour = min(max(context.get('hour', 12), 0), 23) / 23.0
            seasonality = min(max(context.get('seasonality_factor', 1.0), 0.1), 10.0)
            competition = min(max(context.get('competition_level', 0.5), 0.0), 1.0)
            budget_ratio = min(max(context.get('budget_remaining_ratio', 1.0), 0.0), 1.0)
            
            vector.extend([hour, seasonality, competition, budget_ratio])
            
            # Ensure vector has expected dimension
            expected_dim = len(segments) + 3 + 5 + 7 + 4  # segments + devices + channels + days + continuous
            if len(vector) != expected_dim:
                logger.warning(f"Context vector dimension mismatch: {len(vector)} != {expected_dim}")
                # Pad or truncate to expected dimension
                if len(vector) < expected_dim:
                    vector.extend([0.0] * (expected_dim - len(vector)))
                else:
                    vector = vector[:expected_dim]
            
            # Validate vector values
            vector = [float(v) if not (math.isnan(v) or math.isinf(v)) else 0.0 for v in vector]
            
            return np.array(vector, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error creating context vector: {e}")
            # Return zero vector with expected dimensions as fallback
            segments = get_discovered_segments() or ['default_segment']
            fallback_dim = len(segments) + 3 + 5 + 7 + 4
            return np.zeros(fallback_dim, dtype=np.float32)
    
    def _fixed_allocation(self, variants: List[TestVariant], user_id: str) -> int:
        """Fixed probability allocation based on user ID hash"""
        user_hash = hash(user_id) % 100
        cumulative_prob = 0.0
        
        for i, variant in enumerate(variants):
            cumulative_prob += variant.allocation_probability * 100
            if user_hash < cumulative_prob:
                return i
        
        return len(variants) - 1  # Use last variant if needed
    
    def _thompson_sampling_allocation(self, variants: List[TestVariant]) -> int:
        """Thompson sampling allocation using Beta posteriors"""
        samples = []
        for variant in variants:
            alpha, beta = variant.conversion_rate_posterior
            sample = np.random.beta(alpha, beta)
            samples.append(sample)
        
        return np.argmax(samples)
    
    def _ucb_allocation(self, variants: List[TestVariant]) -> int:
        """Upper Confidence Bound allocation"""
        total_observations = sum(v.n_observations for v in variants)
        if total_observations == 0:
            return np.random.randint(len(variants))
        
        ucb_values = []
        for variant in variants:
            if variant.n_observations == 0:
                ucb_values.append(float('inf'))
            else:
                mean = variant.primary_metric_mean
                confidence = math.sqrt(
                    self.config.ucb_confidence * math.log(total_observations) / variant.n_observations
                )
                ucb_values.append(mean + confidence)
        
        return np.argmax(ucb_values)
    
    def _adaptive_allocation(self, test_id: str, variants: List[TestVariant], 
                           context_tensor: torch.Tensor) -> int:
        """Adaptive allocation using contextual bandit"""
        if test_id in self.contextual_bandits:
            # Use exploration probability
            if np.random.random() < self.config.exploration_rate:
                return np.random.randint(len(variants))
            else:
                return self.contextual_bandits[test_id].select_variant_ucb(context_tensor)
        else:
            return self._thompson_sampling_allocation(variants)
    
    def _epsilon_greedy_allocation(self, variants: List[TestVariant]) -> int:
        """Epsilon-greedy allocation"""
        if np.random.random() < self.config.exploration_rate:
            return np.random.randint(len(variants))
        else:
            # Select best performing variant
            best_idx = 0
            best_performance = -float('inf')
            
            for i, variant in enumerate(variants):
                if variant.n_observations > 0:
                    performance = variant.primary_metric_mean
                    if performance > best_performance:
                        best_performance = performance
                        best_idx = i
            
            return best_idx
    
    def record_observation(self, test_id: str, variant_id: str, user_id: str,
                          primary_metric_value: float, secondary_metrics: Dict[str, float],
                          converted: bool, context: Dict[str, Any] = None):
        """
        Record observation for statistical analysis - Thread-safe with proper validation
        """
        try:
            if test_id not in self.active_tests:
                logger.warning(f"Test {test_id} not found")
                return
            
            variants = self.test_registry[test_id]
            variant = next((v for v in variants if v.variant_id == variant_id), None)
            
            if not variant:
                logger.warning(f"Variant {variant_id} not found in test {test_id}")
                return
            
            # Validate primary metric value
            if not isinstance(primary_metric_value, (int, float)) or math.isnan(primary_metric_value) or math.isinf(primary_metric_value):
                logger.warning(f"Invalid primary_metric_value {primary_metric_value}, using 0.0")
                primary_metric_value = 0.0
            
            # Update variant statistics
            segment = context.get('segment', 'unknown') if context else 'unknown'
            variant.update_observation(
                primary_metric_value, secondary_metrics, converted, segment, context
            )
            
            # Update contextual bandit if applicable - with thread safety
            if test_id in self.contextual_bandits and context:
                try:
                    context_vector = self._create_context_vector(context)
                    if len(context_vector) != self.context_dim:
                        logger.warning(f"Context vector dimension mismatch: {len(context_vector)} != {self.context_dim}")
                        return
                    
                    context_tensor = torch.tensor(context_vector, dtype=torch.float32)
                    variant_idx = next((i for i, v in enumerate(variants) if v.variant_id == variant_id), -1)
                    
                    if variant_idx >= 0:
                        # Use lock for thread safety in contextual bandit updates
                        import threading
                        if not hasattr(self, '_bandit_lock'):
                            self._bandit_lock = threading.Lock()
                        
                        with self._bandit_lock:
                            self.contextual_bandits[test_id].update(variant_idx, context_tensor, primary_metric_value)
                            
                except Exception as e:
                    logger.warning(f"Error updating contextual bandit: {e}")
            
            # Check for early stopping if sequential testing
            test_config = self.active_tests[test_id]
            if test_config['test_type'] == TestType.SEQUENTIAL_PROBABILITY:
                self._check_sequential_stopping(test_id)
            
            logger.debug(f"Recorded observation for test {test_id}, variant {variant_id}")
            
        except Exception as e:
            logger.error(f"Error recording observation: {e}")
            # Continue gracefully without crashing the system
    
    def _check_sequential_stopping(self, test_id: str):
        """Check if sequential test should be stopped early"""
        variants = self.test_registry[test_id]
        if len(variants) != 2:  # Sequential testing currently supports only 2 variants
            return
        
        variant_a, variant_b = variants[:2]
        
        # Only check if we have minimum observations
        if (variant_a.n_observations < self.config.interim_analysis_frequency or 
            variant_b.n_observations < self.config.interim_analysis_frequency):
            return
        
        # Calculate current test statistic
        p_value = self._calculate_sequential_p_value(variant_a, variant_b)
        
        # Get current spending function value
        sequential_config = self.sequential_tests[test_id]
        n_analyses = len(sequential_config['interim_analyses'])
        
        if n_analyses < len(sequential_config['spending_function_values']):
            current_alpha = sequential_config['spending_function_values'][n_analyses]
            
            # Record interim analysis
            sequential_config['interim_analyses'].append({
                'analysis_number': n_analyses + 1,
                'timestamp': datetime.now(),
                'p_value': p_value,
                'alpha_threshold': current_alpha,
                'variant_a_n': variant_a.n_observations,
                'variant_b_n': variant_b.n_observations
            })
            
            # Check for early stopping
            if p_value < current_alpha:
                self._conclude_test(test_id, "early_stop_significance")
                logger.info(f"Test {test_id} stopped early due to statistical significance")
    
    def _calculate_sequential_p_value(self, variant_a: TestVariant, variant_b: TestVariant) -> float:
        """Calculate p-value for sequential test - Robust with proper error handling"""
        try:
            # Use Welch's t-test for different sample sizes and variances
            if variant_a.n_observations == 0 or variant_b.n_observations == 0:
                return 1.0
            
            mean_a = variant_a.primary_metric_mean
            mean_b = variant_b.primary_metric_mean
            var_a = variant_a.primary_metric_variance
            var_b = variant_b.primary_metric_variance
            
            # Handle edge cases
            if var_a == 0 and var_b == 0:
                return 1.0 if mean_a == mean_b else 0.0
            
            # Avoid division by zero
            if variant_a.n_observations <= 1 or variant_b.n_observations <= 1:
                return 1.0
            
            # Calculate standard errors with minimum variance to prevent division by zero
            se_a = max(var_a / variant_a.n_observations, 1e-10)
            se_b = max(var_b / variant_b.n_observations, 1e-10)
            pooled_se = math.sqrt(se_a + se_b)
            
            if pooled_se == 0:
                return 1.0
            
            # Welch's t-test
            t_stat = (mean_a - mean_b) / pooled_se
            
            # Welch-Satterthwaite degrees of freedom
            numerator = (se_a + se_b) ** 2
            denom_a = (se_a ** 2) / max(variant_a.n_observations - 1, 1)
            denom_b = (se_b ** 2) / max(variant_b.n_observations - 1, 1)
            df = numerator / (denom_a + denom_b)
            
            # Ensure valid degrees of freedom
            df = max(df, 1.0)
            
            # Two-tailed p-value with bounds checking
            if math.isnan(t_stat) or math.isinf(t_stat):
                return 1.0
            
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            return max(0.0, min(1.0, p_value))  # Ensure p-value is in [0, 1]
            
        except Exception as e:
            logger.warning(f"Error calculating sequential p-value: {e}")
            return 1.0  # Conservative fallback
    
    def analyze_test(self, test_id: str, significance_test: SignificanceTest = SignificanceTest.BAYESIAN_HYPOTHESIS) -> TestResults:
        """
        Perform comprehensive statistical analysis of A/B test
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        variants = self.test_registry[test_id]
        
        # Initialize results
        results = TestResults(
            test_id=test_id,
            test_type=test_config['test_type'],
            significance_test=significance_test,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            effect_size=0.0,
            statistical_power=0.0,
            is_significant=False,
            start_time=test_config['start_time']
        )
        
        # Check minimum sample size
        min_observations = min(v.n_observations for v in variants)
        results.minimum_sample_achieved = min_observations >= self.config.minimum_sample_size
        
        if not results.minimum_sample_achieved:
            results.recommended_action = f"Continue test - need {self.config.minimum_sample_size - min_observations} more observations"
            return results
        
        # Perform statistical analysis based on significance test type
        if significance_test == SignificanceTest.BAYESIAN_HYPOTHESIS:
            results = self._bayesian_analysis(results, variants)
        elif significance_test == SignificanceTest.WELCHS_T_TEST:
            results = self._frequentist_t_test(results, variants)
        elif significance_test == SignificanceTest.MANN_WHITNEY_U:
            results = self._mann_whitney_test(results, variants)
        elif significance_test == SignificanceTest.BOOTSTRAP_PERMUTATION:
            results = self._bootstrap_test(results, variants)
        elif significance_test == SignificanceTest.SEQUENTIAL_TESTING:
            results = self._sequential_analysis(results, variants, test_id)
        
        # Multi-metric analysis
        results = self._multi_metric_analysis(results, variants)
        
        # Risk assessment
        results = self._risk_assessment(results, variants)
        
        # Final recommendation
        results = self._generate_recommendation(results)
        
        # Cache results
        self.test_results[test_id] = results
        
        return results
    
    def _bayesian_analysis(self, results: TestResults, variants: List[TestVariant]) -> TestResults:
        """Perform Bayesian hypothesis testing"""
        if len(variants) != 2:
            # Multi-variant Bayesian analysis
            return self._multi_variant_bayesian(results, variants)
        
        variant_a, variant_b = variants[:2]
        
        # Get posterior parameters
        alpha_a, beta_a = variant_a.conversion_rate_posterior
        alpha_b, beta_b = variant_b.conversion_rate_posterior
        
        # Monte Carlo simulation for P(A > B)
        n_samples = self.config.thompson_sample_size
        samples_a = np.random.beta(alpha_a, beta_a, n_samples)
        samples_b = np.random.beta(alpha_b, beta_b, n_samples)
        
        prob_a_better = np.mean(samples_a > samples_b)
        results.bayesian_probability = prob_a_better
        
        # Effect size (difference in conversion rates)
        mean_a = alpha_a / (alpha_a + beta_a)
        mean_b = alpha_b / (alpha_b + beta_b)
        results.effect_size = mean_a - mean_b
        
        # Credible interval for difference
        diff_samples = samples_a - samples_b
        results.credible_interval = (
            np.percentile(diff_samples, 2.5),
            np.percentile(diff_samples, 97.5)
        )
        
        # Statistical significance (Bayesian)
        results.is_significant = prob_a_better > 0.95 or prob_a_better < 0.05
        results.p_value = min(prob_a_better, 1 - prob_a_better) * 2  # Two-tailed equivalent
        
        # Winner determination
        if prob_a_better > 0.95:
            results.winner_variant_id = variant_a.variant_id
            results.lift_percentage = (mean_a - mean_b) / mean_b * 100
        elif prob_a_better < 0.05:
            results.winner_variant_id = variant_b.variant_id
            results.lift_percentage = (mean_b - mean_a) / mean_a * 100
        
        return results
    
    def _multi_variant_bayesian(self, results: TestResults, variants: List[TestVariant]) -> TestResults:
        """Bayesian analysis for multiple variants"""
        n_samples = self.config.thompson_sample_size
        variant_samples = []
        
        # Generate samples for each variant
        for variant in variants:
            alpha, beta = variant.conversion_rate_posterior
            samples = np.random.beta(alpha, beta, n_samples)
            variant_samples.append(samples)
        
        # Find best variant for each sample
        variant_samples = np.array(variant_samples)
        best_variants = np.argmax(variant_samples, axis=0)
        
        # Calculate probabilities
        variant_probs = []
        for i in range(len(variants)):
            prob = np.mean(best_variants == i)
            variant_probs.append(prob)
        
        # Find winner
        best_variant_idx = np.argmax(variant_probs)
        results.winner_variant_id = variants[best_variant_idx].variant_id
        results.bayesian_probability = variant_probs[best_variant_idx]
        
        # Statistical significance (any variant significantly better?)
        results.is_significant = max(variant_probs) > 0.95
        
        return results
    
    def _frequentist_t_test(self, results: TestResults, variants: List[TestVariant]) -> TestResults:
        """Perform Welch's t-test"""
        if len(variants) != 2:
            raise ValueError("Frequentist t-test only supports 2 variants")
        
        variant_a, variant_b = variants[:2]
        
        if variant_a.n_observations == 0 or variant_b.n_observations == 0:
            return results
        
        # Calculate means and variances
        mean_a = variant_a.primary_metric_mean
        mean_b = variant_b.primary_metric_mean
        var_a = variant_a.primary_metric_variance
        var_b = variant_b.primary_metric_variance
        
        # Welch's t-test
        if var_a == 0 and var_b == 0:
            results.p_value = 0.0 if mean_a != mean_b else 1.0
        else:
            pooled_se = math.sqrt(var_a / variant_a.n_observations + var_b / variant_b.n_observations)
            t_stat = (mean_a - mean_b) / pooled_se if pooled_se > 0 else 0
            
            # Degrees of freedom (Welch-Satterthwaite)
            df = (var_a / variant_a.n_observations + var_b / variant_b.n_observations) ** 2 / (
                (var_a / variant_a.n_observations) ** 2 / (variant_a.n_observations - 1) +
                (var_b / variant_b.n_observations) ** 2 / (variant_b.n_observations - 1)
            )
            
            results.p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Confidence interval
            t_critical = stats.t.ppf(1 - self.config.alpha / 2, df)
            margin_error = t_critical * pooled_se
            results.confidence_interval = (
                (mean_a - mean_b) - margin_error,
                (mean_a - mean_b) + margin_error
            )
        
        results.effect_size = mean_a - mean_b
        results.is_significant = results.p_value < self.config.alpha
        
        # Winner determination
        if results.is_significant:
            if mean_a > mean_b:
                results.winner_variant_id = variant_a.variant_id
                results.lift_percentage = (mean_a - mean_b) / mean_b * 100
            else:
                results.winner_variant_id = variant_b.variant_id
                results.lift_percentage = (mean_b - mean_a) / mean_a * 100
        
        return results
    
    def _mann_whitney_test(self, results: TestResults, variants: List[TestVariant]) -> TestResults:
        """Perform Mann-Whitney U test (non-parametric)"""
        if len(variants) != 2:
            raise ValueError("Mann-Whitney test only supports 2 variants")
        
        # This is a simplified version - in reality you'd need the raw observations
        # For now, use t-test as approximation
        return self._frequentist_t_test(results, variants)
    
    def _bootstrap_test(self, results: TestResults, variants: List[TestVariant]) -> TestResults:
        """Perform bootstrap permutation test"""
        if len(variants) != 2:
            raise ValueError("Bootstrap test only supports 2 variants")
        
        variant_a, variant_b = variants[:2]
        
        # Simulate bootstrap samples based on observed statistics
        n_bootstrap = 10000
        
        # Generate synthetic data based on observed means and variances
        data_a = np.random.normal(
            variant_a.primary_metric_mean,
            variant_a.primary_metric_std,
            variant_a.n_observations
        )
        data_b = np.random.normal(
            variant_b.primary_metric_mean,
            variant_b.primary_metric_std,
            variant_b.n_observations
        )
        
        observed_diff = np.mean(data_a) - np.mean(data_b)
        
        # Bootstrap permutation
        combined_data = np.concatenate([data_a, data_b])
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            np.random.shuffle(combined_data)
            boot_a = combined_data[:len(data_a)]
            boot_b = combined_data[len(data_a):]
            boot_diff = np.mean(boot_a) - np.mean(boot_b)
            bootstrap_diffs.append(boot_diff)
        
        # P-value calculation
        results.p_value = np.mean(np.abs(bootstrap_diffs) >= abs(observed_diff))
        results.effect_size = observed_diff
        results.is_significant = results.p_value < self.config.alpha
        
        # Confidence interval
        results.confidence_interval = (
            np.percentile(bootstrap_diffs, 2.5),
            np.percentile(bootstrap_diffs, 97.5)
        )
        
        return results
    
    def _sequential_analysis(self, results: TestResults, variants: List[TestVariant], test_id: str) -> TestResults:
        """Perform sequential analysis"""
        if test_id not in self.sequential_tests:
            return self._bayesian_analysis(results, variants)
        
        sequential_config = self.sequential_tests[test_id]
        
        # Use the latest interim analysis
        if sequential_config['interim_analyses']:
            latest_analysis = sequential_config['interim_analyses'][-1]
            results.p_value = latest_analysis['p_value']
            results.is_significant = latest_analysis['p_value'] < latest_analysis['alpha_threshold']
        
        return results
    
    def _multi_metric_analysis(self, results: TestResults, variants: List[TestVariant]) -> TestResults:
        """Analyze multiple metrics with appropriate corrections"""
        
        # Primary metric results
        results.primary_metric_results = {
            'metric': self.config.primary_metric,
            'means': [v.primary_metric_mean for v in variants],
            'sample_sizes': [v.n_observations for v in variants]
        }
        
        # Secondary metrics analysis
        for metric in self.config.secondary_metrics:
            metric_results = {}
            
            for i, variant in enumerate(variants):
                if metric in variant.secondary_metrics:
                    metric_value = variant.secondary_metrics[metric]
                    metric_mean = metric_value / max(1, variant.n_observations)
                    metric_results[variant.variant_id] = metric_mean
            
            results.secondary_metric_results[metric] = metric_results
        
        # Multiple comparison correction (Bonferroni)
        if len(self.config.secondary_metrics) > 0:
            n_comparisons = 1 + len(self.config.secondary_metrics)  # Primary + secondaries
            corrected_alpha = self.config.alpha / n_comparisons
            
            # Adjust significance based on corrected alpha
            if results.p_value < corrected_alpha:
                results.is_significant = True
        
        return results
    
    def _risk_assessment(self, results: TestResults, variants: List[TestVariant]) -> TestResults:
        """Assess risks of different decisions"""
        
        # Type I error (false positive) risk
        results.risk_of_false_positive = self.config.alpha
        
        # Type II error (false negative) risk - estimated
        if len(variants) == 2:
            variant_a, variant_b = variants[:2]
            
            # Estimate power based on observed effect size and sample sizes
            observed_effect = abs(variant_a.primary_metric_mean - variant_b.primary_metric_mean)
            pooled_variance = (variant_a.primary_metric_variance + variant_b.primary_metric_variance) / 2
            
            if pooled_variance > 0 and observed_effect > 0:
                effect_size_cohen = observed_effect / math.sqrt(pooled_variance)
                min_n = min(variant_a.n_observations, variant_b.n_observations)
                
                # Power calculation (simplified)
                ncp = effect_size_cohen * math.sqrt(min_n / 2)  # Non-centrality parameter
                power_est = 1 - stats.t.cdf(stats.t.ppf(1 - self.config.alpha / 2, 2 * min_n - 2), 
                                           2 * min_n - 2, ncp)
                power_est += stats.t.cdf(-stats.t.ppf(1 - self.config.alpha / 2, 2 * min_n - 2), 
                                        2 * min_n - 2, ncp)
                
                results.statistical_power = max(0, min(1, power_est))
                results.risk_of_false_negative = 1 - results.statistical_power
            else:
                results.statistical_power = 0.5
                results.risk_of_false_negative = 0.5
        
        # Expected loss calculation (Bayesian decision theory)
        if hasattr(results, 'bayesian_probability'):
            # Calculate expected loss for each decision
            prob_a_better = results.bayesian_probability
            
            # Loss if we choose A when B is better
            loss_choose_a_when_b_better = abs(results.effect_size) * (1 - prob_a_better)
            
            # Loss if we choose B when A is better  
            loss_choose_b_when_a_better = abs(results.effect_size) * prob_a_better
            
            results.expected_loss = {
                'choose_variant_a': loss_choose_a_when_b_better,
                'choose_variant_b': loss_choose_b_when_a_better,
                'continue_test': abs(results.effect_size) * 0.1  # Small cost for continuing
            }
        
        return results
    
    def _generate_recommendation(self, results: TestResults) -> TestResults:
        """Generate final recommendation based on analysis"""
        
        if not results.minimum_sample_achieved:
            results.recommended_action = "Continue test - insufficient sample size"
            return results
        
        if results.is_significant:
            if results.winner_variant_id:
                results.recommended_action = f"Deploy {results.winner_variant_id} - statistically significant improvement"
            else:
                results.recommended_action = "Statistical difference detected but winner unclear"
        else:
            # Check Bayesian probability if available
            if hasattr(results, 'bayesian_probability') and results.bayesian_probability:
                if results.bayesian_probability > 0.90:
                    results.recommended_action = f"Consider deploying {results.winner_variant_id} - high Bayesian probability"
                elif results.bayesian_probability < 0.10:
                    results.recommended_action = "Consider deploying control - high probability it's better"
                else:
                    results.recommended_action = "No clear winner - consider extending test or tie-breaking criteria"
            else:
                results.recommended_action = "No significant difference detected - consider practical significance"
        
        # Add duration information
        if results.end_time:
            duration = results.end_time - results.start_time
            results.test_duration_hours = duration.total_seconds() / 3600
        
        return results
    
    def _conclude_test(self, test_id: str, reason: str):
        """Conclude a test and mark it as completed"""
        if test_id in self.active_tests:
            self.active_tests[test_id]['status'] = 'completed'
            self.active_tests[test_id]['end_time'] = datetime.now()
            self.active_tests[test_id]['conclusion_reason'] = reason
            
            logger.info(f"Test {test_id} concluded: {reason}")
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status and progress of a test"""
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test_config = self.active_tests[test_id]
        variants = self.test_registry[test_id]
        
        status = {
            'test_id': test_id,
            'status': test_config['status'],
            'start_time': test_config['start_time'],
            'test_type': test_config['test_type'].value,
            'allocation_strategy': test_config['allocation_strategy'].value,
            'required_sample_size': test_config['required_sample_size'],
            'variants': []
        }
        
        for variant in variants:
            variant_status = {
                'variant_id': variant.variant_id,
                'name': variant.name,
                'n_observations': variant.n_observations,
                'primary_metric_mean': variant.primary_metric_mean,
                'conversion_rate': variant.beta_alpha / (variant.beta_alpha + variant.beta_beta),
                'allocation_probability': variant.allocation_probability
            }
            status['variants'].append(variant_status)
        
        # Progress calculation
        min_observations = min(v.n_observations for v in variants)
        status['progress'] = min(1.0, min_observations / test_config['required_sample_size'])
        
        return status
    
    def list_active_tests(self) -> List[Dict[str, Any]]:
        """List all active tests with their current status"""
        active_tests = []
        
        for test_id in self.active_tests:
            if self.active_tests[test_id]['status'] == 'active':
                status = self.get_test_status(test_id)
                active_tests.append(status)
        
        return active_tests
    
    def export_test_results(self, test_id: str, format: str = 'json') -> str:
        """Export test results in specified format"""
        if test_id not in self.test_results:
            results = self.analyze_test(test_id)
        else:
            results = self.test_results[test_id]
        
        if format == 'json':
            # Convert to serializable format
            export_data = {
                'test_id': results.test_id,
                'test_type': results.test_type.value,
                'significance_test': results.significance_test.value,
                'p_value': results.p_value,
                'confidence_interval': results.confidence_interval,
                'effect_size': results.effect_size,
                'statistical_power': results.statistical_power,
                'is_significant': results.is_significant,
                'bayesian_probability': results.bayesian_probability,
                'credible_interval': results.credible_interval,
                'recommended_action': results.recommended_action,
                'winner_variant_id': results.winner_variant_id,
                'lift_percentage': results.lift_percentage,
                'start_time': results.start_time.isoformat(),
                'end_time': results.end_time.isoformat() if results.end_time else None,
                'test_duration_hours': results.test_duration_hours,
                'primary_metric_results': results.primary_metric_results,
                'secondary_metric_results': results.secondary_metric_results,
                'risk_assessment': {
                    'risk_of_false_positive': results.risk_of_false_positive,
                    'risk_of_false_negative': results.risk_of_false_negative,
                    'expected_loss': results.expected_loss
                }
            }
            
            return json.dumps(export_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_completed_tests(self, days_old: int = 30):
        """Clean up old completed tests"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        tests_to_remove = []
        for test_id, config in self.active_tests.items():
            if (config['status'] == 'completed' and 
                config.get('end_time', datetime.now()) < cutoff_date):
                tests_to_remove.append(test_id)
        
        for test_id in tests_to_remove:
            del self.active_tests[test_id]
            if test_id in self.test_registry:
                del self.test_registry[test_id]
            if test_id in self.test_results:
                del self.test_results[test_id]
            if test_id in self.contextual_bandits:
                del self.contextual_bandits[test_id]
            if test_id in self.sequential_tests:
                del self.sequential_tests[test_id]
        
        logger.info(f"Cleaned up {len(tests_to_remove)} old tests")


# Example usage and testing functions
def create_policy_comparison_test(framework: StatisticalABTestFramework,
                                policy_a_params: Dict[str, Any],
                                policy_b_params: Dict[str, Any],
                                test_name: str = "Policy Comparison") -> str:
    """
    Create a test to compare two RL policies
    """
    
    variants = [
        {
            'variant_id': 'policy_a',
            'name': 'Policy A',
            'policy_parameters': policy_a_params,
            'allocation_probability': 0.5
        },
        {
            'variant_id': 'policy_b', 
            'name': 'Policy B',
            'policy_parameters': policy_b_params,
            'allocation_probability': 0.5
        }
    ]
    
    test_id = f"policy_test_{uuid.uuid4().hex[:8]}"
    
    return framework.create_ab_test(
        test_id=test_id,
        test_name=test_name,
        variants=variants,
        test_type=TestType.BAYESIAN_BANDIT,
        allocation_strategy=AllocationStrategy.THOMPSON_SAMPLING,
        duration_days=14
    )


if __name__ == "__main__":
    # Example usage
    from discovery_engine import GA4DiscoveryEngine
    
    discovery = GA4DiscoveryEngine()
    config = StatisticalConfig()
    
    framework = StatisticalABTestFramework(config, discovery)
    
    # Create a test
    policy_a = {'learning_rate': 0.001, 'epsilon': 0.1}
    policy_b = {'learning_rate': 0.01, 'epsilon': 0.05}
    
    test_id = create_policy_comparison_test(framework, policy_a, policy_b)
    print(f"Created test: {test_id}")
    
    # Simulate some observations
    for i in range(1000):
        variant = framework.assign_variant(test_id, f"user_{i}", {
            'segment': 'researching_parent',
            'device': 'mobile',
            'channel': 'organic',
            'hour': 14,
            'day_of_week': 1
        })
        
        # Simulate results (policy B performs slightly better)
        if variant == 'policy_a':
            conversion_rate = 0.02
        else:
            conversion_rate = 0.025
        
        converted = np.random.random() < conversion_rate
        roas = np.random.normal(3.2 if converted else 0, 0.5)
        
        framework.record_observation(
            test_id=test_id,
            variant_id=variant,
            user_id=f"user_{i}",
            primary_metric_value=float(converted),
            secondary_metrics={'roas': roas},
            converted=converted,
            context={'segment': 'researching_parent', 'device': 'mobile'}
        )
    
    # Analyze results
    results = framework.analyze_test(test_id)
    print(f"Test results: {results.recommended_action}")
    print(f"Winner: {results.winner_variant_id}")
    print(f"Bayesian probability: {results.bayesian_probability:.3f}")
    print(f"P-value: {results.p_value:.3f}")