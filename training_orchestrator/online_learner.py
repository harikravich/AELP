"""
Online Learning System for Real-Time Agent Improvement

This module implements online learning capabilities that allow agents to learn and improve
while serving real traffic. It balances exploration vs exploitation using Thompson sampling
for bid optimization with safety guardrails and incremental model updates.

Key Features:
- Thompson sampling for multi-armed bandit optimization
- Safe exploration with configurable guardrails
- Incremental model updates without service interruption
- Real-time performance monitoring and adaptation
- Budget and safety constraint enforcement
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import copy
from scipy.stats import beta
try:
    import redis
except ImportError:
    redis = None

try:
    from google.cloud import pubsub_v1, bigquery
except ImportError:
    pubsub_v1 = None
    bigquery = None

try:
    from .rl_agents.base_agent import BaseRLAgent, AgentConfig
except ImportError:
    # Fallback for direct imports - create minimal mock classes for testing
    from dataclasses import dataclass, field
    from typing import List, Dict, Any
    import abc
    
    @dataclass
    class AgentConfig:
        state_dim: int = 128
        action_dim: int = 64
        learning_rate: float = 3e-4
    
    class BaseRLAgent(abc.ABC):
        def __init__(self, config, agent_id):
            self.config = config
            self.agent_id = agent_id
        
        @abc.abstractmethod
        async def select_action(self, state, deterministic=False):
            pass
        
        @abc.abstractmethod
        def update_policy(self, experiences):
            pass
        
        @abc.abstractmethod
        def get_state(self):
            pass
        
        @abc.abstractmethod
        def load_state(self, state):
            pass


logger = logging.getLogger(__name__)


@dataclass
class OnlineLearnerConfig:
    """Configuration for online learning system"""
    
    # Thompson Sampling Parameters
    ts_prior_alpha: float = 1.0  # Prior success count
    ts_prior_beta: float = 1.0   # Prior failure count
    ts_update_rate: float = 0.1  # Learning rate for posterior updates
    ts_exploration_decay: float = 0.995  # Decay rate for exploration
    min_exploration_rate: float = 0.05  # Minimum exploration probability
    
    # Safe Exploration Parameters
    safety_threshold: float = 0.8  # Minimum performance threshold relative to baseline
    safety_window_size: int = 100  # Number of recent episodes for safety evaluation
    safety_violation_limit: int = 5  # Max consecutive safety violations
    emergency_fallback: bool = True  # Enable emergency use baseline
    
    # Budget Safety
    max_budget_risk: float = 0.1  # Max fraction of daily budget to risk on exploration
    budget_safety_margin: float = 0.2  # Safety margin for budget allocation
    max_loss_threshold: float = 0.15  # Max acceptable loss rate
    
    # Online Update Parameters
    online_update_frequency: int = 50  # Episodes between model updates
    update_batch_size: int = 32  # Batch size for incremental updates
    learning_rate_schedule: str = "adaptive"  # "constant", "decay", "adaptive"
    gradient_accumulation_steps: int = 4  # Steps to accumulate gradients
    
    # Performance Monitoring
    performance_window_size: int = 200  # Episodes for performance evaluation
    min_episodes_before_update: int = 20  # Minimum episodes before first update
    convergence_threshold: float = 0.01  # Threshold for detecting convergence
    
    # Multi-Armed Bandit Setup
    bandit_arms: List[str] = field(default_factory=lambda: [
        "conservative", "balanced", "aggressive", "experimental"
    ])
    arm_selection_method: str = "thompson_sampling"  # "epsilon_greedy", "ucb", "thompson_sampling"
    
    # Data Storage and Logging
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1  # Separate DB for online learning
    log_interval: int = 10  # Log metrics every N episodes
    
    # Integration Settings
    enable_real_time_updates: bool = True
    enable_performance_prediction: bool = True
    enable_competitive_intelligence: bool = True


@dataclass
class ExplorationAction:
    """Represents an exploration action with metadata"""
    action: Dict[str, Any]
    arm_id: str
    confidence: float
    risk_level: str  # "low", "medium", "high"
    expected_reward: float
    uncertainty: float
    safety_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyConstraints:
    """Safety constraints for online exploration"""
    max_budget_deviation: float = 0.2  # Max deviation from baseline budget
    min_roi_threshold: float = 0.5  # Minimum ROI relative to baseline
    max_cpa_multiplier: float = 2.0  # Maximum CPA increase allowed
    blacklisted_audiences: List[str] = field(default_factory=list)
    restricted_times: List[Tuple[int, int]] = field(default_factory=list)  # (start_hour, end_hour)
    

class ThompsonSamplerArm:
    """Individual arm for Thompson sampling bandit"""
    
    def __init__(self, arm_id: str, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.arm_id = arm_id
        self.alpha = prior_alpha  # Success count + prior
        self.beta = prior_beta    # Failure count + prior
        self.total_pulls = 0
        self.total_rewards = 0.0
        self.recent_rewards = deque(maxlen=100)
        self.last_updated = datetime.now()
        
    def sample(self) -> float:
        """Sample from Beta distribution for this arm"""
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, reward: float, success: bool = None):
        """Update arm statistics with new observation"""
        if success is None:
            success = reward > 0.5  # Default threshold for binary success
            
        self.total_pulls += 1
        self.total_rewards += reward
        self.recent_rewards.append(reward)
        
        if success:
            self.alpha += 1
        else:
            self.beta += 1
            
        self.last_updated = datetime.now()
    
    def get_mean_reward(self) -> float:
        """Get empirical mean reward"""
        return self.total_rewards / max(1, self.total_pulls)
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for the arm's performance"""
        if self.total_pulls < 10:
            return (0.0, 1.0)  # Wide interval for insufficient data
            
        # Use Beta distribution percentiles
        lower = beta.ppf((1 - confidence) / 2, self.alpha, self.beta)
        upper = beta.ppf(1 - (1 - confidence) / 2, self.alpha, self.beta)
        return (lower, upper)


class OnlineLearner:
    """
    Online learning system that continuously improves agent performance while serving real traffic.
    Implements Thompson sampling for bid optimization with comprehensive safety mechanisms.
    """
    
    def __init__(self, agent: BaseRLAgent, config: OnlineLearnerConfig):
        self.agent = agent
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize Thompson sampling bandits
        self.bandit_arms = {
            arm_id: ThompsonSamplerArm(arm_id, config.ts_prior_alpha, config.ts_prior_beta)
            for arm_id in config.bandit_arms
        }
        
        # Performance tracking
        self.episode_history = deque(maxlen=config.performance_window_size)
        self.baseline_performance = None
        self.safety_violations = 0
        self.consecutive_violations = 0
        
        # Online learning state
        self.online_updates_count = 0
        self.gradient_buffer = []
        self.learning_rate_scheduler = self._create_lr_scheduler()
        
        # Safety monitoring
        self.safety_constraints = SafetyConstraints()
        self.emergency_mode = False
        self.fallback_agent = None
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.update_lock = threading.Lock()
        
        # External services
        self._init_external_services()
        
        # Performance cache
        self.performance_cache = {}
        self.last_cache_update = datetime.now()
        
        self.logger.info(f"Online learner initialized with {len(self.bandit_arms)} arms")
    
    def _init_external_services(self):
        """Initialize Redis and other external services"""
        # Initialize Redis if available
        if redis is not None:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True
                )
                
                # Test connection
                self.redis_client.ping()
                
            except Exception as e:
                self.logger.warning(f"Redis not available: {e}")
                self.redis_client = None
        else:
            self.logger.info("Redis module not available, skipping Redis initialization")
            self.redis_client = None
        
        # Initialize BigQuery if available
        if bigquery is not None:
            try:
                self.bigquery_client = bigquery.Client()
            except Exception as e:
                self.logger.warning(f"BigQuery not available: {e}")
                self.bigquery_client = None
        else:
            self.logger.info("BigQuery module not available, skipping BigQuery initialization")
            self.bigquery_client = None
    
    def _create_lr_scheduler(self) -> Callable[[int], float]:
        """Create learning rate scheduler based on configuration"""
        if self.config.learning_rate_schedule == "constant":
            return lambda step: self.agent.config.learning_rate
        elif self.config.learning_rate_schedule == "decay":
            return lambda step: self.agent.config.learning_rate * (0.995 ** step)
        else:  # adaptive
            return self._adaptive_lr_schedule
    
    def _adaptive_lr_schedule(self, step: int) -> float:
        """Adaptive learning rate based on recent performance"""
        if len(self.episode_history) < 20:
            return self.agent.config.learning_rate
        
        recent_rewards = [ep['reward'] for ep in list(self.episode_history)[-20:]]
        reward_variance = np.var(recent_rewards)
        
        # Higher variance -> lower learning rate for stability
        lr_multiplier = 1.0 / (1.0 + reward_variance)
        return self.agent.config.learning_rate * lr_multiplier
    
    async def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """
        Select action using explore vs exploit strategy with Thompson sampling.
        
        Args:
            state: Current environment state
            deterministic: If True, use exploitation only
            
        Returns:
            Action dictionary with exploration metadata
        """
        # Check if we're in emergency mode
        if self.emergency_mode and self.fallback_agent:
            return await self.fallback_agent.select_action(state, deterministic=True)
        
        # Determine exploration vs exploitation
        if deterministic:
            exploration_action = await self._exploit_action(state)
        else:
            should_explore = await self._should_explore(state)
            if should_explore:
                exploration_action = await self._explore_action(state)
            else:
                exploration_action = await self._exploit_action(state)
        
        # Apply safety constraints
        safe_action = await self._apply_safety_constraints(exploration_action, state)
        
        # Log action selection
        await self._log_action_selection(safe_action, state)
        
        return safe_action.action
    
    async def _should_explore(self, state: Dict[str, Any]) -> bool:
        """Determine if we should explore based on current conditions"""
        # Never explore if in emergency mode
        if self.emergency_mode:
            return False
        
        # Check budget constraints
        budget_info = state.get("budget_constraints", {})
        daily_spent = budget_info.get("daily_spent", 0)
        daily_budget = budget_info.get("daily_budget", 100)
        
        # Don't explore if we've spent too much of our budget
        if daily_spent / daily_budget > (1.0 - self.config.max_budget_risk):
            return False
        
        # Use Thompson sampling to decide exploration
        arm_samples = {arm_id: arm.sample() for arm_id, arm in self.bandit_arms.items()}
        best_arm = max(arm_samples.keys(), key=lambda x: arm_samples[x])
        
        # Explore if experimental/aggressive arms are selected
        return best_arm in ["aggressive", "experimental"]
    
    async def _explore_action(self, state: Dict[str, Any]) -> ExplorationAction:
        """Generate exploration action using Thompson sampling"""
        # Sample from all arms
        arm_samples = {}
        for arm_id, arm in self.bandit_arms.items():
            arm_samples[arm_id] = arm.sample()
        
        # Select best arm
        selected_arm = max(arm_samples.keys(), key=lambda x: arm_samples[x])
        selected_arm_obj = self.bandit_arms[selected_arm]
        
        # Generate base action from agent
        base_action = await self.agent.select_action(state, deterministic=False)
        
        # Modify action based on selected arm
        modified_action = await self._modify_action_for_arm(base_action, selected_arm, state)
        
        # Calculate risk and confidence metrics
        confidence = arm_samples[selected_arm]
        uncertainty = 1.0 / (selected_arm_obj.alpha + selected_arm_obj.beta)
        risk_level = self._calculate_risk_level(modified_action, base_action)
        
        return ExplorationAction(
            action=modified_action,
            arm_id=selected_arm,
            confidence=confidence,
            risk_level=risk_level,
            expected_reward=selected_arm_obj.get_mean_reward(),
            uncertainty=uncertainty,
            safety_score=await self._calculate_safety_score(modified_action, state),
            metadata={
                "base_action": base_action,
                "arm_samples": arm_samples,
                "selection_time": datetime.now().isoformat()
            }
        )
    
    async def _exploit_action(self, state: Dict[str, Any]) -> ExplorationAction:
        """Generate exploitation action using best known strategy"""
        # Use the arm with highest empirical mean
        best_arm_id = max(
            self.bandit_arms.keys(),
            key=lambda x: self.bandit_arms[x].get_mean_reward()
        )
        
        best_arm = self.bandit_arms[best_arm_id]
        
        # Generate conservative action from agent
        action = await self.agent.select_action(state, deterministic=True)
        
        return ExplorationAction(
            action=action,
            arm_id=best_arm_id,
            confidence=1.0,
            risk_level="low",
            expected_reward=best_arm.get_mean_reward(),
            uncertainty=0.0,
            safety_score=1.0,
            metadata={
                "exploitation_mode": True,
                "selection_time": datetime.now().isoformat()
            }
        )
    
    async def _modify_action_for_arm(self, base_action: Dict[str, Any], arm_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Modify base action according to the selected arm's strategy"""
        modified_action = copy.deepcopy(base_action)
        
        if arm_id == "conservative":
            # Reduce budget and use safer bid strategies
            modified_action["budget"] = min(modified_action["budget"] * 0.8, base_action["budget"])
            if modified_action["bid_strategy"] == "cpa":
                modified_action["bid_amount"] *= 0.9
        
        elif arm_id == "balanced":
            # Keep base action mostly unchanged
            pass
        
        elif arm_id == "aggressive":
            # Increase budget and bids for potentially higher returns
            modified_action["budget"] = min(modified_action["budget"] * 1.3, base_action["budget"] * 1.5)
            if modified_action["bid_strategy"] in ["cpc", "cpm"]:
                modified_action["bid_amount"] *= 1.2
            modified_action["audience_size"] = min(1.0, modified_action["audience_size"] * 1.1)
        
        elif arm_id == "experimental":
            # Try different creative types and audience combinations
            creative_types = ["image", "video", "carousel"]
            if modified_action["creative_type"] in creative_types:
                # Try a different creative type
                other_types = [t for t in creative_types if t != modified_action["creative_type"]]
                modified_action["creative_type"] = np.random.choice(other_types)
            
            # Enable A/B testing more often
            modified_action["ab_test_enabled"] = True
            modified_action["ab_test_split"] = np.random.uniform(0.3, 0.7)
        
        return modified_action
    
    def _calculate_risk_level(self, modified_action: Dict[str, Any], base_action: Dict[str, Any]) -> str:
        """Calculate risk level of the modified action"""
        budget_increase = (modified_action["budget"] - base_action["budget"]) / base_action["budget"]
        bid_increase = (modified_action["bid_amount"] - base_action["bid_amount"]) / base_action["bid_amount"]
        
        risk_score = budget_increase + bid_increase
        
        if risk_score > 0.3:
            return "high"
        elif risk_score > 0.1:
            return "medium"
        else:
            return "low"
    
    async def _calculate_safety_score(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Calculate safety score for an action given the current state"""
        score = 1.0
        
        # Budget safety check
        budget_info = state.get("budget_constraints", {})
        daily_budget = budget_info.get("daily_budget", 100)
        if action["budget"] > daily_budget * self.config.budget_safety_margin:
            score -= 0.3
        
        # Bid amount safety check
        historical_avg_bid = state.get("performance_history", {}).get("avg_bid", 1.0)
        if action["bid_amount"] > historical_avg_bid * self.safety_constraints.max_cpa_multiplier:
            score -= 0.2
        
        # Audience safety check
        if action["target_audience"] in self.safety_constraints.blacklisted_audiences:
            score -= 0.5
        
        # Time-based constraints
        current_hour = datetime.now().hour
        for start_hour, end_hour in self.safety_constraints.restricted_times:
            if start_hour <= current_hour <= end_hour:
                score -= 0.2
                break
        
        return max(0.0, score)
    
    async def _apply_safety_constraints(self, exploration_action: ExplorationAction, state: Dict[str, Any]) -> ExplorationAction:
        """Apply safety constraints to exploration action"""
        action = exploration_action.action
        
        # Check if action violates safety constraints
        if exploration_action.safety_score < 0.5:
            self.logger.warning(f"Action failed safety check, falling back to safer option")
            # Fall back to conservative arm
            return await self._exploit_action(state)
        
        # Apply budget caps
        budget_info = state.get("budget_constraints", {})
        max_safe_budget = budget_info.get("daily_budget", 100) * (1 - self.config.budget_safety_margin)
        action["budget"] = min(action["budget"], max_safe_budget)
        
        return exploration_action
    
    async def online_update(self, experiences: List[Dict[str, Any]], force_update: bool = False) -> Dict[str, float]:
        """
        Perform online update of the agent's policy based on recent experiences.
        
        Args:
            experiences: List of recent experiences (state, action, reward, next_state, done)
            force_update: Force update even if conditions aren't met
            
        Returns:
            Dict of update metrics
        """
        if not experiences:
            return {}
        
        # Check if we should update
        if not force_update and not self._should_update():
            return {"status": "skipped", "reason": "conditions_not_met"}
        
        with self.update_lock:
            try:
                # Prepare batch for update
                batch = self._prepare_update_batch(experiences)
                
                if len(batch) < self.config.update_batch_size and not force_update:
                    # Store in gradient buffer for later
                    self.gradient_buffer.extend(batch)
                    return {"status": "buffered", "batch_size": len(batch)}
                
                # Combine with buffered gradients
                if self.gradient_buffer:
                    batch.extend(self.gradient_buffer)
                    self.gradient_buffer = []
                
                # Perform incremental update
                update_metrics = await self._perform_incremental_update(batch)
                
                # Update learning rate
                self.online_updates_count += 1
                new_lr = self.learning_rate_scheduler(self.online_updates_count)
                self._update_learning_rate(new_lr)
                
                # Log update
                await self._log_online_update(update_metrics)
                
                return update_metrics
                
            except Exception as e:
                self.logger.error(f"Online update failed: {e}")
                return {"status": "failed", "error": str(e)}
    
    def _should_update(self) -> bool:
        """Determine if we should perform an online update"""
        # Need minimum number of episodes
        if len(self.episode_history) < self.config.min_episodes_before_update:
            return False
        
        # Don't update too frequently
        episodes_since_update = len(self.episode_history) % self.config.online_update_frequency
        if episodes_since_update != 0:
            return False
        
        # Don't update in emergency mode
        if self.emergency_mode:
            return False
        
        return True
    
    def _prepare_update_batch(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare batch of experiences for online update"""
        # Filter out experiences with safety violations
        safe_experiences = [
            exp for exp in experiences 
            if exp.get("safety_violation", False) == False
        ]
        
        # Sample batch if we have too many experiences
        if len(safe_experiences) > self.config.update_batch_size:
            indices = np.random.choice(
                len(safe_experiences), 
                self.config.update_batch_size, 
                replace=False
            )
            safe_experiences = [safe_experiences[i] for i in indices]
        
        return safe_experiences
    
    async def _perform_incremental_update(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform incremental model update"""
        if not batch:
            return {"status": "empty_batch"}
        
        # Convert batch to format expected by agent
        formatted_batch = []
        for exp in batch:
            formatted_exp = {
                "state": exp["state"],
                "action": exp["action"],
                "reward": exp["reward"],
                "next_state": exp.get("next_state"),
                "done": exp.get("done", False)
            }
            formatted_batch.append(formatted_exp)
        
        # Perform agent update
        agent_metrics = self.agent.update_policy(formatted_batch)
        
        # Update Thompson sampling arms based on outcomes
        for exp in batch:
            arm_id = exp.get("arm_id")
            if arm_id and arm_id in self.bandit_arms:
                success = exp["reward"] > 0
                self.bandit_arms[arm_id].update(exp["reward"], success)
        
        return {
            "status": "completed",
            "batch_size": len(batch),
            "agent_metrics": agent_metrics,
            "update_count": self.online_updates_count
        }
    
    def _update_learning_rate(self, new_lr: float):
        """Update learning rate for agent optimizers"""
        for optimizer in self.agent.__dict__.values():
            if hasattr(optimizer, 'param_groups'):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
    
    async def explore_vs_exploit(self, state: Dict[str, Any]) -> Tuple[str, float]:
        """
        Decide between exploration and exploitation and return strategy with confidence.
        
        Args:
            state: Current environment state
            
        Returns:
            Tuple of (strategy: "explore" or "exploit", confidence: float)
        """
        # Calculate exploration probability using Thompson sampling
        arm_samples = {arm_id: arm.sample() for arm_id, arm in self.bandit_arms.items()}
        
        # Get best arm
        best_arm_id = max(arm_samples.keys(), key=lambda x: arm_samples[x])
        best_sample = arm_samples[best_arm_id]
        
        # Calculate confidence based on arm statistics
        best_arm = self.bandit_arms[best_arm_id]
        confidence = best_sample * (best_arm.alpha / (best_arm.alpha + best_arm.beta))
        
        # Safety checks
        if self.emergency_mode:
            return ("exploit", 1.0)
        
        if not await self._is_safe_to_explore(state):
            return ("exploit", 0.8)
        
        # Decide based on arm type and confidence
        if best_arm_id in ["experimental", "aggressive"] and confidence > 0.6:
            return ("explore", confidence)
        else:
            return ("exploit", confidence)
    
    async def _is_safe_to_explore(self, state: Dict[str, Any]) -> bool:
        """Check if it's safe to explore given current conditions"""
        # Budget safety
        budget_info = state.get("budget_constraints", {})
        utilization = budget_info.get("budget_utilization", 0)
        if utilization > (1.0 - self.config.max_budget_risk):
            return False
        
        # Performance safety
        if self.baseline_performance and len(self.episode_history) > 10:
            recent_performance = np.mean([ep["reward"] for ep in list(self.episode_history)[-10:]])
            if recent_performance < self.baseline_performance * self.config.safety_threshold:
                return False
        
        # Consecutive violations check
        if self.consecutive_violations >= self.config.safety_violation_limit:
            return False
        
        return True
    
    async def safe_exploration(self, state: Dict[str, Any], base_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform safe exploration by modifying base action within safety constraints.
        
        Args:
            state: Current environment state
            base_action: Base action to modify safely
            
        Returns:
            Safely modified action
        """
        safe_action = copy.deepcopy(base_action)
        
        # Apply conservative modifications
        budget_multiplier = np.random.uniform(0.9, 1.1)  # ±10% budget variation
        safe_action["budget"] = base_action["budget"] * budget_multiplier
        
        # Small bid adjustments
        bid_multiplier = np.random.uniform(0.95, 1.05)  # ±5% bid variation
        safe_action["bid_amount"] = base_action["bid_amount"] * bid_multiplier
        
        # Audience size exploration
        if np.random.random() < 0.3:  # 30% chance to adjust audience
            audience_delta = np.random.uniform(-0.1, 0.1)
            safe_action["audience_size"] = np.clip(
                base_action["audience_size"] + audience_delta, 0.1, 1.0
            )
        
        # Ensure safety constraints
        safe_action = await self._enforce_safety_constraints(safe_action, state)
        
        return safe_action
    
    async def _enforce_safety_constraints(self, action: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce all safety constraints on an action"""
        # Budget constraints
        budget_info = state.get("budget_constraints", {})
        max_budget = budget_info.get("daily_budget", 100) * (1 - self.config.budget_safety_margin)
        action["budget"] = min(action["budget"], max_budget)
        
        # Bid constraints
        historical = state.get("performance_history", {})
        max_bid = historical.get("avg_bid", 1.0) * self.safety_constraints.max_cpa_multiplier
        action["bid_amount"] = min(action["bid_amount"], max_bid)
        
        # Audience constraints
        if action["target_audience"] in self.safety_constraints.blacklisted_audiences:
            # Fall back to default safe audience
            action["target_audience"] = "professionals"  # Conservative default
        
        return action
    
    def update_baseline_performance(self, episodes: List[Dict[str, Any]]):
        """Update baseline performance metrics"""
        if not episodes:
            return
        
        rewards = [ep["reward"] for ep in episodes]
        self.baseline_performance = np.mean(rewards)
        
        # Store in Redis for persistence
        if self.redis_client:
            baseline_data = {
                "performance": self.baseline_performance,
                "episodes_count": len(episodes),
                "updated_at": datetime.now().isoformat()
            }
            self.redis_client.hset(
                f"online_learner:baseline:{self.agent.agent_id}",
                mapping=baseline_data
            )
    
    def record_episode(self, episode_data: Dict[str, Any]):
        """Record episode for performance tracking and bandit updates"""
        self.episode_history.append(episode_data)
        
        # Update bandit arm if this episode used exploration
        arm_id = episode_data.get("arm_id")
        if arm_id and arm_id in self.bandit_arms:
            reward = episode_data["reward"]
            success = episode_data.get("success", reward > 0)
            self.bandit_arms[arm_id].update(reward, success)
        
        # Safety monitoring
        if episode_data.get("safety_violation", False):
            self.consecutive_violations += 1
            self.safety_violations += 1
        else:
            self.consecutive_violations = 0
        
        # Check for emergency mode trigger
        if self.consecutive_violations >= self.config.safety_violation_limit:
            self._trigger_emergency_mode()
        elif self.emergency_mode and self.consecutive_violations == 0:
            self._exit_emergency_mode()
    
    def _trigger_emergency_mode(self):
        """Trigger emergency mode - stop exploration, use conservative actions"""
        if not self.emergency_mode:
            self.emergency_mode = True
            self.logger.critical("Emergency mode activated - switching to conservative baseline")
            
            # Create fallback agent with conservative settings if not exists
            if not self.fallback_agent:
                self._create_fallback_agent()
    
    def _exit_emergency_mode(self):
        """Exit emergency mode - resume normal exploration"""
        if self.emergency_mode:
            self.emergency_mode = False
            self.logger.info("Emergency mode deactivated - resuming normal operation")
    
    def _create_fallback_agent(self):
        """Create a conservative fallback agent for emergency situations"""
        # This would create a copy of the agent with very conservative settings
        # For now, we'll use the existing agent in exploitation mode
        self.fallback_agent = self.agent
    
    async def _log_action_selection(self, exploration_action: ExplorationAction, state: Dict[str, Any]):
        """Log action selection details for monitoring and debugging"""
        if self.redis_client and self.online_updates_count % self.config.log_interval == 0:
            log_data = {
                "agent_id": self.agent.agent_id,
                "arm_id": exploration_action.arm_id,
                "confidence": exploration_action.confidence,
                "risk_level": exploration_action.risk_level,
                "safety_score": exploration_action.safety_score,
                "emergency_mode": self.emergency_mode,
                "timestamp": datetime.now().isoformat()
            }
            
            key = f"online_learner:actions:{self.agent.agent_id}"
            self.redis_client.lpush(key, json.dumps(log_data))
            self.redis_client.ltrim(key, 0, 1000)  # Keep last 1000 entries
    
    async def _log_online_update(self, metrics: Dict[str, Any]):
        """Log online update metrics"""
        if self.redis_client:
            log_data = {
                "agent_id": self.agent.agent_id,
                "update_count": self.online_updates_count,
                "metrics": metrics,
                "bandit_stats": {
                    arm_id: {
                        "mean_reward": arm.get_mean_reward(),
                        "total_pulls": arm.total_pulls,
                        "alpha": arm.alpha,
                        "beta": arm.beta
                    }
                    for arm_id, arm in self.bandit_arms.items()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            key = f"online_learner:updates:{self.agent.agent_id}"
            self.redis_client.lpush(key, json.dumps(log_data))
            self.redis_client.ltrim(key, 0, 500)  # Keep last 500 updates
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for monitoring"""
        metrics = {
            "online_updates_count": self.online_updates_count,
            "emergency_mode": self.emergency_mode,
            "safety_violations": self.safety_violations,
            "consecutive_violations": self.consecutive_violations,
            "baseline_performance": self.baseline_performance,
            "episodes_recorded": len(self.episode_history)
        }
        
        # Bandit arm statistics
        arm_stats = {}
        for arm_id, arm in self.bandit_arms.items():
            arm_stats[arm_id] = {
                "mean_reward": arm.get_mean_reward(),
                "total_pulls": arm.total_pulls,
                "confidence_interval": arm.get_confidence_interval(),
                "last_updated": arm.last_updated.isoformat()
            }
        metrics["bandit_arms"] = arm_stats
        
        # Recent performance
        if self.episode_history:
            recent_rewards = [ep["reward"] for ep in list(self.episode_history)[-50:]]
            metrics["recent_performance"] = {
                "mean_reward": np.mean(recent_rewards),
                "std_reward": np.std(recent_rewards),
                "min_reward": np.min(recent_rewards),
                "max_reward": np.max(recent_rewards)
            }
        
        return metrics
    
    async def shutdown(self):
        """Gracefully shutdown the online learner"""
        self.logger.info("Shutting down online learner...")
        
        # Wait for any ongoing updates
        with self.update_lock:
            # Save final state
            if self.redis_client:
                final_state = {
                    "bandit_arms": {
                        arm_id: {
                            "alpha": arm.alpha,
                            "beta": arm.beta,
                            "total_pulls": arm.total_pulls,
                            "total_rewards": arm.total_rewards
                        }
                        for arm_id, arm in self.bandit_arms.items()
                    },
                    "performance_metrics": self.get_performance_metrics(),
                    "shutdown_time": datetime.now().isoformat()
                }
                
                self.redis_client.hset(
                    f"online_learner:final_state:{self.agent.agent_id}",
                    mapping={"data": json.dumps(final_state)}
                )
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Online learner shutdown completed")


# Utility functions for integration

def create_online_learner(agent: BaseRLAgent, config_dict: Dict[str, Any] = None) -> OnlineLearner:
    """Factory function to create online learner with default or custom configuration"""
    config = OnlineLearnerConfig()
    
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return OnlineLearner(agent, config)


async def demo_online_learning():
    """Demo function showing online learning in action"""
    from .rl_agents.ppo_agent import PPOAgent, PPOConfig
    
    # Create agent
    ppo_config = PPOConfig(state_dim=50, action_dim=20)
    agent = PPOAgent(ppo_config, "demo_agent")
    
    # Create online learner
    online_config = OnlineLearnerConfig(
        online_update_frequency=10,
        safety_threshold=0.7,
        max_budget_risk=0.15
    )
    learner = OnlineLearner(agent, online_config)
    
    # Simulate online learning episodes
    for episode in range(100):
        # Mock state
        state = {
            "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.3},
            "performance_history": {"avg_roas": 1.2, "avg_ctr": 0.02},
            "market_context": {"competition_level": 0.6}
        }
        
        # Select action
        action = await learner.select_action(state)
        
        # Mock episode outcome
        reward = np.random.normal(0.5, 0.2)  # Mock reward
        episode_data = {
            "state": state,
            "action": action,
            "reward": reward,
            "success": reward > 0,
            "safety_violation": reward < -0.5
        }
        
        # Record episode
        learner.record_episode(episode_data)
        
        # Trigger online update periodically
        if episode % 20 == 0:
            await learner.online_update([episode_data])
    
    # Get final metrics
    metrics = learner.get_performance_metrics()
    print("Online Learning Demo Results:")
    print(json.dumps(metrics, indent=2))
    
    await learner.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_online_learning())