"""
Episode Management

Handles the execution of individual training episodes across different
environments and phases, with comprehensive logging and state tracking.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np


class EpisodeStatus(Enum):
    """Episode execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SAFETY_VIOLATION = "safety_violation"


@dataclass
class EpisodeConfiguration:
    """Configuration for episode execution"""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_steps: int = 1000
    timeout_seconds: int = 300
    save_observations: bool = True
    save_actions: bool = True
    save_rewards: bool = True
    enable_safety_checks: bool = True
    log_level: str = "INFO"
    enable_delayed_rewards: bool = False  # Enable delayed reward tracking
    delayed_reward_system: Optional[Any] = None  # DelayedRewardSystem instance


@dataclass
class EpisodeMetrics:
    """Metrics collected during episode execution"""
    total_reward: float = 0.0
    total_steps: int = 0
    success_rate: float = 0.0
    budget_spent: float = 0.0
    roi: float = 0.0
    click_through_rate: float = 0.0
    conversion_rate: float = 0.0
    cost_per_acquisition: float = 0.0
    brand_safety_score: float = 1.0
    content_quality_score: float = 1.0
    audience_engagement: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Complete result of an episode execution"""
    episode_id: str
    status: EpisodeStatus
    success: bool
    metrics: EpisodeMetrics
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    observations: List[Any] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    safety_violations: List[str] = field(default_factory=list)
    
    @property
    def total_reward(self) -> float:
        return self.metrics.total_reward
    
    @property
    def budget_spent(self) -> float:
        return self.metrics.budget_spent
    
    @property
    def roi(self) -> float:
        return self.metrics.roi


class EpisodeManager:
    """
    Manages the execution of training episodes across different environments
    with comprehensive logging, safety checks, and performance monitoring.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Episode tracking
        self.active_episodes: Dict[str, EpisodeResult] = {}
        self.completed_episodes: List[EpisodeResult] = []
        
        # Performance tracking
        self.episode_count = 0
        self.total_successful_episodes = 0
        self.total_reward_collected = 0.0
        self.total_budget_spent = 0.0
        
        self.logger.info("Episode Manager initialized")
    
    async def run_episode(self, 
                         agent, 
                         environment, 
                         episode_name: str,
                         episode_config: Optional[EpisodeConfiguration] = None) -> EpisodeResult:
        """
        Run a single training episode
        
        Args:
            agent: The agent to run
            environment: The environment to run in
            episode_name: Name identifier for the episode
            episode_config: Optional episode configuration
            
        Returns:
            EpisodeResult: Complete episode results
        """
        
        if episode_config is None:
            episode_config = EpisodeConfiguration()
        
        episode_config.episode_id = f"{episode_name}_{episode_config.episode_id}"
        
        self.logger.info(f"Starting episode: {episode_config.episode_id}")
        
        # Initialize episode result
        episode_result = EpisodeResult(
            episode_id=episode_config.episode_id,
            status=EpisodeStatus.PENDING,
            success=False,
            metrics=EpisodeMetrics(),
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.0
        )
        
        # Add to active episodes tracking
        self.active_episodes[episode_config.episode_id] = episode_result
        
        try:
            # Set status to running
            episode_result.status = EpisodeStatus.RUNNING
            episode_result.start_time = datetime.now()
            
            # Run the episode with timeout
            await asyncio.wait_for(
                self._execute_episode(agent, environment, episode_config, episode_result),
                timeout=episode_config.timeout_seconds
            )
            
            # Mark as completed
            episode_result.status = EpisodeStatus.COMPLETED
            episode_result.success = True
            
        except asyncio.TimeoutError:
            episode_result.status = EpisodeStatus.TIMEOUT
            episode_result.error_message = f"Episode timed out after {episode_config.timeout_seconds} seconds"
            self.logger.warning(f"Episode {episode_config.episode_id} timed out")
            
        except Exception as e:
            episode_result.status = EpisodeStatus.FAILED
            episode_result.error_message = str(e)
            self.logger.error(f"Episode {episode_config.episode_id} failed: {e}")
            
        finally:
            # Finalize episode
            episode_result.end_time = datetime.now()
            episode_result.duration_seconds = (
                episode_result.end_time - episode_result.start_time
            ).total_seconds()
            
            # Remove from active and add to completed
            self.active_episodes.pop(episode_config.episode_id, None)
            self.completed_episodes.append(episode_result)
            
            # Update global statistics
            self.episode_count += 1
            if episode_result.success:
                self.total_successful_episodes += 1
            self.total_reward_collected += episode_result.total_reward
            self.total_budget_spent += episode_result.budget_spent
            
            self.logger.info(
                f"Episode {episode_config.episode_id} completed. "
                f"Status: {episode_result.status.value}, "
                f"Reward: {episode_result.total_reward:.3f}, "
                f"Duration: {episode_result.duration_seconds:.1f}s"
            )
        
        return episode_result
    
    async def _execute_episode(self, 
                              agent, 
                              environment, 
                              config: EpisodeConfiguration,
                              result: EpisodeResult):
        """Execute the core episode loop"""
        
        # Reset environment
        observation = await self._safe_environment_reset(environment)
        if observation is None:
            raise RuntimeError("Failed to reset environment")
        
        # Initialize episode tracking
        step_count = 0
        total_reward = 0.0
        budget_spent = 0.0
        done = False
        
        # Episode loop
        while not done and step_count < config.max_steps:
            
            # Agent selects action
            action = await self._safe_agent_action(agent, observation)
            if action is None:
                raise RuntimeError("Agent failed to select action")
            
            # Environment step
            step_result = await self._safe_environment_step(environment, action)
            if step_result is None:
                raise RuntimeError("Environment step failed")
            
            next_observation, reward, done, info = step_result
            
            # Update tracking
            step_count += 1
            total_reward += reward
            budget_spent += info.get("budget_spent", 0.0)
            
            # Save data if configured
            if config.save_observations:
                result.observations.append(observation)
            if config.save_actions:
                result.actions.append(action)
            if config.save_rewards:
                result.rewards.append(reward)
            
            # Handle delayed rewards if enabled
            if config.enable_delayed_rewards and config.delayed_reward_system:
                try:
                    touchpoint_id = await config.delayed_reward_system.store_pending_reward(
                        episode_id=config.episode_id,
                        user_id=info.get('user_id', 'unknown'),
                        campaign_id=info.get('campaign_id', 'unknown'),
                        action=action,
                        state=observation,
                        immediate_reward=reward,
                        channel=info.get('channel', 'unknown'),
                        creative_type=info.get('creative_type', 'unknown'),
                        cost=info.get('cost', 0.0),
                        metadata=info
                    )
                    if not hasattr(result, 'touchpoint_ids'):
                        result.touchpoint_ids = []
                    result.touchpoint_ids.append(touchpoint_id)
                except Exception as e:
                    self.logger.warning(f"Failed to store pending reward: {e}")
            
            # Safety checks
            if config.enable_safety_checks:
                safety_violations = await self._check_step_safety(action, info)
                if safety_violations:
                    result.safety_violations.extend(safety_violations)
                    if len(result.safety_violations) > 3:  # Too many violations
                        result.status = EpisodeStatus.SAFETY_VIOLATION
                        break
            
            # Update observation for next step
            observation = next_observation
            
            # Allow other coroutines to run
            await asyncio.sleep(0.001)
        
        # Calculate final metrics
        await self._calculate_episode_metrics(result, total_reward, step_count, budget_spent, info)
        
        # Check for delayed reward updates if enabled
        if config.enable_delayed_rewards and config.delayed_reward_system:
            try:
                delayed_updates = await config.delayed_reward_system.handle_partial_episode(config.episode_id)
                if delayed_updates:
                    result.info['delayed_reward_updates'] = delayed_updates
                    total_delayed_adjustment = sum(update['reward_delta'] for update in delayed_updates)
                    result.info['total_delayed_reward_adjustment'] = total_delayed_adjustment
                    self.logger.info(f"Found {len(delayed_updates)} delayed reward updates for episode {config.episode_id}")
            except Exception as e:
                self.logger.warning(f"Failed to check delayed rewards: {e}")
    
    async def _safe_environment_reset(self, environment):
        """Safely reset environment with error handling"""
        try:
            if hasattr(environment, 'reset') and callable(environment.reset):
                if asyncio.iscoroutinefunction(environment.reset):
                    return await environment.reset()
                else:
                    return environment.reset()
            else:
                self.logger.error("Environment does not have a reset method")
                return None
        except Exception as e:
            self.logger.error(f"Environment reset failed: {e}")
            return None
    
    async def _safe_agent_action(self, agent, observation):
        """Safely get agent action with error handling"""
        try:
            if hasattr(agent, 'select_action') and callable(agent.select_action):
                if asyncio.iscoroutinefunction(agent.select_action):
                    return await agent.select_action(observation)
                else:
                    return agent.select_action(observation)
            else:
                self.logger.error("Agent does not have a select_action method")
                return None
        except Exception as e:
            self.logger.error(f"Agent action selection failed: {e}")
            return None
    
    async def _safe_environment_step(self, environment, action):
        """Safely step environment with error handling"""
        try:
            if hasattr(environment, 'step') and callable(environment.step):
                if asyncio.iscoroutinefunction(environment.step):
                    return await environment.step(action)
                else:
                    return environment.step(action)
            else:
                self.logger.error("Environment does not have a step method")
                return None
        except Exception as e:
            self.logger.error(f"Environment step failed: {e}")
            return None
    
    async def _check_step_safety(self, action, info) -> List[str]:
        """Check for safety violations in the current step"""
        violations = []
        
        # Check budget constraints
        budget_spent = info.get("budget_spent", 0.0)
        if budget_spent > 100.0:  # Example: max $100 per action
            violations.append(f"Budget exceeded: ${budget_spent:.2f}")
        
        # Check content safety
        content_score = info.get("content_safety_score", 1.0)
        if content_score < 0.8:
            violations.append(f"Content safety score too low: {content_score:.2f}")
        
        # Check for inappropriate targeting
        if info.get("inappropriate_targeting", False):
            violations.append("Inappropriate audience targeting detected")
        
        # Check bid amounts
        bid_amount = info.get("bid_amount", 0.0)
        if bid_amount > 50.0:  # Example: max $50 bid
            violations.append(f"Bid amount too high: ${bid_amount:.2f}")
        
        return violations
    
    async def _calculate_episode_metrics(self, 
                                       result: EpisodeResult,
                                       total_reward: float,
                                       step_count: int,
                                       budget_spent: float,
                                       final_info: Dict[str, Any]):
        """Calculate comprehensive episode metrics"""
        
        metrics = result.metrics
        
        # Basic metrics
        metrics.total_reward = total_reward
        metrics.total_steps = step_count
        metrics.budget_spent = budget_spent
        
        # Calculate ROI
        revenue = final_info.get("revenue", 0.0)
        if budget_spent > 0:
            metrics.roi = (revenue - budget_spent) / budget_spent
        else:
            metrics.roi = 0.0
        
        # Campaign performance metrics
        metrics.click_through_rate = final_info.get("click_through_rate", 0.0)
        metrics.conversion_rate = final_info.get("conversion_rate", 0.0)
        metrics.audience_engagement = final_info.get("audience_engagement", 0.0)
        
        # Calculate cost per acquisition
        conversions = final_info.get("conversions", 0)
        if conversions > 0:
            metrics.cost_per_acquisition = budget_spent / conversions
        else:
            metrics.cost_per_acquisition = float('inf')
        
        # Safety and quality scores
        metrics.brand_safety_score = final_info.get("brand_safety_score", 1.0)
        metrics.content_quality_score = final_info.get("content_quality_score", 1.0)
        
        # Success rate (based on achievement of episode goals)
        episode_goals_met = final_info.get("goals_achieved", 0)
        total_goals = final_info.get("total_goals", 1)
        metrics.success_rate = episode_goals_met / total_goals
        
        # Validation metrics for historical validation phase
        if final_info.get("validation_data"):
            metrics.validation_metrics = final_info["validation_data"]
    
    async def run_batch_episodes(self, 
                                agent, 
                                environments: List[Any],
                                episode_configs: List[EpisodeConfiguration],
                                max_concurrent: int = 5) -> List[EpisodeResult]:
        """
        Run multiple episodes concurrently
        
        Args:
            agent: The agent to run
            environments: List of environments
            episode_configs: List of episode configurations
            max_concurrent: Maximum concurrent episodes
            
        Returns:
            List of episode results
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_episode(env, config):
            async with semaphore:
                return await self.run_episode(agent, env, f"batch_{config.episode_id}", config)
        
        # Create tasks for all episodes
        tasks = [
            run_single_episode(env, config) 
            for env, config in zip(environments, episode_configs)
        ]
        
        # Run all episodes
        self.logger.info(f"Starting batch of {len(tasks)} episodes")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch episode {i} failed: {result}")
            else:
                valid_results.append(result)
        
        self.logger.info(f"Batch completed: {len(valid_results)}/{len(tasks)} episodes successful")
        return valid_results
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get comprehensive episode statistics"""
        
        if self.episode_count == 0:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "average_reward": 0.0,
                "total_budget_spent": 0.0,
                "average_duration": 0.0
            }
        
        # Calculate success rate
        success_rate = self.total_successful_episodes / self.episode_count
        
        # Calculate averages
        average_reward = self.total_reward_collected / self.episode_count
        
        # Calculate average duration
        total_duration = sum(ep.duration_seconds for ep in self.completed_episodes)
        average_duration = total_duration / len(self.completed_episodes) if self.completed_episodes else 0.0
        
        # Recent performance (last 50 episodes)
        recent_episodes = self.completed_episodes[-50:]
        recent_success_rate = sum(1 for ep in recent_episodes if ep.success) / len(recent_episodes) if recent_episodes else 0.0
        recent_average_reward = sum(ep.total_reward for ep in recent_episodes) / len(recent_episodes) if recent_episodes else 0.0
        
        return {
            "total_episodes": self.episode_count,
            "successful_episodes": self.total_successful_episodes,
            "success_rate": success_rate,
            "average_reward": average_reward,
            "total_reward_collected": self.total_reward_collected,
            "total_budget_spent": self.total_budget_spent,
            "average_duration": average_duration,
            "recent_success_rate": recent_success_rate,
            "recent_average_reward": recent_average_reward,
            "active_episodes": len(self.active_episodes),
            "completed_episodes": len(self.completed_episodes)
        }
    
    def get_active_episodes(self) -> Dict[str, EpisodeResult]:
        """Get currently active episodes"""
        return self.active_episodes.copy()
    
    def get_completed_episodes(self, limit: Optional[int] = None) -> List[EpisodeResult]:
        """Get completed episodes with optional limit"""
        if limit is None:
            return self.completed_episodes.copy()
        else:
            return self.completed_episodes[-limit:]
    
    def clear_episode_history(self):
        """Clear episode history (for memory management)"""
        self.completed_episodes.clear()
        self.logger.info("Episode history cleared")
    
    async def stop_all_episodes(self):
        """Stop all active episodes"""
        for episode_id in list(self.active_episodes.keys()):
            episode = self.active_episodes[episode_id]
            episode.status = EpisodeStatus.FAILED
            episode.error_message = "Stopped by user request"
            episode.end_time = datetime.now()
            
        self.active_episodes.clear()
        self.logger.info("All active episodes stopped")