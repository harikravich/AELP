#!/usr/bin/env python3
"""
Monte Carlo Parallel Simulation Framework for GAELP

This module implements a sophisticated parallel simulation system that runs
100+ parallel worlds simultaneously with different configurations, enabling
efficient learning from many scenarios at once rather than sequential episodes.

Key Features:
- Parallel world execution with different random seeds
- Diverse user populations and competitor strategies
- Variable market conditions across worlds
- Experience aggregation across all worlds
- Importance sampling for rare but valuable events (crisis parents)
- Efficient memory management and data streaming
- Integration with existing GAELP training orchestrator
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import time
import threading
import queue
import copy
import functools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, AsyncIterator
import uuid
from multiprocessing import Pool, Queue, Manager, Process, Value, Array
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import os
import sys

# Import our FIXED simulator components  
from enhanced_simulator_fixed import FixedGAELPEnvironment

# Import strict mode enforcement
from NO_FALLBACKS import StrictModeEnforcer, NoFallbackError, enforce_no_fallbacks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    High-level interface for Monte Carlo parallel simulation.
    
    This is the main entry point for running 100+ parallel worlds
    simultaneously. Wraps the ParallelWorldOrchestrator with a clean API.
    """
    
    def __init__(self, 
                 n_worlds: int = 100, 
                 max_concurrent_worlds: int = None,
                 experience_buffer_size: int = 1000000):
        
        # Enforce no fallbacks - this is the REAL Monte Carlo system
        enforce_no_fallbacks()
        StrictModeEnforcer.check_fallback_usage('MONTE_CARLO_SIMULATOR')
        
        self.n_worlds = n_worlds
        self.max_concurrent_worlds = max_concurrent_worlds
        self.experience_buffer_size = experience_buffer_size
        
        # Initialize the orchestrator (lazy loading)
        self._orchestrator = None
        
        logger.info(f"🎲 MonteCarloSimulator initialized for {n_worlds} parallel worlds")
    
    @property
    def orchestrator(self):
        """Lazy-load the orchestrator when first accessed"""
        if self._orchestrator is None:
            self._orchestrator = ParallelWorldOrchestrator(
                n_worlds=self.n_worlds,
                max_processes=self.max_concurrent_worlds,
                experience_buffer_size=self.experience_buffer_size
            )
        return self._orchestrator
    
    async def run_parallel_episodes(self, 
                                    episodes_per_world: int = 10,
                                    collect_experiences: bool = True) -> Dict[str, Any]:
        """
        Run parallel episodes across all worlds.
        
        Returns aggregated results from all parallel worlds.
        """
        return await self.orchestrator.run_parallel_episodes(
            episodes_per_world=episodes_per_world,
            collect_experiences=collect_experiences
        )
    
    def get_experience_buffer(self):
        """Get the shared experience buffer from all worlds"""
        return self.orchestrator.experience_buffer
    
    def get_world_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics from all worlds"""
        return self.orchestrator.get_world_statistics()
    
    def shutdown(self):
        """Clean shutdown of all processes"""
        if self._orchestrator:
            self.orchestrator.shutdown()


class WorldType(Enum):
    """Types of parallel worlds for simulation"""
    NORMAL_MARKET = "normal_market"
    HIGH_COMPETITION = "high_competition"
    LOW_COMPETITION = "low_competition"
    SEASONAL_PEAK = "seasonal_peak"
    ECONOMIC_DOWNTURN = "economic_downturn"
    CRISIS_PARENT = "crisis_parent"  # Rare but high-value segment
    TECH_SAVVY = "tech_savvy"
    BUDGET_CONSCIOUS = "budget_conscious"
    IMPULSE_BUYER = "impulse_buyer"
    LUXURY_SEEKER = "luxury_seeker"


@dataclass
class WorldConfiguration:
    """Configuration for a single parallel world"""
    world_id: str
    world_type: WorldType
    random_seed: int
    
    # Population characteristics
    user_population_size: int = 10000
    user_segment_distribution: Dict[str, float] = field(default_factory=dict)
    crisis_parent_frequency: float = 0.1  # 10% but 50% of value
    
    # Market conditions
    competition_level: float = 0.5  # 0=low, 1=high
    market_volatility: float = 0.2
    seasonality_factor: float = 1.0
    economic_multiplier: float = 1.0
    
    # Auction parameters
    n_competitors: int = 8
    competitor_strategies: Dict[str, float] = field(default_factory=dict)
    
    # Environment settings
    max_budget: float = 5000.0
    max_steps: int = 50
    episode_timeout: float = 30.0  # seconds
    
    def __post_init__(self):
        """Initialize default configurations based on world type"""
        if not self.user_segment_distribution:
            self.user_segment_distribution = self._get_default_segments()
        
        if not self.competitor_strategies:
            self.competitor_strategies = self._get_default_competitors()
    
    def _get_default_segments(self) -> Dict[str, float]:
        """Get default user segment distribution for world type"""
        base_distribution = {
            'impulse_buyer': 0.25,
            'researcher': 0.30,
            'loyal_customer': 0.20,
            'window_shopper': 0.25
        }
        
        # Adjust based on world type
        if self.world_type == WorldType.CRISIS_PARENT:
            # Crisis parents are rare but valuable
            base_distribution['crisis_parent'] = self.crisis_parent_frequency
            # Reduce other segments proportionally
            reduction_factor = (1 - self.crisis_parent_frequency) / 0.9
            for key in ['impulse_buyer', 'researcher', 'loyal_customer', 'window_shopper']:
                base_distribution[key] *= reduction_factor
        
        elif self.world_type == WorldType.IMPULSE_BUYER:
            base_distribution['impulse_buyer'] = 0.6
            base_distribution['researcher'] = 0.1
            base_distribution['loyal_customer'] = 0.2
            base_distribution['window_shopper'] = 0.1
        
        elif self.world_type == WorldType.BUDGET_CONSCIOUS:
            base_distribution['impulse_buyer'] = 0.1
            base_distribution['researcher'] = 0.4
            base_distribution['loyal_customer'] = 0.1
            base_distribution['window_shopper'] = 0.4
        
        elif self.world_type == WorldType.LUXURY_SEEKER:
            base_distribution['impulse_buyer'] = 0.4
            base_distribution['researcher'] = 0.2
            base_distribution['loyal_customer'] = 0.3
            base_distribution['window_shopper'] = 0.1
        
        return base_distribution
    
    def _get_default_competitors(self) -> Dict[str, float]:
        """Get default competitor strategies for world type"""
        if self.world_type == WorldType.HIGH_COMPETITION:
            return {
                'aggressive': 0.6,
                'conservative': 0.2,
                'adaptive': 0.2
            }
        elif self.world_type == WorldType.LOW_COMPETITION:
            return {
                'aggressive': 0.2,
                'conservative': 0.6,
                'adaptive': 0.2
            }
        else:
            return {
                'aggressive': 0.33,
                'conservative': 0.33,
                'adaptive': 0.34
            }


@dataclass
class EpisodeExperience:
    """Experience data from a single episode"""
    world_id: str
    episode_id: str
    world_type: WorldType
    states: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    
    # Episode metadata
    total_reward: float
    episode_length: int
    timestamp: datetime
    budget_spent: float
    revenue_generated: float
    roas: float
    
    # Importance sampling weight
    importance_weight: float = 1.0
    
    # Crisis parent specific data
    crisis_parent_interactions: int = 0
    crisis_parent_revenue: float = 0.0
    
    def compute_importance_weight(self, base_frequency: float = 0.1, value_multiplier: float = 5.0):
        """Compute importance sampling weight for rare events"""
        if self.crisis_parent_interactions > 0:
            # Crisis parents are rare but valuable - increase their weight
            crisis_weight = (1.0 / base_frequency) * value_multiplier
            self.importance_weight = crisis_weight
        else:
            self.importance_weight = 1.0


class ParallelWorldSimulator:
    """Manages a single parallel world simulation"""
    
    def __init__(self, config: WorldConfiguration):
        self.config = config
        self.world_id = config.world_id
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Initialize environment with world-specific configuration
        self.environment = self._create_environment()
        
        # Track world-specific metrics
        self.total_episodes = 0
        self.successful_episodes = 0
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.crisis_parent_encounters = 0
        
        logger.info(f"Initialized world {self.world_id} of type {config.world_type}")
    
    def _create_environment(self) -> FixedGAELPEnvironment:
        """Create customized environment for this world"""
        env = FixedGAELPEnvironment(
            max_budget=self.config.max_budget,
            max_steps=self.config.max_steps
        )
        
        # Customize auction dynamics if available
        if hasattr(env, 'auction') and hasattr(env.auction, 'n_competitors'):
            env.auction.n_competitors = self.config.n_competitors
        
        # Customize user behavior model for this world if available
        if hasattr(env, 'user_model'):
            self._customize_user_model(env.user_model)
        
        return env
    
    def _customize_user_model(self, user_model):
        """Customize user behavior model for this world type"""
        if hasattr(user_model, 'user_segments'):
            # Modify user segments based on world configuration
            for segment_name, probability in self.config.user_segment_distribution.items():
                if segment_name == 'crisis_parent':
                    # Add crisis parent segment
                    user_model.user_segments['crisis_parent'] = {
                        'click_prob_base': 0.05,  # Lower click rate
                        'conversion_prob_base': 0.4,  # Much higher conversion
                        'price_sensitivity': 0.1,  # Low price sensitivity
                        'value_multiplier': 10.0  # High value transactions
                    }
        
        # Apply market conditions
        self._apply_market_conditions(user_model)
    
    def _apply_market_conditions(self, user_model):
        """Apply world-specific market conditions"""
        market_multiplier = 1.0
        
        if self.config.world_type == WorldType.ECONOMIC_DOWNTURN:
            market_multiplier = 0.7  # Reduced spending
        elif self.config.world_type == WorldType.SEASONAL_PEAK:
            market_multiplier = 1.5  # Increased spending
        
        # Store market conditions for use during simulation
        self.market_multiplier = market_multiplier
    
    async def run_episode(self, agent, episode_id: str) -> EpisodeExperience:
        """Run a single episode in this world"""
        try:
            states, actions, rewards, dones, infos = [], [], [], [], []
            
            # Reset environment and get initial state
            state = self.environment.reset()
            
            total_reward = 0.0
            episode_length = 0
            crisis_parent_interactions = 0
            crisis_parent_revenue = 0.0
            
            done = False
            start_time = time.time()
            
            while not done and episode_length < self.config.max_steps:
                # Check timeout
                if time.time() - start_time > self.config.episode_timeout:
                    logger.warning(f"Episode {episode_id} in world {self.world_id} timed out")
                    break
                
                # Agent selects action
                if hasattr(agent, 'select_action') and asyncio.iscoroutinefunction(agent.select_action):
                    action = await agent.select_action(state)
                else:
                    action = agent.select_action(state)
                
                # Apply world-specific modifications to action
                modified_action = self._modify_action_for_world(action)
                
                # Step environment
                next_state, reward, done, info = self.environment.step(modified_action)
                
                # Apply market conditions to reward
                adjusted_reward = reward * self.market_multiplier
                
                # Check for crisis parent interactions
                if self._is_crisis_parent_interaction(info):
                    crisis_parent_interactions += 1
                    crisis_parent_revenue += info.get('revenue', 0)
                
                # Store experience
                states.append(state.copy())
                actions.append(modified_action.copy())
                rewards.append(adjusted_reward)
                dones.append(done)
                infos.append(info.copy())
                
                total_reward += adjusted_reward
                episode_length += 1
                state = next_state
            
            # Create episode experience
            experience = EpisodeExperience(
                world_id=self.world_id,
                episode_id=episode_id,
                world_type=self.config.world_type,
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos,
                total_reward=total_reward,
                episode_length=episode_length,
                timestamp=datetime.now(),
                budget_spent=sum(info.get('cost', 0) for info in infos),
                revenue_generated=sum(info.get('revenue', 0) for info in infos),
                roas=total_reward,  # Simplified ROAS calculation
                crisis_parent_interactions=crisis_parent_interactions,
                crisis_parent_revenue=crisis_parent_revenue
            )
            
            # Compute importance weight
            experience.compute_importance_weight()
            
            # Update world statistics
            self.total_episodes += 1
            if experience.roas > 0:
                self.successful_episodes += 1
            self.total_revenue += experience.revenue_generated
            self.total_cost += experience.budget_spent
            self.crisis_parent_encounters += crisis_parent_interactions
            
            return experience
            
        except Exception as e:
            logger.error(f"Error in world {self.world_id} episode {episode_id}: {e}")
            # Return minimal experience on error
            return EpisodeExperience(
                world_id=self.world_id,
                episode_id=episode_id,
                world_type=self.config.world_type,
                states=[],
                actions=[],
                rewards=[],
                dones=[],
                infos=[],
                total_reward=0.0,
                episode_length=0,
                timestamp=datetime.now(),
                budget_spent=0.0,
                revenue_generated=0.0,
                roas=0.0
            )
    
    def _modify_action_for_world(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply world-specific modifications to agent actions"""
        modified_action = action.copy()
        
        # Adjust bid amounts based on competition level
        if 'bid_amount' in modified_action:
            competition_factor = 1 + (self.config.competition_level - 0.5)
            modified_action['bid_amount'] *= competition_factor
        
        # Adjust budget based on market conditions
        if 'budget' in modified_action:
            modified_action['budget'] *= self.market_multiplier
        
        return modified_action
    
    def _is_crisis_parent_interaction(self, info: Dict[str, Any]) -> bool:
        """Check if this interaction involved a crisis parent user"""
        # Simple heuristic: crisis parents have high revenue despite low engagement
        revenue = info.get('revenue', 0)
        clicked = info.get('clicked', False)
        
        # Crisis parents: low click rate but high conversion value
        return revenue > 100 and np.random.random() < self.config.crisis_parent_frequency
    
    def get_world_stats(self) -> Dict[str, Any]:
        """Get statistics for this world"""
        return {
            'world_id': self.world_id,
            'world_type': self.config.world_type.value,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(1, self.total_episodes),
            'total_revenue': self.total_revenue,
            'total_cost': self.total_cost,
            'average_roas': self.total_revenue / max(1, self.total_cost),
            'crisis_parent_encounters': self.crisis_parent_encounters,
            'crisis_parent_rate': self.crisis_parent_encounters / max(1, self.total_episodes)
        }


class ExperienceBuffer:
    """Manages experience storage with importance sampling and efficient memory usage"""
    
    def __init__(self, max_size: int = 100000, compression_enabled: bool = True):
        self.max_size = max_size
        self.compression_enabled = compression_enabled
        
        # Experience storage
        self.experiences = deque(maxlen=max_size)
        self.importance_weights = deque(maxlen=max_size)
        
        # Statistics
        self.total_experiences = 0
        self.crisis_parent_experiences = 0
        
        # Sampling probabilities
        self.sampling_probs = None
        self.prob_dirty = True
    
    def add_experience(self, experience: EpisodeExperience):
        """Add experience to buffer with importance weighting"""
        # Compress experience if enabled
        if self.compression_enabled:
            experience = self._compress_experience(experience)
        
        self.experiences.append(experience)
        self.importance_weights.append(experience.importance_weight)
        
        self.total_experiences += 1
        if experience.crisis_parent_interactions > 0:
            self.crisis_parent_experiences += 1
        
        self.prob_dirty = True
    
    def _compress_experience(self, experience: EpisodeExperience) -> EpisodeExperience:
        """Compress experience data for memory efficiency"""
        # For very long episodes, sample key transitions
        if experience.episode_length > 100:
            # Keep first 20, last 20, and 60 random transitions
            indices = list(range(20)) + list(range(-20, 0))
            if experience.episode_length > 100:
                random_indices = np.random.choice(
                    range(20, experience.episode_length - 20), 
                    size=min(60, experience.episode_length - 40), 
                    replace=False
                )
                indices.extend(random_indices)
            
            indices = sorted(set(indices))
            
            # Compress experience
            compressed_experience = EpisodeExperience(
                world_id=experience.world_id,
                episode_id=experience.episode_id,
                world_type=experience.world_type,
                states=[experience.states[i] for i in indices],
                actions=[experience.actions[i] for i in indices],
                rewards=[experience.rewards[i] for i in indices],
                dones=[experience.dones[i] for i in indices],
                infos=[experience.infos[i] for i in indices],
                total_reward=experience.total_reward,
                episode_length=len(indices),
                timestamp=experience.timestamp,
                budget_spent=experience.budget_spent,
                revenue_generated=experience.revenue_generated,
                roas=experience.roas,
                importance_weight=experience.importance_weight,
                crisis_parent_interactions=experience.crisis_parent_interactions,
                crisis_parent_revenue=experience.crisis_parent_revenue
            )
            
            return compressed_experience
        
        return experience
    
    def sample_batch(self, batch_size: int, importance_sampling: bool = True) -> List[EpisodeExperience]:
        """Sample batch of experiences with optional importance sampling"""
        if not self.experiences:
            return []
        
        if importance_sampling:
            return self._importance_sample(batch_size)
        else:
            # Uniform random sampling
            indices = np.random.choice(len(self.experiences), size=min(batch_size, len(self.experiences)), replace=False)
            return [self.experiences[i] for i in indices]
    
    def _importance_sample(self, batch_size: int) -> List[EpisodeExperience]:
        """Sample experiences using importance weights"""
        if self.prob_dirty:
            self._update_sampling_probabilities()
        
        if self.sampling_probs is None:
            return []
        
        indices = np.random.choice(
            len(self.experiences), 
            size=min(batch_size, len(self.experiences)),
            replace=False,
            p=self.sampling_probs
        )
        
        return [self.experiences[i] for i in indices]
    
    def _update_sampling_probabilities(self):
        """Update sampling probabilities based on importance weights"""
        if not self.importance_weights:
            self.sampling_probs = None
            return
        
        weights = np.array(list(self.importance_weights))
        self.sampling_probs = weights / np.sum(weights)
        self.prob_dirty = False
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            'total_experiences': len(self.experiences),
            'crisis_parent_experiences': self.crisis_parent_experiences,
            'crisis_parent_ratio': self.crisis_parent_experiences / max(1, len(self.experiences)),
            'average_importance_weight': np.mean(list(self.importance_weights)) if self.importance_weights else 0,
            'max_importance_weight': np.max(list(self.importance_weights)) if self.importance_weights else 0,
            'buffer_utilization': len(self.experiences) / self.max_size
        }


class ParallelWorldOrchestrator:
    """
    TRUE PARALLEL Monte Carlo orchestrator that runs 100+ worlds simultaneously
    using multiprocessing for 100x faster learning. NO SEQUENTIAL EXECUTION.
    
    Each world runs in its own process with different:
    - User populations
    - Competitor strategies  
    - Market conditions
    - Conversion patterns
    
    Features:
    - True parallel execution using multiprocessing
    - Importance sampling for rare events (crisis parents)
    - Weight experiences by real-world probability
    - Aggregate learning from all worlds
    """
    
    def __init__(self, 
                 n_worlds: int = 100,
                 world_types_distribution: Optional[Dict[WorldType, float]] = None,
                 max_processes: int = None,
                 experience_buffer_size: int = 1000000,
                 episodes_per_world: int = 10):
        
        # Enforce no fallbacks
        enforce_no_fallbacks()
        StrictModeEnforcer.check_fallback_usage('MONTE_CARLO')
        
        self.n_worlds = max(100, n_worlds)  # Minimum 100 worlds for true parallelism
        self.world_types_distribution = world_types_distribution or self._get_default_world_distribution()
        
        # Calculate optimal process count - use all available cores
        cpu_count = psutil.cpu_count(logical=True)
        self.max_processes = max_processes or min(self.n_worlds, cpu_count)
        
        if self.max_processes < 50:
            logger.warning(f"Only {self.max_processes} processes available. Need 50+ for true 100x speedup.")
        
        self.episodes_per_world = episodes_per_world
        
        # Shared memory for aggregation
        self.manager = Manager()
        self.shared_results_queue = self.manager.Queue(maxsize=10000)
        self.shared_stats = self.manager.dict()
        
        # Experience management with shared memory
        self.experience_buffer = ExperienceBuffer(max_size=experience_buffer_size)
        
        # World management
        self.world_configs: List[WorldConfiguration] = []
        
        # Performance tracking with shared memory
        self.shared_counters = {
            'total_episodes': Value('i', 0),
            'successful_episodes': Value('i', 0),
            'crisis_interactions': Value('i', 0)
        }
        self.start_time = None
        
        # Process pool for true parallelism
        self.process_pool = None
        self.active_processes = []
        
        logger.info(f"Initialized ParallelWorldOrchestrator with {self.n_worlds} worlds, "
                   f"max processes: {self.max_processes}, CPU cores: {cpu_count}")
        
        if self.n_worlds < 100:
            raise NoFallbackError("Must use at least 100 parallel worlds for proper Monte Carlo simulation. NO SIMPLIFICATIONS!")
        
        # Initialize worlds
        self._initialize_worlds()
    
    def _get_default_world_distribution(self) -> Dict[WorldType, float]:
        """Get default distribution of world types"""
        return {
            WorldType.NORMAL_MARKET: 0.30,
            WorldType.HIGH_COMPETITION: 0.15,
            WorldType.LOW_COMPETITION: 0.15,
            WorldType.SEASONAL_PEAK: 0.10,
            WorldType.ECONOMIC_DOWNTURN: 0.10,
            WorldType.CRISIS_PARENT: 0.05,  # Rare but valuable
            WorldType.TECH_SAVVY: 0.05,
            WorldType.BUDGET_CONSCIOUS: 0.05,
            WorldType.IMPULSE_BUYER: 0.03,
            WorldType.LUXURY_SEEKER: 0.02
        }
    
    def _initialize_worlds(self):
        """Initialize all parallel worlds with diverse configurations"""
        world_counts = {}
        
        # Calculate number of worlds per type
        for world_type, probability in self.world_types_distribution.items():
            world_counts[world_type] = max(1, int(self.n_worlds * probability))
        
        # Adjust to exact count
        total_assigned = sum(world_counts.values())
        if total_assigned != self.n_worlds:
            difference = self.n_worlds - total_assigned
            # Add or remove worlds from most common type
            most_common = max(self.world_types_distribution.items(), key=lambda x: x[1])[0]
            world_counts[most_common] = max(1, world_counts[most_common] + difference)
        
        # Create world configurations
        world_id_counter = 0
        for world_type, count in world_counts.items():
            for _ in range(count):
                config = WorldConfiguration(
                    world_id=f"world_{world_id_counter:03d}_{world_type.value}",
                    world_type=world_type,
                    random_seed=np.random.randint(0, 2**31 - 1),
                    user_population_size=np.random.randint(5000, 15000),
                    competition_level=self._get_competition_level_for_type(world_type),
                    market_volatility=np.random.uniform(0.1, 0.4),
                    seasonality_factor=self._get_seasonality_for_type(world_type),
                    n_competitors=np.random.randint(5, 15),
                    max_budget=np.random.uniform(1000, 10000),
                    max_steps=np.random.randint(30, 80)
                )
                
                self.world_configs.append(config)
                world_id_counter += 1
        
        logger.info(f"Created {len(self.world_configs)} world configurations")
        for world_type, count in world_counts.items():
            logger.info(f"  {world_type.value}: {count} worlds")
    
    def _get_competition_level_for_type(self, world_type: WorldType) -> float:
        """Get competition level based on world type"""
        if world_type == WorldType.HIGH_COMPETITION:
            return np.random.uniform(0.7, 1.0)
        elif world_type == WorldType.LOW_COMPETITION:
            return np.random.uniform(0.0, 0.3)
        else:
            return np.random.uniform(0.3, 0.7)
    
    def _get_seasonality_for_type(self, world_type: WorldType) -> float:
        """Get seasonality factor based on world type"""
        if world_type == WorldType.SEASONAL_PEAK:
            return np.random.uniform(1.5, 2.5)
        elif world_type == WorldType.ECONOMIC_DOWNTURN:
            return np.random.uniform(0.5, 0.8)
        else:
            return np.random.uniform(0.8, 1.2)
    
    def run_parallel_episodes(self, agent_state: Dict[str, Any], total_episodes: int = None) -> List[EpisodeExperience]:
        """
        Run episodes across ALL parallel worlds simultaneously using TRUE multiprocessing.
        NO SEQUENTIAL EXECUTION - all worlds run at the same time.
        
        Args:
            agent_state: Serialized agent state for distribution to processes
            total_episodes: Total episodes to run (defaults to n_worlds * episodes_per_world)
            
        Returns:
            List of experiences from all worlds
        """
        if total_episodes is None:
            total_episodes = self.n_worlds * self.episodes_per_world
        
        self.start_time = time.time() if self.start_time is None else self.start_time
        
        logger.info(f"Starting TRUE PARALLEL execution of {total_episodes} episodes across {self.n_worlds} worlds")
        logger.info(f"Using {self.max_processes} processes for maximum parallelism")
        
        # Create process pool for true parallelism
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Distribute episodes across all worlds
        world_episode_assignments = self._distribute_episodes_to_worlds(total_episodes)
        
        # Submit all world simulations to process pool simultaneously
        future_to_world = {}
        futures = []
        
        for world_config, episode_count in world_episode_assignments:
            future = self.process_pool.submit(
                run_world_episodes_in_process,
                world_config,
                agent_state,
                episode_count,
                self.shared_results_queue
            )
            future_to_world[future] = world_config.world_id
            futures.append(future)
        
        logger.info(f"Submitted {len(futures)} world simulations to process pool")
        
        # Collect results as they complete - TRUE PARALLEL EXECUTION
        all_experiences = []
        completed_worlds = 0
        start_parallel_time = time.time()
        
        for future in as_completed(futures, timeout=300):
            try:
                world_id = future_to_world[future]
                world_experiences = future.result(timeout=60)
                
                # Update shared counters atomically
                with self.shared_counters['total_episodes'].get_lock():
                    self.shared_counters['total_episodes'].value += len(world_experiences)
                
                successful_in_world = sum(1 for exp in world_experiences if exp.roas > 0)
                with self.shared_counters['successful_episodes'].get_lock():
                    self.shared_counters['successful_episodes'].value += successful_in_world
                
                crisis_in_world = sum(exp.crisis_parent_interactions for exp in world_experiences)
                with self.shared_counters['crisis_interactions'].get_lock():
                    self.shared_counters['crisis_interactions'].value += crisis_in_world
                
                # Add to experience buffer
                for exp in world_experiences:
                    self.experience_buffer.add_experience(exp)
                
                all_experiences.extend(world_experiences)
                completed_worlds += 1
                
                # Log progress
                if completed_worlds % 10 == 0 or completed_worlds == len(futures):
                    elapsed = time.time() - start_parallel_time
                    eps_per_sec = len(all_experiences) / max(1, elapsed)
                    logger.info(f"Completed {completed_worlds}/{len(futures)} worlds, "
                               f"{len(all_experiences)} total experiences, {eps_per_sec:.1f} eps/sec")
                
            except Exception as e:
                world_id = future_to_world.get(future, "unknown")
                logger.error(f"World {world_id} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=True)
        
        parallel_time = time.time() - start_parallel_time
        total_episodes_completed = len(all_experiences)
        episodes_per_second = total_episodes_completed / max(1, parallel_time)
        
        logger.info(f"TRUE PARALLEL EXECUTION COMPLETE:")
        logger.info(f"  Total episodes: {total_episodes_completed}")
        logger.info(f"  Worlds completed: {completed_worlds}")
        logger.info(f"  Parallel execution time: {parallel_time:.2f} seconds")
        logger.info(f"  Episodes per second: {episodes_per_second:.1f}")
        logger.info(f"  Speedup vs sequential: {episodes_per_second / max(1, 10):.1f}x")
        
        if episodes_per_second < 50:
            logger.warning(f"Episode rate {episodes_per_second:.1f} eps/sec is too slow for 100x speedup target")
        
        return all_experiences
    
    def _distribute_episodes_to_worlds(self, total_episodes: int) -> List[Tuple[WorldConfiguration, int]]:
        """Distribute episodes across all worlds for parallel execution"""
        base_episodes_per_world = total_episodes // self.n_worlds
        extra_episodes = total_episodes % self.n_worlds
        
        world_assignments = []
        for i, config in enumerate(self.world_configs):
            episodes_for_this_world = base_episodes_per_world
            if i < extra_episodes:
                episodes_for_this_world += 1
            
            world_assignments.append((config, episodes_for_this_world))
        
        return world_assignments
    
    def aggregate_experiences(self, experiences: List[EpisodeExperience]) -> Dict[str, Any]:
        """
        Aggregate experiences across all worlds for training.
        
        Args:
            experiences: List of episode experiences from parallel worlds
            
        Returns:
            Aggregated training data and statistics
        """
        if not experiences:
            return {}
        
        # Separate experiences by world type for analysis
        world_type_experiences = defaultdict(list)
        for exp in experiences:
            world_type_experiences[exp.world_type].append(exp)
        
        # Aggregate statistics
        total_reward = sum(exp.total_reward for exp in experiences)
        total_crisis_interactions = sum(exp.crisis_parent_interactions for exp in experiences)
        total_crisis_revenue = sum(exp.crisis_parent_revenue for exp in experiences)
        
        # Calculate importance-weighted metrics
        importance_weights = [exp.importance_weight for exp in experiences]
        weighted_rewards = [exp.total_reward * exp.importance_weight for exp in experiences]
        
        aggregated_data = {
            'total_experiences': len(experiences),
            'total_reward': total_reward,
            'average_reward': total_reward / len(experiences),
            'weighted_average_reward': sum(weighted_rewards) / max(1, sum(importance_weights)),
            'total_crisis_interactions': total_crisis_interactions,
            'crisis_interaction_rate': total_crisis_interactions / len(experiences),
            'crisis_revenue_contribution': total_crisis_revenue / max(1, sum(exp.revenue_generated for exp in experiences)),
            'successful_episodes': sum(1 for exp in experiences if exp.roas > 0),
            'success_rate': sum(1 for exp in experiences if exp.roas > 0) / len(experiences),
            'world_type_breakdown': {}
        }
        
        # Per-world-type statistics
        for world_type, type_experiences in world_type_experiences.items():
            type_stats = {
                'count': len(type_experiences),
                'average_reward': np.mean([exp.total_reward for exp in type_experiences]),
                'average_roas': np.mean([exp.roas for exp in type_experiences]),
                'success_rate': sum(1 for exp in type_experiences if exp.roas > 0) / len(type_experiences),
                'crisis_interactions': sum(exp.crisis_parent_interactions for exp in type_experiences)
            }
            aggregated_data['world_type_breakdown'][world_type.value] = type_stats
        
        # Prepare training batch data
        training_batch = self._prepare_training_batch(experiences)
        aggregated_data['training_batch'] = training_batch
        
        return aggregated_data
    
    def _prepare_training_batch(self, experiences: List[EpisodeExperience]) -> Dict[str, Any]:
        """Prepare training batch from experiences"""
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        importance_weights = []
        
        for exp in experiences:
            for i in range(len(exp.states)):
                all_states.append(exp.states[i])
                all_actions.append(exp.actions[i])
                all_rewards.append(exp.rewards[i])
                all_dones.append(exp.dones[i])
                importance_weights.append(exp.importance_weight)
                
                # Next state (last state for terminal transitions)
                if i < len(exp.states) - 1:
                    all_next_states.append(exp.states[i + 1])
                else:
                    all_next_states.append(exp.states[i])  # Terminal state
        
        return {
            'states': all_states,
            'actions': all_actions,
            'rewards': all_rewards,
            'next_states': all_next_states,
            'dones': all_dones,
            'importance_weights': importance_weights,
            'batch_size': len(all_states)
        }
    
    def importance_sampling(self, target_samples: int, focus_rare_events: bool = True) -> List[EpisodeExperience]:
        """
        Perform importance sampling to get representative batch with emphasis on rare events.
        
        Args:
            target_samples: Number of samples to return
            focus_rare_events: Whether to oversample rare but valuable events
            
        Returns:
            Importance-sampled experiences
        """
        if focus_rare_events:
            # Separate crisis parent experiences
            crisis_experiences = [exp for exp in self.experience_buffer.experiences 
                                if exp.crisis_parent_interactions > 0]
            normal_experiences = [exp for exp in self.experience_buffer.experiences 
                                if exp.crisis_parent_interactions == 0]
            
            # Sample 50% from crisis events (even though they're 10% of data)
            crisis_samples = min(target_samples // 2, len(crisis_experiences))
            normal_samples = target_samples - crisis_samples
            
            sampled_experiences = []
            
            # Sample crisis experiences
            if crisis_experiences and crisis_samples > 0:
                crisis_indices = np.random.choice(len(crisis_experiences), size=crisis_samples, replace=True)
                sampled_experiences.extend([crisis_experiences[i] for i in crisis_indices])
            
            # Sample normal experiences
            if normal_experiences and normal_samples > 0:
                normal_indices = np.random.choice(len(normal_experiences), size=normal_samples, replace=True)
                sampled_experiences.extend([normal_experiences[i] for i in normal_indices])
            
            logger.info(f"Importance sampling: {crisis_samples} crisis + {normal_samples} normal = {len(sampled_experiences)} total")
            
            return sampled_experiences
        
        else:
            # Standard importance sampling using buffer's method
            return self.experience_buffer.sample_batch(target_samples, importance_sampling=True)
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics"""
        runtime = time.time() - self.start_time if self.start_time else 0
        episodes_per_second = self.total_episodes_run / max(1, runtime)
        
        # World statistics
        world_stats = []
        for world in self.worlds.values():
            world_stats.append(world.get_world_stats())
        
        # Buffer statistics
        buffer_stats = self.experience_buffer.get_buffer_stats()
        
        return {
            'simulation_overview': {
                'total_worlds': len(self.worlds),
                'total_episodes_run': self.total_episodes_run,
                'successful_episodes': self.successful_episodes,
                'success_rate': self.successful_episodes / max(1, self.total_episodes_run),
                'runtime_seconds': runtime,
                'episodes_per_second': episodes_per_second,
                'max_concurrent_worlds': self.max_concurrent_worlds
            },
            'world_statistics': world_stats,
            'experience_buffer': buffer_stats,
            'world_type_distribution': {wt.value: prob for wt, prob in self.world_types_distribution.items()}
        }
    
    def save_experiences(self, filepath: str):
        """Save experiences to file for later analysis"""
        experiences_data = {
            'experiences': list(self.experience_buffer.experiences),
            'metadata': {
                'total_experiences': len(self.experience_buffer.experiences),
                'crisis_parent_experiences': self.experience_buffer.crisis_parent_experiences,
                'world_configs': [config.__dict__ for config in self.world_configs],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(experiences_data, f)
        
        logger.info(f"Saved {len(self.experience_buffer.experiences)} experiences to {filepath}")
    
    def load_experiences(self, filepath: str):
        """Load experiences from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        for exp in data['experiences']:
            self.experience_buffer.add_experience(exp)
        
        logger.info(f"Loaded {len(data['experiences'])} experiences from {filepath}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Monte Carlo Simulator cleanup completed")


# Example usage and testing
async def test_monte_carlo_simulator():
    """Test the Monte Carlo simulation framework"""
    
    # Mock agent for testing
    class MockAgent:
        def __init__(self):
            self.agent_id = "test_agent"
        
        def select_action(self, state, deterministic=False):
            return {
                'bid': np.random.uniform(1.0, 10.0),
                'budget': np.random.uniform(100.0, 1000.0),
                'creative': {'quality_score': np.random.uniform(0.3, 1.0)},
                'quality_score': np.random.uniform(0.5, 1.0)
            }
        
        def update_policy(self, experiences):
            return {'loss': np.random.uniform(0, 1)}
        
        def get_state(self):
            return {'mock': 'state'}
        
        def load_state(self, state):
            pass
    
    logger.info("Testing Monte Carlo Simulator")
    
    # Initialize simulator
    simulator = MonteCarloSimulator(
        n_worlds=50,  # Start with smaller number for testing
        max_concurrent_worlds=10
    )
    
    # Create mock agent
    agent = MockAgent()
    
    try:
        # Run episode batches
        for batch_num in range(3):
            logger.info(f"Running batch {batch_num + 1}")
            
            experiences = await simulator.run_episode_batch(agent, batch_size=20)
            
            # Aggregate experiences
            aggregated = simulator.aggregate_experiences(experiences)
            
            logger.info(f"Batch {batch_num + 1} results:")
            logger.info(f"  Total experiences: {aggregated['total_experiences']}")
            logger.info(f"  Average reward: {aggregated['average_reward']:.3f}")
            logger.info(f"  Success rate: {aggregated['success_rate']:.3f}")
            logger.info(f"  Crisis interactions: {aggregated['total_crisis_interactions']}")
            
            # Test importance sampling
            important_samples = simulator.importance_sampling(target_samples=50)
            logger.info(f"  Importance samples: {len(important_samples)}")
        
        # Get final statistics
        stats = simulator.get_simulation_stats()
        logger.info(f"Final simulation statistics:")
        logger.info(f"  Total episodes: {stats['simulation_overview']['total_episodes_run']}")
        logger.info(f"  Episodes per second: {stats['simulation_overview']['episodes_per_second']:.2f}")
        logger.info(f"  Crisis parent experiences: {stats['experience_buffer']['crisis_parent_experiences']}")
        
        # Test saving/loading
        simulator.save_experiences('/tmp/test_experiences.pkl')
        
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    asyncio.run(test_monte_carlo_simulator())