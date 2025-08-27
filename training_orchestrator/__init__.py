"""
GAELP Training Orchestrator

Core module for managing simulation-to-real-world learning progression
for ad campaign agents.
"""

from .core import TrainingOrchestrator
from .phases import TrainingPhase, PhaseManager
from .curriculum import CurriculumScheduler
from .episode_manager import EpisodeManager
from .performance_monitor import PerformanceMonitor
from .safety_monitor import SafetyMonitor
from .journey_timeout import (
    JourneyTimeoutManager,
    TimeoutConfiguration,
    AbandonmentReason,
    AbandonmentPenalty,
    create_timeout_manager
)
from .delayed_reward_system import (
    DelayedRewardSystem, 
    DelayedRewardConfig, 
    AttributionModel, 
    ConversionEvent,
    RewardReplayBuffer,
    integrate_with_episode_manager,
    create_delayed_reward_training_loop
)

__version__ = "1.0.0"
__all__ = [
    "TrainingOrchestrator",
    "TrainingPhase", 
    "PhaseManager",
    "CurriculumScheduler",
    "EpisodeManager",
    "PerformanceMonitor",
    "SafetyMonitor",
    "JourneyTimeoutManager",
    "TimeoutConfiguration",
    "AbandonmentReason",
    "AbandonmentPenalty",
    "create_timeout_manager",
    "DelayedRewardSystem",
    "DelayedRewardConfig",
    "AttributionModel",
    "ConversionEvent",
    "RewardReplayBuffer",
    "integrate_with_episode_manager",
    "create_delayed_reward_training_loop"
]