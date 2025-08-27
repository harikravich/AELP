"""
Configuration management for Training Orchestrator

Provides default configurations and environment-specific settings
for different deployment scenarios.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration"""
    bigquery_project: str = "gaelp-project"
    bigquery_dataset: str = "training_logs"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None


@dataclass
class PubSubConfig:
    """Pub/Sub configuration"""
    project_id: str = "gaelp-project"
    topic_training_events: str = "training-events"
    topic_safety_alerts: str = "safety-alerts"
    subscription_timeout: float = 60.0


@dataclass
class PhaseConfig:
    """Phase-specific configuration"""
    simulation_episodes: int = 1000
    historical_validation_episodes: int = 100
    real_testing_episodes: int = 50
    scaled_deployment_episodes: int = 500


@dataclass
class BudgetConfig:
    """Budget and cost control configuration"""
    real_testing_daily_limit: float = 50.0
    real_testing_episode_limit: float = 10.0
    scaled_deployment_daily_limit: float = 1000.0
    emergency_stop_threshold: float = 5000.0
    budget_alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "warning": 0.8,
        "critical": 0.95
    })


@dataclass
class SafetyConfig:
    """Safety and compliance configuration"""
    content_safety_threshold: float = 0.9
    brand_safety_threshold: float = 0.85
    max_bid_amount: float = 20.0
    require_human_approval_real: bool = True
    anomaly_detection_sensitivity: float = 2.5
    max_violations_per_day: int = 5


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration"""
    enabled: bool = True
    difficulty_progression_rate: float = 0.1
    performance_window_size: int = 50
    adaptation_threshold: float = 0.1
    min_task_episodes: int = 10
    max_task_episodes: int = 100


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    metrics_collection_interval: int = 60  # seconds
    checkpoint_interval: int = 100  # episodes
    performance_analysis_window: int = 50
    trend_analysis_min_episodes: int = 20


@dataclass
class TrainingOrchestratorConfig:
    """Complete training orchestrator configuration"""
    
    # Basic settings
    experiment_name: str = "ad_campaign_training"
    random_seed: int = 42
    environment: str = "development"  # development, staging, production
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    pubsub: PubSubConfig = field(default_factory=PubSubConfig)
    phases: PhaseConfig = field(default_factory=PhaseConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Advanced settings
    enable_distributed_training: bool = False
    max_concurrent_episodes: int = 5
    enable_transfer_learning: bool = True
    
    @classmethod
    def from_environment(cls, env: str = "development") -> "TrainingOrchestratorConfig":
        """Create configuration based on environment"""
        
        config = cls()
        config.environment = env
        
        if env == "development":
            config = cls._setup_development_config(config)
        elif env == "staging":
            config = cls._setup_staging_config(config)
        elif env == "production":
            config = cls._setup_production_config(config)
        
        # Override with environment variables
        config = cls._apply_env_overrides(config)
        
        return config
    
    @staticmethod
    def _setup_development_config(config: "TrainingOrchestratorConfig") -> "TrainingOrchestratorConfig":
        """Setup development environment configuration"""
        
        # Smaller scale for development
        config.phases.simulation_episodes = 100
        config.phases.historical_validation_episodes = 20
        config.phases.real_testing_episodes = 5
        
        # Lower budget limits
        config.budget.real_testing_daily_limit = 10.0
        config.budget.real_testing_episode_limit = 2.0
        
        # More lenient safety
        config.safety.max_violations_per_day = 10
        config.safety.require_human_approval_real = False
        
        # Local services
        config.database.bigquery_project = "gaelp-dev"
        config.database.redis_host = "localhost"
        
        # Debug logging
        config.monitoring.log_level = "DEBUG"
        config.monitoring.checkpoint_interval = 10
        
        return config
    
    @staticmethod
    def _setup_staging_config(config: "TrainingOrchestratorConfig") -> "TrainingOrchestratorConfig":
        """Setup staging environment configuration"""
        
        # Moderate scale for staging
        config.phases.simulation_episodes = 500
        config.phases.historical_validation_episodes = 50
        config.phases.real_testing_episodes = 20
        
        # Moderate budget limits
        config.budget.real_testing_daily_limit = 25.0
        config.budget.real_testing_episode_limit = 5.0
        
        # Standard safety
        config.safety.require_human_approval_real = True
        
        # Staging services
        config.database.bigquery_project = "gaelp-staging"
        config.pubsub.project_id = "gaelp-staging"
        
        return config
    
    @staticmethod
    def _setup_production_config(config: "TrainingOrchestratorConfig") -> "TrainingOrchestratorConfig":
        """Setup production environment configuration"""
        
        # Full scale for production
        config.phases.simulation_episodes = 2000
        config.phases.historical_validation_episodes = 200
        config.phases.real_testing_episodes = 100
        config.phases.scaled_deployment_episodes = 1000
        
        # Production budget limits
        config.budget.real_testing_daily_limit = 100.0
        config.budget.real_testing_episode_limit = 20.0
        config.budget.scaled_deployment_daily_limit = 5000.0
        
        # Strict safety
        config.safety.require_human_approval_real = True
        config.safety.max_violations_per_day = 2
        config.safety.anomaly_detection_sensitivity = 2.0
        
        # Production services
        config.database.bigquery_project = "gaelp-production"
        config.pubsub.project_id = "gaelp-production"
        
        # Production monitoring
        config.monitoring.log_level = "INFO"
        config.enable_distributed_training = True
        config.max_concurrent_episodes = 10
        
        return config
    
    @staticmethod
    def _apply_env_overrides(config: "TrainingOrchestratorConfig") -> "TrainingOrchestratorConfig":
        """Apply environment variable overrides"""
        
        # Database overrides
        if os.getenv("BIGQUERY_PROJECT"):
            config.database.bigquery_project = os.getenv("BIGQUERY_PROJECT")
        
        if os.getenv("REDIS_HOST"):
            config.database.redis_host = os.getenv("REDIS_HOST")
        
        if os.getenv("REDIS_PORT"):
            config.database.redis_port = int(os.getenv("REDIS_PORT"))
        
        # Budget overrides
        if os.getenv("REAL_TESTING_BUDGET_LIMIT"):
            config.budget.real_testing_daily_limit = float(os.getenv("REAL_TESTING_BUDGET_LIMIT"))
        
        if os.getenv("MAX_DAILY_BUDGET"):
            config.budget.scaled_deployment_daily_limit = float(os.getenv("MAX_DAILY_BUDGET"))
        
        # Safety overrides
        if os.getenv("REQUIRE_HUMAN_APPROVAL"):
            config.safety.require_human_approval_real = os.getenv("REQUIRE_HUMAN_APPROVAL").lower() == "true"
        
        # Pub/Sub overrides
        if os.getenv("PUBSUB_PROJECT_ID"):
            config.pubsub.project_id = os.getenv("PUBSUB_PROJECT_ID")
        
        if os.getenv("PUBSUB_TOPIC_TRAINING_EVENTS"):
            config.pubsub.topic_training_events = os.getenv("PUBSUB_TOPIC_TRAINING_EVENTS")
        
        if os.getenv("PUBSUB_TOPIC_SAFETY_ALERTS"):
            config.pubsub.topic_safety_alerts = os.getenv("PUBSUB_TOPIC_SAFETY_ALERTS")
        
        # BigQuery dataset overrides
        if os.getenv("BIGQUERY_DATASET_CAMPAIGN"):
            config.database.bigquery_dataset = os.getenv("BIGQUERY_DATASET_CAMPAIGN")
        
        # Redis cache configuration
        if os.getenv("REDIS_CACHE_HOST"):
            config.database.redis_host = os.getenv("REDIS_CACHE_HOST")
        
        if os.getenv("REDIS_CACHE_PORT"):
            config.database.redis_port = int(os.getenv("REDIS_CACHE_PORT"))
        
        # Monitoring overrides
        if os.getenv("LOG_LEVEL"):
            config.monitoring.log_level = os.getenv("LOG_LEVEL")
        
        if os.getenv("RANDOM_SEED"):
            config.random_seed = int(os.getenv("RANDOM_SEED"))
        
        return config
    
    def to_legacy_config(self) -> Dict[str, Any]:
        """Convert to legacy configuration format for backward compatibility"""
        
        return {
            "experiment_id": f"{self.experiment_name}_{self.random_seed}",
            "simulation_episodes": self.phases.simulation_episodes,
            "historical_validation_episodes": self.phases.historical_validation_episodes,
            "real_testing_budget_limit": self.budget.real_testing_daily_limit,
            "scaled_deployment_threshold": 0.15,
            "curriculum_enabled": self.curriculum.enabled,
            "difficulty_progression_rate": self.curriculum.difficulty_progression_rate,
            "performance_window": self.curriculum.performance_window_size,
            "max_daily_budget": self.budget.scaled_deployment_daily_limit,
            "safety_check_interval": 10,
            "anomaly_threshold": self.safety.anomaly_detection_sensitivity,
            "random_seed": self.random_seed,
            "log_level": self.monitoring.log_level,
            "checkpoint_interval": self.monitoring.checkpoint_interval,
            "bigquery_project": self.database.bigquery_project,
            "bigquery_dataset": self.database.bigquery_dataset,
            "redis_host": self.database.redis_host,
            "redis_port": self.database.redis_port,
            "pubsub_topic": self.pubsub.topic_training_events
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        
        issues = []
        
        # Budget validation
        if self.budget.real_testing_daily_limit <= 0:
            issues.append("Real testing daily limit must be positive")
        
        if self.budget.real_testing_episode_limit > self.budget.real_testing_daily_limit:
            issues.append("Episode budget limit cannot exceed daily limit")
        
        # Phase validation
        if self.phases.simulation_episodes < 10:
            issues.append("Simulation episodes must be at least 10")
        
        if self.phases.historical_validation_episodes < 5:
            issues.append("Historical validation episodes must be at least 5")
        
        # Safety validation
        if self.safety.content_safety_threshold < 0.5:
            issues.append("Content safety threshold too low (minimum 0.5)")
        
        if self.safety.max_violations_per_day < 1:
            issues.append("Max violations per day must be at least 1")
        
        # Curriculum validation
        if self.curriculum.performance_window_size < 10:
            issues.append("Performance window size must be at least 10")
        
        return issues


def load_env_file(env_file: str) -> None:
    """Load environment variables from a .env file"""
    if not os.path.exists(env_file):
        return
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key.strip()] = value


def load_config(config_file: Optional[str] = None, 
               environment: str = "development") -> TrainingOrchestratorConfig:
    """
    Load configuration from file or environment
    
    Args:
        config_file: Optional path to configuration file
        environment: Environment name (development, staging, production)
        
    Returns:
        TrainingOrchestratorConfig: Loaded configuration
    """
    
    # Try to load environment file
    current_dir = Path(__file__).parent
    env_file = current_dir / f".env.{environment}"
    
    if env_file.exists():
        load_env_file(str(env_file))
    
    if config_file and os.path.exists(config_file):
        # Load from file (implementation would depend on file format)
        # For now, return environment-based config
        pass
    
    # Load environment-based configuration
    config = TrainingOrchestratorConfig.from_environment(environment)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        raise ValueError(f"Configuration validation failed: {issues}")
    
    return config


# Predefined configurations for common scenarios
DEVELOPMENT_CONFIG = TrainingOrchestratorConfig.from_environment("development")
STAGING_CONFIG = TrainingOrchestratorConfig.from_environment("staging")
PRODUCTION_CONFIG = TrainingOrchestratorConfig.from_environment("production")

# Quick access configurations
QUICK_TEST_CONFIG = TrainingOrchestratorConfig(
    experiment_name="quick_test",
    phases=PhaseConfig(
        simulation_episodes=50,
        historical_validation_episodes=10,
        real_testing_episodes=3,
        scaled_deployment_episodes=20
    ),
    budget=BudgetConfig(
        real_testing_daily_limit=5.0,
        real_testing_episode_limit=1.0
    ),
    monitoring=MonitoringConfig(
        log_level="DEBUG",
        checkpoint_interval=5
    )
)

RESEARCH_CONFIG = TrainingOrchestratorConfig(
    experiment_name="research_experiment",
    phases=PhaseConfig(
        simulation_episodes=5000,
        historical_validation_episodes=500,
        real_testing_episodes=0,  # Research only, no real testing
        scaled_deployment_episodes=0
    ),
    budget=BudgetConfig(
        real_testing_daily_limit=0.0,
        real_testing_episode_limit=0.0
    ),
    safety=SafetyConfig(
        require_human_approval_real=True,
        max_violations_per_day=50  # More lenient for research
    ),
    curriculum=CurriculumConfig(
        difficulty_progression_rate=0.05,  # Slower progression for research
        performance_window_size=100
    )
)