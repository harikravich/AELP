"""
Configuration settings for GAELP Safety Framework
Provides centralized configuration management for all safety components.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import timedelta


@dataclass
class BudgetConfig:
    """Budget control configuration"""
    default_daily_limit: float = 1000.0
    default_weekly_limit: float = 5000.0
    default_monthly_limit: float = 20000.0
    max_campaign_limit: float = 50000.0
    enable_auto_pause: bool = True
    spend_monitoring_interval: int = 60  # seconds
    violation_cooldown: int = 300  # seconds before allowing resume


@dataclass
class ContentConfig:
    """Content safety configuration"""
    enable_text_moderation: bool = True
    enable_image_moderation: bool = True
    enable_video_moderation: bool = True
    strict_mode: bool = False  # More aggressive filtering
    custom_blocked_keywords: List[str] = field(default_factory=list)
    platform_policies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    age_restriction_threshold: int = 13


@dataclass
class PerformanceConfig:
    """Performance safety configuration"""
    enable_reward_clipping: bool = True
    min_reward: float = -10.0
    max_reward: float = 10.0
    clip_percentile: float = 95.0
    anomaly_threshold: float = 3.0  # Standard deviations
    max_campaign_runtime: timedelta = timedelta(days=30)
    performance_check_interval: int = 300  # seconds


@dataclass
class BehaviorConfig:
    """Agent behavior safety configuration"""
    max_actions_per_hour: int = 100
    repetition_threshold: int = 5
    rapid_change_threshold: int = 10
    enable_ethical_constraints: bool = True
    require_human_approval: List[str] = field(default_factory=lambda: [
        'create_campaign', 'delete_campaign', 'modify_targeting'
    ])


@dataclass
class OperationalConfig:
    """Operational safety configuration"""
    enable_sandbox: bool = True
    enable_graduated_deployment: bool = True
    sandbox_timeout: timedelta = timedelta(hours=4)
    emergency_contact_emails: List[str] = field(default_factory=list)
    audit_log_retention: timedelta = timedelta(days=2555)  # 7 years


@dataclass
class DataSafetyConfig:
    """Data safety and privacy configuration"""
    enable_pii_detection: bool = True
    enable_consent_management: bool = True
    data_retention_days: int = 365
    enable_gdpr_compliance: bool = True
    enable_ccpa_compliance: bool = True
    credential_rotation_days: int = 90


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    health_check_interval: int = 30  # seconds
    dashboard_update_interval: int = 60  # seconds
    alert_webhooks: List[str] = field(default_factory=list)
    critical_alert_emails: List[str] = field(default_factory=list)
    slack_webhook_url: Optional[str] = None
    enable_metrics_export: bool = True


@dataclass
class GAELPSafetyConfig:
    """Main configuration class for GAELP Safety Framework"""
    
    # Component configurations
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    operational: OperationalConfig = field(default_factory=OperationalConfig)
    data_safety: DataSafetyConfig = field(default_factory=DataSafetyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    environment: str = "development"  # development, staging, production
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    
    # Feature flags
    enable_budget_controls: bool = True
    enable_content_safety: bool = True
    enable_performance_safety: bool = True
    enable_operational_safety: bool = True
    enable_data_safety: bool = True
    enable_behavior_safety: bool = True
    
    @classmethod
    def from_environment(cls) -> 'GAELPSafetyConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Environment
        config.environment = os.getenv('GAELP_ENVIRONMENT', 'development')
        config.log_level = os.getenv('GAELP_LOG_LEVEL', 'INFO')
        config.enable_debug_mode = os.getenv('GAELP_DEBUG', 'false').lower() == 'true'
        
        # Budget configuration
        config.budget.default_daily_limit = float(os.getenv('GAELP_DEFAULT_DAILY_BUDGET', '1000.0'))
        config.budget.default_weekly_limit = float(os.getenv('GAELP_DEFAULT_WEEKLY_BUDGET', '5000.0'))
        config.budget.default_monthly_limit = float(os.getenv('GAELP_DEFAULT_MONTHLY_BUDGET', '20000.0'))
        config.budget.max_campaign_limit = float(os.getenv('GAELP_MAX_CAMPAIGN_BUDGET', '50000.0'))
        config.budget.enable_auto_pause = os.getenv('GAELP_AUTO_PAUSE', 'true').lower() == 'true'
        
        # Content configuration
        config.content.strict_mode = os.getenv('GAELP_CONTENT_STRICT_MODE', 'false').lower() == 'true'
        config.content.age_restriction_threshold = int(os.getenv('GAELP_AGE_THRESHOLD', '13'))
        
        # Performance configuration
        config.performance.min_reward = float(os.getenv('GAELP_MIN_REWARD', '-10.0'))
        config.performance.max_reward = float(os.getenv('GAELP_MAX_REWARD', '10.0'))
        config.performance.anomaly_threshold = float(os.getenv('GAELP_ANOMALY_THRESHOLD', '3.0'))
        
        # Behavior configuration
        config.behavior.max_actions_per_hour = int(os.getenv('GAELP_MAX_ACTIONS_PER_HOUR', '100'))
        config.behavior.repetition_threshold = int(os.getenv('GAELP_REPETITION_THRESHOLD', '5'))
        
        # Monitoring configuration
        config.monitoring.health_check_interval = int(os.getenv('GAELP_HEALTH_CHECK_INTERVAL', '30'))
        
        alert_webhooks = os.getenv('GAELP_ALERT_WEBHOOKS', '')
        if alert_webhooks:
            config.monitoring.alert_webhooks = alert_webhooks.split(',')
        
        critical_emails = os.getenv('GAELP_CRITICAL_EMAILS', '')
        if critical_emails:
            config.monitoring.critical_alert_emails = critical_emails.split(',')
        
        config.monitoring.slack_webhook_url = os.getenv('GAELP_SLACK_WEBHOOK')
        
        # Feature flags
        config.enable_budget_controls = os.getenv('GAELP_ENABLE_BUDGET', 'true').lower() == 'true'
        config.enable_content_safety = os.getenv('GAELP_ENABLE_CONTENT', 'true').lower() == 'true'
        config.enable_performance_safety = os.getenv('GAELP_ENABLE_PERFORMANCE', 'true').lower() == 'true'
        config.enable_operational_safety = os.getenv('GAELP_ENABLE_OPERATIONAL', 'true').lower() == 'true'
        config.enable_data_safety = os.getenv('GAELP_ENABLE_DATA_SAFETY', 'true').lower() == 'true'
        config.enable_behavior_safety = os.getenv('GAELP_ENABLE_BEHAVIOR', 'true').lower() == 'true'
        
        return config
    
    @classmethod
    def development_config(cls) -> 'GAELPSafetyConfig':
        """Create development configuration"""
        config = cls()
        config.environment = "development"
        config.enable_debug_mode = True
        config.log_level = "DEBUG"
        
        # Relaxed limits for development
        config.budget.default_daily_limit = 100.0
        config.budget.default_weekly_limit = 500.0
        config.budget.default_monthly_limit = 2000.0
        config.budget.max_campaign_limit = 1000.0
        
        # Less strict content filtering
        config.content.strict_mode = False
        
        # More frequent monitoring
        config.monitoring.health_check_interval = 15
        config.monitoring.dashboard_update_interval = 30
        
        return config
    
    @classmethod
    def staging_config(cls) -> 'GAELPSafetyConfig':
        """Create staging configuration"""
        config = cls()
        config.environment = "staging"
        config.enable_debug_mode = False
        config.log_level = "INFO"
        
        # Production-like limits but smaller
        config.budget.default_daily_limit = 500.0
        config.budget.default_weekly_limit = 2500.0
        config.budget.default_monthly_limit = 10000.0
        config.budget.max_campaign_limit = 5000.0
        
        # Strict content filtering
        config.content.strict_mode = True
        
        # Production-like monitoring
        config.monitoring.health_check_interval = 30
        config.monitoring.dashboard_update_interval = 60
        
        return config
    
    @classmethod
    def production_config(cls) -> 'GAELPSafetyConfig':
        """Create production configuration"""
        config = cls()
        config.environment = "production"
        config.enable_debug_mode = False
        config.log_level = "WARNING"
        
        # Full production limits
        config.budget.default_daily_limit = 10000.0
        config.budget.default_weekly_limit = 50000.0
        config.budget.default_monthly_limit = 200000.0
        config.budget.max_campaign_limit = 500000.0
        
        # Strict content filtering
        config.content.strict_mode = True
        
        # Comprehensive monitoring
        config.monitoring.health_check_interval = 30
        config.monitoring.dashboard_update_interval = 60
        config.monitoring.enable_metrics_export = True
        
        # Enable all safety features
        config.enable_budget_controls = True
        config.enable_content_safety = True
        config.enable_performance_safety = True
        config.enable_operational_safety = True
        config.enable_data_safety = True
        config.enable_behavior_safety = True
        
        return config
    
    def to_safety_configuration(self):
        """Convert to SafetyConfiguration for orchestrator"""
        from safety_orchestrator import SafetyConfiguration
        
        return SafetyConfiguration(
            enable_budget_controls=self.enable_budget_controls,
            enable_content_safety=self.enable_content_safety,
            enable_performance_safety=self.enable_performance_safety,
            enable_operational_safety=self.enable_operational_safety,
            enable_data_safety=self.enable_data_safety,
            enable_behavior_safety=self.enable_behavior_safety,
            max_daily_budget=self.budget.default_daily_limit,
            content_violation_threshold=3,
            performance_anomaly_threshold=self.performance.anomaly_threshold,
            behavior_violation_threshold=self.behavior.repetition_threshold,
            alert_webhooks=self.monitoring.alert_webhooks,
            human_review_required=len(self.behavior.require_human_approval) > 0,
            auto_pause_on_critical=self.budget.enable_auto_pause,
            emergency_contacts=self.monitoring.critical_alert_emails
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Budget validation
        if self.budget.default_daily_limit <= 0:
            issues.append("Default daily budget must be positive")
        
        if self.budget.default_weekly_limit < self.budget.default_daily_limit * 7:
            issues.append("Weekly budget should be at least 7x daily budget")
        
        if self.budget.default_monthly_limit < self.budget.default_weekly_limit * 4:
            issues.append("Monthly budget should be at least 4x weekly budget")
        
        # Performance validation
        if self.performance.min_reward >= self.performance.max_reward:
            issues.append("Min reward must be less than max reward")
        
        if self.performance.anomaly_threshold <= 0:
            issues.append("Anomaly threshold must be positive")
        
        # Behavior validation
        if self.behavior.max_actions_per_hour <= 0:
            issues.append("Max actions per hour must be positive")
        
        if self.behavior.repetition_threshold <= 0:
            issues.append("Repetition threshold must be positive")
        
        # Monitoring validation
        if self.monitoring.health_check_interval <= 0:
            issues.append("Health check interval must be positive")
        
        # Production-specific validation
        if self.environment == "production":
            if not self.monitoring.critical_alert_emails:
                issues.append("Production requires critical alert emails")
            
            if not self.enable_budget_controls:
                issues.append("Production requires budget controls")
            
            if not self.enable_content_safety:
                issues.append("Production requires content safety")
        
        return issues


# Predefined configurations
DEVELOPMENT_CONFIG = GAELPSafetyConfig.development_config()
STAGING_CONFIG = GAELPSafetyConfig.staging_config()
PRODUCTION_CONFIG = GAELPSafetyConfig.production_config()


def get_config(environment: str = None) -> GAELPSafetyConfig:
    """Get configuration based on environment"""
    if environment is None:
        environment = os.getenv('GAELP_ENVIRONMENT', 'development')
    
    if environment == 'development':
        return DEVELOPMENT_CONFIG
    elif environment == 'staging':
        return STAGING_CONFIG
    elif environment == 'production':
        return PRODUCTION_CONFIG
    else:
        # Try to load from environment variables
        return GAELPSafetyConfig.from_environment()


def validate_config(config: GAELPSafetyConfig) -> bool:
    """Validate configuration and log issues"""
    import logging
    logger = logging.getLogger(__name__)
    
    issues = config.validate()
    
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("Configuration validation passed")
        return True


# Environment-specific settings
ENVIRONMENT_SETTINGS = {
    'development': {
        'enable_sandbox_only': True,
        'require_human_approval': False,
        'enable_auto_pause': False,
        'log_all_actions': True
    },
    'staging': {
        'enable_sandbox_only': False,
        'require_human_approval': True,
        'enable_auto_pause': True,
        'log_all_actions': True
    },
    'production': {
        'enable_sandbox_only': False,
        'require_human_approval': True,
        'enable_auto_pause': True,
        'log_all_actions': True,
        'enable_compliance_mode': True
    }
}