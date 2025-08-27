"""
Configuration settings for the agent manager
"""
import os
from typing import List, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "agent_manager"
    user: str = "agent_manager"
    password: str = "password"
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    class Config:
        env_prefix = "DB_"


class KubernetesSettings(BaseSettings):
    """Kubernetes configuration"""
    namespace: str = "agent-training"
    kubeconfig_path: Optional[str] = None
    in_cluster: bool = False
    job_ttl_seconds: int = 86400  # 24 hours
    max_concurrent_jobs: int = 50
    resource_buffer: float = 0.1  # 10% buffer
    
    class Config:
        env_prefix = "K8S_"


class ResourceSettings(BaseSettings):
    """Resource management configuration"""
    default_cpu_request: str = "1"
    default_memory_request: str = "2Gi"
    default_storage_request: str = "10Gi"
    max_cpu_per_job: str = "8"
    max_memory_per_job: str = "32Gi"
    max_gpu_per_job: int = 4
    
    class Config:
        env_prefix = "RESOURCE_"


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""
    prometheus_port: int = 8080
    metrics_retention_days: int = 30
    health_check_interval: int = 60  # seconds
    alert_thresholds: dict = {
        "cpu_usage": 90.0,
        "memory_usage": 85.0,
        "job_failure_rate": 20.0,
        "queue_length": 50
    }
    
    class Config:
        env_prefix = "MONITORING_"


class SecuritySettings(BaseSettings):
    """Security configuration"""
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    allowed_origins: List[str] = ["*"]
    
    class Config:
        env_prefix = "SECURITY_"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    
    class Config:
        env_prefix = "LOG_"


class RedisSettings(BaseSettings):
    """Redis configuration for job queuing"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class Settings(BaseSettings):
    """Main application settings"""
    app_name: str = "GAELP Agent Manager"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    kubernetes: KubernetesSettings = KubernetesSettings()
    resources: ResourceSettings = ResourceSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()
    redis: RedisSettings = RedisSettings()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings