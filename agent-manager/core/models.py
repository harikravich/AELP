"""
Core data models for agent management system
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class AgentType(str, Enum):
    SIMULATION = "simulation"
    REAL_DEPLOYMENT = "real_deployment"
    EVALUATION = "evaluation"
    RESEARCH = "research"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"


# SQLAlchemy Models
class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    type = Column(String(50), nullable=False)
    status = Column(String(50), default=AgentStatus.PENDING)
    version = Column(String(50), nullable=False)
    docker_image = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Configuration and resources
    config = Column(JSON)
    resource_requirements = Column(JSON)
    environment_vars = Column(JSON)
    secrets = Column(JSON)
    
    # Budget and quota
    budget_limit = Column(Float)
    current_cost = Column(Float, default=0.0)
    
    # Relationships
    jobs = relationship("TrainingJob", back_populates="agent")
    deployments = relationship("AgentDeployment", back_populates="agent")


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default=JobStatus.QUEUED)
    priority = Column(Integer, default=5)  # 1-10, 10 being highest
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Configuration
    hyperparameters = Column(JSON)
    training_config = Column(JSON)
    checkpoint_path = Column(String(500))
    
    # Resources
    allocated_resources = Column(JSON)
    resource_usage = Column(JSON)
    cost = Column(Float, default=0.0)
    
    # Kubernetes
    k8s_job_name = Column(String(255))
    k8s_namespace = Column(String(255), default="default")
    
    # Relationships
    agent = relationship("Agent", back_populates="jobs")


class AgentDeployment(Base):
    __tablename__ = "agent_deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")
    
    # Deployment configuration
    replicas = Column(Integer, default=1)
    docker_image = Column(String(255), nullable=False)
    resource_limits = Column(JSON)
    environment_vars = Column(JSON)
    
    # Kubernetes
    k8s_deployment_name = Column(String(255))
    k8s_service_name = Column(String(255))
    k8s_namespace = Column(String(255), default="default")
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime)
    
    # Relationships
    agent = relationship("Agent", back_populates="deployments")


class ResourceQuota(Base):
    __tablename__ = "resource_quotas"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False)
    quota_limit = Column(Float, nullable=False)
    current_usage = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentMetrics(Base):
    __tablename__ = "agent_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Performance metrics
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    gpu_usage = Column(Float)
    
    # Training metrics
    training_loss = Column(Float)
    validation_accuracy = Column(Float)
    epoch = Column(Integer)
    
    # Custom metrics
    custom_metrics = Column(JSON)


# Pydantic Models for API
class ResourceRequirements(BaseModel):
    cpu: str = "1"
    memory: str = "2Gi"
    gpu: Optional[str] = None
    storage: str = "10Gi"


class AgentConfig(BaseModel):
    hyperparameters: Dict[str, Any] = {}
    training_config: Dict[str, Any] = {}
    environment_selection: str = "simulation"
    safety_policy_id: Optional[str] = None
    performance_thresholds: Dict[str, float] = {}


class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: AgentType
    version: str = Field(..., min_length=1)
    docker_image: str = Field(..., min_length=1)
    description: Optional[str] = None
    config: AgentConfig = AgentConfig()
    resource_requirements: ResourceRequirements = ResourceRequirements()
    environment_vars: Dict[str, str] = {}
    secrets: Dict[str, str] = {}
    budget_limit: Optional[float] = None


class AgentResponse(BaseModel):
    id: int
    name: str
    type: AgentType
    status: AgentStatus
    version: str
    docker_image: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: Optional[str]
    config: AgentConfig
    resource_requirements: ResourceRequirements
    budget_limit: Optional[float]
    current_cost: float
    
    class Config:
        from_attributes = True


class TrainingJobCreateRequest(BaseModel):
    agent_id: int
    name: str = Field(..., min_length=1, max_length=255)
    priority: int = Field(default=5, ge=1, le=10)
    hyperparameters: Dict[str, Any] = {}
    training_config: Dict[str, Any] = {}
    resource_requirements: Optional[ResourceRequirements] = None


class TrainingJobResponse(BaseModel):
    id: int
    agent_id: int
    name: str
    status: JobStatus
    priority: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    hyperparameters: Dict[str, Any]
    cost: float
    k8s_job_name: Optional[str]
    
    class Config:
        from_attributes = True


class DeploymentCreateRequest(BaseModel):
    agent_id: int
    name: str = Field(..., min_length=1, max_length=255)
    replicas: int = Field(default=1, ge=1, le=10)
    resource_limits: Optional[ResourceRequirements] = None
    environment_vars: Dict[str, str] = {}


class DeploymentResponse(BaseModel):
    id: int
    agent_id: int
    name: str
    status: str
    replicas: int
    created_at: datetime
    deployed_at: Optional[datetime]
    k8s_deployment_name: Optional[str]
    k8s_service_name: Optional[str]
    
    class Config:
        from_attributes = True


class AgentMetricsResponse(BaseModel):
    timestamp: datetime
    cpu_usage: Optional[float]
    memory_usage: Optional[float]
    gpu_usage: Optional[float]
    training_loss: Optional[float]
    validation_accuracy: Optional[float]
    epoch: Optional[int]
    custom_metrics: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True