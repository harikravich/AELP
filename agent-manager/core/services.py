"""
Core business logic services for agent management
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from .models import (
    Agent, TrainingJob, AgentDeployment, ResourceQuota, AgentMetrics,
    AgentType, AgentStatus, JobStatus,
    AgentCreateRequest, AgentResponse, TrainingJobCreateRequest, 
    TrainingJobResponse, DeploymentCreateRequest, DeploymentResponse
)
from .database import get_db_session

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agents"""
    
    def create_agent(self, agent_request: AgentCreateRequest, created_by: str) -> AgentResponse:
        """Create a new agent"""
        with get_db_session() as db:
            # Check if agent name already exists
            existing_agent = db.query(Agent).filter(Agent.name == agent_request.name).first()
            if existing_agent:
                raise ValueError(f"Agent with name '{agent_request.name}' already exists")
            
            # Create new agent
            agent = Agent(
                name=agent_request.name,
                type=agent_request.type.value,
                version=agent_request.version,
                docker_image=agent_request.docker_image,
                description=agent_request.description,
                config=agent_request.config.dict(),
                resource_requirements=agent_request.resource_requirements.dict(),
                environment_vars=agent_request.environment_vars,
                secrets=agent_request.secrets,
                budget_limit=agent_request.budget_limit,
                created_by=created_by,
                status=AgentStatus.PENDING.value
            )
            
            db.add(agent)
            db.flush()
            db.refresh(agent)
            
            logger.info(f"Created agent {agent.name} with ID {agent.id}")
            return self._agent_to_response(agent)
    
    def get_agent(self, agent_id: int) -> Optional[AgentResponse]:
        """Get agent by ID"""
        with get_db_session() as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                return self._agent_to_response(agent)
            return None
    
    def get_agent_by_name(self, name: str) -> Optional[AgentResponse]:
        """Get agent by name"""
        with get_db_session() as db:
            agent = db.query(Agent).filter(Agent.name == name).first()
            if agent:
                return self._agent_to_response(agent)
            return None
    
    def list_agents(
        self, 
        skip: int = 0, 
        limit: int = 100,
        agent_type: Optional[AgentType] = None,
        status: Optional[AgentStatus] = None,
        created_by: Optional[str] = None
    ) -> List[AgentResponse]:
        """List agents with filtering"""
        with get_db_session() as db:
            query = db.query(Agent)
            
            if agent_type:
                query = query.filter(Agent.type == agent_type.value)
            if status:
                query = query.filter(Agent.status == status.value)
            if created_by:
                query = query.filter(Agent.created_by == created_by)
            
            agents = query.offset(skip).limit(limit).all()
            return [self._agent_to_response(agent) for agent in agents]
    
    def update_agent_status(self, agent_id: int, status: AgentStatus) -> bool:
        """Update agent status"""
        with get_db_session() as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                agent.status = status.value
                agent.updated_at = datetime.utcnow()
                logger.info(f"Updated agent {agent_id} status to {status.value}")
                return True
            return False
    
    def update_agent_cost(self, agent_id: int, cost_delta: float) -> bool:
        """Update agent current cost"""
        with get_db_session() as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                agent.current_cost += cost_delta
                agent.updated_at = datetime.utcnow()
                
                # Check budget limit
                if agent.budget_limit and agent.current_cost > agent.budget_limit:
                    logger.warning(f"Agent {agent_id} exceeded budget limit: {agent.current_cost} > {agent.budget_limit}")
                
                return True
            return False
    
    def delete_agent(self, agent_id: int) -> bool:
        """Delete an agent"""
        with get_db_session() as db:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if agent:
                # Check if agent has running jobs
                running_jobs = db.query(TrainingJob).filter(
                    and_(
                        TrainingJob.agent_id == agent_id,
                        TrainingJob.status == JobStatus.RUNNING.value
                    )
                ).count()
                
                if running_jobs > 0:
                    raise ValueError(f"Cannot delete agent with {running_jobs} running jobs")
                
                db.delete(agent)
                logger.info(f"Deleted agent {agent_id}")
                return True
            return False
    
    def _agent_to_response(self, agent: Agent) -> AgentResponse:
        """Convert Agent model to response"""
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            type=AgentType(agent.type),
            status=AgentStatus(agent.status),
            version=agent.version,
            docker_image=agent.docker_image,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            created_by=agent.created_by,
            description=agent.description,
            config=agent.config or {},
            resource_requirements=agent.resource_requirements or {},
            budget_limit=agent.budget_limit,
            current_cost=agent.current_cost or 0.0
        )


class TrainingJobService:
    """Service for managing training jobs"""
    
    def create_job(self, job_request: TrainingJobCreateRequest) -> TrainingJobResponse:
        """Create a new training job"""
        with get_db_session() as db:
            # Verify agent exists
            agent = db.query(Agent).filter(Agent.id == job_request.agent_id).first()
            if not agent:
                raise ValueError(f"Agent {job_request.agent_id} not found")
            
            # Create job
            job = TrainingJob(
                agent_id=job_request.agent_id,
                name=job_request.name,
                priority=job_request.priority,
                hyperparameters=job_request.hyperparameters,
                training_config=job_request.training_config,
                status=JobStatus.QUEUED.value
            )
            
            if job_request.resource_requirements:
                job.allocated_resources = job_request.resource_requirements.dict()
            
            db.add(job)
            db.flush()
            db.refresh(job)
            
            logger.info(f"Created training job {job.name} with ID {job.id}")
            return self._job_to_response(job)
    
    def get_job(self, job_id: int) -> Optional[TrainingJobResponse]:
        """Get job by ID"""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                return self._job_to_response(job)
            return None
    
    def list_jobs(
        self,
        skip: int = 0,
        limit: int = 100,
        agent_id: Optional[int] = None,
        status: Optional[JobStatus] = None
    ) -> List[TrainingJobResponse]:
        """List training jobs with filtering"""
        with get_db_session() as db:
            query = db.query(TrainingJob)
            
            if agent_id:
                query = query.filter(TrainingJob.agent_id == agent_id)
            if status:
                query = query.filter(TrainingJob.status == status.value)
            
            jobs = query.order_by(TrainingJob.priority.desc(), TrainingJob.created_at).offset(skip).limit(limit).all()
            return [self._job_to_response(job) for job in jobs]
    
    def update_job_status(
        self, 
        job_id: int, 
        status: JobStatus,
        k8s_job_name: Optional[str] = None
    ) -> bool:
        """Update job status"""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.status = status.value
                
                if status == JobStatus.RUNNING and not job.started_at:
                    job.started_at = datetime.utcnow()
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    job.completed_at = datetime.utcnow()
                
                if k8s_job_name:
                    job.k8s_job_name = k8s_job_name
                
                logger.info(f"Updated job {job_id} status to {status.value}")
                return True
            return False
    
    def update_job_cost(self, job_id: int, cost: float) -> bool:
        """Update job cost"""
        with get_db_session() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                old_cost = job.cost or 0.0
                job.cost = cost
                
                # Update agent cost
                cost_delta = cost - old_cost
                agent_service = AgentService()
                agent_service.update_agent_cost(job.agent_id, cost_delta)
                
                return True
            return False
    
    def get_queued_jobs(self, limit: int = 10) -> List[TrainingJobResponse]:
        """Get queued jobs ordered by priority"""
        with get_db_session() as db:
            jobs = db.query(TrainingJob).filter(
                TrainingJob.status == JobStatus.QUEUED.value
            ).order_by(
                TrainingJob.priority.desc(),
                TrainingJob.created_at
            ).limit(limit).all()
            
            return [self._job_to_response(job) for job in jobs]
    
    def _job_to_response(self, job: TrainingJob) -> TrainingJobResponse:
        """Convert TrainingJob model to response"""
        return TrainingJobResponse(
            id=job.id,
            agent_id=job.agent_id,
            name=job.name,
            status=JobStatus(job.status),
            priority=job.priority,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            hyperparameters=job.hyperparameters or {},
            cost=job.cost or 0.0,
            k8s_job_name=job.k8s_job_name
        )


class ResourceService:
    """Service for managing resource quotas and usage"""
    
    def check_quota(self, user_id: str, resource_requirements: Dict[str, Any]) -> bool:
        """Check if user has sufficient quota for resource requirements"""
        with get_db_session() as db:
            for resource_type, required in resource_requirements.items():
                quota = db.query(ResourceQuota).filter(
                    and_(
                        ResourceQuota.user_id == user_id,
                        ResourceQuota.resource_type == resource_type
                    )
                ).first()
                
                if not quota:
                    logger.warning(f"No quota found for user {user_id} and resource {resource_type}")
                    return False
                
                # Parse resource value (e.g., "2Gi" -> 2)
                required_value = self._parse_resource_value(required)
                
                if quota.current_usage + required_value > quota.quota_limit:
                    logger.warning(f"Quota exceeded for user {user_id}: {resource_type}")
                    return False
        
        return True
    
    def allocate_resources(self, user_id: str, resource_requirements: Dict[str, Any]) -> bool:
        """Allocate resources to user"""
        with get_db_session() as db:
            for resource_type, required in resource_requirements.items():
                quota = db.query(ResourceQuota).filter(
                    and_(
                        ResourceQuota.user_id == user_id,
                        ResourceQuota.resource_type == resource_type
                    )
                ).first()
                
                if quota:
                    required_value = self._parse_resource_value(required)
                    quota.current_usage += required_value
                    quota.updated_at = datetime.utcnow()
        
        return True
    
    def deallocate_resources(self, user_id: str, resource_requirements: Dict[str, Any]) -> bool:
        """Deallocate resources from user"""
        with get_db_session() as db:
            for resource_type, required in resource_requirements.items():
                quota = db.query(ResourceQuota).filter(
                    and_(
                        ResourceQuota.user_id == user_id,
                        ResourceQuota.resource_type == resource_type
                    )
                ).first()
                
                if quota:
                    required_value = self._parse_resource_value(required)
                    quota.current_usage = max(0, quota.current_usage - required_value)
                    quota.updated_at = datetime.utcnow()
        
        return True
    
    def _parse_resource_value(self, value: str) -> float:
        """Parse resource value string to float"""
        if isinstance(value, (int, float)):
            return float(value)
        
        value = str(value).lower()
        
        # Handle memory units
        if value.endswith('gi'):
            return float(value[:-2])
        elif value.endswith('mi'):
            return float(value[:-2]) / 1024
        elif value.endswith('g'):
            return float(value[:-1])
        elif value.endswith('m'):
            return float(value[:-1]) / 1024
        
        # Handle CPU units
        if value.endswith('m'):
            return float(value[:-1]) / 1000
        
        try:
            return float(value)
        except ValueError:
            return 0.0


class MetricsService:
    """Service for managing agent metrics"""
    
    def record_metrics(
        self, 
        agent_id: int, 
        metrics: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Record metrics for an agent"""
        with get_db_session() as db:
            metric = AgentMetrics(
                agent_id=agent_id,
                timestamp=timestamp or datetime.utcnow(),
                cpu_usage=metrics.get('cpu_usage'),
                memory_usage=metrics.get('memory_usage'),
                gpu_usage=metrics.get('gpu_usage'),
                training_loss=metrics.get('training_loss'),
                validation_accuracy=metrics.get('validation_accuracy'),
                epoch=metrics.get('epoch'),
                custom_metrics=metrics.get('custom_metrics', {})
            )
            
            db.add(metric)
            return True
    
    def get_metrics(
        self, 
        agent_id: int, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get metrics for an agent"""
        with get_db_session() as db:
            query = db.query(AgentMetrics).filter(AgentMetrics.agent_id == agent_id)
            
            if start_time:
                query = query.filter(AgentMetrics.timestamp >= start_time)
            if end_time:
                query = query.filter(AgentMetrics.timestamp <= end_time)
            
            metrics = query.order_by(AgentMetrics.timestamp.desc()).limit(limit).all()
            
            return [
                {
                    'timestamp': metric.timestamp,
                    'cpu_usage': metric.cpu_usage,
                    'memory_usage': metric.memory_usage,
                    'gpu_usage': metric.gpu_usage,
                    'training_loss': metric.training_loss,
                    'validation_accuracy': metric.validation_accuracy,
                    'epoch': metric.epoch,
                    'custom_metrics': metric.custom_metrics or {}
                }
                for metric in metrics
            ]