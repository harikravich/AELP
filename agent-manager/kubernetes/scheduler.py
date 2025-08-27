"""
Job scheduler for managing training job lifecycle on Kubernetes
"""
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from ..core.services import TrainingJobService, AgentService, ResourceService
from ..core.models import JobStatus, AgentStatus
from .client import KubernetesClient

logger = logging.getLogger(__name__)


@dataclass
class SchedulingDecision:
    """Represents a scheduling decision"""
    job_id: int
    schedule: bool
    reason: str
    estimated_start_time: Optional[datetime] = None
    resource_allocation: Optional[Dict[str, str]] = None


class JobScheduler:
    """Kubernetes-based job scheduler with resource management"""
    
    def __init__(self, namespace: str = "agent-training"):
        self.k8s_client = KubernetesClient(namespace)
        self.job_service = TrainingJobService()
        self.agent_service = AgentService()
        self.resource_service = ResourceService()
        self.namespace = namespace
        
        # Scheduling configuration
        self.max_concurrent_jobs = 10
        self.resource_buffer = 0.1  # 10% buffer for resource allocation
        
    async def schedule_jobs(self) -> List[SchedulingDecision]:
        """Main scheduling loop - process queued jobs"""
        logger.info("Starting job scheduling cycle")
        
        # Get queued jobs ordered by priority
        queued_jobs = self.job_service.get_queued_jobs(limit=50)
        
        if not queued_jobs:
            logger.debug("No queued jobs to schedule")
            return []
        
        # Get current cluster resource usage
        cluster_resources = self.k8s_client.get_resource_usage()
        running_jobs_count = len(self._get_running_jobs())
        
        scheduling_decisions = []
        
        for job in queued_jobs:
            # Check if we can schedule this job
            decision = await self._evaluate_job_for_scheduling(
                job, cluster_resources, running_jobs_count
            )
            
            if decision.schedule:
                # Attempt to start the job
                success = await self._start_training_job(job)
                if success:
                    running_jobs_count += 1
                    decision.reason += " - Job started successfully"
                else:
                    decision.schedule = False
                    decision.reason = "Failed to start job on Kubernetes"
            
            scheduling_decisions.append(decision)
            
            # Break if we've reached max concurrent jobs
            if running_jobs_count >= self.max_concurrent_jobs:
                logger.info(f"Reached max concurrent jobs limit: {self.max_concurrent_jobs}")
                break
        
        logger.info(f"Completed scheduling cycle. Processed {len(scheduling_decisions)} jobs")
        return scheduling_decisions
    
    async def _evaluate_job_for_scheduling(
        self, 
        job, 
        cluster_resources: Dict[str, Any],
        running_jobs_count: int
    ) -> SchedulingDecision:
        """Evaluate whether a job can be scheduled"""
        
        # Check concurrent job limit
        if running_jobs_count >= self.max_concurrent_jobs:
            return SchedulingDecision(
                job_id=job.id,
                schedule=False,
                reason="Max concurrent jobs limit reached"
            )
        
        # Get agent and check status
        agent = self.agent_service.get_agent(job.agent_id)
        if not agent:
            return SchedulingDecision(
                job_id=job.id,
                schedule=False,
                reason="Agent not found"
            )
        
        # Check agent budget
        if agent.budget_limit and agent.current_cost >= agent.budget_limit:
            return SchedulingDecision(
                job_id=job.id,
                schedule=False,
                reason="Agent budget limit exceeded"
            )
        
        # Get resource requirements
        resource_reqs = job.allocated_resources or agent.resource_requirements
        if not resource_reqs:
            resource_reqs = {
                "cpu": "1",
                "memory": "2Gi"
            }
        
        # Check resource availability
        if not self._check_cluster_resources(resource_reqs, cluster_resources):
            return SchedulingDecision(
                job_id=job.id,
                schedule=False,
                reason="Insufficient cluster resources"
            )
        
        # Check user quotas
        if not self.resource_service.check_quota(agent.created_by, resource_reqs):
            return SchedulingDecision(
                job_id=job.id,
                schedule=False,
                reason="User quota exceeded"
            )
        
        return SchedulingDecision(
            job_id=job.id,
            schedule=True,
            reason="Job meets all scheduling criteria",
            estimated_start_time=datetime.utcnow(),
            resource_allocation=resource_reqs
        )
    
    async def _start_training_job(self, job) -> bool:
        """Start a training job on Kubernetes"""
        try:
            # Get agent details
            agent = self.agent_service.get_agent(job.agent_id)
            if not agent:
                logger.error(f"Agent {job.agent_id} not found for job {job.id}")
                return False
            
            # Generate unique job name
            k8s_job_name = f"training-job-{job.id}-{int(datetime.utcnow().timestamp())}"
            
            # Prepare job configuration
            resource_reqs = job.allocated_resources or agent.resource_requirements
            environment_vars = self._prepare_environment_vars(job, agent)
            command, args = self._prepare_command_args(job, agent)
            
            # Create Kubernetes job
            success = self.k8s_client.create_training_job(
                job_name=k8s_job_name,
                agent_name=agent.name,
                docker_image=agent.docker_image,
                command=command,
                args=args,
                resource_requirements=resource_reqs,
                environment_vars=environment_vars,
                secrets=agent.secrets,
                node_selector=self._get_node_selector(job, agent)
            )
            
            if success:
                # Update job status
                self.job_service.update_job_status(
                    job.id, 
                    JobStatus.RUNNING,
                    k8s_job_name=k8s_job_name
                )
                
                # Allocate resources to user
                self.resource_service.allocate_resources(
                    agent.created_by, 
                    resource_reqs
                )
                
                logger.info(f"Started training job {job.id} as {k8s_job_name}")
                return True
            else:
                logger.error(f"Failed to create Kubernetes job for training job {job.id}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting training job {job.id}: {e}")
            return False
    
    def _prepare_environment_vars(self, job, agent) -> Dict[str, str]:
        """Prepare environment variables for the job"""
        env_vars = {
            "AGENT_ID": str(agent.id),
            "AGENT_NAME": agent.name,
            "JOB_ID": str(job.id),
            "JOB_NAME": job.name,
            "GAELP_ENV": "production"
        }
        
        # Add agent environment variables
        if agent.environment_vars:
            env_vars.update(agent.environment_vars)
        
        # Add hyperparameters as environment variables
        if job.hyperparameters:
            for key, value in job.hyperparameters.items():
                env_key = f"HYPERPARAM_{key.upper()}"
                env_vars[env_key] = str(value)
        
        return env_vars
    
    def _prepare_command_args(self, job, agent) -> tuple:
        """Prepare command and arguments for the job"""
        # Default training command
        command = ["python", "-m", "gaelp.training.main"]
        
        args = [
            "--agent-id", str(agent.id),
            "--job-id", str(job.id),
            "--config", json.dumps(job.training_config or {})
        ]
        
        # Add hyperparameters as arguments
        if job.hyperparameters:
            args.extend(["--hyperparams", json.dumps(job.hyperparameters)])
        
        return command, args
    
    def _get_node_selector(self, job, agent) -> Optional[Dict[str, str]]:
        """Get node selector based on job requirements"""
        node_selector = {}
        
        # GPU requirement
        resource_reqs = job.allocated_resources or agent.resource_requirements
        if resource_reqs and "nvidia.com/gpu" in resource_reqs:
            node_selector["accelerator"] = "nvidia-tesla-k80"  # Default GPU type
        
        # Agent type specific scheduling
        if agent.type == "research":
            node_selector["node-type"] = "research"
        elif agent.type == "real_deployment":
            node_selector["node-type"] = "production"
        
        return node_selector if node_selector else None
    
    def _check_cluster_resources(
        self, 
        required: Dict[str, str], 
        available: Dict[str, Any]
    ) -> bool:
        """Check if cluster has sufficient resources"""
        # Parse required CPU
        required_cpu = self._parse_cpu_requirement(required.get("cpu", "0"))
        available_cpu = available.get("cpu_limits", 0) - available.get("cpu_requests", 0)
        
        if required_cpu > available_cpu * (1 - self.resource_buffer):
            return False
        
        # Parse required memory
        required_memory = self._parse_memory_requirement(required.get("memory", "0"))
        available_memory = available.get("memory_limits", 0) - available.get("memory_requests", 0)
        
        if required_memory > available_memory * (1 - self.resource_buffer):
            return False
        
        return True
    
    def _parse_cpu_requirement(self, cpu_str: str) -> float:
        """Parse CPU requirement string"""
        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
    
    def _parse_memory_requirement(self, memory_str: str) -> float:
        """Parse memory requirement string (in GB)"""
        memory_str = memory_str.upper()
        if memory_str.endswith("GI"):
            return float(memory_str[:-2])
        elif memory_str.endswith("G"):
            return float(memory_str[:-1])
        elif memory_str.endswith("MI"):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith("M"):
            return float(memory_str[:-1]) / 1024
        return float(memory_str) / (1024 * 1024 * 1024)
    
    def _get_running_jobs(self) -> List:
        """Get currently running jobs"""
        return self.job_service.list_jobs(status=JobStatus.RUNNING, limit=100)
    
    async def monitor_jobs(self):
        """Monitor running jobs and update their status"""
        logger.info("Starting job monitoring cycle")
        
        running_jobs = self._get_running_jobs()
        
        for job in running_jobs:
            if not job.k8s_job_name:
                continue
            
            # Get Kubernetes job status
            k8s_status = self.k8s_client.get_job_status(job.k8s_job_name)
            
            if not k8s_status:
                logger.warning(f"Could not get status for K8s job {job.k8s_job_name}")
                continue
            
            # Update job status based on Kubernetes status
            new_status = self._determine_job_status(k8s_status)
            
            if new_status != job.status:
                logger.info(f"Updating job {job.id} status from {job.status} to {new_status}")
                self.job_service.update_job_status(job.id, new_status)
                
                # Deallocate resources if job completed
                if new_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    await self._cleanup_completed_job(job)
    
    def _determine_job_status(self, k8s_status: Dict[str, Any]) -> JobStatus:
        """Determine job status from Kubernetes job status"""
        if k8s_status.get("succeeded", 0) > 0:
            return JobStatus.COMPLETED
        elif k8s_status.get("failed", 0) > 0:
            return JobStatus.FAILED
        elif k8s_status.get("active", 0) > 0:
            return JobStatus.RUNNING
        else:
            return JobStatus.RUNNING  # Default to running if unclear
    
    async def _cleanup_completed_job(self, job):
        """Cleanup resources for completed job"""
        try:
            # Get agent for resource deallocation
            agent = self.agent_service.get_agent(job.agent_id)
            if agent:
                resource_reqs = job.allocated_resources or agent.resource_requirements
                if resource_reqs:
                    self.resource_service.deallocate_resources(
                        agent.created_by, 
                        resource_reqs
                    )
            
            # Optional: Delete Kubernetes job after some time
            # self.k8s_client.delete_job(job.k8s_job_name)
            
            logger.info(f"Cleaned up completed job {job.id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up job {job.id}: {e}")


class AutoScaler:
    """Auto-scaling based on job queue and resource utilization"""
    
    def __init__(self, scheduler: JobScheduler):
        self.scheduler = scheduler
        self.k8s_client = scheduler.k8s_client
        
    async def scale_based_on_demand(self):
        """Scale cluster based on job demand"""
        # Get queue length
        queued_jobs = self.scheduler.job_service.get_queued_jobs(limit=100)
        queue_length = len(queued_jobs)
        
        # Get current resource utilization
        cluster_resources = self.k8s_client.get_resource_usage()
        
        # Simple scaling logic
        if queue_length > 10 and self._is_resource_constrained(cluster_resources):
            logger.info("High queue length and resource constraint - scaling up recommended")
            # Here you would integrate with GKE auto-scaler or node pool management
        
        elif queue_length == 0 and self._is_resource_underutilized(cluster_resources):
            logger.info("Empty queue and low utilization - scaling down recommended")
    
    def _is_resource_constrained(self, resources: Dict[str, Any]) -> bool:
        """Check if cluster is resource constrained"""
        cpu_utilization = resources.get("cpu_requests", 0) / max(resources.get("cpu_limits", 1), 1)
        memory_utilization = resources.get("memory_requests", 0) / max(resources.get("memory_limits", 1), 1)
        
        return cpu_utilization > 0.8 or memory_utilization > 0.8
    
    def _is_resource_underutilized(self, resources: Dict[str, Any]) -> bool:
        """Check if cluster is underutilized"""
        cpu_utilization = resources.get("cpu_requests", 0) / max(resources.get("cpu_limits", 1), 1)
        memory_utilization = resources.get("memory_requests", 0) / max(resources.get("memory_limits", 1), 1)
        
        return cpu_utilization < 0.2 and memory_utilization < 0.2