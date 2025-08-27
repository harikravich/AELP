"""
FastAPI application for agent management
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import uvicorn

from ..core.database import create_tables, get_db
from ..core.services import AgentService, TrainingJobService, MetricsService
from ..core.models import (
    AgentCreateRequest, AgentResponse, AgentType, AgentStatus,
    TrainingJobCreateRequest, TrainingJobResponse, JobStatus,
    DeploymentCreateRequest, DeploymentResponse,
    AgentMetricsResponse
)
from ..kubernetes.scheduler import JobScheduler
from ..monitoring.websocket import WebSocketManager
from .auth import get_current_user, User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global services
job_scheduler = None
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Agent Manager service")
    create_tables()
    
    # Initialize job scheduler
    global job_scheduler
    job_scheduler = JobScheduler()
    
    # Start background tasks
    import asyncio
    async def scheduler_loop():
        while True:
            try:
                await job_scheduler.schedule_jobs()
                await job_scheduler.monitor_jobs()
                await asyncio.sleep(30)  # Run every 30 seconds
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    # Start scheduler in background
    asyncio.create_task(scheduler_loop())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Manager service")


# Create FastAPI app
app = FastAPI(
    title="GAELP Agent Manager",
    description="Agent management system for the GAELP platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
agent_service = AgentService()
job_service = TrainingJobService()
metrics_service = MetricsService()


# Agent Management Endpoints

@app.post("/api/v1/agents", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_request: AgentCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new agent"""
    try:
        agent = agent_service.create_agent(agent_request, current_user.username)
        logger.info(f"Created agent {agent.name} for user {current_user.username}")
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/agents", response_model=List[AgentResponse])
async def list_agents(
    skip: int = 0,
    limit: int = 100,
    agent_type: Optional[AgentType] = None,
    status: Optional[AgentStatus] = None,
    current_user: User = Depends(get_current_user)
):
    """List agents with filtering"""
    try:
        agents = agent_service.list_agents(
            skip=skip, 
            limit=limit, 
            agent_type=agent_type, 
            status=status,
            created_by=current_user.username if not current_user.is_admin else None
        )
        return agents
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get agent by ID"""
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check permissions
    if not current_user.is_admin and agent.created_by != current_user.username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return agent


@app.patch("/api/v1/agents/{agent_id}/status")
async def update_agent_status(
    agent_id: int,
    status: AgentStatus,
    current_user: User = Depends(get_current_user)
):
    """Update agent status"""
    # Check agent exists and user has permission
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if not current_user.is_admin and agent.created_by != current_user.username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = agent_service.update_agent_status(agent_id, status)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update agent status")
    
    return {"message": "Agent status updated successfully"}


@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete an agent"""
    # Check agent exists and user has permission
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if not current_user.is_admin and agent.created_by != current_user.username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        success = agent_service.delete_agent(agent_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete agent")
        
        return {"message": "Agent deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Training Job Endpoints

@app.post("/api/v1/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def create_training_job(
    job_request: TrainingJobCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Create a new training job"""
    try:
        # Check agent exists and user has permission
        agent = agent_service.get_agent(job_request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        if not current_user.is_admin and agent.created_by != current_user.username:
            raise HTTPException(status_code=403, detail="Access denied")
        
        job = job_service.create_job(job_request)
        logger.info(f"Created training job {job.name} for agent {job.agent_id}")
        
        # Trigger scheduling in background
        background_tasks.add_task(trigger_scheduling)
        
        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    skip: int = 0,
    limit: int = 100,
    agent_id: Optional[int] = None,
    status: Optional[JobStatus] = None,
    current_user: User = Depends(get_current_user)
):
    """List training jobs with filtering"""
    try:
        # If not admin, filter by user's agents
        if not current_user.is_admin and agent_id:
            agent = agent_service.get_agent(agent_id)
            if not agent or agent.created_by != current_user.username:
                raise HTTPException(status_code=403, detail="Access denied")
        
        jobs = job_service.list_jobs(
            skip=skip, 
            limit=limit, 
            agent_id=agent_id, 
            status=status
        )
        
        # Filter jobs by user if not admin
        if not current_user.is_admin:
            user_jobs = []
            for job in jobs:
                agent = agent_service.get_agent(job.agent_id)
                if agent and agent.created_by == current_user.username:
                    user_jobs.append(job)
            jobs = user_jobs
        
        return jobs
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get training job by ID"""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check permissions
    if not current_user.is_admin:
        agent = agent_service.get_agent(job.agent_id)
        if not agent or agent.created_by != current_user.username:
            raise HTTPException(status_code=403, detail="Access denied")
    
    return job


@app.get("/api/v1/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: int,
    lines: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get training job logs"""
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check permissions
    if not current_user.is_admin:
        agent = agent_service.get_agent(job.agent_id)
        if not agent or agent.created_by != current_user.username:
            raise HTTPException(status_code=403, detail="Access denied")
    
    if not job.k8s_job_name:
        raise HTTPException(status_code=400, detail="Job not started on Kubernetes")
    
    # Get logs from Kubernetes
    logs = job_scheduler.k8s_client.get_pod_logs(job.k8s_job_name, lines)
    if logs is None:
        raise HTTPException(status_code=404, detail="Logs not found")
    
    return {"logs": logs}


# Metrics Endpoints

@app.get("/api/v1/agents/{agent_id}/metrics", response_model=List[AgentMetricsResponse])
async def get_agent_metrics(
    agent_id: int,
    limit: int = 1000,
    current_user: User = Depends(get_current_user)
):
    """Get metrics for an agent"""
    # Check agent exists and user has permission
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if not current_user.is_admin and agent.created_by != current_user.username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        metrics = metrics_service.get_metrics(agent_id, limit=limit)
        return [AgentMetricsResponse(**metric) for metric in metrics]
    except Exception as e:
        logger.error(f"Error getting agent metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/agents/{agent_id}/metrics")
async def record_agent_metrics(
    agent_id: int,
    metrics: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Record metrics for an agent"""
    # Check agent exists and user has permission
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if not current_user.is_admin and agent.created_by != current_user.username:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        success = metrics_service.record_metrics(agent_id, metrics)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record metrics")
        
        # Broadcast metrics via WebSocket
        await websocket_manager.broadcast_agent_metrics(agent_id, metrics)
        
        return {"message": "Metrics recorded successfully"}
    except Exception as e:
        logger.error(f"Error recording agent metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# System Status Endpoints

@app.get("/api/v1/status")
async def get_system_status(current_user: User = Depends(get_current_user)):
    """Get system status"""
    try:
        # Get cluster resources
        cluster_resources = job_scheduler.k8s_client.get_resource_usage()
        
        # Get job counts
        queued_jobs = len(job_service.get_queued_jobs(limit=1000))
        running_jobs = len(job_service.list_jobs(status=JobStatus.RUNNING, limit=1000))
        
        # Get agent counts
        all_agents = agent_service.list_agents(limit=1000)
        active_agents = len([a for a in all_agents if a.status == AgentStatus.RUNNING])
        
        return {
            "cluster_resources": cluster_resources,
            "job_counts": {
                "queued": queued_jobs,
                "running": running_jobs
            },
            "agent_counts": {
                "total": len(all_agents),
                "active": active_agents
            },
            "scheduler_status": "running"
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agent-manager"}


# Background task functions

async def trigger_scheduling():
    """Trigger job scheduling"""
    if job_scheduler:
        try:
            await job_scheduler.schedule_jobs()
        except Exception as e:
            logger.error(f"Error in triggered scheduling: {e}")


# WebSocket endpoint
@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket, agent_id: int):
    """WebSocket endpoint for real-time agent monitoring"""
    await websocket_manager.connect(websocket, agent_id)


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )