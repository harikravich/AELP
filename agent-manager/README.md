# GAELP Agent Manager

The Agent Manager is a core component of the GAELP (Google Ads Enhanced Learning Platform) that handles the complete lifecycle of AI agents from registration to training execution.

## Features

### Agent Lifecycle Management
- **Agent Registration**: Container-based agent deployment with Docker
- **Resource Allocation**: CPU, GPU, memory management with quotas
- **Scaling Policies**: Auto-scale based on workload demand
- **Version Management**: Agent versioning and rollback capabilities
- **Environment Management**: Secure handling of variables and secrets

### Job Scheduling & Orchestration
- **Kubernetes-based Orchestration**: Native Kubernetes job scheduling
- **Priority Queues**: Priority-based training job scheduling
- **Resource Optimization**: Intelligent resource allocation algorithms
- **Auto-scaling**: Dynamic scaling based on queue length and utilization
- **Job Preemption**: Support for preemptible instances and migration

### Monitoring & Observability
- **Real-time Metrics**: Agent performance and resource utilization
- **Health Monitoring**: Automated health checks and recovery
- **WebSocket Support**: Real-time monitoring dashboards
- **Prometheus Integration**: Comprehensive metrics collection
- **Alert Management**: Automated alerting for issues

### Multi-Agent Support
- **Parallel Training**: Coordinate multiple agents simultaneously
- **Experiment Management**: A/B testing and comparison framework
- **Resource Sharing**: Efficient resource sharing between agents
- **Budget Controls**: Per-agent and per-user budget enforcement

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Kubernetes    │    │   PostgreSQL    │
│   REST API      │◄──►│   Cluster       │    │   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   Job           │    │   Redis         │
│   Manager       │    │   Scheduler     │    │   Queue         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │   Resource      │    │   Alert         │
│   Metrics       │    │   Manager       │    │   Manager       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Kubernetes cluster (local or GKE)
- PostgreSQL database
- Redis instance

### Local Development

1. **Clone and Setup**
```bash
cd /home/hariravichandran/AELP/agent-manager
cp .env.example .env
# Edit .env with your configuration
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Infrastructure**
```bash
docker-compose up -d postgres redis
```

4. **Run the Service**
```bash
python main.py
```

### Docker Deployment

```bash
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/deployment.yaml

# Create secrets
kubectl create secret generic agent-manager-secrets \
  --from-literal=db-host=your-db-host \
  --from-literal=db-name=agent_manager \
  --from-literal=db-user=agent_manager \
  --from-literal=db-password=your-password \
  --from-literal=redis-host=your-redis-host \
  --from-literal=jwt-secret=your-jwt-secret
```

## API Usage

### Authentication

All API requests require a JWT token in the Authorization header:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/agents
```

### Creating an Agent

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "name": "campaign-optimizer-v1",
    "type": "real_deployment",
    "version": "1.0.0",
    "docker_image": "gcr.io/gaelp/campaign-optimizer:v1.0.0",
    "description": "Campaign optimization agent for Google Ads",
    "config": {
      "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32
      },
      "environment_selection": "production",
      "performance_thresholds": {
        "accuracy": 0.85
      }
    },
    "resource_requirements": {
      "cpu": "2",
      "memory": "4Gi",
      "gpu": "1"
    },
    "budget_limit": 1000.0
  }'
```

### Starting a Training Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "agent_id": 1,
    "name": "training-run-001",
    "priority": 8,
    "hyperparameters": {
      "epochs": 100,
      "learning_rate": 0.001
    },
    "training_config": {
      "dataset": "google_ads_campaigns_2024",
      "validation_split": 0.2
    }
  }'
```

### Real-time Monitoring

Connect to WebSocket for real-time agent monitoring:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/1'); // Monitor agent ID 1

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Agent metrics:', data);
};
```

## Agent Types

### Simulation Agents
- Training in simulated environments
- No real budget impact
- Used for experimentation and development

### Real Deployment Agents
- Operating with real budgets
- Strict safety controls
- Production monitoring

### Evaluation Agents
- Benchmarking and testing
- Comparative analysis
- Performance validation

### Research Agents
- Experimental features
- Advanced algorithms
- Academic research

## Configuration

### Environment Variables

Key configuration options:

- `ENVIRONMENT`: deployment environment (development/production)
- `K8S_NAMESPACE`: Kubernetes namespace for training jobs
- `K8S_MAX_CONCURRENT_JOBS`: Maximum concurrent training jobs
- `RESOURCE_MAX_CPU_PER_JOB`: Maximum CPU per training job
- `MONITORING_PROMETHEUS_PORT`: Prometheus metrics port

### Resource Quotas

Default resource quotas per user:
- CPU: 10 cores
- Memory: 20 GB
- GPU: 2 units
- Storage: 100 GB

## Monitoring

### Metrics

The system exposes Prometheus metrics at `/metrics`:

- `agent_jobs_total`: Total number of jobs per agent
- `agent_job_duration_seconds`: Job duration histogram
- `agent_resource_usage`: Current resource usage
- `cluster_resource_usage`: Cluster-wide resource usage

### Health Checks

Health endpoint at `/health` returns:
```json
{
  "status": "healthy",
  "service": "agent-manager"
}
```

### Alerts

Automated alerts for:
- High resource utilization (>90% CPU/memory)
- Job failure rate >20%
- Queue length >50 jobs
- Budget overruns

## Integration

### Training Orchestrator
- Job execution coordination
- Model checkpointing
- Result aggregation

### Safety & Policy Engine
- Budget enforcement
- Safety policy validation
- Compliance monitoring

### Benchmark Portal
- Performance result display
- Comparative analysis
- Historical tracking

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black .

# Type checking
mypy .

# Linting
flake8 .
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Add new feature"

# Apply migrations
alembic upgrade head
```

## Production Deployment

### GKE Deployment

1. **Create GKE Cluster**
```bash
gcloud container clusters create gaelp-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10
```

2. **Deploy to GKE**
```bash
# Build and push image
docker build -t gcr.io/YOUR_PROJECT/agent-manager:latest .
docker push gcr.io/YOUR_PROJECT/agent-manager:latest

# Deploy
kubectl apply -f kubernetes/deployment.yaml
```

### Security Considerations

- Use strong JWT secrets in production
- Enable RBAC in Kubernetes
- Use Google Cloud Secret Manager for sensitive data
- Regular security updates and patches

### Scaling

- Horizontal Pod Autoscaler for API pods
- Cluster autoscaler for node scaling
- Resource quotas to prevent resource exhaustion
- Load balancing for high availability

## Troubleshooting

### Common Issues

1. **Jobs stuck in queue**
   - Check resource quotas
   - Verify cluster capacity
   - Review job priority settings

2. **High resource usage**
   - Monitor Prometheus metrics
   - Check for resource leaks
   - Review resource limits

3. **Database connection issues**
   - Verify database credentials
   - Check network connectivity
   - Review connection pool settings

### Logs

View logs using:
```bash
# Docker Compose
docker-compose logs agent-manager-api

# Kubernetes
kubectl logs -f deployment/agent-manager -n gaelp
```

## Support

For issues and questions:
- Check the troubleshooting guide
- Review logs and metrics
- Contact the GAELP team