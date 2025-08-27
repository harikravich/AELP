# GAELP Infrastructure

This directory contains the complete Google Cloud Platform infrastructure setup for the GAELP (Generic Agent Experimentation & Learning Platform) ad campaign learning platform.

## Architecture Overview

The GAELP infrastructure is designed to support:
- AI agent training with GPU compute for LLM simulation
- Real-time ad campaign deployment and monitoring
- BigQuery data storage for campaign performance metrics
- Container orchestration for agent deployment
- API services for environment and agent management
- Safety controls and budget monitoring
- Scalable compute for parallel campaign simulations

## Components

### Core Infrastructure
- **GKE Cluster**: Container orchestration with GPU support for training workloads
- **Cloud Run Services**: Serverless API endpoints for platform services
- **BigQuery**: Data warehousing for campaign metrics and training data
- **Cloud Storage**: Model artifacts, training data, and campaign assets
- **Vertex AI**: ML platform for model management and experimentation
- **IAM**: Security policies and service accounts with least-privilege access
- **Monitoring**: Comprehensive monitoring, alerting, and budget controls

### Networking
- Private VPC with Cloud NAT for secure communication
- Firewall rules for controlled access
- Load balancing for high availability

## Directory Structure

```
infrastructure/
├── terraform/                 # Infrastructure as Code
│   ├── main.tf               # Main Terraform configuration
│   ├── terraform.tfvars.example
│   └── modules/              # Terraform modules
│       ├── networking/       # VPC, subnets, firewall rules
│       ├── gke/             # GKE cluster with node pools
│       ├── cloud_run/       # Cloud Run services
│       ├── bigquery/        # BigQuery datasets and tables
│       ├── storage/         # Cloud Storage buckets
│       ├── iam/             # IAM roles and service accounts
│       ├── monitoring/      # Monitoring, alerting, budgets
│       └── vertex_ai/       # Vertex AI resources
├── kubernetes/              # Kubernetes manifests
│   ├── namespaces.yaml      # Namespace definitions
│   └── training-orchestrator.yaml
├── cloudbuild/              # CI/CD configuration
│   └── build-config.yaml    # Cloud Build pipeline
├── scripts/                 # Setup and management scripts
│   ├── setup.sh            # Infrastructure deployment script
│   └── teardown.sh         # Infrastructure cleanup script
└── README.md               # This file
```

## Quick Start

### Prerequisites
- Google Cloud SDK (`gcloud`) installed and configured
- Terraform >= 1.0 installed
- kubectl installed
- A GCP project with billing enabled

### Setup

1. **Clone and navigate to the infrastructure directory:**
   ```bash
   cd AELP/infrastructure
   ```

2. **Run the setup script:**
   ```bash
   ./scripts/setup.sh --project YOUR_PROJECT_ID --region us-central1
   ```

   This script will:
   - Enable required GCP APIs
   - Initialize and apply Terraform configuration
   - Configure kubectl for the GKE cluster
   - Create Kubernetes namespaces
   - Install GPU drivers

3. **Configure your environment:**
   - Update `terraform/terraform.tfvars` with your specific settings
   - Review and update monitoring alert email addresses
   - Configure ad platform API credentials in Secret Manager

### Manual Setup (Alternative)

1. **Enable required APIs:**
   ```bash
   gcloud services enable container.googleapis.com compute.googleapis.com \
     cloudbuild.googleapis.com bigquery.googleapis.com storage.googleapis.com \
     run.googleapis.com monitoring.googleapis.com aiplatform.googleapis.com
   ```

2. **Initialize Terraform:**
   ```bash
   cd terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your project details
   terraform init
   terraform plan
   terraform apply
   ```

3. **Configure kubectl:**
   ```bash
   gcloud container clusters get-credentials gaelp-cluster --region us-central1
   ```

4. **Deploy Kubernetes resources:**
   ```bash
   kubectl apply -f ../kubernetes/namespaces.yaml
   kubectl apply -f ../kubernetes/training-orchestrator.yaml
   ```

## Infrastructure Components Details

### GKE Cluster
- **CPU Pool**: General workloads with auto-scaling (0-10 nodes)
- **GPU Pool**: ML training with NVIDIA Tesla T4 GPUs (0-5 nodes)
- **Memory Pool**: High-memory workloads for large model inference (0-3 nodes)
- All node pools use preemptible instances for cost optimization

### Cloud Run Services
- **Environment Registry**: Manages simulation environments
- **Agent Management**: Agent lifecycle and deployment
- **Campaign Monitor**: Real-time campaign monitoring
- **Metrics Collector**: Aggregates and processes metrics

### BigQuery Datasets
- **campaign_performance**: Ad campaign metrics and KPIs
- **agent_training**: Training logs and model metrics
- **simulation_results**: Campaign simulation outcomes
- **realtime_metrics**: Streaming metrics (shorter retention)

### Storage Buckets
- **model_artifacts**: ML model checkpoints and weights
- **training_data**: Training datasets and features
- **campaign_assets**: Campaign creatives and configurations
- **simulation_results**: Simulation outputs and reports
- **temp_processing**: Temporary processing files

### Security Features
- Workload Identity for secure GKE to GCP service integration
- KMS encryption for storage and Vertex AI resources
- Private GKE nodes with Cloud NAT for internet access
- Least-privilege IAM roles and service accounts
- Security policies enforcing best practices

### Monitoring and Alerting
- Custom dashboards for GAELP metrics
- Budget alerts for cost control
- Health checks for all services
- Performance monitoring for training jobs
- Error rate alerting for APIs

## Cost Optimization

The infrastructure is designed with cost optimization in mind:
- Preemptible instances for all compute workloads
- Auto-scaling node pools that scale to zero
- Lifecycle policies for storage with automatic tiering
- Regional persistent disks for better cost/performance
- Budget alerts and quota monitoring

## Scaling Considerations

### Horizontal Scaling
- GKE node pools auto-scale based on demand
- Cloud Run services auto-scale with traffic
- BigQuery scales automatically for query workloads

### Vertical Scaling
- GPU nodes for intensive training workloads
- High-memory nodes for large model inference
- Configurable resource limits for all components

## Maintenance

### Updates
- Terraform manages infrastructure changes
- Cloud Build provides CI/CD for application deployments
- GKE auto-updates can be configured through maintenance windows

### Backup and Recovery
- BigQuery automatic backups with configurable retention
- Storage versioning enabled for model artifacts
- Terraform state should be stored in Cloud Storage with versioning

### Monitoring
- Use the provided monitoring dashboard for system health
- Set up custom alerts based on your specific SLAs
- Regular review of cost and usage through billing reports

## Cleanup

To destroy all infrastructure:
```bash
./scripts/teardown.sh --project YOUR_PROJECT_ID
```

**Warning**: This will destroy all resources and cannot be undone.

## Next Steps

1. **Deploy GAELP Applications**: Use the provided Kubernetes manifests as templates
2. **Configure CI/CD**: Set up Cloud Build triggers for your repositories
3. **Set Up Monitoring**: Configure custom metrics and dashboards for your specific use cases
4. **Security Review**: Review and adjust IAM policies based on your security requirements
5. **Cost Optimization**: Monitor usage and adjust resource allocation as needed

## Support

For issues or questions:
1. Check GCP Console for resource status
2. Review Terraform output for configuration details
3. Check Cloud Logging for application and infrastructure logs
4. Use Cloud Monitoring for performance metrics

## Important Configuration Notes

Before deploying to production:
1. Update `terraform.tfvars` with your project-specific values
2. Configure proper email addresses for monitoring alerts
3. Set up your billing account ID for budget alerts
4. Review and adjust resource quotas and limits
5. Configure backup and disaster recovery procedures