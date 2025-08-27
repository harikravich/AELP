#!/bin/bash
set -e

# GAELP Infrastructure Setup Script
# This script sets up the GCP infrastructure for the GAELP platform

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_message $BLUE "Checking prerequisites..."

if ! command_exists gcloud; then
    print_message $RED "Error: gcloud CLI is not installed. Please install it first."
    exit 1
fi

if ! command_exists terraform; then
    print_message $RED "Error: Terraform is not installed. Please install it first."
    exit 1
fi

if ! command_exists kubectl; then
    print_message $RED "Error: kubectl is not installed. Please install it first."
    exit 1
fi

print_message $GREEN "Prerequisites check passed!"

# Set up variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")/terraform"
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
ENVIRONMENT="dev"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -z|--zone)
            ZONE="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -p, --project      GCP Project ID (required)"
            echo "  -r, --region       GCP Region (default: us-central1)"
            echo "  -z, --zone         GCP Zone (default: us-central1-a)"
            echo "  -e, --environment  Environment (default: dev)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            print_message $RED "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if project ID is provided
if [[ -z "$PROJECT_ID" ]]; then
    print_message $RED "Error: Project ID is required. Use -p or --project to specify it."
    exit 1
fi

print_message $BLUE "Setting up GAELP infrastructure..."
print_message $YELLOW "Project ID: $PROJECT_ID"
print_message $YELLOW "Region: $REGION"
print_message $YELLOW "Zone: $ZONE"
print_message $YELLOW "Environment: $ENVIRONMENT"

# Set the GCP project
print_message $BLUE "Setting GCP project..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
print_message $BLUE "Enabling required GCP APIs..."
gcloud services enable \
    container.googleapis.com \
    compute.googleapis.com \
    cloudbuild.googleapis.com \
    bigquery.googleapis.com \
    storage.googleapis.com \
    run.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    cloudresourcemanager.googleapis.com \
    iam.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    cloudscheduler.googleapis.com \
    cloudfunctions.googleapis.com \
    notebooks.googleapis.com \
    cloudkms.googleapis.com

print_message $GREEN "APIs enabled successfully!"

# Initialize Terraform
print_message $BLUE "Initializing Terraform..."
cd "$TERRAFORM_DIR"

if [[ ! -f "terraform.tfvars" ]]; then
    print_message $YELLOW "Creating terraform.tfvars from example..."
    cp terraform.tfvars.example terraform.tfvars
    
    # Update the tfvars file with provided values
    sed -i "s/your-gaelp-project-id/$PROJECT_ID/g" terraform.tfvars
    sed -i "s/us-central1/$REGION/g" terraform.tfvars
    sed -i "s/us-central1-a/$ZONE/g" terraform.tfvars
    sed -i "s/dev/$ENVIRONMENT/g" terraform.tfvars
    
    print_message $YELLOW "Please review and update terraform.tfvars with your specific configuration."
fi

terraform init

# Plan the deployment
print_message $BLUE "Planning Terraform deployment..."
terraform plan \
    -var="project_id=$PROJECT_ID" \
    -var="region=$REGION" \
    -var="zone=$ZONE" \
    -var="environment=$ENVIRONMENT"

# Ask for confirmation
print_message $YELLOW "Do you want to apply this Terraform plan? (y/N)"
read -r response
if [[ "$response" != "y" && "$response" != "Y" ]]; then
    print_message $YELLOW "Deployment cancelled."
    exit 0
fi

# Apply the Terraform configuration
print_message $BLUE "Applying Terraform configuration..."
terraform apply \
    -var="project_id=$PROJECT_ID" \
    -var="region=$REGION" \
    -var="zone=$ZONE" \
    -var="environment=$ENVIRONMENT" \
    -auto-approve

print_message $GREEN "Terraform deployment completed!"

# Get GKE credentials
print_message $BLUE "Configuring kubectl for GKE cluster..."
CLUSTER_NAME=$(terraform output -raw gke_cluster_name)
gcloud container clusters get-credentials "$CLUSTER_NAME" --region="$REGION"

print_message $GREEN "kubectl configured successfully!"

# Create Kubernetes namespaces
print_message $BLUE "Creating Kubernetes namespaces..."
kubectl create namespace gaelp-training --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace gaelp-agents --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace gaelp-data --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace gaelp-monitoring --dry-run=client -o yaml | kubectl apply -f -

# Label namespaces
kubectl label namespace gaelp-training project=gaelp environment="$ENVIRONMENT" --overwrite
kubectl label namespace gaelp-agents project=gaelp environment="$ENVIRONMENT" --overwrite
kubectl label namespace gaelp-data project=gaelp environment="$ENVIRONMENT" --overwrite
kubectl label namespace gaelp-monitoring project=gaelp environment="$ENVIRONMENT" --overwrite

print_message $GREEN "Kubernetes namespaces created!"

# Install NVIDIA GPU drivers on GPU nodes (if any)
print_message $BLUE "Installing NVIDIA GPU drivers..."
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Output important information
print_message $GREEN "=== GAELP Infrastructure Setup Complete! ==="
print_message $BLUE "Cluster Information:"
echo "  GKE Cluster: $CLUSTER_NAME"
echo "  Region: $REGION"
echo "  Project: $PROJECT_ID"

print_message $BLUE "Service URLs:"
terraform output cloud_run_urls

print_message $BLUE "BigQuery Datasets:"
terraform output bigquery_dataset_ids

print_message $BLUE "Storage Buckets:"
terraform output storage_bucket_names

print_message $BLUE "Monitoring Dashboard:"
terraform output monitoring_dashboard_url

print_message $YELLOW "Next Steps:"
echo "1. Review and update monitoring alert email addresses in the Terraform configuration"
echo "2. Configure your ad platform API credentials in Secret Manager"
echo "3. Deploy your GAELP applications to the GKE cluster"
echo "4. Set up your CI/CD pipelines using Cloud Build"
echo "5. Configure budget alerts with your actual billing account"

print_message $GREEN "Setup completed successfully!"