#!/bin/bash
set -e

# GAELP Infrastructure Teardown Script
# This script tears down the GCP infrastructure for the GAELP platform

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

# Set up variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")/terraform"
PROJECT_ID=""
FORCE_DESTROY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_DESTROY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -p, --project      GCP Project ID (required)"
            echo "  -f, --force        Force destroy without confirmation"
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

print_message $YELLOW "=== GAELP Infrastructure Teardown ==="
print_message $YELLOW "Project ID: $PROJECT_ID"

# Warning message
print_message $RED "WARNING: This will destroy ALL GAELP infrastructure in project $PROJECT_ID"
print_message $RED "This action cannot be undone!"

if [[ "$FORCE_DESTROY" != true ]]; then
    print_message $YELLOW "Are you sure you want to continue? Type 'yes' to confirm:"
    read -r response
    if [[ "$response" != "yes" ]]; then
        print_message $YELLOW "Teardown cancelled."
        exit 0
    fi
fi

# Set the GCP project
print_message $BLUE "Setting GCP project..."
gcloud config set project "$PROJECT_ID"

# Change to Terraform directory
cd "$TERRAFORM_DIR"

# Check if Terraform state exists
if [[ ! -f "terraform.tfstate" ]]; then
    print_message $YELLOW "No Terraform state found. Checking for remote state..."
    terraform init
fi

# Plan the destruction
print_message $BLUE "Planning Terraform destruction..."
terraform plan -destroy

if [[ "$FORCE_DESTROY" != true ]]; then
    print_message $YELLOW "Do you want to proceed with destroying the infrastructure? (y/N)"
    read -r response
    if [[ "$response" != "y" && "$response" != "Y" ]]; then
        print_message $YELLOW "Teardown cancelled."
        exit 0
    fi
fi

# Destroy the Terraform-managed infrastructure
print_message $BLUE "Destroying Terraform-managed infrastructure..."
terraform destroy -auto-approve

print_message $GREEN "Terraform resources destroyed successfully!"

# Clean up any remaining resources that might not be managed by Terraform
print_message $BLUE "Cleaning up remaining resources..."

# Delete any remaining GKE clusters
print_message $BLUE "Checking for remaining GKE clusters..."
CLUSTERS=$(gcloud container clusters list --format="value(name,zone)" --filter="name:gaelp-*" 2>/dev/null || true)
if [[ -n "$CLUSTERS" ]]; then
    while IFS=$'\t' read -r cluster_name cluster_zone; do
        print_message $YELLOW "Deleting remaining GKE cluster: $cluster_name in $cluster_zone"
        gcloud container clusters delete "$cluster_name" --zone="$cluster_zone" --quiet || true
    done <<< "$CLUSTERS"
fi

# Delete any remaining Cloud Run services
print_message $BLUE "Checking for remaining Cloud Run services..."
SERVICES=$(gcloud run services list --format="value(metadata.name,metadata.namespace)" --filter="metadata.name:gaelp-*" 2>/dev/null || true)
if [[ -n "$SERVICES" ]]; then
    while IFS=$'\t' read -r service_name service_region; do
        print_message $YELLOW "Deleting remaining Cloud Run service: $service_name in $service_region"
        gcloud run services delete "$service_name" --region="$service_region" --quiet || true
    done <<< "$SERVICES"
fi

# Delete any remaining storage buckets
print_message $BLUE "Checking for remaining storage buckets..."
BUCKETS=$(gsutil ls -p "$PROJECT_ID" 2>/dev/null | grep "gs://gaelp-" | sed 's|gs://||' | sed 's|/$||' || true)
if [[ -n "$BUCKETS" ]]; then
    for bucket in $BUCKETS; do
        print_message $YELLOW "Deleting remaining storage bucket: $bucket"
        gsutil -m rm -r "gs://$bucket" 2>/dev/null || true
    done
fi

# Delete any remaining BigQuery datasets
print_message $BLUE "Checking for remaining BigQuery datasets..."
DATASETS=$(bq ls --format=csv --max_results=1000 | grep "gaelp" | cut -d',' -f1 || true)
if [[ -n "$DATASETS" ]]; then
    for dataset in $DATASETS; do
        print_message $YELLOW "Deleting remaining BigQuery dataset: $dataset"
        bq rm -r -f "$dataset" || true
    done
fi

# Delete any remaining Compute Engine instances
print_message $BLUE "Checking for remaining Compute Engine instances..."
INSTANCES=$(gcloud compute instances list --format="value(name,zone)" --filter="name:gaelp-*" 2>/dev/null || true)
if [[ -n "$INSTANCES" ]]; then
    while IFS=$'\t' read -r instance_name instance_zone; do
        print_message $YELLOW "Deleting remaining Compute Engine instance: $instance_name in $instance_zone"
        gcloud compute instances delete "$instance_name" --zone="$instance_zone" --quiet || true
    done <<< "$INSTANCES"
fi

# Delete any remaining VPC networks
print_message $BLUE "Checking for remaining VPC networks..."
NETWORKS=$(gcloud compute networks list --format="value(name)" --filter="name:gaelp-*" 2>/dev/null || true)
if [[ -n "$NETWORKS" ]]; then
    for network in $NETWORKS; do
        # Delete firewall rules first
        FIREWALL_RULES=$(gcloud compute firewall-rules list --format="value(name)" --filter="network:$network" 2>/dev/null || true)
        if [[ -n "$FIREWALL_RULES" ]]; then
            for rule in $FIREWALL_RULES; do
                print_message $YELLOW "Deleting firewall rule: $rule"
                gcloud compute firewall-rules delete "$rule" --quiet || true
            done
        fi
        
        # Delete subnets
        SUBNETS=$(gcloud compute networks subnets list --format="value(name,region)" --filter="network:$network" 2>/dev/null || true)
        if [[ -n "$SUBNETS" ]]; then
            while IFS=$'\t' read -r subnet_name subnet_region; do
                print_message $YELLOW "Deleting subnet: $subnet_name in $subnet_region"
                gcloud compute networks subnets delete "$subnet_name" --region="$subnet_region" --quiet || true
            done <<< "$SUBNETS"
        fi
        
        # Delete the network
        print_message $YELLOW "Deleting VPC network: $network"
        gcloud compute networks delete "$network" --quiet || true
    done
fi

# Delete any remaining service accounts
print_message $BLUE "Checking for remaining service accounts..."
SERVICE_ACCOUNTS=$(gcloud iam service-accounts list --format="value(email)" --filter="email:gaelp-*" 2>/dev/null || true)
if [[ -n "$SERVICE_ACCOUNTS" ]]; then
    for sa in $SERVICE_ACCOUNTS; do
        print_message $YELLOW "Deleting service account: $sa"
        gcloud iam service-accounts delete "$sa" --quiet || true
    done
fi

# Clean up Terraform state
print_message $BLUE "Cleaning up Terraform state..."
rm -f terraform.tfstate terraform.tfstate.backup

print_message $GREEN "=== GAELP Infrastructure Teardown Complete! ==="
print_message $BLUE "All GAELP resources have been removed from project: $PROJECT_ID"
print_message $YELLOW "Please verify in the GCP Console that all resources have been properly cleaned up."

# Final verification message
print_message $YELLOW "Manual cleanup verification checklist:"
echo "1. Check GCP Console for any remaining resources"
echo "2. Verify all storage buckets are deleted"
echo "3. Check for any remaining IAM roles or policies"
echo "4. Verify BigQuery datasets are removed"
echo "5. Check monitoring dashboards and alert policies"
echo "6. Review any remaining KMS keys"
echo "7. Check billing for any unexpected charges"

print_message $GREEN "Teardown completed successfully!"