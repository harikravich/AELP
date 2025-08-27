#!/bin/bash

# =============================================================================
# GAELP BigQuery Infrastructure Deployment Script
# =============================================================================

set -e

# Configuration
PROJECT_ID=${GAELP_PROJECT_ID:-"gaelp-production"}
DATASET_ID="gaelp_campaigns"
REGION=${GAELP_REGION:-"us-central1"}
SERVICE_ACCOUNT=${GAELP_SERVICE_ACCOUNT:-"gaelp-bigquery@${PROJECT_ID}.iam.gserviceaccount.com"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if gcloud is installed and authenticated
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        error "No active gcloud authentication found. Please run 'gcloud auth login'"
        exit 1
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    success "Prerequisites check completed"
}

# Enable required APIs
enable_apis() {
    log "Enabling required Google Cloud APIs..."
    
    apis=(
        "bigquery.googleapis.com"
        "dataflow.googleapis.com"
        "cloudscheduler.googleapis.com"
        "cloudfunctions.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log "Enabling $api..."
        gcloud services enable $api --quiet
    done
    
    success "All required APIs enabled"
}

# Create BigQuery dataset
create_dataset() {
    log "Creating BigQuery dataset: $DATASET_ID"
    
    # Check if dataset exists
    if bq ls -d $PROJECT_ID:$DATASET_ID &> /dev/null; then
        warning "Dataset $DATASET_ID already exists, skipping creation"
        return 0
    fi
    
    # Create dataset
    bq mk \
        --dataset \
        --description="GAELP Ad Campaign Learning Platform - Campaign data, performance metrics, and agent training data" \
        --location=$REGION \
        --default_table_expiration=94608000 \
        --default_partition_expiration=94608000 \
        $PROJECT_ID:$DATASET_ID
    
    success "Dataset $DATASET_ID created successfully"
}

# Deploy schema files
deploy_schemas() {
    log "Deploying BigQuery schemas..."
    
    schema_files=(
        "schemas/01_core_campaigns.sql"
        "schemas/02_cost_optimization.sql"
    )
    
    for file in "${schema_files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Executing schema file: $file"
            
            # Replace placeholder dataset references
            sed "s/gaelp_campaigns/$PROJECT_ID:$DATASET_ID/g" "$file" > "/tmp/$(basename $file)"
            
            # Execute the SQL file
            bq query \
                --use_legacy_sql=false \
                --max_rows=0 \
                --quiet \
                < "/tmp/$(basename $file)"
            
            success "Schema file $file deployed successfully"
        else
            error "Schema file $file not found"
        fi
    done
}

# Deploy views
deploy_views() {
    log "Deploying BigQuery views..."
    
    view_files=(
        "views/campaign_analytics.sql"
    )
    
    for file in "${view_files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Executing view file: $file"
            
            # Replace placeholder dataset references
            sed "s/gaelp_campaigns/$PROJECT_ID:$DATASET_ID/g" "$file" > "/tmp/$(basename $file)"
            
            # Execute the SQL file
            bq query \
                --use_legacy_sql=false \
                --max_rows=0 \
                --quiet \
                < "/tmp/$(basename $file)"
            
            success "View file $file deployed successfully"
        else
            error "View file $file not found"
        fi
    done
}

# Deploy functions and procedures
deploy_functions() {
    log "Deploying BigQuery functions and procedures..."
    
    function_files=(
        "pipelines/streaming_ingestion.sql"
        "functions/data_exports.sql"
        "functions/monitoring_alerts.sql"
    )
    
    for file in "${function_files[@]}"; do
        if [[ -f "$file" ]]; then
            log "Executing function file: $file"
            
            # Replace placeholder dataset references
            sed "s/gaelp_campaigns/$PROJECT_ID:$DATASET_ID/g" "$file" > "/tmp/$(basename $file)"
            
            # Execute the SQL file
            bq query \
                --use_legacy_sql=false \
                --max_rows=0 \
                --quiet \
                < "/tmp/$(basename $file)"
            
            success "Function file $file deployed successfully"
        else
            error "Function file $file not found"
        fi
    done
}

# Set up IAM permissions
setup_iam() {
    log "Setting up IAM permissions..."
    
    # Create service account if it doesn't exist
    if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT &> /dev/null; then
        log "Creating service account: $SERVICE_ACCOUNT"
        gcloud iam service-accounts create gaelp-bigquery \
            --display-name="GAELP BigQuery Service Account" \
            --description="Service account for GAELP BigQuery operations"
    fi
    
    # Grant necessary permissions
    roles=(
        "roles/bigquery.admin"
        "roles/bigquery.dataEditor"
        "roles/bigquery.jobUser"
        "roles/storage.objectAdmin"
        "roles/dataflow.admin"
        "roles/cloudscheduler.admin"
    )
    
    for role in "${roles[@]}"; do
        log "Granting role $role to $SERVICE_ACCOUNT"
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$SERVICE_ACCOUNT" \
            --role="$role" \
            --quiet
    done
    
    success "IAM permissions configured"
}

# Create Cloud Storage buckets
create_storage_buckets() {
    log "Creating Cloud Storage buckets..."
    
    buckets=(
        "gaelp-training-data-${PROJECT_ID}"
        "gaelp-config-${PROJECT_ID}"
        "gaelp-backups-${PROJECT_ID}"
        "gaelp-exports-${PROJECT_ID}"
    )
    
    for bucket in "${buckets[@]}"; do
        if ! gsutil ls -b gs://$bucket &> /dev/null; then
            log "Creating bucket: gs://$bucket"
            gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$bucket
            
            # Set lifecycle policy for training data bucket
            if [[ $bucket == *"training-data"* ]]; then
                cat > /tmp/lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "NEARLINE"
        },
        "condition": {
          "age": 30
        }
      },
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "COLDLINE"
        },
        "condition": {
          "age": 90
        }
      },
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 365
        }
      }
    ]
  }
}
EOF
                gsutil lifecycle set /tmp/lifecycle.json gs://$bucket
            fi
            
            success "Bucket gs://$bucket created"
        else
            warning "Bucket gs://$bucket already exists"
        fi
    done
}

# Set up monitoring and alerting
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create log-based metrics
    log "Creating log-based metrics..."
    
    # BigQuery slot usage metric
    gcloud logging metrics create gaelp_bigquery_slots \
        --description="BigQuery slot usage for GAELP" \
        --log-filter='resource.type="bigquery_project" AND protoPayload.methodName="jobservice.jobcompleted"' \
        --value-extractor='EXTRACT(protoPayload.serviceData.jobCompletedEvent.job.jobStatistics.totalSlotMs)' \
        --quiet || warning "Log metric gaelp_bigquery_slots may already exist"
    
    # BigQuery cost metric
    gcloud logging metrics create gaelp_bigquery_cost \
        --description="BigQuery cost estimation for GAELP" \
        --log-filter='resource.type="bigquery_project" AND protoPayload.methodName="jobservice.jobcompleted"' \
        --value-extractor='EXTRACT(protoPayload.serviceData.jobCompletedEvent.job.jobStatistics.totalBilledBytes)' \
        --quiet || warning "Log metric gaelp_bigquery_cost may already exist"
    
    success "Monitoring setup completed"
}

# Create scheduled jobs
setup_scheduled_jobs() {
    log "Setting up scheduled jobs..."
    
    # Data lifecycle management job
    gcloud scheduler jobs create http gaelp-data-lifecycle \
        --location=$REGION \
        --schedule="0 2 * * *" \
        --uri="https://cloudfunctions.net/gaelp-data-lifecycle" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"operation": "lifecycle_management"}' \
        --quiet || warning "Scheduled job gaelp-data-lifecycle may already exist"
    
    # Alert processing job
    gcloud scheduler jobs create http gaelp-alert-processing \
        --location=$REGION \
        --schedule="*/5 * * * *" \
        --uri="https://cloudfunctions.net/gaelp-alert-processor" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"operation": "process_alerts"}' \
        --quiet || warning "Scheduled job gaelp-alert-processing may already exist"
    
    # Cost monitoring job
    gcloud scheduler jobs create http gaelp-cost-monitoring \
        --location=$REGION \
        --schedule="0 */6 * * *" \
        --uri="https://cloudfunctions.net/gaelp-cost-monitor" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"operation": "check_budgets"}' \
        --quiet || warning "Scheduled job gaelp-cost-monitoring may already exist"
    
    success "Scheduled jobs created"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    # Check if tables exist
    tables=(
        "campaigns"
        "performance_metrics"
        "agent_episodes"
        "simulation_data"
        "personas"
        "safety_events"
    )
    
    for table in "${tables[@]}"; do
        if bq ls $PROJECT_ID:$DATASET_ID | grep -q $table; then
            success "Table $table exists"
        else
            error "Table $table is missing"
        fi
    done
    
    # Check if views exist
    views=(
        "campaign_performance_summary"
        "agent_performance_comparison"
        "simulation_vs_real_performance"
        "safety_monitoring_dashboard"
        "learning_progress"
    )
    
    for view in "${views[@]}"; do
        if bq ls $PROJECT_ID:$DATASET_ID | grep -q $view; then
            success "View $view exists"
        else
            warning "View $view may be missing"
        fi
    done
    
    success "Deployment validation completed"
}

# Generate configuration file
generate_config() {
    log "Generating configuration file..."
    
    cat > gaelp_bigquery_config.json << EOF
{
  "project_id": "$PROJECT_ID",
  "dataset_id": "$DATASET_ID",
  "region": "$REGION",
  "service_account": "$SERVICE_ACCOUNT",
  "buckets": {
    "training_data": "gaelp-training-data-$PROJECT_ID",
    "config": "gaelp-config-$PROJECT_ID",
    "backups": "gaelp-backups-$PROJECT_ID",
    "exports": "gaelp-exports-$PROJECT_ID"
  },
  "monitoring": {
    "enabled": true,
    "alert_frequency": "5_minutes",
    "cost_budget_alerts": true
  },
  "data_lifecycle": {
    "archive_after_days": 365,
    "delete_simulation_after_days": 180,
    "compress_old_data": true
  },
  "deployment_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "version": "1.0.0"
}
EOF
    
    success "Configuration file generated: gaelp_bigquery_config.json"
}

# Main deployment function
main() {
    log "Starting GAELP BigQuery infrastructure deployment..."
    
    # Change to the directory containing the script
    cd "$(dirname "$0")"
    
    check_prerequisites
    enable_apis
    create_dataset
    setup_iam
    create_storage_buckets
    deploy_schemas
    deploy_views
    deploy_functions
    setup_monitoring
    setup_scheduled_jobs
    validate_deployment
    generate_config
    
    success "GAELP BigQuery infrastructure deployment completed successfully!"
    log "Configuration saved to: gaelp_bigquery_config.json"
    log "Dataset available at: https://console.cloud.google.com/bigquery?project=$PROJECT_ID&ws=!1m5!1m4!4m3!1s$PROJECT_ID!2s$DATASET_ID"
}

# Help function
show_help() {
    cat << EOF
GAELP BigQuery Infrastructure Deployment

Usage: $0 [COMMAND]

Commands:
  deploy          Full deployment (default)
  schemas         Deploy only schemas
  views           Deploy only views
  functions       Deploy only functions
  monitoring      Setup only monitoring
  validate        Validate existing deployment
  clean           Clean up resources (DESTRUCTIVE)
  help            Show this help

Environment Variables:
  GAELP_PROJECT_ID      Google Cloud Project ID (default: gaelp-production)
  GAELP_REGION          Deployment region (default: us-central1)
  GAELP_SERVICE_ACCOUNT Service account email

Examples:
  $0                    # Full deployment
  $0 schemas            # Deploy only schemas
  $0 validate           # Validate deployment
  
  GAELP_PROJECT_ID=my-project $0  # Deploy to specific project
EOF
}

# Clean up function (destructive)
clean_deployment() {
    warning "This will delete ALL GAELP BigQuery resources. This action is irreversible!"
    read -p "Are you sure you want to continue? (type 'DELETE' to confirm): " confirmation
    
    if [[ "$confirmation" != "DELETE" ]]; then
        log "Cleanup cancelled"
        exit 0
    fi
    
    log "Cleaning up GAELP BigQuery resources..."
    
    # Delete dataset (this will delete all tables, views, functions)
    bq rm -r -f $PROJECT_ID:$DATASET_ID || warning "Dataset deletion failed"
    
    # Delete Cloud Storage buckets
    buckets=(
        "gaelp-training-data-${PROJECT_ID}"
        "gaelp-config-${PROJECT_ID}"
        "gaelp-backups-${PROJECT_ID}"
        "gaelp-exports-${PROJECT_ID}"
    )
    
    for bucket in "${buckets[@]}"; do
        gsutil -m rm -r gs://$bucket || warning "Bucket gs://$bucket deletion failed"
    done
    
    # Delete scheduled jobs
    gcloud scheduler jobs delete gaelp-data-lifecycle --location=$REGION --quiet || warning "Job deletion failed"
    gcloud scheduler jobs delete gaelp-alert-processing --location=$REGION --quiet || warning "Job deletion failed"
    gcloud scheduler jobs delete gaelp-cost-monitoring --location=$REGION --quiet || warning "Job deletion failed"
    
    success "Cleanup completed"
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    schemas)
        check_prerequisites
        create_dataset
        deploy_schemas
        ;;
    views)
        check_prerequisites
        deploy_views
        ;;
    functions)
        check_prerequisites
        deploy_functions
        ;;
    monitoring)
        check_prerequisites
        setup_monitoring
        ;;
    validate)
        check_prerequisites
        validate_deployment
        ;;
    clean)
        check_prerequisites
        clean_deployment
        ;;
    help)
        show_help
        ;;
    *)
        error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac