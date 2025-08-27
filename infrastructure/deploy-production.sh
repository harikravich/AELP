#!/bin/bash
# GAELP Production Infrastructure Deployment Script

set -e

echo "🚀 Starting GAELP Production Infrastructure Deployment"
echo "Project: aura-thrive-platform"
echo "Region: us-central1"
echo ""

# Check if we're authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login' first"
    exit 1
fi

echo "✅ Authenticated with gcloud"

# Create production configuration for training orchestrator
echo "📝 Creating production configuration for training orchestrator..."

# Create production environment file
cat > /home/hariravichandran/AELP/training_orchestrator/.env.production << EOF
# GAELP Production Environment Configuration
ENVIRONMENT=production
PROJECT_ID=aura-thrive-platform
REGION=us-central1

# BigQuery Configuration
BIGQUERY_PROJECT=aura-thrive-platform
BIGQUERY_DATASET_CAMPAIGN=campaign_performance
BIGQUERY_DATASET_TRAINING=agent_training
BIGQUERY_DATASET_SIMULATION=simulation_results
BIGQUERY_DATASET_REALTIME=realtime_metrics

# Redis Configuration (will be updated after deployment)
REDIS_CACHE_HOST=localhost
REDIS_CACHE_PORT=6379
REDIS_SESSIONS_HOST=localhost
REDIS_SESSIONS_PORT=6379

# Pub/Sub Configuration
PUBSUB_PROJECT_ID=aura-thrive-platform
PUBSUB_TOPIC_TRAINING_EVENTS=gaelp-training-events
PUBSUB_TOPIC_SAFETY_ALERTS=gaelp-safety-alerts
PUBSUB_TOPIC_CAMPAIGN_EVENTS=gaelp-campaign-events

# Budget and Safety Configuration
REAL_TESTING_BUDGET_LIMIT=100.0
MAX_DAILY_BUDGET=5000.0
REQUIRE_HUMAN_APPROVAL=true

# Monitoring Configuration
LOG_LEVEL=INFO
RANDOM_SEED=42
EOF

echo "✅ Production environment configuration created"

# Check if essential APIs are enabled and suggest manual enablement if needed
echo "🔍 Checking required APIs..."

REQUIRED_APIS=(
    "container.googleapis.com"
    "compute.googleapis.com"
    "bigquery.googleapis.com"
    "storage.googleapis.com"
    "redis.googleapis.com"
    "pubsub.googleapis.com"
    "run.googleapis.com"
    "monitoring.googleapis.com"
    "logging.googleapis.com"
)

MISSING_APIS=()

for api in "${REQUIRED_APIS[@]}"; do
    if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        MISSING_APIS+=("$api")
    fi
done

if [ ${#MISSING_APIS[@]} -gt 0 ]; then
    echo "⚠️  The following APIs need to be enabled manually in the GCP Console:"
    for api in "${MISSING_APIS[@]}"; do
        echo "   - $api"
    done
    echo ""
    echo "Please enable these APIs at: https://console.cloud.google.com/apis/library"
    echo "Then re-run this script."
    exit 1
fi

echo "✅ All required APIs are enabled"

# Deploy core infrastructure using gcloud commands instead of Terraform
echo "🏗️  Deploying core infrastructure..."

# Create VPC Network
echo "Creating VPC network..."
if ! gcloud compute networks describe gaelp-vpc --format="value(name)" 2>/dev/null; then
    gcloud compute networks create gaelp-vpc \
        --subnet-mode=custom \
        --description="GAELP main VPC network"
    echo "✅ VPC network created"
else
    echo "✅ VPC network already exists"
fi

# Create Subnet
echo "Creating subnet..."
if ! gcloud compute networks subnets describe gaelp-subnet --region=us-central1 --format="value(name)" 2>/dev/null; then
    gcloud compute networks subnets create gaelp-subnet \
        --network=gaelp-vpc \
        --range=10.0.0.0/16 \
        --region=us-central1 \
        --secondary-range=gke-pods=10.1.0.0/16,gke-services=10.2.0.0/16 \
        --enable-private-ip-google-access
    echo "✅ Subnet created"
else
    echo "✅ Subnet already exists"
fi

# Create BigQuery datasets
echo "Creating BigQuery datasets..."
DATASETS=("campaign_performance" "agent_training" "simulation_results" "realtime_metrics")

for dataset in "${DATASETS[@]}"; do
    if ! bq ls -d "${dataset}" 2>/dev/null; then
        bq mk --dataset \
            --description="GAELP ${dataset} dataset" \
            --location=us-central1 \
            "${PROJECT_ID}:${dataset}"
        echo "✅ BigQuery dataset ${dataset} created"
    else
        echo "✅ BigQuery dataset ${dataset} already exists"
    fi
done

# Create Pub/Sub topics
echo "Creating Pub/Sub topics..."
TOPICS=("gaelp-training-events" "gaelp-safety-alerts" "gaelp-campaign-events" "gaelp-agent-state-changes" "gaelp-model-updates")

for topic in "${TOPICS[@]}"; do
    if ! gcloud pubsub topics describe "${topic}" --format="value(name)" 2>/dev/null; then
        gcloud pubsub topics create "${topic}"
        echo "✅ Pub/Sub topic ${topic} created"
    else
        echo "✅ Pub/Sub topic ${topic} already exists"
    fi
done

# Create Pub/Sub subscriptions
echo "Creating Pub/Sub subscriptions..."
SUBSCRIPTIONS=(
    "gaelp-training-events-processor:gaelp-training-events"
    "gaelp-safety-alerts-monitor:gaelp-safety-alerts" 
    "gaelp-campaign-events-analyzer:gaelp-campaign-events"
)

for sub_topic in "${SUBSCRIPTIONS[@]}"; do
    IFS=':' read -r subscription topic <<< "$sub_topic"
    if ! gcloud pubsub subscriptions describe "${subscription}" --format="value(name)" 2>/dev/null; then
        gcloud pubsub subscriptions create "${subscription}" \
            --topic="${topic}" \
            --ack-deadline=60
        echo "✅ Pub/Sub subscription ${subscription} created"
    else
        echo "✅ Pub/Sub subscription ${subscription} already exists"
    fi
done

# Create Cloud Storage buckets
echo "Creating Cloud Storage buckets..."
BUCKET_SUFFIX=$(date +%s | tail -c 9)
BUCKETS=(
    "gaelp-model-artifacts-${BUCKET_SUFFIX}"
    "gaelp-training-data-${BUCKET_SUFFIX}"
    "gaelp-campaign-assets-${BUCKET_SUFFIX}"
    "gaelp-simulation-results-${BUCKET_SUFFIX}"
    "gaelp-temp-processing-${BUCKET_SUFFIX}"
)

for bucket in "${BUCKETS[@]}"; do
    if ! gsutil ls -b "gs://${bucket}" 2>/dev/null; then
        gsutil mb -l us-central1 "gs://${bucket}"
        echo "✅ Storage bucket ${bucket} created"
    else
        echo "✅ Storage bucket ${bucket} already exists"
    fi
done

# Update environment file with actual bucket names
cat >> /home/hariravichandran/AELP/training_orchestrator/.env.production << EOF

# Storage Configuration
GCS_BUCKET_MODEL_ARTIFACTS=gaelp-model-artifacts-${BUCKET_SUFFIX}
GCS_BUCKET_TRAINING_DATA=gaelp-training-data-${BUCKET_SUFFIX}
GCS_BUCKET_CAMPAIGN_ASSETS=gaelp-campaign-assets-${BUCKET_SUFFIX}
GCS_BUCKET_SIMULATION_RESULTS=gaelp-simulation-results-${BUCKET_SUFFIX}
GCS_BUCKET_TEMP_PROCESSING=gaelp-temp-processing-${BUCKET_SUFFIX}
EOF

echo "✅ Updated configuration with bucket names"

# Create Redis instances would require more complex setup, suggest manual creation
echo "📋 Next steps for complete deployment:"
echo ""
echo "1. 🔴 Create Redis instances manually in GCP Console:"
echo "   - gaelp-redis-cache (Basic tier, 1GB, us-central1)"
echo "   - gaelp-redis-sessions (Basic tier, 1GB, us-central1)"
echo ""
echo "2. 🔴 Create GKE cluster manually:"
echo "   gcloud container clusters create gaelp-cluster \\"
echo "     --region us-central1 \\"
echo "     --network gaelp-vpc \\"
echo "     --subnetwork gaelp-subnet \\"
echo "     --enable-ip-alias \\"
echo "     --cluster-secondary-range-name gke-pods \\"
echo "     --services-secondary-range-name gke-services \\"
echo "     --enable-autoscaling \\"
echo "     --min-nodes 0 --max-nodes 10 \\"
echo "     --machine-type e2-standard-4 \\"
echo "     --disk-size 50GB \\"
echo "     --enable-autorepair \\"
echo "     --enable-autoupgrade"
echo ""
echo "3. 🟢 Update Redis connection details in:"
echo "   /home/hariravichandran/AELP/training_orchestrator/.env.production"
echo ""
echo "4. 🟢 Deploy training orchestrator with production config:"
echo "   cd /home/hariravichandran/AELP/training_orchestrator"
echo "   python -m training_orchestrator.cli --config production"
echo ""

echo "🎉 Core GAELP infrastructure deployment completed!"
echo "📁 Production configuration saved to: /home/hariravichandran/AELP/training_orchestrator/.env.production"