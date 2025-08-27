# GAELP GCP Infrastructure - Main Configuration
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.84"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# Providers
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Locals
locals {
  common_labels = {
    project     = "gaelp"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Data sources
data "google_client_config" "default" {}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",
    "compute.googleapis.com",
    "cloudbuild.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "run.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudscheduler.googleapis.com",
    "cloudfunctions.googleapis.com",
    "redis.googleapis.com",
    "pubsub.googleapis.com",
    "servicenetworking.googleapis.com",
    "cloudkms.googleapis.com",
    "secretmanager.googleapis.com"
  ])

  service                    = each.value
  disable_dependent_services = false
  disable_on_destroy        = false
}

# Import modules
module "networking" {
  source = "./modules/networking"
  
  project_id = var.project_id
  region     = var.region
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

module "gke" {
  source = "./modules/gke"
  
  project_id     = var.project_id
  region         = var.region
  zone           = var.zone
  network_name   = module.networking.network_name
  subnet_name    = module.networking.subnet_name
  labels         = local.common_labels
  
  depends_on = [module.networking]
}

module "cloud_run" {
  source = "./modules/cloud_run"
  
  project_id = var.project_id
  region     = var.region
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

module "bigquery" {
  source = "./modules/bigquery"
  
  project_id = var.project_id
  region     = var.region
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

module "storage" {
  source = "./modules/storage"
  
  project_id = var.project_id
  region     = var.region
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

module "iam" {
  source = "./modules/iam"
  
  project_id = var.project_id
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

module "monitoring" {
  source = "./modules/monitoring"
  
  project_id = var.project_id
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

module "vertex_ai" {
  source = "./modules/vertex_ai"
  
  project_id = var.project_id
  region     = var.region
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

module "redis" {
  source = "./modules/redis"
  
  project_id   = var.project_id
  region       = var.region
  network_name = module.networking.network_name
  labels       = local.common_labels
  
  depends_on = [module.networking, google_project_service.required_apis]
}

module "pubsub" {
  source = "./modules/pubsub"
  
  project_id = var.project_id
  labels     = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

# Outputs
output "gke_cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = module.gke.cluster_name
}

output "cloud_run_urls" {
  description = "Cloud Run service URLs"
  value       = module.cloud_run.service_urls
}

output "bigquery_dataset_ids" {
  description = "BigQuery dataset IDs"
  value       = module.bigquery.dataset_ids
}

output "storage_bucket_names" {
  description = "Cloud Storage bucket names"
  value       = module.storage.bucket_names
}

output "monitoring_dashboard_url" {
  description = "Monitoring dashboard URL"
  value       = module.monitoring.dashboard_url
}

output "redis_instances" {
  description = "Redis instance details"
  value       = module.redis.redis_instances
  sensitive   = true
}

output "pubsub_topics" {
  description = "Pub/Sub topic names"
  value       = module.pubsub.topic_names
}

output "iam_service_accounts" {
  description = "IAM service account details"
  value       = module.iam.service_accounts
}