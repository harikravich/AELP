# Storage Module for GAELP
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "labels" {
  description = "Common labels"
  type        = map(string)
}

# Random suffix for bucket names to ensure uniqueness
resource "random_string" "bucket_suffix" {
  length  = 8
  upper   = false
  special = false
}

# Model Artifacts Bucket
resource "google_storage_bucket" "model_artifacts" {
  name          = "gaelp-model-artifacts-${random_string.bucket_suffix.result}"
  location      = var.region
  force_destroy = false

  labels = var.labels

  # Versioning for model checkpoints
  versioning {
    enabled = true
  }

  # Lifecycle management
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age                = 30
      with_state         = "ARCHIVED"
      matches_storage_class = ["NEARLINE"]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Security settings
  uniform_bucket_level_access = true

  # Encryption
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage_key.id
  }
}

# Training Data Bucket
resource "google_storage_bucket" "training_data" {
  name          = "gaelp-training-data-${random_string.bucket_suffix.result}"
  location      = var.region
  force_destroy = false

  labels = var.labels

  # Versioning
  versioning {
    enabled = true
  }

  # Lifecycle management
  lifecycle_rule {
    condition {
      age = 180
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  # Security settings
  uniform_bucket_level_access = true

  # Encryption
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage_key.id
  }
}

# Campaign Assets Bucket
resource "google_storage_bucket" "campaign_assets" {
  name          = "gaelp-campaign-assets-${random_string.bucket_suffix.result}"
  location      = var.region
  force_destroy = false

  labels = var.labels

  # Versioning
  versioning {
    enabled = true
  }

  # Lifecycle management
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Security settings
  uniform_bucket_level_access = true

  # Encryption
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage_key.id
  }
}

# Simulation Results Bucket
resource "google_storage_bucket" "simulation_results" {
  name          = "gaelp-simulation-results-${random_string.bucket_suffix.result}"
  location      = var.region
  force_destroy = false

  labels = var.labels

  # Versioning
  versioning {
    enabled = true
  }

  # Lifecycle management
  lifecycle_rule {
    condition {
      age = 60
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 14
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  # Security settings
  uniform_bucket_level_access = true

  # Encryption
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage_key.id
  }
}

# Temporary Processing Bucket
resource "google_storage_bucket" "temp_processing" {
  name          = "gaelp-temp-processing-${random_string.bucket_suffix.result}"
  location      = var.region
  force_destroy = true

  labels = var.labels

  # Lifecycle management for temporary files
  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Security settings
  uniform_bucket_level_access = true

  # Encryption
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage_key.id
  }
}

# KMS Key Ring
resource "google_kms_key_ring" "gaelp_keyring" {
  name     = "gaelp-keyring"
  location = var.region
}

# KMS Crypto Key for Storage Encryption
resource "google_kms_crypto_key" "storage_key" {
  name     = "gaelp-storage-key"
  key_ring = google_kms_key_ring.gaelp_keyring.id

  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }
}

# IAM binding for storage buckets - service accounts
resource "google_storage_bucket_iam_member" "model_artifacts_access" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.project_id}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "training_data_access" {
  bucket = google_storage_bucket.training_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.project_id}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "campaign_assets_access" {
  bucket = google_storage_bucket.campaign_assets.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.project_id}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "simulation_results_access" {
  bucket = google_storage_bucket.simulation_results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.project_id}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "temp_processing_access" {
  bucket = google_storage_bucket.temp_processing.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.project_id}-compute@developer.gserviceaccount.com"
}

# Bucket notifications for processing pipelines
resource "google_pubsub_topic" "storage_notifications" {
  name = "gaelp-storage-notifications"

  labels = var.labels
}

resource "google_storage_notification" "model_artifacts_notification" {
  bucket         = google_storage_bucket.model_artifacts.name
  payload_format = "JSON_API_V1"
  topic          = google_pubsub_topic.storage_notifications.id
  event_types    = ["OBJECT_FINALIZE", "OBJECT_DELETE"]

  depends_on = [google_pubsub_topic_iam_member.storage_publisher]
}

resource "google_pubsub_topic_iam_member" "storage_publisher" {
  topic  = google_pubsub_topic.storage_notifications.id
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:service-${data.google_project.project.number}@gs-project-accounts.iam.gserviceaccount.com"
}

# Project data source
data "google_project" "project" {
  project_id = var.project_id
}

# Outputs
output "bucket_names" {
  description = "Storage bucket names"
  value = {
    model_artifacts    = google_storage_bucket.model_artifacts.name
    training_data     = google_storage_bucket.training_data.name
    campaign_assets   = google_storage_bucket.campaign_assets.name
    simulation_results = google_storage_bucket.simulation_results.name
    temp_processing   = google_storage_bucket.temp_processing.name
  }
}

output "bucket_urls" {
  description = "Storage bucket URLs"
  value = {
    model_artifacts    = google_storage_bucket.model_artifacts.url
    training_data     = google_storage_bucket.training_data.url
    campaign_assets   = google_storage_bucket.campaign_assets.url
    simulation_results = google_storage_bucket.simulation_results.url
    temp_processing   = google_storage_bucket.temp_processing.url
  }
}

output "kms_key_id" {
  description = "KMS key ID for storage encryption"
  value       = google_kms_crypto_key.storage_key.id
}

output "storage_notification_topic" {
  description = "Pub/Sub topic for storage notifications"
  value       = google_pubsub_topic.storage_notifications.name
}