# Vertex AI Module for GAELP (Simplified)
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

# KMS Key Ring for Vertex AI
resource "google_kms_key_ring" "vertex_ai_keyring" {
  name     = "gaelp-vertex-ai-keyring"
  location = var.region
}

# KMS Crypto Key for Vertex AI
resource "google_kms_crypto_key" "vertex_ai_key" {
  name     = "gaelp-vertex-ai-key"
  key_ring = google_kms_key_ring.vertex_ai_keyring.id

  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }
}

# Vertex AI Custom Training Job (placeholder - will be used for agent training)
# Note: Actual training jobs will be created programmatically via API

# Service Account for Vertex AI workloads
resource "google_service_account" "vertex_ai_sa" {
  account_id   = "gaelp-vertex-ai"
  display_name = "GAELP Vertex AI Service Account"
  description  = "Service account for GAELP Vertex AI operations"
}

# IAM bindings for Vertex AI service account
resource "google_project_iam_member" "vertex_ai_bindings" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/storage.objectAdmin",
    "roles/bigquery.dataEditor",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.vertex_ai_sa.email}"
}

# Outputs
output "vertex_ai_service_account" {
  description = "Vertex AI service account email"
  value       = google_service_account.vertex_ai_sa.email
}

output "vertex_ai_kms_key" {
  description = "Vertex AI KMS key ID"
  value       = google_kms_crypto_key.vertex_ai_key.id
}