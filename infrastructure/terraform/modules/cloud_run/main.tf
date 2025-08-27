# Cloud Run Module for GAELP
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

# Service account for Cloud Run services
resource "google_service_account" "cloud_run_sa" {
  account_id   = "gaelp-cloud-run"
  display_name = "GAELP Cloud Run Service Account"
  description  = "Service account for GAELP Cloud Run services"
}

# IAM bindings for Cloud Run service account
resource "google_project_iam_member" "cloud_run_bindings" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectViewer",
    "roles/storage.objectCreator",
    "roles/cloudsql.client",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Environment Registry API Service
resource "google_cloud_run_service" "environment_registry" {
  name     = "environment-registry"
  location = var.region

  template {
    metadata {
      labels = var.labels
      annotations = {
        "autoscaling.knative.dev/maxScale"         = "100"
        "autoscaling.knative.dev/minScale"         = "0"
        "run.googleapis.com/cpu-throttling"        = "false"
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }

    spec {
      container_concurrency = 80
      timeout_seconds      = 300
      service_account_name = google_service_account.cloud_run_sa.email

      containers {
        image = "gcr.io/cloudrun/placeholder"

        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }

        ports {
          container_port = 8080
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "ENVIRONMENT"
          value = "production"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image
    ]
  }
}

# Agent Management API Service
resource "google_cloud_run_service" "agent_management" {
  name     = "agent-management"
  location = var.region

  template {
    metadata {
      labels = var.labels
      annotations = {
        "autoscaling.knative.dev/maxScale"         = "50"
        "autoscaling.knative.dev/minScale"         = "0"
        "run.googleapis.com/cpu-throttling"        = "false"
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }

    spec {
      container_concurrency = 80
      timeout_seconds      = 300
      service_account_name = google_service_account.cloud_run_sa.email

      containers {
        image = "gcr.io/cloudrun/placeholder"

        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }

        ports {
          container_port = 8080
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "ENVIRONMENT"
          value = "production"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image
    ]
  }
}

# Campaign Monitoring API Service
resource "google_cloud_run_service" "campaign_monitor" {
  name     = "campaign-monitor"
  location = var.region

  template {
    metadata {
      labels = var.labels
      annotations = {
        "autoscaling.knative.dev/maxScale"         = "200"
        "autoscaling.knative.dev/minScale"         = "1"
        "run.googleapis.com/cpu-throttling"        = "false"
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }

    spec {
      container_concurrency = 100
      timeout_seconds      = 60
      service_account_name = google_service_account.cloud_run_sa.email

      containers {
        image = "gcr.io/cloudrun/placeholder"

        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }

        ports {
          container_port = 8080
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "ENVIRONMENT"
          value = "production"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image
    ]
  }
}

# Metrics Collection Service
resource "google_cloud_run_service" "metrics_collector" {
  name     = "metrics-collector"
  location = var.region

  template {
    metadata {
      labels = var.labels
      annotations = {
        "autoscaling.knative.dev/maxScale"         = "50"
        "autoscaling.knative.dev/minScale"         = "1"
        "run.googleapis.com/cpu-throttling"        = "false"
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }

    spec {
      container_concurrency = 100
      timeout_seconds      = 300
      service_account_name = google_service_account.cloud_run_sa.email

      containers {
        image = "gcr.io/cloudrun/placeholder"

        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }

        ports {
          container_port = 8080
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "ENVIRONMENT"
          value = "production"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image
    ]
  }
}

# IAM policy for public access (adjust as needed)
data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}

# Apply public access policy to services that need it
resource "google_cloud_run_service_iam_policy" "environment_registry_noauth" {
  location = google_cloud_run_service.environment_registry.location
  project  = google_cloud_run_service.environment_registry.project
  service  = google_cloud_run_service.environment_registry.name

  policy_data = data.google_iam_policy.noauth.policy_data
}

# Outputs
output "service_urls" {
  description = "Cloud Run service URLs"
  value = {
    environment_registry = google_cloud_run_service.environment_registry.status[0].url
    agent_management     = google_cloud_run_service.agent_management.status[0].url
    campaign_monitor     = google_cloud_run_service.campaign_monitor.status[0].url
    metrics_collector    = google_cloud_run_service.metrics_collector.status[0].url
  }
}

output "service_account_email" {
  description = "Cloud Run service account email"
  value       = google_service_account.cloud_run_sa.email
}