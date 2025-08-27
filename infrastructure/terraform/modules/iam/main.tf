# IAM Module for GAELP
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "labels" {
  description = "Common labels"
  type        = map(string)
}

# Service Account for Training Orchestrator
resource "google_service_account" "training_orchestrator" {
  account_id   = "gaelp-training-orchestrator"
  display_name = "GAELP Training Orchestrator"
  description  = "Service account for GAELP training orchestration workloads"
}

# Service Account for Agent Deployment
resource "google_service_account" "agent_deployment" {
  account_id   = "gaelp-agent-deployment"
  display_name = "GAELP Agent Deployment"
  description  = "Service account for GAELP agent deployment and management"
}

# Service Account for Campaign Management
resource "google_service_account" "campaign_management" {
  account_id   = "gaelp-campaign-management"
  display_name = "GAELP Campaign Management"
  description  = "Service account for GAELP campaign management and monitoring"
}

# Service Account for Data Processing
resource "google_service_account" "data_processing" {
  account_id   = "gaelp-data-processing"
  display_name = "GAELP Data Processing"
  description  = "Service account for GAELP data processing pipelines"
}

# Service Account for Monitoring
resource "google_service_account" "monitoring" {
  account_id   = "gaelp-monitoring"
  display_name = "GAELP Monitoring"
  description  = "Service account for GAELP monitoring and alerting"
}

# Custom Role for GAELP Operations
resource "google_project_iam_custom_role" "gaelp_operator" {
  role_id     = "gaelpOperator"
  title       = "GAELP Operator"
  description = "Custom role for GAELP platform operations"

  permissions = [
    "bigquery.datasets.get",
    "bigquery.tables.get",
    "bigquery.tables.getData",
    "bigquery.tables.create",
    "bigquery.tables.update",
    "bigquery.jobs.create",
    "storage.objects.get",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.list",
    "storage.buckets.get",
    "container.clusters.get",
    "container.pods.get",
    "container.pods.list",
    "container.pods.create",
    "container.pods.delete",
    "run.services.get",
    "run.services.list",
    "monitoring.timeSeries.create",
    "monitoring.metricDescriptors.create",
    "logging.logEntries.create",
    "aiplatform.customJobs.create",
    "aiplatform.customJobs.get",
    "aiplatform.customJobs.list"
  ]
}

# IAM bindings for Training Orchestrator
resource "google_project_iam_member" "training_orchestrator_bindings" {
  for_each = toset([
    "roles/container.developer",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/aiplatform.user",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    google_project_iam_custom_role.gaelp_operator.id
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.training_orchestrator.email}"
}

# IAM bindings for Agent Deployment
resource "google_project_iam_member" "agent_deployment_bindings" {
  for_each = toset([
    "roles/container.developer",
    "roles/run.developer",
    "roles/storage.objectViewer",
    "roles/bigquery.dataViewer",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    google_project_iam_custom_role.gaelp_operator.id
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.agent_deployment.email}"
}

# IAM bindings for Campaign Management
resource "google_project_iam_member" "campaign_management_bindings" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectViewer",
    "roles/run.invoker",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor",
    google_project_iam_custom_role.gaelp_operator.id
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.campaign_management.email}"
}

# IAM bindings for Data Processing
resource "google_project_iam_member" "data_processing_bindings" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/pubsub.editor",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
    google_project_iam_custom_role.gaelp_operator.id
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.data_processing.email}"
}

# IAM bindings for Monitoring
resource "google_project_iam_member" "monitoring_bindings" {
  for_each = toset([
    "roles/monitoring.editor",
    "roles/logging.viewer",
    "roles/bigquery.dataViewer",
    "roles/storage.objectViewer",
    "roles/container.viewer",
    "roles/run.viewer"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.monitoring.email}"
}

# Workload Identity bindings for GKE
resource "google_service_account_iam_member" "workload_identity_training" {
  service_account_id = google_service_account.training_orchestrator.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[gaelp-training/training-orchestrator]"
}

resource "google_service_account_iam_member" "workload_identity_agent" {
  service_account_id = google_service_account.agent_deployment.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[gaelp-agents/agent-deployment]"
}

resource "google_service_account_iam_member" "workload_identity_data" {
  service_account_id = google_service_account.data_processing.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[gaelp-data/data-processing]"
}

# Security policies
resource "google_project_organization_policy" "require_shielded_vm" {
  project    = var.project_id
  constraint = "compute.requireShieldedVm"

  boolean_policy {
    enforced = true
  }
}

resource "google_project_organization_policy" "disable_service_account_key_creation" {
  project    = var.project_id
  constraint = "iam.disableServiceAccountKeyCreation"

  boolean_policy {
    enforced = true
  }
}

resource "google_project_organization_policy" "restrict_public_ip" {
  project    = var.project_id
  constraint = "compute.vmExternalIpAccess"

  list_policy {
    deny {
      all = true
    }
  }
}

# Outputs
output "service_accounts" {
  description = "Service account details"
  value = {
    training_orchestrator = {
      email = google_service_account.training_orchestrator.email
      id    = google_service_account.training_orchestrator.id
    }
    agent_deployment = {
      email = google_service_account.agent_deployment.email
      id    = google_service_account.agent_deployment.id
    }
    campaign_management = {
      email = google_service_account.campaign_management.email
      id    = google_service_account.campaign_management.id
    }
    data_processing = {
      email = google_service_account.data_processing.email
      id    = google_service_account.data_processing.id
    }
    monitoring = {
      email = google_service_account.monitoring.email
      id    = google_service_account.monitoring.id
    }
  }
}

output "custom_role_id" {
  description = "Custom GAELP operator role ID"
  value       = google_project_iam_custom_role.gaelp_operator.id
}