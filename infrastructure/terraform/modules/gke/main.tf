# GKE Module for GAELP
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "zone" {
  description = "GCP zone"
  type        = string
}

variable "network_name" {
  description = "VPC network name"
  type        = string
}

variable "subnet_name" {
  description = "Subnet name"
  type        = string
}

variable "labels" {
  description = "Common labels"
  type        = map(string)
}

# GKE Cluster
resource "google_container_cluster" "gaelp_cluster" {
  name     = "gaelp-cluster"
  location = var.region

  # Network configuration
  network    = var.network_name
  subnetwork = var.subnet_name

  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  # IP allocation policy
  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }

  # Network policy
  network_policy {
    enabled = true
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Master auth
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  # Private cluster config
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"

    master_global_access_config {
      enabled = true
    }
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = false
    }

    network_policy_config {
      disabled = false
    }

    gcp_filestore_csi_driver_config {
      enabled = true
    }

    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Maintenance policy
  maintenance_policy {
    recurring_window {
      start_time = "2023-01-01T02:00:00Z"
      end_time   = "2023-01-01T06:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }

  # Logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  # Cluster autoscaling (not enabled by default)
  cluster_autoscaling {
    enabled = false
  }

  resource_labels = var.labels

  lifecycle {
    ignore_changes = [node_pool]
  }
}

# CPU-optimized node pool for general workloads
resource "google_container_node_pool" "cpu_pool" {
  name       = "cpu-pool"
  location   = var.region
  cluster    = google_container_cluster.gaelp_cluster.name
  node_count = 1

  # Autoscaling
  autoscaling {
    min_node_count = 0
    max_node_count = 10
  }

  # Node configuration
  node_config {
    preemptible  = true
    machine_type = "e2-standard-4"
    disk_size_gb = 50
    disk_type    = "pd-ssd"

    # Google service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Labels and tags
    labels = merge(var.labels, {
      node_pool = "cpu-pool"
      workload  = "general"
    })

    tags = ["gke-node", "cpu-pool"]

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# GPU node pool for ML workloads
resource "google_container_node_pool" "gpu_pool" {
  name       = "gpu-pool"
  location   = var.zone
  cluster    = google_container_cluster.gaelp_cluster.name
  node_count = 0

  # Autoscaling
  autoscaling {
    min_node_count = 0
    max_node_count = 5
  }

  # Node configuration
  node_config {
    preemptible  = true
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    # GPU configuration
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }

    # Google service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Labels and tags
    labels = merge(var.labels, {
      node_pool = "gpu-pool"
      workload  = "ml-training"
    })

    tags = ["gke-node", "gpu-pool"]

    # Taints for GPU nodes
    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# High-memory node pool for large model inference
resource "google_container_node_pool" "memory_pool" {
  name       = "memory-pool"
  location   = var.region
  cluster    = google_container_cluster.gaelp_cluster.name
  node_count = 0

  # Autoscaling
  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }

  # Node configuration
  node_config {
    preemptible  = true
    machine_type = "n2-highmem-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    # Google service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Labels and tags
    labels = merge(var.labels, {
      node_pool = "memory-pool"
      workload  = "inference"
    })

    tags = ["gke-node", "memory-pool"]

    # Taints for high-memory nodes
    taint {
      key    = "workload"
      value  = "memory-intensive"
      effect = "NO_SCHEDULE"
    }

    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Upgrade settings
  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# Service account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "gaelp-gke-nodes"
  display_name = "GAELP GKE Node Service Account"
  description  = "Service account for GAELP GKE cluster nodes"
}

# IAM bindings for node service account
resource "google_project_iam_member" "gke_node_bindings" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer",
    "roles/artifactregistry.reader"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Outputs
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.gaelp_cluster.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.gaelp_cluster.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.gaelp_cluster.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "node_service_account" {
  description = "Node service account email"
  value       = google_service_account.gke_nodes.email
}