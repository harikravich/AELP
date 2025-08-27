# BigQuery Module for GAELP
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

# Campaign Performance Dataset
resource "google_bigquery_dataset" "campaign_performance" {
  dataset_id                 = "campaign_performance"
  friendly_name              = "Campaign Performance Data"
  description                = "Dataset for storing ad campaign performance metrics"
  location                   = var.region
  default_table_expiration_ms = 31536000000 # 1 year

  labels = var.labels

  access {
    role          = "OWNER"
    special_group = "projectOwners"
  }

  access {
    role          = "READER"
    special_group = "projectReaders"
  }
}

# Agent Training Dataset
resource "google_bigquery_dataset" "agent_training" {
  dataset_id                 = "agent_training"
  friendly_name              = "Agent Training Data"
  description                = "Dataset for storing agent training metrics and logs"
  location                   = var.region
  default_table_expiration_ms = 15552000000 # 6 months

  labels = var.labels

  access {
    role          = "OWNER"
    special_group = "projectOwners"
  }

  access {
    role          = "READER"
    special_group = "projectReaders"
  }
}

# Simulation Results Dataset
resource "google_bigquery_dataset" "simulation_results" {
  dataset_id                 = "simulation_results"
  friendly_name              = "Simulation Results"
  description                = "Dataset for storing campaign simulation results"
  location                   = var.region
  default_table_expiration_ms = 7776000000 # 3 months

  labels = var.labels

  access {
    role          = "OWNER"
    special_group = "projectOwners"
  }

  access {
    role          = "READER"
    special_group = "projectReaders"
  }
}

# Real-time Metrics Dataset (streaming)
resource "google_bigquery_dataset" "realtime_metrics" {
  dataset_id                 = "realtime_metrics"
  friendly_name              = "Real-time Metrics"
  description                = "Dataset for real-time campaign and agent metrics"
  location                   = var.region
  default_table_expiration_ms = 2592000000 # 30 days

  labels = var.labels

  access {
    role          = "OWNER"
    special_group = "projectOwners"
  }

  access {
    role          = "READER"
    special_group = "projectReaders"
  }
}

# Campaign Performance Table
resource "google_bigquery_table" "campaign_metrics" {
  dataset_id = google_bigquery_dataset.campaign_performance.dataset_id
  table_id   = "campaign_metrics"

  labels = var.labels

  schema = jsonencode([
    {
      name = "campaign_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "agent_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "platform"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "impressions"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "clicks"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "conversions"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "cost"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "revenue"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "ctr"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "cpc"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "roas"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "metadata"
      type = "JSON"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["platform", "agent_id"]
}

# Agent Training Metrics Table
resource "google_bigquery_table" "training_metrics" {
  dataset_id = google_bigquery_dataset.agent_training.dataset_id
  table_id   = "training_metrics"

  labels = var.labels

  schema = jsonencode([
    {
      name = "training_job_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "agent_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "epoch"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "loss"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "accuracy"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "learning_rate"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "gpu_utilization"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "memory_usage"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "training_config"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "model_checkpoint_path"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["agent_id", "training_job_id"]
}

# Simulation Results Table
resource "google_bigquery_table" "simulation_runs" {
  dataset_id = google_bigquery_dataset.simulation_results.dataset_id
  table_id   = "simulation_runs"

  labels = var.labels

  schema = jsonencode([
    {
      name = "simulation_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "agent_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "environment_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "duration_seconds"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "total_reward"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "actions_taken"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "success_rate"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "environment_config"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "agent_config"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "results_summary"
      type = "JSON"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["agent_id", "environment_id"]
}

# Real-time Metrics Table (for streaming)
resource "google_bigquery_table" "realtime_campaign_metrics" {
  dataset_id = google_bigquery_dataset.realtime_metrics.dataset_id
  table_id   = "realtime_campaign_metrics"

  labels = var.labels

  schema = jsonencode([
    {
      name = "event_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "campaign_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "agent_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "event_type"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "platform"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "metric_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "metric_value"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "event_data"
      type = "JSON"
      mode = "NULLABLE"
    }
  ])

  time_partitioning {
    type  = "HOUR"
    field = "timestamp"
  }

  clustering = ["platform", "agent_id", "event_type"]
}

# Data source for current client config removed - using special groups instead

# Outputs
output "dataset_ids" {
  description = "BigQuery dataset IDs"
  value = {
    campaign_performance = google_bigquery_dataset.campaign_performance.dataset_id
    agent_training      = google_bigquery_dataset.agent_training.dataset_id
    simulation_results  = google_bigquery_dataset.simulation_results.dataset_id
    realtime_metrics   = google_bigquery_dataset.realtime_metrics.dataset_id
  }
}

output "table_ids" {
  description = "BigQuery table IDs"
  value = {
    campaign_metrics         = google_bigquery_table.campaign_metrics.table_id
    training_metrics        = google_bigquery_table.training_metrics.table_id
    simulation_runs         = google_bigquery_table.simulation_runs.table_id
    realtime_campaign_metrics = google_bigquery_table.realtime_campaign_metrics.table_id
  }
}