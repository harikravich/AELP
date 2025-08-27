# Monitoring Module for GAELP
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "labels" {
  description = "Common labels"
  type        = map(string)
}

# Notification channels
resource "google_monitoring_notification_channel" "email" {
  display_name = "GAELP Email Alerts"
  type         = "email"

  labels = {
    email_address = "alerts@your-domain.com" # Replace with your email
  }

  user_labels = var.labels
}

resource "google_monitoring_notification_channel" "slack" {
  display_name = "GAELP Slack Alerts"
  type         = "slack"

  labels = {
    channel_name = "#gaelp-alerts"
    url          = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" # Replace with your webhook
  }

  user_labels = var.labels
}

# Budget Alert Policy - commented out for initial deployment
# resource "google_billing_budget" "gaelp_budget" {
#   billing_account = "YOUR_BILLING_ACCOUNT_ID"
#   display_name    = "GAELP Monthly Budget"
#
#   budget_filter {
#     projects = ["projects/${var.project_id}"]
#   }
#
#   amount {
#     specified_amount {
#       currency_code = "USD"
#       units         = "1000"
#     }
#   }
#
#   threshold_rules {
#     threshold_percent = 0.5
#     spend_basis       = "CURRENT_SPEND"
#   }
#
#   all_updates_rule {
#     disable_default_iam_recipients = false
#   }
# }

# GKE Cluster Health Alert
resource "google_monitoring_alert_policy" "gke_cluster_health" {
  display_name = "GKE Cluster Health"
  combiner     = "OR"

  conditions {
    display_name = "GKE cluster is down"

    condition_threshold {
      filter          = "resource.type=\"gke_cluster\" AND resource.label.cluster_name=\"gaelp-cluster\""
      duration        = "300s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.id,
    google_monitoring_notification_channel.slack.id
  ]

  alert_strategy {
    auto_close = "86400s" # 24 hours
  }

  user_labels = var.labels
}

# High CPU Usage Alert
resource "google_monitoring_alert_policy" "high_cpu_usage" {
  display_name = "High CPU Usage"
  combiner     = "OR"

  conditions {
    display_name = "CPU usage above 80%"

    condition_threshold {
      filter          = "resource.type=\"gce_instance\" AND metric.type=\"compute.googleapis.com/instance/cpu/utilization\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.id
  ]

  alert_strategy {
    auto_close = "3600s" # 1 hour
  }

  user_labels = var.labels
}

# Training Job Failure Alert
resource "google_monitoring_alert_policy" "training_job_failure" {
  display_name = "Training Job Failures"
  combiner     = "OR"

  conditions {
    display_name = "Training job failed"

    condition_threshold {
      filter          = "resource.type=\"k8s_container\" AND resource.label.container_name=~\"training-.*\" AND metric.type=\"logging.googleapis.com/log_entry_count\" AND log_entry.severity=\"ERROR\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.id,
    google_monitoring_notification_channel.slack.id
  ]

  alert_strategy {
    auto_close = "7200s" # 2 hours
  }

  user_labels = var.labels
}

# Cloud Run Service Error Rate Alert
resource "google_monitoring_alert_policy" "cloud_run_error_rate" {
  display_name = "Cloud Run High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run service error rate > 5%"

    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.label.response_code_class=\"5xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.05

      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"

        group_by_fields = [
          "resource.label.service_name"
        ]
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.id
  ]

  alert_strategy {
    auto_close = "3600s" # 1 hour
  }

  user_labels = var.labels
}

# BigQuery Job Failure Alert
resource "google_monitoring_alert_policy" "bigquery_job_failure" {
  display_name = "BigQuery Job Failures"
  combiner     = "OR"

  conditions {
    display_name = "BigQuery job failure rate > 10%"

    condition_threshold {
      filter          = "resource.type=\"bigquery_project\" AND metric.type=\"bigquery.googleapis.com/job/num_failed\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.1

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.id
  ]

  alert_strategy {
    auto_close = "3600s" # 1 hour
  }

  user_labels = var.labels
}

# Storage Quota Alert
resource "google_monitoring_alert_policy" "storage_quota" {
  display_name = "Storage Quota Usage"
  combiner     = "OR"

  conditions {
    display_name = "Storage usage > 80% of quota"

    condition_threshold {
      filter          = "resource.type=\"gcs_bucket\" AND metric.type=\"storage.googleapis.com/storage/total_bytes\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8e12 # 800GB in bytes

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [
    google_monitoring_notification_channel.email.id
  ]

  alert_strategy {
    auto_close = "86400s" # 24 hours
  }

  user_labels = var.labels
}

# Custom metrics for GAELP
resource "google_monitoring_metric_descriptor" "agent_performance" {
  description  = "GAELP agent performance metric"
  display_name = "Agent Performance Score"
  type         = "custom.googleapis.com/gaelp/agent_performance"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"

  labels {
    key         = "agent_id"
    value_type  = "STRING"
    description = "The ID of the agent"
  }

  labels {
    key         = "environment_id"
    value_type  = "STRING"
    description = "The ID of the environment"
  }

  labels {
    key         = "campaign_id"
    value_type  = "STRING"
    description = "The ID of the campaign"
  }
}

resource "google_monitoring_metric_descriptor" "training_progress" {
  description  = "GAELP training progress metric"
  display_name = "Training Progress"
  type         = "custom.googleapis.com/gaelp/training_progress"
  metric_kind  = "GAUGE"
  value_type   = "DOUBLE"

  labels {
    key         = "training_job_id"
    value_type  = "STRING"
    description = "The ID of the training job"
  }

  labels {
    key         = "agent_id"
    value_type  = "STRING"
    description = "The ID of the agent being trained"
  }
}

# GAELP Dashboard
resource "google_monitoring_dashboard" "gaelp_dashboard" {
  dashboard_json = jsonencode({
    displayName = "GAELP Platform Dashboard"
    mosaicLayout = {
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "GKE Cluster Status"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"gke_cluster\" AND resource.label.cluster_name=\"gaelp-cluster\""
                      aggregation = {
                        alignmentPeriod  = "60s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                }
              ]
              timeshiftDuration = "0s"
              yAxis = {
                label = "Status"
                scale = "LINEAR"
              }
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Training Jobs Performance"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/gaelp/training_progress\""
                      aggregation = {
                        alignmentPeriod  = "300s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                }
              ]
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Agent Performance"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "metric.type=\"custom.googleapis.com/gaelp/agent_performance\""
                      aggregation = {
                        alignmentPeriod  = "300s"
                        perSeriesAligner = "ALIGN_MEAN"
                      }
                    }
                  }
                  plotType = "LINE"
                }
              ]
            }
          }
        },
        {
          width  = 6
          height = 4
          widget = {
            title = "Cloud Run Request Count"
            xyChart = {
              dataSets = [
                {
                  timeSeriesQuery = {
                    timeSeriesFilter = {
                      filter = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\""
                      aggregation = {
                        alignmentPeriod    = "60s"
                        perSeriesAligner   = "ALIGN_RATE"
                        crossSeriesReducer = "REDUCE_SUM"
                        groupByFields      = ["resource.label.service_name"]
                      }
                    }
                  }
                  plotType = "STACKED_AREA"
                }
              ]
            }
          }
        }
      ]
    }
  })
}

# Outputs
output "dashboard_url" {
  description = "Monitoring dashboard URL"
  value       = "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.gaelp_dashboard.id}?project=${var.project_id}"
}

output "notification_channels" {
  description = "Notification channel IDs"
  value = {
    email = google_monitoring_notification_channel.email.id
    slack = google_monitoring_notification_channel.slack.id
  }
}

output "alert_policies" {
  description = "Alert policy names"
  value = {
    gke_cluster_health    = google_monitoring_alert_policy.gke_cluster_health.name
    high_cpu_usage       = google_monitoring_alert_policy.high_cpu_usage.name
    training_job_failure = google_monitoring_alert_policy.training_job_failure.name
    cloud_run_error_rate = google_monitoring_alert_policy.cloud_run_error_rate.name
    bigquery_job_failure = google_monitoring_alert_policy.bigquery_job_failure.name
    storage_quota        = google_monitoring_alert_policy.storage_quota.name
  }
}