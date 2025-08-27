# Pub/Sub Module for GAELP (Simplified for initial deployment)
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "labels" {
  description = "Common labels"
  type        = map(string)
}

# Training Events Topic
resource "google_pubsub_topic" "training_events" {
  name = "gaelp-training-events"
  labels = var.labels
  message_retention_duration = "86400s" # 24 hours
}

# Safety Alerts Topic
resource "google_pubsub_topic" "safety_alerts" {
  name = "gaelp-safety-alerts"
  labels = var.labels
  message_retention_duration = "604800s" # 7 days
}

# Campaign Events Topic
resource "google_pubsub_topic" "campaign_events" {
  name = "gaelp-campaign-events"
  labels = var.labels
  message_retention_duration = "259200s" # 3 days
}

# Agent State Changes Topic
resource "google_pubsub_topic" "agent_state_changes" {
  name = "gaelp-agent-state-changes"
  labels = var.labels
  message_retention_duration = "86400s" # 24 hours
}

# Model Updates Topic
resource "google_pubsub_topic" "model_updates" {
  name = "gaelp-model-updates"
  labels = var.labels
  message_retention_duration = "604800s" # 7 days
}

# Basic subscriptions for processing
resource "google_pubsub_subscription" "training_events_processor" {
  name  = "gaelp-training-events-processor"
  topic = google_pubsub_topic.training_events.name
  labels = var.labels
  message_retention_duration = "86400s"
  retain_acked_messages      = false
  ack_deadline_seconds = 20
}

resource "google_pubsub_subscription" "safety_alerts_monitor" {
  name  = "gaelp-safety-alerts-monitor"
  topic = google_pubsub_topic.safety_alerts.name
  labels = var.labels
  message_retention_duration = "604800s"
  retain_acked_messages      = true
  ack_deadline_seconds = 10
}

resource "google_pubsub_subscription" "campaign_events_analyzer" {
  name  = "gaelp-campaign-events-analyzer"
  topic = google_pubsub_topic.campaign_events.name
  labels = var.labels
  message_retention_duration = "259200s"
  retain_acked_messages      = false
  ack_deadline_seconds = 60
}

# Outputs
output "topic_names" {
  description = "Pub/Sub topic names"
  value = {
    training_events      = google_pubsub_topic.training_events.name
    safety_alerts       = google_pubsub_topic.safety_alerts.name
    campaign_events     = google_pubsub_topic.campaign_events.name
    agent_state_changes = google_pubsub_topic.agent_state_changes.name
    model_updates       = google_pubsub_topic.model_updates.name
  }
}

output "subscription_names" {
  description = "Pub/Sub subscription names"
  value = {
    training_events_processor = google_pubsub_subscription.training_events_processor.name
    safety_alerts_monitor     = google_pubsub_subscription.safety_alerts_monitor.name
    campaign_events_analyzer  = google_pubsub_subscription.campaign_events_analyzer.name
  }
}