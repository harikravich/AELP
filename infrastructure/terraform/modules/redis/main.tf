# Redis Module for GAELP (Simplified for initial deployment)
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "network_name" {
  description = "VPC network name"
  type        = string
}

variable "labels" {
  description = "Common labels"
  type        = map(string)
}

# Redis Instance for Real-time State Management
resource "google_redis_instance" "gaelp_cache" {
  name           = "gaelp-redis-cache"
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region

  # Redis configuration
  redis_version     = "REDIS_7_0"
  display_name      = "GAELP Redis Cache"
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }

  # Auth
  auth_enabled = true

  # Labels
  labels = var.labels
}

# Redis Instance for Session Management  
resource "google_redis_instance" "gaelp_sessions" {
  name           = "gaelp-redis-sessions"
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region

  # Redis configuration
  redis_version     = "REDIS_7_0"
  display_name      = "GAELP Redis Sessions"
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    timeout = "300"
  }

  # Auth
  auth_enabled = true

  # Labels
  labels = merge(var.labels, {
    purpose = "sessions"
  })
}

# Outputs
output "redis_instances" {
  description = "Redis instance details"
  value = {
    cache = {
      id         = google_redis_instance.gaelp_cache.id
      host       = google_redis_instance.gaelp_cache.host
      port       = google_redis_instance.gaelp_cache.port
      auth_string = google_redis_instance.gaelp_cache.auth_string
    }
    sessions = {
      id         = google_redis_instance.gaelp_sessions.id
      host       = google_redis_instance.gaelp_sessions.host
      port       = google_redis_instance.gaelp_sessions.port
      auth_string = google_redis_instance.gaelp_sessions.auth_string
    }
  }
  sensitive = true
}

output "redis_connection_strings" {
  description = "Redis connection strings"
  value = {
    cache       = "redis://:${google_redis_instance.gaelp_cache.auth_string}@${google_redis_instance.gaelp_cache.host}:${google_redis_instance.gaelp_cache.port}"
    sessions    = "redis://:${google_redis_instance.gaelp_sessions.auth_string}@${google_redis_instance.gaelp_sessions.host}:${google_redis_instance.gaelp_sessions.port}"
  }
  sensitive = true
}