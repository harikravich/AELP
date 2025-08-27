# Networking Module for GAELP
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

# VPC Network
resource "google_compute_network" "gaelp_vpc" {
  name                    = "gaelp-vpc"
  auto_create_subnetworks = false
  routing_mode           = "GLOBAL"
  description            = "GAELP main VPC network"
}

# Subnet for GKE and Compute resources
resource "google_compute_subnetwork" "gaelp_subnet" {
  name          = "gaelp-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.gaelp_vpc.id
  description   = "Main subnet for GAELP resources"

  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.2.0.0/16"
  }

  private_ip_google_access = true
}

# Cloud NAT for private instances
resource "google_compute_router" "gaelp_router" {
  name    = "gaelp-router"
  region  = var.region
  network = google_compute_network.gaelp_vpc.id
}

resource "google_compute_router_nat" "gaelp_nat" {
  name   = "gaelp-nat"
  router = google_compute_router.gaelp_router.name
  region = var.region

  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Firewall rules
resource "google_compute_firewall" "allow_internal" {
  name    = "gaelp-allow-internal"
  network = google_compute_network.gaelp_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8"]
  description   = "Allow internal communication within VPC"
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "gaelp-allow-ssh"
  network = google_compute_network.gaelp_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh-allowed"]
  description   = "Allow SSH access"
}

resource "google_compute_firewall" "allow_http_https" {
  name    = "gaelp-allow-http-https"
  network = google_compute_network.gaelp_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server", "https-server"]
  description   = "Allow HTTP/HTTPS traffic"
}

# Load balancer IP
resource "google_compute_global_address" "gaelp_lb_ip" {
  name         = "gaelp-lb-ip"
  description  = "Global IP for GAELP load balancer"
  address_type = "EXTERNAL"
}

# Outputs
output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.gaelp_vpc.name
}

output "network_id" {
  description = "VPC network ID"
  value       = google_compute_network.gaelp_vpc.id
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.gaelp_subnet.name
}

output "subnet_id" {
  description = "Subnet ID"
  value       = google_compute_subnetwork.gaelp_subnet.id
}

output "load_balancer_ip" {
  description = "Load balancer IP address"
  value       = google_compute_global_address.gaelp_lb_ip.address
}