"""
Metrics collection and monitoring for agents
"""
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import asyncio

logger = logging.getLogger(__name__)

# Prometheus metrics
agent_jobs_total = Counter('agent_jobs_total', 'Total number of agent jobs', ['agent_id', 'status'])
agent_job_duration = Histogram('agent_job_duration_seconds', 'Job duration in seconds', ['agent_id'])
agent_resource_usage = Gauge('agent_resource_usage', 'Resource usage', ['agent_id', 'resource_type'])
agent_cost_total = Counter('agent_cost_total', 'Total cost per agent', ['agent_id'])
cluster_resource_usage = Gauge('cluster_resource_usage', 'Cluster resource usage', ['resource_type', 'usage_type'])


class MetricsCollector:
    """Collects and aggregates metrics for agents and the system"""
    
    def __init__(self):
        self.metrics_cache: Dict[str, Any] = {}
        self.last_collection_time = datetime.utcnow()
        
    def record_job_started(self, agent_id: int, job_id: int):
        """Record job start"""
        agent_jobs_total.labels(agent_id=str(agent_id), status='started').inc()
        logger.info(f"Recorded job start: agent={agent_id}, job={job_id}")
    
    def record_job_completed(self, agent_id: int, job_id: int, duration_seconds: float):
        """Record job completion"""
        agent_jobs_total.labels(agent_id=str(agent_id), status='completed').inc()
        agent_job_duration.labels(agent_id=str(agent_id)).observe(duration_seconds)
        logger.info(f"Recorded job completion: agent={agent_id}, job={job_id}, duration={duration_seconds}s")
    
    def record_job_failed(self, agent_id: int, job_id: int, duration_seconds: float):
        """Record job failure"""
        agent_jobs_total.labels(agent_id=str(agent_id), status='failed').inc()
        agent_job_duration.labels(agent_id=str(agent_id)).observe(duration_seconds)
        logger.info(f"Recorded job failure: agent={agent_id}, job={job_id}, duration={duration_seconds}s")
    
    def record_resource_usage(self, agent_id: int, resource_type: str, usage: float):
        """Record resource usage"""
        agent_resource_usage.labels(agent_id=str(agent_id), resource_type=resource_type).set(usage)
    
    def record_agent_cost(self, agent_id: int, cost: float):
        """Record agent cost"""
        agent_cost_total.labels(agent_id=str(agent_id)).inc(cost)
    
    def record_cluster_resources(self, resources: Dict[str, Any]):
        """Record cluster resource usage"""
        for resource_type in ['cpu', 'memory']:
            requests = resources.get(f'{resource_type}_requests', 0)
            limits = resources.get(f'{resource_type}_limits', 0)
            
            cluster_resource_usage.labels(
                resource_type=resource_type, 
                usage_type='requests'
            ).set(requests)
            
            cluster_resource_usage.labels(
                resource_type=resource_type, 
                usage_type='limits'
            ).set(limits)
    
    def get_agent_metrics_summary(self, agent_id: int) -> Dict[str, Any]:
        """Get metrics summary for an agent"""
        # This would typically query your metrics backend
        # For now, return a mock summary
        return {
            "jobs_total": 10,
            "jobs_completed": 8,
            "jobs_failed": 1,
            "jobs_running": 1,
            "avg_job_duration": 3600,  # seconds
            "total_cost": 125.50,
            "current_cpu_usage": 2.5,  # cores
            "current_memory_usage": 8.0,  # GB
            "last_activity": datetime.utcnow().isoformat()
        }
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get system-wide metrics summary"""
        return {
            "total_agents": 25,
            "active_agents": 15,
            "total_jobs": 150,
            "running_jobs": 12,
            "queued_jobs": 5,
            "cluster_cpu_usage": 45.2,  # percentage
            "cluster_memory_usage": 67.8,  # percentage
            "total_cost_today": 2456.78,
            "avg_job_duration": 2400  # seconds
        }


class HealthMonitor:
    """Monitors system and agent health"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        self.alert_thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "job_failure_rate": 20.0,
            "queue_length": 50
        }
    
    async def check_agent_health(self, agent_id: int) -> Dict[str, Any]:
        """Check health of a specific agent"""
        health_status = {
            "agent_id": agent_id,
            "status": "healthy",
            "checks": {},
            "alerts": [],
            "last_check": datetime.utcnow().isoformat()
        }
        
        try:
            # Check resource usage
            metrics = self.metrics_collector.get_agent_metrics_summary(agent_id)
            
            # CPU check
            cpu_usage = metrics.get("current_cpu_usage", 0)
            if cpu_usage > self.alert_thresholds["cpu_usage"]:
                health_status["status"] = "warning"
                health_status["alerts"].append({
                    "type": "high_cpu_usage",
                    "value": cpu_usage,
                    "threshold": self.alert_thresholds["cpu_usage"]
                })
            health_status["checks"]["cpu_usage"] = cpu_usage
            
            # Memory check
            memory_usage = metrics.get("current_memory_usage", 0)
            if memory_usage > self.alert_thresholds["memory_usage"]:
                health_status["status"] = "warning"
                health_status["alerts"].append({
                    "type": "high_memory_usage",
                    "value": memory_usage,
                    "threshold": self.alert_thresholds["memory_usage"]
                })
            health_status["checks"]["memory_usage"] = memory_usage
            
            # Job failure rate check
            total_jobs = metrics.get("jobs_total", 0)
            failed_jobs = metrics.get("jobs_failed", 0)
            failure_rate = (failed_jobs / max(total_jobs, 1)) * 100
            
            if failure_rate > self.alert_thresholds["job_failure_rate"]:
                health_status["status"] = "critical"
                health_status["alerts"].append({
                    "type": "high_failure_rate",
                    "value": failure_rate,
                    "threshold": self.alert_thresholds["job_failure_rate"]
                })
            health_status["checks"]["job_failure_rate"] = failure_rate
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["alerts"].append({
                "type": "health_check_error",
                "error": str(e)
            })
            logger.error(f"Error checking agent {agent_id} health: {e}")
        
        return health_status
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health_status = {
            "status": "healthy",
            "checks": {},
            "alerts": [],
            "last_check": datetime.utcnow().isoformat()
        }
        
        try:
            metrics = self.metrics_collector.get_system_metrics_summary()
            
            # Cluster resource checks
            cpu_usage = metrics.get("cluster_cpu_usage", 0)
            memory_usage = metrics.get("cluster_memory_usage", 0)
            
            if cpu_usage > self.alert_thresholds["cpu_usage"]:
                health_status["status"] = "warning"
                health_status["alerts"].append({
                    "type": "cluster_high_cpu",
                    "value": cpu_usage,
                    "threshold": self.alert_thresholds["cpu_usage"]
                })
            
            if memory_usage > self.alert_thresholds["memory_usage"]:
                health_status["status"] = "warning"
                health_status["alerts"].append({
                    "type": "cluster_high_memory",
                    "value": memory_usage,
                    "threshold": self.alert_thresholds["memory_usage"]
                })
            
            # Queue length check
            queued_jobs = metrics.get("queued_jobs", 0)
            if queued_jobs > self.alert_thresholds["queue_length"]:
                health_status["status"] = "warning"
                health_status["alerts"].append({
                    "type": "high_queue_length",
                    "value": queued_jobs,
                    "threshold": self.alert_thresholds["queue_length"]
                })
            
            health_status["checks"].update({
                "cluster_cpu_usage": cpu_usage,
                "cluster_memory_usage": memory_usage,
                "queued_jobs": queued_jobs,
                "running_jobs": metrics.get("running_jobs", 0),
                "active_agents": metrics.get("active_agents", 0)
            })
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["alerts"].append({
                "type": "system_health_check_error",
                "error": str(e)
            })
            logger.error(f"Error checking system health: {e}")
        
        return health_status


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.notification_channels = []
    
    async def process_alert(self, alert: Dict[str, Any]):
        """Process and potentially send an alert"""
        alert_key = f"{alert['type']}_{alert.get('agent_id', 'system')}"
        
        # Check if this is a new alert or update
        if alert_key not in self.active_alerts:
            # New alert
            self.active_alerts[alert_key] = {
                **alert,
                "first_seen": datetime.utcnow(),
                "count": 1
            }
            await self._send_notification(alert)
            logger.warning(f"New alert: {alert}")
        else:
            # Update existing alert
            self.active_alerts[alert_key]["count"] += 1
            self.active_alerts[alert_key]["last_seen"] = datetime.utcnow()
    
    async def resolve_alert(self, alert_type: str, agent_id: Optional[int] = None):
        """Resolve an alert"""
        alert_key = f"{alert_type}_{agent_id or 'system'}"
        
        if alert_key in self.active_alerts:
            resolved_alert = self.active_alerts.pop(alert_key)
            resolved_alert["resolved_at"] = datetime.utcnow()
            self.alert_history.append(resolved_alert)
            
            logger.info(f"Resolved alert: {alert_key}")
    
    async def _send_notification(self, alert: Dict[str, Any]):
        """Send notification for alert"""
        # In a real implementation, this would send notifications via:
        # - Email
        # - Slack
        # - PagerDuty
        # - etc.
        
        logger.info(f"Alert notification: {alert['type']} - {alert.get('message', 'No message')}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        return self.alert_history[-limit:]


# Global instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor(metrics_collector)
alert_manager = AlertManager()


def start_metrics_server(port: int = 8080):
    """Start Prometheus metrics server"""
    start_http_server(port)
    logger.info(f"Started Prometheus metrics server on port {port}")