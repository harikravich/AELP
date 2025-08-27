#!/usr/bin/env python3
"""
Main entry point for the Agent Manager service
"""
import asyncio
import logging
import signal
import sys
from typing import Optional

import uvicorn
from prometheus_client import start_http_server

from config.settings import settings
from core.database import create_tables
from monitoring.metrics import start_metrics_server
from kubernetes.scheduler import JobScheduler, AutoScaler
from api.main import app

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper()),
    format=settings.logging.format
)
logger = logging.getLogger(__name__)


class AgentManagerService:
    """Main service orchestrator for the Agent Manager"""
    
    def __init__(self):
        self.scheduler: Optional[JobScheduler] = None
        self.auto_scaler: Optional[AutoScaler] = None
        self.running = False
    
    async def start(self):
        """Start the agent manager service"""
        logger.info("Starting GAELP Agent Manager...")
        
        # Initialize database
        logger.info("Initializing database...")
        create_tables()
        
        # Start metrics server
        if settings.monitoring.prometheus_port:
            logger.info(f"Starting Prometheus metrics server on port {settings.monitoring.prometheus_port}")
            start_metrics_server(settings.monitoring.prometheus_port)
        
        # Initialize scheduler
        logger.info("Initializing job scheduler...")
        self.scheduler = JobScheduler(namespace=settings.kubernetes.namespace)
        self.auto_scaler = AutoScaler(self.scheduler)
        
        # Start background tasks
        self.running = True
        asyncio.create_task(self._scheduler_loop())
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._autoscaler_loop())
        
        logger.info("Agent Manager started successfully")
    
    async def stop(self):
        """Stop the agent manager service"""
        logger.info("Stopping Agent Manager...")
        self.running = False
        
        # Give background tasks time to finish
        await asyncio.sleep(2)
        
        logger.info("Agent Manager stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                if self.scheduler:
                    # Schedule jobs
                    await self.scheduler.schedule_jobs()
                    
                    # Monitor running jobs
                    await self.scheduler.monitor_jobs()
                
                # Wait before next cycle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitoring_loop(self):
        """Health monitoring loop"""
        while self.running:
            try:
                from monitoring.metrics import health_monitor
                
                # Check system health
                system_health = await health_monitor.check_system_health()
                
                if system_health["status"] != "healthy":
                    logger.warning(f"System health check failed: {system_health}")
                
                # Process any alerts
                for alert in system_health.get("alerts", []):
                    from monitoring.metrics import alert_manager
                    await alert_manager.process_alert(alert)
                
                # Wait before next check
                await asyncio.sleep(settings.monitoring.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _autoscaler_loop(self):
        """Auto-scaling loop"""
        while self.running:
            try:
                if self.auto_scaler:
                    await self.auto_scaler.scale_based_on_demand()
                
                # Run auto-scaling less frequently
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in auto-scaler loop: {e}")
                await asyncio.sleep(300)


# Global service instance
service = AgentManagerService()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    asyncio.create_task(service.stop())


async def main():
    """Main function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the service
        await service.start()
        
        # Run the API server
        config = uvicorn.Config(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_level=settings.logging.level.lower(),
            reload=settings.environment == "development"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"Starting API server on {settings.api_host}:{settings.api_port}")
        await server.serve()
        
    except Exception as e:
        logger.error(f"Failed to start Agent Manager: {e}")
        sys.exit(1)
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())