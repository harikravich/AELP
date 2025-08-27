#!/usr/bin/env python3
"""
Production Integration Example for GAELP Training Orchestrator

This example demonstrates how the training orchestrator integrates with
real production services instead of demo/mock services.

Usage:
    python production_example.py --environment production
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any
from datetime import datetime

# Add the training orchestrator to the path
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config, TrainingOrchestratorConfig
from production_adapter import get_production_adapter, ProductionServiceAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTrainingOrchestrator:
    """
    Production-ready training orchestrator that uses real GCP services
    """
    
    def __init__(self, config: TrainingOrchestratorConfig):
        self.config = config
        self.adapter = get_production_adapter()
        self.demo_mode = self.adapter is None
        
        if self.demo_mode:
            logger.warning("🟡 Running in DEMO MODE - production services not available")
        else:
            logger.info("🟢 Running in PRODUCTION MODE - using real GCP services")
    
    def start_training_session(self, agent_id: str, experiment_config: Dict[str, Any]) -> bool:
        """Start a new training session"""
        
        logger.info(f"🚀 Starting training session for agent {agent_id}")
        
        # Create training session metadata
        session_data = {
            "training_job_id": f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": "STARTED",
            "config": experiment_config,
            "environment": self.config.environment
        }
        
        if self.demo_mode:
            # Demo mode - log actions without real service calls
            logger.info("📝 [DEMO] Would store training session in BigQuery")
            logger.info("📝 [DEMO] Would publish training event to Pub/Sub")
            logger.info("📝 [DEMO] Would cache agent state in Redis")
            return True
        else:
            # Production mode - use real services
            success = True
            
            # Store training metrics in BigQuery
            if not self.adapter.store_training_metrics(session_data):
                logger.error("❌ Failed to store training metrics")
                success = False
            
            # Publish training event to Pub/Sub
            if not self.adapter.publish_training_event(session_data):
                logger.error("❌ Failed to publish training event")
                success = False
            
            # Cache initial agent state
            agent_state = {
                "status": "training",
                "session_id": session_data["training_job_id"],
                "last_update": datetime.now().isoformat(),
                "progress": 0.0
            }
            
            if not self.adapter.cache_agent_state(agent_id, agent_state):
                logger.error("❌ Failed to cache agent state")
                success = False
            
            return success
    
    def update_training_progress(self, agent_id: str, progress_data: Dict[str, Any]) -> bool:
        """Update training progress"""
        
        logger.info(f"📊 Updating training progress for agent {agent_id}: {progress_data.get('progress', 0):.1%}")
        
        if self.demo_mode:
            # Demo mode
            logger.info("📝 [DEMO] Would update training metrics in BigQuery")
            logger.info("📝 [DEMO] Would update agent state in Redis")
            return True
        else:
            # Production mode
            success = True
            
            # Update training metrics
            metrics_data = {
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "epoch": progress_data.get("epoch", 0),
                "loss": progress_data.get("loss", 0.0),
                "accuracy": progress_data.get("accuracy", 0.0),
                "learning_rate": progress_data.get("learning_rate", 0.001),
                "training_job_id": progress_data.get("session_id", "unknown")
            }
            
            if not self.adapter.store_training_metrics(metrics_data):
                logger.error("❌ Failed to update training metrics")
                success = False
            
            # Update cached agent state
            agent_state = {
                "status": "training",
                "last_update": datetime.now().isoformat(),
                "progress": progress_data.get("progress", 0.0),
                "current_epoch": progress_data.get("epoch", 0),
                "current_loss": progress_data.get("loss", 0.0)
            }
            
            if not self.adapter.cache_agent_state(agent_id, agent_state):
                logger.error("❌ Failed to update agent state")
                success = False
            
            return success
    
    def complete_training_session(self, agent_id: str, final_metrics: Dict[str, Any]) -> bool:
        """Complete a training session"""
        
        logger.info(f"🏁 Completing training session for agent {agent_id}")
        
        completion_data = {
            "training_job_id": final_metrics.get("session_id", "unknown"),
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": "COMPLETED",
            "final_accuracy": final_metrics.get("final_accuracy", 0.0),
            "total_epochs": final_metrics.get("total_epochs", 0),
            "training_duration": final_metrics.get("duration_seconds", 0)
        }
        
        if self.demo_mode:
            # Demo mode
            logger.info("📝 [DEMO] Would store final training metrics in BigQuery")
            logger.info("📝 [DEMO] Would publish completion event to Pub/Sub")
            logger.info("📝 [DEMO] Would update agent state to 'completed'")
            return True
        else:
            # Production mode
            success = True
            
            # Store final metrics
            if not self.adapter.store_training_metrics(completion_data):
                logger.error("❌ Failed to store final training metrics")
                success = False
            
            # Publish completion event
            if not self.adapter.publish_training_event(completion_data):
                logger.error("❌ Failed to publish completion event")
                success = False
            
            # Update agent state to completed
            agent_state = {
                "status": "completed",
                "last_update": datetime.now().isoformat(),
                "progress": 1.0,
                "final_accuracy": final_metrics.get("final_accuracy", 0.0),
                "completion_time": datetime.now().isoformat()
            }
            
            if not self.adapter.cache_agent_state(agent_id, agent_state):
                logger.error("❌ Failed to update agent state to completed")
                success = False
            
            return success
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current agent status"""
        
        if self.demo_mode:
            # Demo mode - return mock status
            return {
                "agent_id": agent_id,
                "status": "demo",
                "source": "demo_mode",
                "last_update": datetime.now().isoformat()
            }
        else:
            # Production mode - get from Redis cache
            cached_state = self.adapter.get_agent_state(agent_id)
            
            if cached_state:
                return {
                    "agent_id": agent_id,
                    "source": "redis_cache",
                    **cached_state
                }
            else:
                return {
                    "agent_id": agent_id,
                    "status": "unknown",
                    "source": "not_found",
                    "last_update": datetime.now().isoformat()
                }
    
    def run_example_training_workflow(self) -> None:
        """Run an example training workflow"""
        
        logger.info("🎯 Starting example training workflow")
        
        agent_id = "example_agent_001"
        
        # Step 1: Start training session
        experiment_config = {
            "algorithm": "PPO",
            "learning_rate": 0.001,
            "batch_size": 64,
            "max_epochs": 100,
            "environment": "ad_campaign_simulation"
        }
        
        if not self.start_training_session(agent_id, experiment_config):
            logger.error("❌ Failed to start training session")
            return
        
        # Step 2: Simulate training progress updates
        import time
        for epoch in range(1, 6):  # Simulate 5 epochs
            progress_data = {
                "epoch": epoch,
                "progress": epoch / 100.0,  # 5% progress
                "loss": 1.0 - (epoch * 0.1),  # Decreasing loss
                "accuracy": 0.5 + (epoch * 0.05),  # Increasing accuracy
                "learning_rate": 0.001
            }
            
            if not self.update_training_progress(agent_id, progress_data):
                logger.error(f"❌ Failed to update progress for epoch {epoch}")
            
            # Check agent status
            status = self.get_agent_status(agent_id)
            logger.info(f"📋 Agent status: {status.get('status')} - Progress: {status.get('progress', 0):.1%}")
            
            time.sleep(1)  # Brief pause between updates
        
        # Step 3: Complete training session
        final_metrics = {
            "final_accuracy": 0.75,
            "total_epochs": 5,
            "duration_seconds": 300
        }
        
        if not self.complete_training_session(agent_id, final_metrics):
            logger.error("❌ Failed to complete training session")
            return
        
        # Step 4: Final status check
        final_status = self.get_agent_status(agent_id)
        logger.info(f"🎉 Training completed! Final status: {final_status}")
        
        logger.info("✅ Example training workflow completed successfully")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="GAELP Production Integration Example")
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="production",
        help="Environment to run in"
    )
    parser.add_argument(
        "--test-connections",
        action="store_true",
        help="Only test service connections"
    )
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting GAELP Training Orchestrator - Environment: {args.environment}")
    
    # Load configuration
    try:
        config = load_config(environment=args.environment)
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"   BigQuery Project: {config.database.bigquery_project}")
        logger.info(f"   Redis Host: {config.database.redis_host}:{config.database.redis_port}")
        logger.info(f"   Environment: {config.environment}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Create orchestrator
    orchestrator = ProductionTrainingOrchestrator(config)
    
    if args.test_connections:
        # Only test connections
        if orchestrator.adapter:
            logger.info("🔍 Testing production service connections...")
            results = orchestrator.adapter.test_connections()
            
            logger.info("📊 Connection Test Results:")
            for service, connected in results.items():
                status = "✅" if connected else "❌"
                logger.info(f"   {status} {service}")
            
            total = len(results)
            successful = sum(results.values())
            logger.info(f"📈 Overall: {successful}/{total} services connected")
            
            return 0 if successful > 0 else 1
        else:
            logger.warning("⚠️  No production adapter available - all services in demo mode")
            return 1
    else:
        # Run example workflow
        try:
            orchestrator.run_example_training_workflow()
            return 0
        except Exception as e:
            logger.error(f"❌ Example workflow failed: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())