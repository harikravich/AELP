#!/usr/bin/env python3
"""
Integration script for Production Checkpoint Manager with GAELP

This script demonstrates how to integrate the production-grade checkpoint manager
with the existing GAELP training and deployment workflow.
"""

import os
import sys
import logging
import asyncio
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add AELP to path
sys.path.insert(0, '/home/hariravichandran/AELP')

from production_checkpoint_manager import (
    ProductionCheckpointManager,
    ValidationStatus,
    create_production_checkpoint_manager,
    validate_checkpoint_before_deployment,
    emergency_rollback_if_needed
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GAELPProductionIntegration:
    """
    Integration wrapper for GAELP with production checkpoint manager
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "gaelp_production_checkpoints",
        holdout_data_path: str = "gaelp_holdout_data.json"
    ):
        self.checkpoint_manager = create_production_checkpoint_manager(
            checkpoint_dir=checkpoint_dir,
            holdout_data_path=holdout_data_path,
            max_checkpoints=20,  # Keep more checkpoints for production
            auto_rollback=True
        )
        
        self.training_config = {}
        self.current_agent = None
        
        logger.info("GAELP Production Integration initialized")
    
    def save_training_checkpoint(
        self,
        agent,
        episode: int,
        training_metrics: Dict[str, Any],
        validate_immediately: bool = True
    ) -> Optional[str]:
        """
        Save training checkpoint with production validation
        
        Args:
            agent: The RL agent to checkpoint
            episode: Current training episode
            training_metrics: Metrics from training
            validate_immediately: Whether to validate immediately
            
        Returns:
            Checkpoint ID if successful, None otherwise
        """
        
        try:
            # Extract training config from agent
            training_config = self._extract_training_config(agent)
            
            # Generate model version
            model_version = f"gaelp_v{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            logger.info(f"Saving GAELP checkpoint at episode {episode}")
            
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                model=agent,
                model_version=model_version,
                episode=episode,
                training_config=training_config,
                training_metrics=training_metrics,
                validate_immediately=validate_immediately
            )
            
            self.current_agent = agent
            
            logger.info(f"GAELP checkpoint saved: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save GAELP checkpoint: {e}")
            return None
    
    def _extract_training_config(self, agent) -> Dict[str, Any]:
        """Extract training configuration from agent"""
        config = {
            'agent_type': type(agent).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract common RL agent parameters
        if hasattr(agent, 'learning_rate'):
            config['learning_rate'] = agent.learning_rate
        if hasattr(agent, 'epsilon'):
            config['epsilon'] = agent.epsilon
        if hasattr(agent, 'gamma'):
            config['gamma'] = agent.gamma
        if hasattr(agent, 'episodes'):
            config['total_episodes'] = agent.episodes
        if hasattr(agent, 'buffer_size'):
            config['buffer_size'] = agent.buffer_size
        
        # Extract network architecture info
        if hasattr(agent, 'q_network') and hasattr(agent.q_network, 'state_dict'):
            state_dict = agent.q_network.state_dict()
            config['model_parameters'] = sum(p.numel() for p in agent.q_network.parameters() if hasattr(agent.q_network, 'parameters'))
            config['model_layers'] = len([k for k in state_dict.keys() if 'weight' in k])
        
        return config
    
    def validate_and_deploy_checkpoint(
        self,
        checkpoint_id: str,
        force_deploy: bool = False
    ) -> bool:
        """
        Validate checkpoint and deploy if it passes all checks
        
        Args:
            checkpoint_id: ID of checkpoint to validate and deploy
            force_deploy: Force deployment even if validation fails
            
        Returns:
            True if deployment successful
        """
        
        logger.info(f"Validating GAELP checkpoint {checkpoint_id} for production deployment")
        
        try:
            # Use helper function for validation check
            can_deploy, message = validate_checkpoint_before_deployment(
                self.checkpoint_manager, checkpoint_id
            )
            
            logger.info(f"Validation result: {message}")
            
            if can_deploy or force_deploy:
                # Deploy to production
                deploy_success = self.checkpoint_manager.deploy_checkpoint(
                    checkpoint_id, force=force_deploy
                )
                
                if deploy_success:
                    logger.info(f"✅ GAELP model {checkpoint_id} deployed to production")
                    
                    # Log production deployment
                    self._log_production_deployment(checkpoint_id)
                    
                    return True
                else:
                    logger.error(f"❌ Failed to deploy GAELP model {checkpoint_id}")
                    return False
            else:
                logger.warning(f"⚠️  GAELP checkpoint {checkpoint_id} failed validation - not deploying")
                return False
                
        except Exception as e:
            logger.error(f"Error during validation and deployment: {e}")
            return False
    
    def _log_production_deployment(self, checkpoint_id: str):
        """Log production deployment event"""
        try:
            status = self.checkpoint_manager.get_checkpoint_status(checkpoint_id)
            
            deployment_log = {
                'event': 'production_deployment',
                'checkpoint_id': checkpoint_id,
                'model_version': status['model_version'],
                'deployed_at': datetime.now().isoformat(),
                'validation_metrics': status['validation_metrics'],
                'system': 'GAELP'
            }
            
            # Save deployment log
            log_file = f"gaelp_production_deployments.json"
            try:
                import json
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []
            
            logs.append(deployment_log)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"Production deployment logged to {log_file}")
            
        except Exception as e:
            logger.warning(f"Failed to log production deployment: {e}")
    
    def handle_production_emergency(self) -> bool:
        """
        Handle production emergency by rolling back to last known good model
        
        Returns:
            True if emergency rollback successful
        """
        
        logger.warning("🚨 PRODUCTION EMERGENCY - Initiating rollback")
        
        try:
            # Use emergency rollback helper
            rollback_success = emergency_rollback_if_needed(self.checkpoint_manager)
            
            if rollback_success:
                logger.info("✅ Emergency rollback completed successfully")
                
                # Get current production status
                status = self.checkpoint_manager.get_production_status()
                current_checkpoint = status['current_production_checkpoint']
                
                logger.info(f"Production now running checkpoint: {current_checkpoint}")
                
                return True
            else:
                logger.error("❌ Emergency rollback failed")
                return False
                
        except Exception as e:
            logger.error(f"Emergency rollback error: {e}")
            return False
    
    def get_production_status_report(self) -> Dict[str, Any]:
        """Get comprehensive production status report"""
        
        try:
            # Get basic production status
            prod_status = self.checkpoint_manager.get_production_status()
            
            # Get list of all checkpoints
            all_checkpoints = self.checkpoint_manager.list_checkpoints()
            
            # Get validated checkpoints ready for deployment
            validated_checkpoints = self.checkpoint_manager.list_checkpoints(
                status_filter=ValidationStatus.PASSED
            )
            
            # Create comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'production_status': prod_status,
                'total_checkpoints': len(all_checkpoints),
                'validated_checkpoints': len(validated_checkpoints),
                'recent_checkpoints': all_checkpoints[:5],  # Most recent 5
                'system_health': {
                    'has_production_model': prod_status['current_production_checkpoint'] is not None,
                    'rollback_available': prod_status['rollback_available'],
                    'validation_current': prod_status['last_validation_time'] is not None,
                    'emergency_ready': prod_status['rollback_available']
                }
            }
            
            # Add performance summary if we have production model
            if prod_status['current_production_checkpoint']:
                try:
                    current_status = self.checkpoint_manager.get_checkpoint_status(
                        prod_status['current_production_checkpoint']
                    )
                    
                    report['current_model'] = {
                        'checkpoint_id': prod_status['current_production_checkpoint'],
                        'model_version': current_status['model_version'],
                        'deployed_at': current_status['deployed_at'],
                        'validation_metrics': current_status['validation_metrics']
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not get current model details: {e}")
                    report['current_model'] = None
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate status report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def export_production_report(self, output_path: str = None) -> str:
        """Export comprehensive production report"""
        
        if output_path is None:
            output_path = f"gaelp_production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            report = self.get_production_status_report()
            
            # Add checkpoint validation reports for recent checkpoints
            if 'recent_checkpoints' in report:
                report['checkpoint_details'] = []
                
                for checkpoint in report['recent_checkpoints'][:3]:  # Top 3 recent
                    try:
                        validation_report_path = self.checkpoint_manager.export_validation_report(
                            checkpoint['checkpoint_id']
                        )
                        
                        report['checkpoint_details'].append({
                            'checkpoint_id': checkpoint['checkpoint_id'],
                            'validation_report': validation_report_path
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not export validation report for {checkpoint['checkpoint_id']}: {e}")
            
            # Save comprehensive report
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Production report exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export production report: {e}")
            return ""

# Example integration functions

def integrate_with_existing_training():
    """Example: Integrate with existing GAELP training loop"""
    
    logger.info("Demonstrating GAELP training integration")
    
    # Create integration wrapper
    gaelp_integration = GAELPProductionIntegration()
    
    # Mock training loop
    class MockGAELPAgent:
        def __init__(self):
            self.learning_rate = 0.001
            self.epsilon = 0.1
            self.gamma = 0.99
            self.episodes = 0
            self.q_network = torch.nn.Linear(10, 4)
        
        def state_dict(self):
            return self.q_network.state_dict()
        
        def select_action(self, state):
            return {'action': 0, 'confidence': 0.8}
    
    agent = MockGAELPAgent()
    
    # Simulate training checkpoints
    checkpoints_saved = []
    
    for episode in [100, 200, 300, 400, 500]:
        agent.episodes = episode
        
        # Simulate improving metrics
        training_metrics = {
            'roas': 2.0 + (episode / 500) * 2.0,  # Improving from 2.0 to 4.0
            'conversion_rate': 0.08 + (episode / 500) * 0.07,  # Improving
            'ctr': 0.05 + (episode / 500) * 0.05,  # Improving
            'avg_reward': 500 + episode * 2,
            'epsilon': 1.0 - (episode / 500) * 0.9
        }
        
        # Save checkpoint
        checkpoint_id = gaelp_integration.save_training_checkpoint(
            agent=agent,
            episode=episode,
            training_metrics=training_metrics,
            validate_immediately=True
        )
        
        if checkpoint_id:
            checkpoints_saved.append(checkpoint_id)
            logger.info(f"Training checkpoint {episode}: {checkpoint_id}")
            
            # Deploy best performing checkpoint
            if episode == 500:  # Final checkpoint
                success = gaelp_integration.validate_and_deploy_checkpoint(checkpoint_id)
                if success:
                    logger.info("🚀 Best model deployed to production!")
    
    logger.info(f"Training integration complete. Saved {len(checkpoints_saved)} checkpoints")
    return gaelp_integration, checkpoints_saved

def demonstrate_production_monitoring():
    """Demonstrate production monitoring and emergency handling"""
    
    logger.info("Demonstrating production monitoring")
    
    # Create integration (would use existing in practice)
    gaelp_integration = GAELPProductionIntegration()
    
    # Get production status
    status_report = gaelp_integration.get_production_status_report()
    
    logger.info("Production Status Summary:")
    logger.info(f"  Total checkpoints: {status_report.get('total_checkpoints', 0)}")
    logger.info(f"  Validated checkpoints: {status_report.get('validated_checkpoints', 0)}")
    logger.info(f"  Has production model: {status_report.get('system_health', {}).get('has_production_model', False)}")
    logger.info(f"  Rollback available: {status_report.get('system_health', {}).get('rollback_available', False)}")
    
    # Export comprehensive report
    report_path = gaelp_integration.export_production_report()
    logger.info(f"Detailed report exported to: {report_path}")
    
    # Simulate emergency scenario
    logger.info("\nSimulating production emergency...")
    emergency_handled = gaelp_integration.handle_production_emergency()
    
    if emergency_handled:
        logger.info("✅ Emergency handling successful")
    else:
        logger.warning("⚠️  Emergency handling needs attention")
    
    return gaelp_integration

async def demonstrate_async_integration():
    """Demonstrate async integration with GAELP training orchestrator"""
    
    logger.info("Demonstrating async integration")
    
    gaelp_integration = GAELPProductionIntegration()
    
    # Simulate async training workflow
    class AsyncMockAgent:
        def __init__(self):
            self.learning_rate = 0.001
            self.episodes = 0
            self.q_network = torch.nn.Linear(10, 4)
        
        def state_dict(self):
            return self.q_network.state_dict()
        
        async def select_action(self, state):
            # Simulate async action selection
            await asyncio.sleep(0.001)
            return {'action': 0, 'confidence': 0.8}
    
    agent = AsyncMockAgent()
    
    # Async checkpoint saving
    checkpoint_tasks = []
    
    for episode in range(100, 601, 100):
        agent.episodes = episode
        
        metrics = {
            'roas': 3.0 + episode / 1000,
            'conversion_rate': 0.1 + episode / 5000,
            'async_performance': True
        }
        
        # Create checkpoint (non-async part)
        checkpoint_id = gaelp_integration.save_training_checkpoint(
            agent=agent,
            episode=episode,
            training_metrics=metrics,
            validate_immediately=False  # Validate separately
        )
        
        if checkpoint_id:
            logger.info(f"Async checkpoint created: {checkpoint_id}")
    
    logger.info("Async integration demonstration complete")
    return gaelp_integration

def main():
    """Main integration demonstration"""
    
    logger.info("="*60)
    logger.info("GAELP PRODUCTION CHECKPOINT MANAGER INTEGRATION")
    logger.info("="*60)
    
    try:
        # 1. Training Integration
        logger.info("\n1. Training Integration Demo")
        gaelp_integration, checkpoints = integrate_with_existing_training()
        
        # 2. Production Monitoring
        logger.info("\n2. Production Monitoring Demo")
        demonstrate_production_monitoring()
        
        # 3. Async Integration
        logger.info("\n3. Async Integration Demo")
        asyncio.run(demonstrate_async_integration())
        
        logger.info("\n" + "="*60)
        logger.info("INTEGRATION DEMONSTRATION COMPLETE")
        logger.info("="*60)
        
        logger.info("✅ Production Checkpoint Manager successfully integrated with GAELP")
        logger.info("Key features demonstrated:")
        logger.info("  - Training checkpoint validation")
        logger.info("  - Production deployment safety")
        logger.info("  - Emergency rollback capabilities")
        logger.info("  - Comprehensive monitoring")
        logger.info("  - Async workflow support")
        
        logger.info("\n🚀 Ready for production deployment!")
        
    except Exception as e:
        logger.error(f"Integration demonstration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)

if __name__ == "__main__":
    main()