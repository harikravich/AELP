#!/usr/bin/env python3
"""
Test shadow mode integration with GAELP orchestrator
Verifies that shadow mode runs alongside production training
"""

import sys
import logging
import time

# Configure logging to show shadow mode activity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress noisy loggers 
logging.getLogger('discovery_engine').setLevel(logging.ERROR)
logging.getLogger('segment_discovery').setLevel(logging.ERROR)
logging.getLogger('fortified_rl_agent_no_hardcoding').setLevel(logging.WARNING)
logging.getLogger('fortified_environment_no_hardcoding').setLevel(logging.WARNING)
logging.getLogger('auction_gym_integration_fixed').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def test_shadow_mode_integration():
    """Test that shadow mode integrates properly with training"""
    
    try:
        from gaelp_production_orchestrator import GAELPProductionOrchestrator, OrchestratorConfig
        
        # Create config with shadow mode enabled but simplified setup
        config = OrchestratorConfig(
            environment='development',
            dry_run=True,
            enable_shadow_mode=True,
            enable_rl_training=True,  # Enable training to test integration
            enable_online_learning=False,  # Disable to simplify
            enable_google_ads=False,
            enable_ab_testing=False,
            enable_explainability=False
        )
        
        logger.info("🚀 Starting shadow mode integration test...")
        
        # Initialize orchestrator
        orchestrator = GAELPProductionOrchestrator(config)
        
        if not orchestrator.initialize_components():
            logger.error("❌ Failed to initialize components")
            return False
            
        # Check shadow mode is properly set up
        shadow_mode = orchestrator.components.get('shadow_mode')
        if not shadow_mode:
            logger.error("❌ Shadow mode component not found")
            return False
            
        logger.info(f"✅ Shadow mode initialized with {len(shadow_mode.models)} models")
        
        # Test running a single training episode to verify shadow mode integration
        logger.info("🎯 Running single training episode to test shadow mode integration...")
        
        try:
            episode_metrics = orchestrator._run_training_episode(episode=0)
            
            logger.info(f"✅ Training episode completed successfully!")
            logger.info(f"   Episode metrics keys: {list(episode_metrics.keys())}")
            
            # Check if shadow mode metrics are included
            if 'shadow_mode' in episode_metrics:
                shadow_metrics = episode_metrics['shadow_mode']
                logger.info(f"✅ Shadow mode metrics found: {shadow_metrics}")
                
                if shadow_metrics and 'total_shadow_comparisons' in shadow_metrics:
                    comparisons = shadow_metrics['total_shadow_comparisons']
                    logger.info(f"🎉 Shadow mode executed {comparisons} comparisons during training!")
                    
                    if comparisons > 0:
                        avg_divergence = shadow_metrics.get('avg_overall_divergence', 0)
                        logger.info(f"   Average bid divergence: {avg_divergence:.3f}")
                        logger.info(f"   Shadow models active: {shadow_metrics.get('shadow_models_active', [])}")
                        return True
                    else:
                        logger.warning("⚠️ No shadow comparisons made during episode")
                        return False
                else:
                    logger.warning("⚠️ Shadow metrics present but incomplete")
                    return False
            else:
                logger.warning("⚠️ No shadow mode metrics in episode results")
                return False
                
        except Exception as e:
            logger.error(f"❌ Training episode failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("SHADOW MODE INTEGRATION TEST")
    logger.info("=" * 80)
    
    success = test_shadow_mode_integration()
    
    if success:
        logger.info("=" * 80)
        logger.info("🎉 SHADOW MODE INTEGRATION TEST PASSED!")
        logger.info("✅ Shadow mode is successfully wired into production orchestrator")
        logger.info("✅ Parallel testing runs alongside real decisions")
        logger.info("✅ Shadow metrics are captured and reported")
        logger.info("=" * 80)
        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("❌ SHADOW MODE INTEGRATION TEST FAILED!")
        logger.error("Shadow mode is not properly integrated")
        logger.error("=" * 80)
        sys.exit(1)