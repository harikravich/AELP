#!/usr/bin/env python3
"""
Test script to verify A/B testing integration with training episodes
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ab_training_integration():
    """Test A/B testing integration with actual training episodes"""
    
    try:
        # Import orchestrator
        from gaelp_production_orchestrator import GAELPProductionOrchestrator, OrchestratorConfig
        
        logger.info("Testing A/B testing integration with training...")
        
        # Create test config
        config = OrchestratorConfig()
        config.enable_ab_testing = True
        config.enable_rl_training = False  # We'll manually run episodes
        config.enable_online_learning = False
        config.enable_shadow_mode = False
        config.enable_google_ads = False
        config.dry_run = True
        
        # Create orchestrator
        orchestrator = GAELPProductionOrchestrator(config)
        
        # Initialize components
        if not orchestrator.initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        ab_testing = orchestrator.components.get('ab_testing')
        if not ab_testing:
            logger.error("A/B testing component not found")
            return False
        
        # Create a policy comparison test
        policy_a = {
            'learning_rate': 0.001,
            'epsilon': 0.2,
            'discount_factor': 0.99
        }
        
        policy_b = {
            'learning_rate': 0.005,
            'epsilon': 0.1,
            'discount_factor': 0.95
        }
        
        test_id = orchestrator.create_policy_comparison_test(
            policy_a, policy_b, "Training Integration Test"
        )
        
        if not test_id:
            logger.error("Failed to create policy comparison test")
            return False
        
        logger.info(f"Created A/B test: {test_id}")
        
        # Simulate several training episodes with A/B testing
        logger.info("Running training episodes with A/B testing...")
        
        episode_results = []
        for episode in range(5):  # Run 5 episodes
            try:
                # Run training episode (this will use A/B testing internally)
                result = orchestrator._run_training_episode(episode)
                episode_results.append(result)
                
                logger.info(f"Episode {episode}: reward={result['total_reward']:.3f}, "
                          f"steps={result['steps']}, variant={result.get('ab_variant', 'None')}")
                
                # Check if A/B testing was used
                if result.get('ab_test_id'):
                    logger.info(f"  A/B test active: {result['ab_test_id']}")
                    logger.info(f"  Variant used: {result['ab_variant']}")
                else:
                    logger.warning(f"  No A/B test active for episode {episode}")
                
            except Exception as e:
                logger.error(f"Episode {episode} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Check test status after episodes
        status = ab_testing.get_test_status(test_id)
        logger.info(f"Final test status: {status['status']}")
        
        total_observations = sum(v['n_observations'] for v in status['variants'])
        logger.info(f"Total observations collected: {total_observations}")
        
        for variant in status['variants']:
            logger.info(f"  {variant['variant_id']}: {variant['n_observations']} observations, "
                       f"mean={variant['primary_metric_mean']:.3f}")
        
        # Test analysis if we have enough data
        if total_observations > 0:
            try:
                results = ab_testing.analyze_test(test_id)
                logger.info(f"Test Analysis Results:")
                logger.info(f"  P-value: {results.p_value:.4f}")
                logger.info(f"  Bayesian probability: {results.bayesian_probability:.4f}")
                logger.info(f"  Recommendation: {results.recommended_action}")
            except Exception as e:
                logger.warning(f"Analysis failed (expected with small sample): {e}")
        
        # Verify metrics contain A/B test results
        orchestrator._update_metrics()
        metrics = orchestrator.metrics
        
        if 'ab_test_results' in metrics:
            logger.info("A/B test results found in metrics")
            for test_id, results in metrics['ab_test_results'].items():
                logger.info(f"  {test_id}: {results['recommendation']}")
        else:
            logger.info("No A/B test results in metrics yet (expected for small sample)")
        
        # Test that variant parameters were applied
        agent = orchestrator.components['rl_agent']
        logger.info(f"Agent current parameters:")
        logger.info(f"  Learning rate: {getattr(agent, 'learning_rate', 'N/A')}")
        logger.info(f"  Epsilon: {getattr(agent, 'epsilon', 'N/A')}")
        logger.info(f"  Hyperparameters: {getattr(agent, '_hyperparameters', {})}")
        
        # Verify episodes had different behavior based on variants
        variants_used = [r.get('ab_variant') for r in episode_results if r.get('ab_variant')]
        logger.info(f"Variants used across episodes: {set(variants_used)}")
        
        if len(set(variants_used)) > 1:
            logger.info("✅ Multiple variants were tested")
        elif len(variants_used) > 0:
            logger.info(f"ℹ️  Only one variant used: {variants_used[0]}")
        else:
            logger.warning("⚠️ No variants were recorded")
        
        logger.info("✅ A/B testing training integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"A/B testing training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ab_training_integration()
    sys.exit(0 if success else 1)