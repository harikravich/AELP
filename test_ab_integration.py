#!/usr/bin/env python3
"""
Test script to verify A/B testing integration in production orchestrator
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ab_testing_integration():
    """Test A/B testing integration with production orchestrator"""
    
    try:
        # Import orchestrator
        from gaelp_production_orchestrator import GAELPProductionOrchestrator, OrchestratorConfig
        
        logger.info("Testing A/B testing integration...")
        
        # Create test config with A/B testing enabled
        config = OrchestratorConfig()
        config.enable_ab_testing = True
        config.enable_rl_training = False  # Don't run full training for test
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
        
        # Check if A/B testing component was initialized
        if 'ab_testing' not in orchestrator.components:
            logger.error("A/B testing component not found")
            return False
        
        ab_testing = orchestrator.components['ab_testing']
        logger.info(f"A/B testing component initialized: {type(ab_testing)}")
        
        # Test creating a policy comparison test
        policy_a = {
            'learning_rate': 0.001,
            'epsilon': 0.1,
            'discount_factor': 0.99
        }
        
        policy_b = {
            'learning_rate': 0.01,
            'epsilon': 0.05,
            'discount_factor': 0.95
        }
        
        test_id = orchestrator.create_policy_comparison_test(
            policy_a, policy_b, "Learning Rate Comparison Test"
        )
        
        if not test_id:
            logger.error("Failed to create policy comparison test")
            return False
        
        logger.info(f"Created A/B test: {test_id}")
        
        # Test variant assignment
        context = {
            'segment': 'test_segment',
            'device': 'mobile',
            'channel': 'organic',
            'hour': 14,
            'day_of_week': 1
        }
        
        variant = ab_testing.assign_variant(test_id, "test_user_1", context)
        logger.info(f"Assigned variant: {variant}")
        
        # Test recording observations
        for i in range(10):
            user_id = f"test_user_{i}"
            variant = ab_testing.assign_variant(test_id, user_id, context)
            
            # Simulate some performance (policy B slightly better)
            if variant == 'policy_a':
                reward = np.random.normal(2.0, 0.5)
            else:
                reward = np.random.normal(2.2, 0.5)
                
            converted = reward > 1.5
            
            ab_testing.record_observation(
                test_id=test_id,
                variant_id=variant,
                user_id=user_id,
                primary_metric_value=reward,
                secondary_metrics={'ctr': 0.02, 'roas': reward * 1.5},
                converted=converted,
                context=context
            )
        
        logger.info("Recorded 10 observations")
        
        # Test getting test status
        status = ab_testing.get_test_status(test_id)
        logger.info(f"Test status: {status['status']}")
        logger.info(f"Variants: {len(status['variants'])}")
        
        for variant in status['variants']:
            logger.info(f"  {variant['variant_id']}: {variant['n_observations']} observations, "
                       f"mean={variant['primary_metric_mean']:.3f}")
        
        # Test A/B test analysis
        if status['variants'][0]['n_observations'] > 0:
            try:
                results = ab_testing.analyze_test(test_id)
                logger.info(f"Analysis results:")
                logger.info(f"  P-value: {results.p_value:.4f}")
                logger.info(f"  Is significant: {results.is_significant}")
                logger.info(f"  Bayesian probability: {results.bayesian_probability:.4f}")
                logger.info(f"  Effect size: {results.effect_size:.4f}")
                logger.info(f"  Recommendation: {results.recommended_action}")
            except Exception as e:
                logger.warning(f"Analysis failed (expected with small sample): {e}")
        
        # Test context creation
        state = np.array([1, 2, 3.5, 0, 1, 2, 1, 0.7, 0.8, 5.0])
        context = orchestrator._create_ab_context_from_state(state)
        logger.info(f"Created context: {context}")
        
        logger.info("✅ A/B testing integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"A/B testing integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ab_testing_integration()
    sys.exit(0 if success else 1)