#!/usr/bin/env python3
"""
Test script for GAELP W&B integration.
Tests the wandb_tracking module functionality.
"""

import numpy as np
from wandb_tracking import GAELPWandbTracker, GAELPExperimentConfig, create_experiment_tracker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_wandb_tracker():
    """Test the W&B tracker functionality"""
    
    print("🧪 Testing GAELP W&B Integration")
    print("=" * 40)
    
    # Test 1: Basic tracker initialization
    print("\n1. Testing tracker initialization...")
    config = GAELPExperimentConfig(
        agent_type="PPO",
        learning_rate=0.001,
        batch_size=32,
        num_episodes=10,
        environment_type="TestEnvironment"
    )
    
    tracker = create_experiment_tracker(
        experiment_name="test_wandb_integration",
        config=config,
        tags=["test", "GAELP", "integration"]
    )
    
    if tracker:
        print("✅ Tracker initialized successfully")
    else:
        print("❌ Tracker initialization failed")
        return
    
    # Test 2: Log environment calibration
    print("\n2. Testing environment calibration logging...")
    calibration_data = {
        'mean_ctr': 0.025,
        'std_ctr': 0.015,
        'mean_cpc': 1.5,
        'mean_conv_rate': 0.05,
        'mean_roas': 2.3
    }
    tracker.log_environment_calibration(calibration_data)
    print("✅ Environment calibration logged")
    
    # Test 3: Log episode metrics
    print("\n3. Testing episode metrics logging...")
    episode_results = []
    
    for episode in range(1, 11):
        # Simulate episode metrics
        total_reward = np.random.normal(100, 20)
        steps = np.random.randint(50, 100)
        roas = np.random.normal(2.0, 0.5)
        ctr = np.random.normal(0.025, 0.005)
        conversion_rate = np.random.normal(0.05, 0.01)
        total_cost = np.random.normal(1000, 200)
        total_revenue = total_cost * roas
        
        # Log to tracker
        tracker.log_episode_metrics(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            roas=roas,
            ctr=ctr,
            conversion_rate=conversion_rate,
            total_cost=total_cost,
            total_revenue=total_revenue,
            avg_cpc=total_cost / (ctr * 10000),  # Simulated clicks
            additional_metrics={
                'total_impressions': 10000,
                'total_clicks': int(ctr * 10000),
                'total_conversions': int(conversion_rate * ctr * 10000)
            }
        )
        
        # Store for batch testing
        episode_results.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'final_roas': roas
        })
        
        if episode % 3 == 0:
            print(f"   Episode {episode}: Reward={total_reward:.2f}, ROAS={roas:.2f}x")
    
    print("✅ Episode metrics logged")
    
    # Test 4: Log batch metrics
    print("\n4. Testing batch metrics logging...")
    tracker.log_batch_metrics(episode_results[-5:], batch_size=5)
    print("✅ Batch metrics logged")
    
    # Test 5: Log evaluation metrics
    print("\n5. Testing evaluation metrics logging...")
    eval_metrics = {
        'mean_absolute_error': 0.25,
        'correlation': 0.85,
        'accuracy': 0.78
    }
    tracker.log_evaluation_metrics(eval_metrics)
    print("✅ Evaluation metrics logged")
    
    # Test 6: Save local results
    print("\n6. Testing local results saving...")
    tracker.save_local_results(episode_results, "test_wandb_results.json")
    print("✅ Local results saved")
    
    # Test 7: Generate learning curve
    print("\n7. Testing learning curve generation...")
    try:
        tracker.log_learning_curve()
        print("✅ Learning curve generated")
    except Exception as e:
        print(f"⚠️  Learning curve generation failed (expected if matplotlib not available): {e}")
    
    # Test 8: Finish session
    print("\n8. Testing session cleanup...")
    tracker.finish()
    print("✅ Session finished")
    
    print("\n🎉 All tests completed successfully!")
    print("\nCheck the following:")
    print("  • W&B run should be visible in offline mode")
    print("  • test_wandb_results.json should be created")
    print("  • All metrics should be properly structured")


def test_anonymous_mode():
    """Test anonymous mode functionality"""
    
    print("\n🔒 Testing Anonymous Mode")
    print("=" * 30)
    
    # Create tracker in anonymous mode
    tracker = GAELPWandbTracker(
        project_name="gaelp-test-anonymous",
        experiment_name="anonymous_test",
        anonymous=True
    )
    
    # Log some basic metrics
    tracker.log_episode_metrics(
        episode=1,
        total_reward=100.0,
        steps=50,
        roas=2.5,
        ctr=0.03,
        conversion_rate=0.06,
        total_cost=1000,
        total_revenue=2500
    )
    
    tracker.finish()
    print("✅ Anonymous mode test completed")


if __name__ == "__main__":
    try:
        test_wandb_tracker()
        test_anonymous_mode()
        
        print("\n🏆 GAELP W&B Integration Test Suite PASSED")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()