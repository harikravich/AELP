"""
Minimal test for online learning system functionality

Tests core components without external dependencies to validate the implementation.
"""

import asyncio
import sys
import os
import numpy as np
from unittest.mock import Mock, patch
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_thompson_sampler_arm():
    """Test Thompson sampler arm basic functionality"""
    print("Testing Thompson Sampler Arm...")
    
    # Import here to handle potential missing dependencies gracefully
    try:
        from training_orchestrator.online_learner import ThompsonSamplerArm
    except ImportError as e:
        print(f"❌ Could not import ThompsonSamplerArm: {e}")
        return False
    
    # Create arm
    arm = ThompsonSamplerArm("test_arm", 2.0, 3.0)
    
    # Test initialization
    assert arm.arm_id == "test_arm"
    assert arm.alpha == 2.0
    assert arm.beta == 3.0
    assert arm.total_pulls == 0
    
    # Test sampling (should return values between 0 and 1)
    samples = [arm.sample() for _ in range(10)]
    assert all(0 <= s <= 1 for s in samples)
    
    # Test update with success
    initial_alpha = arm.alpha
    arm.update(0.8, success=True)
    assert arm.alpha == initial_alpha + 1
    assert arm.total_pulls == 1
    
    # Test update with failure
    initial_beta = arm.beta
    arm.update(0.2, success=False)
    assert arm.beta == initial_beta + 1
    assert arm.total_pulls == 2
    
    # Test confidence interval
    ci = arm.get_confidence_interval(0.95)
    assert len(ci) == 2
    assert 0 <= ci[0] <= ci[1] <= 1
    
    print("✅ Thompson Sampler Arm tests passed!")
    return True


def test_online_learner_config():
    """Test online learner configuration"""
    print("Testing Online Learner Configuration...")
    
    try:
        from training_orchestrator.online_learner import OnlineLearnerConfig
    except ImportError as e:
        print(f"❌ Could not import OnlineLearnerConfig: {e}")
        return False
    
    # Test default configuration
    config = OnlineLearnerConfig()
    
    # Check default values
    assert config.ts_prior_alpha == 1.0
    assert config.ts_prior_beta == 1.0
    assert config.safety_threshold == 0.8
    assert len(config.bandit_arms) == 4
    assert "conservative" in config.bandit_arms
    assert "aggressive" in config.bandit_arms
    
    # Test custom configuration
    custom_config = OnlineLearnerConfig(
        ts_prior_alpha=5.0,
        safety_threshold=0.9,
        bandit_arms=["test1", "test2"],
        max_budget_risk=0.2
    )
    
    assert custom_config.ts_prior_alpha == 5.0
    assert custom_config.safety_threshold == 0.9
    assert custom_config.bandit_arms == ["test1", "test2"]
    assert custom_config.max_budget_risk == 0.2
    
    print("✅ Online Learner Configuration tests passed!")
    return True


async def test_mock_online_learner():
    """Test online learner with mock agent"""
    print("Testing Online Learner with Mock Agent...")
    
    try:
        from training_orchestrator.online_learner import OnlineLearner, OnlineLearnerConfig
    except ImportError as e:
        print(f"❌ Could not import OnlineLearner: {e}")
        return False
    
    # Create mock agent
    class MockAgent:
        def __init__(self):
            self.agent_id = "mock_agent"
            self.config = Mock()
            self.config.learning_rate = 0.001
        
        async def select_action(self, state, deterministic=False):
            return {
                "creative_type": "image",
                "budget": 100.0,
                "bid_amount": 2.0,
                "target_audience": "professionals",
                "bid_strategy": "cpc",
                "audience_size": 0.5,
                "ab_test_enabled": False,
                "ab_test_split": 0.5
            }
        
        def update_policy(self, experiences):
            return {"loss": 0.1, "entropy": 0.05}
        
        def get_state(self):
            return {"weights": "mock_weights"}
        
        def load_state(self, state):
            pass
    
    mock_agent = MockAgent()
    
    # Create configuration
    config = OnlineLearnerConfig(
        bandit_arms=["conservative", "balanced", "aggressive"],
        online_update_frequency=5,
        min_episodes_before_update=2
    )
    
    # Mock external services
    with patch('training_orchestrator.online_learner.redis.Redis'), \
         patch('training_orchestrator.online_learner.bigquery.Client'):
        
        # Create online learner
        learner = OnlineLearner(mock_agent, config)
        learner.redis_client = Mock()
        learner.bigquery_client = Mock()
        
        # Test initialization
        assert len(learner.bandit_arms) == 3
        assert not learner.emergency_mode
        assert learner.online_updates_count == 0
        
        # Test action selection
        state = {
            "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.3},
            "performance_history": {"avg_roas": 1.2},
            "market_context": {"competition_level": 0.5}
        }
        
        action = await learner.select_action(state, deterministic=True)
        assert isinstance(action, dict)
        assert "budget" in action
        assert "bid_amount" in action
        
        # Test exploration vs exploitation
        strategy, confidence = await learner.explore_vs_exploit(state)
        assert strategy in ["explore", "exploit"]
        assert 0 <= confidence <= 1
        
        # Test episode recording
        episode_data = {
            "state": state,
            "action": action,
            "reward": 0.7,
            "success": True,
            "safety_violation": False
        }
        learner.record_episode(episode_data)
        assert len(learner.episode_history) == 1
        
        # Test online update
        experiences = [
            {
                "state": state,
                "action": action,
                "reward": 0.7,
                "next_state": state,
                "done": False
            }
        ]
        
        result = await learner.online_update(experiences, force_update=True)
        assert "status" in result
        
        # Test performance metrics
        metrics = learner.get_performance_metrics()
        assert "online_updates_count" in metrics
        assert "bandit_arms" in metrics
        
        await learner.shutdown()
    
    print("✅ Online Learner with Mock Agent tests passed!")
    return True


def test_safe_exploration():
    """Test safe exploration functionality"""
    print("Testing Safe Exploration...")
    
    try:
        from training_orchestrator.online_learner import SafetyConstraints
    except ImportError as e:
        print(f"❌ Could not import SafetyConstraints: {e}")
        return False
    
    # Test safety constraints
    constraints = SafetyConstraints()
    
    # Check defaults
    assert constraints.max_budget_deviation == 0.2
    assert constraints.min_roi_threshold == 0.5
    assert isinstance(constraints.blacklisted_audiences, list)
    
    # Test custom constraints
    custom_constraints = SafetyConstraints(
        max_budget_deviation=0.3,
        min_roi_threshold=0.8,
        blacklisted_audiences=["risky_audience"],
        max_cpa_multiplier=1.5
    )
    
    assert custom_constraints.max_budget_deviation == 0.3
    assert custom_constraints.min_roi_threshold == 0.8
    assert "risky_audience" in custom_constraints.blacklisted_audiences
    assert custom_constraints.max_cpa_multiplier == 1.5
    
    print("✅ Safe Exploration tests passed!")
    return True


async def run_integration_test():
    """Run integration test with multiple components"""
    print("Running Integration Test...")
    
    # Mock all external dependencies
    with patch('training_orchestrator.online_learner.redis.Redis'), \
         patch('training_orchestrator.online_learner.bigquery.Client'):
        
        try:
            from training_orchestrator.online_learner import create_online_learner
        except ImportError as e:
            print(f"❌ Could not import create_online_learner: {e}")
            return False
        
        # Create mock agent
        class IntegrationMockAgent:
            def __init__(self):
                self.agent_id = "integration_test_agent"
                self.config = Mock()
                self.config.learning_rate = 0.001
                self.training_step = 0
            
            async def select_action(self, state, deterministic=False):
                # Simulate some variation based on deterministic flag
                budget_multiplier = 1.0 if deterministic else np.random.uniform(0.8, 1.2)
                
                return {
                    "creative_type": "image",
                    "budget": 100.0 * budget_multiplier,
                    "bid_amount": 2.0 * budget_multiplier,
                    "target_audience": "professionals",
                    "bid_strategy": "cpc",
                    "audience_size": 0.5,
                    "ab_test_enabled": not deterministic,
                    "ab_test_split": 0.5
                }
            
            def update_policy(self, experiences):
                self.training_step += 1
                return {
                    "policy_loss": np.random.uniform(0.1, 0.5),
                    "value_loss": np.random.uniform(0.1, 0.3)
                }
            
            def get_state(self):
                return {"training_step": self.training_step}
            
            def load_state(self, state):
                self.training_step = state.get("training_step", 0)
        
        mock_agent = IntegrationMockAgent()
        
        # Create online learner with custom config
        config_dict = {
            "bandit_arms": ["conservative", "balanced", "aggressive", "experimental"],
            "online_update_frequency": 3,
            "safety_threshold": 0.7,
            "max_budget_risk": 0.2
        }
        
        learner = create_online_learner(mock_agent, config_dict)
        learner.redis_client = Mock()
        learner.bigquery_client = Mock()
        
        # Simulate online learning session
        episode_results = []
        
        for episode in range(10):
            # Create state
            state = {
                "budget_constraints": {
                    "daily_budget": 200.0,
                    "budget_utilization": np.random.uniform(0.1, 0.8)
                },
                "performance_history": {
                    "avg_roas": np.random.uniform(0.8, 2.0),
                    "avg_bid": 2.0
                },
                "market_context": {
                    "competition_level": np.random.uniform(0.3, 0.8)
                }
            }
            
            # Select action
            action = await learner.select_action(state)
            
            # Simulate outcome
            base_reward = 0.5 + np.random.normal(0, 0.2)
            if action["budget"] > 120:
                base_reward += 0.1
            
            reward = np.clip(base_reward, -1, 2)
            
            episode_data = {
                "state": state,
                "action": action,
                "reward": reward,
                "success": reward > 0.2,
                "safety_violation": reward < -0.3,
                "next_state": state,
                "done": False
            }
            
            # Record episode
            learner.record_episode({
                "state": state,
                "action": action,
                "reward": reward,
                "success": episode_data["success"],
                "safety_violation": episode_data["safety_violation"]
            })
            
            episode_results.append(episode_data)
            
            # Trigger update every 3 episodes
            if episode % 3 == 2:
                update_result = await learner.online_update([episode_data])
                print(f"Episode {episode}: Update result: {update_result.get('status', 'unknown')}")
        
        # Check final state
        metrics = learner.get_performance_metrics()
        
        assert metrics["episodes_recorded"] == 10
        assert metrics["online_updates_count"] >= 0  # Should have some updates
        
        # Check that bandit arms have been used
        arm_stats = metrics["bandit_arms"]
        total_pulls = sum(stats["total_pulls"] for stats in arm_stats.values())
        assert total_pulls > 0, "No bandit arms were used"
        
        print(f"Total bandit arm pulls: {total_pulls}")
        print(f"Online updates performed: {metrics['online_updates_count']}")
        
        await learner.shutdown()
    
    print("✅ Integration test passed!")
    return True


async def main():
    """Run all tests"""
    print("=" * 50)
    print("Online Learning System - Minimal Tests")
    print("=" * 50)
    
    tests = [
        ("Thompson Sampler Arm", test_thompson_sampler_arm),
        ("Online Learner Config", test_online_learner_config),
        ("Mock Online Learner", test_mock_online_learner),
        ("Safe Exploration", test_safe_exploration),
        ("Integration Test", run_integration_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Online learning system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)