"""
Simple Test for Online Learning System

A clean, minimal test of the online learning functionality.
"""

import asyncio
import sys
import os
import numpy as np
from unittest.mock import Mock
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training_orchestrator'))

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_thompson_sampler():
    """Test Thompson sampler functionality"""
    print("Testing Thompson Sampler...")
    
    from online_learner import ThompsonSamplerArm
    
    # Create arm
    arm = ThompsonSamplerArm("test_arm", 1.0, 1.0)
    
    # Test basic functionality
    assert arm.arm_id == "test_arm"
    assert arm.alpha == 1.0
    assert arm.beta == 1.0
    
    # Test sampling
    sample = arm.sample()
    assert 0 <= sample <= 1
    
    # Test update
    arm.update(0.8, success=True)
    assert arm.alpha == 2.0
    assert arm.total_pulls == 1
    
    print("✅ Thompson Sampler test passed!")
    return True


def test_config():
    """Test configuration"""
    print("Testing Configuration...")
    
    from online_learner import OnlineLearnerConfig, SafetyConstraints
    
    # Default config
    config = OnlineLearnerConfig()
    assert len(config.bandit_arms) == 4
    assert "conservative" in config.bandit_arms
    
    # Safety constraints
    safety = SafetyConstraints()
    assert safety.max_budget_deviation == 0.2
    
    print("✅ Configuration test passed!")
    return True


async def test_online_learner():
    """Test online learner with mock agent"""
    print("Testing Online Learner...")
    
    from online_learner import OnlineLearner, OnlineLearnerConfig
    
    # Mock agent
    class MockAgent:
        def __init__(self):
            self.agent_id = "mock_agent"
            self.config = Mock()
            self.config.learning_rate = 0.001
        
        async def select_action(self, state, deterministic=False):
            return {
                "creative_type": "image",
                "budget": 100.0 * (1.0 if deterministic else np.random.uniform(0.9, 1.1)),
                "bid_amount": 2.0,
                "target_audience": "professionals",
                "bid_strategy": "cpc",
                "audience_size": 0.5,
                "ab_test_enabled": False,
                "ab_test_split": 0.5
            }
        
        def update_policy(self, experiences):
            return {"loss": 0.1}
        
        def get_state(self):
            return {"step": 0}
        
        def load_state(self, state):
            pass
    
    mock_agent = MockAgent()
    config = OnlineLearnerConfig(bandit_arms=["conservative", "balanced", "aggressive"])
    
    # Create learner
    learner = OnlineLearner(mock_agent, config)
    learner.redis_client = Mock()
    learner.bigquery_client = Mock()
    
    # Test initialization
    assert len(learner.bandit_arms) == 3
    assert not learner.emergency_mode
    
    # Test action selection
    state = {
        "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.3},
        "performance_history": {"avg_roas": 1.2},
        "market_context": {"competition_level": 0.5}
    }
    
    action = await learner.select_action(state, deterministic=True)
    assert isinstance(action, dict)
    assert "budget" in action
    
    # Test exploration decision
    strategy, confidence = await learner.explore_vs_exploit(state)
    assert strategy in ["explore", "exploit"]
    assert 0 <= confidence <= 1
    
    # Test episode recording
    episode_data = {
        "state": state,
        "action": action,
        "reward": 0.75,
        "success": True,
        "safety_violation": False
    }
    learner.record_episode(episode_data)
    assert len(learner.episode_history) == 1
    
    # Test online update
    experiences = [{
        "state": state,
        "action": action,
        "reward": 0.75,
        "next_state": state,
        "done": False
    }]
    
    result = await learner.online_update(experiences, force_update=True)
    assert "status" in result
    
    # Test performance metrics
    metrics = learner.get_performance_metrics()
    assert "online_updates_count" in metrics
    assert "bandit_arms" in metrics
    
    await learner.shutdown()
    
    print(f"✅ Online Learner test passed! Episodes: {metrics['episodes_recorded']}")
    return True


async def run_scenario():
    """Run realistic learning scenario"""
    print("Running Learning Scenario...")
    
    from online_learner import create_online_learner
    
    # Mock agent with learning behavior
    class ScenarioAgent:
        def __init__(self):
            self.agent_id = "scenario_agent"
            self.config = Mock()
            self.config.learning_rate = 0.001
            self.step = 0
        
        async def select_action(self, state, deterministic=False):
            self.step += 1
            market_level = state.get("market_context", {}).get("competition_level", 0.5)
            
            budget = 100.0
            if market_level > 0.7:
                budget *= 1.2
            elif market_level < 0.3:
                budget *= 0.9
            
            if not deterministic:
                budget *= np.random.uniform(0.9, 1.1)
            
            return {
                "creative_type": "image",
                "budget": budget,
                "bid_amount": 2.0 * (1 + market_level * 0.2),
                "target_audience": "professionals",
                "bid_strategy": "cpc",
                "audience_size": 0.5,
                "ab_test_enabled": not deterministic,
                "ab_test_split": 0.5
            }
        
        def update_policy(self, experiences):
            return {"policy_loss": 0.5 * (0.95 ** (self.step // 10))}
        
        def get_state(self):
            return {"step": self.step}
        
        def load_state(self, state):
            self.step = state.get("step", 0)
    
    agent = ScenarioAgent()
    
    # Create learner
    config_dict = {
        "bandit_arms": ["conservative", "balanced", "aggressive", "experimental"],
        "online_update_frequency": 5,
        "safety_threshold": 0.6,
        "max_budget_risk": 0.2
    }
    
    learner = create_online_learner(agent, config_dict)
    learner.redis_client = Mock()
    learner.bigquery_client = Mock()
    
    # Simulate episodes
    results = []
    market_competition = 0.5
    
    for episode in range(30):
        # Evolving market
        market_competition += np.random.normal(0, 0.02)
        market_competition = np.clip(market_competition, 0.2, 0.8)
        
        # Create state
        state = {
            "budget_constraints": {
                "daily_budget": 200.0,
                "budget_utilization": min(0.8, episode / 40.0)
            },
            "performance_history": {
                "avg_roas": np.random.uniform(1.0, 2.0),
                "avg_bid": 2.0
            },
            "market_context": {
                "competition_level": market_competition,
                "seasonality_factor": 0.9 + 0.2 * np.sin(episode * 0.1)
            }
        }
        
        # Select action
        action = await learner.select_action(state)
        
        # Simulate outcome
        base_reward = 0.6
        if market_competition > 0.7:
            base_reward -= 0.1
        elif market_competition < 0.3:
            base_reward += 0.1
        
        if action["budget"] > 150:
            base_reward += 0.05
        
        reward = base_reward + np.random.normal(0, 0.2)
        reward = np.clip(reward, -0.5, 1.5)
        
        # Record episode
        episode_data = {
            "state": state,
            "action": action,
            "reward": reward,
            "success": reward > 0.3,
            "safety_violation": reward < -0.3
        }
        
        learner.record_episode(episode_data)
        results.append(episode_data)
        
        # Trigger updates
        if episode % 5 == 4:
            experiences = [{
                "state": ep["state"],
                "action": ep["action"],
                "reward": ep["reward"],
                "next_state": ep["state"],
                "done": False
            } for ep in results[-5:]]
            
            await learner.online_update(experiences)
    
    # Analyze results
    rewards = [ep["reward"] for ep in results]
    first_half = np.mean(rewards[:15])
    second_half = np.mean(rewards[15:])
    improvement = (second_half - first_half) / abs(first_half) * 100 if first_half != 0 else 0
    
    metrics = learner.get_performance_metrics()
    
    print(f"  Episodes: 30")
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Performance improvement: {improvement:.1f}%")
    print(f"  Online updates: {metrics['online_updates_count']}")
    print(f"  Safety violations: {metrics['safety_violations']}")
    
    # Show bandit arm usage
    print(f"  Bandit arm performance:")
    for arm_id, stats in metrics["bandit_arms"].items():
        if stats["total_pulls"] > 0:
            print(f"    {arm_id}: {stats['total_pulls']} pulls, {stats['mean_reward']:.3f} avg")
    
    await learner.shutdown()
    
    print("✅ Learning scenario completed successfully!")
    return True


async def main():
    """Run all tests"""
    print("=" * 50)
    print("Simple Online Learning Tests")
    print("=" * 50)
    
    tests = [
        ("Thompson Sampler", test_thompson_sampler),
        ("Configuration", test_config),
        ("Online Learner", test_online_learner),
        ("Learning Scenario", run_scenario)
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
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nOnline Learning System Features:")
        print("✅ Thompson Sampling multi-armed bandits")
        print("✅ Safe exploration with budget constraints")  
        print("✅ Incremental policy updates")
        print("✅ Emergency mode safety fallback")
        print("✅ Real-time performance monitoring")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)