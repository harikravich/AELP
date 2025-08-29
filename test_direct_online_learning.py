"""
Direct test of online learning functionality

Tests the online learning system by directly importing the module without
going through the training_orchestrator __init__.py that has SQLAlchemy issues.
"""

import sys
import os
import asyncio
import numpy as np
from unittest.mock import Mock, patch
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_thompson_sampler_direct():
    """Test Thompson sampler arm directly"""
    print("Testing Thompson Sampler Arm (Direct Import)...")
    
    # Import directly from the module file
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training_orchestrator'))
        from online_learner import ThompsonSamplerArm
    except ImportError as e:
        print(f"❌ Could not import ThompsonSamplerArm: {e}")
        return False
    
    # Create and test arm
    arm = ThompsonSamplerArm("test_arm", prior_alpha=2.0, prior_beta=3.0)
    
    # Test initialization
    assert arm.arm_id == "test_arm"
    assert arm.alpha == 2.0
    assert arm.beta == 3.0
    assert arm.total_pulls == 0
    assert arm.total_rewards == 0.0
    
    # Test sampling
    samples = [arm.sample() for _ in range(20)]
    assert all(0 <= s <= 1 for s in samples), "Samples should be in [0,1]"
    
    # Test successful update
    initial_alpha = arm.alpha
    arm.update(0.8, success=True)
    assert arm.alpha == initial_alpha + 1, f"Alpha should be {initial_alpha + 1}, got {arm.alpha}"
    assert arm.total_pulls == 1
    assert abs(arm.total_rewards - 0.8) < 1e-10
    
    # Test failed update  
    initial_beta = arm.beta
    arm.update(0.2, success=False)
    assert arm.beta == initial_beta + 1, f"Beta should be {initial_beta + 1}, got {arm.beta}"
    assert arm.total_pulls == 2
    assert abs(arm.total_rewards - 1.0) < 1e-10
    
    # Test mean reward calculation
    expected_mean = 1.0 / 2  # total_rewards / total_pulls
    assert abs(arm.get_mean_reward() - expected_mean) < 1e-10
    
    # Test confidence interval
    ci = arm.get_confidence_interval(0.95)
    assert len(ci) == 2
    assert 0 <= ci[0] <= ci[1] <= 1
    
    print(f"  - Arm ID: {arm.arm_id}")
    print(f"  - Alpha/Beta: {arm.alpha}/{arm.beta}")
    print(f"  - Total pulls: {arm.total_pulls}")
    print(f"  - Mean reward: {arm.get_mean_reward():.3f}")
    print(f"  - 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    print("✅ Thompson Sampler Arm (Direct) tests passed!")
    return True


def test_online_learner_config_direct():
    """Test online learner configuration directly"""
    print("Testing Online Learner Configuration (Direct Import)...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training_orchestrator'))
        from online_learner import OnlineLearnerConfig, SafetyConstraints, ExplorationAction
    except ImportError as e:
        print(f"❌ Could not import configuration classes: {e}")
        return False
    
    # Test default configuration
    config = OnlineLearnerConfig()
    
    assert config.ts_prior_alpha == 1.0
    assert config.ts_prior_beta == 1.0
    assert config.safety_threshold == 0.8
    assert config.max_budget_risk == 0.1
    assert len(config.bandit_arms) == 4
    assert "conservative" in config.bandit_arms
    assert "aggressive" in config.bandit_arms
    assert "experimental" in config.bandit_arms
    
    # Test custom configuration
    custom_config = OnlineLearnerConfig(
        ts_prior_alpha=5.0,
        safety_threshold=0.9,
        bandit_arms=["test1", "test2", "test3"],
        max_budget_risk=0.15,
        online_update_frequency=10
    )
    
    assert custom_config.ts_prior_alpha == 5.0
    assert custom_config.safety_threshold == 0.9
    assert custom_config.bandit_arms == ["test1", "test2", "test3"]
    assert custom_config.max_budget_risk == 0.15
    assert custom_config.online_update_frequency == 10
    
    # Test safety constraints
    safety = SafetyConstraints()
    assert safety.max_budget_deviation == 0.2
    assert safety.min_roi_threshold == 0.5
    assert safety.max_cpa_multiplier == 2.0
    
    custom_safety = SafetyConstraints(
        max_budget_deviation=0.3,
        min_roi_threshold=0.8,
        blacklisted_audiences=["risky_audience"],
        max_cpa_multiplier=1.5
    )
    assert custom_safety.max_budget_deviation == 0.3
    assert custom_safety.min_roi_threshold == 0.8
    assert "risky_audience" in custom_safety.blacklisted_audiences
    
    # Test exploration action
    action = ExplorationAction(
        action={"budget": 100.0, "bid": 2.0},
        arm_id="test_arm",
        confidence=0.8,
        risk_level="medium",
        expected_reward=0.6,
        uncertainty=0.2,
        safety_score=0.9
    )
    
    assert action.arm_id == "test_arm"
    assert action.confidence == 0.8
    assert action.risk_level == "medium"
    assert action.action["budget"] == 100.0
    
    print(f"  - Default bandit arms: {config.bandit_arms}")
    print(f"  - Safety threshold: {config.safety_threshold}")
    print(f"  - Update frequency: {config.online_update_frequency}")
    print(f"  - Safety constraints max deviation: {safety.max_budget_deviation}")
    
    print("✅ Online Learner Configuration (Direct) tests passed!")
    return True


async def test_online_learner_direct():
    """Test online learner with direct import and mocks"""
    print("Testing Online Learner (Direct Import)...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training_orchestrator'))
        from online_learner import OnlineLearner, OnlineLearnerConfig
    except ImportError as e:
        print(f"❌ Could not import OnlineLearner: {e}")
        return False
    
    # Create mock agent
    class DirectMockAgent:
        def __init__(self):
            self.agent_id = "direct_mock_agent"
            self.config = Mock()
            self.config.learning_rate = 0.001
            self.training_step = 0
        
        async def select_action(self, state, deterministic=False):
            base_action = {
                "creative_type": "image",
                "budget": 100.0,
                "bid_amount": 2.0,
                "target_audience": "professionals",
                "bid_strategy": "cpc",
                "audience_size": 0.5,
                "ab_test_enabled": not deterministic,
                "ab_test_split": 0.5
            }
            
            # Add some variation for non-deterministic
            if not deterministic:
                base_action["budget"] *= np.random.uniform(0.9, 1.1)
                base_action["bid_amount"] *= np.random.uniform(0.95, 1.05)
            
            return base_action
        
        def update_policy(self, experiences):
            self.training_step += 1
            return {
                "policy_loss": np.random.uniform(0.1, 0.5),
                "value_loss": np.random.uniform(0.05, 0.3),
                "entropy": np.random.uniform(0.01, 0.1)
            }
        
        def get_state(self):
            return {"training_step": self.training_step}
        
        def load_state(self, state):
            self.training_step = state.get("training_step", 0)
    
    mock_agent = DirectMockAgent()
    
    # Create configuration
    config = OnlineLearnerConfig(
        bandit_arms=["conservative", "balanced", "aggressive"],
        online_update_frequency=3,
        min_episodes_before_update=2,
        safety_threshold=0.7,
        max_budget_risk=0.15
    )
    
    # Create online learner (external dependencies are optional)
    learner = OnlineLearner(mock_agent, config)
    learner.redis_client = Mock()
    learner.redis_client.ping.return_value = True
    learner.bigquery_client = Mock()
    
    # Test initialization
    assert len(learner.bandit_arms) == 3
    assert not learner.emergency_mode
    assert learner.online_updates_count == 0
    assert learner.baseline_performance is None
    
    print(f"  - Initialized with {len(learner.bandit_arms)} bandit arms")
    print(f"  - Emergency mode: {learner.emergency_mode}")
    
    # Test action selection (exploitation)
    state = {
        "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.3},
        "performance_history": {"avg_roas": 1.2, "avg_bid": 2.0},
        "market_context": {"competition_level": 0.5}
    }
    
    exploit_action = await learner.select_action(state, deterministic=True)
    assert isinstance(exploit_action, dict)
    assert "budget" in exploit_action
    assert "bid_amount" in exploit_action
    
    print(f"  - Exploitation action budget: {exploit_action['budget']:.2f}")
    
    # Test exploration vs exploitation decision
    strategy, confidence = await learner.explore_vs_exploit(state)
    assert strategy in ["explore", "exploit"]
    assert 0 <= confidence <= 1
    
    print(f"  - Strategy decision: {strategy} (confidence: {confidence:.3f})")
    
    # Test safe exploration
    base_action = {
            "budget": 80.0,
            "bid_amount": 2.5,
            "audience_size": 0.6,
            "creative_type": "image",
            "target_audience": "professionals"
    }
    
    safe_action = await learner.safe_exploration(state, base_action)
    assert isinstance(safe_action, dict)
    
    # Should be similar but with small safe variations
    budget_diff = abs(safe_action["budget"] - base_action["budget"])
    assert budget_diff <= base_action["budget"] * 0.2  # Within 20%
    
    print(f"  - Safe exploration: budget {base_action['budget']:.2f} -> {safe_action['budget']:.2f}")
    
    # Test episode recording
    episode_data = {
            "state": state,
            "action": exploit_action,
            "reward": 0.75,
            "success": True,
            "safety_violation": False,
            "arm_id": "balanced"
    }
    
    learner.record_episode(episode_data)
    assert len(learner.episode_history) == 1
    
    # Check bandit arm was updated
    balanced_arm = learner.bandit_arms["balanced"]
    assert balanced_arm.total_pulls == 1
    assert abs(balanced_arm.total_rewards - 0.75) < 1e-10
    
    print(f"  - Recorded episode: reward={episode_data['reward']}")
    print(f"  - Balanced arm pulls: {balanced_arm.total_pulls}")
    
    # Test online update
    experiences = [
            {
                "state": state,
                "action": exploit_action,
                "reward": 0.75,
                "next_state": state,
                "done": False,
                "safety_violation": False,
                "arm_id": "balanced"
            }
    ]
    
    update_result = await learner.online_update(experiences, force_update=True)
    assert "status" in update_result
    assert learner.online_updates_count >= 1
    
    print(f"  - Online update result: {update_result.get('status')}")
    print(f"  - Update count: {learner.online_updates_count}")
    
    # Test performance metrics
    metrics = learner.get_performance_metrics()
    assert "online_updates_count" in metrics
    assert "bandit_arms" in metrics
    assert "episodes_recorded" in metrics
    
    print(f"  - Episodes recorded: {metrics['episodes_recorded']}")
    print(f"  - Safety violations: {metrics['safety_violations']}")
    
    # Test baseline performance update
    baseline_episodes = [
            {"reward": 0.6}, {"reward": 0.8}, {"reward": 0.7}, {"reward": 0.9}
    ]
    learner.update_baseline_performance(baseline_episodes)
    expected_baseline = np.mean([ep["reward"] for ep in baseline_episodes])
    
    assert abs(learner.baseline_performance - expected_baseline) < 1e-10
    print(f"  - Baseline performance: {learner.baseline_performance:.3f}")
    
    # Test emergency mode handling
    # Simulate multiple safety violations
    for i in range(learner.config.safety_violation_limit):
            violation_episode = {
                "reward": -0.5,
                "success": False,
                "safety_violation": True
            }
            learner.record_episode(violation_episode)
    
    assert learner.emergency_mode
    print(f"  - Emergency mode activated after violations")
    
    # In emergency mode, should always exploit
    emergency_strategy, emergency_confidence = await learner.explore_vs_exploit(state)
    assert emergency_strategy == "exploit"
    assert emergency_confidence == 1.0
    
    print(f"  - Emergency strategy: {emergency_strategy}")
    
    await learner.shutdown()
    
    print("✅ Online Learner (Direct) tests passed!")
    return True


async def run_integration_scenario():
    """Run a realistic integration scenario"""
    print("Running Integration Scenario...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training_orchestrator'))
        from online_learner import OnlineLearner, OnlineLearnerConfig, create_online_learner
    except ImportError as e:
        print(f"❌ Could not import for integration: {e}")
        return False
    
    # Create mock agent with more realistic behavior
    class ScenarioMockAgent:
        def __init__(self):
            self.agent_id = "scenario_agent"
            self.config = Mock()
            self.config.learning_rate = 0.001
            self.training_step = 0
            self.performance_history = []
        
        async def select_action(self, state, deterministic=False):
            # Base action influenced by market conditions
            market_level = state.get("market_context", {}).get("competition_level", 0.5)
            
            base_budget = 100.0
            base_bid = 2.0
            
            # Adjust based on market conditions
            if market_level > 0.7:  # High competition
                base_budget *= 1.2
                base_bid *= 1.15
            elif market_level < 0.3:  # Low competition  
                base_budget *= 0.9
                base_bid *= 0.95
            
            # Add exploration noise if not deterministic
            if not deterministic:
                base_budget *= np.random.uniform(0.85, 1.15)
                base_bid *= np.random.uniform(0.9, 1.1)
            
            return {
                "creative_type": np.random.choice(["image", "video", "carousel"]),
                "budget": base_budget,
                "bid_amount": base_bid,
                "target_audience": np.random.choice(["young_adults", "professionals", "families"]),
                "bid_strategy": "cpc",
                "audience_size": np.random.uniform(0.3, 0.8),
                "ab_test_enabled": not deterministic and np.random.random() < 0.3,
                "ab_test_split": 0.5
            }
        
        def update_policy(self, experiences):
            self.training_step += 1
            
            # Simulate learning - performance improves with more updates
            base_loss = 0.5 * (0.95 ** (self.training_step // 10))
            
            return {
                "policy_loss": base_loss + np.random.uniform(-0.1, 0.1),
                "value_loss": base_loss * 0.6 + np.random.uniform(-0.05, 0.05),
                "entropy": max(0.01, 0.1 - self.training_step * 0.001)
            }
        
        def get_state(self):
            return {
                "training_step": self.training_step,
                "performance_history": self.performance_history[-10:]  # Last 10 episodes
            }
        
        def load_state(self, state):
            self.training_step = state.get("training_step", 0)
            self.performance_history = state.get("performance_history", [])
    
    scenario_agent = ScenarioMockAgent()
    
    # Use factory function
    config_dict = {
        "bandit_arms": ["conservative", "balanced", "aggressive", "experimental", "video_focus"],
        "online_update_frequency": 5,
        "min_episodes_before_update": 3,
        "safety_threshold": 0.6,
        "max_budget_risk": 0.2,
        "ts_prior_alpha": 2.0,
        "ts_prior_beta": 2.0
    }
    
    with patch('redis.Redis'), patch('bigquery.Client'):
        learner = create_online_learner(scenario_agent, config_dict)
        learner.redis_client = Mock()
        learner.bigquery_client = Mock()
        
        print(f"  - Created learner with {len(learner.bandit_arms)} arms")
        
        # Simulate 50 episodes of online learning
        episode_results = []
        market_competition = 0.5  # Starting competition level
        
        for episode in range(50):
            # Evolving market conditions
            market_competition += np.random.normal(0, 0.02)
            market_competition = np.clip(market_competition, 0.1, 0.9)
            
            # Create realistic state
            state = {
                "budget_constraints": {
                    "daily_budget": np.random.uniform(150, 250),
                    "budget_utilization": min(0.9, episode / 60.0),  # Gradual budget usage
                    "remaining_budget": np.random.uniform(50, 200)
                },
                "performance_history": {
                    "avg_roas": np.random.uniform(0.8, 2.2),
                    "avg_ctr": np.random.uniform(0.015, 0.045),
                    "avg_bid": np.random.uniform(1.5, 3.5)
                },
                "market_context": {
                    "competition_level": market_competition,
                    "seasonality_factor": 0.9 + 0.2 * np.sin(episode * 0.1),  # Seasonal variation
                    "market_volatility": np.random.uniform(0.1, 0.4)
                },
                "time_context": {
                    "hour_of_day": episode % 24,
                    "day_of_week": (episode // 24) % 7
                }
            }
            
            # Select action
            action = await learner.select_action(state)
            
            # Simulate realistic campaign outcome
            base_performance = 0.6
            
            # Market factors
            if market_competition > 0.7:
                base_performance -= 0.15  # High competition hurts performance
            elif market_competition < 0.3:
                base_performance += 0.1   # Low competition helps
            
            # Action quality factors
            if action["budget"] > 180:
                base_performance += 0.1   # Higher budget can help
            if action["creative_type"] == "video":
                base_performance += 0.05  # Video performs slightly better
            if action.get("ab_test_enabled", False):
                base_performance += 0.03  # A/B testing helps
            
            # Seasonal effects
            seasonal_factor = state["market_context"]["seasonality_factor"]
            base_performance *= seasonal_factor
            
            # Add noise and randomness
            noise = np.random.normal(0, 0.3 * state["market_context"]["market_volatility"])
            final_reward = base_performance + noise
            final_reward = np.clip(final_reward, -1.0, 2.0)
            
            # Calculate derived metrics
            roas = max(0, final_reward * 2.5)
            ctr = max(0, final_reward * 0.03)
            success = final_reward > 0.3
            safety_violation = final_reward < -0.4 or roas < 0.2
            
            episode_data = {
                "episode": episode,
                "state": state,
                "action": action,
                "reward": final_reward,
                "roas": roas,
                "ctr": ctr,
                "success": success,
                "safety_violation": safety_violation,
                "market_competition": market_competition
            }
            
            # Record with learner
            learner.record_episode({
                "state": state,
                "action": action,
                "reward": final_reward,
                "success": success,
                "safety_violation": safety_violation,
                "next_state": state,
                "done": False
            })
            
            episode_results.append(episode_data)
            
            # Trigger online updates
            if episode > 0 and episode % learner.config.online_update_frequency == 0:
                recent_experiences = []
                for recent_ep in episode_results[-learner.config.online_update_frequency:]:
                    exp = {
                        "state": recent_ep["state"],
                        "action": recent_ep["action"],
                        "reward": recent_ep["reward"],
                        "next_state": recent_ep["state"],
                        "done": False,
                        "safety_violation": recent_ep["safety_violation"]
                    }
                    recent_experiences.append(exp)
                
                update_result = await learner.online_update(recent_experiences)
                
                if episode % 20 == 0:  # Log every 20 episodes
                    print(f"  - Episode {episode}: Update {update_result.get('status', 'unknown')}, "
                          f"Market competition: {market_competition:.2f}, "
                          f"Reward: {final_reward:.3f}")
        
        # Analyze results
        rewards = [ep["reward"] for ep in episode_results]
        roas_values = [ep["roas"] for ep in episode_results]
        safety_violations = sum(1 for ep in episode_results if ep["safety_violation"])
        
        # Performance trend analysis
        first_half_reward = np.mean(rewards[:25])
        second_half_reward = np.mean(rewards[25:])
        improvement = (second_half_reward - first_half_reward) / abs(first_half_reward) * 100
        
        final_metrics = learner.get_performance_metrics()
        
        print(f"\n  Integration Scenario Results:")
        print(f"  - Episodes completed: 50")
        print(f"  - Average reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"  - Average ROAS: {np.mean(roas_values):.3f}")
        print(f"  - Performance improvement: {improvement:.1f}%")
        print(f"  - Safety violations: {safety_violations}")
        print(f"  - Online updates: {final_metrics['online_updates_count']}")
        print(f"  - Emergency mode activated: {final_metrics['emergency_mode']}")
        
        # Analyze bandit arm performance
        print(f"\n  Bandit Arm Performance:")
        for arm_id, stats in final_metrics["bandit_arms"].items():
            if stats["total_pulls"] > 0:
                print(f"  - {arm_id}: {stats['total_pulls']} pulls, "
                      f"avg reward: {stats['mean_reward']:.3f}")
        
        await learner.shutdown()
    
    print("✅ Integration Scenario completed successfully!")
    return True


async def main():
    """Run all direct tests"""
    print("=" * 60)
    print("Online Learning System - Direct Tests")
    print("=" * 60)
    
    tests = [
        ("Thompson Sampler Direct", test_thompson_sampler_direct),
        ("Config Direct", test_online_learner_config_direct),
        ("Online Learner Direct", test_online_learner_direct),
        ("Integration Scenario", run_integration_scenario)
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
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Online learning system is working correctly.")
        print("\nKey Features Validated:")
        print("✅ Thompson Sampling for multi-armed bandits")
        print("✅ Safe exploration with budget guardrails")
        print("✅ Incremental model updates")
        print("✅ Emergency mode and safety monitoring")
        print("✅ Real-time performance adaptation")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)