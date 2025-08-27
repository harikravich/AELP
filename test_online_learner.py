"""
Test Suite for Online Learning System

Comprehensive tests for the online learning capabilities including:
- Thompson sampling bandit optimization
- Safe exploration with guardrails
- Incremental model updates
- Emergency mode handling
- Performance monitoring
"""

import asyncio
import pytest
import numpy as np
import torch
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

from training_orchestrator.online_learner import (
    OnlineLearner,
    OnlineLearnerConfig,
    ThompsonSamplerArm,
    ExplorationAction,
    SafetyConstraints,
    create_online_learner
)
from training_orchestrator.rl_agents.base_agent import BaseRLAgent, AgentConfig


class MockAgent(BaseRLAgent):
    """Mock agent for testing purposes"""
    
    def __init__(self, config: AgentConfig, agent_id: str):
        self.config = config
        self.agent_id = agent_id
        self.training_step = 0
        self.episode_count = 0
        self.total_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {}
        self.logger = logging.getLogger("MockAgent")
    
    def _setup_networks(self):
        """Mock network setup"""
        pass
    
    def _setup_optimizers(self):
        """Mock optimizer setup"""
        pass
    
    async def select_action(self, state, deterministic=False):
        """Mock action selection"""
        return {
            "creative_type": "image",
            "target_audience": "professionals",
            "bid_strategy": "cpc",
            "budget": 100.0,
            "bid_amount": 2.0,
            "audience_size": 0.5,
            "ab_test_enabled": False,
            "ab_test_split": 0.5
        }
    
    def update_policy(self, experiences):
        """Mock policy update"""
        self.training_step += 1
        return {
            "policy_loss": np.random.uniform(0.1, 0.5),
            "value_loss": np.random.uniform(0.1, 0.3),
            "entropy": np.random.uniform(0.01, 0.1)
        }
    
    def get_state(self):
        """Mock state getter"""
        return {
            "training_step": self.training_step,
            "episode_count": self.episode_count
        }
    
    def load_state(self, state):
        """Mock state loader"""
        self.training_step = state.get("training_step", 0)
        self.episode_count = state.get("episode_count", 0)


class TestThompsonSamplerArm:
    """Test Thompson sampler arm functionality"""
    
    def test_arm_initialization(self):
        """Test arm initialization with priors"""
        arm = ThompsonSamplerArm("test_arm", prior_alpha=2.0, prior_beta=3.0)
        
        assert arm.arm_id == "test_arm"
        assert arm.alpha == 2.0
        assert arm.beta == 3.0
        assert arm.total_pulls == 0
        assert arm.total_rewards == 0.0
    
    def test_arm_sampling(self):
        """Test Beta distribution sampling"""
        arm = ThompsonSamplerArm("test_arm", prior_alpha=5.0, prior_beta=5.0)
        
        samples = [arm.sample() for _ in range(100)]
        
        # Samples should be between 0 and 1
        assert all(0 <= s <= 1 for s in samples)
        
        # With equal alpha and beta, mean should be around 0.5
        mean_sample = np.mean(samples)
        assert 0.3 <= mean_sample <= 0.7  # Allow some variance
    
    def test_arm_update_success(self):
        """Test arm update with successful outcomes"""
        arm = ThompsonSamplerArm("test_arm", prior_alpha=1.0, prior_beta=1.0)
        
        # Update with successful outcomes
        arm.update(0.8, success=True)
        arm.update(0.9, success=True)
        
        assert arm.alpha == 3.0  # 1.0 + 2 successes
        assert arm.beta == 1.0   # No failures
        assert arm.total_pulls == 2
        assert arm.total_rewards == 1.7
    
    def test_arm_update_failure(self):
        """Test arm update with failed outcomes"""
        arm = ThompsonSamplerArm("test_arm", prior_alpha=1.0, prior_beta=1.0)
        
        # Update with failed outcomes
        arm.update(0.2, success=False)
        arm.update(0.1, success=False)
        
        assert arm.alpha == 1.0  # No successes
        assert arm.beta == 3.0   # 1.0 + 2 failures
        assert arm.total_pulls == 2
        assert arm.total_rewards == 0.3
    
    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        arm = ThompsonSamplerArm("test_arm", prior_alpha=10.0, prior_beta=5.0)
        
        # Add some data
        for _ in range(20):
            arm.update(0.7, success=True)
        
        lower, upper = arm.get_confidence_interval(confidence=0.95)
        
        assert 0 <= lower < upper <= 1
        assert upper - lower < 0.5  # Should be reasonably tight with data


class TestOnlineLearner:
    """Test online learner functionality"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing"""
        config = AgentConfig(state_dim=50, action_dim=20)
        return MockAgent(config, "test_agent")
    
    @pytest.fixture  
    def online_config(self):
        """Create test configuration"""
        return OnlineLearnerConfig(
            ts_prior_alpha=1.0,
            ts_prior_beta=1.0,
            online_update_frequency=5,
            min_episodes_before_update=3,
            safety_threshold=0.7,
            max_budget_risk=0.2,
            bandit_arms=["conservative", "balanced", "aggressive", "experimental"]
        )
    
    @pytest.fixture
    def online_learner(self, mock_agent, online_config):
        """Create online learner for testing"""
        # Mock Redis to avoid external dependencies
        with patch('training_orchestrator.online_learner.redis.Redis'):
            learner = OnlineLearner(mock_agent, online_config)
            learner.redis_client = Mock()
            learner.redis_client.ping.return_value = True
            return learner
    
    def test_initialization(self, online_learner, online_config):
        """Test online learner initialization"""
        assert len(online_learner.bandit_arms) == 4
        assert all(isinstance(arm, ThompsonSamplerArm) for arm in online_learner.bandit_arms.values())
        assert online_learner.config == online_config
        assert online_learner.online_updates_count == 0
        assert not online_learner.emergency_mode
    
    @pytest.mark.asyncio
    async def test_action_selection_exploitation(self, online_learner):
        """Test action selection in exploitation mode"""
        state = {
            "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.1},
            "performance_history": {"avg_roas": 1.2},
            "market_context": {"competition_level": 0.5}
        }
        
        action = await online_learner.select_action(state, deterministic=True)
        
        assert isinstance(action, dict)
        assert "creative_type" in action
        assert "budget" in action
        assert "bid_amount" in action
    
    @pytest.mark.asyncio
    async def test_exploration_vs_exploitation_decision(self, online_learner):
        """Test exploration vs exploitation decision making"""
        state = {
            "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.3},
            "performance_history": {"avg_roas": 1.2},
            "market_context": {"competition_level": 0.5}
        }
        
        strategy, confidence = await online_learner.explore_vs_exploit(state)
        
        assert strategy in ["explore", "exploit"]
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio  
    async def test_safe_exploration(self, online_learner):
        """Test safe exploration functionality"""
        state = {
            "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.3},
            "performance_history": {"avg_bid": 2.0, "avg_roas": 1.2}
        }
        
        base_action = {
            "budget": 50.0,
            "bid_amount": 2.0,
            "audience_size": 0.5,
            "creative_type": "image"
        }
        
        safe_action = await online_learner.safe_exploration(state, base_action)
        
        # Should be similar to base action but with small variations
        assert safe_action["budget"] != base_action["budget"]  # Should be modified
        assert abs(safe_action["budget"] - base_action["budget"]) <= base_action["budget"] * 0.2
        assert safe_action["creative_type"] == base_action["creative_type"]  # Should preserve type
    
    def test_episode_recording(self, online_learner):
        """Test episode recording and bandit updates"""
        episode_data = {
            "reward": 0.8,
            "success": True,
            "arm_id": "aggressive",
            "safety_violation": False
        }
        
        initial_pulls = online_learner.bandit_arms["aggressive"].total_pulls
        
        online_learner.record_episode(episode_data)
        
        assert len(online_learner.episode_history) == 1
        assert online_learner.bandit_arms["aggressive"].total_pulls == initial_pulls + 1
        assert online_learner.consecutive_violations == 0
    
    def test_safety_violation_handling(self, online_learner):
        """Test safety violation detection and emergency mode"""
        # Record multiple safety violations
        for i in range(online_learner.config.safety_violation_limit):
            episode_data = {
                "reward": -0.5,
                "success": False,
                "safety_violation": True
            }
            online_learner.record_episode(episode_data)
        
        assert online_learner.emergency_mode
        assert online_learner.consecutive_violations >= online_learner.config.safety_violation_limit
    
    @pytest.mark.asyncio
    async def test_online_update_conditions(self, online_learner):
        """Test conditions for triggering online updates"""
        # Not enough episodes yet
        experiences = [{"state": {}, "action": {}, "reward": 0.5}]
        result = await online_learner.online_update(experiences)
        
        assert result.get("status") == "skipped"
        
        # Add minimum episodes
        for i in range(online_learner.config.min_episodes_before_update + 1):
            online_learner.episode_history.append({"reward": 0.5})
        
        # Still shouldn't update due to frequency
        result = await online_learner.online_update(experiences)
        assert result.get("status") in ["skipped", "buffered"]
    
    @pytest.mark.asyncio
    async def test_force_online_update(self, online_learner):
        """Test forced online update"""
        experiences = [
            {
                "state": {"test": "state"},
                "action": {"test": "action"},
                "reward": 0.7,
                "next_state": {"test": "next_state"},
                "done": False
            }
        ]
        
        result = await online_learner.online_update(experiences, force_update=True)
        
        assert result.get("status") == "completed"
        assert online_learner.online_updates_count == 1
    
    def test_performance_metrics(self, online_learner):
        """Test performance metrics collection"""
        # Add some episode history
        for i in range(10):
            online_learner.episode_history.append({
                "reward": np.random.uniform(0, 1),
                "success": True
            })
        
        metrics = online_learner.get_performance_metrics()
        
        assert "online_updates_count" in metrics
        assert "emergency_mode" in metrics
        assert "bandit_arms" in metrics
        assert "recent_performance" in metrics
        
        # Check bandit arm stats
        for arm_id in online_learner.config.bandit_arms:
            assert arm_id in metrics["bandit_arms"]
            assert "mean_reward" in metrics["bandit_arms"][arm_id]
    
    @pytest.mark.asyncio
    async def test_emergency_mode_behavior(self, online_learner):
        """Test behavior in emergency mode"""
        # Trigger emergency mode
        online_learner._trigger_emergency_mode()
        
        state = {
            "budget_constraints": {"daily_budget": 100, "budget_utilization": 0.5},
            "market_context": {"competition_level": 0.8}
        }
        
        # Should always exploit in emergency mode
        strategy, confidence = await online_learner.explore_vs_exploit(state)
        assert strategy == "exploit"
        assert confidence == 1.0
    
    def test_baseline_performance_update(self, online_learner):
        """Test baseline performance tracking"""
        episodes = [
            {"reward": 0.8},
            {"reward": 0.6}, 
            {"reward": 0.9},
            {"reward": 0.7}
        ]
        
        online_learner.update_baseline_performance(episodes)
        
        expected_baseline = np.mean([ep["reward"] for ep in episodes])
        assert abs(online_learner.baseline_performance - expected_baseline) < 1e-6


class TestIntegrationScenarios:
    """Integration tests for realistic online learning scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_online_learning_cycle(self):
        """Test complete online learning cycle from start to finish"""
        # Create agent and learner
        config = AgentConfig(state_dim=30, action_dim=15)
        agent = MockAgent(config, "integration_test_agent")
        
        online_config = OnlineLearnerConfig(
            online_update_frequency=5,
            min_episodes_before_update=3,
            safety_threshold=0.6
        )
        
        with patch('training_orchestrator.online_learner.redis.Redis'):
            learner = OnlineLearner(agent, online_config)
            learner.redis_client = Mock()
        
        # Simulate episodes with different outcomes
        episode_results = []
        
        for episode in range(20):
            # Create realistic state
            state = {
                "budget_constraints": {
                    "daily_budget": 200.0,
                    "budget_utilization": np.random.uniform(0.1, 0.8)
                },
                "performance_history": {
                    "avg_roas": np.random.uniform(0.8, 2.0),
                    "avg_ctr": np.random.uniform(0.01, 0.05),
                    "avg_bid": np.random.uniform(1.0, 5.0)
                },
                "market_context": {
                    "competition_level": np.random.uniform(0.3, 0.9)
                }
            }
            
            # Select action
            action = await learner.select_action(state)
            
            # Simulate episode outcome based on action
            base_reward = 0.5
            if action.get("bid_amount", 2.0) > 3.0:
                base_reward += 0.2  # Higher bids -> better performance
            if action.get("budget", 100.0) > 150.0:
                base_reward += 0.1  # Higher budget -> more opportunities
            
            # Add noise
            reward = base_reward + np.random.normal(0, 0.2)
            reward = np.clip(reward, -1, 2)  # Realistic reward range
            
            episode_data = {
                "state": state,
                "action": action,
                "reward": reward,
                "success": reward > 0.3,
                "safety_violation": reward < -0.3,
                "next_state": state,  # Simplified
                "done": False
            }
            
            # Record episode
            learner.record_episode(episode_data)
            episode_results.append(episode_data)
            
            # Trigger updates periodically
            if episode % 5 == 4:
                update_result = await learner.online_update([episode_data])
                print(f"Episode {episode}: Update result: {update_result}")
        
        # Check final state
        metrics = learner.get_performance_metrics()
        
        assert metrics["episodes_recorded"] == 20
        assert metrics["online_updates_count"] > 0
        
        # All arms should have been updated
        for arm_stats in metrics["bandit_arms"].values():
            assert arm_stats["total_pulls"] > 0
        
        # Should have baseline performance
        assert learner.baseline_performance is not None
        
        print(f"Final metrics: {json.dumps(metrics, indent=2)}")
        
        await learner.shutdown()
    
    @pytest.mark.asyncio
    async def test_budget_constraint_scenarios(self):
        """Test online learning under various budget constraints"""
        config = AgentConfig(state_dim=20, action_dim=10)
        agent = MockAgent(config, "budget_test_agent")
        
        online_config = OnlineLearnerConfig(
            max_budget_risk=0.1,  # Very conservative
            budget_safety_margin=0.3
        )
        
        with patch('training_orchestrator.online_learner.redis.Redis'):
            learner = OnlineLearner(agent, online_config)
            learner.redis_client = Mock()
        
        # Test high budget utilization scenario
        high_utilization_state = {
            "budget_constraints": {
                "daily_budget": 100.0,
                "budget_utilization": 0.95  # 95% used
            }
        }
        
        strategy, confidence = await learner.explore_vs_exploit(high_utilization_state)
        assert strategy == "exploit"  # Should not explore when budget is almost spent
        
        # Test low budget utilization scenario
        low_utilization_state = {
            "budget_constraints": {
                "daily_budget": 100.0,
                "budget_utilization": 0.1  # Only 10% used
            }
        }
        
        strategy, confidence = await learner.explore_vs_exploit(low_utilization_state)
        # Should be more likely to explore when budget is available
        
        await learner.shutdown()


def run_performance_benchmark():
    """Benchmark online learning performance"""
    import time
    
    print("Running online learning performance benchmark...")
    
    config = AgentConfig(state_dim=100, action_dim=50)
    agent = MockAgent(config, "benchmark_agent")
    
    online_config = OnlineLearnerConfig(
        bandit_arms=["conservative", "balanced", "aggressive", "experimental", "custom1", "custom2"],
        online_update_frequency=10
    )
    
    async def benchmark():
        with patch('training_orchestrator.online_learner.redis.Redis'):
            learner = OnlineLearner(agent, online_config)
            learner.redis_client = Mock()
        
        num_episodes = 1000
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = {
                "budget_constraints": {"daily_budget": 100.0, "budget_utilization": 0.3},
                "performance_history": {"avg_roas": 1.2},
                "market_context": {"competition_level": 0.5}
            }
            
            action = await learner.select_action(state)
            
            episode_data = {
                "state": state,
                "action": action,
                "reward": np.random.normal(0.5, 0.2),
                "success": True,
                "safety_violation": False
            }
            
            learner.record_episode(episode_data)
            
            if episode % 50 == 0:
                await learner.online_update([episode_data])
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Processed {num_episodes} episodes in {duration:.2f} seconds")
        print(f"Throughput: {num_episodes/duration:.1f} episodes/second")
        
        metrics = learner.get_performance_metrics()
        print(f"Updates performed: {metrics['online_updates_count']}")
        
        await learner.shutdown()
    
    asyncio.run(benchmark())


if __name__ == "__main__":
    print("Testing Online Learning System...")
    
    # Run basic tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run performance benchmark
    print("\n" + "="*50)
    run_performance_benchmark()
    
    print("\nOnline learning tests completed!")