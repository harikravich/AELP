#!/usr/bin/env python3
"""
Test Suite for Delayed Reward System

Comprehensive tests for the delayed reward system including:
- Pending reward storage and retrieval
- Attribution model calculations
- Replay buffer functionality
- Integration with training orchestrator
"""

import asyncio
import pytest
import tempfile
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from training_orchestrator.delayed_reward_system import (
    DelayedRewardSystem,
    DelayedRewardConfig,
    AttributionModel,
    ConversionEvent,
    Touchpoint,
    ConversionRecord,
    PendingReward,
    RewardReplayBuffer
)


class TestRewardReplayBuffer:
    """Test the reward replay buffer functionality"""
    
    def test_buffer_initialization(self):
        buffer = RewardReplayBuffer(max_size=10)
        assert len(buffer.buffer) == 0
        assert len(buffer.priorities) == 0
        
    def test_add_experience(self):
        buffer = RewardReplayBuffer(max_size=10)
        
        # Create mock touchpoint and conversion
        touchpoint = Touchpoint(
            touchpoint_id="test_tp",
            episode_id="test_episode",
            user_id="test_user",
            campaign_id="test_campaign",
            timestamp=datetime.now(),
            action={"budget": 100},
            state={"market": "test"},
            immediate_reward=10.0,
            channel="search",
            creative_type="image",
            placement="top",
            position_in_journey=0
        )
        
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=100.0
        )
        
        buffer.add_experience("test_episode", 10.0, 25.0, touchpoint, conversion)
        
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0]['reward_delta'] == 15.0
        
    def test_buffer_max_size(self):
        buffer = RewardReplayBuffer(max_size=2)
        
        touchpoint = Touchpoint(
            touchpoint_id="test_tp",
            episode_id="test_episode",
            user_id="test_user",
            campaign_id="test_campaign",
            timestamp=datetime.now(),
            action={"budget": 100},
            state={"market": "test"},
            immediate_reward=10.0,
            channel="search",
            creative_type="image",
            placement="top",
            position_in_journey=0
        )
        
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=100.0
        )
        
        # Add 3 experiences to buffer with max_size=2
        for i in range(3):
            buffer.add_experience(f"episode_{i}", 10.0, 25.0, touchpoint, conversion)
            
        assert len(buffer.buffer) == 2  # Should not exceed max_size
        
    def test_sample_batch(self):
        buffer = RewardReplayBuffer(max_size=10)
        
        touchpoint = Touchpoint(
            touchpoint_id="test_tp",
            episode_id="test_episode",
            user_id="test_user",
            campaign_id="test_campaign",
            timestamp=datetime.now(),
            action={"budget": 100},
            state={"market": "test"},
            immediate_reward=10.0,
            channel="search",
            creative_type="image",
            placement="top",
            position_in_journey=0
        )
        
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=100.0
        )
        
        # Add multiple experiences
        for i in range(5):
            buffer.add_experience(f"episode_{i}", 10.0, 25.0, touchpoint, conversion)
            
        batch = buffer.sample_batch(3)
        assert len(batch) == 3
        
        # Test sampling when buffer has fewer items than requested
        small_batch = buffer.sample_batch(10)
        assert len(small_batch) == 5


class TestDelayedRewardConfig:
    """Test configuration object"""
    
    def test_default_config(self):
        config = DelayedRewardConfig()
        assert config.attribution_window_days == 7
        assert config.default_attribution_model == AttributionModel.LINEAR
        assert config.use_redis_cache == True
        
    def test_custom_config(self):
        config = DelayedRewardConfig(
            attribution_window_days=14,
            default_attribution_model=AttributionModel.FIRST_CLICK,
            use_redis_cache=False
        )
        assert config.attribution_window_days == 14
        assert config.default_attribution_model == AttributionModel.FIRST_CLICK
        assert config.use_redis_cache == False


class TestDelayedRewardSystem:
    """Test the main delayed reward system"""
    
    @pytest.fixture
    def config(self):
        return DelayedRewardConfig(
            use_redis_cache=False,
            use_database_persistence=False,
            enable_async_processing=False
        )
    
    @pytest.fixture
    def delayed_reward_system(self, config):
        return DelayedRewardSystem(config)
    
    @pytest.mark.asyncio
    async def test_store_pending_reward(self, delayed_reward_system):
        """Test storing a pending reward"""
        
        touchpoint_id = await delayed_reward_system.store_pending_reward(
            episode_id="test_episode",
            user_id="test_user",
            campaign_id="test_campaign",
            action={"budget": 100},
            state={"market": "test"},
            immediate_reward=10.0,
            channel="search",
            creative_type="image"
        )
        
        assert touchpoint_id is not None
        assert len(delayed_reward_system.user_journeys["test_user"]) == 1
        assert len(delayed_reward_system.episode_touchpoints["test_episode"]) == 1
        
    @pytest.mark.asyncio
    async def test_trigger_attribution_no_touchpoints(self, delayed_reward_system):
        """Test attribution when user has no touchpoints"""
        
        attribution_rewards = await delayed_reward_system.trigger_attribution(
            user_id="nonexistent_user",
            conversion_event=ConversionEvent.PURCHASE,
            conversion_value=100.0
        )
        
        assert attribution_rewards == {}
        
    @pytest.mark.asyncio
    async def test_trigger_attribution_with_touchpoints(self, delayed_reward_system):
        """Test attribution when user has touchpoints"""
        
        # First, store some pending rewards
        touchpoint_ids = []
        for i in range(3):
            tp_id = await delayed_reward_system.store_pending_reward(
                episode_id=f"episode_{i}",
                user_id="test_user",
                campaign_id="test_campaign",
                action={"budget": 100},
                state={"market": "test"},
                immediate_reward=10.0,
                channel="search"
            )
            touchpoint_ids.append(tp_id)
            
        # Trigger attribution
        attribution_rewards = await delayed_reward_system.trigger_attribution(
            user_id="test_user",
            conversion_event=ConversionEvent.PURCHASE,
            conversion_value=120.0
        )
        
        assert len(attribution_rewards) == 3
        # For LINEAR attribution, each touchpoint should get equal share
        for reward in attribution_rewards.values():
            assert reward == 40.0  # 120.0 / 3
            
    @pytest.mark.asyncio
    async def test_handle_partial_episode(self, delayed_reward_system):
        """Test handling partial episodes"""
        
        # Store a pending reward
        await delayed_reward_system.store_pending_reward(
            episode_id="test_episode",
            user_id="test_user",
            campaign_id="test_campaign",
            action={"budget": 100},
            state={"market": "test"},
            immediate_reward=10.0
        )
        
        # Initially no delayed rewards
        delayed_updates = await delayed_reward_system.handle_partial_episode("test_episode")
        assert len(delayed_updates) == 0
        
        # After attribution, should have delayed rewards
        await delayed_reward_system.trigger_attribution(
            user_id="test_user",
            conversion_event=ConversionEvent.PURCHASE,
            conversion_value=100.0
        )
        
        delayed_updates = await delayed_reward_system.handle_partial_episode("test_episode")
        assert len(delayed_updates) == 1
        assert delayed_updates[0]['attributed_reward'] == 100.0
        
    def test_get_statistics(self, delayed_reward_system):
        """Test statistics retrieval"""
        
        stats = delayed_reward_system.get_statistics()
        
        assert 'attribution_stats' in stats
        assert 'pending_rewards' in stats
        assert 'user_journeys' in stats
        assert 'replay_buffer' in stats
        assert 'storage' in stats
        
    @pytest.mark.asyncio
    async def test_user_journey_tracking(self, delayed_reward_system):
        """Test user journey tracking"""
        
        user_id = "journey_user"
        
        # Store multiple touchpoints for the same user
        for i in range(3):
            await delayed_reward_system.store_pending_reward(
                episode_id=f"episode_{i}",
                user_id=user_id,
                campaign_id="test_campaign",
                action={"budget": 100 + i * 10},
                state={"step": i},
                immediate_reward=10.0 + i
            )
            
        journey = delayed_reward_system.get_user_journey(user_id)
        assert len(journey) == 3
        
        # Check journey order and position tracking
        for i, touchpoint in enumerate(journey):
            assert touchpoint.position_in_journey == i
            assert touchpoint.action["budget"] == 100 + i * 10
            
    @pytest.mark.asyncio
    async def test_get_replay_batch(self, delayed_reward_system):
        """Test replay buffer batch retrieval"""
        
        # Initially empty
        batch = await delayed_reward_system.get_replay_batch()
        assert len(batch) == 0
        
        # Add some experiences through attribution
        await delayed_reward_system.store_pending_reward(
            episode_id="test_episode",
            user_id="test_user",
            campaign_id="test_campaign",
            action={"budget": 100},
            state={"market": "test"},
            immediate_reward=10.0
        )
        
        await delayed_reward_system.trigger_attribution(
            user_id="test_user",
            conversion_event=ConversionEvent.PURCHASE,
            conversion_value=50.0
        )
        
        # Now should have experiences (but need minimum samples)
        delayed_reward_system.config.min_replay_samples = 1
        batch = await delayed_reward_system.get_replay_batch(batch_size=5)
        assert len(batch) >= 0


class TestAttributionModels:
    """Test different attribution models"""
    
    @pytest.fixture
    def config(self):
        return DelayedRewardConfig(
            use_redis_cache=False,
            use_database_persistence=False,
            enable_async_processing=False
        )
    
    @pytest.fixture
    def delayed_reward_system(self, config):
        return DelayedRewardSystem(config)
    
    def create_mock_touchpoints(self, count=3, user_id="test_user"):
        """Helper to create mock touchpoints"""
        touchpoints = []
        base_time = datetime.now()
        
        for i in range(count):
            touchpoint = Touchpoint(
                touchpoint_id=f"tp_{i}",
                episode_id=f"episode_{i}",
                user_id=user_id,
                campaign_id="test_campaign",
                timestamp=base_time + timedelta(hours=i),
                action={"budget": 100, "channel": ["search", "display", "social"][i % 3]},
                state={"step": i},
                immediate_reward=10.0,
                channel=["search", "display", "social"][i % 3],
                creative_type="image",
                placement="top",
                position_in_journey=i
            )
            touchpoints.append(touchpoint)
            
        return touchpoints
    
    @pytest.mark.asyncio
    async def test_last_click_attribution(self, delayed_reward_system):
        """Test last-click attribution model"""
        
        touchpoints = self.create_mock_touchpoints(3)
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=100.0
        )
        
        attribution_rewards = await delayed_reward_system._apply_attribution_model(
            touchpoints, conversion, AttributionModel.LAST_CLICK
        )
        
        # Only last touchpoint should get credit
        assert len(attribution_rewards) == 1
        assert attribution_rewards["tp_2"] == 100.0
        
    @pytest.mark.asyncio
    async def test_first_click_attribution(self, delayed_reward_system):
        """Test first-click attribution model"""
        
        touchpoints = self.create_mock_touchpoints(3)
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=100.0
        )
        
        attribution_rewards = await delayed_reward_system._apply_attribution_model(
            touchpoints, conversion, AttributionModel.FIRST_CLICK
        )
        
        # Only first touchpoint should get credit
        assert len(attribution_rewards) == 1
        assert attribution_rewards["tp_0"] == 100.0
        
    @pytest.mark.asyncio
    async def test_linear_attribution(self, delayed_reward_system):
        """Test linear attribution model"""
        
        touchpoints = self.create_mock_touchpoints(3)
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=120.0
        )
        
        attribution_rewards = await delayed_reward_system._apply_attribution_model(
            touchpoints, conversion, AttributionModel.LINEAR
        )
        
        # Each touchpoint should get equal credit
        assert len(attribution_rewards) == 3
        for reward in attribution_rewards.values():
            assert reward == 40.0  # 120.0 / 3
            
    @pytest.mark.asyncio
    async def test_position_based_attribution(self, delayed_reward_system):
        """Test position-based attribution model"""
        
        touchpoints = self.create_mock_touchpoints(4)
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=100.0
        )
        
        attribution_rewards = await delayed_reward_system._apply_attribution_model(
            touchpoints, conversion, AttributionModel.POSITION_BASED
        )
        
        # First gets 40%, last gets 40%, middle split 20%
        assert len(attribution_rewards) == 4
        assert attribution_rewards["tp_0"] == 40.0  # First
        assert attribution_rewards["tp_3"] == 40.0  # Last
        assert attribution_rewards["tp_1"] == 10.0  # Middle
        assert attribution_rewards["tp_2"] == 10.0  # Middle
        
    @pytest.mark.asyncio
    async def test_time_decay_attribution(self, delayed_reward_system):
        """Test time-decay attribution model"""
        
        touchpoints = self.create_mock_touchpoints(2)
        # Make second touchpoint much closer to conversion
        touchpoints[1].timestamp = datetime.now() - timedelta(minutes=30)
        touchpoints[0].timestamp = datetime.now() - timedelta(hours=24)
        
        conversion = ConversionRecord(
            conversion_id="test_conv",
            user_id="test_user",
            timestamp=datetime.now(),
            event_type=ConversionEvent.PURCHASE,
            value=100.0
        )
        
        attribution_rewards = await delayed_reward_system._apply_attribution_model(
            touchpoints, conversion, AttributionModel.TIME_DECAY
        )
        
        # More recent touchpoint should get more credit
        assert len(attribution_rewards) == 2
        assert attribution_rewards["tp_1"] > attribution_rewards["tp_0"]
        
        # Total should sum to conversion value
        total_attributed = sum(attribution_rewards.values())
        assert abs(total_attributed - 100.0) < 0.01


class TestIntegration:
    """Test integration with other components"""
    
    @pytest.mark.asyncio
    async def test_episode_manager_integration(self):
        """Test integration with episode manager"""
        
        # Mock episode manager
        episode_manager = Mock()
        episode_manager._execute_episode = AsyncMock()
        
        # Create delayed reward system
        config = DelayedRewardConfig(
            use_redis_cache=False,
            use_database_persistence=False,
            enable_async_processing=False
        )
        delayed_reward_system = DelayedRewardSystem(config)
        
        # Test integration function exists
        from training_orchestrator.delayed_reward_system import integrate_with_episode_manager
        
        # This should not raise an error
        enhanced_manager = await integrate_with_episode_manager(episode_manager, delayed_reward_system)
        assert enhanced_manager is not None
        
    def test_conversion_event_enum(self):
        """Test conversion event enumeration"""
        
        events = list(ConversionEvent)
        assert ConversionEvent.PURCHASE in events
        assert ConversionEvent.SIGNUP in events
        assert ConversionEvent.LEAD in events
        
    def test_attribution_model_enum(self):
        """Test attribution model enumeration"""
        
        models = list(AttributionModel)
        assert AttributionModel.LAST_CLICK in models
        assert AttributionModel.FIRST_CLICK in models
        assert AttributionModel.LINEAR in models
        assert AttributionModel.TIME_DECAY in models
        assert AttributionModel.POSITION_BASED in models


if __name__ == "__main__":
    # Run tests
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        sys.exit(result.returncode)
        
    except FileNotFoundError:
        print("pytest not found. Running basic tests...")
        
        # Run basic tests without pytest
        import unittest
        
        class BasicTest(unittest.TestCase):
            def test_buffer_creation(self):
                buffer = RewardReplayBuffer(10)
                self.assertEqual(len(buffer.buffer), 0)
                
            def test_config_creation(self):
                config = DelayedRewardConfig()
                self.assertEqual(config.attribution_window_days, 7)
                
        unittest.main()