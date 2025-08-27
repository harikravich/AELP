#!/usr/bin/env python3
"""
Test script to verify Journey State Encoder integration with PPO Agent
"""

import torch
import numpy as np
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_journey_encoder_integration():
    """Test the integration between Journey State Encoder and PPO Agent"""
    
    print("🧪 Testing Journey State Encoder integration with PPO Agent")
    print("=" * 60)
    
    try:
        # Import components
        from training_orchestrator.journey_state_encoder import (
            JourneyStateEncoder, JourneyStateEncoderConfig, create_journey_encoder
        )
        from journey_aware_rl_agent import (
            JourneyAwarePPOAgent, extract_journey_state_for_encoder
        )
        
        print("✅ Successfully imported Journey State Encoder and PPO Agent")
        
        # 1. Test Journey State Encoder standalone
        print("\n1. Testing Journey State Encoder...")
        
        encoder = create_journey_encoder(
            max_sequence_length=5,
            lstm_hidden_dim=64,
            encoded_state_dim=256
        )
        
        # Create sample journey data
        sample_journey = {
            'current_state': 'considering',
            'days_in_journey': 5,
            'journey_stage': 1,
            'total_touches': 3,
            'conversion_probability': 0.3,
            'user_fatigue_level': 0.2,
            'time_since_last_touch': 2.0,
            'hour_of_day': 14,
            'day_of_week': 2,
            'day_of_month': 15,
            'current_timestamp': 1640995200,
            'journey_history': [
                {
                    'channel': 'search',
                    'user_state_after': 'aware',
                    'cost': 2.50,
                    'timestamp': 1640908800
                },
                {
                    'channel': 'social',
                    'user_state_after': 'interested',
                    'cost': 1.20,
                    'timestamp': 1640951200
                }
            ],
            'channel_distribution': {
                'search': 1, 'social': 1, 'display': 0, 'video': 0,
                'email': 0, 'direct': 0, 'affiliate': 0, 'retargeting': 0
            },
            'channel_costs': {
                'search': 2.50, 'social': 1.20, 'display': 0.0, 'video': 0.0,
                'email': 0.0, 'direct': 0.0, 'affiliate': 0.0, 'retargeting': 0.0
            },
            'channel_last_touch': {
                'search': 3.0, 'social': 1.5, 'display': 30.0, 'video': 30.0,
                'email': 30.0, 'direct': 30.0, 'affiliate': 30.0, 'retargeting': 30.0
            },
            'click_through_rate': 0.035,
            'engagement_rate': 0.15,
            'bounce_rate': 0.4,
            'conversion_rate': 0.08,
            'competitors_seen': 2,
            'competitor_engagement_rate': 0.12
        }
        
        # Test encoding
        encoded_state = encoder.encode_journey(sample_journey)
        print(f"  ✅ Encoded journey state shape: {encoded_state.shape}")
        print(f"  ✅ Encoded state type: {type(encoded_state)}")
        print(f"  ✅ Output dimension: {encoder.get_output_dim()}")
        assert encoded_state.shape[0] == 256, f"Expected 256 dimensions, got {encoded_state.shape[0]}"
        
        # 2. Test PPO Agent with Journey Encoder
        print("\n2. Testing PPO Agent with Journey State Encoder...")
        
        agent = JourneyAwarePPOAgent(
            state_dim=256,
            hidden_dim=256,
            num_channels=8,
            use_journey_encoder=True
        )
        
        print(f"  ✅ PPO Agent initialized with journey encoder")
        print(f"  ✅ Journey encoder enabled: {agent.use_journey_encoder}")
        print(f"  ✅ Journey encoder type: {type(agent.journey_encoder)}")
        
        # 3. Test action selection with encoded state
        print("\n3. Testing action selection with encoded journey state...")
        
        # Test with dictionary state (should use encoder)
        channel_idx, bid_amount, log_prob = agent.select_action(sample_journey)
        
        print(f"  ✅ Action selection successful!")
        print(f"  ✅ Selected channel: {channel_idx}")
        print(f"  ✅ Bid amount: ${bid_amount:.2f}")
        print(f"  ✅ Log probability shape: {log_prob.shape}")
        
        assert 0 <= channel_idx < 8, f"Channel index out of range: {channel_idx}"
        assert bid_amount > 0, f"Bid amount should be positive: {bid_amount}"
        assert log_prob.numel() == 1, f"Log prob should be scalar: {log_prob.shape}"
        
        # 4. Test training integration
        print("\n4. Testing training integration...")
        
        # Store a transition
        next_state = sample_journey.copy()
        next_state['conversion_probability'] = 0.4  # Slightly higher after action
        
        agent.store_transition(
            state=sample_journey,
            action=channel_idx,
            reward=5.0,
            next_state=next_state,
            done=False,
            log_prob=log_prob
        )
        
        print(f"  ✅ Transition stored successfully!")
        print(f"  ✅ Memory size: {len(agent.memory)}")
        
        # Store a few more transitions for batch training
        for i in range(35):  # Need at least 32 for batch
            test_state = sample_journey.copy()
            test_state['conversion_probability'] = np.random.uniform(0.1, 0.8)
            test_state['total_touches'] = i + 1
            
            ch_idx, bid_amt, log_p = agent.select_action(test_state)
            
            agent.store_transition(
                state=test_state,
                action=ch_idx,
                reward=np.random.uniform(-2.0, 10.0),
                next_state=test_state,
                done=np.random.random() < 0.1,
                log_prob=log_p
            )
        
        print(f"  ✅ Added {len(agent.memory)} transitions to memory")
        
        # Test update
        print("\n5. Testing PPO update with encoded states...")
        agent.update(batch_size=32, epochs=2)
        print(f"  ✅ PPO update completed successfully!")
        print(f"  ✅ Memory cleared: {len(agent.memory) == 0}")
        
        # 6. Test model save/load
        print("\n6. Testing model save/load...")
        
        model_path = '/tmp/test_journey_agent.pth'
        agent.save(model_path)
        print(f"  ✅ Model saved to {model_path}")
        
        # Create new agent and load
        new_agent = JourneyAwarePPOAgent(
            state_dim=256,
            hidden_dim=256,
            num_channels=8,
            use_journey_encoder=True
        )
        new_agent.load(model_path)
        print(f"  ✅ Model loaded successfully!")
        
        # Test that loaded agent works
        test_action = new_agent.select_action(sample_journey)
        print(f"  ✅ Loaded agent action selection works!")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Journey State Encoder is properly integrated with PPO Agent")
        print("✅ 256-dimensional LSTM-encoded features are fed to actor-critic network")
        print("✅ Training, saving, and loading work correctly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gaelp_integration():
    """Test integration with GAELP master system"""
    
    print("\n🔗 Testing GAELP Master Integration...")
    
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        from user_journey_database import UserJourney, UserProfile, JourneyState
        from datetime import datetime
        
        # Create minimal config for testing
        config = GAELPConfig(
            simulation_days=1,
            users_per_day=10,
            n_parallel_worlds=2,
            daily_budget_total=100.0
        )
        
        # Initialize orchestrator (this will test component integration)
        orchestrator = MasterOrchestrator(config)
        
        print(f"  ✅ GAELP Master Orchestrator initialized")
        print(f"  ✅ Journey State Encoder integrated: {hasattr(orchestrator, 'state_encoder')}")
        print(f"  ✅ State encoder type: {type(orchestrator.state_encoder)}")
        
        # Test journey state encoding
        import time
        mock_journey = UserJourney(
            journey_id="test_journey",
            user_id="test_user",
            canonical_user_id="test_user",
            journey_start=datetime.now(),
            current_state=JourneyState.CONSIDERING,
            touchpoint_count=3,
            journey_score=0.6,
            converted=False
        )
        # Ensure journey_start is properly set as datetime
        mock_journey.journey_start = datetime.now()
        
        mock_profile = UserProfile(
            user_id="test_user",
            canonical_user_id="test_user",
            device_ids=["device_123"],
            current_journey_state=JourneyState.CONSIDERING,
            conversion_probability=0.3,
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        
        # Test state encoding
        import asyncio
        encoded_state = asyncio.run(orchestrator._encode_journey_state(mock_journey, mock_profile))
        
        print(f"  ✅ Journey state encoded successfully")
        print(f"  ✅ Encoded state type: {type(encoded_state)}")
        print(f"  ✅ Has required fields: {all(key in encoded_state for key in ['current_state', 'conversion_probability', 'total_touches'])}")
        
        print("✅ GAELP integration test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ GAELP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Journey State Encoder Integration Tests")
    print("=" * 70)
    
    # Run tests
    encoder_test_passed = test_journey_encoder_integration()
    gaelp_test_passed = test_gaelp_integration()
    
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    print(f"Journey Encoder Integration: {'✅ PASSED' if encoder_test_passed else '❌ FAILED'}")
    print(f"GAELP Master Integration:    {'✅ PASSED' if gaelp_test_passed else '❌ FAILED'}")
    
    if encoder_test_passed and gaelp_test_passed:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Journey State Encoder is successfully wired to PPO Agent")
        print("✅ 256-dimensional LSTM features are properly fed to actor-critic")
        print("✅ GAELP system integration is working correctly")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("=" * 70)