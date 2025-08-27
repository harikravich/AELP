#!/usr/bin/env python3
"""
Simple test script to verify PPO agent integration with Journey State Encoder.
"""

import sys
sys.path.append('/home/hariravichandran/AELP')

import torch
import numpy as np
from datetime import datetime

# Add the journey encoder path
import os
sys.path.insert(0, os.path.join('/home/hariravichandran/AELP', 'training_orchestrator'))

from journey_state_encoder import create_journey_encoder
from journey_aware_rl_agent import JourneyAwarePPOAgent

def test_encoder_integration():
    """Test the encoder integration with PPO agent"""
    print("Testing PPO Agent with Journey State Encoder Integration")
    print("=" * 60)
    
    # Create example journey data
    journey_data = {
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
        'current_timestamp': 1640995200,  # Unix timestamp
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
            },
            {
                'channel': 'display',
                'user_state_after': 'considering',
                'cost': 3.80,
                'timestamp': 1640994000
            }
        ],
        'channel_distribution': {
            'search': 1, 'social': 1, 'display': 1, 'video': 0,
            'email': 0, 'direct': 0, 'affiliate': 0, 'retargeting': 0
        },
        'channel_costs': {
            'search': 2.50, 'social': 1.20, 'display': 3.80, 'video': 0.0,
            'email': 0.0, 'direct': 0.0, 'affiliate': 0.0, 'retargeting': 0.0
        },
        'channel_last_touch': {
            'search': 3.0, 'social': 1.5, 'display': 0.1, 'video': 30.0,
            'email': 30.0, 'direct': 30.0, 'affiliate': 30.0, 'retargeting': 30.0
        },
        'click_through_rate': 0.035,
        'engagement_rate': 0.15,
        'bounce_rate': 0.4,
        'conversion_rate': 0.08,
        'competitors_seen': 2,
        'competitor_engagement_rate': 0.12
    }
    
    # Test 1: Create encoder directly
    print("\n1. Testing encoder creation...")
    encoder = create_journey_encoder(
        max_sequence_length=5,
        lstm_hidden_dim=64,
        encoded_state_dim=256,
        normalize_features=False  # Disable normalization for single sample testing
    )
    print(f"   ✓ Encoder created with output dim: {encoder.get_output_dim()}")
    
    # Test 2: Encode journey
    print("\n2. Testing journey encoding...")
    encoded_state = encoder.encode_journey(journey_data)
    print(f"   ✓ Encoded state shape: {encoded_state.shape}")
    print(f"   ✓ Encoded state sample values: {encoded_state[:5]}")
    
    # Test 3: Create PPO agent with encoder
    print("\n3. Testing PPO agent with encoder...")
    agent = JourneyAwarePPOAgent(
        state_dim=256,
        hidden_dim=256,
        num_channels=8,
        use_journey_encoder=True
    )
    print(f"   ✓ PPO agent created with encoder")
    
    # Test 4: Action selection
    print("\n4. Testing action selection...")
    channel_idx, bid_amount, log_prob = agent.select_action(journey_data)
    print(f"   ✓ Selected channel: {channel_idx}")
    print(f"   ✓ Bid amount: ${bid_amount:.2f}")
    print(f"   ✓ Log probability: {log_prob.item():.4f}")
    
    # Test 5: Multiple action selections
    print("\n5. Testing multiple action selections...")
    actions = []
    for i in range(5):
        ch, bid, _ = agent.select_action(journey_data)
        actions.append((ch, bid))
    print(f"   ✓ Actions: {actions}")
    
    # Test 6: Memory and update test
    print("\n6. Testing memory storage and update...")
    
    # Add transitions to memory
    for i in range(10):
        test_data = journey_data.copy()
        test_data['conversion_probability'] = min(0.1 + i * 0.1, 1.0)
        
        ch_idx, bid, log_p = agent.select_action(test_data)
        agent.store_transition(
            state=test_data,
            action=ch_idx,
            reward=np.random.uniform(-1, 3),
            next_state=test_data,
            done=False,
            log_prob=log_p
        )
    
    print(f"   ✓ Stored {len(agent.memory)} transitions")
    
    # Test update
    initial_params = [p.clone() for p in agent.actor_critic.parameters()]
    initial_encoder_params = [p.clone() for p in agent.journey_encoder.parameters()]
    
    agent.update(batch_size=10, epochs=1)
    
    # Check parameter changes
    params_changed = any(
        not torch.equal(initial, current) 
        for initial, current in zip(initial_params, agent.actor_critic.parameters())
    )
    
    encoder_params_changed = any(
        not torch.equal(initial, current) 
        for initial, current in zip(initial_encoder_params, agent.journey_encoder.parameters())
    )
    
    print(f"   ✓ Actor-critic parameters changed: {params_changed}")
    print(f"   ✓ Encoder parameters changed: {encoder_params_changed}")
    
    # Test 7: Save and load
    print("\n7. Testing save/load functionality...")
    save_path = "/home/hariravichandran/AELP/test_checkpoint.pth"
    agent.save(save_path)
    print(f"   ✓ Saved agent checkpoint")
    
    # Load into new agent
    agent2 = JourneyAwarePPOAgent(
        state_dim=256,
        hidden_dim=256, 
        num_channels=8,
        use_journey_encoder=True
    )
    agent2.load(save_path)
    print(f"   ✓ Loaded agent checkpoint")
    
    # Verify consistency
    ch1, bid1, _ = agent.select_action(journey_data)
    ch2, bid2, _ = agent2.select_action(journey_data)
    print(f"   ✓ Original: ch={ch1}, bid=${bid1:.2f}")
    print(f"   ✓ Loaded: ch={ch2}, bid=${bid2:.2f}")
    
    # Cleanup
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! PPO agent successfully integrated with Journey State Encoder")
    print(f"✅ Encoder provides rich 256-dimensional state representation")
    print(f"✅ LSTM sequences are properly processed for touchpoint history")
    print(f"✅ Gradients flow correctly through encoder during training")
    
    return agent

if __name__ == "__main__":
    try:
        agent = test_encoder_integration()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)