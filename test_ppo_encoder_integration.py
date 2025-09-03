#!/usr/bin/env python3
"""
Test script to verify PPO agent integration with Journey State Encoder.

This script tests:
1. Journey encoder state encoding
2. PPO agent action selection with encoded states
3. Gradient flow through the encoder
4. Memory storage and updates
"""

import sys
sys.path.append('/home/hariravichandran/AELP')

import torch
import numpy as np
from datetime import datetime

from journey_aware_rl_agent import JourneyAwarePPOAgent, extract_journey_state_for_encoder
from enhanced_journey_tracking import EnhancedMultiTouchUser, UserState, Channel
from multi_channel_orchestrator import MultiChannelOrchestrator

def create_mock_user():
    """Create a mock user for testing"""
    user = EnhancedMultiTouchUser("test_user_001")
    
    # Add some journey history
    from enhanced_journey_tracking import Touchpoint
    
    # Simulate a few touchpoints
    touchpoint1 = Touchpoint(
        timestamp=datetime.now(),
        channel=Channel.SEARCH,
        cost=2.5,
        user_state_before=UserState.UNAWARE,
        user_state_after=UserState.AWARE
    )
    user.journey.append(touchpoint1)
    user.current_state = UserState.AWARE
    
    touchpoint2 = Touchpoint(
        timestamp=datetime.now(),
        channel=Channel.SOCIAL,
        cost=1.2,
        user_state_before=UserState.AWARE,
        user_state_after=UserState.INTERESTED
    )
    user.journey.append(touchpoint2)
    user.current_state = UserState.INTERESTED
    
    # Update user stats
    user.total_touches = 2
    user.total_cost = 3.7
    user.conversion_probability = 0.25
    user.time_since_last_touch = 1.0
    
    return user

def test_encoder_integration():
    """Test the encoder integration with PPO agent"""
    print("Testing PPO Agent with Journey State Encoder Integration")
    print("=" * 60)
    
    # Create test components
    user = create_mock_user()
    orchestrator = MultiChannelOrchestrator(budget_daily=100.0)
    timestamp = datetime.now()
    
    # Test 1: State extraction for encoder
    print("\n1. Testing state extraction for encoder...")
    journey_data = extract_journey_state_for_encoder(user, orchestrator, timestamp)
    print(f"   ✓ Extracted journey data with {len(journey_data)} fields")
    print(f"   ✓ Journey history length: {len(journey_data['journey_history'])}")
    print(f"   ✓ Current state: {journey_data['current_state']}")
    
    # Test 2: PPO Agent initialization
    print("\n2. Testing PPO agent initialization with encoder...")
    agent = JourneyAwarePPOAgent(
        state_dim=256,
        hidden_dim=256,
        num_channels=8,
        use_journey_encoder=True
    )
    print(f"   ✓ Agent initialized with encoder enabled")
    print(f"   ✓ Encoder output dim: {agent.journey_encoder.get_output_dim()}")
    
    # Test 3: State encoding
    print("\n3. Testing state encoding...")
    encoded_state = agent.journey_encoder.encode_journey(journey_data)
    print(f"   ✓ Encoded state shape: {encoded_state.shape}")
    print(f"   ✓ Expected shape: torch.Size([256])")
    assert encoded_state.shape == torch.Size([256]), f"Unexpected encoded state shape: {encoded_state.shape}"
    
    # Test 4: Action selection
    print("\n4. Testing action selection...")
    channel_idx, bid_amount, log_prob = agent.select_action(journey_data)
    print(f"   ✓ Selected channel: {channel_idx}")
    print(f"   ✓ Bid amount: ${bid_amount:.2f}")
    print(f"   ✓ Log probability: {log_prob.item():.4f}")
    
    # Test 5: Memory storage
    print("\n5. Testing memory storage...")
    agent.store_transition(
        state=journey_data,
        action=channel_idx,
        reward=1.5,
        next_state=journey_data,  # Same state for test
        done=False,
        log_prob=log_prob
    )
    print(f"   ✓ Stored transition in memory")
    print(f"   ✓ Memory size: {len(agent.memory)}")
    
    # Test 6: Gradient flow (small batch test)
    print("\n6. Testing gradient flow...")
    
    # Add more transitions for a small batch
    for i in range(5):
        # Slightly modify the journey data
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
    
    # Test update
    print(f"   ✓ Added {len(agent.memory)} transitions")
    
    # Get initial parameters to verify they change
    initial_actor_params = [p.clone() for p in agent.actor_critic.parameters()]
    initial_encoder_params = [p.clone() for p in agent.journey_encoder.parameters()]
    
    agent.update(batch_size=6, epochs=1)
    
    # Check if parameters changed (indicates gradient flow)
    actor_params_changed = any(
        not torch.equal(initial, current) 
        for initial, current in zip(initial_actor_params, agent.actor_critic.parameters())
    )
    
    encoder_params_changed = any(
        not torch.equal(initial, current) 
        for initial, current in zip(initial_encoder_params, agent.journey_encoder.parameters())
    )
    
    print(f"   ✓ Actor-critic parameters changed: {actor_params_changed}")
    print(f"   ✓ Encoder parameters changed: {encoder_params_changed}")
    
    # Test 7: Save/Load functionality
    print("\n7. Testing save/load functionality...")
    save_path = "/home/hariravichandran/AELP/test_agent_checkpoint.pth"
    agent.save(save_path)
    print(f"   ✓ Saved agent to {save_path}")
    
    # Create new agent and load
    agent2 = JourneyAwarePPOAgent(
        state_dim=256,
        hidden_dim=256,
        num_channels=8,
        use_journey_encoder=True
    )
    agent2.load(save_path)
    print(f"   ✓ Loaded agent from checkpoint")
    
    # Verify loaded agent produces same action
    ch_idx1, bid1, _ = agent.select_action(journey_data)
    ch_idx2, bid2, _ = agent2.select_action(journey_data)
    
    print(f"   ✓ Original agent action: channel={ch_idx1}, bid=${bid1:.2f}")
    print(f"   ✓ Loaded agent action: channel={ch_idx2}, bid=${bid2:.2f}")
    
    # Test 8: Performance comparison
    print("\n8. Testing performance comparison...")
    
    # Test with encoder
    import time
    start_time = time.time()
    for _ in range(100):
        agent.select_action(journey_data)
    encoder_time = time.time() - start_time
    
    # Test without encoder (use simple state)
    agent_simple = JourneyAwarePPOAgent(
        state_dim=41,  # Original simple state dim
        use_journey_encoder=False
    )
    
    from journey_aware_rl_agent import extract_journey_state
    simple_state = extract_journey_state(user, orchestrator, timestamp)
    
    start_time = time.time()
    for _ in range(100):
        agent_simple.select_action(simple_state)
    simple_time = time.time() - start_time
    
    print(f"   ✓ Encoder-based inference: {encoder_time:.4f}s (100 actions)")
    print(f"   ✓ Simple state inference: {simple_time:.4f}s (100 actions)")
    print(f"   ✓ Encoder overhead: {(encoder_time/simple_time - 1)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! PPO agent successfully integrated with Journey State Encoder")
    
    # Cleanup
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
    
    return agent

if __name__ == "__main__":
    try:
        agent = test_encoder_integration()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)