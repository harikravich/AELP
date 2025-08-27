# Delayed Reward System for Multi-Day Conversions

The Delayed Reward System is a critical component of the GAELP Training Orchestrator that handles multi-day conversions and attribution in ad campaign optimization. This system addresses the fundamental challenge that conversions often happen days after the initial ad touchpoint, requiring sophisticated attribution and reward backpropagation mechanisms.

## Problem Statement

In traditional reinforcement learning for ad campaigns, agents only receive immediate rewards (clicks, impressions) but miss the most important delayed signals (purchases, signups) that may occur 1-7 days later. This creates several problems:

1. **Immediate-Reward-Only Training**: Agents optimize for clicks rather than conversions
2. **Attribution Gap**: No mechanism to connect later conversions to earlier ad touchpoints
3. **Partial Episode Problem**: Episodes end before conversions happen
4. **Multi-Touch Attribution**: Multiple touchpoints contribute to a single conversion

## Solution Overview

The Delayed Reward System provides:

- **Pending Reward Storage**: Store touchpoints waiting for attribution
- **Multiple Attribution Models**: Last-click, first-click, linear, time-decay, position-based, data-driven
- **Partial Episode Handling**: Connect conversions to completed episodes
- **Reward Replay Buffer**: Train agents with corrected rewards when conversions are detected
- **Distributed Storage**: Redis caching + database persistence for scalability

## Core Components

### 1. DelayedRewardSystem

Main orchestrator class that manages the entire delayed reward pipeline.

```python
from training_orchestrator import DelayedRewardSystem, DelayedRewardConfig

config = DelayedRewardConfig(
    attribution_window_days=7,
    default_attribution_model=AttributionModel.LINEAR,
    replay_buffer_size=50000
)

delayed_reward_system = DelayedRewardSystem(config)
```

### 2. Touchpoint Tracking

Every agent action creates a touchpoint that's stored for potential future attribution:

```python
touchpoint_id = await delayed_reward_system.store_pending_reward(
    episode_id="episode_123",
    user_id="user_456",
    campaign_id="campaign_789",
    action=agent_action,
    state=environment_state,
    immediate_reward=immediate_reward,
    channel="search",
    creative_type="video"
)
```

### 3. Attribution Trigger

When a conversion is detected, attribution is triggered across relevant touchpoints:

```python
attribution_rewards = await delayed_reward_system.trigger_attribution(
    user_id="user_456",
    conversion_event=ConversionEvent.PURCHASE,
    conversion_value=250.0,
    currency="USD"
)

# Returns: {"touchpoint_1": 100.0, "touchpoint_2": 150.0}
```

### 4. Reward Replay Buffer

Stores experiences with corrected rewards for training:

```python
replay_batch = await delayed_reward_system.get_replay_batch(batch_size=32)

for experience in replay_batch:
    await agent.update_with_corrected_reward(
        state=experience['state'],
        action=experience['action'],
        corrected_reward=experience['attributed_reward'],
        original_reward=experience['original_reward']
    )
```

## Attribution Models

### Linear Attribution (Default)
Equal credit to all touchpoints in the customer journey.

```python
# 3 touchpoints, $120 conversion = $40 each
touchpoint_1: $40
touchpoint_2: $40  
touchpoint_3: $40
```

### Last-Click Attribution
All credit to the final touchpoint before conversion.

```python
# Only last touchpoint gets credit
touchpoint_1: $0
touchpoint_2: $0
touchpoint_3: $120
```

### First-Click Attribution
All credit to the initial touchpoint that started the journey.

```python
# Only first touchpoint gets credit
touchpoint_1: $120
touchpoint_2: $0
touchpoint_3: $0
```

### Time-Decay Attribution
More recent touchpoints get exponentially more credit.

```python
# Half-life of 24 hours (configurable)
touchpoint_1 (48h ago): $20
touchpoint_2 (24h ago): $40
touchpoint_3 (12h ago): $60
```

### Position-Based Attribution
40% first, 40% last, 20% distributed among middle touchpoints.

```python
# U-shaped attribution curve
touchpoint_1: $48  # 40% first
touchpoint_2: $24  # 20% middle
touchpoint_3: $48  # 40% last
```

### Data-Driven Attribution
Uses historical conversion data to weight touchpoints by effectiveness.

```python
# Weights based on channel performance, position, etc.
search_touchpoint: $60    # High conversion channel
display_touchpoint: $30   # Lower conversion channel  
social_touchpoint: $30    # Medium conversion channel
```

## Integration with Training Orchestrator

### Episode Manager Integration

The system integrates seamlessly with the existing episode manager:

```python
from training_orchestrator import integrate_with_episode_manager

# Enhance episode manager with delayed rewards
enhanced_episode_manager = await integrate_with_episode_manager(
    episode_manager, 
    delayed_reward_system
)

# Episodes now automatically track pending rewards
episode_result = await enhanced_episode_manager.run_episode(
    agent, environment, "campaign_episode_1"
)

# Check for delayed reward updates
if 'delayed_rewards' in episode_result.info:
    print(f"Found {len(episode_result.info['delayed_rewards'])} delayed updates")
```

### Training Loop Integration

Complete training loop with delayed reward learning:

```python
async def enhanced_training_loop():
    while training:
        # Regular episode with immediate rewards
        episode_result = await run_episode(agent, environment)
        
        # Store pending rewards for future attribution
        for step in episode_result.steps:
            await delayed_reward_system.store_pending_reward(...)
        
        # Periodically train with delayed rewards
        if episode_count % replay_frequency == 0:
            replay_batch = await delayed_reward_system.get_replay_batch()
            for experience in replay_batch:
                await agent.update_with_corrected_reward(...)
```

## Storage Architecture

### In-Memory Tracking
- User journeys: `Dict[user_id, List[Touchpoint]]`
- Episode touchpoints: `Dict[episode_id, List[touchpoint_id]]` 
- Pending rewards: `Dict[reward_id, PendingReward]`

### Redis Caching (Optional)
```python
config = DelayedRewardConfig(
    use_redis_cache=True,
    redis_host="localhost",
    redis_port=6379,
    redis_ttl_seconds=604800  # 7 days
)
```

### Database Persistence (Optional)
```python
config = DelayedRewardConfig(
    use_database_persistence=True,
    database_url="postgresql://user:pass@localhost/delayed_rewards"
)
```

## Configuration Options

```python
@dataclass
class DelayedRewardConfig:
    # Attribution settings
    attribution_window_days: int = 7
    default_attribution_model: AttributionModel = AttributionModel.LINEAR
    time_decay_half_life_hours: float = 24.0
    
    # Storage settings  
    use_redis_cache: bool = True
    use_database_persistence: bool = True
    max_pending_rewards: int = 100000
    
    # Replay buffer settings
    replay_buffer_size: int = 50000
    min_replay_samples: int = 1000
    replay_batch_size: int = 32
    replay_frequency: int = 100  # Every N episodes
    
    # Performance settings
    batch_attribution_size: int = 1000
    max_concurrent_attributions: int = 10
    enable_async_processing: bool = True
```

## Performance Considerations

### Memory Management
- Automatic cleanup of expired pending rewards
- Configurable buffer sizes with LRU eviction
- Efficient batch processing for attribution

### Scalability
- Redis clustering for distributed caching  
- Database sharding by user_id or time
- Async processing for high-throughput scenarios

### Latency Optimization
- In-memory primary storage with async persistence
- Batch attribution processing
- Configurable cleanup intervals

## Monitoring and Statistics

```python
stats = delayed_reward_system.get_statistics()

print(f"Pending rewards: {stats['pending_rewards']}")
print(f"Conversions attributed: {stats['attribution_stats']['total_conversions_attributed']}")
print(f"Total reward attributed: ${stats['attribution_stats']['total_reward_attributed']:.2f}")
print(f"Avg time to conversion: {stats['replay_buffer']['avg_time_to_conversion']:.1f}h")
print(f"Conversion rate: {stats['replay_buffer']['conversion_rate']:.2%}")
```

## Usage Examples

### Basic Usage

```python
import asyncio
from training_orchestrator import DelayedRewardSystem, DelayedRewardConfig, ConversionEvent

async def basic_example():
    # Initialize system
    config = DelayedRewardConfig(attribution_window_days=7)
    delayed_reward_system = DelayedRewardSystem(config)
    
    # Store pending reward
    touchpoint_id = await delayed_reward_system.store_pending_reward(
        episode_id="episode_1",
        user_id="user_123",
        campaign_id="campaign_456",
        action={"budget": 100, "creative": "video"},
        state={"market_conditions": "high_competition"},
        immediate_reward=2.5,
        channel="search"
    )
    
    # Simulate conversion 2 days later
    await asyncio.sleep(2)  # In reality, this would be 2 days
    
    attribution_rewards = await delayed_reward_system.trigger_attribution(
        user_id="user_123",
        conversion_event=ConversionEvent.PURCHASE,
        conversion_value=150.0
    )
    
    print(f"Attributed ${attribution_rewards[touchpoint_id]:.2f} to touchpoint")

asyncio.run(basic_example())
```

### Advanced Multi-Touch Journey

```python
async def multi_touch_example():
    delayed_reward_system = DelayedRewardSystem(DelayedRewardConfig())
    
    user_id = "user_journey_demo"
    
    # Simulate multi-touch customer journey
    touchpoints = []
    
    # Touch 1: Search ad (awareness)
    tp1 = await delayed_reward_system.store_pending_reward(
        episode_id="episode_search",
        user_id=user_id,
        campaign_id="search_campaign",
        action={"channel": "search", "budget": 50},
        state={"funnel_stage": "awareness"},
        immediate_reward=1.0,
        channel="search"
    )
    touchpoints.append(tp1)
    
    # Touch 2: Display retargeting (consideration)  
    tp2 = await delayed_reward_system.store_pending_reward(
        episode_id="episode_display",
        user_id=user_id,
        campaign_id="retargeting_campaign", 
        action={"channel": "display", "budget": 30},
        state={"funnel_stage": "consideration"},
        immediate_reward=0.5,
        channel="display"
    )
    touchpoints.append(tp2)
    
    # Touch 3: Email (conversion)
    tp3 = await delayed_reward_system.store_pending_reward(
        episode_id="episode_email",
        user_id=user_id,
        campaign_id="email_campaign",
        action={"channel": "email", "budget": 10},
        state={"funnel_stage": "decision"},
        immediate_reward=0.2,
        channel="email"
    )
    touchpoints.append(tp3)
    
    # Conversion happens
    attribution_rewards = await delayed_reward_system.trigger_attribution(
        user_id=user_id,
        conversion_event=ConversionEvent.PURCHASE,
        conversion_value=200.0
    )
    
    print("Attribution Results:")
    for tp_id, reward in attribution_rewards.items():
        print(f"  Touchpoint {tp_id[:8]}: ${reward:.2f}")
    
    # Get user journey
    journey = delayed_reward_system.get_user_journey(user_id)
    print(f"\nComplete journey: {len(journey)} touchpoints")
    for i, tp in enumerate(journey):
        print(f"  {i+1}. {tp.channel} campaign (${tp.immediate_reward} → ${attribution_rewards.get(tp.touchpoint_id, 0):.2f})")

asyncio.run(multi_touch_example())
```

## Testing

Run the comprehensive test suite:

```bash
# Run all delayed reward system tests
python -m pytest tests/test_delayed_reward_system.py -v

# Run specific test categories
python -m pytest tests/test_delayed_reward_system.py::TestAttributionModels -v
python -m pytest tests/test_delayed_reward_system.py::TestDelayedRewardSystem -v
python -m pytest tests/test_delayed_reward_system.py::TestRewardReplayBuffer -v
```

## Demo

Run the interactive demonstration:

```bash
# Run full delayed reward system demo
python examples/delayed_reward_demo.py

# Output shows:
# - Multi-touch customer journeys
# - Real-time conversion attribution  
# - Delayed reward training from replay buffer
# - System performance statistics
```

## Best Practices

### 1. Attribution Model Selection
- **Linear**: Good default for balanced attribution
- **Last-Click**: Use when final touchpoint most important
- **Time-Decay**: Best for long consideration periods
- **Data-Driven**: Use when you have sufficient historical data

### 2. Attribution Window
```python
# E-commerce: 7-14 days
config.attribution_window_days = 7

# B2B: 30-90 days  
config.attribution_window_days = 30

# Mobile apps: 1-3 days
config.attribution_window_days = 1
```

### 3. Replay Training Frequency
```python
# High-volume: Train more frequently
config.replay_frequency = 50  # Every 50 episodes

# Low-volume: Train less frequently  
config.replay_frequency = 200  # Every 200 episodes
```

### 4. Memory Management
```python
# Production settings
config.max_pending_rewards = 1000000
config.replay_buffer_size = 100000
config.cleanup_interval_hours = 12
```

## Troubleshooting

### Common Issues

1. **No Attribution Happening**
   - Check attribution window settings
   - Verify user_id consistency between touchpoints and conversions
   - Ensure conversions are within the time window

2. **Memory Usage Growing**
   - Reduce `max_pending_rewards` and `replay_buffer_size`
   - Decrease `cleanup_interval_hours`
   - Enable database persistence and reduce in-memory storage

3. **Slow Attribution Processing**
   - Enable async processing
   - Increase `batch_attribution_size`
   - Use Redis caching
   - Consider database indexing

4. **Inconsistent Rewards**
   - Check attribution model configuration
   - Verify touchpoint timestamps
   - Review conversion value calculations

### Debug Mode

```python
import logging
logging.getLogger("training_orchestrator.delayed_reward_system").setLevel(logging.DEBUG)

# Enables detailed logging of:
# - Touchpoint storage
# - Attribution calculations  
# - Replay buffer operations
# - Performance metrics
```

## Future Enhancements

1. **Advanced Attribution Models**
   - Shapley value attribution
   - Machine learning-based attribution
   - Cross-device journey tracking

2. **Real-Time Integration**
   - Streaming attribution from external systems
   - Real-time conversion feeds
   - Event-driven architecture

3. **Enhanced Analytics**
   - Attribution model comparison
   - A/B testing framework
   - Conversion funnel analysis

4. **Scalability Improvements**
   - Distributed attribution processing
   - Stream processing with Apache Kafka
   - Auto-scaling based on load

This delayed reward system is a fundamental component for building effective RL agents that optimize for actual business value rather than just intermediate metrics.