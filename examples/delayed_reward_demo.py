#!/usr/bin/env python3
"""
Delayed Reward System Demo

Demonstrates how to use the delayed reward system for multi-day conversions
in ad campaign optimization training.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any

from training_orchestrator import (
    DelayedRewardSystem,
    DelayedRewardConfig,
    AttributionModel,
    ConversionEvent,
    EpisodeManager,
    integrate_with_episode_manager
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockAgent:
    """Mock RL agent for demonstration"""
    
    def __init__(self):
        self.learning_rate = 0.001
        
    async def select_action(self, observation):
        """Select a random action for demo purposes"""
        return {
            'campaign_type': random.choice(['search', 'display', 'social']),
            'budget': random.uniform(50, 200),
            'creative_type': random.choice(['image', 'video', 'carousel']),
            'target_audience': random.choice(['young_adults', 'professionals', 'parents']),
            'bid_strategy': random.choice(['cpc', 'cpm', 'cpa'])
        }
    
    async def update(self, state, action, reward, next_state, done):
        """Mock training update"""
        logger.info(f"Training update: reward={reward:.3f}")
        
    async def update_with_corrected_reward(self, state, action, corrected_reward, original_reward):
        """Update with corrected reward from delayed attribution"""
        reward_delta = corrected_reward - original_reward
        logger.info(f"Corrected reward update: delta={reward_delta:.3f}")


class MockEnvironment:
    """Mock ad campaign environment"""
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 20
        self.user_pool = [f"user_{i}" for i in range(100)]
        
    async def reset(self):
        """Reset environment"""
        self.step_count = 0
        return {
            'market_conditions': random.uniform(0.5, 1.5),
            'competition_level': random.uniform(0.3, 1.0),
            'seasonality': random.uniform(0.8, 1.2)
        }
        
    async def step(self, action):
        """Execute action in environment"""
        self.step_count += 1
        
        # Simulate campaign results
        budget = action['budget']
        impressions = int(budget * random.uniform(50, 150))
        clicks = int(impressions * random.uniform(0.01, 0.05))  # 1-5% CTR
        cost = budget * random.uniform(0.8, 1.0)
        
        # Immediate reward based on CTR and cost efficiency
        ctr = clicks / max(impressions, 1)
        immediate_reward = ctr * 10 + (budget - cost) * 0.1
        
        # Environment info with user tracking
        info = {
            'user_id': random.choice(self.user_pool),
            'campaign_id': f"campaign_{random.randint(1, 10)}",
            'impressions': impressions,
            'clicks': clicks,
            'cost': cost,
            'channel': action['campaign_type'],
            'creative_type': action['creative_type']
        }
        
        next_state = {
            'market_conditions': random.uniform(0.5, 1.5),
            'competition_level': random.uniform(0.3, 1.0),
            'seasonality': random.uniform(0.8, 1.2)
        }
        
        done = self.step_count >= self.max_steps
        
        return next_state, immediate_reward, done, info


async def simulate_conversions(delayed_reward_system: DelayedRewardSystem):
    """Simulate delayed conversions happening over time"""
    
    users = [f"user_{i}" for i in range(100)]
    
    while True:
        await asyncio.sleep(5)  # Check every 5 seconds for demo
        
        # Randomly trigger conversions
        if random.random() < 0.3:  # 30% chance per check
            user_id = random.choice(users)
            
            # Check if user has any touchpoints
            journey = delayed_reward_system.get_user_journey(user_id)
            if journey:
                conversion_value = random.uniform(10, 500)
                conversion_event = random.choice(list(ConversionEvent))
                
                logger.info(f"🎉 Conversion! User {user_id} converted for ${conversion_value:.2f}")
                
                attributed_rewards = await delayed_reward_system.trigger_attribution(
                    user_id=user_id,
                    conversion_event=conversion_event,
                    conversion_value=conversion_value
                )
                
                logger.info(f"Attributed to {len(attributed_rewards)} touchpoints")


async def run_training_episode(agent, environment, episode_manager, delayed_reward_system, episode_id):
    """Run a single training episode with delayed reward tracking"""
    
    logger.info(f"🚀 Starting episode {episode_id}")
    
    state = await environment.reset()
    done = False
    total_immediate_reward = 0
    
    while not done:
        action = await agent.select_action(state)
        next_state, reward, done, info = await environment.step(action)
        
        # Store pending reward for delayed attribution
        touchpoint_id = await delayed_reward_system.store_pending_reward(
            episode_id=episode_id,
            user_id=info['user_id'],
            campaign_id=info['campaign_id'],
            action=action,
            state=state,
            immediate_reward=reward,
            channel=info['channel'],
            creative_type=info['creative_type'],
            cost=info['cost']
        )
        
        logger.info(f"  Step: reward={reward:.3f}, user={info['user_id']}, touchpoint={touchpoint_id[:8]}")
        
        # Train with immediate reward
        await agent.update(state, action, reward, next_state, done)
        
        total_immediate_reward += reward
        state = next_state
        
        await asyncio.sleep(0.1)  # Small delay for demo visibility
    
    logger.info(f"✅ Episode {episode_id} completed. Total immediate reward: {total_immediate_reward:.3f}")
    
    # Check for any delayed reward updates
    delayed_updates = await delayed_reward_system.handle_partial_episode(episode_id)
    if delayed_updates:
        logger.info(f"📈 Found {len(delayed_updates)} delayed reward updates for episode {episode_id}")


async def train_with_replay_buffer(agent, delayed_reward_system):
    """Train agent with delayed reward replay buffer"""
    
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        
        replay_batch = await delayed_reward_system.get_replay_batch(batch_size=5)
        
        if replay_batch:
            logger.info(f"🎓 Training with {len(replay_batch)} delayed reward experiences")
            
            for experience in replay_batch:
                await agent.update_with_corrected_reward(
                    state=experience['state'],
                    action=experience['action'],
                    corrected_reward=experience['attributed_reward'],
                    original_reward=experience['original_reward']
                )


async def print_statistics(delayed_reward_system):
    """Print system statistics periodically"""
    
    while True:
        await asyncio.sleep(15)  # Print every 15 seconds
        
        stats = delayed_reward_system.get_statistics()
        
        print("\n" + "="*60)
        print("📊 DELAYED REWARD SYSTEM STATISTICS")
        print("="*60)
        print(f"Pending rewards: {stats['pending_rewards']}")
        print(f"User journeys: {stats['user_journeys']}")
        print(f"Total touchpoints: {stats['total_touchpoints']}")
        print(f"Conversions attributed: {stats['attribution_stats']['total_conversions_attributed']}")
        print(f"Total reward attributed: ${stats['attribution_stats']['total_reward_attributed']:.2f}")
        
        if stats['replay_buffer']['size'] > 0:
            print(f"Replay buffer size: {stats['replay_buffer']['size']}")
            print(f"Avg reward delta: {stats['replay_buffer']['avg_reward_delta']:.3f}")
            print(f"Avg time to conversion: {stats['replay_buffer']['avg_time_to_conversion']:.1f}h")
            print(f"Conversion rate: {stats['replay_buffer']['conversion_rate']:.2%}")
        
        print("="*60)


async def demo_attribution_models():
    """Demonstrate different attribution models"""
    
    print("\n" + "🧪 ATTRIBUTION MODEL DEMONSTRATION")
    print("="*60)
    
    # Create a simple delayed reward system for testing
    config = DelayedRewardConfig(
        attribution_window_days=1,
        use_redis_cache=False,
        use_database_persistence=False
    )
    
    delayed_reward_system = DelayedRewardSystem(config)
    
    # Create mock touchpoints
    user_id = "demo_user"
    touchpoints = []
    
    for i in range(3):
        touchpoint_id = await delayed_reward_system.store_pending_reward(
            episode_id=f"demo_episode_{i}",
            user_id=user_id,
            campaign_id="demo_campaign",
            action={'budget': 100, 'channel': ['search', 'display', 'social'][i]},
            state={'market': 'test'},
            immediate_reward=10.0,
            channel=['search', 'display', 'social'][i]
        )
        touchpoints.append(touchpoint_id)
        await asyncio.sleep(0.1)  # Small time gap
    
    # Test different attribution models
    conversion_value = 100.0
    
    for model in AttributionModel:
        delayed_reward_system.config.default_attribution_model = model
        
        attributed_rewards = await delayed_reward_system.trigger_attribution(
            user_id=user_id,
            conversion_event=ConversionEvent.PURCHASE,
            conversion_value=conversion_value
        )
        
        print(f"\n{model.value.upper()} Attribution:")
        for touchpoint_id, reward in attributed_rewards.items():
            print(f"  Touchpoint {touchpoint_id[:8]}: ${reward:.2f}")
        
        # Reset for next model
        delayed_reward_system.user_journeys[user_id] = delayed_reward_system.user_journeys[user_id][:3]


async def main():
    """Main demonstration function"""
    
    print("🚀 DELAYED REWARD SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Configuration
    config = DelayedRewardConfig(
        attribution_window_days=7,
        replay_buffer_size=1000,
        min_replay_samples=10,
        replay_batch_size=5,
        use_redis_cache=False,  # Disable for demo
        use_database_persistence=False,  # Disable for demo
        enable_async_processing=True
    )
    
    # Initialize system
    delayed_reward_system = DelayedRewardSystem(config)
    
    # Create mock components
    agent = MockAgent()
    environment = MockEnvironment()
    
    # Demonstrate attribution models
    await demo_attribution_models()
    
    print("\n🎬 Starting live training simulation...")
    print("This will run for 2 minutes with:")
    print("- Training episodes every few seconds")
    print("- Random conversions triggering attribution")
    print("- Periodic statistics updates")
    print("- Delayed reward training from replay buffer")
    
    # Start background tasks
    tasks = [
        asyncio.create_task(simulate_conversions(delayed_reward_system)),
        asyncio.create_task(train_with_replay_buffer(agent, delayed_reward_system)),
        asyncio.create_task(print_statistics(delayed_reward_system))
    ]
    
    # Run training episodes
    try:
        for episode in range(10):
            await run_training_episode(
                agent, environment, None, delayed_reward_system, f"demo_episode_{episode}"
            )
            await asyncio.sleep(3)  # Gap between episodes
        
        # Let background tasks run a bit longer to show delayed effects
        print("\n⏱️  Letting background processes run for 30 seconds...")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    finally:
        # Cancel background tasks
        for task in tasks:
            task.cancel()
        
        # Show final statistics
        final_stats = delayed_reward_system.get_statistics()
        
        print("\n" + "🏁 FINAL STATISTICS")
        print("="*60)
        print(f"Total conversions: {final_stats['attribution_stats']['total_conversions_attributed']}")
        print(f"Total reward attributed: ${final_stats['attribution_stats']['total_reward_attributed']:.2f}")
        print(f"Replay buffer experiences: {final_stats['replay_buffer']['size']}")
        print(f"Unique user journeys: {final_stats['user_journeys']}")
        
        # Shutdown system
        await delayed_reward_system.shutdown()
        
        print("✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())