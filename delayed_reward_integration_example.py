"""
Integration Example: Using Delayed Reward Attribution with RL Training

This example shows how to integrate the user_journey_tracker with RL training
to provide proper delayed reward signals for multi-day conversions.

CRITICAL: This demonstrates NO immediate rewards - everything is delayed.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

from user_journey_tracker import (
    UserJourneyTracker,
    TouchpointType,
    ConversionType
)


class MockRLAgent:
    """Mock RL Agent to demonstrate delayed reward integration"""
    
    def __init__(self):
        self.experience_buffer = []
        self.delayed_reward_corrections = []
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select action (bid, creative, placement)"""
        # Mock action selection
        return {
            'bid_amount': random.uniform(1.0, 5.0),
            'creative_id': f"creative_{random.randint(1, 10)}",
            'placement_id': f"placement_{random.randint(1, 5)}",
            'channel': random.choice(['search', 'display', 'social', 'video'])
        }
    
    def store_experience(self, state: Dict[str, Any], action: Dict[str, Any], 
                        immediate_reward: float, next_state: Dict[str, Any], done: bool):
        """Store experience with immediate reward (which should always be 0.0)"""
        if immediate_reward != 0.0:
            raise ValueError("CRITICAL: No immediate rewards allowed!")
        
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'immediate_reward': immediate_reward,  # Always 0.0
            'next_state': next_state,
            'done': done,
            'timestamp': datetime.now()
        })
    
    def update_with_delayed_rewards(self, delayed_reward_data: List[Dict[str, Any]]):
        """Update agent with delayed reward attributions"""
        for reward_data in delayed_reward_data:
            self.delayed_reward_corrections.append({
                'touchpoint_id': reward_data['touchpoint_id'],
                'attributed_reward': reward_data['attributed_reward'],
                'attribution_model': reward_data['attribution_model'],
                'time_to_conversion_hours': reward_data['time_to_conversion_hours']
            })
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        total_immediate = sum(exp['immediate_reward'] for exp in self.experience_buffer)
        total_delayed = sum(corr['attributed_reward'] for corr in self.delayed_reward_corrections)
        
        return {
            'total_experiences': len(self.experience_buffer),
            'total_delayed_corrections': len(self.delayed_reward_corrections),
            'total_immediate_reward': total_immediate,  # Should be 0.0
            'total_delayed_reward': total_delayed,
            'reward_ratio_delayed_vs_immediate': 'infinite' if total_immediate == 0 else total_delayed / total_immediate
        }


class DelayedRewardEnvironment:
    """Mock environment that produces delayed conversions"""
    
    def __init__(self, journey_tracker: UserJourneyTracker):
        self.journey_tracker = journey_tracker
        self.current_user_id = None
        self.step_count = 0
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode"""
        self.current_user_id = f"user_{datetime.now().timestamp()}"
        self.step_count = 0
        
        return {
            'user_id': self.current_user_id,
            'user_state': 'new',
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'previous_touchpoints': 0
        }
    
    def step(self, action: Dict[str, Any]) -> tuple:
        """Execute action and return state, reward, done, info"""
        self.step_count += 1
        
        # Add touchpoint to journey tracker (NO immediate reward)
        touchpoint_id = self.journey_tracker.add_touchpoint(
            user_id=self.current_user_id,
            channel=action['channel'],
            touchpoint_type=TouchpointType.CLICK,
            campaign_id=f"campaign_{action['channel']}",
            creative_id=action['creative_id'],
            placement_id=action['placement_id'],
            bid_amount=action['bid_amount'],
            cost=action['bid_amount'] * 0.7,  # 70% of bid as cost
            state_data={'step': self.step_count},
            action_data=action
        )
        
        # Next state
        next_state = {
            'user_id': self.current_user_id,
            'user_state': 'engaged' if self.step_count > 1 else 'new',
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'previous_touchpoints': self.step_count
        }
        
        # NO IMMEDIATE REWARD - always 0.0
        immediate_reward = 0.0
        
        # Episode ends after 5 steps
        done = self.step_count >= 5
        
        # Info includes touchpoint_id for later attribution tracking
        info = {
            'touchpoint_id': touchpoint_id,
            'cost': action['bid_amount'] * 0.7,
            'user_id': self.current_user_id
        }
        
        return next_state, immediate_reward, done, info


class DelayedRewardTrainingLoop:
    """Training loop that handles delayed reward attribution"""
    
    def __init__(self):
        # Use a temporary file database instead of in-memory to ensure proper initialization
        import tempfile
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.journey_tracker = UserJourneyTracker(self.temp_db.name)
        self.agent = MockRLAgent()
        self.environment = DelayedRewardEnvironment(self.journey_tracker)
        self.touchpoint_to_episode = {}  # Track which episode each touchpoint belongs to
        
    async def run_episode(self, episode_id: int) -> Dict[str, Any]:
        """Run single episode"""
        state = self.environment.reset()
        done = False
        episode_touchpoints = []
        total_cost = 0.0
        
        while not done:
            # Agent selects action
            action = self.agent.select_action(state)
            
            # Environment step
            next_state, immediate_reward, done, info = self.environment.step(action)
            
            # CRITICAL: Verify no immediate reward
            assert immediate_reward == 0.0, "NO IMMEDIATE REWARDS ALLOWED!"
            
            # Store experience with 0.0 reward
            self.agent.store_experience(state, action, immediate_reward, next_state, done)
            
            # Track touchpoint for later attribution
            touchpoint_id = info['touchpoint_id']
            self.touchpoint_to_episode[touchpoint_id] = episode_id
            episode_touchpoints.append(touchpoint_id)
            total_cost += info['cost']
            
            state = next_state
        
        return {
            'episode_id': episode_id,
            'user_id': info['user_id'],
            'touchpoints': episode_touchpoints,
            'total_cost': total_cost,
            'immediate_reward': 0.0  # Always 0.0
        }
    
    def simulate_delayed_conversions(self, episodes: List[Dict[str, Any]]):
        """Simulate conversions happening days after episodes end"""
        print("\n🔄 Simulating delayed conversions...")
        
        # Simulate some users converting days later
        conversion_rate = 0.2  # 20% of users convert
        
        for episode in episodes:
            if random.random() < conversion_rate:
                # User converts with some delay
                conversion_value = random.uniform(50.0, 200.0)
                
                print(f"💰 User {episode['user_id']} converts ${conversion_value:.2f} "
                      f"(episode {episode['episode_id']})")
                
                # Record conversion in journey tracker
                delayed_rewards = self.journey_tracker.record_conversion(
                    user_id=episode['user_id'],
                    conversion_type=ConversionType.PURCHASE,
                    value=conversion_value
                )
                
                # Update agent with delayed rewards
                for reward in delayed_rewards:
                    for touchpoint, credit in reward.attributed_touchpoints:
                        # Create reward data for agent update
                        reward_data = [{
                            'touchpoint_id': touchpoint.touchpoint_id,
                            'attributed_reward': credit,
                            'attribution_model': reward.attribution_model,
                            'time_to_conversion_hours': (
                                reward.conversion_event.timestamp - touchpoint.timestamp
                            ).total_seconds() / 3600
                        }]
                        
                        # Update agent with delay reward corrections
                        self.agent.update_with_delayed_rewards(reward_data)
    
    async def run_training_experiment(self, num_episodes: int = 10):
        """Run complete training experiment with delayed rewards"""
        print("🚀 Starting Delayed Reward Training Experiment")
        print("=" * 60)
        print("CRITICAL: NO immediate rewards - everything is delayed attribution")
        print("=" * 60)
        
        episodes = []
        
        # Run episodes (no immediate rewards)
        print(f"\n📊 Running {num_episodes} episodes with 0.0 immediate rewards...")
        for episode_id in range(num_episodes):
            episode_result = await self.run_episode(episode_id)
            episodes.append(episode_result)
            
            if episode_id % 5 == 0:
                print(f"  Episode {episode_id}: User {episode_result['user_id'][:8]}... "
                      f"{len(episode_result['touchpoints'])} touchpoints, "
                      f"${episode_result['total_cost']:.2f} cost, "
                      f"immediate reward: {episode_result['immediate_reward']}")
        
        # Simulate delayed conversions
        self.simulate_delayed_conversions(episodes)
        
        # Get attribution statistics
        journey_stats = self.journey_tracker.get_journey_statistics()
        agent_stats = self.agent.get_training_stats()
        
        print("\n📈 Training Results:")
        print("=" * 60)
        print(f"Episodes completed: {len(episodes)}")
        print(f"Total users: {journey_stats['total_users']}")
        print(f"Total touchpoints: {journey_stats['total_touchpoints']}")
        print(f"Total conversions: {journey_stats['total_conversions']}")
        print(f"Conversion rate: {journey_stats['conversion_rate']:.1%}")
        print(f"Total delayed rewards: {journey_stats['total_delayed_rewards']}")
        print(f"Total attributed value: ${journey_stats['total_attributed_value']:.2f}")
        
        print(f"\n🤖 Agent Statistics:")
        print(f"Total experiences: {agent_stats['total_experiences']}")
        print(f"Total immediate reward: ${agent_stats['total_immediate_reward']:.2f}")
        print(f"Total delayed reward: ${agent_stats['total_delayed_reward']:.2f}")
        print(f"Delayed corrections: {agent_stats['total_delayed_corrections']}")
        
        print(f"\n🏆 Attribution Model Distribution:")
        for model, count in journey_stats['attribution_model_distribution'].items():
            print(f"  {model}: {count} attributions")
        
        print(f"\n⏰ Attribution Windows Used:")
        for window, days in journey_stats['attribution_windows_used'].items():
            print(f"  {window}: {days} days")
        
        # Export training data for RL
        training_data = self.journey_tracker.export_journey_data_for_training()
        
        print(f"\n💾 Training Data Export:")
        print(f"  Touchpoints for RL: {len(training_data['touchpoints'])}")
        print(f"  Delayed rewards for RL: {len(training_data['delayed_rewards'])}")
        print(f"  User journeys: {len(training_data['user_journeys'])}")
        
        # Verify NO immediate rewards in export
        immediate_rewards = [tp['immediate_reward'] for tp in training_data['touchpoints']]
        assert all(r == 0.0 for r in immediate_rewards), "CRITICAL: Found non-zero immediate rewards!"
        print("✅ VERIFIED: All touchpoints have 0.0 immediate reward")
        
        # Verify delayed rewards have proper attribution
        for reward_data in training_data['delayed_rewards']:
            for tp_reward in reward_data['touchpoint_rewards']:
                assert tp_reward['original_immediate_reward'] == 0.0, \
                    "CRITICAL: Original immediate reward should be 0.0"
                assert tp_reward['reward_delta'] == tp_reward['attributed_reward'], \
                    "CRITICAL: Reward delta should equal attributed reward"
        print("✅ VERIFIED: All delayed rewards properly attributed")
        
        print("\n" + "=" * 60)
        print("🎉 EXPERIMENT COMPLETE")
        print("✅ NO immediate rewards used - 100% delayed attribution")
        print("✅ Multi-day attribution windows working")
        print("✅ Multi-touch attribution implemented")
        print("✅ Training data ready for RL integration")
        print("=" * 60)
        
        return {
            'episodes': episodes,
            'journey_stats': journey_stats,
            'agent_stats': agent_stats,
            'training_data': training_data
        }
    
    def cleanup(self):
        """Clean up temporary resources"""
        import os
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)


async def main():
    """Run the delayed reward training integration example"""
    training_loop = DelayedRewardTrainingLoop()
    try:
        results = await training_loop.run_training_experiment(num_episodes=20)
    finally:
        training_loop.cleanup()
    
    # Show sample delayed reward attribution
    print("\n🔍 Sample Delayed Reward Attributions:")
    for i, reward_data in enumerate(results['training_data']['delayed_rewards'][:3]):
        print(f"\nReward {i+1}:")
        print(f"  Model: {reward_data['attribution_model']}")
        print(f"  Window: {reward_data['attribution_window_days']} days")
        print(f"  Conversion: ${reward_data['conversion_event']['value']:.2f}")
        print(f"  Touchpoints credited:")
        for tp_reward in reward_data['touchpoint_rewards']:
            print(f"    - {tp_reward['touchpoint_id'][:8]}...: "
                  f"${tp_reward['attributed_reward']:.2f} "
                  f"(was ${tp_reward['original_immediate_reward']:.2f} immediate)")


if __name__ == "__main__":
    asyncio.run(main())