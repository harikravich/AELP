#!/usr/bin/env python3
"""
Updated Training Example Using RecSim-AuctionGym Bridge

This shows how to update training loops to use realistic user segments 
and auction dynamics instead of random generation.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List
import logging
from unittest.mock import Mock

# Import the RecSim-AuctionGym bridge - REQUIRED
from recsim_auction_bridge import RecSimAuctionBridge, UserSegment
from recsim_user_model import RecSimUserModel

# Import training components - REQUIRED
from online_learner import OnlineLearner, OnlineLearnerConfig


class EnhancedMockAgent:
    """
    Updated mock agent that uses RecSim-AuctionGym bridge for action selection
    """
    
    def __init__(self, use_bridge: bool = True):
        self.agent_id = "enhanced_mock_agent"
        self.config = Mock()
        self.config.learning_rate = 0.001
        self.step = 0
        
        # Initialize RecSim bridge
        self.use_bridge = use_bridge and BRIDGE_AVAILABLE
        if self.use_bridge:
            self.bridge = RecSimAuctionBridge()
            print("🎯 Agent using RecSim-AuctionGym bridge for realistic actions")
        else:
            self.bridge = None
            print("⚠️ Agent RecSim REQUIRED: random actions") not available
        
        # Track current user for session continuity
        self.current_user_id = None
        self.current_user_segment = None
    
    async def select_action(self, state: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """
        BEFORE: Random action selection
        AFTER: Use RecSim bridge to generate realistic actions based on user segments
        """
        self.step += 1
        
        if not self.use_bridge:
            # Old way - random actions
            return {
                "creative_type": "image",
                "budget": 100.0 * (1.0 if deterministic else np.random.uniform(0.9, 1.1)),
                "bid_amount": 2.0,
                "target_audience": "professionals", 
                "bid_strategy": "cpc",
                "audience_size": 0.5,
                "product_category": "shoes",
                "method": "random"
            }
        
        # NEW: Generate or continue with realistic user
        if not self.current_user_id or self.step % 10 == 0:  # New user every 10 steps
            self.current_user_id = f"training_user_{self.step}"
            
            # Get user auction signals
            context = {
                'hour': state.get('hour', np.random.randint(9, 22)),
                'device': np.random.choice(['mobile', 'desktop', 'tablet'])
            }
            
            user_signals = self.bridge.user_to_auction_signals(
                user_id=self.current_user_id,
                context=context
            )
            self.current_user_segment = user_signals['segment']
        
        # Generate realistic query and intent
        product_category = np.random.choice(['shoes', 'electronics', 'books', 'clothing'])
        query_data = self.bridge.generate_query_from_state(
            user_id=self.current_user_id,
            product_category=product_category
        )
        
        # Get user-specific auction signals
        auction_signals = self.bridge.user_to_auction_signals(
            user_id=self.current_user_id,
            context={'product_category': product_category}
        )
        
        # Calculate realistic bid using segment-specific logic
        optimal_bid = self.bridge.map_segment_to_bid_value(
            segment=UserSegment(auction_signals['segment']),
            query_intent=query_data['intent'],
            market_context={'competition_level': state.get('competition_level', 0.5)}
        )
        
        # Adjust budget based on user segment and market conditions
        base_budget = 100.0
        if auction_signals['segment'] in ['impulse_buyer', 'loyal_customer']:
            base_budget *= 1.3
        elif auction_signals['segment'] in ['price_conscious', 'window_shopper']:
            base_budget *= 0.7
        
        if not deterministic:
            base_budget *= np.random.uniform(0.9, 1.1)
        
        # Create action based on RecSim insights
        action = {
            "creative_type": "image" if query_data['query_type'] == 'informational' else "video",
            "budget": min(base_budget, state.get('budget_remaining', 1000)),
            "bid_amount": optimal_bid,
            "target_audience": self._map_segment_to_audience(auction_signals['segment']),
            "bid_strategy": "cpc",
            "audience_size": auction_signals['price_sensitivity'],
            "product_category": product_category,
            "quality_score": auction_signals['quality_score'],
            
            # NEW: RecSim-specific fields
            "user_segment": auction_signals['segment'],
            "query_generated": query_data['query'],
            "journey_stage": query_data['journey_stage'],
            "intent_strength": query_data['intent_strength'],
            "suggested_bid": auction_signals['suggested_bid'],
            "participation_probability": auction_signals['participation_probability'],
            "method": "recsim_bridge"
        }
        
        return action
    
    def _map_segment_to_audience(self, segment: str) -> str:
        """Map RecSim segments to audience targeting"""
        mapping = {
            'impulse_buyer': 'shoppers',
            'researcher': 'professionals', 
            'loyal_customer': 'existing_customers',
            'window_shopper': 'browsing_users',
            'price_conscious': 'deal_seekers',
            'brand_loyalist': 'brand_enthusiasts'
        }
        return mapping.get(segment, 'general')
    
    def update_policy(self, experiences: List[Dict]) -> Dict:
        """Enhanced policy update with RecSim insights"""
        if not self.use_bridge or not experiences:
            return {"loss": 0.1}
        
        # Analyze segment performance
        segment_performance = {}
        for exp in experiences:
            action = exp.get('action', {})
            segment = action.get('user_segment', 'unknown')
            
            if segment not in segment_performance:
                segment_performance[segment] = []
            segment_performance[segment].append(exp.get('reward', 0))
        
        # Calculate segment-specific losses
        total_loss = 0
        for segment, rewards in segment_performance.items():
            avg_reward = np.mean(rewards)
            # Higher loss for poor-performing segments
            segment_loss = max(0.01, 1.0 - avg_reward) if avg_reward >= 0 else 1.5
            total_loss += segment_loss
        
        return {
            "loss": total_loss / len(segment_performance) if segment_performance else 0.1,
            "segment_performance": segment_performance,
            "policy_updates": len(experiences)
        }
    
    def get_state(self) -> Dict:
        return {
            "step": self.step,
            "current_user_id": self.current_user_id,
            "current_segment": self.current_user_segment,
            "using_bridge": self.use_bridge
        }
    
    def load_state(self, state: Dict):
        self.step = state.get("step", 0)
        self.current_user_id = state.get("current_user_id")
        self.current_user_segment = state.get("current_segment")


class RealisticEnvironmentSimulator:
    """
    Simulates realistic environment responses using RecSim-AuctionGym bridge
    """
    
    def __init__(self):
        self.use_bridge = BRIDGE_AVAILABLE
        if self.use_bridge:
            self.bridge = RecSimAuctionBridge()
            print("🌟 Environment using RecSim-AuctionGym bridge for realistic responses")
        else:
            print("⚠️ Environment RecSim REQUIRED: random responses") not available
        
        self.episode_count = 0
        
    def simulate_environment_step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        BEFORE: Random environment responses
        AFTER: Use RecSim bridge to simulate realistic user responses and auction outcomes
        """
        if not self.use_bridge:
            # Old fallback
            return {
                'reward': np.random.normal(0.5, 0.2),
                'clicked': np.random.random() < 0.05,
                'converted': np.random.random() < 0.01,
                'cost': action.get('bid_amount', 2.0) * np.random.uniform(0.5, 1.0),
                'revenue': np.random.gamma(2, 30) if np.random.random() < 0.01 else 0,
                'method': 'random'
            }
        
        # NEW: Use RecSim bridge for realistic simulation
        user_id = action.get('current_user_id', f"env_user_{self.episode_count}")
        
        # Check if user participates in auction based on their segment
        if 'participation_probability' in action:
            if np.random.random() > action['participation_probability']:
                return {
                    'reward': -0.1,  # Small penalty for no participation
                    'clicked': False,
                    'converted': False,
                    'cost': 0,
                    'revenue': 0,
                    'reason': 'User did not participate',
                    'method': 'recsim_bridge'
                }
        
        # Run actual auction through the bridge
        context = {
            'hour': np.random.randint(9, 22),
            'device': 'mobile',
            'product_category': action.get('product_category', 'shoes')
        }
        
        # Use the bridge's auction wrapper
        auction_result = self.bridge.auction_wrapper.run_auction(
            your_bid=action.get('bid_amount', 2.0),
            your_quality_score=action.get('quality_score', 0.7),
            context=context
        )
        
        # Calculate reward based on realistic outcomes
        if auction_result.won and auction_result.outcome:
            # User clicked - calculate conversion based on segment
            segment = action.get('user_segment', 'unknown')
            intent_strength = action.get('intent_strength', 0.5)
            
            # Segment-specific conversion rates
            conversion_rates = {
                'impulse_buyer': 0.15,
                'researcher': 0.05, 
                'loyal_customer': 0.20,
                'window_shopper': 0.02,
                'price_conscious': 0.08,
                'brand_loyalist': 0.18
            }
            
            base_conv_rate = conversion_rates.get(segment, 0.05)
            final_conv_rate = base_conv_rate * (0.5 + intent_strength)
            
            converted = np.random.random() < final_conv_rate
            revenue = np.random.gamma(2, 50) if converted else 0
            
            # Calculate ROAS-based reward
            cost = auction_result.price_paid
            reward = (revenue - cost) / max(cost, 0.01)
            
            return {
                'reward': reward,
                'clicked': True,
                'converted': converted,
                'cost': cost,
                'revenue': revenue,
                'position': auction_result.slot_position,
                'competitors': auction_result.competitors,
                'user_segment': segment,
                'journey_stage': action.get('journey_stage', 'unknown'),
                'method': 'recsim_bridge'
            }
        else:
            return {
                'reward': -0.05,  # Small penalty for losing auction or no click
                'clicked': False,
                'converted': False,
                'cost': 0,
                'revenue': 0,
                'auction_won': auction_result.won,
                'position': auction_result.slot_position if auction_result.won else 99,
                'method': 'recsim_bridge'
            }


async def run_enhanced_training_scenario():
    """
    Demonstrate enhanced training scenario with RecSim-AuctionGym bridge
    """
    print("🚀 Enhanced Training Scenario with RecSim-AuctionGym Bridge")
    print("=" * 70)
    
    # Create enhanced components
    agent = EnhancedMockAgent(use_bridge=True)
    env_sim = RealisticEnvironmentSimulator()
    
    # Training configuration  
    config = {
        "bandit_arms": ["conservative", "balanced", "aggressive", "experimental"],
        "online_update_frequency": 10,
        "safety_threshold": 0.6,
        "max_budget_risk": 0.2
    }
    
    if LEARNER_AVAILABLE:
        try:
            from online_learner import create_online_learner
            learner = create_online_learner(agent, config)
            learner.redis_client = Mock()
            learner.bigquery_client = Mock()
            print("✅ Using enhanced online learner")
        except Exception as e:
            print(f"⚠️ Online learner not available: {e}")
            learner = None
    else:
        learner = None
    
    # Run training episodes
    results = {
        'episodes': [],
        'segment_performance': {},
        'journey_insights': {},
        'total_conversions': 0,
        'total_cost': 0,
        'total_revenue': 0
    }
    
    episodes = 50
    print(f"\n📈 Running {episodes} training episodes...")
    
    for episode in range(episodes):
        env_sim.episode_count = episode
        
        # Create realistic state
        state = {
            'budget_remaining': 1000.0 * (1 - episode / episodes),
            'competition_level': 0.3 + 0.4 * np.sin(episode * 0.1),
            'hour': np.random.randint(9, 22),
            'market_context': {
                'seasonality': 0.9 + 0.2 * np.sin(episode * 0.05)
            }
        }
        
        # Agent selects action (using RecSim bridge if available)
        action = await agent.select_action(state, deterministic=False)
        action['current_user_id'] = agent.current_user_id  # Pass user context
        
        # Environment responds (using RecSim bridge if available)
        env_response = env_sim.simulate_environment_step(action)
        
        # Update results
        episode_result = {
            'episode': episode,
            'action': action,
            'response': env_response,
            'reward': env_response['reward']
        }
        results['episodes'].append(episode_result)
        
        # Track metrics
        results['total_conversions'] += int(env_response.get('converted', False))
        results['total_cost'] += env_response.get('cost', 0)
        results['total_revenue'] += env_response.get('revenue', 0)
        
        # Track segment performance (NEW with RecSim)
        if BRIDGE_AVAILABLE and env_response.get('method') == 'recsim_bridge':
            segment = env_response.get('user_segment', 'unknown')
            if segment not in results['segment_performance']:
                results['segment_performance'][segment] = {
                    'episodes': 0,
                    'conversions': 0,
                    'total_reward': 0,
                    'cost': 0,
                    'revenue': 0
                }
            
            results['segment_performance'][segment]['episodes'] += 1
            results['segment_performance'][segment]['conversions'] += int(env_response.get('converted', False))
            results['segment_performance'][segment]['total_reward'] += env_response['reward']
            results['segment_performance'][segment]['cost'] += env_response.get('cost', 0)
            results['segment_performance'][segment]['revenue'] += env_response.get('revenue', 0)
            
            # Track journey stages
            journey_stage = env_response.get('journey_stage', 'unknown')
            results['journey_insights'][journey_stage] = results['journey_insights'].get(journey_stage, 0) + 1
        
        # Online learning update
        if learner and episode % config['online_update_frequency'] == 9:
            recent_episodes = results['episodes'][-config['online_update_frequency']:]
            experiences = [
                {
                    'state': state,
                    'action': ep['action'],
                    'reward': ep['reward'],
                    'next_state': state,  # Simplified
                    'done': False
                }
                for ep in recent_episodes
            ]
            
            await learner.online_update(experiences, force_update=True)
        
        # Progress update
        if episode % 10 == 9:
            recent_rewards = [ep['reward'] for ep in results['episodes'][-10:]]
            avg_reward = np.mean(recent_rewards)
            print(f"  Episode {episode+1}: Avg reward = {avg_reward:.3f}")
    
    return results, agent, env_sim, learner


def analyze_enhanced_results(results: Dict, agent: EnhancedMockAgent):
    """Analyze results with RecSim-specific insights"""
    
    print("\n📊 ENHANCED TRAINING RESULTS")
    print("=" * 50)
    
    # Basic metrics
    episodes = len(results['episodes'])
    avg_reward = np.mean([ep['reward'] for ep in results['episodes']])
    total_conversions = results['total_conversions']
    conversion_rate = total_conversions / episodes if episodes > 0 else 0
    roas = results['total_revenue'] / max(results['total_cost'], 0.01)
    
    print(f"Episodes: {episodes}")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Total Conversions: {total_conversions}")
    print(f"Conversion Rate: {conversion_rate:.2%}")
    print(f"ROAS: {roas:.2f}x")
    print(f"Total Cost: ${results['total_cost']:.2f}")
    print(f"Total Revenue: ${results['total_revenue']:.2f}")
    
    # RecSim-specific insights
    if BRIDGE_AVAILABLE and results['segment_performance']:
        print(f"\n🎯 RecSim Segment Performance:")
        print("-" * 40)
        for segment, perf in results['segment_performance'].items():
            if perf['episodes'] > 0:
                seg_roas = perf['revenue'] / max(perf['cost'], 0.01)
                seg_conv_rate = perf['conversions'] / perf['episodes']
                avg_reward = perf['total_reward'] / perf['episodes']
                
                print(f"  {segment.upper()}:")
                print(f"    Episodes: {perf['episodes']}")
                print(f"    Conversions: {perf['conversions']} ({seg_conv_rate:.1%})")
                print(f"    ROAS: {seg_roas:.2f}x")
                print(f"    Avg Reward: {avg_reward:.3f}")
                print()
        
        print(f"🎪 Journey Stage Distribution:")
        print("-" * 40)
        total_stages = sum(results['journey_insights'].values())
        for stage, count in sorted(results['journey_insights'].items()):
            percentage = count / total_stages * 100 if total_stages > 0 else 0
            print(f"  {stage}: {count} ({percentage:.1f}%)")
        
        print(f"\n🤖 Agent State:")
        print("-" * 20)
        agent_state = agent.get_state()
        print(f"  Using Bridge: {agent_state['using_bridge']}")
        print(f"  Total Steps: {agent_state['step']}")
        if agent_state.get('current_segment'):
            print(f"  Current Segment: {agent_state['current_segment']}")
    
    else:
        from NO_FALLBACKS import StrictModeEnforcer
        StrictModeEnforcer.enforce('RECSIM_BRIDGE_TRAINING', fallback_attempted=True)
        raise RuntimeError("RecSim bridge REQUIRED for training. NO FALLBACKS ALLOWED!")


async def main():
    """Main training demonstration"""
    print("🎯 RecSim-AuctionGym Bridge Training Integration")
    print("=" * 60)
    print("This demonstrates how to update training loops to use")
    print("realistic user segments instead of random generation.")
    print("=" * 60)
    
    # Check availability
    print(f"\n🔍 Component Status:")
    if not BRIDGE_AVAILABLE:
        from NO_FALLBACKS import StrictModeEnforcer
        StrictModeEnforcer.enforce('RECSIM_BRIDGE_AVAILABILITY', fallback_attempted=True)
        raise RuntimeError("RecSim-AuctionGym Bridge MUST be available. NO FALLBACKS!")
    print(f"  RecSim-AuctionGym Bridge: ✅ MANDATORY and Available")
    print(f"  Online Learner: {'✅ Available' if LEARNER_AVAILABLE else '❌ Not Available'}")
    
    # Run enhanced training
    results, agent, env_sim, learner = await run_enhanced_training_scenario()
    
    # Analyze results
    analyze_enhanced_results(results, agent)
    
    # Show comparison
    print(f"\n🔄 BEFORE vs AFTER Comparison:")
    print("=" * 40)
    print("BEFORE (Random):")
    print("  ❌ Fake user segments")
    print("  ❌ Random auction outcomes") 
    print("  ❌ No journey progression")
    print("  ❌ Unrealistic conversion rates")
    
    print("\nAFTER (RecSim-AuctionGym Bridge):")
    if BRIDGE_AVAILABLE:
        print("  ✅ RecSim 6 authentic segments")
        print("  ✅ Realistic auction competition")
        print("  ✅ Journey-aware query generation")
        print("  ✅ Segment-specific bidding logic")
        print("  ✅ Intent-driven conversion rates")
    else:
        print("  ⚠️ Bridge not available - using fallbacks")
    
    # Cleanup
    if learner:
        await learner.shutdown()
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())