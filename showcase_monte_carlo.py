#!/usr/bin/env python3
"""
Monte Carlo Simulator Showcase

Demonstrates the key features with visible output.
"""

import asyncio
import json
import numpy as np
import time
from datetime import datetime

from monte_carlo_simulator import MonteCarloSimulator, WorldType


class ShowcaseAgent:
    def __init__(self):
        self.agent_id = "showcase_agent"
        self.calls = 0
    
    def select_action(self, state, deterministic=False):
        self.calls += 1
        return {
            'bid': np.random.uniform(2.0, 8.0),
            'budget': np.random.uniform(300.0, 1200.0),
            'creative': {'quality_score': np.random.uniform(0.6, 0.95)},
            'quality_score': np.random.uniform(0.7, 0.95)
        }


async def showcase_demo():
    """Showcase the Monte Carlo simulator"""
    
    print("🚀 MONTE CARLO PARALLEL SIMULATION SHOWCASE")
    print("=" * 60)
    
    # Create simulator
    print("Creating Monte Carlo Simulator...")
    print("  📊 Worlds: 25")
    print("  🔄 Max Concurrent: 8")
    print("  🎯 Focus on Crisis Parent scenarios")
    
    # Custom distribution emphasizing crisis parents
    crisis_focused_distribution = {
        WorldType.NORMAL_MARKET: 0.20,
        WorldType.HIGH_COMPETITION: 0.20,
        WorldType.LOW_COMPETITION: 0.15,
        WorldType.SEASONAL_PEAK: 0.10,
        WorldType.ECONOMIC_DOWNTURN: 0.10,
        WorldType.CRISIS_PARENT: 0.15,  # Higher for demo
        WorldType.TECH_SAVVY: 0.04,
        WorldType.BUDGET_CONSCIOUS: 0.03,
        WorldType.IMPULSE_BUYER: 0.02,
        WorldType.LUXURY_SEEKER: 0.01
    }
    
    simulator = MonteCarloSimulator(
        n_worlds=25,
        world_types_distribution=crisis_focused_distribution,
        max_concurrent_worlds=8
    )
    
    agent = ShowcaseAgent()
    
    print(f"✅ Simulator created with {simulator.n_worlds} worlds")
    
    # Show world distribution
    print("\n📋 World Distribution:")
    for world_type, prob in crisis_focused_distribution.items():
        count = int(simulator.n_worlds * prob)
        if count > 0:
            print(f"  {world_type.value}: {count} worlds")
    
    try:
        results = []
        total_crisis_interactions = 0
        
        # Run 3 batches
        for batch_num in range(3):
            print(f"\n🔄 Running Batch {batch_num + 1}/3...")
            batch_start = time.time()
            
            # Run episodes
            experiences = await simulator.run_episode_batch(agent, batch_size=20)
            batch_time = time.time() - batch_start
            
            # Aggregate results
            aggregated = simulator.aggregate_experiences(experiences)
            
            # Count crisis parent interactions
            batch_crisis = sum(exp.crisis_parent_interactions for exp in experiences)
            total_crisis_interactions += batch_crisis
            
            # Display results
            print(f"  📈 Episodes: {len(experiences)}")
            print(f"  💰 Total Reward: {aggregated.get('total_reward', 0):.2f}")
            print(f"  📊 Success Rate: {aggregated.get('success_rate', 0):.1%}")
            print(f"  🚨 Crisis Interactions: {batch_crisis}")
            print(f"  ⏱️  Time: {batch_time:.2f}s ({len(experiences)/batch_time:.1f} episodes/sec)")
            
            results.append({
                'batch': batch_num + 1,
                'experiences': len(experiences),
                'success_rate': aggregated.get('success_rate', 0),
                'crisis_interactions': batch_crisis,
                'time': batch_time
            })
        
        # Final analysis
        print(f"\n📊 FINAL ANALYSIS")
        print("=" * 40)
        
        total_episodes = sum(r['experiences'] for r in results)
        avg_success_rate = np.mean([r['success_rate'] for r in results])
        total_time = sum(r['time'] for r in results)
        
        print(f"🏆 Total Episodes: {total_episodes}")
        print(f"📈 Average Success Rate: {avg_success_rate:.1%}")
        print(f"🚨 Total Crisis Interactions: {total_crisis_interactions}")
        print(f"📍 Crisis Rate: {total_crisis_interactions/total_episodes:.1%}")
        print(f"⚡ Performance: {total_episodes/total_time:.1f} episodes/second")
        
        # Demonstrate importance sampling
        print(f"\n🎯 IMPORTANCE SAMPLING DEMO")
        print("-" * 30)
        
        regular_samples = simulator.experience_buffer.sample_batch(50, importance_sampling=False)
        importance_samples = simulator.importance_sampling(target_samples=50, focus_rare_events=True)
        
        regular_crisis = sum(1 for exp in regular_samples if exp.crisis_parent_interactions > 0)
        importance_crisis = sum(1 for exp in importance_samples if exp.crisis_parent_interactions > 0)
        
        print(f"Regular Sampling: {regular_crisis}/50 ({regular_crisis*2}%) crisis events")
        print(f"Importance Sampling: {importance_crisis}/50 ({importance_crisis*2}%) crisis events")
        
        if regular_crisis > 0:
            boost = importance_crisis / regular_crisis
            print(f"🚀 Importance Boost: {boost:.1f}x more crisis events!")
        
        # Buffer stats
        buffer_stats = simulator.experience_buffer.get_buffer_stats()
        print(f"\n💾 EXPERIENCE BUFFER")
        print("-" * 20)
        print(f"Total Experiences: {buffer_stats['total_experiences']}")
        print(f"Crisis Experiences: {buffer_stats['crisis_parent_experiences']}")
        print(f"Average Weight: {buffer_stats['average_importance_weight']:.2f}")
        print(f"Max Weight: {buffer_stats['max_importance_weight']:.2f}")
        
        # Performance stats
        sim_stats = simulator.get_simulation_stats()
        overview = sim_stats['simulation_overview']
        
        print(f"\n⚡ PERFORMANCE METRICS")
        print("-" * 20)
        print(f"Episodes/Second: {overview['episodes_per_second']:.2f}")
        print(f"Success Rate: {overview['success_rate']:.1%}")
        print(f"Runtime: {overview['runtime_seconds']:.1f}s")
        
        # World performance
        print(f"\n🌍 TOP WORLD PERFORMERS")
        print("-" * 25)
        world_stats = sim_stats['world_statistics']
        top_worlds = sorted(world_stats, key=lambda x: x['success_rate'], reverse=True)[:5]
        
        for world in top_worlds:
            print(f"{world['world_type']}: {world['success_rate']:.1%} success, "
                  f"ROAS {world['average_roas']:.2f}")
        
        # Save results
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': total_episodes,
            'crisis_interactions': total_crisis_interactions,
            'performance': overview['episodes_per_second'],
            'success_rate': avg_success_rate,
            'world_stats': world_stats[:10],  # Top 10 worlds
            'buffer_stats': buffer_stats
        }
        
        with open('/home/hariravichandran/AELP/showcase_results.json', 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to showcase_results.json")
        
        print(f"\n🎉 SHOWCASE COMPLETE!")
        print("Key Features Demonstrated:")
        print("✅ Parallel world simulation")
        print("✅ Crisis parent rare event handling")
        print("✅ Importance sampling for training")
        print("✅ Experience aggregation")
        print("✅ Real-time performance metrics")
        print(f"✅ Scalable to 100+ worlds ({simulator.n_worlds} worlds used)")
        
    finally:
        simulator.cleanup()
        print("🧹 Cleanup complete")


if __name__ == "__main__":
    asyncio.run(showcase_demo())