#!/usr/bin/env python3
"""
Comprehensive Demo of Monte Carlo Parallel Simulation Framework

This demo showcases all the key features:
1. 100+ parallel worlds simulation
2. Different user populations and market conditions
3. Crisis parent rare event simulation (10% frequency, 50% value)
4. Experience aggregation and importance sampling
5. Performance metrics and analytics
"""

import asyncio
import json
import logging
import numpy as np
import time
from datetime import datetime

from monte_carlo_simulator import (
    MonteCarloSimulator, 
    WorldType, 
    WorldConfiguration
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoAgent:
    """Enhanced demo agent that adapts to different market conditions"""
    
    def __init__(self):
        self.agent_id = "demo_agent_v1"
        self.episode_count = 0
        self.world_performance = {}
        
    def select_action(self, state, deterministic=False):
        """Select action with some basic intelligence"""
        self.episode_count += 1
        
        # Extract useful state information
        market_context = state.get("market_context", {})
        budget_info = state.get("budget_constraints", {})
        performance = state.get("performance_history", {})
        
        # Adapt bid based on competition and performance
        base_bid = np.random.uniform(2.0, 8.0)
        
        # Increase bid in high competition markets
        competition = market_context.get("competition_level", 0.5)
        competition_multiplier = 1 + (competition * 0.5)
        
        # Adjust based on past performance
        avg_roas = performance.get("avg_roas", 1.0)
        if avg_roas > 2.0:  # Performing well, increase spend
            performance_multiplier = 1.3
        elif avg_roas < 0.5:  # Performing poorly, decrease spend
            performance_multiplier = 0.7
        else:
            performance_multiplier = 1.0
        
        final_bid = base_bid * competition_multiplier * performance_multiplier
        
        # Budget allocation strategy
        remaining_budget = budget_info.get("remaining_budget", 1000)
        daily_budget = budget_info.get("daily_budget", 500)
        
        # Conservative budget allocation if low budget remaining
        budget_ratio = remaining_budget / max(daily_budget, 1)
        if budget_ratio < 0.3:
            budget_multiplier = 0.5
        elif budget_ratio > 2.0:
            budget_multiplier = 1.5
        else:
            budget_multiplier = 1.0
        
        return {
            'bid': max(0.5, min(20.0, final_bid)),  # Clamp bid
            'budget': max(50.0, min(2000.0, daily_budget * budget_multiplier)),
            'creative': {
                'quality_score': np.random.uniform(0.6, 0.95),
                'price_shown': np.random.uniform(25.0, 200.0),
                'brand_affinity': np.random.uniform(0.3, 0.9),
                'relevance': np.random.uniform(0.7, 0.95)
            },
            'quality_score': np.random.uniform(0.7, 0.95),
            'targeting_precision': np.random.uniform(0.5, 0.9)
        }


async def run_comprehensive_demo():
    """Run comprehensive Monte Carlo simulation demo"""
    
    logger.info("=" * 80)
    logger.info("MONTE CARLO PARALLEL SIMULATION FRAMEWORK DEMO")
    logger.info("=" * 80)
    
    # Initialize simulator with 100 worlds
    logger.info("Initializing Monte Carlo Simulator with 100 parallel worlds...")
    
    # Custom world distribution emphasizing crisis parent scenarios
    custom_distribution = {
        WorldType.NORMAL_MARKET: 0.25,
        WorldType.HIGH_COMPETITION: 0.20,
        WorldType.LOW_COMPETITION: 0.15,
        WorldType.SEASONAL_PEAK: 0.12,
        WorldType.ECONOMIC_DOWNTURN: 0.08,
        WorldType.CRISIS_PARENT: 0.10,  # Higher than default for demo
        WorldType.TECH_SAVVY: 0.04,
        WorldType.BUDGET_CONSCIOUS: 0.03,
        WorldType.IMPULSE_BUYER: 0.02,
        WorldType.LUXURY_SEEKER: 0.01
    }
    
    simulator = MonteCarloSimulator(
        n_worlds=100,
        world_types_distribution=custom_distribution,
        max_concurrent_worlds=20,  # Increase parallelism
        experience_buffer_size=1000000
    )
    
    # Create intelligent demo agent
    agent = DemoAgent()
    
    logger.info(f"Created simulator with {simulator.n_worlds} worlds")
    logger.info("World distribution:")
    for world_type, prob in custom_distribution.items():
        count = int(simulator.n_worlds * prob)
        logger.info(f"  {world_type.value}: {count} worlds ({prob:.1%})")
    
    # Track performance across multiple batches
    all_experiences = []
    batch_results = []
    crisis_parent_encounters = 0
    total_crisis_revenue = 0.0
    
    try:
        # Run multiple episode batches
        n_batches = 5
        episodes_per_batch = 50
        
        for batch_num in range(n_batches):
            batch_start = time.time()
            
            logger.info(f"\n--- Running Batch {batch_num + 1}/{n_batches} ---")
            logger.info(f"Episodes per batch: {episodes_per_batch}")
            
            # Run episode batch across all worlds
            experiences = await simulator.run_episode_batch(
                agent, 
                batch_size=episodes_per_batch
            )
            
            batch_time = time.time() - batch_start
            
            # Aggregate experiences
            aggregated = simulator.aggregate_experiences(experiences)
            
            # Track crisis parent interactions
            batch_crisis_interactions = sum(exp.crisis_parent_interactions for exp in experiences)
            batch_crisis_revenue = sum(exp.crisis_parent_revenue for exp in experiences)
            crisis_parent_encounters += batch_crisis_interactions
            total_crisis_revenue += batch_crisis_revenue
            
            # Store results
            batch_result = {
                'batch_num': batch_num + 1,
                'experiences': len(experiences),
                'total_reward': aggregated.get('total_reward', 0),
                'average_reward': aggregated.get('average_reward', 0),
                'success_rate': aggregated.get('success_rate', 0),
                'crisis_interactions': batch_crisis_interactions,
                'crisis_revenue': batch_crisis_revenue,
                'batch_time': batch_time,
                'episodes_per_second': len(experiences) / batch_time,
                'world_breakdown': aggregated.get('world_type_breakdown', {})
            }
            batch_results.append(batch_result)
            all_experiences.extend(experiences)
            
            # Log batch results
            logger.info(f"Batch {batch_num + 1} Results:")
            logger.info(f"  Episodes: {len(experiences)}")
            logger.info(f"  Total Reward: {aggregated.get('total_reward', 0):.2f}")
            logger.info(f"  Average Reward: {aggregated.get('average_reward', 0):.3f}")
            logger.info(f"  Success Rate: {aggregated.get('success_rate', 0):.2%}")
            logger.info(f"  Crisis Parent Interactions: {batch_crisis_interactions}")
            logger.info(f"  Crisis Revenue: ${batch_crisis_revenue:.2f}")
            logger.info(f"  Time: {batch_time:.2f}s ({len(experiences) / batch_time:.1f} episodes/sec)")
            
            # Show top performing world types
            world_breakdown = aggregated.get('world_type_breakdown', {})
            if world_breakdown:
                logger.info("  Top World Types by Success Rate:")
                sorted_worlds = sorted(
                    world_breakdown.items(), 
                    key=lambda x: x[1].get('success_rate', 0), 
                    reverse=True
                )[:3]
                for world_type, stats in sorted_worlds:
                    logger.info(f"    {world_type}: {stats.get('success_rate', 0):.2%} success, "
                              f"{stats.get('average_reward', 0):.3f} avg reward")
        
        # Final Analysis
        logger.info(f"\n{'=' * 60}")
        logger.info("FINAL SIMULATION ANALYSIS")
        logger.info(f"{'=' * 60}")
        
        # Overall statistics
        total_episodes = sum(batch['experiences'] for batch in batch_results)
        total_reward = sum(batch['total_reward'] for batch in batch_results)
        overall_success_rate = np.mean([batch['success_rate'] for batch in batch_results])
        avg_episodes_per_second = np.mean([batch['episodes_per_second'] for batch in batch_results])
        
        logger.info(f"Total Episodes Run: {total_episodes}")
        logger.info(f"Total Reward: {total_reward:.2f}")
        logger.info(f"Average Reward per Episode: {total_reward / max(1, total_episodes):.3f}")
        logger.info(f"Overall Success Rate: {overall_success_rate:.2%}")
        logger.info(f"Average Performance: {avg_episodes_per_second:.1f} episodes/second")
        
        # Crisis parent analysis
        crisis_parent_rate = crisis_parent_encounters / max(1, total_episodes)
        crisis_revenue_share = total_crisis_revenue / max(1, sum(exp.revenue_generated for exp in all_experiences))
        
        logger.info(f"\nCrisis Parent Analysis:")
        logger.info(f"  Total Crisis Parent Interactions: {crisis_parent_encounters}")
        logger.info(f"  Crisis Parent Rate: {crisis_parent_rate:.2%} (Target: ~10%)")
        logger.info(f"  Crisis Parent Revenue: ${total_crisis_revenue:.2f}")
        logger.info(f"  Crisis Revenue Share: {crisis_revenue_share:.2%} (Target: ~50%)")
        
        if crisis_parent_rate > 0:
            avg_crisis_value = total_crisis_revenue / crisis_parent_encounters
            logger.info(f"  Average Crisis Parent Value: ${avg_crisis_value:.2f}")
        
        # Demonstrate importance sampling
        logger.info(f"\n--- Importance Sampling Demo ---")
        
        # Sample with and without importance sampling
        regular_samples = simulator.experience_buffer.sample_batch(100, importance_sampling=False)
        importance_samples = simulator.importance_sampling(target_samples=100, focus_rare_events=True)
        
        regular_crisis_count = sum(1 for exp in regular_samples if exp.crisis_parent_interactions > 0)
        importance_crisis_count = sum(1 for exp in importance_samples if exp.crisis_parent_interactions > 0)
        
        logger.info(f"Regular Sampling: {regular_crisis_count}/100 ({regular_crisis_count}%) crisis parent experiences")
        logger.info(f"Importance Sampling: {importance_crisis_count}/100 ({importance_crisis_count}%) crisis parent experiences")
        logger.info(f"Importance Sampling Boost: {importance_crisis_count / max(1, regular_crisis_count):.1f}x more crisis events")
        
        # Buffer statistics
        buffer_stats = simulator.experience_buffer.get_buffer_stats()
        logger.info(f"\nExperience Buffer Statistics:")
        logger.info(f"  Total Experiences: {buffer_stats['total_experiences']}")
        logger.info(f"  Crisis Parent Experiences: {buffer_stats['crisis_parent_experiences']}")
        logger.info(f"  Crisis Parent Ratio: {buffer_stats['crisis_parent_ratio']:.2%}")
        logger.info(f"  Average Importance Weight: {buffer_stats['average_importance_weight']:.2f}")
        logger.info(f"  Max Importance Weight: {buffer_stats['max_importance_weight']:.2f}")
        logger.info(f"  Buffer Utilization: {buffer_stats['buffer_utilization']:.1%}")
        
        # Simulation performance stats
        sim_stats = simulator.get_simulation_stats()
        sim_overview = sim_stats['simulation_overview']
        
        logger.info(f"\nSimulation Performance:")
        logger.info(f"  Runtime: {sim_overview['runtime_seconds']:.1f} seconds")
        logger.info(f"  Episodes per Second: {sim_overview['episodes_per_second']:.2f}")
        logger.info(f"  Concurrent Worlds: {sim_overview['max_concurrent_worlds']}")
        logger.info(f"  Success Rate: {sim_overview['success_rate']:.2%}")
        
        # World performance analysis
        logger.info(f"\n--- World Type Performance Analysis ---")
        world_stats = sim_stats['world_statistics']
        
        # Sort by success rate
        sorted_world_stats = sorted(world_stats, key=lambda x: x['success_rate'], reverse=True)
        
        for world_stat in sorted_world_stats[:10]:  # Top 10 worlds
            logger.info(f"{world_stat['world_type']} ({world_stat['world_id']}):")
            logger.info(f"  Success Rate: {world_stat['success_rate']:.2%}")
            logger.info(f"  Average ROAS: {world_stat['average_roas']:.2f}")
            logger.info(f"  Episodes: {world_stat['total_episodes']}")
            logger.info(f"  Crisis Rate: {world_stat['crisis_parent_rate']:.2%}")
        
        # Save detailed results
        results_data = {
            'simulation_overview': sim_stats['simulation_overview'],
            'batch_results': batch_results,
            'crisis_analysis': {
                'total_interactions': crisis_parent_encounters,
                'interaction_rate': crisis_parent_rate,
                'total_revenue': total_crisis_revenue,
                'revenue_share': crisis_revenue_share
            },
            'world_performance': sorted_world_stats,
            'buffer_stats': buffer_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('/home/hariravichandran/AELP/monte_carlo_demo_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to monte_carlo_demo_results.json")
        
        # Save experiences for later analysis
        simulator.save_experiences('/home/hariravichandran/AELP/monte_carlo_experiences.pkl')
        logger.info(f"Experience data saved to monte_carlo_experiences.pkl")
        
        logger.info(f"\n{'=' * 60}")
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info(f"{'=' * 60}")
        
        # Key takeaways
        logger.info("Key Takeaways:")
        logger.info(f"✓ Ran {total_episodes} episodes across {simulator.n_worlds} parallel worlds")
        logger.info(f"✓ Achieved {avg_episodes_per_second:.1f} episodes/second with parallel execution")
        logger.info(f"✓ Crisis parent events occurred at {crisis_parent_rate:.1%} rate (target ~10%)")
        logger.info(f"✓ Importance sampling increased crisis event representation by {importance_crisis_count / max(1, regular_crisis_count):.1f}x")
        logger.info(f"✓ Experience buffer holds {buffer_stats['total_experiences']} experiences")
        logger.info(f"✓ Successfully demonstrated scalable Monte Carlo simulation framework")
        
    finally:
        simulator.cleanup()


async def main():
    """Main demo function"""
    try:
        await run_comprehensive_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())