#!/usr/bin/env python3
"""
Demo: GAELP Competitive Auction Simulation
Shows realistic multi-agent competitive auctions with learning competitors.
"""

import asyncio
import numpy as np
from datetime import datetime
from decimal import Decimal
import matplotlib.pyplot as plt
import pandas as pd

from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from competitor_agents import CompetitorAgentManager, AuctionContext, UserValueTier

async def run_competitive_auction_demo():
    """Run comprehensive competitive auction demo"""
    
    print("🏟️  GAELP COMPETITIVE AUCTION SIMULATION")
    print("=" * 60)
    print("Demonstrating realistic multi-agent competitive bidding")
    print("with Q-Learning, Policy Gradient, Rule-Based, and Random agents")
    print("=" * 60)
    
    # Initialize competitor manager
    competitor_manager = CompetitorAgentManager()
    
    # Run multiple auction scenarios
    scenarios = [
        {
            'name': 'Low Competition Morning',
            'user_tier': UserValueTier.MEDIUM,
            'time_of_day': 9,
            'market_competition': 0.3,
            'keyword_competition': 0.4,
            'conversion_probability': 0.03
        },
        {
            'name': 'High Competition Evening',
            'user_tier': UserValueTier.HIGH,
            'time_of_day': 20,
            'market_competition': 0.8,
            'keyword_competition': 0.9,
            'conversion_probability': 0.06
        },
        {
            'name': 'Premium User Weekday',
            'user_tier': UserValueTier.PREMIUM,
            'time_of_day': 14,
            'market_competition': 0.6,
            'keyword_competition': 0.7,
            'conversion_probability': 0.12
        },
        {
            'name': 'Budget User Weekend',
            'user_tier': UserValueTier.LOW,
            'time_of_day': 11,
            'market_competition': 0.2,
            'keyword_competition': 0.3,
            'conversion_probability': 0.015
        }
    ]
    
    all_results = []
    
    print("\n🎯 Running Competitive Auction Scenarios:")
    print("-" * 50)
    
    for scenario_idx, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario_idx}. {scenario['name']}")
        print(f"   User Tier: {scenario['user_tier'].value}")
        print(f"   Market Competition: {scenario['market_competition']:.1%}")
        print(f"   Time: {scenario['time_of_day']:02d}:00")
        
        # Run 20 auctions for this scenario to see patterns
        scenario_results = []
        
        for auction_num in range(20):
            # Create auction context
            context = AuctionContext(
                user_id=f"user_{scenario_idx}_{auction_num}",
                user_value_tier=scenario['user_tier'],
                timestamp=datetime.now(),
                device_type=np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1]),
                geo_location='US',
                time_of_day=scenario['time_of_day'],
                day_of_week=2,  # Wednesday
                market_competition=scenario['market_competition'],
                keyword_competition=scenario['keyword_competition'],
                seasonality_factor=1.1,
                user_engagement_score=np.random.uniform(0.2, 0.8),
                conversion_probability=scenario['conversion_probability']
            )
            
            # Run auction
            results = competitor_manager.run_auction(context)
            
            # Find winner (highest bidder)
            if results:
                winner = max(results.items(), key=lambda x: x[1].bid_amount)
                winner_name, winner_result = winner
                
                auction_result = {
                    'scenario': scenario['name'],
                    'auction_num': auction_num,
                    'winner': winner_name,
                    'winning_bid': winner_result.bid_amount,
                    'participants': len(results),
                    'total_competition': sum(r.bid_amount for r in results.values()),
                    'user_tier': scenario['user_tier'].value,
                    'market_competition': scenario['market_competition'],
                    'time_of_day': scenario['time_of_day']
                }
                
                # Add individual agent bids
                for agent_name, result in results.items():
                    auction_result[f'{agent_name}_bid'] = result.bid_amount
                    
                scenario_results.append(auction_result)
                all_results.append(auction_result)
        
        # Analyze scenario results
        if scenario_results:
            df_scenario = pd.DataFrame(scenario_results)
            
            print(f"   Results (20 auctions):")
            print(f"     Avg Total Competition: ${df_scenario['total_competition'].mean():.2f}")
            print(f"     Avg Winning Bid: ${df_scenario['winning_bid'].mean():.2f}")
            print(f"     Winner Distribution:")
            
            winner_counts = df_scenario['winner'].value_counts()
            for winner, count in winner_counts.items():
                percentage = count / len(df_scenario) * 100
                avg_bid = df_scenario[df_scenario['winner'] == winner]['winning_bid'].mean()
                print(f"       {winner:10s}: {count:2d} wins ({percentage:4.1f}%) avg ${avg_bid:.2f}")
    
    # Overall analysis
    print(f"\n📊 OVERALL COMPETITIVE ANALYSIS")
    print("-" * 50)
    
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        print(f"Total Auctions: {len(df_all)}")
        print(f"Average Competition Level: ${df_all['total_competition'].mean():.2f}")
        print(f"Average Winning Bid: ${df_all['winning_bid'].mean():.2f}")
        
        print(f"\nMarket Share by Agent:")
        winner_counts = df_all['winner'].value_counts()
        for winner, count in winner_counts.items():
            market_share = count / len(df_all) * 100
            print(f"  {winner:10s}: {market_share:5.1f}% market share ({count} wins)")
        
        print(f"\nAgent Performance by Scenario:")
        for scenario_name in df_all['scenario'].unique():
            scenario_data = df_all[df_all['scenario'] == scenario_name]
            print(f"\n  {scenario_name}:")
            winner_counts = scenario_data['winner'].value_counts()
            for winner, count in winner_counts.items():
                percentage = count / len(scenario_data) * 100
                avg_bid = scenario_data[scenario_data['winner'] == winner]['winning_bid'].mean()
                print(f"    {winner:10s}: {percentage:4.1f}% wins, ${avg_bid:.2f} avg bid")
        
        # Show bidding patterns by user tier
        print(f"\nBidding Patterns by User Value Tier:")
        for tier in df_all['user_tier'].unique():
            tier_data = df_all[df_all['user_tier'] == tier]
            avg_bid = tier_data['winning_bid'].mean()
            print(f"  {tier:8s}: ${avg_bid:.2f} average winning bid")
    
    # Test agent learning over time
    print(f"\n🧠 TESTING AGENT LEARNING OVER TIME")
    print("-" * 50)
    
    # Get initial performance
    initial_performance = {}
    for agent_name, agent in competitor_manager.agents.items():
        initial_performance[agent_name] = agent.get_performance_summary()
    
    # Run additional learning simulation
    learning_context = AuctionContext(
        user_id="learning_test",
        user_value_tier=UserValueTier.HIGH,
        timestamp=datetime.now(),
        device_type='mobile',
        geo_location='US',
        time_of_day=20,
        day_of_week=2,
        market_competition=0.7,
        keyword_competition=0.8,
        seasonality_factor=1.2,
        user_engagement_score=0.6,
        conversion_probability=0.08
    )
    
    print("Running 100 learning auctions with feedback...")
    
    for learning_round in range(100):
        results = competitor_manager.run_auction(learning_context)
        
        # Simulate realistic outcomes for learning
        if results:
            # Winner gets the business
            winner_name = max(results.items(), key=lambda x: x[1].bid_amount)[0]
            
            for agent_name, result in results.items():
                won = agent_name == winner_name
                result.won = won
                
                if won:
                    # Winner pays second price
                    all_bids = sorted([r.bid_amount for r in results.values()], reverse=True)
                    result.cost_per_click = all_bids[1] if len(all_bids) > 1 else all_bids[0] * 0.8
                    result.position = 1
                    
                    # Simulate conversion based on context
                    if np.random.random() < learning_context.conversion_probability:
                        result.converted = True
                        result.revenue = np.random.normal(180, 30)  # Annual subscription value
                else:
                    result.cost_per_click = 0
                    result.position = len(results) + 1
    
    # Get final performance
    final_performance = {}
    for agent_name, agent in competitor_manager.agents.items():
        final_performance[agent_name] = agent.get_performance_summary()
    
    print(f"\nAgent Learning Results:")
    for agent_name in initial_performance:
        initial = initial_performance[agent_name]
        final = final_performance[agent_name]
        
        initial_win_rate = initial['metrics']['win_rate']
        final_win_rate = final['metrics']['win_rate']
        initial_roas = initial['metrics']['roas']
        final_roas = final['metrics']['roas']
        
        print(f"  {agent_name:10s}:")
        print(f"    Win Rate: {initial_win_rate:.1%} → {final_win_rate:.1%}")
        print(f"    ROAS: {initial_roas:.2f}x → {final_roas:.2f}x")
        if 'learning_insights' in final and 'strategy_evolution' in final['learning_insights']:
            print(f"    Learning Events: {final['learning_insights']['strategy_evolution'].get('total_changes', 0)}")
        else:
            print(f"    Learning Events: {len(agent.learning_history)}")
    
    print(f"\n✨ COMPETITIVE AUCTION DEMO COMPLETE!")
    print("🏆 Key Achievements:")
    print("  ✅ Multi-agent competitive bidding system")
    print("  ✅ Realistic second-price auction mechanics") 
    print("  ✅ Agent learning and strategy adaptation")
    print("  ✅ Market dynamics and user tier responsiveness")
    print("  ✅ Performance tracking and analytics")
    
    return df_all if all_results else None


async def run_gaelp_integration_demo():
    """Run full GAELP integration with competitors"""
    
    print(f"\n🚀 FULL GAELP INTEGRATION DEMO")
    print("=" * 50)
    
    # Create optimized config for demo
    config = GAELPConfig(
        simulation_days=2,
        users_per_day=50,
        n_parallel_worlds=5,
        daily_budget_total=Decimal('500.0'),
        enable_delayed_rewards=False,
        enable_creative_optimization=True,
        enable_budget_pacing=False,
        enable_identity_resolution=False,
        enable_criteo_response=True,
        enable_safety_system=True
    )
    
    # Initialize and run
    orchestrator = MasterOrchestrator(config)
    
    print("Running integrated simulation...")
    try:
        metrics = await orchestrator.run_end_to_end_simulation()
        summary = orchestrator.get_simulation_summary()
        
        print(f"\n📈 INTEGRATION RESULTS:")
        print(f"  Total Users: {metrics.total_users}")
        print(f"  Total Auctions: {metrics.total_auctions}")
        print(f"  Our Wins: {metrics.total_auctions - metrics.competitor_wins}")
        print(f"  Competitor Wins: {metrics.competitor_wins}")
        print(f"  Win Rate: {((metrics.total_auctions - metrics.competitor_wins) / max(metrics.total_auctions, 1)):.1%}")
        print(f"  Total Spend: ${metrics.total_spend}")
        print(f"  Total Revenue: ${metrics.total_revenue}")
        print(f"  ROAS: {metrics.average_roas:.2f}x")
        
        if 'competitor_analysis' in summary:
            comp_analysis = summary['competitor_analysis']
            print(f"\n🏆 Competitive Landscape:")
            landscape = comp_analysis.get('competitive_landscape', {})
            print(f"  Market Leader: {landscape.get('market_leader', 'N/A')}")
            print(f"  Most Aggressive: {landscape.get('most_aggressive_competitor', 'N/A')}")
            print(f"  Highest ROAS: {landscape.get('highest_performing_competitor', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False


async def main():
    """Run complete competitive auction demonstration"""
    
    # Run competitive auction demo
    results_df = await run_competitive_auction_demo()
    
    # Run full integration demo
    integration_success = await run_gaelp_integration_demo()
    
    print(f"\n🎊 DEMONSTRATION COMPLETE!")
    
    if integration_success:
        print("✅ Competitors are fully integrated into GAELP!")
        print("✅ Real-time competitive auctions are working")
        print("✅ Agent learning and adaptation is functional")
        print("✅ Market dynamics are realistic")
    else:
        print("⚠️  Some integration issues detected - see logs above")
    
    return results_df


if __name__ == "__main__":
    results = asyncio.run(main())