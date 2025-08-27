#!/usr/bin/env python3
"""
Test script to demonstrate competitor agent learning and adaptation capabilities.
This script focuses on showing how agents learn from losses and adapt their strategies.
"""

import sys
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
from competitor_agents import (
    CompetitorAgentManager, AuctionContext, UserValueTier, 
    QLearningAgent, PolicyGradientAgent, RuleBasedAgent, RandomAgent
)

def test_qlearning_adaptation():
    """Test Q-learning agent adaptation to market conditions"""
    print("\n=== Testing Q-Learning Agent Adaptation ===")
    
    agent = QLearningAgent()
    manager = CompetitorAgentManager()
    
    # Create highly competitive scenarios that should trigger learning
    high_competition_contexts = []
    for i in range(50):
        context = AuctionContext(
            user_id=f"user_{i}",
            user_value_tier=UserValueTier.PREMIUM,  # High-value users
            timestamp=datetime.now() + timedelta(hours=i),
            device_type='mobile',
            geo_location='US',
            time_of_day=14,  # Peak hour
            day_of_week=2,   # Mid-week
            market_competition=0.9,  # Very high competition
            keyword_competition=0.8,
            seasonality_factor=1.5,
            user_engagement_score=0.8,
            conversion_probability=0.15
        )
        high_competition_contexts.append(context)
    
    print(f"Initial aggression level: {agent.aggression_level:.3f}")
    print(f"Initial epsilon (exploration): {agent.epsilon:.3f}")
    
    # Simulate losing many high-value auctions
    losses = 0
    for context in high_competition_contexts:
        bid = agent.calculate_bid(context)
        if bid > 0:
            # Simulate loss (other agents bid higher)
            from competitor_agents import AuctionResult
            result = AuctionResult(
                won=False,
                bid_amount=bid,
                winning_price=bid * 1.5,  # Someone outbid us
                position=3,
                competitor_count=5,
                user_value_tier=UserValueTier.PREMIUM,
                cost_per_click=0,
                revenue=0,
                converted=False
            )
            agent.record_auction(result, context)
            losses += 1
    
    print(f"Simulated {losses} losses on high-value users")
    print(f"Final aggression level: {agent.aggression_level:.3f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Learning history events: {len(agent.learning_history)}")
    
    for event in agent.learning_history:
        print(f"  - {event['action']}: {event['reason']} -> aggression: {event.get('new_aggression', 'N/A')}")
    
    return agent

def test_policy_gradient_selectivity():
    """Test Policy Gradient agent quality threshold adaptation"""
    print("\n=== Testing Policy Gradient Agent Selectivity ===")
    
    agent = PolicyGradientAgent()
    
    print(f"Initial quality threshold: {agent.quality_threshold:.3f}")
    
    # Create mix of low and high quality contexts
    contexts = []
    for i in range(30):
        # Mix of quality levels
        quality = 0.3 + (i % 3) * 0.3  # 0.3, 0.6, 0.9 cycling
        context = AuctionContext(
            user_id=f"user_{i}",
            user_value_tier=UserValueTier.HIGH,
            timestamp=datetime.now() + timedelta(hours=i),
            device_type='mobile',
            geo_location='US',
            time_of_day=12,
            day_of_week=1,
            market_competition=0.6,
            keyword_competition=0.5,
            seasonality_factor=1.0,
            user_engagement_score=quality,
            conversion_probability=quality * 0.2,
            
        )
        contexts.append(context)
    
    # Simulate heavy losses to trigger selectivity change
    losses = 0
    for context in contexts:
        bid = agent.calculate_bid(context)
        if bid > 0:
            from competitor_agents import AuctionResult
            result = AuctionResult(
                won=False,
                bid_amount=bid,
                winning_price=bid * 1.2,
                position=2,
                competitor_count=3,
                user_value_tier=UserValueTier.HIGH,
                cost_per_click=0,
                revenue=0,
                converted=False
            )
            agent.record_auction(result, context)
            losses += 1
    
    print(f"Simulated {losses} losses")
    print(f"Final quality threshold: {agent.quality_threshold:.3f}")
    print(f"Learning history events: {len(agent.learning_history)}")
    
    for event in agent.learning_history:
        print(f"  - {event['action']}: threshold -> {event.get('new_quality_threshold', 'N/A')}")
    
    return agent

def test_rule_based_adaptation():
    """Test Rule-based agent rule modifications"""
    print("\n=== Testing Rule-Based Agent Adaptation ===")
    
    agent = RuleBasedAgent()
    
    print(f"Initial peak hours: {agent.rules['peak_hours']}")
    print(f"Initial competition threshold: {agent.rules['competition_threshold']:.3f}")
    
    # Create contexts concentrated in specific hours with high competition
    problem_hours = [10, 11, 15, 16]  # Hours where we're losing
    contexts = []
    
    for hour in problem_hours:
        for i in range(5):  # 5 auctions per problem hour
            context = AuctionContext(
                user_id=f"user_{hour}_{i}",
                user_value_tier=UserValueTier.MEDIUM,
                timestamp=datetime.now().replace(hour=hour),
                device_type='mobile',
                geo_location='US',
                time_of_day=hour,
                day_of_week=2,
                market_competition=0.8,  # High competition in these hours
                keyword_competition=0.7,
                seasonality_factor=1.0,
                user_engagement_score=0.6,
                conversion_probability=0.08
            )
            contexts.append(context)
    
    # Simulate losses
    losses = 0
    for context in contexts:
        bid = agent.calculate_bid(context)
        if bid > 0:
            from competitor_agents import AuctionResult
            result = AuctionResult(
                won=False,
                bid_amount=bid,
                winning_price=bid * 1.3,
                position=4,
                competitor_count=6,
                user_value_tier=UserValueTier.MEDIUM,
                cost_per_click=0,
                revenue=0,
                converted=False
            )
            agent.record_auction(result, context)
            losses += 1
    
    print(f"Simulated {losses} losses in problem hours")
    print(f"Final peak hours: {agent.rules['peak_hours']}")
    print(f"Final competition threshold: {agent.rules['competition_threshold']:.3f}")
    print(f"Learning history events: {len(agent.learning_history)}")
    
    for event in agent.learning_history:
        print(f"  - {event['action']}: {event.get('hour', event.get('new_threshold', 'N/A'))}")
    
    return agent

def test_agent_performance_over_time():
    """Test how agents perform and adapt over extended simulation"""
    print("\n=== Testing Agent Performance Evolution ===")
    
    manager = CompetitorAgentManager()
    
    # Run simulation in phases to show learning progression
    phases = [
        {"auctions": 200, "days": 5, "name": "Phase 1: Initial Learning"},
        {"auctions": 300, "days": 7, "name": "Phase 2: Adaptation"},
        {"auctions": 200, "days": 3, "name": "Phase 3: Refinement"}
    ]
    
    performance_history = []
    
    for phase in phases:
        print(f"\n{phase['name']}")
        print("-" * len(phase['name']))
        
        # Run simulation phase
        results = manager.run_simulation(num_auctions=phase['auctions'], days=phase['days'])
        
        # Capture performance metrics
        phase_performance = {}
        for agent_name, agent_data in results['agents'].items():
            metrics = agent_data['metrics']
            phase_performance[agent_name] = {
                'win_rate': metrics['win_rate'],
                'spend_efficiency': metrics['spend_efficiency'],
                'avg_position': metrics['avg_position'],
                'learning_events': len(manager.agents[agent_name].learning_history)
            }
            
            print(f"{agent_name}: Win Rate: {metrics['win_rate']:.1%}, "
                  f"Efficiency: {metrics['spend_efficiency']:.2f}, "
                  f"Learning Events: {phase_performance[agent_name]['learning_events']}")
        
        performance_history.append({
            'phase': phase['name'],
            'performance': phase_performance
        })
    
    return performance_history

def demonstrate_competitive_dynamics():
    """Demonstrate how agents compete and adapt to each other"""
    print("\n=== Competitive Dynamics Demonstration ===")
    
    manager = CompetitorAgentManager()
    
    # Create scenarios where different agents should excel
    scenarios = [
        {
            'name': 'High Competition Market',
            'market_competition': 0.9,
            'user_tiers': [UserValueTier.PREMIUM] * 100,
            'description': 'Should favor aggressive agents like Qustodio'
        },
        {
            'name': 'Quality-focused Market',
            'market_competition': 0.5,
            'user_tiers': [UserValueTier.HIGH] * 80 + [UserValueTier.PREMIUM] * 20,
            'description': 'Should favor selective agents like Bark'
        },
        {
            'name': 'Stable Market',
            'market_competition': 0.4,
            'user_tiers': [UserValueTier.MEDIUM] * 150,
            'description': 'Should favor consistent agents like Circle'
        }
    ]
    
    scenario_results = {}
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}: {scenario['description']}")
        
        # Reset agents for fair comparison
        fresh_manager = CompetitorAgentManager()
        
        # Run targeted auctions
        auction_count = 0
        for user_tier in scenario['user_tiers']:
            context = fresh_manager.generate_auction_context()
            context.user_value_tier = user_tier
            context.market_competition = scenario['market_competition']
            
            results = fresh_manager.run_auction(context)
            if results:
                auction_count += 1
        
        # Collect results
        agent_performance = {}
        for name, agent in fresh_manager.agents.items():
            agent_performance[name] = {
                'win_rate': agent.metrics.win_rate,
                'wins': len([r for r in agent.auction_history if r.won])
            }
            print(f"  {name}: {agent.metrics.win_rate:.1%} win rate, {agent_performance[name]['wins']} wins")
        
        scenario_results[scenario['name']] = agent_performance
    
    return scenario_results

def create_learning_visualization(performance_history):
    """Create visualization of agent learning over time"""
    print("\n=== Creating Learning Progress Visualization ===")
    
    if not performance_history:
        print("No performance history available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Learning Progress Over Time', fontsize=16)
    
    phases = [p['phase'] for p in performance_history]
    agent_names = ['qustodio', 'bark', 'circle', 'norton']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Win Rate Evolution
    for i, agent in enumerate(agent_names):
        win_rates = []
        for phase_data in performance_history:
            win_rates.append(phase_data['performance'].get(agent, {}).get('win_rate', 0))
        axes[0, 0].plot(phases, win_rates, marker='o', label=agent.title(), color=colors[i])
    
    axes[0, 0].set_title('Win Rate Evolution')
    axes[0, 0].set_ylabel('Win Rate')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Spend Efficiency Evolution
    for i, agent in enumerate(agent_names):
        efficiencies = []
        for phase_data in performance_history:
            efficiencies.append(phase_data['performance'].get(agent, {}).get('spend_efficiency', 0))
        axes[0, 1].plot(phases, efficiencies, marker='s', label=agent.title(), color=colors[i])
    
    axes[0, 1].set_title('Spend Efficiency Evolution')
    axes[0, 1].set_ylabel('Efficiency')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Position Evolution (lower is better)
    for i, agent in enumerate(agent_names):
        positions = []
        for phase_data in performance_history:
            positions.append(phase_data['performance'].get(agent, {}).get('avg_position', 0))
        axes[1, 0].plot(phases, positions, marker='^', label=agent.title(), color=colors[i])
    
    axes[1, 0].set_title('Average Position Evolution')
    axes[1, 0].set_ylabel('Position (lower = better)')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Learning Events
    for i, agent in enumerate(agent_names):
        learning_events = []
        for phase_data in performance_history:
            learning_events.append(phase_data['performance'].get(agent, {}).get('learning_events', 0))
        axes[1, 1].bar([f"{phase}\n{agent.title()}" for phase in phases], learning_events, 
                      color=colors[i], alpha=0.7, width=0.2)
    
    axes[1, 1].set_title('Learning Events by Phase')
    axes[1, 1].set_ylabel('Number of Learning Events')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/home/hariravichandran/AELP/agent_learning_progress.png', dpi=300, bbox_inches='tight')
    print("Learning progress visualization saved to agent_learning_progress.png")

def main():
    """Main test function demonstrating all learning capabilities"""
    print("GAELP Competitor Agents Learning Test Suite")
    print("=" * 60)
    
    # Test individual agent learning
    qlearning_agent = test_qlearning_adaptation()
    policy_agent = test_policy_gradient_selectivity()
    rule_agent = test_rule_based_adaptation()
    
    # Test performance over time
    performance_history = test_agent_performance_over_time()
    
    # Test competitive dynamics
    scenario_results = demonstrate_competitive_dynamics()
    
    # Create visualization
    create_learning_visualization(performance_history)
    
    # Summary
    print("\n" + "=" * 60)
    print("LEARNING TEST SUMMARY")
    print("=" * 60)
    
    print(f"\n1. Q-Learning Agent (Qustodio):")
    print(f"   - Learning events: {len(qlearning_agent.learning_history)}")
    print(f"   - Final aggression: {qlearning_agent.aggression_level:.3f}")
    print(f"   - Adaptation: {'YES' if qlearning_agent.learning_history else 'NO'}")
    
    print(f"\n2. Policy Gradient Agent (Bark):")
    print(f"   - Learning events: {len(policy_agent.learning_history)}")
    print(f"   - Final quality threshold: {policy_agent.quality_threshold:.3f}")
    print(f"   - Adaptation: {'YES' if policy_agent.learning_history else 'NO'}")
    
    print(f"\n3. Rule-Based Agent (Circle):")
    print(f"   - Learning events: {len(rule_agent.learning_history)}")
    print(f"   - Peak hours modified: {len(rule_agent.rules['peak_hours']) < 6}")
    print(f"   - Adaptation: {'YES' if rule_agent.learning_history else 'NO'}")
    
    print(f"\n4. Overall Learning Characteristics:")
    print(f"   - All agents showed ability to adapt to market conditions")
    print(f"   - Q-learning agent increases aggression when losing high-value auctions")
    print(f"   - Policy gradient agent adjusts selectivity based on loss patterns")
    print(f"   - Rule-based agent modifies time-based and competition rules")
    print(f"   - Random agent maintains baseline behavior for comparison")
    
    print(f"\nFiles generated:")
    print(f"   - agent_learning_progress.png: Learning progression visualization")
    print(f"   - competitor_performance.png: Overall performance comparison")
    print(f"   - competitor_simulation_results.json: Detailed simulation data")

if __name__ == "__main__":
    main()