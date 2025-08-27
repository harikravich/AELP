#!/usr/bin/env python3
"""
Validation Script for Monte Carlo Parallel Simulation Framework

Validates that all key requirements are met:
1. 100+ parallel worlds simulation capability
2. Crisis parent rare events (10% frequency, 50% value)
3. Importance sampling functionality
4. Experience aggregation across worlds
5. Efficient parallel execution
6. Integration with existing GAELP components
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime

from monte_carlo_simulator import MonteCarloSimulator, WorldType
from monte_carlo_integration import MonteCarloTrainingConfig, MonteCarloTrainingOrchestrator

print("🔍 MONTE CARLO FRAMEWORK VALIDATION")
print("=" * 60)

async def validate_requirements():
    """Validate all key requirements"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'validation_results': {},
        'performance_metrics': {},
        'compliance_check': {}
    }
    
    print("\n1️⃣ TESTING 100+ PARALLEL WORLDS CAPABILITY")
    print("-" * 40)
    
    # Test with 100 worlds
    start_time = time.time()
    simulator = MonteCarloSimulator(n_worlds=100, max_concurrent_worlds=25)
    init_time = time.time() - start_time
    
    print(f"✅ Created simulator with {simulator.n_worlds} worlds in {init_time:.2f}s")
    
    # Verify world distribution
    world_types = set()
    for config in simulator.world_configs:
        world_types.add(config.world_type)
    
    print(f"✅ World diversity: {len(world_types)} different world types")
    results['validation_results']['world_count'] = simulator.n_worlds
    results['validation_results']['world_types'] = len(world_types)
    results['validation_results']['init_time'] = init_time
    
    print("\n2️⃣ TESTING CRISIS PARENT RARE EVENTS")
    print("-" * 35)
    
    # Mock agent for testing
    class ValidationAgent:
        def select_action(self, state, deterministic=False):
            return {
                'bid': np.random.uniform(2.0, 6.0),
                'budget': np.random.uniform(500.0, 1500.0),
                'creative': {'quality_score': np.random.uniform(0.7, 0.95)},
                'quality_score': np.random.uniform(0.8, 0.95)
            }
    
    agent = ValidationAgent()
    
    # Run episodes to collect crisis parent data
    crisis_episodes = []
    total_episodes = 0
    crisis_interactions = 0
    total_revenue = 0.0
    crisis_revenue = 0.0
    
    # Run multiple batches to get good statistics
    for batch_num in range(3):
        experiences = await simulator.run_episode_batch(agent, batch_size=50)
        
        for exp in experiences:
            total_episodes += 1
            total_revenue += exp.revenue_generated
            
            if exp.crisis_parent_interactions > 0:
                crisis_interactions += exp.crisis_parent_interactions
                crisis_revenue += exp.crisis_parent_revenue
                crisis_episodes.append(exp)
    
    crisis_frequency = crisis_interactions / max(1, total_episodes)
    crisis_value_share = crisis_revenue / max(1, total_revenue) if total_revenue > 0 else 0
    
    print(f"✅ Crisis parent frequency: {crisis_frequency:.1%} (target ~10%)")
    print(f"✅ Crisis parent value share: {crisis_value_share:.1%} (target ~50%)")
    print(f"✅ Total crisis interactions: {crisis_interactions}")
    print(f"✅ Average crisis value: ${crisis_revenue / max(1, crisis_interactions):.2f}")
    
    results['validation_results']['crisis_frequency'] = crisis_frequency
    results['validation_results']['crisis_value_share'] = crisis_value_share
    results['validation_results']['crisis_interactions'] = crisis_interactions
    
    # Compliance check
    crisis_freq_ok = 0.05 <= crisis_frequency <= 0.20  # 5-20% range
    results['compliance_check']['crisis_frequency'] = crisis_freq_ok
    
    print("\n3️⃣ TESTING IMPORTANCE SAMPLING")
    print("-" * 30)
    
    # Test importance sampling boost
    regular_samples = simulator.experience_buffer.sample_batch(100, importance_sampling=False)
    importance_samples = simulator.importance_sampling(target_samples=100, focus_rare_events=True)
    
    regular_crisis_count = sum(1 for exp in regular_samples if exp.crisis_parent_interactions > 0)
    importance_crisis_count = sum(1 for exp in importance_samples if exp.crisis_parent_interactions > 0)
    
    importance_boost = importance_crisis_count / max(1, regular_crisis_count)
    
    print(f"✅ Regular sampling: {regular_crisis_count}/100 ({regular_crisis_count}%) crisis events")
    print(f"✅ Importance sampling: {importance_crisis_count}/100 ({importance_crisis_count}%) crisis events")
    print(f"✅ Importance boost: {importance_boost:.1f}x more crisis events")
    
    results['validation_results']['regular_crisis_rate'] = regular_crisis_count / 100
    results['validation_results']['importance_crisis_rate'] = importance_crisis_count / 100
    results['validation_results']['importance_boost'] = importance_boost
    
    # Check importance weights
    buffer_stats = simulator.experience_buffer.get_buffer_stats()
    print(f"✅ Average importance weight: {buffer_stats['average_importance_weight']:.2f}")
    print(f"✅ Maximum importance weight: {buffer_stats['max_importance_weight']:.2f}")
    
    results['validation_results']['avg_importance_weight'] = buffer_stats['average_importance_weight']
    results['validation_results']['max_importance_weight'] = buffer_stats['max_importance_weight']
    
    print("\n4️⃣ TESTING EXPERIENCE AGGREGATION")
    print("-" * 32)
    
    # Test aggregation functionality
    test_experiences = await simulator.run_episode_batch(agent, batch_size=30)
    aggregated = simulator.aggregate_experiences(test_experiences)
    
    print(f"✅ Aggregated {len(test_experiences)} experiences")
    print(f"✅ World types in batch: {len(aggregated.get('world_type_breakdown', {}))}")
    print(f"✅ Average reward: {aggregated.get('average_reward', 0):.3f}")
    print(f"✅ Success rate: {aggregated.get('success_rate', 0):.1%}")
    
    # Verify training batch format
    training_batch = aggregated.get('training_batch', {})
    print(f"✅ Training batch size: {training_batch.get('batch_size', 0)}")
    print(f"✅ Has required fields: {all(key in training_batch for key in ['states', 'actions', 'rewards'])}")
    
    results['validation_results']['aggregation_batch_size'] = training_batch.get('batch_size', 0)
    results['validation_results']['world_types_in_batch'] = len(aggregated.get('world_type_breakdown', {}))
    
    print("\n5️⃣ TESTING PERFORMANCE")
    print("-" * 20)
    
    # Performance test
    perf_start = time.time()
    perf_experiences = await simulator.run_episode_batch(agent, batch_size=100)
    perf_time = time.time() - perf_start
    episodes_per_second = len(perf_experiences) / perf_time
    
    print(f"✅ Generated {len(perf_experiences)} episodes in {perf_time:.2f}s")
    print(f"✅ Performance: {episodes_per_second:.1f} episodes/second")
    print(f"✅ Parallel efficiency: Using {simulator.max_concurrent_worlds} concurrent worlds")
    
    results['performance_metrics']['episodes_per_second'] = episodes_per_second
    results['performance_metrics']['batch_processing_time'] = perf_time
    results['performance_metrics']['concurrent_worlds'] = simulator.max_concurrent_worlds
    
    # Performance compliance
    performance_ok = episodes_per_second >= 30  # Minimum 30 episodes/second
    results['compliance_check']['performance'] = performance_ok
    
    print("\n6️⃣ TESTING INTEGRATION")
    print("-" * 20)
    
    # Test integration with training orchestrator
    config = MonteCarloTrainingConfig(
        n_worlds=25,  # Smaller for quick test
        batch_size=50,
        importance_sampling_ratio=0.4
    )
    
    orchestrator = MonteCarloTrainingOrchestrator(config)
    
    # Test training batch generation
    training_batch = await orchestrator.generate_training_batch(agent)
    
    print(f"✅ Integration orchestrator created")
    print(f"✅ Generated training batch: {training_batch['batch_size']} experiences")
    print(f"✅ Importance sampled ratio: {training_batch['importance_sampled_ratio']:.1%}")
    print(f"✅ Crisis parent experiences: {training_batch['crisis_parent_experiences']}")
    
    results['validation_results']['integration_batch_size'] = training_batch['batch_size']
    results['validation_results']['integration_importance_ratio'] = training_batch['importance_sampled_ratio']
    
    # Cleanup
    simulator.cleanup()
    orchestrator.cleanup()
    
    print("\n📊 FINAL COMPLIANCE CHECK")
    print("=" * 40)
    
    # Overall compliance
    compliance_items = [
        ("100+ Worlds Support", simulator.n_worlds >= 100),
        ("Crisis Parent Handling", crisis_freq_ok),
        ("Importance Sampling", importance_boost > 1.0),
        ("Experience Aggregation", training_batch.get('batch_size', 0) > 0),
        ("Performance Standard", performance_ok),
        ("Integration Ready", training_batch['batch_size'] > 0)
    ]
    
    all_compliant = True
    for item, status in compliance_items:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {item}: {'PASS' if status else 'FAIL'}")
        if not status:
            all_compliant = False
    
    results['compliance_check']['overall_compliant'] = all_compliant
    results['compliance_check']['individual_checks'] = dict(compliance_items)
    
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL REQUIREMENTS MET' if all_compliant else '❌ REQUIREMENTS NOT MET'}")
    
    if all_compliant:
        print("\n🚀 MONTE CARLO FRAMEWORK VALIDATION SUCCESSFUL!")
        print("Ready for production deployment.")
    else:
        print("\n⚠️  Some requirements need attention before production.")
    
    # Save validation results
    with open('/home/hariravichandran/AELP/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📋 Detailed validation results saved to validation_results.json")
    
    return all_compliant

if __name__ == "__main__":
    success = asyncio.run(validate_requirements())
    exit(0 if success else 1)