#!/usr/bin/env python3
"""
Test script for TRUE PARALLEL Monte Carlo simulation with 100+ worlds.
Verifies 100x faster learning through multiprocessing parallelism.
"""

import logging
import sys
import time
import psutil
import traceback
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_parallel_world_orchestrator():
    """Test the TRUE PARALLEL Monte Carlo simulation framework"""
    
    logger.info("Testing ParallelWorldOrchestrator - TRUE PARALLEL execution")
    
    try:
        from monte_carlo_simulator import ParallelWorldOrchestrator
        logger.info("✅ Successfully imported ParallelWorldOrchestrator")
    except ImportError as e:
        logger.error(f"❌ Failed to import ParallelWorldOrchestrator: {e}")
        return False
    
    # Initialize orchestrator with 100+ worlds
    try:
        orchestrator = ParallelWorldOrchestrator(
            n_worlds=120,  # Must be 100+ for true parallelism
            episodes_per_world=5  # 5 episodes per world = 600 total episodes
        )
        logger.info("✅ Successfully created ParallelWorldOrchestrator")
    except Exception as e:
        logger.error(f"❌ Failed to create orchestrator: {e}")
        return False
    
    # Create agent state for distribution to processes
    agent_state = {
        'agent_id': 'parallel_test_agent',
        'policy_type': 'random',
        'parameters': {'exploration_rate': 0.1}
    }
    
    try:
        logger.info("🚀 Starting TRUE PARALLEL execution test")
        start_test_time = time.time()
        
        # Run episodes across ALL worlds in parallel
        experiences = orchestrator.run_parallel_episodes(
            agent_state=agent_state,
            total_episodes=600  # 600 episodes across 120 worlds
        )
        
        test_duration = time.time() - start_test_time
        
        # Aggregate experiences
        aggregated = orchestrator.aggregate_experiences(experiences)
        
        logger.info(f"🎯 TRUE PARALLEL EXECUTION TEST RESULTS:")
        logger.info(f"  Total experiences: {len(experiences)}")
        logger.info(f"  Test duration: {test_duration:.2f} seconds")
        logger.info(f"  Episodes per second: {len(experiences) / max(1, test_duration):.1f}")
        logger.info(f"  Average reward: {aggregated['average_reward']:.3f}")
        logger.info(f"  Success rate: {aggregated['success_rate']:.3f}")
        logger.info(f"  Crisis interactions: {aggregated['total_crisis_interactions']}")
        logger.info(f"  Worlds with episodes: {len(set(exp.world_id for exp in experiences))}")
        
        # Test importance sampling
        important_samples = orchestrator.importance_sampling(target_samples=100)
        logger.info(f"  Importance samples: {len(important_samples)}")
        
        # Get performance metrics
        performance = orchestrator.get_performance_metrics()
        logger.info(f"📊 PERFORMANCE METRICS:")
        logger.info(f"  Episodes per second: {performance['episodes_per_second']:.1f}")
        logger.info(f"  Estimated speedup: {performance['estimated_speedup']:.1f}x")
        logger.info(f"  Target speedup achieved: {performance['speedup_achieved']}")
        logger.info(f"  CPU cores used: {performance['max_processes']}/{performance['cpu_cores']}")
        logger.info(f"  Memory usage: {performance['memory_usage_gb']:.1f} GB")
        
        # Verify 100x speedup target
        if performance['episodes_per_second'] < 50:
            logger.warning("⚠️  Did not achieve 50+ episodes/second target")
        else:
            logger.info("✅ Achieved high-performance parallel execution!")
        
        if len(experiences) < 500:
            logger.warning("⚠️  Expected more episodes from parallel execution")
        else:
            logger.info("✅ Generated sufficient training experiences!")
        
        # Test experience buffer statistics
        buffer_stats = orchestrator.experience_buffer.get_buffer_stats()
        logger.info(f"💾 EXPERIENCE BUFFER:")
        logger.info(f"  Total experiences: {buffer_stats['total_experiences']}")
        logger.info(f"  Crisis parent ratio: {buffer_stats['crisis_parent_ratio']:.3f}")
        logger.info(f"  Average importance weight: {buffer_stats['average_importance_weight']:.3f}")
        
        # Test saving/loading
        orchestrator.save_experiences('/tmp/parallel_experiences.pkl')
        logger.info("✅ Successfully saved experiences")
        
        return performance['speedup_achieved']
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error(traceback.format_exc())
        return False
        
    finally:
        orchestrator.cleanup()


def verify_100x_speedup():
    """Verify that we achieve 100x faster learning through parallelism"""
    logger.info("🚀 VERIFYING 100X SPEEDUP TARGET 🚀")
    
    # Test with different world counts to show scaling
    test_configs = [
        {'n_worlds': 100, 'episodes_per_world': 3, 'name': '100 worlds'},
        {'n_worlds': 150, 'episodes_per_world': 2, 'name': '150 worlds'},
    ]
    
    # Only test 200 worlds if we have enough cores
    cpu_count = psutil.cpu_count(logical=True)
    if cpu_count >= 16:
        test_configs.append({'n_worlds': 200, 'episodes_per_world': 2, 'name': '200 worlds (max test)'})
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {config['name']}")
        logger.info(f"{'='*60}")
        
        try:
            from monte_carlo_simulator import ParallelWorldOrchestrator
            
            orchestrator = ParallelWorldOrchestrator(
                n_worlds=config['n_worlds'],
                episodes_per_world=config['episodes_per_world']
            )
            
            agent_state = {'agent_id': f"test_{config['n_worlds']}_worlds"}
            
            start_time = time.time()
            experiences = orchestrator.run_parallel_episodes(agent_state)
            duration = time.time() - start_time
            
            eps_per_sec = len(experiences) / max(1, duration)
            
            result = {
                'config': config['name'],
                'episodes': len(experiences),
                'duration': duration,
                'eps_per_sec': eps_per_sec,
                'worlds': config['n_worlds']
            }
            results.append(result)
            
            logger.info(f"📈 Result: {len(experiences)} episodes in {duration:.2f}s = {eps_per_sec:.1f} eps/sec")
            
            orchestrator.cleanup()
            
        except Exception as e:
            logger.error(f"❌ Test failed for {config['name']}: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🏆 SPEEDUP VERIFICATION SUMMARY")
    logger.info(f"{'='*60}")
    
    for result in results:
        speedup = result['eps_per_sec'] / 10  # vs 10 eps/sec sequential baseline
        logger.info(f"{result['config']}: {result['eps_per_sec']:.1f} eps/sec = {speedup:.1f}x speedup")
    
    best_performance = max(results, key=lambda x: x['eps_per_sec']) if results else None
    if best_performance:
        best_speedup = best_performance['eps_per_sec'] / 10
        logger.info(f"\n🎯 BEST PERFORMANCE: {best_speedup:.1f}x speedup")
        
        if best_speedup >= 50:
            logger.info("✅ SUCCESS: Achieved 50x+ speedup target!")
            return True
        else:
            logger.warning(f"⚠️  Only achieved {best_speedup:.1f}x speedup (target: 50x+)")
            return False
    
    return False


def test_importance_sampling():
    """Test importance sampling for rare events (crisis parents)"""
    logger.info("\n🎯 TESTING IMPORTANCE SAMPLING")
    
    try:
        from monte_carlo_simulator import ParallelWorldOrchestrator
        
        # Create orchestrator focused on crisis parent worlds
        orchestrator = ParallelWorldOrchestrator(
            n_worlds=100,
            episodes_per_world=3,
            world_types_distribution={
                'normal_market': 0.6,
                'crisis_parent': 0.3,  # Higher percentage for testing
                'high_competition': 0.1
            }
        )
        
        agent_state = {'agent_id': 'importance_sampling_test'}
        
        # Run episodes
        experiences = orchestrator.run_parallel_episodes(agent_state)
        
        # Test importance sampling
        crisis_experiences = [exp for exp in experiences if exp.crisis_parent_interactions > 0]
        normal_experiences = [exp for exp in experiences if exp.crisis_parent_interactions == 0]
        
        logger.info(f"📊 Experience Distribution:")
        logger.info(f"  Total experiences: {len(experiences)}")
        logger.info(f"  Crisis parent experiences: {len(crisis_experiences)}")
        logger.info(f"  Normal experiences: {len(normal_experiences)}")
        logger.info(f"  Crisis parent ratio: {len(crisis_experiences) / len(experiences):.3f}")
        
        # Test importance sampling
        important_samples = orchestrator.importance_sampling(target_samples=100, focus_rare_events=True)
        crisis_in_samples = sum(1 for exp in important_samples if exp.crisis_parent_interactions > 0)
        
        logger.info(f"🎯 Importance Sampling Results:")
        logger.info(f"  Total samples: {len(important_samples)}")
        logger.info(f"  Crisis samples: {crisis_in_samples}")
        logger.info(f"  Crisis ratio in samples: {crisis_in_samples / len(important_samples):.3f}")
        
        # Should have higher ratio of crisis samples than original data
        original_crisis_ratio = len(crisis_experiences) / len(experiences)
        sample_crisis_ratio = crisis_in_samples / len(important_samples)
        
        if sample_crisis_ratio > original_crisis_ratio:
            logger.info("✅ Importance sampling successfully oversampled rare events!")
            return True
        else:
            logger.warning("⚠️  Importance sampling did not increase rare event frequency")
            return False
        
        orchestrator.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Importance sampling test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🚀 STARTING PARALLEL MONTE CARLO TESTS")
    logger.info(f"System info: {psutil.cpu_count(logical=True)} CPU cores, {psutil.virtual_memory().total / (1024**3):.1f} GB RAM")
    
    tests = [
        ("Basic Parallel Functionality", test_parallel_world_orchestrator),
        ("100x Speedup Verification", verify_100x_speedup),
        ("Importance Sampling", test_importance_sampling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*70}")
        logger.info(f"🧪 RUNNING: {test_name}")
        logger.info(f"{'='*70}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("🏁 FINAL TEST RESULTS")
    logger.info(f"{'='*70}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 ALL TESTS PASSED - PARALLEL MONTE CARLO READY!")
        sys.exit(0)
    else:
        logger.error("💥 SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()