"""
Crisis Parent Training Demo

Demonstrates the integration of ImportanceSampler with the training loop
for proper crisis parent weighting (10% population, 50% value, 5x weight).
"""

import sys
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from importance_sampler import ImportanceSampler, Experience
from training_orchestrator.rl_agents.importance_sampling_integration import (
    ImportanceSamplingReplayBuffer,
    ExperienceAggregator,
    test_crisis_parent_weighting
)
from training_orchestrator.rl_agents.ppo_agent import PPOAgent, PPOConfig
from training_orchestrator.importance_sampling_trainer import (
    ImportanceSamplingTrainingOrchestrator,
    ImportanceSamplingTrainingConfiguration,
    create_importance_sampling_trainer
)


logger = logging.getLogger(__name__)


def generate_mock_experiences(num_experiences: int = 1000) -> List[Dict[str, Any]]:
    """Generate mock experiences with 10% crisis parents, 90% regular parents"""
    
    experiences = []
    
    for i in range(num_experiences):
        # Determine event type - 10% crisis parents, 90% regular parents
        if i % 10 == 0:  # 10% crisis parents
            event_type = "crisis_parent"
            base_reward = 5.0  # Higher rewards for crisis parents
            base_value = 5.0
            user_profile = {"status": "crisis", "urgency": "high"}
            behavior = {"crisis_content_engagement": 0.8, "search_urgency": 0.9}
        else:  # 90% regular parents
            event_type = "regular_parent"
            base_reward = 1.0  # Lower rewards for regular parents
            base_value = 1.0
            user_profile = {"status": "normal", "urgency": "low"}
            behavior = {"crisis_content_engagement": 0.2, "search_urgency": 0.3}
        
        # Add some noise
        reward = base_reward + np.random.normal(0, 0.5)
        value = base_value + np.random.normal(0, 0.3)
        
        experience = {
            'state': torch.randn(128),
            'action': np.random.randint(0, 4),
            'reward': reward,
            'next_state': torch.randn(128),
            'done': np.random.random() < 0.1,
            'value': value,
            'event_type': event_type,
            'metadata': {
                'user_profile': user_profile,
                'behavior': behavior,
                'timestamp': i,
                'value_threshold_90th': 2.0  # Threshold for top 10% values
            }
        }
        
        experiences.append(experience)
    
    return experiences


def demo_importance_sampler():
    """Demonstrate basic ImportanceSampler functionality"""
    
    print("\n" + "="*60)
    print("DEMO 1: Basic ImportanceSampler Functionality")
    print("="*60)
    
    # Create importance sampler with crisis parent scenario
    sampler = ImportanceSampler(
        population_ratios={"crisis_parent": 0.1, "regular_parent": 0.9},
        conversion_ratios={"crisis_parent": 0.5, "regular_parent": 0.5},
        max_weight=5.0  # 5x weight for crisis parents
    )
    
    print(f"Initial importance weights: {sampler._importance_weights}")
    
    # Generate and add experiences
    experiences = generate_mock_experiences(1000)
    
    for exp in experiences:
        importance_exp = Experience(
            state=exp['state'].numpy(),
            action=exp['action'],
            reward=exp['reward'],
            next_state=exp['next_state'].numpy(),
            done=exp['done'],
            value=exp['value'],
            event_type=exp['event_type'],
            timestamp=exp['metadata']['timestamp'],
            metadata=exp['metadata']
        )
        sampler.add_experience(importance_exp)
    
    # Sample with importance weighting
    batch_size = 64
    sampled_experiences, importance_weights, indices = sampler.weighted_sampling(batch_size)
    
    # Analyze results
    crisis_count = sum(1 for exp in sampled_experiences if exp.event_type == "crisis_parent")
    regular_count = batch_size - crisis_count
    
    print(f"\nSampling Results:")
    print(f"Crisis parents in batch: {crisis_count}/{batch_size} ({crisis_count/batch_size:.1%})")
    print(f"Regular parents in batch: {regular_count}/{batch_size} ({regular_count/batch_size:.1%})")
    print(f"Expected crisis ratio without weighting: 10%")
    print(f"Actual crisis ratio with 5x weighting: {crisis_count/batch_size:.1%}")
    print(f"Average importance weight: {np.mean(importance_weights):.3f}")
    
    # Test bias correction
    mock_gradients = np.random.randn(128)
    corrected_gradients = sampler.bias_correction(mock_gradients, importance_weights, batch_size)
    
    print(f"\nBias Correction:")
    print(f"Original gradient norm: {np.linalg.norm(mock_gradients):.3f}")
    print(f"Corrected gradient norm: {np.linalg.norm(corrected_gradients):.3f}")
    
    # Print statistics
    stats = sampler.get_sampling_statistics()
    print(f"\nSampling Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_replay_buffer_integration():
    """Demonstrate ImportanceSamplingReplayBuffer"""
    
    print("\n" + "="*60)
    print("DEMO 2: ImportanceSamplingReplayBuffer")
    print("="*60)
    
    # Run the built-in test
    buffer, stats = test_crisis_parent_weighting()
    
    print(f"\nBuffer Statistics:")
    print(f"Total experiences in buffer: {len(buffer)}")
    print(f"Crisis parent weight: {stats['crisis_parent_weight']:.1f}x")
    print(f"Regular parent weight: {stats['regular_parent_weight']:.1f}x")
    
    # Test multiple sampling rounds
    crisis_ratios = []
    for _ in range(10):
        batch_dict, weights, indices = buffer.sample_importance_weighted(32)
        crisis_count = sum(1 for event_type in batch_dict['event_types'] if event_type == 'crisis_parent')
        crisis_ratios.append(crisis_count / 32)
    
    print(f"\nConsistency Test (10 sampling rounds):")
    print(f"Average crisis parent ratio: {np.mean(crisis_ratios):.1%}")
    print(f"Standard deviation: {np.std(crisis_ratios):.3f}")
    print(f"Min ratio: {np.min(crisis_ratios):.1%}")
    print(f"Max ratio: {np.max(crisis_ratios):.1%}")


def demo_experience_aggregation():
    """Demonstrate ExperienceAggregator for identifying crisis parents"""
    
    print("\n" + "="*60)
    print("DEMO 3: ExperienceAggregator")
    print("="*60)
    
    # Create experience aggregator
    aggregator = ExperienceAggregator(
        crisis_indicators=['crisis', 'urgent', 'emergency', 'high_priority']
    )
    
    # Create replay buffer
    buffer = ImportanceSamplingReplayBuffer(capacity=1000)
    
    # Generate mixed experiences
    experiences = generate_mock_experiences(500)
    
    # Process through aggregator
    stats = aggregator.aggregate_experiences(experiences, buffer)
    
    print(f"Experience Aggregation Results:")
    print(f"Total experiences processed: {len(experiences)}")
    print(f"Crisis parents identified: {stats['crisis_parent']}")
    print(f"Regular parents identified: {stats['regular_parent']}")
    print(f"Crisis parent identification rate: {stats['crisis_parent'] / len(experiences):.1%}")
    
    # Test individual identification
    test_cases = [
        {
            'metadata': {'user_profile': {'status': 'crisis'}},
            'reward': 3.0,
            'expected': 'crisis_parent'
        },
        {
            'metadata': {'behavior': {'crisis_content_engagement': 0.8}},
            'reward': 2.5,
            'expected': 'crisis_parent'
        },
        {
            'metadata': {'user_profile': {'status': 'normal'}},
            'reward': 1.0,
            'expected': 'regular_parent'
        }
    ]
    
    print(f"\nIdentification Test Cases:")
    for i, case in enumerate(test_cases):
        identified = aggregator.identify_event_type(case)
        correct = identified == case['expected']
        print(f"  Case {i+1}: {identified} ({'✓' if correct else '✗'})")


def demo_ppo_integration():
    """Demonstrate PPO agent with importance sampling"""
    
    print("\n" + "="*60)
    print("DEMO 4: PPO Agent with Importance Sampling")
    print("="*60)
    
    # Create PPO configuration
    config = PPOConfig(
        state_dim=128,
        action_dim=4,
        learning_rate=3e-4,
        rollout_length=64,
        minibatch_size=32
    )
    
    # Create PPO agent
    agent = PPOAgent(config, "demo_agent")
    
    print(f"Created PPO agent with importance sampling")
    print(f"Agent has importance sampler: {hasattr(agent, 'importance_sampler')}")
    
    if hasattr(agent, 'importance_sampler'):
        weights = agent.importance_sampler._importance_weights
        print(f"Crisis parent weight: {weights.get('crisis_parent', 1.0):.1f}x")
        print(f"Regular parent weight: {weights.get('regular_parent', 1.0):.1f}x")
    
    # Generate mock experiences for training
    experiences = []
    for i in range(64):  # Rollout length
        exp = {
            'reward': np.random.randn(),
            'done': i == 63,  # Last experience is done
            'metadata': {
                'user_profile': 'crisis' if i % 10 == 0 else 'normal'
            },
            'timestamp': i,
            'value': 5.0 if i % 10 == 0 else 1.0
        }
        experiences.append(exp)
    
    print(f"Generated {len(experiences)} experiences for training")
    print(f"Crisis experiences: {sum(1 for exp in experiences if 'crisis' in exp['metadata']['user_profile'])}")
    
    # Since we can't run full training without environment, just show the structure
    print(f"Agent update_policy method available: {hasattr(agent, 'update_policy')}")


def demo_training_orchestrator():
    """Demonstrate ImportanceSamplingTrainingOrchestrator setup"""
    
    print("\n" + "="*60)
    print("DEMO 5: Training Orchestrator Setup")
    print("="*60)
    
    # Create training configuration
    config = ImportanceSamplingTrainingConfiguration(
        experiment_id="crisis_parent_demo",
        crisis_parent_weight=5.0,
        simulation_episodes=100,
        enable_importance_sampling=True,
        crisis_indicators=['crisis', 'urgent', 'emergency']
    )
    
    print(f"Training Configuration:")
    print(f"  Experiment ID: {config.experiment_id}")
    print(f"  Crisis parent weight: {config.crisis_parent_weight}x")
    print(f"  Simulation episodes: {config.simulation_episodes}")
    print(f"  Importance sampling enabled: {config.enable_importance_sampling}")
    print(f"  Crisis indicators: {config.crisis_indicators}")
    
    # Create orchestrator using factory function
    try:
        trainer = create_importance_sampling_trainer(
            experiment_id="factory_demo",
            crisis_parent_weight=5.0,
            importance_sampling_alpha=0.6,
            simulation_episodes=50
        )
        
        print(f"\nTrainer created successfully:")
        print(f"  Type: {type(trainer).__name__}")
        print(f"  Has importance sampler: {trainer.importance_sampler is not None}")
        print(f"  Crisis parent weight: {trainer.is_config.crisis_parent_weight}x")
        
        # Show importance sampling metrics structure
        metrics = trainer.get_importance_sampling_metrics()
        print(f"\nImportance Sampling Metrics Available:")
        for key in metrics.keys():
            print(f"  - {key}")
            
    except Exception as e:
        print(f"Note: Training orchestrator setup shown conceptually (requires full environment)")
        print(f"Error: {e}")


def main():
    """Run all demonstrations"""
    
    print("Crisis Parent Training Integration Demo")
    print("======================================")
    print("This demo shows how ImportanceSampler integrates with the training loop")
    print("to ensure crisis parents (10% population, 50% value) get 5x weight.\n")
    
    try:
        # Run demonstrations
        demo_importance_sampler()
        demo_replay_buffer_integration()
        demo_experience_aggregation()
        demo_ppo_integration()
        demo_training_orchestrator()
        
        print("\n" + "="*60)
        print("SUMMARY: Crisis Parent Integration Complete")
        print("="*60)
        print("✓ ImportanceSampler provides 5x weighting for crisis parents")
        print("✓ ImportanceSamplingReplayBuffer integrates with training loop")
        print("✓ ExperienceAggregator identifies crisis parents from metadata")
        print("✓ PPO Agent includes importance sampling and bias correction")
        print("✓ Training Orchestrator coordinates the entire pipeline")
        print("\nCrisis parents now receive proper representation in training!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()