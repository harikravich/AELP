#!/usr/bin/env python3
"""
Example usage of the Model Versioning and Experiment Tracking System for GAELP.
Demonstrates integration with training pipelines, A/B testing, and rollback procedures.
"""

import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from model_versioning import (
    ModelVersioningSystem, 
    ExperimentType, 
    ModelStatus,
    create_versioning_system
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockGAELPModel:
    """Mock GAELP model for demonstration purposes"""
    
    def __init__(self, algorithm: str, hyperparams: Dict[str, Any]):
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.weights = np.random.randn(100)  # Mock model weights
        self.training_history = []
    
    def train(self, episodes: int = 100):
        """Mock training process"""
        for episode in range(episodes):
            # Simulate training metrics
            reward = np.random.normal(1000 + episode * 5, 200)
            roas = np.random.normal(3.0 + episode * 0.01, 0.5)
            
            self.training_history.append({
                'episode': episode,
                'reward': reward,
                'roas': roas
            })
        
        # Add some noise to weights to simulate learning
        self.weights += np.random.randn(100) * 0.1
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics"""
        if not self.training_history:
            return {}
        
        rewards = [h['reward'] for h in self.training_history]
        roas_values = [h['roas'] for h in self.training_history]
        
        return {
            'final_reward': rewards[-1],
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'final_roas': roas_values[-1],
            'avg_roas': np.mean(roas_values),
            'max_roas': np.max(roas_values),
            'training_episodes': len(self.training_history),
            'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0
        }


def demonstrate_basic_experiment_tracking():
    """Demonstrate basic experiment tracking and model versioning"""
    logger.info("=== Demonstrating Basic Experiment Tracking ===")
    
    # Initialize versioning system
    versioning = create_versioning_system(
        models_dir="./demo_models",
        experiments_dir="./demo_experiments"
    )
    
    # Start a new experiment
    experiment_id = versioning.track_experiment(
        name="GAELP_PPO_Baseline",
        experiment_type=ExperimentType.TRAINING,
        config={
            "algorithm": "PPO",
            "learning_rate": 0.001,
            "batch_size": 32,
            "gamma": 0.99,
            "clip_param": 0.2,
            "environment": "EnhancedGAELP",
            "max_episodes": 500
        },
        description="Baseline PPO training on GAELP environment",
        tags=["ppo", "baseline", "gaelp"],
        wandb_config={
            "project": "gaelp-demo",
            "entity": "research-team"
        }
    )
    
    logger.info(f"Started experiment: {experiment_id}")
    
    # Create and train a mock model
    model = MockGAELPModel("PPO", {
        "learning_rate": 0.001,
        "batch_size": 32,
        "gamma": 0.99,
        "clip_param": 0.2
    })
    
    # Simulate training
    logger.info("Training model...")
    model.train(episodes=100)
    
    # Get training metrics
    metrics = model.get_final_metrics()
    logger.info(f"Training completed. Final ROAS: {metrics['final_roas']:.2f}")
    
    # Save model version
    model_id = versioning.save_model_version(
        model_obj=model,
        model_name="gaelp_ppo_agent",
        config=model.hyperparams,
        metrics=metrics,
        experiment_id=experiment_id,
        tags=["ppo", "baseline", "v1"],
        description="Baseline PPO agent trained for 100 episodes",
        status=ModelStatus.VALIDATION
    )
    
    logger.info(f"Saved model version: {model_id}")
    
    # Update experiment with results
    versioning.update_experiment(
        experiment_id=experiment_id,
        metrics=metrics,
        status="completed",
        model_versions=[model_id]
    )
    
    return versioning, experiment_id, model_id


def demonstrate_hyperparameter_sweep():
    """Demonstrate hyperparameter sweep with multiple model versions"""
    logger.info("=== Demonstrating Hyperparameter Sweep ===")
    
    versioning = create_versioning_system()
    
    # Start hyperparameter sweep experiment
    experiment_id = versioning.track_experiment(
        name="GAELP_PPO_Hyperparameter_Sweep",
        experiment_type=ExperimentType.HYPERPARAMETER_SWEEP,
        config={
            "algorithm": "PPO",
            "sweep_params": {
                "learning_rate": [0.001, 0.003, 0.01],
                "batch_size": [16, 32, 64],
                "clip_param": [0.1, 0.2, 0.3]
            }
        },
        description="Hyperparameter sweep for PPO on GAELP",
        tags=["ppo", "sweep", "hyperopt"]
    )
    
    # Test different hyperparameter combinations
    learning_rates = [0.001, 0.003, 0.01]
    batch_sizes = [16, 32, 64]
    
    best_model_id = None
    best_roas = 0
    model_versions = []
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            logger.info(f"Testing LR={lr}, Batch Size={batch_size}")
            
            # Create model with specific hyperparameters
            model = MockGAELPModel("PPO", {
                "learning_rate": lr,
                "batch_size": batch_size,
                "gamma": 0.99,
                "clip_param": 0.2
            })
            
            # Train model
            model.train(episodes=50)  # Shorter training for sweep
            metrics = model.get_final_metrics()
            
            # Save model version
            model_id = versioning.save_model_version(
                model_obj=model,
                model_name="gaelp_ppo_agent",
                config=model.hyperparams,
                metrics=metrics,
                experiment_id=experiment_id,
                tags=["ppo", "sweep", f"lr_{lr}", f"batch_{batch_size}"],
                description=f"PPO agent with LR={lr}, Batch={batch_size}",
                status=ModelStatus.TESTING
            )
            
            model_versions.append(model_id)
            
            # Track best model
            if metrics['final_roas'] > best_roas:
                best_roas = metrics['final_roas']
                best_model_id = model_id
    
    # Update experiment with sweep results
    versioning.update_experiment(
        experiment_id=experiment_id,
        metrics={
            "best_model": best_model_id,
            "best_roas": best_roas,
            "total_combinations_tested": len(model_versions)
        },
        status="completed",
        model_versions=model_versions
    )
    
    # Promote best model to validation
    if best_model_id:
        versioning.models_metadata[best_model_id].status = ModelStatus.VALIDATION
        versioning._save_metadata()
    
    logger.info(f"Sweep completed. Best model: {best_model_id} (ROAS: {best_roas:.2f})")
    
    return versioning, experiment_id, model_versions, best_model_id


def demonstrate_ab_testing(versioning, model_a_id: str, model_b_id: str):
    """Demonstrate A/B testing between two model versions"""
    logger.info("=== Demonstrating A/B Testing ===")
    
    # Run A/B test
    test_id = versioning.run_ab_test(
        test_name="PPO_Baseline_vs_Tuned",
        model_a_version=model_a_id,
        model_b_version=model_b_id,
        traffic_split={"model_a": 0.5, "model_b": 0.5},
        duration_hours=24.0
    )
    
    logger.info(f"A/B test completed: {test_id}")
    
    # Get test results
    test_result = versioning.ab_tests[test_id]
    logger.info(f"Winner: {test_result.winner}")
    logger.info(f"Model A ROAS: {test_result.metrics['model_a']['roas']:.2f}")
    logger.info(f"Model B ROAS: {test_result.metrics['model_b']['roas']:.2f}")
    
    return test_id


def demonstrate_model_comparison(versioning, model_ids: List[str]):
    """Demonstrate model version comparison"""
    logger.info("=== Demonstrating Model Comparison ===")
    
    if len(model_ids) < 2:
        logger.warning("Need at least 2 models for comparison")
        return
    
    # Compare first two models
    comparison = versioning.compare_versions(
        model_ids[0],
        model_ids[1],
        metrics=["final_roas", "avg_reward", "max_reward"]
    )
    
    logger.info("Model Comparison Results:")
    logger.info(f"Version 1: {comparison['version1']['model_id']}")
    logger.info(f"Version 2: {comparison['version2']['model_id']}")
    logger.info(f"Overall Winner: {comparison['overall_winner']}")
    
    for metric, diff in comparison['differences'].items():
        logger.info(f"{metric}: {diff['absolute_diff']:.2f} absolute, {diff['relative_diff']:.1f}% relative")
    
    return comparison


def demonstrate_rollback(versioning, target_model_id: str):
    """Demonstrate model rollback functionality"""
    logger.info("=== Demonstrating Model Rollback ===")
    
    # Simulate a problematic deployment
    logger.info("Simulating problematic model deployment...")
    
    # Perform rollback
    success = versioning.rollback(
        target_version=target_model_id,
        reason="Production model showing degraded ROAS performance"
    )
    
    if success:
        logger.info(f"Successfully rolled back to version: {target_model_id}")
    else:
        logger.error("Rollback failed")
    
    return success


def demonstrate_model_lineage(versioning, model_id: str):
    """Demonstrate model lineage tracking"""
    logger.info("=== Demonstrating Model Lineage ===")
    
    lineage = versioning.get_model_lineage(model_id)
    
    logger.info(f"Model: {lineage['model']['model_id']}")
    logger.info(f"Parents: {len(lineage['parents'])}")
    logger.info(f"Children: {len(lineage['children'])}")
    logger.info(f"Lineage Depth: {lineage['depth']}")
    
    return lineage


def demonstrate_experiment_reporting(versioning, experiment_id: str):
    """Demonstrate experiment reporting"""
    logger.info("=== Demonstrating Experiment Reporting ===")
    
    # Get experiment results
    results = versioning.get_experiment_results(experiment_id)
    
    logger.info(f"Experiment: {results['experiment']['name']}")
    logger.info(f"Total Models: {results['summary']['total_models']}")
    logger.info(f"Total A/B Tests: {results['summary']['total_ab_tests']}")
    
    if results['summary']['duration_hours']:
        logger.info(f"Duration: {results['summary']['duration_hours']:.1f} hours")
    
    # Export detailed report
    report_path = versioning.export_experiment_report(experiment_id)
    logger.info(f"Detailed report exported to: {report_path}")
    
    return results


def main():
    """Run all demonstrations"""
    logger.info("Starting GAELP Model Versioning System Demonstration")
    
    try:
        # 1. Basic experiment tracking
        versioning, baseline_exp_id, baseline_model_id = demonstrate_basic_experiment_tracking()
        
        # 2. Hyperparameter sweep
        versioning, sweep_exp_id, sweep_models, best_model_id = demonstrate_hyperparameter_sweep()
        
        # 3. Model comparison
        if len(sweep_models) >= 2:
            comparison = demonstrate_model_comparison(versioning, sweep_models[:2])
        
        # 4. A/B testing
        if baseline_model_id and best_model_id:
            test_id = demonstrate_ab_testing(versioning, baseline_model_id, best_model_id)
        
        # 5. Model lineage
        lineage = demonstrate_model_lineage(versioning, baseline_model_id)
        
        # 6. Rollback demonstration
        rollback_success = demonstrate_rollback(versioning, baseline_model_id)
        
        # 7. Experiment reporting
        baseline_results = demonstrate_experiment_reporting(versioning, baseline_exp_id)
        sweep_results = demonstrate_experiment_reporting(versioning, sweep_exp_id)
        
        # 8. Show production models
        production_models = versioning.get_production_models()
        logger.info(f"Current production models: {len(production_models)}")
        
        # 9. Show model history
        history = versioning.get_model_history("gaelp_ppo")
        logger.info(f"Total GAELP PPO model versions: {len(history)}")
        
        logger.info("Demonstration completed successfully!")
        
        return {
            "versioning_system": versioning,
            "experiments": [baseline_exp_id, sweep_exp_id],
            "models": {
                "baseline": baseline_model_id,
                "best": best_model_id,
                "all_sweep": sweep_models
            },
            "production_models": len(production_models),
            "total_versions": len(history)
        }
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    demo_results = main()
    
    print("\n=== Demonstration Summary ===")
    print(f"Experiments created: {len(demo_results['experiments'])}")
    print(f"Model versions: {len(demo_results['models']['all_sweep'])}")
    print(f"Production models: {demo_results['production_models']}")
    print(f"Total versions tracked: {demo_results['total_versions']}")