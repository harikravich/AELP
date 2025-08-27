#!/usr/bin/env python3
"""
Demo script showing GAELP W&B integration with simulated training.
This demonstrates the complete tracking workflow without requiring full GAELP infrastructure.
"""

import numpy as np
import asyncio
import logging
from wandb_tracking import GAELPWandbTracker, GAELPExperimentConfig, create_experiment_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulatedGAELPTrainer:
    """Simulated GAELP trainer for demonstrating W&B integration"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or "gaelp_wandb_demo"
        
        # Initialize W&B tracking
        self._init_wandb_tracking()
        
        # Simulated environment calibration
        self.calibration_stats = {
            'mean_ctr': 0.025,
            'std_ctr': 0.015,
            'mean_cpc': 1.5,
            'mean_conv_rate': 0.05,
            'mean_roas': 2.3
        }
        
        if self.wandb_tracker:
            self.wandb_tracker.log_environment_calibration(self.calibration_stats)
        
        logger.info("Initialized simulated GAELP trainer with W&B tracking")
    
    def _init_wandb_tracking(self):
        """Initialize W&B tracking"""
        config = GAELPExperimentConfig(
            agent_type="PPO",
            learning_rate=0.001,
            batch_size=32,
            num_episodes=50,
            max_steps_per_episode=100,
            environment_type="SimulatedGAELP",
            reward_function="ROAS_optimized",
            # Additional hyperparameters
            gamma=0.99,
            epsilon=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01
        )
        
        self.wandb_tracker = create_experiment_tracker(
            experiment_name=self.experiment_name,
            config=config,
            tags=["GAELP", "RL", "demo", "advertising", "PPO"]
        )
    
    async def simulate_episode(self, episode_num: int) -> dict:
        """Simulate a single training episode"""
        
        # Simulate episode progression with realistic dynamics
        base_performance = 1.0 + (episode_num / 100)  # Gradual improvement
        noise = np.random.normal(0, 0.1)
        performance_factor = base_performance + noise
        
        # Simulate metrics with realistic correlations
        total_impressions = np.random.randint(8000, 12000)
        ctr = max(0.01, np.random.normal(0.025 * performance_factor, 0.005))
        total_clicks = int(total_impressions * ctr)
        
        conversion_rate = max(0.01, np.random.normal(0.05 * performance_factor, 0.01))
        total_conversions = int(total_clicks * conversion_rate)
        
        avg_cpc = max(0.5, np.random.normal(1.5 / performance_factor, 0.3))
        total_cost = total_clicks * avg_cpc
        
        roas = max(0.1, np.random.normal(2.3 * performance_factor, 0.5))
        total_revenue = total_cost * roas
        
        # RL-specific metrics
        steps = np.random.randint(50, 100)
        # Reward based on ROAS and efficiency
        reward_per_step = (roas - 1.0) * 10 + np.random.normal(0, 1)
        total_reward = reward_per_step * steps
        
        # Simulate some async processing time
        await asyncio.sleep(0.01)
        
        return {
            'episode': episode_num,
            'total_reward': total_reward,
            'steps': steps,
            'final_roas': roas,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            'avg_cpc': avg_cpc,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_conversions': total_conversions
        }
    
    async def train(self, num_episodes: int = 50):
        """Simulate full training loop with W&B tracking"""
        
        logger.info(f"Starting simulated training for {num_episodes} episodes")
        results = []
        
        for episode in range(1, num_episodes + 1):
            # Simulate episode
            episode_metrics = await self.simulate_episode(episode)
            results.append(episode_metrics)
            
            # Log to W&B
            if self.wandb_tracker:
                self.wandb_tracker.log_episode_metrics(
                    episode=episode,
                    total_reward=episode_metrics['total_reward'],
                    steps=episode_metrics['steps'],
                    roas=episode_metrics['final_roas'],
                    ctr=episode_metrics['ctr'],
                    conversion_rate=episode_metrics['conversion_rate'],
                    total_cost=episode_metrics['total_cost'],
                    total_revenue=episode_metrics['total_revenue'],
                    avg_cpc=episode_metrics['avg_cpc'],
                    additional_metrics={
                        'total_impressions': episode_metrics['total_impressions'],
                        'total_clicks': episode_metrics['total_clicks'],
                        'total_conversions': episode_metrics['total_conversions']
                    }
                )
            
            # Log batch metrics every 10 episodes
            if episode % 10 == 0:
                recent_results = results[-10:]
                if self.wandb_tracker:
                    self.wandb_tracker.log_batch_metrics(recent_results, batch_size=10)
                
                # Log progress
                avg_reward = np.mean([r['total_reward'] for r in recent_results])
                avg_roas = np.mean([r['final_roas'] for r in recent_results])
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.2f}, Avg ROAS={avg_roas:.2f}x")
        
        return results
    
    def simulate_evaluation(self, num_samples: int = 100):
        """Simulate evaluation on 'real' data"""
        
        logger.info("Simulating evaluation on real data...")
        
        # Simulate evaluation metrics
        predictions = []
        for _ in range(num_samples):
            actual_roas = np.random.normal(2.3, 0.5)
            predicted_roas = actual_roas + np.random.normal(0, 0.3)  # Some prediction error
            error = abs(predicted_roas - actual_roas)
            
            predictions.append({
                'predicted': predicted_roas,
                'actual': actual_roas,
                'error': error
            })
        
        import pandas as pd
        pred_df = pd.DataFrame(predictions)
        
        eval_metrics = {
            'mean_absolute_error': pred_df['error'].mean(),
            'correlation': pred_df['predicted'].corr(pred_df['actual']),
            'accuracy': (pred_df['error'] < 1.0).mean(),
            'rmse': np.sqrt(np.mean(pred_df['error']**2))
        }
        
        # Log to W&B
        if self.wandb_tracker:
            self.wandb_tracker.log_evaluation_metrics(eval_metrics)
        
        logger.info(f"Evaluation Results: MAE={eval_metrics['mean_absolute_error']:.2f}, "
                   f"Correlation={eval_metrics['correlation']:.3f}")
        
        return eval_metrics
    
    def finish_training(self, results: list):
        """Finish training and cleanup"""
        if self.wandb_tracker:
            # Save results locally
            self.wandb_tracker.save_local_results(results, f"{self.experiment_name}_results.json")
            
            # Finish W&B session (includes learning curve generation)
            self.wandb_tracker.finish()
            
            logger.info("Finished W&B tracking session")


async def run_demo():
    """Run the complete W&B integration demo"""
    
    print("🚀 GAELP W&B Integration Demo")
    print("=" * 50)
    print("Features demonstrated:")
    print("  • Experiment configuration tracking")
    print("  • Episode-by-episode metrics logging")
    print("  • Batch metrics aggregation")
    print("  • Environment calibration tracking")
    print("  • Evaluation metrics logging")
    print("  • Learning curve visualization")
    print("  • Local results backup")
    print("  • Anonymous/offline mode support")
    print()
    
    # Create trainer
    trainer = SimulatedGAELPTrainer("gaelp_integration_demo")
    
    # Phase 1: Training
    print("📚 Phase 1: Training Simulation")
    results = await trainer.train(num_episodes=50)
    
    # Phase 2: Evaluation
    print("\n🎯 Phase 2: Evaluation Simulation")
    eval_metrics = trainer.simulate_evaluation(num_samples=200)
    
    # Phase 3: Analysis
    print("\n📊 Phase 3: Results Analysis")
    final_results = results[-10:]
    avg_final_reward = np.mean([r['total_reward'] for r in final_results])
    avg_final_roas = np.mean([r['final_roas'] for r in final_results])
    
    print(f"Final Performance (last 10 episodes):")
    print(f"  • Average Reward: {avg_final_reward:.2f}")
    print(f"  • Average ROAS: {avg_final_roas:.2f}x")
    print(f"  • Evaluation MAE: {eval_metrics['mean_absolute_error']:.2f}")
    print(f"  • Evaluation Correlation: {eval_metrics['correlation']:.3f}")
    
    # Show learning progression
    print("\n📈 Learning Progression:")
    for i in range(0, len(results), 10):
        batch = results[i:i+10]
        if batch:
            avg_roas = np.mean([r['final_roas'] for r in batch])
            avg_reward = np.mean([r['total_reward'] for r in batch])
            print(f"  Episodes {i+1}-{i+10}: ROAS={avg_roas:.2f}x, Reward={avg_reward:.2f}")
    
    # Cleanup
    trainer.finish_training(results)
    
    print("\n🎉 Demo completed successfully!")
    print("\nFiles created:")
    print("  • W&B offline run data in ./wandb/")
    print("  • Local results backup: gaelp_integration_demo_results.json")
    print("  • Learning curve visualization in W&B logs")
    print("\nTo sync to W&B cloud, run: wandb sync <run_directory>")


if __name__ == "__main__":
    asyncio.run(run_demo())