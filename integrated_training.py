#!/usr/bin/env python3
"""
Integrated GAELP training with enhanced simulator and real data.
Combines AuctionGym-style dynamics with real campaign data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import asyncio
import logging

from enhanced_simulator import EnhancedGAELPEnvironment
from training_orchestrator.rl_agents.agent_factory import AgentFactory, AgentFactoryConfig, AgentType
from training_orchestrator.checkpoint_manager import CheckpointManager
from wandb_tracking import GAELPWandbTracker, GAELPExperimentConfig, create_experiment_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedGAELPTrainer:
    """Trains GAELP agents with realistic simulator calibrated on real data"""
    
    def __init__(self, experiment_name: str = None, enable_wandb: bool = True):
        # Load real data for calibration
        self.real_data = pd.read_csv('data/aggregated_data.csv')
        
        # Initialize enhanced environment
        self.env = EnhancedGAELPEnvironment()
        
        # Calibrate environment with real data statistics
        calibration_stats = self._calibrate_environment()
        
        # Checkpoint manager (before agent creation)
        self.checkpoint_manager = CheckpointManager()
        
        # Initialize RL agent
        self.agent = self._create_agent()
        
        # Initialize W&B tracking
        self.wandb_tracker = None
        if enable_wandb:
            self._init_wandb_tracking(experiment_name, calibration_stats)
        
    def _calibrate_environment(self):
        """Calibrate simulator with real data statistics"""
        
        # Extract real statistics
        real_stats = {
            'mean_ctr': self.real_data['ctr'].mean(),
            'std_ctr': self.real_data['ctr'].std(),
            'mean_cpc': self.real_data['cpc'].mean(),
            'mean_conv_rate': self.real_data['conversion_rate'].mean(),
            'mean_roas': self.real_data['roas'].mean()
        }
        
        # Update simulator parameters
        self.env.calibrator.benchmarks.update(real_stats)
        
        logger.info(f"Calibrated simulator with real data: mean CTR={real_stats['mean_ctr']:.3%}")
        
        return real_stats
    
    def _create_agent(self):
        """Create RL agent for training"""
        
        config = AgentFactoryConfig(
            agent_type=AgentType.PPO,
            agent_id="gaelp_integrated",
            state_dim=128,
            action_dim=64,
            enable_state_processing=True,
            enable_reward_engineering=True
        )
        
        factory = AgentFactory(config)
        agent = factory.create_agent()
        
        # Load checkpoint if exists
        loaded_episode = self.checkpoint_manager.load_latest_checkpoint(agent)
        if loaded_episode:
            logger.info(f"Loaded checkpoint from episode {loaded_episode}")
        
        return agent
    
    def _init_wandb_tracking(self, experiment_name: str, calibration_stats: Dict[str, Any]):
        """Initialize Weights & Biases tracking"""
        try:
            # Create experiment configuration
            config = GAELPExperimentConfig(
                agent_type="PPO",
                learning_rate=0.001,
                batch_size=32,
                environment_type="EnhancedGAELP",
                reward_function="ROAS_optimized",
                max_steps_per_episode=100,
                # Add calibration stats to config
                **{f"calibration_{k}": v for k, v in calibration_stats.items()}
            )
            
            # Create tracker
            self.wandb_tracker = create_experiment_tracker(
                experiment_name=experiment_name,
                config=config,
                tags=["GAELP", "RL", "advertising", "PPO", "integrated_training"]
            )
            
            # Log environment calibration data
            self.wandb_tracker.log_environment_calibration(calibration_stats)
            
            logger.info("Initialized W&B tracking for GAELP training")
            
        except Exception as e:
            logger.warning(f"Failed to initialize W&B tracking: {e}")
            self.wandb_tracker = None
    
    async def train_episode(self, episode_num: int) -> Dict[str, Any]:
        """Train one episode with enhanced environment"""
        
        obs = self.env.reset()
        episode_reward = 0
        episode_data = []
        
        for step in range(100):  # Max 100 steps per episode
            # Get action from agent
            state = self._obs_to_state(obs)
            action_values = await self.agent.select_action(state)
            
            # Convert agent action to environment action
            env_action = {
                'bid': float(action_values.get('bid', 1.0)),
                'budget': 1000,
                'quality_score': float(action_values.get('quality', 0.7)),
                'creative': {
                    'quality_score': float(action_values.get('creative_quality', 0.6)),
                    'price_shown': float(action_values.get('price', 50))
                }
            }
            
            # Step environment
            next_obs, reward, done, info = self.env.step(env_action)
            
            # Store experience
            episode_data.append({
                'state': state,
                'action': action_values,
                'reward': reward,
                'next_state': self._obs_to_state(next_obs),
                'done': done
            })
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        # Update agent with episode data
        if len(episode_data) > 0:
            self.agent.update_policy(episode_data)
        
        # Calculate additional metrics for tracking
        total_impressions = obs.get('impressions', 0)
        total_clicks = obs.get('clicks', 0)
        total_conversions = obs.get('conversions', 0)
        total_cost = obs.get('total_cost', 0)
        total_revenue = obs.get('total_revenue', 0)
        
        # Calculate derived metrics
        ctr = total_clicks / max(total_impressions, 1)
        conversion_rate = total_conversions / max(total_clicks, 1)
        avg_cpc = total_cost / max(total_clicks, 1)
        final_roas = obs.get('roas', 0)
        
        metrics = {
            'episode': episode_num,
            'total_reward': episode_reward,
            'steps': len(episode_data),
            'final_roas': final_roas,
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            'avg_cpc': avg_cpc,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_conversions': total_conversions
        }
        
        # Log to W&B if available
        if self.wandb_tracker:
            self.wandb_tracker.log_episode_metrics(
                episode=episode_num,
                total_reward=episode_reward,
                steps=len(episode_data),
                roas=final_roas,
                ctr=ctr,
                conversion_rate=conversion_rate,
                total_cost=total_cost,
                total_revenue=total_revenue,
                avg_cpc=avg_cpc,
                additional_metrics={
                    'total_impressions': total_impressions,
                    'total_clicks': total_clicks,
                    'total_conversions': total_conversions
                }
            )
        
        return metrics
    
    def _obs_to_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert environment observation to agent state"""
        
        return {
            'cost': obs.get('total_cost', 0),
            'revenue': obs.get('total_revenue', 0),
            'impressions': obs.get('impressions', 0),
            'clicks': obs.get('clicks', 0),
            'conversions': obs.get('conversions', 0),
            'cpc': obs.get('avg_cpc', 0),
            'roas': obs.get('roas', 0)
        }
    
    async def train(self, num_episodes: int = 100):
        """Main training loop"""
        
        logger.info(f"Starting integrated training for {num_episodes} episodes")
        
        results = []
        
        for episode in range(1, num_episodes + 1):
            metrics = await self.train_episode(episode)
            results.append(metrics)
            
            # Log progress
            if episode % 10 == 0:
                recent_results = results[-10:]
                recent_rewards = [r['total_reward'] for r in recent_results]
                recent_roas = [r['final_roas'] for r in recent_results]
                
                logger.info(
                    f"Episode {episode}: "
                    f"Avg Reward={np.mean(recent_rewards):.3f}, "
                    f"Avg ROAS={np.mean(recent_roas):.2f}x"
                )
                
                # Log batch metrics to W&B
                if self.wandb_tracker:
                    self.wandb_tracker.log_batch_metrics(recent_results, batch_size=10)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    self.agent,
                    episode,
                    {'avg_reward': np.mean(recent_rewards), 'avg_roas': np.mean(recent_roas)}
                )
        
        return results
    
    def evaluate_on_real_data(self, n_samples: int = 100) -> Dict[str, Any]:
        """Evaluate agent performance on real data samples"""
        
        logger.info("Evaluating on real data...")
        
        # Sample real campaigns
        real_samples = self.real_data.sample(n_samples)
        
        predictions = []
        for _, campaign in real_samples.iterrows():
            # Create state from real campaign
            state = {
                'impressions': campaign['impressions'],
                'clicks': campaign['clicks'],
                'cost': campaign['cost'],
                'conversions': campaign['conversions'],
                'revenue': campaign['revenue']
            }
            
            # Get agent's action
            action = self.agent.select_action(state)
            
            # Predict performance
            predicted_roas = action.get('expected_roas', 0)
            actual_roas = campaign['roas']
            
            predictions.append({
                'predicted': predicted_roas,
                'actual': actual_roas,
                'error': abs(predicted_roas - actual_roas)
            })
        
        # Calculate metrics
        pred_df = pd.DataFrame(predictions)
        metrics = {
            'mean_absolute_error': pred_df['error'].mean(),
            'correlation': pred_df['predicted'].corr(pred_df['actual']),
            'accuracy': (pred_df['error'] < 1.0).mean()  # Within 1x ROAS
        }
        
        logger.info(f"Evaluation Results: MAE={metrics['mean_absolute_error']:.2f}, Correlation={metrics['correlation']:.3f}")
        
        # Log evaluation metrics to W&B
        if self.wandb_tracker:
            self.wandb_tracker.log_evaluation_metrics(metrics)
        
        return metrics
    
    def finish_training(self, results: List[Dict[str, Any]]):
        """Clean up and finalize training session"""
        if self.wandb_tracker:
            # Save results locally as backup
            self.wandb_tracker.save_local_results(results)
            
            # Finish W&B session
            self.wandb_tracker.finish()
            
            logger.info("Finished W&B tracking session")


async def main():
    """Run integrated training with enhanced simulator and real data"""
    
    print("🚀 GAELP Integrated Training System")
    print("=" * 50)
    print("Using:")
    print("  • Enhanced auction simulator (AuctionGym-style)")
    print("  • Real campaign data calibration")
    print("  • Persistent learning with checkpoints")
    print("  • Realistic user behavior modeling")
    print("  • Weights & Biases experiment tracking")
    print()
    
    trainer = IntegratedGAELPTrainer(experiment_name="gaelp_integrated_demo")
    
    # Train for initial episodes
    print("Phase 1: Initial Training on Enhanced Simulator")
    results = await trainer.train(num_episodes=50)
    
    # Evaluate on real data
    print("\nPhase 2: Evaluation on Real Data")
    eval_metrics = trainer.evaluate_on_real_data(n_samples=100)
    
    print(f"\n📊 Training Complete!")
    print(f"Final Performance:")
    print(f"  Average ROAS: {np.mean([r['final_roas'] for r in results[-10:]]):.2f}x")
    print(f"  Real Data MAE: {eval_metrics['mean_absolute_error']:.2f}")
    print(f"  Correlation with Real: {eval_metrics['correlation']:.3f}")
    
    # Show learning progression
    print("\n📈 Learning Progression:")
    for i in range(0, len(results), 10):
        batch = results[i:i+10]
        if batch:
            avg_roas = np.mean([r['final_roas'] for r in batch])
            avg_reward = np.mean([r['total_reward'] for r in batch])
            print(f"  Episodes {i+1}-{i+10}: ROAS={avg_roas:.2f}x, Reward={avg_reward:.3f}")
    
    # Finish training and clean up W&B session
    trainer.finish_training(results)


if __name__ == "__main__":
    asyncio.run(main())