#!/usr/bin/env python3
"""
Weights & Biases integration for GAELP experiment tracking.
Tracks RL training metrics, performance, and hyperparameters.
"""

import wandb
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class GAELPWandbTracker:
    """Weights & Biases tracker for GAELP experiments"""
    
    def __init__(
        self,
        project_name: str = "gaelp-rl-training",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        anonymous: bool = True
    ):
        """
        Initialize W&B tracking for GAELP experiments
        
        Args:
            project_name: W&B project name
            experiment_name: Experiment run name
            config: Configuration dictionary to track
            tags: List of tags for the experiment
            anonymous: Use anonymous mode if no API key available
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.config = config or {}
        self.tags = tags or []
        self.anonymous = anonymous
        self.run = None
        self.episode_metrics = []
        
        # Initialize W&B run
        self._init_wandb()
    
    def _generate_experiment_name(self) -> str:
        """Generate a unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"gaelp_training_{timestamp}"
    
    def _init_wandb(self):
        """Initialize Weights & Biases run"""
        try:
            # Set anonymous mode if no API key
            if self.anonymous or not os.getenv('WANDB_API_KEY'):
                os.environ['WANDB_MODE'] = 'offline'
                logger.info("Using W&B in offline/anonymous mode")
            
            # Initialize run
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                tags=self.tags,
                reinit=True
            )
            
            # Log system info
            self._log_system_info()
            
            logger.info(f"Initialized W&B tracking for project: {self.project_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}. Continuing without tracking.")
            self.run = None
    
    def _log_system_info(self):
        """Log system and environment information"""
        if not self.run:
            return
        
        try:
            import platform
            import psutil
            
            system_info = {
                "system/platform": platform.platform(),
                "system/python_version": platform.python_version(),
                "system/cpu_count": psutil.cpu_count(),
                "system/memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            }
            
            wandb.config.update(system_info)
            
        except Exception as e:
            logger.warning(f"Could not log system info: {e}")
    
    def log_training_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        if not self.run:
            return
        
        try:
            wandb.config.update({
                "training": config
            })
            logger.info("Logged training configuration to W&B")
        except Exception as e:
            logger.warning(f"Failed to log training config: {e}")
    
    def log_episode_metrics(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        roas: float,
        ctr: float,
        conversion_rate: float,
        total_cost: float,
        total_revenue: float,
        avg_cpc: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Log metrics for a single episode
        
        Args:
            episode: Episode number
            total_reward: Total reward for the episode
            steps: Number of steps in the episode
            roas: Return on Ad Spend
            ctr: Click-through rate
            conversion_rate: Conversion rate
            total_cost: Total cost for the episode
            total_revenue: Total revenue for the episode
            avg_cpc: Average cost per click
            additional_metrics: Additional custom metrics
        """
        if not self.run:
            return
        
        try:
            # Core RL metrics
            metrics = {
                "episode/number": episode,
                "episode/total_reward": total_reward,
                "episode/steps": steps,
                "episode/reward_per_step": total_reward / max(steps, 1),
                
                # Business metrics
                "performance/roas": roas,
                "performance/ctr": ctr,
                "performance/conversion_rate": conversion_rate,
                "performance/total_cost": total_cost,
                "performance/total_revenue": total_revenue,
                "performance/profit": total_revenue - total_cost,
                
                # Efficiency metrics
                "efficiency/cost_per_conversion": total_cost / max(conversion_rate * steps, 1),
                "efficiency/revenue_per_step": total_revenue / max(steps, 1),
            }
            
            # Add CPC if available
            if avg_cpc is not None:
                metrics["performance/avg_cpc"] = avg_cpc
            
            # Add additional metrics
            if additional_metrics:
                for key, value in additional_metrics.items():
                    metrics[f"custom/{key}"] = value
            
            # Log to W&B
            wandb.log(metrics, step=episode)
            
            # Store for local analysis
            self.episode_metrics.append(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log episode metrics: {e}")
    
    def log_batch_metrics(self, episode_batch: List[Dict[str, Any]], batch_size: int = 10):
        """Log aggregated metrics for a batch of episodes"""
        if not self.run or not episode_batch:
            return
        
        try:
            # Calculate batch statistics
            rewards = [ep['total_reward'] for ep in episode_batch]
            roas_values = [ep.get('final_roas', 0) for ep in episode_batch]
            steps = [ep.get('steps', 0) for ep in episode_batch]
            
            batch_metrics = {
                f"batch/avg_reward_{batch_size}ep": np.mean(rewards),
                f"batch/std_reward_{batch_size}ep": np.std(rewards),
                f"batch/avg_roas_{batch_size}ep": np.mean(roas_values),
                f"batch/std_roas_{batch_size}ep": np.std(roas_values),
                f"batch/avg_steps_{batch_size}ep": np.mean(steps),
                f"batch/reward_trend": np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0,
            }
            
            # Log batch metrics
            episode_num = episode_batch[-1].get('episode', 0)
            wandb.log(batch_metrics, step=episode_num)
            
            logger.info(f"Logged batch metrics for episodes {episode_num-len(episode_batch)+1}-{episode_num}")
            
        except Exception as e:
            logger.warning(f"Failed to log batch metrics: {e}")
    
    def log_evaluation_metrics(self, eval_results: Dict[str, Any]):
        """Log evaluation results against real data"""
        if not self.run:
            return
        
        try:
            eval_metrics = {}
            for key, value in eval_results.items():
                eval_metrics[f"evaluation/{key}"] = value
            
            wandb.log(eval_metrics)
            logger.info("Logged evaluation metrics to W&B")
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters"""
        if not self.run:
            return
        
        try:
            wandb.config.update({
                "hyperparameters": hyperparams
            })
            logger.info("Logged hyperparameters to W&B")
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_model_artifacts(self, model_path: str, model_name: str = "gaelp_agent"):
        """Log model artifacts"""
        if not self.run:
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"{model_name}_{self.experiment_name}",
                type="model",
                description=f"GAELP RL agent model from {self.experiment_name}"
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
            logger.info(f"Logged model artifact: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")
    
    def log_environment_calibration(self, calibration_data: Dict[str, Any]):
        """Log environment calibration statistics"""
        if not self.run:
            return
        
        try:
            cal_metrics = {}
            for key, value in calibration_data.items():
                cal_metrics[f"calibration/{key}"] = value
            
            wandb.log(cal_metrics)
            logger.info("Logged environment calibration data to W&B")
            
        except Exception as e:
            logger.warning(f"Failed to log calibration data: {e}")
    
    def log_learning_curve(self):
        """Generate and log learning curve visualization"""
        if not self.run or not self.episode_metrics:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            episodes = [m["episode/number"] for m in self.episode_metrics]
            rewards = [m["episode/total_reward"] for m in self.episode_metrics]
            roas_values = [m["performance/roas"] for m in self.episode_metrics]
            
            # Create learning curve plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Reward progression
            ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
            # Moving average
            window = min(10, len(rewards) // 4)
            if window > 1:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(episodes[window-1:], moving_avg, color='red', label=f'Moving Average ({window})')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Learning Curve - Reward Progression')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # ROAS progression
            ax2.plot(episodes, roas_values, alpha=0.3, color='green', label='Episode ROAS')
            if window > 1:
                roas_avg = np.convolve(roas_values, np.ones(window)/window, mode='valid')
                ax2.plot(episodes[window-1:], roas_avg, color='orange', label=f'Moving Average ({window})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('ROAS')
            ax2.set_title('Learning Curve - ROAS Progression')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log to W&B
            wandb.log({"learning_curve": wandb.Image(fig)})
            plt.close(fig)
            
            logger.info("Generated and logged learning curve visualization")
            
        except Exception as e:
            logger.warning(f"Failed to generate learning curve: {e}")
    
    def save_local_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save results locally as backup"""
        try:
            if filename is None:
                filename = f"gaelp_results_{self.experiment_name}.json"
            
            output_data = {
                "experiment_name": self.experiment_name,
                "project_name": self.project_name,
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "results": results,
                "summary": self._generate_summary(results)
            }
            
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Saved results locally to {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save local results: {e}")
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not results:
            return {}
        
        try:
            rewards = [r.get('total_reward', 0) for r in results]
            roas_values = [r.get('final_roas', 0) for r in results]
            
            return {
                "total_episodes": len(results),
                "avg_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "max_reward": np.max(rewards),
                "avg_roas": np.mean(roas_values),
                "max_roas": np.max(roas_values),
                "final_10_avg_reward": np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
                "final_10_avg_roas": np.mean(roas_values[-10:]) if len(roas_values) >= 10 else np.mean(roas_values)
            }
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return {}
    
    def finish(self):
        """Finish the W&B run"""
        try:
            if self.run:
                # Log final learning curve
                self.log_learning_curve()
                
                # Finish the run
                wandb.finish()
                logger.info("Finished W&B tracking session")
        except Exception as e:
            logger.warning(f"Error finishing W&B run: {e}")


class GAELPExperimentConfig:
    """Configuration class for GAELP experiments"""
    
    def __init__(
        self,
        agent_type: str = "PPO",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_episodes: int = 100,
        max_steps_per_episode: int = 100,
        environment_type: str = "EnhancedGAELP",
        reward_function: str = "ROAS_optimized",
        **kwargs
    ):
        self.agent_type = agent_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.environment_type = environment_type
        self.reward_function = reward_function
        
        # Additional hyperparameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for W&B logging"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def create_experiment_tracker(
    experiment_name: str = None,
    config: GAELPExperimentConfig = None,
    tags: List[str] = None
) -> GAELPWandbTracker:
    """
    Factory function to create a GAELP experiment tracker
    
    Args:
        experiment_name: Name for the experiment
        config: Experiment configuration
        tags: Tags for the experiment
    
    Returns:
        Initialized GAELPWandbTracker
    """
    config_dict = config.to_dict() if config else {}
    
    return GAELPWandbTracker(
        experiment_name=experiment_name,
        config=config_dict,
        tags=tags,
        anonymous=True  # Default to anonymous mode
    )