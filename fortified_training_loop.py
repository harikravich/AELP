#!/usr/bin/env python3
"""
FORTIFIED TRAINING LOOP FOR GAELP
Complete training system with all components integrated
"""

import os
import sys
import logging
import time
import numpy as np
from datetime import datetime
import torch
import ray
from typing import Dict, List, Any, Optional
import json

# Add GAELP to path
sys.path.insert(0, '/home/hariravichandran/AELP')

# Import all fortified components
from fortified_rl_agent import FortifiedRLAgent, EnrichedJourneyState
from fortified_environment import FortifiedGAELPEnvironment
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fortified_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@ray.remote
class ParallelEnvironment:
    """Ray actor for parallel environment execution"""
    
    def __init__(self, env_id: int, config: Dict[str, Any]):
        self.env_id = env_id
        self.config = config
        self.env = FortifiedGAELPEnvironment(
            max_budget=config.get('max_budget', 10000),
            max_steps=config.get('max_steps', 1000),
            use_real_ga4_data=config.get('use_ga4', True),
            is_parallel=True  # This is a Ray parallel environment
        )
        self.metrics = {
            'episodes': 0,
            'total_reward': 0,
            'conversions': 0,
            'revenue': 0
        }
    
    def collect_experience(self, agent: FortifiedRLAgent, num_steps: int) -> List[Dict]:
        """Collect experience from environment"""
        experiences = []
        
        state = self.env.reset()
        state_obj = self.env.current_state
        
        for _ in range(num_steps):
            # Get action from agent
            action = agent.select_action(state_obj, explore=True)
            
            # Execute in environment
            next_state, reward, done, info = self.env.step(action)
            next_state_obj = self.env.current_state
            
            # Store experience
            experience = {
                'state': state_obj,
                'action': action,
                'reward': reward,
                'next_state': next_state_obj,
                'done': done,
                'info': info
            }
            experiences.append(experience)
            
            # Update metrics
            self.metrics['total_reward'] += reward
            # Track both immediate and total conversions (including delayed)
            metrics = info.get('metrics', {})
            if metrics.get('total_conversions', 0) > 0:
                self.metrics['conversions'] += metrics.get('total_conversions', 0)
                self.metrics['revenue'] += metrics.get('total_revenue', 0)
            
            if done:
                self.metrics['episodes'] += 1
                state = self.env.reset()
                state_obj = self.env.current_state
            else:
                state = next_state
                state_obj = next_state_obj
        
        return experiences
    
    def get_metrics(self) -> Dict:
        """Get environment metrics"""
        return self.metrics

class FortifiedTrainingSystem:
    """
    Complete training system with all components integrated
    """
    
    def __init__(self,
                 num_environments: int = 16,
                 total_episodes: int = 10000,
                 batch_size: int = 256,
                 checkpoint_interval: int = 100):
        
        self.num_environments = num_environments
        self.total_episodes = total_episodes
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        logger.info("=" * 80)
        logger.info("FORTIFIED GAELP TRAINING SYSTEM")
        logger.info("=" * 80)
        
        # Initialize Ray for parallel training
        if not ray.is_initialized():
            ray.init(num_cpus=num_environments + 2)
        
        # Initialize all components
        logger.info("Initializing components...")
        
        # Discovery Engine - run discovery ONCE before parallel processes
        self.discovery = DiscoveryEngine(write_enabled=True, cache_only=False)
        patterns = self.discovery.discover_all_patterns()  # Discover and save patterns
        logger.info(f"Main process discovered {len(patterns.segments)} segments")
        
        self.creative_selector = CreativeSelector()
        self.attribution = AttributionEngine()
        self.budget_pacer = BudgetPacer()
        self.identity_resolver = IdentityResolver()
        self.pm = ParameterManager()
        
        # Initialize fortified agent
        self.agent = FortifiedRLAgent(
            discovery_engine=self.discovery,
            creative_selector=self.creative_selector,
            attribution_engine=self.attribution,
            budget_pacer=self.budget_pacer,
            identity_resolver=self.identity_resolver,
            parameter_manager=self.pm
        )
        
        # Create parallel environments
        logger.info(f"Creating {num_environments} parallel environments...")
        self.environments = [
            ParallelEnvironment.remote(
                env_id=i,
                config={
                    'max_budget': 10000,
                    'max_steps': 1000,
                    'use_ga4': True
                }
            )
            for i in range(num_environments)
        ]
        
        # Training metrics
        self.training_metrics = {
            'episodes_completed': 0,
            'total_experiences': 0,
            'total_conversions': 0,
            'total_revenue': 0.0,
            'avg_reward': 0.0,
            'best_roas': 0.0,
            'training_losses': [],
            'epsilon_history': [],
            'performance_history': []
        }
        
        logger.info("Training system initialized successfully")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting fortified training...")
        start_time = time.time()
        
        episodes_per_env = self.total_episodes // self.num_environments
        steps_per_collection = 100
        
        for episode in range(episodes_per_env):
            # Collect experiences in parallel
            logger.info(f"\n--- Episode {episode}/{episodes_per_env} ---")
            
            experience_futures = [
                env.collect_experience.remote(
                    self.agent,
                    steps_per_collection
                )
                for env in self.environments
            ]
            
            # Wait for all environments
            all_experiences = ray.get(experience_futures)
            
            # Process and store experiences
            for env_experiences in all_experiences:
                for exp in env_experiences:
                    self.agent.store_experience(
                        state=exp['state'],
                        action=exp['action'],
                        reward=exp['reward'],
                        next_state=exp['next_state'],
                        done=exp['done']
                    )
                    
                    # Update agent's performance history
                    self.agent.update_performance_history(
                        creative_id=exp['action']['creative_id'],
                        channel=exp['action']['channel'],
                        result=exp['info'].get('auction_result', {})
                    )
            
            self.training_metrics['total_experiences'] += len(all_experiences) * steps_per_collection
            
            # Train agent when buffer is ready
            if len(self.agent.replay_buffer) >= self.batch_size:
                train_metrics = self.agent.train(batch_size=self.batch_size)
                
                if train_metrics:
                    self.training_metrics['training_losses'].append(train_metrics)
                    self.training_metrics['epsilon_history'].append(train_metrics['epsilon'])
                    
                    logger.info(f"Training - Loss (bid): {train_metrics['loss_bid']:.4f}, "
                              f"Loss (creative): {train_metrics['loss_creative']:.4f}, "
                              f"Loss (channel): {train_metrics['loss_channel']:.4f}, "
                              f"Epsilon: {train_metrics['epsilon']:.4f}")
            
            # Get metrics from environments
            if episode % 10 == 0:
                self._log_progress(episode, episodes_per_env)
            
            # Checkpoint
            if episode % self.checkpoint_interval == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Early stopping if converged
            if self._check_convergence():
                logger.info("Training converged early!")
                break
        
        # Final checkpoint
        self._save_checkpoint(episodes_per_env)
        
        # Training complete
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        logger.info(f"Total experiences: {self.training_metrics['total_experiences']:,}")
        logger.info(f"Final epsilon: {self.agent.epsilon:.4f}")
        logger.info("=" * 80)
        
        # Save final metrics
        self._save_final_report()
    
    def _log_progress(self, episode: int, episodes_per_env: int):
        """Log training progress"""
        # Get metrics from all environments
        metric_futures = [env.get_metrics.remote() for env in self.environments]
        env_metrics = ray.get(metric_futures)
        
        # Aggregate metrics
        total_conversions = sum(m['conversions'] for m in env_metrics)
        total_revenue = sum(m['revenue'] for m in env_metrics)
        total_reward = sum(m['total_reward'] for m in env_metrics)
        episodes_completed = sum(m['episodes'] for m in env_metrics)
        
        # Use the actual number of episodes for progress
        total_episodes = self.total_episodes
        
        self.training_metrics['total_conversions'] = total_conversions
        self.training_metrics['total_revenue'] = total_revenue
        
        # Calculate performance metrics
        if total_episodes > 0:
            avg_reward = total_reward / total_episodes
            self.training_metrics['avg_reward'] = avg_reward
        
        # Log channel performance
        if hasattr(self.agent, 'channel_performance'):
            logger.info("\nChannel Performance:")
            for channel, perf in self.agent.channel_performance.items():
                if perf.get('spend', 0) > 0:
                    roas = perf['revenue'] / perf['spend']
                    logger.info(f"  {channel}: ROAS={roas:.2f}x, "
                              f"Spend=${perf['spend']:.2f}, "
                              f"Revenue=${perf['revenue']:.2f}")
                    self.training_metrics['best_roas'] = max(
                        self.training_metrics['best_roas'], roas
                    )
        
        # Log creative performance
        if hasattr(self.agent, 'creative_performance'):
            top_creatives = sorted(
                self.agent.creative_performance.items(),
                key=lambda x: x[1].get('ctr', 0),
                reverse=True
            )[:5]
            
            if top_creatives:
                logger.info("\nTop 5 Creatives by CTR:")
                for creative_id, perf in top_creatives:
                    logger.info(f"  Creative {creative_id}: "
                              f"CTR={perf['ctr']*100:.2f}%, "
                              f"CVR={perf['cvr']*100:.2f}%")
        
        # Progress bar (avoid division by zero)
        progress = (episode / max(total_episodes, 1)) * 100
        logger.info(f"\nProgress: {progress:.1f}% [{episode}/{total_episodes}]")
        logger.info(f"Conversions: {total_conversions:,}")
        logger.info(f"Revenue: ${total_revenue:,.2f}")
        logger.info(f"Avg Reward: {self.training_metrics['avg_reward']:.2f}")
        logger.info(f"Best ROAS: {self.training_metrics['best_roas']:.2f}x")
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        if len(self.training_metrics['training_losses']) < 100:
            return False
        
        # Check if loss has plateaued
        recent_losses = self.training_metrics['training_losses'][-100:]
        bid_losses = [l['loss_bid'] for l in recent_losses]
        
        if len(bid_losses) >= 100:
            # Calculate variance of recent losses
            variance = np.var(bid_losses[-50:])
            if variance < 0.001 and self.agent.epsilon <= self.agent.epsilon_min:
                return True
        
        return False
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"fortified_agent_episode_{episode}.pt"
        )
        
        self.agent.save_model(checkpoint_path)
        
        # Save metrics
        metrics_path = os.path.join(
            checkpoint_dir,
            f"metrics_episode_{episode}.json"
        )
        
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_report(self):
        """Save final training report"""
        report = {
            'training_config': {
                'num_environments': self.num_environments,
                'total_episodes': self.total_episodes,
                'batch_size': self.batch_size
            },
            'final_metrics': self.training_metrics,
            'agent_performance': {
                'channel_performance': self.agent.channel_performance,
                'creative_performance': self.agent.creative_performance,
                'final_epsilon': self.agent.epsilon
            },
            'component_stats': {
                'discovered_segments': len(self.discovery.discover_all_patterns().user_patterns.get('segments', {})),
                'num_creatives': len(self.creative_selector.creatives),
                'attribution_models': ['linear', 'time_decay', 'position_based']
            }
        }
        
        with open('fortified_training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Final report saved: fortified_training_report.json")
    
    def cleanup(self):
        """Clean up resources"""
        ray.shutdown()
        logger.info("Training system cleaned up")

def main():
    """Main entry point"""
    # Configuration
    config = {
        'num_environments': 16,
        'total_episodes': 10000,
        'batch_size': 256,
        'checkpoint_interval': 100
    }
    
    # Create and run training system
    training_system = FortifiedTrainingSystem(**config)
    
    try:
        training_system.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
    finally:
        training_system.cleanup()

if __name__ == "__main__":
    main()