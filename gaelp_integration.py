#!/usr/bin/env python3
"""
GAELP Integration Module
Connects multi-touch journey tracking with existing GAELP components
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import wandb
import logging

# Import journey tracking components
from enhanced_journey_tracking import (
    EnhancedMultiTouchUser, 
    MultiTouchJourneySimulator,
    Channel, UserState, TouchpointType
)
from multi_channel_orchestrator import (
    MultiChannelOrchestrator,
    JourneyAwareRLEnvironment
)
from journey_aware_rl_agent import (
    JourneyAwarePPOAgent,
    JourneyState,
    extract_journey_state
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GAELPIntegration:
    """Main integration class for GAELP with journey tracking"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize GAELP integration"""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        logger.info("Initializing GAELP components...")
        
        # Journey simulator
        self.simulator = MultiTouchJourneySimulator(
            num_users=self.config.get('num_users', 1000),
            time_horizon_days=self.config.get('time_horizon_days', 30)
        )
        
        # Multi-channel orchestrator
        self.orchestrator = MultiChannelOrchestrator(
            budget_daily=self.config.get('budget_daily', 1000.0),
            learning_rate=self.config.get('orchestrator_lr', 0.01)
        )
        
        # RL Environment
        self.env = JourneyAwareRLEnvironment(
            self.simulator,
            self.orchestrator
        )
        
        # Journey-aware agent
        self.agent = JourneyAwarePPOAgent(
            learning_rate=self.config.get('agent_lr', 3e-4),
            gamma=self.config.get('gamma', 0.99)
        )
        
        # Initialize Weights & Biases
        if self.config.get('use_wandb', True):
            self._init_wandb()
        
        # Metrics tracking
        self.metrics = {
            'conversions': [],
            'costs': [],
            'revenues': [],
            'journey_lengths': [],
            'roas': [],
            'cac': []
        }
        
        logger.info("GAELP integration initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'num_users': 1000,
            'time_horizon_days': 30,
            'budget_daily': 1000.0,
            'orchestrator_lr': 0.01,
            'agent_lr': 3e-4,
            'gamma': 0.99,
            'use_wandb': True,
            'wandb_project': 'gaelp-journey-tracking',
            'wandb_entity': None,
            'training_episodes': 100,
            'eval_episodes': 20,
            'checkpoint_freq': 10,
            'batch_size': 32,
            'update_epochs': 4
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        try:
            # Try to initialize wandb
            wandb.init(
                project=self.config.get('wandb_project', 'gaelp-journey-tracking'),
                entity=self.config.get('wandb_entity'),
                config=self.config,
                tags=['multi-touch', 'journey-aware', 'ppo'],
                mode='offline' if not os.environ.get('WANDB_API_KEY') else 'online'
            )
            logger.info("Weights & Biases initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
            logger.info("Running in offline mode - metrics will be saved locally")
            # Set wandb to offline mode
            os.environ['WANDB_MODE'] = 'offline'
            wandb.init(
                project=self.config.get('wandb_project', 'gaelp-journey-tracking'),
                entity=self.config.get('wandb_entity'),
                config=self.config,
                tags=['multi-touch', 'journey-aware', 'ppo'],
                mode='offline'
            )
    
    def simulate_journeys(self) -> pd.DataFrame:
        """Simulate customer journeys"""
        logger.info("Simulating customer journeys...")
        
        # Run simulation
        touchpoints_df = self.simulator.simulate_journeys()
        
        # Get statistics
        stats = self.simulator.get_journey_statistics()
        
        logger.info(f"Simulated {len(touchpoints_df)} touchpoints")
        logger.info(f"Conversion rate: {stats['conversion_rate']:.2%}")
        logger.info(f"Average touches to conversion: {stats['avg_touches_to_conversion']:.1f}")
        
        # Log to wandb
        if self.config.get('use_wandb'):
            wandb.log({
                'simulation/total_users': stats['total_users'],
                'simulation/converted_users': stats['converted_users'],
                'simulation/conversion_rate': stats['conversion_rate'],
                'simulation/avg_touches': stats['avg_touches_to_conversion'],
                'simulation/avg_cac': stats['avg_cost_per_conversion'],
                'simulation/avg_ltv': stats['avg_ltv'],
                'simulation/roi': stats['roi']
            })
        
        return touchpoints_df
    
    def train_agent(self, episodes: Optional[int] = None):
        """Train the journey-aware RL agent"""
        episodes = episodes or self.config.get('training_episodes', 100)
        
        logger.info(f"Training agent for {episodes} episodes...")
        
        best_reward = -float('inf')
        
        for episode in range(episodes):
            # Reset environment
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_conversions = 0
            episode_cost = 0
            
            # Get initial state
            user = self.env.current_user
            state = extract_journey_state(user, self.orchestrator, self.env.simulator.current_date)
            
            while episode_length < 1000:
                # Select action
                channel_idx, bid_amount, log_prob = self.agent.select_action(state)
                
                # Create action array
                action = np.zeros(len(Channel))
                action[channel_idx] = bid_amount
                
                # Step environment
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                # Track metrics
                episode_reward += reward
                episode_length += 1
                episode_cost += info.get('cost', 0)
                if info.get('converted', False):
                    episode_conversions += 1
                
                # Get next state
                next_user = self.env.current_user
                next_state = extract_journey_state(
                    next_user, self.orchestrator, self.env.simulator.current_date
                )
                
                # Store transition
                self.agent.store_transition(
                    state, channel_idx, reward, next_state, done, log_prob
                )
                
                if done:
                    break
                
                state = next_state
            
            # Update agent
            if len(self.agent.memory) >= self.config.get('batch_size', 32):
                self.agent.update(
                    batch_size=self.config.get('batch_size', 32),
                    epochs=self.config.get('update_epochs', 4)
                )
            
            # Track metrics
            cac = episode_cost / max(episode_conversions, 1)
            self.metrics['conversions'].append(episode_conversions)
            self.metrics['costs'].append(episode_cost)
            self.metrics['cac'].append(cac)
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                          f"Conversions={episode_conversions}, CAC=${cac:.2f}")
            
            # Log to wandb
            if self.config.get('use_wandb'):
                wandb.log({
                    'train/episode': episode,
                    'train/reward': episode_reward,
                    'train/conversions': episode_conversions,
                    'train/cost': episode_cost,
                    'train/cac': cac,
                    'train/episode_length': episode_length
                })
            
            # Save checkpoint
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_checkpoint(f"best_model_ep{episode}.pth")
            
            if episode % self.config.get('checkpoint_freq', 10) == 0:
                self.save_checkpoint(f"checkpoint_ep{episode}.pth")
        
        logger.info(f"Training completed. Best reward: {best_reward:.2f}")
    
    def evaluate(self, episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the trained agent"""
        episodes = episodes or self.config.get('eval_episodes', 20)
        
        logger.info(f"Evaluating agent for {episodes} episodes...")
        
        total_reward = 0
        total_conversions = 0
        total_cost = 0
        total_revenue = 0
        journey_lengths = []
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            episode_length = 0
            
            user = self.env.current_user
            state = extract_journey_state(user, self.orchestrator, self.env.simulator.current_date)
            
            while episode_length < 1000:
                # Select action (no exploration during eval)
                with torch.no_grad():
                    channel_idx, bid_amount, _ = self.agent.select_action(state)
                
                action = np.zeros(len(Channel))
                action[channel_idx] = bid_amount
                
                obs, reward, done, truncated, info = self.env.step(action)
                
                total_reward += reward
                total_cost += info.get('cost', 0)
                episode_length += 1
                
                if info.get('converted', False):
                    total_conversions += 1
                    total_revenue += 120  # Aura LTV
                    journey_lengths.append(episode_length)
                
                if done:
                    break
                
                next_user = self.env.current_user
                state = extract_journey_state(
                    next_user, self.orchestrator, self.env.simulator.current_date
                )
        
        # Calculate metrics
        avg_reward = total_reward / episodes
        conversion_rate = total_conversions / episodes
        avg_cac = total_cost / max(total_conversions, 1)
        roas = total_revenue / max(total_cost, 1)
        avg_journey = np.mean(journey_lengths) if journey_lengths else 0
        
        results = {
            'avg_reward': avg_reward,
            'total_conversions': total_conversions,
            'conversion_rate': conversion_rate,
            'avg_cac': avg_cac,
            'roas': roas,
            'avg_journey_length': avg_journey
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Average Reward: {avg_reward:.2f}")
        logger.info(f"  Conversions: {total_conversions} ({conversion_rate:.2%} rate)")
        logger.info(f"  Average CAC: ${avg_cac:.2f}")
        logger.info(f"  ROAS: {roas:.2f}x")
        logger.info(f"  Avg Journey Length: {avg_journey:.1f} touches")
        
        # Log to wandb
        if self.config.get('use_wandb'):
            wandb.log({
                'eval/avg_reward': avg_reward,
                'eval/conversions': total_conversions,
                'eval/conversion_rate': conversion_rate,
                'eval/cac': avg_cac,
                'eval/roas': roas,
                'eval/avg_journey': avg_journey
            })
        
        return results
    
    def run_attribution_analysis(self, touchpoints_df: pd.DataFrame) -> Dict[str, Any]:
        """Run multi-touch attribution analysis"""
        logger.info("Running multi-touch attribution analysis...")
        
        # Group by user and calculate attribution
        user_journeys = touchpoints_df.groupby('user_id').agg({
            'channel': lambda x: list(x),
            'cost': 'sum',
            'conversion_probability': 'last',
            'ltv': 'last',
            'journey_length': 'last'
        })
        
        # Calculate channel attribution
        channel_attribution = {}
        
        for channel in Channel:
            channel_name = channel.value
            
            # First-touch attribution
            first_touch = sum(1 for channels in user_journeys['channel'] 
                            if channels and channels[0] == channel_name)
            
            # Last-touch attribution
            last_touch = sum(1 for channels in user_journeys['channel']
                           if channels and channels[-1] == channel_name)
            
            # Linear attribution
            linear = sum(channels.count(channel_name) / len(channels)
                        for channels in user_journeys['channel'] if channels)
            
            channel_attribution[channel_name] = {
                'first_touch': first_touch,
                'last_touch': last_touch,
                'linear': linear
            }
        
        # Calculate path analysis
        common_paths = {}
        for channels in user_journeys['channel']:
            if len(channels) >= 2:
                path = ' -> '.join(channels[:5])  # First 5 touches
                common_paths[path] = common_paths.get(path, 0) + 1
        
        # Sort paths by frequency
        top_paths = sorted(common_paths.items(), key=lambda x: x[1], reverse=True)[:10]
        
        attribution_results = {
            'channel_attribution': channel_attribution,
            'top_conversion_paths': top_paths,
            'avg_journey_length': user_journeys['journey_length'].mean(),
            'avg_cost_per_journey': user_journeys['cost'].mean()
        }
        
        logger.info("Attribution analysis completed")
        
        return attribution_results
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join('/home/hariravichandran/AELP/checkpoints', filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        self.agent.save(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join('/home/hariravichandran/AELP/checkpoints', filename)
        
        if os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("=" * 60)
        report.append("GAELP Multi-Touch Journey Tracking Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Configuration
        report.append("Configuration:")
        report.append(f"  Users: {self.config['num_users']}")
        report.append(f"  Time Horizon: {self.config['time_horizon_days']} days")
        report.append(f"  Daily Budget: ${self.config['budget_daily']}")
        report.append("")
        
        # Training metrics
        if self.metrics['conversions']:
            report.append("Training Metrics:")
            report.append(f"  Total Episodes: {len(self.metrics['conversions'])}")
            report.append(f"  Avg Conversions: {np.mean(self.metrics['conversions']):.2f}")
            report.append(f"  Avg CAC: ${np.mean(self.metrics['cac']):.2f}")
            report.append("")
        
        # Channel performance
        report.append("Channel Performance:")
        for channel, perf in self.orchestrator.channel_performance.items():
            if perf['spend'] > 0:
                report.append(f"  {channel.value}:")
                report.append(f"    Spend: ${perf['spend']:.2f}")
                report.append(f"    Conversions: {perf['conversions']}")
                report.append(f"    ROAS: {perf['roas']:.2f}x")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = '/home/hariravichandran/AELP/gaelp_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved: {report_path}")
        
        return report_text

def main():
    """Main integration pipeline"""
    print("🚀 GAELP Integration Pipeline")
    print("=" * 60)
    
    # Initialize integration
    gaelp = GAELPIntegration()
    
    # 1. Simulate journeys
    print("\n📊 Phase 1: Journey Simulation")
    touchpoints_df = gaelp.simulate_journeys()
    touchpoints_df.to_csv('/home/hariravichandran/AELP/gaelp_touchpoints.csv', index=False)
    
    # 2. Train agent
    print("\n🤖 Phase 2: Agent Training")
    gaelp.train_agent(episodes=50)
    
    # 3. Evaluate
    print("\n📈 Phase 3: Evaluation")
    eval_results = gaelp.evaluate(episodes=10)
    
    # 4. Attribution analysis
    print("\n🔍 Phase 4: Attribution Analysis")
    attribution = gaelp.run_attribution_analysis(touchpoints_df)
    
    # 5. Generate report
    print("\n📝 Phase 5: Report Generation")
    report = gaelp.generate_report()
    print(report)
    
    # Save final results
    results = {
        'evaluation': eval_results,
        'attribution': attribution,
        'config': gaelp.config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('/home/hariravichandran/AELP/gaelp_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ GAELP integration pipeline completed successfully!")
    print(f"   • Touchpoints saved: gaelp_touchpoints.csv")
    print(f"   • Results saved: gaelp_results.json")
    print(f"   • Report saved: gaelp_report.txt")
    print(f"   • Model checkpoints: checkpoints/")
    
    return gaelp

if __name__ == "__main__":
    gaelp = main()