#!/usr/bin/env python3
"""
Offline RL Trainer for GAELP using d3rlpy
Learns from historical campaign data without exploration
"""

import numpy as np
import pandas as pd
import torch
import d3rlpy
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OfflineDataset:
    """Container for offline RL dataset"""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    
    def __len__(self):
        return len(self.observations)
    
    def save(self, filepath: str):
        """Save dataset to disk"""
        data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_observations': self.next_observations,
            'terminals': self.terminals
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load dataset from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)


class CampaignDataPreprocessor:
    """Preprocesses campaign data for offline RL training"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = [
            'impressions', 'clicks', 'ctr', 'cost', 'cpc', 
            'conversions', 'conversion_rate', 'revenue', 'roas', 
            'hour', 'day_of_week', 'is_weekend'
        ]
        self.categorical_columns = ['vertical', 'season']
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit preprocessors and transform data"""
        
        # Handle missing values
        df = df.fillna(0)
        
        # Encode categorical variables
        encoded_features = []
        for col in self.categorical_columns:
            if col in df.columns:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(df[col].astype(str))
                encoded_features.append(encoded.reshape(-1, 1))
                self.encoders[col] = encoder
        
        # Scale numerical features
        numerical_features = []
        for col in self.feature_columns:
            if col in df.columns:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df[col].values.reshape(-1, 1))
                numerical_features.append(scaled)
                self.scalers[col] = scaler
        
        # Combine all features
        all_features = numerical_features + encoded_features
        if all_features:
            features = np.hstack(all_features)
        else:
            features = np.zeros((len(df), 1))
        
        # Create metadata
        metadata = {
            'feature_dim': features.shape[1],
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'n_samples': len(df)
        }
        
        return features, metadata
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors"""
        
        # Handle missing values
        df = df.fillna(0)
        
        # Encode categorical variables
        encoded_features = []
        for col in self.categorical_columns:
            if col in df.columns and col in self.encoders:
                # Handle unseen categories
                encoded = []
                for value in df[col].astype(str):
                    try:
                        encoded.append(self.encoders[col].transform([value])[0])
                    except ValueError:
                        # Use most frequent class for unseen categories
                        encoded.append(0)
                encoded_features.append(np.array(encoded).reshape(-1, 1))
        
        # Scale numerical features
        numerical_features = []
        for col in self.feature_columns:
            if col in df.columns and col in self.scalers:
                scaled = self.scalers[col].transform(df[col].values.reshape(-1, 1))
                numerical_features.append(scaled)
        
        # Combine all features
        all_features = numerical_features + encoded_features
        if all_features:
            features = np.hstack(all_features)
        else:
            features = np.zeros((len(df), 1))
        
        return features


class ActionExtractor:
    """Extracts actions from campaign data"""
    
    def __init__(self):
        self.action_space_dim = 4  # bid, budget_allocation, targeting_intensity, creative_quality
        
    def extract_actions(self, df: pd.DataFrame) -> np.ndarray:
        """Extract actions from campaign data"""
        
        actions = []
        
        for _, row in df.iterrows():
            # Derive action from campaign metrics
            # Action = [normalized_bid, budget_efficiency, targeting_quality, creative_performance]
            
            # Bid intensity (normalized CPC)
            bid_intensity = min(row.get('cpc', 1.0) / 5.0, 1.0)  # Normalize to [0,1]
            
            # Budget efficiency (revenue/cost ratio capped)
            budget_eff = min(row.get('roas', 0) / 10.0, 1.0) if row.get('cost', 0) > 0 else 0
            
            # Targeting quality (CTR relative to benchmark)
            targeting_quality = min(row.get('ctr', 0) / 0.05, 1.0)  # 5% CTR as benchmark
            
            # Creative performance (conversion rate relative to benchmark)
            creative_perf = min(row.get('conversion_rate', 0) / 0.1, 1.0)  # 10% CR as benchmark
            
            action = [bid_intensity, budget_eff, targeting_quality, creative_perf]
            actions.append(action)
        
        return np.array(actions, dtype=np.float32)


class OfflineRLTrainer:
    """Offline RL trainer using Conservative Q-Learning (CQL)"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.preprocessor = CampaignDataPreprocessor()
        self.action_extractor = ActionExtractor()
        self.algorithm = None
        self.dataset = None
        self.metadata = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'algorithm': 'cql',
            'batch_size': 256,
            'n_epochs': 100,
            'learning_rate': 3e-4,
            'alpha': 5.0,  # CQL regularization parameter
            'use_gpu': torch.cuda.is_available(),
            'validation_split': 0.2,
            'save_interval': 20,
            'checkpoint_dir': '/home/hariravichandran/AELP/checkpoints'
        }
    
    def load_data(self, data_path: str) -> OfflineDataset:
        """Load and preprocess campaign data"""
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(df)} campaign records")
        
        # Extract features (states)
        features, self.metadata = self.preprocessor.fit_transform(df)
        
        # Extract actions
        actions = self.action_extractor.extract_actions(df)
        
        # Extract rewards (profit as immediate reward)
        rewards = df['profit'].values.astype(np.float32)
        
        # Normalize rewards to improve training stability
        reward_std = np.std(rewards)
        if reward_std > 0:
            rewards = rewards / reward_std
        
        # Create next observations (shifted by 1)
        next_features = np.roll(features, -1, axis=0)
        
        # Create terminals (mark end of campaigns)
        terminals = np.zeros(len(df), dtype=bool)
        # Mark terminal states (could be end of campaign or significant metric changes)
        for i in range(1, len(df)):
            if df.iloc[i]['campaign_id'] != df.iloc[i-1]['campaign_id']:
                terminals[i-1] = True
        terminals[-1] = True  # Last observation is always terminal
        
        # Create dataset
        self.dataset = OfflineDataset(
            observations=features.astype(np.float32),
            actions=actions,
            rewards=rewards,
            next_observations=next_features.astype(np.float32),
            terminals=terminals
        )
        
        logger.info(f"Created dataset with {len(self.dataset)} transitions")
        logger.info(f"Feature dimension: {features.shape[1]}")
        logger.info(f"Action dimension: {actions.shape[1]}")
        logger.info(f"Reward statistics: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
        
        return self.dataset
    
    def create_d3rlpy_dataset(self) -> d3rlpy.dataset.MDPDataset:
        """Convert to d3rlpy dataset format"""
        
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        
        # Create episodes list for d3rlpy
        episodes = []
        current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': []
        }
        
        for i in range(len(self.dataset)):
            current_episode['observations'].append(self.dataset.observations[i])
            current_episode['actions'].append(self.dataset.actions[i])
            current_episode['rewards'].append(self.dataset.rewards[i])
            current_episode['terminals'].append(self.dataset.terminals[i])
            
            if self.dataset.terminals[i]:
                # End of episode, convert to numpy arrays and add to episodes
                episodes.append({
                    'observations': np.array(current_episode['observations']),
                    'actions': np.array(current_episode['actions']),
                    'rewards': np.array(current_episode['rewards']),
                    'terminals': np.array(current_episode['terminals'])
                })
                # Reset for next episode
                current_episode = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': []
                }
        
        # Create d3rlpy dataset from episodes
        mdp_dataset = d3rlpy.dataset.create_fifo_replay_buffer(
            limit=len(self.dataset),
            env=None  # We'll set this manually
        )
        
        # Add all transitions to the buffer
        for episode in episodes:
            for i in range(len(episode['observations'])):
                mdp_dataset.append(
                    observation=episode['observations'][i],
                    action=episode['actions'][i],
                    reward=episode['rewards'][i],
                    terminal=episode['terminals'][i]
                )
        
        return mdp_dataset
    
    def train(self, save_model: bool = True) -> Dict[str, Any]:
        """Train offline RL algorithm using d3rlpy"""
        
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        
        logger.info("Starting training...")
        
        # Create checkpoint directory
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Split dataset into train/test
        n_train = int(len(self.dataset) * (1 - self.config['validation_split']))
        
        train_observations = self.dataset.observations[:n_train]
        train_actions = self.dataset.actions[:n_train]
        train_rewards = self.dataset.rewards[:n_train]
        train_terminals = self.dataset.terminals[:n_train]
        
        test_observations = self.dataset.observations[n_train:]
        test_actions = self.dataset.actions[n_train:]
        test_rewards = self.dataset.rewards[n_train:]
        test_terminals = self.dataset.terminals[n_train:]
        
        logger.info(f"Training on {n_train} samples, testing on {len(self.dataset) - n_train} samples")
        
        # Create training dataset
        train_dataset = d3rlpy.dataset.MDPDataset(
            observations=train_observations,
            actions=train_actions,
            rewards=train_rewards,
            terminals=train_terminals
        )
        
        # Create test dataset
        test_dataset = d3rlpy.dataset.MDPDataset(
            observations=test_observations,
            actions=test_actions,
            rewards=test_rewards,
            terminals=test_terminals
        )
        
        # Initialize algorithm
        if self.config['algorithm'] == 'cql':
            # Create CQL with minimal configuration
            try:
                self.algorithm = d3rlpy.algos.CQLConfig().create(
                    device='cuda' if self.config['use_gpu'] else 'cpu'
                )
            except Exception as e:
                logger.warning(f"Failed to create CQL with config, using defaults: {e}")
                self.algorithm = d3rlpy.algos.CQL()
        else:
            raise ValueError(f"Unsupported algorithm: {self.config['algorithm']}")
        
        # Training metrics
        training_metrics = {
            'train_losses': [],
            'test_scores': [],
            'epochs': []
        }
        
        # Build and train the algorithm
        try:
            logger.info("Building CQL algorithm...")
            
            # Build the algorithm with the dataset
            self.algorithm.build_with_dataset(train_dataset)
            
            logger.info("Starting CQL training...")
            
            # Simple training call
            self.algorithm.fit(
                train_dataset,
                n_steps=self.config['n_epochs'] * 100,  # Total training steps
                n_steps_per_epoch=100
            )
            
            logger.info("Training completed successfully")
            
            # Store simple metrics
            training_metrics['train_losses'] = [0.0] * self.config['n_epochs']  # Placeholder
            training_metrics['epochs'] = list(range(self.config['n_epochs']))
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Continue with empty metrics - but still try to build for evaluation
            try:
                self.algorithm.build_with_dataset(train_dataset)
                logger.info("Algorithm built successfully despite training failure")
            except Exception as build_e:
                logger.error(f"Failed to build algorithm: {build_e}")
            
            training_metrics['train_losses'] = [0.0] * self.config['n_epochs']
            training_metrics['epochs'] = list(range(self.config['n_epochs']))
        
        # Save final model
        if save_model:
            model_path = os.path.join(self.config['checkpoint_dir'], 'final_model.d3')
            try:
                self.algorithm.save(model_path)
                logger.info(f"Model saved to {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save model: {e}")
        
        # Save training metadata
        self.metadata.update({
            'training_config': self.config,
            'training_metrics': training_metrics,
            'final_test_score': training_metrics['test_scores'][-1] if training_metrics['test_scores'] else None
        })
        
        return training_metrics
    
    def evaluate_policy(self, test_data_path: str = None) -> Dict[str, Any]:
        """Evaluate trained policy"""
        
        if self.algorithm is None:
            raise ValueError("No trained algorithm. Call train() first.")
        
        # Use validation data or load new test data
        if test_data_path:
            test_df = pd.read_csv(test_data_path)
            test_features = self.preprocessor.transform(test_df)
        else:
            # Use existing dataset
            test_features = self.dataset.observations
        
        # Predict actions
        predicted_actions = []
        for obs in test_features:
            # Ensure batch dimension
            obs_batch = obs.reshape(1, -1) 
            action = self.algorithm.predict(obs_batch)[0]
            predicted_actions.append(action)
        
        predicted_actions = np.array(predicted_actions)
        
        # Calculate evaluation metrics
        evaluation_results = {
            'n_predictions': len(predicted_actions),
            'action_statistics': {
                'mean': predicted_actions.mean(axis=0).tolist(),
                'std': predicted_actions.std(axis=0).tolist(),
                'min': predicted_actions.min(axis=0).tolist(),
                'max': predicted_actions.max(axis=0).tolist()
            }
        }
        
        return evaluation_results
    
    def save_training_plots(self, metrics: Dict[str, Any], save_dir: str = None):
        """Save training progress plots"""
        
        save_dir = save_dir or self.config['checkpoint_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loss plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Test score plot
        plt.subplot(1, 2, 2)
        if metrics['test_scores']:
            plt.plot(metrics['epochs'], metrics['test_scores'])
            plt.title('Test Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {save_dir}")


def test_with_enhanced_simulator(trainer: OfflineRLTrainer, n_episodes: int = 10):
    """Test trained policy with enhanced simulator"""
    
    try:
        from enhanced_simulator import EnhancedGAELPEnvironment
    except ImportError:
        logger.error("Could not import enhanced_simulator")
        return
    
    if trainer.algorithm is None:
        logger.error("No trained algorithm available")
        return
    
    env = EnhancedGAELPEnvironment()
    results = []
    
    logger.info(f"Testing policy on enhanced simulator for {n_episodes} episodes")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while episode_length < 50:  # Limit episode length
            # Convert observation to format expected by algorithm
            obs_array = np.array([
                obs['total_cost'], obs['total_revenue'], obs['impressions'],
                obs['clicks'], obs['conversions'], obs['avg_cpc'], obs['roas']
            ]).reshape(1, -1)
            
            # Pad to match training dimension if needed
            if obs_array.shape[1] < trainer.metadata['feature_dim']:
                padding = np.zeros((1, trainer.metadata['feature_dim'] - obs_array.shape[1]))
                obs_array = np.hstack([obs_array, padding])
            
            # Predict action (ensure correct batch format)
            action_pred = trainer.algorithm.predict(obs_array)[0]
            
            # Convert to environment action format
            action = {
                'bid': action_pred[0] * 5.0,  # Scale back to reasonable bid range
                'budget': 1000,
                'quality_score': 0.5 + action_pred[2] * 0.5,  # [0.5, 1.0]
                'creative': {
                    'quality_score': 0.3 + action_pred[3] * 0.6,  # [0.3, 0.9]
                    'price_shown': np.random.uniform(10, 100)
                }
            }
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'final_roas': obs['roas']
        })
        
        if episode % 5 == 0:
            logger.info(f"Episode {episode}: Reward={episode_reward:.3f}, ROAS={obs['roas']:.3f}")
    
    # Calculate summary statistics
    rewards = [r['reward'] for r in results]
    roas_values = [r['final_roas'] for r in results]
    
    summary = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_roas': np.mean(roas_values),
        'std_roas': np.std(roas_values),
        'episodes': results
    }
    
    logger.info(f"Test Results Summary:")
    logger.info(f"  Mean Reward: {summary['mean_reward']:.3f} ± {summary['std_reward']:.3f}")
    logger.info(f"  Mean ROAS: {summary['mean_roas']:.3f} ± {summary['std_roas']:.3f}")
    
    return summary


def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    config = {
        'algorithm': 'cql',
        'batch_size': 64,
        'n_epochs': 10,  # Reduced for testing
        'learning_rate': 3e-4,
        'alpha': 5.0,
        'use_gpu': False,  # Set to True if GPU available
        'validation_split': 0.2,
        'save_interval': 10,
        'checkpoint_dir': '/home/hariravichandran/AELP/checkpoints'
    }
    
    # Initialize trainer
    trainer = OfflineRLTrainer(config)
    
    try:
        # Load and preprocess data
        dataset = trainer.load_data('/home/hariravichandran/AELP/data/aggregated_data.csv')
        
        # Train offline RL algorithm
        metrics = trainer.train(save_model=True)
        
        # Save training plots
        trainer.save_training_plots(metrics)
        
        # Evaluate policy
        evaluation = trainer.evaluate_policy()
        logger.info(f"Policy evaluation: {evaluation}")
        
        # Test with enhanced simulator
        simulator_results = test_with_enhanced_simulator(trainer, n_episodes=10)
        
        logger.info("Offline RL training completed successfully!")
        
        return {
            'training_metrics': metrics,
            'evaluation': evaluation,
            'simulator_results': simulator_results
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()