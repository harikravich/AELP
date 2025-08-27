"""
Checkpoint manager for persistent learning across runs.
Enables the agent to remember what it learned.
"""

import os
import json
import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from google.cloud import storage
from dotenv import load_dotenv
import io

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CheckpointManager:
    """Manages model checkpoints and learning persistence."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Track learning history
        self.history_file = self.checkpoint_dir / "learning_history.json"
        self.strategies_file = self.checkpoint_dir / "discovered_strategies.json"
        self.latest_checkpoint = self.checkpoint_dir / "latest_checkpoint.pth"
        
        # Initialize GCS client if bucket is configured
        self.gcs_bucket = os.getenv('GCS_BUCKET')
        self.storage_client = None
        self.bucket = None
        
        if self.gcs_bucket:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(self.gcs_bucket)
                logger.info(f"Connected to GCS bucket: {self.gcs_bucket}")
            except Exception as e:
                logger.warning(f"Could not connect to GCS: {e}. Using local storage only.")
        
    def save_checkpoint(self, agent, episode: int, metrics: Dict[str, Any]) -> str:
        """Save model checkpoint with learning state."""
        
        checkpoint_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'model_state_dict': agent.actor.state_dict() if hasattr(agent, 'actor') else {},
            'critic_state_dict': agent.critic.state_dict() if hasattr(agent, 'critic') else {},
            'optimizer_state_dict': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else {},
            'metrics': metrics,
            'training_step': getattr(agent, 'training_step', 0),
            'exploration_rate': getattr(agent, 'exploration_rate', 1.0),
        }
        
        # Save locally first
        episode_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pth"
        torch.save(checkpoint_data, episode_path)
        torch.save(checkpoint_data, self.latest_checkpoint)
        
        # Save to GCS if available
        if self.bucket:
            try:
                # Save checkpoint to GCS
                buffer = io.BytesIO()
                torch.save(checkpoint_data, buffer)
                buffer.seek(0)
                
                blob_name = f"checkpoints/checkpoint_episode_{episode}.pth"
                blob = self.bucket.blob(blob_name)
                blob.upload_from_file(buffer)
                
                # Also save as latest
                buffer.seek(0)
                latest_blob = self.bucket.blob("checkpoints/latest_checkpoint.pth")
                latest_blob.upload_from_file(buffer)
                
                logger.info(f"Saved checkpoint to GCS at episode {episode}")
            except Exception as e:
                logger.error(f"Failed to save to GCS: {e}")
        
        logger.info(f"Saved checkpoint locally at episode {episode}")
        
        # Update learning history
        self._update_learning_history(episode, metrics)
        
        return str(episode_path)
    
    def load_latest_checkpoint(self, agent) -> Optional[int]:
        """Load the most recent checkpoint if it exists."""
        
        checkpoint = None
        
        # Try loading from GCS first
        if self.bucket:
            try:
                blob = self.bucket.blob("checkpoints/latest_checkpoint.pth")
                if blob.exists():
                    buffer = io.BytesIO()
                    blob.download_to_file(buffer)
                    buffer.seek(0)
                    checkpoint = torch.load(buffer, map_location='cpu')
                    logger.info("Loaded checkpoint from GCS")
            except Exception as e:
                logger.warning(f"Could not load from GCS: {e}")
        
        # Fall back to local if GCS didn't work
        if checkpoint is None and self.latest_checkpoint.exists():
            try:
                checkpoint = torch.load(self.latest_checkpoint, map_location='cpu')
                logger.info("Loaded checkpoint from local storage")
            except Exception as e:
                logger.error(f"Failed to load local checkpoint: {e}")
        
        if checkpoint is None:
            logger.info("No previous checkpoint found. Starting fresh.")
            return None
            
        try:
            # Restore model states
            if hasattr(agent, 'actor') and 'model_state_dict' in checkpoint:
                agent.actor.load_state_dict(checkpoint['model_state_dict'])
            
            if hasattr(agent, 'critic') and 'critic_state_dict' in checkpoint:
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            if hasattr(agent, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            if hasattr(agent, 'training_step'):
                agent.training_step = checkpoint.get('training_step', 0)
            
            if hasattr(agent, 'exploration_rate'):
                agent.exploration_rate = checkpoint.get('exploration_rate', 1.0)
            
            episode = checkpoint.get('episode', 0)
            
            logger.info(f"Loaded checkpoint from episode {episode}")
            logger.info(f"Resuming with training_step={checkpoint.get('training_step', 0)}")
            
            # Load discovered strategies
            self._load_discovered_strategies(agent)
            
            return episode
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None
    
    def _update_learning_history(self, episode: int, metrics: Dict[str, Any]):
        """Update the learning history file."""
        
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        # Keep last 1000 episodes
        history = history[-1000:]
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def save_discovered_strategy(self, strategy: Dict[str, Any]):
        """Save a newly discovered high-performing strategy."""
        
        strategies = []
        if self.strategies_file.exists():
            with open(self.strategies_file, 'r') as f:
                strategies = json.load(f)
        
        # Add new strategy if it's good enough
        if strategy.get('roas', 0) > 3.0:  # Only save strategies with 3x+ ROAS
            strategies.append({
                **strategy,
                'discovered_at': datetime.now().isoformat()
            })
            
            # Keep best 50 strategies
            strategies = sorted(strategies, key=lambda x: x.get('roas', 0), reverse=True)[:50]
            
            with open(self.strategies_file, 'w') as f:
                json.dump(strategies, f, indent=2)
            
            logger.info(f"Saved new strategy with {strategy.get('roas', 0):.2f}x ROAS")
    
    def _load_discovered_strategies(self, agent):
        """Load previously discovered strategies into agent memory."""
        
        if not self.strategies_file.exists():
            return
        
        try:
            with open(self.strategies_file, 'r') as f:
                strategies = json.load(f)
            
            # Store strategies in agent for exploitation
            if hasattr(agent, 'known_strategies'):
                agent.known_strategies = strategies
                logger.info(f"Loaded {len(strategies)} known strategies")
                
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learning so far."""
        
        if not self.history_file.exists():
            return {'total_episodes': 0, 'best_roas': 0, 'avg_roas': 0}
        
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        
        if not history:
            return {'total_episodes': 0, 'best_roas': 0, 'avg_roas': 0}
        
        all_roas = [h['metrics'].get('roas', 0) for h in history]
        
        return {
            'total_episodes': len(history),
            'best_roas': max(all_roas),
            'avg_roas': sum(all_roas) / len(all_roas),
            'latest_episode': history[-1]['episode'],
            'total_training_time': len(history) * 10  # Approximate minutes
        }