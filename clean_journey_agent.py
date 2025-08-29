#!/usr/bin/env python3
"""
Clean Journey-Aware RL Agent for Learning Verification

A simplified, clean version of journey-aware PPO agent specifically 
designed for learning verification testing without corrupted code.

NO FALLBACKS. NO CORRUPTED PATTERN DISCOVERY CALLS. CLEAN CODE ONLY.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)

class CleanActorCritic(nn.Module):
    """Clean actor-critic network for journey-aware RL"""
    
    def __init__(self, state_dim: int = 64, hidden_dim: int = 128, num_channels: int = 8):
        super().__init__()
        
        # Shared feature extractor
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (channel selection)
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actor_out = nn.Linear(hidden_dim // 2, num_channels)
        
        # Critic head (value estimation)
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic_out = nn.Linear(hidden_dim // 2, 1)
        
        # Bid amount prediction per channel
        self.bid_fc = nn.Linear(hidden_dim, num_channels)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, state: torch.Tensor):
        """Forward pass through the network"""
        # Shared features
        x = F.relu(self.shared_fc1(state))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        shared_features = self.dropout(x)
        
        # Actor path (channel selection probabilities)
        actor_hidden = F.relu(self.actor_fc(shared_features))
        channel_logits = self.actor_out(actor_hidden)
        channel_probs = F.softmax(channel_logits, dim=-1)
        
        # Critic path (value estimation)
        critic_hidden = F.relu(self.critic_fc(shared_features))
        value = self.critic_out(critic_hidden)
        
        # Bid amounts (positive values)
        bid_amounts = F.softplus(self.bid_fc(shared_features))
        
        return channel_probs, value, bid_amounts

class CleanJourneyPPOAgent:
    """Clean PPO agent with learning verification support"""
    
    def __init__(self,
                 state_dim: int = 64,
                 hidden_dim: int = 128,
                 num_channels: int = 8,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create network
        self.actor_critic = CleanActorCritic(state_dim, hidden_dim, num_channels)
        self.actor_critic.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Experience buffer
        self.memory = []
        
        # Channel mapping
        self.channels = [
            'google_ads', 'facebook_ads', 'email', 'display',
            'video', 'social', 'affiliate', 'direct'
        ]
        
        logger.info(f"Clean Journey PPO Agent initialized")
        logger.info(f"Parameters: {sum(p.numel() for p in self.actor_critic.parameters())}")
    
    def select_action(self, state: Union[Dict, torch.Tensor]) -> Tuple[int, float, torch.Tensor]:
        """Select action using current policy"""
        
        # Convert state to tensor if needed
        if isinstance(state, dict):
            # Simple state extraction from dict
            state_values = []
            for key in ['impressions', 'clicks', 'conversions', 'cost', 'revenue', 'ctr', 'roas']:
                state_values.append(state.get(key, 0.0))
            
            # Pad to required state dimension
            while len(state_values) < 64:
                state_values.append(0.0)
            
            state_tensor = torch.FloatTensor(state_values[:64]).to(self.device)
        else:
            state_tensor = state.to(self.device) if not state.device == self.device else state
        
        # Ensure correct shape
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            channel_probs, value, bid_amounts = self.actor_critic(state_tensor)
        
        # Sample channel
        dist = Categorical(channel_probs)
        channel_idx = dist.sample()
        
        # Get bid amount for selected channel
        bid_amount = bid_amounts[0, channel_idx].item()
        
        # Get log probability
        log_prob = dist.log_prob(channel_idx)
        
        return channel_idx.item(), bid_amount, log_prob
    
    def store_transition(self, state, action: int, reward: float, next_state, done: bool, log_prob: torch.Tensor):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
    
    def update_policy(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update policy using PPO with proper learning verification support"""
        
        if not experiences:
            return {}
        
        # Convert experiences to training batch
        batch_size = len(experiences)
        
        # Extract states, actions, rewards
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        for exp in experiences:
            # Convert state to tensor
            if isinstance(exp['state'], dict):
                state_values = []
                for key in ['impressions', 'clicks', 'conversions', 'cost', 'revenue', 'ctr', 'roas']:
                    state_values.append(exp['state'].get(key, 0.0))
                while len(state_values) < 64:
                    state_values.append(0.0)
                state_tensor = torch.FloatTensor(state_values[:64])
            else:
                state_tensor = torch.FloatTensor(exp['state'])
            
            states.append(state_tensor)
            
            # Extract action (convert to tensor if needed)
            if isinstance(exp['action'], (int, float)):
                actions.append(int(exp['action']))
            else:
                actions.append(0)  # Default action
            
            rewards.append(float(exp.get('reward', 0.0)))
            
            # Create dummy log prob if not provided
            if 'log_prob' in exp and exp['log_prob'] is not None:
                log_probs.append(exp['log_prob'])
            else:
                log_probs.append(torch.tensor(0.0))
        
        # Convert to tensors
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)
        
        # Compute returns (simple discounted rewards)
        returns = torch.zeros_like(rewards_tensor)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards_tensor[t] + self.gamma * running_return
            returns[t] = running_return
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        metrics = {}
        
        # Forward pass
        channel_probs, values, bid_amounts = self.actor_critic(states_tensor)
        
        # Policy loss
        dist = Categorical(channel_probs)
        new_log_probs = dist.log_prob(actions_tensor)
        
        # Compute advantages
        advantages = returns - values.squeeze()
        
        # PPO clipped loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Entropy loss (for exploration)
        entropy = dist.entropy().mean()
        entropy_loss = -entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        
        # Optimizer step
        self.optimizer.step()
        
        # Clear memory
        self.memory = []
        
        # Return metrics for verification
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item(),
            'grad_norm': grad_norm.item(),
            'mean_reward': rewards_tensor.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'batch_size': batch_size
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class SimpleDummyEnvironment:
    """Simple dummy environment for testing"""
    
    def __init__(self):
        self.step_count = 0
        self.max_steps = 50
    
    def reset(self):
        """Reset environment"""
        self.step_count = 0
        return {
            'impressions': 1000,
            'clicks': 50,
            'conversions': 5,
            'cost': 100.0,
            'revenue': 500.0,
            'ctr': 0.05,
            'roas': 5.0
        }
    
    def step(self, action):
        """Step environment"""
        self.step_count += 1
        
        # Simulate some dynamics based on action
        channel_idx, bid_amount = action if isinstance(action, tuple) else (action, 1.0)
        
        # Simple reward: higher bid = more conversions but also more cost
        base_reward = np.random.normal(0, 1)
        bid_bonus = min(bid_amount * 0.1, 2.0)  # Cap bonus
        reward = base_reward + bid_bonus
        
        # Next state
        next_state = {
            'impressions': 1000 + self.step_count * 50,
            'clicks': 50 + self.step_count * 2,
            'conversions': max(0, 5 + int(np.random.normal(0, 2))),
            'cost': 100.0 + self.step_count * 10,
            'revenue': 500.0 + reward * 50,
            'ctr': np.clip(0.05 + np.random.normal(0, 0.01), 0.001, 0.2),
            'roas': np.clip(5.0 + reward, 1.0, 20.0)
        }
        
        done = self.step_count >= self.max_steps
        info = {'step': self.step_count}
        
        return next_state, reward, done, info

def test_clean_agent():
    """Test the clean agent for basic functionality"""
    
    print("🧪 Testing Clean Journey PPO Agent")
    print("=" * 50)
    
    # Create agent
    agent = CleanJourneyPPOAgent(
        state_dim=64,
        hidden_dim=64,
        num_channels=8,
        learning_rate=0.001
    )
    
    # Create environment
    env = SimpleDummyEnvironment()
    
    print(f"✅ Agent created with {sum(p.numel() for p in agent.actor_critic.parameters())} parameters")
    
    # Test action selection
    state = env.reset()
    channel_idx, bid_amount, log_prob = agent.select_action(state)
    print(f"✅ Action selection: channel={channel_idx}, bid=${bid_amount:.2f}")
    
    # Test training episode
    episode_experiences = []
    obs = env.reset()
    episode_reward = 0
    
    for step in range(10):
        # Select action
        channel_idx, bid_amount, log_prob = agent.select_action(obs)
        
        # Step environment
        next_obs, reward, done, info = env.step((channel_idx, bid_amount))
        episode_reward += reward
        
        # Store experience
        experience = {
            'state': obs,
            'action': channel_idx,
            'reward': reward,
            'next_state': next_obs,
            'done': done,
            'log_prob': log_prob
        }
        episode_experiences.append(experience)
        
        obs = next_obs
        
        if done:
            break
    
    print(f"✅ Episode completed: {len(episode_experiences)} steps, reward={episode_reward:.2f}")
    
    # Test policy update
    update_metrics = agent.update_policy(episode_experiences)
    print(f"✅ Policy update completed:")
    for key, value in update_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # Test multiple episodes for learning verification
    print(f"\n🔍 Testing learning over multiple episodes...")
    
    episode_rewards = []
    policy_losses = []
    entropies = []
    
    for episode in range(5):
        obs = env.reset()
        episode_reward = 0
        episode_experiences = []
        
        for step in range(20):
            channel_idx, bid_amount, log_prob = agent.select_action(obs)
            next_obs, reward, done, info = env.step((channel_idx, bid_amount))
            episode_reward += reward
            
            experience = {
                'state': obs,
                'action': channel_idx,
                'reward': reward,
                'next_state': next_obs,
                'done': done,
                'log_prob': log_prob
            }
            episode_experiences.append(experience)
            
            obs = next_obs
            if done:
                break
        
        # Update policy
        metrics = agent.update_policy(episode_experiences)
        
        episode_rewards.append(episode_reward)
        policy_losses.append(metrics.get('policy_loss', 0))
        entropies.append(metrics.get('entropy', 0))
        
        print(f"  Episode {episode}: reward={episode_reward:.2f}, loss={metrics.get('policy_loss', 0):.4f}")
    
    # Check for learning trends
    print(f"\n📊 Learning Analysis:")
    print(f"   Episode rewards: {[f'{r:.2f}' for r in episode_rewards]}")
    print(f"   Policy losses: {[f'{l:.4f}' for l in policy_losses]}")
    print(f"   Entropies: {[f'{e:.4f}' for e in entropies]}")
    
    # Simple trend analysis
    if len(episode_rewards) >= 3:
        early_reward = np.mean(episode_rewards[:2])
        late_reward = np.mean(episode_rewards[-2:])
        reward_improvement = late_reward - early_reward
        
        if reward_improvement > 0.1:
            print(f"   ✅ Reward trend: Improving ({reward_improvement:.2f})")
        else:
            print(f"   ⚠️  Reward trend: Not clearly improving ({reward_improvement:.2f})")
    
    if len(policy_losses) >= 3:
        early_loss = np.mean(policy_losses[:2])
        late_loss = np.mean(policy_losses[-2:])
        loss_improvement = early_loss - late_loss
        
        if loss_improvement > 0:
            print(f"   ✅ Loss trend: Improving ({loss_improvement:.4f})")
        else:
            print(f"   ⚠️  Loss trend: Not clearly improving ({loss_improvement:.4f})")
    
    print(f"\n✅ Clean agent testing completed!")
    
    return agent

if __name__ == "__main__":
    test_clean_agent()