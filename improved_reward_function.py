#!/usr/bin/env python3
"""
Improved Reward Function for GAELP that balances:
1. Volume (impressions, clicks, conversions)
2. CAC (Customer Acquisition Cost)
3. ROAS (Return on Ad Spend)
4. Market Share
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ImprovedRewardCalculator:
    """
    Reward function that incentivizes both high volume and low CAC
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Weight configuration for different objectives
        self.weights = {
            'volume': config.get('volume_weight', 0.3),      # NEW: Volume incentive
            'cac': config.get('cac_weight', 0.3),            # NEW: CAC efficiency
            'roas': config.get('roas_weight', 0.2),          # Existing: Revenue/Cost
            'market_share': config.get('market_share_weight', 0.1),  # NEW: Win rate
            'exploration': config.get('exploration_weight', 0.05),
            'diversity': config.get('diversity_weight', 0.05)
        }
        
        # Target metrics for normalization
        self.targets = {
            'daily_impressions': config.get('target_impressions', 10000),
            'daily_clicks': config.get('target_clicks', 300),
            'daily_conversions': config.get('target_conversions', 30),
            'target_cac': config.get('target_cac', 50.0),  # Target $50 CAC
            'max_cac': config.get('max_cac', 150.0),       # Max acceptable $150 CAC
            'target_roas': config.get('target_roas', 3.0),
            'target_win_rate': config.get('target_win_rate', 0.3)
        }
        
        # Volume scaling factors
        self.volume_scales = {
            'impression_value': 0.01,  # Each impression worth $0.01 in reward
            'click_value': 0.50,       # Each click worth $0.50 in reward
            'conversion_value': 20.0   # Each conversion worth $20 in reward
        }
        
        # Episode tracking
        self.episode_metrics = {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spend': 0.0,
            'revenue': 0.0,
            'auctions_entered': 0,
            'auctions_won': 0
        }
    
    def calculate_reward(self, context: Dict, tracker=None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective reward balancing volume and CAC
        
        Args:
            context: Dictionary containing:
                - won: bool, whether auction was won
                - cost: float, amount paid for impression
                - click_occurred: bool
                - conversion_occurred: bool
                - conversion_value: float
                - impressions_today: int (optional)
                - conversions_today: int (optional)
                - spend_today: float (optional)
        """
        components = {}
        
        # Update episode metrics
        self._update_episode_metrics(context)
        
        # 1. VOLUME COMPONENT (30% weight) - Incentivize scale
        components['volume'] = self._calculate_volume_reward(context)
        
        # 2. CAC COMPONENT (30% weight) - Incentivize efficiency
        components['cac'] = self._calculate_cac_reward(context)
        
        # 3. ROAS COMPONENT (20% weight) - Profitability
        components['roas'] = self._calculate_roas_reward(context)
        
        # 4. MARKET SHARE COMPONENT (10% weight) - Win rate
        components['market_share'] = self._calculate_market_share_reward(context)
        
        # 5. EXPLORATION COMPONENT (5% weight)
        components['exploration'] = self._calculate_exploration_reward(context, tracker)
        
        # 6. DIVERSITY COMPONENT (5% weight)
        components['diversity'] = self._calculate_diversity_reward(tracker)
        
        # Calculate weighted total
        total_reward = sum(self.weights[key] * components[key] for key in components.keys())
        
        # Add bonus for achieving both volume AND efficiency
        if components['volume'] > 0.5 and components['cac'] > 0.5:
            synergy_bonus = 0.2  # 20% bonus for achieving both
            total_reward *= (1 + synergy_bonus)
            components['synergy_bonus'] = synergy_bonus
        
        # Log detailed breakdown periodically
        if context.get('step', 0) % 100 == 0:
            logger.info(f"Reward breakdown at step {context.get('step', 0)}:")
            logger.info(f"  Volume: {components['volume']:.3f} (weight: {self.weights['volume']})")
            logger.info(f"  CAC: {components['cac']:.3f} (weight: {self.weights['cac']})")
            logger.info(f"  ROAS: {components['roas']:.3f} (weight: {self.weights['roas']})")
            logger.info(f"  Market Share: {components['market_share']:.3f} (weight: {self.weights['market_share']})")
            logger.info(f"  Total Reward: {total_reward:.3f}")
            logger.info(f"  Episode Stats - Impressions: {self.episode_metrics['impressions']}, "
                       f"Conversions: {self.episode_metrics['conversions']}, "
                       f"CAC: ${self._get_current_cac():.2f}")
        
        return total_reward, components
    
    def _update_episode_metrics(self, context: Dict):
        """Update running metrics for the episode"""
        if context.get('won', False):
            self.episode_metrics['auctions_won'] += 1
            self.episode_metrics['impressions'] += 1
            self.episode_metrics['spend'] += context.get('cost', 0)
            
            if context.get('click_occurred', False):
                self.episode_metrics['clicks'] += 1
            
            if context.get('conversion_occurred', False):
                self.episode_metrics['conversions'] += 1
                self.episode_metrics['revenue'] += context.get('conversion_value', 0)
        
        self.episode_metrics['auctions_entered'] += 1
    
    def _calculate_volume_reward(self, context: Dict) -> float:
        """
        Reward for achieving volume targets
        Returns value between -1 and 1
        """
        volume_score = 0.0
        
        # Immediate volume reward for this step
        if context.get('won', False):
            # Impression reward
            volume_score += self.volume_scales['impression_value']
            
            if context.get('click_occurred', False):
                # Click reward
                volume_score += self.volume_scales['click_value']
            
            if context.get('conversion_occurred', False):
                # Conversion reward (biggest volume reward)
                volume_score += self.volume_scales['conversion_value']
        
        # Normalize to [-1, 1] range
        # Max possible per step: 0.01 + 0.50 + 20.0 = 20.51
        normalized_score = np.tanh(volume_score / 10.0)  # Sigmoid-like normalization
        
        # Add progress bonus based on daily targets
        daily_impressions = context.get('impressions_today', self.episode_metrics['impressions'])
        daily_conversions = context.get('conversions_today', self.episode_metrics['conversions'])
        
        impression_progress = min(1.0, daily_impressions / self.targets['daily_impressions'])
        conversion_progress = min(1.0, daily_conversions / self.targets['daily_conversions'])
        
        progress_bonus = (impression_progress + conversion_progress) / 2
        
        return normalized_score * 0.7 + progress_bonus * 0.3
    
    def _calculate_cac_reward(self, context: Dict) -> float:
        """
        Reward for maintaining low CAC
        Returns value between -1 and 1
        """
        current_cac = self._get_current_cac()
        
        if current_cac <= 0:
            # No conversions yet, neutral reward
            return 0.0
        
        # Calculate CAC efficiency score
        if current_cac <= self.targets['target_cac']:
            # Below target CAC - excellent!
            cac_score = 1.0
        elif current_cac <= self.targets['max_cac']:
            # Between target and max - linear decay
            ratio = (current_cac - self.targets['target_cac']) / \
                   (self.targets['max_cac'] - self.targets['target_cac'])
            cac_score = 1.0 - ratio  # Linear from 1.0 to 0.0
        else:
            # Above max CAC - negative reward
            excess_ratio = (current_cac - self.targets['max_cac']) / self.targets['max_cac']
            cac_score = -np.tanh(excess_ratio)  # Negative, capped at -1
        
        # Add marginal CAC consideration (reward for improving CAC)
        if context.get('conversion_occurred', False):
            marginal_cac = context.get('cost', 0) / 1  # Cost for this single conversion
            if marginal_cac < current_cac:
                # This conversion improved our CAC!
                improvement_bonus = 0.2
                cac_score = min(1.0, cac_score + improvement_bonus)
        
        return cac_score
    
    def _calculate_roas_reward(self, context: Dict) -> float:
        """
        Reward for maintaining good ROAS
        Returns value between -1 and 1
        """
        revenue = context.get('revenue', 0.0)
        cost = context.get('cost', 0.01)  # Avoid division by zero
        
        if cost <= 0:
            return 0.0
        
        immediate_roas = revenue / cost
        
        # Calculate episode ROAS
        if self.episode_metrics['spend'] > 0:
            episode_roas = self.episode_metrics['revenue'] / self.episode_metrics['spend']
        else:
            episode_roas = 0.0
        
        # Combine immediate and episode ROAS
        combined_roas = 0.3 * immediate_roas + 0.7 * episode_roas
        
        # Normalize based on target
        if combined_roas >= self.targets['target_roas']:
            roas_score = 1.0
        elif combined_roas > 0:
            roas_score = combined_roas / self.targets['target_roas']
        else:
            roas_score = -0.5  # Negative but not too harsh
        
        return np.tanh(roas_score)  # Smooth normalization
    
    def _calculate_market_share_reward(self, context: Dict) -> float:
        """
        Reward for winning auctions (market share)
        Returns value between -1 and 1
        """
        if self.episode_metrics['auctions_entered'] == 0:
            return 0.0
        
        win_rate = self.episode_metrics['auctions_won'] / self.episode_metrics['auctions_entered']
        
        # Compare to target win rate
        if win_rate >= self.targets['target_win_rate']:
            share_score = 1.0
        else:
            share_score = win_rate / self.targets['target_win_rate']
        
        # Penalize very low win rates
        if win_rate < 0.05:
            share_score = -0.5
        
        return share_score
    
    def _calculate_exploration_reward(self, context: Dict, tracker) -> float:
        """Reward for exploring new strategies"""
        if tracker and hasattr(tracker, 'get_exploration_score'):
            return tracker.get_exploration_score()
        return 0.0
    
    def _calculate_diversity_reward(self, tracker) -> float:
        """Reward for diverse actions"""
        if tracker and hasattr(tracker, 'get_diversity_score'):
            return tracker.get_diversity_score()
        return 0.0
    
    def _get_current_cac(self) -> float:
        """Calculate current Customer Acquisition Cost"""
        if self.episode_metrics['conversions'] == 0:
            return 0.0
        return self.episode_metrics['spend'] / self.episode_metrics['conversions']
    
    def reset_episode_metrics(self):
        """Reset metrics at the start of a new episode"""
        self.episode_metrics = {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spend': 0.0,
            'revenue': 0.0,
            'auctions_entered': 0,
            'auctions_won': 0
        }


# Example usage and testing
def test_reward_function():
    """Test the improved reward function with various scenarios"""
    
    calculator = ImprovedRewardCalculator()
    
    print("Testing Improved Reward Function")
    print("=" * 50)
    
    # Scenario 1: High volume, good CAC
    context1 = {
        'won': True,
        'cost': 2.0,
        'click_occurred': True,
        'conversion_occurred': True,
        'conversion_value': 100.0,
        'step': 100
    }
    reward1, components1 = calculator.calculate_reward(context1)
    print(f"\nScenario 1 - High volume, good CAC:")
    print(f"  Total Reward: {reward1:.3f}")
    print(f"  Components: {components1}")
    
    # Scenario 2: Low volume, expensive CAC
    calculator.episode_metrics['spend'] = 500
    calculator.episode_metrics['conversions'] = 2
    context2 = {
        'won': True,
        'cost': 50.0,
        'click_occurred': False,
        'conversion_occurred': False,
        'conversion_value': 0.0,
        'step': 200
    }
    reward2, components2 = calculator.calculate_reward(context2)
    print(f"\nScenario 2 - Low volume, expensive CAC:")
    print(f"  Current CAC: ${calculator._get_current_cac():.2f}")
    print(f"  Total Reward: {reward2:.3f}")
    print(f"  Components: {components2}")
    
    # Scenario 3: Good balance
    calculator.reset_episode_metrics()
    for i in range(10):
        context3 = {
            'won': True,
            'cost': 3.0,
            'click_occurred': i % 3 == 0,
            'conversion_occurred': i % 10 == 0,
            'conversion_value': 150.0 if i % 10 == 0 else 0,
            'step': 300 + i
        }
        reward3, components3 = calculator.calculate_reward(context3)
    
    print(f"\nScenario 3 - Balanced approach after 10 steps:")
    print(f"  Metrics: {calculator.episode_metrics}")
    print(f"  Current CAC: ${calculator._get_current_cac():.2f}")
    print(f"  Total Reward: {reward3:.3f}")
    print(f"  Components: {components3}")


if __name__ == "__main__":
    test_reward_function()