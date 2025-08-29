"""
Channel Effectiveness Model
Learns which channels work best for different segments and contexts
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

@dataclass 
class ChannelPerformance:
    """Track performance metrics for a channel"""
    channel_name: str
    
    # Overall metrics
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    revenue: float = 0.0
    
    # Segment-specific performance
    segment_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based performance
    hourly_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Device performance
    device_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Competition metrics
    avg_competition_level: float = 0.0
    win_rate: float = 0.0
    avg_cpc: float = 0.0
    
    def get_roas(self, segment: Optional[str] = None) -> float:
        """Get ROAS overall or for segment"""
        if segment and segment in self.segment_metrics:
            metrics = self.segment_metrics[segment]
            return metrics.get('revenue', 0) / max(0.01, metrics.get('spend', 0.01))
        return self.revenue / max(0.01, self.spend)
    
    def get_ctr(self, segment: Optional[str] = None) -> float:
        """Get CTR overall or for segment"""
        if segment and segment in self.segment_metrics:
            metrics = self.segment_metrics[segment]
            return metrics.get('clicks', 0) / max(1, metrics.get('impressions', 1))
        return self.clicks / max(1, self.impressions)

class ChannelEffectivenessModel:
    """Models channel effectiveness and learns optimal channel selection"""
    
    def __init__(self):
        self.channels: Dict[str, ChannelPerformance] = {}
        
        # Channel characteristics (realistic modeling)
        self.channel_characteristics = {
            'google': {
                'type': 'search',
                'intent_level': 'high',
                'avg_cpc': 3.50,
                'peak_hours': [9, 10, 11, 14, 15, 16, 20, 21],
                'strong_segments': ['urgent_thorough_high-intent', 'patient_comparing_high-intent'],
                'devices': {'desktop': 0.45, 'mobile': 0.50, 'tablet': 0.05}
            },
            'facebook': {
                'type': 'social',
                'intent_level': 'medium',
                'avg_cpc': 2.20,
                'peak_hours': [12, 13, 19, 20, 21, 22],
                'strong_segments': ['patient_exploring_browsing', 'immediate_quick_browsing'],
                'devices': {'mobile': 0.75, 'desktop': 0.20, 'tablet': 0.05}
            },
            'tiktok': {
                'type': 'social',
                'intent_level': 'low',
                'avg_cpc': 1.80,
                'peak_hours': [18, 19, 20, 21, 22, 23],
                'strong_segments': ['immediate_quick_browsing', 'urgent_quick_browsing'],
                'devices': {'mobile': 0.95, 'desktop': 0.04, 'tablet': 0.01}
            },
            'bing': {
                'type': 'search',
                'intent_level': 'high',
                'avg_cpc': 2.80,
                'peak_hours': [9, 10, 11, 14, 15, 16],
                'strong_segments': ['patient_thorough_high-intent'],
                'devices': {'desktop': 0.60, 'mobile': 0.35, 'tablet': 0.05}
            }
        }
        
        # Initialize channels
        for channel_name in self.channel_characteristics:
            self.channels[channel_name] = ChannelPerformance(channel_name)
        
        # Learning parameters (Multi-Armed Bandit with UCB)
        self.channel_selections: Dict[str, int] = {ch: 0 for ch in self.channels}
        self.channel_rewards: Dict[str, float] = {ch: 0.0 for ch in self.channels}
    
    def select_channel(self, segment: str, context: Dict[str, Any], 
                      explore: bool = True) -> str:
        """Select optimal channel using Upper Confidence Bound (UCB) algorithm"""
        
        hour = context.get('hour', 12)
        device = context.get('device', 'mobile')
        budget_remaining = context.get('budget_remaining', float('inf'))
        
        # Calculate UCB scores for each channel
        ucb_scores = {}
        total_selections = sum(self.channel_selections.values())
        
        for channel_name, channel in self.channels.items():
            characteristics = self.channel_characteristics[channel_name]
            
            # Base score from historical performance
            if self.channel_selections[channel_name] > 0:
                avg_reward = self.channel_rewards[channel_name] / self.channel_selections[channel_name]
                exploration_bonus = np.sqrt(2 * np.log(max(1, total_selections)) / self.channel_selections[channel_name])
            else:
                # Unseen channel gets high exploration bonus
                avg_reward = 0
                exploration_bonus = float('inf') if explore else 0
            
            # Context adjustments
            context_multiplier = 1.0
            
            # Hour bonus
            if hour in characteristics['peak_hours']:
                context_multiplier *= 1.2
            
            # Segment affinity
            if segment in characteristics['strong_segments']:
                context_multiplier *= 1.3
            elif any(seg_part in segment for seg_part in ['urgent', 'high-intent']):
                if characteristics['intent_level'] == 'high':
                    context_multiplier *= 1.15
            
            # Device match
            device_match = characteristics['devices'].get(device, 0.1)
            context_multiplier *= (0.8 + device_match * 0.4)
            
            # Budget considerations
            if budget_remaining < 100 and characteristics['avg_cpc'] > 3:
                context_multiplier *= 0.5  # Penalize expensive channels when budget low
            
            # Calculate final UCB score
            if exploration_bonus == float('inf'):
                ucb_scores[channel_name] = float('inf')
            else:
                ucb_scores[channel_name] = (avg_reward * context_multiplier) + exploration_bonus
        
        # Select channel with highest UCB score
        if explore and np.random.random() < 0.1:
            # 10% pure exploration
            return np.random.choice(list(self.channels.keys()))
        else:
            return max(ucb_scores.keys(), key=lambda k: ucb_scores[k])
    
    def record_outcome(self, channel: str, segment: str, context: Dict[str, Any],
                       outcome: Dict[str, Any]):
        """Record the outcome of a channel selection"""
        
        if channel not in self.channels:
            return
        
        channel_perf = self.channels[channel]
        
        # Update overall metrics
        channel_perf.impressions += outcome.get('impressions', 0)
        channel_perf.clicks += outcome.get('clicks', 0)
        channel_perf.conversions += outcome.get('conversions', 0)
        channel_perf.spend += outcome.get('cost', 0)
        channel_perf.revenue += outcome.get('revenue', 0)
        
        # Update segment-specific metrics
        if segment not in channel_perf.segment_metrics:
            channel_perf.segment_metrics[segment] = {
                'impressions': 0, 'clicks': 0, 'conversions': 0,
                'spend': 0, 'revenue': 0
            }
        
        seg_metrics = channel_perf.segment_metrics[segment]
        seg_metrics['impressions'] += outcome.get('impressions', 0)
        seg_metrics['clicks'] += outcome.get('clicks', 0)
        seg_metrics['conversions'] += outcome.get('conversions', 0)
        seg_metrics['spend'] += outcome.get('cost', 0)
        seg_metrics['revenue'] += outcome.get('revenue', 0)
        
        # Update hourly performance
        hour = context.get('hour', 12)
        if hour not in channel_perf.hourly_performance:
            channel_perf.hourly_performance[hour] = {
                'impressions': 0, 'clicks': 0, 'conversions': 0
            }
        
        hour_metrics = channel_perf.hourly_performance[hour]
        hour_metrics['impressions'] += outcome.get('impressions', 0)
        hour_metrics['clicks'] += outcome.get('clicks', 0)
        hour_metrics['conversions'] += outcome.get('conversions', 0)
        
        # Update UCB tracking
        self.channel_selections[channel] += 1
        
        # Calculate reward (ROAS-based with CTR component)
        if outcome.get('impressions', 0) > 0:
            ctr = outcome.get('clicks', 0) / outcome.get('impressions', 1)
            roas = outcome.get('revenue', 0) / max(0.01, outcome.get('cost', 0.01))
            reward = (ctr * 0.3) + (roas * 0.7)  # Weighted reward
        else:
            reward = 0
        
        self.channel_rewards[channel] += reward
        
        # Update competition metrics
        if outcome.get('won', False):
            channel_perf.win_rate = (
                (channel_perf.win_rate * (channel_perf.impressions - 1) + 1) /
                channel_perf.impressions
            )
        
        if outcome.get('cost', 0) > 0 and outcome.get('clicks', 0) > 0:
            cpc = outcome['cost'] / outcome['clicks']
            # Exponential moving average
            channel_perf.avg_cpc = channel_perf.avg_cpc * 0.95 + cpc * 0.05
    
    def get_channel_recommendation(self, segment: str) -> Dict[str, Any]:
        """Get channel recommendations for a segment"""
        
        recommendations = []
        
        for channel_name, channel in self.channels.items():
            score = 0
            reasons = []
            
            # Check if we have data for this segment
            if segment in channel.segment_metrics:
                seg_metrics = channel.segment_metrics[segment]
                if seg_metrics['impressions'] > 10:
                    roas = seg_metrics['revenue'] / max(0.01, seg_metrics['spend'])
                    ctr = seg_metrics['clicks'] / max(1, seg_metrics['impressions'])
                    
                    score = (roas / 3.0) * 0.6 + (ctr / 0.05) * 0.4  # Normalize and weight
                    
                    if roas > 3:
                        reasons.append(f"High ROAS ({roas:.1f}x)")
                    if ctr > 0.03:
                        reasons.append(f"Good CTR ({ctr:.1%})")
            
            # Check channel characteristics match
            characteristics = self.channel_characteristics[channel_name]
            if segment in characteristics['strong_segments']:
                score += 0.2
                reasons.append("Strong segment match")
            
            recommendations.append({
                'channel': channel_name,
                'score': score,
                'reasons': reasons,
                'data_points': channel.segment_metrics.get(segment, {}).get('impressions', 0)
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'primary': recommendations[0] if recommendations else None,
            'alternatives': recommendations[1:3] if len(recommendations) > 1 else [],
            'avoid': [r for r in recommendations if r['score'] < 0.3]
        }
    
    def get_insights(self) -> Dict[str, Any]:
        """Get channel performance insights"""
        
        insights = {
            'channel_performance': {},
            'best_segments_by_channel': {},
            'peak_hours_discovered': {},
            'learning_progress': {
                'total_selections': sum(self.channel_selections.values()),
                'channel_selections': self.channel_selections.copy()
            }
        }
        
        for channel_name, channel in self.channels.items():
            # Overall performance
            insights['channel_performance'][channel_name] = {
                'impressions': channel.impressions,
                'spend': round(channel.spend, 2),
                'revenue': round(channel.revenue, 2),
                'roas': round(channel.get_roas(), 2),
                'ctr': round(channel.get_ctr(), 4),
                'avg_cpc': round(channel.avg_cpc, 2)
            }
            
            # Best performing segments
            if channel.segment_metrics:
                best_segments = sorted(
                    channel.segment_metrics.items(),
                    key=lambda x: x[1].get('revenue', 0) / max(0.01, x[1].get('spend', 0.01)),
                    reverse=True
                )[:3]
                
                insights['best_segments_by_channel'][channel_name] = [
                    {
                        'segment': seg,
                        'roas': round(metrics['revenue'] / max(0.01, metrics['spend']), 2)
                    }
                    for seg, metrics in best_segments
                ]
            
            # Peak hours discovered
            if channel.hourly_performance:
                peak_hours = sorted(
                    channel.hourly_performance.items(),
                    key=lambda x: x[1].get('clicks', 0) / max(1, x[1].get('impressions', 1)),
                    reverse=True
                )[:3]
                
                insights['peak_hours_discovered'][channel_name] = [
                    hour for hour, _ in peak_hours
                ]
        
        return insights

# Global instance
channel_model = ChannelEffectivenessModel()