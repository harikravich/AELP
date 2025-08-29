"""
Creative Effectiveness Model
Learns which creative messages work for which segments and channels
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class CreativeVariant:
    """A creative message variant with performance tracking"""
    creative_id: str
    channel: str
    message_type: str  # 'fear', 'benefit', 'social_proof', 'authority', 'urgency'
    
    # Message components
    headline: str
    body: str
    cta: str
    visual_style: str  # 'emotional', 'clinical', 'lifestyle', 'comparison'
    
    # Performance tracking
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    revenue: float = 0.0
    
    # Segment-specific performance
    segment_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_ctr(self, segment: Optional[str] = None) -> float:
        """Get CTR overall or for specific segment"""
        if segment and segment in self.segment_performance:
            perf = self.segment_performance[segment]
            return perf['clicks'] / max(1, perf['impressions'])
        return self.clicks / max(1, self.impressions)
    
    def get_cvr(self, segment: Optional[str] = None) -> float:
        """Get conversion rate overall or for specific segment"""
        if segment and segment in self.segment_performance:
            perf = self.segment_performance[segment]
            return perf['conversions'] / max(1, perf['clicks'])
        return self.conversions / max(1, self.clicks)

class CreativeEffectivenessModel:
    """Models creative effectiveness across segments and channels"""
    
    def __init__(self):
        self.creatives: Dict[str, CreativeVariant] = {}
        self.message_embeddings: Dict[str, np.ndarray] = {}
        self.segment_preferences: Dict[str, Dict[str, float]] = {}
        self.channel_effectiveness: Dict[str, Dict[str, float]] = {}
        
        # Initialize realistic creative variants
        self._initialize_creatives()
        
        # Thompson sampling parameters for exploration
        self.alpha: Dict[str, float] = {}  # Success counts
        self.beta: Dict[str, float] = {}   # Failure counts
    
    def _initialize_creatives(self):
        """Initialize realistic creative variants based on behavioral health marketing"""
        
        # Google Search creatives
        self.add_creative(CreativeVariant(
            creative_id="google_crisis_1",
            channel="google",
            message_type="urgency",
            headline="Is Your Teen in Crisis? Get Help Now",
            body="AI-powered monitoring detects mood changes before they escalate. Trusted by 50,000 parents.",
            cta="Start Free Trial",
            visual_style="clinical"
        ))
        
        self.add_creative(CreativeVariant(
            creative_id="google_prevention_1",
            channel="google",
            message_type="benefit",
            headline="Prevent Teen Mental Health Issues",
            body="Clinical psychologists recommend early detection. Our AI spots warning signs you might miss.",
            cta="Learn More",
            visual_style="clinical"
        ))
        
        self.add_creative(CreativeVariant(
            creative_id="google_authority_1",
            channel="google",
            message_type="authority",
            headline="CDC-Recommended Screen Time Monitoring",
            body="Follow AAP guidelines with AI assistance. Know when device use becomes problematic.",
            cta="See Guidelines",
            visual_style="clinical"
        ))
        
        # Facebook/Instagram creatives
        self.add_creative(CreativeVariant(
            creative_id="facebook_social_1",
            channel="facebook",
            message_type="social_proof",
            headline="Join 50,000 Parents Who Sleep Better",
            body="\"Finally, I know my daughter is safe online without being invasive\" - Sarah M.",
            cta="Try It Free",
            visual_style="lifestyle"
        ))
        
        self.add_creative(CreativeVariant(
            creative_id="facebook_fear_1",
            channel="facebook",
            message_type="fear",
            headline="What's Really Happening at 2AM?",
            body="70% of teen mental health issues start online. Know the signs before it's too late.",
            cta="Protect Your Teen",
            visual_style="emotional"
        ))
        
        # TikTok creatives (younger parents)
        self.add_creative(CreativeVariant(
            creative_id="tiktok_trend_1",
            channel="tiktok",
            message_type="benefit",
            headline="Be the Parent Who Gets It 💯",
            body="AI helps you understand their digital world without invading privacy",
            cta="Start Protecting",
            visual_style="lifestyle"
        ))
        
    def add_creative(self, creative: CreativeVariant):
        """Add a creative variant to the model"""
        self.creatives[creative.creative_id] = creative
        # Initialize Thompson sampling parameters
        self.alpha[creative.creative_id] = 1.0
        self.beta[creative.creative_id] = 1.0
    
    def select_creative(self, channel: str, segment: str, 
                        context: Dict[str, Any]) -> CreativeVariant:
        """Select best creative using Thompson sampling for exploration/exploitation"""
        
        # Filter creatives for channel
        channel_creatives = [c for c in self.creatives.values() if c.channel == channel]
        
        if not channel_creatives:
            # Fallback to any creative
            channel_creatives = list(self.creatives.values())
        
        # Thompson sampling
        best_creative = None
        best_sample = -1
        
        for creative in channel_creatives:
            # Sample from Beta distribution
            sample = np.random.beta(
                self.alpha.get(creative.creative_id, 1),
                self.beta.get(creative.creative_id, 1)
            )
            
            # Boost sample based on segment match
            if segment in creative.segment_performance:
                seg_perf = creative.segment_performance[segment]
                if seg_perf['impressions'] > 10:
                    # Use actual performance for this segment
                    segment_ctr = seg_perf['clicks'] / max(1, seg_perf['impressions'])
                    sample = sample * 0.7 + segment_ctr * 0.3
            
            # Context bonuses
            hour = context.get('hour', 12)
            if creative.message_type == 'urgency' and hour in [22, 23, 0, 1, 2]:
                sample *= 1.2  # Urgency works better late night
            
            if creative.message_type == 'authority' and segment == 'thorough_researcher':
                sample *= 1.15  # Researchers respond to authority
            
            if best_sample < sample:
                best_sample = sample
                best_creative = creative
        
        return best_creative or channel_creatives[0]
    
    def record_impression(self, creative_id: str, segment: str):
        """Record an impression"""
        if creative_id in self.creatives:
            creative = self.creatives[creative_id]
            creative.impressions += 1
            
            if segment not in creative.segment_performance:
                creative.segment_performance[segment] = {
                    'impressions': 0, 'clicks': 0, 'conversions': 0, 'revenue': 0
                }
            creative.segment_performance[segment]['impressions'] += 1
    
    def record_click(self, creative_id: str, segment: str):
        """Record a click and update Thompson sampling parameters"""
        if creative_id in self.creatives:
            creative = self.creatives[creative_id]
            creative.clicks += 1
            
            if segment in creative.segment_performance:
                creative.segment_performance[segment]['clicks'] += 1
            
            # Update Thompson sampling (success)
            self.alpha[creative_id] += 1
    
    def record_conversion(self, creative_id: str, segment: str, value: float):
        """Record a conversion"""
        if creative_id in self.creatives:
            creative = self.creatives[creative_id]
            creative.conversions += 1
            creative.revenue += value
            
            if segment in creative.segment_performance:
                creative.segment_performance[segment]['conversions'] += 1
                creative.segment_performance[segment]['revenue'] += value
    
    def get_creative_insights(self) -> Dict[str, Any]:
        """Get insights about creative performance"""
        insights = {
            'total_creatives': len(self.creatives),
            'by_channel': {},
            'by_message_type': {},
            'top_performers': []
        }
        
        # Aggregate by channel
        for channel in ['google', 'facebook', 'tiktok']:
            channel_creatives = [c for c in self.creatives.values() if c.channel == channel]
            if channel_creatives:
                total_impr = sum(c.impressions for c in channel_creatives)
                total_clicks = sum(c.clicks for c in channel_creatives)
                insights['by_channel'][channel] = {
                    'impressions': total_impr,
                    'clicks': total_clicks,
                    'ctr': total_clicks / max(1, total_impr)
                }
        
        # Top performers
        sorted_creatives = sorted(self.creatives.values(), 
                                 key=lambda c: c.get_ctr(), 
                                 reverse=True)
        
        for creative in sorted_creatives[:5]:
            insights['top_performers'].append({
                'id': creative.creative_id,
                'channel': creative.channel,
                'headline': creative.headline[:50],
                'ctr': round(creative.get_ctr(), 4),
                'impressions': creative.impressions
            })
        
        return insights
    
    def calculate_effectiveness_score(self, creative_id: str, segment: str) -> float:
        """Calculate effectiveness score for creative-segment pair"""
        if creative_id not in self.creatives:
            return 0.0
        
        creative = self.creatives[creative_id]
        
        # Get segment-specific performance
        if segment in creative.segment_performance:
            perf = creative.segment_performance[segment]
            if perf['impressions'] < 10:
                # Not enough data, use overall performance
                ctr = creative.get_ctr()
                cvr = creative.get_cvr()
            else:
                ctr = perf['clicks'] / max(1, perf['impressions'])
                cvr = perf['conversions'] / max(1, perf['clicks'])
        else:
            ctr = creative.get_ctr()
            cvr = creative.get_cvr()
        
        # Weighted score
        effectiveness = (ctr * 0.3) + (cvr * 0.7)
        return effectiveness

# Global instance
creative_model = CreativeEffectivenessModel()