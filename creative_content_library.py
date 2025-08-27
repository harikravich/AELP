#!/usr/bin/env python3
"""
Creative Content Library - Actual ad copy, images, and creative elements
Tracks performance of real creative content
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import hashlib

@dataclass
class CreativeAsset:
    """Represents an actual ad creative"""
    creative_id: str
    channel: str
    format: str  # 'display', 'video', 'carousel', 'story'
    
    # Text content
    headline: str
    body_copy: str
    cta_text: str
    
    # Visual elements
    primary_image: str  # URL or description
    thumbnail: str
    background_color: str
    
    # Targeting hints
    emotional_tone: str  # 'urgent', 'supportive', 'informative', 'empowering'
    value_prop: str  # Main value being communicated
    
    # Performance tracking
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    
    @property
    def ctr(self) -> float:
        return self.clicks / max(1, self.impressions)
    
    @property
    def conversion_rate(self) -> float:
        return self.conversions / max(1, self.clicks)
    
    @property
    def cpc(self) -> float:
        return self.spend / max(1, self.clicks)
    
    @property
    def roas(self) -> float:
        revenue = self.conversions * 119.99  # Aura annual subscription
        return revenue / max(0.01, self.spend)


class CreativeContentLibrary:
    """Manages actual creative content for campaigns"""
    
    def __init__(self):
        self.creatives = {}
        self._initialize_real_creatives()
        
    def _initialize_real_creatives(self):
        """Initialize with actual Aura ad creatives"""
        
        # Google Search Ads
        google_search_creatives = [
            CreativeAsset(
                creative_id=self._generate_id("GS001"),
                channel="google",
                format="search",
                headline="Is Your Teen Okay? Know for Sure",
                body_copy="Aura monitors your child's digital wellbeing. See concerning patterns in texts, social media & browsing. Get alerts that matter.",
                cta_text="Start Free Trial",
                primary_image="",
                thumbnail="",
                background_color="#4A90E2",
                emotional_tone="urgent",
                value_prop="peace_of_mind"
            ),
            CreativeAsset(
                creative_id=self._generate_id("GS002"),
                channel="google",
                format="search",
                headline="Teen Mental Health Crisis: Take Action",
                body_copy="1 in 5 teens struggle with mental health. Aura helps you spot warning signs early through smart monitoring. Trusted by 100K+ parents.",
                cta_text="Protect Your Teen",
                primary_image="",
                thumbnail="",
                background_color="#E94B3C",
                emotional_tone="urgent",
                value_prop="early_intervention"
            ),
            CreativeAsset(
                creative_id=self._generate_id("GS003"),
                channel="google",
                format="search",
                headline="Smart Parental Controls That Actually Work",
                body_copy="Beyond screen time limits. Aura uses AI to detect cyberbullying, depression signals & risky behavior. Stay connected, not intrusive.",
                cta_text="Try Aura Free",
                primary_image="",
                thumbnail="",
                background_color="#2ECC71",
                emotional_tone="empowering",
                value_prop="smart_monitoring"
            ),
        ]
        
        # Facebook/Instagram Creatives
        facebook_creatives = [
            CreativeAsset(
                creative_id=self._generate_id("FB001"),
                channel="facebook",
                format="carousel",
                headline="The Signs Were There. We Just Didn't See Them.",
                body_copy="Sarah's mom shares how Aura helped her identify her daughter's anxiety before it became a crisis. Real-time alerts for concerning behavior patterns across all their devices.",
                cta_text="Learn How Aura Helps",
                primary_image="testimonial_sarah_mom.jpg",
                thumbnail="aura_dashboard_preview.jpg",
                background_color="#6C5CE7",
                emotional_tone="supportive",
                value_prop="testimonial"
            ),
            CreativeAsset(
                creative_id=self._generate_id("FB002"),
                channel="facebook",
                format="video",
                headline="What Your Teen Isn't Telling You",
                body_copy="70% of teens hide mental health struggles from parents. Watch how Aura bridges the communication gap without invading privacy.",
                cta_text="Watch 2-Min Demo",
                primary_image="teen_parent_conversation.mp4",
                thumbnail="video_thumbnail.jpg",
                background_color="#00B894",
                emotional_tone="informative",
                value_prop="education"
            ),
            CreativeAsset(
                creative_id=self._generate_id("FB003"),
                channel="facebook",
                format="display",
                headline="Sleep Issues? Excessive Screen Time? Mood Swings?",
                body_copy="These could be signs of digital stress. Aura's behavioral AI spots patterns you might miss. Join 100,000+ parents protecting their teens' mental health.",
                cta_text="Get Started Today",
                primary_image="warning_signs_infographic.jpg",
                thumbnail="aura_app_icon.jpg",
                background_color="#FDCB6E",
                emotional_tone="urgent",
                value_prop="pattern_recognition"
            ),
        ]
        
        # TikTok Creatives
        tiktok_creatives = [
            CreativeAsset(
                creative_id=self._generate_id("TT001"),
                channel="tiktok",
                format="story",
                headline="POV: You Finally Understand Your Teen",
                body_copy="Aura decoded my daughter's digital life. Not spying, just understanding. The alerts saved us. #ParentingWin #MentalHealthMatters",
                cta_text="Try Free for 30 Days",
                primary_image="parent_relief_moment.mp4",
                thumbnail="tiktok_cover.jpg",
                background_color="#FF6B6B",
                emotional_tone="empowering",
                value_prop="understanding"
            ),
            CreativeAsset(
                creative_id=self._generate_id("TT002"),
                channel="tiktok",
                format="story",
                headline="Therapist Explains Teen Warning Signs",
                body_copy="Dr. Kim breaks down 5 digital behaviors that signal teen depression. Aura monitors all of them automatically. Knowledge is power.",
                cta_text="Learn More",
                primary_image="therapist_explanation.mp4",
                thumbnail="dr_kim_thumbnail.jpg",
                background_color="#4ECDC4",
                emotional_tone="informative",
                value_prop="expert_endorsed"
            ),
        ]
        
        # Store all creatives
        for creative in google_search_creatives + facebook_creatives + tiktok_creatives:
            self.creatives[creative.creative_id] = creative
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID for creative"""
        return f"{prefix}_{hashlib.md5(prefix.encode()).hexdigest()[:8]}"
    
    def get_creative_for_context(self, channel: str, behavior_cluster: str, 
                                 time_of_day: int) -> CreativeAsset:
        """Select best creative for context"""
        
        channel_creatives = [c for c in self.creatives.values() if c.channel == channel]
        
        if not channel_creatives:
            # Fallback to any creative
            channel_creatives = list(self.creatives.values())
        
        # Smart selection based on performance and context
        if any(c.impressions > 100 for c in channel_creatives):
            # Use Thompson Sampling for explore/exploit
            best_creative = None
            best_score = -1
            
            for creative in channel_creatives:
                # Calculate Beta distribution parameters
                alpha = creative.clicks + 1
                beta = creative.impressions - creative.clicks + 1
                
                # Sample from Beta distribution
                score = np.random.beta(alpha, beta)
                
                # Boost score based on context match
                if "urgent" in behavior_cluster.lower() and creative.emotional_tone == "urgent":
                    score *= 1.2
                elif "high_intent" in behavior_cluster.lower() and creative.emotional_tone == "empowering":
                    score *= 1.15
                
                if score > best_score:
                    best_score = score
                    best_creative = creative
            
            return best_creative
        else:
            # Random selection for initial exploration
            return random.choice(channel_creatives)
    
    def record_impression(self, creative_id: str, cost: float):
        """Record an impression for a creative"""
        if creative_id in self.creatives:
            self.creatives[creative_id].impressions += 1
            self.creatives[creative_id].spend += cost
    
    def record_click(self, creative_id: str):
        """Record a click for a creative"""
        if creative_id in self.creatives:
            self.creatives[creative_id].clicks += 1
    
    def record_conversion(self, creative_id: str):
        """Record a conversion for a creative"""
        if creative_id in self.creatives:
            self.creatives[creative_id].conversions += 1
    
    def get_top_performers(self, channel: str, metric: str = "ctr", limit: int = 3) -> List[CreativeAsset]:
        """Get top performing creatives for a channel"""
        
        channel_creatives = [c for c in self.creatives.values() 
                            if c.channel == channel and c.impressions > 0]
        
        if metric == "ctr":
            sorted_creatives = sorted(channel_creatives, key=lambda c: c.ctr, reverse=True)
        elif metric == "conversion_rate":
            sorted_creatives = sorted(channel_creatives, key=lambda c: c.conversion_rate, reverse=True)
        elif metric == "roas":
            sorted_creatives = sorted(channel_creatives, key=lambda c: c.roas, reverse=True)
        else:
            sorted_creatives = channel_creatives
        
        return sorted_creatives[:limit]
    
    def get_creative_details(self, creative_id: str) -> Optional[CreativeAsset]:
        """Get full details for a creative"""
        return self.creatives.get(creative_id)


# Global instance
import numpy as np
creative_library = CreativeContentLibrary()