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
    """Manages actual creative content for campaigns
    
    NOW ENHANCED WITH:
    - LLM-powered creative generation
    - Infinite headline variations
    - Dynamic creative optimization
    - A/B testing framework
    """
    
    def __init__(self, enable_llm_generation: bool = True):
        self.creatives = {}
        self.enable_llm_generation = enable_llm_generation
        self.llm_generator = None
        
        # Initialize LLM generator if enabled
        if enable_llm_generation:
            try:
                from hybrid_llm_rl_integration import CreativeGenerator, LLMStrategyConfig
                config = LLMStrategyConfig(
                    model="gpt-4o-mini",
                    temperature=0.9,  # Higher for creativity
                    use_caching=True
                )
                self.llm_generator = CreativeGenerator(config)
                print("✅ CreativeLibrary enhanced with LLM generation capabilities")
            except Exception as e:
                print(f"⚠️ LLM creative generation not available: {e}")
                self.llm_generator = None
        
        self._initialize_real_creatives()
        
        # Track generated creatives separately
        self.generated_creatives = {}
        self.creative_performance = {}  # Track which creatives work best
        
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
            # Use any creative if needed
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
        # Check both original and generated creatives
        if creative_id in self.creatives:
            return self.creatives[creative_id]
        return self.generated_creatives.get(creative_id)
    
    def generate_creative_variation(self, 
                                   base_creative: CreativeAsset,
                                   theme: str = None,
                                   segment: str = "concerned_parents") -> CreativeAsset:
        """Generate a new creative variation using LLM.
        
        Args:
            base_creative: Creative to base variation on
            theme: Optional theme override
            segment: Target segment
            
        Returns:
            New CreativeAsset with generated content
        """
        if not self.llm_generator:
            # NO FALLBACKS - LLM generator is REQUIRED
            raise RuntimeError("LLM generator is REQUIRED for creative variations. NO FALLBACKS.")
        
        # Use LLM to generate variation
        if not theme:
            theme = base_creative.value_prop
        
        # Determine emotional tone based on performance
        if base_creative.ctr < 0.01:
            tone = "urgent"
        elif base_creative.conversion_rate < 0.02:
            tone = "empowering"
        else:
            tone = base_creative.emotional_tone
        
        # Generate new headline
        new_headline = self.llm_generator.generate_headline(theme, segment, tone)
        
        # Generate complete ad copy
        ad_copy = self.llm_generator.generate_ad_copy(new_headline, theme)
        
        # Create new creative asset
        new_creative = CreativeAsset(
            creative_id=self._generate_id(f"LLM_{len(self.generated_creatives)}"),
            channel=base_creative.channel,
            format=base_creative.format,
            headline=new_headline,
            body_copy=ad_copy.get("description", base_creative.body_copy),
            cta_text=ad_copy.get("cta", base_creative.cta_text),
            primary_image=base_creative.primary_image,  # Reuse visual assets
            thumbnail=base_creative.thumbnail,
            background_color=base_creative.background_color,
            emotional_tone=tone,
            value_prop=theme
        )
        
        # Store generated creative
        self.generated_creatives[new_creative.creative_id] = new_creative
        return new_creative
    
    def generate_creative_batch(self, 
                               channel: str,
                               n_variations: int = 5,
                               themes: List[str] = None) -> List[CreativeAsset]:
        """Generate multiple creative variations for testing.
        
        Args:
            channel: Target channel
            n_variations: Number of variations to generate
            themes: Optional list of themes to use
            
        Returns:
            List of generated CreativeAssets
        """
        if not themes:
            themes = ["safety", "balance", "trust", "education", "peace", "control"]
        
        # Get base creatives for the channel
        channel_creatives = [c for c in self.creatives.values() if c.channel == channel]
        if not channel_creatives:
            channel_creatives = list(self.creatives.values())[:3]
        
        generated = []
        for i in range(n_variations):
            base = random.choice(channel_creatives)
            theme = random.choice(themes)
            variation = self.generate_creative_variation(base, theme)
            generated.append(variation)
        
        return generated
    
    def get_winning_creative(self, channel: str, metric: str = "roas") -> CreativeAsset:
        """Get the best performing creative, considering both original and generated.
        
        Args:
            channel: Channel to filter by
            metric: Performance metric to optimize
            
        Returns:
            Best performing creative
        """
        # Combine original and generated creatives
        all_creatives = list(self.creatives.values()) + list(self.generated_creatives.values())
        channel_creatives = [c for c in all_creatives if c.channel == channel]
        
        if not channel_creatives:
            channel_creatives = all_creatives
        
        # Sort by metric
        if metric == "ctr":
            best = max(channel_creatives, key=lambda c: c.ctr)
        elif metric == "conversion_rate":
            best = max(channel_creatives, key=lambda c: c.conversion_rate)
        else:  # roas
            best = max(channel_creatives, key=lambda c: c.roas)
        
        return best
    
    def run_creative_tournament(self, channel: str, rounds: int = 10) -> Dict[str, any]:
        """Run a tournament to find best creative through iterative testing.
        
        Args:
            channel: Channel to test on
            rounds: Number of tournament rounds
            
        Returns:
            Tournament results and winner
        """
        # Start with existing creatives
        contestants = [c for c in self.creatives.values() if c.channel == channel][:5]
        
        # Add some generated variations
        if self.llm_generator:
            contestants.extend(self.generate_creative_batch(channel, 5))
        
        results = {
            "rounds": [],
            "winner": None,
            "performance_lift": 0.0
        }
        
        for round_num in range(rounds):
            # Simulate performance (in real system, this would be actual data)
            for creative in contestants:
                # Simulate impressions and clicks
                creative.impressions += random.randint(100, 1000)
                creative.clicks += random.randint(1, int(creative.impressions * 0.05))
                creative.conversions += random.randint(0, int(creative.clicks * 0.1))
                creative.spend += creative.clicks * random.uniform(0.5, 2.0)
            
            # Find worst performer
            worst = min(contestants, key=lambda c: c.roas)
            
            # Generate new challenger based on best performer
            best = max(contestants, key=lambda c: c.roas)
            if self.llm_generator and round_num < rounds - 1:
                # Create variation of winner
                challenger = self.generate_creative_variation(best)
                # Replace worst with challenger
                contestants.remove(worst)
                contestants.append(challenger)
            
            results["rounds"].append({
                "round": round_num + 1,
                "best_roas": best.roas,
                "worst_roas": worst.roas,
                "replaced": worst.creative_id if round_num < rounds - 1 else None
            })
        
        # Final winner
        winner = max(contestants, key=lambda c: c.roas)
        baseline_roas = max([c.roas for c in self.creatives.values() if c.channel == channel], default=1.0)
        
        results["winner"] = winner.creative_id
        results["winner_headline"] = winner.headline
        results["performance_lift"] = (winner.roas / baseline_roas - 1) * 100 if baseline_roas > 0 else 0
        
        return results


# Global instance
import numpy as np
# Enable LLM generation for infinite creative variations
creative_library = CreativeContentLibrary(enable_llm_generation=True)