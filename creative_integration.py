"""
Creative Integration Module for GAELP
Connects the CreativeSelector system to existing simulations and replaces empty ad_content dictionaries.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

# Import from existing modules
from creative_selector import (
    CreativeSelector, UserState, UserSegment, JourneyStage,
    Creative, ImpressionData, LandingPageType
)


class SimulationPersona(Enum):
    """Map simulation personas to user segments"""
    CONCERNED_PARENT = "concerned_parent"
    TECH_SAVVY_PARENT = "tech_savvy_parent"
    CRISIS_PARENT = "crisis_parent"
    RESEARCHER = "researcher"
    PRICE_CONSCIOUS = "price_conscious"


@dataclass
class SimulationContext:
    """Context data from various simulation systems"""
    user_id: str
    persona: str
    channel: str
    device_type: str = "desktop"
    time_of_day: str = "afternoon"
    session_count: int = 1
    price_sensitivity: float = 0.5
    urgency_score: float = 0.5
    technical_level: float = 0.5
    conversion_probability: float = 0.5
    previous_interactions: List[str] = None
    geo_location: str = "US"
    
    def __post_init__(self):
        if self.previous_interactions is None:
            self.previous_interactions = []


class CreativeIntegration:
    """
    Main integration class that bridges CreativeSelector with simulation systems
    """
    
    def __init__(self):
        self.creative_selector = CreativeSelector()
        self.persona_to_segment_map = {
            SimulationPersona.CONCERNED_PARENT.value: UserSegment.CRISIS_PARENTS,
            SimulationPersona.TECH_SAVVY_PARENT.value: UserSegment.RESEARCHERS,
            SimulationPersona.CRISIS_PARENT.value: UserSegment.CRISIS_PARENTS,
            SimulationPersona.RESEARCHER.value: UserSegment.RESEARCHERS,
            SimulationPersona.PRICE_CONSCIOUS.value: UserSegment.PRICE_CONSCIOUS,
            # Default mappings for common simulation cases
            "impulse_buyer": UserSegment.PRICE_CONSCIOUS,
            "bargain_hunter": UserSegment.PRICE_CONSCIOUS,
            "premium_buyer": UserSegment.RESEARCHERS,
            "casual_browser": UserSegment.RETARGETING,
        }
        
        # Track impression history for fatigue modeling
        self.user_impressions: Dict[str, List[str]] = {}
    
    def get_targeted_ad_content(self, context: SimulationContext) -> Dict[str, Any]:
        """
        Get targeted ad content using CreativeSelector instead of empty {}
        Returns rich ad content with headline, description, CTA, landing page, etc.
        """
        
        # Convert simulation context to UserState
        user_state = self._convert_to_user_state(context)
        
        # Select optimal creative
        creative, selection_reason = self.creative_selector.select_creative(user_state)
        
        # Convert creative to simulation-compatible ad_content
        ad_content = self._convert_creative_to_ad_content(creative, context)
        
        # Add selection metadata
        ad_content['selection_reason'] = selection_reason
        ad_content['creative_id'] = creative.id
        
        return ad_content
    
    def _convert_to_user_state(self, context: SimulationContext) -> UserState:
        """Convert SimulationContext to UserState for CreativeSelector"""
        
        # Map persona to user segment
        segment = self.persona_to_segment_map.get(
            context.persona.lower(), 
            UserSegment.RETARGETING  # Default fallback
        )
        
        # Determine journey stage based on context
        journey_stage = self._determine_journey_stage(context)
        
        return UserState(
            user_id=context.user_id,
            segment=segment,
            journey_stage=journey_stage,
            device_type=context.device_type,
            time_of_day=context.time_of_day,
            previous_interactions=context.previous_interactions,
            conversion_probability=context.conversion_probability,
            urgency_score=context.urgency_score,
            price_sensitivity=context.price_sensitivity,
            technical_level=context.technical_level,
            session_count=context.session_count,
            last_seen=time.time(),
            geo_location=context.geo_location
        )
    
    def _determine_journey_stage(self, context: SimulationContext) -> JourneyStage:
        """Determine user journey stage from simulation context"""
        
        # New users start at awareness
        if context.session_count == 1:
            return JourneyStage.AWARENESS
        
        # Users with previous interactions are in consideration
        if context.previous_interactions and context.session_count <= 3:
            return JourneyStage.CONSIDERATION
        
        # Frequent users are in decision phase
        if context.session_count > 3:
            return JourneyStage.DECISION
        
        # High urgency means decision phase
        if context.urgency_score > 0.8:
            return JourneyStage.DECISION
        
        # Default to consideration
        return JourneyStage.CONSIDERATION
    
    def _convert_creative_to_ad_content(self, creative: Creative, context: SimulationContext) -> Dict[str, Any]:
        """Convert Creative object to simulation-compatible ad_content dictionary"""
        
        # Base ad content with rich information
        ad_content = {
            # Creative content
            'headline': creative.headline,
            'description': creative.description,
            'cta': creative.cta,
            'image_url': creative.image_url,
            'landing_page': creative.landing_page.value,
            
            # Simulation compatibility fields (what existing code expects)
            'creative_quality': self._calculate_creative_quality(creative),
            'price_shown': self._determine_price_shown(creative, context),
            'brand_match': self._calculate_brand_match(creative, context),
            'relevance_score': self._calculate_relevance_score(creative, context),
            'product_id': self._determine_product_id(creative),
            
            # Additional targeting fields
            'quality_score': self._calculate_creative_quality(creative),
            'trust_signals': self._calculate_trust_signals(creative),
            'urgency_messaging': self._calculate_urgency_messaging(creative),
            'social_proof': self._calculate_social_proof(creative),
            'landing_page_match': self._calculate_landing_page_match(creative, context),
            
            # Creative metadata
            'creative_type': creative.creative_type.value,
            'segment': creative.segment.value,
            'journey_stage': creative.journey_stage.value,
            'tags': creative.tags,
            'priority': creative.priority
        }
        
        return ad_content
    
    def _calculate_creative_quality(self, creative: Creative) -> float:
        """Calculate creative quality score (0.0-1.0)"""
        base_quality = 0.5
        
        # Boost for high priority creatives
        priority_boost = min(creative.priority / 10.0, 0.3)
        
        # Boost for rich content (headline + description)
        content_boost = 0.1 if len(creative.headline) > 20 and len(creative.description) > 30 else 0
        
        # Tag-based quality indicators
        quality_tags = ["premium", "advanced", "professional", "expert"]
        tag_boost = 0.1 if any(tag in creative.tags for tag in quality_tags) else 0
        
        return min(base_quality + priority_boost + content_boost + tag_boost, 1.0)
    
    def _determine_price_shown(self, creative: Creative, context: SimulationContext) -> float:
        """Determine price to show based on creative and context"""
        
        # Price-conscious users see lower prices
        if creative.segment == UserSegment.PRICE_CONSCIOUS:
            if "free" in creative.tags or "trial" in creative.tags:
                return 0.0  # Free trial
            return 9.99  # Discounted price
        
        # Crisis situations show premium pricing (immediate value)
        if creative.segment == UserSegment.CRISIS_PARENTS:
            return 19.99
        
        # Researchers see full pricing
        if creative.segment == UserSegment.RESEARCHERS:
            return 29.99  # Full feature pricing
        
        # Default pricing
        return 14.99
    
    def _calculate_brand_match(self, creative: Creative, context: SimulationContext) -> float:
        """Calculate how well creative matches user's brand affinity"""
        
        # High match for segment-appropriate creatives
        if creative.segment.value == context.persona.lower():
            return 0.9
        
        # Medium match for related segments
        if (creative.segment == UserSegment.CRISIS_PARENTS and 
            context.persona in ["concerned_parent", "tech_savvy_parent"]):
            return 0.7
        
        # Lower match for unrelated segments
        return 0.4
    
    def _calculate_relevance_score(self, creative: Creative, context: SimulationContext) -> float:
        """Calculate content relevance score"""
        relevance = 0.5  # Base relevance
        
        # Exact persona match
        if creative.segment.value in context.persona.lower():
            relevance += 0.3
        
        # Channel relevance (social, search, etc.)
        if context.channel.lower() in ["social", "facebook", "instagram"]:
            if creative.creative_type.value in ["video", "carousel"]:
                relevance += 0.2
        elif context.channel.lower() in ["search", "google"]:
            if creative.creative_type.value in ["text_ad"]:
                relevance += 0.2
        
        # Device relevance
        if context.device_type == "mobile":
            if creative.creative_type.value in ["banner", "text_ad"]:
                relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _determine_product_id(self, creative: Creative) -> str:
        """Determine product ID based on creative"""
        return f"gaelp_{creative.segment.value}_{creative.journey_stage.value}"
    
    def _calculate_trust_signals(self, creative: Creative) -> float:
        """Calculate trust signal strength"""
        trust = 0.5  # Base trust
        
        trust_indicators = ["certified", "verified", "trusted", "secure", "protection"]
        if any(indicator in creative.description.lower() for indicator in trust_indicators):
            trust += 0.3
        
        if "99.9%" in creative.description or "real-time" in creative.description:
            trust += 0.2
        
        return min(trust, 1.0)
    
    def _calculate_urgency_messaging(self, creative: Creative) -> float:
        """Calculate urgency level in messaging"""
        urgency = 0.3  # Base urgency
        
        urgency_words = ["immediate", "now", "urgent", "limited time", "expires", "today"]
        urgency_count = sum(1 for word in urgency_words 
                           if word in (creative.headline + " " + creative.description).lower())
        
        return min(urgency + (urgency_count * 0.2), 1.0)
    
    def _calculate_social_proof(self, creative: Creative) -> float:
        """Calculate social proof strength"""
        social_proof = 0.4  # Base social proof
        
        proof_indicators = ["parents trust", "users", "families", "recommended", "rated"]
        if any(indicator in creative.description.lower() for indicator in proof_indicators):
            social_proof += 0.3
        
        if "10,000" in creative.description or "millions" in creative.description:
            social_proof += 0.2
        
        return min(social_proof, 1.0)
    
    def _calculate_landing_page_match(self, creative: Creative, context: SimulationContext) -> float:
        """Calculate how well landing page matches ad content"""
        
        # Emergency setup pages have high match for crisis situations
        if (creative.landing_page == LandingPageType.EMERGENCY_SETUP and 
            context.urgency_score > 0.7):
            return 0.9
        
        # Comparison pages for researchers
        if (creative.landing_page == LandingPageType.COMPARISON_GUIDE and 
            creative.segment == UserSegment.RESEARCHERS):
            return 0.8
        
        # Free trial for price-conscious
        if (creative.landing_page == LandingPageType.FREE_TRIAL and 
            creative.segment == UserSegment.PRICE_CONSCIOUS):
            return 0.9
        
        # Default good match
        return 0.6
    
    def track_impression(self, user_id: str, creative_id: str, clicked: bool = False, 
                        converted: bool = False, engagement_time: float = 0.0, 
                        cost: float = 0.0):
        """Track impression for fatigue modeling and performance optimization"""
        
        # Track in CreativeSelector
        self.creative_selector.track_impression(
            creative_id=creative_id,
            user_id=user_id,
            clicked=clicked,
            converted=converted,
            engagement_time=engagement_time,
            cost=cost
        )
        
        # Track locally for quick lookups
        if user_id not in self.user_impressions:
            self.user_impressions[user_id] = []
        self.user_impressions[user_id].append(creative_id)
    
    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Get creative performance report"""
        return self.creative_selector.get_performance_report(days)
    
    def get_fatigue_analysis(self, user_id: str) -> Dict[str, float]:
        """Get fatigue analysis for specific user"""
        return self.creative_selector.get_fatigue_analysis(user_id)
    
    def create_ab_test(self, test_name: str, variants: List[Dict[str, Any]]):
        """Create A/B test for creative optimization"""
        from creative_selector import ABTestVariant
        
        ab_variants = []
        for variant_data in variants:
            ab_variants.append(ABTestVariant(
                variant_id=variant_data['id'],
                name=variant_data['name'],
                traffic_split=variant_data['traffic_split'],
                creative_overrides=variant_data.get('overrides', {}),
                landing_page_override=variant_data.get('landing_page_override'),
                active=variant_data.get('active', True)
            ))
        
        self.creative_selector.create_ab_test(test_name, ab_variants)


# Helper functions for easy integration with existing code
def get_creative_integration() -> CreativeIntegration:
    """Get singleton instance of CreativeIntegration"""
    if not hasattr(get_creative_integration, '_instance'):
        get_creative_integration._instance = CreativeIntegration()
    return get_creative_integration._instance


def replace_empty_ad_content(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace empty ad_content {} with rich targeted content
    
    Usage:
        # Instead of: ad_content = {}
        ad_content = replace_empty_ad_content({
            'user_id': 'user_123',
            'persona': 'crisis_parent',
            'channel': 'search',
            'device_type': 'mobile'
        })
    """
    integration = get_creative_integration()
    
    # Create SimulationContext from provided context
    sim_context = SimulationContext(
        user_id=context.get('user_id', f'user_{int(time.time())}'),
        persona=context.get('persona', 'concerned_parent'),
        channel=context.get('channel', 'search'),
        device_type=context.get('device_type', 'desktop'),
        time_of_day=context.get('time_of_day', 'afternoon'),
        session_count=context.get('session_count', 1),
        price_sensitivity=context.get('price_sensitivity', 0.5),
        urgency_score=context.get('urgency_score', 0.5),
        technical_level=context.get('technical_level', 0.5),
        conversion_probability=context.get('conversion_probability', 0.5),
        previous_interactions=context.get('previous_interactions', []),
        geo_location=context.get('geo_location', 'US')
    )
    
    return integration.get_targeted_ad_content(sim_context)


# Example usage and testing
if __name__ == "__main__":
    print("=== Creative Integration Demo ===\n")
    
    integration = CreativeIntegration()
    
    # Test different personas
    test_contexts = [
        SimulationContext(
            user_id="crisis_parent_1",
            persona="crisis_parent",
            channel="search",
            device_type="mobile",
            urgency_score=0.9,
            session_count=1
        ),
        SimulationContext(
            user_id="researcher_1",
            persona="researcher",
            channel="social",
            device_type="desktop",
            technical_level=0.9,
            session_count=3
        ),
        SimulationContext(
            user_id="price_conscious_1",
            persona="price_conscious",
            channel="display",
            device_type="mobile",
            price_sensitivity=0.9,
            session_count=2
        )
    ]
    
    for context in test_contexts:
        print(f"--- {context.persona.upper()} USER ---")
        ad_content = integration.get_targeted_ad_content(context)
        
        print(f"Headline: {ad_content['headline']}")
        print(f"CTA: {ad_content['cta']}")
        print(f"Landing Page: {ad_content['landing_page']}")
        print(f"Price: ${ad_content['price_shown']}")
        print(f"Quality Score: {ad_content['creative_quality']:.2f}")
        print(f"Selection Reason: {ad_content['selection_reason']}")
        print()
        
        # Track impression
        integration.track_impression(
            user_id=context.user_id,
            creative_id=ad_content['creative_id'],
            clicked=True,
            engagement_time=30.0
        )
    
    # Show performance report
    print("=== Performance Report ===")
    report = integration.get_performance_report(1)
    print(f"Total Impressions: {report['total_impressions']}")
    print(f"Total Clicks: {report['total_clicks']}")
    
    for creative_id, perf in report['creative_performance'].items():
        if perf['impressions'] > 0:
            print(f"{creative_id}: {perf['clicks']}/{perf['impressions']} CTR={perf['ctr']:.2%}")