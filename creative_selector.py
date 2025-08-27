"""
Creative Selection System for GAELP
Dynamic ad creative selection based on user state, journey stage, and performance metrics.
Includes A/B testing, creative fatigue tracking, and landing page optimization.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import random
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict


class UserSegment(Enum):
    CRISIS_PARENTS = "crisis_parents"
    RESEARCHERS = "researchers"
    PRICE_CONSCIOUS = "price_conscious"
    RETARGETING = "retargeting"


class JourneyStage(Enum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    DECISION = "decision"
    RETENTION = "retention"


class CreativeType(Enum):
    HERO_IMAGE = "hero_image"
    VIDEO = "video"
    CAROUSEL = "carousel"
    TEXT_AD = "text_ad"
    BANNER = "banner"


class LandingPageType(Enum):
    EMERGENCY_SETUP = "emergency_setup"
    COMPARISON_GUIDE = "comparison_guide"
    FREE_TRIAL = "free_trial"
    FEATURE_DEEP_DIVE = "feature_deep_dive"


@dataclass
class Creative:
    """Individual creative asset with metadata"""
    id: str
    segment: UserSegment
    journey_stage: JourneyStage
    creative_type: CreativeType
    headline: str
    description: str
    cta: str
    image_url: str
    landing_page: LandingPageType
    priority: int = 1
    tags: List[str] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class UserState:
    """User state and context for creative selection"""
    user_id: str
    segment: UserSegment
    journey_stage: JourneyStage
    device_type: str
    time_of_day: str
    previous_interactions: List[str]
    conversion_probability: float
    urgency_score: float
    price_sensitivity: float
    technical_level: float
    session_count: int
    last_seen: float
    geo_location: str = "US"
    
    def get_context_hash(self) -> str:
        """Generate hash for A/B test bucket assignment"""
        context = f"{self.user_id}_{self.segment.value}_{self.journey_stage.value}"
        return hashlib.md5(context.encode()).hexdigest()


@dataclass
class ImpressionData:
    """Track creative impressions and performance"""
    creative_id: str
    user_id: str
    timestamp: float
    clicked: bool = False
    converted: bool = False
    engagement_time: float = 0.0
    cost: float = 0.0


@dataclass
class ABTestVariant:
    """A/B test variant configuration"""
    variant_id: str
    name: str
    traffic_split: float
    creative_overrides: Dict[str, Any]
    landing_page_override: Optional[LandingPageType] = None
    active: bool = True


class CreativeSelector:
    """
    Main creative selection system with fatigue tracking and A/B testing
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.creatives: Dict[str, Creative] = {}
        self.impressions: List[ImpressionData] = []
        self.user_impression_history: Dict[str, List[str]] = defaultdict(list)
        self.creative_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'impressions': 0, 'clicks': 0, 'conversions': 0, 'ctr': 0.0, 'cvr': 0.0
        })
        self.ab_tests: Dict[str, ABTestVariant] = {}
        self.fatigue_threshold = 3  # Max times to show same creative
        self.fatigue_decay_hours = 24  # Hours before fatigue resets
        
        self._initialize_creatives()
        if config_file:
            self.load_config(config_file)
    
    def _initialize_creatives(self):
        """Initialize default creative library"""
        
        # Crisis Parents Creatives
        self.add_creative(Creative(
            id="crisis_parent_emergency_1",
            segment=UserSegment.CRISIS_PARENTS,
            journey_stage=JourneyStage.AWARENESS,
            creative_type=CreativeType.HERO_IMAGE,
            headline="Protect Your Child Online - Setup in 5 Minutes",
            description="Immediate protection from harmful content. No technical setup required.",
            cta="Get Protected Now",
            image_url="/images/emergency_protection.jpg",
            landing_page=LandingPageType.EMERGENCY_SETUP,
            priority=10,
            tags=["urgent", "protection", "immediate"]
        ))
        
        self.add_creative(Creative(
            id="crisis_parent_video_1",
            segment=UserSegment.CRISIS_PARENTS,
            journey_stage=JourneyStage.CONSIDERATION,
            creative_type=CreativeType.VIDEO,
            headline="See How Fast GAELP Blocks Threats",
            description="Watch real-time protection in action. 99.9% threat detection rate.",
            cta="Watch Demo",
            image_url="/videos/protection_demo.mp4",
            landing_page=LandingPageType.EMERGENCY_SETUP,
            priority=8,
            tags=["demo", "real-time", "effective"]
        ))
        
        # Researcher Creatives
        self.add_creative(Creative(
            id="researcher_comparison_1",
            segment=UserSegment.RESEARCHERS,
            journey_stage=JourneyStage.CONSIDERATION,
            creative_type=CreativeType.CAROUSEL,
            headline="Compare AI Safety Solutions Side-by-Side",
            description="Detailed analysis of GAELP vs competitors. Performance benchmarks included.",
            cta="View Comparison",
            image_url="/images/comparison_charts.jpg",
            landing_page=LandingPageType.COMPARISON_GUIDE,
            priority=9,
            tags=["comparison", "data", "benchmarks"]
        ))
        
        self.add_creative(Creative(
            id="researcher_technical_1",
            segment=UserSegment.RESEARCHERS,
            journey_stage=JourneyStage.DECISION,
            creative_type=CreativeType.TEXT_AD,
            headline="Advanced AI Training Environment",
            description="Full API access, custom environments, detailed analytics dashboard.",
            cta="Explore Features",
            image_url="/images/technical_dashboard.jpg",
            landing_page=LandingPageType.FEATURE_DEEP_DIVE,
            priority=7,
            tags=["technical", "api", "advanced"]
        ))
        
        # Price-Conscious Creatives
        self.add_creative(Creative(
            id="price_conscious_trial_1",
            segment=UserSegment.PRICE_CONSCIOUS,
            journey_stage=JourneyStage.AWARENESS,
            creative_type=CreativeType.BANNER,
            headline="Free 30-Day Trial - No Credit Card Required",
            description="Try all premium features free. Cancel anytime. No hidden fees.",
            cta="Start Free Trial",
            image_url="/images/free_trial_badge.jpg",
            landing_page=LandingPageType.FREE_TRIAL,
            priority=10,
            tags=["free", "trial", "no-commitment"]
        ))
        
        self.add_creative(Creative(
            id="price_conscious_value_1",
            segment=UserSegment.PRICE_CONSCIOUS,
            journey_stage=JourneyStage.DECISION,
            creative_type=CreativeType.HERO_IMAGE,
            headline="50% Less Than Competitors",
            description="Same protection, half the cost. See detailed pricing comparison.",
            cta="Compare Prices",
            image_url="/images/pricing_comparison.jpg",
            landing_page=LandingPageType.COMPARISON_GUIDE,
            priority=8,
            tags=["value", "pricing", "savings"]
        ))
        
        # Retargeting Creatives
        self.add_creative(Creative(
            id="retargeting_urgency_1",
            segment=UserSegment.RETARGETING,
            journey_stage=JourneyStage.DECISION,
            creative_type=CreativeType.VIDEO,
            headline="Limited Time: 40% Off Premium",
            description="Offer expires in 48 hours. Don't miss out on advanced protection.",
            cta="Claim Discount",
            image_url="/videos/urgency_offer.mp4",
            landing_page=LandingPageType.FREE_TRIAL,
            priority=10,
            tags=["urgent", "discount", "limited-time"]
        ))
        
        self.add_creative(Creative(
            id="retargeting_abandoned_1",
            segment=UserSegment.RETARGETING,
            journey_stage=JourneyStage.RETENTION,
            creative_type=CreativeType.TEXT_AD,
            headline="Complete Your Setup in 2 Minutes",
            description="Your account is 90% complete. Finish setup and get protected today.",
            cta="Complete Setup",
            image_url="/images/progress_bar.jpg",
            landing_page=LandingPageType.EMERGENCY_SETUP,
            priority=9,
            tags=["completion", "progress", "quick"]
        ))
    
    def add_creative(self, creative: Creative):
        """Add creative to library"""
        self.creatives[creative.id] = creative
    
    def select_creative(self, user_state: UserState) -> Tuple[Creative, str]:
        """
        Main method to select optimal creative for user
        Returns: (selected_creative, reason)
        """
        
        # Check for A/B test assignment
        ab_variant = self._get_ab_test_variant(user_state)
        
        # Get candidate creatives
        candidates = self._get_candidate_creatives(user_state)
        
        if not candidates:
            # Fallback to any creative for segment
            candidates = [c for c in self.creatives.values() 
                         if c.segment == user_state.segment]
        
        if not candidates:
            # Ultimate fallback
            candidates = list(self.creatives.values())
        
        # Filter out fatigued creatives
        fresh_candidates = self._filter_fatigued_creatives(candidates, user_state.user_id)
        
        if fresh_candidates:
            candidates = fresh_candidates
        
        # Apply A/B test overrides if applicable
        if ab_variant:
            candidates = self._apply_ab_overrides(candidates, ab_variant)
        
        # Score and select best creative
        selected = self._score_and_select(candidates, user_state)
        
        # Generate selection reason
        reason = self._generate_selection_reason(selected, user_state, ab_variant)
        
        return selected, reason
    
    def _get_candidate_creatives(self, user_state: UserState) -> List[Creative]:
        """Get creatives matching user segment and journey stage"""
        candidates = []
        
        for creative in self.creatives.values():
            # Primary match: segment and stage
            if (creative.segment == user_state.segment and 
                creative.journey_stage == user_state.journey_stage):
                candidates.append(creative)
            
            # Secondary match: segment only (lower priority)
            elif creative.segment == user_state.segment:
                candidates.append(creative)
        
        return candidates
    
    def _filter_fatigued_creatives(self, candidates: List[Creative], user_id: str) -> List[Creative]:
        """Remove creatives that user has seen too often recently"""
        current_time = time.time()
        cutoff_time = current_time - (self.fatigue_decay_hours * 3600)
        
        # Get recent impressions for this user
        recent_impressions = [
            imp for imp in self.impressions 
            if (imp.user_id == user_id and imp.timestamp > cutoff_time)
        ]
        
        # Count impressions per creative
        impression_counts = defaultdict(int)
        for imp in recent_impressions:
            impression_counts[imp.creative_id] += 1
        
        # Filter out fatigued creatives
        fresh_candidates = [
            creative for creative in candidates
            if impression_counts[creative.id] < self.fatigue_threshold
        ]
        
        return fresh_candidates
    
    def _score_and_select(self, candidates: List[Creative], user_state: UserState) -> Creative:
        """Score candidates and select the best one"""
        if not candidates:
            return list(self.creatives.values())[0]  # Fallback
        
        scored_candidates = []
        
        for creative in candidates:
            score = self._calculate_creative_score(creative, user_state)
            scored_candidates.append((creative, score))
        
        # Sort by score (highest first) and select
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]
    
    def _calculate_creative_score(self, creative: Creative, user_state: UserState) -> float:
        """Calculate relevance score for creative given user state"""
        score = creative.priority  # Base score
        
        # Performance boost
        perf = self.creative_performance[creative.id]
        if perf['impressions'] > 10:  # Only if we have enough data
            score += perf['ctr'] * 100  # CTR boost
            score += perf['cvr'] * 200  # CVR boost (more important)
        
        # Context-specific boosts
        if user_state.segment == UserSegment.CRISIS_PARENTS:
            if user_state.urgency_score > 0.8:
                if "urgent" in creative.tags:
                    score += 5
            if "immediate" in creative.tags:
                score += 3
        
        elif user_state.segment == UserSegment.RESEARCHERS:
            if user_state.technical_level > 0.7:
                if "technical" in creative.tags:
                    score += 4
            if "comparison" in creative.tags and user_state.journey_stage == JourneyStage.CONSIDERATION:
                score += 3
        
        elif user_state.segment == UserSegment.PRICE_CONSCIOUS:
            if user_state.price_sensitivity > 0.8:
                if "free" in creative.tags or "value" in creative.tags:
                    score += 5
        
        elif user_state.segment == UserSegment.RETARGETING:
            if user_state.session_count > 3:
                if "urgent" in creative.tags:
                    score += 4
        
        # Device-specific adjustments
        if user_state.device_type == "mobile":
            if creative.creative_type in [CreativeType.BANNER, CreativeType.TEXT_AD]:
                score += 2
        else:
            if creative.creative_type in [CreativeType.VIDEO, CreativeType.CAROUSEL]:
                score += 2
        
        # Time-based adjustments
        if user_state.time_of_day in ["evening", "night"]:
            if user_state.segment == UserSegment.CRISIS_PARENTS:
                score += 3  # Parents more active in evening
        
        return score
    
    def track_impression(self, creative_id: str, user_id: str, clicked: bool = False, 
                        converted: bool = False, engagement_time: float = 0.0, 
                        cost: float = 0.0):
        """Track creative impression and update performance metrics"""
        impression = ImpressionData(
            creative_id=creative_id,
            user_id=user_id,
            timestamp=time.time(),
            clicked=clicked,
            converted=converted,
            engagement_time=engagement_time,
            cost=cost
        )
        
        self.impressions.append(impression)
        self.user_impression_history[user_id].append(creative_id)
        
        # Update performance metrics
        self._update_performance_metrics(impression)
    
    def _update_performance_metrics(self, impression: ImpressionData):
        """Update performance metrics for creative"""
        perf = self.creative_performance[impression.creative_id]
        perf['impressions'] += 1
        
        if impression.clicked:
            perf['clicks'] += 1
        
        if impression.converted:
            perf['conversions'] += 1
        
        # Recalculate rates
        if perf['impressions'] > 0:
            perf['ctr'] = perf['clicks'] / perf['impressions']
            perf['cvr'] = perf['conversions'] / perf['impressions']
    
    def calculate_fatigue(self, creative_id: str, user_id: str) -> float:
        """
        Calculate creative fatigue score for user (0.0 = fresh, 1.0 = completely fatigued)
        """
        current_time = time.time()
        cutoff_time = current_time - (self.fatigue_decay_hours * 3600)
        
        # Count recent impressions
        recent_count = sum(1 for imp in self.impressions 
                          if (imp.creative_id == creative_id and 
                              imp.user_id == user_id and 
                              imp.timestamp > cutoff_time))
        
        # Calculate fatigue score
        fatigue_score = min(recent_count / self.fatigue_threshold, 1.0)
        return fatigue_score
    
    def _get_ab_test_variant(self, user_state: UserState) -> Optional[ABTestVariant]:
        """Get A/B test variant for user if applicable"""
        if not self.ab_tests:
            return None
        
        # Use consistent hash-based assignment
        context_hash = user_state.get_context_hash()
        hash_int = int(context_hash[:8], 16)  # Use first 8 chars as int
        bucket = (hash_int % 100) / 100.0  # Convert to 0-1 range
        
        cumulative_split = 0.0
        for variant in self.ab_tests.values():
            if not variant.active:
                continue
            
            cumulative_split += variant.traffic_split
            if bucket <= cumulative_split:
                return variant
        
        return None
    
    def _apply_ab_overrides(self, candidates: List[Creative], 
                           variant: ABTestVariant) -> List[Creative]:
        """Apply A/B test overrides to candidate creatives"""
        # This is a simplified implementation
        # In practice, you might modify creative properties or filter candidates
        return candidates
    
    def _generate_selection_reason(self, creative: Creative, user_state: UserState, 
                                  ab_variant: Optional[ABTestVariant]) -> str:
        """Generate human-readable reason for creative selection"""
        reasons = []
        
        # Basic match
        reasons.append(f"Segment: {user_state.segment.value}")
        reasons.append(f"Stage: {user_state.journey_stage.value}")
        
        # A/B test
        if ab_variant:
            reasons.append(f"A/B Test: {ab_variant.name}")
        
        # Performance
        perf = self.creative_performance[creative.id]
        if perf['impressions'] > 10:
            reasons.append(f"CTR: {perf['ctr']:.2%}")
        
        # Context factors
        if user_state.urgency_score > 0.8:
            reasons.append("High urgency")
        
        if user_state.price_sensitivity > 0.8:
            reasons.append("Price sensitive")
        
        return " | ".join(reasons)
    
    def create_ab_test(self, test_name: str, variants: List[ABTestVariant]):
        """Create new A/B test with variants"""
        for variant in variants:
            self.ab_tests[f"{test_name}_{variant.variant_id}"] = variant
    
    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate performance report for creatives"""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_impressions = [imp for imp in self.impressions if imp.timestamp > cutoff_time]
        
        report = {
            'period_days': days,
            'total_impressions': len(recent_impressions),
            'total_clicks': sum(1 for imp in recent_impressions if imp.clicked),
            'total_conversions': sum(1 for imp in recent_impressions if imp.converted),
            'creative_performance': {}
        }
        
        for creative_id, creative in self.creatives.items():
            creative_imps = [imp for imp in recent_impressions if imp.creative_id == creative_id]
            
            if creative_imps:
                clicks = sum(1 for imp in creative_imps if imp.clicked)
                conversions = sum(1 for imp in creative_imps if imp.converted)
                
                report['creative_performance'][creative_id] = {
                    'headline': creative.headline,
                    'segment': creative.segment.value,
                    'impressions': len(creative_imps),
                    'clicks': clicks,
                    'conversions': conversions,
                    'ctr': clicks / len(creative_imps) if creative_imps else 0,
                    'cvr': conversions / len(creative_imps) if creative_imps else 0
                }
        
        return report
    
    def get_fatigue_analysis(self, user_id: str) -> Dict[str, float]:
        """Get fatigue scores for all creatives for a specific user"""
        return {
            creative_id: self.calculate_fatigue(creative_id, user_id)
            for creative_id in self.creatives.keys()
        }
    
    def save_config(self, filename: str):
        """Save current configuration to file"""
        config = {
            'creatives': {cid: asdict(creative) for cid, creative in self.creatives.items()},
            'ab_tests': {tid: asdict(test) for tid, test in self.ab_tests.items()},
            'settings': {
                'fatigue_threshold': self.fatigue_threshold,
                'fatigue_decay_hours': self.fatigue_decay_hours
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filename: str):
        """Load configuration from file"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Load creatives
            for cid, creative_data in config.get('creatives', {}).items():
                creative_data['segment'] = UserSegment(creative_data['segment'])
                creative_data['journey_stage'] = JourneyStage(creative_data['journey_stage'])
                creative_data['creative_type'] = CreativeType(creative_data['creative_type'])
                creative_data['landing_page'] = LandingPageType(creative_data['landing_page'])
                self.creatives[cid] = Creative(**creative_data)
            
            # Load A/B tests
            for tid, test_data in config.get('ab_tests', {}).items():
                if test_data.get('landing_page_override'):
                    test_data['landing_page_override'] = LandingPageType(test_data['landing_page_override'])
                self.ab_tests[tid] = ABTestVariant(**test_data)
            
            # Load settings
            settings = config.get('settings', {})
            self.fatigue_threshold = settings.get('fatigue_threshold', self.fatigue_threshold)
            self.fatigue_decay_hours = settings.get('fatigue_decay_hours', self.fatigue_decay_hours)
        
        except FileNotFoundError:
            print(f"Config file {filename} not found, using defaults")
        except Exception as e:
            print(f"Error loading config: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize selector
    selector = CreativeSelector()
    
    # Create example user states
    crisis_parent = UserState(
        user_id="parent_123",
        segment=UserSegment.CRISIS_PARENTS,
        journey_stage=JourneyStage.AWARENESS,
        device_type="mobile",
        time_of_day="evening",
        previous_interactions=[],
        conversion_probability=0.7,
        urgency_score=0.9,
        price_sensitivity=0.4,
        technical_level=0.3,
        session_count=1,
        last_seen=time.time()
    )
    
    researcher = UserState(
        user_id="researcher_456",
        segment=UserSegment.RESEARCHERS,
        journey_stage=JourneyStage.CONSIDERATION,
        device_type="desktop",
        time_of_day="afternoon",
        previous_interactions=["visited_comparison_page"],
        conversion_probability=0.5,
        urgency_score=0.3,
        price_sensitivity=0.6,
        technical_level=0.9,
        session_count=3,
        last_seen=time.time() - 3600
    )
    
    # Test creative selection
    print("=== Creative Selection Demo ===\n")
    
    # Crisis parent selection
    creative, reason = selector.select_creative(crisis_parent)
    print(f"Crisis Parent Creative:")
    print(f"  Selected: {creative.headline}")
    print(f"  CTA: {creative.cta}")
    print(f"  Landing Page: {creative.landing_page.value}")
    print(f"  Reason: {reason}\n")
    
    # Researcher selection
    creative, reason = selector.select_creative(researcher)
    print(f"Researcher Creative:")
    print(f"  Selected: {creative.headline}")
    print(f"  CTA: {creative.cta}")
    print(f"  Landing Page: {creative.landing_page.value}")
    print(f"  Reason: {reason}\n")
    
    # Track some impressions
    selector.track_impression(creative.id, researcher.user_id, clicked=True, engagement_time=45.0)
    
    # Test fatigue
    for i in range(4):  # Show same creative multiple times
        creative, _ = selector.select_creative(crisis_parent)
        selector.track_impression(creative.id, crisis_parent.user_id)
    
    print("=== Fatigue Analysis ===")
    fatigue = selector.get_fatigue_analysis(crisis_parent.user_id)
    for creative_id, score in fatigue.items():
        if score > 0:
            print(f"  {creative_id}: {score:.2f}")
    
    # Create A/B test
    ab_variants = [
        ABTestVariant("control", "Control Group", 0.5, {}),
        ABTestVariant("test", "New Headlines", 0.5, {"headline_boost": True})
    ]
    selector.create_ab_test("headline_test", ab_variants)
    
    print(f"\n=== A/B Test Created ===")
    print(f"Test variants: {len(selector.ab_tests)}")
    
    # Performance report
    print(f"\n=== Performance Report ===")
    report = selector.get_performance_report(1)  # Last 1 day
    print(f"Total impressions: {report['total_impressions']}")
    print(f"Total clicks: {report['total_clicks']}")
    
    # Save configuration
    selector.save_config("creative_config.json")
    print(f"\nConfiguration saved to creative_config.json")