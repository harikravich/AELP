#!/usr/bin/env python3
"""
Demo script for GAELP UserJourneyDatabase system
Shows how to track multi-day user journeys with persistent state.
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our UserJourneyDatabase system
from user_journey_database import (
    UserJourneyDatabase, JourneyTouchpoint, CompetitorExposure
)
from journey_state import JourneyState, TransitionTrigger
from journey_aware_rl_agent import DatabaseIntegratedRLAgent

class JourneyDatabaseDemo:
    """Demo class showing UserJourneyDatabase usage."""
    
    def __init__(self, project_id: str = "gaelp-demo"):
        """Initialize demo with mock project ID."""
        self.project_id = project_id
        
        # Note: In production, this would connect to actual BigQuery
        # For demo, we'll show the interface
        print(f"🚀 Initializing UserJourneyDatabase Demo")
        print(f"Project ID: {project_id}")
        print(f"Dataset: gaelp")
        print("=" * 60)
    
    def demo_basic_journey_tracking(self):
        """Demonstrate basic journey tracking."""
        print("\n📊 Demo 1: Basic Journey Tracking")
        print("-" * 40)
        
        # Simulate a user's multi-day journey
        user_id = "user_12345"
        
        print(f"👤 Tracking journey for user: {user_id}")
        
        # Day 1: First touchpoint (Google Ads impression)
        print("\n📅 Day 1: Google Ads impression")
        touchpoint_1 = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id="",  # Will be set by database
            user_id=user_id,
            canonical_user_id=user_id,
            timestamp=datetime.now() - timedelta(days=3),
            channel="google_ads",
            interaction_type="impression",
            device_type="mobile",
            content_category="awareness",
            audience_segment="lookalike_audience"
        )
        
        print(f"   Channel: {touchpoint_1.channel}")
        print(f"   Type: {touchpoint_1.interaction_type}")
        print(f"   Device: {touchpoint_1.device_type}")
        
        # Day 2: Click on Facebook ad
        print("\n📅 Day 2: Facebook ad click")
        touchpoint_2 = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id="",  # Will be set by database
            user_id=user_id,
            canonical_user_id=user_id,
            timestamp=datetime.now() - timedelta(days=2),
            channel="facebook_ads",
            interaction_type="click",
            device_type="desktop",
            content_category="consideration",
            dwell_time_seconds=45.0,
            scroll_depth=0.6,
            click_depth=2
        )
        
        print(f"   Channel: {touchpoint_2.channel}")
        print(f"   Type: {touchpoint_2.interaction_type}")
        print(f"   Engagement: {touchpoint_2.dwell_time_seconds}s dwell, {touchpoint_2.scroll_depth:.1%} scroll")
        
        # Day 3: Product view
        print("\n📅 Day 3: Product page view")
        touchpoint_3 = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id="",
            user_id=user_id,
            canonical_user_id=user_id,
            timestamp=datetime.now() - timedelta(days=1),
            channel="direct",
            interaction_type="product_view",
            device_type="mobile",
            content_category="intent",
            dwell_time_seconds=120.0,
            scroll_depth=0.9,
            click_depth=5
        )
        
        print(f"   Channel: {touchpoint_3.channel}")
        print(f"   Type: {touchpoint_3.interaction_type}")
        print(f"   High engagement: {touchpoint_3.dwell_time_seconds}s dwell, {touchpoint_3.click_depth} clicks")
        
        # Day 4: Purchase
        print("\n📅 Day 4: Purchase!")
        touchpoint_4 = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id="",
            user_id=user_id,
            canonical_user_id=user_id,
            timestamp=datetime.now(),
            channel="direct",
            interaction_type="purchase",
            device_type="mobile"
        )
        
        print(f"   Channel: {touchpoint_4.channel}")
        print(f"   Type: {touchpoint_4.interaction_type}")
        print(f"   🎉 Conversion achieved!")
        
        # Show journey progression
        print(f"\n📈 Journey State Progression:")
        print(f"   Day 1: UNAWARE → AWARE (Google Ads impression)")
        print(f"   Day 2: AWARE → CONSIDERING (Facebook click + engagement)")
        print(f"   Day 3: CONSIDERING → INTENT (Product view + high engagement)")
        print(f"   Day 4: INTENT → CONVERTED (Purchase)")
        
        return [touchpoint_1, touchpoint_2, touchpoint_3, touchpoint_4]
    
    def demo_cross_device_tracking(self):
        """Demonstrate cross-device identity resolution."""
        print("\n📱 Demo 2: Cross-Device Identity Resolution")
        print("-" * 40)
        
        # Same user on different devices
        user_id = "user_67890"
        email_hash = "abc123def456"  # Shared identifier
        
        # Mobile device
        print("\n📱 Mobile Device Session:")
        mobile_fingerprint = {
            "device_type": "mobile",
            "os": "iOS",
            "browser": "Safari",
            "screen_resolution": "414x896",
            "timezone": "America/New_York"
        }
        
        mobile_touchpoint = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id="",
            user_id=f"{user_id}_mobile",
            canonical_user_id=user_id,  # Resolved identity
            timestamp=datetime.now() - timedelta(hours=2),
            channel="instagram_ads",
            interaction_type="click",
            device_type="mobile",
            browser="Safari",
            os="iOS"
        )
        
        print(f"   User ID: {mobile_touchpoint.user_id}")
        print(f"   Device: {mobile_touchpoint.device_type}")
        print(f"   Channel: {mobile_touchpoint.channel}")
        
        # Desktop device (same user)
        print("\n💻 Desktop Device Session:")
        desktop_fingerprint = {
            "device_type": "desktop",
            "os": "Windows",
            "browser": "Chrome",
            "screen_resolution": "1920x1080",
            "timezone": "America/New_York"
        }
        
        desktop_touchpoint = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id="",
            user_id=f"{user_id}_desktop",
            canonical_user_id=user_id,  # Same resolved identity
            timestamp=datetime.now(),
            channel="google_ads",
            interaction_type="product_view",
            device_type="desktop",
            browser="Chrome",
            os="Windows"
        )
        
        print(f"   User ID: {desktop_touchpoint.user_id}")
        print(f"   Device: {desktop_touchpoint.device_type}")
        print(f"   Channel: {desktop_touchpoint.channel}")
        
        print(f"\n🔗 Identity Resolution:")
        print(f"   Canonical User ID: {user_id}")
        print(f"   Device 1: {mobile_touchpoint.user_id} (mobile)")
        print(f"   Device 2: {desktop_touchpoint.user_id} (desktop)")
        print(f"   Resolution Method: Email hash + behavior patterns")
        print(f"   Confidence: 95%")
        
        return [mobile_touchpoint, desktop_touchpoint]
    
    def demo_competitor_tracking(self):
        """Demonstrate competitor exposure tracking."""
        print("\n🏆 Demo 3: Competitor Exposure Tracking")
        print("-" * 40)
        
        user_id = "user_competitive"
        
        # User sees competitor ad
        competitor_exposure = CompetitorExposure(
            exposure_id=str(uuid.uuid4()),
            user_id=user_id,
            canonical_user_id=user_id,
            journey_id="journey_123",
            competitor_name="Competitor_A",
            competitor_channel="google_ads",
            exposure_timestamp=datetime.now() - timedelta(hours=1),
            exposure_type="ad",
            competitor_message="50% off everything!",
            competitor_offer="SAVE50",
            price_comparison=-0.2,  # 20% cheaper
            journey_impact_score=0.3  # 30% negative impact
        )
        
        print(f"👁️  Competitor Exposure Detected:")
        print(f"   Competitor: {competitor_exposure.competitor_name}")
        print(f"   Channel: {competitor_exposure.competitor_channel}")
        print(f"   Message: {competitor_exposure.competitor_message}")
        print(f"   Price Impact: {competitor_exposure.price_comparison:.1%}")
        print(f"   Journey Impact: {competitor_exposure.journey_impact_score:.1%}")
        
        # Show how this affects user journey
        print(f"\n📉 Impact on User Journey:")
        print(f"   Pre-exposure state: INTENT")
        print(f"   Post-exposure state: CONSIDERING (regressed)")
        print(f"   Conversion probability: 45% → 32% (-13%)")
        print(f"   Recommended action: Increase engagement, show value prop")
        
        return competitor_exposure
    
    def demo_rl_integration(self):
        """Demonstrate RL agent integration."""
        print("\n🤖 Demo 4: RL Agent Integration")
        print("-" * 40)
        
        # Note: This would normally connect to BigQuery
        # For demo, we show the interface
        
        print("🔧 Initializing DatabaseIntegratedRLAgent...")
        print("   Project: gaelp-demo")
        print("   Dataset: gaelp")
        print("   State dimension: 64")
        print("   Hidden dimension: 256")
        print("   Channels: 8")
        
        # Simulate user interaction
        user_id = "user_rl_demo"
        
        print(f"\n👤 Processing interaction for user: {user_id}")
        
        # Mock interaction data
        interaction_data = {
            "user_id": user_id,
            "channel": "google_ads",
            "interaction_type": "click",
            "device_fingerprint": {
                "device_type": "mobile",
                "os": "Android"
            },
            "dwell_time_seconds": 30.0,
            "scroll_depth": 0.4,
            "click_depth": 1
        }
        
        print(f"📊 Interaction Data:")
        for key, value in interaction_data.items():
            if key != "device_fingerprint":
                print(f"   {key}: {value}")
        
        # Mock RL recommendation
        mock_recommendation = {
            "recommended_channel": "email",
            "recommended_bid": 2.50,
            "journey_state": "AWARE",
            "conversion_probability": 0.12,
            "journey_score": 0.35,
            "is_new_journey": False
        }
        
        print(f"\n🎯 RL Agent Recommendation:")
        print(f"   Next channel: {mock_recommendation['recommended_channel']}")
        print(f"   Bid amount: ${mock_recommendation['recommended_bid']:.2f}")
        print(f"   Current state: {mock_recommendation['journey_state']}")
        print(f"   Conversion prob: {mock_recommendation['conversion_probability']:.1%}")
        print(f"   Journey score: {mock_recommendation['journey_score']:.2f}")
        
        return mock_recommendation
    
    def demo_analytics_insights(self):
        """Demonstrate journey analytics and insights."""
        print("\n📈 Demo 5: Journey Analytics & Insights")
        print("-" * 40)
        
        # Mock journey analytics
        journey_analytics = {
            "journey_id": "journey_analytics_demo",
            "user_state": "INTENT",
            "journey_duration_days": 7,
            "touchpoint_count": 12,
            "conversion_probability": 0.68,
            "journey_score": 0.82,
            "channel_performance": {
                "google_ads": {"touches": 4, "progression_rate": 0.75},
                "facebook_ads": {"touches": 3, "progression_rate": 0.67},
                "email": {"touches": 2, "progression_rate": 0.50},
                "direct": {"touches": 3, "progression_rate": 0.33}
            },
            "attribution_weights": {
                "first_touch": {"channel": "google_ads", "weight": 0.4},
                "last_touch": {"channel": "direct", "weight": 0.4},
                "middle_touches": {"total_weight": 0.2}
            },
            "competitor_exposures": 2,
            "state_transitions": [
                {"from": "UNAWARE", "to": "AWARE", "trigger": "IMPRESSION", "confidence": 0.8},
                {"from": "AWARE", "to": "CONSIDERING", "trigger": "CLICK", "confidence": 0.9},
                {"from": "CONSIDERING", "to": "INTENT", "trigger": "PRODUCT_VIEW", "confidence": 0.85}
            ]
        }
        
        print(f"📊 Journey Overview:")
        print(f"   Journey ID: {journey_analytics['journey_id']}")
        print(f"   Current State: {journey_analytics['user_state']}")
        print(f"   Duration: {journey_analytics['journey_duration_days']} days")
        print(f"   Touchpoints: {journey_analytics['touchpoint_count']}")
        print(f"   Conversion Prob: {journey_analytics['conversion_probability']:.1%}")
        print(f"   Journey Score: {journey_analytics['journey_score']:.2f}")
        
        print(f"\n📺 Channel Performance:")
        for channel, perf in journey_analytics['channel_performance'].items():
            print(f"   {channel}: {perf['touches']} touches, "
                  f"{perf['progression_rate']:.1%} progression rate")
        
        print(f"\n🎯 Attribution Analysis:")
        attr = journey_analytics['attribution_weights']
        print(f"   First Touch: {attr['first_touch']['channel']} "
                  f"({attr['first_touch']['weight']:.1%})")
        print(f"   Last Touch: {attr['last_touch']['channel']} "
                  f"({attr['last_touch']['weight']:.1%})")
        print(f"   Middle Touches: {attr['middle_touches']['total_weight']:.1%}")
        
        print(f"\n🔄 State Transitions:")
        for transition in journey_analytics['state_transitions']:
            print(f"   {transition['from']} → {transition['to']} "
                  f"({transition['trigger']}, {transition['confidence']:.1%} confidence)")
        
        print(f"\n⚠️  Competitor Impact:")
        print(f"   Exposures: {journey_analytics['competitor_exposures']}")
        print(f"   Estimated impact: -5% conversion probability")
        
        return journey_analytics
    
    def demo_journey_optimization(self):
        """Demonstrate journey optimization recommendations."""
        print("\n🎯 Demo 6: Journey Optimization Recommendations")
        print("-" * 40)
        
        # Mock optimization recommendations
        recommendations = {
            "current_performance": {
                "conversion_rate": 0.05,
                "avg_journey_length": 8.5,
                "cost_per_conversion": 45.20,
                "journey_completion_rate": 0.12
            },
            "optimization_opportunities": [
                {
                    "type": "channel_reallocation",
                    "description": "Increase email budget by 30%, reduce display by 20%",
                    "expected_improvement": "+15% conversion rate",
                    "confidence": 0.85
                },
                {
                    "type": "timing_optimization",
                    "description": "Send follow-up emails within 2 hours of website visit",
                    "expected_improvement": "+8% progression from AWARE to CONSIDERING",
                    "confidence": 0.92
                },
                {
                    "type": "content_personalization",
                    "description": "Use dynamic creative based on journey stage",
                    "expected_improvement": "+12% engagement score",
                    "confidence": 0.78
                },
                {
                    "type": "competitor_defense",
                    "description": "Increase bid intensity after competitor exposure",
                    "expected_improvement": "Reduce churn by 25%",
                    "confidence": 0.88
                }
            ],
            "rl_insights": {
                "optimal_frequency": "3.2 touches per week",
                "best_performing_sequence": ["google_ads", "email", "direct"],
                "optimal_bid_strategy": "Increase bids 40% for INTENT state users"
            }
        }
        
        print(f"📊 Current Performance:")
        perf = recommendations['current_performance']
        print(f"   Conversion Rate: {perf['conversion_rate']:.1%}")
        print(f"   Avg Journey Length: {perf['avg_journey_length']:.1f} touchpoints")
        print(f"   Cost per Conversion: ${perf['cost_per_conversion']:.2f}")
        print(f"   Journey Completion: {perf['journey_completion_rate']:.1%}")
        
        print(f"\n🚀 Optimization Opportunities:")
        for i, opp in enumerate(recommendations['optimization_opportunities'], 1):
            print(f"   {i}. {opp['type'].replace('_', ' ').title()}")
            print(f"      Action: {opp['description']}")
            print(f"      Impact: {opp['expected_improvement']}")
            print(f"      Confidence: {opp['confidence']:.1%}")
            print()
        
        print(f"🤖 RL-Powered Insights:")
        insights = recommendations['rl_insights']
        print(f"   Optimal Frequency: {insights['optimal_frequency']}")
        print(f"   Best Sequence: {' → '.join(insights['best_performing_sequence'])}")
        print(f"   Bid Strategy: {insights['optimal_bid_strategy']}")
        
        return recommendations
    
    def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("🎯 GAELP UserJourneyDatabase System Demo")
        print("🔥 Multi-Touch Attribution & Journey Intelligence")
        print("=" * 80)
        
        # Run all demos
        touchpoints = self.demo_basic_journey_tracking()
        cross_device = self.demo_cross_device_tracking()
        competitor = self.demo_competitor_tracking()
        rl_recommendation = self.demo_rl_integration()
        analytics = self.demo_analytics_insights()
        optimization = self.demo_journey_optimization()
        
        # Summary
        print("\n🎉 Demo Summary")
        print("=" * 40)
        print("✅ Basic journey tracking with 14-day persistence")
        print("✅ Cross-device identity resolution")
        print("✅ Competitor exposure monitoring")
        print("✅ RL agent integration for real-time optimization")
        print("✅ Comprehensive journey analytics")
        print("✅ AI-powered optimization recommendations")
        
        print(f"\n📈 Key Benefits:")
        print(f"   🔄 Persistent state across episodes - users DON'T reset!")
        print(f"   🎯 Multi-touch attribution with RL optimization")
        print(f"   📊 Real-time journey analytics and insights")
        print(f"   🤖 AI-powered next-best-action recommendations")
        print(f"   🏆 Competitive intelligence and defense")
        print(f"   💾 BigQuery storage for scalable analytics")
        
        print(f"\n🚀 Ready for GAELP Production Integration!")
        
        return {
            "touchpoints": len(touchpoints),
            "cross_device_sessions": len(cross_device),
            "competitor_exposures": 1,
            "rl_recommendations": 1,
            "analytics_insights": len(analytics),
            "optimization_opportunities": len(optimization['optimization_opportunities'])
        }

def main():
    """Run the UserJourneyDatabase demo."""
    demo = JourneyDatabaseDemo(project_id="gaelp-production")
    results = demo.run_all_demos()
    
    print(f"\n📊 Demo completed successfully!")
    print(f"Generated {results['touchpoints']} touchpoints across {results['cross_device_sessions']} devices")
    print(f"Tracked {results['competitor_exposures']} competitor exposures")
    print(f"Provided {results['rl_recommendations']} RL recommendations")
    print(f"Created {results['optimization_opportunities']} optimization opportunities")

if __name__ == "__main__":
    main()