#!/usr/bin/env python3
"""
Realistic Aura Parental Controls Competitive Bidding Simulation
Simulates real-world competitive advertising environment with actual product data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from enum import Enum

# Import our existing components
from enhanced_journey_tracking import EnhancedMultiTouchUser, Channel, UserState, TouchpointType
from multi_channel_orchestrator import MultiChannelOrchestrator
from auction_gym_integration import AuctionGymWrapper, AuctionResult
from aura_campaign_simulator import AuraProduct, AuraUserSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== REALISTIC MARKET DATA ==========

class Competitor(Enum):
    """Major competitors in parental controls market"""
    QUSTODIO = "Qustodio"  # Market leader, $99/year
    BARK = "Bark"  # Strong brand, $99-$144/year
    CIRCLE = "Circle"  # Router-based, $129/year
    NORTON_FAMILY = "Norton Family"  # Part of Norton 360, $49/year
    GOOGLE_FAMILY = "Google Family Link"  # Free tier + premium
    APPLE_SCREENTIME = "Apple Screen Time"  # Built-in, free
    NET_NANNY = "Net Nanny"  # Legacy player, $54-$89/year
    KASPERSKY = "Kaspersky Safe Kids"  # $14.99/year basic

@dataclass
class CompetitorProfile:
    """Competitor bidding profile"""
    name: str
    budget_daily: float
    avg_cpc: float
    quality_score: float  # 1-10, affects ad rank
    conversion_rate: float
    ltv: float
    aggressiveness: float  # 0-1, how much they outbid

COMPETITOR_PROFILES = {
    Competitor.QUSTODIO: CompetitorProfile(
        name="Qustodio",
        budget_daily=5000,
        avg_cpc=3.50,
        quality_score=8.5,
        conversion_rate=0.045,
        ltv=198,  # $99 * 2 year retention
        aggressiveness=0.8
    ),
    Competitor.BARK: CompetitorProfile(
        name="Bark",
        budget_daily=4500,
        avg_cpc=4.20,
        quality_score=8.0,
        conversion_rate=0.040,
        ltv=216,  # $144 * 1.5 year retention
        aggressiveness=0.85
    ),
    Competitor.CIRCLE: CompetitorProfile(
        name="Circle",
        budget_daily=2500,
        avg_cpc=2.80,
        quality_score=7.0,
        conversion_rate=0.030,
        ltv=193,  # $129 * 1.5 year retention
        aggressiveness=0.6
    ),
    Competitor.NORTON_FAMILY: CompetitorProfile(
        name="Norton Family",
        budget_daily=3000,
        avg_cpc=2.20,
        quality_score=7.5,
        conversion_rate=0.035,
        ltv=98,  # $49 * 2 year retention
        aggressiveness=0.5
    ),
    Competitor.NET_NANNY: CompetitorProfile(
        name="Net Nanny",
        budget_daily=1500,
        avg_cpc=2.50,
        quality_score=6.5,
        conversion_rate=0.025,
        ltv=133,  # $89 * 1.5 year retention
        aggressiveness=0.4
    )
}

# ========== AUCTION DYNAMICS ==========

class RealAuctionEnvironment:
    """Simulates real Google Ads/Facebook Ads auction environment"""
    
    def __init__(self):
        self.competitors = COMPETITOR_PROFILES
        self.auction_wrapper = AuctionGymWrapper(max_competitors=len(self.competitors))
        
        # Real-world auction parameters
        self.reserve_price = 0.50  # Minimum bid
        self.quality_weight = 0.4  # How much quality score matters
        self.position_ctr_decay = [1.0, 0.6, 0.4, 0.3, 0.2]  # CTR by position
        
    def run_auction(self, our_bid: float, our_quality: float, 
                   channel: Channel, keyword: str, time_of_day: int) -> AuctionResult:
        """Run realistic second-price auction with quality scores"""
        
        # Get competitor bids for this context
        competitor_bids = self._get_competitor_bids(channel, keyword, time_of_day)
        
        # Calculate ad rank (bid * quality_score)
        our_ad_rank = our_bid * our_quality
        
        competitor_ranks = [
            (comp_bid * prof.quality_score, comp_bid, prof) 
            for prof, comp_bid in competitor_bids
        ]
        
        # Add our bid
        all_ranks = [(our_ad_rank, our_bid, None)] + competitor_ranks
        all_ranks.sort(reverse=True, key=lambda x: x[0])
        
        # Determine position and price
        our_position = None
        price_paid = 0
        
        for i, (rank, bid, profile) in enumerate(all_ranks):
            if profile is None:  # This is us
                our_position = i + 1
                if i < len(all_ranks) - 1:
                    # Pay just enough to beat next competitor
                    next_rank = all_ranks[i + 1][0]
                    price_paid = (next_rank / our_quality) + 0.01
                else:
                    price_paid = self.reserve_price
                break
        
        # Calculate expected CTR based on position
        if our_position and our_position <= len(self.position_ctr_decay):
            position_multiplier = self.position_ctr_decay[our_position - 1]
        else:
            position_multiplier = 0.1  # Very low CTR for positions 6+
        
        # Simulate click outcome
        base_ctr = self._get_base_ctr(channel, keyword)
        actual_ctr = base_ctr * position_multiplier * (our_quality / 10)
        clicked = np.random.random() < actual_ctr
        
        return AuctionResult(
            won=(our_position is not None and our_position <= 5),
            price_paid=price_paid if clicked else 0,
            slot_position=our_position or 99,
            total_slots=5,
            competitors=len(competitor_bids),
            estimated_ctr=actual_ctr,
            true_ctr=actual_ctr,
            outcome=clicked,
            revenue=0  # Set later if conversion happens
        )
    
    def _get_competitor_bids(self, channel: Channel, keyword: str, 
                            time_of_day: int) -> List[Tuple[CompetitorProfile, float]]:
        """Get competitor bids based on context"""
        bids = []
        
        for competitor, profile in self.competitors.items():
            # Check if competitor is active this hour
            if np.random.random() > 0.7:  # 70% chance competitor is bidding
                # Adjust bid based on keyword value
                keyword_multiplier = self._get_keyword_value(keyword)
                
                # Time of day adjustment
                tod_multiplier = 1.0
                if 20 <= time_of_day <= 22:  # Prime time
                    tod_multiplier = 1.3
                elif 2 <= time_of_day <= 6:  # Night
                    tod_multiplier = 0.5
                
                # Channel adjustment
                channel_multiplier = 1.0
                if channel == Channel.SEARCH:
                    channel_multiplier = 1.2  # Search is premium
                elif channel == Channel.DISPLAY:
                    channel_multiplier = 0.7
                
                # Calculate bid with noise
                base_bid = profile.avg_cpc * keyword_multiplier * tod_multiplier * channel_multiplier
                noise = np.random.normal(1.0, 0.15)  # 15% variance
                bid = max(self.reserve_price, base_bid * noise * profile.aggressiveness)
                
                bids.append((profile, bid))
        
        return bids
    
    def _get_keyword_value(self, keyword: str) -> float:
        """Get keyword value multiplier based on intent"""
        high_intent_keywords = [
            'best parental control app', 'parental control software',
            'monitor kids phone', 'block inappropriate content',
            'screen time limits app', 'child safety app'
        ]
        
        medium_intent_keywords = [
            'parental controls', 'kids online safety',
            'internet filter', 'family safety', 'digital wellness'
        ]
        
        if any(term in keyword.lower() for term in high_intent_keywords):
            return 1.5
        elif any(term in keyword.lower() for term in medium_intent_keywords):
            return 1.0
        else:
            return 0.7
    
    def _get_base_ctr(self, channel: Channel, keyword: str) -> float:
        """Get base CTR for channel and keyword"""
        base_ctrs = {
            Channel.SEARCH: 0.035,  # 3.5% base CTR for search
            Channel.SOCIAL: 0.015,  # 1.5% for social
            Channel.DISPLAY: 0.008,  # 0.8% for display
            Channel.VIDEO: 0.025,  # 2.5% for video
            Channel.RETARGETING: 0.045,  # 4.5% for retargeting
        }
        
        ctr = base_ctrs.get(channel, 0.01)
        
        # Adjust for keyword relevance
        keyword_mult = self._get_keyword_value(keyword)
        return ctr * (0.7 + 0.3 * keyword_mult)

# ========== REALISTIC USER BEHAVIOR ==========

@dataclass
class RealisticAuraUser:
    """Realistic Aura user with authentic parent persona"""
    user_id: str
    persona: str  # From AuraUserSimulator segments
    household_income: int
    num_children: int
    children_ages: List[int]
    tech_comfort: float  # 0-1 scale
    safety_concern: float  # 0-1 scale
    price_sensitivity: float  # 0-1 scale
    research_tendency: float  # 0-1, how much they research
    brand_loyalty: float  # 0-1, stickiness to first solution
    
    # Journey tracking
    touchpoints: List[Dict] = field(default_factory=list)
    current_state: UserState = UserState.UNAWARE
    days_in_journey: int = 0
    
    # Behavioral patterns
    peak_research_hours: List[int] = field(default_factory=list)
    preferred_channels: List[Channel] = field(default_factory=list)
    trigger_event: Optional[str] = None
    
    def __post_init__(self):
        """Initialize realistic behavior patterns"""
        # Set peak hours based on persona
        if self.persona == 'concerned_parent':
            self.peak_research_hours = [20, 21, 22]  # After kids in bed
            self.preferred_channels = [Channel.SEARCH, Channel.SOCIAL]
        elif self.persona == 'tech_savvy_parent':
            self.peak_research_hours = [12, 13, 19, 20]
            self.preferred_channels = [Channel.SEARCH, Channel.VIDEO]
        elif self.persona == 'crisis_parent':
            self.peak_research_hours = list(range(24))  # Any time
            self.preferred_channels = [Channel.SEARCH, Channel.RETARGETING]
        else:
            self.peak_research_hours = [19, 20, 21]
            self.preferred_channels = [Channel.SOCIAL, Channel.DISPLAY]
    
    def evaluate_ad(self, ad_content: Dict, channel: Channel, competitor: Optional[str]) -> float:
        """Evaluate ad relevance and quality"""
        relevance = 0.5  # Base relevance
        
        # Channel preference
        if channel in self.preferred_channels:
            relevance += 0.2
        
        # Message resonance based on persona
        if self.persona == 'concerned_parent' and 'safety' in str(ad_content).lower():
            relevance += 0.3
        elif self.persona == 'tech_savvy_parent' and 'features' in str(ad_content).lower():
            relevance += 0.2
        elif self.persona == 'crisis_parent' and 'immediate' in str(ad_content).lower():
            relevance += 0.4
        
        # Price sensitivity
        if 'free trial' in str(ad_content).lower() and self.price_sensitivity > 0.7:
            relevance += 0.2
        
        # Competitor consideration (realistic multi-option evaluation)
        if competitor and self.research_tendency > 0.6:
            relevance *= 0.8  # Reduce relevance if researching multiple options
        
        return min(1.0, relevance)
    
    def decide_conversion(self, landing_experience: Dict, price_shown: float, 
                         competitors_seen: List[str]) -> bool:
        """Realistic conversion decision"""
        base_conv_rate = {
            'concerned_parent': 0.08,
            'tech_savvy_parent': 0.05,
            'crisis_parent': 0.15,
            'new_parent': 0.03,
            'budget_conscious': 0.02
        }.get(self.persona, 0.04)
        
        # Adjust for user state
        state_multipliers = {
            UserState.UNAWARE: 0.1,
            UserState.AWARE: 0.3,
            UserState.INTERESTED: 0.6,
            UserState.CONSIDERING: 0.8,
            UserState.INTENT: 1.2
        }
        
        conv_rate = base_conv_rate * state_multipliers.get(self.current_state, 0.5)
        
        # Price impact
        if price_shown > 100 and self.price_sensitivity > 0.7:
            conv_rate *= 0.5
        elif price_shown < 50 and self.safety_concern > 0.8:
            conv_rate *= 0.7  # Too cheap might seem unreliable
        
        # Competitor impact
        if len(competitors_seen) > 3 and self.research_tendency > 0.7:
            conv_rate *= 0.6  # Analysis paralysis
        
        # Journey length impact
        if self.days_in_journey > 14:
            conv_rate *= 1.3  # More likely to convert after research
        elif self.days_in_journey < 2 and self.persona != 'crisis_parent':
            conv_rate *= 0.5  # Too early in journey
        
        return np.random.random() < conv_rate

# ========== COMPLETE SIMULATION ==========

class RealisticAuraSimulation:
    """Complete realistic simulation with all components"""
    
    def __init__(self, num_users: int = 10000, simulation_days: int = 30, 
                 daily_budget: float = 1000):
        self.num_users = num_users
        self.simulation_days = simulation_days
        self.daily_budget = daily_budget
        
        # Initialize components
        self.auction_env = RealAuctionEnvironment()
        self.aura_simulator = AuraUserSimulator()
        self.orchestrator = MultiChannelOrchestrator(budget_daily=daily_budget)
        
        # Initialize users
        self.users = self._create_realistic_users(num_users)
        
        # Tracking
        self.results = {
            'auctions': [],
            'conversions': [],
            'costs': [],
            'revenues': [],
            'competitor_wins': {comp.value: 0 for comp in Competitor}
        }
        
        # Keywords to bid on
        self.keywords = [
            'parental control app',
            'monitor kids phone',
            'screen time limits',
            'block inappropriate content',
            'kids online safety',
            'family internet filter',
            'child phone monitoring',
            'digital wellbeing kids',
            'youtube kids alternative',
            'social media monitoring'
        ]
    
    def _create_realistic_users(self, num_users: int) -> List[RealisticAuraUser]:
        """Create realistic user population"""
        users = []
        
        # Distribute users across personas based on market research
        persona_distribution = {
            'concerned_parent': 0.35,
            'tech_savvy_parent': 0.20,
            'new_parent': 0.15,
            'crisis_parent': 0.10,
            'budget_conscious': 0.20
        }
        
        for i in range(num_users):
            # Select persona
            persona = np.random.choice(
                list(persona_distribution.keys()),
                p=list(persona_distribution.values())
            )
            
            # Generate realistic demographics
            household_income = np.random.normal(75000, 25000)
            num_children = np.random.choice([1, 2, 3, 4], p=[0.3, 0.5, 0.15, 0.05])
            children_ages = sorted([np.random.randint(5, 18) for _ in range(num_children)])
            
            # Set behavioral traits based on persona
            if persona == 'concerned_parent':
                tech_comfort = np.random.uniform(0.4, 0.7)
                safety_concern = np.random.uniform(0.8, 1.0)
                price_sensitivity = np.random.uniform(0.2, 0.5)
                research_tendency = np.random.uniform(0.5, 0.8)
            elif persona == 'tech_savvy_parent':
                tech_comfort = np.random.uniform(0.7, 1.0)
                safety_concern = np.random.uniform(0.6, 0.9)
                price_sensitivity = np.random.uniform(0.4, 0.7)
                research_tendency = np.random.uniform(0.7, 1.0)
            elif persona == 'crisis_parent':
                tech_comfort = np.random.uniform(0.3, 0.8)
                safety_concern = np.random.uniform(0.9, 1.0)
                price_sensitivity = np.random.uniform(0.1, 0.4)
                research_tendency = np.random.uniform(0.2, 0.5)
            else:
                tech_comfort = np.random.uniform(0.3, 0.7)
                safety_concern = np.random.uniform(0.5, 0.8)
                price_sensitivity = np.random.uniform(0.6, 0.9)
                research_tendency = np.random.uniform(0.4, 0.7)
            
            user = RealisticAuraUser(
                user_id=f"user_{i:05d}",
                persona=persona,
                household_income=household_income,
                num_children=num_children,
                children_ages=children_ages,
                tech_comfort=tech_comfort,
                safety_concern=safety_concern,
                price_sensitivity=price_sensitivity,
                research_tendency=research_tendency,
                brand_loyalty=np.random.uniform(0.3, 0.8)
            )
            
            users.append(user)
        
        return users
    
    def run_simulation(self) -> pd.DataFrame:
        """Run complete simulation"""
        logger.info(f"Starting realistic Aura simulation: {self.num_users} users, {self.simulation_days} days")
        
        all_events = []
        
        for day in range(self.simulation_days):
            daily_spend = 0
            daily_conversions = 0
            
            # Simulate hourly auctions
            for hour in range(24):
                if daily_spend >= self.daily_budget:
                    break
                
                # Select active users this hour
                active_users = [
                    u for u in self.users 
                    if hour in u.peak_research_hours and np.random.random() < 0.1
                ]
                
                for user in active_users[:100]:  # Limit to 100 users per hour for performance
                    if daily_spend >= self.daily_budget:
                        break
                    
                    # Select keyword and channel
                    keyword = np.random.choice(self.keywords)
                    channel = np.random.choice(user.preferred_channels)
                    
                    # Determine our bid
                    our_bid = self._calculate_optimal_bid(user, channel, keyword, hour)
                    our_quality = 7.5  # Aura's quality score
                    
                    # Run auction
                    auction_result = self.auction_env.run_auction(
                        our_bid, our_quality, channel, keyword, hour
                    )
                    
                    if auction_result.won and auction_result.outcome:  # We won and got click
                        daily_spend += auction_result.price_paid
                        
                        # Update user journey
                        user.touchpoints.append({
                            'day': day,
                            'hour': hour,
                            'channel': channel.value,
                            'keyword': keyword,
                            'position': auction_result.slot_position,
                            'cpc': auction_result.price_paid,
                            'competitors': auction_result.competitors
                        })
                        
                        # Progress user state
                        user.current_state = self._progress_user_state(user.current_state)
                        user.days_in_journey = day - (user.touchpoints[0]['day'] if user.touchpoints else day)
                        
                        # Check for conversion
                        competitors_seen = [tp.get('competitor') for tp in user.touchpoints[-5:] if tp.get('competitor')]
                        
                        if user.decide_conversion({'quality': our_quality}, 180, competitors_seen):  # $180 annual
                            daily_conversions += 1
                            revenue = 180 if user.persona != 'budget_conscious' else 144
                            
                            all_events.append({
                                'day': day,
                                'hour': hour,
                                'user_id': user.user_id,
                                'persona': user.persona,
                                'channel': channel.value,
                                'keyword': keyword,
                                'event': 'conversion',
                                'revenue': revenue,
                                'cost': sum(tp['cpc'] for tp in user.touchpoints),
                                'touches': len(user.touchpoints),
                                'journey_days': user.days_in_journey,
                                'competitors_faced': auction_result.competitors
                            })
                            
                            # Reset user for new journey
                            user.current_state = UserState.UNAWARE
                            user.touchpoints = []
                        else:
                            all_events.append({
                                'day': day,
                                'hour': hour,
                                'user_id': user.user_id,
                                'persona': user.persona,
                                'channel': channel.value,
                                'keyword': keyword,
                                'event': 'click',
                                'revenue': 0,
                                'cost': auction_result.price_paid,
                                'touches': len(user.touchpoints),
                                'journey_days': user.days_in_journey,
                                'competitors_faced': auction_result.competitors
                            })
            
            logger.info(f"Day {day}: Spend=${daily_spend:.2f}, Conversions={daily_conversions}")
        
        return pd.DataFrame(all_events)
    
    def _calculate_optimal_bid(self, user: RealisticAuraUser, channel: Channel, 
                               keyword: str, hour: int) -> float:
        """Calculate optimal bid based on user value and competition"""
        # Base bid on expected value
        base_value = 180  # Annual subscription value
        
        # Adjust for persona conversion likelihood
        persona_multipliers = {
            'crisis_parent': 1.5,
            'concerned_parent': 1.2,
            'tech_savvy_parent': 1.0,
            'new_parent': 0.7,
            'budget_conscious': 0.5
        }
        
        expected_value = base_value * persona_multipliers.get(user.persona, 1.0)
        
        # Adjust for user state
        state_multipliers = {
            UserState.INTENT: 1.5,
            UserState.CONSIDERING: 1.2,
            UserState.INTERESTED: 0.8,
            UserState.AWARE: 0.5,
            UserState.UNAWARE: 0.3
        }
        
        expected_value *= state_multipliers.get(user.current_state, 0.5)
        
        # Calculate bid as percentage of expected value
        bid = expected_value * 0.02  # Bid 2% of expected value
        
        # Adjust for time of day
        if 20 <= hour <= 22:  # Prime time
            bid *= 1.3
        elif 2 <= hour <= 6:  # Night
            bid *= 0.5
        
        # Cap bid
        return min(bid, 10.0)  # Max $10 CPC
    
    def _progress_user_state(self, current_state: UserState) -> UserState:
        """Progress user through journey states"""
        progression = {
            UserState.UNAWARE: UserState.AWARE,
            UserState.AWARE: UserState.INTERESTED,
            UserState.INTERESTED: UserState.CONSIDERING,
            UserState.CONSIDERING: UserState.INTENT,
            UserState.INTENT: UserState.INTENT  # Stay here until conversion
        }
        
        # Probabilistic progression
        if np.random.random() < 0.6:  # 60% chance to progress
            return progression.get(current_state, current_state)
        return current_state
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze simulation results"""
        conversions = df[df['event'] == 'conversion']
        
        analysis = {
            'total_conversions': len(conversions),
            'total_revenue': conversions['revenue'].sum(),
            'total_cost': df['cost'].sum(),
            'avg_cac': df['cost'].sum() / max(len(conversions), 1),
            'roas': conversions['revenue'].sum() / max(df['cost'].sum(), 1),
            'avg_journey_days': conversions['journey_days'].mean() if len(conversions) > 0 else 0,
            'avg_touches': conversions['touches'].mean() if len(conversions) > 0 else 0,
            'conversion_rate': len(conversions) / df['user_id'].nunique(),
            
            # By persona
            'conversions_by_persona': conversions.groupby('persona').size().to_dict() if len(conversions) > 0 else {},
            'cac_by_persona': conversions.groupby('persona')['cost'].mean().to_dict() if len(conversions) > 0 else {},
            
            # By channel
            'conversions_by_channel': conversions.groupby('channel').size().to_dict() if len(conversions) > 0 else {},
            'cpc_by_channel': df.groupby('channel')['cost'].mean().to_dict(),
            
            # Competition analysis
            'avg_competitors_faced': df['competitors_faced'].mean(),
            'high_competition_hours': df.groupby('hour')['competitors_faced'].mean().nlargest(3).to_dict()
        }
        
        return analysis

def main():
    """Run realistic Aura simulation"""
    print("🎯 Realistic Aura Parental Controls Simulation")
    print("=" * 60)
    print("Simulating competitive bidding environment with:")
    print("  • 5 major competitors (Qustodio, Bark, Circle, etc.)")
    print("  • Real auction dynamics (second-price with quality scores)")
    print("  • Realistic parent personas and behaviors")
    print("  • Multi-touch journeys (7-14 days typical)")
    print("  • Target: $100 CAC for $180 annual subscription")
    print("=" * 60)
    
    # Run simulation
    sim = RealisticAuraSimulation(
        num_users=5000,
        simulation_days=30,
        daily_budget=1000
    )
    
    results_df = sim.run_simulation()
    analysis = sim.analyze_results(results_df)
    
    # Print results
    print("\n📊 SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total Conversions: {analysis['total_conversions']}")
    print(f"Total Revenue: ${analysis['total_revenue']:.2f}")
    print(f"Total Cost: ${analysis['total_cost']:.2f}")
    print(f"Average CAC: ${analysis['avg_cac']:.2f}")
    print(f"ROAS: {analysis['roas']:.2f}x")
    print(f"Conversion Rate: {analysis['conversion_rate']:.2%}")
    print(f"Avg Journey: {analysis['avg_journey_days']:.1f} days, {analysis['avg_touches']:.1f} touches")
    print(f"Avg Competition: {analysis['avg_competitors_faced']:.1f} competitors per auction")
    
    print("\n🎭 Conversions by Persona:")
    for persona, count in analysis['conversions_by_persona'].items():
        cac = analysis['cac_by_persona'].get(persona, 0)
        print(f"  {persona}: {count} conversions, ${cac:.2f} CAC")
    
    print("\n📱 Performance by Channel:")
    for channel, count in analysis['conversions_by_channel'].items():
        cpc = analysis['cpc_by_channel'].get(channel, 0)
        print(f"  {channel}: {count} conversions, ${cpc:.2f} avg CPC")
    
    print("\n⏰ High Competition Hours:")
    for hour, competitors in analysis['high_competition_hours'].items():
        print(f"  {hour:02d}:00 - {competitors:.1f} avg competitors")
    
    # Save detailed results
    results_df.to_csv('/home/hariravichandran/AELP/realistic_aura_results.csv', index=False)
    
    with open('/home/hariravichandran/AELP/realistic_aura_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print("\n✅ Simulation complete!")
    print("  • Detailed results: realistic_aura_results.csv")
    print("  • Analysis: realistic_aura_analysis.json")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if analysis['avg_cac'] < 100:
        print("  ✅ CAC is within target! Current strategy is working.")
    else:
        print(f"  ⚠️ CAC (${analysis['avg_cac']:.2f}) exceeds target.")
        print("  Consider:")
        print("    • Focus on high-intent keywords")
        print("    • Increase retargeting budget")
        print("    • Optimize for crisis/concerned parent personas")
    
    if analysis['roas'] < 1.5:
        print("  ⚠️ ROAS is below 1.5x. Need to improve conversion rate or reduce costs.")
    
    return results_df, analysis

if __name__ == "__main__":
    results_df, analysis = main()