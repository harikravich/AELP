#!/usr/bin/env python3
"""
Example of Updated Simulation Using RecSim-AuctionGym Bridge

This demonstrates how to replace fake users and random auction logic 
with the RecSim-AuctionGym bridge for realistic behavior.
"""

import numpy as np
from typing import Dict, Any, List
import logging

# Import the RecSim-AuctionGym bridge
try:
    from recsim_auction_bridge import RecSimAuctionBridge, UserSegment
    from recsim_user_model import RecSimUserModel
    BRIDGE_AVAILABLE = True
except ImportError:
    logging.warning("RecSim-AuctionGym bridge not available. Using fallback.")
    BRIDGE_AVAILABLE = False

class UpdatedGAELPSimulation:
    """
    Example of how to update existing simulations to use RecSim-AuctionGym bridge
    instead of random user generation and fake auction logic.
    """
    
    def __init__(self):
        # Initialize RecSim-AuctionGym bridge
        if BRIDGE_AVAILABLE:
            self.bridge = RecSimAuctionBridge()
            print("✅ Using RecSim-AuctionGym bridge for realistic simulation")
        else:
            self.bridge = None
            print("❌ Bridge not available - using fallback simulation")
        
        # Track active user sessions
        self.active_users = {}
        self.session_results = []
        
    def generate_realistic_user(self, user_id: str = None) -> Dict[str, Any]:
        """
        BEFORE: Create fake users with random.choice(['segment1', 'segment2'])
        AFTER: Use RecSimUserModel to generate users with authentic segments
        """
        if not BRIDGE_AVAILABLE:
            # Fallback to old method
            return {
                'user_id': user_id or f"fake_user_{np.random.randint(1000, 9999)}",
                'segment': np.random.choice(['impulse', 'researcher', 'loyal']),
                'fake': True
            }
        
        # NEW: Generate user through RecSim bridge
        user_id = user_id or f"recsim_user_{len(self.active_users)}"
        
        # Let the bridge generate a user with realistic RecSim 6 segments
        auction_signals = self.bridge.user_to_auction_signals(
            user_id=user_id,
            context={'hour': np.random.randint(0, 24), 'device': 'mobile'}
        )
        
        user_profile = {
            'user_id': user_id,
            'segment': auction_signals['segment'],  # One of IMPULSE_BUYER, RESEARCHER, etc.
            'suggested_bid': auction_signals['suggested_bid'],
            'quality_score': auction_signals['quality_score'],
            'participation_prob': auction_signals['participation_probability'],
            'interest_level': auction_signals['interest_level'],
            'fatigue_level': auction_signals['fatigue_level'],
            'price_sensitivity': auction_signals['price_sensitivity']
        }
        
        self.active_users[user_id] = user_profile
        return user_profile
    
    def run_realistic_auction(self, user_id: str, bid: float, 
                             product_category: str = "shoes") -> Dict[str, Any]:
        """
        BEFORE: Random auction logic with fixed competitors
        AFTER: Use AuctionGym calls through the bridge for realistic competition
        """
        if not BRIDGE_AVAILABLE:
            # Fallback to old random auction
            return {
                'won': np.random.random() < 0.3,
                'price_paid': bid * np.random.uniform(0.5, 0.9),
                'position': np.random.randint(1, 6),
                'fake': True
            }
        
        # NEW: Generate realistic query based on user state
        query_data = self.bridge.generate_query_from_state(
            user_id=user_id,
            product_category=product_category
        )
        
        # Get user-specific auction signals
        context = {
            'hour': np.random.randint(0, 24),
            'device': np.random.choice(['mobile', 'desktop']),
            'query': query_data['query'],
            'brand_query': query_data['brand_query']
        }
        
        auction_signals = self.bridge.user_to_auction_signals(user_id, context)
        
        # Use map_segment_to_bid_value for realistic bidding
        user_segment = UserSegment(auction_signals['segment'])
        optimal_bid = self.bridge.map_segment_to_bid_value(
            segment=user_segment,
            query_intent=query_data['intent'],
            market_context=context
        )
        
        # Simulate auction participation based on user behavior
        if np.random.random() > auction_signals['participation_probability']:
            return {
                'won': False,
                'price_paid': 0,
                'reason': 'User did not participate in auction',
                'query_generated': query_data['query'],
                'journey_stage': query_data['journey_stage']
            }
        
        # Use the actual auction wrapper through the bridge
        auction_result = self.bridge.auction_wrapper.run_auction(
            your_bid=min(bid, optimal_bid * 1.2),  # Cap at 120% of optimal
            your_quality_score=auction_signals['quality_score'],
            context=context
        )
        
        return {
            'won': auction_result.won,
            'price_paid': auction_result.price_paid,
            'position': auction_result.slot_position,
            'competitors': auction_result.competitors,
            'estimated_ctr': auction_result.estimated_ctr,
            'outcome': auction_result.outcome,
            'revenue': auction_result.revenue,
            'query_generated': query_data['query'],
            'journey_stage': query_data['journey_stage'],
            'user_segment': auction_signals['segment'],
            'intent_strength': query_data['intent_strength']
        }
    
    def simulate_user_session(self, num_interactions: int = 5) -> Dict[str, Any]:
        """
        BEFORE: Random user interactions without realistic journey progression
        AFTER: Use RecSim bridge to simulate complete user sessions with journey stages
        """
        if not BRIDGE_AVAILABLE:
            # Old way - random interactions
            results = {
                'interactions': num_interactions,
                'conversions': np.random.randint(0, 2),
                'total_cost': np.random.uniform(10, 100),
                'total_revenue': np.random.uniform(0, 200),
                'method': 'random_fallback'
            }
            return results
        
        # NEW: Use bridge's simulate_user_auction_session
        user_id = f"session_{len(self.session_results)}"
        
        session_result = self.bridge.simulate_user_auction_session(
            user_id=user_id,
            num_queries=num_interactions,
            product_category="shoes"  # Can be parameterized
        )
        
        # Extract insights
        analytics = {
            'interactions': len(session_result['queries']),
            'auctions_participated': len(session_result['auctions']),
            'conversions': session_result['conversions'],
            'total_cost': session_result['total_cost'],
            'total_revenue': session_result['total_revenue'],
            'roas': session_result.get('roas', 0),
            'queries_generated': [q['query'] for q in session_result['queries']],
            'journey_stages': [q['journey_stage'] for q in session_result['queries']],
            'user_segment': session_result['queries'][0]['user_segment'] if session_result['queries'] else 'unknown',
            'method': 'recsim_bridge'
        }
        
        self.session_results.append(analytics)
        return analytics
    
    def run_training_simulation(self, episodes: int = 100) -> Dict[str, Any]:
        """
        BEFORE: Training loops with random user generation
        AFTER: Realistic training with RecSim 6 segments flowing through to auction participation
        """
        print(f"🎯 Running {episodes} episodes with {'RecSim-AuctionGym bridge' if BRIDGE_AVAILABLE else 'fallback simulation'}")
        
        results = {
            'episodes': episodes,
            'total_conversions': 0,
            'total_cost': 0,
            'total_revenue': 0,
            'segment_performance': {},
            'journey_insights': {}
        }
        
        for episode in range(episodes):
            # Generate user for this episode
            user = self.generate_realistic_user()
            
            # Run session for this user
            session = self.simulate_user_session(num_interactions=np.random.randint(1, 8))
            
            # Update results
            results['total_conversions'] += session['conversions']
            results['total_cost'] += session['total_cost']
            results['total_revenue'] += session['total_revenue']
            
            # Track segment performance (NEW)
            if BRIDGE_AVAILABLE and session['method'] == 'recsim_bridge':
                segment = session['user_segment']
                if segment not in results['segment_performance']:
                    results['segment_performance'][segment] = {
                        'episodes': 0,
                        'conversions': 0,
                        'cost': 0,
                        'revenue': 0
                    }
                
                results['segment_performance'][segment]['episodes'] += 1
                results['segment_performance'][segment]['conversions'] += session['conversions']
                results['segment_performance'][segment]['cost'] += session['total_cost']
                results['segment_performance'][segment]['revenue'] += session['total_revenue']
                
                # Track journey stage insights (NEW)
                for stage in session.get('journey_stages', []):
                    results['journey_insights'][stage] = results['journey_insights'].get(stage, 0) + 1
        
        # Calculate summary metrics
        results['avg_roas'] = results['total_revenue'] / max(results['total_cost'], 0.01)
        results['conversion_rate'] = results['total_conversions'] / episodes
        
        return results
    
    def get_bridge_analytics(self) -> Dict[str, Any]:
        """Get detailed analytics from the RecSim-AuctionGym bridge"""
        if not BRIDGE_AVAILABLE:
            return {'message': 'Bridge analytics not available'}
        
        return self.bridge.get_bridge_analytics()


def demonstrate_integration():
    """Demonstrate the key integration points"""
    print("🔄 RecSim-AuctionGym Bridge Integration Demo")
    print("=" * 60)
    
    sim = UpdatedGAELPSimulation()
    
    # 1. Show user generation differences
    print("\n1️⃣ USER GENERATION:")
    print("-" * 30)
    
    for i in range(3):
        user = sim.generate_realistic_user(f"demo_user_{i}")
        if BRIDGE_AVAILABLE:
            print(f"User {i+1}: {user['segment']} (Bid: ${user['suggested_bid']:.2f}, Interest: {user['interest_level']:.2f})")
        else:
            print(f"User {i+1}: {user['segment']} (Fallback)")
    
    # 2. Show auction differences
    print("\n2️⃣ AUCTION SIMULATION:")
    print("-" * 30)
    
    test_user = sim.generate_realistic_user("auction_test")
    auction_result = sim.run_realistic_auction("auction_test", bid=2.50)
    
    if BRIDGE_AVAILABLE:
        print(f"Query Generated: '{auction_result['query_generated']}'")
        print(f"Journey Stage: {auction_result['journey_stage']}")
        print(f"Won Auction: {auction_result['won']}")
        print(f"Position: {auction_result['position']}")
        print(f"Competitors: {auction_result['competitors']}")
        print(f"User Segment: {auction_result['user_segment']}")
    else:
        print(f"Fallback auction result: {auction_result}")
    
    # 3. Run training simulation
    print("\n3️⃣ TRAINING SIMULATION:")
    print("-" * 30)
    
    results = sim.run_training_simulation(episodes=20)
    
    print(f"Episodes: {results['episodes']}")
    print(f"Total Conversions: {results['total_conversions']}")
    print(f"Average ROAS: {results['avg_roas']:.2f}x")
    print(f"Conversion Rate: {results['conversion_rate']:.2%}")
    
    if BRIDGE_AVAILABLE:
        print("\n📊 RecSim Segment Performance:")
        for segment, perf in results['segment_performance'].items():
            if perf['episodes'] > 0:
                segment_roas = perf['revenue'] / max(perf['cost'], 0.01)
                print(f"  {segment}: {perf['episodes']} episodes, {segment_roas:.2f}x ROAS")
        
        print("\n🎯 Journey Stage Distribution:")
        total_stages = sum(results['journey_insights'].values())
        for stage, count in results['journey_insights'].items():
            percentage = count / total_stages * 100 if total_stages > 0 else 0
            print(f"  {stage}: {count} ({percentage:.1f}%)")
        
        # Show bridge analytics
        bridge_analytics = sim.get_bridge_analytics()
        if 'total_sessions' in bridge_analytics:
            print(f"\n🔍 Bridge Analytics:")
            print(f"  Total Sessions: {bridge_analytics['total_sessions']}")
            print(f"  Query Generation Stats: {len(bridge_analytics['query_generation_stats'])} segments tracked")
    
    print("\n✅ Integration demonstration completed!")
    
    return results


if __name__ == "__main__":
    results = demonstrate_integration()
    
    print("\n" + "="*60)
    print("📋 INTEGRATION SUMMARY:")
    print("="*60)
    print("Key changes made to use RecSim-AuctionGym bridge:")
    print("1️⃣ Replaced fake user generation with RecSimUserModel")
    print("2️⃣ Replaced random auction logic with actual AuctionGym calls")
    print("3️⃣ Used user_to_auction_signals() for realistic bidding")
    print("4️⃣ Used generate_query_from_state() for authentic queries")
    print("5️⃣ Connected map_segment_to_bid_value() to bidding decisions")
    print("6️⃣ RecSim 6 segments now flow through to auction participation")
    print()
    if BRIDGE_AVAILABLE:
        print("✅ RecSim-AuctionGym bridge is working!")
        print("   - IMPULSE_BUYER, RESEARCHER, LOYAL_CUSTOMER, etc. are active")
        print("   - Realistic queries generated based on journey stage")
        print("   - Auction participation based on user segment behavior")
    else:
        print("⚠️ RecSim-AuctionGym bridge not available")
        print("   - Install dependencies: recsim_user_model.py, recsim_auction_bridge.py")
        print("   - Check that auction_gym_integration.py is available")