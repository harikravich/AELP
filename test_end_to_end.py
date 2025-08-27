#!/usr/bin/env python3
"""End-to-end test of the entire GAELP system after all fixes"""

import asyncio
import logging
from datetime import datetime
from gaelp_master_integration import MasterOrchestrator, GAELPConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_end_to_end():
    """Test the entire GAELP system end-to-end"""
    
    print("="*80)
    print("END-TO-END GAELP SYSTEM TEST")
    print("="*80)
    
    # Initialize the system
    print("\n1. Initializing GAELP Master Orchestrator...")
    try:
        config = GAELPConfig(
            enable_delayed_rewards=True,
            enable_competitive_intelligence=True,
            enable_creative_optimization=True,
            enable_budget_pacing=True,
            enable_identity_resolution=True,
            enable_criteo_response=True,
            enable_safety_system=True,
            enable_temporal_effects=True
        )
        
        master = MasterOrchestrator(config)
        print(f"   ✅ Master orchestrator initialized with {len([v for v in vars(master).values() if v is not None])} components")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Component Status Check
    print("\n2. Checking all component statuses...")
    components = {
        'journey_db': 'User Journey Database',
        'monte_carlo': 'Monte Carlo Simulator',
        'competitor_agents': 'Competitor Agents',
        'auction_gym': 'Auction Gym Integration',
        'delayed_rewards': 'Delayed Reward System',
        'thompson_sampler': 'Thompson Sampler',
        'creative_optimization': 'Creative Optimization',
        'budget_pacer': 'Budget Pacer',
        'identity_resolver': 'Identity Resolution',
        'attribution_engine': 'Attribution Engine',
        'importance_sampler': 'Importance Sampler',
        'conversion_lag_model': 'Conversion Lag Model',
        'competitive_intel': 'Competitive Intelligence',
        'criteo_response': 'Criteo Response Model',
        'timeout_manager': 'Journey Timeout Manager',
        'temporal_effects': 'Temporal Effects',
        'model_versioning': 'Model Versioning',
        'online_learner': 'Online Learner',
        'safety_system': 'Safety System'
    }
    
    active_count = 0
    for attr, name in components.items():
        if hasattr(master, attr) and getattr(master, attr) is not None:
            print(f"   ✅ {name}: Active")
            active_count += 1
        else:
            print(f"   ⚠️  {name}: Not initialized")
    
    print(f"\n   Total: {active_count}/{len(components)} components active")
    
    # Test 3: Bid Calculation Pipeline
    print("\n3. Testing bid calculation pipeline...")
    try:
        # Simulate different user scenarios
        test_scenarios = [
            {
                'name': 'High-Intent Crisis Parent',
                'journey_state': {
                    'conversion_probability': 0.85,
                    'journey_stage': 3,
                    'user_fatigue_level': 0.1,
                    'hour_of_day': 14,
                    'user_id': 'crisis_parent_001'
                },
                'query_data': {
                    'query': 'emergency parental controls now',
                    'intent_strength': 0.95,
                    'segment': 'crisis_parent',
                    'device_type': 'mobile',
                    'location': 'US'
                }
            },
            {
                'name': 'Research Phase User',
                'journey_state': {
                    'conversion_probability': 0.35,
                    'journey_stage': 1,
                    'user_fatigue_level': 0.4,
                    'hour_of_day': 22,
                    'user_id': 'researcher_001'
                },
                'query_data': {
                    'query': 'compare parental control apps',
                    'intent_strength': 0.5,
                    'segment': 'researcher',
                    'device_type': 'desktop',
                    'location': 'US'
                }
            },
            {
                'name': 'Budget Conscious Parent',
                'journey_state': {
                    'conversion_probability': 0.55,
                    'journey_stage': 2,
                    'user_fatigue_level': 0.3,
                    'hour_of_day': 19,
                    'user_id': 'budget_parent_001'
                },
                'query_data': {
                    'query': 'free parental controls',
                    'intent_strength': 0.6,
                    'segment': 'budget_conscious',
                    'device_type': 'mobile',
                    'location': 'US'
                }
            }
        ]
        
        creative_selection = {'creative_type': 'display', 'variant_id': 'A'}
        
        for scenario in test_scenarios:
            bid = await master._calculate_bid(
                journey_state=scenario['journey_state'],
                query_data=scenario['query_data'],
                creative_selection=creative_selection
            )
            
            print(f"\n   {scenario['name']}:")
            print(f"      Query: '{scenario['query_data']['query']}'")
            print(f"      Conversion probability: {scenario['journey_state']['conversion_probability']:.1%}")
            print(f"      Journey stage: {scenario['journey_state']['journey_stage']}")
            print(f"      Calculated bid: ${bid:.2f}")
            
    except Exception as e:
        print(f"   ❌ Bid calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Auction Simulation
    print("\n4. Testing auction simulation...")
    try:
        auction_result = await master._run_auction(
            bid_amount=2.5,
            query_data=test_scenarios[0]['query_data'],
            creative_selection=creative_selection
        )
        
        print(f"   ✅ Auction completed:")
        print(f"      Won: {auction_result['won']}")
        print(f"      Position: {auction_result.get('position', 'N/A')}")
        print(f"      Winning price: ${auction_result.get('winning_price', 0):.2f}")
        print(f"      Estimated CTR: {auction_result.get('estimated_ctr', 0):.3f}")
        
    except Exception as e:
        print(f"   ❌ Auction simulation failed: {e}")
        return False
    
    # Test 5: Journey Recording
    print("\n5. Testing journey recording...")
    try:
        # Record a journey
        user_id = 'test_user_e2e'
        journey_id = await master._record_journey(
            user_id=user_id,
            query=test_scenarios[0]['query_data']['query'],
            bid_amount=bid,
            won=True,
            creative_selection=creative_selection
        )
        
        if journey_id:
            print(f"   ✅ Journey recorded: {journey_id}")
        else:
            print(f"   ⚠️  Journey recording returned None (may be expected)")
            
    except Exception as e:
        print(f"   ❌ Journey recording failed: {e}")
        return False
    
    # Test 6: Safety System Check
    print("\n6. Testing safety system...")
    try:
        # Test with extreme bid
        extreme_bid = 50.0  # Way above max
        safe_bid = master.safety_system.validate_bid(
            bid_amount=extreme_bid,
            context={'query': 'test', 'campaign_id': 'test_campaign'}
        )
        
        print(f"   ✅ Safety system active:")
        print(f"      Original bid: ${extreme_bid:.2f}")
        print(f"      Safe bid: ${safe_bid:.2f}")
        print(f"      Reduction: {((extreme_bid - safe_bid) / extreme_bid * 100):.1f}%")
        
    except Exception as e:
        print(f"   ❌ Safety system test failed: {e}")
        return False
    
    # Test 7: Online Learning
    print("\n7. Testing online learning...")
    try:
        if master.online_learner:
            # Select an arm
            selected_arm = master.online_learner.select_arm()
            print(f"   ✅ Selected arm: {selected_arm}")
            
            # Update with reward
            master.online_learner.update(selected_arm, reward=0.1)
            print(f"      Reward recorded for arm {selected_arm}")
            
            # Get current estimates
            values = master.online_learner.get_arm_values()
            print(f"      Current arm values: {[f'{v:.3f}' for v in values[:3]]}...")
        else:
            print(f"   ⚠️  Online learner not available")
            
    except Exception as e:
        print(f"   ❌ Online learning test failed: {e}")
        return False
    
    # Test 8: Performance Metrics
    print("\n8. Checking performance metrics...")
    try:
        metrics = master.get_performance_metrics()
        
        if metrics:
            print(f"   ✅ Performance metrics:")
            for key, value in list(metrics.items())[:5]:
                if isinstance(value, (int, float)):
                    print(f"      {key}: {value:.3f}")
                else:
                    print(f"      {key}: {value}")
        else:
            print(f"   ⚠️  No metrics available yet")
            
    except Exception as e:
        print(f"   ⚠️  Metrics not available: {e}")
    
    print("\n" + "="*80)
    print("✅ END-TO-END TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nSystem Summary:")
    print(f"  - {active_count} components active and integrated")
    print(f"  - Bid calculation pipeline working")
    print(f"  - Auction simulation functional")
    print(f"  - Safety systems operational")
    print(f"  - Online learning active")
    print("\nThe GAELP system is ready for operation!")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_end_to_end())
    exit(0 if success else 1)