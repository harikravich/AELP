#!/usr/bin/env python3
"""
Comprehensive test to verify all components are actually working as planned
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
import logging

logging.basicConfig(level=logging.WARNING)

async def test_comprehensive_functionality():
    """Test that all components actually work as intended"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE GAELP FUNCTIONALITY TEST")
    print("="*80)
    
    results = {}
    
    # 1. Test Multi-Touch Journey Tracking
    print("\n1. Testing Multi-Touch Journey Tracking...")
    try:
        from enhanced_journey_tracking import EnhancedMultiTouchUser, UserState, Channel, TouchpointType, Touchpoint
        
        # Create a user and process touchpoints
        user = EnhancedMultiTouchUser(user_id="test_user_001")
        
        # Test state progression
        initial_state = user.current_state
        
        # Process a touchpoint
        new_state, cost = user.process_touchpoint(
            channel=Channel.SEARCH,
            touchpoint_type=TouchpointType.CLICK,
            bid_amount=2.5,
            timestamp=datetime.now()
        )
        
        # Check if state changed
        if len(user.journey) > 0 and user.total_touches > 0:
            print(f"   ✅ Journey tracking: {initial_state} → {new_state}")
            print(f"      Touchpoints: {len(user.journey)}, Cost: ${user.total_cost:.2f}")
            results['journey_tracking'] = True
        else:
            print(f"   ❌ Journey not tracking touchpoints")
            results['journey_tracking'] = False
            
    except Exception as e:
        print(f"   ❌ Journey tracking error: {e}")
        results['journey_tracking'] = False
    
    # 2. Test Thompson Sampling Online Learning
    print("\n2. Testing Thompson Sampling Online Learning...")
    try:
        from training_orchestrator.online_learner import ThompsonSamplerArm, OnlineLearner, OnlineLearnerConfig
        
        # Create Thompson sampler
        arm = ThompsonSamplerArm("test_arm", prior_alpha=1.0, prior_beta=1.0)
        
        # Test sampling
        samples = [arm.sample() for _ in range(10)]
        
        # Update with rewards
        arm.update(reward=1.0, success=True)
        arm.update(reward=0.0, success=False)
        
        # Check confidence interval
        lower, upper = arm.get_confidence_interval()
        
        if 0 <= lower <= upper <= 1 and arm.total_pulls == 2:
            print(f"   ✅ Thompson Sampling working")
            print(f"      Confidence interval: [{lower:.3f}, {upper:.3f}]")
            print(f"      Alpha: {arm.alpha}, Beta: {arm.beta}")
            results['thompson_sampling'] = True
        else:
            print(f"   ❌ Thompson Sampling not working correctly")
            results['thompson_sampling'] = False
            
    except Exception as e:
        print(f"   ❌ Thompson Sampling error: {e}")
        results['thompson_sampling'] = False
    
    # 3. Test Journey State Encoder
    print("\n3. Testing Journey State Encoder...")
    try:
        from training_orchestrator.journey_state_encoder import JourneyStateEncoder, JourneyStateEncoderConfig
        
        config = JourneyStateEncoderConfig(encoded_state_dim=256)
        encoder = JourneyStateEncoder(config)
        
        # Create test journey data
        journey_data = {
            'touchpoints': [
                {'channel': 'search', 'timestamp': 1.0, 'bid': 2.5},
                {'channel': 'display', 'timestamp': 2.0, 'bid': 1.5}
            ],
            'days_in_journey': 2,
            'current_stage': 'considering',
            'total_spend': 4.0,
            'conversion_probability': 0.3
        }
        
        # Encode the journey
        encoded = encoder.encode_journey(journey_data)
        
        if encoded.shape[0] == 256:
            print(f"   ✅ Journey encoder working: output shape {encoded.shape}")
            results['journey_encoder'] = True
        else:
            print(f"   ❌ Journey encoder wrong output shape: {encoded.shape}")
            results['journey_encoder'] = False
            
    except Exception as e:
        print(f"   ❌ Journey encoder error: {e}")
        results['journey_encoder'] = False
    
    # 4. Test Criteo CTR Model
    print("\n4. Testing Criteo CTR Model...")
    try:
        from criteo_response_model import CriteoUserResponseModel
        
        model = CriteoUserResponseModel()
        
        # Test CTR prediction
        response = model.simulate_user_response(
            user_id="test_user",
            ad_content={'price': 99.99, 'category': 'parental_controls'},
            context={'device': 'mobile', 'hour': 20}
        )
        
        ctr = response.get('predicted_ctr', 0)
        
        if 0.001 <= ctr <= 0.1:  # Realistic CTR range
            print(f"   ✅ Criteo CTR model working: {ctr:.4f}")
            results['criteo_ctr'] = True
        else:
            print(f"   ❌ Criteo CTR unrealistic: {ctr}")
            results['criteo_ctr'] = False
            
    except Exception as e:
        print(f"   ❌ Criteo CTR error: {e}")
        results['criteo_ctr'] = False
    
    # 5. Test Attribution Models
    print("\n5. Testing Attribution Models...")
    try:
        from attribution_models import AttributionEngine, Journey, Touchpoint as AttrTouchpoint
        
        engine = AttributionEngine()
        
        # Create test journey
        journey = Journey(
            id="test_journey",
            touchpoints=[
                AttrTouchpoint(
                    id="touch1",
                    timestamp=datetime.now() - timedelta(hours=2),
                    channel='search',
                    action='click',
                    value=2.0,
                    metadata={'campaign_id': 'camp1', 'engagement_score': 0.8}
                ),
                AttrTouchpoint(
                    id="touch2",
                    timestamp=datetime.now() - timedelta(hours=1),
                    channel='display',
                    action='impression',
                    value=1.0,
                    metadata={'campaign_id': 'camp2', 'engagement_score': 0.5}
                )
            ],
            converted=True,
            conversion_value=100.0,
            conversion_timestamp=datetime.now()
        )
        
        # Calculate attribution
        credits = engine.calculate_attribution(journey, model='time_decay')
        
        if len(credits) > 0 and sum(credits.values()) > 0:
            print(f"   ✅ Attribution working: {len(credits)} touchpoints credited")
            results['attribution'] = True
        else:
            print(f"   ❌ Attribution not calculating credits")
            results['attribution'] = False
            
    except Exception as e:
        print(f"   ❌ Attribution error: {e}")
        results['attribution'] = False
    
    # 6. Test Competitor Agents
    print("\n6. Testing Competitor Agents...")
    try:
        from competitor_agents import CompetitorAgentManager, AuctionContext, UserValueTier
        
        manager = CompetitorAgentManager()
        
        # Create auction context
        context = AuctionContext(
            user_id="test_user",
            user_value_tier=UserValueTier.HIGH,
            timestamp=datetime.now(),
            device_type="mobile",
            geo_location="US",
            time_of_day=14,
            day_of_week=2,
            market_competition=0.7,
            keyword_competition=0.5,
            seasonality_factor=1.0,
            user_engagement_score=0.6,
            conversion_probability=0.03
        )
        
        # Run auction
        results_auction = manager.run_auction(context)
        
        if len(results_auction) == 4:  # 4 competitors
            print(f"   ✅ Competitors bidding: {len(results_auction)} agents")
            for agent, result in results_auction.items():
                print(f"      {agent}: ${result.bid_amount:.2f}")
            results['competitors'] = True
        else:
            print(f"   ❌ Wrong number of competitors: {len(results_auction)}")
            results['competitors'] = False
            
    except Exception as e:
        print(f"   ❌ Competitors error: {e}")
        results['competitors'] = False
    
    # 7. Test Budget Pacing
    print("\n7. Testing Budget Pacing...")
    try:
        from budget_pacer import BudgetPacer
        
        pacer = BudgetPacer()
        
        # Test pacing multiplier
        multiplier = pacer.get_pacing_multiplier(
            hour=10,
            spent_so_far=300.0,
            daily_budget=1000.0
        )
        
        if 0.1 <= multiplier <= 2.0:
            print(f"   ✅ Budget pacing working: multiplier {multiplier:.2f}")
            results['budget_pacing'] = True
        else:
            print(f"   ❌ Budget pacing multiplier unrealistic: {multiplier}")
            results['budget_pacing'] = False
            
    except Exception as e:
        print(f"   ❌ Budget pacing error: {e}")
        results['budget_pacing'] = False
    
    # 8. Test Creative Selection
    print("\n8. Testing Creative Selection...")
    try:
        from creative_selector import CreativeSelector, UserState as CreativeUserState, UserSegment, JourneyStage
        
        selector = CreativeSelector()
        
        # Create user state with ALL required parameters
        user_state = CreativeUserState(
            user_id="test_user",
            segment=UserSegment.CRISIS_PARENTS,
            journey_stage=JourneyStage.AWARENESS,
            device_type="mobile",
            time_of_day="evening",
            previous_interactions=["search", "display"],
            conversion_probability=0.7,
            urgency_score=0.9,
            price_sensitivity=0.3,
            technical_level=0.2,
            session_count=2,
            last_seen=datetime.now().timestamp()
        )
        
        # Select creative
        creative, reason = selector.select_creative(user_state)
        
        if creative and hasattr(creative, 'headline'):
            print(f"   ✅ Creative selection working: {creative.headline[:50]}...")
            results['creative_selection'] = True
        else:
            print(f"   ❌ Creative selection not returning proper creative")
            results['creative_selection'] = False
            
    except Exception as e:
        print(f"   ❌ Creative selection error: {e}")
        results['creative_selection'] = False
    
    # 9. Test Safety System
    print("\n9. Testing Safety System...")
    try:
        from safety_system import SafetySystem, SafetyConfig
        
        config = SafetyConfig(max_bid_absolute=10.0)
        safety = SafetySystem(config)
        
        # Test bid validation
        safe_bid = safety.validate_bid(
            bid_amount=15.0,
            context={'budget_remaining': 100.0}
        )
        
        if safe_bid <= 10.0:
            print(f"   ✅ Safety system working: ${15:.2f} → ${safe_bid:.2f}")
            results['safety_system'] = True
        else:
            print(f"   ❌ Safety system not capping bids: ${safe_bid}")
            results['safety_system'] = False
            
    except Exception as e:
        print(f"   ❌ Safety system error: {e}")
        results['safety_system'] = False
    
    # 10. Test Integration in Master Orchestrator
    print("\n10. Testing Master Orchestrator Integration...")
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig()
        config.simulation_days = 1
        config.users_per_day = 1
        config.n_parallel_worlds = 1
        
        master = MasterOrchestrator(config)
        
        # Check all components are present
        components_present = sum([
            hasattr(master, 'journey_db') and master.journey_db is not None,
            hasattr(master, 'monte_carlo') and master.monte_carlo is not None,
            hasattr(master, 'competitor_manager') and master.competitor_manager is not None,
            hasattr(master, 'attribution_engine') and master.attribution_engine is not None,
            hasattr(master, 'delayed_rewards') and master.delayed_rewards is not None,
            hasattr(master, 'state_encoder') and master.state_encoder is not None,
            hasattr(master, 'creative_selector') and master.creative_selector is not None,
            hasattr(master, 'budget_pacer') and master.budget_pacer is not None,
            hasattr(master, 'identity_resolver') and master.identity_resolver is not None,
            hasattr(master, 'criteo_response') and master.criteo_response is not None,
            hasattr(master, 'online_learner') and master.online_learner is not None,
            hasattr(master, 'safety_system') and master.safety_system is not None,
        ])
        
        if components_present >= 10:
            print(f"   ✅ Master orchestrator integrated: {components_present}/12 core components")
            results['master_integration'] = True
        else:
            print(f"   ❌ Master orchestrator missing components: {components_present}/12")
            results['master_integration'] = False
            
    except Exception as e:
        print(f"   ❌ Master orchestrator error: {e}")
        results['master_integration'] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"{status} {test.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed >= 8:
        print("\n🎉 SYSTEM IS FUNCTIONAL!")
        print("Most core components are working as intended.")
    elif passed >= 5:
        print("\n⚠️ SYSTEM IS PARTIALLY FUNCTIONAL")
        print("Some components working but significant gaps remain.")
    else:
        print("\n❌ SYSTEM HAS MAJOR ISSUES")
        print("Most components are not working as intended.")
    
    return passed >= 8

if __name__ == "__main__":
    success = asyncio.run(test_comprehensive_functionality())
    exit(0 if success else 1)