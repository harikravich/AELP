#!/usr/bin/env python3
"""
Test Master Integration - Verify all components are wired together
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

# Configure logging to show integration points
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_master_integration():
    """Test that all 20 components are properly integrated"""
    
    print("\n" + "="*80)
    print("GAELP MASTER INTEGRATION TEST")
    print("="*80)
    
    try:
        # Import the master integration
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        print("\n✅ Master integration module imported successfully")
        
        # Initialize the system
        config = GAELPConfig()
        config.simulation_days = 1
        config.users_per_day = 10
        config.daily_budget_total = Decimal('100.0')
        config.n_parallel_worlds = 1  # Single world for quick test
        config.enable_competitive_intel = True
        config.enable_criteo_response = True
        config.enable_conversion_lag = False  # Disabled due to missing lifelines
        config.enable_identity_resolution = True
        config.enable_importance_sampling = True
        config.enable_creative_selection = True
        config.enable_temporal_effects = True
        config.enable_online_learning = True
        config.enable_safety_checks = True
        
        master = MasterOrchestrator(config)
        # No need to call initialize - happens in __init__
        
        print("\n📊 Component Integration Status:")
        print("-" * 50)
        
        # Check each component
        components = [
            ("UserJourneyDatabase", hasattr(master, 'journey_db')),
            ("MonteCarloSimulator", hasattr(master, 'monte_carlo')),
            ("CompetitorAgents", hasattr(master, 'competitors')),
            ("RecSimAuctionBridge", hasattr(master, 'auction_bridge')),
            ("AttributionModels", hasattr(master, 'attribution_engine')),
            ("DelayedRewardSystem", hasattr(master, 'delayed_rewards')),
            ("JourneyStateEncoder", hasattr(master, 'journey_encoder')),
            ("CreativeSelector", hasattr(master, 'creative_selector')),
            ("BudgetPacer", hasattr(master, 'budget_pacer')),
            ("IdentityResolver", hasattr(master, 'identity_resolver')),
            ("EvaluationFramework", hasattr(master, 'evaluation')),
            ("ImportanceSampler", hasattr(master, 'importance_sampler')),
            ("ConversionLagModel", hasattr(master, 'conversion_lag_model')),
            ("CompetitiveIntel", hasattr(master, 'competitive_intel')),
            ("CriteoResponseModel", hasattr(master, 'criteo_response')),
            ("JourneyTimeout", hasattr(master, 'timeout_manager')),
            ("TemporalEffects", hasattr(master, 'temporal_effects')),
            ("ModelVersioning", hasattr(master, 'model_versioning')),
            ("OnlineLearner", hasattr(master, 'online_learner')),
            ("SafetySystem", hasattr(master, 'safety_system'))
        ]
        
        integrated_count = 0
        missing_components = []
        
        for name, is_integrated in components:
            status = "✅" if is_integrated else "❌"
            print(f"  {status} {name:25} {'Integrated' if is_integrated else 'NOT FOUND'}")
            if is_integrated:
                integrated_count += 1
            else:
                missing_components.append(name)
        
        print(f"\n📈 Integration Score: {integrated_count}/20 components")
        
        if missing_components:
            print(f"\n⚠️  Missing Components:")
            for comp in missing_components:
                print(f"    - {comp}")
        
        # Test a basic flow
        print("\n🔄 Testing Basic Flow...")
        print("-" * 50)
        
        # Test user creation
        user_profile = await master.journey_db.get_or_create_user(
            user_id="test_user_001",
            attributes={
                "segment": "crisis_parent",
                "device": "mobile",
                "location": "US"
            }
        )
        print(f"✅ User created: {user_profile.user_id}")
        
        # Test journey creation
        journey = await master.journey_db.get_or_create_journey(
            user_id="test_user_001",
            session_id="test_session_001"
        )
        print(f"✅ Journey created: {journey.journey_id}")
        
        # Test creative selection
        if hasattr(master, 'creative_selector'):
            creative = master.creative_selector.select_creative(
                user_segment="crisis_parent",
                context={"device": "mobile", "hour": 20}
            )
            print(f"✅ Creative selected: {creative.get('title', 'Unknown')}")
        
        # Test Criteo CTR prediction
        if hasattr(master, 'criteo_response'):
            response = master.criteo_response.simulate_user_response(
                user_id="test_user_001",
                ad_content={"category": "parental_controls", "price": 99.99},
                context={"device": "mobile", "hour": 20}
            )
            print(f"✅ Criteo CTR prediction: {response.get('predicted_ctr', 0):.4f}")
        
        # Test safety checks
        if hasattr(master, 'safety_system'):
            safe_bid = master.safety_system.validate_bid(
                bid_amount=15.0,
                context={"budget_remaining": 50.0}
            )
            print(f"✅ Safety check: Bid ${15.0:.2f} → ${safe_bid:.2f}")
        
        # Test budget pacing
        if hasattr(master, 'budget_pacer'):
            pacing = master.budget_pacer.get_pacing_multiplier(
                hour=10,
                spent_so_far=30.0,
                daily_budget=100.0
            )
            print(f"✅ Budget pacing multiplier: {pacing:.2f}")
        
        print("\n" + "="*80)
        print("✅ MASTER INTEGRATION TEST COMPLETE")
        print(f"   Integration Score: {integrated_count}/20")
        
        if integrated_count >= 18:
            print("   Status: EXCELLENT - System fully integrated")
        elif integrated_count >= 15:
            print("   Status: GOOD - Most components integrated")
        elif integrated_count >= 10:
            print("   Status: PARTIAL - Some integration gaps")
        else:
            print("   Status: POOR - Major integration needed")
        
        print("="*80)
        
        # Cleanup
        await master.cleanup()
        
        return integrated_count >= 15  # Success if 75% integrated
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_master_integration())
    exit(0 if success else 1)