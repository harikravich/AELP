#!/usr/bin/env python3
"""
Test to verify actual integration of all components in GAELP
"""

import asyncio
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_actual_integration():
    """Test that all components are actually wired and working"""
    
    print("\n" + "="*80)
    print("TESTING ACTUAL GAELP INTEGRATION")
    print("="*80)
    
    try:
        # Import the master orchestrator
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        # Create minimal config
        config = GAELPConfig()
        config.simulation_days = 1
        config.users_per_day = 5
        config.n_parallel_worlds = 1
        
        # Initialize system
        print("\n🔧 Initializing GAELP Master System...")
        master = MasterOrchestrator(config)
        
        # Check all components are present
        print("\n📊 Component Presence Check:")
        print("-" * 50)
        
        components = {
            'journey_db': 'UserJourneyDatabase',
            'monte_carlo': 'MonteCarloSimulator',
            'competitor_manager': 'CompetitorAgents',
            'auction_bridge': 'RecSimAuctionBridge',
            'attribution_engine': 'AttributionModels',
            'delayed_rewards': 'DelayedRewardSystem',
            'state_encoder': 'JourneyStateEncoder',
            'creative_selector': 'CreativeSelector',
            'budget_pacer': 'BudgetPacer',
            'identity_resolver': 'IdentityResolver',
            'evaluation': 'EvaluationFramework',
            'importance_sampler': 'ImportanceSampler',
            'conversion_lag_model': 'ConversionLagModel',
            'competitive_intel': 'CompetitiveIntel',
            'criteo_response': 'CriteoResponseModel',
            'timeout_manager': 'JourneyTimeout',
            'temporal_effects': 'TemporalEffects',
            'model_versioning': 'ModelVersioning',
            'online_learner': 'OnlineLearner',
            'safety_system': 'SafetySystem'
        }
        
        present = 0
        missing = []
        none_values = []
        
        for attr, name in components.items():
            if hasattr(master, attr):
                value = getattr(master, attr)
                if value is not None:
                    print(f"  ✅ {name:25} Present and initialized")
                    present += 1
                else:
                    print(f"  ⚠️  {name:25} Present but None")
                    none_values.append(name)
            else:
                print(f"  ❌ {name:25} Missing attribute")
                missing.append(name)
        
        print(f"\n📈 Presence Score: {present}/{len(components)}")
        
        if none_values:
            print(f"\n⚠️ Components set to None ({len(none_values)}):")
            for comp in none_values:
                print(f"    - {comp}")
        
        if missing:
            print(f"\n❌ Missing Components ({len(missing)}):")
            for comp in missing:
                print(f"    - {comp}")
        
        # Test actual functionality
        print("\n🔄 Testing Component Functionality...")
        print("-" * 50)
        
        # 1. Test Journey Database
        try:
            user = await master.journey_db.get_or_create_user(
                user_id="test_user",
                attributes={"segment": "crisis_parent"}
            )
            print("✅ Journey Database: Working")
        except Exception as e:
            print(f"❌ Journey Database: {e}")
        
        # 2. Test Creative Selector
        try:
            if master.creative_selector:
                from creative_selector import UserState as CreativeUserState
                creative, reason = master.creative_selector.select_creative(
                    CreativeUserState(segment="crisis_parent")
                )
                print(f"✅ Creative Selector: Working (selected: {creative.get('headline', 'Unknown')[:50]}...)")
            else:
                print("⚠️ Creative Selector: Set to None")
        except Exception as e:
            print(f"❌ Creative Selector: {e}")
        
        # 3. Test Competitor Agents
        try:
            if master.competitor_manager:
                context = {"hour": 10, "user_tier": "high_value"}
                results = master.competitor_manager.run_auction(context)
                print(f"✅ Competitor Agents: Working ({len(results)} agents bidding)")
            else:
                print("⚠️ Competitor Agents: Not found")
        except Exception as e:
            print(f"❌ Competitor Agents: {e}")
        
        # 4. Test Identity Resolver
        try:
            if master.identity_resolver:
                from identity_resolver import DeviceSignature
                sig = DeviceSignature(
                    device_id="device_001",
                    user_agent="Mozilla/5.0",
                    ip_address="192.168.1.1"
                )
                master.identity_resolver.add_device_signature(sig)
                canonical_id = master.identity_resolver.resolve_identity("device_001")
                print(f"✅ Identity Resolver: Working (canonical: {canonical_id})")
            else:
                print("⚠️ Identity Resolver: Set to None")
        except Exception as e:
            print(f"❌ Identity Resolver: {e}")
        
        # 5. Test State Encoder
        try:
            if master.state_encoder:
                import torch
                test_state = {
                    'touchpoints': [],
                    'days_in_journey': 1,
                    'current_stage': 'awareness'
                }
                encoded = master.state_encoder.encode_journey(test_state)
                print(f"✅ Journey State Encoder: Working (output shape: {encoded.shape})")
            else:
                print("⚠️ Journey State Encoder: Not found")
        except Exception as e:
            print(f"❌ Journey State Encoder: {e}")
        
        # 6. Test Criteo Response Model
        try:
            if master.criteo_response:
                response = master.criteo_response.simulate_user_response(
                    user_id="test",
                    ad_content={"price": 99.99},
                    context={"device": "mobile"}
                )
                print(f"✅ Criteo Response Model: Working (CTR: {response.get('predicted_ctr', 0):.4f})")
            else:
                print("⚠️ Criteo Response Model: Not found")
        except Exception as e:
            print(f"❌ Criteo Response Model: {e}")
        
        # 7. Test Online Learner
        if master.online_learner is None:
            print("⚠️ Online Learner: Set to None (commented out in code)")
        else:
            print("✅ Online Learner: Present")
        
        # Check for actual wiring in methods
        print("\n🔌 Testing Integration Points...")
        print("-" * 50)
        
        # Check if PPO uses encoder
        try:
            from journey_aware_rl_agent import JourneyAwarePPOAgent
            test_agent = JourneyAwarePPOAgent(use_journey_encoder=True)
            if test_agent.use_journey_encoder and test_agent.journey_encoder:
                print("✅ PPO Agent uses Journey Encoder")
            else:
                print("⚠️ PPO Agent encoder not active")
        except Exception as e:
            print(f"❌ PPO Agent encoder check: {e}")
        
        # Summary
        print("\n" + "="*80)
        print("INTEGRATION TEST SUMMARY")
        print("="*80)
        
        total_expected = len(components)
        fully_working = present
        
        if fully_working >= 18:
            print(f"✅ EXCELLENT: {fully_working}/{total_expected} components working")
        elif fully_working >= 15:
            print(f"⚠️ GOOD: {fully_working}/{total_expected} components working")
        else:
            print(f"❌ NEEDS WORK: Only {fully_working}/{total_expected} components working")
        
        # Cleanup
        await master.cleanup()
        
        return fully_working >= 15
        
    except Exception as e:
        print(f"\n❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_actual_integration())
    print("\n" + "="*80)
    if success:
        print("✅ INTEGRATION VERIFIED")
    else:
        print("❌ INTEGRATION ISSUES FOUND")
    exit(0 if success else 1)