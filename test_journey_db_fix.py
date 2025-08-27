#!/usr/bin/env python3
"""
Test if Journey Database fixes are working
"""

import asyncio
from datetime import datetime
import logging

logging.basicConfig(level=logging.WARNING)

async def test_journey_db_fix():
    """Test the Journey Database fixes"""
    
    print("\n" + "="*80)
    print("TESTING JOURNEY DATABASE FIXES")
    print("="*80)
    
    # Test 1: Direct UserJourneyDatabase initialization
    print("\n1. Testing UserJourneyDatabase initialization...")
    try:
        from user_journey_database import UserJourneyDatabase
        
        # This should not crash even without BigQuery credentials
        db = UserJourneyDatabase(
            project_id="test-project",
            dataset_id="test-dataset",
            timeout_days=14
        )
        
        if db.bigquery_available:
            print("   ⚠️  BigQuery is available (unexpected)")
        else:
            print("   ✅ BigQuery unavailable - using in-memory storage")
            
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Test get_or_create_user method
    print("\n2. Testing get_or_create_user method...")
    try:
        user_profile, is_new = db.get_or_create_user(
            user_id="test_user_001",
            attributes={"segment": "crisis_parent", "device": "mobile"}
        )
        
        if user_profile and user_profile.user_id == "test_user_001":
            print(f"   ✅ User created: {user_profile.user_id}")
            print(f"      Is new: {is_new}")
            print(f"      Attributes: {user_profile.attributes}")
        else:
            print("   ❌ User not created properly")
            return False
            
        # Test getting existing user
        user_profile2, is_new2 = db.get_or_create_user(
            user_id="test_user_001",
            attributes={"segment": "crisis_parent"}
        )
        
        if not is_new2:
            print(f"   ✅ Existing user retrieved correctly")
        else:
            print("   ❌ Should have returned existing user")
            return False
            
    except Exception as e:
        print(f"   ❌ get_or_create_user failed: {e}")
        return False
    
    # Test 3: Test get_or_create_journey method
    print("\n3. Testing get_or_create_journey method...")
    try:
        journey, is_new = db.get_or_create_journey(
            user_id="test_user_001",
            channel="search",
            device_fingerprint={"device_id": "device_001", "browser": "chrome"}
        )
        
        if journey and journey.canonical_user_id:
            print(f"   ✅ Journey created: {journey.journey_id}")
            print(f"      User: {journey.canonical_user_id}")
            print(f"      State: {journey.current_state}")
        else:
            print("   ❌ Journey not created properly")
            return False
            
    except Exception as e:
        print(f"   ❌ get_or_create_journey failed: {e}")
        return False
    
    # Test 4: Test with MasterOrchestrator
    print("\n4. Testing with MasterOrchestrator...")
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig()
        config.simulation_days = 1
        config.users_per_day = 1
        config.n_parallel_worlds = 1
        
        master = MasterOrchestrator(config)
        
        # Test journey creation through master
        journey, is_new = master.journey_db.get_or_create_journey(
            user_id="master_test_user",
            channel="display",
            device_fingerprint={"device_id": "master_device"}
        )
        
        if journey:
            print(f"   ✅ Master orchestrator can create journeys: {journey.journey_id}")
        else:
            print("   ❌ Master orchestrator journey creation failed")
            return False
            
        # Test user creation through master
        user, is_new = master.journey_db.get_or_create_user(
            user_id="master_test_user_2",
            attributes={"segment": "researcher"}
        )
        
        if user:
            print(f"   ✅ Master orchestrator can create users: {user.user_id}")
        else:
            print("   ❌ Master orchestrator user creation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ MasterOrchestrator test failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ ALL JOURNEY DATABASE TESTS PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_journey_db_fix())
    exit(0 if success else 1)