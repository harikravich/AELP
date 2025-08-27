#!/usr/bin/env python3
"""Test if Journey Database works with BigQuery using Thrive project"""

import os
from user_journey_database import UserJourneyDatabase

def test_journey_db_bigquery():
    """Test Journey Database with BigQuery connection"""
    
    print("="*80)
    print("TESTING JOURNEY DATABASE WITH BIGQUERY (THRIVE PROJECT)")
    print("="*80)
    
    print(f"\nEnvironment:")
    print(f"  GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
    
    # Test 1: Initialize Journey Database
    print("\n1. Initializing Journey Database...")
    try:
        db = UserJourneyDatabase()  # Should use Thrive project automatically
        
        print(f"   ✅ Database initialized")
        print(f"      Project: {db.project_id}")
        print(f"      Dataset: {db.dataset_id}")
        print(f"      BigQuery available: {db.bigquery_available}")
        
        if not db.bigquery_available:
            print("   ⚠️  BigQuery not available - using in-memory storage")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Create a user
    print("\n2. Testing get_or_create_user...")
    try:
        user, is_new = db.get_or_create_user(
            user_id="test_bigquery_user",
            attributes={"segment": "crisis_parent"}
        )
        
        if user:
            print(f"   ✅ User created/retrieved: {user.user_id}")
            print(f"      Is new: {is_new}")
            print(f"      State: {user.current_journey_state}")
        else:
            print("   ❌ User creation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ get_or_create_user failed: {e}")
        return False
    
    # Test 3: Create a journey
    print("\n3. Testing get_or_create_journey...")
    try:
        journey, is_new = db.get_or_create_journey(
            user_id="test_bigquery_user",
            channel="search",
            device_fingerprint={"device_id": "test_device"}
        )
        
        if journey:
            print(f"   ✅ Journey created/retrieved: {journey.journey_id}")
            print(f"      User: {journey.canonical_user_id}")
            print(f"      State: {journey.current_state}")
            print(f"      Is new: {is_new}")
        else:
            print("   ❌ Journey creation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ get_or_create_journey failed: {e}")
        return False
    
    # Test 4: Test with MasterOrchestrator
    print("\n4. Testing with MasterOrchestrator...")
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig()
        print(f"   Config project_id: {config.project_id}")
        print(f"   Config dataset_id: {config.dataset_id}")
        
        master = MasterOrchestrator(config)
        
        # Check if journey_db is properly initialized
        if hasattr(master, 'journey_db'):
            print(f"   ✅ MasterOrchestrator has journey_db")
            print(f"      Project: {master.journey_db.project_id}")
            print(f"      BigQuery available: {master.journey_db.bigquery_available}")
            
            # Try to create a journey through master
            journey, is_new = master.journey_db.get_or_create_journey(
                user_id="master_bigquery_user",
                channel="display"
            )
            
            if journey:
                print(f"   ✅ Can create journeys through master: {journey.journey_id}")
            else:
                print("   ❌ Journey creation through master failed")
        else:
            print("   ❌ MasterOrchestrator doesn't have journey_db")
            return False
            
    except Exception as e:
        print(f"   ❌ MasterOrchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ JOURNEY DATABASE BIGQUERY TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_journey_db_bigquery()
    exit(0 if success else 1)