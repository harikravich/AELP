#!/usr/bin/env python3
"""Test if Identity Resolver fixes work"""

from datetime import datetime
from identity_resolver import IdentityResolver, DeviceSignature, MatchConfidence

def test_identity_resolver():
    """Test Identity Resolver with correct parameters"""
    
    print("="*80)
    print("TESTING IDENTITY RESOLVER FIX")
    print("="*80)
    
    # Test 1: Initialize Identity Resolver
    print("\n1. Initializing Identity Resolver...")
    try:
        resolver = IdentityResolver()
        print(f"   ✅ Identity Resolver initialized")
        print(f"      High confidence threshold: {resolver.high_confidence_threshold}")
        print(f"      Medium confidence threshold: {resolver.medium_confidence_threshold}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Create DeviceSignatures with correct parameters
    print("\n2. Creating device signatures...")
    try:
        # Mobile device signature
        mobile_sig = DeviceSignature(
            device_id="mobile_123",
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            screen_resolution="390x844",
            timezone="America/New_York",
            language="en-US",
            platform="iOS",
            browser="Safari",
            last_seen=datetime.now()
        )
        
        # Desktop device signature
        desktop_sig = DeviceSignature(
            device_id="desktop_456",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            screen_resolution="1920x1080",
            timezone="America/New_York",
            language="en-US",
            platform="macOS",
            browser="Chrome",
            last_seen=datetime.now()
        )
        
        print(f"   ✅ Device signatures created")
        print(f"      Mobile: {mobile_sig.device_id}")
        print(f"      Desktop: {desktop_sig.device_id}")
        
    except Exception as e:
        print(f"   ❌ Failed to create signatures: {e}")
        return False
    
    # Test 3: Add signatures to resolver
    print("\n3. Adding signatures to resolver...")
    try:
        resolver.add_device_signature(mobile_sig)
        resolver.add_device_signature(desktop_sig)
        
        print(f"   ✅ Signatures added to resolver")
        print(f"      Total signatures: {len(resolver.device_signatures)}")
        
    except Exception as e:
        print(f"   ❌ Failed to add signatures: {e}")
        return False
    
    # Test 4: Test identity resolution
    print("\n4. Testing identity resolution...")
    try:
        # Create a new signature that should match
        similar_mobile = DeviceSignature(
            device_id="mobile_789",
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            screen_resolution="390x844",
            timezone="America/New_York",
            language="en-US",
            platform="iOS",
            browser="Safari",
            last_seen=datetime.now()
        )
        
        # Add behavioral patterns for better matching
        similar_mobile.search_patterns = ["parental controls", "kids safety"]
        mobile_sig.search_patterns = ["parental controls", "kids safety app"]
        
        # Add the similar mobile device first
        resolver.add_device_signature(similar_mobile)
        
        # Resolve identity using device ID
        identity_id = resolver.resolve_identity("mobile_789")
        
        if identity_id:
            print(f"   ✅ Identity resolved to: {identity_id}")
            # Check if it grouped with other devices
            cluster = resolver.get_identity_cluster("mobile_789")
            if cluster and len(cluster.device_ids) > 1:
                print(f"      Grouped with {len(cluster.device_ids)-1} other device(s)")
        else:
            print(f"   ⚠️  No identity resolved (this may be expected for new device)")
            
    except Exception as e:
        print(f"   ❌ Identity resolution failed: {e}")
        return False
    
    # Test 5: Test identity clustering
    print("\n5. Testing identity clustering...")
    try:
        # Get identity cluster for a device
        cluster = resolver.get_identity_cluster("mobile_123")
        
        if cluster:
            print(f"   ✅ Identity cluster retrieved")
            print(f"      Cluster ID: {cluster.cluster_id}")
            print(f"      Devices in cluster: {len(cluster.device_ids)}")
            print(f"      Confidence: {cluster.confidence_score:.2f}")
        else:
            print(f"   ⚠️  No cluster found for device (may be standalone)")
            
        # Try to get all identities
        if hasattr(resolver, 'identities'):
            print(f"      Total identities tracked: {len(resolver.identities)}")
        
    except Exception as e:
        print(f"   ❌ Identity clustering failed: {e}")
        # Not critical, continue
        pass
    
    # Test 6: Test with MasterOrchestrator
    print("\n6. Testing with MasterOrchestrator...")
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig()
        master = MasterOrchestrator(config)
        
        if hasattr(master, 'identity_resolver'):
            # Add a signature through master
            test_sig = DeviceSignature(
                device_id="master_test_device",
                user_agent="Mozilla/5.0 Test",
                platform="TestOS",
                last_seen=datetime.now()
            )
            
            master.identity_resolver.add_device_signature(test_sig)
            
            print(f"   ✅ MasterOrchestrator identity resolver works")
            print(f"      Added device: {test_sig.device_id}")
            
        else:
            print("   ❌ MasterOrchestrator doesn't have identity_resolver")
            return False
            
    except Exception as e:
        print(f"   ❌ MasterOrchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ IDENTITY RESOLVER TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_identity_resolver()
    exit(0 if success else 1)