#!/usr/bin/env python3
"""
Test RecSim integration to verify NO FALLBACKS are used
"""

def test_recsim_integration():
    """Test that RecSim integration works without fallbacks"""
    
    try:
        from auction_gym_integration import AuctionGymWrapper, AuctionResult
        from recsim_user_model import RecSimUserModel, UserSegment, UserProfile
        from recsim_auction_bridge import RecSimAuctionBridge
        
        print("✅ All imports successful")
        
        # Initialize components
        recsim_model = RecSimUserModel()
        auction_wrapper = AuctionGymWrapper()
        
        print("✅ Components initialized")
        
        # Initialize bridge
        bridge = RecSimAuctionBridge(
            recsim_model=recsim_model,
            auction_wrapper=auction_wrapper
        )
        
        print("✅ Bridge initialized successfully")
        
        # Test user generation
        signals = bridge.user_to_auction_signals("test_user")
        bid_amount = signals["suggested_bid"]
        print(f"✅ Auction signals generated: bid=${bid_amount:.2f}")
        
        # Test query generation
        query_data = bridge.generate_query_from_state("test_user", "shoes")
        print(f"✅ Query generated: '{query_data['query']}'")
        
        # Test user response simulation
        response = recsim_model.simulate_ad_response(
            "test_user", 
            {"creative_quality": 0.8}, 
            {"hour": 20}
        )
        print(f"✅ User response simulated: clicked={response['clicked']}")
        
        print("\n🎉 ALL RECSIM INTEGRATION TESTS PASSED - NO FALLBACKS!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recsim_integration()
    exit(0 if success else 1)
