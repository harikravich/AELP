#!/usr/bin/env python3
"""
Debug single auction to see why competition is not working
"""

import logging
import numpy as np
from auction_gym_integration_fixed import FixedAuctionGymIntegration

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_single_auction():
    """Debug a single auction execution"""
    
    print("\n" + "="*60)
    print("🔍 DEBUG SINGLE AUCTION EXECUTION")
    print("="*60)
    
    # Initialize auction
    auction = FixedAuctionGymIntegration()
    
    # Run single auction with moderate bid
    our_bid = 5.00
    query_context = {
        'query_value': our_bid * 1.5,
        'user_segment': 2,
        'device_type': 1,
        'channel_index': 0,
        'stage': 2,
        'touchpoints': 3,
        'competition_level': 0.7,
        'hour': 14,
        'cvr': 0.02,
        'ltv': 199.98
    }
    
    print(f"\nRunning auction with our bid: ${our_bid:.2f}")
    print(f"Query value: ${query_context['query_value']:.2f}")
    print(f"Competition level: {query_context['competition_level']:.1f}")
    
    # Run auction
    result = auction.run_auction(our_bid, query_context)
    
    print(f"\nResult: {'WON' if result['won'] else 'LOST'}")
    if result['won']:
        print(f"Position: {result['position']}")
        print(f"Price paid: ${result['cost']:.2f}")
        print(f"CPC: ${result['cpc']:.2f}")
    print(f"Competitors: {result['competitors']}")
    
    # Show competitor insights
    insights = auction.auction_wrapper.get_competitor_insights()
    print(f"\nCompetitor Details:")
    for name, info in insights.items():
        print(f"  {name}: {info['type']}, Budget: ${info['budget']:.0f}")
    
if __name__ == "__main__":
    debug_single_auction()