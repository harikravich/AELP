# RecSim-AuctionGym Bridge

## Overview

The RecSim-AuctionGym Bridge (`recsim_auction_bridge.py`) successfully connects RecSim user segments to the AuctionGym bidding system, creating a realistic simulation where user behavior drives auction participation and query generation.

## Key Features

### 🎯 User Segment Mapping
- Maps 6 distinct user segments to auction participation patterns:
  - **IMPULSE_BUYER**: High bid volatility, quick decisions, mobile-first
  - **RESEARCHER**: Consistent bidding, high participation, desktop-focused
  - **LOYAL_CUSTOMER**: Premium bidding, selective participation, brand-focused
  - **WINDOW_SHOPPER**: Low bids, minimal participation, browsing behavior
  - **PRICE_CONSCIOUS**: Conservative bidding, price-driven decisions
  - **BRAND_LOYALIST**: Premium bids for preferred brands, high conversion

### 🔍 Query Generation System
- Generates realistic search queries based on user journey stages:
  - **Awareness**: "what is running shoes", "how to choose sneakers"
  - **Consideration**: "sneakers comparison", "best shoes for running"
  - **Purchase**: "buy sneakers", "sneakers for sale", "discount shoes"
  - **Loyalty**: "nike sneakers", "brand official store"
  - **Re-engagement**: "shoes deals", "sneakers sale"

### 💰 Auction Integration
- Converts user signals into bidding parameters:
  - Interest level → bid multipliers
  - Fatigue level → participation probability
  - Price sensitivity → bid constraints
  - Brand affinity → quality score adjustments

## Core Methods

### `user_to_auction_signals(user_id, context)`
Converts user profile and state to auction bidding signals:
```python
signals = bridge.user_to_auction_signals("user_123", {
    'hour': 20, 'device': 'mobile'
})
# Returns: suggested_bid, quality_score, participation_probability, etc.
```

### `generate_query_from_state(user_id, product_category, brand)`
Generates contextual search queries based on user journey stage:
```python
query_data = bridge.generate_query_from_state(
    user_id="user_123", 
    product_category="sneakers",
    brand="nike"
)
# Returns: query, journey_stage, intent_strength, query_type
```

### `map_segment_to_bid_value(segment, query_intent, market_context)`
Maps user segment and query intent to appropriate bid values:
```python
bid_value = bridge.map_segment_to_bid_value(
    segment=UserSegment.IMPULSE_BUYER,
    query_intent=query_intent,
    market_context={'competition_level': 0.6}
)
# Returns: optimized bid value for segment/intent combination
```

### `simulate_user_auction_session(user_id, num_queries, product_category)`
Runs complete user session with multiple queries and auctions:
```python
session = bridge.simulate_user_auction_session(
    user_id="user_123",
    num_queries=5,
    product_category="running_shoes"
)
# Returns: complete session analytics including costs, revenue, conversions
```

## Architecture

### Segment Bid Profiles
Each user segment has distinct bidding characteristics:
- **Base bid range**: Segment-appropriate bid amounts
- **Bid volatility**: How much bids fluctuate
- **Participation rate**: Likelihood to enter auctions
- **Quality score range**: Ad quality expectations
- **Preferred slots**: Position preferences
- **Time sensitivity**: Decision-making speed
- **Budget depletion rate**: Spending patterns

### Journey Stage Mapping
User segments map to different journey stage probabilities:
- **Impulse Buyers**: 55% purchase stage, 20% awareness
- **Researchers**: 45% consideration, 35% awareness
- **Loyal Customers**: 40% purchase, 35% loyalty
- **Window Shoppers**: 50% consideration, 40% awareness
- **Price Conscious**: 55% consideration, 25% awareness
- **Brand Loyalists**: 40% loyalty, 35% purchase

### Query Templates
Each journey stage has specific query patterns:
- **Informational queries**: "how to", "what is", "guide"
- **Navigational queries**: "[brand] official", "[brand] store"
- **Transactional queries**: "buy", "sale", "discount", "order"

## Integration Benefits

### 🔄 Bidirectional Data Flow
- User behavior influences auction participation
- Auction results update user state (fatigue, interest)
- Query performance feeds back into bidding decisions

### 📊 Realistic Simulation
- Segments behave according to research-backed patterns
- Journey stages drive appropriate query generation
- Market dynamics affect bidding strategies

### 🎛️ Configurable Parameters
- Adjustable bid ranges per segment
- Customizable journey stage probabilities
- Flexible query templates and categories

## Usage Examples

### Basic Usage
```python
from recsim_auction_bridge import RecSimAuctionBridge

# Initialize bridge
bridge = RecSimAuctionBridge()

# Generate auction signals for a user
signals = bridge.user_to_auction_signals("user_123")
print(f"Suggested bid: ${signals['suggested_bid']:.2f}")

# Generate query based on user state
query = bridge.generate_query_from_state("user_123", "shoes")
print(f"Query: '{query['query']}' (Stage: {query['journey_stage']})")

# Run complete session
session = bridge.simulate_user_auction_session("user_123", num_queries=3)
print(f"Session ROAS: {session.get('roas', 0):.2f}x")
```

### Advanced Analytics
```python
# Run multiple sessions across segments
for segment in UserSegment:
    for i in range(10):
        bridge.simulate_user_auction_session(f"{segment.value}_{i}", 5)

# Get comprehensive analytics
analytics = bridge.get_bridge_analytics()
print(f"Total sessions: {analytics['total_sessions']}")
print("Segment performance:", analytics['segment_performance'])
```

## Fallback Support

The bridge includes comprehensive fallback implementations when dependencies are unavailable:
- **RecSim fallback**: Simplified user model with basic segment behavior
- **AuctionGym fallback**: Basic auction simulation with win/loss outcomes
- **Full functionality**: Works with or without external dependencies

## Files

- **`recsim_auction_bridge.py`**: Main bridge implementation
- **`demo_recsim_auction_bridge.py`**: Comprehensive demonstration script
- **Integration with existing**: `recsim_user_model.py`, `auction_gym_integration.py`

## Testing

Run the test suite:
```bash
python3 recsim_auction_bridge.py
```

Run the demonstration:
```bash
python3 demo_recsim_auction_bridge.py
```

This bridge successfully connects GAELP's two main simulation components, enabling realistic user-driven auction dynamics and query generation patterns.