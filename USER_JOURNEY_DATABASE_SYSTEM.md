# GAELP UserJourneyDatabase System

## 🎯 Overview

The GAELP UserJourneyDatabase system is a comprehensive multi-touch attribution and user journey tracking platform that maintains persistent user state across episodes. This is CRITICAL for the RL system to learn from multi-touch journeys, as users do NOT reset between episodes.

## 🔥 Key Features

### ✅ Persistent Journey Tracking
- **14-day timeout management** - Journeys persist across multiple episodes
- **Multi-day journey state** - Users maintain state between interactions
- **No episode resets** - Critical for RL learning from real user behavior
- **Journey progression tracking** through states: UNAWARE → AWARE → CONSIDERING → INTENT → CONVERTED

### ✅ Cross-Device Identity Resolution
- **Device fingerprinting** for identity matching
- **Email/phone hash matching** for cross-session identification
- **Behavioral pattern matching** for anonymous user linking
- **Confidence scoring** for identity resolution quality

### ✅ Comprehensive State Management
- **Journey state transitions** with confidence scoring
- **Trigger-based state changes** (impressions, clicks, engagement, etc.)
- **State progression validation** with configurable thresholds
- **Engagement scoring** based on dwell time, scroll depth, click patterns

### ✅ Multi-Touch Attribution
- **First-touch attribution** for awareness campaigns
- **Last-touch attribution** for conversion campaigns  
- **Time-decay attribution** for temporal weighting
- **Position-based attribution** for journey position weighting

### ✅ Competitor Intelligence
- **Competitor exposure tracking** across all channels
- **Impact assessment** on journey progression
- **Competitive defense strategies** with bid adjustments
- **Message and offer intelligence** gathering

### ✅ RL Integration
- **Real-time action recommendations** based on journey state
- **Expected reward calculation** for optimization
- **Training data generation** from completed journeys
- **State-action-reward sequences** for policy learning

### ✅ BigQuery Analytics Platform
- **Scalable data storage** with intelligent partitioning
- **Real-time streaming ingestion** from touchpoint events
- **Materialized views** for fast analytics queries
- **Cost-optimized querying** with clustering strategies

## 🗃️ Schema Design

### Core Tables

#### `users`
- Identity resolution and canonical user mapping
- Demographics and segmentation data
- Current journey state and scoring
- Device fingerprint history

#### `user_journeys` 
- Individual journey instances with 14-day timeout
- Journey progression and state transitions
- Conversion tracking and attribution
- Journey scoring and probability calculations

#### `journey_touchpoints`
- Individual interaction events within journeys
- Engagement metrics and behavioral data
- Attribution weights and channel performance
- Device and context information

#### `journey_state_transitions`
- State change events with trigger analysis
- Confidence scoring and prediction data
- ML model outputs and recommendations
- Transition pattern analysis

#### `competitor_exposures`
- Competitive intelligence and impact tracking
- Message analysis and offer comparison
- Journey impact assessment
- Defensive strategy recommendations

#### `channel_history`
- Historical performance by channel and date
- Attribution metrics and conversion tracking
- Quality scores and engagement analysis
- Optimization recommendations

## 📊 Files Created

### Core System Files

1. **`/home/hariravichandran/AELP/infrastructure/bigquery/schemas/03_journey_schema.sql`**
   - Complete BigQuery schema definition
   - Partitioning and clustering strategies
   - Materialized views for analytics
   - Data retention policies

2. **`/home/hariravichandran/AELP/journey_state.py`**
   - Journey state management and transitions
   - Engagement scoring algorithms
   - Conversion probability calculations
   - State validation and progression logic

3. **`/home/hariravichandran/AELP/user_journey_database.py`**
   - Main database integration class
   - BigQuery connectivity and operations
   - Identity resolution and journey management
   - Analytics and insights generation

### Integration and Demo Files

4. **`/home/hariravichandran/AELP/journey_aware_rl_agent.py`** (updated)
   - Enhanced with DatabaseIntegratedRLAgent class
   - Real-time journey processing
   - RL action recommendations
   - Conversion tracking and training data generation

5. **`/home/hariravichandran/AELP/demo_user_journey_database.py`**
   - Comprehensive demonstration script
   - Multi-day journey examples
   - Cross-device tracking scenarios
   - Competitor intelligence demos

6. **`/home/hariravichandran/AELP/test_journey_database.py`**
   - Complete test suite for all components
   - Validates functionality without BigQuery dependency
   - Regression testing for core algorithms

## 🚀 Usage Examples

### Basic Journey Tracking

```python
from user_journey_database import UserJourneyDatabase
from journey_state import TransitionTrigger

# Initialize database
db = UserJourneyDatabase(project_id="your-project")

# Get or create journey
journey, is_new = db.get_or_create_journey(
    user_id="user123",
    channel="google_ads",
    interaction_type="click"
)

# Update with new touchpoint
updated_journey = db.update_journey(
    journey_id=journey.journey_id,
    touchpoint=touchpoint,
    trigger=TransitionTrigger.CLICK
)
```

### RL Agent Integration

```python
from journey_aware_rl_agent import DatabaseIntegratedRLAgent

# Initialize RL agent with database
agent = DatabaseIntegratedRLAgent(
    bigquery_project_id="your-project"
)

# Process user interaction and get recommendation
recommendation, expected_reward = agent.process_user_interaction(
    user_id="user123",
    channel="google_ads", 
    interaction_type="click",
    dwell_time_seconds=45.0,
    scroll_depth=0.6
)

# Record conversion
agent.record_conversion(
    user_id="user123",
    conversion_value=50.0,
    conversion_type="purchase"
)
```

### Analytics and Insights

```python
# Get comprehensive journey analytics
analytics = db.get_journey_analytics(journey_id)

# Includes:
# - Journey progression and state history
# - Channel performance and attribution
# - Conversion probability predictions
# - Optimization recommendations
```

## 🎯 Critical Features for RL Learning

### 1. Persistent State Across Episodes
- **No user resets** between episodes
- **14-day journey timeout** maintains context
- **Cross-session continuity** for realistic learning

### 2. Multi-Touch Attribution
- **Credit assignment** across multiple touchpoints
- **Temporal attribution** with decay functions  
- **Channel contribution** analysis for optimization

### 3. Real-Time State Updates
- **Immediate state transitions** based on interactions
- **Confidence-scored predictions** for next states
- **Dynamic probability updates** for conversion

### 4. Comprehensive Context
- **Competitor exposure impact** on journey progression
- **Device and channel history** for personalization
- **Engagement patterns** for quality assessment

## 🔧 Production Integration

### Prerequisites
- Google Cloud Project with BigQuery enabled
- Service account credentials for BigQuery access
- Python 3.8+ with required dependencies

### Deployment Steps
1. Deploy BigQuery schema using `03_journey_schema.sql`
2. Configure service account permissions
3. Initialize UserJourneyDatabase with project credentials
4. Integrate with existing GAELP RL training pipeline
5. Set up streaming data ingestion from touchpoint events

### Monitoring and Maintenance
- **Journey timeout cleanup** runs daily
- **Data quality validation** on ingestion
- **Performance monitoring** for query optimization
- **Cost monitoring** and partition management

## 📈 Expected Impact

### For RL System
- **Realistic user behavior** learning from persistent journeys
- **Multi-touch optimization** instead of single-touch decisions
- **Temporal dynamics** understanding for timing optimization
- **Competitive intelligence** for defensive strategies

### For Business Intelligence
- **True attribution** across complex customer journeys
- **Cross-device insights** for omnichannel optimization
- **Competitive analysis** for market positioning
- **Predictive analytics** for conversion probability

### For Performance
- **Conversion rate improvements** through better targeting
- **Cost efficiency** through attribution-based optimization
- **Customer lifetime value** increase through journey optimization
- **Competitive advantage** through intelligence gathering

## 🎉 Ready for Production

The UserJourneyDatabase system is fully implemented and tested, providing the critical foundation for GAELP's RL system to learn from realistic, persistent user journeys without episode resets. This enables true multi-touch attribution and optimization that reflects real-world customer behavior.

**All tests pass ✅ - Ready for immediate integration!**