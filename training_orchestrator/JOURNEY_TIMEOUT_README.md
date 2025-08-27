# Journey Timeout and Abandonment System

## Overview

The Journey Timeout and Abandonment System ensures proper termination of user journeys in GAELP's training orchestrator. It prevents zombie journeys from running indefinitely and calculates appropriate abandonment penalties for training feedback.

## Key Features

### ✅ **Timeout Management**
- **14-day default timeout** for all journeys
- **Configurable timeout periods** per journey or globally
- **Automatic expiration detection** with background monitoring
- **Timeout extension capability** for high-engagement journeys

### ✅ **Abandonment Detection**
- **Multiple abandonment reasons**: timeout, inactivity, competitor conversion, budget exhaustion, fatigue
- **Sophisticated penalty calculation** based on journey state, cost, and opportunity
- **Real-time abandonment event emission** for training feedback
- **Comprehensive abandonment analytics** and reporting

### ✅ **Zombie Journey Cleanup**
- **Automated stale data cleanup** with configurable retention periods
- **Batch processing** for efficient resource utilization  
- **Memory and cache management** to prevent resource leaks
- **BigQuery integration** for persistent data storage

### ✅ **Training Integration**
- **Seamless integration** with TrainingOrchestrator
- **Real-time metrics tracking** for training feedback
- **Asynchronous background processing** without blocking training
- **Redis caching** for high-performance state management

## Core Components

### JourneyTimeoutManager
Main orchestrator for timeout and abandonment logic:
- `check_timeouts()` - Monitor and process expired journeys
- `mark_abandoned()` - Mark journeys as abandoned with penalty calculation
- `calculate_abandonment_cost()` - Calculate penalties and opportunity costs
- `cleanup_stale_data()` - Clean up zombie journeys and old data

### TimeoutConfiguration
Configurable timeout behavior:
- `default_timeout_days` - Global timeout period (default: 14 days)
- `inactivity_threshold_hours` - Inactivity detection threshold (default: 72 hours)
- `cleanup_batch_size` - Batch size for cleanup operations (default: 1000)
- `abandonment_check_interval_minutes` - Check frequency (default: 30 minutes)

### AbandonmentPenalty
Penalty calculation structure:
- `penalty_amount` - Direct cost penalty based on spend and state
- `opportunity_cost` - Lost conversion value penalty
- `abandonment_reason` - Classification of abandonment cause
- `days_active` / `touchpoint_count` - Journey engagement metrics

## Usage Examples

### Standalone Usage
```python
from training_orchestrator import create_timeout_manager, AbandonmentReason

# Create timeout manager
timeout_manager = create_timeout_manager(
    timeout_days=14,
    project_id="your-project",
    dataset_id="gaelp"
)

# Start monitoring
await timeout_manager.start()

# Register journey
await timeout_manager.register_journey(
    journey_id="journey-123",
    start_time=datetime.now(),
    user_id="user-456"
)

# Check timeouts
timed_out = await timeout_manager.check_timeouts()

# Manual abandonment
penalty = await timeout_manager.mark_abandoned(
    journey_id="journey-123",
    reason=AbandonmentReason.COMPETITOR_CONVERSION
)
```

### Training Orchestrator Integration
```python
from training_orchestrator import TrainingOrchestrator, TrainingConfiguration

config = TrainingConfiguration(
    journey_timeout_days=14,
    inactivity_threshold_hours=72,
    cleanup_stale_data_days=30
)

orchestrator = TrainingOrchestrator(config)
await orchestrator.start_journey_monitoring()

# Register training journey
await orchestrator.register_training_journey(
    journey_id="training-journey-789",
    user_id="training-user-123"
)

# Check timeouts during training
timed_out = await orchestrator.check_journey_timeouts()

# Get abandonment analytics
analytics = await orchestrator.get_journey_abandonment_analytics(days=30)
```

## Penalty Calculation Logic

The abandonment penalty system uses multiple factors:

### Base Penalty Components
1. **Cost Penalty**: 15% of total journey spend
2. **State Penalty**: Weighted by journey progression (INTENT = 80%, CONSIDERING = 50%, etc.)
3. **Time Decay**: Longer journeys have higher penalties
4. **Opportunity Cost**: Lost conversion value based on probability

### Reason-Specific Multipliers
- **Timeout**: 1.0x (baseline)
- **Competitor Conversion**: 1.5x (high penalty)
- **Budget Exhausted**: 0.6x (reduced penalty)
- **User Fatigue**: 0.7x (moderate penalty)
- **Manual Termination**: 0.3x (minimal penalty)

### Example Calculations
- **Early Stage Timeout**: $25 spend, AWARE state → ~$7 penalty
- **High-Intent Competitor Loss**: $150 spend, INTENT state → ~$180 penalty
- **Budget Exhausted**: $75 spend, CONSIDERING state → ~$45 penalty

## Configuration Options

### Timeout Settings
```python
TimeoutConfiguration(
    default_timeout_days=14,           # Journey timeout period
    inactivity_threshold_hours=72,     # Inactivity detection
    max_journey_duration_days=90,      # Maximum allowed duration
    abandonment_check_interval_minutes=30  # Check frequency
)
```

### Penalty Weights
```python
state_penalty_weights = {
    'UNAWARE': 0.1,      # Low penalty for early abandonment
    'AWARE': 0.2,
    'CONSIDERING': 0.5,
    'INTENT': 0.8,       # High penalty for late-stage abandonment
    'CONVERTED': 0.0     # No penalty for successful conversion
}
```

## Integration Points

### BigQuery Storage
- **journey_abandonments** table for penalty records
- **user_journeys** table updates for abandonment status
- **Automated archiving** of old records

### Redis Caching
- **Active timeout tracking** for high performance
- **Abandonment event streaming** for real-time processing
- **Configuration persistence** across restarts

### Training Feedback
- **Real-time penalty calculation** affects agent rewards
- **Abandonment analytics** inform training strategies
- **Performance metrics** track timeout management effectiveness

## Monitoring and Analytics

### Key Metrics
- **Active Journeys**: Currently monitored journeys
- **Timeout Rate**: Percentage of journeys that timeout
- **Average Penalty**: Mean abandonment cost
- **Cleanup Efficiency**: Stale data removal statistics

### Analytics Queries
- **Abandonment by Reason**: Breakdown of abandonment causes
- **Penalty Trends**: Penalty amounts over time
- **State Distribution**: Where journeys typically abandon
- **Cost Analysis**: Relationship between spend and abandonment

## Error Handling

### Fault Tolerance
- **Graceful degradation** when external services unavailable
- **Automatic retry logic** for transient failures
- **Circuit breaker patterns** for service protection
- **Comprehensive logging** for debugging

### Recovery Mechanisms  
- **State persistence** in Redis for crash recovery
- **Checkpoint creation** for long-running operations
- **Manual intervention tools** for edge cases
- **Data consistency checks** for integrity

## Performance Optimization

### Scalability Features
- **Asynchronous processing** for non-blocking operations
- **Batch operations** for efficient database usage
- **Connection pooling** for resource management
- **Caching strategies** for frequently accessed data

### Resource Management
- **Memory-efficient data structures** for large-scale operations
- **Background cleanup** to prevent resource leaks
- **Rate limiting** for external service calls
- **Monitoring hooks** for performance tracking

## Files Overview

- **`journey_timeout.py`** - Main timeout manager implementation
- **`examples/journey_timeout_example.py`** - Usage demonstrations
- **Core integration** in `training_orchestrator/core.py`
- **Module exports** in `training_orchestrator/__init__.py`

The journey timeout system ensures that no user journey runs forever, provides proper cost accounting for abandoned journeys, and maintains clean data for optimal training performance.