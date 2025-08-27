# Budget Pacer System Documentation

## Overview

The GAELP Budget Pacer is an advanced budget management system designed to prevent early budget exhaustion and optimize spend distribution throughout the day. It provides intelligent pacing algorithms, real-time monitoring, and automated safety controls.

## Core Features

### 1. Hourly Budget Allocation (`allocate_hourly_budget()`)

Distributes daily budgets across 24 hours using sophisticated strategies:

**Available Strategies:**
- `EVEN_DISTRIBUTION`: Spreads budget evenly across all hours
- `PERFORMANCE_WEIGHTED`: Allocates more budget to high-performing hours
- `HISTORICAL_PATTERN`: Uses historical spending patterns to optimize allocation
- `PREDICTIVE_ML`: Uses machine learning to predict optimal hourly distribution
- `ADAPTIVE_HYBRID`: Combines multiple strategies based on available data

**Example:**
```python
from budget_pacer import BudgetPacer, ChannelType, PacingStrategy
from decimal import Decimal

pacer = BudgetPacer()
allocations = pacer.allocate_hourly_budget(
    campaign_id="campaign_001",
    channel=ChannelType.GOOGLE_ADS,
    daily_budget=Decimal('1000.00'),
    strategy=PacingStrategy.ADAPTIVE_HYBRID
)
```

### 2. Intraday Pacing Protection (`check_pace()`)

Prevents frontloading with multiple protection mechanisms:

**Frontload Protection:**
- First 4 hours limited to 50% of normal hourly allocation
- Maximum 15% of daily budget per hour
- Velocity limits prevent rapid spending

**Pace Monitoring:**
- Calculates pace ratio: (spend_percentage / time_percentage)
- Generates alerts when pace exceeds thresholds
- Automatic interventions for critical overpacing

**Example:**
```python
pace_ratio, alert = pacer.check_pace("campaign_001", ChannelType.GOOGLE_ADS)
if pace_ratio > 1.5:
    print(f"Warning: Spending {pace_ratio:.1f}x faster than target pace")
```

### 3. Dynamic Budget Reallocation (`reallocate_unused()`)

Intelligently redistributes budget between channels based on performance:

**Reallocation Logic:**
- Identifies underperforming channels with unused budget
- Reallocates to high-performing channels
- Considers 20% performance difference threshold
- Maintains overall daily budget constraints

**Example:**
```python
reallocation_results = await pacer.reallocate_unused("campaign_001")
for channel, amount in reallocation_results.items():
    action = "increased" if amount > 0 else "decreased"
    print(f"{channel}: Budget {action} by ${abs(amount):.2f}")
```

### 4. Channel-Specific Budget Management

Supports multiple marketing channels with individual controls:

**Supported Channels:**
- Google Ads
- Facebook Ads
- TikTok Ads
- Display Advertising
- Native Advertising
- Video Advertising
- Search Marketing
- Shopping Campaigns

**Channel Features:**
- Individual budget allocations
- Channel-specific pacing strategies
- Performance tracking per channel
- Independent circuit breakers

### 5. Circuit Breakers for Overspending (`emergency_stop()`)

Automated protection against runaway spending:

**Circuit Breaker States:**
- `CLOSED`: Normal operation
- `OPEN`: Spending blocked due to violations
- `HALF_OPEN`: Testing recovery after cooldown

**Trigger Conditions:**
- Spending exceeds 90% of daily budget
- Multiple rapid overspending events
- Unusual spending velocity patterns
- Performance degradation below thresholds

**Example:**
```python
# Automatic trigger
can_spend, reason = pacer.can_spend("campaign_001", ChannelType.GOOGLE_ADS, Decimal('100.00'))
if not can_spend and "circuit breaker" in reason:
    print("Circuit breaker has opened - spending blocked")

# Manual emergency stop
success = await pacer.emergency_stop("campaign_001", "Manual intervention required")
```

### 6. Predictive Pacing with Machine Learning

Uses historical data to predict optimal budget distribution:

**ML Features:**
- Linear regression models for conversion rate prediction
- Cost-per-click forecasting
- Hourly performance optimization
- Confidence scoring based on data quality

**Data Requirements:**
- Minimum 24 hours of historical data for basic predictions
- 168+ hours (1 week) for full ML functionality
- Click, conversion, and spend data for each hour

**Model Outputs:**
- Predicted conversion rates by hour
- Forecasted cost-per-click trends
- Performance multipliers for budget weighting
- Confidence scores for prediction reliability

## Safety Features

### Spend Authorization System

Multi-layer authorization prevents unauthorized or unsafe spending:

```python
can_spend, reason = pacer.can_spend(
    campaign_id="campaign_001",
    channel=ChannelType.GOOGLE_ADS,
    amount=Decimal('150.00')
)

if can_spend:
    # Record the transaction
    transaction = SpendTransaction(
        campaign_id="campaign_001",
        channel=ChannelType.GOOGLE_ADS,
        amount=Decimal('150.00'),
        timestamp=datetime.utcnow(),
        clicks=60,
        conversions=5
    )
    pacer.record_spend(transaction)
else:
    print(f"Spend blocked: {reason}")
```

### Real-time Monitoring and Alerts

Continuous monitoring with automated alert generation:

**Alert Types:**
- `warning_overpacing`: Pace ratio > 1.5x
- `critical_overpacing`: Pace ratio > 2.0x
- `emergency_stop_required`: Spending > 90% of budget
- `circuit_breaker_open`: Channel spending blocked
- `performance_degradation`: Conversion rates declining

**Alert Handling:**
```python
async def alert_handler(alert):
    if alert.severity == "critical":
        # Implement immediate action
        await emergency_response(alert)
    elif alert.severity == "high":
        # Implement corrective measures
        await adjust_bids(alert)
```

### Performance Tracking and Optimization

Comprehensive performance analytics:

**Tracked Metrics:**
- Click-through rates (CTR)
- Conversion rates
- Cost-per-click (CPC)
- Cost-per-acquisition (CPA)
- Return on ad spend (ROAS)

**Optimization Features:**
- Automatic bid adjustments based on performance
- Budget reallocation to high-performing channels
- Historical pattern learning
- Predictive performance modeling

## Integration Examples

### Basic Integration

```python
import asyncio
from budget_pacer import BudgetPacer, ChannelType, PacingStrategy
from decimal import Decimal

async def main():
    # Initialize pacer
    pacer = BudgetPacer()
    
    # Set up campaign
    allocations = pacer.allocate_hourly_budget(
        campaign_id="my_campaign",
        channel=ChannelType.GOOGLE_ADS,
        daily_budget=Decimal('1000.00'),
        strategy=PacingStrategy.ADAPTIVE_HYBRID
    )
    
    # Authorization check before spending
    can_spend, reason = pacer.can_spend(
        "my_campaign", 
        ChannelType.GOOGLE_ADS, 
        Decimal('50.00')
    )
    
    if can_spend:
        print("Spend authorized - proceed with transaction")
    else:
        print(f"Spend blocked: {reason}")

asyncio.run(main())
```

### Advanced Integration with Multiple Channels

```python
async def setup_multichannel_campaign():
    pacer = BudgetPacer()
    campaign_id = "multichannel_campaign"
    
    # Set up multiple channels
    channels = [
        (ChannelType.GOOGLE_ADS, Decimal('2000.00'), PacingStrategy.PREDICTIVE_ML),
        (ChannelType.FACEBOOK_ADS, Decimal('1500.00'), PacingStrategy.PERFORMANCE_WEIGHTED),
        (ChannelType.TIKTOK_ADS, Decimal('1000.00'), PacingStrategy.ADAPTIVE_HYBRID),
    ]
    
    for channel, budget, strategy in channels:
        allocations = pacer.allocate_hourly_budget(
            campaign_id, channel, budget, strategy
        )
        print(f"Set up {channel.value} with ${budget} budget")
    
    # Simulate spending and monitoring
    while True:
        # Check if spending is allowed
        for channel, budget, _ in channels:
            spend_amount = Decimal('25.00')  # Example spend
            can_spend, reason = pacer.can_spend(campaign_id, channel, spend_amount)
            
            if can_spend:
                # Record transaction (implement actual spending logic)
                print(f"Authorized ${spend_amount} spend on {channel.value}")
            else:
                print(f"Blocked spend on {channel.value}: {reason}")
        
        # Check for budget reallocation opportunities
        reallocation = await pacer.reallocate_unused(campaign_id)
        if reallocation:
            print(f"Reallocated budget: {reallocation}")
        
        await asyncio.sleep(300)  # Check every 5 minutes
```

## Configuration Options

### Pacing Parameters

```python
pacer = BudgetPacer()

# Customize pacing parameters
pacer.max_hourly_spend_pct = 0.12  # Max 12% per hour (default: 15%)
pacer.frontload_protection_hours = 6  # Protect first 6 hours (default: 4)
pacer.emergency_stop_threshold = 0.85  # Stop at 85% of budget (default: 90%)
pacer.pace_warning_threshold = 1.3  # Alert at 1.3x pace (default: 1.5x)
```

### Performance Thresholds

```python
# Set performance-based limits
min_conversion_rate = 0.02  # 2% minimum
max_cost_per_click = 5.00   # $5.00 maximum
target_roas = 3.0          # 3:1 return on ad spend

# These can be integrated into authorization logic
```

## Best Practices

### 1. Gradual Rollout
- Start with conservative pacing limits
- Monitor performance for 7-14 days
- Gradually optimize based on learnings

### 2. Channel-Specific Strategies
- Use ML prediction for high-volume channels
- Use performance weighting for established channels
- Use even distribution for new channels

### 3. Safety First
- Always implement circuit breakers
- Set conservative emergency stop thresholds
- Monitor alerts closely during initial deployment

### 4. Performance Optimization
- Regularly review reallocation opportunities
- Analyze historical patterns weekly
- Adjust strategies based on seasonal trends

### 5. Integration Testing
- Test all authorization paths
- Verify emergency stop functionality
- Validate alert mechanisms

## Monitoring and Maintenance

### Key Metrics to Track
- Daily budget utilization rates
- Pacing alert frequency
- Circuit breaker activation rates
- Performance metric trends
- Reallocation success rates

### Regular Maintenance Tasks
- Review and update ML models weekly
- Analyze spending patterns monthly
- Optimize pacing strategies quarterly
- Update safety thresholds as needed

### Troubleshooting Common Issues

**High Pacing Alert Frequency:**
- Review hourly allocations
- Check for unusual traffic patterns
- Verify budget limits are appropriate

**Circuit Breaker Activation:**
- Investigate spending velocity
- Review bid management settings
- Check for external factors (seasonality, events)

**Poor Performance Optimization:**
- Ensure sufficient historical data
- Verify conversion tracking accuracy
- Review channel attribution models

## API Reference

### Core Methods

#### `allocate_hourly_budget(campaign_id, channel, daily_budget, strategy)`
Allocates daily budget across 24 hours using specified strategy.

**Parameters:**
- `campaign_id` (str): Unique campaign identifier
- `channel` (ChannelType): Marketing channel
- `daily_budget` (Decimal): Total daily budget
- `strategy` (PacingStrategy): Allocation strategy

**Returns:** List[HourlyAllocation]

#### `check_pace(campaign_id, channel)`
Checks current spending pace and returns ratio and alerts.

**Returns:** Tuple[float, Optional[PacingAlert]]

#### `can_spend(campaign_id, channel, amount)`
Authorizes spending based on pacing and safety rules.

**Returns:** Tuple[bool, str]

#### `record_spend(transaction)`
Records a spend transaction for pacing analysis.

**Parameters:**
- `transaction` (SpendTransaction): Transaction details

#### `reallocate_unused(campaign_id)`
Reallocates unused budget between channels.

**Returns:** Dict[ChannelType, Decimal]

#### `emergency_stop(campaign_id, reason)`
Emergency stop all spending for a campaign.

**Returns:** bool

### Data Classes

#### `SpendTransaction`
```python
@dataclass
class SpendTransaction:
    campaign_id: str
    channel: ChannelType
    amount: Decimal
    timestamp: datetime
    clicks: int = 0
    conversions: int = 0
    cost_per_click: float = 0.0
    conversion_rate: float = 0.0
```

#### `PacingAlert`
```python
@dataclass
class PacingAlert:
    alert_type: str
    campaign_id: str
    channel: Optional[ChannelType]
    current_spend: Decimal
    pace_ratio: float
    recommended_action: str
    severity: str
```

## Conclusion

The GAELP Budget Pacer provides comprehensive protection against early budget exhaustion while optimizing spend distribution for maximum performance. Its advanced algorithms, safety features, and flexible configuration options make it suitable for campaigns of any scale.

For additional support or custom implementations, refer to the integration examples and test files provided with this system.