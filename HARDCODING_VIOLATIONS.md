# HARDCODING VIOLATIONS IN GAELP TRAINING

## CRITICAL: These violate CLAUDE.md requirements

### 1. HARDCODED CHANNELS
**File:** fortified_rl_agent.py
- **Line 451:** `channels = ['organic', 'paid_search', 'social', 'display', 'email']`
- **VIOLATION:** Channels should be discovered from GA4/discovered_patterns.json
- **FIX:** Read from `patterns.channels.keys()`

### 2. HARDCODED BID RANGES  
**File:** fortified_rl_agent.py
- **Line 31-32:** 
  ```python
  MIN_BID = 0.50
  MAX_BID = 10.00
  ```
- **VIOLATION:** Bid ranges should be learned from auction data
- **FIX:** Calculate from discovered_patterns bid_ranges or learn from auction competition

### 3. HARDCODED BUDGETS
**File:** fortified_environment.py
- **Line 43-44:**
  ```python
  max_budget: float = 10000.0
  max_steps: int = 1000
  ```
- **VIOLATION:** Budget should come from configuration or be discovered
- **FIX:** Use ParameterManager or discover from patterns

**File:** fortified_rl_agent.py  
- **Line 82:** `remaining_budget: float = 1000.0`
- **VIOLATION:** Default budget hardcoded
- **FIX:** Initialize from environment or patterns

### 4. HARDCODED NORMALIZATION CONSTANTS
**File:** fortified_rl_agent.py
- **Line 115:** `min(self.touchpoints_seen / 20.0, 1.0)`
- **Line 148:** `self.avg_position_last_10 / 10.0`  
- **Line 154:** `min(self.remaining_budget / 1000.0, 1.0)`
- **Line 164:** `len(self.touchpoint_credits) / 10.0`
- **Line 165:** `self.expected_conversion_value / 200.0`
- **Line 168:** `self.ab_test_variant / 10.0`
- **Line 177:** `min(self.competitor_impressions_seen / 10.0, 1.0)`
- **VIOLATION:** Normalization divisors are arbitrary magic numbers
- **FIX:** Calculate from actual data statistics (mean, std, max)

### 5. HARDCODED CONVERSION VALUE
**File:** fortified_environment.py
- **Line 239:** `'avg_conversion_value': np.mean(conversion_values) if conversion_values else 100.0`
- **VIOLATION:** Fallback value of 100.0 is hardcoded
- **FIX:** Use discovered average from patterns

### 6. HARDCODED HYPERPARAMETERS
**File:** fortified_rl_agent.py
- **Line 199, 206:** `Dropout(0.1)`
- **VIOLATION:** Dropout rate hardcoded
- **FIX:** Use ParameterManager for all hyperparameters

### 7. HARDCODED DEFAULT VALUES
**File:** fortified_rl_agent.py
- **Line 46:** `days_since_first_touch: float = 0.0`
- **Line 73:** `competition_level: float = 0.5`
- **Line 74:** `avg_competitor_bid: float = 0.0`
- **Line 75:** `win_rate_last_10: float = 0.0`
- **Line 85:** `cross_device_confidence: float = 0.0`
- **Line 94:** `expected_conversion_value: float = 0.0`
- **Line 101:** `conversion_probability: float = 0.02`
- **Line 108:** `competitor_fatigue_level: float = 0.0`
- **VIOLATION:** Default values should be discovered or calculated
- **FIX:** Initialize from discovered patterns or calculate from data

### 8. HARDCODED BATCH SIZE
**File:** fortified_environment.py
- **Line 84:** `batch_size=100`
- **VIOLATION:** Batch size hardcoded
- **FIX:** Use ParameterManager

### 9. HARDCODED USER/DEVICE IDs
**File:** fortified_environment.py
- **Line 340:** `user_id = f"user_{datetime.now().timestamp()}_{np.random.randint(10000)}"`
- **Line 346:** `"device_id": f"device_{np.random.randint(10000)}"`
- **VIOLATION:** ID ranges hardcoded
- **FIX:** Use UUID or discovered ranges

## IMPACT
These hardcoded values prevent the system from:
1. **Learning** optimal bid ranges from actual auction dynamics
2. **Discovering** channels from real GA4 data
3. **Adapting** to different budget scales
4. **Normalizing** based on actual data distributions
5. **Configuring** hyperparameters dynamically

## REQUIRED ACTIONS
1. Remove ALL hardcoded values
2. Replace with discovered patterns from discovered_patterns.json
3. Use ParameterManager for configuration
4. Calculate statistics from actual data
5. Learn optimal values through training

## FILES AFFECTED
- fortified_rl_agent.py (MOST violations)
- fortified_environment.py  
- fortified_training_loop.py
- gaelp_master_integration.py
- training_orchestrator/rl_agent_*.py

## VERIFICATION
Run: `grep -r "MIN_BID\|MAX_BID\|1000\|100\|0\.[0-9]\|'organic'\|'paid_search'" --include="*.py" .`