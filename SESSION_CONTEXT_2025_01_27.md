# GAELP Session Context - January 27, 2025

## CRITICAL STATUS: MASSIVE VIOLATIONS FOUND

### What to Tell Next Session:

**START WITH THIS:**
"The codebase has 576+ violations of CLAUDE.md. Run `python3 NO_FALLBACKS.py` to see them. DO NOT work on anything else until these are fixed. The system is full of fallbacks and hardcoded values that violate the core requirements."

## CodeGPT MCP Status
- API Key: `sk-a83d1ada-756a-4eb8-a8f6-7a6b6907a799`
- Wrapper Script: `/tmp/codegpt_wrapper.sh` (may need recreating)
- Status: Connected but tools not visible yet
- Fix needed: May need CODEGPT_ORG_ID environment variable

## CRITICAL VIOLATIONS FOUND

### 1. Fallback Code (FORBIDDEN)
- `gaelp_master_integration.py`: 7+ fallback methods
- `gaelp_dynamic_budget_optimizer.py`: `_fallback_allocation()` method
- `enhanced_simulator.py`: Fallback competitors
- `performance_driven_budget_optimizer.py`: Safe fallback allocation

### 2. Feature Flags Set to False (FORBIDDEN)
```python
# enhanced_simulator.py
CREATIVE_INTEGRATION_AVAILABLE = False
AUCTION_INTEGRATION_AVAILABLE = False  
RECSIM_BRIDGE_AVAILABLE = False
RECSIM_AVAILABLE = False

# attribution_models.py
CONVERSION_LAG_MODEL_AVAILABLE = False
```

### 3. Hardcoded Values (FORBIDDEN)
- Bid amounts: `bid_amount *= 1.2`, `bid_amount *= 0.85`
- Conversion rates: `cvr = 0.02`
- Budget values: `budget_risk=0.1`
- Thresholds: `threshold=0.8`
- Magic numbers everywhere: 30 day windows, 0.95 probabilities, etc.

### 4. Mock/Dummy Code (FORBIDDEN)
- `crisis_parent_training_demo.py`: `generate_mock_experiences()`
- `display_bot_filter.py`: `generate_mock_placement_data()`
- Multiple mock implementations instead of real ones

## TODO List (IN ORDER OF PRIORITY)

1. **Fix CodeGPT MCP connection** - needs proper wrapper script
2. **Fix ALL fallback violations in gaelp_master_integration.py** (576 violations found)
3. **Remove ALL hardcoded values** from entire codebase
4. **Fix enhanced_simulator.py** - remove CREATIVE_INTEGRATION_AVAILABLE = False
5. **Fix gaelp_dynamic_budget_optimizer.py** - remove _fallback_allocation
6. **Implement PROPER RecSim integration** (currently RECSIM_AVAILABLE = False)
7. **Implement PROPER AuctionGym integration** (currently AUCTION_INTEGRATION_AVAILABLE = False)
8. **Verify RL agents actually learn** (not just random actions)
9. **Run NO_FALLBACKS.py and fix ALL violations**

## Commands to Run IMMEDIATELY

```bash
# SEE ALL VIOLATIONS
python3 NO_FALLBACKS.py

# Check for fallbacks
grep -r "fallback\|simplified\|mock\|dummy" --include="*.py" . | grep -v test_

# Check for disabled features
grep -r "_AVAILABLE = False" --include="*.py" .

# Verify components
python3 verify_all_components.py --strict
```

## The Brutal Truth

The GAELP implementation is **NOT WORKING PROPERLY**:
- RecSim integration is **DISABLED** 
- AuctionGym integration is **DISABLED**
- Creative integration is **DISABLED**
- System falls back to simplified/random behavior
- 576+ violations of core requirements
- Hardcoded values everywhere instead of learned parameters

## Next Steps

1. DO NOT add new features
2. DO NOT write new code
3. FIX the existing violations first
4. Make RecSim actually work
5. Make AuctionGym actually work
6. Remove ALL fallbacks
7. Remove ALL hardcoded values
8. Then verify with NO_FALLBACKS.py

## Remember CLAUDE.md

- **NEVER** implement fallback code
- **NEVER** use simplified versions
- **NEVER** use mock implementations
- **NEVER** skip components that are "hard"
- If something doesn't work, FIX IT PROPERLY

The user said: "WTF. Take out all fallbacks and make sure the primary system is working across the board"

THIS IS NOT DONE YET.