# CRITICAL INSTRUCTIONS FOR CLAUDE CODE

## ABSOLUTE REQUIREMENTS - NO EXCEPTIONS

### 1. NO FALLBACKS OR SIMPLIFICATIONS
- **NEVER** implement fallback code
- **NEVER** use simplified versions
- **NEVER** use mock implementations
- **NEVER** skip components that are "hard"
- If something doesn't work, FIX IT PROPERLY

### 2. MANDATORY IMPLEMENTATIONS
When implementing GAELP or any RL system:
- **MUST** use proper Reinforcement Learning (Q-learning/PPO), NOT bandits
- **MUST** use RecSim for user simulation, NOT simple random
- **MUST** use AuctionGym for auctions, NOT simplified mechanics
- **MUST** implement ALL components, not just the easy ones
- **MUST** handle errors by FIXING them, not bypassing

### 3. TESTING REQUIREMENTS
- **MUST** test every component thoroughly
- **MUST** verify components actually work, not just compile
- **MUST** ensure data flows through entire system
- **MUST** check that learning actually happens

### 4. FORBIDDEN PATTERNS
NEVER write code containing:
- `fallback`
- `simplified`
- `mock` (except in actual test files)
- `dummy`
- `TODO` or `FIXME` (fix it NOW)
- `not available`
- `_AVAILABLE = False`
- `try/except` that ignores errors
- **HARDCODED VALUES** - NO hardcoded segments, categories, or constants
- **STATIC LISTS** - System must discover/learn these at runtime
- **FIXED PARAMETERS** - All parameters must be learned or configured

### 5. WHEN BLOCKED
If you encounter a difficult implementation:
1. DO NOT simplify
2. DO NOT skip
3. DO NOT use fallbacks
4. Research the proper solution
5. Implement it correctly
6. Test thoroughly

### 6. ARCHITECTURE DECISIONS
- User journeys require REINFORCEMENT LEARNING, not bandits
- Auctions require proper second-price mechanics
- Attribution requires multi-touch models
- Everything must handle delayed rewards
- **NO HARDCODING** - System discovers segments, categories, parameters at runtime
- **DYNAMIC LEARNING** - All thresholds, weights, and parameters are learned
- **NEVER DOWNSCALE** - Always solve problems as designed, no simplifications

## ENFORCEMENT
The NO_FALLBACKS.py module will raise errors if you try to use fallbacks.
This is intentional. Fix the actual problem.

## COMMANDS TO RUN
When working on GAELP:
```bash
# Check for fallback code
grep -r "fallback\|simplified\|mock\|dummy" --include="*.py" .

# Run strict mode test
python3 NO_FALLBACKS.py

# Verify all components
python3 verify_all_components.py --strict
```

## REMEMBER
The user explicitly said: "WTF. Take out all fallbacks and make sure the primary system is working across the board"

This means:
- NO SHORTCUTS
- NO SIMPLIFICATIONS  
- EVERYTHING MUST WORK PROPERLY
- FIX THE HARD PROBLEMS
- **BE BRUTALLY HONEST** - Tell the truth about what's working and what isn't

If you're about to write a fallback, STOP and implement it properly instead.