# STRICT SUBAGENT RULES - NO EXCEPTIONS

## ABSOLUTE PROHIBITIONS
1. **NO SIMPLIFICATIONS** - Ever. If something is complex, implement it properly.
2. **NO HARDCODING** - All values must be discovered, learned, or configured.
3. **NO DUMMY/MOCK DATA** - Only real, working implementations.
4. **NO FALLBACKS** - If primary approach fails, FIX IT, don't bypass.
5. **NO TODO/FIXME** - Complete everything before marking done.
6. **NO SHORTCUTS** - Full implementation only.

## MANDATORY REQUIREMENTS
1. **TEST BEFORE COMPLETING** - Must run actual tests with output
2. **SHOW EVIDENCE** - Provide test results proving it works
3. **INTEGRATION TEST** - Must work with rest of system
4. **NO ISOLATED FIXES** - Changes must integrate properly

## VERIFICATION CHECKLIST
Before any subagent marks a task complete, they MUST:

### 1. Run These Tests:
```python
# Test the specific component
python3 -c "from [module] import [component]; [test_code]; print('✅ Working')"

# Test integration with system
python3 -c "from gaelp_master_integration import MasterOrchestrator, GAELPConfig; config = GAELPConfig(); orch = MasterOrchestrator(config); print('✅ System still initializes')"

# Run a training step
python3 -c "[test that training still works]"
```

### 2. Verify No Regressions:
```bash
# Check for forbidden patterns
grep -r "fallback\|simplified\|mock\|dummy\|TODO\|FIXME" --include="*.py" [changed_files]

# Verify no hardcoding
grep -r "hardcoded\|fixed_value\|HARDCODED" --include="*.py" [changed_files]
```

### 3. Provide Evidence:
- Show actual output from tests
- Show performance metrics (if applicable)
- Show that system still runs end-to-end

## SPECIFIC RULES FOR OUR 7-DAY SPRINT

### For RL State Dimension Fix:
- MUST handle variable dimensions properly
- MUST NOT corrupt neural network learning
- MUST test with actual training loop
- MUST show loss decreasing

### For Fantasy State Removal:
- MUST use ONLY platform-available metrics
- MUST verify with actual API documentation
- MUST NOT assume perfect tracking
- MUST handle missing data gracefully

### For Dashboard Fixes:
- MUST connect to real data sources
- MUST NOT show mock data
- MUST handle real-time updates
- MUST show actual metrics flowing through

### For Parallel Training:
- MUST actually run parallel environments
- MUST show speedup metrics
- MUST maintain learning stability
- MUST NOT break experience replay

## ENFORCEMENT MECHANISM

### Before Starting:
```python
# Subagent must acknowledge these rules
print("I acknowledge: NO simplifications, NO hardcoding, NO dummies, NO fallbacks")
print("I will test everything before marking complete")
```

### After Completion:
```python
# Subagent must provide test results
print("Test Results:")
print("1. Component test: [PASS/FAIL with output]")
print("2. Integration test: [PASS/FAIL with output]") 
print("3. No forbidden patterns: [PASS/FAIL with grep results]")
print("4. Performance: [metrics]")
```

## CONSEQUENCES
If a subagent violates these rules:
1. Task marked as FAILED, not complete
2. Must fix properly, no excuses
3. No credit for partial/simplified work

## REMEMBER
The user said: "WTF. Take out all fallbacks and make sure the primary system is working across the board"

This means:
- EVERYTHING must work properly
- NO shortcuts allowed
- Test results required
- Full implementation only