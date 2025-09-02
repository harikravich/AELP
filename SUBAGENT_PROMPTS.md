# SUBAGENT TASK PROMPTS - STRICT ENFORCEMENT

## TASK 1: Fix RL State Dimension Hashing

**AGENT**: hardcode-eliminator

**PROMPT**:
```
Fix the RL state dimension hashing issue in training_orchestrator/rl_agent_proper.py line 278.

CURRENT BROKEN CODE:
target_idx = hash(f"feature_{i}") % self.state_dim
state_vector[target_idx] += val * 0.1

This corrupts neural network learning by hashing features into existing dimensions.

REQUIRED FIX:
1. Implement proper state vector standardization
2. Pad or truncate cleanly without corruption
3. Maintain consistent dimensions for neural network

MANDATORY TESTS:
1. Create test showing state vectors maintain consistency
2. Run actual training loop showing loss decreases
3. Verify no dimension mismatches during training
4. Run for 100 training steps without errors

FORBIDDEN:
- NO simplified workarounds
- NO ignoring extra features
- NO hardcoded dimensions
- Must handle ANY number of features properly

You MUST provide test output showing training works after your fix.
Read SUBAGENT_RULES.md first and acknowledge.
```

## TASK 2: Remove Fantasy State Data

**AGENT**: fallback-eliminator

**PROMPT**:
```
Replace fantasy state data in gaelp_master_integration.py lines 2024-2031.

CURRENT FANTASY DATA:
- user.touchpoints (impossible to track cross-platform)
- user.fatigue_level (can't measure)
- user.lifetime_value (can't predict exactly)

REQUIRED REPLACEMENT:
Use ONLY metrics available from real platform APIs:
- campaign_ctr (last 7 days)
- campaign_cpc
- impression_share
- budget_utilization
- time_of_day
- device_performance

MANDATORY VERIFICATION:
1. Check Google Ads API documentation
2. Check Facebook Marketing API documentation
3. Verify each metric is actually available
4. Show code fetching real metrics (even if simulated for now)

MANDATORY TESTS:
1. Show system runs with new realistic state
2. Show RL agent can learn with new state
3. Verify state dimension is consistent
4. Run 50 training steps successfully

FORBIDDEN:
- NO perfect user tracking
- NO impossible metrics
- NO assumptions about data availability
- Must handle missing data gracefully

Read SUBAGENT_RULES.md first and acknowledge.
```

## TASK 3: Dashboard Data Architecture

**AGENT**: project-coordinator

**PROMPT**:
```
Fix dashboard data architecture in gaelp_live_dashboard_enhanced.py.

PROBLEMS TO FIX:
1. Four redundant tracking systems (platform_tracking, channel_tracking, etc.)
2. String/float conversion errors in update_from_realistic_step()
3. Enterprise sections show mock data instead of real data

REQUIRED IMPLEMENTATION:
1. Create single UnifiedDataManager class
2. Fix ALL type conversion issues
3. Connect ALL 6 enterprise sections to real data sources
4. No mock data anywhere

MANDATORY TESTS:
1. Start dashboard and show it runs
2. Process 10 step results without errors
3. Show real data flowing to all sections
4. Verify no type conversion errors

VERIFICATION CHECKLIST:
- Creative Studio: Connected to real creative_tracking
- Audience Hub: Shows ML-discovered segments
- War Room: Real auction data
- Attribution Center: Real attribution logic
- AI Arena: Real RL training metrics
- Executive Dashboard: Real calculated KPIs

FORBIDDEN:
- NO mock data in ANY section
- NO simplified placeholders
- NO "will implement later"
- Everything must work NOW

Read SUBAGENT_RULES.md first and acknowledge.
```

## ENFORCEMENT PROTOCOL

Before launching any subagent:
1. They must read SUBAGENT_RULES.md
2. They must acknowledge the rules
3. They must show test results before marking complete
4. We verify their work independently

After subagent completes:
1. We run their tests ourselves
2. We check for forbidden patterns
3. We verify integration works
4. Only then mark as complete