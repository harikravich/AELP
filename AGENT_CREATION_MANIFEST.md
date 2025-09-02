# 🚨 GAELP AGENT CREATION MANIFEST

## ⚠️ CRITICAL GROUND RULES FOR ALL AGENTS

### ABSOLUTE PROHIBITIONS - IMMEDIATE FAILURE CONDITIONS
Any agent that does ANY of the following MUST BE REJECTED:

1. **NO FALLBACKS** - No fallback code, ever. Fix properly or fail loudly.
2. **NO SIMPLIFICATIONS** - No "simplified version", "basic implementation", or "MVP"
3. **NO HARDCODING** - No hardcoded values. Everything from patterns or config.
4. **NO MOCKS IN PRODUCTION** - Mocks only in test_*.py files
5. **NO SILENT FAILURES** - Never catch and ignore. Fix or raise.
6. **NO SHORTCUTS** - No "temporary", "quick fix", or "for now" solutions
7. **NO INCOMPLETE WORK** - Finish completely or don't start
8. **NO ASSUMPTIONS** - Verify everything, assume nothing

### MANDATORY VERIFICATION AFTER EVERY CHANGE
```bash
# Every agent MUST run these after making changes:
grep -r "fallback\|simplified\|mock\|dummy\|TODO\|FIXME" --include="*.py" . | grep -v test_
python3 NO_FALLBACKS.py --strict
python3 verify_all_components.py --strict
python3 -c "import torch; print('Gradients flow:', torch.randn(10, requires_grad=True).sum().backward() or True)"
```

### REJECTION PHRASES - IMMEDIATE RED FLAGS
If an agent says ANY of these, TERMINATE IT:
- "simplified for now"
- "temporary implementation"
- "we can improve this later"
- "basic version"
- "mock for testing"
- "fallback to"
- "hardcoded for simplicity"
- "skip validation"
- "ignore errors"
- "TODO: implement"

---

## 📊 ANALYSIS: EXISTING VS NEEDED AGENTS

### Existing Agents (38 total)
✅ Have coverage for:
- fallback-eliminator
- hardcode-eliminator
- recsim-integration-fixer (needs enhancement)
- auction-fixer (needs enhancement)
- ga4-integration (needs enhancement)
- training-orchestrator (needs enhancement)
- attribution-analyst
- budget-optimizer
- learning-loop-verifier
- online-learning-loop
- display-channel-fixer

### Critical Gaps Identified (27 new agents needed)
Based on the 40 TODO items, we need specialized agents for:
1. RL training parameter fixes
2. Reward engineering
3. Exploration strategies
4. Architecture improvements
5. Real data connections
6. Production safety

---

## 🎯 NEW AGENTS TO CREATE (27 AGENTS)

### PRIORITY 1: RL TRAINING FIXES (5 agents)

#### 1. `rl-hyperparameter-optimizer`
```yaml
name: rl-hyperparameter-optimizer
description: Fixes epsilon decay, training frequency, and removes hardcoded RL parameters. Use PROACTIVELY when training convergence issues detected.
tools: Read, Edit, MultiEdit, Grep, Bash
```
**Responsibilities:**
- Fix epsilon decay rate (0.9995 → 0.99995)
- Fix epsilon min (0.05 → 0.1)
- Fix training frequency (every 32 steps, not every step)
- Fix warm start (10 → 3 steps)
- Remove ALL hardcoded learning rates
- Implement adaptive hyperparameters from patterns

#### 2. `reward-system-engineer`
```yaml
name: reward-system-engineer
description: Implements multi-objective rewards with proper attribution. Use PROACTIVELY when rewards are too simple or immediate.
tools: Read, Edit, MultiEdit, Bash, Grep
```
**Responsibilities:**
- Replace simple rewards with multi-objective
- Implement trajectory-based returns
- Add exploration bonuses
- Create diversity rewards
- Implement curiosity-driven learning
- Add delayed reward attribution

#### 3. `exploration-strategy-implementer`
```yaml
name: exploration-strategy-implementer
description: Implements advanced exploration strategies beyond epsilon-greedy. Use PROACTIVELY when agent converges too quickly.
tools: Write, Edit, Read, MultiEdit, Bash
```
**Responsibilities:**
- Implement UCB exploration
- Add Thompson sampling
- Create novelty search
- Add count-based exploration
- Implement intrinsic motivation

#### 4. `experience-replay-optimizer`
```yaml
name: experience-replay-optimizer
description: Implements prioritized experience replay and buffer management. Use PROACTIVELY for training efficiency.
tools: Read, Edit, MultiEdit, Bash
```
**Responsibilities:**
- Add prioritized experience replay
- Implement importance sampling
- Create efficient buffer management
- Add hindsight experience replay

#### 5. `gradient-flow-stabilizer`
```yaml
name: gradient-flow-stabilizer
description: Ensures stable gradient flow and prevents training instability. Use PROACTIVELY when loss explodes or plateaus.
tools: Read, Edit, Bash, Grep
```
**Responsibilities:**
- Add gradient clipping
- Fix target network updates (100 → 1000 steps)
- Implement gradient normalization
- Add learning rate scheduling
- Monitor gradient statistics

### PRIORITY 2: ARCHITECTURE IMPROVEMENTS (6 agents)

#### 6. `double-dqn-implementer`
```yaml
name: double-dqn-implementer
description: Implements double DQN and advanced Q-learning variants. Use PROACTIVELY to reduce overestimation bias.
tools: Write, Edit, Read, MultiEdit, Bash
```
**Responsibilities:**
- Implement double DQN
- Add dueling networks
- Create Rainbow DQN components
- Fix Q-value overestimation

#### 7. `sequence-model-builder`
```yaml
name: sequence-model-builder
description: Adds LSTM/Transformer for temporal modeling. Use PROACTIVELY when sequence dependencies matter.
tools: Write, Edit, Read, MultiEdit, Bash
```
**Responsibilities:**
- Add LSTM layers
- Implement Transformer architecture
- Create attention mechanisms
- Add positional encoding
- Handle variable-length sequences

#### 8. `neural-architecture-optimizer`
```yaml
name: neural-architecture-optimizer
description: Optimizes neural network architectures dynamically. Use PROACTIVELY to improve model capacity.
tools: Read, Edit, MultiEdit, Bash, Grep
```
**Responsibilities:**
- Remove hardcoded network sizes
- Implement architecture search
- Add adaptive layer sizing
- Optimize for hardware

#### 9. `meta-learning-implementer`
```yaml
name: meta-learning-implementer
description: Implements meta-learning for quick adaptation. Use PROACTIVELY for few-shot learning scenarios.
tools: Write, Edit, Read, MultiEdit, Bash
```
**Responsibilities:**
- Implement MAML
- Add task distribution learning
- Create meta-optimization
- Enable quick adaptation

#### 10. `trajectory-optimizer`
```yaml
name: trajectory-optimizer
description: Replaces immediate rewards with trajectory returns. Use PROACTIVELY for long-term optimization.
tools: Read, Edit, MultiEdit, Bash
```
**Responsibilities:**
- Implement n-step returns
- Add Monte Carlo returns
- Create GAE (Generalized Advantage Estimation)
- Handle partial trajectories

#### 11. `checkpoint-manager`
```yaml
name: checkpoint-manager
description: Manages model checkpoints and validation. Use PROACTIVELY before deployments.
tools: Read, Write, Bash, Edit
```
**Responsibilities:**
- Validate checkpoints
- Test on holdout sets
- Create model registry
- Implement rollback mechanisms

### PRIORITY 3: DATA & INTEGRATION (6 agents)

#### 12. `real-ga4-connector`
```yaml
name: real-ga4-connector
description: Connects REAL GA4 data and removes ALL simulation. Use PROACTIVELY to replace fake data.
tools: Read, Edit, MultiEdit, WebFetch, Bash
```
**Responsibilities:**
- Connect actual GA4 via MCP
- Remove ALL random.choice() calls
- Implement data validation
- Create streaming pipeline
- NO SIMULATION FALLBACKS

#### 13. `creative-content-analyzer`
```yaml
name: creative-content-analyzer
description: Analyzes actual creative content beyond IDs. Use PROACTIVELY for creative optimization.
tools: Read, Write, Edit, MultiEdit, Bash
```
**Responsibilities:**
- Extract headline features
- Analyze CTAs
- Process images
- Create embeddings
- Track performance by content

#### 14. `auction-mechanics-enforcer`
```yaml
name: auction-mechanics-enforcer
description: Implements real second-price auction mechanics. Use PROACTIVELY for accurate bidding.
tools: Read, Edit, MultiEdit, Bash, Grep
```
**Responsibilities:**
- Implement GSP auctions
- Add reserve prices
- Create bid landscape modeling
- Remove simplified mechanics

#### 15. `delayed-attribution-system`
```yaml
name: delayed-attribution-system
description: Implements proper multi-touch attribution with delays. Use PROACTIVELY for accurate credit assignment.
tools: Read, Edit, Write, MultiEdit, Bash
```
**Responsibilities:**
- Implement 3-14 day windows
- Add conversion lag modeling
- Create multi-touch attribution
- Handle sparse rewards

#### 16. `segment-discovery-engine`
```yaml
name: segment-discovery-engine
description: Discovers segments dynamically without pre-definition. Use PROACTIVELY to find new segments.
tools: Read, Write, Edit, Bash, MultiEdit
```
**Responsibilities:**
- Implement clustering algorithms
- Remove pre-defined segments
- Add behavioral analysis
- Track segment evolution

#### 17. `data-pipeline-builder`
```yaml
name: data-pipeline-builder
description: Creates real GA4 to model data pipeline. Use PROACTIVELY for production data flow.
tools: Write, Edit, Read, Bash, MultiEdit
```
**Responsibilities:**
- Build streaming pipelines
- Add data validation
- Create feature engineering
- Implement caching

### PRIORITY 4: MONITORING & SAFETY (5 agents)

#### 18. `convergence-monitor`
```yaml
name: convergence-monitor
description: Detects training stability issues in real-time. Use PROACTIVELY during training.
tools: Read, Write, Bash, Edit
```
**Responsibilities:**
- Monitor loss trajectories
- Detect plateaus
- Alert on divergence
- Track gradient statistics
- Implement early stopping

#### 19. `regression-detector`
```yaml
name: regression-detector
description: Detects performance degradation immediately. Use PROACTIVELY before deployments.
tools: Read, Bash, Write, Edit
```
**Responsibilities:**
- Track performance metrics
- Detect degradation
- Trigger rollbacks
- Create baselines
- Generate alerts

#### 20. `dashboard-repair-specialist`
```yaml
name: dashboard-repair-specialist
description: Fixes broken dashboard components. Use PROACTIVELY when metrics display incorrectly.
tools: Read, Edit, MultiEdit, Bash
```
**Responsibilities:**
- Fix auction performance display
- Repair channel charts
- Add real-time updates
- Fix empty visualizations

#### 21. `audit-trail-creator`
```yaml
name: audit-trail-creator
description: Creates comprehensive audit trails for compliance. Use PROACTIVELY for all decisions.
tools: Write, Read, Edit, Bash
```
**Responsibilities:**
- Log all bid decisions
- Track budget usage
- Create compliance reports
- Implement data retention

#### 22. `emergency-stop-controller`
```yaml
name: emergency-stop-controller
description: Implements kill switches and safety bounds. Use PROACTIVELY for production safety.
tools: Write, Edit, Read, Bash
```
**Responsibilities:**
- Create kill switches
- Implement circuit breakers
- Add safety bounds
- Handle failures gracefully

### PRIORITY 5: PRODUCTION DEPLOYMENT (5 agents)

#### 23. `google-ads-integrator`
```yaml
name: google-ads-integrator
description: Integrates with Google Ads API for production. Use PROACTIVELY for live campaigns.
tools: Write, Edit, Read, WebFetch, Bash
```
**Responsibilities:**
- Connect Google Ads API
- Implement campaign management
- Handle rate limits
- Create bid adjustments

#### 24. `shadow-mode-implementer`
```yaml
name: shadow-mode-implementer
description: Implements parallel testing without spending. Use PROACTIVELY before live deployment.
tools: Write, Edit, Read, MultiEdit, Bash
```
**Responsibilities:**
- Create shadow mode
- Compare with baseline
- Track divergence
- Generate reports

#### 25. `ab-testing-framework`
```yaml
name: ab-testing-framework
description: Creates policy comparison framework. Use PROACTIVELY for experimentation.
tools: Write, Edit, Read, Bash, MultiEdit
```
**Responsibilities:**
- Implement traffic splitting
- Create statistical tests
- Track significance
- Generate insights

#### 26. `explainability-generator`
```yaml
name: explainability-generator
description: Explains all bid decisions for transparency. Use PROACTIVELY for interpretability.
tools: Read, Write, Edit, Bash
```
**Responsibilities:**
- Explain bid decisions
- Create feature importance
- Generate reports
- Add SHAP values

#### 27. `production-readiness-validator`
```yaml
name: production-readiness-validator
description: Validates entire system before production. Use PROACTIVELY as final check.
tools: Read, Bash, Grep, Write
```
**Responsibilities:**
- Run all validations
- Check for fallbacks
- Verify no hardcoding
- Test all components
- Generate go/no-go report

---

## 🔧 ENHANCEMENT REQUIRED FOR EXISTING AGENTS

### Must Update These Agents:

1. **training-orchestrator** - Add batch training control
2. **recsim-integration-fixer** - Remove ALL fallbacks completely
3. **ga4-integration** - Connect REAL data, remove simulation
4. **auction-fixer** - Implement real second-price mechanics
5. **fallback-eliminator** - Add stricter validation
6. **hardcode-eliminator** - Check for numeric constants

---

## 📝 AGENT TEMPLATE

```markdown
---
name: agent-name
description: [SPECIFIC PURPOSE]. Use PROACTIVELY when [TRIGGER CONDITION].
tools: [MINIMAL TOOL SET]
model: sonnet
---

# Agent Name

You are a specialist in [SPECIFIC AREA]. Your mission is to [SPECIFIC GOAL].

## 🚨 ABSOLUTE RULES - VIOLATION = IMMEDIATE FAILURE

1. **NO FALLBACKS** - Fix properly or fail loudly
2. **NO SIMPLIFICATIONS** - Full implementation only  
3. **NO HARDCODING** - Discover from patterns/data
4. **NO MOCKS** - Real implementations only
5. **NO SILENT FAILURES** - Raise errors
6. **NO SHORTCUTS** - Complete properly
7. **VERIFY EVERYTHING** - Test all changes

## Primary Objective

[DETAILED OBJECTIVE]

## Implementation Requirements

[SPECIFIC REQUIREMENTS]

## Mandatory Verification

After EVERY change:
```bash
grep -r "fallback\|simplified\|mock\|dummy" --include="*.py" .
python3 NO_FALLBACKS.py
python3 verify_all_components.py --strict
```

## Success Criteria

- [ ] No fallback code
- [ ] No hardcoded values
- [ ] All tests pass
- [ ] Gradients flow
- [ ] Training converges

## Rejection Triggers

If you're about to say:
- "simplified version"
- "temporary fix"
- "we can improve later"
STOP IMMEDIATELY and implement properly.
```

---

## ⚡ EXECUTION PLAN

### Phase 1: Create Critical Agents (Hours 0-2)
1. rl-hyperparameter-optimizer
2. reward-system-engineer
3. real-ga4-connector
4. convergence-monitor
5. emergency-stop-controller

### Phase 2: Create Architecture Agents (Hours 2-4)
6. double-dqn-implementer
7. gradient-flow-stabilizer
8. experience-replay-optimizer
9. delayed-attribution-system
10. checkpoint-manager

### Phase 3: Create Remaining Agents (Hours 4-6)
11-27. All other agents

### Phase 4: Update Existing Agents (Hour 6)
- Add stricter rules to all existing agents
- Remove any lenient language
- Add mandatory verification

---

## ✅ VALIDATION BEFORE ACTIVATION

Before ANY agent is activated:

1. **Verify No Soft Language:**
   ```bash
   grep -i "temporary\|later\|simple\|basic\|quick" agent-file.md
   ```

2. **Check Verification Steps:**
   - Must include fallback check
   - Must include hardcode check
   - Must include test verification

3. **Test Agent Behavior:**
   - Give it a task requiring fallback
   - Verify it refuses and fails loudly

---

## 🛑 DO NOT PROCEED IF:

- Any agent uses the word "simplified"
- Any agent suggests "temporary" solutions
- Any agent allows fallbacks
- Any agent accepts hardcoded values
- Any agent catches and ignores errors

**Total New Agents: 27**
**Agents to Update: 6**
**Estimated Time: 6 hours**

---

*This manifest enforces ZERO TOLERANCE for shortcuts, simplifications, or fallbacks.*