# GAELP PRODUCTION DEPLOYMENT - 7-DAY PLAN
## Target: CPA < $100 with $500/day budget

## VERIFIED ISSUES FROM ANALYSIS
### ✅ Confirmed Problems:
1. **RL state dimension hashing corruption** (rl_agent_proper.py:278) - breaks learning
2. **Fantasy state data** - using user.lifetime_value, perfect tracking (unrealistic)
3. **Dashboard has 4 redundant tracking systems** - needs UnifiedDataManager
4. **String/float conversion errors** in update_from_realistic_step()
5. **6 enterprise sections using mock data** instead of real data
6. **Training too slow** - needs 50x parallelization for 48-hour training

### ❌ Non-Issues (Already Working):
- Criteo model AUC 0.827 (good, not overfitted)
- RecSim/AuctionGym integration working
- GA4 data pipeline functional
- Platform API scaffolding ready

## DAY-BY-DAY EXECUTION PLAN

### DAY 1: Fix Critical Training Blockers
- [ ] Fix RL state dimension hashing corruption in rl_agent_proper.py
- [ ] Replace fantasy state with platform-observable metrics only
- [ ] Add dense reward signals for faster convergence

### DAY 2: Dashboard Data Architecture
- [ ] Create UnifiedDataManager to eliminate 4 redundant tracking systems
- [ ] Fix string/float conversion issues in update_from_realistic_step
- [ ] Connect all 6 enterprise sections to real data sources

### DAY 3: Training Acceleration
- [ ] Implement 50x parallel environment training
- [ ] Add experience replay prioritization
- [ ] Bootstrap with GA4 historical data

### DAY 4: Platform Integration
- [ ] Connect Google Ads API with OAuth
- [ ] Connect Facebook Marketing API
- [ ] Add TikTok and Bing APIs

### DAY 5-6: 48-Hour Training Sprint
- [ ] Run 100,000 episodes with parallel training
- [ ] Monitor convergence via dashboard
- [ ] Validate with small budget test ($50/day)

### DAY 7: Production Deployment
- [ ] Deploy with $500/day budget across platforms
- [ ] Monitor CPA and optimize in real-time
- [ ] Target: CPA < $100 with high volume

## SPECIFIC FIXES NEEDED

### 1. Fix RL State Dimension (rl_agent_proper.py:278)
```python
# BROKEN - Hashing corrupts learning:
target_idx = hash(f"feature_{i}") % self.state_dim

# FIX - Pad or truncate safely:
def _standardize_state_vector(self, state_vector):
    if len(state_vector) < self.state_dim:
        return np.pad(state_vector, (0, self.state_dim - len(state_vector)))
    else:
        return state_vector[:self.state_dim]
```

### 2. Replace Fantasy State (gaelp_master_integration.py:2030)
```python
# FANTASY (current):
user_data = {
    'touchpoints_seen': user.touchpoints,      # ❌ Can't track cross-platform
    'ad_fatigue_level': user.fatigue_level,    # ❌ Can't measure
    'estimated_ltv': user.lifetime_value       # ❌ Can't predict exactly
}

# REALISTIC (fix):
realistic_state = {
    'campaign_ctr': last_7_days_ctr,           # ✅ Available from APIs
    'campaign_cpc': last_7_days_cpc,           # ✅ Available
    'impression_share': current_impression_share, # ✅ Available
    'budget_utilization': spend / daily_budget,   # ✅ Available
    'time_of_day': current_hour,               # ✅ Available
    'device_performance': mobile_vs_desktop_ctr,  # ✅ Available
}
```

### 3. Dashboard UnifiedDataManager
```python
class UnifiedDataManager:
    """Single source of truth for all dashboard data"""
    def __init__(self):
        self.metrics = MetricsStore()
        self.channels = ChannelManager()  # Replaces 4 tracking systems
        self.auctions = AuctionTracker()
        self.creatives = CreativePerformanceTracker()
```

### 4. Training Acceleration
```python
# Current: 1 environment = weeks to train
# Target: 50 parallel environments = 48 hours
parallel_envs = ParallelWorldSimulator(n_worlds=50)
```

## SUCCESS METRICS
- **CPA Target**: < $100
- **Volume Target**: 1000+ conversions/week
- **Budget**: $500/day across all platforms
- **Training Time**: 48 hours (not weeks)
- **Platforms**: Google, Facebook, TikTok, Bing

## NOTES
- Dashboard currently not running (needs `python3 gaelp_live_dashboard_enhanced.py`)
- Checkpoint loading fixed with weights_only=False
- Array shape error in replay buffer fixed with np.stack
- System initializes successfully but needs above fixes for production