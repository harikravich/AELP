# Complete GAELP System Architecture
## Mapping the ENTIRE Learning System

## 🎯 The Core Problem We're Solving

**Teaching an RL agent to optimize ad campaigns BEFORE real-world deployment**
- Users take 3-14 days to convert
- Multiple touchpoints across channels
- Competitive auctions with 5+ bidders
- $100 CAC target, $180 LTV product
- Learn from simulations to avoid wasting real money

## 🧠 Current RL System Status

### What We Have:
1. **PPO Agent** (`journey_aware_rl_agent.py`)
   - 41-feature state space
   - LSTM for sequence processing
   - Journey-aware reward shaping
   - But NOT connected to real data!

2. **Training Infrastructure** (`gaelp_integration.py`)
   - Checkpointing system
   - Wandb tracking (offline mode)
   - Episode management
   - But using fake data!

3. **Offline RL** (`offline_rl_trainer.py`)
   - D3rlpy CQL implementation
   - Can learn from historical data
   - But no real historical data!

### ❌ What's NOT Working:
- **No persistent user tracking** across episodes
- **No multi-day journeys** (resets each episode)
- **No competitive learning** (competitors don't adapt)
- **No real attribution** (last-click only)
- **Reward is immediate** (not delayed conversion)

## 🏗️ Complete Architecture Design

### 1. USER JOURNEY PERSISTENCE LAYER

```python
class UserJourneyDatabase:
    """
    Maintains user state across episodes and days
    This is CRITICAL - users don't reset between episodes!
    """
    
    users = {}  # user_id -> UserJourney
    
    class UserJourney:
        user_id: str
        created_at: datetime
        
        # RecSim state (persists)
        segment: UserSegment  # Can change over time!
        attention_span: float
        fatigue_level: float
        
        # Journey state
        touchpoints: List[Touchpoint]
        current_stage: Stage  # UNAWARE -> AWARE -> CONSIDERING -> INTENT -> CONVERTED
        days_in_journey: int
        
        # Cross-channel tracking
        devices_seen: Set[Device]
        channels_exposed: Set[Channel]
        competitors_seen: Set[Competitor]
        
        # Probabilistic matching
        match_confidence: float  # How sure we are this is the same user
        possible_aliases: List[str]  # Other IDs that might be this user
```

### 2. MONTE CARLO SIMULATION APPROACH

```python
class MonteCarloSimulation:
    """
    Run MANY parallel simulations to learn faster
    """
    
    def run_episode_batch(self, n_simulations=100):
        """
        Instead of 1 sequential episode, run 100 parallel worlds
        """
        worlds = []
        for i in range(n_simulations):
            # Each world has different:
            # - User population sample
            # - Competitor strategies
            # - Market conditions
            # - Random seeds
            world = SimulationWorld(
                users=self.sample_users(1000),
                competitors=self.sample_competitor_strategies(),
                seasonality=self.sample_time_period(),
                market_noise=np.random.normal(1.0, 0.1)
            )
            worlds.append(world)
        
        # Run all worlds in parallel
        results = parallel_map(self.run_world, worlds)
        
        # Agent learns from ALL experiences
        return aggregate_experiences(results)
    
    def importance_sampling(self, experiences):
        """
        Weight experiences by how likely they are in reality
        """
        weights = []
        for exp in experiences:
            # Crisis parents are rare but valuable
            if exp.user.is_crisis:
                weight = 0.1 * 10  # 10% probability * 10x value
            else:
                weight = 0.9 * 1
            weights.append(weight)
        
        return weighted_learning(experiences, weights)
```

### 3. MULTI-DAY JOURNEY TRACKING

```python
class MultiDayEnvironment(gym.Env):
    """
    Environment that maintains state across days
    """
    
    def __init__(self):
        self.day = 0
        self.user_db = UserJourneyDatabase()
        self.budget_remaining = {}  # Per day budgets
        
    def step(self, action):
        """
        Each step is an HOUR, not a full episode
        """
        hour = self.current_hour
        
        # Get users active this hour
        active_users = self.get_active_users(hour)
        
        for user in active_users:
            # Check if user is continuing journey
            journey = self.user_db.get_journey(user.id)
            
            if journey.days_in_journey > 14:
                # Journey timeout, mark as lost
                journey.status = 'LOST'
                reward = -journey.total_cost  # Negative reward for waste
                
            elif journey.is_converted:
                # Already converted, skip
                continue
                
            else:
                # User is still in journey
                # Run auction for this user
                auction_result = self.run_auction(user, action)
                
                if auction_result.won:
                    # Update journey
                    journey.add_touchpoint(
                        channel=action.channel,
                        bid=auction_result.price,
                        position=auction_result.position
                    )
                    
                    # Check for conversion (DELAYED REWARD)
                    if self.check_conversion(journey):
                        # Conversion might happen days later!
                        reward = self.calculate_attribution_reward(journey)
                    else:
                        # No reward yet, journey continues
                        reward = -auction_result.price  # Just the cost
        
        # Advance time
        if hour == 23:
            self.day += 1
            self.reset_daily_budgets()
        
        return state, reward, done, info
```

### 4. STATE REPRESENTATION FOR SEQUENTIAL DECISIONS

```python
class SequentialState:
    """
    State must capture journey history and future potential
    """
    
    # User journey features (THIS IS KEY)
    journey_features = {
        'touchpoint_sequence': [ch1, ch2, ch3],  # Last 3 channels
        'days_since_first_touch': 5,
        'total_spend_on_user': 12.50,
        'competitor_exposures': ['Qustodio', 'Bark'],
        'stage_progression': 'AWARE->CONSIDERING',
        'engagement_trend': 'increasing',  # or decreasing, stable
    }
    
    # User segment features (from RecSim)
    user_features = {
        'segment': 'researcher',
        'attention_span': 8.0,
        'fatigue_level': 0.3,
        'price_sensitivity': 0.7,
        'urgency_score': 0.2,  # Derived from behavior
    }
    
    # Market features (from AuctionGym)
    market_features = {
        'avg_cpc_trend': 'rising',
        'competitor_aggression': 0.8,
        'time_of_day': 14,
        'day_of_week': 'Tuesday',
        'seasonality': 'back_to_school',
    }
    
    # Budget features
    budget_features = {
        'daily_remaining': 450.00,
        'monthly_remaining': 8500.00,
        'current_roi': 2.3,
        'pacing': 'on_track',  # or ahead, behind
    }
    
    def to_tensor(self):
        # Flatten all features into RL state vector
        # Use embeddings for categorical
        # Normalize numerics
        return torch.tensor([...])
```

### 5. REWARD FUNCTION WITH ATTRIBUTION

```python
class AttributionRewardFunction:
    """
    Sophisticated reward that handles multi-touch attribution
    """
    
    def calculate_reward(self, journey, conversion_value=180):
        if not journey.is_converted:
            # No conversion yet, just return cost
            return -journey.last_touchpoint.cost
        
        # CONVERSION! Now attribute credit
        attribution = self.multi_touch_attribution(journey)
        
        # Our touchpoints
        our_credit = sum(
            credit for tp, credit in attribution.items()
            if tp.advertiser == 'Aura'
        )
        
        # Calculate reward
        attributed_value = conversion_value * our_credit
        total_cost = sum(tp.cost for tp in journey.touchpoints if tp.advertiser == 'Aura')
        
        reward = attributed_value - total_cost
        
        # Bonuses/Penalties
        if journey.days_to_conversion < 3:
            reward *= 1.2  # Quick conversion bonus
        
        if total_cost > 100:  # Over CAC target
            reward *= 0.8  # Penalty
        
        return reward
    
    def multi_touch_attribution(self, journey):
        """
        Data-driven attribution (DDA) style
        """
        credits = {}
        
        # Time decay
        for i, touchpoint in enumerate(journey.touchpoints):
            time_weight = 0.5 ** ((journey.conversion_time - touchpoint.time).days / 7)
            
            # Position weight (first and last touch get extra)
            if i == 0:
                position_weight = 1.3  # First touch bonus
            elif i == len(journey.touchpoints) - 1:
                position_weight = 1.5  # Last touch bonus
            else:
                position_weight = 1.0
            
            # Channel effectiveness
            channel_weight = {
                'search': 1.2,  # High intent
                'retargeting': 1.4,  # Very effective
                'social': 0.8,  # Lower intent
                'display': 0.6  # Awareness only
            }.get(touchpoint.channel, 1.0)
            
            credits[touchpoint] = time_weight * position_weight * channel_weight
        
        # Normalize
        total = sum(credits.values())
        return {tp: c/total for tp, c in credits.items()}
```

### 6. LEARNING SYSTEM INTEGRATION

```python
class GAELPLearningSystem:
    """
    Connects ALL components for learning
    """
    
    def __init__(self):
        # Data sources
        self.recsim = RecSimUserModel()
        self.auctiongym = AuctionGymWrapper()
        self.criteo = CriteoDataLoader()
        
        # Persistence
        self.user_db = UserJourneyDatabase()
        self.competitor_memory = CompetitorMemory()
        
        # Learning
        self.agent = JourneyAwarePPOAgent()
        self.replay_buffer = PrioritizedReplayBuffer()
        
    def training_loop(self):
        """
        Complete training process
        """
        for epoch in range(1000):
            # 1. Run Monte Carlo simulations
            experiences = self.monte_carlo.run_episode_batch(
                n_simulations=100,
                days_per_simulation=30
            )
            
            # 2. Store in replay buffer with importance weights
            for exp in experiences:
                priority = self.calculate_priority(exp)
                self.replay_buffer.add(exp, priority)
            
            # 3. Sample batch for training
            batch = self.replay_buffer.sample(
                batch_size=256,
                importance_sampling=True
            )
            
            # 4. Update agent
            self.agent.update(batch)
            
            # 5. Update competitor models (they learn too!)
            self.competitor_memory.update_strategies(experiences)
            
            # 6. Evaluate on hold-out scenarios
            if epoch % 10 == 0:
                metrics = self.evaluate()
                
                # 7. Adjust simulation difficulty
                if metrics['win_rate'] > 0.7:
                    self.increase_competition()
```

### 7. COMPETITIVE DYNAMICS

```python
class CompetitorAgents:
    """
    Competitors that actually learn and adapt
    """
    
    def __init__(self):
        self.competitors = {
            'Qustodio': QLearningAgent(aggressive=True),
            'Bark': PolicyGradientAgent(budget=5000),
            'Circle': RuleBased(strategy='defensive'),
            'Norton': RandomAgent(),  # Some don't adapt
        }
    
    def update_strategies(self, market_results):
        """
        Competitors learn from losses
        """
        for name, agent in self.competitors.items():
            # Did they lose high-value users?
            losses = market_results.get_losses(name)
            
            if losses.include_crisis_parents():
                # They'll bid more aggressively next time
                agent.increase_aggression(0.1)
            
            # Update their models
            agent.learn(losses)
```

## 📊 Complete Data Flow

```
INITIALIZATION:
RecSim UserModel → Generate 10K users with segments
    ↓
UserJourneyDatabase → Persist users across episodes
    ↓
CompetitorAgents → Initialize with strategies

EACH HOUR:
Get active users for this hour
    ↓
For each user:
    Check journey state (continuing or new)
    ↓
    Generate search query (based on state + segment)
    ↓
    Run AuctionGym (all competitors bid)
    ↓
    Winner shows ad (creative selected by state)
    ↓
    User response (click/ignore based on RecSim)
    ↓
    Update journey (add touchpoint)
    ↓
    Check conversion (probability based on journey)
    ↓
    Calculate reward (with attribution)
    ↓
Store experience in replay buffer

LEARNING:
Sample batch from replay buffer
    ↓
Calculate TD error or Monte Carlo returns
    ↓
Update agent policy (PPO)
    ↓
Update value function
    ↓
Update competitor models

EVALUATION:
Run on test user population
    ↓
Measure: CAC, ROAS, Win Rate, Journey Length
    ↓
Compare to baselines
```

## 🎮 The ACTUAL Simulation We Should Run

```python
# Day 1, Hour 20 (8 PM)
user_lisa = recsim.create_user(segment='concerned_parent')
lisa_journey = user_db.create_journey(user_lisa)

# Lisa searches after kids are in bed
query = "screen time limits app"
auction = auctiongym.run(
    query=query,
    bidders=[aura_agent, qustodio_agent, bark_agent],
    user_signals=lisa_journey.get_signals()
)

# Aura bids $2.50, Qustodio $3.20, Bark $2.90
# Qustodio wins, Lisa clicks, browses, leaves

# Day 2, Hour 13 (1 PM lunch break)
# Lisa continues research
query = "qustodio reviews"  # Branded search!
# Aura can bid on competitor terms
auction = auctiongym.run(...)
# Aura wins with $1.80 bid
# Shows comparison landing page
# Lisa signs up for email list

# Day 5, Hour 21
# Triggering event: Lisa finds inappropriate YouTube history
lisa_journey.trigger_crisis_event()
# Urgency spike, segment shifts to semi-crisis

query = "block youtube kids phone immediately"
# All competitors bid HIGH (they detect crisis)
# Aura uses emergency creative
# Lisa clicks multiple ads (comparison shopping under pressure)

# Day 6, Hour 9
# Retargeting opportunity
# Lisa gets email "Still unprotected?"
# Clicks through, converts
# Journey: 6 days, 4 searches, 7 touchpoints, $8.70 CAC

# ATTRIBUTION:
# 20% first touch (awareness)
# 30% email (trigger)
# 50% last click (conversion)
# Aura gets 60% credit = $108 attributed value
# Reward = $108 - $8.70 = $99.30
```

## ✅ Checklist: Is Our System Complete?

- [ ] Users persist across episodes? 
- [ ] Multi-day journeys tracked?
- [ ] Real auction dynamics?
- [ ] Competitors adapt?
- [ ] Attribution modeling?
- [ ] Delayed rewards?
- [ ] Monte Carlo parallelization?
- [ ] State includes history?
- [ ] Creative selection?
- [ ] Budget pacing?
- [ ] Seasonality?
- [ ] Device/channel coordination?

## 🚨 Critical Missing Pieces

1. **User Identity Resolution**
   - How do we know mobile user = desktop user?
   - Probabilistic matching based on patterns

2. **Conversion Lag**
   - Some users convert 30+ days later
   - Need to handle partial episodes

3. **Creative Fatigue**
   - Same ad shown 10x loses effectiveness
   - Need creative rotation strategy

4. **Competitive Intelligence**
   - How much do we know about competitor bids?
   - Should we model partial observability?

5. **Budget Pacing**
   - Can't spend all budget in first hour
   - Need intraday pacing strategy

## Next Steps

1. **Validate architecture** with small-scale test
2. **Implement user persistence layer**
3. **Connect RecSim → AuctionGym → Attribution**
4. **Add Monte Carlo parallelization**
5. **Train with increasing competition levels**