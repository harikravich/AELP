# GAELP RL Agent Training Architecture

```ascii
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    GAELP TRAINING SYSTEM                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── REAL DATA SOURCES ───────────────────────┐
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   GA4 MCP    │    │ Criteo CTR   │    │  Historical  │     │
│  │   Real Data  │    │    Model     │    │   Campaign   │     │
│  │              │    │              │    │     Data     │     │
│  │ 751K users   │    │ 0.5%-8% CTR  │    │              │     │
│  │ Dec 2024     │    │ predictions  │    │  Spend/ROI   │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                    │                    │              │
│         └────────────────────┼────────────────────┘              │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │ Pattern Discovery│                         │
│                    │    (KMeans)      │                         │
│                    └─────────┬────────┘                         │
│                              ▼                                   │
│                 ┌───────────────────────┐                       │
│                 │  discovered_patterns  │                       │
│                 │        .json          │                       │
│                 └───────────┬───────────┘                       │
└─────────────────────────────┼───────────────────────────────────┘
                              ▼
┌─────────────────────── TRAINING LOOP ───────────────────────────────────────────────────────┐
│                                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              EPISODE LOOP (1000 episodes/day)                         │  │
│  │                                                                                       │  │
│  │  ┌────────────┐         ┌──────────────────────────────────┐      ┌──────────────┐  │  │
│  │  │Environment │◄────────│                                  │      │              │  │  │
│  │  │   Reset    │         │         RL AGENT                 │      │  EXPERIENCE  │  │  │
│  │  └─────┬──────┘         │     (Double DQN)                 │      │    REPLAY    │  │  │
│  │        │                │                                  │      │    BUFFER    │  │  │
│  │        ▼                │  ┌──────────────────────────┐   │      │              │  │  │
│  │  ┌────────────┐         │  │      Q-Network           │   │      │  Size: 20K   │  │  │
│  │  │   STATE    │────────▶│  │   512→256→128 neurons    │   │◄─────│  Batch: 32   │  │  │
│  │  │            │         │  │   Dropout: 30%           │   │      │  Priority:   │  │  │
│  │  │ [10 dims]  │         │  │   Learning Rate: 0.0001  │   │      │  TD Error    │  │  │
│  │  └────────────┘         │  └──────────┬───────────────┘   │      └──────▲───────┘  │  │
│  │                         │              │                   │             │           │  │
│  │        ┌────────────────┼──────────────▼───────────────────┼─────────────┘           │  │
│  │        │                │         ACTION                   │                         │  │
│  │        │                │    [Bid, Creative, Target]      │                         │  │
│  │        │                └──────────────────────────────────┘                         │  │
│  │        ▼                                                                             │  │
│  │  ┌────────────┐         ┌────────────────┐        ┌──────────────┐                 │  │
│  │  │  AUCTION   │────────▶│    REWARD      │───────▶│    STORE     │                 │  │
│  │  │ SIMULATOR  │         │   CALCULATOR   │        │  EXPERIENCE  │                 │  │
│  │  │            │         │                │        │              │                 │  │
│  │  │ GSP Rules  │         │ Immediate: -$  │        │ (s,a,r,s',d) │                 │  │
│  │  │ Competitors│         │ Delayed: +CVR  │        └──────────────┘                 │  │
│  │  └────────────┘         └────────────────┘                                          │  │
│  │                                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                              │
│                              Every 32 Steps                Every 1000 Steps                 │
│                                    ▼                              ▼                         │
│                         ┌──────────────────┐          ┌──────────────────┐                 │
│                         │  TRAIN Q-NETWORK │          │ UPDATE TARGET NET│                 │
│                         │   (Batch=32)     │          │    (Soft Update) │                 │
│                         └──────────────────┘          └──────────────────┘                 │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── ENRICHMENT COMPONENTS ───────────────────────────────────────────────┐
│                                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   Attribution    │  │  Budget          │  │    Creative      │  │   Competitor     │   │
│  │     Engine       │  │  Optimizer       │  │    Analyzer      │  │  Intelligence    │   │
│  │                  │  │                  │  │                  │  │                  │   │
│  │ Multi-touch      │  │ Hourly pacing    │  │ 20 variants      │  │ Win rates        │   │
│  │ 14-day window    │  │ ROI optimization │  │ A/B testing      │  │ Bid patterns     │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│           │                      │                      │                      │             │
│           └──────────────────────┴──────────────────────┴──────────────────────┘             │
│                                              ▼                                               │
│                                    ┌──────────────────┐                                     │
│                                    │  STATE ENRICHER  │                                     │
│                                    │                  │                                     │
│                                    │ Adds context to  │                                     │
│                                    │  raw state vector│                                     │
│                                    └──────────────────┘                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── PARALLEL SYSTEMS ────────────────────────────────────────────────────┐
│                                                                                              │
│  ┌──────────────────────────────┐          ┌──────────────────────────────┐                │
│  │      SHADOW MODE              │          │    A/B TESTING FRAMEWORK     │                │
│  │                               │          │                              │                │
│  │  ┌──────────┐  ┌──────────┐  │          │  ┌─────────┐  ┌─────────┐   │                │
│  │  │ Policy A │  │ Policy B │  │          │  │Variant A│  │Variant B│   │                │
│  │  └────┬─────┘  └────┬─────┘  │          │  └────┬────┘  └────┬────┘   │                │
│  │       │              │        │          │       │             │        │                │
│  │       └──────┬───────┘        │          │       └──────┬──────┘        │                │
│  │              ▼                │          │              ▼               │                │
│  │     ┌──────────────┐          │          │    ┌──────────────┐         │                │
│  │     │  COMPARISON  │          │          │    │ SIGNIFICANCE │         │                │
│  │     │   No Risk    │          │          │    │   TESTING    │         │                │
│  │     └──────────────┘          │          │    └──────────────┘         │                │
│  └───────────────────────────────┘          └──────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── STATE VECTOR DETAILS ────────────────────────────────────────────────┐
│                                                                                              │
│  State[10] = [                                                                              │
│      0: Journey Stage      (0=awareness, 1=consideration, 2=decision, 3=purchase)           │
│      1: Touchpoints Seen   (0-20 ad interactions)                                           │
│      2: Days Since First   (0-30 days, time decay)                                          │
│      3: User Segment       (cluster_0, cluster_1, etc. from GA4)                            │
│      4: Device Type        (0=mobile, 1=desktop, 2=tablet)                                  │
│      5: Channel           (0=display, 1=search, 2=social, 3=organic)                        │
│      6: Creative ID       (0-19, which ad variant)                                          │
│      7: Competition       (0.0-1.0, auction pressure)                                       │
│      8: Budget Remaining  (0.0-1.0, percentage left)                                        │
│      9: Current Bid       (normalized bid amount)                                           │
│  ]                                                                                           │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── KEY METRICS & PARAMETERS ────────────────────────────────────────────┐
│                                                                                              │
│  REAL DATA (GA4 December 2024):           │  TRAINING PARAMETERS:                           │
│  ├─ Total Users: 751,234                  │  ├─ Episodes/Day: ~1000                        │
│  ├─ Total Sessions: 960,419               │  ├─ Steps/Episode: Max 1000                    │
│  ├─ Total Conversions: 12,252             │  ├─ Learning Rate: 0.0001                      │
│  ├─ Overall CVR: 1.28%                    │  ├─ Epsilon: 0.3 → 0.1 (decay: 0.99995)        │
│  │                                        │  ├─ Discount (γ): 0.99                         │
│  ├─ CHANNEL PERFORMANCE:                  │  ├─ Buffer Size: 20,000                        │
│  │  ├─ Paid Search: 2.65% CVR ✓           │  ├─ Batch Size: 32                             │
│  │  ├─ Paid Social: 1.76% CVR             │  ├─ Target Update: Every 1000 steps            │
│  │  ├─ Display: 0.047% CVR ✗ (BROKEN!)    │  └─ Soft Update (τ): 0.001                     │
│  │  └─ Branded: 3.56% CVR ✓✓              │                                                 │
│  │                                        │  DISCOVERED SEGMENTS:                          │
│  └─ DEVICE PERFORMANCE:                   │  ├─ High Intent: 3.56% CVR                     │
│     ├─ Desktop: 1.89% CVR                 │  ├─ Paid Searchers: 2.65% CVR                  │
│     ├─ Mobile: 0.87% CVR                  │  ├─ Social Browsers: 1.76% CVR                 │
│     └─ Tablet: 1.16% CVR                  │  └─ Organic Researchers: 0.30% CVR             │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── CRITEO CTR MODEL ────────────────────────────────────────────────────┐
│                                                                                              │
│                     Input Features (39)                                                     │
│                            │                                                                │
│        ┌───────────────────┼───────────────────┐                                           │
│        ▼                   ▼                   ▼                                           │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐                                       │
│  │Numerical │       │Categorical│       │ Context  │                                       │
│  │    13    │       │    26     │       │ Features │                                       │
│  └─────┬────┘       └─────┬─────┘       └─────┬────┘                                       │
│        └───────────────────┼───────────────────┘                                           │
│                            ▼                                                                │
│                 ┌──────────────────┐                                                       │
│                 │ Gradient Boosting│                                                       │
│                 │   Classifier     │                                                       │
│                 └─────────┬────────┘                                                       │
│                           ▼                                                                │
│                    CTR: 0.5% - 8%                                                          │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## System Flow Summary

1. **Real GA4 data** provides user patterns and channel performance
2. **Pattern discovery** clusters users into behavioral segments
3. **Environment** simulates auctions using discovered patterns
4. **RL Agent** learns optimal bidding through Double DQN
5. **Experience replay** enables learning from past episodes
6. **Attribution** handles delayed conversion rewards
7. **Components** enrich state with context and constraints
8. **Parallel systems** test policies safely without real money

The agent trains on **simulated auctions** informed by **real GA4 patterns**, not live ads.
Display channel needs urgent fixing (0.047% CVR vs 2.65% for paid search).