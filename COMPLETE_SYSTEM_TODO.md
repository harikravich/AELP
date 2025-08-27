# Complete To-Do List: Behavioral Health Marketing System

## PHASE 1: Fix the Persona System (Week 1)

### 1.1 PersonaFactory Overhaul
- [ ] Remove ALL hardcoded segments (crisis_parent, researcher, etc.)
- [ ] Build concern level spectrum (0-10 scale, continuous not discrete)
- [ ] Create trigger event generator:
  - Found concerning search history
  - Report card came home
  - Friend's kid had incident
  - Saw news article
  - Therapist recommendation
  - School counselor suggestion
- [ ] Implement concern progression model:
  - Curious (0-2) → Interested (3-4) → Worried (5-6) → Anxious (7-8) → Crisis (9-10)
  - Time-based progression (concern can increase/decrease)
- [ ] Add realistic parent attributes:
  - Income level (affects price sensitivity)
  - Tech savviness (affects research depth)
  - Number of kids
  - Previous mental health exposure
  - Geographic location (CA vs TX different attitudes)

### 1.2 Search Query Generation
- [ ] Map concern levels to actual search queries:
  - Level 1-2: "screen time recommendations teenagers"
  - Level 3-4: "too much screen time effects"
  - Level 5-6: "teen won't get off phone help"
  - Level 7-8: "signs of depression in teenagers"
  - Level 9-10: "found self harm content child phone"
- [ ] Add query refinement patterns (how people search multiple times)
- [ ] Include voice search patterns (different from typed)

### 1.3 Time-of-Day Patterns
- [ ] 2-4am: Crisis searches (found something)
- [ ] 6-8am: Morning worry searches (before school)
- [ ] 10-11am: Work break research
- [ ] 3-5pm: After school concerns
- [ ] 8-10pm: Bedtime battles
- [ ] 11pm-1am: Can't sleep worry searches

## PHASE 2: Connect Real Data (Week 2)

### 2.1 Criteo Integration
- [ ] Load the 45M impressions dataset properly
- [ ] Extract CTR patterns by:
  - Ad position
  - Time of day
  - Device type
  - User history
- [ ] Build CTR prediction model
- [ ] Calibrate with parental control specific data

### 2.2 Google Analytics Connection
- [ ] Set up GA4 API connection
- [ ] Extract historical Aura data:
  - Conversion paths
  - Time lag distributions
  - Drop-off points
  - Feature page visits → conversion correlation
- [ ] Identify high-value search terms
- [ ] Map content consumption to conversion

### 2.3 Competitor Intelligence
- [ ] Scrape competitor pricing (Bark: $14/mo, Qustodio: $55/yr)
- [ ] Document feature comparisons
- [ ] Analyze competitor ad copy (SpyFu/SEMrush)
- [ ] Model switching costs and barriers

## PHASE 3: Multi-Week Journey Engine (Week 3)

### 3.1 Journey State Machine
- [ ] States: Unaware → Triggered → Researching → Comparing → Deciding → Purchased → Retained
- [ ] Transition probabilities based on:
  - Concern level
  - Number of touchpoints
  - Time since trigger
  - Competitor exposure
- [ ] Journey abandonment modeling

### 3.2 Memory System
- [ ] Track every ad seen (creative, message, channel)
- [ ] Remember search queries
- [ ] Store competitor visits
- [ ] Model ad fatigue (same ad loses effectiveness)

### 3.3 Decision Dynamics
- [ ] Spouse approval requirement (for >$10/mo)
- [ ] Budget constraints (wait for paycheck)
- [ ] Free trial behavior (who converts after trial)
- [ ] Urgency decay (crisis feeling fades after 72 hours)

## PHASE 4: Ad Creative Engine (Week 4)

### 4.1 Compliant Copy Generation
- [ ] Build template library:
  - Crisis: "Get immediate insights into your teen's digital wellbeing"
  - Worry: "73% of parents don't know these warning signs"
  - Prevention: "Build healthy habits before problems start"
- [ ] A/B test variants for each concern level
- [ ] Channel-specific formatting

### 4.2 Visual Assets
- [ ] Dashboard screenshots (showing insights, not teen data)
- [ ] Clinician testimonials
- [ ] Statistical graphics
- [ ] Family happiness imagery (outcome focused)

### 4.3 Landing Page Variants
- [ ] Crisis: Immediate trial, speak to specialist
- [ ] Research: Full feature comparison
- [ ] Price-sensitive: Discount emphasis
- [ ] Clinical: Expert endorsements

## PHASE 5: Channel Simulators (Week 5)

### 5.1 Google Ads Simulator
- [ ] Keyword quality scores
- [ ] Bid auction mechanics
- [ ] Ad rank calculations
- [ ] Budget pacing throughout day
- [ ] Competitor bid responses

### 5.2 Meta Ads Simulator
- [ ] Audience overlap penalties
- [ ] Frequency capping
- [ ] Lookalike audience quality
- [ ] Creative fatigue curves
- [ ] iOS 14.5+ signal loss

### 5.3 TikTok Simulator
- [ ] Viral coefficient modeling
- [ ] Creator partnership effects
- [ ] Trend participation boost
- [ ] Young parent audience concentration

### 5.4 Emerging Channels
- [ ] Reddit: Community sentiment effects
- [ ] Spotify: Podcast host reads
- [ ] School newsletters: Trust multiplier
- [ ] NextDoor: Local parent network effects

## PHASE 6: Competition Dynamics (Week 6)

### 6.1 Competitor Behaviors
- [ ] Bark: Aggressive SEM, owns "bark parental controls"
- [ ] Qustodio: Content marketing focus
- [ ] Life360: Location selling point
- [ ] Circle: Router-based advantage

### 6.2 Bidding Wars
- [ ] Branded term defense
- [ ] Competitor conquesting
- [ ] Dayparting battles
- [ ] Retargeting pool fights

## PHASE 7: RL Agent Upgrades (Week 7)

### 7.1 State Space Expansion
- [ ] Journey stage (trigger → research → compare → decide)
- [ ] Concern level (0-10)
- [ ] Previous touchpoints (list of ads seen)
- [ ] Time since trigger
- [ ] Competitor exposure count
- [ ] Channel performance history

### 7.2 Action Space
- [ ] Bid amount (continuous)
- [ ] Creative selection (50+ variants)
- [ ] Channel allocation
- [ ] Audience targeting parameters
- [ ] Frequency capping decisions

### 7.3 Reward Engineering
- [ ] Immediate: -CPC for clicks
- [ ] Delayed: -CAC for conversions
- [ ] LTV bonus for high-value segments
- [ ] Penalty for oversaturation

## PHASE 8: Scale Infrastructure (Week 8)

### 8.1 Performance Requirements
- [ ] Handle 100K concurrent visitors
- [ ] Track 1M daily events
- [ ] Store 90-day attribution window
- [ ] Real-time bid decisions (<100ms)

### 8.2 Data Pipeline
- [ ] Event streaming (Kafka/Pub/Sub)
- [ ] Real-time aggregation
- [ ] Attribution calculation
- [ ] Model retraining pipeline

## PHASE 9: MCP Integration (Week 9)

### 9.1 API Connections
- [ ] Google Ads API via MCP
- [ ] Facebook Marketing API
- [ ] TikTok Ads API
- [ ] Google Analytics 4

### 9.2 Bid Management
- [ ] Real-time bid adjustments
- [ ] Budget pacing
- [ ] Creative rotation
- [ ] Audience updates

### 9.3 Feedback Loop
- [ ] Import actual performance data
- [ ] Compare to simulation
- [ ] Retrain models
- [ ] Adjust strategies

## PHASE 10: Production Launch (Week 10)

### 10.1 Testing Protocol
- [ ] Shadow mode (run parallel, don't bid)
- [ ] 1% traffic test
- [ ] 10% rollout
- [ ] 50% A/B test
- [ ] Full deployment

### 10.2 Monitoring
- [ ] Real-time CAC tracking
- [ ] Channel performance dashboards
- [ ] Anomaly detection
- [ ] Automated alerts

### 10.3 Safety Mechanisms
- [ ] Max bid caps
- [ ] Daily spend limits
- [ ] CAC circuit breakers
- [ ] Manual override controls

## Success Metrics

### Month 1 Targets:
- CAC: <$120
- Volume: 7,500 conversions
- Spend: $900K

### Month 6 Targets:
- CAC: <$105
- Volume: 24,000 conversions
- Spend: $2.5M

### Month 12 Targets:
- CAC: <$100
- Volume: 42,000 conversions
- Spend: $4.2M

## NO SHORTCUTS ALLOWED
- ✗ No hardcoded segments
- ✗ No fake probability distributions
- ✗ No simplified user behavior
- ✗ No single-touch attribution
- ✗ No instant conversions
- ✗ No ignoring compliance
- ✗ No skipping competition
- ✗ No assuming perfect data

## This is a $50M/year system. Build it right.