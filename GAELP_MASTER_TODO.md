# GAELP Master TODO - Building a Self-Learning Performance Marketing Agent
Last Updated: January 22, 2025

## Mission Statement
Build an AI agent that learns to become the world's best direct-to-consumer performance marketer, starting with Aura Balance (behavioral health monitoring for parents).

## Key Product Insights (from aura.com/parental-controls analysis)
- **Current Page Gaps**: No crisis messaging, weak clinical authority, Balance feature buried
- **Price Points**: $10/month (Kids), $32/month (Family) with 14-day trial
- **Trust Signals**: Mom's Choice Award, 4.7 App Store rating (but no clinical endorsements)
- **Critical Limitation**: Balance only works on iOS (must be prominent in targeting)
- **Opportunity**: Position as behavioral health solution, not just parental controls

## Current Status
- ✅ 20 components integrated and working
- ✅ RL foundation (DQN/PPO) implemented
- ✅ User journey simulation with realistic triggers
- ✅ Multi-touch attribution working
- ❌ No creative optimization/generation
- ❌ Limited action space (only bidding)
- ❌ No real data integration
- ❌ No production deployment path

## Deployment Strategy
- **Phase 1**: Personal account testing with personal funds
- **Phase 2**: Cross-account tracking (personal ads → Aura GA4)
- **Phase 3**: Scale to company accounts after proven success
- **Target**: aura.com/parental-controls (point of sale)
- **Key**: Test intermediate landing pages for conversion optimization

---

## PHASE 0: Fix Current System Issues [IMMEDIATE]
**Goal: Get existing system stable and learning properly**

### 0.1 Dashboard & Monitoring Fixes
- [ ] Fix "winning 100% of bids" - auction mechanics broken
- [ ] Fix conversion tracking - no conversions happening
- [ ] Fix spend not increasing - budget pacing broken
- [ ] Fix competitor analysis not updating
- [ ] Fix AI insights chart - dead
- [ ] Fix attribution visualization

### 0.2 Learning Verification
- [ ] Verify RL agent is actually learning (not random)
- [ ] Add learning curve visualization
- [ ] Implement proper exploration/exploitation balance
- [ ] Add checkpointing for trained models
- [ ] Create performance benchmarks

---

## PHASE 0.5: Personal Account Setup & Landing Page System [IMMEDIATE - DAY 1-2]
**Goal: Set up personal testing infrastructure with conversion tracking**

### 0.5.1 Personal Ad Account Setup
- [ ] Create/configure personal Google Ads account
- [ ] Create/configure personal Facebook Ads account
- [ ] Set up billing with personal card ($1000 initial limit)
- [ ] Configure conversion tracking pixels
- [ ] Set up UTM parameter system for cross-account tracking

### 0.5.2 Landing Page Testing Infrastructure with Social Scanner Tool
- [ ] Set up landing page hosting (Vercel/Netlify)
- [ ] Create domain for testing (e.g., teen-social-scanner.com, find-secret-accounts.com)
- [ ] **BUILD FREE SOCIAL SCANNER TOOL** (Primary Conversion Driver):
  ```python
  scanner_features = {
      'account_discovery': {
          'cross_platform_search': True,  # Find linked accounts
          'username_variations': True,     # Check sarah_123, sarah.123
          'reverse_image_search': True,    # Find profile pic reuse
          'finsta_detection': True,        # Identify hidden accounts
      },
      'risk_assessment': {
          'public_exposure_score': True,   # How findable by strangers
          'adult_follower_check': True,    # Inappropriate followers
          'location_disclosure': True,     # Sharing location publicly
          'personal_info_audit': True,     # Info in bios/posts
      },
      'ml_demonstrations': {
          'behavioral_patterns': True,     # "Posts 73% more at 2am"
          'mood_indicators': True,         # "Language suggests anxiety"
          'network_risks': True,          # "3 followers likely fake"
          'content_risks': True,          # "12 posts show location"
      }
  }
  ```
- [ ] Implement scanner backend:
  - Instagram public API integration
  - TikTok public profile scraping
  - Google Vision API for reverse image search
  - OSINT tools for account discovery
  - ML models for risk scoring
- [ ] Build landing page generator system based on Aura gaps:
  ```python
  landing_pages = {
      # BEHAVIORAL HEALTH FOCUSED (main opportunity)
      'behavioral_crisis': '/teen-behavioral-crisis-help',  # Detect mood changes, self-harm risks
      'mental_health_monitor': '/teen-mental-health-ai',  # AI detecting depression/anxiety
      'cdc_guidelines': '/cdc-screen-time-guidelines',  # Authority positioning
      'therapist_recommended': '/therapist-recommended-monitoring',  # Clinical backing
      
      # BALANCE FEATURE HERO (currently buried on main site)
      'balance_first': '/ai-wellness-insights',  # Lead with Balance, not controls
      'social_persona': '/understand-teen-social-life',  # Social monitoring angle
      'sleep_patterns': '/teen-sleep-disruption-detector',  # Specific concern
      
      # COMPARISON/CONQUEST (exploit competitor weaknesses)
      'vs_bark': '/aura-vs-bark-behavioral-health',  # Position on behavioral, not just safety
      'vs_therapy': '/cheaper-than-therapy',  # Cost comparison to counseling
      
      # EDUCATION BRIDGES (missing on current site)
      'warning_signs': '/teen-depression-warning-signs',  # Education → Product
      'quiz_funnel': '/is-my-teen-at-risk-quiz',  # Interactive assessment
      'parent_guide': '/digital-parenting-mental-health-guide',  # Resource → Trial
      
      # iOS SPECIFIC (since Balance only works on iOS)
      'iphone_parents': '/iphone-family-wellness',  # Target iOS families explicitly
  }
  ```
- [ ] Implement A/B testing framework on landing pages
- [ ] Add heatmap tracking (Hotjar/FullStory)
- [ ] Create conversion tracking:
  - Page views
  - Time on page
  - Scroll depth
  - Button clicks
  - Form starts
  - Trial signups
  - Purchases

### 0.5.3 Cross-Account Attribution Pipeline
- [ ] Build attribution bridge:
  ```
  Personal Ad → Landing Page → Aura.com → GA4
  ```
- [ ] Implement server-side tracking:
  - Capture click ID from ad platforms
  - Pass through to Aura via URL parameters
  - Store in cookie/localStorage
  - Send to GA4 via Measurement Protocol
- [ ] Create unified reporting dashboard:
  - Personal ad spend
  - Landing page metrics
  - Aura conversions
  - True ROAS calculation
- [ ] Handle iOS 14.5+ attribution loss:
  - Implement CAPI (Conversions API)
  - Use enhanced conversions
  - Model missing data

### 0.5.4 Reward Signal Configuration
- [ ] Define reward hierarchy:
  ```python
  rewards = {
      'impression': 0.0,  # No reward
      'click': 0.1,  # Small positive
      'landing_page_view': 0.2,
      'engagement_30s': 0.3,
      'quiz_complete': 0.5,
      'email_capture': 1.0,
      'trial_start': 5.0,
      'paid_conversion': 10.0,  # PRIMARY REWARD
      'month_2_retention': 20.0  # ULTIMATE REWARD
  }
  ```
- [ ] Implement delayed reward attribution
- [ ] Create value modeling for partial conversions

---

## PHASE 1: GA4 Data Integration [CRITICAL PATH - DAY 3-4]
**Goal: Use real Aura data to ground truth the simulation**

### 1.1 MCP GA4 Connector
- [ ] Build MCP connector for GA4 API
- [ ] Pull historical conversion paths (last 90 days)
- [ ] Extract real attribution sequences
- [ ] Get actual CAC by channel/campaign
- [ ] Identify real user segments from behavior

### 1.2 Data Pipeline
- [ ] Create data ingestion pipeline (hourly pulls)
- [ ] Build data warehouse (BigQuery tables)
- [ ] Implement data validation/cleaning
- [ ] Create feedback loop from production → simulation

### 1.3 Simulation Calibration
- [ ] Replace synthetic data with GA4 patterns
- [ ] Calibrate conversion rates to match reality
- [ ] Adjust attribution windows based on real data
- [ ] Tune competitor behavior to match market share

---

## PHASE 2: Creative Intelligence System [GAME CHANGER - DAY 5-7]
**Goal: Build the brain that optimizes creative performance**

### 2.1 Creative Generation Engine
- [ ] Integrate LLM for headline generation (Claude/GPT-4)
- [ ] Build template system for ad variations
- [ ] Create message testing framework based on Aura gaps:
  - **Behavioral Health Angles** (main focus):
    - "AI detects mood changes before you do"
    - "Know if your teen is really okay"
    - "Catch warning signs early"
    - "Your teen's digital therapist"
  - **Clinical Authority** (missing on current site):
    - "CDC-recommended monitoring"
    - "Designed with child psychologists"
    - "AAP screen time guidelines built-in"
    - "Therapist-recommended solution"
  - **Crisis vs Prevention messaging**:
    - Crisis: "Is your teen in crisis? Know now"
    - Prevention: "Prevent problems before they start"
    - Concern: "Something feels off with your teen?"
  - **Balance Feature Hero** (undersold currently):
    - "AI that understands your teen's emotions"
    - "See your teen's wellness score"
    - "Track mood patterns invisibly"
  - **iOS Targeting** (critical limitation):
    - "iPhone families only" (be upfront)
    - "Works with your teen's iPhone"
- [ ] Implement creative DNA tracking (what elements work)

### 2.2 Creative Performance Learning
- [ ] Build creative fatigue model
- [ ] Track performance decay curves
- [ ] Implement multi-armed bandit for creative selection
- [ ] Create segment-specific creative mapping
- [ ] Build creative interaction effects model

### 2.3 Visual Asset Management
- [ ] Integrate with DALL-E/Midjourney for image generation
- [ ] Build image performance tracking
- [ ] Create visual A/B testing framework
- [ ] Implement color/emotion analysis

### 2.4 Landing Page Optimization Learning
- [ ] Build landing page variant generator:
  ```python
  page_elements = {
      'hero': ['crisis_help', 'peace_of_mind', 'clinical_backing'],
      'social_proof': ['testimonials', 'stats', 'logos', 'reviews'],
      'cta_style': ['urgent', 'gentle', 'professional', 'emotional'],
      'form_fields': ['minimal', 'detailed', 'progressive'],
      'content_depth': ['tldr', 'educational', 'comprehensive'],
      'pricing': ['hidden', 'prominent', 'anchored', 'compared']
  }
  ```
- [ ] Learn conversion patterns:
  - Crisis parents → Direct CTA works
  - Researchers → Need education first
  - Price-sensitive → Comparison tables help
- [ ] Multi-variate testing framework:
  - Test element combinations
  - Learn interaction effects
  - Discover segment-specific preferences
- [ ] Progressive disclosure testing:
  - When to show pricing
  - When to ask for email
  - When to mention competitors

---

## PHASE 3: Expanded Action & State Space [CRITICAL - DAY 8-10]
**Goal: Give the agent full control over marketing levers**

### 3.1 Richer Action Space
```python
actions = {
    'bid_adjustment': [-50%, ..., +200%],  # More granular
    'audience_targeting': {
        'expand': bool,
        'narrow': bool,
        'lookalike_%': [1, 2, 5, 10]
    },
    'creative_selection': {
        'variant': [0...100],  # Dynamic pool
        'rotation': ['even', 'optimized', 'sequential']
    },
    'budget_allocation': {
        'channel_shift': [-20%, ..., +20%],
        'campaign_shift': [-30%, ..., +30%],
        'dayparting': [hourly_multipliers]
    },
    'campaign_structure': {
        'split_test': bool,
        'consolidate': bool
    }
}
```

### 3.2 Enhanced State Representation
```python
state = {
    'user_context': {
        'journey_stage': [...],
        'concern_level': [...],
        'trigger_recency': [...]
    },
    'creative_performance': {
        'ctr_by_creative': [...],
        'fatigue_scores': [...],
        'interaction_history': [...]
    },
    'competitive_landscape': {
        'competitor_bids': [...],
        'market_saturation': [...],
        'share_of_voice': [...]
    },
    'platform_signals': {
        'facebook_learning_phase': bool,
        'google_quality_score': float,
        'tiktok_engagement_rate': float
    },
    'temporal_context': {
        'hour_of_day': [...],
        'day_of_week': [...],
        'seasonality': [...]
    }
}
```

---

## PHASE 4: Channel-Specific Intelligence [REQUIRED - DAY 11-14]
**Goal: Master each platform's unique dynamics**

### 4.1 Facebook/Meta Optimization
- [ ] Model learning phase (50 conversions/week)
- [ ] Implement audience overlap penalties
- [ ] Build frequency cap optimization
- [ ] Handle iOS 14.5+ signal loss (critical for Balance/iOS only)
- [ ] Advantage+ campaign automation
- [ ] Target iOS users specifically (Balance limitation)
- [ ] Parent interest targeting:
  - Behavioral: Parents of teens 13-17
  - Interests: Mental health awareness, therapy, counseling
  - Lookalikes: From iOS app installers

### 4.2 Google Ads Mastery
- [ ] Quality Score optimization model
- [ ] Keyword expansion/pruning logic
- [ ] Search term mining from queries:
  - Behavioral health keywords (high intent)
  - "Teen depression signs"
  - "Is my teen okay"
  - "Teen mental health monitoring"
  - "Digital wellness for teens"
- [ ] Performance Max integration
- [ ] YouTube creative requirements
- [ ] Target iPhone/iOS searches specifically

### 4.3 TikTok Growth Hacking
- [ ] Viral coefficient modeling
- [ ] Creator partnership simulation
- [ ] Spark Ads optimization
- [ ] Trend-jacking timing model

### 4.4 Emerging Channels
- [ ] Reddit community sentiment modeling
- [ ] School newsletter partnerships
- [ ] Therapist referral networks
- [ ] Parent Facebook groups

---

## PHASE 5: Production Deployment System [END GOAL - DAY 15-21]
**Goal: Deploy the trained agent with real money on personal accounts**

### 5.1 Workflow Orchestration System
- [ ] Build recommendation engine UI (Streamlit dashboard)
- [ ] Create Slack bot for approvals:
  ```python
  @bot.command
  async def agent_recommendation(ctx):
      return {
          'action': 'increase_bid',
          'params': {'keyword': 'teen crisis help', 'adjustment': +40},
          'reasoning': 'High intent searches up 30%, competitor dropped out',
          'expected_impact': '+15 conversions/day',
          'risk_level': 'low',
          'buttons': ['APPROVE', 'MODIFY', 'REJECT', 'MORE_INFO']
      }
  ```
- [ ] Create approval workflow for:
  - Creative changes (auto-approve if CTR > baseline)
  - Budget increases > $100/day (require approval)
  - New campaign launches (always require approval)
  - Audience expansions > 20% (require approval)
- [ ] Implement safety checks:
  - Max spend limits ($1000/day personal account)
  - Performance guardrails (pause if CPA > $150)
  - Anomaly detection (alert if CTR drops 50%)
  - Brand safety (keyword blacklists)
- [ ] Build rollback mechanisms:
  - One-click pause all campaigns
  - Restore previous settings
  - Emergency budget caps

### 5.2 API Integrations
- [ ] Facebook Marketing API
  - Campaign creation
  - Bid adjustments
  - Creative uploads
  - Performance data pull
- [ ] Google Ads API
  - Similar capabilities
- [ ] TikTok Business API
- [ ] Analytics webhooks

### 5.3 Production Learning Loop
```
1. Agent recommends action
2. Human approves/modifies
3. Action deployed to platform
4. Wait for performance data (1-7 days)
5. Agent observes results
6. Updates model weights
7. Repeat
```

### 5.4 A/B Testing Framework
- [ ] Holdout group management
- [ ] Statistical significance testing
- [ ] Incrementality measurement
- [ ] Causal inference for attribution

---

## PHASE 6: Aura Balance Specific Optimizations
**Goal: Dominate the behavioral health monitoring category**

### 6.1 Message Testing Matrix
- [ ] Behavioral Health Positioning (PRIMARY):
  - "Detect depression and anxiety early"
  - "AI-powered mood monitoring"
  - "Your teen's digital wellness score"
  - "Behavioral pattern recognition"
  - "Mental health early warning system"
- [ ] Authority signals (MISSING on current site):
  - "CDC screen time guidelines integrated"
  - "Designed with child psychologists"
  - "AAP-aligned recommendations"
  - "Used by 10,000 therapists" (aspirational)
- [ ] Balance Feature Emphasis:
  - "Understand your teen's social persona"
  - "Track sleep disruption patterns"
  - "See behavior changes in real-time"
  - "AI insights into digital wellbeing"
- [ ] Privacy messaging:
  - "Supportive, not surveillance"
  - "Guide, don't spy"
  - "Wellness monitoring, not invasion"
- [ ] Value Comparisons:
  - "Costs less than one therapy session"
  - "Cheaper than missing the warning signs"
  - "1/10th the cost of counseling"
- [ ] iOS Exclusive Angle:
  - "Premium iPhone family solution"
  - "Designed for Apple families"
  - "Works seamlessly with Screen Time"

### 6.2 Trigger-Based Campaigns
- [ ] School incident response ads
- [ ] News event reactive campaigns
- [ ] Seasonal (back-to-school, summer break)
- [ ] Competitive conquesting (when Bark has issues)

### 6.3 Trust Building Content
- [ ] Parent testimonial campaigns
- [ ] Clinical expert endorsements
- [ ] Case study development
- [ ] Educational content series

---

## PHASE 7: Scale Testing & Optimization
**Goal: Prove the system at scale**

### 7.1 Load Testing
- [ ] Simulate 100K daily visitors
- [ ] Handle $150K daily budget
- [ ] Process 40K monthly conversions
- [ ] 20+ touchpoint journeys

### 7.2 Performance Optimization
- [ ] Optimize simulation speed (GPU acceleration)
- [ ] Implement distributed training
- [ ] Build caching layers
- [ ] Stream processing for real-time data

### 7.3 Cost Optimization
- [ ] Minimize API calls
- [ ] Optimize compute usage
- [ ] Implement smart sampling
- [ ] Build cost prediction models

---

## PHASE 8: Advanced Features [FUTURE]
**Goal: Push beyond human performance**

### 8.1 Cross-Channel Journey Orchestration
- [ ] Unified user identity across channels
- [ ] Sequential messaging strategies
- [ ] Channel handoff optimization
- [ ] Omnichannel budget allocation

### 8.2 Predictive Analytics
- [ ] LTV prediction by segment
- [ ] Churn prediction and prevention
- [ ] Demand forecasting
- [ ] Competitive response prediction

### 8.3 Advanced RL Techniques
- [ ] Hierarchical RL for strategy/tactics
- [ ] Multi-agent RL for competitive dynamics
- [ ] Inverse RL from expert marketers
- [ ] Meta-learning for quick adaptation

---

## Success Metrics

### Simulation Success
- [ ] Agent consistently beats baseline by 50%+ ROAS
- [ ] Discovers non-obvious strategies
- [ ] Adapts to market changes within 24 hours
- [ ] Generates novel creative combinations

### Production Success
- [ ] Reduce CAC by 30% in 90 days
- [ ] Increase conversion rate by 25%
- [ ] Achieve 4:1 ROAS consistently
- [ ] Scale to $5M monthly spend profitably

---

## Risk Mitigation

### Technical Risks
- **Risk**: Creative generation produces inappropriate content
- **Mitigation**: Human review queue, content filters, brand guidelines

- **Risk**: Agent makes expensive mistakes
- **Mitigation**: Gradual rollout, spend limits, performance guardrails

- **Risk**: Platform APIs change/break
- **Mitigation**: Abstraction layers, fallback strategies, monitoring

### Business Risks
- **Risk**: Competitors copy strategies
- **Mitigation**: Continuous learning, proprietary data advantage

- **Risk**: Platform policy violations
- **Mitigation**: Compliance checks, conservative initial approach

---

## Critical Tracking Challenges & Solutions

### Cross-Account Attribution Issues
**Challenge**: Personal ad account → Company GA4 tracking
**Solutions**:
1. **Server-side GTM**: 
   - Deploy GTM server container on personal domain
   - Forward events to Aura GA4 with proper client_id
2. **URL Parameter Passthrough**:
   - `?gaelp_uid={unique_id}&source={personal}&campaign={test_1}`
   - Store in localStorage, survive navigation
3. **Measurement Protocol**:
   - Direct server-to-server event sending
   - Bypass client-side tracking restrictions
4. **Conversion Import**:
   - Upload offline conversions back to ad platforms
   - Match via click_id or user identifiers

### Landing Page → Aura.com Handoff
**Challenge**: Losing tracking when redirecting to main site
**Solutions**:
1. **Iframe Embedding**: Embed Aura checkout in landing page
2. **Subdomain Strategy**: host.aura.com for testing
3. **PostMessage API**: Cross-domain communication
4. **Webhook System**: Real-time conversion notifications

---

## Implementation Order (AGGRESSIVE TIMELINE)

### Sprint 1 (Days 1-7): Foundation & Testing Infrastructure
- **Day 1-2**: Personal account setup, landing page system
- **Day 3-4**: GA4 integration, cross-account tracking
- **Day 5-7**: Creative intelligence system, LLM integration

### Sprint 2 (Days 8-14): Intelligence & Optimization  
- **Day 8-10**: Expanded action/state space
- **Day 11-14**: Channel-specific intelligence

### Sprint 3 (Days 15-21): Production Deployment
- **Day 15-17**: Workflow system, API integrations
- **Day 18-19**: First live test with $100/day
- **Day 20-21**: Analyze results, iterate

### Sprint 4 (Days 22-30): Scale & Learn
- **Day 22-25**: Scale to $500/day
- **Day 26-28**: Multi-channel testing
- **Day 29-30**: Performance analysis, optimization

---

## Required Resources

### Technical
- GA4 API access ✅
- Facebook Marketing API access (needed)
- Google Ads API access (needed)
- GPT-4/Claude API for creative generation
- GPU compute for training (current: GCP instance)

### Human
- 1 engineer (you) for core development
- 1 performance marketer for validation (optional but helpful)
- Legal review for compliance (before production)

### Financial
- ~$10K/month for API costs during development
- $50K initial test budget for production
- $5K/month for compute/infrastructure

---

## Key Learning Objectives for the Agent

### Must Discover Through Testing
1. **Creative-Segment Fit**:
   - Behavioral health messaging vs. safety messaging effectiveness
   - Clinical authority impact (CDC/AAP) on conversion
   - Balance feature as hero vs. supporting role
   - Crisis messaging threshold (when does it scare vs. motivate?)
   - iOS-exclusive positioning (premium vs. limitation)

2. **Landing Page Flow**:
   - Behavioral health quiz vs. general safety quiz
   - Balance demo vs. full feature overview
   - Clinical backing placement (hero vs. trust section)
   - Price anchoring against therapy costs
   - iOS requirement disclosure timing

3. **Timing Patterns**:
   - 2am crisis searches → immediate conversion
   - Weekend research sessions → delayed conversion
   - School hours → parent browsing patterns

4. **Competitive Dynamics**:
   - Bark's weakness on behavioral health (safety only)
   - Qustodio's lack of AI insights
   - Life360's location-only focus
   - Market gap: Clinical backing + AI behavioral analysis

5. **Channel Synergies**:
   - Google Search → Facebook retargeting paths
   - Email capture → nurture sequence timing
   - Trial → paid conversion optimization

---

## Next Immediate Actions (DO TODAY)

### Hour 1-2: Fix Critical Issues
1. Fix the "winning 100% of bids" issue in auction mechanics
2. Fix conversion tracking (verify data flow)

### Hour 3-4: Personal Account Setup  
3. Create personal Google Ads account
4. Set up conversion tracking pixels
5. Configure UTM parameters

### Hour 5-6: GA4 Integration
6. Get GA4 MCP connector working
7. Pull first batch of real Aura data
8. Verify data quality

### Hour 7-8: Creative System
9. Build creative generation prototype with Claude API
10. Generate first 10 headline variants

### Tomorrow: Landing Pages & Testing
11. Set up Vercel/Netlify hosting
12. Build first landing page variant
13. Deploy first $100 test campaign
14. Expand RL agent action space
15. Run overnight training with real data

---

## Notes

- **Behavioral Health Focus**: Position as mental health solution, not parental controls
- **Balance Feature**: Currently undersold - this is our wedge into the market
- **iOS Limitation**: Must be transparent about iOS-only for Balance features
- **Clinical Authority Gap**: Current site lacks this - huge opportunity
- **Price Anchoring**: $32/month seems cheap vs. $150/session therapy
- The creative intelligence system is the KEY differentiator
- Without real data integration, the agent won't learn properly
- Manual workflow is not a weakness - it's required for safety
- Start small ($100/day) and scale based on performance

Remember: The goal is not to replace human marketers but to augment them with superhuman pattern recognition and testing velocity.

## Competitive Positioning Map
- **Aura Strength**: AI behavioral insights (Balance)
- **Bark**: Safety/content filtering only
- **Qustodio**: Traditional monitoring
- **Life360**: Location tracking focus
- **Our Opportunity**: Own "behavioral health + AI" positioning