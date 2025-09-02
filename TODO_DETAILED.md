# GAELP Production Sprint - Detailed TODO List
**Last Updated:** September 1, 2025
**Priority:** CRITICAL - Production Sprint Day 2

## 🚨 IMMEDIATE FIXES REQUIRED

### 1. Fix Training Auction Parameters ⚡ CRITICAL
**Problem:** RL agent has won 0 auctions after 20,000+ episodes
**Root Cause:** Simulator using $0.50-$5 bids vs real CPCs of $20-50
**Actions:**
- [ ] Check current training status: `ps aux | grep parallel_training`
- [ ] If still 0 wins, stop training immediately
- [ ] Update `enhanced_simulator_fixed.py`:
  - Change bid ranges from (0.5, 5.0) to (5.0, 50.0)
  - Adjust competition bids to match real CPCs:
    - Brand keywords: $5-15
    - Non-brand: $20-50
    - Competitor: $30-80
- [ ] Update `parallel_training_accelerator.py` reward scaling
- [ ] Restart training with realistic parameters

## 📊 DATA EXTRACTION & INTEGRATION

### 2. Complete GA4 Data Extraction
**Status:** Partially complete (August week 1 done)
**Missing:** 11+ months of historical data

#### Campaign Data Extraction
- [ ] Extract all campaigns for last 12 months
- [ ] Categorize by product (PC, AV, Identity, Balance)
- [ ] Identify campaign types (Search, Social, Display, PMax)
- [ ] Track performance metrics (CPC, CVR, CPA, ROAS)
- [ ] Map which campaigns drive trials vs D2P

#### Product Performance
- [ ] Extract all product SKUs with "FT" (trials)
- [ ] Extract all product SKUs with "No DBOO" (D2P)
- [ ] Calculate daily averages by product category
- [ ] Track revenue per product line
- [ ] Identify top performers

#### Creative/AB Testing
- [ ] Extract all campaigns with "ab-" prefix
- [ ] Identify test variants and winners
- [ ] Track landing page tests (68% off, welcome back, etc.)
- [ ] Document pricing test results
- [ ] Map creative performance to conversion rates

#### User Journey Extraction
- [ ] Multi-touch attribution paths
- [ ] Average touchpoints before conversion
- [ ] Time to conversion distribution
- [ ] Channel sequences
- [ ] Device transition patterns

### 3. Build GA4 + Internal Data Merger
**Challenge:** GA4 missing 31% of conversions (app data)
**Solution:** Merge GA4 with internal subscription data

- [ ] Create data pipeline architecture
- [ ] Map GA4 events to internal metrics:
  - GA4 'purchase' → D2P + Post-trial conversions
  - Internal trials → Web + App trial starts
  - Internal subscribers → Include app store subs
- [ ] Build cohort tracking system:
  - Track trial starts by date
  - Monitor 14-day conversion windows
  - Calculate true trial-to-paid rates
- [ ] Create unified dataset for training

## 🤖 GAELP TRAINING ADJUSTMENTS

### 4. Implement Realistic Training Parameters
**Current:** Synthetic data with wrong assumptions
**Target:** Real-world Aura metrics

#### Update Simulator
- [ ] Set trial conversion rate to 70% (not 30%)
- [ ] Implement 14-day attribution window
- [ ] Add trial vs D2P funnel logic
- [ ] Use actual product distribution:
  - PC: 7% of volume
  - AV: 20% of volume  
  - D2P: 55% of transactions
  - Trials: 45% of transactions

#### Bidding Strategy Updates
- [ ] Implement dual bidding strategies:
  - Trials: Max bid = Target CPA × 0.70
  - D2P: Max bid = Target CPA × 0.95
- [ ] Add product-specific bid adjustments
- [ ] Implement iOS targeting for Balance
- [ ] Add time-of-day bid modifiers

#### Reward Function Improvements
- [ ] Account for delayed trial conversions
- [ ] Weight D2P conversions higher (immediate revenue)
- [ ] Add LTV considerations (D2P has lower churn)
- [ ] Penalize excessive trial acquisition if conversion drops

### 5. Balance/Thrive Specific Optimizations
**Opportunity:** Small volume but 100% CVR on targeted campaigns
**Constraint:** iOS only

- [ ] Create iOS-specific bidding strategy
- [ ] Target behavioral health keywords
- [ ] Leverage PC customer base for cross-sell
- [ ] Track Balance adoption from PC users
- [ ] Build lookalike audiences from converters

## 📈 VALIDATION & TESTING

### 6. Validate Data Accuracy
- [ ] Cross-check GA4 totals with internal reporting
- [ ] Verify trial conversion rates by cohort
- [ ] Confirm revenue per conversion calculations
- [ ] Validate device/platform breakdowns
- [ ] Check for seasonal patterns

### 7. Test GAELP with Real Data
- [ ] Create test scenarios with actual campaigns
- [ ] Validate bid recommendations against historical performance
- [ ] Test attribution window handling
- [ ] Verify product categorization accuracy
- [ ] Check creative selection logic

## 📋 DOCUMENTATION & REPORTING

### 8. Create Comprehensive Documentation
- [ ] Document GA4 tracking capabilities and gaps
- [ ] Create data dictionary for all metrics
- [ ] Map campaign naming conventions
- [ ] Document trial vs D2P identification rules
- [ ] Create troubleshooting guide

### 9. Build Performance Dashboard
- [ ] Real-time training metrics
- [ ] Auction win rate tracking
- [ ] Conversion rate by funnel type
- [ ] Revenue attribution reporting
- [ ] Product performance comparison

## 🎯 SUCCESS METRICS

### Key Performance Indicators
1. **Training Progress**
   - [ ] Achieve >0% auction win rate
   - [ ] Complete 100,000 episodes
   - [ ] Reduce loss below 0.01
   - [ ] Achieve stable policy

2. **Data Completeness**
   - [ ] Extract 12 months of GA4 data
   - [ ] Map 95%+ of campaigns to products
   - [ ] Track 100% of trial cohorts
   - [ ] Merge app + web data successfully

3. **Model Performance**
   - [ ] Beat baseline CPA by 20%
   - [ ] Maintain 70% trial conversion
   - [ ] Optimize bid allocation across products
   - [ ] Identify winning creatives

## 🔄 Daily Checklist

### Every Session Should:
1. Check training status and metrics
2. Review auction win rate
3. Validate data extraction progress
4. Test latest model updates
5. Document findings in SESSION_SUMMARY.md
6. Update this TODO list

## 📝 Notes for Next Session

### Critical Context
- User has 1,134 paying subscribers/day in August
- 70% trial conversion is CORRECT (not 30%)
- GA4 missing all trial starts (534/day)
- Real CPCs are $20-50, not $2-5
- Mobile/iOS is 61% of business
- Balance only works on iOS
- PC is gateway product despite lower volume

### Quick Commands
```bash
# Check training
ps aux | grep parallel_training
tail -n 100 training_output.log

# Monitor GPU/CPU
htop
nvidia-smi

# Check extraction
ls -la ga4_extracted_data/
python3 july_ga4_mapping_final.py

# Run tests
python3 verify_all_components.py --strict
```

### File Locations
- Training: `/home/hariravichandran/AELP/parallel_training_accelerator.py`
- Monitor: `/home/hariravichandran/AELP/monitor_training.py`
- GA4 Scripts: `/home/hariravichandran/AELP/ga4_*.py`
- Data: `/home/hariravichandran/AELP/ga4_extracted_data/`

---
**Remember:** NO FALLBACKS, NO SIMPLIFICATIONS - Fix the real problems!