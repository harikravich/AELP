# GAELP Production Ad Account Setup

## 🚨 CRITICAL WARNING: REAL MONEY SYSTEM

**This system creates REAL ad accounts with REAL billing that will charge REAL MONEY to your credit card. There are NO sandbox or test modes. Every click, impression, and conversion costs actual dollars.**

## Overview

Complete production infrastructure for behavioral health advertising campaigns with:

- **Google Ads account** with proper billing and API access
- **Facebook Business Manager** with ad account and pixel setup
- **Conversion tracking** on all landing pages with iOS 14.5+ compliance
- **UTM parameter system** with gaelp_uid for attribution
- **Budget safeguards** with automated campaign pausing
- **Real-time monitoring** and emergency stops

## Budget Protection

### Limits Enforced
- **Daily:** $100 across all platforms
- **Monthly:** $3,000 total spend
- **Emergency Stop:** $5,000 absolute maximum

### Safety Features
- ✅ Real-time spend monitoring every 15 minutes
- ✅ Automated campaign pausing at budget thresholds
- ✅ Email/SMS alerts at 75%, 90%, 100% of budget
- ✅ Emergency kill switch at $5,000
- ✅ Platform-specific limits (Google: $60/day, Facebook: $40/day)

## Files Structure

```
/home/hariravichandran/AELP/
├── production_ad_account_manager.py     # Main account setup system
├── conversion_tracking_pixels.py        # Landing page tracking implementation
├── budget_safety_monitor.py            # Real-time budget protection
├── setup_production_ads.py             # Complete setup orchestrator
├── quick_start_production_ads.py       # Setup demonstration and runner
└── ~/.config/gaelp/                    # Generated setup files and credentials
```

## Quick Start

### 1. Prerequisites Check
```bash
python3 quick_start_production_ads.py
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Complete Setup
```bash
python3 setup_production_ads.py
```

## Detailed Setup Process

### Step 1: Google Ads Account Setup
- OAuth authentication with Google account
- Developer token request and approval
- Customer account creation with billing
- Conversion action setup for behavioral health
- Budget limit configuration ($60/day)
- API access validation

### Step 2: Facebook Business Manager Setup
- Business Manager creation and verification
- Ad account setup with billing
- Facebook Pixel implementation
- Domain verification (teen-wellness-monitor.com)
- iOS 14.5+ compliance configuration
- Conversions API setup

### Step 3: Conversion Tracking Implementation
- Landing page pixel code generation
- Enhanced conversions for iOS privacy
- Server-side Conversions API backup
- Event configuration:
  - PageView (all visits)
  - Lead (email signups) - $5 value
  - AddToCart (trial starts) - $25 value  
  - Purchase (subscriptions) - $99 value

### Step 4: UTM Parameter System
- Standard UTM parameters (source, medium, campaign, content, term)
- GAELP custom parameters:
  - `gaelp_uid`: Unique session identifier
  - `gaelp_test`: Test variant tracking
  - `gaelp_agent`: Agent version
  - `gaelp_world`: Simulation world
  - `gaelp_sig`: Verification signature

### Step 5: Budget Monitoring Setup
- Real-time spend tracking database
- Platform API integration for spend data
- Alert system configuration (email/SMS)
- Emergency action automation
- Monitoring service daemon

## Campaign Structure

### Google Ads Campaigns
```json
{
  "name": "Behavioral_Health_Search_Test",
  "type": "SEARCH",
  "daily_budget": 25.0,
  "ad_groups": [
    {
      "name": "Crisis_Keywords",
      "keywords": [
        "\"teen depression help\"",
        "\"is my teen okay\"",
        "[teen mental health crisis]"
      ]
    }
  ]
}
```

### Facebook Campaigns  
```json
{
  "name": "iOS_Parents_Test", 
  "objective": "OUTCOME_LEADS",
  "daily_budget": 25.0,
  "ad_sets": [
    {
      "name": "Crisis_Parents",
      "targeting": {
        "age_min": 35,
        "age_max": 55,
        "interests": ["Mental health", "Parenting"],
        "behaviors": ["Parents (Teens 13-17)"]
      }
    }
  ]
}
```

## Post-Setup Launch Process

### 1. Start Budget Monitoring (CRITICAL)
```bash
cd ~/.config/gaelp/production_setup/
python3 start_budget_monitoring.py
```
**⚠️ MUST BE RUNNING BEFORE LAUNCHING CAMPAIGNS**

### 2. Deploy Landing Page Tracking
- Upload `landing_page_template.html` to web server
- Verify all tracking pixels fire correctly
- Test conversion events in platform interfaces

### 3. Launch Test Campaigns
```bash
python3 launch_test_campaigns.py
```
- Start with $50/day total budget
- Monitor performance hourly for first 24 hours
- Validate conversion attribution

### 4. Monitor and Scale
- Check spend daily
- Review conversion tracking
- Scale based on performance data
- Maintain budget discipline

## API Credentials Management

Credentials are stored securely in encrypted SQLite database:
```
~/.config/gaelp/ad_accounts/credentials.db
```

### Credential Types Stored
- Google Ads: OAuth refresh token, customer ID, developer token
- Facebook: Access token, app ID, app secret, ad account ID
- All tokens encrypted and access-controlled

## Budget Monitoring Details

### Monitoring Frequency
- **Real-time:** Every 15 minutes
- **Hourly:** Detailed spend reports
- **Daily:** Comprehensive analysis and alerts

### Alert Thresholds
- **75%:** Warning email sent
- **90%:** Danger alert with SMS
- **100%:** Emergency - campaigns paused automatically
- **110%:** Kill switch - all advertising stopped

### Emergency Actions
1. **Budget Warning (75%):** Email alert sent
2. **Budget Danger (90%):** Email + SMS alerts
3. **Budget Emergency (100%):** All campaigns paused automatically
4. **Kill Switch (110%):** Complete ad spend stoppage

## Conversion Tracking Implementation

### Client-Side Tracking
```html
<!-- Google Analytics 4 with Enhanced Conversions -->
<script async src="https://www.googletagmanager.com/gtag/js?id=AW-XXXXXXXXX"></script>

<!-- Facebook Pixel with Advanced Matching -->
<script>fbq('init', 'XXXXXXXXXXXXXXXXX', {
  em: 'hashed_email',
  fn: 'hashed_first_name'
});</script>

<!-- GAELP Custom Tracking -->
<script>
window.gaelp_track = function(event_name, event_data) {
  // Enhanced tracking with GAELP parameters
};
</script>
```

### Server-Side Conversions API
- Facebook Conversions API for iOS 14.5+ accuracy
- Enhanced data sharing with hashed PII
- Backup tracking for client-side blocking
- Real-time conversion validation

## Security and Compliance

### Data Protection
- All PII hashed before transmission
- Secure credential storage with encryption
- GDPR/CCPA compliant data handling
- Regular security audits

### Platform Compliance
- Google Ads policy adherence
- Facebook advertising standards
- Health claims compliance
- Age-appropriate targeting

## Troubleshooting

### Common Setup Issues

**OAuth Failures:**
- Verify Google account access (hari@aura.com)
- Check developer token approval status
- Confirm Facebook app configuration

**Budget Monitoring Not Working:**
- Check platform API credentials
- Verify spend tracking database permissions
- Test email/SMS notification settings

**Conversion Tracking Issues:**
- Validate pixel installation on landing pages
- Check enhanced conversions setup
- Test server-side API connections

### Support Resources
- Platform documentation and support
- Setup logs in `~/.config/gaelp/`
- Budget monitoring alerts for issues

## Cost Management

### Expected Costs

**Setup Phase:**
- Google Ads: $0 (setup only)
- Facebook: $0 (setup only)
- Total: $0

**Testing Phase (First Week):**
- Daily budget: $50
- Weekly spend: ~$350
- Learning and optimization

**Production Phase:**
- Daily budget: $100
- Monthly spend: ~$3,000
- Scaled campaign performance

### Cost Control Measures
- Prepaid card usage recommended
- Daily spend limits enforced
- Real-time monitoring prevents overspend
- Emergency stops at multiple thresholds
- Manual approval required for budget increases

## Performance Monitoring

### Key Metrics Tracked
- **Spend:** Daily, weekly, monthly across platforms
- **Performance:** CTR, CPC, CPM, CPA, ROAS
- **Conversions:** Volume, value, attribution
- **Attribution:** Cross-device, cross-platform tracking

### Reporting
- Real-time dashboard with spend and performance
- Daily email reports with key metrics
- Weekly analysis of campaign performance
- Monthly budget and ROI review

## Important Reminders

### Before Launch
- ✅ Budget monitoring service running
- ✅ Landing page tracking deployed and tested
- ✅ Conversion events validating in platforms
- ✅ Emergency contact information updated
- ✅ Prepaid card or spending limits set

### During Operation  
- ✅ Check spend daily
- ✅ Monitor email alerts closely
- ✅ Validate conversion attribution
- ✅ Respond to budget warnings immediately
- ✅ Keep monitoring service running

### Emergency Procedures
- **Budget overage:** Campaigns pause automatically
- **Conversion tracking fails:** Fix immediately to maintain attribution
- **Platform policy violation:** Address quickly to avoid suspension
- **Technical issues:** Check logs and restart monitoring

---

## 🚨 FINAL WARNING

**THIS SYSTEM USES REAL MONEY. Every click costs actual dollars. You are responsible for all advertising costs. Monitor spend carefully and keep budget protection active at all times.**

**Generated:** 2025-08-23
**Version:** Production v1.0
**Contact:** Budget alerts sent to hari@aura.com