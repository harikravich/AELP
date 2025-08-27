# GAELP Production Safety Framework Deployment Guide

## 🚨 CRITICAL: REAL MONEY SYSTEM 🚨

This is the **PRODUCTION-READY** GAELP Safety Framework that handles **REAL MONEY** transactions, **ACTUAL REGULATORY COMPLIANCE**, and **LIVE AD CAMPAIGNS**. This system has been upgraded from mock demonstrations to handle real financial transactions with proper safety controls.

## ⚠️ BEFORE DEPLOYMENT ⚠️

**READ THIS ENTIRE DOCUMENT** before deploying to production. This system:
- Processes real credit card transactions through Stripe
- Implements actual GDPR, CCPA, and COPPA compliance
- Can trigger emergency stops that affect real campaigns
- Sends real alerts to emergency contacts
- Logs all actions for regulatory audit

## Production Components Upgraded

### 1. Real Money Budget Controls (`production_budget_controls.py`)
- **Stripe integration** for actual payment processing
- **Real-time fraud detection** with transaction blocking
- **Multi-tier emergency stops** with financial impact assessment
- **BigQuery audit logging** for all financial transactions
- **Cloud Monitoring integration** for real-time metrics
- **Regulatory compliance** checks for spending limits

**CRITICAL SAFETY FEATURES:**
- Master kill switch for immediate system shutdown
- Automatic campaign pausing on budget violations
- Real-time fraud score calculation
- Emergency stop with payment method disabling

### 2. AI-Powered Content Safety (`production_content_safety.py`)
- **OpenAI GPT-4 moderation** for text content
- **Google Vision AI** for image and video analysis
- **Perspective API** for toxicity detection
- **Platform-specific policy engines** (Google Ads, Facebook, etc.)
- **GDPR/CCPA/COPPA compliance** checking
- **Multi-language support** with cultural sensitivity

**REAL AI INTEGRATIONS:**
- OpenAI Moderation API
- Google Cloud Vision API
- Google Cloud Video Intelligence API
- Google Cloud Natural Language API
- Perspective API for toxicity

### 3. Emergency Control Systems (`emergency_controls.py`)
- **Multi-channel emergency notifications** (Slack, SMS, email, phone calls)
- **Regulatory compliance engine** with GDPR/CCPA/COPPA
- **Crisis management workflows** with escalation procedures
- **Legal notification systems** for compliance violations
- **Post-mortem automation** for critical incidents

**EMERGENCY RESPONSE LEVELS:**
- **LOW**: Increased monitoring, on-call team notified
- **MEDIUM**: Reduced spending, management notified
- **HIGH**: All campaigns paused, executive team notified
- **CRITICAL**: Complete system shutdown, board notified
- **CATASTROPHIC**: Total lockdown, regulatory authorities notified

### 4. Production Monitoring (`production_monitoring.py`)
- **Prometheus metrics** with real-time dashboards
- **Real-time alerting** with PagerDuty/Slack integration
- **Performance monitoring** with SLA tracking
- **Health checks** with automatic failover
- **Compliance monitoring** with regulatory reporting

**MONITORING CAPABILITIES:**
- Real-time budget violation tracking
- Content moderation performance metrics
- Fraud detection accuracy monitoring
- System health scoring
- Regulatory compliance dashboards

## Pre-Deployment Checklist

### 🔐 Security Requirements
- [ ] Stripe API keys configured (live keys, not test)
- [ ] Google Cloud service account with appropriate permissions
- [ ] OpenAI API key with sufficient credits
- [ ] All API keys stored in Google Secret Manager
- [ ] Network security groups configured for production
- [ ] SSL certificates installed and validated

### 💰 Financial Setup
- [ ] Stripe account configured for live transactions
- [ ] Payment methods verified and active
- [ ] Daily/monthly spending limits configured
- [ ] Bank account linked for settlements
- [ ] Fraud monitoring thresholds set
- [ ] Backup payment methods configured

### 🚨 Emergency Contacts
- [ ] Primary safety engineer contact configured
- [ ] Engineering manager emergency contact
- [ ] Executive team emergency contacts
- [ ] Legal team for compliance violations
- [ ] PR team for crisis communications
- [ ] All contacts tested with sample alerts

### 📊 Monitoring Setup
- [ ] Grafana dashboard deployed
- [ ] PagerDuty integration configured
- [ ] Slack alerting channels created
- [ ] Email notification lists configured
- [ ] Monitoring retention policies set
- [ ] Alert escalation procedures tested

### ⚖️ Regulatory Compliance
- [ ] GDPR data processing agreements signed
- [ ] CCPA compliance procedures documented
- [ ] COPPA parental consent workflows implemented
- [ ] Data retention policies configured
- [ ] Privacy policy updated with safety disclosures
- [ ] Regulatory reporting procedures established

## Environment Variables

Create a `.env` file with the following production configuration:

```bash
# Environment
GAELP_ENVIRONMENT=production

# Financial Controls
STRIPE_API_KEY=sk_live_your_stripe_key_here
MAX_DAILY_GLOBAL_SPEND=1000000.0
FRAUD_THRESHOLD=0.7

# Payment Methods
PRIMARY_STRIPE_PAYMENT_METHOD_ID=pm_your_primary_payment_method
PRIMARY_CARD_LAST_FOUR=1234
PRIMARY_CARD_DAILY_LIMIT=100000.0
PRIMARY_CARD_MONTHLY_LIMIT=2000000.0

BACKUP_STRIPE_PAYMENT_METHOD_ID=pm_your_backup_payment_method
BACKUP_CARD_LAST_FOUR=5678
BACKUP_CARD_DAILY_LIMIT=50000.0
BACKUP_CARD_MONTHLY_LIMIT=1000000.0

# AI Services
OPENAI_API_KEY=sk-your_openai_key_here
PERSPECTIVE_API_KEY=your_perspective_api_key

# Google Cloud
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
BIGQUERY_DATASET=safety_audit

# Monitoring & Alerting
PROMETHEUS_ENABLED=true
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
PAGERDUTY_INTEGRATION_KEY=your-pagerduty-key

# Emergency Contacts
EMERGENCY_CONTACT_1_NAME="Safety Engineer"
EMERGENCY_CONTACT_1_EMAIL=safety@yourcompany.com
EMERGENCY_CONTACT_1_PHONE=+1-555-0001
EMERGENCY_CONTACT_1_SLACK=U1234567890

EMERGENCY_CONTACT_2_NAME="Engineering Manager"
EMERGENCY_CONTACT_2_EMAIL=engineering-manager@yourcompany.com
EMERGENCY_CONTACT_2_PHONE=+1-555-0002
EMERGENCY_CONTACT_2_SLACK=U2345678901

EMERGENCY_CONTACT_EXEC_NAME="CTO"
EMERGENCY_CONTACT_EXEC_EMAIL=cto@yourcompany.com
EMERGENCY_CONTACT_EXEC_PHONE=+1-555-0003

# Compliance
GDPR_ENABLED=true
CCPA_ENABLED=true
COPPA_ENABLED=true
REGULATORY_NOTIFICATIONS=true

# Feature Flags
ENABLE_REAL_MONEY=true
ENABLE_AI_MODERATION=true
ENABLE_EMERGENCY_STOPS=true
ENABLE_MONITORING=true
CONTENT_MODERATION_STRICT=true
```

## Deployment Steps

### 1. Install Dependencies
```bash
cd /home/hariravichandran/AELP/safety_framework
pip install -r production_requirements.txt
```

### 2. Validate Configuration
```bash
python deploy_production.py --check-config
```

### 3. Deploy to Production
```bash
python deploy_production.py
```

### 4. Verify Deployment
The deployment script will:
- Initialize all safety systems
- Register payment methods
- Test emergency systems
- Create monitoring dashboards
- Perform health checks

## Production Usage

### Campaign Validation
```python
from production_integration import ProductionSafetyOrchestrator, SafetyValidationRequest

# Create validation request
request = SafetyValidationRequest(
    request_id="campaign_001",
    request_type="campaign_creation",
    campaign_data={
        "id": "campaign_001",
        "title": "Summer Sale Campaign",
        "description": "Limited time summer sale offers",
        "target_audience": "adults_25_54",
        "min_age": 18
    },
    financial_data={
        "daily_limit": 5000.0,
        "weekly_limit": 25000.0,
        "monthly_limit": 100000.0,
        "total_limit": 500000.0,
        "payment_method_id": "primary_production_card",
        "min_roi": 0.15,
        "max_cpa": 50.0
    },
    content_data={
        "title": "Amazing Summer Sale - Up to 70% Off!",
        "description": "Don't miss our biggest sale of the year...",
        "ad_copy": "Limited time offer. Shop now and save big!"
    },
    user_id="user_123",
    target_platforms=["google_ads", "facebook_ads"],
    gdpr_applicable=True,
    ccpa_applicable=True
)

# Validate campaign
response = await orchestrator.validate_campaign_creation(request)

if response.can_proceed:
    print("Campaign approved for deployment")
else:
    print(f"Campaign rejected: {response.violations}")
```

### Emergency Stop
```python
from emergency_controls import EmergencyLevel

# Trigger emergency stop
stop_id = await orchestrator.trigger_emergency_stop(
    level=EmergencyLevel.HIGH,
    reason="Fraudulent activity detected",
    triggered_by="fraud_detection_system",
    context={
        "affected_campaigns": ["campaign_001", "campaign_002"],
        "fraud_score": 0.95,
        "estimated_financial_impact": 50000.0
    }
)
```

## Monitoring Dashboards

The system automatically creates Grafana dashboards for:

### Safety Overview Dashboard
- System health score
- Active campaigns count
- Budget utilization
- Emergency stops timeline
- Compliance status

### Financial Controls Dashboard
- Real-time spending by campaign
- Budget violations tracking
- Fraud detection alerts
- Payment method health
- ROI tracking

### Content Safety Dashboard
- Moderation queue size
- Approval/rejection rates
- Violation types breakdown
- AI service performance
- Platform policy compliance

### Emergency Response Dashboard
- Active emergency stops
- Emergency contact status
- Response time metrics
- Escalation tracking
- Recovery procedures

## Alert Configurations

### Critical Alerts (PagerDuty + Phone)
- Emergency stops triggered
- Critical budget violations
- Payment processing failures
- System health below 0.5
- Compliance violations

### Warning Alerts (Slack + Email)
- High budget violation rates
- Content moderation queue backup
- System health below 0.7
- Unusual spending patterns
- API service degradation

### Info Alerts (Slack only)
- Daily spending summaries
- Weekly safety reports
- Successful emergency recoveries
- System maintenance notifications

## Regulatory Compliance

### GDPR Compliance
- Automatic consent verification
- Data processing lawful basis validation
- Right to erasure implementation
- Data breach notification (72-hour rule)
- Cross-border data transfer controls

### CCPA Compliance
- California resident identification
- Opt-out mechanism validation
- Personal information sale controls
- Non-discrimination enforcement
- Consumer rights verification

### COPPA Compliance
- Age verification for users under 13
- Parental consent validation
- Child data protection measures
- Educational content guidelines
- Safe harbor provisions

## Recovery Procedures

### System Recovery
1. **Assess Impact**: Determine scope of incident
2. **Stabilize Systems**: Ensure no ongoing damage
3. **Restore Services**: Gradually bring systems online
4. **Validate Operations**: Confirm all systems working
5. **Document Incident**: Complete post-mortem analysis

### Financial Recovery
1. **Stop Spending**: Immediate campaign pause
2. **Assess Damage**: Calculate financial impact
3. **Secure Accounts**: Disable compromised payment methods
4. **Investigate**: Determine root cause
5. **Recover Funds**: Initiate chargebacks/refunds if needed

### Data Breach Response
1. **Contain Breach**: Stop data exposure
2. **Assess Impact**: Determine data affected
3. **Notify Authorities**: GDPR 72-hour notification
4. **Notify Users**: Individual notifications
5. **Remediate**: Fix security vulnerabilities

## Production Support

### 24/7 Emergency Hotline
- **Phone**: +1-XXX-XXX-XXXX
- **Email**: emergency@yourcompany.com
- **Slack**: #safety-emergency

### Escalation Matrix
1. **L1**: Safety Engineer (0-15 minutes)
2. **L2**: Engineering Manager (15-30 minutes)
3. **L3**: Engineering Director (30-60 minutes)
4. **L4**: CTO (1-2 hours)
5. **L5**: CEO (2+ hours)

### Documentation
- **Runbooks**: `/docs/production_runbooks/`
- **API Documentation**: `/docs/api/`
- **Compliance Guides**: `/docs/compliance/`
- **Emergency Procedures**: `/docs/emergency/`

## Testing in Production

### Safety Testing
- Monthly emergency stop drills
- Quarterly compliance audits
- Annual penetration testing
- Continuous chaos engineering
- Real-time monitoring validation

### Performance Testing
- Load testing with real payment methods
- Stress testing emergency systems
- Disaster recovery testing
- API rate limit testing
- Database failover testing

## Maintenance Windows

### Scheduled Maintenance
- **Weekly**: System health checks (Sundays 2-4 AM UTC)
- **Monthly**: Security updates (First Saturday 2-6 AM UTC)
- **Quarterly**: Major upgrades (Planned with 30-day notice)

### Emergency Maintenance
- Critical security patches: Immediate deployment
- System failures: Maximum 4-hour response time
- Data corruption: Immediate emergency response

## Legal Disclaimers

⚠️ **IMPORTANT LEGAL NOTICES** ⚠️

1. **Financial Liability**: This system processes real money. Ensure proper insurance coverage.
2. **Regulatory Compliance**: You are responsible for compliance with local laws.
3. **Data Protection**: Ensure GDPR/CCPA compliance in your jurisdiction.
4. **Emergency Procedures**: Test all emergency contacts and procedures.
5. **Audit Requirements**: Maintain audit logs as required by law.

## Support & Contact

For production deployment support:
- **Technical Issues**: engineering@yourcompany.com
- **Compliance Questions**: legal@yourcompany.com
- **Emergency Support**: +1-XXX-XXX-XXXX

---

**⚠️ FINAL WARNING ⚠️**

This is a PRODUCTION system handling REAL MONEY. Do not deploy without:
1. Complete testing in staging environment
2. All emergency procedures validated
3. Full team training completed
4. Legal and compliance approval
5. Executive sign-off on financial limits

**Deploy responsibly. Money and compliance are not optional.**