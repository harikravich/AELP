# GAELP Safety Framework

A comprehensive safety framework for the Generic Agent Experimentation & Learning Platform (GAELP) ad campaign learning system. This framework implements critical safety mechanisms to protect against financial loss, content violations, and unethical behavior.

## 🚨 Critical Safety Features

### 1. Budget Controls
- **Daily/Weekly/Monthly spending limits** with real-time monitoring
- **Automatic campaign pausing** when limits are exceeded
- **Rollback mechanisms** for failed campaigns
- **Spend anomaly detection** for unusual patterns
- **Emergency stop capabilities** across all campaigns

### 2. Content Safety
- **Content moderation** for text, images, and videos
- **Brand safety compliance** checking
- **Age-appropriate content filtering**
- **Platform policy validation** (Google Ads, Facebook, etc.)
- **Prohibited content detection** (illegal, harmful, misleading)

### 3. Performance Safety
- **Reward clipping** to prevent exploitation
- **Performance anomaly detection** for unusual metrics
- **A/B test statistical validation**
- **Campaign timeout mechanisms**
- **Reward hacking prevention**

### 4. Operational Safety
- **Sandbox environments** for safe testing
- **Graduated deployment** (simulation → small budget → full deployment)
- **Human approval workflows** for high-risk campaigns
- **Comprehensive audit logging**
- **Emergency stop procedures**

### 5. Data Safety & Privacy
- **PII detection and masking**
- **Secure credential management** for ad APIs
- **GDPR/CCPA compliance** features
- **Data retention policies**
- **Privacy protection measures**

### 6. Agent Behavior Safety
- **Action space constraints** (reasonable budgets, ethical targeting)
- **Behavior pattern monitoring** for stuck or repetitive actions
- **Ethical targeting guidelines** (no discrimination)
- **Human-in-the-loop interventions**
- **Violation tracking and escalation**

## 🏗️ Architecture

```
GAELP Safety Framework
├── Budget Controls          (budget_controls.py)
├── Content Safety          (content_safety.py)
├── Performance Safety      (performance_safety.py)
├── Operational Safety      (operational_safety.py)
├── Data Safety            (data_safety.py)
├── Agent Behavior Safety  (agent_behavior_safety.py)
└── Safety Orchestrator    (safety_orchestrator.py)
```

## 🚀 Quick Start

```python
from safety_framework.integration import create_gaelp_safety_integration

# Initialize safety framework
safety = create_gaelp_safety_integration({
    'max_daily_budget': 5000.0,
    'auto_pause_on_critical': True,
    'human_review_required': True
})

await safety.initialize()

# Validate a new campaign
campaign_data = {
    'id': 'campaign_123',
    'title': 'My Ad Campaign',
    'budget': 1000.0,
    'targeting': {'age_range': {'min': 18, 'max': 65}},
    'platform': 'google_ads'
}

result = await safety.validate_new_campaign(campaign_data, 'user_123')
if result['valid']:
    print("Campaign approved for deployment")
else:
    print(f"Campaign rejected: {result['violations']}")
```

## 🔧 Integration with GAELP Components

### Environment Registry Integration
```python
from safety_framework.integration import EnvironmentRegistryMiddleware

# Add safety validation to environment submissions
middleware = EnvironmentRegistryMiddleware(safety)
validation = await middleware.validate_environment_submission(env_data, user_id)
```

### Training Orchestrator Integration
```python
from safety_framework.integration import TrainingOrchestratorMiddleware

# Monitor agent actions during training
middleware = TrainingOrchestratorMiddleware(safety)
action_result = await middleware.validate_training_action(agent_id, action_type, params)
```

### MCP Integration Safety
```python
from safety_framework.integration import MCPIntegrationMiddleware

# Validate external API calls
middleware = MCPIntegrationMiddleware(safety)
api_result = await middleware.validate_external_api_call('google_ads_api', params)
```

## 📊 Safety Dashboard

Get real-time safety status:

```python
dashboard = safety.get_safety_dashboard()
print(f"System Health: {dashboard['overall_status']['health_score']}")
print(f"Active Violations: {dashboard['system_metrics']['active_violations']}")
```

## 🚨 Emergency Procedures

### Emergency Stop All Campaigns
```python
stop_result = await safety.emergency_stop_all_campaigns(
    reason="Suspicious activity detected",
    initiated_by="security_system"
)
```

### Emergency Pause Single Campaign
```python
pause_result = await safety.emergency_pause_campaign(
    campaign_id="campaign_123",
    reason="Budget anomaly detected",
    initiated_by="automated_system"
)
```

## 🔒 Security Features

### Budget Protection
- Multi-level budget limits (hourly, daily, weekly, monthly, total)
- Real-time spend monitoring with sub-second response times
- Automatic campaign pausing on limit violations
- Spend velocity monitoring for unusual patterns

### Content Protection
- Multi-platform policy compliance (Google, Facebook, Microsoft)
- Brand safety categorization and filtering
- Age-appropriate content validation
- Prohibited content detection with confidence scoring

### Behavioral Protection
- Agent action constraint enforcement
- Repetitive behavior detection and intervention
- Discriminatory targeting prevention
- Ethical guideline compliance monitoring

## 📈 Monitoring & Alerting

### Built-in Metrics
- System health score (0-1)
- Active violation count
- Budget utilization rates
- Content approval rates
- Performance anomaly rates

### Alert Integration
```python
# Register webhook for critical alerts
safety.register_alert_webhook("https://your-webhook.com/alerts")

# Register custom alert handler
async def custom_alert_handler(event):
    print(f"ALERT: {event.description}")
    
safety.register_human_review_callback(custom_alert_handler)
```

## 🛡️ Compliance Features

### GDPR/CCPA Compliance
- Automatic PII detection and masking
- User consent management
- Data retention policy enforcement
- Right to be forgotten implementation

### Industry Standards
- SOC 2 Type II compatible audit logging
- ISO 27001 security controls
- PCI DSS compliance for payment data
- COPPA compliance for child safety

## 🔧 Configuration

### Environment Variables
```bash
# Safety Framework Configuration
GAELP_SAFETY_MAX_DAILY_BUDGET=10000.0
GAELP_SAFETY_ENABLE_HUMAN_REVIEW=true
GAELP_SAFETY_AUTO_PAUSE_ON_CRITICAL=true
GAELP_SAFETY_ALERT_WEBHOOK_URL=https://your-webhook.com
GAELP_SAFETY_LOG_LEVEL=INFO
```

### Configuration File
```python
safety_config = {
    'enable_budget_controls': True,
    'enable_content_safety': True,
    'enable_performance_safety': True,
    'enable_operational_safety': True,
    'enable_data_safety': True,
    'enable_behavior_safety': True,
    'max_daily_budget': 10000.0,
    'content_violation_threshold': 3,
    'performance_anomaly_threshold': 0.8,
    'behavior_violation_threshold': 5,
    'human_review_required': True,
    'auto_pause_on_critical': True
}
```

## 🧪 Testing

Run the test suite:
```bash
pip install -r requirements.txt
pytest tests/ -v --cov=safety_framework
```

## 📝 Audit Logging

All safety events are automatically logged with:
- Event timestamp and unique ID
- User/agent responsible for action
- Campaign and context information
- Safety violations and actions taken
- Resolution status and timeline

## 🚧 Deployment Recommendations

### Production Setup
1. **Enable all safety modules** in production
2. **Set conservative limits** initially
3. **Monitor dashboards** continuously
4. **Configure human review** for high-risk actions
5. **Test emergency procedures** regularly

### Staging/Development
1. **Use sandbox environments** for testing
2. **Enable graduated deployment** for new features
3. **Test violation scenarios** thoroughly
4. **Validate alert mechanisms**

## 📞 Support & Emergency Contacts

### Emergency Procedures
1. **Immediate**: Call emergency stop API
2. **Within 5 minutes**: Notify human reviewers
3. **Within 15 minutes**: Contact GAELP operations team
4. **Within 1 hour**: Generate incident report

### Contact Information
- **Emergency Hotline**: [Emergency Contact]
- **Technical Support**: [Support Contact]
- **Security Team**: [Security Contact]

## ⚠️ Important Notes

### Critical Requirements
- **NEVER** deploy campaigns without safety validation
- **ALWAYS** monitor spending in real-time
- **IMMEDIATELY** investigate critical violations
- **REGULARLY** review and update safety thresholds

### Legal Compliance
This framework helps with compliance but does not guarantee it. Always consult with legal counsel for:
- Platform policy interpretation
- Privacy law compliance
- Advertising regulation adherence
- Data protection requirements

---

**Remember: Safety is not optional. Every campaign must pass safety validation before deployment.**