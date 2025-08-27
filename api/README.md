# GAELP API Documentation

## Overview

The GAELP (Generalized Ad Environment Learning Platform) API provides a comprehensive suite of endpoints for managing ad campaign optimization through reinforcement learning. The platform bridges simulation and real-world environments, enabling safe training and deployment of AI agents for advertising optimization.

## Architecture

The GAELP API consists of five main components:

### 1. Environment API (`/v1/environments`)
- **Purpose**: Manages both simulated and real ad environments
- **Key Operations**: Reset environments, execute campaign steps, render visualizations, collect audience feedback
- **Use Cases**: Training environments, A/B testing, campaign simulation

### 2. Agent API (`/v1/agents`)  
- **Purpose**: Manages AI agents that optimize ad campaigns
- **Key Operations**: Agent registration, campaign selection, policy updates, creative generation
- **Use Cases**: RL agent deployment, automated optimization, creative generation

### 3. Campaign Management API (`/v1/campaigns`)
- **Purpose**: Full lifecycle management of ad campaigns
- **Key Operations**: Campaign CRUD, performance tracking, status control, historical analysis
- **Use Cases**: Campaign execution, performance monitoring, optimization

### 4. Simulation API (`/v1/simulation`)
- **Purpose**: Detailed user persona simulation and response modeling
- **Key Operations**: Persona creation, response simulation, batch testing, market simulation
- **Use Cases**: Safe training data generation, campaign pre-testing, market analysis

### 5. Safety & Budget API (`/v1/safety`)
- **Purpose**: Safety validation and budget protection mechanisms
- **Key Operations**: Spending limits, safety validation, emergency controls, audit logging
- **Use Cases**: Risk management, compliance, budget protection

## Quick Start

### Authentication

All API requests require authentication via API key or Bearer token:

```bash
# Using API Key
curl -H "X-API-Key: your-api-key" https://api.gaelp.dev/v1/campaigns

# Using Bearer Token
curl -H "Authorization: Bearer your-jwt-token" https://api.gaelp.dev/v1/campaigns
```

### Base URLs

- **Production**: `https://api.gaelp.dev`
- **Staging**: `https://staging-api.gaelp.dev`

### Rate Limits

- **Standard Tier**: 1,000 requests/hour
- **Premium Tier**: 10,000 requests/hour
- **Enterprise Tier**: Custom limits

## Core Workflows

### 1. Simulation-to-Real Training Pipeline

```python
# Step 1: Create diverse personas
persona_response = requests.post(
    "https://api.gaelp.dev/v1/simulation/personas",
    json={
        "persona_name": "Tech-Savvy Millennial",
        "demographics": {"age": 28, "income": 75000},
        "interests": [{"category": "fitness", "interest_level": 0.9}]
    }
)

# Step 2: Run batch simulations
simulation_response = requests.post(
    "https://api.gaelp.dev/v1/simulation/batch-simulate",
    json={
        "personas": [persona_response.json()["persona_id"]],
        "campaigns": [campaign_config],
        "batch_config": {"sample_size_per_persona": 1000}
    }
)

# Step 3: Train agent on simulation results
training_response = requests.post(
    "https://api.gaelp.dev/v1/agents/agent-123/update-policy",
    json={
        "performance_data": simulation_response.json()["aggregated_results"]
    }
)

# Step 4: Deploy to real environment
real_deployment = requests.post(
    "https://api.gaelp.dev/v1/environments/real-env/step",
    json={"ad_campaign": optimized_campaign}
)
```

### 2. Campaign Safety Validation

```python
# Validate campaign before launch
safety_check = requests.post(
    "https://api.gaelp.dev/v1/safety/campaign-safety",
    json={
        "campaign": campaign_config,
        "validation_level": "strict"
    }
)

if safety_check.json()["validation_result"] == "approved":
    # Safe to launch
    launch_response = requests.post(
        "https://api.gaelp.dev/v1/campaigns",
        json=campaign_config
    )
```

### 3. Automated Budget Management

```python
# Set spending limits
limits_response = requests.post(
    "https://api.gaelp.dev/v1/safety/spending-limits",
    json={
        "limit_scope": "account",
        "limits": {
            "daily_budget": 1000.0,
            "emergency_stop_threshold": 5000.0
        },
        "alert_thresholds": {
            "warning_threshold_percentage": 80
        }
    }
)

# Monitor spending status
status_response = requests.get(
    "https://api.gaelp.dev/v1/safety/spending-status?include_projections=true"
)

# Emergency controls if needed
if status_response.json()["velocity_analysis"]["velocity_status"] == "critical":
    emergency_response = requests.post(
        "https://api.gaelp.dev/v1/safety/emergency-controls",
        json={
            "action": "pause_all",
            "reason": "Spending velocity too high"
        }
    )
```

## API Features

### Advanced Simulation Capabilities

- **Realistic Persona Modeling**: Detailed demographic, psychographic, and behavioral modeling
- **Emotional Response Simulation**: Models user emotions and sentiment toward ads
- **Attention Modeling**: Simulates visual attention and cognitive load
- **Market-Level Simulation**: Full market dynamics including competition and external factors

### Safety and Compliance

- **Multi-Layer Validation**: Content safety, brand safety, compliance, fraud detection
- **Real-Time Monitoring**: Spending velocity analysis and anomaly detection
- **Emergency Controls**: Automatic stops and gradual recovery mechanisms
- **Audit Logging**: Complete audit trail for compliance and analysis

### Intelligent Agent Features

- **Multi-Algorithm Support**: DQN, PPO, SAC, A3C, Genetic Algorithms, Bayesian Optimization
- **Specialization**: Agents can specialize by vertical, budget range, campaign type
- **Creative Generation**: AI-powered creative variant generation
- **Dynamic Optimization**: Real-time targeting and budget optimization

### Enterprise Integration

- **Webhook Support**: Real-time notifications for key events
- **MCP Connectors**: Integration with major ad platforms (Facebook, Google, LinkedIn)
- **Batch Processing**: Efficient bulk operations for large-scale deployments
- **Custom Validation**: Configurable safety and validation rules

## Error Handling

The API uses standard HTTP status codes and provides detailed error responses:

```json
{
  "error": "Invalid campaign configuration",
  "code": "INVALID_REQUEST",
  "details": {
    "field": "targeting.demographics.age_range",
    "message": "Age range must be between 13 and 65"
  },
  "request_id": "req_123456789"
}
```

Common status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

## SDKs and Libraries

- **Python SDK**: `pip install gaelp-python`
- **Node.js SDK**: `npm install gaelp-node`
- **Go SDK**: Available via Go modules
- **CLI Tool**: `pip install gaelp-cli`

## Monitoring and Analytics

The API provides comprehensive monitoring through:

- **Performance Metrics**: Real-time campaign performance tracking
- **Agent Analytics**: Training progress and model performance
- **Cost Analysis**: Detailed spending breakdowns and projections
- **Safety Metrics**: Compliance scores and risk assessments

## Support and Resources

- **Documentation**: Full API reference at `/docs`
- **Status Page**: System status at `status.gaelp.dev`
- **Community**: Discord server for developers
- **Enterprise Support**: 24/7 support for enterprise customers

## Changelog

### v1.0.0 (Latest)
- Initial release with full feature set
- All five API modules available
- Comprehensive safety and budget controls
- Advanced simulation capabilities

See `/v1/changelog` for detailed version history.

---

*For complete API specifications, see the OpenAPI documents in this directory.*