# GAELP LLM Integration Guide

## Overview

This guide covers the integration of real LLM APIs (Claude, GPT) for authentic user persona simulation in GAELP (Google Ad Experiment Learning Platform). The integration replaces mock personas with LLM-powered personas that provide realistic, consistent user behavior simulation.

## Architecture

### Core Components

1. **LLM Persona Service** (`llm_persona_service.py`)
   - Main service orchestrating LLM interactions
   - Handles multiple LLM providers (Anthropic Claude, OpenAI GPT)
   - Manages rate limiting, cost tracking, and caching
   - Provides fallback mechanisms for reliability

2. **Persona Factory** (`persona_factory.py`)
   - Generates diverse, realistic user personas
   - Creates detailed demographic and psychological profiles
   - Provides templates for common user types
   - Supports targeted persona creation for specific campaigns

3. **Integration Layer** (`run_full_demo_llm.py`)
   - Updated demo showcasing LLM-powered personas
   - Maintains backward compatibility with existing training pipeline
   - Provides rich logging and analytics for LLM interactions

### Key Features

- **Authentic User Simulation**: Real LLM responses based on detailed persona profiles
- **Dynamic State Management**: Personas evolve over time (fatigue, engagement changes)
- **Cost Control**: Built-in rate limiting and budget management
- **Reliability**: Automatic fallback to heuristic responses if LLM unavailable
- **Multi-Provider Support**: Primary/fallback provider configuration
- **Comprehensive Analytics**: Detailed tracking of persona behavior and LLM usage

## Setup Instructions

### 1. Environment Setup

Run the automated setup script:

```bash
./setup_llm_environment.sh
```

This will:
- Install required dependencies
- Start Redis (if Docker available)
- Create configuration templates
- Run basic connectivity tests

### 2. API Key Configuration

Edit the `.env` file created by the setup script:

```bash
# Required: At least one API key
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Cost and rate limiting
MAX_DAILY_COST=50.0
REQUESTS_PER_MINUTE=60
REQUESTS_PER_HOUR=1000
```

#### Getting API Keys

- **Anthropic Claude**: Visit [console.anthropic.com](https://console.anthropic.com/)
- **OpenAI GPT**: Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 3. Redis Setup

Redis is required for persona state management and caching:

**Option 1: Docker (Recommended)**
```bash
docker run -d --name gaelp-redis -p 6379:6379 redis:alpine
```

**Option 2: Local Installation**
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
redis-server
```

### 4. Initial Configuration

Run the interactive setup:

```bash
python3 llm_integration_setup.py
```

This will:
- Test API connectivity
- Create test personas
- Validate LLM responses
- Generate configuration files

## Usage Examples

### Basic Demo

Run the full LLM-powered demo:

```bash
python3 run_full_demo_llm.py
```

### Quick Test

Test the integration without running full training:

```bash
python3 llm_integration_setup.py quick
```

### Programmatic Usage

```python
from llm_persona_service import LLMPersonaService, LLMPersonaConfig
from persona_factory import PersonaTemplates

# Initialize service
config = LLMPersonaConfig(
    primary_provider="anthropic",
    anthropic_api_key="your_key",
    max_daily_cost=20.0
)
service = LLMPersonaService(config)

# Create personas
persona = PersonaTemplates.tech_early_adopter()
await service.create_persona(persona)

# Test campaign
campaign = {
    "creative_type": "video",
    "message": "Revolutionary new product!",
    "category": "technology",
    "budget": 25.0
}

# Get response
response = await service.respond_to_ad(persona.persona_id, campaign)
print(f"Engagement: {response['engagement_score']}")
print(f"Reasoning: {response['reasoning']}")
```

## Persona System

### Persona Components

Each persona consists of:

1. **Demographics**: Age, gender, income, education, location, employment
2. **Psychology**: Personality traits (Big 5), values, interests, shopping behavior
3. **State**: Current engagement level, fatigue, interaction history

### Persona States

- **Fresh**: New to seeing ads, curious and open
- **Engaged**: Actively interested in relevant ads
- **Fatigued**: Tired from seeing too many ads
- **Blocked**: Actively avoiding advertisements
- **Recovered**: Recovered from fatigue, selective engagement

### Persona Templates

Pre-built persona types for quick testing:

```python
from persona_factory import PersonaTemplates

# Individual templates
tech_adopter = PersonaTemplates.tech_early_adopter()
family_budget = PersonaTemplates.budget_conscious_family()
luxury_consumer = PersonaTemplates.luxury_consumer()
student = PersonaTemplates.student()
retiree = PersonaTemplates.retiree()

# Diverse test cohort
cohort = PersonaTemplates.get_diverse_test_cohort()
```

## LLM Providers

### Anthropic Claude

- **Models**: claude-3-sonnet-20240229 (default), claude-3-opus-20240229
- **Strengths**: Excellent reasoning, nuanced responses, safety
- **Cost**: ~$15-30 per million tokens
- **Rate Limits**: Varies by tier

### OpenAI GPT

- **Models**: gpt-4 (default), gpt-3.5-turbo
- **Strengths**: Fast responses, good creativity, wide knowledge
- **Cost**: ~$30-60 per million tokens for GPT-4
- **Rate Limits**: Varies by tier

### Provider Selection

The service automatically handles provider selection:

1. Primary provider attempts first
2. Fallback provider if primary fails
3. Heuristic fallback if all LLMs fail

## Cost Management

### Built-in Controls

- **Daily Limits**: Configurable maximum daily spend
- **Rate Limiting**: Requests per minute/hour/day limits
- **Caching**: Reduces redundant API calls
- **Efficient Prompting**: Optimized prompts to minimize token usage

### Cost Estimation

Typical costs for different usage levels:

- **Development/Testing**: $1-5 per day
- **Small-scale training**: $10-20 per day
- **Production training**: $50-200 per day

### Monitoring

Track costs in real-time:

```python
# Get cost analytics
analytics = await service.get_service_analytics()
print(f"Today's cost: ${analytics['daily_cost']}")
```

## Integration Points

### Training Orchestrator

The LLM personas integrate seamlessly with the existing training orchestrator:

```python
# In training_orchestrator/core.py
async def _run_simulation_phase(self, agent, environments):
    # Uses LLM personas automatically if configured
    simulation_env = environments.get("simulation")
    for persona in simulation_env["personas"]:
        response = await persona.respond_to_ad(campaign)
        # Process response normally
```

### Safety Framework

LLM responses are subject to existing safety checks:

- Content safety validation
- Brand safety compliance
- Budget limit enforcement
- Anomaly detection

### Benchmark Portal

Persona analytics integrate with the benchmark system:

- Response authenticity metrics
- Persona state evolution tracking
- Cross-campaign behavior analysis

## Monitoring and Analytics

### Service Health

Monitor service health:

```python
health = await service.health_check()
print(f"Service: {health['service_status']}")
print(f"Providers: {health['providers']}")
print(f"Redis: {health['redis_status']}")
```

### Persona Analytics

Track individual persona behavior:

```python
analytics = await service.get_persona_analytics(persona_id)
print(f"Interactions: {analytics['total_interactions']}")
print(f"CTR: {analytics['ctr']:.3f}")
print(f"Current state: {analytics['current_state']}")
```

### Campaign Analytics

Analyze campaign performance across personas:

```python
results = await manager.run_test_campaign(campaign_config)
print(f"Overall CTR: {results['ctr']:.3f}")
print(f"ROAS: {results['roas']:.2f}x")
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```
   Error: "Authentication failed"
   Solution: Check API keys in .env file
   ```

2. **Redis Connection**
   ```
   Error: "Redis connection failed"
   Solution: Ensure Redis is running on localhost:6379
   ```

3. **Rate Limiting**
   ```
   Error: "Rate limit exceeded"
   Solution: Reduce requests_per_minute in config
   ```

4. **High Costs**
   ```
   Error: "Daily cost limit exceeded"
   Solution: Increase max_daily_cost or optimize usage
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python3 run_full_demo_llm.py
```

### Fallback Testing

Test fallback behavior by temporarily removing API keys:

```bash
unset ANTHROPIC_API_KEY
unset OPENAI_API_KEY
python3 run_full_demo_llm.py
```

## Performance Optimization

### Caching Strategy

- **Response Caching**: Similar campaigns cached for 5 minutes
- **Persona Caching**: Persona state cached for 24 hours
- **Health Check Caching**: Provider health cached for 1 minute

### Rate Optimization

- **Batch Requests**: Group similar persona responses
- **Async Processing**: Non-blocking API calls
- **Smart Retries**: Exponential backoff for failures

### Cost Optimization

- **Prompt Engineering**: Minimal but effective prompts
- **Response Format**: Structured JSON to reduce tokens
- **Provider Selection**: Cost-based provider routing

## Production Deployment

### Environment Configuration

```bash
# Production settings
MAX_DAILY_COST=200.0
REQUESTS_PER_MINUTE=100
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

### Scaling Considerations

- **Redis Clustering**: For high-volume deployments
- **Load Balancing**: Multiple service instances
- **Provider Redundancy**: Multiple API keys per provider

### Security

- **API Key Rotation**: Regular key rotation schedule
- **Network Security**: VPC/firewall protection
- **Audit Logging**: Comprehensive request logging

## Migration Guide

### From Mock Personas

1. Install LLM dependencies
2. Configure API keys
3. Update imports in training code
4. Test with small persona cohort
5. Gradually scale up usage

### Backward Compatibility

The integration maintains full backward compatibility:

- Existing training code works unchanged
- Mock personas available as fallback
- Same response format and metrics

## Support and Resources

### Documentation

- API Documentation: See docstrings in service files
- Examples: Check `llm_integration_setup.py`
- Tests: Run `python3 -m pytest tests/`

### Community

- Issues: Report on GAELP repository
- Discussions: Use GitHub Discussions
- Updates: Follow release notes

### Professional Support

For enterprise deployments:
- Custom persona development
- Optimization consulting
- 24/7 support options