# GAELP MCP Connectors Integration Guide

This guide explains how GAELP agents can use the MCP connectors to deploy and monitor real advertising campaigns.

## Overview

The MCP connectors provide a secure, standardized way for GAELP to:
- Deploy campaigns to Meta (Facebook/Instagram) and Google Ads
- Monitor campaign performance in real-time
- Manage creative assets and audience targeting
- Control spending and ensure compliance

## Integration Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GAELP Agent   │    │  MCP Connector  │    │  Ad Platform    │
│                 │◄──►│                 │◄──►│                 │
│ - Experiment    │    │ - Authentication│    │ - Meta Ads      │
│ - Optimize      │    │ - Rate Limiting │    │ - Google Ads    │
│ - Monitor       │    │ - Safety Checks │    │ - TikTok Ads    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Setup and Configuration

### 1. Initialize MCP Connection

```python
# Example GAELP agent code
import asyncio
from mcp_client import MCPClient

async def setup_advertising_connectors():
    # Connect to Meta Ads MCP
    meta_client = MCPClient('meta-ads-connector')
    await meta_client.connect()
    
    # Configure Meta credentials
    await meta_client.call_tool('meta_connect', {
        'appId': 'your-app-id',
        'appSecret': 'your-app-secret',
        'accessToken': 'your-access-token',
        'accountId': 'act_123456789',
        'businessAccountId': '123456789'
    })
    
    # Set spending limits for safety
    await meta_client.call_tool('meta_set_spending_limits', {
        'limits': {
            'dailyLimit': 100,
            'monthlyLimit': 3000,
            'campaignLimit': 500,
            'currency': 'USD',
            'alertThresholds': {
                'warning': 80,
                'critical': 95
            }
        }
    })
    
    # Connect to Google Ads MCP
    google_client = MCPClient('google-ads-connector')
    await google_client.connect()
    
    await google_client.call_tool('google_connect', {
        'customerId': '123-456-7890',
        'developerToken': 'your-developer-token',
        'accessToken': 'your-oauth-token',
        'refreshToken': 'your-refresh-token'
    })
    
    return meta_client, google_client
```

### 2. Campaign Deployment Workflow

```python
class CampaignDeploymentAgent:
    def __init__(self, meta_client, google_client):
        self.meta_client = meta_client
        self.google_client = google_client
    
    async def deploy_experiment(self, experiment_config):
        """Deploy an A/B test campaign across platforms"""
        
        # Step 1: Validate content against platform policies
        for creative in experiment_config['creatives']:
            meta_validation = await self.meta_client.call_tool('meta_validate_content', {
                'creative': creative
            })
            
            if not meta_validation['data']['valid']:
                raise ValueError(f"Meta policy violation: {meta_validation['data']['errors']}")
            
            google_validation = await self.google_client.call_tool('google_validate_content', {
                'creative': creative
            })
            
            if not google_validation['data']['valid']:
                raise ValueError(f"Google policy violation: {google_validation['data']['errors']}")
        
        # Step 2: Create campaigns on both platforms
        campaigns = {}
        
        # Deploy to Meta
        meta_campaign = await self.meta_client.call_tool('meta_create_campaign', {
            'campaign': {
                'name': f"GAELP_Experiment_{experiment_config['id']}_Meta",
                'objective': {'type': experiment_config['objective']},
                'budget': experiment_config['budget'],
                'targeting': experiment_config['targeting'],
                'creatives': experiment_config['creatives'],
                'status': 'PAUSED'  # Start paused for safety
            }
        })
        campaigns['meta'] = meta_campaign['data']['id']
        
        # Deploy to Google
        google_campaign = await self.google_client.call_tool('google_create_campaign', {
            'campaign': {
                'name': f"GAELP_Experiment_{experiment_config['id']}_Google",
                'objective': {'type': experiment_config['objective']},
                'budget': experiment_config['budget'],
                'targeting': experiment_config['targeting'],
                'creatives': experiment_config['creatives'],
                'status': 'PAUSED'
            }
        })
        campaigns['google'] = google_campaign['data']['id']
        
        # Step 3: Upload creatives
        for creative in experiment_config['creatives']:
            await self.meta_client.call_tool('meta_upload_creative', {
                'creative': creative
            })
            await self.google_client.call_tool('google_upload_creative', {
                'creative': creative
            })
        
        return campaigns
    
    async def start_campaigns(self, campaign_ids):
        """Start campaigns after final safety checks"""
        
        # Final spending check
        meta_spending = await self.meta_client.call_tool('meta_get_spending_status')
        if meta_spending['data']['alerts']:
            for alert in meta_spending['data']['alerts']:
                if alert['level'] == 'CRITICAL':
                    raise ValueError(f"Cannot start: {alert['message']}")
        
        # Start campaigns
        await self.meta_client.call_tool('meta_resume_campaign', {
            'campaignId': campaign_ids['meta']
        })
        
        await self.google_client.call_tool('google_resume_campaign', {
            'campaignId': campaign_ids['google']
        })
```

### 3. Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self, meta_client, google_client):
        self.meta_client = meta_client
        self.google_client = google_client
    
    async def collect_performance_data(self, campaign_ids, date_range):
        """Collect performance data from all platforms"""
        
        performance_data = {}
        
        # Get Meta performance
        meta_performance = await self.meta_client.call_tool('meta_get_campaign_performance', {
            'campaignId': campaign_ids['meta'],
            'dateRange': date_range,
            'metrics': ['impressions', 'clicks', 'spend', 'conversions', 'ctr', 'cpc']
        })
        performance_data['meta'] = meta_performance['data']
        
        # Get Google performance  
        google_performance = await self.google_client.call_tool('google_get_campaign_performance', {
            'campaignId': campaign_ids['google'],
            'dateRange': date_range,
            'metrics': ['impressions', 'clicks', 'cost_micros', 'conversions', 'ctr', 'average_cpc']
        })
        performance_data['google'] = google_performance['data']
        
        return performance_data
    
    async def optimize_campaigns(self, campaign_ids, performance_data):
        """Optimize campaigns based on performance"""
        
        # Example optimization logic
        for platform, data in performance_data.items():
            client = self.meta_client if platform == 'meta' else self.google_client
            campaign_id = campaign_ids[platform]
            
            # If CTR is low, try different creative
            if data['ctr'] < 1.0:  # Less than 1% CTR
                # Pause underperforming campaign
                await client.call_tool(f'{platform}_pause_campaign', {
                    'campaignId': campaign_id
                })
                
                # Create new campaign with different creative
                # (implementation depends on GAELP's optimization strategy)
            
            # If CPC is too high, adjust bidding
            if data['cpc'] > 2.0:  # More than $2 CPC
                await client.call_tool(f'{platform}_update_campaign', {
                    'campaignId': campaign_id,
                    'updates': {
                        'bidStrategy': {
                            'type': 'TARGET_CPA',
                            'amount': 10.0  # Target $10 CPA
                        }
                    }
                })
```

### 4. Safety and Compliance Integration

```python
class SafetyManager:
    def __init__(self, meta_client, google_client):
        self.meta_client = meta_client
        self.google_client = google_client
    
    async def continuous_monitoring(self):
        """Continuously monitor for safety violations"""
        
        while True:
            # Check spending status
            await self.check_spending_limits()
            
            # Check campaign performance for anomalies
            await self.check_performance_anomalies()
            
            # Wait before next check
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def check_spending_limits(self):
        """Check if spending is within limits"""
        
        for platform, client in [('meta', self.meta_client), ('google', self.google_client)]:
            spending_status = await client.call_tool(f'{platform}_get_spending_status')
            
            for alert in spending_status['data']['alerts']:
                if alert['level'] == 'CRITICAL':
                    # Emergency pause all campaigns
                    campaigns = await client.call_tool(f'{platform}_list_campaigns')
                    for campaign in campaigns['data']:
                        if campaign['status'] == 'ACTIVE':
                            await client.call_tool(f'{platform}_pause_campaign', {
                                'campaignId': campaign['id']
                            })
                    
                    # Alert GAELP safety system
                    await self.alert_safety_system(f"Critical spending alert: {alert['message']}")
    
    async def emergency_stop(self, reason):
        """Emergency stop all campaigns"""
        
        for platform, client in [('meta', self.meta_client), ('google', self.google_client)]:
            campaigns = await client.call_tool(f'{platform}_list_campaigns')
            
            for campaign in campaigns['data']:
                if campaign['status'] == 'ACTIVE':
                    await client.call_tool(f'{platform}_pause_campaign', {
                        'campaignId': campaign['id']
                    })
        
        await self.alert_safety_system(f"Emergency stop triggered: {reason}")
```

## Integration with GAELP Components

### Training Orchestrator Integration

```python
# In GAELP Training Orchestrator
class ExperimentRunner:
    async def run_ad_experiment(self, experiment_config):
        # Initialize MCP connectors
        meta_client, google_client = await setup_advertising_connectors()
        
        # Deploy campaigns
        deployer = CampaignDeploymentAgent(meta_client, google_client)
        campaign_ids = await deployer.deploy_experiment(experiment_config)
        
        # Start monitoring
        monitor = PerformanceMonitor(meta_client, google_client)
        safety_manager = SafetyManager(meta_client, google_client)
        
        # Start safety monitoring in background
        asyncio.create_task(safety_manager.continuous_monitoring())
        
        # Start campaigns
        await deployer.start_campaigns(campaign_ids)
        
        # Collect data over experiment duration
        performance_history = []
        for day in range(experiment_config['duration_days']):
            await asyncio.sleep(24 * 60 * 60)  # Wait 24 hours
            
            performance = await monitor.collect_performance_data(
                campaign_ids,
                {'start': f'2024-01-{day+1:02d}', 'end': f'2024-01-{day+1:02d}'}
            )
            performance_history.append(performance)
            
            # Optimize based on performance
            await monitor.optimize_campaigns(campaign_ids, performance)
        
        # Stop campaigns and return results
        await deployer.stop_campaigns(campaign_ids)
        return performance_history
```

### BigQuery Integration

```python
# Store performance data in BigQuery
class DataStorage:
    def __init__(self, bigquery_client):
        self.bigquery = bigquery_client
    
    async def store_performance_data(self, campaign_ids, performance_data):
        """Store performance data in BigQuery for analysis"""
        
        rows_to_insert = []
        
        for platform, data in performance_data.items():
            row = {
                'timestamp': datetime.utcnow().isoformat(),
                'platform': platform,
                'campaign_id': campaign_ids[platform],
                'impressions': data['impressions'],
                'clicks': data['clicks'],
                'spend': data['spend'],
                'conversions': data['conversions'],
                'ctr': data['ctr'],
                'cpc': data['cpc']
            }
            rows_to_insert.append(row)
        
        # Insert into BigQuery
        table_ref = self.bigquery.dataset('gaelp').table('campaign_performance')
        errors = self.bigquery.insert_rows_json(table_ref, rows_to_insert)
        
        if errors:
            raise ValueError(f"BigQuery insert failed: {errors}")
```

## Best Practices

### 1. Error Handling
- Always check MCP responses for success status
- Implement retry logic for transient failures
- Log all API interactions for debugging
- Have fallback mechanisms for critical operations

### 2. Security
- Store credentials securely (encrypted)
- Use least-privilege access tokens
- Rotate tokens regularly
- Monitor for unauthorized access

### 3. Cost Control
- Set conservative spending limits initially
- Monitor spending in real-time
- Implement automatic pause triggers
- Alert on unusual spending patterns

### 4. Performance Optimization
- Batch API calls where possible
- Cache frequently accessed data
- Use appropriate retry strategies
- Monitor connector health

### 5. Compliance
- Validate all content before deployment
- Keep audit logs of all changes
- Implement approval workflows for high-spend campaigns
- Regular compliance checks

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   ```python
   # Check token validity
   response = await client.call_tool('meta_test_connection')
   if not response['success']:
       # Refresh tokens or re-authenticate
   ```

2. **Rate Limiting**
   ```python
   # MCP connectors handle this automatically, but monitor for delays
   ```

3. **Campaign Approval Delays**
   ```python
   # Check campaign status regularly
   campaign = await client.call_tool('meta_get_campaign', {'campaignId': campaign_id})
   if campaign['data']['status'] == 'PENDING_REVIEW':
       # Wait for approval before collecting performance data
   ```

4. **Spending Alerts**
   ```python
   # Implement spending monitoring
   status = await client.call_tool('meta_get_spending_status')
   for alert in status['data']['alerts']:
       if alert['level'] == 'CRITICAL':
           # Take immediate action
   ```

## Performance Benchmarks

Expected performance for MCP connectors:

- **API Response Time**: < 2 seconds for most operations
- **Campaign Creation**: < 30 seconds end-to-end
- **Performance Data Retrieval**: < 5 seconds
- **Rate Limiting**: Automatically handled per platform limits
- **Uptime**: > 99.9% availability

## Monitoring and Alerting

Set up monitoring for:

- MCP connector health
- API response times
- Error rates
- Spending thresholds
- Campaign performance anomalies
- Compliance violations

This integration enables GAELP to safely and effectively use real advertising platforms for AI agent training and optimization while maintaining strict safety and compliance controls.