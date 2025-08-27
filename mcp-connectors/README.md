# GAELP MCP Connectors

This directory contains Model Context Protocol (MCP) connectors for advertising platforms that enable GAELP to deploy and monitor real ad campaigns.

## Overview

The MCP connectors provide a standardized interface for GAELP to interact with external advertising platforms. They handle authentication, rate limiting, error recovery, and provide safety controls for campaign management.

## Available Connectors

### 1. Meta Ads Connector (`meta-ads/`)
- **Platform**: Facebook & Instagram Ads
- **API Version**: v18.0 (configurable)
- **Features**:
  - Campaign creation, management, and monitoring
  - Creative upload and testing
  - Audience management
  - Performance analytics
  - Content policy validation
  - Spending controls

### 2. Google Ads Connector (`google-ads/`)
- **Platform**: Google Ads (Search, Display, Shopping)
- **API Version**: v14 (configurable)
- **Features**:
  - Campaign and ad group management
  - Responsive and text ad creation
  - User list (audience) management
  - Performance reporting
  - Budget controls
  - Policy compliance checking

## Architecture

```
mcp-connectors/
├── shared/
│   ├── types.ts           # Common interfaces and types
│   ├── utils.ts           # Shared utilities and helpers
│   └── base-connector.ts  # Base connector class
├── meta-ads/
│   ├── meta-connector.ts  # Meta Ads API integration
│   └── meta-mcp-server.ts # MCP server for Meta Ads
├── google-ads/
│   ├── google-connector.ts  # Google Ads API integration
│   └── google-mcp-server.ts # MCP server for Google Ads
└── tsconfig.json         # TypeScript configuration
```

## Key Features

### 🔐 Security & Authentication
- OAuth 2.0 and API key management
- Encrypted credential storage
- Token refresh handling
- Audit logging for all requests

### 🛡️ Safety Controls
- Spending limits and monitoring
- Real-time budget alerts
- Campaign approval workflows
- Content policy validation

### 🚦 Rate Limiting & Reliability
- Automatic rate limiting per platform
- Exponential backoff retry logic
- Circuit breaker patterns
- Health monitoring

### 📊 Performance Monitoring
- Real-time metrics collection
- Cost tracking and optimization
- A/B testing support
- Conversion attribution

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Build the Connectors
```bash
npm run build:mcp
```

### 3. Start Meta Ads MCP Server
```bash
npm run start:meta-mcp
```

### 4. Start Google Ads MCP Server
```bash
npm run start:google-mcp
```

## Configuration

### Meta Ads Configuration
```typescript
interface MetaConfig {
  appId: string;           // Meta App ID
  appSecret: string;       // Meta App Secret
  accessToken: string;     // Long-lived access token
  accountId: string;       // Ad Account ID
  businessAccountId: string; // Business Account ID
  rateLimitPerSecond?: number; // Default: 10
  apiVersion?: string;     // Default: v18.0
}
```

### Google Ads Configuration
```typescript
interface GoogleAdsConfig {
  customerId: string;      // Google Ads Customer ID
  developerToken: string;  // Developer token
  accessToken: string;     // OAuth access token
  refreshToken?: string;   // OAuth refresh token
  clientId?: string;       // OAuth client ID
  clientSecret?: string;   // OAuth client secret
  rateLimitPerSecond?: number; // Default: 10
  apiVersion?: string;     // Default: v14
}
```

## Available MCP Tools

### Connection Management
- `{platform}_connect` - Establish API connection
- `{platform}_test_connection` - Test connectivity
- `{platform}_health_check` - Get system health status

### Campaign Management
- `{platform}_create_campaign` - Create new campaigns
- `{platform}_update_campaign` - Modify existing campaigns
- `{platform}_get_campaign` - Retrieve campaign details
- `{platform}_list_campaigns` - List campaigns with filters
- `{platform}_pause_campaign` - Pause campaigns
- `{platform}_resume_campaign` - Resume campaigns
- `{platform}_delete_campaign` - Delete campaigns

### Creative Management
- `{platform}_upload_creative` - Upload ad creatives
- `{platform}_get_creative` - Retrieve creative details
- `{platform}_list_creatives` - List creatives
- `{platform}_delete_creative` - Delete creatives

### Performance Monitoring
- `{platform}_get_campaign_performance` - Get campaign metrics
- `{platform}_get_account_performance` - Get account-level metrics

### Audience Management
- `{platform}_create_audience` - Create custom audiences
- `{platform}_get_audience` - Retrieve audience details
- `{platform}_list_audiences` - List audiences
- `{platform}_delete_audience` - Delete audiences

### Safety & Compliance
- `{platform}_validate_content` - Validate ad content
- `{platform}_set_spending_limits` - Configure spending controls
- `{platform}_get_spending_status` - Check spending status

## Usage Examples

### Creating a Campaign
```typescript
// Connect to Meta Ads
await callTool('meta_connect', {
  appId: 'your-app-id',
  appSecret: 'your-app-secret',
  accessToken: 'your-access-token',
  accountId: 'act_123456789',
  businessAccountId: '123456789'
});

// Create campaign
await callTool('meta_create_campaign', {
  campaign: {
    name: 'GAELP Test Campaign',
    objective: { type: 'TRAFFIC' },
    budget: {
      amount: 50,
      currency: 'USD',
      type: 'DAILY'
    },
    status: 'PAUSED'
  }
});
```

### Monitoring Performance
```typescript
// Get campaign performance
await callTool('meta_get_campaign_performance', {
  campaignId: '123456789',
  dateRange: {
    start: '2024-01-01',
    end: '2024-01-07'
  },
  metrics: ['impressions', 'clicks', 'spend', 'conversions']
});
```

### Setting Spending Limits
```typescript
// Configure spending controls
await callTool('meta_set_spending_limits', {
  limits: {
    dailyLimit: 100,
    monthlyLimit: 3000,
    campaignLimit: 500,
    currency: 'USD',
    alertThresholds: {
      warning: 80,
      critical: 95
    }
  }
});
```

## Error Handling

The connectors implement comprehensive error handling:

- **Rate Limiting**: Automatic backoff and retry
- **Authentication**: Token refresh and re-authentication
- **API Errors**: Structured error responses with codes
- **Network Issues**: Connection timeouts and retries
- **Validation**: Input validation and sanitization

## Security Considerations

1. **Credential Management**: Never log or expose API keys
2. **Rate Limiting**: Respect platform API limits
3. **Spending Controls**: Always verify budget limits
4. **Content Validation**: Check ads against platform policies
5. **Audit Logging**: Track all API operations

## Development

### Running in Development Mode
```bash
# Meta Ads MCP Server
npm run dev:meta

# Google Ads MCP Server
npm run dev:google
```

### Building for Production
```bash
npm run build:mcp
```

### Testing Connections
```bash
# Test Meta Ads connection
echo '{"method": "tools/call", "params": {"name": "meta_test_connection", "arguments": {}}}' | npm run start:meta-mcp

# Test Google Ads connection
echo '{"method": "tools/call", "params": {"name": "google_test_connection", "arguments": {}}}' | npm run start:google-mcp
```

## Integration with GAELP

The MCP connectors integrate with GAELP components:

- **Training Orchestrator**: Campaign deployment and monitoring
- **Safety & Policy**: Budget controls and content validation
- **BigQuery Storage**: Performance data integration
- **Benchmark Portal**: Service configuration and monitoring

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify API keys and tokens
   - Check token expiration
   - Ensure proper scopes/permissions

2. **Rate Limiting**
   - Reduce request frequency
   - Check rate limit configuration
   - Monitor API quotas

3. **Campaign Creation Failures**
   - Validate campaign parameters
   - Check account permissions
   - Verify budget settings

4. **Performance Data Missing**
   - Allow time for data processing
   - Check date ranges
   - Verify campaign is active

### Logging

All connectors use structured logging with different levels:
- `INFO`: Normal operations
- `WARN`: Non-critical issues
- `ERROR`: Failures and exceptions
- `DEBUG`: Detailed debugging (enable with `DEBUG=true`)

## License

MIT License - See LICENSE file for details.