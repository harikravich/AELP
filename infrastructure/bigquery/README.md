# GAELP BigQuery Data Architecture

This directory contains the complete BigQuery data architecture for the GAELP (Google Ad Environment Learning Platform) system, designed to store, process, and analyze ad campaign learning data at scale.

## Architecture Overview

The GAELP BigQuery infrastructure supports both simulated and real campaign data for training reinforcement learning agents on ad campaign optimization. The system is designed for:

- **Scalability**: Handle millions of campaign interactions
- **Real-time Processing**: Stream campaign data and agent actions
- **Cost Optimization**: Intelligent partitioning and materialized views
- **Data Quality**: Comprehensive validation and monitoring
- **Analytics**: Rich insights for agent performance and campaign effectiveness

## Dataset Structure

### Core Tables

#### `campaigns`
- **Purpose**: Central registry of all campaigns (simulated and real)
- **Key Features**: Agent assignments, configurations, platform integration
- **Partitioning**: By creation date
- **Clustering**: By agent_id, campaign_type, status

#### `performance_metrics`
- **Purpose**: Time-series performance data (impressions, clicks, conversions, spend)
- **Key Features**: Multiple granularities (hourly/daily), actual vs simulated data
- **Partitioning**: By timestamp date
- **Clustering**: By campaign_id, granularity, metric_type

#### `agent_episodes`
- **Purpose**: Detailed RL training data (states, actions, rewards)
- **Key Features**: Complete episode sequences for model training
- **Partitioning**: By timestamp date
- **Clustering**: By campaign_id, agent_id, episode_number

#### `simulation_data`
- **Purpose**: LLM persona responses and simulated user behavior
- **Key Features**: Creative evaluation, targeting tests, A/B testing
- **Partitioning**: By simulation timestamp
- **Clustering**: By campaign_id, persona_id, interaction_type

#### `personas`
- **Purpose**: User personas for simulation-based testing
- **Key Features**: Demographics, psychographics, behavioral patterns
- **Clustering**: By active status

#### `safety_events`
- **Purpose**: Safety monitoring and compliance tracking
- **Key Features**: Automated detection, severity classification, resolution tracking
- **Partitioning**: By detection date
- **Clustering**: By severity, event_type, campaign_id

### Analytics Views

#### `campaign_performance_summary`
- Aggregated campaign metrics with performance grades
- Success rates and efficiency calculations
- Time-based performance analysis

#### `agent_performance_comparison`
- Agent rankings and performance tiers
- Success rates and risk assessments
- Trend analysis and benchmarking

#### `simulation_vs_real_performance`
- Prediction accuracy evaluation
- Sim-to-real transfer analysis
- Model validation metrics

#### `safety_monitoring_dashboard`
- Real-time safety event tracking
- Trend analysis and unresolved issues
- Financial impact assessment

#### `learning_progress`
- Agent training progression
- Reward trend analysis
- Convergence indicators

### Optimization Features

#### Materialized Views
- `daily_campaign_performance`: Pre-aggregated daily metrics
- `agent_leaderboard`: Real-time agent rankings
- Automatic refresh and cost optimization

#### Streaming Infrastructure
- Real-time data ingestion from campaign platforms
- Stream processing for agent actions
- Data validation and quality monitoring

#### Cost Management
- Intelligent partitioning and clustering
- Data lifecycle management
- Query cost tracking and optimization
- Budget monitoring and alerting

## Deployment

### Prerequisites

1. **Google Cloud SDK**: Install and authenticate
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. **Permissions**: Ensure you have the following roles:
- BigQuery Admin
- Storage Admin
- Cloud Scheduler Admin
- IAM Admin

### Quick Start

1. **Set Environment Variables**:
```bash
export GAELP_PROJECT_ID="your-project-id"
export GAELP_REGION="us-central1"
```

2. **Deploy Full Infrastructure**:
```bash
./deploy.sh
```

3. **Validate Deployment**:
```bash
./deploy.sh validate
```

### Deployment Options

```bash
# Full deployment
./deploy.sh deploy

# Deploy only schemas
./deploy.sh schemas

# Deploy only views
./deploy.sh views

# Deploy only functions
./deploy.sh functions

# Setup monitoring
./deploy.sh monitoring

# Validate deployment
./deploy.sh validate

# Clean up (DESTRUCTIVE)
./deploy.sh clean
```

### Configuration

The deployment generates a configuration file `gaelp_bigquery_config.json` with all connection details and settings.

## Data Flow

### 1. Real-time Ingestion
```
Campaign Platforms → Pub/Sub → Dataflow → BigQuery Streaming
```

### 2. Agent Training
```
Agent Actions → Cloud Functions → BigQuery → ML Training Pipeline
```

### 3. Simulation
```
LLM Personas → Simulation Engine → BigQuery → Analytics
```

### 4. Monitoring
```
BigQuery → Cloud Monitoring → Alerting → Auto-remediation
```

## Key Features

### Real-time Streaming
- Campaign performance metrics ingestion
- Agent action tracking
- Data validation and quality monitoring
- Automatic error handling and retry logic

### Data Quality
- Comprehensive validation rules
- Anomaly detection for campaigns and agents
- Data lineage tracking
- Quality score calculation

### Cost Optimization
- Partition pruning for efficient queries
- Clustered tables for reduced scan costs
- Materialized views for expensive aggregations
- Query cost tracking and budget alerts

### Safety & Compliance
- Automated safety event detection
- Budget and performance threshold monitoring
- Compliance tracking and audit logging
- Automated remediation actions

### Analytics & Insights
- Real-time campaign health monitoring
- Agent performance benchmarking
- Simulation vs real-world analysis
- Learning progress tracking

## Monitoring & Alerting

### Real-time Dashboards
- System health overview
- Campaign performance monitoring
- Agent anomaly detection
- Cost and budget tracking

### Automated Alerts
- Performance degradation
- Budget threshold violations
- Data quality issues
- System health problems

### Alert Configuration
Customize alerts through the `alert_subscriptions` table:
- Email, Slack, or webhook delivery
- Severity-based filtering
- Agent/campaign-specific monitoring
- Quiet hours and throttling

## Best Practices

### Query Optimization
1. Always use partition filters
2. Leverage clustering columns
3. Use materialized views for repeated calculations
4. Monitor query costs and optimize expensive queries

### Data Management
1. Implement proper data retention policies
2. Use streaming inserts for real-time data
3. Validate data quality at ingestion
4. Monitor storage costs and optimize

### Security
1. Use service accounts with minimal permissions
2. Implement column-level security
3. Audit data access and modifications
4. Encrypt sensitive data

### Performance
1. Design queries for your access patterns
2. Use appropriate data types
3. Implement proper indexing strategies
4. Monitor and optimize slow queries

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Verify service account permissions
   - Check IAM roles assignment
   - Ensure APIs are enabled

2. **Data Quality Issues**
   - Check validation logs in `data_quality_log`
   - Review streaming table error rates
   - Validate source data formats

3. **Performance Problems**
   - Monitor query costs in `query_cost_tracking`
   - Check partition filtering usage
   - Optimize clustering strategies

4. **Cost Overruns**
   - Review budget alerts in `cost_budgets`
   - Analyze expensive queries
   - Implement data archival policies

### Support

For technical support and questions:
1. Check the monitoring dashboard for system status
2. Review safety events for automatic alerts
3. Analyze query performance metrics
4. Contact the GAELP team for assistance

## Contributing

When adding new features or modifications:
1. Update schema documentation
2. Add appropriate validation rules
3. Implement monitoring and alerting
4. Update deployment scripts
5. Test thoroughly in development environment

## License

This infrastructure is part of the GAELP system and subject to the project's license terms.