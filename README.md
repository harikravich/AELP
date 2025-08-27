# GAELP Training Orchestrator

A comprehensive training orchestrator for ad campaign agents that manages the complete simulation-to-real-world learning progression.

## Overview

The Training Orchestrator is the core component of GAELP (Generative Autonomous Expert Learning Platform) that coordinates the interaction between agents and environments during training. It implements a four-phase progression system that safely transitions ad campaign agents from simulation training to real-world deployment.

## Four-Phase Training Pipeline

### Phase 1: Simulation Training
- Agent learns on LLM user persona responses
- High-volume, low-cost experimentation
- Basic ad optimization (creative, targeting, budget allocation)
- Curriculum learning (simple to complex scenarios)

### Phase 2: Historical Data Validation
- Test agent on real historical campaign datasets
- Validate simulation learnings against actual performance
- Identify gaps between simulation and reality

### Phase 3: Small Budget Real Testing
- Deploy agent with strict budget limits ($10-50/day)
- Real ad platform integration via MCP connectors
- Continuous safety monitoring
- Performance comparison with baseline campaigns

### Phase 4: Scaled Deployment
- Increase budgets based on performance thresholds
- Multi-campaign management
- Advanced optimization strategies
- Transfer learning across different products/audiences

## Key Features

### Episode Management
- Reset environments, run campaigns, collect rewards
- Comprehensive logging and state tracking
- Batch episode execution with concurrency control
- Error handling and recovery mechanisms

### Multi-Environment Coordination
- Simulation environments with LLM personas
- Historical validation environments
- Real ad platform environments
- Seamless transitions between phases

### Curriculum Scheduling
- Progressive difficulty scaling
- Task-based learning progression
- Performance-driven curriculum adaptation
- Prerequisites and graduation criteria

### Performance Monitoring
- Real-time performance tracking
- Trend analysis and projections
- Graduation criteria assessment
- Anomaly detection

### Safety Integration
- Budget controls and limits
- Content and brand safety validation
- Human approval workflows
- Violation tracking and alerts

### Checkpoint Management
- Save/restore agent state
- Experiment reproducibility
- Recovery from failures
- Version control integration

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AELP

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## Quick Start

```python
import asyncio
from training_orchestrator import TrainingOrchestrator
from training_orchestrator.config import DEVELOPMENT_CONFIG
from training_orchestrator.core import TrainingConfiguration

# Create configuration
config = TrainingConfiguration(**DEVELOPMENT_CONFIG.to_legacy_config())

# Initialize orchestrator
orchestrator = TrainingOrchestrator(config)

# Create your agent and environments
agent = YourAdCampaignAgent()
environments = {
    "simulation": YourSimulationEnvironment(),
    "historical": YourHistoricalEnvironment(),
    "real": YourRealEnvironment(),
    "scaled": YourScaledEnvironment()
}

# Start training
async def main():
    success = await orchestrator.start_training(agent, environments)
    if success:
        print("Training completed successfully!")
        metrics = orchestrator.get_metrics()
        print(f"Final metrics: {metrics}")

# Run training
asyncio.run(main())
```

## Configuration

The system supports multiple configuration modes:

### Development Configuration
```python
from training_orchestrator.config import DEVELOPMENT_CONFIG

# Smaller scale, local services, debug logging
config = DEVELOPMENT_CONFIG
```

### Production Configuration
```python
from training_orchestrator.config import PRODUCTION_CONFIG

# Full scale, production services, strict safety
config = PRODUCTION_CONFIG
```

### Environment Variables
```bash
# Database
export BIGQUERY_PROJECT="your-project"
export REDIS_HOST="your-redis-host"

# Budget limits
export REAL_TESTING_BUDGET_LIMIT="50.0"
export MAX_DAILY_BUDGET="1000.0"

# Safety
export REQUIRE_HUMAN_APPROVAL="true"
export LOG_LEVEL="INFO"
```

## Core Components

### TrainingOrchestrator
Main orchestrator class that coordinates the entire training process.

```python
orchestrator = TrainingOrchestrator(config)
await orchestrator.start_training(agent, environments)
```

### PhaseManager
Manages transitions between training phases based on graduation criteria.

```python
phase_manager = orchestrator.phase_manager
can_graduate, reason = phase_manager.can_graduate_phase(phase, metrics, episode_count)
```

### EpisodeManager
Handles execution of individual training episodes with comprehensive logging.

```python
episode_manager = orchestrator.episode_manager
result = await episode_manager.run_episode(agent, environment, "episode_1")
```

### CurriculumScheduler
Manages progressive difficulty scaling and task scheduling.

```python
curriculum = orchestrator.curriculum_scheduler
task = curriculum.get_current_task(phase)
```

### PerformanceMonitor
Tracks performance metrics and analyzes trends for graduation decisions.

```python
monitor = orchestrator.performance_monitor
assessment = monitor.get_graduation_assessment(phase)
```

### SafetyMonitor
Enforces safety constraints, especially during real-world testing.

```python
safety = orchestrator.safety_monitor
is_safe = await safety.check_safety_constraints(episode_result, phase)
```

## Agent Interface

Your agent must implement the following interface:

```python
class YourAgent:
    async def select_action(self, observation):
        """Select action based on observation"""
        return action_dict
    
    def get_state(self):
        """Get agent state for checkpointing"""
        return state_dict
    
    def load_state(self, state):
        """Load agent state from checkpoint"""
        pass
```

## Environment Interface

Your environments must implement:

```python
class YourEnvironment:
    async def reset(self):
        """Reset environment and return initial observation"""
        return observation_dict
    
    async def step(self, action):
        """Execute action and return (observation, reward, done, info)"""
        return next_obs, reward, done, info_dict
```

## Graduation Criteria

### Simulation Phase
- Minimum average reward: 0.7
- Success rate: 80%
- Performance improvement rate: 5%
- Consistency over 50 episodes

### Historical Validation
- Historical performance match: 95%
- Prediction accuracy: 85%
- Correlation with actual: 80%

### Real Testing
- 5 consecutive positive ROI campaigns
- Minimum 5% ROI threshold
- Budget efficiency: 80%
- Zero safety violations

### Scaled Deployment
- 15% sustained performance improvement
- 90% multi-campaign success rate
- 80% transfer learning effectiveness

## Safety Features

### Budget Controls
- Daily and per-episode limits
- Real-time spending tracking
- Emergency stop mechanisms
- Alert thresholds

### Content Safety
- Content quality scoring
- Brand safety validation
- Forbidden keyword detection
- Human approval workflows

### Anomaly Detection
- Performance anomaly detection
- Statistical outlier identification
- Automated intervention triggers
- Alert notifications

## Integration Points

### BigQuery Storage
- Episode data logging
- Performance metrics storage
- Historical analysis
- Compliance reporting

### Safety & Policy Engine
- Content validation
- Policy compliance
- Risk assessment
- Approval workflows

### Agent Manager
- Job lifecycle management
- Resource allocation
- Scaling decisions
- Health monitoring

### Environment Registry
- Environment discovery
- Version management
- Configuration distribution
- Health checks

## Monitoring and Observability

### Metrics Collection
- Episode-level metrics
- Phase progression tracking
- Performance trends
- Safety compliance

### Logging
- Structured logging with levels
- Correlation IDs
- Performance profiling
- Error tracking

### Alerting
- Safety violations
- Performance degradation
- Budget thresholds
- System failures

## Examples

See `example_training_run.py` for complete examples including:
- Full training pipeline execution
- Phase-specific analysis
- Mock agent and environment implementations
- Configuration examples

## Best Practices

1. **Reproducibility**: Always set random seeds and log all configurations
2. **Safety First**: Implement comprehensive safety checks before real testing
3. **Gradual Scaling**: Respect the four-phase progression and graduation criteria
4. **Monitoring**: Implement comprehensive logging and alerting
5. **Testing**: Test thoroughly in simulation before real deployment

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black training_orchestrator/
isort training_orchestrator/

# Type checking
mypy training_orchestrator/

# Run example
python example_training_run.py
```

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure safety considerations are addressed
5. Test thoroughly in development environment

## License

[Your License Here]

## Support

For questions or issues, please [contact the team] or [create an issue].