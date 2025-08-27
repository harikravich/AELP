"""
Example Training Run

Demonstrates how to use the Training Orchestrator for ad campaign
agent training with the four-phase progression.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Training orchestrator imports
from training_orchestrator import TrainingOrchestrator
from training_orchestrator.config import TrainingOrchestratorConfig, DEVELOPMENT_CONFIG
from training_orchestrator.core import TrainingConfiguration


# Mock Agent class for demonstration
class MockAdCampaignAgent:
    """Mock agent for demonstration purposes"""
    
    def __init__(self, name: str = "MockAgent"):
        self.name = name
        self.state = {"episode_count": 0, "total_reward": 0.0}
        
    async def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select an action based on observation"""
        
        # Mock action selection - in reality this would be ML-based
        action = {
            "creative_id": "creative_001",
            "target_audience": ["tech_enthusiasts", "age_25_45"],
            "budget_allocation": min(100.0, observation.get("available_budget", 50.0)),
            "bid_amount": 2.50,
            "campaign_duration": 3  # days
        }
        
        return action
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing"""
        return self.state.copy()
    
    def load_state(self, state: Dict[str, Any]):
        """Load agent state from checkpoint"""
        self.state = state.copy()


# Mock Environment classes for demonstration
class MockSimulationEnvironment:
    """Mock simulation environment using LLM personas"""
    
    def __init__(self):
        self.episode_count = 0
        self.personas = ["budget_conscious", "tech_savvy", "brand_focused", "performance_driven"]
        
    async def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode"""
        self.episode_count += 1
        
        return {
            "available_budget": 100.0,
            "target_market": "technology",
            "competition_level": 0.5,
            "persona_type": self.personas[self.episode_count % len(self.personas)],
            "market_conditions": "normal"
        }
    
    async def step(self, action: Dict[str, Any]) -> tuple:
        """Execute action and return results"""
        
        # Mock environment response
        reward = self._calculate_reward(action)
        done = True  # Each episode is one campaign
        
        info = {
            "budget_spent": action.get("budget_allocation", 0.0),
            "click_through_rate": 0.03 + (reward * 0.02),
            "conversion_rate": 0.02 + (reward * 0.01),
            "content_safety_score": 0.95,
            "brand_safety_score": 0.90,
            "revenue": action.get("budget_allocation", 0.0) * (1.0 + reward),
            "goals_achieved": 3 if reward > 0.5 else 1,
            "total_goals": 3
        }
        
        next_observation = await self.reset() if done else {}
        
        return next_observation, reward, done, info
    
    def _calculate_reward(self, action: Dict[str, Any]) -> float:
        """Calculate reward based on action quality"""
        
        # Mock reward calculation
        base_reward = 0.3
        
        # Reward good budget allocation
        budget = action.get("budget_allocation", 0.0)
        if 50.0 <= budget <= 100.0:
            base_reward += 0.2
        
        # Reward appropriate bid amount
        bid = action.get("bid_amount", 0.0)
        if 1.0 <= bid <= 5.0:
            base_reward += 0.2
        
        # Reward good targeting
        targeting = action.get("target_audience", [])
        if len(targeting) >= 1:
            base_reward += 0.3
        
        return min(1.0, base_reward)


class MockHistoricalEnvironment:
    """Mock historical validation environment"""
    
    def __init__(self):
        self.episode_count = 0
        self.historical_benchmarks = [0.75, 0.82, 0.68, 0.91, 0.77]
        
    async def reset(self) -> Dict[str, Any]:
        """Reset for historical validation episode"""
        self.episode_count += 1
        
        return {
            "historical_campaign_data": {
                "budget": 200.0,
                "duration": 7,
                "target_audience_size": 50000,
                "expected_performance": self.historical_benchmarks[self.episode_count % len(self.historical_benchmarks)]
            },
            "validation_mode": True
        }
    
    async def step(self, action: Dict[str, Any]) -> tuple:
        """Validate action against historical data"""
        
        historical_data = action.get("historical_campaign_data", {})
        expected_performance = historical_data.get("expected_performance", 0.7)
        
        # Compare agent's approach to historical success
        performance_match = min(1.0, abs(expected_performance - 0.1))  # Mock validation
        
        reward = performance_match
        done = True
        
        info = {
            "historical_performance_match": performance_match,
            "prediction_accuracy": 0.85 + (performance_match * 0.1),
            "correlation_with_actual": 0.8 + (performance_match * 0.15),
            "validation_data": {
                "historical_performance_match": performance_match,
                "prediction_accuracy": 0.85 + (performance_match * 0.1),
                "correlation_with_actual": 0.8 + (performance_match * 0.15)
            }
        }
        
        return {}, reward, done, info


class MockRealEnvironment:
    """Mock real testing environment (simulates real ad platforms)"""
    
    def __init__(self):
        self.episode_count = 0
        
    async def reset(self) -> Dict[str, Any]:
        """Reset for real campaign episode"""
        self.episode_count += 1
        
        return {
            "platform": "google_ads",
            "real_budget": 10.0,  # Small budget for testing
            "safety_monitoring": True,
            "human_approval": True  # Required for real campaigns
        }
    
    async def step(self, action: Dict[str, Any]) -> tuple:
        """Execute real campaign (mock)"""
        
        # Mock real campaign execution
        budget_spent = min(action.get("budget_allocation", 0.0), 10.0)
        
        # Mock ROI calculation
        roi = 0.05 + (0.1 * (1.0 if budget_spent > 0 else 0.0))  # 5-15% ROI
        revenue = budget_spent * (1.0 + roi)
        
        reward = roi
        done = True
        
        info = {
            "budget_spent": budget_spent,
            "revenue": revenue,
            "roi": roi,
            "click_through_rate": 0.025,
            "conversion_rate": 0.015,
            "content_safety_score": 0.98,
            "brand_safety_score": 0.95,
            "human_approval": True,
            "safety_compliance": 1.0,
            "budget_efficiency": 0.85
        }
        
        return {}, reward, done, info


async def run_training_example():
    """Run a complete training example"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GAELP Training Orchestrator Example")
    
    # Load configuration
    config = DEVELOPMENT_CONFIG
    logger.info(f"Using configuration: {config.experiment_name} in {config.environment} mode")
    
    # Convert to legacy format for compatibility
    legacy_config = TrainingConfiguration(**config.to_legacy_config())
    
    # Initialize training orchestrator
    orchestrator = TrainingOrchestrator(legacy_config)
    
    # Create mock agent
    agent = MockAdCampaignAgent("AdCampaignAgent_v1")
    
    # Create mock environments for each phase
    environments = {
        "simulation": MockSimulationEnvironment(),
        "historical": MockHistoricalEnvironment(),
        "real": MockRealEnvironment(),
        "scaled": MockSimulationEnvironment()  # Use simulation for scaled demo
    }
    
    logger.info("Initialized agent and environments")
    
    try:
        # Start training
        logger.info("=" * 60)
        logger.info("STARTING TRAINING ORCHESTRATION")
        logger.info("=" * 60)
        
        success = await orchestrator.start_training(agent, environments)
        
        if success:
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            
            # Get final metrics
            final_metrics = orchestrator.get_metrics()
            logger.info(f"Final Training Metrics:")
            logger.info(f"  Total Episodes: {final_metrics.total_episodes}")
            logger.info(f"  Successful Episodes: {final_metrics.successful_episodes}")
            logger.info(f"  Success Rate: {final_metrics.successful_episodes / max(1, final_metrics.total_episodes):.3f}")
            logger.info(f"  Average Reward: {final_metrics.average_reward:.3f}")
            logger.info(f"  Total Budget Spent: ${final_metrics.budget_spent:.2f}")
            logger.info(f"  Safety Violations: {final_metrics.safety_violations}")
            logger.info(f"  Final Phase: {final_metrics.current_phase.value}")
            
        else:
            logger.error("Training failed!")
            return False
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        orchestrator.stop_training()
        return False
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return False
    
    return True


async def run_phase_specific_example():
    """Run example focusing on specific phase analysis"""
    
    logger = logging.getLogger(__name__)
    logger.info("Running phase-specific analysis example")
    
    # Load quick test configuration
    from training_orchestrator.config import QUICK_TEST_CONFIG
    config = TrainingConfiguration(**QUICK_TEST_CONFIG.to_legacy_config())
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(config)
    
    # Get phase manager for analysis
    phase_manager = orchestrator.phase_manager
    
    # Demonstrate phase progression checking
    from training_orchestrator.phases import TrainingPhase
    
    # Simulate some performance data
    simulation_performance = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87]
    
    # Check graduation criteria
    can_graduate, reason = phase_manager.can_graduate_phase(
        TrainingPhase.SIMULATION,
        {"average_reward": 0.8, "success_rate": 0.85, "performance_improvement_rate": 0.1, "performance_consistency": 0.9},
        len(simulation_performance)
    )
    
    logger.info(f"Simulation phase graduation check: {can_graduate}")
    logger.info(f"Reason: {reason}")
    
    # Get curriculum progress
    curriculum_progress = orchestrator.curriculum_scheduler.get_curriculum_progress(TrainingPhase.SIMULATION)
    logger.info(f"Curriculum progress: {curriculum_progress}")
    
    # Get performance summary
    performance_summary = orchestrator.performance_monitor.get_performance_summary(TrainingPhase.SIMULATION)
    logger.info(f"Performance summary: {performance_summary}")


async def main():
    """Main function to run examples"""
    
    print("GAELP Training Orchestrator Examples")
    print("=" * 50)
    print("1. Full Training Example")
    print("2. Phase Analysis Example")
    print("3. Both Examples")
    
    choice = input("Select example to run (1-3): ").strip()
    
    if choice == "1":
        await run_training_example()
    elif choice == "2":
        await run_phase_specific_example()
    elif choice == "3":
        await run_training_example()
        print("\n" + "=" * 50 + "\n")
        await run_phase_specific_example()
    else:
        print("Invalid choice. Running full example...")
        await run_training_example()


if __name__ == "__main__":
    # Check if running in async context
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())