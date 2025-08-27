"""
Example usage of the Journey Timeout and Abandonment System

Demonstrates how to use the journey timeout manager for handling 
zombie journeys and calculating abandonment penalties.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

from training_orchestrator import (
    TrainingOrchestrator,
    TrainingConfiguration,
    JourneyTimeoutManager,
    TimeoutConfiguration,
    AbandonmentReason,
    create_timeout_manager
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_timeout_manager():
    """Demonstrate the standalone timeout manager functionality"""
    
    print("=== Journey Timeout Manager Demo ===\n")
    
    # Create a standalone timeout manager for testing
    timeout_manager = create_timeout_manager(
        timeout_days=14,
        project_id="gaelp-demo",
        dataset_id="gaelp_test"
    )
    
    try:
        # Start the timeout manager
        await timeout_manager.start()
        
        # Register some test journeys
        journey_ids = []
        for i in range(5):
            journey_id = f"demo-journey-{i}-{uuid.uuid4().hex[:8]}"
            
            # Some journeys are already expired for demo
            start_time = datetime.now() - timedelta(days=15 + i)
            
            timeout_at = await timeout_manager.register_journey(
                journey_id=journey_id,
                start_time=start_time,
                user_id=f"demo-user-{i}"
            )
            
            journey_ids.append(journey_id)
            print(f"Registered journey {journey_id}")
            print(f"  Start time: {start_time}")
            print(f"  Timeout at: {timeout_at}")
            print(f"  Already expired: {datetime.now() >= timeout_at}\n")
        
        # Check for timeouts
        print("Checking for timed out journeys...")
        timed_out = await timeout_manager.check_timeouts()
        print(f"Found {len(timed_out)} timed out journeys: {timed_out}\n")
        
        # Manually mark a journey as abandoned for different reasons
        if journey_ids:
            test_journey = journey_ids[0]
            print(f"Manually abandoning journey {test_journey} due to competitor conversion...")
            
            # Create sample journey data for penalty calculation
            journey_data = {
                'journey_start': datetime.now() - timedelta(days=7),
                'total_cost': 125.50,
                'touchpoint_count': 8,
                'current_state': 'INTENT',
                'conversion_probability': 0.45,
                'expected_conversion_value': 200.0
            }
            
            penalty = await timeout_manager.mark_abandoned(
                journey_id=test_journey,
                reason=AbandonmentReason.COMPETITOR_CONVERSION,
                journey_data=journey_data
            )
            
            print(f"Abandonment penalty calculated:")
            print(f"  Journey ID: {penalty.journey_id}")
            print(f"  Reason: {penalty.abandonment_reason.value}")
            print(f"  Days active: {penalty.days_active}")
            print(f"  Total cost: ${penalty.total_cost:.2f}")
            print(f"  Touchpoints: {penalty.touchpoint_count}")
            print(f"  Last state: {penalty.last_state}")
            print(f"  Penalty amount: ${penalty.penalty_amount:.2f}")
            print(f"  Opportunity cost: ${penalty.opportunity_cost:.2f}\n")
        
        # Get analytics (would query BigQuery in real usage)
        print("Getting abandonment analytics...")
        analytics = await timeout_manager.get_abandonment_analytics(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        print("Analytics summary:")
        for key, value in analytics.get('summary', {}).items():
            print(f"  {key}: {value}")
        
        # Extend timeout for a journey
        if len(journey_ids) > 1:
            extend_journey = journey_ids[1]
            print(f"\nExtending timeout for journey {extend_journey} by 7 days...")
            
            success = await timeout_manager.extend_timeout(
                journey_id=extend_journey,
                additional_days=7,
                reason="high_engagement_detected"
            )
            
            if success:
                status = timeout_manager.get_timeout_status(extend_journey)
                if status:
                    print(f"Extension successful. New timeout: {status['timeout_at']}")
                    print(f"Days remaining: {status['days_remaining']}")
        
        # Cleanup stale data
        print("\nCleaning up stale data...")
        cleanup_stats = await timeout_manager.cleanup_stale_data(older_than_days=1)
        print(f"Cleanup statistics: {cleanup_stats}")
        
    finally:
        # Stop the timeout manager
        await timeout_manager.stop()
    
    print("\n=== Timeout Manager Demo Complete ===")


async def demonstrate_training_integration():
    """Demonstrate timeout manager integration with training orchestrator"""
    
    print("\n=== Training Integration Demo ===\n")
    
    # Create training configuration with timeout settings
    config = TrainingConfiguration(
        experiment_id=f"timeout-demo-{uuid.uuid4().hex[:8]}",
        bigquery_project="gaelp-demo",
        bigquery_dataset="training_logs",
        journey_timeout_days=14,
        inactivity_threshold_hours=72,
        cleanup_stale_data_days=30
    )
    
    # Create training orchestrator (this would initialize the timeout manager)
    try:
        orchestrator = TrainingOrchestrator(config)
        
        # Start journey monitoring (in real usage, this would be part of training loop)
        await orchestrator.start_journey_monitoring()
        
        # Simulate registering journeys during training
        training_journeys = []
        for i in range(3):
            journey_id = f"training-journey-{i}-{uuid.uuid4().hex[:8]}"
            
            timeout_at = await orchestrator.register_training_journey(
                journey_id=journey_id,
                user_id=f"training-user-{i}"
            )
            
            training_journeys.append(journey_id)
            print(f"Registered training journey {journey_id} with timeout: {timeout_at}")
        
        # Get current metrics
        metrics = orchestrator.get_metrics()
        print(f"\nCurrent training metrics:")
        print(f"  Active journeys: {metrics.active_journeys}")
        print(f"  Timed out journeys: {metrics.timed_out_journeys}")
        print(f"  Total abandonment penalty: ${metrics.total_abandonment_penalty:.2f}")
        print(f"  Zombie journeys cleaned: {metrics.zombie_journeys_cleaned}")
        
        # Check timeouts
        print("\nChecking for timeouts during training...")
        timed_out = await orchestrator.check_journey_timeouts()
        print(f"Timed out journeys: {timed_out}")
        
        # Get abandonment analytics
        print("\nGetting abandonment analytics...")
        analytics = await orchestrator.get_journey_abandonment_analytics(days=7)
        print(f"Analytics period: last 7 days")
        print(f"Total abandonments: {analytics.get('summary', {}).get('total_abandonments', 0)}")
        
        # Stop monitoring
        await orchestrator.stop_journey_monitoring()
        
    except Exception as e:
        logger.error(f"Error in training integration demo: {e}")
        print(f"Note: Some features may not work without proper BigQuery/Redis setup")
    
    print("\n=== Training Integration Demo Complete ===")


async def demonstrate_penalty_calculation():
    """Demonstrate different abandonment penalty calculations"""
    
    print("\n=== Penalty Calculation Demo ===\n")
    
    timeout_manager = create_timeout_manager(timeout_days=14)
    
    # Test different abandonment scenarios
    scenarios = [
        {
            'name': 'Early Stage Timeout',
            'reason': AbandonmentReason.TIMEOUT,
            'data': {
                'journey_start': datetime.now() - timedelta(days=3),
                'total_cost': 25.0,
                'touchpoint_count': 2,
                'current_state': 'AWARE',
                'conversion_probability': 0.1,
                'expected_conversion_value': 100.0
            }
        },
        {
            'name': 'High-Intent Competitor Loss',
            'reason': AbandonmentReason.COMPETITOR_CONVERSION,
            'data': {
                'journey_start': datetime.now() - timedelta(days=10),
                'total_cost': 150.0,
                'touchpoint_count': 12,
                'current_state': 'INTENT',
                'conversion_probability': 0.7,
                'expected_conversion_value': 300.0
            }
        },
        {
            'name': 'Budget Exhausted',
            'reason': AbandonmentReason.BUDGET_EXHAUSTED,
            'data': {
                'journey_start': datetime.now() - timedelta(days=5),
                'total_cost': 75.0,
                'touchpoint_count': 6,
                'current_state': 'CONSIDERING',
                'conversion_probability': 0.3,
                'expected_conversion_value': 150.0
            }
        },
        {
            'name': 'User Fatigue',
            'reason': AbandonmentReason.FATIGUE,
            'data': {
                'journey_start': datetime.now() - timedelta(days=20),
                'total_cost': 200.0,
                'touchpoint_count': 25,
                'current_state': 'CONSIDERING',
                'conversion_probability': 0.15,
                'expected_conversion_value': 120.0
            }
        }
    ]
    
    for scenario in scenarios:
        journey_id = f"penalty-demo-{uuid.uuid4().hex[:8]}"
        
        penalty = timeout_manager.calculate_abandonment_cost(
            journey_id=journey_id,
            reason=scenario['reason'],
            journey_data=scenario['data']
        )
        
        print(f"Scenario: {scenario['name']}")
        print(f"  Reason: {penalty.abandonment_reason.value}")
        print(f"  Days active: {penalty.days_active}")
        print(f"  Total cost: ${penalty.total_cost:.2f}")
        print(f"  Last state: {penalty.last_state}")
        print(f"  Conversion probability lost: {penalty.conversion_probability_lost:.1%}")
        print(f"  Penalty amount: ${penalty.penalty_amount:.2f}")
        print(f"  Opportunity cost: ${penalty.opportunity_cost:.2f}")
        print(f"  Total impact: ${penalty.penalty_amount + penalty.opportunity_cost:.2f}\n")
    
    print("=== Penalty Calculation Demo Complete ===")


async def main():
    """Run all demonstration examples"""
    
    print("Journey Timeout and Abandonment System Demo")
    print("=" * 50)
    
    try:
        # Run demonstrations
        await demonstrate_penalty_calculation()
        await demonstrate_timeout_manager()
        await demonstrate_training_integration()
        
        print("\nAll demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\nDemo encountered an error: {e}")
        print("Note: Some features require proper BigQuery/Redis configuration")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())