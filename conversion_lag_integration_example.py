#!/usr/bin/env python3
"""
Conversion Lag Model Integration Example

This example demonstrates how to wire the Conversion Lag Model with:
1. Journey Timeout Manager for abandonment decisions
2. Attribution Models for dynamic window sizing
3. Delayed Reward System for timing predictions
4. Enhanced handling of censored data and 30+ day conversions

Key Integration Points:
- Use predict_conversion_time() for each journey
- Handle right-censored data with handle_censored_data()
- Calculate hazard rates with calculate_hazard_rate()
- Adjust attribution windows based on predictions
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import all required components
from conversion_lag_model import ConversionLagModel, ConversionJourney
from attribution_models import AttributionEngine, Journey, Touchpoint
from training_orchestrator.journey_timeout import JourneyTimeoutManager, TimeoutConfiguration
from training_orchestrator.delayed_reward_system import DelayedRewardSystem, DelayedRewardConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversionLagIntegrationDemo:
    """
    Demonstration of fully integrated conversion lag model system.
    """
    
    def __init__(self):
        """Initialize all components with conversion lag model integration."""
        
        # 1. Initialize Conversion Lag Model
        self.conversion_lag_model = ConversionLagModel(
            attribution_window_days=30,
            timeout_threshold_days=45,
            model_type='weibull'
        )
        
        # 2. Initialize Journey Timeout Manager with conversion lag support
        timeout_config = TimeoutConfiguration(
            enable_conversion_lag_model=True,
            conversion_lag_model_type='weibull',
            attribution_window_days=30,
            timeout_threshold_days=45
        )
        self.timeout_manager = JourneyTimeoutManager(timeout_config)
        
        # 3. Initialize Attribution Engine with conversion lag model
        self.attribution_engine = AttributionEngine(
            conversion_lag_model=self.conversion_lag_model
        )
        
        # 4. Initialize Delayed Reward System with conversion lag support
        delayed_reward_config = DelayedRewardConfig(
            enable_conversion_lag_model=True,
            dynamic_attribution_windows=True,
            conversion_timeout_threshold_days=45
        )
        self.delayed_reward_system = DelayedRewardSystem(delayed_reward_config)
        
        logger.info("Conversion lag integration system initialized")
    
    async def demonstrate_full_integration(self):
        """
        Demonstrate the complete integrated workflow.
        """
        logger.info("=== Starting Conversion Lag Integration Demo ===")
        
        # Step 1: Generate sample journey data and train the model
        await self._train_models_with_sample_data()
        
        # Step 2: Demonstrate journey registration with predictions
        await self._demonstrate_journey_registration()
        
        # Step 3: Demonstrate dynamic attribution windows
        await self._demonstrate_dynamic_attribution()
        
        # Step 4: Demonstrate conversion predictions and timeout decisions
        await self._demonstrate_conversion_predictions()
        
        # Step 5: Demonstrate censored data handling
        await self._demonstrate_censored_data_handling()
        
        logger.info("=== Conversion Lag Integration Demo Complete ===")
    
    async def _train_models_with_sample_data(self):
        """Train the conversion lag model with sample data."""
        logger.info("Training conversion lag model with sample data...")
        
        # Create sample journeys for training
        sample_journeys = self._create_sample_journeys(1000)
        
        # Train the conversion lag model
        self.conversion_lag_model.fit(sample_journeys)
        
        # Train the timeout manager's model
        await self.timeout_manager.train_conversion_lag_model(sample_journeys)
        
        # Train the delayed reward system's model
        await self.delayed_reward_system.train_conversion_lag_model()
        
        # Get training insights
        insights = self.conversion_lag_model.get_conversion_insights(sample_journeys)
        logger.info(f"Model training completed. Insights: {insights}")
    
    async def _demonstrate_journey_registration(self):
        """Demonstrate journey registration with conversion predictions."""
        logger.info("Demonstrating journey registration with predictions...")
        
        # Register a new journey
        journey_id = "demo_journey_001"
        user_id = "user_12345"
        start_time = datetime.now()
        
        # Sample touchpoints
        touchpoints = [
            {
                'timestamp': start_time,
                'channel': 'search',
                'action_data': '{"budget": 100, "keywords": ["shoes", "running"]}'
            },
            {
                'timestamp': start_time + timedelta(hours=2),
                'channel': 'display',
                'action_data': '{"budget": 75, "creative": "banner_001"}'
            }
        ]
        
        # Sample features
        features = {
            'user_engagement_score': 0.7,
            'product_interest_score': 0.8,
            'demographic_score': 0.6
        }
        
        # Register with timeout manager (gets conversion predictions)
        prediction_data = await self.timeout_manager.register_journey_for_conversion_prediction(
            journey_id=journey_id,
            user_id=user_id,
            start_time=start_time,
            touchpoints=touchpoints,
            features=features
        )
        
        if prediction_data:
            logger.info(f"Journey {journey_id} registered with predictions:")
            logger.info(f"  - Recommended timeout: {prediction_data['recommended_timeout_days']} days")
            logger.info(f"  - Conversion probabilities (7 days): {prediction_data.get('conversion_probabilities', [])[:7]}")
        
        # Register with delayed reward system
        touchpoint_id = await self.delayed_reward_system.store_pending_reward(
            episode_id=journey_id,
            user_id=user_id,
            campaign_id="campaign_001",
            action={"budget": 100, "channel": "search"},
            state={"market_conditions": "high_competition"},
            immediate_reward=25.0,
            channel="search"
        )
        
        logger.info(f"Touchpoint registered in delayed reward system: {touchpoint_id}")
    
    async def _demonstrate_dynamic_attribution(self):
        """Demonstrate dynamic attribution window calculation."""
        logger.info("Demonstrating dynamic attribution windows...")
        
        # Create a sample journey
        journey = self._create_sample_attribution_journey()
        
        # Calculate attribution with dynamic window
        attributions, window_days = self.attribution_engine.calculate_attribution_with_dynamic_window(
            journey=journey,
            model_name='time_decay',
            use_dynamic_window=True
        )
        
        logger.info(f"Dynamic attribution results:")
        logger.info(f"  - Attribution window: {window_days} days")
        logger.info(f"  - Touchpoints included: {len(attributions)}")
        logger.info(f"  - Attribution weights: {attributions}")
        
        # Get conversion timing insights
        insights = self.attribution_engine.get_conversion_timing_insights(journey)
        if insights:
            logger.info(f"  - Peak conversion day: {insights['peak_conversion_day']}")
            logger.info(f"  - Median conversion day: {insights['median_conversion_day']}")
            logger.info(f"  - Max conversion probability: {insights['max_conversion_probability']:.3f}")
    
    async def _demonstrate_conversion_predictions(self):
        """Demonstrate conversion timing predictions."""
        logger.info("Demonstrating conversion timing predictions...")
        
        user_id = "user_demo_123"
        
        # Store some touchpoints for the user
        for i in range(3):
            await self.delayed_reward_system.store_pending_reward(
                episode_id=f"episode_demo_{i}",
                user_id=user_id,
                campaign_id="demo_campaign",
                action={"budget": 50 + i * 10, "channel": ["search", "display", "social"][i]},
                state={"step": i},
                immediate_reward=10.0 + i * 2,
                channel=["search", "display", "social"][i]
            )
        
        # Get conversion predictions
        predictions = await self.delayed_reward_system.predict_conversion_timing(user_id)
        
        if predictions:
            logger.info(f"Conversion predictions for user {user_id}:")
            logger.info(f"  - Peak conversion day: {predictions['peak_conversion_day']}")
            logger.info(f"  - Median conversion day: {predictions['median_conversion_day']}")
            logger.info(f"  - Recommended attribution window: {predictions['recommended_attribution_window']}")
            logger.info(f"  - 7-day conversion probabilities: {predictions['conversion_probabilities_7_days']}")
        
        # Calculate dynamic attribution window
        dynamic_window = await self.delayed_reward_system.calculate_dynamic_attribution_window(user_id)
        logger.info(f"  - Dynamic attribution window: {dynamic_window} days")
    
    async def _demonstrate_censored_data_handling(self):
        """Demonstrate handling of right-censored data."""
        logger.info("Demonstrating censored data handling...")
        
        # Handle censored data in timeout manager
        censored_stats = await self.timeout_manager.handle_censored_journey_data()
        logger.info(f"Timeout manager censored data stats: {censored_stats}")
        
        # Handle censored data in delayed reward system
        delayed_censored_stats = await self.delayed_reward_system.handle_censored_data_update()
        logger.info(f"Delayed reward system censored data stats: {delayed_censored_stats}")
        
        # Demonstrate timeout decisions for long-running journeys
        long_journey_id = "long_journey_001"
        long_user_id = "user_long_123"
        long_start_time = datetime.now() - timedelta(days=35)  # 35 days ago
        
        prediction_data = await self.timeout_manager.register_journey_for_conversion_prediction(
            journey_id=long_journey_id,
            user_id=long_user_id,
            start_time=long_start_time,
            touchpoints=[{
                'timestamp': long_start_time,
                'channel': 'search',
                'action_data': '{"budget": 200}'
            }],
            features={'user_engagement_score': 0.3}  # Low engagement
        )
        
        if prediction_data:
            logger.info(f"Long journey prediction (35 days old):")
            logger.info(f"  - Recommended timeout: {prediction_data['recommended_timeout_days']} days")
            
            # Check if this journey should be abandoned
            if prediction_data['recommended_timeout_days'] < 35:
                logger.info(f"  - Journey {long_journey_id} should be considered for abandonment")
    
    def _create_sample_journeys(self, count: int) -> List[ConversionJourney]:
        """Create sample conversion journeys for training."""
        import random
        
        journeys = []
        current_time = datetime.now()
        
        for i in range(count):
            start_time = current_time - timedelta(days=random.randint(1, 60))
            
            # Simulate conversion patterns
            if random.random() < 0.25:  # 25% conversion rate
                # Converted journey
                duration = np.random.exponential(scale=7)  # Most convert quickly
                if random.random() < 0.15:  # 15% take much longer
                    duration += np.random.exponential(scale=25)
                
                end_time = start_time + timedelta(days=duration)
                converted = True
                is_censored = False
            else:
                # Non-converted journey
                if random.random() < 0.6:  # 60% of non-converted are ongoing
                    end_time = None
                    converted = False
                    is_censored = True
                    duration = None
                else:  # 40% abandoned
                    duration = random.randint(30, 90)
                    end_time = start_time + timedelta(days=duration)
                    converted = False
                    is_censored = True
            
            # Create touchpoints
            num_touchpoints = np.random.poisson(2) + 1
            touchpoints = []
            for j in range(num_touchpoints):
                touchpoints.append({
                    'channel': random.choice(['email', 'web', 'social', 'search', 'display']),
                    'timestamp': start_time + timedelta(days=random.randint(0, int(duration or 30)))
                })
            
            # Create features
            features = {
                'user_engagement_score': random.uniform(0, 1),
                'product_interest_score': random.uniform(0, 1),
                'demographic_score': random.uniform(0, 1)
            }
            
            journey = ConversionJourney(
                user_id=f'user_{i}',
                start_time=start_time,
                end_time=end_time,
                converted=converted,
                duration_days=duration,
                touchpoints=touchpoints,
                features=features,
                is_censored=is_censored
            )
            
            journeys.append(journey)
        
        return journeys
    
    def _create_sample_attribution_journey(self) -> Journey:
        """Create a sample journey for attribution demonstration."""
        base_time = datetime.now() - timedelta(days=10)
        
        touchpoints = [
            Touchpoint(
                id="tp_1",
                timestamp=base_time,
                channel="search",
                action="search_ad",
                value=25.0
            ),
            Touchpoint(
                id="tp_2", 
                timestamp=base_time + timedelta(days=2),
                channel="display",
                action="banner_ad",
                value=15.0
            ),
            Touchpoint(
                id="tp_3",
                timestamp=base_time + timedelta(days=5),
                channel="email",
                action="promotional_email",
                value=10.0
            ),
            Touchpoint(
                id="tp_4",
                timestamp=base_time + timedelta(days=8),
                channel="social",
                action="social_ad",
                value=20.0
            )
        ]
        
        return Journey(
            id="attribution_demo_journey",
            touchpoints=touchpoints,
            conversion_value=150.0,
            conversion_timestamp=base_time + timedelta(days=9),
            converted=True
        )
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'conversion_lag_model': {
                'is_fitted': self.conversion_lag_model.is_fitted if self.conversion_lag_model else False,
                'model_type': self.conversion_lag_model.model_type if self.conversion_lag_model else None
            },
            'timeout_manager': {
                'active_timeouts': len(self.timeout_manager._active_timeouts),
                'journey_cache': len(self.timeout_manager._journey_data_cache)
            },
            'attribution_engine': {
                'dynamic_attribution_enabled': self.attribution_engine.dynamic_attribution_enabled,
                'available_models': list(self.attribution_engine.models.keys())
            },
            'delayed_reward_system': self.delayed_reward_system.get_statistics()
        }
        
        return stats


async def main():
    """Main demonstration function."""
    
    # Initialize the integration demo
    demo = ConversionLagIntegrationDemo()
    
    try:
        # Run the full integration demonstration
        await demo.demonstrate_full_integration()
        
        # Print system statistics
        stats = await demo.get_system_statistics()
        print("\n=== System Statistics ===")
        print(f"Conversion Lag Model: {stats['conversion_lag_model']}")
        print(f"Timeout Manager: {stats['timeout_manager']}")
        print(f"Attribution Engine: {stats['attribution_engine']}")
        print(f"Delayed Reward System: {stats['delayed_reward_system']}")
        
    except Exception as e:
        logger.error(f"Error in integration demo: {e}")
        raise
    
    finally:
        # Cleanup
        if hasattr(demo.timeout_manager, 'stop'):
            await demo.timeout_manager.stop()
        
        if hasattr(demo.delayed_reward_system, 'shutdown'):
            await demo.delayed_reward_system.shutdown()


if __name__ == "__main__":
    print("=== Conversion Lag Model Integration Demo ===")
    print("This demo shows how to wire the Conversion Lag Model with:")
    print("1. Journey Timeout Manager for abandonment decisions")
    print("2. Attribution Models for dynamic window sizing")  
    print("3. Delayed Reward System for timing predictions")
    print("4. Enhanced handling of censored data and 30+ day conversions")
    print()
    
    asyncio.run(main())
    print("\nDemo completed successfully!")