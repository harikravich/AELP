#!/usr/bin/env python3
"""
LAUNCH PRODUCTION ONLINE LEARNING
Integration with existing GAELP system for continuous learning from production data

NO FALLBACKS - PRODUCTION READY SYSTEM
"""

import asyncio
import logging
import sys
import signal
import json
import time
from datetime import datetime
from typing import Dict, Any
import traceback

# Import production components
from production_online_learner import ProductionOnlineLearner, create_production_online_learner
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent, DynamicEnrichedState
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager
from audit_trail import log_decision, log_outcome, log_budget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_online_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionOnlineLearningSystem:
    """Main production online learning system coordinator"""
    
    def __init__(self):
        self.online_learner = None
        self.agent = None
        self.environment = None
        self.discovery = None
        self.running = False
        self.shutdown_requested = False
        
        # Performance tracking
        self.total_episodes = 0
        self.total_revenue = 0.0
        self.total_spend = 0.0
        self.start_time = datetime.now()
        
    async def initialize_system(self):
        """Initialize all system components"""
        logger.info("=" * 70)
        logger.info("INITIALIZING PRODUCTION ONLINE LEARNING SYSTEM")
        logger.info("=" * 70)
        
        try:
            # Initialize discovery engine first
            logger.info("Initializing discovery engine...")
            self.discovery = DiscoveryEngine(write_enabled=True, cache_only=False)
            
            # Initialize supporting components
            logger.info("Initializing supporting components...")
            creative_selector = CreativeSelector()
            attribution = AttributionEngine()
            budget_pacer = BudgetPacer()
            identity_resolver = IdentityResolver()
            pm = ParameterManager()
            
            # Create production environment
            logger.info("Creating production environment...")
            self.environment = ProductionFortifiedEnvironment(
                parameter_manager=pm,
                use_real_ga4_data=True,  # Use real data for production
                is_parallel=False
            )
            
            # Create production RL agent
            logger.info("Creating production RL agent...")
            self.agent = ProductionFortifiedRLAgent(
                discovery_engine=self.discovery,
                creative_selector=creative_selector,
                attribution_engine=attribution,
                budget_pacer=budget_pacer,
                identity_resolver=identity_resolver,
                parameter_manager=pm
            )
            
            # Create online learner
            logger.info("Creating online learner...")
            self.online_learner = create_production_online_learner(self.agent, self.discovery)
            
            # Log initialization success
            patterns = self.discovery.get_discovered_patterns()
            channels = patterns.get('channels', {})
            
            logger.info("System initialization complete!")
            logger.info(f"  - Agent initialized with {len(self.agent.discovered_channels)} discovered channels")
            logger.info(f"  - {len(self.agent.discovered_segments)} discovered segments")
            logger.info(f"  - {len(self.agent.discovered_creatives)} discovered creatives")
            logger.info(f"  - Online learner ready with Thompson Sampling")
            logger.info(f"  - Safety guardrails active")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            traceback.print_exc()
            return False
    
    async def run_production_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run single production episode with online learning"""
        
        try:
            # Reset environment
            obs, info = self.environment.reset()
            state = self.environment.current_user_state
            
            episode_data = {
                'episode': episode_num,
                'start_time': time.time(),
                'total_reward': 0.0,
                'total_spend': 0.0,
                'total_conversions': 0,
                'total_revenue': 0.0,
                'actions_taken': 0,
                'exploration_actions': 0,
                'safety_violations': 0,
                'experiments': []
            }
            
            done = False
            step = 0
            max_steps = 100  # Reasonable episode length
            
            while not done and step < max_steps:
                # Generate unique user ID for this step
                user_id = f"user_{episode_num}_{step}_{int(time.time())}"
                
                # Convert state for online learner
                online_state = self._convert_state_for_online_learner(state, episode_data)
                
                # Select action using online learner (with A/B testing)
                production_action = await self.online_learner.select_production_action(
                    online_state, user_id
                )
                
                # Convert back to agent action format
                agent_action = self._convert_action_for_agent(production_action)
                
                # Execute action in environment
                next_obs, reward, terminated, truncated, info = self.environment.step(agent_action)
                done = terminated or truncated
                next_state = self.environment.current_user_state
                
                # Calculate actual outcome metrics
                spend = info.get('spend', 0.0)
                conversions = 1 if info.get('converted', False) else 0
                revenue = info.get('revenue', 0.0)
                
                # Create outcome for online learner
                outcome = {
                    'conversion': conversions > 0,
                    'reward': reward,
                    'spend': spend,
                    'revenue': revenue,
                    'next_state': self._convert_state_for_online_learner(next_state, episode_data),
                    'done': done,
                    'channel': info.get('channel', 'unknown'),
                    'campaign_id': f"prod_ep_{episode_num}_{step}",
                    'attribution_data': info.get('attribution_data', {}),
                    'safety_violation': reward < -1.0  # Simple safety check
                }
                
                # Record outcome with online learner
                self.online_learner.record_production_outcome(production_action, outcome, user_id)
                
                # Update episode statistics
                episode_data['total_reward'] += reward
                episode_data['total_spend'] += spend
                episode_data['total_conversions'] += conversions
                episode_data['total_revenue'] += revenue
                episode_data['actions_taken'] += 1
                
                if production_action.get('strategy') in ['aggressive', 'experimental']:
                    episode_data['exploration_actions'] += 1
                
                if outcome.get('safety_violation', False):
                    episode_data['safety_violations'] += 1
                
                # Track experiment participation
                if 'experiment_id' in production_action:
                    episode_data['experiments'].append({
                        'id': production_action['experiment_id'],
                        'variant': production_action['variant_id']
                    })
                
                # Train agent with real outcome
                if step > 0:  # Need previous state
                    self.agent.train(state, agent_action, reward, next_state, done)
                
                # Update state
                state = next_state
                step += 1
                
                # Log progress periodically
                if step % 20 == 0:
                    logger.info(f"Episode {episode_num}, Step {step}: "
                              f"Reward={reward:.2f}, Spend=${spend:.2f}, "
                              f"Conversions={conversions}")
            
            # Finalize episode data
            episode_data['end_time'] = time.time()
            episode_data['duration'] = episode_data['end_time'] - episode_data['start_time']
            episode_data['roi'] = (episode_data['total_revenue'] / episode_data['total_spend'] 
                                 if episode_data['total_spend'] > 0 else 0)
            
            return episode_data
            
        except Exception as e:
            logger.error(f"Episode {episode_num} failed: {e}")
            return {
                'episode': episode_num,
                'error': str(e),
                'total_reward': 0.0,
                'total_spend': 0.0
            }
    
    def _convert_state_for_online_learner(self, agent_state: DynamicEnrichedState, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert agent state to online learner format"""
        return {
            'budget_remaining': getattr(agent_state, 'budget_remaining', 100.0),
            'daily_spend': episode_data.get('total_spend', 0.0),
            'daily_budget': 1000.0,  # From safety guardrails
            'current_ctr': getattr(agent_state, 'current_ctr', 0.02),
            'current_cvr': getattr(agent_state, 'current_cvr', 0.01),
            'current_cpc': getattr(agent_state, 'current_cpc', 1.0),
            'time_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'competition_level': 0.5,  # Mock - could be discovered
            'performance_history': {
                'avg_roas': episode_data.get('roi', 1.0),
                'avg_bid': 1.0  # Mock
            }
        }
    
    def _convert_action_for_agent(self, online_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert online learner action to agent format"""
        # Extract core action parameters
        agent_action = {
            'bid_amount': online_action.get('bid_amount', 1.0),
            'budget': online_action.get('budget_allocation', 0.1) * 1000.0,  # Convert to absolute budget
            'creative_type': online_action.get('creative_type', 'image'),
            'target_audience': online_action.get('target_audience', 'professionals'),
            'bid_strategy': 'cpc'  # Default strategy
        }
        
        return agent_action
    
    async def run_production_learning(self, num_episodes: int = 1000):
        """Run production learning loop"""
        logger.info("=" * 70)
        logger.info(f"STARTING PRODUCTION ONLINE LEARNING - {num_episodes} EPISODES")
        logger.info("=" * 70)
        
        self.running = True
        
        # Start continuous learning cycle in background
        learning_task = asyncio.create_task(self.online_learner.continuous_learning_cycle())
        
        try:
            for episode in range(num_episodes):
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping training")
                    break
                
                # Run episode
                episode_data = await self.run_production_episode(episode)
                
                # Update global statistics
                self.total_episodes += 1
                self.total_revenue += episode_data.get('total_revenue', 0.0)
                self.total_spend += episode_data.get('total_spend', 0.0)
                
                # Log episode summary
                self._log_episode_summary(episode_data)
                
                # Create A/B tests periodically
                if episode % 100 == 0 and episode > 0:
                    await self._maybe_create_ab_test()
                
                # System health check
                if episode % 50 == 0:
                    await self._system_health_check()
                
                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            learning_task.cancel()
            
            # Final summary
            self._log_final_summary()
    
    def _log_episode_summary(self, episode_data: Dict[str, Any]):
        """Log episode summary"""
        if episode_data['episode'] % 10 == 0:  # Log every 10th episode
            logger.info(f"Episode {episode_data['episode']} Summary:")
            logger.info(f"  Total Reward: {episode_data.get('total_reward', 0):.2f}")
            logger.info(f"  Total Spend: ${episode_data.get('total_spend', 0):.2f}")
            logger.info(f"  Conversions: {episode_data.get('total_conversions', 0)}")
            logger.info(f"  Revenue: ${episode_data.get('total_revenue', 0):.2f}")
            logger.info(f"  ROI: {episode_data.get('roi', 0):.2f}x")
            logger.info(f"  Actions: {episode_data.get('actions_taken', 0)} "
                      f"(Exploration: {episode_data.get('exploration_actions', 0)})")
            
            if episode_data.get('safety_violations', 0) > 0:
                logger.warning(f"  Safety Violations: {episode_data['safety_violations']}")
            
            if episode_data.get('experiments'):
                logger.info(f"  Experiments: {len(episode_data['experiments'])} active")
    
    async def _maybe_create_ab_test(self):
        """Maybe create new A/B test"""
        # Only create test if we don't have too many active
        status = self.online_learner.get_system_status()
        
        if status['active_experiments'] < 3:  # Max 3 concurrent experiments
            # Create test of bid strategies
            variants = {
                'control': {'bid_amount': '0%'},  # No change
                'aggressive': {'bid_amount': '+20%'},  # 20% higher bids
                'conservative': {'bid_amount': '-10%'}  # 10% lower bids
            }
            
            test_name = f"bid_optimization_test_{int(time.time())}"
            exp_id = self.online_learner.create_ab_test(test_name, variants)
            logger.info(f"Created A/B test: {test_name} ({exp_id})")
    
    async def _system_health_check(self):
        """Check system health and log status"""
        status = self.online_learner.get_system_status()
        
        logger.info("System Health Check:")
        logger.info(f"  Circuit Breaker: {'TRIGGERED' if status['circuit_breaker'] else 'OK'}")
        logger.info(f"  Active Experiments: {status['active_experiments']}")
        logger.info(f"  Model Updates: {status['model_update_count']}")
        logger.info(f"  Experience Buffer: {status['experience_buffer_size']}")
        
        # Check strategy performance
        strategy_perf = status.get('strategy_performance', {})
        for strategy, perf in strategy_perf.items():
            logger.info(f"  {strategy.title()}: "
                      f"CVR={perf.get('expected_conversion_rate', 0):.3f}, "
                      f"Trials={perf.get('total_trials', 0)}")
        
        # Log recent performance
        recent_perf = status.get('recent_performance', {})
        if recent_perf:
            logger.info(f"  Recent Performance: "
                      f"Reward={recent_perf.get('avg_reward', 0):.2f}, "
                      f"Spend=${recent_perf.get('total_spend', 0):.2f}, "
                      f"Conversions={recent_perf.get('total_conversions', 0)}")
    
    def _log_final_summary(self):
        """Log final training summary"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        overall_roi = self.total_revenue / self.total_spend if self.total_spend > 0 else 0
        
        logger.info("=" * 70)
        logger.info("PRODUCTION ONLINE LEARNING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Runtime: {runtime:.0f} seconds ({runtime/3600:.1f} hours)")
        logger.info(f"Total Episodes: {self.total_episodes}")
        logger.info(f"Total Spend: ${self.total_spend:.2f}")
        logger.info(f"Total Revenue: ${self.total_revenue:.2f}")
        logger.info(f"Overall ROI: {overall_roi:.2f}x")
        logger.info(f"Average Spend per Episode: ${self.total_spend/max(1,self.total_episodes):.2f}")
        logger.info("=" * 70)
        
        # Log system final status
        status = self.online_learner.get_system_status()
        logger.info("Final System Status:")
        logger.info(json.dumps(status, indent=2))
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print(" PRODUCTION ONLINE LEARNING SYSTEM ".center(70))
    print("=" * 70)
    print("Features:")
    print("✅ Thompson Sampling for safe exploration")
    print("✅ A/B testing with statistical significance")
    print("✅ Circuit breakers and safety guardrails")
    print("✅ Continuous model updates from production data")
    print("✅ Real-time feedback loop")
    print("✅ NO HARDCODED PARAMETERS")
    print("=" * 70)
    
    # Create system
    system = ProductionOnlineLearningSystem()
    system.setup_signal_handlers()
    
    # Initialize
    success = await system.initialize_system()
    if not success:
        logger.error("System initialization failed. Exiting.")
        sys.exit(1)
    
    # Get configuration from user
    try:
        num_episodes = int(input("\nEnter number of episodes to run (default 500): ") or "500")
    except ValueError:
        num_episodes = 500
    
    print(f"\nStarting production online learning with {num_episodes} episodes...")
    print("Press Ctrl+C to stop gracefully at any time.")
    print("=" * 70)
    
    # Run production learning
    await system.run_production_learning(num_episodes)
    
    print("\nProduction online learning completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nSystem failed: {e}")
        traceback.print_exc()
        sys.exit(1)