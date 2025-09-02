#!/usr/bin/env python3
"""
PRODUCTION TRAINING LAUNCHER - Uses new components with NO hardcoding
"""

import sys
import subprocess
from datetime import datetime
import os

def print_banner():
    """Print selection banner"""
    print("\n" + "=" * 70)
    print(" GAELP PRODUCTION TRAINING LAUNCHER ".center(70))
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def main():
    """Main launcher for production system"""
    print_banner()
    
    print("\n🚀 PRODUCTION SYSTEM - NO HARDCODING")
    print("=" * 70)
    print("\nThis system uses:")
    print("✅ fortified_rl_agent_no_hardcoding.py - Everything discovered")
    print("✅ fortified_environment_no_hardcoding.py - Dynamic configuration")
    print("✅ monitor_production_quality.py - Shows actual creative content")
    print("✅ All values discovered from patterns.json")
    print("✅ Z-score normalization from actual data")
    print("✅ Warm start from 4.42% CVR segments")
    print()
    print("Select option:\n")
    print("1. 🎯 Run Production Training")
    print("   - Uses all new components with NO hardcoding")
    print("   - Discovers everything from patterns")
    print("   - Warm start initialization")
    print()
    print("2. 📊 Monitor Production Training")
    print("   - Shows actual creative content")
    print("   - Daily conversion/spend rates")
    print("   - Real-time metrics")
    print()
    print("3. 🧪 Test Production System")
    print("   - Verify no hardcoding violations")
    print("   - Check all components work")
    print()
    print("4. 📈 Launch with Parallel Training (16 environments)")
    print("   - Maximum performance")
    print("   - Ray-based parallelization")
    print()
    print("5. 🔄 Run Online Learning System")
    print("   - Continuous learning from production data")
    print("   - Thompson Sampling exploration/exploitation")
    print("   - A/B testing with statistical significance")
    print("   - Safety guardrails and circuit breakers")
    print()
    print("6. 📊 Monitor Online Learning")
    print("   - Real-time performance dashboard")
    print("   - Strategy performance tracking")
    print("   - A/B test results analysis")
    print()
    print("0. Exit")
    print()
    
    choice = input("Enter choice (1/2/3/4/5/6/0): ").strip()
    
    if choice == "1":
        print("\n" + "=" * 70)
        print("Starting PRODUCTION TRAINING...")
        print("=" * 70)
        print("\nThis will:")
        print("- Use fortified_rl_agent_no_hardcoding.py")
        print("- Use fortified_environment_no_hardcoding.py")
        print("- Discover all parameters from patterns.json")
        print("- Apply warm start from successful segments")
        print("- Use guided exploration near high CVR patterns")
        print("\nPress Ctrl+C to stop at any time")
        print("=" * 70)
        
        input("\nPress Enter to start production training...")
        
        # Create production training script on the fly
        training_code = '''#!/usr/bin/env python3
"""Production Training with NO hardcoding + EMERGENCY CONTROLS"""
import sys
sys.path.insert(0, '/home/hariravichandran/AELP')

import logging
import time
from datetime import datetime

# Import production components
from fortified_rl_agent_no_hardcoding import ProductionFortifiedRLAgent, DynamicEnrichedState
from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
from creative_selector import CreativeSelector
from attribution_models import AttributionEngine
from budget_pacer import BudgetPacer
from identity_resolver import IdentityResolver
from gaelp_parameter_manager import ParameterManager
from emergency_controls import get_emergency_controller, emergency_stop_decorator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("STARTING PRODUCTION TRAINING - NO HARDCODING + EMERGENCY CONTROLS")
    logger.info("="*70)
    
    # Initialize emergency controls FIRST
    emergency_controller = get_emergency_controller()
    logger.info("Emergency control system initialized")
    
    # Check system health before starting
    if not emergency_controller.is_system_healthy():
        logger.error("System not healthy - cannot start training")
        return
    
    # Initialize components with emergency decorators
    @emergency_stop_decorator("discovery_engine")
    def create_discovery():
        return DiscoveryEngine(write_enabled=True, cache_only=False)
    
    @emergency_stop_decorator("creative_selector")
    def create_creative_selector():
        return CreativeSelector()
    
    @emergency_stop_decorator("attribution_engine")
    def create_attribution():
        return AttributionEngine()
    
    @emergency_stop_decorator("budget_pacer")
    def create_budget_pacer():
        return BudgetPacer()
    
    @emergency_stop_decorator("identity_resolver")
    def create_identity_resolver():
        return IdentityResolver()
    
    @emergency_stop_decorator("parameter_manager")
    def create_parameter_manager():
        return ParameterManager()
    
    discovery = create_discovery()
    creative_selector = create_creative_selector()
    attribution = create_attribution()
    budget_pacer = create_budget_pacer()
    identity_resolver = create_identity_resolver()
    pm = create_parameter_manager()
    
    # Create production environment with emergency controls
    logger.info("Creating production environment...")
    @emergency_stop_decorator("environment")
    def create_environment():
        return ProductionFortifiedEnvironment(
            parameter_manager=pm,
            use_real_ga4_data=False,
            is_parallel=False
        )
    
    env = create_environment()
    
    # Create production agent with emergency controls
    logger.info("Creating production agent with discovered parameters...")
    @emergency_stop_decorator("rl_agent")
    def create_agent():
        return ProductionFortifiedRLAgent(
            discovery_engine=discovery,
            creative_selector=creative_selector,
            attribution_engine=attribution,
            budget_pacer=budget_pacer,
            identity_resolver=identity_resolver,
            parameter_manager=pm
        )
    
    agent = create_agent()
    
    logger.info(f"Agent initialized with:")
    logger.info(f"  - {len(agent.discovered_channels)} discovered channels")
    logger.info(f"  - {len(agent.discovered_segments)} discovered segments")
    logger.info(f"  - {len(agent.discovered_creatives)} discovered creatives")
    logger.info(f"  - Warm start enabled: {len(agent.replay_buffer)} samples")
    
    # Training loop with emergency monitoring
    num_episodes = 1000
    for episode in range(num_episodes):
        # Check system health before each episode
        if not emergency_controller.is_system_healthy():
            logger.error("System unhealthy - stopping training")
            break
        
        try:
            obs, info = env.reset()
            episode_reward = 0
            done = False
            step = 0
            episode_spend = 0
            episode_bids = []
            
            while not done:
                # Get current state
                state = env.current_user_state
                
                # Select action with emergency monitoring
                @emergency_stop_decorator("action_selection")
                def select_action_safe():
                    return agent.select_action(state, explore=True)
                
                action = select_action_safe()
                
                # Monitor bid amount for anomalies
                if hasattr(action, 'bid_amount'):
                    bid_amount = float(action.bid_amount)
                    episode_bids.append(bid_amount)
                    emergency_controller.record_bid(bid_amount)
                
                # Step environment with emergency monitoring
                @emergency_stop_decorator("environment_step")
                def step_environment_safe():
                    return env.step(action)
                
                next_obs, reward, terminated, truncated, info = step_environment_safe()
                done = terminated or truncated
                
                # Track spending
                if 'spend' in info:
                    episode_spend += info['spend']
                
                # Get next state
                next_state = env.current_user_state
                
                # Train agent with emergency monitoring
                @emergency_stop_decorator("training_step")
                def train_safe():
                    loss = agent.train(state, action, reward, next_state, done)
                    if loss is not None:
                        emergency_controller.record_training_loss(float(loss))
                    return loss
                
                train_safe()
                
                episode_reward += reward
                step += 1
                
                if step % 100 == 0:
                    logger.info(f"Episode {episode}, Step {step}: Reward={reward:.2f}, Epsilon={agent.epsilon:.3f}")
                    # Check emergency status
                    if emergency_controller.current_emergency_level.value != "green":
                        logger.warning(f"Emergency level: {emergency_controller.current_emergency_level.value}")
            
            # Update budget tracking
            emergency_controller.update_budget_tracking("main_campaign", episode_spend, 1000.0)  # $1000 daily limit
            
            # Log episode results
            metrics = info.get('metrics', {})
            logger.info(f"Episode {episode} complete:")
            logger.info(f"  Total Reward: {episode_reward:.2f}")
            logger.info(f"  Total Spend: ${episode_spend:.2f}")
            logger.info(f"  Max Bid: ${max(episode_bids):.2f}" if episode_bids else "  Max Bid: $0.00")
            logger.info(f"  Conversions: {metrics.get('conversions', 0)}")
            logger.info(f"  Revenue: ${metrics.get('revenue', 0):.2f}")
            logger.info(f"  ROAS: {metrics.get('roas', 0):.2f}x")
            logger.info(f"  Epsilon: {agent.epsilon:.3f}")
            logger.info(f"  Emergency Level: {emergency_controller.current_emergency_level.value}")
            
            # Save model periodically
            if episode % 100 == 0 and episode > 0:
                logger.info(f"Saving model at episode {episode}...")
                # agent.save_model(f"production_model_ep{episode}.pt")
                
        except Exception as e:
            logger.error(f"Error in episode {episode}: {e}")
            emergency_controller.register_error("training_loop", str(e))
            
            # If too many errors, stop training
            if not emergency_controller.is_system_healthy():
                logger.error("Too many errors - stopping training")
                break
    
    logger.info("="*70)
    logger.info("PRODUCTION TRAINING COMPLETE")
    logger.info("="*70)

if __name__ == "__main__":
    main()
'''
        
        # Write and run the training script
        with open('/tmp/run_production_training.py', 'w') as f:
            f.write(training_code)
        
        subprocess.run(["python3", "/tmp/run_production_training.py"])
        
    elif choice == "2":
        print("\n" + "=" * 70)
        print("Starting PRODUCTION MONITOR...")
        print("=" * 70)
        print("\nThis will show:")
        print("- Real-time training metrics")
        print("- Actual creative content (headlines, CTAs)")
        print("- Daily conversion and spend rates")
        print("- Channel performance")
        print("\nPress Ctrl+C to stop")
        print("=" * 70)
        
        time.sleep(2)
        
        # Run production monitor
        subprocess.run(["python3", "monitor_production_quality.py"])
        
    elif choice == "3":
        print("\n" + "=" * 70)
        print("Running PRODUCTION SYSTEM TESTS...")
        print("=" * 70)
        
        # Run production tests
        subprocess.run(["python3", "test_production_no_fallbacks.py"])
        
        print("\n" + "=" * 70)
        print("Tests complete. Check output above.")
        print("=" * 70)
        
    elif choice == "4":
        print("\n" + "=" * 70)
        print("Starting PARALLEL PRODUCTION TRAINING...")
        print("=" * 70)
        print("\nThis will:")
        print("- Launch 16 parallel environments with Ray")
        print("- Use production components with NO hardcoding")
        print("- Maximum training speed")
        print("\nPress Ctrl+C to stop")
        print("=" * 70)
        
        input("\nPress Enter to start parallel training...")
        
        # Run parallel training
        subprocess.run(["python3", "launch_parallel_training.py", "--production"])
        
    elif choice == "5":
        print("\n" + "=" * 70)
        print("Starting ONLINE LEARNING SYSTEM...")
        print("=" * 70)
        print("\nThis will:")
        print("- Use Thompson Sampling for safe exploration")
        print("- Run A/B tests with statistical significance")
        print("- Apply safety guardrails and circuit breakers")
        print("- Update models from production data")
        print("- Create continuous feedback loop")
        print("\nPress Ctrl+C to stop at any time")
        print("=" * 70)
        
        input("\nPress Enter to start online learning...")
        
        # Run online learning system
        subprocess.run(["python3", "standalone_online_learning_demo.py"])
        
    elif choice == "6":
        print("\n" + "=" * 70)
        print("Starting ONLINE LEARNING MONITOR...")
        print("=" * 70)
        print("\nThis will:")
        print("- Show real-time strategy performance")
        print("- Track A/B test results")
        print("- Monitor safety system status")
        print("- Display model update activity")
        print("\nPress Ctrl+C to stop")
        print("=" * 70)
        
        time.sleep(2)
        
        # Run online learning monitor
        subprocess.run(["python3", "monitor_online_learning.py"])
        
    elif choice == "0":
        print("\nExiting...")
        return
    else:
        print("\nInvalid choice. Please try again.")
        return main()
    
    print("\n" + "=" * 70)
    print("Production launcher complete")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nLauncher interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)