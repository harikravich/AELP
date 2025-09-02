#!/usr/bin/env python3
"""
Diagnostic script to trace through GAELP simulation and verify if agent is actually learning
"""

import sys
import torch
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_simulation_components():
    """Check if all simulation components are properly connected"""
    
    logger.info("=" * 80)
    logger.info("GAELP LEARNING DIAGNOSTIC")
    logger.info("=" * 80)
    
    # 1. Check RecSim Integration
    logger.info("\n1. CHECKING RECSIM INTEGRATION:")
    try:
        from recsim_user_model import RecSimUserModel, UserSegment
        user_model = RecSimUserModel()
        logger.info("✓ RecSim user model loaded")
        
        # Test user simulation
        test_user = user_model.generate_user(UserSegment.IMPULSE_BUYER)
        logger.info(f"✓ Generated test user with click_prob={test_user.get('click_propensity', 0):.3f}")
    except Exception as e:
        logger.error(f"✗ RecSim integration FAILED: {e}")
        return False
    
    # 2. Check Auction System
    logger.info("\n2. CHECKING AUCTION SYSTEM:")
    try:
        from fixed_auction_system import FixedAuctionSystem
        auction = FixedAuctionSystem()
        
        # Run test auction
        result = auction.run_auction(
            your_bid=2.5,
            quality_score=7.5,
            context={'user_segment': 'test'},
            user_id='test_user'
        )
        logger.info(f"✓ Auction system working - Won: {result['won']}, Position: {result['position']}")
        
        # Check if competitors are bidding
        if 'competitors' in result:
            competitor_bids = [c['bid'] for c in result['competitors']]
            logger.info(f"✓ Competitors bidding: {competitor_bids[:3]}...")
        else:
            logger.warning("⚠ No competitor information in auction result")
            
    except Exception as e:
        logger.error(f"✗ Auction system FAILED: {e}")
        return False
    
    # 3. Check User Journey Database
    logger.info("\n3. CHECKING USER JOURNEY DATABASE:")
    try:
        from user_journey_database import UserJourneyDatabase
        db = UserJourneyDatabase()
        logger.info(f"✓ User journey database initialized with {db.get_active_user_count()} users")
    except Exception as e:
        logger.error(f"✗ User journey database FAILED: {e}")
        return False
    
    # 4. Check Criteo Response Model
    logger.info("\n4. CHECKING CRITEO RESPONSE MODEL:")
    try:
        from criteo_response_model import CriteoUserResponseModel
        criteo = CriteoUserResponseModel()
        
        # Test response prediction
        test_response = criteo.predict_response(
            user_features={'segment': 'test'},
            ad_features={'creative_type': 'test'},
            context={'time_of_day': 12}
        )
        logger.info(f"✓ Criteo model predicting CTR={test_response.get('ctr', 0):.4f}")
    except Exception as e:
        logger.error(f"✗ Criteo response model FAILED: {e}")
        return False
    
    return True

def check_rl_agent_learning():
    """Check if RL agent is actually updating weights"""
    
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING RL AGENT LEARNING")
    logger.info("=" * 80)
    
    try:
        from training_orchestrator.rl_agent_proper import RLAgent, JourneyState
        
        # Create agent
        agent = RLAgent(state_dim=17, bid_actions=10, creative_actions=5)
        
        # Get initial weights
        initial_weights = {}
        for name, param in agent.q_network.named_parameters():
            initial_weights[name] = param.data.clone()
        
        logger.info(f"✓ RL Agent initialized with {len(initial_weights)} parameter groups")
        
        # Create fake experiences and train
        logger.info("\nGenerating training experiences...")
        for i in range(100):
            # Create fake state
            state = JourneyState(
                stage=np.random.randint(1, 4),
                touchpoints_seen=np.random.randint(0, 10),
                days_since_first_touch=np.random.uniform(0, 30),
                ad_fatigue_level=np.random.uniform(0, 1),
                segment='test',
                device='desktop',
                hour_of_day=12,
                day_of_week=3,
                previous_clicks=np.random.randint(0, 5),
                previous_impressions=np.random.randint(0, 20),
                estimated_ltv=np.random.uniform(50, 200),
                competition_level=np.random.uniform(0, 1),
                channel_performance=np.random.uniform(0, 1)
            )
            
            # Store experience
            action = np.random.randint(0, 10)
            reward = np.random.uniform(-1, 1)
            next_state = state  # Simplified for testing
            
            agent.store_experience(state, action, reward, next_state, done=False)
        
        logger.info(f"✓ Generated {len(agent.replay_buffer)} experiences")
        
        # Train the agent
        logger.info("\nTraining agent...")
        for epoch in range(10):
            agent.train_dqn(batch_size=32)
        
        # Check if weights changed
        logger.info("\nChecking weight updates...")
        weights_changed = False
        max_change = 0
        for name, param in agent.q_network.named_parameters():
            change = torch.abs(param.data - initial_weights[name]).max().item()
            max_change = max(max_change, change)
            if change > 1e-6:
                weights_changed = True
                logger.info(f"  {name}: max change = {change:.6f}")
        
        if weights_changed:
            logger.info(f"\n✓ WEIGHTS ARE UPDATING! Max change: {max_change:.6f}")
            return True
        else:
            logger.error(f"\n✗ WEIGHTS NOT UPDATING! Max change: {max_change:.9f}")
            return False
            
    except Exception as e:
        logger.error(f"✗ RL Agent check FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_reward_flow():
    """Trace reward flow from environment to agent"""
    
    logger.info("\n" + "=" * 80)
    logger.info("TRACING REWARD FLOW")
    logger.info("=" * 80)
    
    try:
        from enhanced_simulator_fixed import FixedGAELPEnvironment
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        # Create minimal config
        config = GAELPConfig(
            enable_dashboard=False,
            enable_monte_carlo=False,
            enable_online_learning=False
        )
        
        # Create orchestrator
        logger.info("Creating master orchestrator...")
        orchestrator = MasterOrchestrator(config)
        
        # Run a single step and trace rewards
        logger.info("\nRunning simulation step...")
        result = orchestrator.step_fixed_environment()
        
        if 'reward' in result:
            logger.info(f"✓ Reward generated: {result['reward']:.4f}")
            
            # Check reward components
            if 'metrics' in result:
                metrics = result['metrics']
                logger.info("  Reward components:")
                logger.info(f"    - Impressions: {metrics.get('impressions', 0)}")
                logger.info(f"    - Clicks: {metrics.get('clicks', 0)}")
                logger.info(f"    - Conversions: {metrics.get('conversions', 0)}")
                logger.info(f"    - Cost: ${metrics.get('cost', 0):.2f}")
                logger.info(f"    - Revenue: ${metrics.get('revenue', 0):.2f}")
        else:
            logger.warning("⚠ No reward in simulation result")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Reward flow check FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks"""
    
    logger.info("Starting GAELP Learning Diagnostics...")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Run checks
    components_ok = check_simulation_components()
    learning_ok = check_rl_agent_learning()
    reward_ok = check_reward_flow()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    
    if components_ok:
        logger.info("✓ Simulation components: WORKING")
    else:
        logger.error("✗ Simulation components: BROKEN")
    
    if learning_ok:
        logger.info("✓ RL Agent learning: WORKING")
    else:
        logger.error("✗ RL Agent learning: NOT WORKING")
    
    if reward_ok:
        logger.info("✓ Reward flow: WORKING")
    else:
        logger.error("✗ Reward flow: BROKEN")
    
    if components_ok and learning_ok and reward_ok:
        logger.info("\n✅ SYSTEM READY FOR TRAINING")
    else:
        logger.error("\n❌ SYSTEM NOT READY - FIX ISSUES ABOVE")
    
    return components_ok and learning_ok and reward_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)