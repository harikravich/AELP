#!/usr/bin/env python3
"""
SIMPLIFIED SHADOW MODE SYSTEM TEST
Test the core shadow mode functionality
"""

import asyncio
import logging
import json
import time
import sys
import os
from datetime import datetime
import numpy as np

# Add current directory to Python path
sys.path.insert(0, '/home/hariravichandran/AELP')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_shadow_mode_basic():
    """Test basic shadow mode functionality"""
    logger.info("="*80)
    logger.info("SHADOW MODE BASIC FUNCTIONALITY TEST")
    logger.info("="*80)
    
    # Test 1: State management
    logger.info("\n1. Testing State Management...")
    try:
        from shadow_mode_state import DynamicEnrichedState, create_synthetic_state_for_testing, batch_create_synthetic_states
        
        # Create synthetic state
        state = create_synthetic_state_for_testing()
        assert isinstance(state, DynamicEnrichedState)
        logger.info(f"✅ Created synthetic state: {state.segment_name}")
        
        # Test vector conversion
        vector = state.to_vector()
        assert len(vector) == 53
        logger.info(f"✅ Vector conversion: {len(vector)} dimensions")
        
        # Test batch creation
        batch_states = batch_create_synthetic_states(100)
        assert len(batch_states) == 100
        logger.info(f"✅ Batch creation: {len(batch_states)} states")
        
        # Test serialization
        state_dict = state.to_dict()
        restored = DynamicEnrichedState.from_dict(state_dict)
        assert restored.segment_name == state.segment_name
        logger.info("✅ Serialization works")
        
        logger.info("State Management: PASSED")
        
    except Exception as e:
        logger.error(f"State Management: FAILED - {e}")
        return False
    
    # Test 2: Environment simulation
    logger.info("\n2. Testing Environment Simulation...")
    try:
        from shadow_mode_environment import ShadowModeEnvironment
        from gaelp_parameter_manager import ParameterManager
        
        pm = ParameterManager()
        env = ShadowModeEnvironment(pm)
        
        # Test reset
        obs, info = env.reset()
        assert 'state' in obs
        logger.info("✅ Environment reset works")
        
        # Test step
        class MockAction:
            def __init__(self):
                self.bid_amount = 2.5
                self.creative_id = 25
                self.channel = 'paid_search'
        
        action = MockAction()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert isinstance(reward, (int, float))
        logger.info(f"✅ Environment step works, reward: {reward:.3f}")
        
        # Test metrics
        metrics = env.get_environment_metrics()
        assert 'win_rate' in metrics
        logger.info(f"✅ Environment metrics: {metrics['win_rate']:.3f} win rate")
        
        logger.info("Environment Simulation: PASSED")
        
    except Exception as e:
        logger.error(f"Environment Simulation: FAILED - {e}")
        return False
    
    # Test 3: Performance test
    logger.info("\n3. Testing Performance...")
    try:
        # Batch creation performance
        start_time = time.time()
        large_batch = batch_create_synthetic_states(1000)
        creation_time = time.time() - start_time
        assert len(large_batch) == 1000
        assert creation_time < 2.0
        logger.info(f"✅ Created 1000 states in {creation_time:.3f}s")
        
        # Vector conversion performance
        start_time = time.time()
        vectors = [state.to_vector() for state in large_batch[:100]]
        conversion_time = time.time() - start_time
        assert len(vectors) == 100
        assert conversion_time < 0.5
        logger.info(f"✅ Converted 100 vectors in {conversion_time:.3f}s")
        
        logger.info("Performance Test: PASSED")
        
    except Exception as e:
        logger.error(f"Performance Test: FAILED - {e}")
        return False
    
    # Test 4: Decision making simulation
    logger.info("\n4. Testing Decision Making...")
    try:
        # Create multiple synthetic models
        models = {}
        
        class SimpleModel:
            def __init__(self, name, bid_bias=1.0):
                self.name = name
                self.bid_bias = bid_bias
            
            def make_decision(self, user_state, context):
                base_bid = 2.0 * self.bid_bias
                value_multiplier = 1.0 + (user_state.segment_cvr - 0.02) * 10
                bid = base_bid * value_multiplier
                
                return {
                    'bid_amount': max(0.5, min(10.0, bid)),
                    'creative_id': np.random.randint(0, 50),
                    'channel': np.random.choice(['paid_search', 'display', 'social']),
                    'confidence': np.random.uniform(0.3, 0.9)
                }
        
        models['production'] = SimpleModel('production', bid_bias=1.0)
        models['shadow'] = SimpleModel('shadow', bid_bias=1.2)
        models['baseline'] = SimpleModel('baseline', bid_bias=0.8)
        
        # Test decisions
        test_state = create_synthetic_state_for_testing()
        test_context = {
            'competition_level': 0.6,
            'avg_competitor_bid': 2.5,
            'is_peak_hour': True
        }
        
        decisions = {}
        for model_name, model in models.items():
            decisions[model_name] = model.make_decision(test_state, test_context)
        
        assert len(decisions) == 3
        
        # Analyze divergence
        prod_bid = decisions['production']['bid_amount']
        shadow_bid = decisions['shadow']['bid_amount']
        divergence = abs(shadow_bid - prod_bid) / prod_bid
        
        logger.info(f"✅ Production bid: ${prod_bid:.2f}")
        logger.info(f"✅ Shadow bid: ${shadow_bid:.2f}")
        logger.info(f"✅ Bid divergence: {divergence:.3f}")
        
        logger.info("Decision Making: PASSED")
        
    except Exception as e:
        logger.error(f"Decision Making: FAILED - {e}")
        return False
    
    # Test 5: Data persistence
    logger.info("\n5. Testing Data Persistence...")
    try:
        import sqlite3
        from pathlib import Path
        
        # Create test database
        db_path = 'test_shadow_simple.db'
        conn = sqlite3.connect(db_path)
        
        # Create table
        conn.execute('''
            CREATE TABLE test_decisions (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                bid_amount REAL,
                timestamp TEXT
            )
        ''')
        
        # Insert test data
        for model_name, decision in decisions.items():
            conn.execute(
                'INSERT INTO test_decisions (model_name, bid_amount, timestamp) VALUES (?, ?, ?)',
                (model_name, decision['bid_amount'], datetime.now().isoformat())
            )
        
        conn.commit()
        
        # Verify data
        results = conn.execute('SELECT COUNT(*) FROM test_decisions').fetchone()[0]
        assert results == 3
        
        conn.close()
        Path(db_path).unlink()  # Cleanup
        
        logger.info("✅ Database operations work")
        logger.info("Data Persistence: PASSED")
        
    except Exception as e:
        logger.error(f"Data Persistence: FAILED - {e}")
        return False
    
    logger.info("\n" + "="*80)
    logger.info("ALL BASIC SHADOW MODE TESTS PASSED!")
    logger.info("="*80)
    
    return True

async def test_shadow_mode_integration():
    """Test shadow mode integration with minimal config"""
    logger.info("\n" + "="*80)
    logger.info("SHADOW MODE INTEGRATION TEST")
    logger.info("="*80)
    
    try:
        from shadow_mode_manager import ShadowModeManager, ShadowTestConfiguration
        
        # Create minimal config
        config = ShadowTestConfiguration(
            test_name="Integration_Test",
            duration_hours=0.01,  # 36 seconds
            models={
                'production': {
                    'model_id': 'test_production',
                    'bid_bias': 1.0,
                    'exploration_rate': 0.05
                },
                'shadow': {
                    'model_id': 'test_shadow',
                    'bid_bias': 1.2,
                    'exploration_rate': 0.12
                }
            },
            min_sample_size=5,
            save_all_decisions=True
        )
        
        logger.info(f"Created test config: {config.test_name}")
        
        # Initialize manager
        manager = ShadowModeManager(config)
        logger.info(f"Initialized manager with {len(manager.models)} models")
        
        # Run very short test
        logger.info("Running short integration test...")
        await manager.run_shadow_testing()
        
        # Check results
        results = manager.get_test_results()
        assert 'performance_report' in results
        
        logger.info("✅ Integration test completed")
        
        # Cleanup
        from pathlib import Path
        if Path(manager.db_path).exists():
            Path(manager.db_path).unlink()
        
        logger.info("INTEGRATION TEST: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"INTEGRATION TEST: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    logger.info("Starting Shadow Mode System Tests...")
    start_time = datetime.now()
    
    try:
        # Run basic tests
        basic_passed = await test_shadow_mode_basic()
        
        if basic_passed:
            # Run integration test
            integration_passed = await test_shadow_mode_integration()
            
            if integration_passed:
                logger.info("\n🎉 ALL SHADOW MODE TESTS PASSED!")
                success = True
            else:
                logger.error("\n❌ Integration test failed")
                success = False
        else:
            logger.error("\n❌ Basic tests failed")
            success = False
        
        duration = datetime.now() - start_time
        logger.info(f"\nTotal test duration: {duration.total_seconds():.2f} seconds")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)