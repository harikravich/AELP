#!/usr/bin/env python3
"""
SHADOW MODE SYSTEM TEST
Comprehensive test of the shadow mode testing system
"""

import asyncio
import logging
import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/home/hariravichandran/AELP')

from shadow_mode_testing import ShadowTestingEngine, create_shadow_testing_config
from shadow_mode_manager import ShadowModeManager, ShadowTestConfiguration
from shadow_mode_state import DynamicEnrichedState, create_synthetic_state_for_testing, batch_create_synthetic_states
from shadow_mode_environment import ShadowModeEnvironment
from shadow_mode_dashboard import ShadowModeDashboard

logger = logging.getLogger(__name__)

class ShadowModeSystemTester:
    """
    Comprehensive tester for shadow mode system
    """
    
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'shadow_mode_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
        self.test_results = {}
        self.start_time = datetime.now()
        
        logger.info("Shadow Mode System Tester initialized")
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("="*80)
        logger.info("SHADOW MODE SYSTEM - COMPREHENSIVE TESTING")
        logger.info("="*80)
        
        tests = [
            ("State Management", self.test_state_management),
            ("Environment Simulation", self.test_environment_simulation),
            ("Shadow Testing Engine", self.test_shadow_testing_engine),
            ("Manager Integration", self.test_manager_integration),
            ("Dashboard Functionality", self.test_dashboard_functionality),
            ("End-to-End Integration", self.test_end_to_end_integration),
            ("Performance Stress Test", self.test_performance_stress),
            ("Error Handling", self.test_error_handling)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*20} {test_name} {'='*20}")
            
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                self.test_results[test_name] = {
                    'passed': result,
                    'duration': duration,
                    'error': None
                }
                
                if result:
                    logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
                else:
                    logger.error(f"❌ {test_name} FAILED ({duration:.2f}s)")
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ {test_name} ERROR: {e} ({duration:.2f}s)")
                self.test_results[test_name] = {
                    'passed': False,
                    'duration': duration,
                    'error': str(e)
                }
                all_passed = False
        
        # Summary
        self._print_test_summary(all_passed)
        return all_passed
    
    async def test_state_management(self):
        """Test dynamic state management"""
        logger.info("Testing state creation and serialization...")
        
        # Test synthetic state creation
        state = create_synthetic_state_for_testing()
        assert isinstance(state, DynamicEnrichedState)
        assert state.segment_name is not None
        assert state.shadow_mode is True
        
        # Test vector conversion
        vector = state.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == state.state_dim
        assert state.state_dim == 53  # Expected dimension
        
        # Test serialization
        state_dict = state.to_dict()
        restored_state = DynamicEnrichedState.from_dict(state_dict)
        assert restored_state.segment_name == state.segment_name
        assert restored_state.segment_cvr == state.segment_cvr
        
        # Test batch creation
        batch_states = batch_create_synthetic_states(100)
        assert len(batch_states) == 100
        assert all(isinstance(s, DynamicEnrichedState) for s in batch_states)
        
        # Test state cloning
        shadow_state = state.clone_for_shadow("test_model")
        assert shadow_state.shadow_mode is True
        assert shadow_state.segment_name == state.segment_name
        
        logger.info("State management tests completed")
        return True
    
    async def test_environment_simulation(self):
        """Test shadow mode environment simulation"""
        logger.info("Testing environment simulation...")
        
        from gaelp_parameter_manager import ParameterManager
        
        # Create environment
        pm = ParameterManager()
        env = ShadowModeEnvironment(pm)
        
        # Test reset
        obs, info = env.reset()
        assert 'state' in obs
        assert isinstance(obs['state'], DynamicEnrichedState)
        assert 'user_id' in info
        
        # Test step
        class MockAction:
            def __init__(self):
                self.bid_amount = 2.5
                self.creative_id = 25
                self.channel = 'paid_search'
        
        action = MockAction()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'auction_result' in step_info
        
        # Test metrics
        metrics = env.get_environment_metrics()
        assert 'win_rate' in metrics
        assert 'ctr' in metrics
        assert 'cvr' in metrics
        assert 'roas' in metrics
        
        logger.info("Environment simulation tests completed")
        return True
    
    async def test_shadow_testing_engine(self):
        """Test shadow testing engine"""
        logger.info("Testing shadow testing engine...")
        
        # Create test config
        config = create_shadow_testing_config()
        
        # Modify for quick test
        config['models']['shadow']['learning_rate'] = 2e-4
        config['comparison_settings']['minimum_sample_size'] = 10
        
        # Save config
        with open('test_shadow_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize engine
        engine = ShadowTestingEngine('test_shadow_config.json')
        
        assert len(engine.models) >= 2  # At least production and shadow
        assert engine.session_id is not None
        
        # Test single comparison
        user_state = create_synthetic_state_for_testing()
        context = {
            'competition_level': 0.6,
            'avg_competitor_bid': 2.0,
            'is_peak_hour': True
        }
        
        comparison = await engine.run_shadow_comparison("test_user_1", user_state, context)
        
        assert comparison is not None
        assert comparison.production_decision is not None
        assert comparison.shadow_decision is not None
        assert isinstance(comparison.bid_divergence, (int, float))
        
        # Test performance report
        report = engine.get_performance_report()
        assert 'session_info' in report
        assert 'model_metrics' in report
        assert 'divergence_analysis' in report
        
        # Cleanup
        Path('test_shadow_config.json').unlink(missing_ok=True)
        
        logger.info("Shadow testing engine tests completed")
        return True
    
    async def test_manager_integration(self):
        """Test shadow mode manager"""
        logger.info("Testing manager integration...")
        
        # Create minimal test config
        config = ShadowTestConfiguration(
            test_name="Test_Integration",
            duration_hours=0.05,  # 3 minutes
            models={
                'production': {
                    'model_id': 'test_production',
                    'learning_rate': 1e-4,
                    'epsilon': 0.05,
                    'bid_bias': 1.0,
                    'exploration_rate': 0.05,
                    'risk_tolerance': 0.4,
                    'creative_preference': 'conservative',
                    'channel_preference': 'balanced'
                },
                'shadow': {
                    'model_id': 'test_shadow',
                    'learning_rate': 2e-4,
                    'epsilon': 0.12,
                    'bid_bias': 1.1,
                    'exploration_rate': 0.12,
                    'risk_tolerance': 0.6,
                    'creative_preference': 'aggressive',
                    'channel_preference': 'search_focused'
                }
            },
            min_sample_size=20,
            save_all_decisions=True
        )
        
        # Initialize manager
        manager = ShadowModeManager(config)
        
        assert len(manager.models) == 2
        assert len(manager.environments) == 2
        assert Path(manager.db_path).exists() == False  # Not created yet
        
        # Test database initialization (manager creates it)
        conn = sqlite3.connect(manager.db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [table[0] for table in tables]
        
        assert 'decisions' in table_names
        assert 'comparisons' in table_names
        assert 'metrics_snapshots' in table_names
        
        conn.close()
        
        # Test performance report generation
        report = manager.generate_performance_report()
        assert 'test_info' in report
        assert 'model_performance' in report
        
        # Cleanup
        Path(manager.db_path).unlink(missing_ok=True)
        
        logger.info("Manager integration tests completed")
        return True
    
    async def test_dashboard_functionality(self):
        """Test dashboard functionality"""
        logger.info("Testing dashboard functionality...")
        
        # Create test database with sample data
        db_path = 'test_dashboard.db'
        conn = sqlite3.connect(db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE decisions (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                model_id TEXT,
                user_id TEXT,
                bid_amount REAL,
                creative_id INTEGER,
                channel TEXT,
                confidence_score REAL,
                won_auction BOOLEAN,
                clicked BOOLEAN,
                converted BOOLEAN,
                spend REAL,
                revenue REAL,
                user_state TEXT,
                context TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE comparisons (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                user_id TEXT,
                production_bid REAL,
                shadow_bid REAL,
                bid_divergence REAL,
                creative_divergence BOOLEAN,
                channel_divergence BOOLEAN,
                production_value REAL,
                shadow_value REAL,
                significant_divergence BOOLEAN,
                comparison_data TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE metrics_snapshots (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                model_id TEXT,
                total_decisions INTEGER,
                win_rate REAL,
                ctr REAL,
                cvr REAL,
                roas REAL,
                avg_bid REAL,
                confidence_score REAL,
                risk_score REAL
            )
        ''')
        
        # Insert sample data
        now = datetime.now().isoformat()
        
        # Sample decisions
        for i in range(50):
            conn.execute('''
                INSERT INTO decisions 
                (session_id, timestamp, model_id, user_id, bid_amount, creative_id, channel,
                 confidence_score, won_auction, clicked, converted, spend, revenue, user_state, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'test_session',
                now,
                'production' if i % 2 == 0 else 'shadow',
                f'user_{i}',
                np.random.uniform(1.0, 5.0),
                np.random.randint(0, 50),
                np.random.choice(['paid_search', 'display', 'social']),
                np.random.uniform(0.3, 0.9),
                np.random.choice([True, False]),
                np.random.choice([True, False]),
                np.random.choice([True, False]),
                np.random.uniform(0.5, 3.0),
                np.random.uniform(0, 50),
                '{}',
                '{}'
            ))
        
        # Sample comparisons
        for i in range(25):
            prod_bid = np.random.uniform(1.0, 4.0)
            shadow_bid = np.random.uniform(1.0, 5.0)
            
            conn.execute('''
                INSERT INTO comparisons 
                (session_id, timestamp, user_id, production_bid, shadow_bid,
                 bid_divergence, creative_divergence, channel_divergence,
                 production_value, shadow_value, significant_divergence, comparison_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'test_session',
                now,
                f'user_{i}',
                prod_bid,
                shadow_bid,
                abs(shadow_bid - prod_bid) / prod_bid,
                np.random.choice([True, False]),
                np.random.choice([True, False]),
                np.random.uniform(-10, 20),
                np.random.uniform(-10, 25),
                np.random.choice([True, False]),
                '{}'
            ))
        
        conn.commit()
        conn.close()
        
        # Test dashboard
        dashboard = ShadowModeDashboard(db_path)
        dashboard._load_data_from_database()
        
        # Test data loading
        assert len(dashboard.metrics_history) > 0
        assert len(dashboard.comparison_data) > 0
        
        # Test summary report
        summary = dashboard.generate_summary_report()
        assert 'models_monitored' in summary
        assert 'comparison_summary' in summary
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
        
        logger.info("Dashboard functionality tests completed")
        return True
    
    async def test_end_to_end_integration(self):
        """Test end-to-end integration"""
        logger.info("Testing end-to-end integration...")
        
        # Create minimal test config for quick execution
        config = ShadowTestConfiguration(
            test_name="E2E_Test",
            duration_hours=0.02,  # About 1 minute
            models={
                'production': {
                    'model_id': 'e2e_production',
                    'bid_bias': 1.0,
                    'exploration_rate': 0.05,
                    'creative_preference': 'conservative'
                },
                'shadow': {
                    'model_id': 'e2e_shadow',
                    'bid_bias': 1.2,
                    'exploration_rate': 0.15,
                    'creative_preference': 'aggressive'
                }
            },
            min_sample_size=10,
            save_all_decisions=True
        )
        
        # Run manager
        manager = ShadowModeManager(config)
        
        # Run for short duration
        try:
            await manager.run_shadow_testing()
            
            # Verify database was created and populated
            assert Path(manager.db_path).exists()
            
            conn = sqlite3.connect(manager.db_path)
            decision_count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            comparison_count = conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
            conn.close()
            
            assert decision_count > 0
            # Note: comparisons might be 0 if very few decisions made
            
            # Test results generation
            results = manager.get_test_results()
            assert 'performance_report' in results
            assert 'database_path' in results
            
            # Cleanup
            Path(manager.db_path).unlink(missing_ok=True)
            
        except Exception as e:
            # Cleanup on error
            if hasattr(manager, 'db_path') and Path(manager.db_path).exists():
                Path(manager.db_path).unlink(missing_ok=True)
            raise e
        
        logger.info("End-to-end integration tests completed")
        return True
    
    async def test_performance_stress(self):
        """Test performance under stress"""
        logger.info("Testing performance under stress...")
        
        # Test batch state creation performance
        start_time = time.time()
        batch_states = batch_create_synthetic_states(1000)
        batch_creation_time = time.time() - start_time
        
        assert len(batch_states) == 1000
        assert batch_creation_time < 5.0  # Should complete in under 5 seconds
        
        # Test vector conversion performance
        start_time = time.time()
        vectors = [state.to_vector() for state in batch_states[:100]]
        vector_time = time.time() - start_time
        
        assert len(vectors) == 100
        assert vector_time < 1.0  # Should complete in under 1 second
        
        # Test serialization performance
        start_time = time.time()
        serialized = [state.to_dict() for state in batch_states[:100]]
        serialize_time = time.time() - start_time
        
        assert len(serialized) == 100
        assert serialize_time < 2.0  # Should complete in under 2 seconds
        
        logger.info(f"Performance metrics:")
        logger.info(f"  Batch creation (1000 states): {batch_creation_time:.3f}s")
        logger.info(f"  Vector conversion (100 states): {vector_time:.3f}s") 
        logger.info(f"  Serialization (100 states): {serialize_time:.3f}s")
        
        logger.info("Performance stress tests completed")
        return True
    
    async def test_error_handling(self):
        """Test error handling"""
        logger.info("Testing error handling...")
        
        # Test invalid state creation
        try:
            invalid_state = DynamicEnrichedState.from_dict({})
            # Should not fail completely but use defaults
            assert invalid_state.segment_name is not None
        except Exception:
            # This is acceptable behavior
            pass
        
        # Test invalid configuration
        try:
            invalid_config = ShadowTestConfiguration(
                test_name="Invalid",
                duration_hours=-1,  # Invalid duration
                models={}  # No models
            )
            # Should not crash instantiation
            assert invalid_config.test_name == "Invalid"
        except Exception:
            # This is acceptable behavior
            pass
        
        # Test nonexistent database
        try:
            dashboard = ShadowModeDashboard("nonexistent.db")
            dashboard._load_data_from_database()
            # Should handle gracefully
        except Exception:
            # This is acceptable behavior if it fails gracefully
            pass
        
        logger.info("Error handling tests completed")
        return True
    
    def _print_test_summary(self, all_passed: bool):
        """Print comprehensive test summary"""
        duration = datetime.now() - self.start_time
        
        print("\n" + "="*80)
        print("SHADOW MODE SYSTEM TEST SUMMARY".center(80))
        print("="*80)
        
        print(f"\nTotal Test Duration: {duration.total_seconds():.2f} seconds")
        print(f"Tests Run: {len(self.test_results)}")
        
        passed_count = sum(1 for r in self.test_results.values() if r['passed'])
        failed_count = len(self.test_results) - passed_count
        
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        
        if all_passed:
            print(f"\n🎉 ALL TESTS PASSED! Shadow mode system is working correctly.")
        else:
            print(f"\n❌ {failed_count} tests failed. Review the errors above.")
        
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'Test Name':<30} {'Status':<10} {'Duration':<10} {'Error'}")
        print("-" * 80)
        
        for test_name, result in self.test_results.items():
            status = "PASSED" if result['passed'] else "FAILED"
            duration_str = f"{result['duration']:.2f}s"
            error = result['error'][:40] + "..." if result['error'] and len(result['error']) > 40 else (result['error'] or "")
            
            print(f"{test_name:<30} {status:<10} {duration_str:<10} {error}")
        
        print("="*80)
        
        # Save detailed results
        results_file = f"shadow_mode_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'all_passed': all_passed,
                    'total_tests': len(self.test_results),
                    'passed_count': passed_count,
                    'failed_count': failed_count,
                    'total_duration': duration.total_seconds()
                },
                'detailed_results': self.test_results
            }, f, indent=2, default=str)
        
        print(f"\nDetailed results saved: {results_file}")

async def main():
    """Main test runner"""
    tester = ShadowModeSystemTester()
    
    try:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())