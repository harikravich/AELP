#!/usr/bin/env python3
"""
SHADOW MODE LAUNCHER
Complete launcher for production-grade shadow mode testing
"""

import asyncio
import logging
import argparse
import json
import time
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/home/hariravichandran/AELP')

from shadow_mode_manager import ShadowModeManager, ShadowTestConfiguration
from shadow_mode_dashboard import ShadowModeDashboard
from emergency_controls import get_emergency_controller, emergency_stop_decorator

logger = logging.getLogger(__name__)

class ShadowModeLauncher:
    """
    Complete launcher for shadow mode testing
    """
    
    def __init__(self):
        self.emergency_controller = get_emergency_controller()
        self.manager: Optional[ShadowModeManager] = None
        self.dashboard: Optional[ShadowModeDashboard] = None
        self.dashboard_process: Optional[subprocess.Popen] = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'shadow_mode_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Shadow Mode Launcher initialized")
    
    def create_test_configurations(self) -> Dict[str, ShadowTestConfiguration]:
        """Create predefined test configurations"""
        
        configs = {}
        
        # Quick validation test (15 minutes)
        configs['quick'] = ShadowTestConfiguration(
            test_name="Quick_Validation_Test",
            duration_hours=0.25,  # 15 minutes
            models={
                'production': {
                    'model_id': 'production_gaelp_v1.2',
                    'learning_rate': 1e-4,
                    'epsilon': 0.05,
                    'bid_bias': 1.0,
                    'exploration_rate': 0.05,
                    'risk_tolerance': 0.4,
                    'creative_preference': 'conservative',
                    'channel_preference': 'balanced',
                    'description': 'Current production model'
                },
                'shadow': {
                    'model_id': 'shadow_gaelp_v2.0',
                    'learning_rate': 2e-4,
                    'epsilon': 0.12,
                    'bid_bias': 1.1,
                    'exploration_rate': 0.12,
                    'risk_tolerance': 0.6,
                    'creative_preference': 'aggressive',
                    'channel_preference': 'search_focused',
                    'description': 'New experimental model'
                }
            },
            traffic_percentage=1.0,
            comparison_threshold=0.15,
            statistical_confidence=0.90,
            min_sample_size=50,
            save_all_decisions=True,
            real_time_reporting=True
        )
        
        # Standard production test (2 hours)
        configs['standard'] = ShadowTestConfiguration(
            test_name="Standard_Production_Test",
            duration_hours=2.0,
            models={
                'production': {
                    'model_id': 'production_gaelp_v1.2',
                    'learning_rate': 1e-4,
                    'epsilon': 0.05,
                    'bid_bias': 1.0,
                    'exploration_rate': 0.05,
                    'risk_tolerance': 0.4,
                    'creative_preference': 'conservative',
                    'channel_preference': 'balanced',
                    'description': 'Current production model'
                },
                'shadow': {
                    'model_id': 'shadow_gaelp_v2.0',
                    'learning_rate': 2e-4,
                    'epsilon': 0.12,
                    'bid_bias': 1.1,
                    'exploration_rate': 0.12,
                    'risk_tolerance': 0.6,
                    'creative_preference': 'aggressive',
                    'channel_preference': 'search_focused',
                    'description': 'New experimental model'
                },
                'baseline': {
                    'model_id': 'random_baseline_v1.0',
                    'learning_rate': 1e-4,
                    'epsilon': 0.3,
                    'bid_bias': 0.8,
                    'exploration_rate': 0.3,
                    'risk_tolerance': 0.5,
                    'creative_preference': 'balanced',
                    'channel_preference': 'balanced',
                    'description': 'Random baseline for comparison'
                }
            },
            traffic_percentage=1.0,
            comparison_threshold=0.15,
            statistical_confidence=0.95,
            min_sample_size=200,
            save_all_decisions=True,
            real_time_reporting=True
        )
        
        # Comprehensive A/B test (8 hours)
        configs['comprehensive'] = ShadowTestConfiguration(
            test_name="Comprehensive_AB_Test",
            duration_hours=8.0,
            models={
                'production': {
                    'model_id': 'production_gaelp_v1.2',
                    'learning_rate': 1e-4,
                    'epsilon': 0.05,
                    'bid_bias': 1.0,
                    'exploration_rate': 0.05,
                    'risk_tolerance': 0.4,
                    'creative_preference': 'conservative',
                    'channel_preference': 'balanced',
                    'description': 'Current production model'
                },
                'shadow_v1': {
                    'model_id': 'shadow_gaelp_v2.0',
                    'learning_rate': 2e-4,
                    'epsilon': 0.12,
                    'bid_bias': 1.1,
                    'exploration_rate': 0.12,
                    'risk_tolerance': 0.6,
                    'creative_preference': 'aggressive',
                    'channel_preference': 'search_focused',
                    'description': 'Aggressive experimental model'
                },
                'shadow_v2': {
                    'model_id': 'shadow_gaelp_v2.1',
                    'learning_rate': 1.5e-4,
                    'epsilon': 0.08,
                    'bid_bias': 0.95,
                    'exploration_rate': 0.08,
                    'risk_tolerance': 0.3,
                    'creative_preference': 'balanced',
                    'channel_preference': 'display_focused',
                    'description': 'Conservative experimental model'
                },
                'baseline': {
                    'model_id': 'random_baseline_v1.0',
                    'learning_rate': 1e-4,
                    'epsilon': 0.3,
                    'bid_bias': 0.8,
                    'exploration_rate': 0.3,
                    'risk_tolerance': 0.5,
                    'creative_preference': 'balanced',
                    'channel_preference': 'balanced',
                    'description': 'Random baseline'
                }
            },
            traffic_percentage=1.0,
            comparison_threshold=0.10,
            statistical_confidence=0.99,
            min_sample_size=1000,
            save_all_decisions=True,
            real_time_reporting=True
        )
        
        return configs
    
    def display_menu(self):
        """Display interactive menu"""
        print("\n" + "="*80)
        print("GAELP SHADOW MODE TESTING - PRODUCTION GRADE".center(80))
        print("="*80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        print("\n🎯 SHADOW MODE OPTIONS:")
        print("1. Quick Validation Test (15 minutes)")
        print("   - Production vs Shadow model")
        print("   - Fast validation of new model")
        print("   - 90% statistical confidence")
        print()
        
        print("2. Standard Production Test (2 hours)")
        print("   - Production vs Shadow vs Baseline")
        print("   - Comprehensive comparison")
        print("   - 95% statistical confidence")
        print()
        
        print("3. Comprehensive A/B Test (8 hours)")
        print("   - Multiple shadow model variants")
        print("   - Full statistical analysis")
        print("   - 99% statistical confidence")
        print()
        
        print("4. Custom Configuration")
        print("   - Define your own test parameters")
        print("   - Advanced options")
        print()
        
        print("5. 📊 Monitor Existing Test")
        print("   - Real-time dashboard")
        print("   - View ongoing test results")
        print()
        
        print("6. 📈 Analyze Previous Test")
        print("   - Load and analyze test results")
        print("   - Generate reports")
        print()
        
        print("0. Exit")
        print()
    
    async def run_shadow_test(self, config: ShadowTestConfiguration, with_dashboard: bool = True):
        """Run shadow test with optional dashboard"""
        
        logger.info(f"Starting shadow test: {config.test_name}")
        logger.info(f"Duration: {config.duration_hours} hours")
        logger.info(f"Models: {list(config.models.keys())}")
        
        # Initialize emergency controls
        if not self.emergency_controller.is_system_healthy():
            logger.error("Emergency system indicates unhealthy state - cannot start")
            return False
        
        # Create manager
        self.manager = ShadowModeManager(config)
        
        # Start dashboard if requested
        dashboard_task = None
        if with_dashboard:
            dashboard_task = asyncio.create_task(self._run_dashboard())
        
        try:
            # Run shadow testing
            await self.manager.run_shadow_testing()
            
            # Get results
            results = self.manager.get_test_results()
            
            # Display summary
            self._display_test_results(results)
            
            logger.info("Shadow testing completed successfully")
            return True
            
        except KeyboardInterrupt:
            logger.info("Shadow testing interrupted by user")
            if self.manager:
                self.manager.stop_testing()
            return False
            
        except Exception as e:
            logger.error(f"Error in shadow testing: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if dashboard_task:
                dashboard_task.cancel()
    
    async def _run_dashboard(self):
        """Run dashboard in background"""
        try:
            # Wait a bit for manager to create database
            await asyncio.sleep(10)
            
            if self.manager and hasattr(self.manager, 'db_path'):
                dashboard = ShadowModeDashboard(self.manager.db_path)
                
                # Run dashboard updates in loop
                while self.manager.is_running:
                    dashboard._load_data_from_database()
                    dashboard.update_dashboard(0)
                    
                    # Save periodic snapshots
                    if int(time.time()) % 600 == 0:  # Every 10 minutes
                        dashboard.save_dashboard_snapshot()
                    
                    await asyncio.sleep(30)  # Update every 30 seconds
                    
        except Exception as e:
            logger.error(f"Error in dashboard: {e}")
    
    def _display_test_results(self, results: Dict[str, Any]):
        """Display test results summary"""
        print("\n" + "="*80)
        print("SHADOW TESTING RESULTS".center(80))
        print("="*80)
        
        perf_report = results['performance_report']
        
        print(f"\nTest: {perf_report['test_info']['test_name']}")
        print(f"Duration: {perf_report['test_info']['runtime_minutes']:.1f} minutes")
        print(f"Total Comparisons: {results['comparison_count']}")
        
        # Model performance
        print(f"\n{'MODEL PERFORMANCE':<50}")
        print("-" * 80)
        print(f"{'Model':<20} {'Decisions':<12} {'Win Rate':<10} {'CTR':<8} {'CVR':<8} {'ROAS':<8} {'Risk':<8}")
        print("-" * 80)
        
        for model_name, metrics in perf_report['model_performance'].items():
            print(f"{model_name:<20} "
                  f"{metrics['total_decisions']:<12} "
                  f"{metrics['win_rate']:<10.3f} "
                  f"{metrics['ctr']:<8.3f} "
                  f"{metrics['cvr']:<8.3f} "
                  f"{metrics['roas']:<8.2f} "
                  f"{metrics['risk_score']:<8.3f}")
        
        # Comparisons
        if 'comparisons' in perf_report:
            comp = perf_report['comparisons']
            print(f"\nCOMPARISONS:")
            print(f"  Total Comparisons: {comp['total_comparisons']}")
            print(f"  Significant Divergences: {comp['significant_divergences']}")
            
            if comp['total_comparisons'] > 0:
                divergence_rate = comp['significant_divergences'] / comp['total_comparisons']
                print(f"  Divergence Rate: {divergence_rate:.1%}")
        
        # Statistical results
        if perf_report.get('statistical_results'):
            stats = perf_report['statistical_results']
            print(f"\nSTATISTICAL ANALYSIS:")
            print(f"  Average Bid Divergence: {stats.get('avg_bid_divergence', 0):.3f}")
            print(f"  Significant Divergence Rate: {stats.get('significant_divergence_rate', 0):.1%}")
            print(f"  Average Value Difference: ${stats.get('avg_value_difference', 0):.2f}")
        
        print(f"\nDatabase saved: {results['database_path']}")
        print("="*80)
    
    def create_custom_config(self) -> Optional[ShadowTestConfiguration]:
        """Create custom configuration interactively"""
        print("\n" + "="*60)
        print("CUSTOM SHADOW TEST CONFIGURATION")
        print("="*60)
        
        try:
            # Basic settings
            test_name = input("Test Name: ").strip() or "Custom_Shadow_Test"
            duration = float(input("Duration (hours) [2.0]: ") or "2.0")
            
            # Model configuration
            models = {}
            
            print(f"\nConfigure models (minimum 2 required):")
            model_count = int(input("Number of models [2]: ") or "2")
            
            for i in range(model_count):
                print(f"\nModel {i+1}:")
                model_name = input(f"  Model name: ").strip() or f"model_{i+1}"
                model_id = input(f"  Model ID: ").strip() or f"{model_name}_v1.0"
                
                # Advanced settings
                print(f"  Advanced settings (press Enter for defaults):")
                learning_rate = float(input(f"    Learning rate [1e-4]: ") or "1e-4")
                epsilon = float(input(f"    Epsilon [0.1]: ") or "0.1")
                bid_bias = float(input(f"    Bid bias [1.0]: ") or "1.0")
                
                models[model_name] = {
                    'model_id': model_id,
                    'learning_rate': learning_rate,
                    'epsilon': epsilon,
                    'bid_bias': bid_bias,
                    'exploration_rate': epsilon,
                    'risk_tolerance': 0.5,
                    'creative_preference': 'balanced',
                    'channel_preference': 'balanced',
                    'description': f'Custom model {i+1}'
                }
            
            # Statistical settings
            print(f"\nStatistical settings:")
            threshold = float(input("Comparison threshold [0.15]: ") or "0.15")
            confidence = float(input("Statistical confidence [0.95]: ") or "0.95")
            min_samples = int(input("Minimum sample size [100]: ") or "100")
            
            # Create configuration
            config = ShadowTestConfiguration(
                test_name=test_name,
                duration_hours=duration,
                models=models,
                traffic_percentage=1.0,
                comparison_threshold=threshold,
                statistical_confidence=confidence,
                min_sample_size=min_samples,
                save_all_decisions=True,
                real_time_reporting=True
            )
            
            print(f"\nConfiguration created successfully!")
            print(f"Test: {config.test_name}")
            print(f"Duration: {config.duration_hours} hours")
            print(f"Models: {list(config.models.keys())}")
            
            return config
            
        except (ValueError, KeyboardInterrupt):
            print("\nConfiguration cancelled.")
            return None
    
    def monitor_existing_test(self):
        """Monitor existing test with dashboard"""
        print("\n" + "="*60)
        print("MONITOR EXISTING SHADOW TEST")
        print("="*60)
        
        # List available databases
        db_files = list(Path('.').glob('shadow_testing_*.db'))
        
        if not db_files:
            print("No shadow test databases found in current directory.")
            return
        
        print("\nAvailable test databases:")
        for i, db_file in enumerate(db_files, 1):
            print(f"{i}. {db_file.name}")
        
        try:
            choice = int(input(f"\nSelect database (1-{len(db_files)}): "))
            if 1 <= choice <= len(db_files):
                db_path = str(db_files[choice - 1])
                
                print(f"\nStarting dashboard for: {db_path}")
                print("Press Ctrl+C to stop monitoring.")
                
                # Start dashboard
                dashboard = ShadowModeDashboard(db_path)
                dashboard.start_monitoring()
            else:
                print("Invalid selection.")
                
        except (ValueError, KeyboardInterrupt):
            print("\nMonitoring cancelled.")
    
    def analyze_previous_test(self):
        """Analyze results from previous test"""
        print("\n" + "="*60)
        print("ANALYZE PREVIOUS SHADOW TEST")
        print("="*60)
        
        # List available databases
        db_files = list(Path('.').glob('shadow_testing_*.db'))
        
        if not db_files:
            print("No shadow test databases found in current directory.")
            return
        
        print("\nAvailable test databases:")
        for i, db_file in enumerate(db_files, 1):
            # Get file info
            stat = db_file.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            size_mb = stat.st_size / (1024 * 1024)
            print(f"{i}. {db_file.name} ({size_mb:.1f}MB, modified {mod_time.strftime('%Y-%m-%d %H:%M')})")
        
        try:
            choice = int(input(f"\nSelect database (1-{len(db_files)}): "))
            if 1 <= choice <= len(db_files):
                db_path = str(db_files[choice - 1])
                
                print(f"\nAnalyzing: {db_path}")
                
                # Create dashboard for analysis
                dashboard = ShadowModeDashboard(db_path)
                dashboard._load_data_from_database()
                
                # Generate and display summary
                summary = dashboard.generate_summary_report()
                print("\nSUMMARY REPORT:")
                print("=" * 50)
                print(json.dumps(summary, indent=2, default=str))
                
                # Save snapshot
                snapshot_file = dashboard.save_dashboard_snapshot()
                print(f"\nDashboard snapshot saved: {snapshot_file}")
                
                # Ask if user wants interactive dashboard
                if input("\nStart interactive dashboard? (y/N): ").lower().startswith('y'):
                    dashboard.start_monitoring()
            else:
                print("Invalid selection.")
                
        except (ValueError, KeyboardInterrupt):
            print("\nAnalysis cancelled.")
    
    async def run_interactive(self):
        """Run interactive mode"""
        configs = self.create_test_configurations()
        
        while True:
            self.display_menu()
            
            try:
                choice = input("Enter choice (0-6): ").strip()
                
                if choice == "0":
                    print("\nExiting...")
                    break
                
                elif choice == "1":
                    # Quick test
                    print(f"\nStarting Quick Validation Test...")
                    with_dashboard = input("Start real-time dashboard? (Y/n): ").lower() != 'n'
                    await self.run_shadow_test(configs['quick'], with_dashboard)
                
                elif choice == "2":
                    # Standard test
                    print(f"\nStarting Standard Production Test...")
                    with_dashboard = input("Start real-time dashboard? (Y/n): ").lower() != 'n'
                    await self.run_shadow_test(configs['standard'], with_dashboard)
                
                elif choice == "3":
                    # Comprehensive test
                    print(f"\nStarting Comprehensive A/B Test...")
                    with_dashboard = input("Start real-time dashboard? (Y/n): ").lower() != 'n'
                    await self.run_shadow_test(configs['comprehensive'], with_dashboard)
                
                elif choice == "4":
                    # Custom config
                    custom_config = self.create_custom_config()
                    if custom_config:
                        with_dashboard = input("Start real-time dashboard? (Y/n): ").lower() != 'n'
                        await self.run_shadow_test(custom_config, with_dashboard)
                
                elif choice == "5":
                    # Monitor existing
                    self.monitor_existing_test()
                
                elif choice == "6":
                    # Analyze previous
                    self.analyze_previous_test()
                
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\nOperation cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Error: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.manager:
            self.manager.stop_testing()
        
        if self.dashboard_process:
            self.dashboard_process.terminate()

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='GAELP Shadow Mode Testing - Production Grade',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_shadow_mode.py                    # Interactive mode
  python launch_shadow_mode.py --test quick       # Quick validation test
  python launch_shadow_mode.py --test standard    # Standard production test
  python launch_shadow_mode.py --monitor db.db    # Monitor existing test
        """
    )
    
    parser.add_argument('--test', choices=['quick', 'standard', 'comprehensive'],
                       help='Run predefined test')
    parser.add_argument('--config', type=str, help='Path to custom config JSON file')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable real-time dashboard')
    parser.add_argument('--monitor', type=str, help='Monitor existing test database')
    parser.add_argument('--analyze', type=str, help='Analyze previous test database')
    
    args = parser.parse_args()
    
    launcher = ShadowModeLauncher()
    
    try:
        if args.monitor:
            # Monitor mode
            if Path(args.monitor).exists():
                dashboard = ShadowModeDashboard(args.monitor)
                dashboard.start_monitoring()
            else:
                print(f"Database not found: {args.monitor}")
                
        elif args.analyze:
            # Analysis mode
            if Path(args.analyze).exists():
                dashboard = ShadowModeDashboard(args.analyze)
                dashboard._load_data_from_database()
                summary = dashboard.generate_summary_report()
                print(json.dumps(summary, indent=2, default=str))
                dashboard.save_dashboard_snapshot()
            else:
                print(f"Database not found: {args.analyze}")
                
        elif args.test:
            # Predefined test
            configs = launcher.create_test_configurations()
            config = configs.get(args.test)
            if config:
                await launcher.run_shadow_test(config, not args.no_dashboard)
            else:
                print(f"Unknown test: {args.test}")
                
        elif args.config:
            # Custom config file
            if Path(args.config).exists():
                with open(args.config) as f:
                    config_data = json.load(f)
                config = ShadowTestConfiguration(**config_data)
                await launcher.run_shadow_test(config, not args.no_dashboard)
            else:
                print(f"Config file not found: {args.config}")
                
        else:
            # Interactive mode
            await launcher.run_interactive()
    
    except KeyboardInterrupt:
        print("\n\nShadow mode testing interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        launcher.cleanup()

if __name__ == "__main__":
    asyncio.run(main())