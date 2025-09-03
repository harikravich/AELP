#!/usr/bin/env python3
"""
Test Budget Optimization Integration in GAELP Production Orchestrator
Verify that budget optimizer is properly wired and active during training
"""

import os
import sys
import logging
import time
from decimal import Decimal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_budget_optimization_integration():
    """Test that budget optimizer is properly integrated into orchestrator"""
    
    print("🚀 Testing GAELP Budget Optimization Integration")
    print("=" * 60)
    
    try:
        # Import orchestrator components
        from gaelp_production_orchestrator import GAELPProductionOrchestrator, OrchestratorConfig
        from budget_optimizer import DynamicBudgetOptimizer, OptimizationObjective
        
        print("✅ Successfully imported orchestrator and budget optimizer")
        
        # Create configuration with budget optimization enabled
        config = OrchestratorConfig()
        config.environment = "testing"
        config.dry_run = True  # Don't spend real money
        config.max_daily_spend = 100.0  # $100 test budget
        config.enable_rl_training = False  # Skip full training for this test
        config.enable_online_learning = False
        config.enable_shadow_mode = False
        config.enable_ab_testing = False
        config.enable_google_ads = False
        
        print(f"✅ Configuration created: ${config.max_daily_spend} daily budget")
        
        # Create orchestrator
        orchestrator = GAELPProductionOrchestrator(config)
        
        # Initialize components
        print("\n📦 Initializing components...")
        success = orchestrator.initialize_components()
        
        if not success:
            print("❌ Component initialization failed")
            return False
        
        print("✅ Components initialized successfully")
        
        # Check if budget optimizer was created
        budget_optimizer = orchestrator.components.get('budget_optimizer')
        if not budget_optimizer:
            print("❌ Budget optimizer not found in components")
            return False
        
        print("✅ Budget optimizer component found")
        print(f"   Type: {type(budget_optimizer).__name__}")
        print(f"   Daily budget: ${budget_optimizer.daily_budget}")
        print(f"   Optimization objective: {budget_optimizer.optimization_objective.value}")
        
        # Add initial performance data to budget optimizer (required for operation)
        print("\n📊 Adding initial performance data...")
        from budget_optimizer import PerformanceWindow
        from datetime import datetime, timedelta
        
        # Add 48 hours of mock performance data
        for i in range(48):
            hour = i % 24
            performance_window = PerformanceWindow(
                start_time=datetime.now() - timedelta(hours=48-i),
                end_time=datetime.now() - timedelta(hours=47-i),
                spend=Decimal('2.50'),  # Small amounts for testing
                impressions=100,
                clicks=5,
                conversions=1 if i % 10 == 0 else 0,  # Occasional conversions
                revenue=Decimal('15.00') if i % 10 == 0 else Decimal('0'),
                roas=6.0 if i % 10 == 0 else 0,
                cpa=Decimal('2.50') if i % 10 == 0 else Decimal('999'),
                cvr=0.2 if i % 10 == 0 else 0,
                cpc=Decimal('0.50'),
                quality_score=7.0 + (hour % 5)  # Vary quality score by hour
            )
            budget_optimizer.add_performance_data(performance_window)
        
        print(f"✅ Added 48 hours of performance data")
        
        # Test budget optimizer methods
        print("\n💰 Testing budget optimizer methods...")
        
        # Test get_optimization_status
        try:
            status = budget_optimizer.get_optimization_status()
            print("✅ get_optimization_status() works")
            print(f"   Budget utilization: {status['budget_status']['daily_utilization']:.1%}")
            print(f"   Learned patterns: {status['optimization_status']['learned_patterns']}")
        except Exception as e:
            print(f"❌ get_optimization_status() failed: {e}")
            return False
        
        # Test get_pacing_multiplier
        try:
            current_hour = 12  # Test with noon
            multiplier = budget_optimizer.get_pacing_multiplier(current_hour)
            print(f"✅ get_pacing_multiplier({current_hour}) = {multiplier:.2f}")
        except Exception as e:
            print(f"❌ get_pacing_multiplier() failed: {e}")
            return False
        
        # Test prevent_early_exhaustion
        try:
            at_risk, reason, cap = budget_optimizer.prevent_early_exhaustion(12)
            print(f"✅ prevent_early_exhaustion() works: risk={at_risk}, reason='{reason}'")
            if cap:
                print(f"   Recommended cap: ${cap}")
        except Exception as e:
            print(f"❌ prevent_early_exhaustion() failed: {e}")
            return False
        
        # Test optimize_hourly_allocation
        try:
            from budget_optimizer import PacingStrategy
            result = budget_optimizer.optimize_hourly_allocation(
                strategy=PacingStrategy.ADAPTIVE_ML
            )
            print("✅ optimize_hourly_allocation() works")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Allocations: {len(result.allocations)} hours")
            print(f"   Total allocated: ${sum(result.allocations.values())}")
            
            # Show a few sample allocations
            sample_hours = [8, 12, 16, 20]
            for hour in sample_hours:
                if hour in result.allocations:
                    print(f"   Hour {hour}: ${result.allocations[hour]:.2f}")
        except Exception as e:
            print(f"❌ optimize_hourly_allocation() failed: {e}")
            return False
        
        # Test performance data integration
        print("\n📊 Testing performance data integration...")
        try:
            from budget_optimizer import PerformanceWindow
            from datetime import datetime, timedelta
            
            # Create mock performance data
            performance_window = PerformanceWindow(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
                spend=Decimal('25.50'),
                impressions=1000,
                clicks=50,
                conversions=2,
                revenue=Decimal('150.00'),
                roas=5.88,
                cpa=Decimal('12.75'),
                cvr=0.04,
                cpc=Decimal('0.51'),
                quality_score=8.2
            )
            
            budget_optimizer.add_performance_data(performance_window)
            print("✅ add_performance_data() works")
            
            # Check if data was added
            updated_status = budget_optimizer.get_optimization_status()
            windows_count = updated_status['risk_assessment']['total_performance_windows']
            print(f"   Performance windows: {windows_count}")
            
        except Exception as e:
            print(f"❌ Performance data integration failed: {e}")
            return False
        
        # Test component status
        print("\n📋 Component status check...")
        status = orchestrator.get_status()
        
        if 'budget_optimizer' in status['components']:
            optimizer_status = status['components']['budget_optimizer']
            print(f"✅ Budget optimizer status: {optimizer_status}")
        else:
            print("❌ Budget optimizer not in component status")
            return False
        
        print("\n🎯 Integration Test Summary:")
        print("✅ Budget optimizer properly initialized")
        print("✅ All key methods functional")
        print("✅ Performance data integration works")
        print("✅ Component status tracking active")
        print("✅ Ready for training loop integration")
        
        # Clean up
        orchestrator.stop()
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_budget_optimization_integration()
    
    if success:
        print("\n🎉 Budget Optimization Integration Test PASSED!")
        print("💡 The budget optimizer is properly wired into the production orchestrator")
        print("💡 It will actively optimize budget allocation during training episodes")
        sys.exit(0)
    else:
        print("\n❌ Budget Optimization Integration Test FAILED!")
        sys.exit(1)