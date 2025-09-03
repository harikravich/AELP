#!/usr/bin/env python3
"""
Quick verification that budget optimizer is properly wired into orchestrator
"""

def verify_budget_integration():
    """Verify budget optimization is properly integrated"""
    
    try:
        from gaelp_production_orchestrator import GAELPProductionOrchestrator, OrchestratorConfig
        
        # Create test config
        config = OrchestratorConfig()
        config.dry_run = True
        config.max_daily_spend = 500.0
        
        # Create orchestrator
        orchestrator = GAELPProductionOrchestrator(config)
        
        # Initialize components
        if not orchestrator.initialize_components():
            print("❌ Component initialization failed")
            return False
        
        # Check budget optimizer
        budget_optimizer = orchestrator.components.get('budget_optimizer')
        if not budget_optimizer:
            print("❌ Budget optimizer not found")
            return False
        
        # Verify it has the right budget
        if budget_optimizer.daily_budget != 500.0:
            print(f"❌ Wrong daily budget: {budget_optimizer.daily_budget}")
            return False
        
        # Check methods exist
        required_methods = [
            'optimize_hourly_allocation',
            'get_pacing_multiplier',
            'prevent_early_exhaustion',
            'add_performance_data',
            'get_optimization_status'
        ]
        
        for method_name in required_methods:
            if not hasattr(budget_optimizer, method_name):
                print(f"❌ Missing method: {method_name}")
                return False
        
        orchestrator.stop()
        
        print("✅ Budget optimization is properly wired into orchestrator")
        print(f"   Daily budget: ${budget_optimizer.daily_budget}")
        print(f"   All required methods present")
        print(f"   Component status: {orchestrator.component_status.get('budget_optimizer', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_budget_integration()
    exit(0 if success else 1)