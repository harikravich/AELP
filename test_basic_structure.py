"""
Basic structure test for Training Orchestrator

Tests the core structure without external dependencies
"""

import sys
import os
from pathlib import Path

# Add the AELP directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that basic imports work"""
    try:
        # Test enum imports (should work without external deps)
        from training_orchestrator.phases import TrainingPhase
        print("✅ TrainingPhase enum imported successfully")
        
        # Test configuration (should work without external deps)
        from training_orchestrator.config import TrainingOrchestratorConfig
        print("✅ TrainingOrchestratorConfig imported successfully")
        
        # Create a basic config
        config = TrainingOrchestratorConfig()
        print(f"✅ Config created: {config.experiment_name}")
        
        # Test validation
        issues = config.validate()
        print(f"✅ Config validation: {len(issues)} issues found")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_phases():
    """Test phase definitions"""
    try:
        from training_orchestrator.phases import TrainingPhase
        
        phases = list(TrainingPhase)
        expected_phases = ["simulation", "historical_validation", "real_testing", "scaled_deployment"]
        
        phase_values = [phase.value for phase in phases]
        
        for expected in expected_phases:
            if expected in phase_values:
                print(f"✅ Phase {expected} defined correctly")
            else:
                print(f"❌ Phase {expected} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Phase test failed: {e}")
        return False

def test_config_environments():
    """Test different environment configurations"""
    try:
        from training_orchestrator.config import (
            DEVELOPMENT_CONFIG,
            STAGING_CONFIG, 
            PRODUCTION_CONFIG,
            QUICK_TEST_CONFIG
        )
        
        configs = {
            "development": DEVELOPMENT_CONFIG,
            "staging": STAGING_CONFIG,
            "production": PRODUCTION_CONFIG,
            "quick_test": QUICK_TEST_CONFIG
        }
        
        for name, config in configs.items():
            issues = config.validate()
            if not issues:
                print(f"✅ {name.title()} config is valid")
            else:
                print(f"❌ {name.title()} config has issues: {issues}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config environment test failed: {e}")
        return False

def test_file_structure():
    """Test that all expected files exist"""
    
    base_dir = Path(__file__).parent / "training_orchestrator"
    
    expected_files = [
        "__init__.py",
        "core.py",
        "phases.py",
        "episode_manager.py",
        "curriculum.py",
        "performance_monitor.py",
        "safety_monitor.py",
        "config.py",
        "cli.py"
    ]
    
    for file_name in expected_files:
        file_path = base_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True

def main():
    """Run all basic tests"""
    
    print("🧪 Running basic structure tests for GAELP Training Orchestrator")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_imports),
        ("Phase Definitions", test_phases),
        ("Config Environments", test_config_environments),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}:")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic structure tests PASSED!")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run full example: python example_training_run.py") 
        print("   3. Test CLI: python -m training_orchestrator.cli --help")
        return True
    else:
        print("❌ Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)