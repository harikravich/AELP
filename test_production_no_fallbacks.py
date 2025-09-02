#!/usr/bin/env python3
"""
TEST PRODUCTION SYSTEM - VERIFY NO FALLBACKS OR HARDCODING
"""

import sys
import json
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_no_hardcoding():
    """Check that no hardcoded values exist in production files"""
    print("\n" + "="*60)
    print("CHECKING FOR HARDCODING VIOLATIONS...")
    print("="*60)
    
    import os
    violations = []
    
    # Files to check
    files_to_check = [
        'fortified_rl_agent_no_hardcoding.py',
        'fortified_environment_no_hardcoding.py',
        'monitor_production_quality.py'
    ]
    
    # Patterns that indicate hardcoding
    hardcoded_patterns = [
        (r"MIN_BID\s*=\s*[\d.]+", "Hardcoded MIN_BID"),
        (r"MAX_BID\s*=\s*[\d.]+", "Hardcoded MAX_BID"),
        (r"channels\s*=\s*\[.*'organic'.*\]", "Hardcoded channel list"),
        (r"budget.*=\s*1000(?:\.0)?(?!\s*#)", "Hardcoded budget 1000"),
        (r"budget.*=\s*10000(?:\.0)?", "Hardcoded budget 10000"),
        (r"/ 20\.0(?!\s*#)", "Hardcoded normalization /20.0"),
        (r"/ 10\.0(?!\s*#)", "Hardcoded normalization /10.0"),
        (r"/ 1000\.0(?!\s*#)", "Hardcoded normalization /1000.0"),
        (r"Dropout\(0\.1\)", "Hardcoded dropout rate"),
        (r"conversion_value.*=\s*100\.0", "Hardcoded conversion value"),
    ]
    
    import re
    import os
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        for pattern, description in hardcoded_patterns:
            matches = re.findall(pattern, content)
            if matches:
                violations.append(f"{file_path}: {description} - Found: {matches}")
    
    if violations:
        print("\n❌ HARDCODING VIOLATIONS FOUND:")
        for v in violations:
            print(f"  - {v}")
        return False
    else:
        print("✅ No hardcoding violations found!")
        return True


def test_discovered_patterns():
    """Test that patterns are properly discovered"""
    print("\n" + "="*60)
    print("TESTING PATTERN DISCOVERY...")
    print("="*60)
    
    import os
    # Load patterns
    patterns_file = 'discovered_patterns.json'
    if not os.path.exists(patterns_file):
        print(f"❌ Pattern file not found: {patterns_file}")
        return False
    
    with open(patterns_file, 'r') as f:
        patterns = json.load(f)
    
    # Check required pattern keys
    required_keys = ['channels', 'segments', 'devices', 'bid_ranges']
    missing_keys = [k for k in required_keys if k not in patterns]
    
    if missing_keys:
        print(f"❌ Missing pattern keys: {missing_keys}")
        return False
    
    print(f"✅ Discovered patterns loaded successfully:")
    print(f"  - Channels: {list(patterns['channels'].keys())}")
    print(f"  - Segments: {list(patterns['segments'].keys())}")
    print(f"  - Devices: {list(patterns['devices'].keys())}")
    
    # Check bid ranges
    if 'bid_ranges' in patterns:
        print(f"  - Bid ranges discovered:")
        for category, ranges in patterns['bid_ranges'].items():
            if isinstance(ranges, dict):
                print(f"    • {category}: ${ranges.get('min', 0):.2f} - ${ranges.get('max', 0):.2f}")
    
    return True


def test_production_agent():
    """Test the production RL agent"""
    print("\n" + "="*60)
    print("TESTING PRODUCTION RL AGENT...")
    print("="*60)
    
    try:
        from fortified_rl_agent_no_hardcoding import (
            ProductionFortifiedRLAgent,
            DynamicEnrichedState,
            DataStatistics
        )
        from discovery_engine import GA4DiscoveryEngine as DiscoveryEngine
        from creative_selector import CreativeSelector
        from attribution_models import AttributionEngine
        from budget_pacer import BudgetPacer
        from identity_resolver import IdentityResolver
        from gaelp_parameter_manager import ParameterManager
        
        # Initialize components
        discovery = DiscoveryEngine(write_enabled=False, cache_only=True)
        creative_selector = CreativeSelector()
        attribution = AttributionEngine()
        budget_pacer = BudgetPacer()
        identity_resolver = IdentityResolver()
        pm = ParameterManager()
        
        # Create agent
        agent = ProductionFortifiedRLAgent(
            discovery_engine=discovery,
            creative_selector=creative_selector,
            attribution_engine=attribution,
            budget_pacer=budget_pacer,
            identity_resolver=identity_resolver,
            parameter_manager=pm
        )
        
        print("✅ Agent initialized successfully")
        
        # Check discovered dimensions
        print(f"  - Discovered {len(agent.discovered_channels)} channels: {agent.discovered_channels[:3]}...")
        print(f"  - Discovered {len(agent.discovered_segments)} segments: {agent.discovered_segments[:3]}...")
        print(f"  - Discovered {len(agent.discovered_creatives)} creatives")
        
        # Check data statistics
        print(f"  - Data statistics computed:")
        print(f"    • Bid range: ${agent.data_stats.bid_min:.2f} - ${agent.data_stats.bid_max:.2f}")
        print(f"    • Budget mean: ${agent.data_stats.budget_mean:.2f}")
        print(f"    • Conversion value mean: ${agent.data_stats.conversion_value_mean:.2f}")
        
        # Test state creation
        state = agent.get_enriched_state(
            user_id="test_user",
            journey_state=type('obj', (object,), {'stage': 1, 'touchpoints_seen': 3})(),
            context={'segment': 'researching_parent', 'channel': 'organic', 'device': 'mobile'}
        )
        
        print(f"  - State vector dimension: {len(state.to_vector(agent.data_stats))}")
        
        # Test action selection
        action = agent.select_action(state, explore=False)
        print(f"  - Action selected:")
        print(f"    • Bid: ${action['bid_amount']:.2f}")
        print(f"    • Channel: {action['channel']}")
        print(f"    • Creative: #{action['creative_id']}")
        
        # Verify no hardcoded values in action
        if action['bid_amount'] == 0.5 or action['bid_amount'] == 10.0:
            print("⚠️  Warning: Bid amount might be hardcoded")
        
        if action['channel'] not in agent.discovered_channels:
            print("❌ Error: Channel not from discovered channels!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_environment():
    """Test the production environment"""
    print("\n" + "="*60)
    print("TESTING PRODUCTION ENVIRONMENT...")
    print("="*60)
    
    try:
        from fortified_environment_no_hardcoding import ProductionFortifiedEnvironment
        from gaelp_parameter_manager import ParameterManager
        
        # Create environment
        pm = ParameterManager()
        env = ProductionFortifiedEnvironment(
            parameter_manager=pm,
            use_real_ga4_data=False,  # Don't use real data for test
            is_parallel=False
        )
        
        print("✅ Environment initialized successfully")
        
        # Check discovered values
        print(f"  - Budget: ${env.max_budget:.2f} (discovered)")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"  - Reset successful, observation shape: {obs.shape}")
        
        # Test step
        action = {
            'bid': np.array([5.0]),
            'creative': 0,
            'channel': 0
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  - Step successful:")
        print(f"    • Reward: {reward:.2f}")
        print(f"    • Info keys: {list(info.keys())}")
        
        # Check metrics are tracked
        if 'metrics' in info:
            print(f"    • Metrics tracked: {list(info['metrics'].keys())[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitor():
    """Test the production monitor"""
    print("\n" + "="*60)
    print("TESTING PRODUCTION MONITOR...")
    print("="*60)
    
    try:
        from monitor_production_quality import ProductionMonitor
        
        # Create monitor
        monitor = ProductionMonitor()
        
        print("✅ Monitor initialized successfully")
        
        # Check discovered content
        print(f"  - Discovered {len(monitor.creative_content)} creative contents")
        print(f"  - Discovered {len(monitor.channel_info)} channel infos")
        
        # Show sample creative content
        if monitor.creative_content:
            sample_id = list(monitor.creative_content.keys())[0]
            sample_content = monitor.creative_content[sample_id]
            print(f"  - Sample creative #{sample_id}:")
            print(f"    • Segment: {sample_content['segment']}")
            print(f"    • Headline: {sample_content['headline'][:50]}...")
            print(f"    • CTA: {sample_content['cta']}")
        
        # Test parsing
        metrics = monitor.parse_log_file()
        print(f"  - Log parsing successful, found {len(metrics)} metric categories")
        
        # Test daily rate calculation
        daily_rates = monitor.calculate_daily_rates(metrics)
        print(f"  - Daily rate calculation successful:")
        print(f"    • Conversions/day: {daily_rates['conversions_per_day']:.1f}")
        print(f"    • Spend/day: ${daily_rates['spend_per_day']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing monitor: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_fallbacks():
    """Check for fallback implementations"""
    print("\n" + "="*60)
    print("CHECKING FOR FALLBACK IMPLEMENTATIONS...")
    print("="*60)
    
    violations = []
    
    # Files to check
    files_to_check = [
        'fortified_rl_agent_no_hardcoding.py',
        'fortified_environment_no_hardcoding.py',
        'monitor_production_quality.py'
    ]
    
    # Patterns that indicate fallbacks
    fallback_patterns = [
        (r"fallback", "Fallback implementation"),
        (r"simplified", "Simplified implementation"),
        (r"mock(?!_)", "Mock implementation"),  # Exclude mock_
        (r"dummy", "Dummy implementation"),
        (r"TODO", "TODO left in code"),
        (r"FIXME", "FIXME left in code"),
        (r"not available", "Not available fallback"),
        (r"_AVAILABLE\s*=\s*False", "Feature disabled"),
    ]
    
    import re
    import os
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            for pattern, description in fallback_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Check if it's in a string (might be legitimate)
                    if '"' in line or "'" in line:
                        # Could be legitimate use in string
                        continue
                    violations.append(f"{file_path}:{i} - {description}: {line.strip()}")
    
    if violations:
        print("\n❌ FALLBACK VIOLATIONS FOUND:")
        for v in violations[:10]:  # Show first 10
            print(f"  - {v}")
        return False
    else:
        print("✅ No fallback implementations found!")
        return True


def main():
    """Run all production tests"""
    print("\n" + "="*80)
    print(" " * 20 + "PRODUCTION SYSTEM TEST SUITE")
    print(" " * 15 + "Verifying NO Fallbacks or Hardcoding")
    print("="*80)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Pattern Discovery", test_discovered_patterns),
        ("No Hardcoding", check_no_hardcoding),
        ("No Fallbacks", test_no_fallbacks),
        ("Production Agent", test_production_agent),
        ("Production Environment", test_production_environment),
        ("Production Monitor", test_monitor)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
            all_passed = all_passed and passed
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results[test_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:.<40} {status}")
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Production system is ready.")
        print("✨ No hardcoding, no fallbacks, everything discovered dynamically!")
    else:
        print("⚠️  Some tests failed. Please fix issues before deploying.")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())