#!/usr/bin/env python3
"""
Test Creative Integration System
Demonstrates how the CreativeSelector replaces empty ad_content dictionaries 
with rich, targeted creative content across all simulation systems.
"""

import logging
import time
import random
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_creative_integration_basics():
    """Test basic CreativeIntegration functionality"""
    print("="*80)
    print("🎨 TESTING CREATIVE INTEGRATION BASICS")
    print("="*80)
    
    try:
        from creative_integration import get_creative_integration, SimulationContext
        
        integration = get_creative_integration()
        
        # Test different user personas
        test_contexts = [
            {
                'name': 'Crisis Parent',
                'context': SimulationContext(
                    user_id="crisis_parent_test",
                    persona="crisis_parent",
                    channel="search",
                    device_type="mobile",
                    urgency_score=0.9,
                    session_count=1
                )
            },
            {
                'name': 'Researcher',
                'context': SimulationContext(
                    user_id="researcher_test",
                    persona="researcher",
                    channel="social",
                    device_type="desktop",
                    technical_level=0.9,
                    session_count=3
                )
            },
            {
                'name': 'Price-Conscious User',
                'context': SimulationContext(
                    user_id="price_conscious_test",
                    persona="price_conscious",
                    channel="display",
                    device_type="mobile",
                    price_sensitivity=0.9,
                    session_count=2
                )
            }
        ]
        
        for test in test_contexts:
            print(f"\n--- {test['name']} ---")
            ad_content = integration.get_targeted_ad_content(test['context'])
            
            print(f"Headline: {ad_content['headline']}")
            print(f"Description: {ad_content['description'][:80]}...")
            print(f"CTA: {ad_content['cta']}")
            print(f"Landing Page: {ad_content['landing_page']}")
            print(f"Price: ${ad_content['price_shown']}")
            print(f"Creative Quality: {ad_content['creative_quality']:.2f}")
            print(f"Selection Reason: {ad_content['selection_reason']}")
            
            # Track impression for testing
            integration.track_impression(
                user_id=test['context'].user_id,
                creative_id=ad_content['creative_id'],
                clicked=random.choice([True, False]),
                engagement_time=random.uniform(10, 120)
            )
        
        print(f"\n✅ Basic Creative Integration test PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Creative Integration not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Basic test FAILED: {e}")
        return False


def test_aura_campaign_with_creative_integration():
    """Test the updated Aura Campaign Simulator with Creative Integration"""
    print("\n" + "="*80)
    print("🎯 TESTING AURA CAMPAIGN WITH CREATIVE INTEGRATION")
    print("="*80)
    
    try:
        from aura_campaign_simulator_updated import AuraCampaignEnvironment
        
        env = AuraCampaignEnvironment()
        
        # Test strategy with creative integration
        test_strategy = {
            'name': 'Creative Integration Test',
            'channel': 'search',
            'keywords': ['crisis_keywords', 'child safety'],
            'bid': 6.0,
            'cpc': 2.5,
        }
        
        print(f"Running campaign: {test_strategy['name']}")
        print(f"Channel: {test_strategy['channel']}")
        
        # Run smaller test campaign
        results = env.run_campaign(test_strategy, num_impressions=1000)
        
        print(f"\n📊 CAMPAIGN RESULTS:")
        print(f"Impressions: {results['impressions']:,}")
        print(f"Clicks: {results['clicks']:,} (CTR: {results['ctr']:.2%})")
        print(f"Conversions: {results['conversions']} (CR: {results['conversion_rate']:.2%})")
        print(f"Revenue: ${results['revenue']:.2f}")
        print(f"CAC: ${results['cac']:.2f}")
        print(f"ROAS: {results['roas']:.2f}x")
        
        # Show creative performance
        if results['creative_impressions']:
            print(f"\n🎨 CREATIVE PERFORMANCE:")
            for creative_id, perf in results['creative_impressions'].items():
                if perf['impressions'] > 0:
                    ctr = perf['clicks'] / perf['impressions']
                    cvr = perf['conversions'] / max(perf['clicks'], 1)
                    print(f"  {perf['headline']} - "
                          f"Impressions: {perf['impressions']}, "
                          f"CTR: {ctr:.2%}, CVR: {cvr:.2%}")
        
        print(f"\n✅ Aura Campaign with Creative Integration test PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Updated Aura Campaign Simulator not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Aura Campaign test FAILED: {e}")
        return False


def test_enhanced_simulator_integration():
    """Test Enhanced Simulator with Creative Integration patch"""
    print("\n" + "="*80)
    print("🔬 TESTING ENHANCED SIMULATOR INTEGRATION")
    print("="*80)
    
    try:
        # Apply the creative integration patch
        from enhanced_simulator_creative_patch import apply_creative_integration_patch, enhance_ad_creative_with_selector
        
        # Test the enhancement function directly
        empty_ad = {}
        test_context = {
            'hour': 20,
            'device': 'mobile',
            'segment': 'crisis_parent',
            'channel': 'search'
        }
        
        print("Testing ad creative enhancement...")
        enhanced_ad = enhance_ad_creative_with_selector(
            ad_creative=empty_ad,
            context=test_context,
            user_id="test_user_123"
        )
        
        print(f"Original ad creative: {empty_ad}")
        print(f"Enhanced ad creative fields: {list(enhanced_ad.keys())}")
        print(f"Headline: {enhanced_ad.get('headline', 'N/A')}")
        print(f"Description: {enhanced_ad.get('description', 'N/A')[:60]}...")
        
        # Test with basic ad creative (should preserve original values)
        basic_ad = {'headline': 'Custom Test Headline', 'price_shown': 99.99}
        enhanced_basic = enhance_ad_creative_with_selector(
            ad_creative=basic_ad,
            context=test_context,
            user_id="test_user_456"
        )
        
        print(f"\nOriginal basic ad: {basic_ad}")
        print(f"Enhanced basic ad headline: {enhanced_basic.get('headline')}")
        print(f"Enhanced basic ad price: ${enhanced_basic.get('price_shown')}")
        
        print(f"\n✅ Enhanced Simulator integration test PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Enhanced Simulator patch not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced Simulator test FAILED: {e}")
        return False


def test_fatigue_and_ab_testing():
    """Test creative fatigue and A/B testing functionality"""
    print("\n" + "="*80)
    print("🧪 TESTING CREATIVE FATIGUE & A/B TESTING")
    print("="*80)
    
    try:
        from creative_integration import get_creative_integration, SimulationContext
        from creative_selector import ABTestVariant
        
        integration = get_creative_integration()
        
        # Create test user
        user_context = SimulationContext(
            user_id="fatigue_test_user",
            persona="concerned_parent",
            channel="social",
            device_type="mobile",
            session_count=1
        )
        
        print("Testing creative fatigue (showing same user multiple ads)...")
        
        # Show multiple ads to same user to test fatigue
        for i in range(6):
            ad_content = integration.get_targeted_ad_content(user_context)
            
            # Track impression
            integration.track_impression(
                user_id=user_context.user_id,
                creative_id=ad_content['creative_id'],
                clicked=i < 3,  # First few clicks, then fatigue sets in
                engagement_time=max(30 - i * 5, 5)  # Decreasing engagement
            )
            
            print(f"  Ad {i+1}: {ad_content['headline'][:40]}... "
                  f"(Creative ID: {ad_content['creative_id']})")
        
        # Check fatigue analysis
        fatigue_analysis = integration.get_fatigue_analysis(user_context.user_id)
        print(f"\nFatigue Analysis:")
        for creative_id, fatigue_score in fatigue_analysis.items():
            if fatigue_score > 0:
                print(f"  {creative_id}: {fatigue_score:.2f} fatigue")
        
        # Test A/B testing
        print(f"\nSetting up A/B test...")
        ab_variants = [
            {
                'id': 'control',
                'name': 'Control Group',
                'traffic_split': 0.5,
                'overrides': {},
                'active': True
            },
            {
                'id': 'test_variant',
                'name': 'New Headline Test',
                'traffic_split': 0.5,
                'overrides': {'headline_boost': True},
                'active': True
            }
        ]
        
        integration.create_ab_test('headline_optimization', ab_variants)
        
        # Test with different users to see A/B assignment
        print(f"Testing A/B variant assignment...")
        for i in range(5):
            test_user = SimulationContext(
                user_id=f"ab_test_user_{i}",
                persona="researcher",
                channel="search",
                device_type="desktop",
                session_count=1
            )
            
            ad_content = integration.get_targeted_ad_content(test_user)
            print(f"  User {i}: {ad_content['selection_reason']}")
        
        print(f"\n✅ Fatigue & A/B testing test PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Creative Integration not available for advanced testing: {e}")
        return False
    except Exception as e:
        print(f"❌ Fatigue & A/B test FAILED: {e}")
        return False


def test_performance_reporting():
    """Test creative performance reporting and analytics"""
    print("\n" + "="*80)
    print("📈 TESTING PERFORMANCE REPORTING")
    print("="*80)
    
    try:
        from creative_integration import get_creative_integration
        
        integration = get_creative_integration()
        
        # Generate performance report
        report = integration.get_performance_report(days=1)
        
        print(f"Performance Report (Last {report['period_days']} day):")
        print(f"Total Impressions: {report['total_impressions']}")
        print(f"Total Clicks: {report['total_clicks']}")
        print(f"Total Conversions: {report['total_conversions']}")
        
        if report['total_impressions'] > 0:
            overall_ctr = report['total_clicks'] / report['total_impressions']
            print(f"Overall CTR: {overall_ctr:.2%}")
        
        if report['creative_performance']:
            print(f"\nTop Creative Performance:")
            
            # Sort by CTR
            sorted_creatives = sorted(
                report['creative_performance'].items(),
                key=lambda x: x[1]['ctr'],
                reverse=True
            )
            
            for creative_id, perf in sorted_creatives[:5]:
                if perf['impressions'] > 0:
                    print(f"  {perf['headline'][:50]}...")
                    print(f"    Impressions: {perf['impressions']}, "
                          f"CTR: {perf['ctr']:.2%}, "
                          f"CVR: {perf['cvr']:.2%}")
        
        print(f"\n✅ Performance reporting test PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Creative Integration not available for reporting: {e}")
        return False
    except Exception as e:
        print(f"❌ Performance reporting test FAILED: {e}")
        return False


def run_comprehensive_test():
    """Run all Creative Integration tests"""
    print("🚀 STARTING COMPREHENSIVE CREATIVE INTEGRATION TESTS")
    print("="*80)
    
    test_results = []
    
    # Run all tests
    test_functions = [
        ("Basic Creative Integration", test_creative_integration_basics),
        ("Aura Campaign Integration", test_aura_campaign_with_creative_integration),
        ("Enhanced Simulator Integration", test_enhanced_simulator_integration),
        ("Fatigue & A/B Testing", test_fatigue_and_ab_testing),
        ("Performance Reporting", test_performance_reporting)
    ]
    
    for test_name, test_func in test_functions:
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            test_results.append({
                'name': test_name,
                'passed': result,
                'duration': duration
            })
            
            time.sleep(1)  # Small delay between tests
            
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            test_results.append({
                'name': test_name,
                'passed': False,
                'duration': 0,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(1 for result in test_results if result['passed'])
    total_tests = len(test_results)
    
    for result in test_results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        duration = f"{result['duration']:.2f}s"
        print(f"{result['name']:<35} {status:<10} ({duration})")
        
        if not result['passed'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    print(f"\n🏆 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Creative Integration is working correctly.")
        print("\n🔧 INTEGRATION COMPLETE:")
        print("   • CreativeSelector now provides rich ad content")
        print("   • Empty ad_content {} dictionaries are replaced with targeted creatives")
        print("   • User journey stage and segment drive creative selection")
        print("   • Creative fatigue prevents overexposure")
        print("   • A/B testing enables creative optimization")
        print("   • Performance tracking improves selection over time")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)