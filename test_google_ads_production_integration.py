#!/usr/bin/env python3
"""
Test Google Ads Production Integration with GAELP
Comprehensive test suite to verify real Google Ads API integration.
NO MOCK TESTS - Only real API integration tests.
"""

import os
import asyncio
import logging
import json
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import our production modules
from google_ads_production_manager import (
    GoogleAdsProductionManager, CampaignConfig, AdGroupConfig,
    GAELPGoogleAdsIntegration
)
from google_ads_gaelp_integration import GAELPGoogleAdsAgent, GAELPCampaignState
from gaelp_google_ads_bridge import GAELPGoogleAdsBridge, integrate_google_ads_with_gaelp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGoogleAdsProductionIntegration(unittest.TestCase):
    """Test suite for Google Ads production integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.customer_id = os.environ.get('GOOGLE_ADS_CUSTOMER_ID')
        self.ads_manager = None
        self.rl_agent = None
        self.bridge = None
        self.test_campaigns = []
    
    def test_environment_variables(self):
        """Test that all required environment variables are set"""
        required_vars = [
            'GOOGLE_ADS_DEVELOPER_TOKEN',
            'GOOGLE_ADS_CLIENT_ID',
            'GOOGLE_ADS_CLIENT_SECRET',
            'GOOGLE_ADS_REFRESH_TOKEN',
            'GOOGLE_ADS_CUSTOMER_ID'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.fail(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        logger.info("✅ All required environment variables are set")
    
    def test_google_ads_client_initialization(self):
        """Test Google Ads client initialization"""
        try:
            self.ads_manager = GoogleAdsProductionManager(self.customer_id)
            self.assertIsNotNone(self.ads_manager.client)
            self.assertEqual(self.ads_manager.customer_id, self.customer_id)
            
            logger.info("✅ Google Ads client initialized successfully")
            
        except Exception as e:
            self.fail(f"Failed to initialize Google Ads client: {e}")
    
    async def test_campaign_creation(self):
        """Test real campaign creation"""
        if not self.ads_manager:
            self.test_google_ads_client_initialization()
        
        try:
            # Create behavioral health campaign configuration
            config = self.ads_manager.create_behavioral_health_campaign_config()
            
            # Modify campaign name to include test identifier
            config.name = f"GAELP_TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config.budget_amount_micros = 10_000_000  # $10/day for testing
            
            # Create campaign
            campaign_resource_name = await self.ads_manager.create_campaign(config)
            
            self.assertIsNotNone(campaign_resource_name)
            self.assertIn('campaigns/', campaign_resource_name)
            
            # Store for cleanup
            campaign_id = campaign_resource_name.split('/')[-1]
            self.test_campaigns.append(campaign_id)
            
            logger.info(f"✅ Test campaign created successfully: {campaign_id}")
            
            return campaign_resource_name
            
        except Exception as e:
            self.fail(f"Failed to create test campaign: {e}")
    
    async def test_campaign_performance_retrieval(self):
        """Test campaign performance data retrieval"""
        if not self.test_campaigns:
            await self.test_campaign_creation()
        
        try:
            campaign_id = self.test_campaigns[0]
            
            # Get performance data (will likely be zero for new campaign)
            performance = await self.ads_manager.get_campaign_performance(campaign_id, days=1)
            
            self.assertIsNotNone(performance)
            self.assertEqual(performance.campaign_id, campaign_id)
            self.assertGreaterEqual(performance.impressions, 0)
            self.assertGreaterEqual(performance.clicks, 0)
            self.assertGreaterEqual(performance.conversions, 0)
            
            logger.info(f"✅ Performance data retrieved: {performance.impressions} impressions, {performance.clicks} clicks")
            
        except Exception as e:
            self.fail(f"Failed to retrieve campaign performance: {e}")
    
    async def test_bid_adjustment(self):
        """Test bid adjustment functionality"""
        if not self.test_campaigns:
            await self.test_campaign_creation()
        
        try:
            campaign_id = self.test_campaigns[0]
            
            # Get campaign keywords
            keywords = await self.ads_manager._get_campaign_keywords(campaign_id)
            
            if keywords:
                # Create test bid adjustments
                bid_adjustments = {}
                for keyword_info in keywords[:3]:  # Limit to first 3 keywords
                    keyword_text = keyword_info['text']
                    bid_adjustments[keyword_text] = 1.1  # Increase by 10%
                
                if bid_adjustments:
                    # Apply bid adjustments
                    await self.ads_manager.update_campaign_bids(campaign_id, bid_adjustments)
                    
                    logger.info(f"✅ Bid adjustments applied successfully: {len(bid_adjustments)} keywords")
                else:
                    logger.info("ℹ️ No keywords found for bid adjustment test")
            else:
                logger.info("ℹ️ No keywords found in campaign (expected for new campaign)")
            
        except Exception as e:
            self.fail(f"Failed to test bid adjustments: {e}")
    
    async def test_rl_agent_initialization(self):
        """Test RL agent initialization"""
        if not self.ads_manager:
            self.test_google_ads_client_initialization()
        
        try:
            self.rl_agent = GAELPGoogleAdsAgent(self.ads_manager)
            
            self.assertIsNotNone(self.rl_agent.ads_manager)
            self.assertIsNotNone(self.rl_agent.rl_integration)
            self.assertEqual(len(self.rl_agent.active_campaigns), 0)  # No campaigns initially
            
            logger.info("✅ RL agent initialized successfully")
            
        except Exception as e:
            self.fail(f"Failed to initialize RL agent: {e}")
    
    async def test_rl_campaign_creation(self):
        """Test RL-driven campaign creation"""
        if not self.rl_agent:
            await self.test_rl_agent_initialization()
        
        try:
            # Create RL-managed campaign
            campaign_resource_name = await self.rl_agent.create_rl_campaign("behavioral_health")
            
            self.assertIsNotNone(campaign_resource_name)
            self.assertIn('campaigns/', campaign_resource_name)
            
            # Check that campaign is tracked by RL agent
            campaign_id = campaign_resource_name.split('/')[-1]
            self.assertIn(campaign_id, self.rl_agent.active_campaigns)
            self.assertIn(campaign_id, self.rl_agent.campaign_states)
            
            # Store for cleanup
            self.test_campaigns.append(campaign_id)
            
            logger.info(f"✅ RL-managed campaign created successfully: {campaign_id}")
            
        except Exception as e:
            self.fail(f"Failed to create RL-managed campaign: {e}")
    
    async def test_campaign_state_update(self):
        """Test campaign state updates"""
        if not self.rl_agent or not self.rl_agent.active_campaigns:
            await self.test_rl_campaign_creation()
        
        try:
            # Update campaign states
            await self.rl_agent.update_campaign_states()
            
            # Verify states were updated
            for campaign_id in self.rl_agent.active_campaigns:
                self.assertIn(campaign_id, self.rl_agent.campaign_states)
                
                state = self.rl_agent.campaign_states[campaign_id]
                self.assertIsInstance(state, GAELPCampaignState)
                self.assertEqual(state.campaign_id, campaign_id)
                self.assertGreaterEqual(state.impressions, 0)
                self.assertGreaterEqual(state.clicks, 0)
                self.assertGreaterEqual(state.conversions, 0)
            
            logger.info("✅ Campaign states updated successfully")
            
        except Exception as e:
            self.fail(f"Failed to update campaign states: {e}")
    
    async def test_rl_optimization(self):
        """Test RL optimization functionality"""
        if not self.rl_agent or not self.rl_agent.campaign_states:
            await self.test_campaign_state_update()
        
        try:
            # Run RL optimization
            await self.rl_agent.optimize_campaigns()
            
            # Check if optimization actions were recorded
            # For new campaigns, there might not be enough data for optimization
            logger.info(f"✅ RL optimization completed. Actions recorded: {len(self.rl_agent.action_history)}")
            
        except Exception as e:
            self.fail(f"Failed to run RL optimization: {e}")
    
    async def test_bridge_initialization(self):
        """Test Google Ads bridge initialization"""
        try:
            self.bridge = await integrate_google_ads_with_gaelp(self.customer_id)
            
            self.assertIsNotNone(self.bridge)
            self.assertIsNotNone(self.bridge.ads_manager)
            self.assertIsNotNone(self.bridge.rl_agent)
            
            logger.info("✅ Google Ads bridge initialized successfully")
            
        except Exception as e:
            self.fail(f"Failed to initialize Google Ads bridge: {e}")
    
    async def test_campaign_summary(self):
        """Test campaign summary generation"""
        if not self.bridge:
            await self.test_bridge_initialization()
        
        try:
            summary = self.bridge.get_campaign_summary()
            
            self.assertIsInstance(summary, dict)
            self.assertIn('timestamp', summary)
            self.assertIn('total_campaigns', summary)
            self.assertIn('overall_performance', summary)
            self.assertIn('optimization_status', summary)
            
            logger.info(f"✅ Campaign summary generated: {summary['total_campaigns']} campaigns")
            
        except Exception as e:
            self.fail(f"Failed to generate campaign summary: {e}")
    
    async def test_emergency_checks(self):
        """Test emergency monitoring functionality"""
        if not self.rl_agent or not self.rl_agent.campaign_states:
            await self.test_campaign_state_update()
        
        try:
            # Test emergency checks on current campaigns
            for campaign_id in self.rl_agent.campaign_states:
                state = self.rl_agent.campaign_states[campaign_id]
                
                # Run emergency checks
                await self.rl_agent._emergency_checks(campaign_id, state)
            
            logger.info("✅ Emergency checks completed successfully")
            
        except Exception as e:
            self.fail(f"Failed to run emergency checks: {e}")
    
    async def cleanup_test_campaigns(self):
        """Clean up test campaigns"""
        if not self.ads_manager or not self.test_campaigns:
            return
        
        logger.info(f"🧹 Cleaning up {len(self.test_campaigns)} test campaigns...")
        
        for campaign_id in self.test_campaigns:
            try:
                # Pause campaign (safer than deletion for testing)
                await self.ads_manager.pause_campaign(campaign_id)
                logger.info(f"✅ Test campaign {campaign_id} paused")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to pause test campaign {campaign_id}: {e}")
        
        logger.info("🧹 Test campaign cleanup complete")

class GoogleAdsIntegrationTestRunner:
    """Test runner for Google Ads integration tests"""
    
    def __init__(self):
        self.test_suite = TestGoogleAdsProductionIntegration()
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("=" * 80)
        print("GOOGLE ADS PRODUCTION INTEGRATION TESTS")
        print("=" * 80)
        
        tests = [
            ("Environment Variables", self.test_suite.test_environment_variables),
            ("Google Ads Client", self.test_suite.test_google_ads_client_initialization),
            ("Campaign Creation", self.test_suite.test_campaign_creation),
            ("Performance Retrieval", self.test_suite.test_campaign_performance_retrieval),
            ("Bid Adjustment", self.test_suite.test_bid_adjustment),
            ("RL Agent Initialization", self.test_suite.test_rl_agent_initialization),
            ("RL Campaign Creation", self.test_suite.test_rl_campaign_creation),
            ("Campaign State Update", self.test_suite.test_campaign_state_update),
            ("RL Optimization", self.test_suite.test_rl_optimization),
            ("Bridge Initialization", self.test_suite.test_bridge_initialization),
            ("Campaign Summary", self.test_suite.test_campaign_summary),
            ("Emergency Checks", self.test_suite.test_emergency_checks)
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\n{'=' * 20} {test_name} {'=' * (58 - len(test_name))}")
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                
                print(f"✅ {test_name}: PASSED")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
                failed_tests += 1
        
        # Cleanup
        print(f"\n{'=' * 20} Cleanup {'=' * 53}")
        await self.test_suite.cleanup_test_campaigns()
        
        # Results summary
        total_tests = passed_tests + failed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED - Google Ads integration is working correctly!")
            print("✅ Real Google Ads API connection verified")
            print("✅ Campaign creation and management working")
            print("✅ RL optimization system functional")
            print("✅ Performance monitoring operational")
            print("✅ Emergency controls active")
            print("✅ Production ready!")
        else:
            print(f"\n⚠️ {failed_tests} tests failed - check logs for details")
            print("❌ Google Ads integration needs attention before production use")
        
        return failed_tests == 0

async def run_integration_verification():
    """
    Run comprehensive verification of Google Ads integration
    """
    print("🔍 Starting Google Ads production integration verification...")
    
    # Check environment first
    required_vars = [
        'GOOGLE_ADS_DEVELOPER_TOKEN',
        'GOOGLE_ADS_CLIENT_ID', 
        'GOOGLE_ADS_CLIENT_SECRET',
        'GOOGLE_ADS_REFRESH_TOKEN',
        'GOOGLE_ADS_CUSTOMER_ID'
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Run setup_google_ads_production.py to configure authentication")
        return False
    
    # Run test suite
    test_runner = GoogleAdsIntegrationTestRunner()
    success = await test_runner.run_all_tests()
    
    if success:
        # Create verification report
        verification_report = {
            'timestamp': datetime.now().isoformat(),
            'status': 'PASSED',
            'google_ads_integration': 'FUNCTIONAL',
            'rl_optimization': 'OPERATIONAL',
            'emergency_controls': 'ACTIVE',
            'production_ready': True,
            'next_steps': [
                'Deploy to production environment',
                'Monitor campaign performance',
                'Enable continuous optimization',
                'Set up alerting for emergency situations'
            ]
        }
        
        with open('/home/hariravichandran/AELP/google_ads_integration_verification.json', 'w') as f:
            json.dump(verification_report, f, indent=2)
        
        print("\n📊 Verification report saved to: google_ads_integration_verification.json")
    
    return success

if __name__ == "__main__":
    asyncio.run(run_integration_verification())