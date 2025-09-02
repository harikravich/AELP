#!/usr/bin/env python3
"""
Google Ads Production Runner for GAELP
Main entry point to start Google Ads production campaign management.
NO FALLBACKS - Real Google Ads API integration only.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Optional

# Import our production Google Ads components
from gaelp_google_ads_bridge import GAELPGoogleAdsBridge, integrate_google_ads_with_gaelp
from setup_google_ads_production import GoogleAdsAuthenticator
from verify_google_ads_integration import main as verify_integration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoogleAdsProductionRunner:
    """Main runner for Google Ads production integration with GAELP"""
    
    def __init__(self):
        self.bridge: Optional[GAELPGoogleAdsBridge] = None
        self.running = False
        
    async def initialize(self, customer_id: str = None) -> bool:
        """Initialize Google Ads production system"""
        print("🚀 Initializing Google Ads production system for GAELP...")
        
        # Verify credentials are configured
        if not self._check_credentials():
            print("❌ Google Ads credentials not configured")
            print("Run: python setup_google_ads_production.py")
            return False
        
        # Run integration verification
        print("🔍 Verifying integration...")
        if not verify_integration():
            print("❌ Integration verification failed")
            return False
        
        # Initialize bridge
        try:
            self.bridge = await integrate_google_ads_with_gaelp(customer_id)
            print("✅ Google Ads bridge initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Ads bridge: {e}")
            return False
    
    def _check_credentials(self) -> bool:
        """Check if all required credentials are configured"""
        required_vars = [
            'GOOGLE_ADS_DEVELOPER_TOKEN',
            'GOOGLE_ADS_CLIENT_ID',
            'GOOGLE_ADS_CLIENT_SECRET',
            'GOOGLE_ADS_REFRESH_TOKEN',
            'GOOGLE_ADS_CUSTOMER_ID'
        ]
        
        return all(os.environ.get(var) for var in required_vars)
    
    async def start_production_campaigns(self, num_campaigns: int = 1) -> bool:
        """Start production campaigns with RL optimization"""
        if not self.bridge:
            print("❌ Google Ads bridge not initialized")
            return False
        
        print(f"🚀 Starting {num_campaigns} production campaign(s)...")
        
        try:
            # Create initial campaigns
            campaigns = await self.bridge.create_and_optimize_campaign_batch(num_campaigns)
            
            print(f"✅ Created {len(campaigns)} production campaigns:")
            for i, campaign_resource_name in enumerate(campaigns, 1):
                campaign_id = campaign_resource_name.split('/')[-1]
                print(f"   {i}. Campaign ID: {campaign_id}")
            
            return len(campaigns) > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to start production campaigns: {e}")
            return False
    
    async def start_continuous_optimization(self) -> None:
        """Start continuous optimization system"""
        if not self.bridge:
            raise RuntimeError("Google Ads bridge not initialized")
        
        print("🔄 Starting continuous optimization system...")
        print("   - Optimization every 4 hours")
        print("   - Performance monitoring every hour")  
        print("   - Emergency monitoring every 15 minutes")
        
        self.running = True
        
        try:
            await self.bridge.start_continuous_optimization()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping continuous optimization...")
            await self.bridge.stop_optimization()
            self.running = False
        except Exception as e:
            logger.error(f"❌ Error in continuous optimization: {e}")
            self.running = False
            raise
    
    def get_status(self) -> dict:
        """Get current system status"""
        if not self.bridge:
            return {'status': 'not_initialized'}
        
        return self.bridge.get_campaign_summary()
    
    async def force_optimization(self) -> dict:
        """Force immediate optimization of all campaigns"""
        if not self.bridge:
            raise RuntimeError("Google Ads bridge not initialized")
        
        return await self.bridge.force_optimization()

async def main():
    """Main function to run Google Ads production system"""
    print("=" * 80)
    print("GOOGLE ADS PRODUCTION SYSTEM FOR GAELP")
    print("=" * 80)
    print("Real Google Ads campaign management with RL optimization")
    print("NO FALLBACKS - Production API integration only")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Google Ads Production Runner for GAELP')
    parser.add_argument('--setup', action='store_true', help='Run setup process')
    parser.add_argument('--verify', action='store_true', help='Verify integration')
    parser.add_argument('--campaigns', type=int, default=1, help='Number of campaigns to create')
    parser.add_argument('--customer-id', type=str, help='Google Ads customer ID')
    parser.add_argument('--continuous', action='store_true', help='Start continuous optimization')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--optimize', action='store_true', help='Force optimization')
    
    args = parser.parse_args()
    
    # Handle setup
    if args.setup:
        print("🔧 Running Google Ads authentication setup...")
        authenticator = GoogleAdsAuthenticator()
        credentials = authenticator.setup_oauth_credentials()
        
        if credentials:
            authenticator.save_credentials_to_env(credentials)
            print("✅ Setup complete! Restart your session to load new credentials.")
        else:
            print("❌ Setup failed")
        
        return
    
    # Handle verification
    if args.verify:
        print("🔍 Running integration verification...")
        success = verify_integration()
        sys.exit(0 if success else 1)
    
    # Initialize runner
    runner = GoogleAdsProductionRunner()
    
    # Initialize system
    if not await runner.initialize(args.customer_id):
        print("❌ Failed to initialize Google Ads production system")
        print("\nTroubleshooting:")
        print("1. Run: python run_google_ads_production.py --setup")
        print("2. Run: python run_google_ads_production.py --verify")
        print("3. Check your Google Ads account permissions")
        sys.exit(1)
    
    # Handle status request
    if args.status:
        print("\n📊 System Status:")
        status = runner.get_status()
        
        print(f"   Total campaigns: {status.get('total_campaigns', 0)}")
        print(f"   Overall spend: ${status.get('overall_performance', {}).get('total_spend', 0):.2f}")
        print(f"   Overall conversions: {status.get('overall_performance', {}).get('total_conversions', 0):.1f}")
        print(f"   Optimization running: {status.get('optimization_status', {}).get('running', False)}")
        
        return
    
    # Handle force optimization
    if args.optimize:
        print("\n🎯 Force optimization...")
        results = await runner.force_optimization()
        
        print(f"   Campaigns processed: {results['campaigns_processed']}")
        print(f"   Optimizations applied: {results['optimizations_applied']}")
        
        if results.get('errors'):
            print(f"   Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"     - {error}")
        
        return
    
    # Create campaigns
    if not await runner.start_production_campaigns(args.campaigns):
        print("❌ Failed to start production campaigns")
        sys.exit(1)
    
    # Show initial status
    print("\n📊 Initial System Status:")
    initial_status = runner.get_status()
    print(f"   Active campaigns: {initial_status.get('total_campaigns', 0)}")
    print(f"   System ready: ✅")
    
    # Start continuous optimization if requested
    if args.continuous:
        print("\n🚀 Starting continuous optimization...")
        print("   Press Ctrl+C to stop")
        
        try:
            await runner.start_continuous_optimization()
        except KeyboardInterrupt:
            print("\n✅ Optimization stopped successfully")
    
    else:
        # Show final status and exit
        print("\n✅ Production campaigns created and running")
        print("\nNext steps:")
        print("1. Monitor campaign performance in Google Ads")
        print("2. Run with --continuous to start optimization")
        print("3. Use --status to check system status")
        print("4. Use --optimize to force immediate optimization")
        
        print(f"\nCommands:")
        print(f"   Status:     python run_google_ads_production.py --status")
        print(f"   Optimize:   python run_google_ads_production.py --optimize")
        print(f"   Continuous: python run_google_ads_production.py --continuous")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)