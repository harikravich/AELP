#!/usr/bin/env python3
"""
Quick Start: GAELP Production Ad Account Setup
REAL MONEY - REAL ACCOUNTS - NO FALLBACKS

This script demonstrates the complete production ad account setup process.

REQUIREMENTS:
✅ Real credit card for billing
✅ Google account access (hari@aura.com)
✅ Facebook Business Manager access
✅ Business information for verification
✅ Domain ownership (teen-wellness-monitor.com)

DELIVERS:
✅ Google Ads Customer ID with $100/day limit
✅ Facebook Ad Account ID with proper setup
✅ Conversion tracking on landing pages
✅ UTM parameter system with gaelp_uid
✅ Budget monitoring and emergency stops
✅ API credentials securely stored
✅ Campaign structure ready for launch
"""

import os
import sys
from pathlib import Path

def check_prerequisites():
    """Check system prerequisites"""
    
    print("🔍 CHECKING PREREQUISITES")
    print("="*50)
    
    checks = {
        'Python version': sys.version_info >= (3, 8),
        'Required modules': True,  # Will check during import
        'Config directory': True,
        'Network access': True
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check config directory
    config_dir = Path.home() / '.config' / 'gaelp'
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Config directory: {config_dir}")
    
    # Check required modules
    required_modules = [
        'google.ads.googleads',
        'facebook_business',
        'requests',
        'sqlite3',
        'smtplib'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n📋 Install requirements with:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("✅ All required modules available")
    
    return True

def show_setup_overview():
    """Show complete setup overview"""
    
    print("\n" + "="*80)
    print("📋 PRODUCTION AD ACCOUNT SETUP OVERVIEW")
    print("="*80)
    
    print("\n🎯 WHAT THIS SETUP CREATES:")
    print("• Google Ads account with REAL billing ($100/day limit)")
    print("• Facebook Business Manager with ad account")
    print("• Conversion tracking pixels for all landing pages")
    print("• UTM parameter system with gaelp_uid tracking")
    print("• Real-time budget monitoring and emergency stops")
    print("• API access for programmatic campaign management")
    print("• Campaign blueprints ready for launch")
    
    print("\n💰 BUDGET PROTECTION:")
    print("• Daily limit: $100 across all platforms")
    print("• Monthly limit: $3000 total spend")
    print("• Emergency stop: $5000 absolute maximum")
    print("• Automated campaign pausing at thresholds")
    print("• Email/SMS alerts for budget concerns")
    print("• Real-time spend monitoring every 15 minutes")
    
    print("\n⚠️  REQUIREMENTS:")
    print("• REAL credit card (prepaid recommended)")
    print("• Business verification information")
    print("• Access to hari@aura.com Google account")
    print("• Facebook Business Manager access")
    print("• Domain ownership verification")
    print("• Phone number for SMS alerts")
    
    print("\n🚨 IMPORTANT REMINDERS:")
    print("• This spends REAL MONEY on REAL ads")
    print("• You are responsible for all costs")
    print("• Budget limits prevent overspend but YOU must monitor")
    print("• Test thoroughly before scaling spend")
    print("• Keep budget monitoring running at all times")

def run_safety_check():
    """Run comprehensive safety check"""
    
    print("\n" + "="*80)
    print("🛡️  SAFETY CHECK - REAL MONEY PROTECTION")
    print("="*80)
    
    safety_items = [
        "I understand this creates REAL ad accounts with REAL billing",
        "I have a credit card or prepaid card ready for billing setup",
        "I will monitor ad spend daily to avoid overspending",
        "I understand campaigns will be paused automatically at budget limits",
        "I have access to hari@aura.com for Google account setup",
        "I have Facebook Business Manager access for ad account creation",
        "I own teen-wellness-monitor.com for domain verification",
        "I will keep budget monitoring running before launching campaigns",
        "I accept responsibility for all advertising costs incurred"
    ]
    
    print("Please confirm each safety item:")
    
    for i, item in enumerate(safety_items, 1):
        while True:
            response = input(f"\n{i}. {item}\n   Confirm (y/n): ").strip().lower()
            if response == 'y':
                print("   ✅ Confirmed")
                break
            elif response == 'n':
                print("   ❌ Safety check failed")
                print(f"\n❌ Setup cancelled - safety requirement not met")
                return False
            else:
                print("   Please enter 'y' or 'n'")
    
    print("\n✅ ALL SAFETY CHECKS PASSED")
    return True

def demonstrate_setup_process():
    """Demonstrate the complete setup process"""
    
    print("\n" + "="*80)
    print("🚀 SETUP PROCESS DEMONSTRATION")
    print("="*80)
    
    setup_steps = [
        {
            'step': 'Initialize Production Manager',
            'description': 'Set up secure credential storage and budget limits',
            'command': 'ProductionAdAccountManager()',
            'time_estimate': '1 minute'
        },
        {
            'step': 'Google Ads Account Setup',
            'description': 'OAuth authentication, developer token, billing setup',
            'command': 'setup_google_ads_account()',
            'time_estimate': '10-15 minutes'
        },
        {
            'step': 'Facebook Business Manager Setup',
            'description': 'Business creation, ad account, pixel setup, domain verification',
            'command': 'setup_facebook_business_manager()',
            'time_estimate': '10-15 minutes'
        },
        {
            'step': 'Conversion Tracking Implementation',
            'description': 'Landing page pixels, server-side API, enhanced conversions',
            'command': 'implement_conversion_tracking()',
            'time_estimate': '5 minutes'
        },
        {
            'step': 'UTM Parameter System',
            'description': 'GAELP tracking parameters, URL builder, signature validation',
            'command': 'implement_utm_tracking_system()',
            'time_estimate': '2 minutes'
        },
        {
            'step': 'Budget Safety Monitoring',
            'description': 'Real-time spend tracking, alerts, emergency pausing',
            'command': 'setup_budget_monitoring()',
            'time_estimate': '3 minutes'
        },
        {
            'step': 'Campaign Structure Creation',
            'description': 'Initial test campaigns with proper tracking URLs',
            'command': 'create_initial_campaigns()',
            'time_estimate': '5 minutes'
        },
        {
            'step': 'Complete Validation',
            'description': 'Verify all components are properly configured',
            'command': 'validate_complete_setup()',
            'time_estimate': '2 minutes'
        }
    ]
    
    total_time = 0
    print(f"📋 COMPLETE SETUP PROCESS ({len(setup_steps)} steps):\n")
    
    for i, step in enumerate(setup_steps, 1):
        print(f"{i}. {step['step']}")
        print(f"   📝 {step['description']}")
        print(f"   ⚡ {step['command']}")
        print(f"   ⏱️  Estimated time: {step['time_estimate']}")
        
        # Extract time estimate
        time_parts = step['time_estimate'].split('-')
        if len(time_parts) == 2:
            total_time += int(time_parts[1].split()[0])
        else:
            total_time += int(time_parts[0].split()[0])
        
        print()
    
    print(f"⏱️  TOTAL ESTIMATED TIME: ~{total_time} minutes")
    print("⚠️  Actual time may vary based on platform approval times")

def show_post_setup_instructions():
    """Show what to do after setup is complete"""
    
    print("\n" + "="*80)
    print("📋 POST-SETUP INSTRUCTIONS")
    print("="*80)
    
    print("\n🚀 IMMEDIATE NEXT STEPS:")
    print("1. Start budget monitoring service (CRITICAL)")
    print("   python3 start_budget_monitoring.py")
    print()
    print("2. Deploy landing page tracking code")
    print("   Upload landing_page_template.html to web server")
    print("   Test conversion tracking is firing")
    print()
    print("3. Launch initial test campaigns")
    print("   python3 launch_test_campaigns.py")
    print("   Start with $50/day total budget")
    print()
    print("4. Monitor performance closely")
    print("   Check spend every hour on first day")
    print("   Validate conversions are tracking")
    print("   Monitor email alerts")
    
    print("\n⚠️  CRITICAL MONITORING:")
    print("• Budget monitoring MUST be running before launching campaigns")
    print("• Check email alerts and respond to budget warnings")
    print("• Review spend daily - you are responsible for all costs")
    print("• Test conversion tracking thoroughly before scaling")
    print("• Keep emergency contact information updated")
    
    print("\n📊 SUCCESS METRICS:")
    print("• Conversion tracking firing correctly")
    print("• Budget limits preventing overspend")
    print("• Campaign performance data flowing")
    print("• API access working for both platforms")
    print("• GAELP parameters capturing properly")

def main():
    """Main quick start demonstration"""
    
    print("🚀 GAELP PRODUCTION AD ACCOUNT SETUP - QUICK START")
    print("="*80)
    print("REAL MONEY - REAL ACCOUNTS - COMPLETE INFRASTRUCTURE")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met - please install requirements")
        return
    
    # Step 2: Show setup overview
    show_setup_overview()
    
    # Step 3: Run safety check
    if not run_safety_check():
        return
    
    # Step 4: Demonstrate setup process
    demonstrate_setup_process()
    
    # Step 5: Show post-setup instructions
    show_post_setup_instructions()
    
    # Final confirmation
    print("\n" + "="*80)
    print("🎯 READY TO PROCEED WITH PRODUCTION SETUP")
    print("="*80)
    
    proceed = input("\nRun complete production setup now? (y/n): ").strip().lower()
    
    if proceed == 'y':
        print("\n🚀 Starting complete production setup...")
        print("⚠️  REAL MONEY WILL BE INVOLVED")
        
        try:
            # Import and run the complete setup
            from setup_production_ads import ProductionAdSetupOrchestrator
            
            orchestrator = ProductionAdSetupOrchestrator()
            result = orchestrator.run_complete_setup()
            
            if result['success']:
                print(f"\n🎉 SETUP COMPLETE!")
                print("Check the generated README for next steps")
            else:
                print(f"\n❌ Setup failed: {result.get('error', 'Unknown error')}")
                
        except ImportError as e:
            print(f"\n❌ Import error: {e}")
            print("Make sure all production setup files are in place")
        except Exception as e:
            print(f"\n❌ Setup error: {e}")
            
    else:
        print("\n⏹️  Setup cancelled")
        print("\nTo run setup later:")
        print("python3 setup_production_ads.py")
    
    print(f"\n📁 All files located in: {Path.home() / '.config' / 'gaelp'}")
    print("⚠️  REMEMBER: This involves REAL MONEY - monitor carefully!")

if __name__ == "__main__":
    main()