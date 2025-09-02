#!/usr/bin/env python3
"""
Check if you have Google Ads access and what accounts are available
"""

import webbrowser
import sys

def check_access():
    print("=" * 80)
    print("GOOGLE ADS ACCESS CHECK")
    print("=" * 80)
    print()
    print("Let's check if you have Google Ads access...")
    print()
    print("1. FIRST: Check if you have access to Google Ads")
    print("   Opening Google Ads in your browser...")
    print()
    
    # Open Google Ads
    url = "https://ads.google.com"
    print(f"   URL: {url}")
    print()
    print("   If you can log in and see campaigns/accounts, you have access!")
    print("   Look for your Customer ID in the top right (format: XXX-XXX-XXXX)")
    print()
    
    response = input("Do you want to open Google Ads now? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open(url)
    
    print()
    print("2. SECOND: Check API Access")
    print("   Go to: https://ads.google.com/aw/apicenter")
    print()
    print("   This is where you can:")
    print("   - Apply for API access (Basic access is free)")
    print("   - Get your Developer Token")
    print("   - See API usage and limits")
    print()
    
    response = input("Do you want to check API Center? (y/n): ")
    if response.lower() == 'y':
        webbrowser.open("https://ads.google.com/aw/apicenter")
    
    print()
    print("3. WHAT YOU NEED FOR THE MCP INTEGRATION:")
    print("   - Developer Token (from API Center)")
    print("   - OAuth2 credentials (from Google Cloud Console)")
    print("   - Customer ID (from Google Ads account)")
    print("   - Refresh Token (generated via OAuth flow)")
    print()
    print("4. IF YOU DON'T HAVE ACCESS:")
    print("   - You'll need someone to add you to the Google Ads account")
    print("   - OR create a test account (free) at:")
    print("     https://developers.google.com/google-ads/api/docs/first-call/dev-token#test-account")
    print()
    
    has_access = input("Do you have access to Google Ads? (y/n/not sure): ")
    
    if has_access.lower() == 'y':
        print()
        print("Great! Let's collect the information we have:")
        print()
        customer_id = input("Customer ID (if you have it, format XXX-XXX-XXXX): ").strip()
        developer_token = input("Developer Token (if you have it): ").strip()
        
        if customer_id or developer_token:
            print()
            print("Here's what we have so far:")
            if customer_id:
                print(f"  Customer ID: {customer_id}")
            if developer_token:
                print(f"  Developer Token: {developer_token[:10]}..." if len(developer_token) > 10 else developer_token)
            print()
            print("Next steps:")
            print("1. We need to set up OAuth2 credentials in Google Cloud Console")
            print("2. Generate a refresh token")
            print("3. Configure the MCP server with these credentials")
        else:
            print()
            print("No problem! We can get these from the Google Ads interface.")
    
    elif has_access.lower() == 'n':
        print()
        print("Options to get access:")
        print()
        print("1. TEST ACCOUNT (Recommended for development):")
        print("   - Free to create")
        print("   - No real money spent")
        print("   - Perfect for testing GAELP")
        print("   - Create at: https://developers.google.com/google-ads/api/docs/first-call/dev-token#test-account")
        print()
        print("2. PRODUCTION ACCOUNT:")
        print("   - Need to be added by account admin")
        print("   - Uses real money for ads")
        print("   - Better for analyzing real campaign data")
    
    else:
        print()
        print("Let's find out! Try opening https://ads.google.com")
        print("If you can log in with your Google account and see any campaigns or accounts,")
        print("then you have access. Otherwise, we'll need to set up a test account.")

if __name__ == "__main__":
    check_access()