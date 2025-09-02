#!/usr/bin/env python3
"""
Extract August 2025 GA4 Data - REAL Implementation
Using MCP tools to pull actual Aura data
"""

import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_august_campaigns():
    """Extract August campaign performance"""
    logger.info("Extracting August 2025 campaign data...")
    
    # We'll call the MCP GA4 tools directly from Claude Code
    # This script serves as documentation of what we need
    
    campaign_request = {
        'startDate': '2025-08-01',
        'endDate': '2025-08-31',
        'dimensions': [
            {'name': 'campaignName'},
            {'name': 'source'},
            {'name': 'medium'},
            {'name': 'date'}
        ],
        'metrics': [
            {'name': 'sessions'},
            {'name': 'totalUsers'},
            {'name': 'conversions'},
            {'name': 'purchaseRevenue'},
            {'name': 'totalAdCost'},
            {'name': 'averageSessionDuration'},
            {'name': 'bounceRate'}
        ]
    }
    
    logger.info("Campaign request prepared:")
    logger.info(json.dumps(campaign_request, indent=2))
    
    return campaign_request


def extract_august_products():
    """Extract August product performance"""
    logger.info("Extracting August 2025 product data...")
    
    product_request = {
        'startDate': '2025-08-01',
        'endDate': '2025-08-31',
        'dimensions': [
            {'name': 'itemName'},
            {'name': 'itemCategory'},
            {'name': 'date'}
        ],
        'metrics': [
            {'name': 'itemsPurchased'},
            {'name': 'itemRevenue'},
            {'name': 'itemsViewed'},
            {'name': 'cartToViewRate'},
            {'name': 'purchaseToViewRate'}
        ]
    }
    
    logger.info("Product request prepared:")
    logger.info(json.dumps(product_request, indent=2))
    
    return product_request


def extract_august_events():
    """Extract August event data (AB tests, etc)"""
    logger.info("Extracting August 2025 event data...")
    
    event_request = {
        'startDate': '2025-08-01',
        'endDate': '2025-08-31',
        'eventName': None  # Get all events
    }
    
    logger.info("Event request prepared:")
    logger.info(json.dumps(event_request, indent=2))
    
    return event_request


def extract_august_behavior():
    """Extract August user behavior"""
    logger.info("Extracting August 2025 user behavior...")
    
    behavior_request = {
        'startDate': '2025-08-01',
        'endDate': '2025-08-31'
    }
    
    logger.info("Behavior request prepared:")
    logger.info(json.dumps(behavior_request, indent=2))
    
    return behavior_request


def extract_august_landing_pages():
    """Extract August landing page performance"""
    logger.info("Extracting August 2025 landing page data...")
    
    landing_request = {
        'startDate': '2025-08-01',
        'endDate': '2025-08-31',
        'dimensions': [
            {'name': 'landingPagePlusQueryString'},
            {'name': 'source'},
            {'name': 'medium'}
        ],
        'metrics': [
            {'name': 'sessions'},
            {'name': 'bounceRate'},
            {'name': 'conversions'},
            {'name': 'averageSessionDuration'}
        ]
    }
    
    logger.info("Landing page request prepared:")
    logger.info(json.dumps(landing_request, indent=2))
    
    return landing_request


def extract_august_devices():
    """Extract August device/platform data"""
    logger.info("Extracting August 2025 device data...")
    
    device_request = {
        'startDate': '2025-08-01', 
        'endDate': '2025-08-31',
        'dimensions': [
            {'name': 'deviceCategory'},
            {'name': 'operatingSystem'},
            {'name': 'platform'}
        ],
        'metrics': [
            {'name': 'sessions'},
            {'name': 'totalUsers'},
            {'name': 'conversions'},
            {'name': 'purchaseRevenue'}
        ]
    }
    
    logger.info("Device request prepared:")
    logger.info(json.dumps(device_request, indent=2))
    
    return device_request


def main():
    """Prepare all extraction requests"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          AUGUST 2025 GA4 DATA EXTRACTION SETUP                ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Preparing extraction requests for:
    1. Campaign Performance
    2. Product Sales (SKUs)
    3. Events (AB tests, etc)
    4. User Behavior
    5. Landing Pages
    6. Device/Platform
    
    """)
    
    # Prepare all requests
    requests = {
        'campaigns': extract_august_campaigns(),
        'products': extract_august_products(),
        'events': extract_august_events(),
        'behavior': extract_august_behavior(),
        'landing_pages': extract_august_landing_pages(),
        'devices': extract_august_devices()
    }
    
    # Save request templates
    output_dir = Path("ga4_extracted_data")
    output_dir.mkdir(exist_ok=True)
    
    request_file = output_dir / "august_2025_requests.json"
    with open(request_file, 'w') as f:
        json.dump(requests, f, indent=2)
    
    print(f"✅ Request templates saved to: {request_file}")
    print("\n📊 Now execute these requests using MCP GA4 tools to get real data")
    print("\nNext steps:")
    print("1. Run campaign report to get all August campaigns")
    print("2. Run product report to verify PC sales (~60/day)")
    print("3. Extract events for AB test data")
    print("4. Get user behavior for journey patterns")
    print("5. Analyze landing page performance")
    print("6. Check device breakdown (iOS for Balance!)")


if __name__ == "__main__":
    main()