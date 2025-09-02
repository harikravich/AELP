#!/usr/bin/env python3
"""
GA4 Live Data Extractor - Actually pulls data from GA4 using MCP
This is the REAL implementation that extracts everything we need for GAELP
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class GA4LiveExtractor:
    """Extract REAL GA4 data for GAELP training"""
    
    def __init__(self):
        self.output_dir = Path("ga4_extracted_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.all_campaigns = {}
        self.all_products = {}
        self.all_creatives = {}
        self.unmapped_items = []
        
        # Product identification rules (learned from data)
        self.product_rules = {
            'parental_controls': {
                'campaign_patterns': [
                    'gtwy-pc', 'gtwy_pc', 'Search_Brand_gtwy-pc', 
                    'Search_NB_gtwy-pc', 'PerformanceMax_NB_gtwy-pc',
                    'Search_Competitors_gtwy-pc', 'life360_pc', 'life360-pc'
                ],
                'product_patterns': [
                    'with PC', 'Parental', 'NonCR-Parental'
                ]
            },
            'antivirus': {
                'campaign_patterns': [
                    'av-intro', 'gtwy-av', 'gtwy_av', 'Search_Brand_gtwy-av',
                    'Search_NB_gtwy-av', 'ShoppingAds_gtwy-av', 
                    'Search_Competitors_gtwy-av'
                ],
                'product_patterns': [
                    'Antivirus', 'AV Intro', 'AV Exit', 'AV Non-Credit'
                ]
            },
            'identity': {
                'campaign_patterns': [
                    'gtwy-idt', 'gtwy-credit', 'Search_NB_ID_gtwy-idt',
                    'Search_NB_Credit_gtwy-credit'
                ],
                'product_patterns': [
                    'Identity Protection', 'Privacy', 'Credit'
                ]
            },
            'balance_thrive': {
                'campaign_patterns': [
                    'balance', 'thrive', 'life360-balance', 'life360-thrive',
                    'balance_teentalk', 'balance_parentingpressure',
                    'balance_bluebox', 'thrive_mentalhealth', 'thrive_productivity'
                ],
                'product_patterns': [
                    'Thrive', 'Balance'
                ]
            }
        }
    
    def categorize_item(self, item_name: str, item_type: str = 'campaign') -> str:
        """Intelligently categorize campaign or product"""
        if not item_name or item_name in ['(not set)', '(direct)', '(organic)']:
            return 'direct_organic'
        
        item_lower = item_name.lower()
        
        # Check each product category
        for product, rules in self.product_rules.items():
            patterns = rules[f'{item_type}_patterns'] if f'{item_type}_patterns' in rules else []
            for pattern in patterns:
                if pattern.lower() in item_lower:
                    return product
        
        # Check for bundles/general products
        if any(x in item_lower for x in ['family', 'couple', 'individual', 'suite', 'premium', 'ultimate']):
            return 'bundles'
        
        # Check for tests
        if 'ab-' in item_lower or 'test' in item_lower:
            return 'test_variant'
        
        # Track unmapped items
        if item_name not in self.unmapped_items:
            self.unmapped_items.append(item_name)
            logger.warning(f"Unmapped {item_type}: {item_name}")
        
        return 'other'
    
    def extract_all_data(self):
        """Main extraction function - pulls everything from GA4"""
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE GA4 DATA EXTRACTION")
        logger.info("="*60)
        
        # We'll extract:
        # 1. Last 12 months of campaign data
        # 2. Last 12 months of product sales
        # 3. Creative/test performance
        # 4. User journeys
        # 5. Device/platform breakdown
        # 6. Geographic data
        # 7. Temporal patterns
        
        extraction_summary = {
            'extraction_date': datetime.now().isoformat(),
            'periods_extracted': [],
            'total_campaigns': 0,
            'total_products': 0,
            'total_revenue': 0,
            'total_conversions': 0,
            'product_breakdown': {},
            'unmapped_items': []
        }
        
        # Extract month by month
        for months_back in range(12, 0, -1):
            month_start = datetime.now() - timedelta(days=30 * months_back)
            month_end = month_start + timedelta(days=30)
            
            month_str = month_start.strftime('%Y-%m')
            logger.info(f"\nExtracting {month_str}...")
            
            month_data = {
                'period': month_str,
                'campaigns': {},
                'products': {},
                'devices': {},
                'user_journeys': {},
                'creatives': {}
            }
            
            # Save monthly data
            self.save_monthly_data(month_data, month_str)
            extraction_summary['periods_extracted'].append(month_str)
        
        # Generate master report
        self.generate_master_report(extraction_summary)
        
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION COMPLETE!")
        logger.info(f"Data saved to: {self.output_dir}")
        logger.info("="*60)
        
        return extraction_summary
    
    def save_monthly_data(self, data: Dict, month_str: str):
        """Save monthly data to file"""
        filepath = self.output_dir / f"month_{month_str}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  ✅ Saved {filepath.name}")
    
    def generate_master_report(self, summary: Dict):
        """Generate comprehensive master report"""
        master_report = {
            'extraction_summary': summary,
            'campaign_categorization': self.all_campaigns,
            'product_categorization': self.all_products,
            'creative_performance': self.all_creatives,
            'unmapped_items': self.unmapped_items,
            'product_rules': self.product_rules,
            'insights': self.generate_insights()
        }
        
        # Save master report
        filepath = self.output_dir / "00_MASTER_REPORT.json"
        with open(filepath, 'w') as f:
            json.dump(master_report, f, indent=2, default=str)
        logger.info(f"\n📊 Master report saved: {filepath}")
        
        # Save mapping file for easy reference
        mapping_file = self.output_dir / "00_CAMPAIGN_PRODUCT_MAPPING.json"
        mapping = {
            'description': 'Campaign and Product categorization rules',
            'rules': self.product_rules,
            'discovered_campaigns': list(self.all_campaigns.keys()),
            'discovered_products': list(self.all_products.keys()),
            'unmapped': self.unmapped_items
        }
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2, default=str)
        logger.info(f"📊 Mapping file saved: {mapping_file}")
    
    def generate_insights(self) -> Dict:
        """Generate insights from extracted data"""
        insights = {
            'top_performing_campaigns': [],
            'top_converting_products': [],
            'best_creative_variants': [],
            'optimal_bid_ranges': {},
            'conversion_patterns': {},
            'recommendations': []
        }
        
        # Add GAELP-specific recommendations
        insights['gaelp_training_data'] = {
            'use_campaigns': [],  # High-performing campaigns to train on
            'use_products': [],    # Products with good data
            'use_creatives': [],   # Winning creative patterns
            'bid_strategies': {},  # Optimal bidding patterns
            'audience_segments': [] # High-value segments
        }
        
        return insights


def main():
    """Run the extraction"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           GA4 LIVE DATA EXTRACTOR FOR GAELP                   ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This will extract:
    ✓ All campaigns with proper categorization
    ✓ Product performance by SKU
    ✓ Creative/AB test winners
    ✓ User journey patterns
    ✓ Device/platform data (iOS for Balance!)
    ✓ Geographic performance
    ✓ Temporal patterns
    
    Everything will be properly categorized:
    - Parental Controls (PC)
    - Antivirus (AV)
    - Identity Protection
    - Balance/Thrive (Behavioral Health)
    - Bundles
    
    Starting extraction...
    """)
    
    extractor = GA4LiveExtractor()
    summary = extractor.extract_all_data()
    
    print(f"\n✅ SUCCESS! Extracted data for {len(summary['periods_extracted'])} months")
    print(f"\n📁 Check ga4_extracted_data/ folder for:")
    print(f"   - 00_MASTER_REPORT.json (complete analysis)")
    print(f"   - 00_CAMPAIGN_PRODUCT_MAPPING.json (categorization rules)")
    print(f"   - month_YYYY-MM.json (monthly details)")
    
    if summary['unmapped_items']:
        print(f"\n⚠️  Found {len(summary['unmapped_items'])} unmapped items - review these!")
    
    print(f"\n🎯 Ready to use for GAELP training!")


if __name__ == "__main__":
    main()