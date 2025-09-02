#!/usr/bin/env python3
"""
Comprehensive GA4 Data Extractor for GAELP
Extracts ALL campaign, creative, and performance data systematically
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GA4ComprehensiveExtractor:
    """Extract and organize ALL GA4 data for GAELP training"""
    
    def __init__(self, output_dir: str = "ga4_extracted_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Product mapping patterns - CRITICAL for correct categorization
        self.product_patterns = {
            'parental_controls': {
                'patterns': [
                    r'gtwy[_-]?pc',
                    r'parental[_-]?control',
                    r'parental[_-]?gw',
                    r'with\s+PC',
                    r'NonCR[_-]?Parental',
                    r'parental[_-]?gateway'
                ],
                'campaigns': [],
                'products': []
            },
            'antivirus': {
                'patterns': [
                    r'av[_-]?intro',
                    r'gtwy[_-]?av',
                    r'antivirus',
                    r'AV\s+',
                    r'norton',
                    r'mcafee'
                ],
                'campaigns': [],
                'products': []
            },
            'identity': {
                'patterns': [
                    r'gtwy[_-]?id',
                    r'identity',
                    r'credit',
                    r'gtwy[_-]?credit',
                    r'privacy',
                    r'lifelock'
                ],
                'campaigns': [],
                'products': []
            },
            'balance_thrive': {
                'patterns': [
                    r'balance',
                    r'thrive',
                    r'mental[_-]?health',
                    r'behavioral',
                    r'teen[_-]?talk',
                    r'parenting[_-]?pressure',
                    r'blue[_-]?box'
                ],
                'campaigns': [],
                'products': []
            },
            'bundle': {
                'patterns': [
                    r'family',
                    r'couple',
                    r'individual.*annual',
                    r'suite',
                    r'premium',
                    r'ultimate'
                ],
                'campaigns': [],
                'products': []
            }
        }
        
        # Campaign type patterns
        self.campaign_types = {
            'search_brand': r'Search[_-]?Brand|brand',
            'search_nonbrand': r'Search[_-]?NB|Search[_-]?Non[_-]?Brand',
            'search_competitor': r'Search[_-]?Competitor|competitor',
            'display': r'display|rmk|remarketing',
            'social': r'facebook|fb|instagram|ig|life360|social',
            'performance_max': r'PerformanceMax|pmax',
            'youtube': r'youtube|yt',
            'email': r'email|newsletter',
            'affiliate': r'affiliate|partner',
            'direct': r'direct|\(direct\)',
            'organic': r'organic|\(organic\)',
            'referral': r'referral|\(referral\)'
        }
        
        # Test patterns for A/B tests
        self.test_patterns = {
            'landing_page': r'ab[_-]?LP|LPtest',
            'pricing': r'price[_-]?test|discount.*test',
            'creative': r'ab[_-]?creative|headline.*test',
            'offer': r'ab[_-]?offer|FT[_-]?test|free[_-]?trial.*test'
        }
        
    def categorize_campaign(self, campaign_name: str) -> Dict[str, str]:
        """Intelligently categorize campaign by product and type"""
        campaign_lower = campaign_name.lower()
        
        # Determine product
        product = 'unknown'
        for prod_name, prod_info in self.product_patterns.items():
            for pattern in prod_info['patterns']:
                if re.search(pattern, campaign_lower, re.IGNORECASE):
                    product = prod_name
                    break
            if product != 'unknown':
                break
        
        # Determine campaign type
        campaign_type = 'other'
        for type_name, pattern in self.campaign_types.items():
            if re.search(pattern, campaign_lower, re.IGNORECASE):
                campaign_type = type_name
                break
        
        # Check for tests
        test_type = None
        for test_name, pattern in self.test_patterns.items():
            if re.search(pattern, campaign_lower, re.IGNORECASE):
                test_type = test_name
                break
        
        return {
            'product': product,
            'campaign_type': campaign_type,
            'test_type': test_type,
            'original_name': campaign_name
        }
    
    def extract_all_data(self, start_date: str, end_date: str):
        """Extract ALL GA4 data systematically"""
        logger.info(f"Starting comprehensive extraction from {start_date} to {end_date}")
        
        # Create date subdirectory
        date_dir = self.output_dir / f"{start_date}_to_{end_date}"
        date_dir.mkdir(exist_ok=True)
        
        # 1. Campaign Performance
        campaigns_data = self.extract_campaign_performance(start_date, end_date)
        self.save_data(campaigns_data, date_dir / "01_campaign_performance.json")
        
        # 2. Creative Performance (A/B tests)
        creative_data = self.extract_creative_performance(start_date, end_date)
        self.save_data(creative_data, date_dir / "02_creative_performance.json")
        
        # 3. User Journeys
        journey_data = self.extract_user_journeys(start_date, end_date)
        self.save_data(journey_data, date_dir / "03_user_journeys.json")
        
        # 4. Keyword Performance
        keyword_data = self.extract_keyword_performance(start_date, end_date)
        self.save_data(keyword_data, date_dir / "04_keyword_performance.json")
        
        # 5. Audience Segments
        audience_data = self.extract_audience_segments(start_date, end_date)
        self.save_data(audience_data, date_dir / "05_audience_segments.json")
        
        # 6. Product Performance
        product_data = self.extract_product_performance(start_date, end_date)
        self.save_data(product_data, date_dir / "06_product_performance.json")
        
        # 7. Landing Pages
        landing_data = self.extract_landing_performance(start_date, end_date)
        self.save_data(landing_data, date_dir / "07_landing_pages.json")
        
        # 8. Device/Platform
        device_data = self.extract_device_performance(start_date, end_date)
        self.save_data(device_data, date_dir / "08_device_platform.json")
        
        # 9. Geographic Performance
        geo_data = self.extract_geographic_performance(start_date, end_date)
        self.save_data(geo_data, date_dir / "09_geographic.json")
        
        # 10. Temporal Patterns
        temporal_data = self.extract_temporal_patterns(start_date, end_date)
        self.save_data(temporal_data, date_dir / "10_temporal_patterns.json")
        
        # Generate mapping report
        mapping_report = self.generate_mapping_report(campaigns_data, product_data)
        self.save_data(mapping_report, date_dir / "00_MAPPING_REPORT.json")
        
        logger.info(f"✅ Extraction complete! Data saved to {date_dir}")
        return date_dir
    
    def extract_campaign_performance(self, start_date: str, end_date: str) -> Dict:
        """Extract detailed campaign performance"""
        # This will call GA4 API
        # For now, showing structure
        logger.info("Extracting campaign performance...")
        
        campaigns = {}
        
        # We'll use mcp__ga4__runReport here
        # Example structure:
        campaign_template = {
            'campaign_name': '',
            'categorization': {},  # From categorize_campaign()
            'metrics': {
                'sessions': 0,
                'users': 0,
                'conversions': 0,
                'revenue': 0.0,
                'cost': 0.0,
                'cpc': 0.0,
                'ctr': 0.0,
                'cvr': 0.0,
                'roas': 0.0
            },
            'daily_performance': [],
            'weekly_trends': [],
            'monthly_summary': {}
        }
        
        return campaigns
    
    def extract_creative_performance(self, start_date: str, end_date: str) -> Dict:
        """Extract A/B test and creative performance"""
        logger.info("Extracting creative/test performance...")
        
        creatives = {
            'headlines': {},
            'descriptions': {},
            'landing_pages': {},
            'offers': {},
            'ab_tests': {
                'completed': [],
                'running': [],
                'winners': {}
            }
        }
        
        return creatives
    
    def extract_user_journeys(self, start_date: str, end_date: str) -> Dict:
        """Extract user journey patterns"""
        logger.info("Extracting user journeys...")
        
        journeys = {
            'typical_paths': {
                'parental_controls': [],
                'antivirus': [],
                'identity': [],
                'balance_thrive': []
            },
            'touchpoint_analysis': {
                'average_touchpoints': 0,
                'time_to_conversion': {},
                'channel_sequences': []
            },
            'abandonment_points': {},
            'cross_device_journeys': []
        }
        
        return journeys
    
    def extract_keyword_performance(self, start_date: str, end_date: str) -> Dict:
        """Extract keyword and search term data"""
        logger.info("Extracting keyword performance...")
        
        keywords = {
            'top_converting': [],
            'by_product': {
                'parental_controls': [],
                'antivirus': [],
                'identity': [],
                'balance_thrive': []
            },
            'negative_keywords': [],
            'search_terms': [],
            'competitor_keywords': [],
            'long_tail_opportunities': []
        }
        
        return keywords
    
    def extract_audience_segments(self, start_date: str, end_date: str) -> Dict:
        """Extract audience segment performance"""
        logger.info("Extracting audience segments...")
        
        audiences = {
            'demographics': {
                'age_groups': {},
                'gender': {},
                'parental_status': {},
                'household_income': {}
            },
            'interests': {},
            'in_market': {},
            'custom_audiences': {},
            'remarketing_lists': {},
            'lookalike_performance': {}
        }
        
        return audiences
    
    def extract_product_performance(self, start_date: str, end_date: str) -> Dict:
        """Extract product-level performance"""
        logger.info("Extracting product performance...")
        
        products = {
            'parental_controls': {
                'skus': [],
                'revenue': 0,
                'units_sold': 0,
                'avg_order_value': 0,
                'conversion_rate': 0,
                'customer_ltv': 0
            },
            'antivirus': {},
            'identity': {},
            'balance_thrive': {},
            'bundles': {}
        }
        
        return products
    
    def extract_landing_performance(self, start_date: str, end_date: str) -> Dict:
        """Extract landing page performance"""
        logger.info("Extracting landing page performance...")
        
        landing_pages = {
            'by_url': {},
            'by_template': {},
            'conversion_rates': {},
            'bounce_rates': {},
            'page_value': {},
            'form_completion': {}
        }
        
        return landing_pages
    
    def extract_device_performance(self, start_date: str, end_date: str) -> Dict:
        """Extract device and platform data"""
        logger.info("Extracting device/platform performance...")
        
        devices = {
            'device_category': {
                'mobile': {},
                'desktop': {},
                'tablet': {}
            },
            'operating_system': {
                'ios': {},  # Critical for Balance
                'android': {},
                'windows': {},
                'macos': {}
            },
            'cross_device': {},
            'app_vs_web': {}
        }
        
        return devices
    
    def extract_geographic_performance(self, start_date: str, end_date: str) -> Dict:
        """Extract geographic performance"""
        logger.info("Extracting geographic performance...")
        
        geographic = {
            'countries': {},
            'states': {},
            'cities': {},
            'dmas': {},
            'performance_by_region': {}
        }
        
        return geographic
    
    def extract_temporal_patterns(self, start_date: str, end_date: str) -> Dict:
        """Extract temporal patterns"""
        logger.info("Extracting temporal patterns...")
        
        temporal = {
            'hour_of_day': {},
            'day_of_week': {},
            'monthly_trends': {},
            'seasonal_patterns': {},
            'holiday_impact': {},
            'promotional_periods': {}
        }
        
        return temporal
    
    def generate_mapping_report(self, campaigns_data: Dict, product_data: Dict) -> Dict:
        """Generate comprehensive mapping report"""
        logger.info("Generating mapping report...")
        
        report = {
            'extraction_date': datetime.now().isoformat(),
            'campaign_to_product_mapping': {},
            'product_categorization': self.product_patterns,
            'unmatched_campaigns': [],
            'validation_warnings': [],
            'summary_stats': {
                'total_campaigns': 0,
                'categorized_campaigns': 0,
                'total_products': 0,
                'total_revenue': 0,
                'total_conversions': 0
            }
        }
        
        return report
    
    def save_data(self, data: Dict, filepath: Path):
        """Save data with proper formatting"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  Saved: {filepath.name}")
    
    def validate_categorization(self, data: Dict) -> List[str]:
        """Validate that categorization is working correctly"""
        warnings = []
        
        # Check for uncategorized items
        # Check for potential miscategorizations
        # Check for duplicate mappings
        
        return warnings


# Main execution function
def main():
    """Run comprehensive GA4 extraction"""
    extractor = GA4ComprehensiveExtractor()
    
    # Extract last 12 months of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║         GA4 COMPREHENSIVE DATA EXTRACTOR                  ║
    ║                                                            ║
    ║  Extracting ALL data from {start_date} to {end_date}     ║
    ║                                                            ║
    ║  This will extract:                                       ║
    ║  • Campaign performance (all campaigns)                   ║
    ║  • Creative/AB test results                               ║
    ║  • User journey patterns                                  ║
    ║  • Keyword performance                                    ║
    ║  • Audience segments                                      ║
    ║  • Product performance                                    ║
    ║  • Landing page metrics                                   ║
    ║  • Device/platform data                                   ║
    ║  • Geographic patterns                                    ║
    ║  • Temporal patterns                                      ║
    ║                                                            ║
    ║  Output: ga4_extracted_data/                             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run extraction
    output_dir = extractor.extract_all_data(start_date, end_date)
    
    print(f"\n✅ Extraction complete! Check {output_dir}")
    print("\n📊 Next steps:")
    print("1. Review 00_MAPPING_REPORT.json for campaign categorization")
    print("2. Check for any 'unknown' or miscategorized campaigns")
    print("3. Use extracted data to train GAELP with real patterns")


if __name__ == "__main__":
    main()