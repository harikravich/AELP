#!/usr/bin/env python3
"""
Google Ads Integration for GAELP
Pull real CPC, bid, and competition data from Google Ads
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

class GoogleAdsIntegration:
    """
    Connect to Google Ads API to get real bid/CPC data
    """
    
    def __init__(self, customer_id: str = None):
        """
        Initialize Google Ads client
        
        Args:
            customer_id: Google Ads customer ID (without hyphens)
        """
        # Configuration for Google Ads API
        self.config = {
            'developer_token': os.environ.get('GOOGLE_ADS_DEVELOPER_TOKEN'),
            'client_id': os.environ.get('GOOGLE_ADS_CLIENT_ID'),
            'client_secret': os.environ.get('GOOGLE_ADS_CLIENT_SECRET'),
            'refresh_token': os.environ.get('GOOGLE_ADS_REFRESH_TOKEN'),
            'customer_id': customer_id or os.environ.get('GOOGLE_ADS_CUSTOMER_ID')
        }
        
        # Initialize client (will need credentials)
        self.client = None
        self.customer_id = self.config['customer_id']
        
    def setup_credentials(self):
        """
        Setup Google Ads API credentials
        """
        print("=" * 80)
        print("GOOGLE ADS API SETUP")
        print("=" * 80)
        print("\nTo use Google Ads API, you need:")
        print("1. Developer Token from Google Ads API Center")
        print("2. OAuth2 credentials (client_id, client_secret, refresh_token)")
        print("3. Customer ID for the Google Ads account")
        print("\nSteps to get credentials:")
        print("1. Go to: https://ads.google.com/aw/apicenter")
        print("2. Apply for API access (Basic access is free)")
        print("3. Create OAuth2 credentials in Google Cloud Console")
        print("4. Use the oauth2 flow to get refresh token")
        print("\nSet these environment variables:")
        print("- GOOGLE_ADS_DEVELOPER_TOKEN")
        print("- GOOGLE_ADS_CLIENT_ID")
        print("- GOOGLE_ADS_CLIENT_SECRET")
        print("- GOOGLE_ADS_REFRESH_TOKEN")
        print("- GOOGLE_ADS_CUSTOMER_ID")
        
    def get_keyword_performance(self, date_range_days: int = 30) -> List[Dict]:
        """
        Get keyword performance data including CPCs and competition
        
        Args:
            date_range_days: Number of days to look back
            
        Returns:
            List of keyword performance data
        """
        if not self.client:
            print("❌ Google Ads client not initialized. Run setup_credentials() first.")
            return []
        
        ga_service = self.client.get_service("GoogleAdsService")
        
        # Query for keyword performance
        query = """
            SELECT
                campaign.name,
                ad_group.name,
                ad_group_criterion.keyword.text,
                ad_group_criterion.keyword.match_type,
                metrics.average_cpc,
                metrics.cost_micros,
                metrics.clicks,
                metrics.impressions,
                metrics.conversions,
                metrics.search_impression_share,
                metrics.search_top_impression_share,
                metrics.search_absolute_top_impression_share,
                metrics.search_rank_lost_impression_share,
                metrics.search_budget_lost_impression_share
            FROM
                keyword_view
            WHERE
                segments.date DURING LAST_{days}_DAYS
                AND campaign.status = 'ENABLED'
                AND ad_group.status = 'ENABLED'
            ORDER BY
                metrics.cost_micros DESC
        """.format(days=date_range_days)
        
        try:
            response = ga_service.search_stream(
                customer_id=self.customer_id, 
                query=query
            )
            
            keywords = []
            for batch in response:
                for row in batch.results:
                    keyword_data = {
                        'campaign': row.campaign.name,
                        'ad_group': row.ad_group.name,
                        'keyword': row.ad_group_criterion.keyword.text,
                        'match_type': row.ad_group_criterion.keyword.match_type.name,
                        'avg_cpc': row.metrics.average_cpc / 1_000_000,  # Convert micros to dollars
                        'total_cost': row.metrics.cost_micros / 1_000_000,
                        'clicks': row.metrics.clicks,
                        'impressions': row.metrics.impressions,
                        'conversions': row.metrics.conversions,
                        'impression_share': row.metrics.search_impression_share,
                        'top_impression_share': row.metrics.search_top_impression_share,
                        'absolute_top_share': row.metrics.search_absolute_top_impression_share,
                        'rank_lost_share': row.metrics.search_rank_lost_impression_share,
                        'budget_lost_share': row.metrics.search_budget_lost_impression_share
                    }
                    keywords.append(keyword_data)
            
            return keywords
            
        except GoogleAdsException as ex:
            print(f"❌ Google Ads API error: {ex}")
            return []
    
    def get_auction_insights(self) -> List[Dict]:
        """
        Get auction insights to see competitor bidding behavior
        
        Returns:
            List of competitor insights
        """
        if not self.client:
            print("❌ Google Ads client not initialized")
            return []
        
        ga_service = self.client.get_service("GoogleAdsService")
        
        # Query for auction insights
        query = """
            SELECT
                campaign.name,
                auction_insight_search_impression_share,
                auction_insight_overlap_rate,
                auction_insight_outranking_share,
                auction_insight_position_above_rate,
                auction_insight_top_impression_percentage,
                auction_insight_absolute_top_impression_percentage
            FROM
                campaign_auction_insight
            WHERE
                segments.date DURING LAST_30_DAYS
        """
        
        try:
            response = ga_service.search_stream(
                customer_id=self.customer_id,
                query=query
            )
            
            insights = []
            for batch in response:
                for row in batch.results:
                    # Process auction insights
                    insight = {
                        'campaign': row.campaign.name,
                        'impression_share': row.auction_insight_search_impression_share,
                        'overlap_rate': row.auction_insight_overlap_rate,
                        'outranking_share': row.auction_insight_outranking_share,
                        'position_above_rate': row.auction_insight_position_above_rate,
                        'top_impression_pct': row.auction_insight_top_impression_percentage,
                        'absolute_top_pct': row.auction_insight_absolute_top_impression_percentage
                    }
                    insights.append(insight)
            
            return insights
            
        except GoogleAdsException as ex:
            print(f"❌ Error getting auction insights: {ex}")
            return []
    
    def estimate_keyword_bids(self, keywords: List[str]) -> Dict[str, Dict]:
        """
        Get bid estimates for keywords using Keyword Planner
        
        Args:
            keywords: List of keywords to estimate
            
        Returns:
            Dictionary of keyword -> bid estimates
        """
        if not self.client:
            print("❌ Google Ads client not initialized")
            return {}
        
        keyword_plan_idea_service = self.client.get_service("KeywordPlanIdeaService")
        keyword_competition_level_enum = self.client.enums.KeywordPlanCompetitionLevelEnum
        
        request = self.client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = self.customer_id
        request.keyword_plan_network = self.client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
        
        # Add keywords to request
        for keyword in keywords:
            request.keyword_seed.keywords.append(keyword)
        
        try:
            response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
            
            estimates = {}
            for idea in response:
                keyword_text = idea.text
                estimates[keyword_text] = {
                    'avg_monthly_searches': idea.keyword_idea_metrics.avg_monthly_searches,
                    'competition': idea.keyword_idea_metrics.competition.name,
                    'competition_index': idea.keyword_idea_metrics.competition_index,
                    'low_top_page_bid': idea.keyword_idea_metrics.low_top_of_page_bid_micros / 1_000_000,
                    'high_top_page_bid': idea.keyword_idea_metrics.high_top_of_page_bid_micros / 1_000_000,
                }
            
            return estimates
            
        except GoogleAdsException as ex:
            print(f"❌ Error estimating keyword bids: {ex}")
            return {}
    
    def analyze_behavioral_health_keywords(self) -> Dict:
        """
        Analyze behavioral health specific keywords and competition
        
        Returns:
            Analysis of behavioral health keyword landscape
        """
        # Key behavioral health keywords to analyze
        target_keywords = [
            "parental controls app",
            "screen time app",
            "family safety app",
            "monitor kids phone",
            "track child location",
            "limit screen time",
            "block inappropriate content",
            "family tracker",
            "parental control software",
            "digital wellbeing app",
            "bark app",
            "qustodio",
            "circle home plus",
            "norton family",
            "screen time parental control"
        ]
        
        # Get bid estimates
        estimates = self.estimate_keyword_bids(target_keywords)
        
        # Analyze competition levels
        analysis = {
            'high_competition': [],
            'medium_competition': [],
            'low_competition': [],
            'avg_cpc_by_category': {},
            'recommended_bids': {}
        }
        
        for keyword, data in estimates.items():
            if data['competition'] == 'HIGH':
                analysis['high_competition'].append({
                    'keyword': keyword,
                    'bid_range': (data['low_top_page_bid'], data['high_top_page_bid']),
                    'monthly_searches': data['avg_monthly_searches']
                })
            elif data['competition'] == 'MEDIUM':
                analysis['medium_competition'].append({
                    'keyword': keyword,
                    'bid_range': (data['low_top_page_bid'], data['high_top_page_bid']),
                    'monthly_searches': data['avg_monthly_searches']
                })
            else:
                analysis['low_competition'].append({
                    'keyword': keyword,
                    'bid_range': (data['low_top_page_bid'], data['high_top_page_bid']),
                    'monthly_searches': data['avg_monthly_searches']
                })
        
        # Calculate category averages
        if analysis['high_competition']:
            high_bids = [k['bid_range'][1] for k in analysis['high_competition']]
            analysis['avg_cpc_by_category']['high'] = sum(high_bids) / len(high_bids)
        
        if analysis['medium_competition']:
            med_bids = [k['bid_range'][1] for k in analysis['medium_competition']]
            analysis['avg_cpc_by_category']['medium'] = sum(med_bids) / len(med_bids)
        
        if analysis['low_competition']:
            low_bids = [k['bid_range'][1] for k in analysis['low_competition']]
            analysis['avg_cpc_by_category']['low'] = sum(low_bids) / len(low_bids)
        
        # Recommended bid ranges for GAELP
        analysis['recommended_bids'] = {
            'base_bid_range': (3.0, 15.0),
            'crisis_intent_multiplier': 2.0,
            'brand_terms_range': (10.0, 20.0),
            'generic_terms_range': (5.0, 12.0),
            'long_tail_range': (2.0, 7.0),
            'max_bid_cap': 25.0
        }
        
        return analysis


def main():
    """
    Main function to run Google Ads analysis
    """
    print("=" * 80)
    print("GOOGLE ADS INTEGRATION FOR GAELP")
    print("=" * 80)
    
    # Initialize integration
    ads = GoogleAdsIntegration()
    
    # Check if credentials are set
    if not ads.config['developer_token']:
        ads.setup_credentials()
        print("\n⚠️ Please set up credentials and run again.")
        return
    
    # Get keyword performance
    print("\n📊 Fetching keyword performance data...")
    keywords = ads.get_keyword_performance()
    
    if keywords:
        print(f"✅ Found {len(keywords)} keywords")
        
        # Show top CPCs
        print("\n💰 Top 10 Keywords by CPC:")
        sorted_keywords = sorted(keywords, key=lambda x: x['avg_cpc'], reverse=True)[:10]
        for kw in sorted_keywords:
            print(f"  {kw['keyword']}: ${kw['avg_cpc']:.2f} ({kw['clicks']} clicks)")
    
    # Get auction insights
    print("\n🎯 Fetching auction insights...")
    insights = ads.get_auction_insights()
    
    if insights:
        print(f"✅ Found auction insights for {len(insights)} campaigns")
    
    # Analyze behavioral health keywords
    print("\n🔍 Analyzing behavioral health keywords...")
    analysis = ads.analyze_behavioral_health_keywords()
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'keywords': keywords,
        'insights': insights,
        'analysis': analysis
    }
    
    with open('google_ads_data.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Results saved to google_ads_data.json")
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR GAELP TRAINING:")
    print("=" * 80)
    
    if analysis and 'recommended_bids' in analysis:
        recs = analysis['recommended_bids']
        print(f"\n📈 Bid Configuration:")
        print(f"  Base range: ${recs['base_bid_range'][0]:.2f} - ${recs['base_bid_range'][1]:.2f}")
        print(f"  Crisis multiplier: {recs['crisis_intent_multiplier']}x")
        print(f"  Brand terms: ${recs['brand_terms_range'][0]:.2f} - ${recs['brand_terms_range'][1]:.2f}")
        print(f"  Generic terms: ${recs['generic_terms_range'][0]:.2f} - ${recs['generic_terms_range'][1]:.2f}")
        print(f"  Long tail: ${recs['long_tail_range'][0]:.2f} - ${recs['long_tail_range'][1]:.2f}")
        print(f"  Max bid cap: ${recs['max_bid_cap']:.2f}")


if __name__ == "__main__":
    main()