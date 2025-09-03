#!/usr/bin/env python3
"""
Pull data from Aura Google Ads account for GAELP training
Read-only access to fetch campaigns, performance, keywords, etc.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pickle

print("="*60)
print("AURA GOOGLE ADS DATA EXTRACTOR")
print("="*60)

# Configuration
CUSTOMER_ID = os.getenv("GOOGLE_ADS_CUSTOMER_ID", "").replace("-", "")
DEVELOPER_TOKEN = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
CLIENT_ID = os.getenv("GOOGLE_ADS_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_ADS_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("GOOGLE_ADS_REFRESH_TOKEN")

# Check if we have credentials
if not all([CUSTOMER_ID, DEVELOPER_TOKEN, CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
    print("\n❌ Missing Google Ads credentials!")
    print("Please set the following environment variables:")
    print("  GOOGLE_ADS_CUSTOMER_ID")
    print("  GOOGLE_ADS_DEVELOPER_TOKEN")
    print("  GOOGLE_ADS_CLIENT_ID")
    print("  GOOGLE_ADS_CLIENT_SECRET")
    print("  GOOGLE_ADS_REFRESH_TOKEN")
    print("\nRun setup_google_ads_mcp.sh to configure these.")
    exit(1)

print(f"\n📊 Connecting to Google Ads account: {CUSTOMER_ID}")

# Initialize the Google Ads client
try:
    # Create credentials dictionary
    credentials = {
        "developer_token": DEVELOPER_TOKEN,
        "refresh_token": REFRESH_TOKEN,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "login_customer_id": CUSTOMER_ID,
        "use_proto_plus": True
    }
    
    # Initialize client
    client = GoogleAdsClient.load_from_dict(credentials)
    print("✅ Successfully connected to Google Ads API")
    
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure you have read access to the Aura account")
    print("2. Check that your developer token is approved")
    print("3. Verify OAuth credentials are correct")
    exit(1)

# Create data directory
data_dir = Path("/home/hariravichandran/AELP/data/aura_google_ads")
data_dir.mkdir(exist_ok=True, parents=True)

print(f"\n💾 Data will be saved to: {data_dir}")

def fetch_account_info(client, customer_id):
    """Fetch basic account information"""
    print("\n🔍 Fetching account information...")
    
    ga_service = client.get_service("GoogleAdsService")
    
    query = """
        SELECT
            customer.id,
            customer.descriptive_name,
            customer.currency_code,
            customer.time_zone
        FROM customer
        LIMIT 1
    """
    
    try:
        response = ga_service.search_stream(customer_id=customer_id, query=query)
        for batch in response:
            for row in batch.results:
                return {
                    "customer_id": row.customer.id,
                    "account_name": row.customer.descriptive_name,
                    "currency": row.customer.currency_code,
                    "timezone": row.customer.time_zone
                }
    except GoogleAdsException as ex:
        print(f"❌ Error fetching account info: {ex}")
        return None

def fetch_campaigns(client, customer_id):
    """Fetch all campaigns with performance data"""
    print("\n📈 Fetching campaigns...")
    
    ga_service = client.get_service("GoogleAdsService")
    
    # Last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    query = f"""
        SELECT
            campaign.id,
            campaign.name,
            campaign.status,
            campaign.advertising_channel_type,
            campaign.bidding_strategy_type,
            campaign_budget.amount_micros,
            metrics.impressions,
            metrics.clicks,
            metrics.cost_micros,
            metrics.conversions,
            metrics.conversions_value,
            metrics.average_cpc,
            metrics.average_cpm,
            metrics.ctr,
            metrics.conversion_rate
        FROM campaign
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY metrics.impressions DESC
    """
    
    campaigns = []
    try:
        response = ga_service.search_stream(customer_id=customer_id, query=query)
        for batch in response:
            for row in batch.results:
                campaigns.append({
                    "campaign_id": row.campaign.id,
                    "campaign_name": row.campaign.name,
                    "status": row.campaign.status.name,
                    "channel": row.campaign.advertising_channel_type.name,
                    "bidding_strategy": row.campaign.bidding_strategy_type.name,
                    "budget": row.campaign_budget.amount_micros / 1_000_000 if row.campaign_budget else 0,
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                    "cost": row.metrics.cost_micros / 1_000_000,
                    "conversions": row.metrics.conversions,
                    "conversion_value": row.metrics.conversions_value,
                    "avg_cpc": row.metrics.average_cpc / 1_000_000 if row.metrics.average_cpc else 0,
                    "avg_cpm": row.metrics.average_cpm / 1_000_000 if row.metrics.average_cpm else 0,
                    "ctr": row.metrics.ctr,
                    "cvr": row.metrics.conversion_rate
                })
                
        print(f"  Found {len(campaigns)} campaigns")
        return campaigns
        
    except GoogleAdsException as ex:
        print(f"❌ Error fetching campaigns: {ex}")
        return []

def fetch_keywords(client, customer_id, limit=1000):
    """Fetch keyword performance data"""
    print("\n🔑 Fetching keyword data...")
    
    ga_service = client.get_service("GoogleAdsService")
    
    # Last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    query = f"""
        SELECT
            ad_group.id,
            ad_group.name,
            ad_group_criterion.keyword.text,
            ad_group_criterion.keyword.match_type,
            ad_group_criterion.status,
            metrics.impressions,
            metrics.clicks,
            metrics.cost_micros,
            metrics.conversions,
            metrics.average_cpc,
            metrics.ctr,
            metrics.conversion_rate
        FROM keyword_view
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
            AND ad_group_criterion.status != 'REMOVED'
        ORDER BY metrics.impressions DESC
        LIMIT {limit}
    """
    
    keywords = []
    try:
        response = ga_service.search_stream(customer_id=customer_id, query=query)
        for batch in response:
            for row in batch.results:
                keywords.append({
                    "ad_group_id": row.ad_group.id,
                    "ad_group_name": row.ad_group.name,
                    "keyword": row.ad_group_criterion.keyword.text,
                    "match_type": row.ad_group_criterion.keyword.match_type.name,
                    "status": row.ad_group_criterion.status.name,
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                    "cost": row.metrics.cost_micros / 1_000_000,
                    "conversions": row.metrics.conversions,
                    "avg_cpc": row.metrics.average_cpc / 1_000_000 if row.metrics.average_cpc else 0,
                    "ctr": row.metrics.ctr,
                    "cvr": row.metrics.conversion_rate
                })
                
        print(f"  Found {len(keywords)} keywords")
        return keywords
        
    except GoogleAdsException as ex:
        print(f"❌ Error fetching keywords: {ex}")
        return []

def fetch_ad_performance(client, customer_id):
    """Fetch ad creative performance"""
    print("\n📝 Fetching ad creative performance...")
    
    ga_service = client.get_service("GoogleAdsService")
    
    # Last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    query = f"""
        SELECT
            ad_group.id,
            ad_group_ad.ad.id,
            ad_group_ad.ad.type,
            ad_group_ad.ad.final_urls,
            ad_group_ad.ad.responsive_search_ad.headlines,
            ad_group_ad.ad.responsive_search_ad.descriptions,
            metrics.impressions,
            metrics.clicks,
            metrics.cost_micros,
            metrics.conversions,
            metrics.ctr,
            metrics.conversion_rate
        FROM ad_group_ad
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
            AND ad_group_ad.status = 'ENABLED'
        ORDER BY metrics.impressions DESC
        LIMIT 500
    """
    
    ads = []
    try:
        response = ga_service.search_stream(customer_id=customer_id, query=query)
        for batch in response:
            for row in batch.results:
                # Extract headlines and descriptions
                headlines = []
                descriptions = []
                
                if hasattr(row.ad_group_ad.ad, 'responsive_search_ad'):
                    rsa = row.ad_group_ad.ad.responsive_search_ad
                    headlines = [h.text for h in rsa.headlines]
                    descriptions = [d.text for d in rsa.descriptions]
                
                ads.append({
                    "ad_group_id": row.ad_group.id,
                    "ad_id": row.ad_group_ad.ad.id,
                    "ad_type": row.ad_group_ad.ad.type.name,
                    "final_urls": list(row.ad_group_ad.ad.final_urls),
                    "headlines": headlines,
                    "descriptions": descriptions,
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                    "cost": row.metrics.cost_micros / 1_000_000,
                    "conversions": row.metrics.conversions,
                    "ctr": row.metrics.ctr,
                    "cvr": row.metrics.conversion_rate
                })
                
        print(f"  Found {len(ads)} ads")
        return ads
        
    except GoogleAdsException as ex:
        print(f"❌ Error fetching ads: {ex}")
        return []

def fetch_audience_data(client, customer_id):
    """Fetch audience targeting and performance"""
    print("\n👥 Fetching audience data...")
    
    ga_service = client.get_service("GoogleAdsService")
    
    # Last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    query = f"""
        SELECT
            ad_group.id,
            ad_group.name,
            ad_group_audience_view.criterion.user_list.name,
            metrics.impressions,
            metrics.clicks,
            metrics.conversions,
            metrics.ctr,
            metrics.conversion_rate
        FROM ad_group_audience_view
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY metrics.impressions DESC
        LIMIT 100
    """
    
    audiences = []
    try:
        response = ga_service.search_stream(customer_id=customer_id, query=query)
        for batch in response:
            for row in batch.results:
                audiences.append({
                    "ad_group_id": row.ad_group.id,
                    "ad_group_name": row.ad_group.name,
                    "audience_name": row.ad_group_audience_view.criterion.user_list.name if hasattr(row.ad_group_audience_view.criterion, 'user_list') else "Unknown",
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                    "conversions": row.metrics.conversions,
                    "ctr": row.metrics.ctr,
                    "cvr": row.metrics.conversion_rate
                })
                
        print(f"  Found {len(audiences)} audience segments")
        return audiences
        
    except GoogleAdsException as ex:
        print(f"❌ Error fetching audiences: {ex}")
        return []

def fetch_device_performance(client, customer_id):
    """Fetch performance by device"""
    print("\n📱 Fetching device performance...")
    
    ga_service = client.get_service("GoogleAdsService")
    
    # Last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    query = f"""
        SELECT
            segments.device,
            metrics.impressions,
            metrics.clicks,
            metrics.cost_micros,
            metrics.conversions,
            metrics.ctr,
            metrics.conversion_rate
        FROM campaign
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    """
    
    devices = {}
    try:
        response = ga_service.search_stream(customer_id=customer_id, query=query)
        for batch in response:
            for row in batch.results:
                device = row.segments.device.name
                if device not in devices:
                    devices[device] = {
                        "device": device,
                        "impressions": 0,
                        "clicks": 0,
                        "cost": 0,
                        "conversions": 0
                    }
                devices[device]["impressions"] += row.metrics.impressions
                devices[device]["clicks"] += row.metrics.clicks
                devices[device]["cost"] += row.metrics.cost_micros / 1_000_000
                devices[device]["conversions"] += row.metrics.conversions
        
        # Calculate CTR and CVR
        device_list = []
        for device_data in devices.values():
            device_data["ctr"] = device_data["clicks"] / device_data["impressions"] if device_data["impressions"] > 0 else 0
            device_data["cvr"] = device_data["conversions"] / device_data["clicks"] if device_data["clicks"] > 0 else 0
            device_list.append(device_data)
            
        print(f"  Found data for {len(device_list)} device types")
        return device_list
        
    except GoogleAdsException as ex:
        print(f"❌ Error fetching device data: {ex}")
        return []

def create_training_dataset(all_data):
    """Create training dataset from fetched data"""
    print("\n🧠 Creating training dataset...")
    
    training_data = {
        "timestamp": datetime.now().isoformat(),
        "account_info": all_data["account_info"],
        "performance_summary": {},
        "campaign_patterns": [],
        "keyword_insights": [],
        "creative_patterns": [],
        "audience_segments": [],
        "device_distribution": []
    }
    
    # Calculate performance summary
    if all_data["campaigns"]:
        campaigns_df = pd.DataFrame(all_data["campaigns"])
        training_data["performance_summary"] = {
            "total_impressions": int(campaigns_df["impressions"].sum()),
            "total_clicks": int(campaigns_df["clicks"].sum()),
            "total_cost": float(campaigns_df["cost"].sum()),
            "total_conversions": float(campaigns_df["conversions"].sum()),
            "avg_ctr": float(campaigns_df["ctr"].mean()),
            "avg_cvr": float(campaigns_df["cvr"].mean()),
            "avg_cpc": float(campaigns_df["avg_cpc"].mean()),
            "campaigns_count": len(campaigns_df)
        }
        
        # Extract campaign patterns
        for channel in campaigns_df["channel"].unique():
            channel_data = campaigns_df[campaigns_df["channel"] == channel]
            training_data["campaign_patterns"].append({
                "channel": channel,
                "count": len(channel_data),
                "avg_ctr": float(channel_data["ctr"].mean()),
                "avg_cvr": float(channel_data["cvr"].mean()),
                "avg_cpc": float(channel_data["avg_cpc"].mean()),
                "total_spend": float(channel_data["cost"].sum())
            })
    
    # Process keywords
    if all_data["keywords"]:
        keywords_df = pd.DataFrame(all_data["keywords"])
        
        # Top performing keywords
        top_keywords = keywords_df.nlargest(20, "conversions")
        for _, kw in top_keywords.iterrows():
            training_data["keyword_insights"].append({
                "keyword": kw["keyword"],
                "match_type": kw["match_type"],
                "ctr": float(kw["ctr"]),
                "cvr": float(kw["cvr"]),
                "avg_cpc": float(kw["avg_cpc"]),
                "conversions": float(kw["conversions"])
            })
    
    # Process ad creatives
    if all_data["ads"]:
        ads_df = pd.DataFrame(all_data["ads"])
        
        # Top performing ads
        top_ads = ads_df.nlargest(10, "conversions")
        for _, ad in top_ads.iterrows():
            training_data["creative_patterns"].append({
                "ad_type": ad["ad_type"],
                "headlines": ad["headlines"][:3] if ad["headlines"] else [],
                "ctr": float(ad["ctr"]),
                "cvr": float(ad["cvr"]),
                "conversions": float(ad["conversions"])
            })
    
    # Process audiences
    if all_data["audiences"]:
        for audience in all_data["audiences"][:20]:
            training_data["audience_segments"].append({
                "name": audience["audience_name"],
                "ctr": float(audience["ctr"]),
                "cvr": float(audience["cvr"]),
                "conversions": float(audience["conversions"])
            })
    
    # Process device data
    if all_data["devices"]:
        training_data["device_distribution"] = all_data["devices"]
    
    return training_data

# Main execution
if __name__ == "__main__":
    print("\n🚀 Starting data extraction...")
    
    # Fetch all data
    all_data = {
        "account_info": fetch_account_info(client, CUSTOMER_ID),
        "campaigns": fetch_campaigns(client, CUSTOMER_ID),
        "keywords": fetch_keywords(client, CUSTOMER_ID),
        "ads": fetch_ad_performance(client, CUSTOMER_ID),
        "audiences": fetch_audience_data(client, CUSTOMER_ID),
        "devices": fetch_device_performance(client, CUSTOMER_ID)
    }
    
    # Save raw data
    print("\n💾 Saving raw data...")
    
    # Save as JSON
    json_file = data_dir / "aura_google_ads_raw.json"
    with open(json_file, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"  Saved to {json_file}")
    
    # Save campaigns as CSV for easy viewing
    if all_data["campaigns"]:
        campaigns_df = pd.DataFrame(all_data["campaigns"])
        csv_file = data_dir / "aura_campaigns.csv"
        campaigns_df.to_csv(csv_file, index=False)
        print(f"  Saved campaigns to {csv_file}")
    
    # Save keywords as CSV
    if all_data["keywords"]:
        keywords_df = pd.DataFrame(all_data["keywords"])
        csv_file = data_dir / "aura_keywords.csv"
        keywords_df.to_csv(csv_file, index=False)
        print(f"  Saved keywords to {csv_file}")
    
    # Create training dataset
    training_data = create_training_dataset(all_data)
    
    # Save training data
    training_file = data_dir / "aura_training_data.json"
    with open(training_file, "w") as f:
        json.dump(training_data, f, indent=2)
    print(f"\n✅ Training dataset saved to {training_file}")
    
    # Save as pickle for model training
    pickle_file = data_dir / "aura_training_data.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(training_data, f)
    print(f"✅ Pickle file saved to {pickle_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA EXTRACTION COMPLETE")
    print("="*60)
    
    if all_data["account_info"]:
        print(f"\nAccount: {all_data['account_info']['account_name']}")
        print(f"Customer ID: {all_data['account_info']['customer_id']}")
    
    if training_data["performance_summary"]:
        summary = training_data["performance_summary"]
        print(f"\nPerformance Summary (Last 30 days):")
        print(f"  Campaigns: {summary['campaigns_count']}")
        print(f"  Impressions: {summary['total_impressions']:,}")
        print(f"  Clicks: {summary['total_clicks']:,}")
        print(f"  Cost: ${summary['total_cost']:,.2f}")
        print(f"  Conversions: {summary['total_conversions']:,.0f}")
        print(f"  Avg CTR: {summary['avg_ctr']*100:.2f}%")
        print(f"  Avg CVR: {summary['avg_cvr']*100:.2f}%")
        print(f"  Avg CPC: ${summary['avg_cpc']:.2f}")
    
    print("\n📊 Data ready for GAELP training!")
    print(f"   Raw data: {json_file}")
    print(f"   Training data: {training_file}")
    print(f"   Pickle file: {pickle_file}")
    
    print("\n🎯 Next steps:")
    print("1. Review the extracted data")
    print("2. Run: python3 integrate_aura_data_to_gaelp.py")
    print("3. Retrain the model with real Aura patterns")