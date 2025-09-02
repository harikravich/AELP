#!/usr/bin/env python3
"""
Analyze August GA4 Data - Categorize and Count
"""

import json
from pathlib import Path

# Product data from extraction (first week of August)
product_data = [
    ("Aura-NonCR-Parental-GW-direct-to-pay_d23", 177, 17698.23),
    ("Aura.com with PC Test | Family Annual", 90, 34560),
    ("Aura-NonCR-Parental-GW-pricetest_d23", 52, 5199.48),
    (".com with PC Annual", 48, 5760),
    ("Aura-NonCR-Parental-GW-direct-to-pay_monthly", 28, 308),
    (".com with PC", 18, 234),
]

# Campaign data from extraction (some key ones)
campaign_data = [
    ("av-intro", 1055, 987.2, 81685.34),
    ("GG_Aura_Search_Brand", 952, 727.55, 90514.08),
    ("GG_Aura_Search_NB_General", 666, 510.44, 41452.88),
    ("GG_Aura_Search_Brand_gtwy-pc", 63, 44.00, 3625.74),
    ("GG_Aura_Search_NB_gtwy-pc", 41, 23.16, 1468.39),
    ("GG_Aura_Search_Brand_gtwy-av", 40, 24.42, 2169.61),
    ("GG_Aura_Search_NB_gtwy-av", 61, 35.28, 2783.29),
    ("GG_Aura_PerformanceMax_NB_gtwy-pc", 10, 4.70, 436.25),
    ("GG_Aura_Search_Competitors_gtwy-pc", 1, 0.17, 17.16),
    ("life360_pc", 2, 0, 0),
    ("balance_parentingpressure_osaw", 1, 1, 99.99),
    ("balance_parentingpressure_ow", 1, 1, 99.99),
    ("balance_teentalk_ow", 1, 1, 99.99),
    ("thrive_mentalhealth", 8, 8, 23.92),
    ("thrive_productivity", 5, 5, 14.95),
]

def analyze_data():
    """Analyze and categorize the data"""
    
    print("\n" + "="*60)
    print("AUGUST 2025 GA4 DATA ANALYSIS (First Week)")
    print("="*60)
    
    # Analyze PC Products
    print("\n📊 PARENTAL CONTROLS (PC) PRODUCTS:")
    print("-" * 40)
    total_pc_sales = 0
    total_pc_revenue = 0
    for name, sales, revenue in product_data:
        if 'PC' in name or 'Parental' in name:
            total_pc_sales += sales
            total_pc_revenue += revenue
            print(f"  {name}: {sales} sales, ${revenue:,.2f}")
    
    print(f"\n  TOTAL PC: {total_pc_sales} sales in 7 days = ~{total_pc_sales/7:.1f} per day")
    print(f"  Revenue: ${total_pc_revenue:,.2f}")
    
    # Analyze other products
    print("\n📊 OTHER KEY PRODUCTS (First Week):")
    print("-" * 40)
    other_products = [
        ("D2C Aura 2022 - Up to 50%", 1018, 109302),
        ("D2C Aura 2022 LP FT", 969, 114034),
        ("Aura AV Exit Pop - Antivirus - $19.99", 489, 34225.11),
        ("AV Intro Pricing - Antivirus, No DBOO - $35.99/yr", 397, 27786.03),
    ]
    
    for name, sales, revenue in other_products:
        print(f"  {name}: {sales} sales, ${revenue:,.2f}")
    
    # Analyze campaigns
    print("\n📊 CAMPAIGN PERFORMANCE (By Product Category):")
    print("-" * 40)
    
    pc_campaigns = []
    av_campaigns = []
    balance_campaigns = []
    general_campaigns = []
    
    for name, sessions, conversions, revenue in campaign_data:
        name_lower = name.lower()
        if 'pc' in name_lower or 'parental' in name_lower:
            pc_campaigns.append((name, sessions, conversions, revenue))
        elif 'av' in name_lower or 'antivirus' in name_lower:
            av_campaigns.append((name, sessions, conversions, revenue))
        elif 'balance' in name_lower or 'thrive' in name_lower:
            balance_campaigns.append((name, sessions, conversions, revenue))
        else:
            general_campaigns.append((name, sessions, conversions, revenue))
    
    print("\nPC Campaigns:")
    for name, sessions, conversions, revenue in pc_campaigns:
        cvr = (conversions / sessions * 100) if sessions > 0 else 0
        print(f"  {name}: {sessions} sessions, {conversions:.0f} conv, {cvr:.2f}% CVR, ${revenue:,.2f}")
    
    print("\nAV Campaigns:")
    for name, sessions, conversions, revenue in av_campaigns:
        cvr = (conversions / sessions * 100) if sessions > 0 else 0
        print(f"  {name}: {sessions} sessions, {conversions:.0f} conv, {cvr:.2f}% CVR, ${revenue:,.2f}")
    
    print("\nBalance/Thrive Campaigns:")
    for name, sessions, conversions, revenue in balance_campaigns:
        cvr = (conversions / sessions * 100) if sessions > 0 else 0
        print(f"  {name}: {sessions} sessions, {conversions:.0f} conv, {cvr:.2f}% CVR, ${revenue:,.2f}")
    
    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 40)
    print(f"1. PC Sales: ~{total_pc_sales/7:.0f}/day (matches your estimate of 50-60/day)")
    print(f"2. Top Campaign: av-intro with {987:.0f} conversions")
    print(f"3. Balance/Thrive: Active with small-scale campaigns")
    print(f"4. PC campaigns have good CVR: 69.8% for brand search")
    print(f"5. Multiple test variants running (A/B tests)")
    
    # Save analysis
    output_dir = Path("ga4_extracted_data")
    output_dir.mkdir(exist_ok=True)
    
    analysis = {
        "period": "2025-08-01 to 2025-08-07",
        "pc_products": {
            "total_sales": total_pc_sales,
            "daily_average": total_pc_sales / 7,
            "revenue": total_pc_revenue
        },
        "campaign_categories": {
            "pc": len(pc_campaigns),
            "av": len(av_campaigns),
            "balance_thrive": len(balance_campaigns),
            "general": len(general_campaigns)
        },
        "insights": [
            f"PC sales averaging {total_pc_sales/7:.0f} per day",
            "AV campaigns driving most volume",
            "Balance/Thrive campaigns active but small scale",
            "Multiple A/B tests running"
        ]
    }
    
    with open(output_dir / "august_week1_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Analysis saved to {output_dir}/august_week1_analysis.json")


if __name__ == "__main__":
    analyze_data()