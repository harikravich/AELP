#!/usr/bin/env python3
"""
Visualization script for GAELP multi-touch attribution analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

def load_results():
    """Load GAELP results"""
    with open('/home/hariravichandran/AELP/gaelp_results.json', 'r') as f:
        results = json.load(f)
    
    touchpoints_df = pd.read_csv('/home/hariravichandran/AELP/gaelp_touchpoints.csv')
    
    return results, touchpoints_df

def plot_attribution_comparison(attribution_data):
    """Plot comparison of different attribution models"""
    channels = list(attribution_data.keys())
    first_touch = [attribution_data[ch]['first_touch'] for ch in channels]
    last_touch = [attribution_data[ch]['last_touch'] for ch in channels]
    linear = [attribution_data[ch]['linear'] for ch in channels]
    
    x = np.arange(len(channels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, first_touch, width, label='First Touch', color='#2E86AB')
    ax.bar(x, last_touch, width, label='Last Touch', color='#A23B72')
    ax.bar(x + width, linear, width, label='Linear', color='#F18F01')
    
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Attribution Credit', fontsize=12)
    ax.set_title('Multi-Touch Attribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/hariravichandran/AELP/attribution_comparison.png', dpi=150)
    plt.show()

def plot_journey_paths(top_paths):
    """Plot top conversion paths"""
    paths = [p[0] for p in top_paths[:10]]
    counts = [p[1] for p in top_paths[:10]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(paths)), counts, color='#4A6FA5')
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels(paths)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('Top 10 Customer Journey Paths', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, count in enumerate(counts):
        ax.text(count + 0.5, i, str(count), va='center')
    
    plt.tight_layout()
    plt.savefig('/home/hariravichandran/AELP/journey_paths.png', dpi=150)
    plt.show()

def plot_performance_metrics(touchpoints_df):
    """Plot performance metrics over time"""
    # Group by day
    touchpoints_df['day'] = pd.to_numeric(touchpoints_df['day'])
    daily_metrics = touchpoints_df.groupby('day').agg({
        'cost': 'sum',
        'conversion_probability': 'mean',
        'ltv': 'mean',
        'journey_length': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Daily spend
    axes[0, 0].plot(daily_metrics['day'], daily_metrics['cost'], marker='o', color='#E63946')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Daily Spend ($)')
    axes[0, 0].set_title('Daily Ad Spend')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Conversion probability
    axes[0, 1].plot(daily_metrics['day'], daily_metrics['conversion_probability'], 
                    marker='s', color='#06A77D')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Avg Conversion Probability')
    axes[0, 1].set_title('Conversion Probability Trend')
    axes[0, 1].grid(True, alpha=0.3)
    
    # LTV trend
    axes[1, 0].plot(daily_metrics['day'], daily_metrics['ltv'], 
                    marker='^', color='#F77F00')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Average LTV ($)')
    axes[1, 0].set_title('Customer Lifetime Value Trend')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Journey length
    axes[1, 1].plot(daily_metrics['day'], daily_metrics['journey_length'], 
                    marker='d', color='#5C415D')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Avg Journey Length')
    axes[1, 1].set_title('Customer Journey Length')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('GAELP Performance Metrics Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/hariravichandran/AELP/performance_metrics.png', dpi=150)
    plt.show()

def plot_channel_efficiency(touchpoints_df):
    """Plot channel efficiency metrics"""
    channel_metrics = touchpoints_df.groupby('channel').agg({
        'cost': 'sum',
        'conversion_probability': 'mean',
        'journey_length': 'count'
    }).reset_index()
    
    channel_metrics['cost_per_touch'] = channel_metrics['cost'] / channel_metrics['journey_length']
    channel_metrics['efficiency_score'] = (channel_metrics['conversion_probability'] / 
                                          (channel_metrics['cost_per_touch'] + 0.01))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cost per touchpoint
    axes[0].bar(channel_metrics['channel'], channel_metrics['cost_per_touch'], 
                color='#457B9D')
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Cost per Touchpoint ($)')
    axes[0].set_title('Channel Cost Efficiency')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Efficiency score
    axes[1].bar(channel_metrics['channel'], channel_metrics['efficiency_score'], 
                color='#1D3557')
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Efficiency Score')
    axes[1].set_title('Channel Performance Score')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Channel Efficiency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/hariravichandran/AELP/channel_efficiency.png', dpi=150)
    plt.show()

def generate_attribution_report(results):
    """Generate text report of attribution analysis"""
    report = []
    report.append("=" * 60)
    report.append("GAELP Multi-Touch Attribution Analysis Report")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().isoformat()}\n")
    
    # Evaluation metrics
    eval_metrics = results['evaluation']
    report.append("Model Performance:")
    report.append(f"  • Average Reward: {eval_metrics['avg_reward']:.2f}")
    report.append(f"  • Conversion Rate: {eval_metrics['conversion_rate']:.2%}")
    report.append(f"  • Customer Acquisition Cost: ${eval_metrics['avg_cac']:.2f}")
    report.append(f"  • Return on Ad Spend: {eval_metrics['roas']:.2f}x")
    report.append(f"  • Average Journey Length: {eval_metrics['avg_journey_length']:.1f} touches\n")
    
    # Attribution comparison
    attribution = results['attribution']['channel_attribution']
    report.append("Attribution Model Comparison:")
    report.append("Channel      | First Touch | Last Touch | Linear")
    report.append("-" * 50)
    
    for channel, data in attribution.items():
        report.append(f"{channel:12} | {data['first_touch']:11.0f} | "
                     f"{data['last_touch']:10.0f} | {data['linear']:6.1f}")
    
    report.append("\nTop Customer Journey Paths:")
    for i, (path, count) in enumerate(results['attribution']['top_conversion_paths'][:5], 1):
        report.append(f"  {i}. {path} ({count} occurrences)")
    
    # Key insights
    report.append("\nKey Insights:")
    
    # Find most effective first-touch channel
    first_touch_winner = max(attribution.items(), 
                            key=lambda x: x[1]['first_touch'])[0]
    report.append(f"  • Best first-touch channel: {first_touch_winner}")
    
    # Find most effective last-touch channel
    last_touch_winner = max(attribution.items(), 
                           key=lambda x: x[1]['last_touch'])[0]
    report.append(f"  • Best last-touch channel: {last_touch_winner}")
    
    # Find most balanced channel (linear)
    linear_winner = max(attribution.items(), 
                       key=lambda x: x[1]['linear'])[0]
    report.append(f"  • Most consistent performer: {linear_winner}")
    
    report.append("\nRecommendations:")
    report.append("  1. Increase budget allocation to high-performing channels")
    report.append("  2. Optimize journey paths with highest conversion rates")
    report.append("  3. Reduce touches for users showing high intent")
    report.append("  4. Implement cross-channel coordination for better synergy")
    
    report_text = "\n".join(report)
    
    # Save report
    with open('/home/hariravichandran/AELP/attribution_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text

def main():
    """Main visualization pipeline"""
    print("📊 GAELP Attribution Visualization")
    print("=" * 60)
    
    # Load results
    results, touchpoints_df = load_results()
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    print("\n📈 Generating visualizations...")
    
    # 1. Attribution comparison
    print("  • Attribution model comparison...")
    plot_attribution_comparison(results['attribution']['channel_attribution'])
    
    # 2. Journey paths
    print("  • Top journey paths...")
    plot_journey_paths(results['attribution']['top_conversion_paths'])
    
    # 3. Performance metrics
    print("  • Performance metrics over time...")
    plot_performance_metrics(touchpoints_df)
    
    # 4. Channel efficiency
    print("  • Channel efficiency analysis...")
    plot_channel_efficiency(touchpoints_df)
    
    # 5. Generate report
    print("\n📝 Generating attribution report...")
    report = generate_attribution_report(results)
    print(report)
    
    print("\n✅ Visualization complete!")
    print("   • Attribution comparison: attribution_comparison.png")
    print("   • Journey paths: journey_paths.png")
    print("   • Performance metrics: performance_metrics.png")
    print("   • Channel efficiency: channel_efficiency.png")
    print("   • Report: attribution_report.txt")

if __name__ == "__main__":
    main()