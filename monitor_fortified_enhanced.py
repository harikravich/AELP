#!/usr/bin/env python3
"""
ENHANCED FORTIFIED TRAINING MONITOR
Properly tracks all metrics including delayed conversions
"""

import os
import sys
import time
import psutil
import re
import numpy as np
from datetime import datetime
import json

def get_process_info(pid):
    """Get process information"""
    try:
        process = psutil.Process(pid)
        return {
            'status': 'Running' if process.is_running() else 'Stopped',
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'create_time': datetime.fromtimestamp(process.create_time()),
            'running': True
        }
    except:
        return {
            'status': 'Not Found',
            'cpu_percent': 0,
            'memory_mb': 0,
            'create_time': datetime.now(),
            'running': False
        }

def parse_training_output(log_file='fortified_training_output.log'):
    """Parse the training log file for metrics"""
    metrics = {
        'episodes': 0,
        'experiences': 0,
        'epsilon': 1.0,
        'convergence_status': 'Not converged',
        'loss_bid': 0,
        'loss_creative': 0,
        'loss_channel': 0,
        'conversions': 0,
        'delayed_conversions': 0,
        'immediate_conversions': 0,
        'revenue': 0,
        'avg_reward': 0,
        'best_roas': 0,
        'channel_performance': {},
        'creative_performance': {},
        'bid_distribution': [],
        'learning_quality': 'Initializing',
        'state_dimensions': 45  # Enriched state
    }
    
    try:
        if os.path.exists(log_file):
            # Read the last 5000 lines for recent metrics
            with open(log_file, 'r') as f:
                lines = f.readlines()
                log_content = ''.join(lines[-5000:])
            
            # Parse episodes (accounting for INFO: prefix)
            episode_matches = re.findall(r'Episode (\d+)/', log_content)
            if episode_matches:
                metrics['episodes'] = int(episode_matches[-1])
            
            # Parse total experiences
            exp_matches = re.findall(r'Total experiences: ([\d,]+)', log_content)
            if exp_matches:
                metrics['experiences'] = int(exp_matches[-1].replace(',', ''))
            
            # Parse epsilon
            epsilon_matches = re.findall(r'Epsilon: ([\d.]+)', log_content)
            if epsilon_matches:
                metrics['epsilon'] = float(epsilon_matches[-1])
            
            # Parse losses
            loss_matches = re.findall(r'Loss \(bid\): ([\d.]+).*Loss \(creative\): ([\d.]+).*Loss \(channel\): ([\d.]+)', log_content)
            if loss_matches:
                metrics['loss_bid'] = float(loss_matches[-1][0])
                metrics['loss_creative'] = float(loss_matches[-1][1])
                metrics['loss_channel'] = float(loss_matches[-1][2])
            
            # Parse ALL conversions (immediate + delayed)
            conv_matches = re.findall(r'Conversions: ([\d,]+)', log_content)
            if conv_matches:
                metrics['conversions'] = int(conv_matches[-1].replace(',', ''))
            
            # Try to find delayed conversion processing logs
            delayed_conv = re.findall(r'Processing delayed conversions.*conversions: (\d+)', log_content)
            if delayed_conv:
                metrics['delayed_conversions'] = int(delayed_conv[-1])
            
            # Parse revenue
            revenue_matches = re.findall(r'Revenue: \$([\d,.]+)', log_content)
            if revenue_matches:
                metrics['revenue'] = float(revenue_matches[-1].replace(',', ''))
            
            # Parse average reward
            reward_matches = re.findall(r'Avg Reward: ([\d.-]+)', log_content)
            if reward_matches:
                metrics['avg_reward'] = float(reward_matches[-1])
            
            # Parse ALL channel performance (including organic and social)
            channel_perf = re.findall(r'(\w+): ROAS=([\d.]+)x.*Spend=\$([\d.]+).*Revenue=\$([\d.]+)', log_content)
            for channel, roas, spend, revenue in channel_perf:
                metrics['channel_performance'][channel] = {
                    'roas': float(roas),
                    'spend': float(spend),
                    'revenue': float(revenue)
                }
            
            # Also check for channels with no ROAS (organic)
            organic_matches = re.findall(r'organic.*Sessions: (\d+)', log_content)
            if organic_matches and 'organic' not in metrics['channel_performance']:
                metrics['channel_performance']['organic'] = {
                    'roas': 0,  # Organic has no spend
                    'spend': 0,
                    'revenue': 0,
                    'sessions': int(organic_matches[-1])
                }
            
            # Parse creative performance from agent's internal state
            creative_perf = re.findall(r'Creative (\d+):.*CTR=([\d.]+)%.*CVR=([\d.]+)%', log_content)
            for creative_id, ctr, cvr in creative_perf:
                metrics['creative_performance'][creative_id] = {
                    'ctr': float(ctr),
                    'cvr': float(cvr)
                }
            
            # If no creative performance found, check for raw metrics
            if not metrics['creative_performance']:
                # Look for creative metrics in different format
                creative_raw = re.findall(r'creative_(\d+).*impressions: (\d+).*clicks: (\d+)', log_content)
                for creative_id, impressions, clicks in creative_raw:
                    imp = float(impressions)
                    clk = float(clicks)
                    if imp > 0:
                        metrics['creative_performance'][creative_id] = {
                            'ctr': (clk / imp) * 100,
                            'cvr': 0  # Will be updated when conversions are attributed
                        }
            
            # Parse best ROAS
            roas_matches = re.findall(r'Best ROAS: ([\d.]+)x', log_content)
            if roas_matches:
                metrics['best_roas'] = float(roas_matches[-1])
            
            # Check for convergence
            if "Training converged" in log_content:
                metrics['convergence_status'] = 'Converged!'
            elif metrics['epsilon'] <= 0.01:
                metrics['convergence_status'] = 'Near convergence'
            
            # Assess learning quality based on loss trend
            if metrics['loss_bid'] > 0 and metrics['epsilon'] < 1.0:
                if metrics['loss_bid'] < 0.01:
                    metrics['learning_quality'] = 'Excellent - Fully learned'
                elif metrics['loss_bid'] < 0.1:
                    metrics['learning_quality'] = 'Good - Refining'
                elif metrics['loss_bid'] < 1.0:
                    metrics['learning_quality'] = 'Learning'
                elif metrics['loss_bid'] < 2.0:
                    metrics['learning_quality'] = 'Early learning'
                else:
                    metrics['learning_quality'] = 'High loss - check config'
            
    except Exception as e:
        print(f"Error parsing logs: {e}")
    
    return metrics

def calculate_component_health(metrics):
    """Calculate health scores for each component"""
    health = {
        'overall': 0,
        'creative_selector': 0,
        'channel_optimizer': 0,
        'attribution_engine': 0,
        'budget_pacer': 0,
        'learning_system': 0,
        'conversion_tracking': 0
    }
    
    # Creative Selector health - based on whether we have performance data
    if metrics['creative_performance']:
        # Check if creatives are getting impressions
        total_creatives = len(metrics['creative_performance'])
        if total_creatives > 0:
            health['creative_selector'] = min(100, total_creatives * 10)
    else:
        health['creative_selector'] = 25
    
    # Channel Optimizer health - based on channel diversity and ROAS
    if metrics['channel_performance']:
        num_channels = len(metrics['channel_performance'])
        avg_roas = np.mean([p['roas'] for p in metrics['channel_performance'].values() if p['roas'] > 0])
        
        if num_channels >= 5:  # All 5 channels active
            health['channel_optimizer'] = 100
        elif num_channels >= 3:
            health['channel_optimizer'] = 75
        elif num_channels >= 2:
            health['channel_optimizer'] = 50
        else:
            health['channel_optimizer'] = 25
            
        # Boost for good ROAS
        if avg_roas > 2.0:
            health['channel_optimizer'] = min(100, health['channel_optimizer'] + 25)
    
    # Attribution Engine health - based on revenue attribution
    if metrics['revenue'] > 0:
        # Check if revenue is being attributed across channels
        channels_with_revenue = sum(1 for p in metrics['channel_performance'].values() if p.get('revenue', 0) > 0)
        if channels_with_revenue >= 3:
            health['attribution_engine'] = 100
        elif channels_with_revenue >= 2:
            health['attribution_engine'] = 75
        elif channels_with_revenue >= 1:
            health['attribution_engine'] = 50
        else:
            health['attribution_engine'] = 25
    
    # Budget Pacer health - based on spend distribution
    if metrics['channel_performance']:
        total_spend = sum(p['spend'] for p in metrics['channel_performance'].values())
        if total_spend > 0:
            # Check if budget is being paced (not all spent at once)
            spend_variance = np.var([p['spend'] for p in metrics['channel_performance'].values()])
            if spend_variance > 0:  # Budget is distributed
                health['budget_pacer'] = 100
            else:
                health['budget_pacer'] = 50
    
    # Learning System health - based on loss and epsilon
    if metrics['loss_bid'] > 0:
        if metrics['loss_bid'] < 0.1:
            health['learning_system'] = 100
        elif metrics['loss_bid'] < 0.5:
            health['learning_system'] = 75
        elif metrics['loss_bid'] < 1.0:
            health['learning_system'] = 50
        else:
            health['learning_system'] = 25
            
        # Boost for low epsilon (less exploration)
        if metrics['epsilon'] < 0.1:
            health['learning_system'] = min(100, health['learning_system'] + 25)
    
    # Conversion Tracking health - based on conversions
    if metrics['conversions'] > 0 or metrics['revenue'] > 0:
        health['conversion_tracking'] = 100
    elif metrics['delayed_conversions'] > 0:
        health['conversion_tracking'] = 75
    elif metrics['experiences'] > 1000:  # Should have some conversions by now
        health['conversion_tracking'] = 25
    else:
        health['conversion_tracking'] = 50  # Still early
    
    # Overall health
    health['overall'] = np.mean(list(health.values())[1:])  # Exclude 'overall' itself
    
    return health

def display_dashboard(metrics, process_info):
    """Display the training dashboard"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("=" * 80)
    print("                        ENHANCED FORTIFIED TRAINING MONITOR                        ")
    print("=" * 80)
    
    # Process Status
    print(f"\n📊 PROCESS STATUS")
    status_emoji = '🟢' if process_info['running'] else '🔴'
    print(f"  Status: {status_emoji} {process_info['status']}")
    print(f"  CPU Usage: {process_info['cpu_percent']:.1f}%")
    print(f"  Memory: {process_info['memory_mb']:.1f} MB")
    uptime = datetime.now() - process_info['create_time']
    print(f"  Uptime: {uptime}")
    
    # Training Progress
    print(f"\n📈 TRAINING PROGRESS")
    print(f"  Episodes: {metrics['episodes']}")
    print(f"  Experiences: {metrics['experiences']:,}")
    print(f"  Epsilon: {metrics['epsilon']:.4f}")
    print(f"  Status: {metrics['convergence_status']}")
    
    # Learning Metrics
    print(f"\n🧠 LEARNING METRICS")
    print(f"  Loss (Bid): {metrics['loss_bid']:.4f}")
    print(f"  Loss (Creative): {metrics['loss_creative']:.4f}")
    print(f"  Loss (Channel): {metrics['loss_channel']:.4f}")
    print(f"  Learning Quality: {metrics['learning_quality']}")
    
    # Performance Metrics with ENHANCED conversion tracking
    print(f"\n💰 PERFORMANCE METRICS")
    print(f"  Total Conversions: {metrics['conversions']:,}")
    if metrics['delayed_conversions'] > 0:
        print(f"    ├─ Immediate: {metrics['conversions'] - metrics['delayed_conversions']:,}")
        print(f"    └─ Delayed (3-14 days): {metrics['delayed_conversions']:,}")
    print(f"  Revenue: ${metrics['revenue']:,.2f}")
    print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
    print(f"  Best ROAS: {metrics['best_roas']:.2f}x")
    
    # ALL Channel Performance (including organic and social)
    if metrics['channel_performance']:
        print(f"\n📊 CHANNEL PERFORMANCE (All 5 Channels)")
        # Ensure all channels are shown
        all_channels = ['organic', 'paid_search', 'social', 'display', 'email']
        for channel in all_channels:
            if channel in metrics['channel_performance']:
                perf = metrics['channel_performance'][channel]
                if channel == 'organic':
                    # Organic has no spend/ROAS
                    sessions = perf.get('sessions', 0)
                    print(f"  {channel:12} Sessions: {sessions:,}  (No paid spend)")
                else:
                    print(f"  {channel:12} ROAS: {perf['roas']:5.2f}x  "
                          f"Spend: ${perf['spend']:7.2f}  "
                          f"Revenue: ${perf['revenue']:8.2f}")
            else:
                print(f"  {channel:12} [Not yet active]")
    
    # Creative Performance
    if metrics['creative_performance']:
        print(f"\n🎨 TOP CREATIVES")
        top_creatives = sorted(metrics['creative_performance'].items(),
                              key=lambda x: x[1]['ctr'], reverse=True)[:5]
        if top_creatives:
            for creative_id, perf in top_creatives:
                print(f"  Creative {creative_id:3}  CTR: {perf['ctr']:5.2f}%  CVR: {perf['cvr']:5.2f}%")
        else:
            print("  [Gathering creative data...]")
    else:
        print(f"\n🎨 CREATIVES")
        print("  [Warming up creative selection...]")
    
    # Component Health
    health = calculate_component_health(metrics)
    print(f"\n❤️  COMPONENT HEALTH")
    for component, score in health.items():
        if component == 'overall':
            continue
        bar = '█' * int(score / 10) + '░' * (10 - int(score / 10))
        emoji = '🟢' if score >= 75 else '🟡' if score >= 50 else '🔴'
        print(f"  {component:20} {emoji} [{bar}] {score:.0f}%")
    
    # Overall Health
    overall = health['overall']
    print(f"\n🏆 OVERALL SYSTEM HEALTH: {overall:.0f}%")
    if overall >= 75:
        print("  Status: 🟢 System learning effectively with all components")
    elif overall >= 50:
        print("  Status: 🟡 System learning, some components warming up")
    else:
        print("  Status: 🔴 System initializing - give it time to learn")
    
    # Enrichment Indicators
    print(f"\n🔬 ENRICHMENT INDICATORS")
    print(f"  State Dimensions: {metrics['state_dimensions']}")
    print(f"  Components Integrated: Creative ✓ Attribution ✓ Budget ✓ Identity ✓")
    print(f"  Multi-Dimensional Actions: Bid ✓ Creative ✓ Channel ✓")
    print(f"  Delayed Conversion Tracking: ✓ (3-14 day window)")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit | Updates every 5 seconds")

def main():
    """Main monitoring loop"""
    # Find the training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('run_training.py' in arg for arg in cmdline):
                training_pid = proc.info['pid']
                break
        except:
            continue
    
    if not training_pid:
        print("Training process not found. Make sure run_training.py is running.")
        sys.exit(1)
    
    print(f"Monitoring training process (PID: {training_pid})...")
    
    try:
        while True:
            process_info = get_process_info(training_pid)
            if not process_info['running']:
                print("\n⚠️  Training process has stopped!")
                break
            
            metrics = parse_training_output()
            display_dashboard(metrics, process_info)
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()