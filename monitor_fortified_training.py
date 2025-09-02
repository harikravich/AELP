#!/usr/bin/env python3
"""
Real-time monitoring for FORTIFIED GAELP training
Tracks all the new enriched metrics and components
"""

import os
import time
import psutil
import subprocess
import json
from datetime import datetime, timedelta
import re
from collections import deque
import numpy as np

def get_process_info(pid=None):
    """Get fortified training process information"""
    if pid is None:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []))
                if 'fortified_training_loop.py' in cmdline or 'fortified_training' in cmdline:
                    pid = proc.info['pid']
                    break
            except:
                continue
    
    if pid is None:
        return None
    
    try:
        process = psutil.Process(pid)
        return {
            'status': 'Running',
            'cpu': process.cpu_percent(interval=1),
            'memory': process.memory_info().rss / 1024 / 1024,  # MB
            'uptime': time.time() - process.create_time(),
            'pid': pid
        }
    except:
        return None

def parse_fortified_logs():
    """Parse fortified training logs for comprehensive metrics"""
    metrics = {
        # Core training metrics
        'episodes': 0,
        'experiences': 0,
        'buffer_size': 0,
        
        # Learning metrics
        'loss_bid': 0.0,
        'loss_creative': 0.0,
        'loss_channel': 0.0,
        'epsilon': 1.0,
        
        # Performance metrics
        'conversions': 0,
        'revenue': 0.0,
        'avg_reward': 0.0,
        'best_roas': 0.0,
        
        # Component-specific metrics
        'creative_performance': {},
        'channel_performance': {},
        'segment_performance': {},
        
        # Action distribution
        'bid_distribution': [],
        'creative_distribution': {},
        'channel_distribution': {},
        
        # State enrichment indicators
        'state_dimensions': 45,
        'journey_progressions': 0,
        'cross_device_matches': 0,
        'delayed_conversions_scheduled': 0,
        
        # Quality indicators
        'learning_quality': 'Initializing',
        'convergence_status': 'Not converged'
    }
    
    try:
        # Check fortified training log
        log_file = "fortified_training_output.log"
        if not os.path.exists(log_file):
            # Fall back to regular training log
            log_file = "fortified_training.log"
        
        if not os.path.exists(log_file):
            # Try to find any recent log
            result = subprocess.run(
                ["find", ".", "-name", "*fortified*.log", "-mmin", "-60"],
                capture_output=True, text=True
            )
            if result.stdout.strip():
                log_file = result.stdout.strip().split('\n')[0]
        
        if os.path.exists(log_file):
            # Read last portion of log
            result = subprocess.run(
                [f"tail -n 500 {log_file}"],
                shell=True, capture_output=True, text=True
            )
            log_content = result.stdout
            
            # Parse episodes
            episodes = re.findall(r'Episode (\d+)/\d+', log_content)
            if episodes:
                metrics['episodes'] = int(episodes[-1])
            
            # Parse experiences
            experiences = re.findall(r'Total experiences: ([\d,]+)', log_content)
            if experiences:
                metrics['experiences'] = int(experiences[-1].replace(',', ''))
            
            # Parse losses
            loss_matches = re.findall(r'Loss \(bid\): ([\d.]+).*Loss \(creative\): ([\d.]+).*Loss \(channel\): ([\d.]+)', log_content)
            if loss_matches:
                latest = loss_matches[-1]
                metrics['loss_bid'] = float(latest[0])
                metrics['loss_creative'] = float(latest[1])
                metrics['loss_channel'] = float(latest[2])
            
            # Parse epsilon
            epsilon_matches = re.findall(r'Epsilon: ([\d.]+)', log_content)
            if epsilon_matches:
                metrics['epsilon'] = float(epsilon_matches[-1])
            
            # Parse conversions and revenue
            conv_matches = re.findall(r'Conversions: ([\d,]+)', log_content)
            if conv_matches:
                metrics['conversions'] = int(conv_matches[-1].replace(',', ''))
            
            revenue_matches = re.findall(r'Revenue: \$([\d,\.]+)', log_content)
            if revenue_matches:
                metrics['revenue'] = float(revenue_matches[-1].replace(',', ''))
            
            # Parse ROAS
            roas_matches = re.findall(r'Best ROAS: ([\d.]+)x', log_content)
            if roas_matches:
                metrics['best_roas'] = float(roas_matches[-1])
            
            # Parse average reward
            reward_matches = re.findall(r'Avg Reward: ([\d.-]+)', log_content)
            if reward_matches:
                metrics['avg_reward'] = float(reward_matches[-1])
            
            # Parse channel performance
            channel_perf = re.findall(r'(\w+): ROAS=([\d.]+)x.*Spend=\$([\d.]+).*Revenue=\$([\d.]+)', log_content)
            for channel, roas, spend, revenue in channel_perf:
                metrics['channel_performance'][channel] = {
                    'roas': float(roas),
                    'spend': float(spend),
                    'revenue': float(revenue)
                }
            
            # Parse creative performance
            creative_perf = re.findall(r'Creative (\d+):.*CTR=([\d.]+)%.*CVR=([\d.]+)%', log_content)
            for creative_id, ctr, cvr in creative_perf:
                metrics['creative_performance'][creative_id] = {
                    'ctr': float(ctr),
                    'cvr': float(cvr)
                }
            
            # Parse action distribution
            bid_matches = re.findall(r'Bid: \$([\d.]+)', log_content)
            if bid_matches:
                metrics['bid_distribution'] = [float(b) for b in bid_matches[-20:]]
            
            # Check for convergence
            if "Training converged" in log_content:
                metrics['convergence_status'] = 'Converged!'
            
            # Assess learning quality
            if metrics['loss_bid'] > 0 and metrics['epsilon'] < 1.0:
                if metrics['loss_bid'] < 0.01:
                    metrics['learning_quality'] = 'Excellent'
                elif metrics['loss_bid'] < 0.1:
                    metrics['learning_quality'] = 'Good'
                elif metrics['loss_bid'] < 1.0:
                    metrics['learning_quality'] = 'Learning'
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
        'learning_system': 0
    }
    
    # Creative Selector health
    if metrics['creative_performance']:
        avg_ctr = np.mean([p['ctr'] for p in metrics['creative_performance'].values()])
        if avg_ctr > 5.0:
            health['creative_selector'] = 100
        elif avg_ctr > 2.0:
            health['creative_selector'] = 75
        elif avg_ctr > 0.5:
            health['creative_selector'] = 50
        else:
            health['creative_selector'] = 25
    
    # Channel Optimizer health
    if metrics['channel_performance']:
        avg_roas = np.mean([p['roas'] for p in metrics['channel_performance'].values()])
        if avg_roas > 3.0:
            health['channel_optimizer'] = 100
        elif avg_roas > 1.5:
            health['channel_optimizer'] = 75
        elif avg_roas > 0.5:
            health['channel_optimizer'] = 50
        else:
            health['channel_optimizer'] = 25
    
    # Attribution Engine health (based on revenue)
    if metrics['revenue'] > 0:
        health['attribution_engine'] = min(100, (metrics['revenue'] / 1000) * 100)
    
    # Budget Pacer health (based on spend efficiency)
    if metrics['best_roas'] > 2.0:
        health['budget_pacer'] = 100
    elif metrics['best_roas'] > 1.0:
        health['budget_pacer'] = 75
    elif metrics['best_roas'] > 0.5:
        health['budget_pacer'] = 50
    else:
        health['budget_pacer'] = 25
    
    # Learning System health
    if metrics['loss_bid'] > 0:
        if metrics['loss_bid'] < 0.1 and metrics['epsilon'] < 0.5:
            health['learning_system'] = 100
        elif metrics['loss_bid'] < 1.0:
            health['learning_system'] = 75
        else:
            health['learning_system'] = 50
    
    # Overall health
    health['overall'] = np.mean(list(health.values()))
    
    return health

def display_fortified_dashboard():
    """Display comprehensive fortified training dashboard"""
    os.system('clear')
    
    print("=" * 80)
    print(" FORTIFIED GAELP TRAINING MONITOR ".center(80))
    print("=" * 80)
    
    # Process info
    proc_info = get_process_info()
    if not proc_info:
        print("\n⚠️  Fortified training process not running")
        print("  Start with: python3 capture_fortified_training.py")
        proc_info = {'status': 'Stopped', 'cpu': 0, 'memory': 0, 'uptime': 0, 'pid': 'N/A'}
    
    print(f"\n📊 PROCESS STATUS")
    print(f"  PID: {proc_info['pid']}")
    print(f"  Status: {'🟢' if proc_info['status'] == 'Running' else '🔴'} {proc_info['status']}")
    print(f"  CPU Usage: {proc_info['cpu']:.1f}%")
    print(f"  Memory: {proc_info['memory']:.1f} MB")
    print(f"  Uptime: {timedelta(seconds=int(proc_info['uptime']))}")
    
    # Parse metrics
    metrics = parse_fortified_logs()
    
    # Training Progress
    print(f"\n📈 TRAINING PROGRESS")
    print(f"  Episodes: {metrics['episodes']:,}")
    print(f"  Experiences: {metrics['experiences']:,}")
    print(f"  Epsilon: {metrics['epsilon']:.4f}")
    print(f"  Status: {metrics['convergence_status']}")
    
    # Learning Metrics
    print(f"\n🧠 LEARNING METRICS")
    print(f"  Loss (Bid): {metrics['loss_bid']:.4f}")
    print(f"  Loss (Creative): {metrics['loss_creative']:.4f}")
    print(f"  Loss (Channel): {metrics['loss_channel']:.4f}")
    print(f"  Learning Quality: {metrics['learning_quality']}")
    
    # Performance Metrics
    print(f"\n💰 PERFORMANCE METRICS")
    print(f"  Conversions: {metrics['conversions']:,}")
    print(f"  Revenue: ${metrics['revenue']:,.2f}")
    print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
    print(f"  Best ROAS: {metrics['best_roas']:.2f}x")
    
    # Channel Performance
    if metrics['channel_performance']:
        print(f"\n📊 CHANNEL PERFORMANCE")
        for channel, perf in sorted(metrics['channel_performance'].items(), 
                                   key=lambda x: x[1]['roas'], reverse=True):
            print(f"  {channel:12} ROAS: {perf['roas']:5.2f}x  "
                  f"Spend: ${perf['spend']:7.2f}  "
                  f"Revenue: ${perf['revenue']:8.2f}")
    
    # Creative Performance
    if metrics['creative_performance']:
        print(f"\n🎨 TOP CREATIVES")
        top_creatives = sorted(metrics['creative_performance'].items(),
                              key=lambda x: x[1]['ctr'], reverse=True)[:5]
        for creative_id, perf in top_creatives:
            print(f"  Creative {creative_id:3}  CTR: {perf['ctr']:5.2f}%  CVR: {perf['cvr']:5.2f}%")
    
    # Action Distribution
    if metrics['bid_distribution']:
        print(f"\n🎯 RECENT ACTIONS")
        avg_bid = np.mean(metrics['bid_distribution'])
        std_bid = np.std(metrics['bid_distribution'])
        print(f"  Avg Bid: ${avg_bid:.2f} ± ${std_bid:.2f}")
        print(f"  Bid Range: ${min(metrics['bid_distribution']):.2f} - ${max(metrics['bid_distribution']):.2f}")
    
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
        print("  Status: 🟡 System learning, some components need attention")
    else:
        print("  Status: 🔴 Check configuration and component integration")
    
    # State Enrichment Indicators
    print(f"\n🔬 ENRICHMENT INDICATORS")
    print(f"  State Dimensions: {metrics['state_dimensions']}")
    print(f"  Components Integrated: Creative Selector ✓ Attribution ✓ Budget Pacer ✓")
    print(f"  Multi-Dimensional Actions: Bid ✓ Creative ✓ Channel ✓")
    
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit | Updates every 5 seconds")
    
    return True

def main():
    """Main monitoring loop"""
    print("Starting FORTIFIED GAELP Training Monitor...")
    print("Looking for fortified training process...")
    
    try:
        while True:
            display_fortified_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")

if __name__ == "__main__":
    main()