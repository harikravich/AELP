#!/usr/bin/env python3
"""Real-time training monitor for GAELP parallel training"""

import os
import time
import psutil
import subprocess
from datetime import datetime, timedelta
import re

def get_process_info(pid=None):
    """Get process information"""
    # Auto-detect training process if no PID given
    if pid is None:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline', []))
                if 'launch_parallel_training.py' in cmdline or 'parallel_training' in cmdline:
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

def parse_ray_logs():
    """Parse Ray logs for training metrics"""
    metrics = {
        'episodes': 0,
        'users_created': 0,
        'auctions': 0,
        'wins': 0,
        'losses': 0,
        'avg_bid': 0,
        'positions': [],
        'bid_variance': 0,
        'exploration_rate': 0,
        'quality_score': 0
    }
    
    try:
        # Try to get metrics from training output log - check multiple possible log files
        log_files = ["training_working.log", "training_final.log", "training_restart.log", 
                     "training_live.log", "training_test.log", "training.log"]
        terminal_output = ""
        
        # First check regular log files
        for log_file in log_files:
            if os.path.exists(log_file):
                result = subprocess.run(
                    [f"tail -500 {log_file} 2>/dev/null"],
                    shell=True, capture_output=True, text=True, timeout=1
                )
                if result.stdout:
                    terminal_output = result.stdout
                    break
        
        # Also check training_output.log if exists
        if not terminal_output and os.path.exists("training_output.log"):
            result = subprocess.run(
                ["tail -1000 training_output.log 2>/dev/null"],
                shell=True, capture_output=True, text=True, timeout=1
            )
            if result.stdout:
                terminal_output = result.stdout
        
        # Also check Ray session logs if no regular logs found
        if not terminal_output:
            result = subprocess.run(
                ["find /tmp/ray -name 'python-core-driver-*.log' -mmin -5 2>/dev/null | head -1"],
                shell=True, capture_output=True, text=True, timeout=1
            )
            if result.stdout.strip():
                ray_log = result.stdout.strip()
                result = subprocess.run(
                    [f"tail -1000 {ray_log} 2>/dev/null"],
                    shell=True, capture_output=True, text=True, timeout=1
                )
                if result.stdout:
                    terminal_output = result.stdout
        
        if not terminal_output:
            # Try to find any log file with training in name
            result = subprocess.run(
                ["tail -500 training*.log 2>/dev/null | head -500"],
                shell=True, capture_output=True, text=True, timeout=1
            )
            terminal_output = result.stdout
        
        # Count users created - check full log
        if os.path.exists("training_output.log"):
            try:
                result = subprocess.run(
                    ["grep -c 'Created new persistent user' training_output.log 2>/dev/null || echo 0"],
                    shell=True, capture_output=True, text=True, timeout=1
                )
                user_count = result.stdout.strip()
                metrics['users_created'] = int(user_count) if user_count else len(re.findall(r'Created new persistent user', terminal_output))
            except:
                users = re.findall(r'Created new persistent user', terminal_output)
                metrics['users_created'] = len(users)
        else:
            users = re.findall(r'Created new persistent user', terminal_output)
            metrics['users_created'] = len(users)
        
        # Parse auction results - look for WON and Lost patterns
        wins = re.findall(r'WON! Position (\d+), paid \$([\d.]+)', terminal_output)
        losses = re.findall(r'Lost\. Position (\d+)', terminal_output)
        
        # Also check for alternative format from enhanced_simulator_fixed
        if not wins:
            wins = re.findall(r'Step \d+: WON! Position (\d+), paid \$([\d.]+)', terminal_output)
        if not losses:
            losses = re.findall(r'Step \d+: Lost\. Position (\d+)', terminal_output)
            
        # If still no auction data, check full log
        if not wins and os.path.exists("training_output.log"):
            try:
                result = subprocess.run(
                    ["grep 'Step.*WON' training_output.log 2>/dev/null | tail -100"],
                    shell=True, capture_output=True, text=True, timeout=1
                )
                if result.stdout:
                    wins = re.findall(r'Step \d+: WON! Position (\d+), paid \$([\d.]+)', result.stdout)
                    
                result = subprocess.run(
                    ["grep 'Step.*Lost' training_output.log 2>/dev/null | tail -100"],
                    shell=True, capture_output=True, text=True, timeout=1
                )
                if result.stdout:
                    losses = re.findall(r'Step \d+: Lost\. Position (\d+)', result.stdout)
            except:
                pass
        
        # Parse bids and positions from terminal - look for auction logging
        bids = re.findall(r'Running auction with bid=\$?([\d.]+)', terminal_output)
        if not bids:
            bids = re.findall(r'bid=\$?([\d.]+)', terminal_output)
            
        # Also check full log for bids
        if not bids and os.path.exists("training_output.log"):
            try:
                result = subprocess.run(
                    ["grep 'Running auction with bid' training_output.log 2>/dev/null | tail -100"],
                    shell=True, capture_output=True, text=True, timeout=1
                )
                if result.stdout:
                    bids = re.findall(r'bid=\$?([\d.]+)', result.stdout)
            except:
                pass
                
        positions = re.findall(r'Position (\d+)', terminal_output)
        
        if bids:
            bid_floats = [float(b) for b in bids[-50:]]
            metrics['avg_bid'] = sum(bid_floats) / len(bid_floats)
            metrics['auctions'] = len(bids)
            
            # Calculate variance for exploration metric
            if len(bid_floats) > 1:
                import numpy as np
                metrics['bid_variance'] = np.std(bid_floats)
                metrics['exploration_rate'] = (metrics['bid_variance'] / (metrics['avg_bid'] + 0.01)) * 100
        
        # Process wins and losses
        if wins:
            metrics['wins'] = len(wins)
            win_positions = [int(p) for p, _ in wins]
            metrics['positions'] = win_positions[-30:]  # Last 30 positions
            
        if losses:
            metrics['losses'] = len(losses)
            
        metrics['auctions'] = metrics['wins'] + metrics['losses']
            
    except Exception as e:
        # Fallback to checking Ray logs
        try:
            user_count = subprocess.run(
                ["grep -c 'Created new persistent user' /tmp/ray/session*/logs/*.log 2>/dev/null || echo 0"],
                shell=True, capture_output=True, text=True
            ).stdout.strip()
            metrics['users_created'] = int(user_count) if user_count else 0
        except:
            pass
    
    # Parse episodes and training info
    episodes = re.findall(r'Episode (\d+) complete', terminal_output)
    if episodes:
        metrics['episodes'] = len(episodes)
    
    # Check for parallel training steps
    step_logs = re.findall(r'Step (\d+)/\d+: (\d+) experiences collected', terminal_output)
    if step_logs:
        metrics['episodes'] = len(step_logs) * 100  # Steps * environments
        
    # Calculate quality score with lower thresholds for early detection
    quality_score = 0
    if metrics['exploration_rate'] > 5:  # Lower threshold
        quality_score += 1  # Good exploration
    if metrics['bid_variance'] > 0.3:  # Lower threshold
        quality_score += 1  # Diverse bidding
    if metrics['wins'] > 0:
        quality_score += 1  # Getting some wins
    if metrics['users_created'] > 10:  # Much lower threshold
        quality_score += 1  # Making progress
    
    metrics['quality_score'] = quality_score
    
    return metrics

def estimate_progress(uptime_seconds, target_episodes=100000):
    """Estimate training progress"""
    # Rough estimate: 500-600 episodes per minute with 16 environments
    episodes_per_second = 10  # Conservative
    estimated_episodes = uptime_seconds * episodes_per_second
    
    progress = min(100, (estimated_episodes / target_episodes) * 100)
    eta_seconds = (target_episodes - estimated_episodes) / episodes_per_second if episodes_per_second > 0 else 0
    
    return {
        'estimated_episodes': int(estimated_episodes),
        'progress_pct': progress,
        'eta': timedelta(seconds=int(eta_seconds))
    }

def display_dashboard():
    """Display training dashboard"""
    os.system('clear')
    
    print("=" * 70)
    print(" GAELP PARALLEL TRAINING MONITOR ".center(70))
    print("=" * 70)
    
    # Process info
    proc_info = get_process_info()
    if not proc_info:
        print("\n⚠️  Training process not running")
        print("  Analyzing log files for recent training data...")
        # Create fake proc_info from log analysis
        proc_info = {
            'status': 'Stopped',
            'cpu': 0,
            'memory': 0,
            'uptime': 0,
            'pid': 'N/A'
        }
    
    print(f"\n📊 PROCESS STATUS")
    print(f"  PID: {proc_info['pid']}")
    print(f"  Status: ✅ {proc_info['status']}")
    print(f"  CPU Usage: {proc_info['cpu']:.1f}%")
    print(f"  Memory: {proc_info['memory']:.1f} MB")
    print(f"  Uptime: {timedelta(seconds=int(proc_info['uptime']))}")
    
    # Progress estimation
    progress = estimate_progress(proc_info['uptime'])
    print(f"\n📈 TRAINING PROGRESS")
    print(f"  Target: 100,000 episodes")
    print(f"  Estimated Complete: {progress['estimated_episodes']:,} ({progress['progress_pct']:.1f}%)")
    print(f"  ETA: {progress['eta']}")
    
    # Progress bar
    bar_width = 50
    filled = int(bar_width * progress['progress_pct'] / 100)
    bar = '█' * filled + '░' * (bar_width - filled)
    print(f"  [{bar}] {progress['progress_pct']:.1f}%")
    
    # Training metrics
    metrics = parse_ray_logs()
    print(f"\n🎯 LEARNING METRICS")
    print(f"  Users Created: {metrics['users_created']:,}")
    print(f"  Auctions Run: {metrics['auctions']}")
    
    # Show win/loss breakdown
    if metrics['wins'] > 0 or metrics['losses'] > 0:
        print(f"  ├─ Wins: {metrics['wins']}")
        print(f"  └─ Losses: {metrics['losses']}")
    
    if metrics['avg_bid'] > 0:
        print(f"\n💰 BIDDING BEHAVIOR")
        print(f"  Avg Bid (last 50): ${metrics['avg_bid']:.2f}")
        print(f"  Bid Variance: ${metrics['bid_variance']:.2f}")
        print(f"  Exploration Rate: {metrics['exploration_rate']:.1f}%")
        
        # Show bid range
        if metrics['bid_variance'] > 0:
            min_bid = metrics['avg_bid'] - (2 * metrics['bid_variance'])
            max_bid = metrics['avg_bid'] + (2 * metrics['bid_variance'])
            print(f"  Bid Range: ${max(0, min_bid):.2f} - ${max_bid:.2f}")
    
    if metrics['positions']:
        print(f"\n🏆 AUCTION PERFORMANCE")
        avg_position = sum(metrics['positions']) / len(metrics['positions'])
        print(f"  Avg Position (last 30): {avg_position:.1f}")
        
        # Position distribution
        top3 = sum(1 for p in metrics['positions'] if p <= 3)
        print(f"  Top 3 Positions: {top3}/{len(metrics['positions'])}")
        
        total_auctions = metrics['wins'] + metrics['losses']
        if total_auctions > 0:
            win_pct = (metrics['wins'] / total_auctions) * 100
            print(f"  Win Rate: {win_pct:.1f}% ({metrics['wins']} wins / {metrics['losses']} losses)")
        else:
            print(f"  Win Rate: {metrics['wins']}/{len(metrics['positions'])}")
        
        # Show learning trend
        if len(metrics['positions']) >= 10:
            first_half = metrics['positions'][:len(metrics['positions'])//2]
            second_half = metrics['positions'][len(metrics['positions'])//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg < first_avg - 0.5:
                print(f"  📈 IMPROVING! Position {first_avg:.1f} → {second_avg:.1f}")
            elif second_avg > first_avg + 0.5:
                print(f"  📉 Exploring: Position {first_avg:.1f} → {second_avg:.1f}")
            else:
                print(f"  ➡️ Stable: Position ~{first_avg:.1f}")
    
    # Quality assessment
    print(f"\n✅ TRAINING QUALITY")
    quality_indicators = []
    
    if metrics['exploration_rate'] > 10:
        quality_indicators.append("✅ Good exploration")
    elif metrics['exploration_rate'] > 5:
        quality_indicators.append("⚠️ Low exploration")
    else:
        quality_indicators.append("❌ No exploration detected")
    
    if metrics['wins'] > 0:
        win_pct = (metrics['wins'] / max(1, len(metrics['positions']))) * 100
        if win_pct > 20:
            quality_indicators.append(f"✅ Winning {win_pct:.0f}% auctions")
        else:
            quality_indicators.append(f"⚠️ Low win rate ({win_pct:.0f}%)")
    
    for indicator in quality_indicators:
        print(f"  {indicator}")
    
    # Overall quality score
    score = metrics['quality_score']
    print(f"\n  Quality Score: {'⭐' * score}{'☆' * (4-score)} ({score}/4)")
    
    if score >= 3:
        print("  Status: 🟢 Learning effectively")
    elif score >= 2:
        print("  Status: 🟡 Learning slowly")
    elif progress['estimated_episodes'] < 1000:
        print("  Status: 🔵 Too early to assess")
    else:
        print("  Status: 🔴 Check configuration")
    
    return True  # Always return True to keep monitoring
    
    # Speed metrics
    if proc_info['uptime'] > 0:
        episodes_per_min = (progress['estimated_episodes'] / proc_info['uptime']) * 60
        print(f"\n⚡ SPEED")
        print(f"  Episodes/min: {episodes_per_min:.0f}")
        print(f"  Time to 100k: {progress['eta']}")
    
    print("\n" + "=" * 70)
    print("Press Ctrl+C to exit | Updates every 10 seconds")
    
    return True

def main():
    """Main monitoring loop"""
    print("Starting GAELP Training Monitor...")
    
    try:
        while True:
            if not display_dashboard():
                print("\nTraining process not found. Exiting...")
                break
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")

if __name__ == "__main__":
    main()