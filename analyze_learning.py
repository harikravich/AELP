#!/usr/bin/env python3
"""
Analyze what the RL agent is actually learning from training
"""

import os
import re
import numpy as np
from collections import defaultdict, Counter
import json

def analyze_training_log(log_file="training_output.log"):
    """Analyze training patterns to understand what's being learned"""
    
    if not os.path.exists(log_file):
        print(f"❌ {log_file} not found")
        return
    
    # Read recent log data
    with open(log_file, 'r') as f:
        lines = f.readlines()[-10000:]  # Last 10k lines
    
    # Extract data
    bids = []
    wins = []
    losses = []
    positions = []
    prices_paid = []
    user_segments = []
    devices = []
    hours = []
    
    for line in lines:
        # Extract bids
        bid_match = re.search(r'bid=\$?([\d.]+)', line)
        if bid_match:
            bids.append(float(bid_match.group(1)))
        
        # Extract wins
        win_match = re.search(r'WON! Position (\d+), paid \$([\d.]+)', line)
        if win_match:
            position = int(win_match.group(1))
            price = float(win_match.group(2))
            wins.append((position, price))
            positions.append(position)
            prices_paid.append(price)
        
        # Extract losses
        loss_match = re.search(r'Lost\. Position (\d+)', line)
        if loss_match:
            losses.append(int(loss_match.group(1)))
        
        # Extract user segments
        segment_match = re.search(r'segment[\'"]:\s*[\'"]([^\'\"]+)', line)
        if segment_match:
            user_segments.append(segment_match.group(1))
        
        # Extract devices
        device_match = re.search(r'device[\'"]:\s*[\'"]([^\'\"]+)', line)
        if device_match:
            devices.append(device_match.group(1))
        
        # Extract hour
        hour_match = re.search(r'hour_of_day[\'"]:\s*(\d+)', line)
        if hour_match:
            hours.append(int(hour_match.group(1)))
    
    print("=" * 70)
    print(" GAELP RL TRAINING ANALYSIS - WHAT ARE WE LEARNING? ".center(70))
    print("=" * 70)
    
    # 1. BIDDING STRATEGY LEARNING
    print("\n📊 BIDDING STRATEGY LEARNING")
    if bids:
        recent_bids = bids[-100:]
        early_bids = bids[:100] if len(bids) > 200 else []
        
        print(f"  Total bids placed: {len(bids)}")
        print(f"  Bid range: ${min(bids):.2f} - ${max(bids):.2f}")
        print(f"  Average bid: ${np.mean(bids):.2f}")
        print(f"  Bid std dev: ${np.std(bids):.2f}")
        
        if early_bids:
            print(f"\n  Early training (first 100):")
            print(f"    Avg: ${np.mean(early_bids):.2f}, Std: ${np.std(early_bids):.2f}")
            print(f"  Recent training (last 100):")
            print(f"    Avg: ${np.mean(recent_bids):.2f}, Std: ${np.std(recent_bids):.2f}")
            
            # Check if converging or exploring
            if np.std(recent_bids) < np.std(early_bids) * 0.5:
                print("  ➡️ CONVERGING: Agent is settling on optimal bid range")
            elif np.std(recent_bids) > np.std(early_bids) * 1.5:
                print("  🔄 EXPLORING: Agent is still exploring bid space")
            else:
                print("  ⚖️ BALANCED: Agent maintains exploration/exploitation balance")
    
    # 2. WIN RATE OPTIMIZATION
    print("\n🏆 WIN RATE OPTIMIZATION")
    if wins or losses:
        total_auctions = len(wins) + len(losses)
        win_rate = len(wins) / total_auctions * 100
        
        print(f"  Total auctions: {total_auctions}")
        print(f"  Wins: {len(wins)} ({win_rate:.1f}%)")
        print(f"  Losses: {len(losses)} ({100-win_rate:.1f}%)")
        
        if positions:
            avg_position = np.mean(positions)
            print(f"  Average winning position: {avg_position:.1f}")
            
            # Position distribution
            position_counts = Counter(positions)
            print(f"  Position distribution:")
            for pos in sorted(position_counts.keys())[:5]:
                count = position_counts[pos]
                pct = count / len(positions) * 100
                print(f"    Position {pos}: {count} times ({pct:.1f}%)")
        
        # Check learning progress
        if len(wins) > 100:
            early_wins = wins[:50]
            recent_wins = wins[-50:]
            early_win_positions = [w[0] for w in early_wins]
            recent_win_positions = [w[0] for w in recent_wins]
            
            print(f"\n  Position improvement:")
            print(f"    Early avg position: {np.mean(early_win_positions):.1f}")
            print(f"    Recent avg position: {np.mean(recent_win_positions):.1f}")
            
            if np.mean(recent_win_positions) < np.mean(early_win_positions) - 0.5:
                print("    ✅ IMPROVING: Getting better positions over time")
            elif np.mean(recent_win_positions) > np.mean(early_win_positions) + 0.5:
                print("    ⚠️ DEGRADING: Positions getting worse")
            else:
                print("    ➡️ STABLE: Maintaining position")
    
    # 3. COST EFFICIENCY LEARNING
    print("\n💰 COST EFFICIENCY LEARNING")
    if prices_paid:
        print(f"  Total cost: ${sum(prices_paid):.2f}")
        print(f"  Average CPC: ${np.mean(prices_paid):.2f}")
        print(f"  CPC range: ${min(prices_paid):.2f} - ${max(prices_paid):.2f}")
        
        # ROI calculation (assuming $100 LTV per conversion)
        if wins:
            roi_data = []
            for position, price in wins:
                # Better positions = higher conversion rate
                conversion_rate = 0.05 / position  # 5% for pos 1, 2.5% for pos 2, etc.
                expected_value = 100 * conversion_rate
                roi = (expected_value - price) / price * 100
                roi_data.append(roi)
            
            if roi_data:
                print(f"\n  Expected ROI:")
                print(f"    Average: {np.mean(roi_data):.1f}%")
                print(f"    Best: {max(roi_data):.1f}%")
                print(f"    Worst: {min(roi_data):.1f}%")
    
    # 4. CONTEXTUAL LEARNING
    print("\n🎯 CONTEXTUAL LEARNING")
    
    if user_segments:
        segment_counts = Counter(user_segments)
        print(f"  User segments targeted:")
        for segment, count in segment_counts.most_common(4):
            print(f"    {segment}: {count} times")
    
    if devices:
        device_counts = Counter(devices)
        print(f"  Device targeting:")
        for device, count in device_counts.most_common():
            print(f"    {device}: {count} times")
    
    if hours:
        print(f"  Time targeting:")
        print(f"    Most active hours: {Counter(hours).most_common(3)}")
    
    # 5. LEARNING SIGNALS
    print("\n🧠 LEARNING SIGNALS")
    
    # Check for bid diversity (exploration)
    if bids:
        unique_bids = len(set([round(b, 2) for b in bids]))
        exploration_score = unique_bids / len(bids) * 100
        print(f"  Bid diversity: {unique_bids} unique values ({exploration_score:.1f}% exploration)")
        
        if exploration_score > 20:
            print("    ✅ Good exploration: Agent trying different strategies")
        elif exploration_score > 10:
            print("    ⚠️ Limited exploration: May be converging too fast")
        else:
            print("    ❌ Poor exploration: Agent may be stuck")
    
    # Check for patterns in bidding
    if len(bids) > 100:
        # Look for cyclical patterns
        recent_100 = bids[-100:]
        bid_changes = [recent_100[i+1] - recent_100[i] for i in range(len(recent_100)-1)]
        volatility = np.std(bid_changes)
        
        print(f"  Bid volatility: ${volatility:.3f}")
        if volatility > 1.0:
            print("    🔄 High volatility: Active exploration")
        elif volatility > 0.1:
            print("    ⚖️ Moderate volatility: Balanced learning")
        else:
            print("    📍 Low volatility: Converged strategy")
    
    # 6. WHAT THE AGENT IS LEARNING
    print("\n📚 KEY LEARNINGS")
    
    learnings = []
    
    # Bid-to-position relationship
    if bids and positions and len(bids) == len(positions):
        correlation = np.corrcoef(bids[-100:], positions[-100:])[0, 1]
        if abs(correlation) > 0.3:
            learnings.append(f"Bid-position correlation: {correlation:.2f} (learning bid impact)")
    
    # Time-based patterns
    if hours and len(hours) > 50:
        hour_performance = defaultdict(list)
        for i, hour in enumerate(hours):
            if i < len(wins):
                hour_performance[hour].append(1)  # Win
            elif i < len(wins) + len(losses):
                hour_performance[hour].append(0)  # Loss
        
        best_hours = []
        for hour, results in hour_performance.items():
            if len(results) > 5:
                win_rate = sum(results) / len(results)
                if win_rate > 0.7:
                    best_hours.append(hour)
        
        if best_hours:
            learnings.append(f"Best hours discovered: {best_hours}")
    
    # Segment-specific strategies
    if user_segments and bids and len(user_segments) == len(bids):
        segment_bids = defaultdict(list)
        for segment, bid in zip(user_segments[-100:], bids[-100:]):
            segment_bids[segment].append(bid)
        
        for segment, seg_bids in segment_bids.items():
            if len(seg_bids) > 10:
                avg_bid = np.mean(seg_bids)
                learnings.append(f"{segment}: avg bid ${avg_bid:.2f}")
    
    if learnings:
        for learning in learnings[:5]:
            print(f"  ✓ {learning}")
    else:
        print("  ⚠️ No clear patterns detected yet - needs more training")
    
    print("\n" + "=" * 70)
    
    # Summary
    if bids and (wins or losses):
        print("\n🎯 TRAINING SUMMARY:")
        print(f"  The agent is learning to:")
        
        if win_rate > 50:
            print(f"  • Win auctions efficiently ({win_rate:.0f}% win rate)")
        else:
            print(f"  • Improve win rate (currently {win_rate:.0f}%)")
        
        if avg_position < 3:
            print(f"  • Secure top positions (avg position {avg_position:.1f})")
        else:
            print(f"  • Compete for better positions (currently {avg_position:.1f})")
        
        if exploration_score > 15:
            print(f"  • Explore bid strategies ({exploration_score:.0f}% diversity)")
        else:
            print(f"  • Exploit learned strategies (converged)")
        
        if prices_paid:
            avg_cpc = np.mean(prices_paid)
            print(f"  • Optimize CPC (currently ${avg_cpc:.2f})")

if __name__ == "__main__":
    analyze_training_log()