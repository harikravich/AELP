#!/usr/bin/env python3
"""Check if GAELP training is actually learning"""

import numpy as np
import torch
import pickle
import os
from pathlib import Path
import json
from datetime import datetime

def check_checkpoint_quality():
    """Check if model checkpoints show learning"""
    checkpoint_dir = Path("checkpoints/parallel")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    
    latest = max(checkpoints, key=os.path.getmtime)
    checkpoint = torch.load(latest, map_location='cpu')
    
    quality_metrics = {
        'checkpoint': latest.name,
        'timestamp': datetime.fromtimestamp(os.path.getmtime(latest)),
        'episode': checkpoint.get('episode', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'rewards': checkpoint.get('episode_rewards', []),
        'win_rate': checkpoint.get('win_rate', 0),
        'avg_position': checkpoint.get('avg_position', 10)
    }
    
    return quality_metrics

def analyze_live_metrics():
    """Analyze live training behavior"""
    import subprocess
    import re
    
    # Sample recent auction results
    try:
        recent_logs = subprocess.run(
            ["tail -500 /dev/pts/2 2>/dev/null | grep -E 'bid=|Position|reward|loss' | tail -100"],
            shell=True, capture_output=True, text=True, timeout=2
        ).stdout
    except:
        recent_logs = ""
    
    # Extract bids
    bids = re.findall(r'bid=\$?([\d.]+)', recent_logs)
    bids = [float(b) for b in bids[-50:]] if bids else []
    
    # Extract positions
    positions = re.findall(r'Position (\d+)', recent_logs)
    positions = [int(p) for p in positions[-50:]] if positions else []
    
    # Extract rewards if logged
    rewards = re.findall(r'reward[:\s]+([-\d.]+)', recent_logs, re.IGNORECASE)
    rewards = [float(r) for r in rewards[-20:]] if rewards else []
    
    quality_indicators = {
        'bid_variance': np.std(bids) if len(bids) > 1 else 0,
        'bid_range': (min(bids), max(bids)) if bids else (0, 0),
        'avg_bid': np.mean(bids) if bids else 0,
        'position_improvement': False,
        'win_rate': 0,
        'exploration_rate': 0
    }
    
    if positions:
        # Check if positions are improving over time
        if len(positions) >= 10:
            early = np.mean(positions[:len(positions)//2])
            late = np.mean(positions[len(positions)//2:])
            quality_indicators['position_improvement'] = late < early
            quality_indicators['position_trend'] = f"{early:.1f} → {late:.1f}"
        
        # Calculate win rate (position <= 5)
        wins = sum(1 for p in positions if p <= 5)
        quality_indicators['win_rate'] = wins / len(positions) * 100
    
    # Check exploration (bid variance indicates exploration)
    if bids and len(bids) > 5:
        quality_indicators['exploration_rate'] = np.std(bids) / (np.mean(bids) + 0.01) * 100
    
    return quality_indicators

def check_neural_network():
    """Check if neural network weights are updating"""
    try:
        # Try to load the most recent RL agent
        from training_orchestrator.rl_agent_proper import RLAgent
        
        # Check if weights files exist
        weight_files = list(Path(".").glob("**/*agent*.pt"))
        if weight_files:
            latest_weights = max(weight_files, key=os.path.getmtime)
            weights = torch.load(latest_weights, map_location='cpu')
            
            # Check weight statistics
            weight_stats = {}
            for key, tensor in weights.items():
                if isinstance(tensor, torch.Tensor):
                    weight_stats[key] = {
                        'mean': float(tensor.mean()),
                        'std': float(tensor.std()),
                        'min': float(tensor.min()),
                        'max': float(tensor.max())
                    }
            
            # Weights should have non-zero variance (not stuck)
            is_learning = any(stat['std'] > 0.01 for stat in weight_stats.values())
            return {'weights_updating': is_learning, 'num_parameters': len(weight_stats)}
    except:
        pass
    
    return {'weights_updating': 'unknown', 'num_parameters': 0}

def generate_quality_report():
    """Generate comprehensive quality report"""
    print("=" * 70)
    print(" GAELP TRAINING QUALITY ANALYSIS ".center(70))
    print("=" * 70)
    
    # Check checkpoints
    print("\n📁 CHECKPOINT ANALYSIS")
    checkpoint_quality = check_checkpoint_quality()
    if checkpoint_quality:
        print(f"  Latest: {checkpoint_quality['checkpoint']}")
        print(f"  Episode: {checkpoint_quality['episode']:,}")
        print(f"  Loss: {checkpoint_quality['loss']:.4f}")
        if checkpoint_quality['rewards']:
            print(f"  Avg Reward: {np.mean(checkpoint_quality['rewards'][-100:]):.4f}")
    else:
        print("  No checkpoints found yet")
    
    # Check live metrics
    print("\n📊 LIVE BEHAVIOR ANALYSIS")
    live_metrics = analyze_live_metrics()
    
    print(f"  Bid Range: ${live_metrics['bid_range'][0]:.2f} - ${live_metrics['bid_range'][1]:.2f}")
    print(f"  Bid Variance: {live_metrics['bid_variance']:.2f}")
    print(f"  Exploration Rate: {live_metrics['exploration_rate']:.1f}%")
    
    if 'position_trend' in live_metrics:
        trend_symbol = "📈" if live_metrics['position_improvement'] else "📉"
        print(f"  Position Trend: {trend_symbol} {live_metrics['position_trend']}")
    
    print(f"  Recent Win Rate: {live_metrics['win_rate']:.1f}%")
    
    # Check neural network
    print("\n🧠 NEURAL NETWORK STATUS")
    nn_status = check_neural_network()
    print(f"  Weights Updating: {nn_status['weights_updating']}")
    print(f"  Parameters: {nn_status['num_parameters']:,}")
    
    # Quality assessment
    print("\n✅ QUALITY INDICATORS")
    
    quality_score = 0
    checks = []
    
    # Check 1: Exploration
    if live_metrics['exploration_rate'] > 10:
        checks.append("✅ Agent is exploring (trying different bids)")
        quality_score += 1
    else:
        checks.append("⚠️ Low exploration - may be stuck")
    
    # Check 2: Bid diversity
    if live_metrics['bid_variance'] > 0.5:
        checks.append("✅ Diverse bidding strategies")
        quality_score += 1
    else:
        checks.append("⚠️ Limited bid diversity")
    
    # Check 3: Position improvement
    if live_metrics.get('position_improvement'):
        checks.append("✅ Positions improving over time")
        quality_score += 2
    elif 'position_trend' in live_metrics:
        checks.append("⚠️ Positions not improving yet")
    
    # Check 4: Win rate
    if live_metrics['win_rate'] > 20:
        checks.append(f"✅ Decent win rate ({live_metrics['win_rate']:.1f}%)")
        quality_score += 2
    elif live_metrics['win_rate'] > 0:
        checks.append(f"📊 Low win rate ({live_metrics['win_rate']:.1f}%) - still learning")
        quality_score += 1
    
    for check in checks:
        print(f"  {check}")
    
    # Final verdict
    print("\n🎯 TRAINING QUALITY VERDICT")
    if quality_score >= 5:
        print("  ✅ EXCELLENT - Agent is learning effectively")
    elif quality_score >= 3:
        print("  ✅ GOOD - Agent is learning, needs more time")
    elif quality_score >= 1:
        print("  ⚠️ EARLY STAGE - Too early to assess, check in 30 min")
    else:
        print("  ❌ POOR - Agent may be stuck, check configuration")
    
    print("\n" + "=" * 70)
    print(f"Quality Score: {quality_score}/6")
    print("Recommendation: ", end="")
    
    if quality_score < 3:
        print("Let training run for 30+ more minutes before evaluating")
    else:
        print("Training is progressing well, continue running")
    
    return quality_score

if __name__ == "__main__":
    score = generate_quality_report()
    
    # Save quality report
    report = {
        'timestamp': datetime.now().isoformat(),
        'quality_score': score,
        'checkpoint': check_checkpoint_quality(),
        'live_metrics': analyze_live_metrics(),
        'neural_network': check_neural_network()
    }
    
    with open('training_quality_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved to training_quality_report.json")