#!/usr/bin/env python3
"""
FORTIFIED GAELP TRAINING MONITOR WITH COMPREHENSIVE INSIGHTS
Shows everything: all channels, CAC trends, learning progress, and actionable insights
"""

import os
import sys
import time
import psutil
import re
import numpy as np
from datetime import datetime, timedelta
import json
from collections import deque, defaultdict

class InsightfulMonitor:
    """Monitor with comprehensive insights and trends"""
    
    def __init__(self):
        # Historical data for trends (keep last 100 updates)
        self.conversion_history = deque(maxlen=100)
        self.cac_history = defaultdict(lambda: deque(maxlen=100))
        self.roas_history = defaultdict(lambda: deque(maxlen=100))
        self.creative_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_history = deque(maxlen=100)
        self.revenue_history = deque(maxlen=100)
        
        # Learning milestones
        self.learning_milestones = []
        self.best_performing = {
            'creative': None,
            'channel': None,
            'segment': None,
            'bid_range': None
        }
        
    def get_process_info(self, pid=None):
        """Get training process information"""
        if pid is None:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    if 'fortified_training' in cmdline or 'capture_fortified' in cmdline:
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
                'cpu': process.cpu_percent(interval=0.1),
                'memory': process.memory_info().rss / 1024 / 1024,  # MB
                'uptime': time.time() - process.create_time(),
                'pid': pid
            }
        except:
            return None
    
    def parse_training_output(self, log_file='fortified_training_output.log'):
        """Parse training log for comprehensive metrics"""
        metrics = {
            # Core metrics
            'episodes': 0,
            'experiences': 0,
            'epsilon': 1.0,
            'convergence_status': 'Not converged',
            
            # Learning metrics
            'loss_bid': 0.0,
            'loss_creative': 0.0,
            'loss_channel': 0.0,
            'learning_rate': 0.0,
            
            # Performance metrics
            'total_conversions': 0,
            'immediate_conversions': 0,
            'delayed_conversions': 0,
            'total_revenue': 0.0,
            'total_spend': 0.0,
            'overall_roas': 0.0,
            'avg_reward': 0.0,
            
            # Channel metrics (ALL 5 channels)
            'channels': {
                'organic': {'spend': 0, 'revenue': 0, 'roas': 0, 'conversions': 0, 'sessions': 0, 'cac': 0},
                'paid_search': {'spend': 0, 'revenue': 0, 'roas': 0, 'conversions': 0, 'clicks': 0, 'cac': 0},
                'social': {'spend': 0, 'revenue': 0, 'roas': 0, 'conversions': 0, 'impressions': 0, 'cac': 0},
                'display': {'spend': 0, 'revenue': 0, 'roas': 0, 'conversions': 0, 'impressions': 0, 'cac': 0},
                'email': {'spend': 0, 'revenue': 0, 'roas': 0, 'conversions': 0, 'opens': 0, 'cac': 0}
            },
            
            # Creative metrics
            'creatives': {},
            
            # Segment metrics
            'segments': {},
            
            # Bid insights
            'bid_stats': {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'optimal_range': (0, 0)
            },
            
            # Volume metrics
            'impressions': 0,
            'clicks': 0,
            'ctr': 0.0,
            'conversion_rate': 0.0,
            
            # Agent learning insights
            'learned_patterns': [],
            'exploration_vs_exploitation': 0.0,  # ratio
            'action_diversity': 0.0
        }
        
        try:
            if not os.path.exists(log_file):
                return metrics
            
            # Read last portion of log
            with open(log_file, 'r') as f:
                lines = f.readlines()
                log_content = ''.join(lines[-10000:])  # More lines for better data
            
            # Parse episodes
            episode_matches = re.findall(r'Episode (\d+)/', log_content)
            if episode_matches:
                metrics['episodes'] = int(episode_matches[-1])
            
            # Parse experiences
            exp_matches = re.findall(r'Total experiences: ([\d,]+)', log_content)
            if exp_matches:
                metrics['experiences'] = int(exp_matches[-1].replace(',', ''))
            
            # Parse epsilon (exploration rate)
            epsilon_matches = re.findall(r'Epsilon: ([\d.]+)', log_content)
            if epsilon_matches:
                metrics['epsilon'] = float(epsilon_matches[-1])
                metrics['exploration_vs_exploitation'] = metrics['epsilon']
            
            # Parse losses
            loss_matches = re.findall(r'Loss \(bid\): ([\d.]+).*Loss \(creative\): ([\d.]+).*Loss \(channel\): ([\d.]+)', log_content)
            if loss_matches:
                latest = loss_matches[-1]
                metrics['loss_bid'] = float(latest[0])
                metrics['loss_creative'] = float(latest[1])
                metrics['loss_channel'] = float(latest[2])
            
            # Parse conversions (with detail)
            conv_matches = re.findall(r'Conversions: ([\d,]+)', log_content)
            if conv_matches:
                metrics['total_conversions'] = int(conv_matches[-1].replace(',', ''))
            
            # Try to find delayed vs immediate
            delayed_matches = re.findall(r'Delayed conversions: (\d+)', log_content)
            if delayed_matches:
                metrics['delayed_conversions'] = int(delayed_matches[-1])
                metrics['immediate_conversions'] = metrics['total_conversions'] - metrics['delayed_conversions']
            
            # Parse revenue
            revenue_matches = re.findall(r'Revenue: \$?([\d,\.]+)', log_content)
            if revenue_matches:
                metrics['total_revenue'] = float(revenue_matches[-1].replace(',', ''))
            
            # Parse ALL channel performance
            channel_patterns = [
                (r'organic: ROAS=([\d.]+)x.*Spend=\$?([\d.]+).*Revenue=\$?([\d.]+)', 'organic'),
                (r'paid_search: ROAS=([\d.]+)x.*Spend=\$?([\d.]+).*Revenue=\$?([\d.]+)', 'paid_search'),
                (r'social: ROAS=([\d.]+)x.*Spend=\$?([\d.]+).*Revenue=\$?([\d.]+)', 'social'),
                (r'display: ROAS=([\d.]+)x.*Spend=\$?([\d.]+).*Revenue=\$?([\d.]+)', 'display'),
                (r'email: ROAS=([\d.]+)x.*Spend=\$?([\d.]+).*Revenue=\$?([\d.]+)', 'email')
            ]
            
            total_spend = 0
            for pattern, channel in channel_patterns:
                matches = re.findall(pattern, log_content)
                if matches:
                    latest = matches[-1]
                    metrics['channels'][channel]['roas'] = float(latest[0])
                    metrics['channels'][channel]['spend'] = float(latest[1])
                    metrics['channels'][channel]['revenue'] = float(latest[2])
                    total_spend += float(latest[1])
                    
                    # Calculate CAC
                    if metrics['channels'][channel]['revenue'] > 0:
                        # Estimate conversions from revenue (assuming $100 per conversion)
                        est_conversions = metrics['channels'][channel]['revenue'] / 100
                        metrics['channels'][channel]['conversions'] = int(est_conversions)
                        if est_conversions > 0:
                            metrics['channels'][channel]['cac'] = metrics['channels'][channel]['spend'] / est_conversions
            
            metrics['total_spend'] = total_spend
            if total_spend > 0:
                metrics['overall_roas'] = metrics['total_revenue'] / total_spend
            
            # Parse creative performance
            creative_matches = re.findall(r'Creative (\d+):.*CTR=([\d.]+)%.*CVR=([\d.]+)%', log_content)
            for creative_id, ctr, cvr in creative_matches:
                metrics['creatives'][creative_id] = {
                    'ctr': float(ctr),
                    'cvr': float(cvr),
                    'performance_score': float(ctr) * float(cvr)  # Combined metric
                }
            
            # Parse bid statistics
            bid_matches = re.findall(r'Bid: \$?([\d.]+)', log_content)
            if bid_matches:
                bids = [float(b) for b in bid_matches[-100:]]  # Last 100 bids
                metrics['bid_stats']['mean'] = np.mean(bids)
                metrics['bid_stats']['std'] = np.std(bids)
                metrics['bid_stats']['min'] = min(bids)
                metrics['bid_stats']['max'] = max(bids)
                # Optimal range is mean ± 1 std
                metrics['bid_stats']['optimal_range'] = (
                    max(0, metrics['bid_stats']['mean'] - metrics['bid_stats']['std']),
                    metrics['bid_stats']['mean'] + metrics['bid_stats']['std']
                )
            
            # Calculate CTR and conversion rate
            if metrics['total_conversions'] > 0 and metrics['experiences'] > 0:
                metrics['conversion_rate'] = (metrics['total_conversions'] / max(1, metrics['experiences'])) * 100
            
            # Detect learned patterns
            if metrics['epsilon'] < 0.5:
                metrics['learned_patterns'].append("Exploitation phase - agent is confident")
            if metrics['loss_bid'] < 0.1:
                metrics['learned_patterns'].append("Bid strategy optimized")
            if metrics['loss_creative'] < 0.1:
                metrics['learned_patterns'].append("Creative selection mastered")
            if metrics['loss_channel'] < 0.1:
                metrics['learned_patterns'].append("Channel allocation optimized")
            
            # Check for convergence
            if metrics['epsilon'] <= 0.01:
                metrics['convergence_status'] = 'Converged!'
            elif metrics['epsilon'] <= 0.1:
                metrics['convergence_status'] = 'Near convergence'
            elif metrics['epsilon'] <= 0.5:
                metrics['convergence_status'] = 'Learning'
            else:
                metrics['convergence_status'] = 'Exploring'
            
        except Exception as e:
            print(f"Error parsing logs: {e}")
        
        return metrics
    
    def calculate_insights(self, metrics):
        """Generate actionable insights from metrics"""
        insights = {
            'top_performing_channel': None,
            'worst_performing_channel': None,
            'best_creative': None,
            'cac_trend': 'stable',
            'volume_trend': 'stable',
            'key_learnings': [],
            'recommendations': [],
            'warnings': []
        }
        
        # Find best/worst channels by ROAS
        channel_roas = [(ch, data['roas']) for ch, data in metrics['channels'].items() if data['roas'] > 0]
        if channel_roas:
            channel_roas.sort(key=lambda x: x[1], reverse=True)
            insights['top_performing_channel'] = {
                'name': channel_roas[0][0],
                'roas': channel_roas[0][1],
                'cac': metrics['channels'][channel_roas[0][0]]['cac']
            }
            if len(channel_roas) > 1:
                insights['worst_performing_channel'] = {
                    'name': channel_roas[-1][0],
                    'roas': channel_roas[-1][1],
                    'cac': metrics['channels'][channel_roas[-1][0]]['cac']
                }
        
        # Find best creative
        if metrics['creatives']:
            best_creative = max(metrics['creatives'].items(), 
                              key=lambda x: x[1]['performance_score'])
            insights['best_creative'] = {
                'id': best_creative[0],
                'ctr': best_creative[1]['ctr'],
                'cvr': best_creative[1]['cvr']
            }
        
        # Track CAC trends
        for channel, data in metrics['channels'].items():
            if data['cac'] > 0:
                self.cac_history[channel].append(data['cac'])
                if len(self.cac_history[channel]) > 10:
                    recent = list(self.cac_history[channel])[-10:]
                    if recent[-1] > recent[0] * 1.1:
                        insights['cac_trend'] = 'increasing'
                    elif recent[-1] < recent[0] * 0.9:
                        insights['cac_trend'] = 'decreasing'
        
        # Volume trends
        self.volume_history.append(metrics['total_conversions'])
        if len(self.volume_history) > 10:
            recent = list(self.volume_history)[-10:]
            if recent[-1] > recent[0] * 1.2:
                insights['volume_trend'] = 'growing'
            elif recent[-1] < recent[0] * 0.8:
                insights['volume_trend'] = 'declining'
        
        # Key learnings based on agent behavior
        if metrics['epsilon'] < 0.1:
            insights['key_learnings'].append("✓ Agent has converged on optimal strategy")
        if metrics['loss_bid'] < 0.5:
            insights['key_learnings'].append("✓ Bidding strategy is effective")
        if len([c for c, d in metrics['channels'].items() if d['roas'] > 2]) >= 3:
            insights['key_learnings'].append("✓ Multiple channels achieving 2x+ ROAS")
        if metrics['delayed_conversions'] > metrics['immediate_conversions']:
            insights['key_learnings'].append("✓ Majority of conversions are delayed (3-14 days)")
        
        # Recommendations
        if insights['top_performing_channel']:
            insights['recommendations'].append(
                f"↑ Scale {insights['top_performing_channel']['name']} - achieving {insights['top_performing_channel']['roas']:.1f}x ROAS"
            )
        if insights['worst_performing_channel'] and insights['worst_performing_channel']['roas'] < 1.0:
            insights['recommendations'].append(
                f"↓ Reduce {insights['worst_performing_channel']['name']} spend - only {insights['worst_performing_channel']['roas']:.1f}x ROAS"
            )
        if metrics['epsilon'] > 0.5:
            insights['recommendations'].append("⏳ Allow more training time for convergence")
        
        # Warnings
        if metrics['overall_roas'] < 1.0:
            insights['warnings'].append("⚠️ Overall ROAS below 1.0x - losing money")
        if metrics['total_conversions'] == 0 and metrics['episodes'] > 10:
            insights['warnings'].append("⚠️ No conversions tracked - check tracking setup")
        if all(d['cac'] == 0 for d in metrics['channels'].values()):
            insights['warnings'].append("⚠️ CAC not being calculated properly")
        
        return insights
    
    def display_dashboard(self, metrics, process_info, insights):
        """Display comprehensive dashboard with insights"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 100)
        print("                          FORTIFIED GAELP - INTELLIGENT TRAINING MONITOR                          ")
        print("=" * 100)
        
        # Process Status
        if process_info:
            status_emoji = '🟢' if process_info['status'] == 'Running' else '🔴'
            print(f"\n📊 SYSTEM STATUS")
            print(f"  Status: {status_emoji} {process_info['status']} | CPU: {process_info['cpu']:.1f}% | Memory: {process_info['memory']:.1f}MB | Uptime: {timedelta(seconds=int(process_info['uptime']))}")
        
        # Training Progress
        print(f"\n🎯 TRAINING PROGRESS")
        progress_bar = '█' * int((1 - metrics['epsilon']) * 20) + '░' * int(metrics['epsilon'] * 20)
        print(f"  Episodes: {metrics['episodes']:,} | Experiences: {metrics['experiences']:,}")
        print(f"  Learning: [{progress_bar}] {(1-metrics['epsilon'])*100:.1f}% | Status: {metrics['convergence_status']}")
        print(f"  Exploration Rate: {metrics['epsilon']:.4f} | Losses: Bid={metrics['loss_bid']:.3f} Creative={metrics['loss_creative']:.3f} Channel={metrics['loss_channel']:.3f}")
        
        # Performance Overview
        print(f"\n💰 PERFORMANCE OVERVIEW")
        print(f"  Total Conversions: {metrics['total_conversions']:,} (Immediate: {metrics['immediate_conversions']:,} | Delayed: {metrics['delayed_conversions']:,})")
        print(f"  Total Revenue: ${metrics['total_revenue']:,.2f} | Total Spend: ${metrics['total_spend']:,.2f}")
        print(f"  Overall ROAS: {metrics['overall_roas']:.2f}x | Conversion Rate: {metrics['conversion_rate']:.2f}%")
        
        # ALL Channel Performance with CAC
        print(f"\n📊 CHANNEL PERFORMANCE (All 5 Channels)")
        print(f"  {'Channel':<12} {'ROAS':>8} {'Spend':>10} {'Revenue':>10} {'Conv':>6} {'CAC':>8} {'Status':<20}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*8} {'-'*20}")
        
        for channel in ['organic', 'paid_search', 'social', 'display', 'email']:
            data = metrics['channels'][channel]
            if data['spend'] > 0 or data['revenue'] > 0:
                status = '🟢 Active' if data['roas'] > 1.5 else '🟡 Learning' if data['roas'] > 0 else '🔴 Inactive'
                print(f"  {channel:<12} {data['roas']:>7.2f}x ${data['spend']:>9.2f} ${data['revenue']:>9.2f} {data['conversions']:>6} ${data['cac']:>7.2f} {status:<20}")
            else:
                print(f"  {channel:<12} {'--':>8} {'$0':>10} {'$0':>10} {'0':>6} {'--':>8} 🔴 Not Active")
        
        # Creative Performance
        if metrics['creatives']:
            print(f"\n🎨 TOP PERFORMING CREATIVES")
            sorted_creatives = sorted(metrics['creatives'].items(), 
                                    key=lambda x: x[1]['performance_score'], 
                                    reverse=True)[:5]
            for creative_id, perf in sorted_creatives:
                score_bar = '█' * int(perf['performance_score'] * 10) + '░' * (10 - int(perf['performance_score'] * 10))
                print(f"  Creative {creative_id:>3}: CTR={perf['ctr']:>5.2f}% CVR={perf['cvr']:>5.2f}% [{score_bar}]")
        
        # Bid Intelligence
        if metrics['bid_stats']['mean'] > 0:
            print(f"\n🎯 BIDDING INTELLIGENCE")
            print(f"  Current Range: ${metrics['bid_stats']['min']:.2f} - ${metrics['bid_stats']['max']:.2f}")
            print(f"  Optimal Range: ${metrics['bid_stats']['optimal_range'][0]:.2f} - ${metrics['bid_stats']['optimal_range'][1]:.2f}")
            print(f"  Mean Bid: ${metrics['bid_stats']['mean']:.2f} ± ${metrics['bid_stats']['std']:.2f}")
        
        # Key Insights
        print(f"\n🧠 KEY INSIGHTS & LEARNINGS")
        if insights['top_performing_channel']:
            print(f"  🏆 Best Channel: {insights['top_performing_channel']['name']} ({insights['top_performing_channel']['roas']:.1f}x ROAS, ${insights['top_performing_channel']['cac']:.2f} CAC)")
        if insights['worst_performing_channel']:
            print(f"  ⚠️  Worst Channel: {insights['worst_performing_channel']['name']} ({insights['worst_performing_channel']['roas']:.1f}x ROAS)")
        if insights['best_creative']:
            print(f"  🎨 Best Creative: #{insights['best_creative']['id']} (CTR: {insights['best_creative']['ctr']:.1f}%, CVR: {insights['best_creative']['cvr']:.1f}%)")
        print(f"  📈 CAC Trend: {insights['cac_trend'].upper()} | Volume Trend: {insights['volume_trend'].upper()}")
        
        # What the agent has learned
        if insights['key_learnings']:
            print(f"\n  Agent Has Learned:")
            for learning in insights['key_learnings']:
                print(f"    {learning}")
        
        # Recommendations
        if insights['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            for rec in insights['recommendations']:
                print(f"  {rec}")
        
        # Warnings
        if insights['warnings']:
            print(f"\n⚠️  WARNINGS")
            for warning in insights['warnings']:
                print(f"  {warning}")
        
        # Learned Patterns
        if metrics['learned_patterns']:
            print(f"\n🔬 LEARNING MILESTONES")
            for pattern in metrics['learned_patterns']:
                print(f"  ✓ {pattern}")
        
        print("\n" + "=" * 100)
        print("Updates every 5 seconds | Press Ctrl+C to exit")
    
    def run(self):
        """Main monitoring loop"""
        print("Starting FORTIFIED GAELP Intelligent Monitor...")
        print("Looking for training process...")
        
        try:
            while True:
                process_info = self.get_process_info()
                metrics = self.parse_training_output()
                insights = self.calculate_insights(metrics)
                
                self.display_dashboard(metrics, process_info, insights)
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user.")
            # Save final insights
            self.save_insights(metrics, insights)
    
    def save_insights(self, metrics, insights):
        """Save insights to file for later analysis"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'insights': insights,
            'trends': {
                'cac_history': {k: list(v) for k, v in self.cac_history.items()},
                'volume_history': list(self.volume_history),
                'revenue_history': list(self.revenue_history)
            }
        }
        
        with open('training_insights.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Insights saved to training_insights.json")

def main():
    monitor = InsightfulMonitor()
    monitor.run()

if __name__ == "__main__":
    main()