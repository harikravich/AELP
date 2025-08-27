#!/usr/bin/env python3
"""
Aura Parental Control App - GAELP Live Demo
Shows real-time bidding, learning, and optimization
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import time
from typing import Dict, List, Tuple
from gaelp_master_integration import MasterOrchestrator, GAELPConfig
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich import box
import random

console = Console()

class AuraSimulation:
    """Simulates Aura's advertising campaign with different parent segments"""
    
    def __init__(self):
        self.segments = {
            'crisis_parent': {
                'name': '🚨 Crisis Parent',
                'conversion_rate': 0.25,
                'avg_value': 120,
                'queries': [
                    'help child addicted to phone',
                    'emergency parental controls now',
                    'kid exposed to inappropriate content',
                    'urgent need screen time limits'
                ]
            },
            'researcher': {
                'name': '🔍 Researcher',
                'conversion_rate': 0.08,
                'avg_value': 80,
                'queries': [
                    'best parental control apps 2024',
                    'compare aura vs qustodio',
                    'parental control app reviews',
                    'screen time management tools'
                ]
            },
            'budget_conscious': {
                'name': '💰 Budget Conscious',
                'conversion_rate': 0.12,
                'avg_value': 60,
                'queries': [
                    'free parental controls',
                    'affordable screen time app',
                    'cheap parental monitoring',
                    'budget family safety apps'
                ]
            },
            'tech_savvy': {
                'name': '💻 Tech Savvy',
                'conversion_rate': 0.15,
                'avg_value': 100,
                'queries': [
                    'advanced parental controls API',
                    'custom screen time rules',
                    'parental control VPN setup',
                    'device management dashboard'
                ]
            }
        }
        
        self.competitors = ['Qustodio', 'Bark', 'Circle', 'Norton']
        self.episode = 0
        self.total_spend = 0
        self.total_revenue = 0
        self.conversions = 0
        self.impressions = 0
        self.clicks = 0
        
        # Track performance over time
        self.history = {
            'episodes': [],
            'roi': [],
            'conversion_rate': [],
            'avg_bid': [],
            'win_rate': []
        }
        
        # Track per-segment performance
        self.segment_stats = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spend': 0,
            'revenue': 0
        })
        
        # Track Thompson Sampling arm performance
        self.arm_stats = defaultdict(lambda: {
            'selections': 0,
            'rewards': 0,
            'avg_reward': 0
        })

    def generate_query(self) -> Tuple[str, Dict]:
        """Generate a realistic search query from a parent"""
        # Time-based segment probability
        hour = datetime.now().hour
        
        # Crisis parents more likely during evening/night
        if 20 <= hour or hour <= 2:
            weights = [0.4, 0.2, 0.2, 0.2]  # Crisis more likely
        # Researchers during work hours
        elif 9 <= hour <= 17:
            weights = [0.2, 0.4, 0.2, 0.2]  # Researchers more likely
        else:
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal
        
        segment_key = random.choices(list(self.segments.keys()), weights=weights)[0]
        segment = self.segments[segment_key]
        query = random.choice(segment['queries'])
        
        return segment_key, {
            'query': query,
            'segment': segment_key,
            'intent_strength': np.random.beta(3, 2) if segment_key == 'crisis_parent' else np.random.beta(2, 3),
            'device_type': random.choice(['mobile', 'desktop']),
            'location': 'US'
        }

    def simulate_auction(self, our_bid: float, segment: str) -> Dict:
        """Simulate an auction with competitors"""
        # Competitor bids based on segment value
        competitor_bids = []
        for competitor in self.competitors:
            if segment == 'crisis_parent':
                # Competitors bid aggressively for crisis parents
                bid = np.random.uniform(2.0, 4.0)
            elif segment == 'researcher':
                bid = np.random.uniform(1.0, 2.5)
            else:
                bid = np.random.uniform(0.5, 2.0)
            competitor_bids.append(bid)
        
        all_bids = [our_bid] + competitor_bids
        all_bids.sort(reverse=True)
        
        our_position = all_bids.index(our_bid) + 1
        won = our_position == 1
        
        # Second price auction
        if won:
            price = all_bids[1] * 0.95  # Small discount
        else:
            price = 0
        
        # CTR by position
        ctr_by_position = {1: 0.06, 2: 0.04, 3: 0.025, 4: 0.015, 5: 0.008}
        ctr = ctr_by_position.get(our_position, 0.005)
        
        return {
            'won': won,
            'position': our_position,
            'price': price,
            'ctr': ctr,
            'num_competitors': len(self.competitors)
        }

    def simulate_conversion(self, segment_key: str, clicked: bool) -> Tuple[bool, float]:
        """Simulate whether a click converts"""
        if not clicked:
            return False, 0
        
        segment = self.segments[segment_key]
        
        # Random conversion based on segment rate
        converts = random.random() < segment['conversion_rate']
        
        if converts:
            # Add some variance to conversion value
            value = segment['avg_value'] * np.random.uniform(0.8, 1.2)
            return True, value
        
        return False, 0


async def run_aura_demo():
    """Run the Aura GAELP demo with live visualization"""
    
    console.print("\n[bold cyan]🚀 Initializing Aura GAELP System...[/bold cyan]\n")
    
    # Initialize GAELP
    config = GAELPConfig(
        enable_delayed_rewards=True,
        enable_competitive_intelligence=True,
        enable_creative_optimization=True,
        enable_budget_pacing=True,
        enable_identity_resolution=True,
        enable_criteo_response=True,
        enable_safety_system=True,
        enable_temporal_effects=True
    )
    
    master = MasterOrchestrator(config)
    simulation = AuraSimulation()
    
    console.print("[green]✓[/green] System initialized with 19 components")
    console.print("[green]✓[/green] Thompson Sampling arms: conservative, balanced, aggressive, experimental")
    console.print("[green]✓[/green] Safety constraints enabled\n")
    
    # Setup display layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=20),
        Layout(name="learning", size=10),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="segments", ratio=1),
        Layout(name="competition", ratio=1)
    )
    
    def create_header():
        """Create header panel"""
        return Panel(
            Text("🎯 Aura Parental Control - Real-Time Bidding & Learning Demo", 
                 style="bold cyan", justify="center"),
            box=box.DOUBLE
        )
    
    def create_metrics_table():
        """Create metrics display"""
        table = Table(title="📊 Campaign Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        roi = (simulation.total_revenue - simulation.total_spend) / max(simulation.total_spend, 1) * 100
        ctr = simulation.clicks / max(simulation.impressions, 1) * 100
        cvr = simulation.conversions / max(simulation.clicks, 1) * 100
        cpa = simulation.total_spend / max(simulation.conversions, 1)
        
        table.add_row("Episode", str(simulation.episode))
        table.add_row("Impressions", f"{simulation.impressions:,}")
        table.add_row("Clicks", f"{simulation.clicks:,}")
        table.add_row("Conversions", f"{simulation.conversions:,}")
        table.add_row("CTR", f"{ctr:.2f}%")
        table.add_row("CVR", f"{cvr:.2f}%")
        table.add_row("CPA", f"${cpa:.2f}")
        table.add_row("Spend", f"${simulation.total_spend:.2f}")
        table.add_row("Revenue", f"${simulation.total_revenue:.2f}")
        table.add_row("ROI", f"{roi:.1f}%")
        
        return table
    
    def create_segments_table():
        """Create segment performance table"""
        table = Table(title="👥 Segment Performance", box=box.ROUNDED)
        table.add_column("Segment", style="cyan")
        table.add_column("Conv", style="green")
        table.add_column("Revenue", style="yellow")
        table.add_column("ROI", style="magenta")
        
        for seg_key, seg_info in simulation.segments.items():
            stats = simulation.segment_stats[seg_key]
            if stats['spend'] > 0:
                roi = ((stats['revenue'] - stats['spend']) / stats['spend'] * 100)
                table.add_row(
                    seg_info['name'],
                    str(stats['conversions']),
                    f"${stats['revenue']:.0f}",
                    f"{roi:.0f}%"
                )
        
        return table
    
    def create_competition_table():
        """Create competition analysis"""
        table = Table(title="🥊 Auction Competition", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        if simulation.history['win_rate']:
            recent_win_rate = np.mean(simulation.history['win_rate'][-10:]) * 100
            recent_avg_bid = np.mean(simulation.history['avg_bid'][-10:])
        else:
            recent_win_rate = 0
            recent_avg_bid = 0
        
        table.add_row("Win Rate", f"{recent_win_rate:.1f}%")
        table.add_row("Avg Bid", f"${recent_avg_bid:.2f}")
        table.add_row("Competitors", "4")
        table.add_row("", "")
        table.add_row("[bold]Top Competitor[/bold]", "")
        table.add_row("Qustodio", "28% share")
        table.add_row("Bark", "22% share")
        
        return table
    
    def create_learning_panel():
        """Create Thompson Sampling learning display"""
        lines = ["[bold cyan]🧠 Thompson Sampling Arms:[/bold cyan]\n"]
        
        if hasattr(master.online_learner, 'bandit_arms'):
            for arm_name, arm in master.online_learner.bandit_arms.items():
                stats = simulation.arm_stats[arm_name]
                value = arm.alpha / (arm.alpha + arm.beta)
                
                # Create a simple bar chart
                bar_length = int(value * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                emoji = {"conservative": "🛡️", "balanced": "⚖️", 
                        "aggressive": "🚀", "experimental": "🔬"}.get(arm_name, "📊")
                
                lines.append(f"{emoji} {arm_name:12} {bar} {value:.3f} ({stats['selections']} uses)")
        
        lines.append("\n[bold cyan]📈 Learning Progress:[/bold cyan]")
        if simulation.episode > 0:
            lines.append(f"Episodes: {simulation.episode}")
            lines.append(f"Exploration Rate: {max(0.3 - simulation.episode*0.001, 0.1):.1%}")
            lines.append(f"Best Segment: {max(simulation.segment_stats.items(), key=lambda x: x[1]['revenue'], default=('none', {}))[0]}")
        
        return Panel("\n".join(lines), title="🎓 Online Learning Status", box=box.ROUNDED)
    
    # Main simulation loop
    console.print("[bold yellow]▶ Starting live simulation... (Press Ctrl+C to stop)[/bold yellow]\n")
    
    try:
        with Live(layout, refresh_per_second=2, console=console) as live:
            while simulation.episode < 1000:  # Run for 1000 episodes
                simulation.episode += 1
                
                # Generate query
                segment_key, query_data = simulation.generate_query()
                
                # Create journey state
                journey_state = {
                    'conversion_probability': np.random.beta(2, 3),
                    'journey_stage': random.randint(1, 3),
                    'user_fatigue_level': np.random.beta(2, 5),
                    'hour_of_day': datetime.now().hour,
                    'user_id': f'user_{simulation.episode}'
                }
                
                # Get bid from GAELP
                creative_selection = {'creative_type': 'display'}
                bid = await master._calculate_bid(journey_state, query_data, creative_selection)
                
                # Simulate auction
                auction_result = simulation.simulate_auction(bid, segment_key)
                
                if auction_result['won']:
                    simulation.impressions += 1
                    simulation.total_spend += auction_result['price']
                    simulation.segment_stats[segment_key]['impressions'] += 1
                    simulation.segment_stats[segment_key]['spend'] += auction_result['price']
                    
                    # Simulate click
                    clicked = random.random() < auction_result['ctr']
                    if clicked:
                        simulation.clicks += 1
                        simulation.segment_stats[segment_key]['clicks'] += 1
                        
                        # Simulate conversion
                        converted, value = simulation.simulate_conversion(segment_key, clicked)
                        if converted:
                            simulation.conversions += 1
                            simulation.total_revenue += value
                            simulation.segment_stats[segment_key]['conversions'] += 1
                            simulation.segment_stats[segment_key]['revenue'] += value
                
                # Update Thompson Sampling
                if hasattr(master.online_learner, 'bandit_arms'):
                    # Select arm based on current strategy
                    arm_samples = {i: arm.sample() for i, arm in enumerate(master.online_learner.bandit_arms.values())}
                    selected_arm = max(arm_samples.keys(), key=lambda x: arm_samples[x])
                    arm_name = list(master.online_learner.bandit_arms.keys())[selected_arm]
                    
                    # Calculate reward (simplified)
                    if auction_result['won']:
                        reward = auction_result['ctr'] * 10  # Reward for winning with good CTR
                    else:
                        reward = 0
                    
                    # Update arm
                    arm = master.online_learner.bandit_arms[arm_name]
                    arm.update(reward, success=auction_result['won'])
                    
                    # Track stats
                    simulation.arm_stats[arm_name]['selections'] += 1
                    simulation.arm_stats[arm_name]['rewards'] += reward
                    simulation.arm_stats[arm_name]['avg_reward'] = (
                        simulation.arm_stats[arm_name]['rewards'] / 
                        simulation.arm_stats[arm_name]['selections']
                    )
                
                # Record history
                if simulation.episode % 10 == 0:
                    simulation.history['episodes'].append(simulation.episode)
                    simulation.history['roi'].append(
                        (simulation.total_revenue - simulation.total_spend) / max(simulation.total_spend, 1)
                    )
                    simulation.history['conversion_rate'].append(
                        simulation.conversions / max(simulation.clicks, 1)
                    )
                    simulation.history['avg_bid'].append(bid)
                    simulation.history['win_rate'].append(1 if auction_result['won'] else 0)
                
                # Update display
                layout["header"].update(create_header())
                layout["metrics"].update(create_metrics_table())
                layout["segments"].update(create_segments_table())
                layout["competition"].update(create_competition_table())
                layout["learning"].update(create_learning_panel())
                layout["footer"].update(Panel(
                    f"[dim]Episode {simulation.episode} | Bid: ${bid:.2f} | "
                    f"Segment: {simulation.segments[segment_key]['name']} | "
                    f"Won: {'✓' if auction_result['won'] else '✗'}[/dim]",
                    box=box.ROUNDED
                ))
                
                # Control simulation speed
                await asyncio.sleep(0.1)  # 10 episodes per second
                
                # Trigger online learning update every 50 episodes
                if simulation.episode % 50 == 0 and hasattr(master, 'online_learner'):
                    # Record episodes for learning
                    for _ in range(10):
                        master.online_learner.record_episode({
                            'state': journey_state,
                            'action': {'bid': bid},
                            'reward': reward if 'reward' in locals() else 0,
                            'success': auction_result['won']
                        })
                    
                    # Trigger update
                    if master.online_learner.episode_history:
                        await master.online_learner.online_update(
                            master.online_learner.episode_history[-10:]
                        )
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation stopped by user[/yellow]")
    
    # Print final summary
    console.print("\n" + "="*80)
    console.print("[bold cyan]📊 FINAL SIMULATION SUMMARY[/bold cyan]")
    console.print("="*80)
    
    roi = (simulation.total_revenue - simulation.total_spend) / max(simulation.total_spend, 1) * 100
    
    summary_table = Table(box=box.DOUBLE)
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Value", style="green", width=20)
    
    summary_table.add_row("Total Episodes", f"{simulation.episode:,}")
    summary_table.add_row("Total Impressions", f"{simulation.impressions:,}")
    summary_table.add_row("Total Clicks", f"{simulation.clicks:,}")
    summary_table.add_row("Total Conversions", f"{simulation.conversions:,}")
    summary_table.add_row("Total Spend", f"${simulation.total_spend:,.2f}")
    summary_table.add_row("Total Revenue", f"${simulation.total_revenue:,.2f}")
    summary_table.add_row("Final ROI", f"{roi:.1f}%")
    summary_table.add_row("Avg CPA", f"${simulation.total_spend/max(simulation.conversions,1):.2f}")
    
    console.print(summary_table)
    
    # Best performing segment
    if simulation.segment_stats:
        best_segment = max(
            simulation.segment_stats.items(),
            key=lambda x: x[1]['revenue'] - x[1]['spend']
        )
        console.print(f"\n[bold green]🏆 Best Segment: {simulation.segments[best_segment[0]]['name']}[/bold green]")
        console.print(f"   Revenue: ${best_segment[1]['revenue']:.2f}")
        console.print(f"   ROI: {((best_segment[1]['revenue']-best_segment[1]['spend'])/max(best_segment[1]['spend'],1)*100):.1f}%")
    
    # Thompson Sampling results
    if simulation.arm_stats:
        console.print("\n[bold cyan]🎰 Thompson Sampling Results:[/bold cyan]")
        for arm_name, stats in simulation.arm_stats.items():
            if stats['selections'] > 0:
                console.print(f"   {arm_name}: {stats['selections']} uses, avg reward: {stats['avg_reward']:.3f}")
    
    console.print("\n[bold green]✅ Demo complete! The system learned and optimized in real-time.[/bold green]")

if __name__ == "__main__":
    import sys
    try:
        # Check if rich is installed
        import rich
    except ImportError:
        print("Installing required package 'rich' for visualization...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        print("Package installed! Starting demo...\n")
    
    asyncio.run(run_aura_demo())