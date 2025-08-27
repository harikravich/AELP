#!/usr/bin/env python3
"""
GAELP Console Monitor - Real-time training visualization
Shows live metrics, charts, and progress in terminal
"""

import asyncio
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import threading

# Rich library for beautiful console output
try:
    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.text import Text
    from rich.align import Align
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for beautiful console output...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'rich'])
    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.text import Text
    from rich.align import Align
    from rich import box

# Import GAELP components
from training_orchestrator import TrainingOrchestrator
from training_orchestrator.config import TrainingOrchestratorConfig, DEVELOPMENT_CONFIG

class GAELPConsoleMonitor:
    """Real-time console monitoring for GAELP training"""
    
    def __init__(self):
        self.console = Console()
        self.training_data = {
            "current_phase": "Initializing",
            "episode": 0,
            "total_episodes": 0,
            "current_roas": 0.0,
            "phase_average_roas": 0.0,
            "total_spend": 0.0,
            "total_revenue": 0.0,
            "agent_exploration": 0.1,
            "safety_violations": 0,
            "graduation_progress": 0.0,
            "recent_performance": [],
            "phase_history": [],
            "current_campaign": {},
            "live_metrics": {}
        }
        
        self.layout = Layout()
        self.setup_layout()
        
    def setup_layout(self):
        """Create the console layout"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split_column(
            Layout(name="status", size=8),
            Layout(name="performance"),
        )
        
        self.layout["right"].split_column(
            Layout(name="campaign", size=10),
            Layout(name="safety"),
        )

    def create_header(self) -> Panel:
        """Create header panel"""
        title = Text("🎯 GAELP - Ad Campaign Learning Platform", style="bold cyan")
        subtitle = Text(f"Live Training Monitor • {datetime.now().strftime('%H:%M:%S')}", style="dim")
        header_text = Text()
        header_text.append(title)
        header_text.append("\n")
        header_text.append(subtitle)
        return Panel(Align.center(header_text), box=box.DOUBLE)

    def create_status_panel(self) -> Panel:
        """Create status panel"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="white", width=20)
        
        # Phase and episode info
        table.add_row("Phase", f"[bold]{self.training_data['current_phase']}[/bold]")
        table.add_row("Episode", f"{self.training_data['episode']}/{self.training_data['total_episodes']}")
        
        # Performance metrics
        table.add_row("Current ROAS", f"[green]{self.training_data['current_roas']:.2f}x[/green]")
        table.add_row("Phase Avg ROAS", f"[yellow]{self.training_data['phase_average_roas']:.2f}x[/yellow]")
        
        # Financial metrics
        table.add_row("Total Spend", f"[red]${self.training_data['total_spend']:.2f}[/red]")
        table.add_row("Total Revenue", f"[green]${self.training_data['total_revenue']:.2f}[/green]")
        
        # Agent state
        table.add_row("Exploration", f"{self.training_data['agent_exploration']:.3f}")
        
        return Panel(table, title="📊 Current Status", box=box.ROUNDED)

    def create_performance_panel(self) -> Panel:
        """Create performance visualization"""
        if not self.training_data['recent_performance']:
            return Panel("No performance data yet...", title="📈 Performance Trend")
            
        # Create ASCII chart
        values = self.training_data['recent_performance'][-20:]  # Last 20 episodes
        if len(values) < 2:
            chart_text = "Collecting data..."
        else:
            # Simple ASCII bar chart
            max_val = max(values) if values else 1
            min_val = min(values) if values else 0
            chart_lines = []
            
            # Create 10 rows for the chart
            for row in range(10, 0, -1):
                line = ""
                threshold = min_val + (max_val - min_val) * (row / 10)
                for val in values:
                    if val >= threshold:
                        line += "█"
                    else:
                        line += " "
                chart_lines.append(f"{threshold:4.1f}│{line}")
            
            chart_text = "\n".join(chart_lines)
            chart_text += f"\n     └{'─' * len(values)}"
            chart_text += f"\n      Recent Episodes (ROAS)"
        
        return Panel(chart_text, title="📈 Performance Trend", box=box.ROUNDED)

    def create_campaign_panel(self) -> Panel:
        """Create current campaign details"""
        if not self.training_data['current_campaign']:
            return Panel("No active campaign...", title="🎯 Current Campaign")
            
        campaign = self.training_data['current_campaign']
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Property", style="cyan", width=12)
        table.add_column("Value", style="white", width=15)
        
        table.add_row("Creative", campaign.get('creative_type', 'N/A'))
        table.add_row("Audience", campaign.get('target_audience', 'N/A'))
        table.add_row("Budget", f"${campaign.get('budget', 0):.2f}")
        table.add_row("Bid Strategy", campaign.get('bid_strategy', 'N/A'))
        
        # Add performance if available
        if 'performance' in campaign:
            perf = campaign['performance']
            table.add_row("", "")  # Separator
            table.add_row("Impressions", f"{perf.get('impressions', 0):,}")
            table.add_row("Clicks", f"{perf.get('clicks', 0):,}")
            table.add_row("Conversions", f"{perf.get('conversions', 0)}")
            table.add_row("CTR", f"{perf.get('ctr', 0)*100:.2f}%")
        
        return Panel(table, title="🎯 Current Campaign", box=box.ROUNDED)

    def create_safety_panel(self) -> Panel:
        """Create safety monitoring panel"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Check", style="cyan", width=15)
        table.add_column("Status", width=12)
        
        # Safety status indicators
        violations = self.training_data['safety_violations']
        if violations == 0:
            safety_status = "[green]✓ SAFE[/green]"
        elif violations < 3:
            safety_status = "[yellow]⚠ WARNING[/yellow]"
        else:
            safety_status = "[red]✗ VIOLATION[/red]"
            
        table.add_row("Safety Status", safety_status)
        table.add_row("Violations", f"[red]{violations}[/red]")
        
        # Budget status
        spend_ratio = self.training_data['total_spend'] / 1000  # Assume $1000 limit
        if spend_ratio < 0.8:
            budget_status = "[green]✓ GOOD[/green]"
        elif spend_ratio < 0.95:
            budget_status = "[yellow]⚠ HIGH[/yellow]"
        else:
            budget_status = "[red]✗ LIMIT[/red]"
            
        table.add_row("Budget", budget_status)
        table.add_row("Spend Ratio", f"{spend_ratio*100:.1f}%")
        
        # Graduation progress
        progress = self.training_data['graduation_progress']
        if progress >= 1.0:
            grad_status = "[green]✓ READY[/green]"
        elif progress >= 0.7:
            grad_status = "[yellow]◗ CLOSE[/yellow]"
        else:
            grad_status = "[blue]◯ LEARNING[/blue]"
            
        table.add_row("Graduation", grad_status)
        table.add_row("Progress", f"{progress*100:.0f}%")
        
        return Panel(table, title="🛡️ Safety Monitor", box=box.ROUNDED)

    def create_footer(self) -> Panel:
        """Create footer with controls"""
        phase_history = " → ".join(self.training_data['phase_history'])
        if not phase_history:
            phase_history = "Starting up..."
            
        footer_text = Text()
        footer_text.append("Phase Progress: ", style="dim")
        footer_text.append(phase_history, style="bold")
        footer_text.append(" • Press Ctrl+C to stop", style="dim")
        
        return Panel(Align.center(footer_text), box=box.ROUNDED)

    def update_display(self):
        """Update all display panels"""
        self.layout["header"].update(self.create_header())
        self.layout["status"].update(self.create_status_panel())
        self.layout["performance"].update(self.create_performance_panel())
        self.layout["campaign"].update(self.create_campaign_panel())
        self.layout["safety"].update(self.create_safety_panel())
        self.layout["footer"].update(self.create_footer())

    async def run_with_monitoring(self):
        """Run GAELP training with live monitoring"""
        
        # Create mock agent for demo
        agent = self.create_mock_agent()
        
        with Live(self.layout, refresh_per_second=4) as live:
            try:
                # Phase 1: Simulation
                await self.run_phase_with_monitoring("Simulation Training", 20, agent, "simulation")
                
                # Phase 2: Historical Validation  
                await self.run_phase_with_monitoring("Historical Validation", 10, agent, "historical")
                
                # Phase 3: Real Testing
                await self.run_phase_with_monitoring("Small Budget Testing", 15, agent, "real_testing")
                
                # Phase 4: Scaled Deployment
                await self.run_phase_with_monitoring("Scaled Deployment", 10, agent, "scaled")
                
                # Final summary
                self.training_data['current_phase'] = "Training Complete!"
                self.update_display()
                
                # Show final results
                self.console.print("\n")
                self.console.print("🎉 [bold green]TRAINING COMPLETE![/bold green]")
                self.console.print(f"Final ROAS: [bold]{self.training_data['current_roas']:.2f}x[/bold]")
                self.console.print(f"Total Revenue: [green]${self.training_data['total_revenue']:.2f}[/green]")
                self.console.print(f"Total Spend: [red]${self.training_data['total_spend']:.2f}[/red]")
                self.console.print("Agent is ready for production deployment! 🚀")
                
                # Keep display for a moment
                await asyncio.sleep(3)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Training stopped by user[/yellow]")

    async def run_phase_with_monitoring(self, phase_name: str, episodes: int, agent, phase_type: str):
        """Run a training phase with live monitoring"""
        self.training_data['current_phase'] = phase_name
        self.training_data['total_episodes'] = episodes
        self.training_data['episode'] = 0
        
        phase_performance = []
        
        for episode in range(1, episodes + 1):
            self.training_data['episode'] = episode
            
            # Simulate episode
            campaign, performance = await self.simulate_episode(agent, phase_type, episode)
            
            # Update training data
            roas = performance['revenue'] / performance['cost'] if performance['cost'] > 0 else 0
            self.training_data['current_roas'] = roas
            self.training_data['current_campaign'] = {**campaign, 'performance': performance}
            self.training_data['total_spend'] += performance['cost']
            self.training_data['total_revenue'] += performance['revenue']
            self.training_data['recent_performance'].append(roas)
            phase_performance.append(roas)
            
            # Update averages
            self.training_data['phase_average_roas'] = sum(phase_performance) / len(phase_performance)
            
            # Update agent state
            await agent.update_policy(roas, performance)
            self.training_data['agent_exploration'] = agent.policy_state['exploration']
            
            # Update graduation progress
            if len(phase_performance) >= 5:
                recent_avg = sum(phase_performance[-5:]) / 5
                if phase_type == "simulation":
                    self.training_data['graduation_progress'] = min(recent_avg / 1.5, 1.0)
                elif phase_type == "real_testing":
                    self.training_data['graduation_progress'] = min(recent_avg / 2.0, 1.0)
                else:
                    self.training_data['graduation_progress'] = 1.0
            
            # Update display
            self.update_display()
            
            # Realistic delay
            await asyncio.sleep(0.5)
        
        # Mark phase complete
        self.training_data['phase_history'].append(f"{phase_name} ({self.training_data['phase_average_roas']:.1f}x)")

    def create_mock_agent(self):
        """Create a mock agent for demonstration"""
        class MockAgent:
            def __init__(self):
                self.policy_state = {"exploration": 0.1}
                self.performance_history = []
                
            async def select_action(self, context):
                import random
                return {
                    "creative_type": random.choice(["image", "video", "carousel"]),
                    "target_audience": random.choice(["young_adults", "professionals", "families"]),
                    "budget": random.uniform(10, 50),
                    "bid_strategy": random.choice(["cpc", "cpm", "cpa"])
                }
                
            async def update_policy(self, reward, performance):
                self.performance_history.append(reward)
                if len(self.performance_history) > 10:
                    avg_recent = sum(self.performance_history[-10:]) / 10
                    if avg_recent > 2.0:
                        self.policy_state["exploration"] *= 0.95
                    else:
                        self.policy_state["exploration"] *= 1.05
                        
        return MockAgent()

    async def simulate_episode(self, agent, phase_type, episode):
        """Simulate a training episode"""
        import random
        
        campaign = await agent.select_action({"phase": phase_type, "episode": episode})
        
        # Simulate performance based on phase and agent learning
        base_roas = 1.0
        if phase_type == "simulation":
            base_roas = 2.0 + (episode * 0.1)  # Learning in simulation
        elif phase_type == "historical":
            base_roas = 3.0 + random.uniform(-1, 1)
        elif phase_type == "real_testing":
            base_roas = 3.5 + random.uniform(-1, 2)
        else:  # scaled
            base_roas = 4.0 + random.uniform(-1, 3)
            
        # Add some randomness
        roas = base_roas * random.uniform(0.3, 2.0)
        cost = campaign['budget']
        revenue = cost * roas
        
        performance = {
            "cost": cost,
            "revenue": revenue,
            "impressions": random.randint(1000, 10000),
            "clicks": random.randint(50, 500),
            "conversions": random.randint(2, 25),
            "ctr": random.uniform(0.01, 0.05)
        }
        
        return campaign, performance

async def main():
    """Main function"""
    monitor = GAELPConsoleMonitor()
    
    # Clear screen and show intro
    os.system('clear' if os.name == 'posix' else 'cls')
    
    console = Console()
    console.print("🎯 [bold cyan]GAELP Ad Campaign Learning Platform[/bold cyan]")
    console.print("Real-time Training Monitor")
    console.print("\nStarting live training demonstration...")
    console.print("Press Ctrl+C at any time to stop\n")
    
    await asyncio.sleep(2)
    
    await monitor.run_with_monitoring()

if __name__ == "__main__":
    asyncio.run(main())