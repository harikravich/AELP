#!/usr/bin/env python3
"""
GAELP Performance CLI Tool
Quick command-line performance analysis and reporting
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import plotly.graph_objects as go
import plotly.express as px

console = Console()

class PerformanceCLI:
    """Command-line performance analysis tool"""
    
    def __init__(self):
        self.data_path = Path("/home/hariravichandran/AELP")
        self.reports_path = Path("/home/hariravichandran/AELP/performance_reports")
        self.reports_path.mkdir(exist_ok=True)
    
    def load_data(self) -> Dict[str, Any]:
        """Load all available performance data"""
        data = {}
        
        try:
            # Load learning history
            learning_history_path = self.data_path / "learning_history.json"
            if learning_history_path.exists():
                with open(learning_history_path, 'r') as f:
                    data['learning_history'] = json.load(f)
            
            # Load RL learnings
            rl_analysis_path = self.data_path / "rl_learnings_analysis.json"
            if rl_analysis_path.exists():
                with open(rl_analysis_path, 'r') as f:
                    data['rl_analysis'] = json.load(f)
            
            # Load real data
            real_data_path = self.data_path / "data" / "aggregated_data.csv"
            if real_data_path.exists():
                data['real_data'] = pd.read_csv(real_data_path)
            
            # Load benchmarks
            metadata_path = self.data_path / "data" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data['benchmarks'] = json.load(f).get('benchmarks', {})
                    
        except Exception as e:
            console.print(f"[red]Error loading data: {str(e)}[/red]")
            
        return data
    
    def analyze_learning_progress(self, data: Dict[str, Any]):
        """Analyze and display learning progress"""
        if 'learning_history' not in data or 'campaigns_created' not in data['learning_history']:
            console.print("[red]No learning data available[/red]")
            return
        
        campaigns = data['learning_history']['campaigns_created']
        df = pd.DataFrame(campaigns)
        
        if df.empty:
            console.print("[red]No campaign data available[/red]")
            return
        
        # Calculate summary statistics
        total_campaigns = len(df)
        avg_roas = df['actual_roas'].mean()
        best_roas = df['actual_roas'].max()
        worst_roas = df['actual_roas'].min()
        std_roas = df['actual_roas'].std()
        
        # Calculate improvement trend
        window_size = max(5, len(df) // 10)
        df['roas_rolling'] = df['actual_roas'].rolling(window=window_size, min_periods=1).mean()
        improvement = df['roas_rolling'].iloc[-1] - df['roas_rolling'].iloc[0] if len(df) > 1 else 0
        
        # Create summary table
        table = Table(title="Learning Progress Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Campaigns", str(total_campaigns))
        table.add_row("Average ROAS", f"{avg_roas:.3f}x")
        table.add_row("Best ROAS", f"{best_roas:.3f}x")
        table.add_row("Worst ROAS", f"{worst_roas:.3f}x")
        table.add_row("ROAS Std Dev", f"{std_roas:.3f}")
        table.add_row("Improvement Trend", f"{improvement:+.3f}x")
        
        console.print(table)
        
        # Learning efficiency analysis
        if len(df) > 10:
            recent_performance = df['actual_roas'].tail(10).mean()
            early_performance = df['actual_roas'].head(10).mean()
            learning_efficiency = (recent_performance - early_performance) / early_performance * 100
            
            console.print(f"\n[bold blue]Learning Efficiency:[/bold blue]")
            console.print(f"Early Performance (first 10): {early_performance:.3f}x ROAS")
            console.print(f"Recent Performance (last 10): {recent_performance:.3f}x ROAS")
            console.print(f"Improvement: {learning_efficiency:+.1f}%")
    
    def analyze_strategies(self, data: Dict[str, Any]):
        """Analyze discovered strategies"""
        if 'rl_analysis' not in data or 'strategies' not in data['rl_analysis']:
            console.print("[red]No strategy data available[/red]")
            return
        
        strategies = data['rl_analysis']['strategies']
        
        # Create strategy table
        table = Table(title="Discovered Strategies Performance")
        table.add_column("Strategy", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("ROAS", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Tests", style="blue")
        
        # Sort by performance
        strategies_sorted = sorted(strategies, key=lambda x: x['performance'], reverse=True)
        
        for strategy in strategies_sorted:
            evidence = strategy.get('evidence', [])
            test_count = "Unknown"
            for e in evidence:
                if "Tested" in e and "times" in e:
                    test_count = e.split()[1]
                    break
            
            table.add_row(
                strategy['name'].replace('_', ' ').title(),
                strategy['description'][:50] + "..." if len(strategy['description']) > 50 else strategy['description'],
                f"{strategy['performance']:.2f}x",
                f"{strategy['confidence']:.1f}",
                test_count
            )
        
        console.print(table)
        
        # Best strategy highlight
        best_strategy = strategies_sorted[0]
        panel = Panel(
            f"[bold green]{best_strategy['name'].replace('_', ' ').title()}[/bold green]\n\n"
            f"{best_strategy['description']}\n\n"
            f"[bold]Performance:[/bold] {best_strategy['performance']:.2f}x ROAS\n"
            f"[bold]Confidence:[/bold] {best_strategy['confidence']:.1f}",
            title="🏆 Best Performing Strategy",
            border_style="green"
        )
        console.print("\n", panel)
    
    def compare_with_benchmarks(self, data: Dict[str, Any]):
        """Compare agent performance with industry benchmarks"""
        if 'benchmarks' not in data or 'learning_history' not in data:
            console.print("[red]No comparison data available[/red]")
            return
        
        campaigns = data.get('learning_history', {}).get('campaigns_created', [])
        if not campaigns:
            console.print("[red]No campaign data for comparison[/red]")
            return
        
        df = pd.DataFrame(campaigns)
        agent_performance = {
            'avg_roas': df['actual_roas'].mean(),
            'avg_ctr': df['actual_ctr'].mean(),
            'total_revenue': df['revenue'].sum(),
            'total_cost': df['cost'].sum()
        }
        
        # Create comparison table
        table = Table(title="Agent vs Industry Benchmarks")
        table.add_column("Industry", style="cyan")
        table.add_column("Benchmark ROAS", style="yellow")
        table.add_column("Agent ROAS", style="green")
        table.add_column("Performance", style="bold")
        
        benchmarks = data['benchmarks']
        for industry, benchmark in benchmarks.items():
            benchmark_roas = benchmark.get('avg_roas', 0)
            agent_roas = agent_performance['avg_roas']
            
            if agent_roas > benchmark_roas:
                performance = f"[green]+{((agent_roas/benchmark_roas - 1) * 100):.1f}%[/green]"
            else:
                performance = f"[red]{((agent_roas/benchmark_roas - 1) * 100):.1f}%[/red]"
            
            table.add_row(
                industry.title(),
                f"{benchmark_roas:.2f}x",
                f"{agent_roas:.2f}x",
                performance
            )
        
        console.print(table)
    
    def generate_quick_report(self, data: Dict[str, Any], save_to_file: bool = False):
        """Generate a quick performance report"""
        report_lines = []
        report_lines.append("GAELP Agent Performance Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Learning progress
        if 'learning_history' in data and 'campaigns_created' in data['learning_history']:
            campaigns = data['learning_history']['campaigns_created']
            df = pd.DataFrame(campaigns)
            
            if not df.empty:
                report_lines.append("LEARNING PROGRESS")
                report_lines.append("-" * 20)
                report_lines.append(f"Total Campaigns: {len(df)}")
                report_lines.append(f"Average ROAS: {df['actual_roas'].mean():.3f}x")
                report_lines.append(f"Best ROAS: {df['actual_roas'].max():.3f}x")
                report_lines.append(f"Total Revenue: ${df['revenue'].sum():.2f}")
                report_lines.append(f"Total Cost: ${df['cost'].sum():.2f}")
                report_lines.append(f"Net Profit: ${(df['revenue'].sum() - df['cost'].sum()):.2f}")
                report_lines.append("")
        
        # Strategy analysis
        if 'rl_analysis' in data and 'strategies' in data['rl_analysis']:
            strategies = data['rl_analysis']['strategies']
            if strategies:
                best_strategy = max(strategies, key=lambda x: x['performance'])
                report_lines.append("BEST STRATEGY DISCOVERED")
                report_lines.append("-" * 25)
                report_lines.append(f"Strategy: {best_strategy['name'].replace('_', ' ').title()}")
                report_lines.append(f"Performance: {best_strategy['performance']:.2f}x ROAS")
                report_lines.append(f"Confidence: {best_strategy['confidence']:.1f}")
                report_lines.append(f"Description: {best_strategy['description']}")
                report_lines.append("")
        
        # Key insights
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-" * 12)
        
        if 'learning_history' in data and 'campaigns_created' in data['learning_history']:
            campaigns = data['learning_history']['campaigns_created']
            df = pd.DataFrame(campaigns)
            
            if not df.empty:
                # Learning trend
                if len(df) > 10:
                    recent_avg = df['actual_roas'].tail(10).mean()
                    early_avg = df['actual_roas'].head(10).mean()
                    if recent_avg > early_avg:
                        report_lines.append("✓ Agent is showing positive learning trend")
                    else:
                        report_lines.append("⚠ Agent learning may have plateaued")
                
                # Performance consistency
                roas_std = df['actual_roas'].std()
                if roas_std < 0.5:
                    report_lines.append("✓ Performance is consistent")
                else:
                    report_lines.append("⚠ Performance shows high variation")
                
                # ROAS threshold analysis
                high_performers = df[df['actual_roas'] > 2.0]
                if len(high_performers) > len(df) * 0.3:
                    report_lines.append("✓ Strong ROAS performance (>2x)")
                else:
                    report_lines.append("⚠ Consider optimizing for higher ROAS")
        
        # Display report
        report_text = "\n".join(report_lines)
        console.print(Panel(report_text, title="📊 Performance Report", border_style="blue"))
        
        # Save to file if requested
        if save_to_file:
            report_file = self.reports_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report_text)
            console.print(f"\n[green]Report saved to: {report_file}[/green]")
    
    def save_performance_charts(self, data: Dict[str, Any]):
        """Generate and save performance visualization charts"""
        if 'learning_history' not in data or 'campaigns_created' not in data['learning_history']:
            console.print("[red]No data available for charts[/red]")
            return
        
        campaigns = data['learning_history']['campaigns_created']
        df = pd.DataFrame(campaigns)
        
        if df.empty:
            console.print("[red]No campaign data available[/red]")
            return
        
        console.print("📊 Generating performance charts...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GAELP Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # ROAS over time
        axes[0, 0].plot(df.index, df['actual_roas'], 'o-', alpha=0.7, label='Individual Campaigns')
        window_size = max(5, len(df) // 10)
        rolling_roas = df['actual_roas'].rolling(window=window_size, min_periods=1).mean()
        axes[0, 0].plot(df.index, rolling_roas, 'r-', linewidth=2, label='Trend')
        axes[0, 0].set_title('ROAS Performance Over Time')
        axes[0, 0].set_xlabel('Campaign Number')
        axes[0, 0].set_ylabel('ROAS')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROAS distribution
        axes[0, 1].hist(df['actual_roas'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(df['actual_roas'].mean(), color='red', linestyle='--', label=f'Mean: {df["actual_roas"].mean():.2f}')
        axes[0, 1].set_title('ROAS Distribution')
        axes[0, 1].set_xlabel('ROAS')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Revenue vs Cost
        axes[1, 0].scatter(df['cost'], df['revenue'], alpha=0.6, c=df['actual_roas'], cmap='viridis')
        axes[1, 0].plot([df['cost'].min(), df['cost'].max()], [df['cost'].min(), df['cost'].max()], 'r--', label='Break-even')
        axes[1, 0].set_title('Revenue vs Cost (colored by ROAS)')
        axes[1, 0].set_xlabel('Cost ($)')
        axes[1, 0].set_ylabel('Revenue ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Conversion performance
        if 'actual_conversions' in df.columns:
            axes[1, 1].plot(df.index, df['actual_conversions'], 'go-', alpha=0.7)
            axes[1, 1].set_title('Conversions Over Time')
            axes[1, 1].set_xlabel('Campaign Number')
            axes[1, 1].set_ylabel('Conversions')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        chart_file = self.reports_path / f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]Charts saved to: {chart_file}[/green]")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="GAELP Performance Analysis CLI")
    parser.add_argument('--action', choices=['summary', 'strategies', 'benchmarks', 'report', 'charts'], 
                       default='summary', help='Analysis action to perform')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    cli = PerformanceCLI()
    
    console.print("[bold blue]🎯 GAELP Performance Analysis CLI[/bold blue]")
    console.print("=" * 50)
    
    # Load data
    with console.status("[bold green]Loading performance data...", spinner="dots"):
        data = cli.load_data()
    
    # Perform requested analysis
    if args.action == 'summary':
        cli.analyze_learning_progress(data)
    elif args.action == 'strategies':
        cli.analyze_strategies(data)
    elif args.action == 'benchmarks':
        cli.compare_with_benchmarks(data)
    elif args.action == 'report':
        cli.generate_quick_report(data, save_to_file=args.save)
    elif args.action == 'charts':
        cli.save_performance_charts(data)
    
    console.print(f"\n[dim]Analysis completed at {datetime.now().strftime('%H:%M:%S')}[/dim]")

if __name__ == "__main__":
    main()