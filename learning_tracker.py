#!/usr/bin/env python3
"""
GAELP Learning Tracker
Tracks and visualizes what the agent learns over time
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

class LearningTracker:
    """Tracks agent learnings across training phases"""
    
    def __init__(self):
        self.learning_history = {
            "campaigns_created": [],
            "performance_metrics": [],
            "discovered_strategies": [],
            "failed_experiments": [],
            "optimization_insights": []
        }
        
        self.phase_learnings = {
            "simulation": {
                "episodes": 0,
                "avg_roas": 0.0,
                "best_creative": None,
                "best_audience": None,
                "key_insights": []
            },
            "historical": {
                "episodes": 0,
                "avg_roas": 0.0,
                "validation_accuracy": 0.0,
                "transferable_learnings": []
            },
            "real_testing": {
                "episodes": 0,
                "avg_roas": 0.0,
                "real_world_discoveries": [],
                "safety_violations": 0
            },
            "scaled": {
                "episodes": 0,
                "avg_roas": 0.0,
                "optimization_strategies": [],
                "roi_improvement": 0.0
            }
        }
    
    def record_campaign(self, phase: str, campaign: Dict[str, Any], performance: Dict[str, Any]):
        """Record a campaign and its performance"""
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "campaign_id": campaign.get("id"),
            "creative_type": campaign.get("creative", {}).get("type"),
            "target_audience": campaign.get("target_audience"),
            "budget": campaign.get("budget", 0),
            "actual_roas": performance.get("roas", 0),
            "actual_ctr": performance.get("ctr", 0),
            "actual_conversions": performance.get("conversions", 0),
            "cost": performance.get("cost", 0),
            "revenue": performance.get("revenue", 0)
        }
        
        self.learning_history["campaigns_created"].append(record)
        self.learning_history["performance_metrics"].append(performance)
        
        # Update phase learnings
        phase_data = self.phase_learnings[phase]
        phase_data["episodes"] += 1
        
        # Running average ROAS
        prev_avg = phase_data["avg_roas"]
        n = phase_data["episodes"]
        phase_data["avg_roas"] = (prev_avg * (n-1) + performance.get("roas", 0)) / n
        
        # Track best performing combinations
        if performance.get("roas", 0) > 3.0:
            if not phase_data.get("best_creative") or performance["roas"] > prev_avg:
                phase_data["best_creative"] = campaign.get("creative", {}).get("type")
                phase_data["best_audience"] = campaign.get("target_audience")
    
    def discover_pattern(self, pattern_type: str, description: str, confidence: float):
        """Record a discovered pattern or strategy"""
        
        discovery = {
            "timestamp": datetime.now().isoformat(),
            "type": pattern_type,
            "description": description,
            "confidence": confidence,
            "episodes_to_discover": len(self.learning_history["campaigns_created"])
        }
        
        self.learning_history["discovered_strategies"].append(discovery)
        
        # Add to current phase insights
        current_phase = self.get_current_phase()
        if current_phase in self.phase_learnings:
            self.phase_learnings[current_phase].setdefault("key_insights", []).append(description)
    
    def record_failure(self, campaign_id: str, failure_reason: str, lesson_learned: str):
        """Record failed experiments and lessons learned"""
        
        failure = {
            "timestamp": datetime.now().isoformat(),
            "campaign_id": campaign_id,
            "reason": failure_reason,
            "lesson": lesson_learned,
            "cost_of_failure": 0.0  # Could track actual cost
        }
        
        self.learning_history["failed_experiments"].append(failure)
    
    def analyze_learnings(self) -> Dict[str, Any]:
        """Analyze all learnings to extract insights"""
        
        if not self.learning_history["campaigns_created"]:
            return {"status": "No campaigns to analyze"}
        
        df = pd.DataFrame(self.learning_history["campaigns_created"])
        
        analysis = {
            "total_campaigns": len(df),
            "phases_completed": df["phase"].nunique(),
            "average_roas": df["actual_roas"].mean(),
            "best_campaign": None,
            "worst_campaign": None,
            "creative_performance": {},
            "audience_performance": {},
            "budget_optimization": {},
            "learning_curve": {},
            "key_discoveries": []
        }
        
        # Best and worst campaigns
        if not df.empty:
            best_idx = df["actual_roas"].idxmax()
            worst_idx = df["actual_roas"].idxmin()
            analysis["best_campaign"] = df.loc[best_idx].to_dict()
            analysis["worst_campaign"] = df.loc[worst_idx].to_dict()
        
        # Creative type performance
        for creative in df["creative_type"].unique():
            creative_data = df[df["creative_type"] == creative]
            analysis["creative_performance"][creative] = {
                "avg_roas": creative_data["actual_roas"].mean(),
                "avg_ctr": creative_data["actual_ctr"].mean(),
                "count": len(creative_data)
            }
        
        # Audience performance
        for audience in df["target_audience"].unique():
            if pd.notna(audience):
                audience_data = df[df["target_audience"] == audience]
                analysis["audience_performance"][audience] = {
                    "avg_roas": audience_data["actual_roas"].mean(),
                    "avg_conversions": audience_data["actual_conversions"].mean(),
                    "optimal_budget": audience_data[audience_data["actual_roas"] > 2.5]["budget"].mean()
                }
        
        # Learning curve analysis (ROAS improvement over time)
        for phase in df["phase"].unique():
            phase_data = df[df["phase"] == phase]
            if len(phase_data) > 1:
                # Calculate improvement from first to last campaign
                first_roas = phase_data.iloc[0]["actual_roas"]
                last_roas = phase_data.iloc[-1]["actual_roas"]
                improvement = ((last_roas - first_roas) / first_roas * 100) if first_roas > 0 else 0
                
                analysis["learning_curve"][phase] = {
                    "start_roas": first_roas,
                    "end_roas": last_roas,
                    "improvement_pct": improvement,
                    "episodes": len(phase_data)
                }
        
        # Extract key discoveries
        analysis["key_discoveries"] = self.extract_key_discoveries(df)
        
        return analysis
    
    def extract_key_discoveries(self, df: pd.DataFrame) -> List[str]:
        """Extract key discoveries from the data"""
        
        discoveries = []
        
        # Creative insights
        creative_perf = df.groupby("creative_type")["actual_roas"].mean()
        if len(creative_perf) > 1:
            best_creative = creative_perf.idxmax()
            worst_creative = creative_perf.idxmin()
            if creative_perf[best_creative] > creative_perf[worst_creative] * 1.2:
                discoveries.append(
                    f"📊 {best_creative.title()} ads outperform {worst_creative} by "
                    f"{(creative_perf[best_creative] / creative_perf[worst_creative] - 1) * 100:.0f}%"
                )
        
        # Audience insights
        if "target_audience" in df.columns:
            audience_perf = df.groupby("target_audience")["actual_roas"].mean()
            if len(audience_perf) > 1:
                best_audience = audience_perf.idxmax()
                discoveries.append(f"🎯 {best_audience} audience shows highest engagement")
        
        # Budget insights
        if len(df) > 10:
            high_performers = df[df["actual_roas"] > df["actual_roas"].quantile(0.75)]
            if not high_performers.empty:
                optimal_budget = high_performers["budget"].mean()
                discoveries.append(f"💰 Optimal budget discovered: ${optimal_budget:.2f}/day")
        
        # Learning progression
        early_campaigns = df.head(5)["actual_roas"].mean()
        recent_campaigns = df.tail(5)["actual_roas"].mean()
        if recent_campaigns > early_campaigns * 1.5:
            discoveries.append(
                f"📈 {(recent_campaigns / early_campaigns - 1) * 100:.0f}% ROAS improvement through learning"
            )
        
        # Pattern discoveries from strategy history
        for strategy in self.learning_history["discovered_strategies"][-3:]:
            discoveries.append(f"🔍 {strategy['description']}")
        
        return discoveries
    
    def get_current_phase(self) -> str:
        """Determine current training phase"""
        if not self.learning_history["campaigns_created"]:
            return "simulation"
        
        last_campaign = self.learning_history["campaigns_created"][-1]
        return last_campaign.get("phase", "simulation")
    
    def generate_report(self) -> str:
        """Generate comprehensive learning report"""
        
        analysis = self.analyze_learnings()
        
        report = []
        report.append("=" * 70)
        report.append("GAELP AGENT LEARNING REPORT")
        report.append("=" * 70)
        report.append(f"\n📊 OVERALL STATISTICS")
        report.append(f"Total Campaigns: {analysis['total_campaigns']}")
        report.append(f"Average ROAS: {analysis['average_roas']:.2f}x")
        report.append(f"Phases Completed: {analysis['phases_completed']}")
        
        if analysis.get("best_campaign"):
            report.append(f"\n🏆 BEST CAMPAIGN")
            best = analysis["best_campaign"]
            report.append(f"ID: {best.get('campaign_id')}")
            report.append(f"ROAS: {best.get('actual_roas', 0):.2f}x")
            report.append(f"Creative: {best.get('creative_type')}")
            report.append(f"Audience: {best.get('target_audience')}")
        
        report.append(f"\n📈 CREATIVE PERFORMANCE")
        for creative, perf in analysis.get("creative_performance", {}).items():
            report.append(f"{creative}: {perf['avg_roas']:.2f}x avg ROAS ({perf['count']} campaigns)")
        
        report.append(f"\n🎯 AUDIENCE INSIGHTS")
        for audience, perf in analysis.get("audience_performance", {}).items():
            if pd.notna(perf['optimal_budget']):
                report.append(f"{audience}: {perf['avg_roas']:.2f}x ROAS, ${perf['optimal_budget']:.2f} optimal budget")
        
        report.append(f"\n📈 LEARNING PROGRESSION")
        for phase, curve in analysis.get("learning_curve", {}).items():
            report.append(f"{phase}: {curve['start_roas']:.2f}x → {curve['end_roas']:.2f}x "
                         f"({curve['improvement_pct']:.0f}% improvement)")
        
        report.append(f"\n🔍 KEY DISCOVERIES")
        for discovery in analysis.get("key_discoveries", []):
            report.append(f"• {discovery}")
        
        report.append(f"\n💡 FAILED EXPERIMENTS & LESSONS")
        for failure in self.learning_history["failed_experiments"][-3:]:
            report.append(f"• {failure['reason']}: {failure['lesson']}")
        
        return "\n".join(report)
    
    def visualize_learning(self):
        """Create visualization of learning progress"""
        
        if not self.learning_history["campaigns_created"]:
            console.print("[yellow]No data to visualize yet[/yellow]")
            return
        
        df = pd.DataFrame(self.learning_history["campaigns_created"])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("GAELP Agent Learning Progress", fontsize=16)
        
        # 1. ROAS over time
        ax1 = axes[0, 0]
        for phase in df["phase"].unique():
            phase_data = df[df["phase"] == phase]
            ax1.plot(range(len(phase_data)), phase_data["actual_roas"], marker='o', label=phase)
        ax1.set_xlabel("Campaign #")
        ax1.set_ylabel("ROAS")
        ax1.set_title("ROAS Learning Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Creative performance
        ax2 = axes[0, 1]
        creative_perf = df.groupby("creative_type")["actual_roas"].mean().sort_values()
        ax2.barh(creative_perf.index, creative_perf.values)
        ax2.set_xlabel("Average ROAS")
        ax2.set_title("Performance by Creative Type")
        
        # 3. Budget vs ROAS
        ax3 = axes[1, 0]
        ax3.scatter(df["budget"], df["actual_roas"], alpha=0.6)
        ax3.set_xlabel("Budget ($)")
        ax3.set_ylabel("ROAS")
        ax3.set_title("Budget Optimization")
        ax3.grid(True, alpha=0.3)
        
        # 4. Phase comparison
        ax4 = axes[1, 1]
        phase_perf = df.groupby("phase")["actual_roas"].mean().sort_values()
        ax4.bar(phase_perf.index, phase_perf.values)
        ax4.set_xlabel("Phase")
        ax4.set_ylabel("Average ROAS")
        ax4.set_title("Performance by Phase")
        
        plt.tight_layout()
        plt.savefig("learning_progress.png", dpi=100, bbox_inches='tight')
        console.print("[green]Learning visualization saved to learning_progress.png[/green]")
        
        return fig

def simulate_learning_session():
    """Simulate a complete learning session"""
    
    tracker = LearningTracker()
    
    console.print(Panel.fit(
        "[bold]GAELP Learning Tracker Demo[/bold]\n"
        "Tracking what the agent learns across campaigns",
        border_style="cyan"
    ))
    
    # Simulate campaigns with learning
    phases = [
        ("simulation", 20, 1.5, 3.5),  # phase, episodes, min_roas, max_roas
        ("historical", 10, 2.0, 4.0),
        ("real_testing", 15, 2.5, 4.5),
        ("scaled", 10, 3.0, 5.0)
    ]
    
    campaign_id = 0
    
    for phase, episodes, min_roas, max_roas in phases:
        console.print(f"\n[bold yellow]═══ {phase.upper()} PHASE ═══[/bold yellow]")
        
        for episode in track(range(episodes), description=f"Running {phase}..."):
            campaign_id += 1
            
            # Agent creates campaign (gets better over time)
            learning_factor = episode / episodes  # 0 to 1
            
            campaign = {
                "id": f"campaign_{campaign_id}",
                "creative": {
                    "type": np.random.choice(["image", "video", "carousel"], 
                                           p=[0.4, 0.4, 0.2] if learning_factor < 0.5 else [0.2, 0.5, 0.3])
                },
                "target_audience": np.random.choice(["professionals", "young_adults", "families"],
                                                  p=[0.5, 0.3, 0.2] if learning_factor > 0.3 else [0.33, 0.33, 0.34]),
                "budget": 30 + learning_factor * 40  # Learns optimal budget
            }
            
            # Simulate performance (improves with learning)
            base_roas = min_roas + (max_roas - min_roas) * learning_factor
            noise = np.random.normal(0, 0.3)
            actual_roas = max(0.5, base_roas + noise)
            
            performance = {
                "roas": actual_roas,
                "ctr": 0.02 + learning_factor * 0.03,
                "conversions": int(5 + learning_factor * 20),
                "cost": campaign["budget"],
                "revenue": campaign["budget"] * actual_roas
            }
            
            # Record campaign
            tracker.record_campaign(phase, campaign, performance)
            
            # Discover patterns periodically
            if episode % 5 == 4:
                if phase == "simulation":
                    tracker.discover_pattern(
                        "creative_optimization",
                        f"Video ads show {np.random.randint(20, 40)}% higher engagement for {campaign['target_audience']}",
                        confidence=0.75 + learning_factor * 0.2
                    )
                elif phase == "real_testing":
                    tracker.discover_pattern(
                        "timing_pattern",
                        f"Morning campaigns (6-9am) perform {np.random.randint(15, 30)}% better",
                        confidence=0.8 + learning_factor * 0.15
                    )
            
            # Record some failures for learning
            if actual_roas < 1.5 and episode % 3 == 0:
                tracker.record_failure(
                    campaign["id"],
                    "Low ROAS due to poor audience match",
                    "Need better audience segmentation for this creative type"
                )
    
    # Generate and display report
    console.print("\n")
    report = tracker.generate_report()
    console.print(Panel(report, title="📊 Learning Report", border_style="green"))
    
    # Save detailed learning history
    with open("learning_history.json", "w") as f:
        json.dump(tracker.learning_history, f, indent=2, default=str)
    
    console.print("\n[dim]Detailed learning history saved to learning_history.json[/dim]")
    
    # Create visualization
    tracker.visualize_learning()
    
    return tracker

if __name__ == "__main__":
    tracker = simulate_learning_session()
    
    console.print("\n[bold cyan]🎯 Agent Learning Complete![/bold cyan]")
    console.print(f"Total discoveries: {len(tracker.learning_history['discovered_strategies'])}")
    console.print(f"Lessons from failures: {len(tracker.learning_history['failed_experiments'])}")