#!/usr/bin/env python3
"""
GAELP Performance Visualization Dashboard
Real-time performance monitoring for advertising agents
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Tuple
import asyncio
import threading
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="GAELP Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PerformanceDashboard:
    """Real-time performance visualization dashboard for GAELP agents"""
    
    def __init__(self):
        self.data_path = Path("/home/hariravichandran/AELP")
        self.reports_path = Path("/home/hariravichandran/AELP/performance_reports")
        self.reports_path.mkdir(exist_ok=True)
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
            
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
            
            # Load campaign history
            campaign_history_path = self.data_path / "campaign_history.json"
            if campaign_history_path.exists():
                with open(campaign_history_path, 'r') as f:
                    data['campaign_history'] = json.load(f)
            
            # Load real data comparison
            real_data_path = self.data_path / "data" / "aggregated_data.csv"
            if real_data_path.exists():
                data['real_data'] = pd.read_csv(real_data_path)
            
            # Load benchmarks
            metadata_path = self.data_path / "data" / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data['benchmarks'] = json.load(f).get('benchmarks', {})
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
        return data
    
    def create_learning_curves(self, data: Dict[str, Any]) -> go.Figure:
        """Create interactive learning curves showing agent improvement"""
        if 'learning_history' not in data or 'campaigns_created' not in data['learning_history']:
            return go.Figure().add_annotation(text="No learning data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        campaigns = data['learning_history']['campaigns_created']
        df = pd.DataFrame(campaigns)
        
        if df.empty:
            return go.Figure().add_annotation(text="No campaign data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate rolling averages
        window_size = max(5, len(df) // 10)
        df['roas_rolling'] = df['actual_roas'].rolling(window=window_size, min_periods=1).mean()
        df['ctr_rolling'] = df['actual_ctr'].rolling(window=window_size, min_periods=1).mean()
        df['conversion_rolling'] = df['actual_conversions'].rolling(window=window_size, min_periods=1).mean()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROAS Over Time', 'CTR Over Time', 'Conversions Over Time', 'Revenue Trend'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # ROAS trend
        fig.add_trace(
            go.Scatter(x=df.index, y=df['actual_roas'], mode='markers', name='ROAS (Individual)',
                      opacity=0.6, marker=dict(size=4, color='lightblue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['roas_rolling'], mode='lines', name='ROAS (Trend)',
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # CTR trend
        fig.add_trace(
            go.Scatter(x=df.index, y=df['actual_ctr'], mode='markers', name='CTR (Individual)',
                      opacity=0.6, marker=dict(size=4, color='lightgreen')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ctr_rolling'], mode='lines', name='CTR (Trend)',
                      line=dict(color='green', width=3)),
            row=1, col=2
        )
        
        # Conversions trend
        fig.add_trace(
            go.Scatter(x=df.index, y=df['actual_conversions'], mode='markers', name='Conversions (Individual)',
                      opacity=0.6, marker=dict(size=4, color='lightyellow')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['conversion_rolling'], mode='lines', name='Conversions (Trend)',
                      line=dict(color='orange', width=3)),
            row=2, col=1
        )
        
        # Revenue and cost comparison
        fig.add_trace(
            go.Scatter(x=df.index, y=df['revenue'], mode='lines', name='Revenue',
                      line=dict(color='green', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['cost'], mode='lines', name='Cost',
                      line=dict(color='red', width=2)),
            row=2, col=2, secondary_y=True
        )
        
        fig.update_layout(
            title="Agent Learning Progress Over Time",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_strategy_performance_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create chart showing discovered strategy performance"""
        if 'rl_analysis' not in data or 'strategies' not in data['rl_analysis']:
            return go.Figure().add_annotation(text="No strategy data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        strategies = data['rl_analysis']['strategies']
        df = pd.DataFrame(strategies)
        
        if df.empty:
            return go.Figure().add_annotation(text="No strategy analysis available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Sort by performance
        df = df.sort_values('performance', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Color scale based on confidence
        colors = px.colors.sequential.Viridis
        
        fig.add_trace(go.Bar(
            y=df['name'],
            x=df['performance'],
            orientation='h',
            text=[f"{perf:.2f}x ROAS" for perf in df['performance']],
            textposition='inside',
            marker=dict(
                color=df['confidence'],
                colorscale='Viridis',
                colorbar=dict(title="Confidence Level"),
                cmin=0,
                cmax=1
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         'Performance: %{x:.2f}x ROAS<br>' +
                         'Confidence: %{marker.color:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Discovered Strategy Performance Rankings",
            xaxis_title="ROAS Performance",
            yaxis_title="Strategy",
            height=400,
            margin=dict(l=250)
        )
        
        return fig
    
    def create_real_vs_sim_comparison(self, data: Dict[str, Any]) -> go.Figure:
        """Compare simulator performance vs real data benchmarks"""
        if 'benchmarks' not in data or 'learning_history' not in data:
            return go.Figure().add_annotation(text="No comparison data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Calculate agent performance
        campaigns = data.get('learning_history', {}).get('campaigns_created', [])
        if not campaigns:
            return go.Figure().add_annotation(text="No campaign data for comparison", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        df = pd.DataFrame(campaigns)
        agent_performance = {
            'avg_roas': df['actual_roas'].mean(),
            'avg_ctr': df['actual_ctr'].mean(),
            'avg_conversions': df['actual_conversions'].mean(),
            'avg_cost': df['cost'].mean()
        }
        
        # Create comparison chart
        benchmarks = data['benchmarks']
        
        metrics = ['avg_roas', 'avg_ctr']
        industries = list(benchmarks.keys())
        
        fig = go.Figure()
        
        # Add benchmark data for each industry
        for industry in industries:
            fig.add_trace(go.Bar(
                name=f"{industry.title()} Benchmark",
                x=metrics,
                y=[benchmarks[industry].get(metric, 0) for metric in metrics],
                opacity=0.7
            ))
        
        # Add agent performance
        fig.add_trace(go.Bar(
            name="Agent Performance",
            x=metrics,
            y=[agent_performance.get(metric, 0) for metric in metrics],
            marker_color='red',
            opacity=0.9
        ))
        
        fig.update_layout(
            title="Agent Performance vs Industry Benchmarks",
            xaxis_title="Metrics",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_campaign_heatmap(self, data: Dict[str, Any]) -> go.Figure:
        """Create heatmap of campaign performance by creative type and audience"""
        if 'learning_history' not in data or 'campaigns_created' not in data['learning_history']:
            return go.Figure().add_annotation(text="No campaign data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        campaigns = data['learning_history']['campaigns_created']
        df = pd.DataFrame(campaigns)
        
        if df.empty or 'creative_type' not in df.columns or 'target_audience' not in df.columns:
            return go.Figure().add_annotation(text="Insufficient campaign data for heatmap", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Create pivot table
        heatmap_data = df.groupby(['creative_type', 'target_audience'])['actual_roas'].mean().reset_index()
        pivot_table = heatmap_data.pivot(index='creative_type', columns='target_audience', values='actual_roas')
        
        # Fill NaN values with 0
        pivot_table = pivot_table.fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlGn',
            text=np.round(pivot_table.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Average ROAS")
        ))
        
        fig.update_layout(
            title="Campaign Performance Heatmap: ROAS by Creative Type & Audience",
            xaxis_title="Target Audience",
            yaxis_title="Creative Type",
            height=400
        )
        
        return fig
    
    def create_learning_efficiency_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Show how quickly the agent learns and improves"""
        if 'learning_history' not in data or 'campaigns_created' not in data['learning_history']:
            return go.Figure().add_annotation(text="No learning data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        campaigns = data['learning_history']['campaigns_created']
        df = pd.DataFrame(campaigns)
        
        if df.empty:
            return go.Figure().add_annotation(text="No campaign data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Calculate learning efficiency metrics
        df['campaign_number'] = range(1, len(df) + 1)
        
        # Rolling maximum (best performance so far)
        df['best_roas_so_far'] = df['actual_roas'].expanding().max()
        
        # Calculate improvement rate
        window = max(5, len(df) // 20)
        df['improvement_rate'] = df['actual_roas'].rolling(window=window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0
        )
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Learning Progress: Best Performance Achieved', 'Learning Rate Over Time'),
            vertical_spacing=0.12
        )
        
        # Best performance trend
        fig.add_trace(
            go.Scatter(x=df['campaign_number'], y=df['best_roas_so_far'], 
                      mode='lines', name='Best ROAS Achieved',
                      line=dict(color='green', width=3)),
            row=1, col=1
        )
        
        # Current performance for comparison
        fig.add_trace(
            go.Scatter(x=df['campaign_number'], y=df['actual_roas'], 
                      mode='markers', name='Individual Campaign ROAS',
                      marker=dict(size=4, color='lightblue', opacity=0.6)),
            row=1, col=1
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=df['campaign_number'], y=df['improvement_rate'], 
                      mode='lines', name='Learning Rate',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Agent Learning Efficiency Analysis",
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Campaign Number", row=2, col=1)
        fig.update_yaxes(title_text="ROAS", row=1, col=1)
        fig.update_yaxes(title_text="Improvement Rate", row=2, col=1)
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str):
        """Save visualization to performance_reports directory"""
        try:
            # Save as HTML
            html_path = self.reports_path / f"{filename}.html"
            fig.write_html(str(html_path))
            
            # Save as PNG
            png_path = self.reports_path / f"{filename}.png"
            fig.write_image(str(png_path), width=1200, height=800)
            
            st.success(f"Visualization saved to {self.reports_path}")
            
        except Exception as e:
            st.error(f"Error saving visualization: {str(e)}")
    
    def display_summary_metrics(self, data: Dict[str, Any]):
        """Display key performance metrics in the sidebar"""
        if 'learning_history' not in data or 'campaigns_created' not in data['learning_history']:
            st.sidebar.error("No performance data available")
            return
        
        campaigns = data['learning_history']['campaigns_created']
        df = pd.DataFrame(campaigns)
        
        if df.empty:
            st.sidebar.error("No campaign data available")
            return
        
        # Calculate summary metrics
        total_campaigns = len(df)
        avg_roas = df['actual_roas'].mean()
        best_roas = df['actual_roas'].max()
        total_revenue = df['revenue'].sum()
        total_cost = df['cost'].sum()
        roi = ((total_revenue - total_cost) / total_cost) * 100
        
        # RL analysis metrics
        strategies_discovered = 0
        best_strategy_performance = 0
        if 'rl_analysis' in data and 'strategies' in data['rl_analysis']:
            strategies_discovered = len(data['rl_analysis']['strategies'])
            if strategies_discovered > 0:
                best_strategy_performance = max(s['performance'] for s in data['rl_analysis']['strategies'])
        
        st.sidebar.markdown("### 📊 Performance Summary")
        st.sidebar.metric("Total Campaigns", total_campaigns)
        st.sidebar.metric("Average ROAS", f"{avg_roas:.2f}x")
        st.sidebar.metric("Best ROAS", f"{best_roas:.2f}x")
        st.sidebar.metric("Total ROI", f"{roi:.1f}%")
        st.sidebar.metric("Strategies Discovered", strategies_discovered)
        if best_strategy_performance > 0:
            st.sidebar.metric("Best Strategy ROAS", f"{best_strategy_performance:.2f}x")
    
    def run_dashboard(self):
        """Main dashboard interface"""
        st.title("🚀 GAELP Performance Dashboard")
        st.markdown("Real-time monitoring of advertising agent learning and performance")
        
        # Sidebar controls
        st.sidebar.title("Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Load data
        with st.spinner("Loading performance data..."):
            data = self.load_data()
        
        # Display summary metrics
        self.display_summary_metrics(data)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Learning Curves", 
            "🏆 Strategy Rankings", 
            "⚖️ Sim vs Real", 
            "🔥 Performance Heatmap", 
            "📈 Learning Efficiency"
        ])
        
        with tab1:
            st.subheader("Agent Learning Progress Over Time")
            st.markdown("Track how the agent's performance improves with each campaign")
            
            learning_fig = self.create_learning_curves(data)
            st.plotly_chart(learning_fig, use_container_width=True)
            
            if st.button("💾 Save Learning Curves", key="save_learning"):
                self.save_visualization(learning_fig, "learning_curves")
        
        with tab2:
            st.subheader("Discovered Strategy Performance Rankings")
            st.markdown("See which strategies the agent has discovered and how well they perform")
            
            strategy_fig = self.create_strategy_performance_chart(data)
            st.plotly_chart(strategy_fig, use_container_width=True)
            
            # Show strategy details
            if 'rl_analysis' in data and 'strategies' in data['rl_analysis']:
                st.subheader("Strategy Details")
                strategies_df = pd.DataFrame(data['rl_analysis']['strategies'])
                st.dataframe(strategies_df[['name', 'description', 'performance', 'confidence']], 
                           use_container_width=True)
            
            if st.button("💾 Save Strategy Rankings", key="save_strategies"):
                self.save_visualization(strategy_fig, "strategy_rankings")
        
        with tab3:
            st.subheader("Simulator vs Real Data Performance")
            st.markdown("Compare agent performance against industry benchmarks")
            
            comparison_fig = self.create_real_vs_sim_comparison(data)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            if st.button("💾 Save Comparison Chart", key="save_comparison"):
                self.save_visualization(comparison_fig, "sim_vs_real_comparison")
        
        with tab4:
            st.subheader("Campaign Performance Heatmap")
            st.markdown("Visualize ROAS performance by creative type and target audience")
            
            heatmap_fig = self.create_campaign_heatmap(data)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            if st.button("💾 Save Heatmap", key="save_heatmap"):
                self.save_visualization(heatmap_fig, "performance_heatmap")
        
        with tab5:
            st.subheader("Learning Efficiency Analysis")
            st.markdown("Analyze how quickly and efficiently the agent is learning")
            
            efficiency_fig = self.create_learning_efficiency_chart(data)
            st.plotly_chart(efficiency_fig, use_container_width=True)
            
            if st.button("💾 Save Efficiency Chart", key="save_efficiency"):
                self.save_visualization(efficiency_fig, "learning_efficiency")
        
        # Real-time updates
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main entry point for the dashboard"""
    dashboard = PerformanceDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()