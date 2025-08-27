# GAELP Performance Visualization Dashboard

A comprehensive real-time performance monitoring system for GAELP advertising agents, featuring both web-based interactive dashboards and command-line analysis tools.

## 🚀 Quick Start

### Web Dashboard (Recommended)
```bash
# Launch the interactive web dashboard
python launch_dashboard.py
```
Dashboard will be available at: http://localhost:8501

### Command Line Analysis
```bash
# Quick performance summary
python performance_cli.py --action summary

# Strategy analysis
python performance_cli.py --action strategies

# Generate full report
python performance_cli.py --action report --save

# Create performance charts
python performance_cli.py --action charts
```

## 📊 Dashboard Features

### 1. Learning Curves Tab
- **Real-time training metrics**: Track ROAS, CTR, conversions over time
- **Trend analysis**: Rolling averages and learning progression
- **Performance visualization**: Multi-metric comparison charts
- **Learning efficiency**: Rate of improvement analysis

### 2. Strategy Rankings Tab
- **Discovered strategies**: Performance-ranked strategy list
- **Strategy details**: Description, confidence levels, test results
- **Performance comparison**: Visual ranking by ROAS performance
- **Evidence tracking**: Number of tests and consistency metrics

### 3. Simulator vs Real Tab
- **Benchmark comparison**: Agent performance vs industry standards
- **Cross-industry analysis**: Compare against retail, finance, travel, B2B
- **Performance gaps**: Identify areas for improvement
- **Real-world validation**: How well the agent performs in practice

### 4. Performance Heatmap Tab
- **Campaign analysis**: ROAS by creative type and audience
- **Pattern identification**: High-performing combinations
- **Visual insights**: Heat-mapped performance data
- **Optimization opportunities**: Spot underperforming segments

### 5. Learning Efficiency Tab
- **Learning speed**: How quickly the agent improves
- **Best performance tracking**: Peak performance achieved over time
- **Improvement rate**: Rate of learning and adaptation
- **Efficiency metrics**: Learning curves and optimization trends

## 🛠️ Installation

### Automatic Installation
The launch script will automatically install required dependencies:
```bash
python launch_dashboard.py
```

### Manual Installation
```bash
# Install dashboard requirements
pip install -r requirements_dashboard.txt

# Install base requirements if not already installed
pip install -r requirements.txt
```

### Required Dependencies
- **streamlit**: Web dashboard framework
- **plotly**: Interactive visualizations
- **pandas**: Data processing
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Additional plotting capabilities
- **rich**: Enhanced CLI output

## 📈 Data Sources

The dashboard automatically loads data from:

1. **learning_history.json**: Campaign performance data
2. **rl_learnings_analysis.json**: Discovered strategies and insights
3. **campaign_history.json**: Historical campaign data
4. **data/aggregated_data.csv**: Real performance benchmarks
5. **data/metadata.json**: Industry benchmark data

## 🎯 Key Metrics Tracked

### Performance Metrics
- **ROAS (Return on Ad Spend)**: Revenue / Cost ratio
- **CTR (Click-Through Rate)**: Clicks / Impressions
- **Conversion Rate**: Conversions / Clicks
- **Revenue**: Total revenue generated
- **Cost**: Total advertising spend
- **ROI**: Return on Investment percentage

### Learning Metrics
- **Strategies Discovered**: Number of effective strategies found
- **Learning Efficiency**: Rate of performance improvement
- **Consistency**: Performance variation analysis
- **Best Performance**: Peak ROAS achieved
- **Trend Analysis**: Learning direction and speed

### Comparison Metrics
- **Industry Benchmarks**: Performance vs industry standards
- **Simulator Accuracy**: Real vs simulated performance
- **Cross-Platform Performance**: Multi-environment comparison

## 🔄 Real-Time Features

### Auto-Refresh
- **30-second updates**: Automatic data refresh
- **Live monitoring**: Real-time performance tracking
- **Status indicators**: Last update timestamps
- **Manual refresh**: On-demand data reload

### Interactive Elements
- **Zoom and pan**: Detailed chart exploration
- **Hover information**: Detailed metric tooltips
- **Filter options**: Time range and metric selection
- **Export capabilities**: Save charts and reports

## 💾 Export and Reporting

### Web Dashboard Exports
- **HTML files**: Interactive charts for sharing
- **PNG images**: High-resolution static charts
- **Performance reports**: Saved to `performance_reports/` directory

### CLI Exports
- **Text reports**: Detailed performance analysis
- **Chart images**: Matplotlib-generated visualizations
- **Data exports**: CSV and JSON formats
- **Timestamped files**: Automatic file versioning

## 🔧 Configuration

### Dashboard Settings
- **Port configuration**: Default 8501, configurable
- **Auto-refresh interval**: 30 seconds default
- **Chart themes**: Multiple color schemes available
- **Export formats**: HTML, PNG, PDF support

### CLI Options
```bash
# Available actions
--action summary      # Performance overview
--action strategies   # Strategy analysis
--action benchmarks   # Industry comparison
--action report       # Full report generation
--action charts       # Visualization export

# Save results to file
--save               # Save output to performance_reports/
```

## 📊 Performance Analysis Examples

### Learning Progress Analysis
```python
# Track improvement over time
recent_performance = df['actual_roas'].tail(10).mean()
early_performance = df['actual_roas'].head(10).mean()
improvement = (recent_performance - early_performance) / early_performance * 100
```

### Strategy Effectiveness
```python
# Identify best strategies
best_strategy = max(strategies, key=lambda x: x['performance'])
consistency = strategy['confidence'] * 100
```

### Benchmark Comparison
```python
# Compare against industry standards
performance_ratio = agent_roas / benchmark_roas
outperformance = (performance_ratio - 1) * 100
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Change port in launch command
   streamlit run dashboard.py --server.port 8502
   ```

2. **Missing data files**:
   - Ensure training runs have generated data files
   - Check file permissions in project directory
   - Verify JSON files are valid format

3. **Import errors**:
   ```bash
   # Reinstall requirements
   pip install -r requirements_dashboard.txt --force-reinstall
   ```

4. **Performance issues**:
   - Large datasets may slow initial load
   - Consider data sampling for very large files
   - Use CLI for quick analysis of large datasets

### Debug Mode
```bash
# Run with debug information
streamlit run dashboard.py --logger.level debug
```

## 🔮 Future Enhancements

### Planned Features
- **Real-time streaming**: Live data updates during training
- **A/B test tracking**: Experiment comparison tools
- **Prediction models**: Performance forecasting
- **Alert system**: Performance threshold notifications
- **Multi-agent comparison**: Side-by-side agent analysis
- **Custom metrics**: User-defined KPI tracking

### Integration Roadmap
- **BigQuery integration**: Large-scale data analysis
- **MLflow tracking**: Experiment management
- **Slack notifications**: Performance alerts
- **API endpoints**: Programmatic access
- **Mobile dashboard**: Responsive design improvements

## 📝 Usage Tips

### Best Practices
1. **Regular monitoring**: Check dashboard daily during training
2. **Baseline comparison**: Always compare against benchmarks
3. **Strategy validation**: Verify high-performing strategies
4. **Export key insights**: Save important findings
5. **Trend analysis**: Focus on learning direction, not just current performance

### Performance Indicators
- **Positive learning**: Upward trending ROAS
- **Strategy discovery**: Increasing strategy count
- **Consistency**: Decreasing performance variation
- **Efficiency**: Faster achievement of performance goals

### Alert Conditions
- **Performance degradation**: Declining ROAS trend
- **High variation**: Inconsistent strategy performance
- **Benchmark gaps**: Significant underperformance vs industry
- **Learning plateau**: Flat improvement curves

## 🤝 Contributing

To enhance the dashboard:
1. Add new visualization types in `dashboard.py`
2. Extend CLI analysis in `performance_cli.py`
3. Create new metric calculations
4. Add data source integrations
5. Improve export formats

---

**Generated for GAELP Project** | Real-time Advertising Agent Performance Monitoring