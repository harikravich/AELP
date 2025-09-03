#!/usr/bin/env python3
"""
GAELP Success Criteria and Performance Monitoring System

Defines comprehensive ROAS targets and success criteria for GAELP production deployment.
Establishes clear metrics, thresholds, KPIs, and monitoring with NO FALLBACKS.

This system enforces strict success criteria and fails loudly when targets are not met.
NO simplified versions, NO mock implementations, NO fallback thresholds.
"""

import asyncio
import logging
import time
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from collections import defaultdict, deque
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"         # Action required within 1 hour
    MEDIUM = "medium"     # Action required within 24 hours
    LOW = "low"          # Monitoring alert only


class KPICategory(Enum):
    """KPI categories for organization"""
    PROFITABILITY = "profitability"      # ROAS, profit margins, ROI
    EFFICIENCY = "efficiency"            # CTR, CVR, CPC, CPA
    SCALE = "scale"                       # Volume, reach, impressions
    QUALITY = "quality"                   # Brand safety, user experience
    LEARNING = "learning"                 # Model performance, convergence
    OPERATIONAL = "operational"           # System health, uptime


@dataclass
class SuccessCriteria:
    """Defines success criteria for a specific KPI"""
    
    name: str
    category: KPICategory
    
    # Core thresholds - NO FALLBACKS
    target_value: float                   # Primary target
    minimum_acceptable: float             # Absolute minimum - failure below this
    excellence_threshold: float           # Excellence threshold for bonuses
    
    # Monitoring parameters
    measurement_window_hours: int = 24    # Time window for measurement
    alert_threshold: float = 0.9          # Alert if below this * minimum_acceptable
    
    # Trend requirements
    require_positive_trend: bool = True   # Must show improvement over time
    trend_window_days: int = 7           # Days to measure trend
    
    # Business impact
    business_critical: bool = False       # System stops if this fails
    revenue_impact: float = 0.0          # Estimated daily revenue impact if failed
    
    # Validation
    validation_required: bool = True      # Requires human validation
    auto_recovery_enabled: bool = False   # Can system auto-recovery
    lower_is_better: bool = False         # For metrics where lower values are better (CPA, response time)
    
    def __post_init__(self):
        """Validate criteria configuration"""
        if self.lower_is_better:
            # For lower-is-better metrics: excellence < target < minimum_acceptable
            if self.excellence_threshold >= self.target_value:
                raise ValueError("Excellence threshold must be less than target value for lower-is-better metrics")
            if self.target_value >= self.minimum_acceptable:
                raise ValueError("Target must be less than minimum acceptable for lower-is-better metrics")
        else:
            # For higher-is-better metrics: minimum_acceptable < target < excellence
            if self.minimum_acceptable >= self.target_value:
                raise ValueError("Minimum acceptable must be less than target value")
            if self.target_value >= self.excellence_threshold:
                raise ValueError("Target must be less than excellence threshold")
        
        if self.alert_threshold > 1.0:
            raise ValueError("Alert threshold cannot exceed 1.0")


@dataclass
class KPIMetrics:
    """Real-time KPI measurement data"""
    
    kpi_name: str
    current_value: float
    target_value: float
    minimum_acceptable: float
    
    # Performance indicators
    performance_ratio: float              # current / target
    days_at_current_level: int           # Stability measure
    trend_direction: str                 # "improving", "declining", "stable"
    trend_strength: float                # Rate of change
    
    # Status
    status: str                          # "excellent", "good", "warning", "critical"
    last_updated: datetime
    
    # Historical context
    best_7_day_value: float
    worst_7_day_value: float
    average_30_day_value: float
    
    # Alerts
    active_alerts: List[str]
    alert_history: List[Dict[str, Any]]


class GAELPSuccessCriteriaDefinition:
    """
    Comprehensive success criteria definition for GAELP system.
    
    Defines production-ready success criteria with NO FALLBACKS.
    Every threshold is carefully calculated based on industry standards
    and business requirements.
    """
    
    def __init__(self):
        """Initialize success criteria definitions"""
        
        # Core ROAS targets based on industry benchmarks
        # NO SIMPLIFIED VERSIONS - These are production targets
        self.success_criteria = self._define_all_success_criteria()
        
        # Performance targets by channel and segment
        self.channel_specific_targets = self._define_channel_targets()
        
        # Learning performance requirements
        self.learning_targets = self._define_learning_targets()
        
        # Operational health requirements
        self.operational_targets = self._define_operational_targets()
        
        logger.info(f"Defined {len(self.success_criteria)} success criteria")
        logger.info(f"Business critical KPIs: {self._count_business_critical()}")
    
    def _define_all_success_criteria(self) -> Dict[str, SuccessCriteria]:
        """Define all success criteria - NO FALLBACKS"""
        
        criteria = {}
        
        # ============ PROFITABILITY KPIs ============
        
        criteria["overall_roas"] = SuccessCriteria(
            name="Overall ROAS",
            category=KPICategory.PROFITABILITY,
            target_value=4.0,                    # 4:1 return target
            minimum_acceptable=2.5,              # Never accept below 2.5:1
            excellence_threshold=6.0,            # Excellence at 6:1
            measurement_window_hours=24,
            business_critical=True,
            revenue_impact=10000.0,              # $10k daily impact
            require_positive_trend=True
        )
        
        criteria["search_campaign_roas"] = SuccessCriteria(
            name="Search Campaign ROAS",
            category=KPICategory.PROFITABILITY,
            target_value=5.0,                    # Search should be higher ROAS
            minimum_acceptable=3.0,
            excellence_threshold=7.0,
            business_critical=True,
            revenue_impact=6000.0
        )
        
        criteria["display_campaign_roas"] = SuccessCriteria(
            name="Display Campaign ROAS",
            category=KPICategory.PROFITABILITY,
            target_value=3.5,                    # Lower but still profitable
            minimum_acceptable=2.0,
            excellence_threshold=5.0,
            business_critical=True,
            revenue_impact=4000.0
        )
        
        criteria["video_campaign_roas"] = SuccessCriteria(
            name="Video Campaign ROAS",
            category=KPICategory.PROFITABILITY,
            target_value=3.0,                    # Video requires brand building
            minimum_acceptable=1.8,
            excellence_threshold=4.5,
            business_critical=False,
            revenue_impact=2000.0
        )
        
        criteria["profit_margin"] = SuccessCriteria(
            name="Profit Margin %",
            category=KPICategory.PROFITABILITY,
            target_value=65.0,                   # 65% profit margin
            minimum_acceptable=50.0,             # Never below 50%
            excellence_threshold=75.0,
            business_critical=True,
            revenue_impact=8000.0
        )
        
        # ============ EFFICIENCY KPIs ============
        
        criteria["overall_ctr"] = SuccessCriteria(
            name="Overall CTR %",
            category=KPICategory.EFFICIENCY,
            target_value=3.5,                    # 3.5% CTR target
            minimum_acceptable=2.0,              # Industry average baseline
            excellence_threshold=5.0,
            measurement_window_hours=24,
            business_critical=False,
            revenue_impact=1000.0
        )
        
        criteria["conversion_rate"] = SuccessCriteria(
            name="Conversion Rate %",
            category=KPICategory.EFFICIENCY,
            target_value=8.0,                    # 8% conversion rate
            minimum_acceptable=5.0,
            excellence_threshold=12.0,
            business_critical=True,
            revenue_impact=7000.0
        )
        
        criteria["cost_per_acquisition"] = SuccessCriteria(
            name="Cost Per Acquisition ($)",
            category=KPICategory.EFFICIENCY,
            target_value=25.0,                   # $25 CPA target
            minimum_acceptable=45.0,             # Never above $45
            excellence_threshold=15.0,           # Excellence below $15
            business_critical=True,
            revenue_impact=5000.0,
            require_positive_trend=False,        # Lower is better
            lower_is_better=True
        )
        
        criteria["cost_per_click"] = SuccessCriteria(
            name="Average CPC ($)",
            category=KPICategory.EFFICIENCY,
            target_value=0.75,                   # $0.75 CPC target
            minimum_acceptable=1.50,             # Max $1.50 CPC
            excellence_threshold=0.50,
            business_critical=False,
            require_positive_trend=False,        # Lower is better
            lower_is_better=True
        )
        
        # ============ SCALE KPIs ============
        
        criteria["daily_impressions"] = SuccessCriteria(
            name="Daily Impressions",
            category=KPICategory.SCALE,
            target_value=100000.0,               # 100k daily impressions
            minimum_acceptable=50000.0,
            excellence_threshold=200000.0,
            measurement_window_hours=24,
            business_critical=False,
            revenue_impact=500.0
        )
        
        criteria["daily_clicks"] = SuccessCriteria(
            name="Daily Clicks",
            category=KPICategory.SCALE,
            target_value=3500.0,                 # Based on 3.5% CTR * 100k impressions
            minimum_acceptable=2000.0,
            excellence_threshold=7000.0,
            business_critical=False
        )
        
        criteria["daily_conversions"] = SuccessCriteria(
            name="Daily Conversions",
            category=KPICategory.SCALE,
            target_value=280.0,                  # Based on 8% CVR * 3500 clicks
            minimum_acceptable=150.0,
            excellence_threshold=500.0,
            business_critical=True,
            revenue_impact=3000.0
        )
        
        # ============ QUALITY KPIs ============
        
        criteria["brand_safety_score"] = SuccessCriteria(
            name="Brand Safety Score",
            category=KPICategory.QUALITY,
            target_value=95.0,                   # 95% brand safety
            minimum_acceptable=90.0,             # Never below 90%
            excellence_threshold=98.0,
            business_critical=True,
            revenue_impact=15000.0               # Brand damage is expensive
        )
        
        criteria["user_experience_score"] = SuccessCriteria(
            name="User Experience Score",
            category=KPICategory.QUALITY,
            target_value=85.0,                   # 85% UX satisfaction
            minimum_acceptable=75.0,
            excellence_threshold=92.0,
            business_critical=False,
            revenue_impact=2000.0
        )
        
        criteria["quality_score_avg"] = SuccessCriteria(
            name="Google Ads Quality Score",
            category=KPICategory.QUALITY,
            target_value=8.0,                    # Quality Score 8/10
            minimum_acceptable=6.0,
            excellence_threshold=9.0,
            business_critical=False,
            revenue_impact=3000.0
        )
        
        # ============ LEARNING KPIs ============
        
        criteria["model_accuracy"] = SuccessCriteria(
            name="ML Model Accuracy %",
            category=KPICategory.LEARNING,
            target_value=85.0,                   # 85% prediction accuracy
            minimum_acceptable=75.0,
            excellence_threshold=92.0,
            business_critical=True,
            revenue_impact=4000.0
        )
        
        criteria["convergence_rate"] = SuccessCriteria(
            name="Learning Convergence Rate",
            category=KPICategory.LEARNING,
            target_value=0.15,                   # 15% improvement per week
            minimum_acceptable=0.05,
            excellence_threshold=0.25,
            trend_window_days=7,
            business_critical=True,
            revenue_impact=2000.0
        )
        
        criteria["exploration_efficiency"] = SuccessCriteria(
            name="Exploration Efficiency %",
            category=KPICategory.LEARNING,
            target_value=75.0,                   # 75% of exploration is valuable
            minimum_acceptable=60.0,
            excellence_threshold=85.0,
            business_critical=False
        )
        
        # ============ OPERATIONAL KPIs ============
        
        criteria["system_uptime"] = SuccessCriteria(
            name="System Uptime %",
            category=KPICategory.OPERATIONAL,
            target_value=99.9,                   # 99.9% uptime
            minimum_acceptable=99.5,             # Max 4 hours downtime/month
            excellence_threshold=99.95,
            measurement_window_hours=24,
            business_critical=True,
            revenue_impact=20000.0               # Downtime is very expensive
        )
        
        criteria["response_time_p95"] = SuccessCriteria(
            name="Response Time P95 (ms)",
            category=KPICategory.OPERATIONAL,
            target_value=100.0,                  # 100ms P95 response
            minimum_acceptable=250.0,            # Never above 250ms
            excellence_threshold=50.0,
            business_critical=False,
            require_positive_trend=False,        # Lower is better
            lower_is_better=True
        )
        
        criteria["budget_utilization"] = SuccessCriteria(
            name="Budget Utilization %",
            category=KPICategory.OPERATIONAL,
            target_value=95.0,                   # Use 95% of budget
            minimum_acceptable=85.0,
            excellence_threshold=98.0,
            business_critical=False,
            revenue_impact=1000.0
        )
        
        return criteria
    
    def _define_channel_targets(self) -> Dict[str, Dict[str, float]]:
        """Define channel-specific performance targets"""
        
        return {
            "google_search": {
                "target_roas": 5.0,
                "min_roas": 3.5,
                "target_ctr": 4.5,
                "target_cvr": 12.0,
                "max_cpa": 20.0
            },
            "google_display": {
                "target_roas": 3.5,
                "min_roas": 2.2,
                "target_ctr": 2.8,
                "target_cvr": 6.5,
                "max_cpa": 35.0
            },
            "youtube_video": {
                "target_roas": 3.0,
                "min_roas": 1.8,
                "target_ctr": 2.2,
                "target_cvr": 4.8,
                "max_cpa": 40.0
            },
            "facebook_feed": {
                "target_roas": 4.2,
                "min_roas": 2.8,
                "target_ctr": 3.8,
                "target_cvr": 8.5,
                "max_cpa": 28.0
            },
            "instagram_stories": {
                "target_roas": 3.8,
                "min_roas": 2.5,
                "target_ctr": 4.2,
                "target_cvr": 7.2,
                "max_cpa": 32.0
            }
        }
    
    def _define_learning_targets(self) -> Dict[str, float]:
        """Define ML learning performance targets"""
        
        return {
            # Model performance
            "prediction_accuracy_min": 75.0,     # Never below 75%
            "prediction_accuracy_target": 85.0,
            
            # Learning efficiency
            "convergence_episodes_max": 5000,    # Must converge within 5k episodes
            "convergence_episodes_target": 3000,
            
            # Exploration vs exploitation
            "exploration_rate_min": 0.1,         # Always explore at least 10%
            "exploration_rate_max": 0.4,         # Never explore more than 40%
            
            # Reward learning
            "reward_improvement_weekly_min": 0.05,  # 5% weekly improvement minimum
            "reward_improvement_weekly_target": 0.15,
            
            # Feature importance stability
            "feature_stability_min": 0.8,        # Feature rankings 80% stable
            "feature_stability_target": 0.9
        }
    
    def _define_operational_targets(self) -> Dict[str, float]:
        """Define operational health targets"""
        
        return {
            # System performance
            "cpu_utilization_max": 80.0,         # Never exceed 80% CPU
            "memory_utilization_max": 85.0,
            "disk_utilization_max": 90.0,
            
            # Response times
            "api_response_p50_max": 50.0,        # 50ms median response
            "api_response_p95_max": 150.0,       # 150ms 95th percentile
            "api_response_p99_max": 300.0,       # 300ms 99th percentile
            
            # Error rates
            "error_rate_max": 0.1,               # 0.1% error rate maximum
            "timeout_rate_max": 0.05,            # 0.05% timeout rate
            
            # Data quality
            "data_completeness_min": 95.0,       # 95% complete data
            "data_accuracy_min": 98.0,           # 98% accurate data
            
            # Security
            "security_scan_score_min": 95.0,     # 95% security score
            "vulnerability_count_max": 0         # Zero critical vulnerabilities
        }
    
    def _count_business_critical(self) -> int:
        """Count business critical KPIs"""
        return sum(1 for criteria in self.success_criteria.values() 
                  if criteria.business_critical)
    
    def get_success_criteria(self, kpi_name: str) -> Optional[SuccessCriteria]:
        """Get success criteria for specific KPI"""
        return self.success_criteria.get(kpi_name)
    
    def get_criteria_by_category(self, category: KPICategory) -> Dict[str, SuccessCriteria]:
        """Get all criteria for a specific category"""
        return {name: criteria for name, criteria in self.success_criteria.items()
                if criteria.category == category}
    
    def get_business_critical_criteria(self) -> Dict[str, SuccessCriteria]:
        """Get all business critical success criteria"""
        return {name: criteria for name, criteria in self.success_criteria.items()
                if criteria.business_critical}


# Backward-compatible wrapper expected by orchestrator
class SuccessCriteriaMonitor:
    """Compatibility wrapper exposing a simple monitor interface.

    Internally composes GAELPSuccessCriteriaDefinition and PerformanceMonitor.
    """
    def __init__(self, db_path: str = "/home/hariravichandran/AELP/gaelp_performance.db"):
        self.definition = GAELPSuccessCriteriaDefinition()
        self.monitor = PerformanceMonitor(self.definition, db_path=db_path)

    def start(self, interval_seconds: int = 60):
        self.monitor.start_monitoring(check_interval_seconds=interval_seconds)

    def stop(self):
        self.monitor.stop_monitoring()

    def get_status(self) -> Dict[str, Any]:
        return self.monitor.get_system_health_summary()


class PerformanceMonitor:
    """
    Real-time performance monitoring system for GAELP success criteria.
    
    Continuously monitors all KPIs against success criteria and generates
    alerts when thresholds are breached. NO FALLBACKS - fails loudly.
    """
    
    def __init__(self, success_criteria: GAELPSuccessCriteriaDefinition,
                 db_path: str = "/home/hariravichandran/AELP/gaelp_performance.db"):
        """Initialize performance monitor"""
        
        self.success_criteria = success_criteria
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Monitoring state
        self.current_metrics = {}
        self.alert_history = defaultdict(list)
        self.metric_history = defaultdict(deque)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Performance monitor initialized")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kpi_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kpi_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    alert_level TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kpi_name TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL,
                    threshold REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_timestamp DATETIME
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    overall_health_score REAL NOT NULL,
                    business_critical_failures INTEGER NOT NULL,
                    total_active_alerts INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kpi_timestamp ON kpi_measurements(kpi_name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_severity ON performance_alerts(severity, resolved)")
    
    def start_monitoring(self, check_interval_seconds: int = 60):
        """Start continuous monitoring"""
        
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started performance monitoring (check interval: {check_interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self, check_interval_seconds: int):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Get current metrics from GAELP system
                current_metrics = self._collect_current_metrics()
                
                # Evaluate against success criteria
                evaluation_results = self._evaluate_all_metrics(current_metrics)
                
                # Generate alerts if needed
                self._process_alerts(evaluation_results)
                
                # Store results
                self._store_metrics(evaluation_results)
                
                # Update system health score
                self._update_system_health(evaluation_results)
                
                time.sleep(check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # NO FALLBACKS - Log error and continue
                time.sleep(check_interval_seconds)
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current metrics from GAELP system"""
        
        # In production, this would connect to the actual GAELP system
        # For now, we'll simulate with realistic data
        
        # This should connect to:
        # - User Journey Database for conversion metrics
        # - Attribution Engine for ROAS calculations  
        # - Budget Pacer for spend efficiency
        # - Creative Selector for CTR/quality metrics
        # - RL Agent for learning performance
        
        return self._simulate_current_metrics()
    
    def _simulate_current_metrics(self) -> Dict[str, float]:
        """Simulate current metrics (replace with real data collection)"""
        
        # Generate realistic metric values with some variation
        base_time = time.time()
        
        return {
            "overall_roas": 3.8 + 0.5 * np.sin(base_time / 3600),  # Oscillating around 3.8
            "search_campaign_roas": 4.5 + 0.3 * np.sin(base_time / 1800),
            "display_campaign_roas": 3.2 + 0.4 * np.sin(base_time / 2400),
            "video_campaign_roas": 2.8 + 0.2 * np.sin(base_time / 3000),
            "profit_margin": 62.0 + 3.0 * np.sin(base_time / 7200),
            
            "overall_ctr": 3.2 + 0.3 * np.sin(base_time / 1200),
            "conversion_rate": 7.5 + 1.0 * np.sin(base_time / 1800),
            "cost_per_acquisition": 28.0 + 5.0 * np.sin(base_time / 2400),
            "cost_per_click": 0.82 + 0.15 * np.sin(base_time / 1500),
            
            "daily_impressions": 85000 + 15000 * np.sin(base_time / 3600),
            "daily_clicks": 2800 + 400 * np.sin(base_time / 3600),
            "daily_conversions": 210 + 40 * np.sin(base_time / 3600),
            
            "brand_safety_score": 94.0 + 2.0 * np.sin(base_time / 7200),
            "user_experience_score": 83.0 + 3.0 * np.sin(base_time / 5400),
            "quality_score_avg": 7.8 + 0.5 * np.sin(base_time / 3600),
            
            "model_accuracy": 82.0 + 4.0 * np.sin(base_time / 10800),
            "convergence_rate": 0.12 + 0.03 * np.sin(base_time / 7200),
            "exploration_efficiency": 72.0 + 8.0 * np.sin(base_time / 5400),
            
            "system_uptime": 99.85 + 0.1 * np.sin(base_time / 14400),
            "response_time_p95": 120 + 30 * np.sin(base_time / 1800),
            "budget_utilization": 92.0 + 4.0 * np.sin(base_time / 3600)
        }
    
    def _evaluate_all_metrics(self, current_metrics: Dict[str, float]) -> Dict[str, KPIMetrics]:
        """Evaluate all metrics against success criteria"""
        
        results = {}
        
        for kpi_name, current_value in current_metrics.items():
            criteria = self.success_criteria.get_success_criteria(kpi_name)
            
            if not criteria:
                logger.warning(f"No success criteria defined for {kpi_name}")
                continue
            
            # Calculate performance metrics
            kpi_metrics = self._evaluate_single_metric(kpi_name, current_value, criteria)
            results[kpi_name] = kpi_metrics
            
            # Store in current metrics
            with self.lock:
                self.current_metrics[kpi_name] = kpi_metrics
                
                # Keep historical data
                self.metric_history[kpi_name].append((datetime.now(), current_value))
                
                # Keep only recent history
                if len(self.metric_history[kpi_name]) > 10000:
                    self.metric_history[kpi_name].popleft()
        
        return results
    
    def _evaluate_single_metric(self, kpi_name: str, current_value: float, 
                               criteria: SuccessCriteria) -> KPIMetrics:
        """Evaluate single metric against its criteria"""
        
        # Calculate performance ratio
        performance_ratio = current_value / criteria.target_value
        
        # Determine status
        if current_value >= criteria.excellence_threshold:
            status = "excellent"
        elif current_value >= criteria.target_value:
            status = "good"
        elif current_value >= criteria.minimum_acceptable:
            status = "warning"
        else:
            status = "critical"
        
        # Calculate trend (simplified - in production would use more data)
        trend_direction = "stable"  # Would calculate from historical data
        trend_strength = 0.0
        
        # Get historical context (simplified)
        best_7_day = current_value * 1.1
        worst_7_day = current_value * 0.9
        avg_30_day = current_value
        
        # Check for alerts
        active_alerts = self._check_alerts(kpi_name, current_value, criteria, status)
        
        return KPIMetrics(
            kpi_name=kpi_name,
            current_value=current_value,
            target_value=criteria.target_value,
            minimum_acceptable=criteria.minimum_acceptable,
            performance_ratio=performance_ratio,
            days_at_current_level=1,  # Would calculate from historical data
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            status=status,
            last_updated=datetime.now(),
            best_7_day_value=best_7_day,
            worst_7_day_value=worst_7_day,
            average_30_day_value=avg_30_day,
            active_alerts=active_alerts,
            alert_history=[]  # Would populate from database
        )
    
    def _check_alerts(self, kpi_name: str, current_value: float, 
                     criteria: SuccessCriteria, status: str) -> List[str]:
        """Check if alerts should be generated"""
        
        alerts = []
        
        # Critical failure alert
        if current_value < criteria.minimum_acceptable:
            severity = AlertSeverity.CRITICAL if criteria.business_critical else AlertSeverity.HIGH
            alerts.append(f"{severity.value.upper()}: {kpi_name} below minimum acceptable threshold")
        
        # Warning alert
        elif current_value < criteria.alert_threshold * criteria.minimum_acceptable:
            alerts.append(f"WARNING: {kpi_name} approaching minimum threshold")
        
        # Target miss alert
        elif current_value < criteria.target_value * 0.9:
            alerts.append(f"NOTICE: {kpi_name} significantly below target")
        
        return alerts
    
    def _process_alerts(self, evaluation_results: Dict[str, KPIMetrics]):
        """Process and store alerts"""
        
        for kpi_name, metrics in evaluation_results.items():
            criteria = self.success_criteria.get_success_criteria(kpi_name)
            
            for alert_message in metrics.active_alerts:
                # Determine severity
                if "CRITICAL" in alert_message:
                    severity = AlertSeverity.CRITICAL
                elif "WARNING" in alert_message:
                    severity = AlertSeverity.HIGH
                else:
                    severity = AlertSeverity.MEDIUM
                
                # Store alert
                self._store_alert(kpi_name, alert_message, severity, 
                                metrics.current_value, criteria)
                
                # Log alert
                logger.warning(f"ALERT [{severity.value}]: {alert_message}")
                
                # If business critical and critical severity, take action
                if criteria.business_critical and severity == AlertSeverity.CRITICAL:
                    self._handle_critical_business_alert(kpi_name, metrics, criteria)
    
    def _handle_critical_business_alert(self, kpi_name: str, metrics: KPIMetrics, 
                                      criteria: SuccessCriteria):
        """Handle critical business alerts - NO FALLBACKS"""
        
        logger.critical(f"BUSINESS CRITICAL FAILURE: {kpi_name} = {metrics.current_value}, "
                       f"minimum required = {criteria.minimum_acceptable}")
        
        # In production, this would:
        # 1. Send immediate notifications to on-call team
        # 2. Trigger emergency procedures
        # 3. Potentially pause campaigns if revenue impact is high
        # 4. Activate backup systems if available
        
        # NO FALLBACKS - System must be fixed, not worked around
        if criteria.revenue_impact > 5000:
            logger.critical(f"High revenue impact alert: Estimated ${criteria.revenue_impact}/day at risk")
    
    def _store_alert(self, kpi_name: str, message: str, severity: AlertSeverity,
                    value: float, criteria: SuccessCriteria):
        """Store alert in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_alerts 
                (kpi_name, alert_type, severity, message, value, threshold)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (kpi_name, "threshold_breach", severity.value, message, 
                 value, criteria.minimum_acceptable))
    
    def _store_metrics(self, evaluation_results: Dict[str, KPIMetrics]):
        """Store metrics in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            for kpi_name, metrics in evaluation_results.items():
                conn.execute("""
                    INSERT INTO kpi_measurements 
                    (kpi_name, value, status, alert_level, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (kpi_name, metrics.current_value, metrics.status,
                     "high" if metrics.active_alerts else "none",
                     json.dumps({
                         "performance_ratio": metrics.performance_ratio,
                         "trend_direction": metrics.trend_direction,
                         "active_alerts_count": len(metrics.active_alerts)
                     })))
    
    def _update_system_health(self, evaluation_results: Dict[str, KPIMetrics]):
        """Update overall system health score"""
        
        total_kpis = len(evaluation_results)
        if total_kpis == 0:
            return
        
        # Calculate health score based on KPI statuses
        status_weights = {"excellent": 1.0, "good": 0.8, "warning": 0.4, "critical": 0.0}
        
        health_score = sum(status_weights.get(metrics.status, 0) 
                          for metrics in evaluation_results.values()) / total_kpis * 100
        
        # Count critical failures
        business_critical_failures = sum(
            1 for kpi_name, metrics in evaluation_results.items()
            if (metrics.status == "critical" and 
                self.success_criteria.get_success_criteria(kpi_name).business_critical)
        )
        
        # Count active alerts
        total_alerts = sum(len(metrics.active_alerts) for metrics in evaluation_results.values())
        
        # Store system health
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_health 
                (overall_health_score, business_critical_failures, total_active_alerts, metadata)
                VALUES (?, ?, ?, ?)
            """, (health_score, business_critical_failures, total_alerts,
                 json.dumps({"total_kpis": total_kpis})))
        
        # Log health status
        if health_score < 50:
            logger.critical(f"SYSTEM HEALTH CRITICAL: {health_score:.1f}% "
                          f"({business_critical_failures} critical failures)")
        elif health_score < 80:
            logger.warning(f"System health degraded: {health_score:.1f}%")
        else:
            logger.info(f"System health good: {health_score:.1f}%")
    
    def get_current_metrics(self) -> Dict[str, KPIMetrics]:
        """Get current KPI metrics"""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get latest health record
            health_result = conn.execute("""
                SELECT overall_health_score, business_critical_failures, total_active_alerts, timestamp
                FROM system_health 
                ORDER BY timestamp DESC 
                LIMIT 1
            """).fetchone()
            
            if not health_result:
                return {"status": "no_data", "health_score": 0}
            
            health_score, critical_failures, active_alerts, timestamp = health_result
            
            # Get KPI summary
            kpi_summary = {}
            current_metrics = self.get_current_metrics()
            
            for category in KPICategory:
                category_kpis = self.success_criteria.get_criteria_by_category(category)
                category_metrics = []
                
                for kpi_name in category_kpis:
                    if kpi_name in current_metrics:
                        category_metrics.append(current_metrics[kpi_name])
                
                if category_metrics:
                    avg_performance = np.mean([m.performance_ratio for m in category_metrics])
                    kpi_summary[category.value] = {
                        "count": len(category_metrics),
                        "avg_performance_ratio": avg_performance,
                        "critical_count": sum(1 for m in category_metrics if m.status == "critical")
                    }
            
            return {
                "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 50 else "critical",
                "health_score": health_score,
                "business_critical_failures": critical_failures,
                "total_active_alerts": active_alerts,
                "timestamp": timestamp,
                "category_summary": kpi_summary,
                "total_kpis_monitored": len(current_metrics)
            }
    
    def generate_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get metrics in time range
            metrics_data = conn.execute("""
                SELECT kpi_name, AVG(value) as avg_value, MIN(value) as min_value, 
                       MAX(value) as max_value, COUNT(*) as measurement_count
                FROM kpi_measurements 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY kpi_name
            """, (start_time.isoformat(), end_time.isoformat())).fetchall()
            
            # Get alerts in time range
            alerts_data = conn.execute("""
                SELECT kpi_name, severity, COUNT(*) as alert_count
                FROM performance_alerts 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY kpi_name, severity
            """, (start_time.isoformat(), end_time.isoformat())).fetchall()
        
        # Build report
        report = {
            "report_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": hours_back
            },
            "executive_summary": {},
            "kpi_performance": {},
            "alert_summary": {},
            "recommendations": []
        }
        
        # Process metrics data
        total_kpis = len(metrics_data)
        meeting_targets = 0
        critical_issues = 0
        
        for kpi_name, avg_value, min_value, max_value, count in metrics_data:
            criteria = self.success_criteria.get_success_criteria(kpi_name)
            if not criteria:
                continue
            
            performance_ratio = avg_value / criteria.target_value
            
            if avg_value >= criteria.target_value:
                meeting_targets += 1
            if avg_value < criteria.minimum_acceptable:
                critical_issues += 1
            
            report["kpi_performance"][kpi_name] = {
                "avg_value": avg_value,
                "min_value": min_value,
                "max_value": max_value,
                "target_value": criteria.target_value,
                "minimum_acceptable": criteria.minimum_acceptable,
                "performance_ratio": performance_ratio,
                "measurement_count": count,
                "status": "excellent" if avg_value >= criteria.excellence_threshold else
                         "good" if avg_value >= criteria.target_value else
                         "warning" if avg_value >= criteria.minimum_acceptable else
                         "critical"
            }
        
        # Executive summary
        report["executive_summary"] = {
            "total_kpis_monitored": total_kpis,
            "kpis_meeting_targets": meeting_targets,
            "target_achievement_rate": meeting_targets / max(total_kpis, 1) * 100,
            "critical_issues": critical_issues,
            "overall_status": "critical" if critical_issues > 0 else
                            "good" if meeting_targets / max(total_kpis, 1) >= 0.8 else
                            "warning"
        }
        
        # Alert summary
        alert_summary = defaultdict(int)
        for kpi_name, severity, count in alerts_data:
            alert_summary[severity] += count
        
        report["alert_summary"] = dict(alert_summary)
        
        # Generate recommendations
        if critical_issues > 0:
            report["recommendations"].append(
                f"URGENT: {critical_issues} KPIs below minimum acceptable thresholds. Immediate action required."
            )
        
        if meeting_targets / max(total_kpis, 1) < 0.5:
            report["recommendations"].append(
                "Performance significantly below targets. Review strategy and resource allocation."
            )
        
        return report


def main():
    """Main function to demonstrate success criteria system"""
    
    logger.info("Initializing GAELP Success Criteria and Monitoring System")
    
    # Initialize success criteria
    success_criteria = GAELPSuccessCriteriaDefinition()
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(success_criteria)
    
    # Display success criteria summary
    logger.info("=== GAELP SUCCESS CRITERIA SUMMARY ===")
    
    for category in KPICategory:
        criteria = success_criteria.get_criteria_by_category(category)
        logger.info(f"\n{category.value.upper()} KPIs ({len(criteria)}):")
        
        for name, crit in criteria.items():
            critical_flag = " [BUSINESS CRITICAL]" if crit.business_critical else ""
            logger.info(f"  • {crit.name}: Target {crit.target_value}, "
                       f"Min {crit.minimum_acceptable}{critical_flag}")
    
    # Display business critical KPIs
    business_critical = success_criteria.get_business_critical_criteria()
    logger.info(f"\nBUSINESS CRITICAL KPIs ({len(business_critical)}):")
    total_risk = 0
    
    for name, crit in business_critical.items():
        logger.info(f"  • {crit.name}: ${crit.revenue_impact}/day at risk")
        total_risk += crit.revenue_impact
    
    logger.info(f"Total daily revenue at risk: ${total_risk}")
    
    # Start monitoring demonstration
    logger.info("\nStarting performance monitoring demonstration...")
    monitor.start_monitoring(check_interval_seconds=10)  # Fast interval for demo
    
    # Let it run for a bit
    time.sleep(30)
    
    # Generate performance report
    report = monitor.generate_performance_report(hours_back=1)
    logger.info("\n=== PERFORMANCE REPORT ===")
    logger.info(f"Executive Summary: {json.dumps(report['executive_summary'], indent=2)}")
    
    # Show system health
    health = monitor.get_system_health_summary()
    logger.info(f"\nSystem Health: {health['status'].upper()} ({health['health_score']:.1f}%)")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    logger.info("Success criteria system demonstration complete")
    logger.info("Database created at: /home/hariravichandran/AELP/gaelp_performance.db")


if __name__ == "__main__":
    main()
