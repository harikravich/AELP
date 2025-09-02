#!/usr/bin/env python3
"""
Success Criteria Validation Script

Validates that all GAELP success criteria are properly configured and that
the monitoring system works correctly. NO FALLBACKS - all tests must pass.

This script ensures that:
1. All success criteria are mathematically valid
2. All business-critical KPIs have proper thresholds
3. Alert systems function correctly  
4. Monitoring captures all required metrics
5. Escalation procedures are properly defined
"""

import sys
import json
import logging
import sqlite3
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Import our success criteria system
from gaelp_success_criteria_monitor import (
    GAELPSuccessCriteriaDefinition, 
    PerformanceMonitor,
    KPICategory,
    AlertSeverity,
    SuccessCriteria
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SuccessCriteriaValidator:
    """
    Comprehensive validator for GAELP success criteria system.
    
    Performs rigorous validation with NO FALLBACKS - all tests must pass
    for production deployment approval.
    """
    
    def __init__(self):
        """Initialize validator"""
        self.success_criteria = GAELPSuccessCriteriaDefinition()
        self.monitor = PerformanceMonitor(self.success_criteria)
        self.validation_results = {}
        self.critical_failures = []
        
        # Load configuration
        try:
            with open('/home/hariravichandran/AELP/success_criteria_config.json', 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {}
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        
        logger.info("Starting comprehensive success criteria validation")
        logger.info("=" * 60)
        
        validation_steps = [
            ("Mathematical Validity", self._validate_mathematical_consistency),
            ("Business Logic", self._validate_business_logic),
            ("Coverage Analysis", self._validate_coverage_completeness),
            ("Alert Configuration", self._validate_alert_configuration),
            ("Monitoring System", self._validate_monitoring_system),
            ("Database Schema", self._validate_database_schema),
            ("Performance Tests", self._validate_performance_requirements),
            ("Integration Tests", self._validate_integration_readiness),
            ("Production Readiness", self._validate_production_readiness)
        ]
        
        all_passed = True
        
        for step_name, validation_func in validation_steps:
            logger.info(f"\n--- {step_name} ---")
            try:
                result = validation_func()
                self.validation_results[step_name] = result
                
                if result["passed"]:
                    logger.info(f"✅ {step_name}: PASSED")
                    if result.get("warnings"):
                        for warning in result["warnings"]:
                            logger.warning(f"⚠️  {warning}")
                else:
                    logger.error(f"❌ {step_name}: FAILED")
                    all_passed = False
                    for error in result.get("errors", []):
                        logger.error(f"   • {error}")
                        self.critical_failures.append(f"{step_name}: {error}")
                        
            except Exception as e:
                logger.error(f"❌ {step_name}: EXCEPTION - {e}")
                all_passed = False
                self.critical_failures.append(f"{step_name}: Exception - {e}")
        
        # Generate final report
        self._generate_validation_report(all_passed)
        
        return all_passed
    
    def _validate_mathematical_consistency(self) -> Dict[str, Any]:
        """Validate mathematical consistency of all success criteria"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        for kpi_name, criteria in self.success_criteria.success_criteria.items():
            
            # Check threshold ordering
            if criteria.minimum_acceptable >= criteria.target_value:
                result["errors"].append(
                    f"{kpi_name}: minimum_acceptable ({criteria.minimum_acceptable}) "
                    f">= target_value ({criteria.target_value})"
                )
                result["passed"] = False
            
            if criteria.target_value >= criteria.excellence_threshold:
                result["errors"].append(
                    f"{kpi_name}: target_value ({criteria.target_value}) "
                    f">= excellence_threshold ({criteria.excellence_threshold})"
                )
                result["passed"] = False
            
            # Check alert threshold validity
            if criteria.alert_threshold > 1.0 or criteria.alert_threshold < 0.1:
                result["errors"].append(
                    f"{kpi_name}: alert_threshold ({criteria.alert_threshold}) outside valid range [0.1, 1.0]"
                )
                result["passed"] = False
            
            # Check time windows
            if criteria.measurement_window_hours <= 0:
                result["errors"].append(
                    f"{kpi_name}: measurement_window_hours must be positive"
                )
                result["passed"] = False
            
            if criteria.trend_window_days <= 0:
                result["errors"].append(
                    f"{kpi_name}: trend_window_days must be positive"
                )
                result["passed"] = False
            
            # Check revenue impact for business critical KPIs
            if criteria.business_critical and criteria.revenue_impact <= 0:
                result["warnings"].append(
                    f"{kpi_name}: business_critical KPI should have positive revenue_impact"
                )
            
            # Validate thresholds make business sense
            if kpi_name.endswith("_roas"):
                if criteria.minimum_acceptable < 1.0:
                    result["warnings"].append(
                        f"{kpi_name}: ROAS minimum below 1.0 means losing money"
                    )
                if criteria.target_value < 2.0:
                    result["warnings"].append(
                        f"{kpi_name}: ROAS target below 2.0 may not be profitable"
                    )
            
            # Check percentage KPIs
            if any(keyword in kpi_name.lower() for keyword in ['rate', 'score', 'utilization']):
                if criteria.excellence_threshold > 100:
                    result["errors"].append(
                        f"{kpi_name}: excellence_threshold > 100 for percentage metric"
                    )
                    result["passed"] = False
        
        logger.info(f"Validated {len(self.success_criteria.success_criteria)} success criteria")
        return result
    
    def _validate_business_logic(self) -> Dict[str, Any]:
        """Validate business logic and requirements"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        # Check that we have business critical KPIs
        business_critical = self.success_criteria.get_business_critical_criteria()
        if len(business_critical) < 5:
            result["errors"].append(
                f"Insufficient business critical KPIs: {len(business_critical)} < 5"
            )
            result["passed"] = False
        
        # Check that essential KPIs are present and business critical
        essential_kpis = [
            "overall_roas",
            "profit_margin", 
            "conversion_rate",
            "brand_safety_score",
            "system_uptime"
        ]
        
        for essential_kpi in essential_kpis:
            if essential_kpi not in self.success_criteria.success_criteria:
                result["errors"].append(f"Missing essential KPI: {essential_kpi}")
                result["passed"] = False
            elif not self.success_criteria.success_criteria[essential_kpi].business_critical:
                result["errors"].append(f"Essential KPI not marked business_critical: {essential_kpi}")
                result["passed"] = False
        
        # Check revenue impact calculations
        total_daily_risk = sum(
            criteria.revenue_impact 
            for criteria in business_critical.values()
        )
        
        if total_daily_risk > 100000:  # $100k daily risk threshold
            result["warnings"].append(
                f"High total daily revenue at risk: ${total_daily_risk:,.0f}"
            )
        
        # Validate category distribution
        category_counts = {}
        for criteria in self.success_criteria.success_criteria.values():
            category_counts[criteria.category] = category_counts.get(criteria.category, 0) + 1
        
        required_categories = [
            KPICategory.PROFITABILITY,
            KPICategory.EFFICIENCY, 
            KPICategory.QUALITY,
            KPICategory.OPERATIONAL
        ]
        
        for category in required_categories:
            if category_counts.get(category, 0) < 2:
                result["errors"].append(
                    f"Insufficient KPIs in category {category.value}: {category_counts.get(category, 0)} < 2"
                )
                result["passed"] = False
        
        logger.info(f"Business critical KPIs: {len(business_critical)}")
        logger.info(f"Total daily revenue at risk: ${total_daily_risk:,.0f}")
        
        return result
    
    def _validate_coverage_completeness(self) -> Dict[str, Any]:
        """Validate that all important metrics are covered"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        # Check channel-specific coverage
        if hasattr(self.success_criteria, 'channel_specific_targets'):
            channels = self.success_criteria.channel_specific_targets.keys()
            logger.info(f"Channel-specific targets defined for: {list(channels)}")
            
            required_channels = ['google_search', 'google_display', 'facebook_feed']
            for channel in required_channels:
                if channel not in channels:
                    result["warnings"].append(f"Missing channel-specific targets for {channel}")
        
        # Check that we cover the full funnel
        funnel_stages = ['impressions', 'clicks', 'conversions']
        for stage in funnel_stages:
            stage_kpis = [kpi for kpi in self.success_criteria.success_criteria.keys() 
                         if stage in kpi]
            if not stage_kpis:
                result["warnings"].append(f"No KPIs found for funnel stage: {stage}")
        
        # Check learning metrics coverage
        learning_kpis = [kpi for kpi in self.success_criteria.success_criteria.keys()
                        if self.success_criteria.success_criteria[kpi].category == KPICategory.LEARNING]
        
        if len(learning_kpis) < 3:
            result["warnings"].append(f"Limited learning metrics coverage: {len(learning_kpis)} KPIs")
        
        return result
    
    def _validate_alert_configuration(self) -> Dict[str, Any]:
        """Validate alert configuration and escalation procedures"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        if not self.config:
            result["errors"].append("Configuration file not loaded")
            result["passed"] = False
            return result
        
        # Check alert configuration structure
        alert_config = self.config.get('gaelp_success_criteria_config', {}).get('alert_configuration', {})
        
        required_sections = ['notification_channels', 'escalation_procedures']
        for section in required_sections:
            if section not in alert_config:
                result["errors"].append(f"Missing alert configuration section: {section}")
                result["passed"] = False
        
        # Check notification channels
        notification_channels = alert_config.get('notification_channels', {})
        required_severities = ['critical', 'high', 'medium', 'low']
        
        for severity in required_severities:
            if severity not in notification_channels:
                result["errors"].append(f"Missing notification channels for severity: {severity}")
                result["passed"] = False
            elif not notification_channels[severity]:
                result["errors"].append(f"Empty notification channels for severity: {severity}")
                result["passed"] = False
        
        # Check escalation procedures
        escalation_procedures = alert_config.get('escalation_procedures', {})
        critical_procedures = ['critical_business_failure', 'roas_below_minimum', 'system_uptime_failure']
        
        for procedure in critical_procedures:
            if procedure not in escalation_procedures:
                result["warnings"].append(f"Missing escalation procedure: {procedure}")
            else:
                proc_config = escalation_procedures[procedure]
                if 'immediate_actions' not in proc_config or not proc_config['immediate_actions']:
                    result["errors"].append(f"No immediate actions defined for {procedure}")
                    result["passed"] = False
        
        return result
    
    def _validate_monitoring_system(self) -> Dict[str, Any]:
        """Validate monitoring system functionality"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        try:
            # Test database connection and schema
            test_db_path = "/tmp/test_gaelp_performance.db"
            test_monitor = PerformanceMonitor(self.success_criteria, db_path=test_db_path)
            
            # Test metrics collection
            current_metrics = test_monitor.get_current_metrics()
            
            # Test system health summary
            health_summary = test_monitor.get_system_health_summary()
            if health_summary.get("status") == "no_data":
                # Start monitoring briefly to generate some data
                test_monitor.start_monitoring(check_interval_seconds=1)
                import time
                time.sleep(3)
                test_monitor.stop_monitoring()
                
                # Try again
                health_summary = test_monitor.get_system_health_summary()
            
            # Validate health summary structure
            required_fields = ["status", "health_score", "total_kpis_monitored"]
            for field in required_fields:
                if field not in health_summary:
                    result["errors"].append(f"Missing field in health summary: {field}")
                    result["passed"] = False
            
            # Test performance report generation
            report = test_monitor.generate_performance_report(hours_back=1)
            
            required_report_sections = ["executive_summary", "kpi_performance", "alert_summary"]
            for section in required_report_sections:
                if section not in report:
                    result["errors"].append(f"Missing section in performance report: {section}")
                    result["passed"] = False
            
            logger.info("Monitoring system functionality validated")
            
        except Exception as e:
            result["errors"].append(f"Monitoring system test failed: {e}")
            result["passed"] = False
        
        return result
    
    def _validate_database_schema(self) -> Dict[str, Any]:
        """Validate database schema and structure"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        try:
            test_db_path = "/tmp/test_gaelp_performance_schema.db"
            test_monitor = PerformanceMonitor(self.success_criteria, db_path=test_db_path)
            
            # Check that tables exist
            with sqlite3.connect(test_db_path) as conn:
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['kpi_measurements', 'performance_alerts', 'system_health']
                for table in required_tables:
                    if table not in tables:
                        result["errors"].append(f"Missing database table: {table}")
                        result["passed"] = False
                
                # Check table schemas
                for table in required_tables:
                    if table in tables:
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [col[1] for col in cursor.fetchall()]
                        
                        if table == 'kpi_measurements':
                            required_columns = ['kpi_name', 'value', 'timestamp', 'status']
                            for col in required_columns:
                                if col not in columns:
                                    result["errors"].append(f"Missing column {col} in {table}")
                                    result["passed"] = False
                
                # Check indexes exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes = [row[0] for row in cursor.fetchall()]
                
                if not any('kpi_timestamp' in idx for idx in indexes):
                    result["warnings"].append("Performance index on kpi_timestamp not found")
            
            logger.info("Database schema validation completed")
            
        except Exception as e:
            result["errors"].append(f"Database schema validation failed: {e}")
            result["passed"] = False
        
        return result
    
    def _validate_performance_requirements(self) -> Dict[str, Any]:
        """Validate performance requirements are achievable"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        # Check ROAS targets are realistic
        roas_kpis = {k: v for k, v in self.success_criteria.success_criteria.items() 
                    if 'roas' in k.lower()}
        
        for kpi_name, criteria in roas_kpis.items():
            # Industry benchmarks validation
            if criteria.target_value > 10.0:
                result["warnings"].append(
                    f"{kpi_name}: Very high ROAS target ({criteria.target_value}x) may be unrealistic"
                )
            
            if criteria.minimum_acceptable < 1.2:
                result["warnings"].append(
                    f"{kpi_name}: Very low minimum ROAS ({criteria.minimum_acceptable}x) may indicate thin margins"
                )
        
        # Check conversion rate targets
        cvr_kpis = {k: v for k, v in self.success_criteria.success_criteria.items() 
                   if 'conversion' in k.lower()}
        
        for kpi_name, criteria in cvr_kpis.items():
            if criteria.target_value > 20.0:
                result["warnings"].append(
                    f"{kpi_name}: High conversion target ({criteria.target_value}%) may be challenging"
                )
        
        # Check system performance requirements
        system_kpis = {k: v for k, v in self.success_criteria.success_criteria.items() 
                      if v.category == KPICategory.OPERATIONAL}
        
        uptime_kpi = system_kpis.get('system_uptime')
        if uptime_kpi and uptime_kpi.target_value > 99.95:
            result["warnings"].append(
                f"Very high uptime target ({uptime_kpi.target_value}%) requires robust infrastructure"
            )
        
        return result
    
    def _validate_integration_readiness(self) -> Dict[str, Any]:
        """Validate integration readiness with GAELP components"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        # Check that success criteria align with GAELP component capabilities
        
        # User Journey Database integration
        journey_metrics = ['daily_conversions', 'conversion_rate']
        for metric in journey_metrics:
            if metric not in self.success_criteria.success_criteria:
                result["warnings"].append(f"Missing journey-related metric: {metric}")
        
        # Attribution Engine integration  
        attribution_metrics = ['overall_roas', 'profit_margin']
        for metric in attribution_metrics:
            if metric not in self.success_criteria.success_criteria:
                result["errors"].append(f"Missing attribution-related metric: {metric}")
                result["passed"] = False
        
        # RL Agent integration
        learning_metrics = ['model_accuracy', 'convergence_rate']
        for metric in learning_metrics:
            if metric not in self.success_criteria.success_criteria:
                result["warnings"].append(f"Missing learning-related metric: {metric}")
        
        # Budget Pacer integration
        budget_metrics = ['budget_utilization', 'cost_per_acquisition']
        for metric in budget_metrics:
            if metric not in self.success_criteria.success_criteria:
                result["warnings"].append(f"Missing budget-related metric: {metric}")
        
        return result
    
    def _validate_production_readiness(self) -> Dict[str, Any]:
        """Validate overall production readiness"""
        
        result = {"passed": True, "errors": [], "warnings": []}
        
        # Check that all business critical KPIs have proper monitoring
        business_critical = self.success_criteria.get_business_critical_criteria()
        
        high_risk_kpis = [name for name, criteria in business_critical.items() 
                         if criteria.revenue_impact > 5000]
        
        if len(high_risk_kpis) > 10:
            result["warnings"].append(
                f"Large number of high-risk KPIs ({len(high_risk_kpis)}) may create alert fatigue"
            )
        
        # Check alert configuration completeness
        if not self.config:
            result["errors"].append("Production configuration not available")
            result["passed"] = False
        
        # Validate monitoring intervals
        monitoring_config = self.config.get('gaelp_success_criteria_config', {}).get('monitoring_configuration', {})
        
        if monitoring_config:
            check_intervals = monitoring_config.get('check_intervals', {})
            if check_intervals.get('critical_kpis_seconds', 300) > 60:
                result["warnings"].append("Critical KPI check interval > 60 seconds may delay issue detection")
        
        # Check data retention policy
        if monitoring_config:
            retention = monitoring_config.get('data_retention', {})
            if retention.get('raw_metrics_days', 30) < 90:
                result["warnings"].append("Raw metrics retention < 90 days may limit analysis")
        
        # Overall readiness assessment
        if len(self.critical_failures) == 0 and len(business_critical) >= 5:
            logger.info("✅ System meets production readiness criteria")
        else:
            result["errors"].append("System not ready for production deployment")
            result["passed"] = False
        
        return result
    
    def _generate_validation_report(self, all_passed: bool):
        """Generate comprehensive validation report"""
        
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION REPORT SUMMARY")
        logger.info("=" * 60)
        
        total_kpis = len(self.success_criteria.success_criteria)
        business_critical_count = len(self.success_criteria.get_business_critical_criteria())
        
        logger.info(f"Total KPIs Defined: {total_kpis}")
        logger.info(f"Business Critical KPIs: {business_critical_count}")
        
        # Category breakdown
        category_counts = {}
        for criteria in self.success_criteria.success_criteria.values():
            category_counts[criteria.category.value] = category_counts.get(criteria.category.value, 0) + 1
        
        logger.info("\nKPI Distribution by Category:")
        for category, count in category_counts.items():
            logger.info(f"  • {category}: {count}")
        
        # Validation results summary
        passed_count = sum(1 for result in self.validation_results.values() if result["passed"])
        total_tests = len(self.validation_results)
        
        logger.info(f"\nValidation Tests: {passed_count}/{total_tests} passed")
        
        if self.critical_failures:
            logger.error(f"\nCRITICAL FAILURES ({len(self.critical_failures)}):")
            for i, failure in enumerate(self.critical_failures, 1):
                logger.error(f"  {i}. {failure}")
        
        # Final verdict
        if all_passed:
            logger.info("\n🎉 SUCCESS: All validation tests passed!")
            logger.info("✅ System is ready for production deployment")
        else:
            logger.error("\n❌ VALIDATION FAILED")
            logger.error("🚫 System is NOT ready for production")
            logger.error(f"   {len(self.critical_failures)} critical issues must be resolved")
        
        logger.info("=" * 60)
        
        # Save detailed report
        report_path = "/home/hariravichandran/AELP/success_criteria_validation_report.json"
        detailed_report = {
            "timestamp": datetime.now().isoformat(),
            "validation_passed": all_passed,
            "total_kpis": total_kpis,
            "business_critical_kpis": business_critical_count,
            "category_distribution": category_counts,
            "validation_results": self.validation_results,
            "critical_failures": self.critical_failures,
            "tests_passed": f"{passed_count}/{total_tests}",
            "production_ready": all_passed
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(detailed_report, f, indent=2)
            logger.info(f"Detailed report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


def main():
    """Main validation execution"""
    
    logger.info("GAELP Success Criteria Validation System")
    logger.info("Validating all success criteria with NO FALLBACKS")
    
    validator = SuccessCriteriaValidator()
    
    # Run complete validation
    success = validator.run_full_validation()
    
    # Exit with appropriate code
    if success:
        logger.info("✅ Validation completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Validation failed - production deployment blocked")
        sys.exit(1)


if __name__ == "__main__":
    main()