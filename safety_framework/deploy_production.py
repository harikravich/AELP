#!/usr/bin/env python3
"""
Production Deployment Script for GAELP Safety Framework
Deploys production-ready safety systems with real money controls.
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, Any
from datetime import datetime

from production_integration import ProductionSafetyOrchestrator, ProductionSafetyConfig
from production_budget_controls import PaymentMethod
from emergency_controls import EmergencyContact, ComplianceRegion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDeployment:
    """Handles production deployment of GAELP safety systems"""
    
    def __init__(self):
        self.config = None
        self.orchestrator = None
        
    def load_configuration(self) -> ProductionSafetyConfig:
        """Load production configuration from environment and config files"""
        
        # Load from environment variables
        config = ProductionSafetyConfig(
            environment=os.getenv('GAELP_ENVIRONMENT', 'production'),
            
            # Financial controls
            stripe_api_key=os.getenv('STRIPE_API_KEY', ''),
            max_daily_global_spend=float(os.getenv('MAX_DAILY_GLOBAL_SPEND', '1000000.0')),
            fraud_detection_threshold=float(os.getenv('FRAUD_THRESHOLD', '0.7')),
            
            # AI services
            openai_api_key=os.getenv('OPENAI_API_KEY', ''),
            perspective_api_key=os.getenv('PERSPECTIVE_API_KEY', ''),
            content_moderation_strict=os.getenv('CONTENT_MODERATION_STRICT', 'true').lower() == 'true',
            
            # Cloud services
            gcp_project_id=os.getenv('GOOGLE_CLOUD_PROJECT', ''),
            bigquery_dataset=os.getenv('BIGQUERY_DATASET', 'safety_audit'),
            
            # Monitoring and alerting
            prometheus_enabled=os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true',
            slack_bot_token=os.getenv('SLACK_BOT_TOKEN', ''),
            pagerduty_integration_key=os.getenv('PAGERDUTY_INTEGRATION_KEY', ''),
            
            # Compliance
            gdpr_enabled=os.getenv('GDPR_ENABLED', 'true').lower() == 'true',
            ccpa_enabled=os.getenv('CCPA_ENABLED', 'true').lower() == 'true',
            coppa_enabled=os.getenv('COPPA_ENABLED', 'true').lower() == 'true',
            regulatory_notifications_enabled=os.getenv('REGULATORY_NOTIFICATIONS', 'true').lower() == 'true',
            
            # Feature flags
            enable_real_money_controls=os.getenv('ENABLE_REAL_MONEY', 'true').lower() == 'true',
            enable_ai_content_moderation=os.getenv('ENABLE_AI_MODERATION', 'true').lower() == 'true',
            enable_emergency_stops=os.getenv('ENABLE_EMERGENCY_STOPS', 'true').lower() == 'true',
            enable_real_time_monitoring=os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
            
            # Emergency contacts
            emergency_contacts=self._load_emergency_contacts()
        )
        
        return config
    
    def _load_emergency_contacts(self) -> list:
        """Load emergency contacts from configuration"""
        # This would typically load from a secure configuration file
        # For demo, using environment variables
        contacts = []
        
        # Load primary emergency contact
        if os.getenv('EMERGENCY_CONTACT_1_EMAIL'):
            contacts.append({
                'contact_id': 'primary_emergency',
                'name': os.getenv('EMERGENCY_CONTACT_1_NAME', 'Primary Emergency Contact'),
                'role': 'Safety Engineer',
                'email': os.getenv('EMERGENCY_CONTACT_1_EMAIL'),
                'phone': os.getenv('EMERGENCY_CONTACT_1_PHONE', ''),
                'slack_user_id': os.getenv('EMERGENCY_CONTACT_1_SLACK', ''),
                'escalation_level': 1,
                'regions': ['US', 'EU', 'UK']
            })
        
        # Load secondary emergency contact
        if os.getenv('EMERGENCY_CONTACT_2_EMAIL'):
            contacts.append({
                'contact_id': 'secondary_emergency',
                'name': os.getenv('EMERGENCY_CONTACT_2_NAME', 'Secondary Emergency Contact'),
                'role': 'Engineering Manager',
                'email': os.getenv('EMERGENCY_CONTACT_2_EMAIL'),
                'phone': os.getenv('EMERGENCY_CONTACT_2_PHONE', ''),
                'slack_user_id': os.getenv('EMERGENCY_CONTACT_2_SLACK', ''),
                'escalation_level': 2,
                'regions': ['US', 'EU', 'UK']
            })
        
        # Load executive emergency contact
        if os.getenv('EMERGENCY_CONTACT_EXEC_EMAIL'):
            contacts.append({
                'contact_id': 'executive_emergency',
                'name': os.getenv('EMERGENCY_CONTACT_EXEC_NAME', 'Executive Emergency Contact'),
                'role': 'CTO',
                'email': os.getenv('EMERGENCY_CONTACT_EXEC_EMAIL'),
                'phone': os.getenv('EMERGENCY_CONTACT_EXEC_PHONE', ''),
                'escalation_level': 4,
                'regions': ['US', 'EU', 'UK']
            })
        
        return contacts
    
    def validate_configuration(self, config: ProductionSafetyConfig) -> bool:
        """Validate production configuration"""
        issues = []
        
        # Critical validations for production
        if config.environment == 'production':
            if not config.stripe_api_key:
                issues.append("Stripe API key is required for production")
            
            if not config.gcp_project_id:
                issues.append("GCP project ID is required for production")
            
            if not config.emergency_contacts:
                issues.append("Emergency contacts are required for production")
            
            if not config.slack_bot_token and not config.pagerduty_integration_key:
                issues.append("At least one alerting channel (Slack or PagerDuty) is required")
            
            if config.max_daily_global_spend < 100000:  # $100k minimum for production
                issues.append("Production daily spend limit seems too low")
        
        # AI service validations
        if config.enable_ai_content_moderation and not config.openai_api_key:
            issues.append("OpenAI API key required for AI content moderation")
        
        # Compliance validations
        if config.gdpr_enabled and not config.regulatory_notifications_enabled:
            issues.append("Regulatory notifications should be enabled for GDPR compliance")
        
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    async def initialize_payment_methods(self, orchestrator: ProductionSafetyOrchestrator):
        """Initialize production payment methods"""
        logger.info("Initializing production payment methods...")
        
        # This would typically load from secure configuration
        # For demo, using environment variables for primary payment method
        primary_payment_config = {
            'payment_id': 'primary_production_card',
            'method_type': 'credit_card',
            'stripe_payment_method_id': os.getenv('PRIMARY_STRIPE_PAYMENT_METHOD_ID', ''),
            'last_four': os.getenv('PRIMARY_CARD_LAST_FOUR', '****'),
            'is_active': True,
            'daily_limit': float(os.getenv('PRIMARY_CARD_DAILY_LIMIT', '100000.0')),
            'monthly_limit': float(os.getenv('PRIMARY_CARD_MONTHLY_LIMIT', '2000000.0')),
            'risk_level': 'low',
            'country_code': 'US',
            'verification_status': 'verified'
        }
        
        if primary_payment_config['stripe_payment_method_id']:
            primary_payment = PaymentMethod(**primary_payment_config)
            
            success = await orchestrator.budget_controller.register_payment_method(primary_payment)
            if success:
                logger.info("Primary payment method registered successfully")
            else:
                logger.error("Failed to register primary payment method")
                return False
        
        # Register backup payment method if configured
        backup_payment_config = {
            'payment_id': 'backup_production_card',
            'method_type': 'credit_card',
            'stripe_payment_method_id': os.getenv('BACKUP_STRIPE_PAYMENT_METHOD_ID', ''),
            'last_four': os.getenv('BACKUP_CARD_LAST_FOUR', '****'),
            'is_active': True,
            'daily_limit': float(os.getenv('BACKUP_CARD_DAILY_LIMIT', '50000.0')),
            'monthly_limit': float(os.getenv('BACKUP_CARD_MONTHLY_LIMIT', '1000000.0')),
            'risk_level': 'medium',
            'country_code': 'US',
            'verification_status': 'verified'
        }
        
        if backup_payment_config['stripe_payment_method_id']:
            backup_payment = PaymentMethod(**backup_payment_config)
            
            success = await orchestrator.budget_controller.register_payment_method(backup_payment)
            if success:
                logger.info("Backup payment method registered successfully")
            else:
                logger.warning("Failed to register backup payment method")
        
        return True
    
    async def run_production_health_checks(self, orchestrator: ProductionSafetyOrchestrator) -> bool:
        """Run comprehensive production health checks"""
        logger.info("Running production health checks...")
        
        # Get system status
        status = await orchestrator.get_system_status()
        
        # Check overall system health
        if status['system']['status'] != 'operational':
            logger.error(f"System status is {status['system']['status']}, expected 'operational'")
            return False
        
        # Check component health
        components = status['components']
        for component, health in components.items():
            if not health['operational']:
                logger.error(f"Component {component} is not operational")
                return False
        
        # Check critical thresholds
        if status['system']['emergency_active']:
            logger.error("System is in emergency mode")
            return False
        
        if status['system']['master_kill_switch']:
            logger.error("Master kill switch is active")
            return False
        
        logger.info("All production health checks passed")
        return True
    
    async def deploy_to_production(self) -> bool:
        """Deploy safety systems to production"""
        try:
            logger.info("=" * 60)
            logger.info("GAELP PRODUCTION SAFETY SYSTEM DEPLOYMENT")
            logger.info("=" * 60)
            
            # Load and validate configuration
            logger.info("Loading production configuration...")
            self.config = self.load_configuration()
            
            if not self.validate_configuration(self.config):
                logger.error("Configuration validation failed")
                return False
            
            # Initialize production orchestrator
            logger.info("Initializing production safety orchestrator...")
            self.orchestrator = ProductionSafetyOrchestrator(self.config)
            
            # Initialize all safety systems
            logger.info("Initializing safety systems...")
            init_success = await self.orchestrator.initialize()
            if not init_success:
                logger.error("Safety system initialization failed")
                return False
            
            # Initialize payment methods
            if self.config.enable_real_money_controls:
                payment_success = await self.initialize_payment_methods(self.orchestrator)
                if not payment_success:
                    logger.error("Payment method initialization failed")
                    return False
            
            # Run production health checks
            health_success = await self.run_production_health_checks(self.orchestrator)
            if not health_success:
                logger.error("Production health checks failed")
                return False
            
            # Final status report
            status = await self.orchestrator.get_system_status()
            logger.info("=" * 60)
            logger.info("PRODUCTION DEPLOYMENT SUCCESSFUL")
            logger.info("=" * 60)
            logger.info(f"System Status: {status['system']['status']}")
            logger.info(f"Components Operational: {len([c for c in status['components'].values() if c['operational']])}/{len(status['components'])}")
            logger.info(f"Real Money Controls: {'ENABLED' if self.config.enable_real_money_controls else 'DISABLED'}")
            logger.info(f"AI Content Moderation: {'ENABLED' if self.config.enable_ai_content_moderation else 'DISABLED'}")
            logger.info(f"Emergency Controls: {'ENABLED' if self.config.enable_emergency_stops else 'DISABLED'}")
            logger.info(f"Real-time Monitoring: {'ENABLED' if self.config.enable_real_time_monitoring else 'DISABLED'}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return False
    
    async def test_emergency_systems(self):
        """Test emergency systems in safe mode"""
        if not self.orchestrator:
            logger.error("Orchestrator not initialized")
            return False
        
        logger.info("Testing emergency systems (safe mode)...")
        
        try:
            # Test low-level emergency stop
            from emergency_controls import EmergencyLevel
            stop_id = await self.orchestrator.trigger_emergency_stop(
                level=EmergencyLevel.LOW,
                reason="Production deployment test",
                triggered_by="deployment_script",
                context={'test_mode': True}
            )
            
            if stop_id:
                logger.info(f"Emergency stop test successful: {stop_id}")
                return True
            else:
                logger.error("Emergency stop test failed")
                return False
                
        except Exception as e:
            logger.error(f"Emergency system test failed: {e}")
            return False
    
    def create_monitoring_dashboard_config(self) -> Dict[str, Any]:
        """Create monitoring dashboard configuration"""
        return {
            "dashboard": {
                "title": "GAELP Production Safety Dashboard",
                "panels": [
                    {
                        "title": "System Health Score",
                        "type": "gauge",
                        "metric": "gaelp_system_health_score",
                        "thresholds": [0.7, 0.9]
                    },
                    {
                        "title": "Budget Violations",
                        "type": "graph",
                        "metric": "rate(gaelp_budget_violations_total[5m])",
                        "alert_threshold": 5.0
                    },
                    {
                        "title": "Content Moderation Rate",
                        "type": "graph",
                        "metric": "rate(gaelp_content_moderation_total[5m])"
                    },
                    {
                        "title": "Emergency Stops",
                        "type": "stat",
                        "metric": "gaelp_emergency_stops_total"
                    },
                    {
                        "title": "Fraud Risk Score",
                        "type": "gauge",
                        "metric": "max(gaelp_fraud_risk_score)",
                        "thresholds": [0.5, 0.8]
                    }
                ]
            },
            "alerts": [
                {
                    "name": "High Budget Violation Rate",
                    "condition": "rate(gaelp_budget_violations_total[5m]) > 5",
                    "severity": "warning"
                },
                {
                    "name": "Emergency Stop Triggered",
                    "condition": "increase(gaelp_emergency_stops_total[1m]) > 0",
                    "severity": "critical"
                },
                {
                    "name": "System Health Low",
                    "condition": "gaelp_system_health_score < 0.7",
                    "severity": "warning"
                }
            ]
        }


async def main():
    """Main deployment function"""
    deployment = ProductionDeployment()
    
    # Deploy to production
    success = await deployment.deploy_to_production()
    
    if success:
        logger.info("Testing emergency systems...")
        test_success = await deployment.test_emergency_systems()
        
        if test_success:
            logger.info("Creating monitoring configuration...")
            dashboard_config = deployment.create_monitoring_dashboard_config()
            
            # Save dashboard configuration
            with open('production_dashboard_config.json', 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            logger.info("Dashboard configuration saved to production_dashboard_config.json")
            
            logger.info("🎉 PRODUCTION DEPLOYMENT COMPLETE! 🎉")
            logger.info("GAELP Safety Framework is now operational with:")
            logger.info("✅ Real money controls")
            logger.info("✅ AI-powered content safety")
            logger.info("✅ Emergency stop systems")
            logger.info("✅ Regulatory compliance")
            logger.info("✅ Real-time monitoring")
            
            # Keep the system running
            logger.info("System is now running. Press Ctrl+C to shutdown.")
            try:
                while True:
                    await asyncio.sleep(60)
                    status = await deployment.orchestrator.get_system_status()
                    logger.info(f"System Status: {status['system']['status']} | "
                              f"Validations: {status['metrics']['validations_processed']} | "
                              f"Uptime: {status['system']['uptime_seconds']:.0f}s")
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                return True
        else:
            logger.error("Emergency system testing failed")
            return False
    else:
        logger.error("Production deployment failed")
        return False


if __name__ == "__main__":
    # Set up environment
    if len(sys.argv) > 1 and sys.argv[1] == "--check-config":
        # Configuration check mode
        deployment = ProductionDeployment()
        config = deployment.load_configuration()
        valid = deployment.validate_configuration(config)
        sys.exit(0 if valid else 1)
    
    # Run deployment
    success = asyncio.run(main())
    sys.exit(0 if success else 1)