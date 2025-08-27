"""
Integration module for GAELP Safety Framework
Provides easy integration with other GAELP components and external systems.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from datetime import datetime, timedelta

from .safety_orchestrator import ComprehensiveSafetyOrchestrator, SafetyConfiguration
from .budget_controls import BudgetLimits, SpendRecord
from .content_safety import ContentItem, ContentType
from .performance_safety import PerformanceDataPoint, PerformanceMetric
from .operational_safety import EmergencyLevel
from .agent_behavior_safety import AgentAction, ActionType

logger = logging.getLogger(__name__)


class GAELPSafetyIntegration:
    """
    Main integration class for GAELP Safety Framework.
    Provides simplified API for other GAELP components.
    """
    
    def __init__(self, config: SafetyConfiguration = None):
        self.orchestrator = ComprehensiveSafetyOrchestrator(config)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the safety framework"""
        if self._initialized:
            return
        
        await self.orchestrator.start_monitoring()
        self._initialized = True
        logger.info("GAELP Safety Framework initialized")
    
    async def shutdown(self):
        """Shutdown the safety framework"""
        await self.orchestrator.stop_monitoring()
        self._initialized = False
        logger.info("GAELP Safety Framework shutdown")
    
    # Campaign Safety API
    
    async def validate_new_campaign(self, campaign_data: Dict[str, Any], 
                                  creator_id: str) -> Dict[str, Any]:
        """
        Validate a new campaign for safety compliance.
        
        Args:
            campaign_data: Campaign configuration
            creator_id: ID of the user/agent creating the campaign
            
        Returns:
            Dict with validation results
        """
        try:
            is_valid, violations = await self.orchestrator.validate_campaign_creation(
                campaign_data, creator_id
            )
            
            return {
                'valid': is_valid,
                'violations': violations,
                'recommendations': self._generate_recommendations(violations),
                'approved_for_deployment': is_valid
            }
        except Exception as e:
            logger.error(f"Campaign validation failed: {e}")
            return {
                'valid': False,
                'violations': [f"System error: {str(e)}"],
                'approved_for_deployment': False
            }
    
    async def register_campaign_budget(self, campaign_id: str, 
                                     daily_limit: float, weekly_limit: float,
                                     monthly_limit: float, total_limit: float) -> bool:
        """Register campaign with budget controls"""
        try:
            if not self.orchestrator.budget_controller:
                logger.warning("Budget controls not enabled")
                return False
            
            limits = BudgetLimits(
                daily_limit=Decimal(str(daily_limit)),
                weekly_limit=Decimal(str(weekly_limit)),
                monthly_limit=Decimal(str(monthly_limit)),
                campaign_limit=Decimal(str(total_limit))
            )
            
            return await self.orchestrator.budget_controller.register_campaign(campaign_id, limits)
        except Exception as e:
            logger.error(f"Budget registration failed: {e}")
            return False
    
    async def record_campaign_spend(self, campaign_id: str, amount: float,
                                  platform: str, transaction_id: str,
                                  description: str = "") -> Dict[str, Any]:
        """Record campaign spending"""
        try:
            if not self.orchestrator.budget_controller:
                return {'success': False, 'error': 'Budget controls not enabled'}
            
            spend = SpendRecord(
                campaign_id=campaign_id,
                amount=Decimal(str(amount)),
                timestamp=datetime.utcnow(),
                platform=platform,
                transaction_id=transaction_id,
                description=description
            )
            
            success, violation = await self.orchestrator.budget_controller.record_spend(spend)
            
            return {
                'success': success,
                'violation': violation.description if violation else None,
                'campaign_paused': violation is not None
            }
        except Exception as e:
            logger.error(f"Spend recording failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Content Safety API
    
    async def moderate_ad_content(self, content: str, content_type: str,
                                campaign_id: str, platform: str = None) -> Dict[str, Any]:
        """Moderate ad content for safety"""
        try:
            if not self.orchestrator.content_safety:
                return {'approved': True, 'message': 'Content safety not enabled'}
            
            content_item = ContentItem(
                content_id=f"{campaign_id}_{content_type}_{datetime.utcnow().timestamp()}",
                content_type=ContentType(content_type.lower()),
                content=content,
                campaign_id=campaign_id
            )
            
            is_approved, violations = await self.orchestrator.content_safety.moderate_content(
                content_item, platform
            )
            
            return {
                'approved': is_approved,
                'violations': [
                    {
                        'type': v.violation_type.value,
                        'severity': v.severity.value,
                        'description': v.description,
                        'suggestions': v.remediation_suggestions
                    }
                    for v in violations
                ],
                'safe_for_all_audiences': is_approved and len(violations) == 0
            }
        except Exception as e:
            logger.error(f"Content moderation failed: {e}")
            return {'approved': False, 'error': str(e)}
    
    # Performance Monitoring API
    
    async def report_campaign_performance(self, campaign_id: str, 
                                        metrics: Dict[str, float]) -> Dict[str, Any]:
        """Report campaign performance metrics"""
        try:
            if not self.orchestrator.performance_safety:
                return {'processed': False, 'message': 'Performance safety not enabled'}
            
            results = []
            for metric_name, value in metrics.items():
                try:
                    # Map metric name to enum
                    metric_enum = getattr(PerformanceMetric, metric_name.upper())
                    
                    data_point = PerformanceDataPoint(
                        campaign_id=campaign_id,
                        metric=metric_enum,
                        value=value,
                        timestamp=datetime.utcnow()
                    )
                    
                    result = await self.orchestrator.performance_safety.process_performance_data(data_point)
                    results.append(result)
                except AttributeError:
                    logger.warning(f"Unknown metric: {metric_name}")
            
            # Check for any anomalies
            anomalies_detected = any(r.get('anomaly_detected', False) for r in results)
            
            return {
                'processed': True,
                'anomalies_detected': anomalies_detected,
                'results': results,
                'recommendations': self._generate_performance_recommendations(results)
            }
        except Exception as e:
            logger.error(f"Performance reporting failed: {e}")
            return {'processed': False, 'error': str(e)}
    
    # Agent Action Validation API
    
    async def validate_agent_action(self, agent_id: str, action_type: str,
                                   parameters: Dict[str, Any],
                                   campaign_id: str = None) -> Dict[str, Any]:
        """Validate agent action for safety compliance"""
        try:
            if not self.orchestrator.behavior_safety:
                return {'allowed': True, 'message': 'Behavior safety not enabled'}
            
            action = AgentAction(
                action_id=f"{agent_id}_{action_type}_{datetime.utcnow().timestamp()}",
                agent_id=agent_id,
                action_type=ActionType(action_type.lower()),
                parameters=parameters,
                timestamp=datetime.utcnow(),
                campaign_id=campaign_id
            )
            
            result = await self.orchestrator.behavior_safety.validate_and_execute_action(action)
            
            return {
                'allowed': result['action_allowed'],
                'executed': result['action_executed'],
                'violations': result['violations'],
                'interventions': result['interventions'],
                'warnings': result['warnings']
            }
        except Exception as e:
            logger.error(f"Action validation failed: {e}")
            return {'allowed': False, 'error': str(e)}
    
    # Emergency Controls API
    
    async def emergency_pause_campaign(self, campaign_id: str, reason: str,
                                     initiated_by: str) -> Dict[str, Any]:
        """Emergency pause a specific campaign"""
        try:
            success = False
            
            # Pause via budget controller
            if self.orchestrator.budget_controller:
                success = await self.orchestrator.budget_controller.pause_campaign(
                    campaign_id, reason
                )
            
            return {
                'success': success,
                'campaign_id': campaign_id,
                'reason': reason,
                'initiated_by': initiated_by,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Emergency pause failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def emergency_stop_all_campaigns(self, reason: str, 
                                         initiated_by: str) -> Dict[str, Any]:
        """Emergency stop all campaigns"""
        try:
            stop_id = await self.orchestrator.emergency_stop_system(
                reason, initiated_by, EmergencyLevel.HIGH
            )
            
            return {
                'success': bool(stop_id),
                'stop_id': stop_id,
                'reason': reason,
                'initiated_by': initiated_by,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # Monitoring and Reporting API
    
    def get_safety_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive safety dashboard"""
        return self.orchestrator.get_comprehensive_safety_status()
    
    def get_campaign_safety_report(self, campaign_id: str) -> Dict[str, Any]:
        """Get safety report for specific campaign"""
        return self.orchestrator.get_campaign_safety_report(campaign_id)
    
    def get_agent_safety_status(self, agent_id: str) -> Dict[str, Any]:
        """Get safety status for specific agent"""
        if self.orchestrator.behavior_safety:
            return asyncio.create_task(
                self.orchestrator.behavior_safety.get_agent_safety_status(agent_id)
            )
        return {'status': 'behavior_safety_disabled'}
    
    # Configuration and Management API
    
    def register_alert_webhook(self, webhook_url: str, webhook_secret: str = None):
        """Register webhook for safety alerts"""
        async def webhook_callback(event):
            # This would send HTTP webhook
            logger.info(f"Webhook alert: {event}")
        
        self.orchestrator.register_alert_callback(webhook_callback)
    
    def register_human_review_callback(self, callback: Callable):
        """Register callback for human review requests"""
        self.orchestrator.register_human_review_callback(callback)
    
    # Utility methods
    
    def _generate_recommendations(self, violations: List[str]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        for violation in violations:
            if 'budget' in violation.lower():
                recommendations.append("Consider reducing campaign budget or extending timeline")
            elif 'content' in violation.lower():
                recommendations.append("Review and modify ad content to comply with platform policies")
            elif 'targeting' in violation.lower():
                recommendations.append("Adjust targeting parameters to avoid discriminatory practices")
            elif 'performance' in violation.lower():
                recommendations.append("Monitor campaign performance and adjust optimization strategy")
        
        return recommendations
    
    def _generate_performance_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for result in results:
            if result.get('anomaly_detected'):
                recommendations.append("Performance anomaly detected - consider campaign review")
            if result.get('reward_clipped'):
                recommendations.append("Reward values were clipped - review optimization bounds")
        
        return recommendations


# Middleware helpers for different GAELP components

class EnvironmentRegistryMiddleware:
    """Safety middleware for Environment Registry"""
    
    def __init__(self, safety_integration: GAELPSafetyIntegration):
        self.safety = safety_integration
    
    async def validate_environment_submission(self, env_data: Dict[str, Any],
                                            submitter_id: str) -> Dict[str, Any]:
        """Validate environment submission for safety"""
        # Check if environment involves real money or user data
        if env_data.get('real_money', False):
            # Treat as campaign validation
            return await self.safety.validate_new_campaign(env_data, submitter_id)
        else:
            # Basic safety checks for simulation environments
            return {'valid': True, 'violations': []}


class TrainingOrchestratorMiddleware:
    """Safety middleware for Training Orchestrator"""
    
    def __init__(self, safety_integration: GAELPSafetyIntegration):
        self.safety = safety_integration
    
    async def validate_training_action(self, agent_id: str, action_type: str,
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training action"""
        return await self.safety.validate_agent_action(agent_id, action_type, parameters)
    
    async def report_training_metrics(self, agent_id: str, episode_metrics: Dict[str, Any]):
        """Report training metrics for monitoring"""
        # Convert training metrics to performance metrics
        performance_metrics = {}
        for key, value in episode_metrics.items():
            if key in ['reward', 'success_rate', 'efficiency']:
                performance_metrics[key] = value
        
        if performance_metrics:
            return await self.safety.report_campaign_performance(agent_id, performance_metrics)
        return {'processed': False, 'message': 'No relevant metrics'}


class MCPIntegrationMiddleware:
    """Safety middleware for MCP Integration"""
    
    def __init__(self, safety_integration: GAELPSafetyIntegration):
        self.safety = safety_integration
    
    async def validate_external_api_call(self, api_name: str, parameters: Dict[str, Any],
                                       campaign_id: str = None) -> Dict[str, Any]:
        """Validate external API calls"""
        # Check for spending actions
        if api_name in ['google_ads_api', 'facebook_ads_api', 'microsoft_ads_api']:
            if 'budget' in parameters or 'bid' in parameters:
                # Treat as agent action
                return await self.safety.validate_agent_action(
                    f"api_{api_name}", 'set_budget', parameters, campaign_id
                )
        
        return {'allowed': True, 'message': 'API call validated'}


# Factory function for easy setup

def create_gaelp_safety_integration(config: Dict[str, Any] = None) -> GAELPSafetyIntegration:
    """
    Factory function to create GAELP Safety Integration with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured GAELPSafetyIntegration instance
    """
    safety_config = SafetyConfiguration()
    
    if config:
        # Override default configuration
        for key, value in config.items():
            if hasattr(safety_config, key):
                setattr(safety_config, key, value)
    
    return GAELPSafetyIntegration(safety_config)


# Example usage and integration patterns

async def example_integration():
    """Example of how to integrate safety framework"""
    
    # Create safety integration
    safety = create_gaelp_safety_integration({
        'max_daily_budget': 5000.0,
        'auto_pause_on_critical': True,
        'human_review_required': True
    })
    
    # Initialize
    await safety.initialize()
    
    try:
        # Validate new campaign
        campaign_data = {
            'id': 'campaign_123',
            'title': 'Test Ad Campaign',
            'budget': 1000.0,
            'targeting': {'age_range': {'min': 18, 'max': 65}},
            'platform': 'google_ads'
        }
        
        validation_result = await safety.validate_new_campaign(campaign_data, 'user_123')
        print(f"Campaign validation: {validation_result}")
        
        # Register budget if validated
        if validation_result['valid']:
            budget_registered = await safety.register_campaign_budget(
                'campaign_123', 100.0, 700.0, 3000.0, 1000.0
            )
            print(f"Budget registered: {budget_registered}")
        
        # Record some spending
        spend_result = await safety.record_campaign_spend(
            'campaign_123', 50.0, 'google_ads', 'txn_123', 'Ad spend'
        )
        print(f"Spend recorded: {spend_result}")
        
        # Get safety dashboard
        dashboard = safety.get_safety_dashboard()
        print(f"Safety status: {dashboard['overall_status']}")
        
    finally:
        # Shutdown
        await safety.shutdown()


if __name__ == "__main__":
    asyncio.run(example_integration())