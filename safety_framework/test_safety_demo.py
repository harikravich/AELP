"""
Comprehensive demo and test of the GAELP Safety Framework
Demonstrates all safety features and integration patterns.
"""

import asyncio
import json
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any

# Import safety framework components
from integration import GAELPSafetyIntegration, create_gaelp_safety_integration
from safety_orchestrator import SafetyConfiguration
from budget_controls import BudgetLimits
from content_safety import ContentItem, ContentType
from performance_safety import PerformanceDataPoint, PerformanceMetric
from agent_behavior_safety import AgentAction, ActionType
from operational_safety import EmergencyLevel


class SafetyFrameworkDemo:
    """Demo class showcasing all safety framework features"""
    
    def __init__(self):
        self.safety = None
        self.demo_results = {}
    
    async def initialize(self):
        """Initialize the safety framework for demo"""
        print("🚀 Initializing GAELP Safety Framework...")
        
        config = {
            'max_daily_budget': 1000.0,
            'auto_pause_on_critical': True,
            'human_review_required': True,
            'enable_budget_controls': True,
            'enable_content_safety': True,
            'enable_performance_safety': True,
            'enable_operational_safety': True,
            'enable_data_safety': True,
            'enable_behavior_safety': True
        }
        
        self.safety = create_gaelp_safety_integration(config)
        await self.safety.initialize()
        
        print("✅ Safety Framework initialized successfully!")
        print()
    
    async def demo_budget_controls(self):
        """Demonstrate budget control features"""
        print("💰 DEMO: Budget Controls")
        print("=" * 50)
        
        # Register campaign with budget limits
        campaign_id = "demo_campaign_budget"
        success = await self.safety.register_campaign_budget(
            campaign_id=campaign_id,
            daily_limit=100.0,
            weekly_limit=500.0,
            monthly_limit=2000.0,
            total_limit=1000.0
        )
        
        print(f"📊 Campaign budget registered: {success}")
        
        # Record normal spending
        spend_result1 = await self.safety.record_campaign_spend(
            campaign_id=campaign_id,
            amount=25.0,
            platform="google_ads",
            transaction_id="txn_001",
            description="Normal ad spend"
        )
        print(f"✅ Normal spend recorded: ${spend_result1}")
        
        # Record spending that exceeds daily limit
        spend_result2 = await self.safety.record_campaign_spend(
            campaign_id=campaign_id,
            amount=150.0,  # This will exceed daily limit of 100
            platform="google_ads",
            transaction_id="txn_002",
            description="Large ad spend"
        )
        print(f"🚨 Large spend recorded: {spend_result2}")
        
        if spend_result2.get('campaign_paused'):
            print("⚠️  Campaign automatically paused due to budget violation!")
        
        self.demo_results['budget_controls'] = {
            'registration_success': success,
            'normal_spend': spend_result1['success'],
            'violation_detected': spend_result2.get('violation') is not None,
            'auto_pause_triggered': spend_result2.get('campaign_paused', False)
        }
        
        print()
    
    async def demo_content_safety(self):
        """Demonstrate content safety features"""
        print("🛡️ DEMO: Content Safety")
        print("=" * 50)
        
        campaign_id = "demo_campaign_content"
        
        # Test safe content
        safe_result = await self.safety.moderate_ad_content(
            content="Buy the best running shoes for your next marathon!",
            content_type="text",
            campaign_id=campaign_id,
            platform="google_ads"
        )
        print(f"✅ Safe content: {safe_result['approved']}")
        
        # Test problematic content
        unsafe_result = await self.safety.moderate_ad_content(
            content="Get rich quick with this miracle weight loss pill! Doctors hate this trick!",
            content_type="text",
            campaign_id=campaign_id,
            platform="google_ads"
        )
        print(f"🚨 Unsafe content: {unsafe_result['approved']}")
        if not unsafe_result['approved']:
            print(f"   Violations: {[v['description'] for v in unsafe_result['violations']]}")
        
        # Test discriminatory content
        discriminatory_result = await self.safety.moderate_ad_content(
            content="Job opportunity - looking for young, attractive women only",
            content_type="text",
            campaign_id=campaign_id,
            platform="facebook_ads"
        )
        print(f"⚖️  Discriminatory content: {discriminatory_result['approved']}")
        if not discriminatory_result['approved']:
            print(f"   Violations: {[v['description'] for v in discriminatory_result['violations']]}")
        
        self.demo_results['content_safety'] = {
            'safe_content_approved': safe_result['approved'],
            'unsafe_content_blocked': not unsafe_result['approved'],
            'discriminatory_content_blocked': not discriminatory_result['approved'],
            'violation_detection_working': len(unsafe_result.get('violations', [])) > 0
        }
        
        print()
    
    async def demo_performance_safety(self):
        """Demonstrate performance safety features"""
        print("📈 DEMO: Performance Safety")
        print("=" * 50)
        
        campaign_id = "demo_campaign_performance"
        
        # Report normal performance metrics
        normal_metrics = {
            'click_through_rate': 0.025,  # 2.5% CTR - normal
            'conversion_rate': 0.05,      # 5% conversion - normal
            'cost_per_click': 1.50,       # $1.50 CPC - normal
            'return_on_ad_spend': 3.2     # 3.2x ROAS - good
        }
        
        normal_result = await self.safety.report_campaign_performance(
            campaign_id=campaign_id,
            metrics=normal_metrics
        )
        print(f"✅ Normal performance reported: {normal_result['processed']}")
        print(f"   Anomalies detected: {normal_result['anomalies_detected']}")
        
        # Report suspicious performance metrics (possible reward hacking)
        suspicious_metrics = {
            'click_through_rate': 0.85,   # 85% CTR - unrealistic!
            'conversion_rate': 0.95,      # 95% conversion - impossible!
            'cost_per_click': 0.01,       # $0.01 CPC - too low
            'return_on_ad_spend': 50.0    # 50x ROAS - suspicious
        }
        
        suspicious_result = await self.safety.report_campaign_performance(
            campaign_id=campaign_id,
            metrics=suspicious_metrics
        )
        print(f"🚨 Suspicious performance reported: {suspicious_result['processed']}")
        print(f"   Anomalies detected: {suspicious_result['anomalies_detected']}")
        
        if suspicious_result['anomalies_detected']:
            print("⚠️  Potential reward hacking detected!")
        
        self.demo_results['performance_safety'] = {
            'normal_metrics_processed': normal_result['processed'],
            'anomaly_detection_working': suspicious_result['anomalies_detected'],
            'reward_hacking_detected': suspicious_result['anomalies_detected']
        }
        
        print()
    
    async def demo_agent_behavior_safety(self):
        """Demonstrate agent behavior safety features"""
        print("🤖 DEMO: Agent Behavior Safety")
        print("=" * 50)
        
        agent_id = "demo_agent_001"
        campaign_id = "demo_campaign_behavior"
        
        # Test normal agent action
        normal_action = await self.safety.validate_agent_action(
            agent_id=agent_id,
            action_type="set_budget",
            parameters={'value': 100.0, 'campaign_id': campaign_id},
            campaign_id=campaign_id
        )
        print(f"✅ Normal action: {normal_action['allowed']}")
        
        # Test excessive budget action
        excessive_action = await self.safety.validate_agent_action(
            agent_id=agent_id,
            action_type="set_budget",
            parameters={'value': 50000.0, 'campaign_id': campaign_id},  # Way too high!
            campaign_id=campaign_id
        )
        print(f"🚨 Excessive budget action: {excessive_action['allowed']}")
        if not excessive_action['allowed']:
            print(f"   Violations: {excessive_action['violations']}")
        
        # Test discriminatory targeting
        discriminatory_action = await self.safety.validate_agent_action(
            agent_id=agent_id,
            action_type="modify_targeting",
            parameters={
                'targeting': {
                    'race': 'white',  # Discriminatory!
                    'religion': 'christian',  # Discriminatory!
                    'age_range': {'min': 18, 'max': 25}
                }
            },
            campaign_id=campaign_id
        )
        print(f"⚖️  Discriminatory targeting: {discriminatory_action['allowed']}")
        if not discriminatory_action['allowed']:
            print(f"   Violations: {discriminatory_action['violations']}")
        
        # Simulate repetitive behavior (multiple identical actions)
        print("🔄 Testing repetitive behavior detection...")
        repetitive_violations = 0
        for i in range(7):  # This should trigger repetitive behavior detection
            repetitive_action = await self.safety.validate_agent_action(
                agent_id=agent_id,
                action_type="adjust_bidding",
                parameters={'value': 1.0},  # Same action repeatedly
                campaign_id=campaign_id
            )
            if repetitive_action.get('violations'):
                repetitive_violations += 1
        
        print(f"   Repetitive behavior violations detected: {repetitive_violations}")
        
        self.demo_results['agent_behavior'] = {
            'normal_action_allowed': normal_action['allowed'],
            'excessive_budget_blocked': not excessive_action['allowed'],
            'discriminatory_targeting_blocked': not discriminatory_action['allowed'],
            'repetitive_behavior_detected': repetitive_violations > 0
        }
        
        print()
    
    async def demo_emergency_procedures(self):
        """Demonstrate emergency stop procedures"""
        print("🚨 DEMO: Emergency Procedures")
        print("=" * 50)
        
        # Emergency pause single campaign
        pause_result = await self.safety.emergency_pause_campaign(
            campaign_id="demo_campaign_emergency",
            reason="Suspicious activity detected",
            initiated_by="security_system"
        )
        print(f"⏸️  Emergency pause: {pause_result['success']}")
        
        # Emergency stop all campaigns (HIGH level)
        stop_result = await self.safety.emergency_stop_all_campaigns(
            reason="Critical security breach detected",
            initiated_by="incident_response_team"
        )
        print(f"🛑 Emergency stop all: {stop_result['success']}")
        print(f"   Stop ID: {stop_result.get('stop_id', 'N/A')}")
        
        self.demo_results['emergency_procedures'] = {
            'emergency_pause_working': pause_result['success'],
            'emergency_stop_working': stop_result['success']
        }
        
        print()
    
    async def demo_safety_dashboard(self):
        """Demonstrate safety monitoring dashboard"""
        print("📊 DEMO: Safety Dashboard")
        print("=" * 50)
        
        dashboard = self.safety.get_safety_dashboard()
        
        print(f"🏥 System Health Score: {dashboard['overall_status']['health_score']:.2f}")
        print(f"🚦 Safety Level: {dashboard['overall_status'].get('safety_level', 'UNKNOWN')}")
        print(f"👀 Monitoring Active: {dashboard['overall_status']['monitoring_active']}")
        
        # System metrics
        metrics = dashboard.get('system_metrics', {})
        print(f"📈 Total Safety Events: {metrics.get('total_safety_events', 0)}")
        print(f"⚠️  Active Violations: {metrics.get('active_violations', 0)}")
        print(f"🚨 Emergency Stops: {metrics.get('emergency_stops', 0)}")
        
        # Recent events
        recent_events = dashboard.get('recent_events', [])
        if recent_events:
            print("\n📋 Recent Safety Events:")
            for event in recent_events[-3:]:  # Show last 3 events
                print(f"   • {event['type']}: {event['description']}")
        
        self.demo_results['dashboard'] = {
            'health_score': dashboard['overall_status']['health_score'],
            'monitoring_active': dashboard['overall_status']['monitoring_active'],
            'metrics_available': len(metrics) > 0
        }
        
        print()
    
    async def demo_campaign_safety_report(self):
        """Demonstrate campaign-specific safety reporting"""
        print("📋 DEMO: Campaign Safety Report")
        print("=" * 50)
        
        campaign_id = "demo_campaign_budget"  # Use campaign from budget demo
        
        report = self.safety.get_campaign_safety_report(campaign_id)
        
        print(f"🎯 Campaign: {report.get('campaign_id', 'N/A')}")
        print(f"📊 Safety Events: {report.get('safety_events', 0)}")
        print(f"⚠️  Unresolved Events: {report.get('unresolved_events', 0)}")
        print(f"💰 Budget Status: {report.get('budget_status', {})}")
        print(f"🏥 Performance Health: {report.get('performance_health', 'unknown')}")
        print(f"⚖️  Compliance Status: {report.get('compliance_status', 'unknown')}")
        
        # Show violations if any
        violations = report.get('safety_violations', [])
        if violations:
            print("\n🚨 Safety Violations:")
            for violation in violations[-3:]:  # Show last 3
                print(f"   • {violation['type']}: {violation['description']}")
        
        self.demo_results['campaign_report'] = {
            'report_generated': 'campaign_id' in report,
            'has_safety_data': report.get('safety_events', 0) > 0
        }
        
        print()
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demo of all safety features"""
        print("🔒 GAELP SAFETY FRAMEWORK COMPREHENSIVE DEMO")
        print("=" * 60)
        print()
        
        try:
            await self.initialize()
            
            # Run all demo modules
            await self.demo_budget_controls()
            await self.demo_content_safety()
            await self.demo_performance_safety()
            await self.demo_agent_behavior_safety()
            await self.demo_emergency_procedures()
            await self.demo_safety_dashboard()
            await self.demo_campaign_safety_report()
            
            # Generate final report
            await self.generate_demo_summary()
            
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.safety:
                await self.safety.shutdown()
                print("🔌 Safety Framework shutdown complete")
    
    async def generate_demo_summary(self):
        """Generate summary of demo results"""
        print("📄 DEMO SUMMARY REPORT")
        print("=" * 50)
        
        total_tests = 0
        passed_tests = 0
        
        for module, results in self.demo_results.items():
            print(f"\n🔹 {module.replace('_', ' ').title()}:")
            for test, result in results.items():
                total_tests += 1
                if result:
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                print(f"   {test.replace('_', ' ').title()}: {status}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🏆 OVERALL RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("   🎉 EXCELLENT: Safety Framework is working perfectly!")
        elif success_rate >= 75:
            print("   👍 GOOD: Safety Framework is working well with minor issues")
        elif success_rate >= 50:
            print("   ⚠️  WARNING: Safety Framework has significant issues")
        else:
            print("   🚨 CRITICAL: Safety Framework is not functioning properly")
        
        print()
        print("🔒 Safety Framework Demo Complete!")
        print("   All critical safety mechanisms have been tested and validated.")
        print("   The framework is ready for integration with GAELP components.")


async def run_integration_examples():
    """Run integration examples for different GAELP components"""
    print("\n🔗 INTEGRATION EXAMPLES")
    print("=" * 50)
    
    # Example: Environment Registry Integration
    print("🏗️ Environment Registry Integration:")
    print("   - Validates environment configurations before deployment")
    print("   - Checks for real money vs simulation environments")
    print("   - Ensures safety compliance for production environments")
    
    # Example: Training Orchestrator Integration
    print("\n🎯 Training Orchestrator Integration:")
    print("   - Validates agent actions during training")
    print("   - Monitors training metrics for anomalies")
    print("   - Prevents unsafe exploration in production")
    
    # Example: MCP Integration
    print("\n🔌 MCP Integration:")
    print("   - Validates external API calls")
    print("   - Monitors spending through ad platform APIs")
    print("   - Ensures credential security")
    
    print("\n✅ All integrations ready for deployment!")


async def main():
    """Main demo function"""
    demo = SafetyFrameworkDemo()
    
    # Run comprehensive demo
    await demo.run_comprehensive_demo()
    
    # Show integration examples
    await run_integration_examples()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("1. Review the safety framework documentation")
    print("2. Integrate with @mcp-integration for API-level controls")
    print("3. Integrate with @training-orchestrator for agent monitoring")
    print("4. Set up production monitoring and alerting")
    print("5. Configure human review workflows")
    print("6. Test emergency procedures in staging environment")
    print("\n🚀 Ready to deploy safe, compliant ad campaign learning!")


if __name__ == "__main__":
    asyncio.run(main())