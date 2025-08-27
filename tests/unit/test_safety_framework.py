"""
Unit tests for Safety Framework components.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import pytest

from safety_framework.budget_monitor import BudgetMonitor, BudgetViolation
from safety_framework.content_safety import ContentSafetyValidator, SafetyScore
from safety_framework.policy_enforcer import PolicyEnforcer, PolicyViolation
from safety_framework.emergency_stop import EmergencyStopController
from safety_framework.audit_logger import AuditLogger


class TestBudgetMonitor:
    """Test suite for BudgetMonitor functionality."""

    @pytest.fixture
    def budget_monitor(self, mock_redis):
        """Create BudgetMonitor instance with mocked dependencies."""
        with patch("safety_framework.budget_monitor.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            monitor = BudgetMonitor(
                redis_url="redis://localhost:6379",
                default_daily_limit=1000.0,
                default_total_limit=10000.0
            )
            return monitor

    @pytest.mark.unit
    async def test_set_budget_limits(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test setting budget limits for an agent."""
        limits = {
            "daily_limit": 500.0,
            "total_limit": 5000.0,
            "emergency_stop_threshold": 0.95
        }
        
        await budget_monitor.set_budget_limits(agent_id, limits)
        
        # Verify limits were stored
        budget_monitor.redis.set.assert_called()

    @pytest.mark.unit
    async def test_track_spending_within_limits(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test tracking spending within budget limits."""
        # Set budget limits
        limits = {"daily_limit": 1000.0, "total_limit": 10000.0}
        await budget_monitor.set_budget_limits(agent_id, limits)
        
        # Track spending within limits
        spending = 100.0
        violation = await budget_monitor.track_spending(agent_id, spending)
        
        assert violation is None
        budget_monitor.redis.incr.assert_called()

    @pytest.mark.unit
    async def test_track_spending_exceeds_daily_limit(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test tracking spending that exceeds daily limit."""
        # Set low daily limit
        limits = {"daily_limit": 100.0, "total_limit": 10000.0}
        await budget_monitor.set_budget_limits(agent_id, limits)
        
        # Mock current daily spending
        budget_monitor.redis.get.return_value = "90.0"  # Already spent $90
        
        # Try to spend $50 more (would exceed $100 limit)
        spending = 50.0
        violation = await budget_monitor.track_spending(agent_id, spending)
        
        assert violation is not None
        assert violation.violation_type == "daily_limit_exceeded"
        assert violation.current_amount == 140.0
        assert violation.limit_amount == 100.0

    @pytest.mark.unit
    async def test_track_spending_exceeds_total_limit(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test tracking spending that exceeds total limit."""
        # Set low total limit
        limits = {"daily_limit": 1000.0, "total_limit": 500.0}
        await budget_monitor.set_budget_limits(agent_id, limits)
        
        # Mock current total spending
        budget_monitor.redis.get.side_effect = ["50.0", "450.0"]  # Daily: $50, Total: $450
        
        # Try to spend $100 more (would exceed $500 total limit)
        spending = 100.0
        violation = await budget_monitor.track_spending(agent_id, spending)
        
        assert violation is not None
        assert violation.violation_type == "total_limit_exceeded"
        assert violation.current_amount == 550.0
        assert violation.limit_amount == 500.0

    @pytest.mark.unit
    async def test_emergency_stop_threshold(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test emergency stop threshold triggering."""
        # Set budget with emergency stop at 90%
        limits = {"daily_limit": 1000.0, "emergency_stop_threshold": 0.9}
        await budget_monitor.set_budget_limits(agent_id, limits)
        
        # Mock current spending at 85%
        budget_monitor.redis.get.return_value = "850.0"
        
        # Try to spend amount that would exceed 90% threshold
        spending = 100.0  # Would bring total to $950 (95%)
        violation = await budget_monitor.track_spending(agent_id, spending)
        
        assert violation is not None
        assert violation.violation_type == "emergency_stop_threshold"
        assert violation.severity == "critical"

    @pytest.mark.unit
    async def test_get_spending_summary(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test getting spending summary for an agent."""
        # Mock spending data
        budget_monitor.redis.get.side_effect = ["150.0", "1250.0"]  # Daily, Total
        budget_monitor.redis.hget.side_effect = ["1000.0", "10000.0"]  # Limits
        
        summary = await budget_monitor.get_spending_summary(agent_id)
        
        assert summary["daily_spent"] == 150.0
        assert summary["total_spent"] == 1250.0
        assert summary["daily_limit"] == 1000.0
        assert summary["total_limit"] == 10000.0
        assert summary["daily_utilization"] == 0.15
        assert summary["total_utilization"] == 0.125

    @pytest.mark.unit
    async def test_reset_daily_budget(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test resetting daily budget at midnight."""
        await budget_monitor.reset_daily_budget(agent_id)
        
        # Verify daily spending was reset
        budget_monitor.redis.delete.assert_called()

    @pytest.mark.unit
    async def test_budget_forecasting(
        self,
        budget_monitor: BudgetMonitor,
        agent_id: str
    ):
        """Test budget forecasting based on current spending patterns."""
        # Mock historical spending data
        historical_data = [
            {"date": "2024-01-01", "amount": 100.0},
            {"date": "2024-01-02", "amount": 120.0},
            {"date": "2024-01-03", "amount": 110.0},
            {"date": "2024-01-04", "amount": 130.0},
        ]
        
        with patch.object(budget_monitor, 'get_historical_spending') as mock_hist:
            mock_hist.return_value = historical_data
            
            forecast = await budget_monitor.forecast_spending(agent_id, days=7)
            
            assert "projected_daily_average" in forecast
            assert "projected_weekly_total" in forecast
            assert "risk_level" in forecast
            assert forecast["projected_daily_average"] > 0


class TestContentSafetyValidator:
    """Test suite for ContentSafetyValidator functionality."""

    @pytest.fixture
    def content_validator(self):
        """Create ContentSafetyValidator instance."""
        return ContentSafetyValidator(
            api_key="test-key",
            safety_threshold=0.8
        )

    @pytest.mark.unit
    async def test_validate_safe_content(
        self,
        content_validator: ContentSafetyValidator
    ):
        """Test validation of safe content."""
        safe_content = {
            "headline": "Discover Amazing Fitness Technology",
            "description": "Transform your workout with our innovative fitness tracker"
        }
        
        with patch.object(content_validator, '_call_safety_api') as mock_api:
            mock_api.return_value = {
                "safety_score": 0.95,
                "categories": {
                    "toxic": 0.01,
                    "hate_speech": 0.02,
                    "sexual": 0.01,
                    "violence": 0.01
                }
            }
            
            result = await content_validator.validate_content(safe_content)
            
            assert result.is_safe is True
            assert result.safety_score == 0.95
            assert result.violations == []

    @pytest.mark.unit
    async def test_validate_unsafe_content(
        self,
        content_validator: ContentSafetyValidator
    ):
        """Test validation of unsafe content."""
        unsafe_content = {
            "headline": "Inappropriate content example",
            "description": "Content that violates safety policies"
        }
        
        with patch.object(content_validator, '_call_safety_api') as mock_api:
            mock_api.return_value = {
                "safety_score": 0.3,
                "categories": {
                    "toxic": 0.8,
                    "hate_speech": 0.1,
                    "sexual": 0.05,
                    "violence": 0.05
                }
            }
            
            result = await content_validator.validate_content(unsafe_content)
            
            assert result.is_safe is False
            assert result.safety_score == 0.3
            assert len(result.violations) > 0
            assert any("toxic" in v.category for v in result.violations)

    @pytest.mark.unit
    async def test_validate_content_multiple_violations(
        self,
        content_validator: ContentSafetyValidator
    ):
        """Test content with multiple safety violations."""
        problematic_content = {
            "headline": "Problematic ad headline",
            "description": "Content with multiple issues"
        }
        
        with patch.object(content_validator, '_call_safety_api') as mock_api:
            mock_api.return_value = {
                "safety_score": 0.2,
                "categories": {
                    "toxic": 0.9,
                    "hate_speech": 0.85,
                    "sexual": 0.1,
                    "violence": 0.75
                }
            }
            
            result = await content_validator.validate_content(problematic_content)
            
            assert result.is_safe is False
            assert len(result.violations) == 3  # toxic, hate_speech, violence
            
            violation_categories = [v.category for v in result.violations]
            assert "toxic" in violation_categories
            assert "hate_speech" in violation_categories
            assert "violence" in violation_categories

    @pytest.mark.unit
    async def test_batch_content_validation(
        self,
        content_validator: ContentSafetyValidator
    ):
        """Test batch validation of multiple content pieces."""
        content_batch = [
            {"headline": "Safe content 1", "description": "Safe description 1"},
            {"headline": "Safe content 2", "description": "Safe description 2"},
            {"headline": "Unsafe content", "description": "Problematic description"}
        ]
        
        with patch.object(content_validator, '_call_safety_api') as mock_api:
            mock_api.side_effect = [
                {"safety_score": 0.95, "categories": {"toxic": 0.01}},
                {"safety_score": 0.92, "categories": {"toxic": 0.02}},
                {"safety_score": 0.3, "categories": {"toxic": 0.9}}
            ]
            
            results = await content_validator.validate_content_batch(content_batch)
            
            assert len(results) == 3
            assert results[0].is_safe is True
            assert results[1].is_safe is True
            assert results[2].is_safe is False

    @pytest.mark.unit
    async def test_content_improvement_suggestions(
        self,
        content_validator: ContentSafetyValidator
    ):
        """Test getting content improvement suggestions."""
        unsafe_content = {
            "headline": "Problematic headline",
            "description": "Needs improvement"
        }
        
        with patch.object(content_validator, '_get_improvement_suggestions') as mock_suggestions:
            mock_suggestions.return_value = [
                "Remove potentially offensive language",
                "Use more inclusive terminology",
                "Focus on positive messaging"
            ]
            
            suggestions = await content_validator.get_improvement_suggestions(unsafe_content)
            
            assert len(suggestions) == 3
            assert "offensive language" in suggestions[0]
            assert "inclusive terminology" in suggestions[1]


class TestPolicyEnforcer:
    """Test suite for PolicyEnforcer functionality."""

    @pytest.fixture
    def policy_enforcer(self, mock_redis):
        """Create PolicyEnforcer instance with mocked dependencies."""
        with patch("safety_framework.policy_enforcer.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            enforcer = PolicyEnforcer(
                redis_url="redis://localhost:6379",
                policy_config_path="test_policies.yaml"
            )
            return enforcer

    @pytest.mark.unit
    async def test_enforce_budget_policy(
        self,
        policy_enforcer: PolicyEnforcer,
        agent_id: str
    ):
        """Test enforcement of budget policies."""
        campaign_request = {
            "budget": {
                "daily_budget": 2000.0,  # Exceeds policy limit
                "total_budget": 20000.0
            }
        }
        
        # Mock policy configuration
        with patch.object(policy_enforcer, 'get_policy_limits') as mock_limits:
            mock_limits.return_value = {
                "max_daily_budget": 1000.0,
                "max_total_budget": 10000.0
            }
            
            violations = await policy_enforcer.enforce_policies(agent_id, campaign_request)
            
            assert len(violations) == 2  # Both daily and total exceed limits
            assert any(v.policy_type == "budget_limit" for v in violations)

    @pytest.mark.unit
    async def test_enforce_content_policy(
        self,
        policy_enforcer: PolicyEnforcer,
        agent_id: str
    ):
        """Test enforcement of content policies."""
        campaign_request = {
            "creative": {
                "headline": "Test headline",
                "description": "Test description"
            }
        }
        
        # Mock content safety check
        with patch.object(policy_enforcer, 'validate_content_safety') as mock_safety:
            mock_safety.return_value = SafetyScore(
                is_safe=False,
                safety_score=0.3,
                violations=[{"category": "toxic", "score": 0.8}]
            )
            
            violations = await policy_enforcer.enforce_policies(agent_id, campaign_request)
            
            assert len(violations) > 0
            assert any(v.policy_type == "content_safety" for v in violations)

    @pytest.mark.unit
    async def test_enforce_targeting_policy(
        self,
        policy_enforcer: PolicyEnforcer,
        agent_id: str
    ):
        """Test enforcement of targeting policies."""
        campaign_request = {
            "targeting": {
                "demographics": {
                    "age_range": [13, 17]  # Targeting minors - policy violation
                }
            }
        }
        
        violations = await policy_enforcer.enforce_policies(agent_id, campaign_request)
        
        assert len(violations) > 0
        assert any(v.policy_type == "targeting_restriction" for v in violations)

    @pytest.mark.unit
    async def test_human_approval_required(
        self,
        policy_enforcer: PolicyEnforcer,
        agent_id: str
    ):
        """Test cases requiring human approval."""
        high_risk_campaign = {
            "budget": {"daily_budget": 900.0},  # Near limit
            "creative": {
                "headline": "Sensitive topic",
                "description": "Requires review"
            }
        }
        
        approval_required = await policy_enforcer.requires_human_approval(
            agent_id, 
            high_risk_campaign
        )
        
        assert approval_required is True

    @pytest.mark.unit
    async def test_policy_exemption_handling(
        self,
        policy_enforcer: PolicyEnforcer,
        agent_id: str
    ):
        """Test handling of policy exemptions."""
        # Grant exemption for higher budget
        exemption = {
            "agent_id": agent_id,
            "policy_type": "budget_limit",
            "exemption_value": 2000.0,
            "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
        
        await policy_enforcer.grant_exemption(exemption)
        
        campaign_request = {
            "budget": {"daily_budget": 1500.0}  # Would normally violate policy
        }
        
        violations = await policy_enforcer.enforce_policies(agent_id, campaign_request)
        
        # Should not have budget violation due to exemption
        budget_violations = [v for v in violations if v.policy_type == "budget_limit"]
        assert len(budget_violations) == 0


class TestEmergencyStopController:
    """Test suite for EmergencyStopController functionality."""

    @pytest.fixture
    def emergency_controller(self, mock_redis):
        """Create EmergencyStopController instance with mocked dependencies."""
        with patch("safety_framework.emergency_stop.redis.Redis") as mock_redis_cls:
            mock_redis_cls.return_value = mock_redis
            
            controller = EmergencyStopController(
                redis_url="redis://localhost:6379"
            )
            return controller

    @pytest.mark.unit
    async def test_trigger_emergency_stop_budget(
        self,
        emergency_controller: EmergencyStopController,
        agent_id: str
    ):
        """Test triggering emergency stop due to budget violation."""
        violation = BudgetViolation(
            agent_id=agent_id,
            violation_type="emergency_stop_threshold",
            current_amount=950.0,
            limit_amount=1000.0,
            severity="critical"
        )
        
        await emergency_controller.trigger_emergency_stop(agent_id, violation)
        
        # Verify emergency stop was triggered
        emergency_controller.redis.set.assert_called()
        
        # Verify agent status was updated
        status = await emergency_controller.get_emergency_status(agent_id)
        assert status["emergency_active"] is True
        assert status["reason"] == "budget_violation"

    @pytest.mark.unit
    async def test_trigger_emergency_stop_safety(
        self,
        emergency_controller: EmergencyStopController,
        agent_id: str
    ):
        """Test triggering emergency stop due to safety violation."""
        safety_violation = {
            "violation_type": "critical_content_safety",
            "safety_score": 0.1,
            "categories": {"hate_speech": 0.95}
        }
        
        await emergency_controller.trigger_emergency_stop(agent_id, safety_violation)
        
        status = await emergency_controller.get_emergency_status(agent_id)
        assert status["emergency_active"] is True
        assert status["reason"] == "safety_violation"

    @pytest.mark.unit
    async def test_release_emergency_stop(
        self,
        emergency_controller: EmergencyStopController,
        agent_id: str
    ):
        """Test releasing emergency stop after review."""
        # First trigger emergency stop
        await emergency_controller.trigger_emergency_stop(
            agent_id, 
            {"violation_type": "test", "severity": "high"}
        )
        
        # Then release it
        release_reason = "Manual review completed - violation resolved"
        await emergency_controller.release_emergency_stop(agent_id, release_reason)
        
        status = await emergency_controller.get_emergency_status(agent_id)
        assert status["emergency_active"] is False
        assert "resolved" in status["status"].lower()

    @pytest.mark.unit
    async def test_emergency_stop_prevention(
        self,
        emergency_controller: EmergencyStopController,
        agent_id: str
    ):
        """Test prevention of actions during emergency stop."""
        # Trigger emergency stop
        await emergency_controller.trigger_emergency_stop(
            agent_id,
            {"violation_type": "test", "severity": "critical"}
        )
        
        # Try to perform action
        can_proceed = await emergency_controller.can_proceed_with_action(
            agent_id,
            "campaign_launch"
        )
        
        assert can_proceed is False

    @pytest.mark.unit
    async def test_emergency_escalation(
        self,
        emergency_controller: EmergencyStopController,
        agent_id: str
    ):
        """Test emergency escalation for critical violations."""
        critical_violation = {
            "violation_type": "severe_content_safety",
            "severity": "critical",
            "requires_escalation": True
        }
        
        with patch.object(emergency_controller, 'notify_escalation_team') as mock_notify:
            await emergency_controller.trigger_emergency_stop(agent_id, critical_violation)
            
            # Verify escalation team was notified
            mock_notify.assert_called_once()


class TestAuditLogger:
    """Test suite for AuditLogger functionality."""

    @pytest.fixture
    def audit_logger(self, mock_bigquery):
        """Create AuditLogger instance with mocked dependencies."""
        with patch("safety_framework.audit_logger.bigquery.Client") as mock_bq_cls:
            mock_bq_cls.return_value = mock_bigquery
            
            logger = AuditLogger(
                project_id="test-project",
                dataset_id="audit_logs"
            )
            return logger

    @pytest.mark.unit
    async def test_log_safety_violation(
        self,
        audit_logger: AuditLogger,
        agent_id: str
    ):
        """Test logging of safety violations."""
        violation = {
            "agent_id": agent_id,
            "violation_type": "content_safety",
            "severity": "high",
            "details": {"safety_score": 0.2},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await audit_logger.log_safety_violation(violation)
        
        # Verify log was stored
        audit_logger.bigquery.insert_rows.assert_called()

    @pytest.mark.unit
    async def test_log_policy_enforcement(
        self,
        audit_logger: AuditLogger,
        agent_id: str
    ):
        """Test logging of policy enforcement actions."""
        enforcement_action = {
            "agent_id": agent_id,
            "policy_type": "budget_limit",
            "action": "campaign_blocked",
            "reason": "Daily budget exceeded",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await audit_logger.log_policy_enforcement(enforcement_action)
        
        audit_logger.bigquery.insert_rows.assert_called()

    @pytest.mark.unit
    async def test_log_emergency_stop(
        self,
        audit_logger: AuditLogger,
        agent_id: str
    ):
        """Test logging of emergency stop events."""
        emergency_event = {
            "agent_id": agent_id,
            "trigger_reason": "Critical safety violation",
            "severity": "critical",
            "auto_triggered": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await audit_logger.log_emergency_stop(emergency_event)
        
        audit_logger.bigquery.insert_rows.assert_called()

    @pytest.mark.unit
    async def test_generate_compliance_report(
        self,
        audit_logger: AuditLogger,
        agent_id: str
    ):
        """Test generating compliance reports."""
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        # Mock query results
        mock_results = [
            {"violation_type": "budget_exceeded", "count": 5},
            {"violation_type": "content_safety", "count": 2},
            {"violation_type": "targeting_restriction", "count": 1}
        ]
        
        audit_logger.bigquery.query.return_value.result.return_value = mock_results
        
        report = await audit_logger.generate_compliance_report(
            agent_id,
            start_date,
            end_date
        )
        
        assert "total_violations" in report
        assert "violation_breakdown" in report
        assert "compliance_score" in report
        assert report["total_violations"] == 8