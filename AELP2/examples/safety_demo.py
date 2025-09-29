#!/usr/bin/env python3
"""
AELP2 Safety System Demonstration

This script demonstrates how to use the AELP2 safety gates and HITL system
for validating actions and ensuring compliance with safety policies.

REQUIRED ENVIRONMENT VARIABLES:
- AELP2_MIN_WIN_RATE: Minimum win rate threshold (e.g., "0.15")
- AELP2_MAX_CAC: Maximum Customer Acquisition Cost (e.g., "50.0")
- AELP2_MIN_ROAS: Minimum Return on Ad Spend (e.g., "2.0")
- AELP2_MAX_SPEND_VELOCITY: Maximum spend velocity (e.g., "1000.0")
- AELP2_APPROVAL_TIMEOUT: HITL approval timeout in seconds (e.g., "3600")

OPTIONAL ENVIRONMENT VARIABLES:
- AELP2_MAX_DAILY_SPEND: Maximum daily spend limit
- AELP2_SAFETY_LOG_PATH: Path to safety event log file
- AELP2_BLOCKED_CONTENT: Comma-separated list of blocked content patterns
- AELP2_REQUIRE_AGE_VERIFICATION: Set to "true" to require age verification
"""

import os
from typing import Dict, Any

from AELP2.core.safety.hitl import (
    validate_action_safety,
    get_safety_gates,
    get_hitl_queue,
    get_policy_checker,
    get_event_logger,
    SafetyConfigurationError,
    ApprovalStatus,
    emergency_stop
)


def setup_demo_environment():
    """Set up demonstration environment variables."""
    demo_config = {
        'AELP2_MIN_WIN_RATE': '0.15',
        'AELP2_MAX_CAC': '50.0',
        'AELP2_MIN_ROAS': '2.0',
        'AELP2_MAX_SPEND_VELOCITY': '1000.0',
        'AELP2_APPROVAL_TIMEOUT': '3600',
        'AELP2_MAX_DAILY_SPEND': '5000.0',
        'AELP2_BLOCKED_CONTENT': 'gambling,weapon,hate,drug',
        'AELP2_REQUIRE_AGE_VERIFICATION': 'true',
        'AELP2_SAFETY_LOG_PATH': '/tmp/aelp2_safety_demo.log'
    }
    
    print("Setting up demo environment variables...")
    for key, value in demo_config.items():
        os.environ[key] = value
        print(f"  {key} = {value}")
    print()


def demo_safety_gates():
    """Demonstrate safety gates functionality."""
    print("=== SAFETY GATES DEMONSTRATION ===")
    
    try:
        gates = get_safety_gates()
        print("✓ Safety gates initialized successfully")
        print(f"  Configured thresholds: {len(gates.thresholds)}")
        
        # Test with good metrics
        good_metrics = {
            'win_rate': 0.25,
            'spend': 1000.0,
            'conversions': 25,  # CAC = 40.0
            'revenue': 2500.0,  # ROAS = 2.5
            'spend_velocity': 800.0,
            'daily_spend': 4000.0
        }
        
        passed, violations = gates.evaluate_gates(good_metrics)
        print(f"\nGood metrics evaluation:")
        print(f"  Passed: {passed}")
        print(f"  Violations: {len(violations)}")
        
        # Test with problematic metrics
        bad_metrics = {
            'win_rate': 0.10,  # Below threshold
            'spend': 1000.0,
            'conversions': 15,  # CAC = 66.67 (too high)
            'revenue': 1500.0,  # ROAS = 1.5 (too low)
            'spend_velocity': 1200.0,  # Too high
            'daily_spend': 6000.0  # Too high
        }
        
        passed, violations = gates.evaluate_gates(bad_metrics)
        print(f"\nProblematic metrics evaluation:")
        print(f"  Passed: {passed}")
        print(f"  Violations: {len(violations)}")
        for violation in violations:
            print(f"    - {violation.gate_name}: {violation.actual_value} {violation.operator} {violation.threshold_value}")
        
    except SafetyConfigurationError as e:
        print(f"✗ Safety gates configuration error: {e}")
    
    print()


def demo_policy_checker():
    """Demonstrate policy checker functionality."""
    print("=== POLICY CHECKER DEMONSTRATION ===")
    
    try:
        checker = get_policy_checker()
        print("✓ Policy checker initialized successfully")
        
        # Test compliant creative
        good_creative = {
            'headline': 'Amazing Product for Adults',
            'description': 'High-quality features and benefits',
            'call_to_action': 'Learn More',
            'targeting': {
                'min_age': 25,
                'audiences': ['general_interest']
            },
            'content_type': 'general'
        }
        
        compliant, issues = checker.check_policy_compliance(good_creative)
        print(f"\nCompliant creative evaluation:")
        print(f"  Compliant: {compliant}")
        print(f"  Issues: {len(issues)}")
        
        # Test problematic creative
        bad_creative = {
            'headline': 'Win Big at Our Gambling Site!',  # Blocked content
            'description': 'Place bets and make money fast',
            'targeting': {
                'min_age': 16,  # Below 18
                'audiences': ['minors', 'young_adults']  # Restricted audience
            },
            'content_type': 'gambling',  # Age-restricted
            'age_verified': False
        }
        
        compliant, issues = checker.check_policy_compliance(bad_creative)
        print(f"\nProblematic creative evaluation:")
        print(f"  Compliant: {compliant}")
        print(f"  Issues: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
        
    except Exception as e:
        print(f"✗ Policy checker error: {e}")
    
    print()


def demo_hitl_approval():
    """Demonstrate HITL approval system."""
    print("=== HITL APPROVAL SYSTEM DEMONSTRATION ===")
    
    try:
        hitl_queue = get_hitl_queue()
        print("✓ HITL approval queue initialized successfully")
        print(f"  Timeout: {hitl_queue.approval_timeout_seconds} seconds")
        
        # Request approval
        action = {
            'type': 'creative_change',
            'creative_id': 'creative_123',
            'changes': ['headline', 'description']
        }
        
        context = {
            'agent_id': 'demo_agent',
            'priority': 'high',
            'requester': 'safety_demo'
        }
        
        approval_id = hitl_queue.request_approval(action, context)
        print(f"\nApproval requested:")
        print(f"  Approval ID: {approval_id}")
        
        # Check status
        status = hitl_queue.check_approval_status(approval_id)
        print(f"  Status: {status}")
        
        # Get pending requests
        pending = hitl_queue.get_pending_requests()
        print(f"  Pending requests: {len(pending)}")
        
        # Simulate approval
        success = hitl_queue.approve_request(approval_id, 'demo_reviewer', 'Looks good for demo')
        print(f"\nApproval attempt:")
        print(f"  Success: {success}")
        
        status = hitl_queue.check_approval_status(approval_id)
        print(f"  New status: {status}")
        
    except SafetyConfigurationError as e:
        print(f"✗ HITL configuration error: {e}")
    
    print()


def demo_safety_validation():
    """Demonstrate comprehensive safety validation."""
    print("=== COMPREHENSIVE SAFETY VALIDATION DEMONSTRATION ===")
    
    # Test safe action
    safe_action = {
        'type': 'bid_adjustment',
        'bid_change': 0.05
    }
    
    safe_metrics = {
        'win_rate': 0.25,
        'spend': 1000.0,
        'conversions': 25,
        'revenue': 2500.0,
        'spend_velocity': 800.0
    }
    
    safe_context = {
        'agent_id': 'demo_agent',
        'risk_level': 'low'
    }
    
    print("Testing safe action:")
    is_safe, violations, approval_id = validate_action_safety(safe_action, safe_metrics, safe_context)
    print(f"  Safe: {is_safe}")
    print(f"  Violations: {len(violations)}")
    print(f"  Approval required: {approval_id is not None}")
    
    # Test risky creative change
    risky_action = {
        'type': 'creative_change',
        'creative': {
            'headline': 'Great Health Product',  # Health content
            'description': 'Medical-grade treatment for all conditions',
            'targeting': {'min_age': 25}
        }
    }
    
    risky_context = {
        'agent_id': 'demo_agent',
        'risk_level': 'high'
    }
    
    print("\nTesting risky creative change:")
    is_safe, violations, approval_id = validate_action_safety(risky_action, safe_metrics, risky_context)
    print(f"  Safe: {is_safe}")
    print(f"  Violations: {len(violations)}")
    for violation in violations:
        print(f"    - {violation}")
    print(f"  Approval required: {approval_id is not None}")
    if approval_id:
        print(f"  Approval ID: {approval_id}")
    
    print()


def demo_safety_events():
    """Demonstrate safety event logging."""
    print("=== SAFETY EVENT LOGGING DEMONSTRATION ===")
    
    try:
        logger = get_event_logger()
        print("✓ Safety event logger initialized successfully")
        
        # Get recent events
        recent_events = logger.get_recent_events(hours=1)
        print(f"\nRecent events (last hour): {len(recent_events)}")
        
        for event in recent_events[-5:]:  # Show last 5 events
            print(f"  - {event.event_type.value} ({event.severity.value}) at {event.timestamp}")
        
        # Get critical events
        critical_events = logger.get_critical_events(hours=24)
        print(f"\nCritical events (last 24 hours): {len(critical_events)}")
        
    except Exception as e:
        print(f"✗ Safety event logging error: {e}")
    
    print()


def demo_emergency_stop():
    """Demonstrate emergency stop functionality."""
    print("=== EMERGENCY STOP DEMONSTRATION ===")
    
    try:
        reason = "Demo emergency stop - testing safety system"
        context = {
            'demo': True,
            'trigger': 'manual_test',
            'metrics': {
                'spend_velocity': 5000.0,
                'anomaly_detected': True
            }
        }
        
        print("Triggering emergency stop (demo)...")
        emergency_stop(reason, context)
        print("✓ Emergency stop triggered successfully")
        
        # Check that it was logged
        logger = get_event_logger()
        critical_events = logger.get_critical_events(hours=1)
        emergency_events = [e for e in critical_events if 'emergency_stop' in e.event_type.value]
        
        if emergency_events:
            print(f"  Emergency stop events logged: {len(emergency_events)}")
            latest = emergency_events[-1]
            print(f"  Latest event ID: {latest.event_id}")
            print(f"  Reason: {latest.metadata.get('reason', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Emergency stop error: {e}")
    
    print()


def main():
    """Run the complete safety system demonstration."""
    print("AELP2 Safety System Demonstration")
    print("=" * 50)
    print()
    
    # Set up environment
    setup_demo_environment()
    
    # Run demonstrations
    demo_safety_gates()
    demo_policy_checker()
    demo_hitl_approval()
    demo_safety_validation()
    demo_safety_events()
    demo_emergency_stop()
    
    print("=" * 50)
    print("Demonstration completed successfully!")
    print()
    print("Key takeaways:")
    print("- All safety thresholds are configurable via environment variables")
    print("- No hardcoded values or fallbacks - system fails fast if misconfigured")
    print("- HITL approval system prevents unauthorized high-risk actions")
    print("- Policy checker ensures content compliance")
    print("- Comprehensive event logging provides audit trail")
    print("- Emergency stop capability for critical situations")
    print()
    print("For production use:")
    print("1. Set all required environment variables")
    print("2. Configure appropriate thresholds for your business")
    print("3. Set up monitoring for safety events")
    print("4. Implement human review workflows for approvals")


if __name__ == '__main__':
    main()
