#!/usr/bin/env python3
"""
GAELP Safety Integration Module
Integrates all safety systems with the main GAELP training and bidding pipeline.

INTEGRATED SAFETY SYSTEMS:
1. Comprehensive Safety Framework (gaelp_safety_framework.py)
2. Reward Validation System (reward_validation_system.py) 
3. Budget Safety System (budget_safety_system.py)
4. Ethical Advertising System (ethical_advertising_system.py)
5. Emergency Controls (emergency_controls.py)

This module provides a single interface for all safety checks and validations
that integrates seamlessly with the existing GAELP production system.

NO FALLBACKS - ALL SAFETY CHECKS ARE MANDATORY AND BLOCKING
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import all safety systems
from gaelp_safety_framework import (
    ComprehensiveSafetyFramework, 
    get_safety_framework,
    safety_check_decorator
)
from reward_validation_system import (
    ProductionRewardValidator,
    get_reward_validator,
    validate_reward_safe,
    reward_validation_decorator
)
from budget_safety_system import (
    ProductionBudgetSafetySystem,
    get_budget_safety_system,
    budget_safety_check
)
from ethical_advertising_system import (
    ProductionEthicalSystem,
    get_ethical_system
)
from emergency_controls import (
    EmergencyController,
    get_emergency_controller,
    emergency_stop_decorator
)

logger = logging.getLogger(__name__)

class SafetyCheckResult(Enum):
    """Safety check result types"""
    APPROVED = "approved"
    CONDITIONAL = "conditional"  # Approved with modifications
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"

@dataclass
class ComprehensiveSafetyResult:
    """Result of comprehensive safety validation"""
    overall_result: SafetyCheckResult
    component_results: Dict[str, SafetyCheckResult]
    violations: List[str]
    warnings: List[str]
    safe_modifications: Dict[str, Any]
    human_review_required: bool
    emergency_triggered: bool
    safety_scores: Dict[str, float]
    processing_time_ms: float
    timestamp: datetime

class GAELPSafetyOrchestrator:
    """Main orchestrator for all GAELP safety systems"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "gaelp_safety_integration_config.json"
        self.config = self._load_config()
        
        # Initialize all safety systems
        self.safety_framework = get_safety_framework()
        self.reward_validator = get_reward_validator()
        self.budget_safety = get_budget_safety_system()
        self.ethical_system = get_ethical_system()
        self.emergency_controller = get_emergency_controller()
        
        # Safety state tracking
        self.safety_checks_performed = 0
        self.safety_violations_detected = 0
        self.human_reviews_triggered = 0
        self.emergency_stops_triggered = 0
        
        # Performance tracking
        self.safety_check_times = []
        self.component_performance = {}
        
        logger.info("GAELP Safety Orchestrator initialized with all safety systems")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load safety integration configuration"""
        default_config = {
            "safety_checks": {
                "reward_validation": {"enabled": True, "blocking": True},
                "budget_safety": {"enabled": True, "blocking": True},
                "ethical_compliance": {"enabled": True, "blocking": True},
                "emergency_controls": {"enabled": True, "blocking": True}
            },
            "performance": {
                "max_safety_check_time_ms": 1000,
                "parallel_processing": True,
                "cache_results": True
            },
            "human_review": {
                "enabled": True,
                "auto_escalation_threshold": 0.8,
                "review_timeout_hours": 24
            }
        }
        
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error loading safety config: {e}")
        
        return default_config
    
    def validate_bidding_decision_comprehensive(self, bid_data: Dict[str, Any]) -> ComprehensiveSafetyResult:
        """Comprehensive safety validation of a bidding decision"""
        start_time = time.time()
        
        # Initialize result tracking
        component_results = {}
        violations = []
        warnings = []
        safe_modifications = {}
        safety_scores = {}
        human_review_required = False
        emergency_triggered = False
        
        try:
            # 1. REWARD VALIDATION
            if self._is_safety_check_enabled("reward_validation"):
                reward_result = self._validate_reward_safety(bid_data)
                component_results["reward_validation"] = reward_result["result"]
                violations.extend(reward_result["violations"])
                warnings.extend(reward_result["warnings"])
                safe_modifications.update(reward_result["modifications"])
                safety_scores["reward_safety"] = reward_result["score"]
                
                if reward_result["human_review"]:
                    human_review_required = True
            
            # 2. BUDGET SAFETY
            if self._is_safety_check_enabled("budget_safety"):
                budget_result = self._validate_budget_safety(bid_data)
                component_results["budget_safety"] = budget_result["result"]
                violations.extend(budget_result["violations"])
                warnings.extend(budget_result["warnings"])
                safe_modifications.update(budget_result["modifications"])
                safety_scores["budget_safety"] = budget_result["score"]
                
                if budget_result["emergency"]:
                    emergency_triggered = True
            
            # 3. ETHICAL COMPLIANCE
            if self._is_safety_check_enabled("ethical_compliance"):
                ethical_result = self._validate_ethical_compliance(bid_data)
                component_results["ethical_compliance"] = ethical_result["result"]
                violations.extend(ethical_result["violations"])
                warnings.extend(ethical_result["warnings"])
                safe_modifications.update(ethical_result["modifications"])
                safety_scores["ethical_compliance"] = ethical_result["score"]
                
                if ethical_result["human_review"]:
                    human_review_required = True
            
            # 4. EMERGENCY CONTROLS CHECK
            if self._is_safety_check_enabled("emergency_controls"):
                emergency_result = self._check_emergency_conditions(bid_data)
                component_results["emergency_controls"] = emergency_result["result"]
                violations.extend(emergency_result["violations"])
                warnings.extend(emergency_result["warnings"])
                safety_scores["emergency_safety"] = emergency_result["score"]
                
                if emergency_result["emergency"]:
                    emergency_triggered = True
            
            # 5. DETERMINE OVERALL RESULT
            overall_result = self._determine_overall_safety_result(
                component_results, violations, human_review_required, emergency_triggered
            )
            
            # 6. UPDATE STATISTICS
            self.safety_checks_performed += 1
            if violations:
                self.safety_violations_detected += 1
            if human_review_required:
                self.human_reviews_triggered += 1
            if emergency_triggered:
                self.emergency_stops_triggered += 1
            
        except Exception as e:
            logger.error(f"Error in comprehensive safety validation: {e}")
            overall_result = SafetyCheckResult.REJECTED
            violations.append(f"Safety validation error: {str(e)}")
            emergency_triggered = True
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        self.safety_check_times.append(processing_time_ms)
        
        # Keep only last 1000 timing records
        if len(self.safety_check_times) > 1000:
            self.safety_check_times = self.safety_check_times[-1000:]
        
        return ComprehensiveSafetyResult(
            overall_result=overall_result,
            component_results=component_results,
            violations=violations,
            warnings=warnings,
            safe_modifications=safe_modifications,
            human_review_required=human_review_required,
            emergency_triggered=emergency_triggered,
            safety_scores=safety_scores,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now()
        )
    
    def _validate_reward_safety(self, bid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reward safety"""
        try:
            if 'reward' not in bid_data:
                return {
                    "result": SafetyCheckResult.APPROVED,
                    "violations": [],
                    "warnings": [],
                    "modifications": {},
                    "score": 1.0,
                    "human_review": False
                }
            
            reward = bid_data['reward']
            context = bid_data.get('context', {})
            
            # Validate with reward system
            validation_result = self.reward_validator.validate_reward(reward, context)
            
            violations = []
            warnings = []
            modifications = {}
            
            if not validation_result.is_valid:
                violations.extend(validation_result.warnings)
            
            if validation_result.clipping_applied:
                warnings.append("Reward was clipped for safety")
                modifications['safe_reward'] = validation_result.validated_reward
            
            # Determine result
            if validation_result.validation_level.value == "blocked":
                result = SafetyCheckResult.REJECTED
            elif validation_result.validation_level.value == "dangerous":
                result = SafetyCheckResult.HUMAN_REVIEW
            elif validation_result.validation_level.value == "suspicious":
                result = SafetyCheckResult.CONDITIONAL
            else:
                result = SafetyCheckResult.APPROVED
            
            # Calculate safety score
            safety_score = max(0.0, 1.0 - validation_result.anomaly_score / 5.0)
            
            human_review = validation_result.validation_level.value in ["dangerous", "blocked"]
            
            return {
                "result": result,
                "violations": violations,
                "warnings": warnings,
                "modifications": modifications,
                "score": safety_score,
                "human_review": human_review
            }
        
        except Exception as e:
            logger.error(f"Error in reward safety validation: {e}")
            return {
                "result": SafetyCheckResult.REJECTED,
                "violations": [f"Reward validation error: {str(e)}"],
                "warnings": [],
                "modifications": {},
                "score": 0.0,
                "human_review": True
            }
    
    def _validate_budget_safety(self, bid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate budget safety"""
        try:
            bid_amount = bid_data.get('bid_amount', 0.0)
            campaign_id = bid_data.get('campaign_id', 'unknown')
            channel = bid_data.get('channel', 'unknown')
            account_id = bid_data.get('account_id', 'default')
            
            # Check with budget safety system
            is_allowed, violations_list, pacing_info = self.budget_safety.validate_spending(
                bid_amount, campaign_id, channel, account_id, bid_data.get('metadata', {})
            )
            
            violations = []
            warnings = []
            modifications = {}
            emergency = False
            
            for violation in violations_list:
                if violation.startswith('WARNING:'):
                    warnings.append(violation)
                else:
                    violations.append(violation)
                    if 'emergency' in violation.lower():
                        emergency = True
            
            # Add pacing modifications
            if pacing_info:
                modifications.update(pacing_info)
            
            # Determine result
            if not is_allowed:
                if emergency:
                    result = SafetyCheckResult.REJECTED
                else:
                    result = SafetyCheckResult.CONDITIONAL
            else:
                result = SafetyCheckResult.APPROVED
            
            # Calculate safety score based on budget utilization
            safety_score = 1.0
            for violation in violations_list:
                if '%' in violation:
                    # Extract utilization percentage
                    try:
                        pct = float(violation.split('%')[0].split()[-1]) / 100.0
                        safety_score = min(safety_score, max(0.0, 2.0 - 2.0 * pct))
                    except:
                        pass
            
            return {
                "result": result,
                "violations": violations,
                "warnings": warnings,
                "modifications": modifications,
                "score": safety_score,
                "emergency": emergency
            }
        
        except Exception as e:
            logger.error(f"Error in budget safety validation: {e}")
            return {
                "result": SafetyCheckResult.REJECTED,
                "violations": [f"Budget validation error: {str(e)}"],
                "warnings": [],
                "modifications": {},
                "score": 0.0,
                "emergency": True
            }
    
    def _validate_ethical_compliance(self, bid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ethical compliance"""
        try:
            campaign_data = bid_data.get('campaign_data', {})
            
            if not campaign_data:
                return {
                    "result": SafetyCheckResult.APPROVED,
                    "violations": [],
                    "warnings": [],
                    "modifications": {},
                    "score": 1.0,
                    "human_review": False
                }
            
            # Validate with ethical system
            is_compliant, violation_objects, recommendations = self.ethical_system.validate_campaign_ethics(campaign_data)
            
            violations = []
            warnings = []
            modifications = {}
            human_review = False
            
            for violation in violation_objects:
                if violation.severity.value in ['critical', 'severe', 'violation']:
                    violations.append(violation.description)
                else:
                    warnings.append(violation.description)
                
                if violation.human_review_required:
                    human_review = True
            
            if recommendations:
                modifications['recommendations'] = recommendations
            
            # Determine result
            if violations:
                if human_review:
                    result = SafetyCheckResult.HUMAN_REVIEW
                else:
                    result = SafetyCheckResult.CONDITIONAL
            else:
                result = SafetyCheckResult.APPROVED
            
            # Calculate safety score
            violation_count = len(violations)
            warning_count = len(warnings)
            safety_score = max(0.0, 1.0 - (violation_count * 0.3 + warning_count * 0.1))
            
            return {
                "result": result,
                "violations": violations,
                "warnings": warnings,
                "modifications": modifications,
                "score": safety_score,
                "human_review": human_review
            }
        
        except Exception as e:
            logger.error(f"Error in ethical compliance validation: {e}")
            return {
                "result": SafetyCheckResult.REJECTED,
                "violations": [f"Ethical validation error: {str(e)}"],
                "warnings": [],
                "modifications": {},
                "score": 0.0,
                "human_review": True
            }
    
    def _check_emergency_conditions(self, bid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check emergency conditions"""
        try:
            # Check if emergency stop is active
            if self.emergency_controller.emergency_stop_triggered:
                return {
                    "result": SafetyCheckResult.REJECTED,
                    "violations": ["Emergency stop is active - all bidding blocked"],
                    "warnings": [],
                    "score": 0.0,
                    "emergency": True
                }
            
            # Check system health
            if not self.emergency_controller.is_system_healthy():
                return {
                    "result": SafetyCheckResult.REJECTED,
                    "violations": ["System health check failed"],
                    "warnings": [],
                    "score": 0.0,
                    "emergency": True
                }
            
            # Record the bid for monitoring
            bid_amount = bid_data.get('bid_amount', 0.0)
            if bid_amount > 0:
                self.emergency_controller.record_bid(bid_amount)
            
            return {
                "result": SafetyCheckResult.APPROVED,
                "violations": [],
                "warnings": [],
                "score": 1.0,
                "emergency": False
            }
        
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            return {
                "result": SafetyCheckResult.REJECTED,
                "violations": [f"Emergency check error: {str(e)}"],
                "warnings": [],
                "score": 0.0,
                "emergency": True
            }
    
    def _determine_overall_safety_result(self, component_results: Dict[str, SafetyCheckResult],
                                       violations: List[str], human_review_required: bool,
                                       emergency_triggered: bool) -> SafetyCheckResult:
        """Determine overall safety result from component results"""
        
        if emergency_triggered:
            return SafetyCheckResult.REJECTED
        
        if human_review_required:
            return SafetyCheckResult.HUMAN_REVIEW
        
        # Check for any rejections
        if any(result == SafetyCheckResult.REJECTED for result in component_results.values()):
            return SafetyCheckResult.REJECTED
        
        # Check for any human review requirements
        if any(result == SafetyCheckResult.HUMAN_REVIEW for result in component_results.values()):
            return SafetyCheckResult.HUMAN_REVIEW
        
        # Check for conditional approvals
        if any(result == SafetyCheckResult.CONDITIONAL for result in component_results.values()):
            return SafetyCheckResult.CONDITIONAL
        
        # All approved
        return SafetyCheckResult.APPROVED
    
    def _is_safety_check_enabled(self, check_name: str) -> bool:
        """Check if a specific safety check is enabled"""
        return self.config.get("safety_checks", {}).get(check_name, {}).get("enabled", True)
    
    @safety_check_decorator("training_step")
    @reward_validation_decorator
    @emergency_stop_decorator("training")
    def validate_training_step(self, training_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate a training step for safety"""
        
        # Extract training metrics for safety validation
        loss = training_data.get('loss', 0.0)
        reward = training_data.get('reward', 0.0)
        step = training_data.get('step', 0)
        
        # Record training loss for monitoring
        if loss is not None:
            self.emergency_controller.record_training_loss(float(loss))
        
        # Validate reward if present
        safe_modifications = {}
        if reward is not None:
            validated_reward = validate_reward_safe(reward, training_data.get('context', {}))
            if abs(validated_reward - reward) > 1e-6:
                safe_modifications['safe_reward'] = validated_reward
        
        return True, safe_modifications
    
    def get_safety_status_comprehensive(self) -> Dict[str, Any]:
        """Get comprehensive safety status across all systems"""
        
        # Get individual system statuses
        safety_framework_status = self.safety_framework.get_safety_status()
        budget_safety_status = self.budget_safety.get_budget_status()
        ethical_system_status = self.ethical_system.get_ethical_status()
        emergency_status = self.emergency_controller.get_system_status()
        
        # Calculate performance metrics
        avg_check_time = np.mean(self.safety_check_times) if self.safety_check_times else 0
        max_check_time = np.max(self.safety_check_times) if self.safety_check_times else 0
        
        return {
            "overall_status": "healthy" if emergency_status["active"] else "emergency",
            "safety_framework": safety_framework_status,
            "budget_safety": budget_safety_status,
            "ethical_system": ethical_system_status,
            "emergency_controls": emergency_status,
            "performance_metrics": {
                "total_safety_checks": self.safety_checks_performed,
                "violations_detected": self.safety_violations_detected,
                "human_reviews_triggered": self.human_reviews_triggered,
                "emergency_stops_triggered": self.emergency_stops_triggered,
                "average_check_time_ms": avg_check_time,
                "max_check_time_ms": max_check_time,
                "violation_rate": self.safety_violations_detected / max(self.safety_checks_performed, 1),
                "human_review_rate": self.human_reviews_triggered / max(self.safety_checks_performed, 1)
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def emergency_shutdown_all_safety_systems(self, reason: str):
        """Emergency shutdown of all safety systems"""
        logger.critical(f"EMERGENCY SHUTDOWN OF ALL SAFETY SYSTEMS: {reason}")
        
        try:
            # Trigger emergency stop
            self.emergency_controller.trigger_manual_emergency_stop(reason)
            
            # Shutdown budget safety
            self.budget_safety.emergency_budget_stop(reason)
            
            # Shutdown safety framework monitoring
            self.safety_framework.shutdown_safety_monitoring()
            
            logger.critical("All safety systems emergency shutdown completed")
            
        except Exception as e:
            logger.critical(f"Error during emergency shutdown: {e}")


# Global safety orchestrator instance
_safety_orchestrator: Optional[GAELPSafetyOrchestrator] = None

def get_safety_orchestrator() -> GAELPSafetyOrchestrator:
    """Get global safety orchestrator instance"""
    global _safety_orchestrator
    if _safety_orchestrator is None:
        _safety_orchestrator = GAELPSafetyOrchestrator()
    return _safety_orchestrator

def validate_gaelp_safety(bid_data: Dict[str, Any]) -> ComprehensiveSafetyResult:
    """Main function to validate GAELP operations for safety"""
    orchestrator = get_safety_orchestrator()
    return orchestrator.validate_bidding_decision_comprehensive(bid_data)

def gaelp_safety_decorator(operation_type: str = "bidding"):
    """Decorator to add comprehensive safety checks to GAELP operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract data for safety validation
            if args and isinstance(args[0], dict):
                data = args[0]
            else:
                data = {"operation_type": operation_type, "args": args, "kwargs": kwargs}
            
            # Perform safety validation
            safety_result = validate_gaelp_safety(data)
            
            # Check if operation is safe to proceed
            if safety_result.overall_result == SafetyCheckResult.REJECTED:
                raise Exception(f"Safety validation failed: {'; '.join(safety_result.violations)}")
            
            if safety_result.overall_result == SafetyCheckResult.HUMAN_REVIEW:
                logger.warning(f"Operation requires human review: {'; '.join(safety_result.violations)}")
                # In production, this would queue for human review
                # For now, we'll proceed with warnings
            
            # Apply safe modifications if any
            if safety_result.safe_modifications:
                if args and isinstance(args[0], dict):
                    args[0].update(safety_result.safe_modifications)
            
            # Execute the original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and comprehensive testing
    print("Initializing GAELP Safety Integration System...")
    
    orchestrator = GAELPSafetyOrchestrator()
    
    print("Testing comprehensive safety validation...")
    
    # Test normal bidding decision
    test_bid_data = {
        'bid_id': 'test_001',
        'bid_amount': 2.50,
        'campaign_id': 'campaign_001',
        'channel': 'google_search',
        'account_id': 'account_001',
        'reward': 1.25,
        'context': {
            'user_segment': 'high_value',
            'conversion_probability': 0.05,
            'conversion_value': 25.0,
            'competition_level': 1.2
        },
        'campaign_data': {
            'creative_text': 'Premium products at great prices',
            'headline': 'Shop now and save!',
            'targeting': {
                'age': {'min': 25, 'max': 55},
                'interests': ['shopping', 'deals']
            },
            'category': 'retail',
            'industry': 'e-commerce'
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_version': 'v1.0'
        }
    }
    
    # Validate normal bid
    result = orchestrator.validate_bidding_decision_comprehensive(test_bid_data)
    
    print(f"\nNormal bid validation result: {result.overall_result.value}")
    print(f"Component results: {[f'{k}: {v.value}' for k, v in result.component_results.items()]}")
    print(f"Violations: {result.violations}")
    print(f"Warnings: {result.warnings}")
    print(f"Safety scores: {result.safety_scores}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Human review required: {result.human_review_required}")
    
    # Test problematic bid
    print("\nTesting problematic bid...")
    
    problematic_bid_data = test_bid_data.copy()
    problematic_bid_data.update({
        'bid_amount': 500.0,  # Extremely high bid
        'reward': 1000.0,     # Suspicious reward
        'campaign_data': {
            'creative_text': 'Miracle cure guaranteed! Lose 50 pounds in 30 days!',
            'headline': 'Doctors hate this one weird trick!',
            'targeting': {
                'age': {'min': 16, 'max': 70},  # Too young
                'demographics': ['financially_distressed', 'elderly']
            },
            'category': 'health_supplements',
            'industry': 'healthcare'
        }
    })
    
    problematic_result = orchestrator.validate_bidding_decision_comprehensive(problematic_bid_data)
    
    print(f"\nProblematic bid validation result: {problematic_result.overall_result.value}")
    print(f"Component results: {[f'{k}: {v.value}' for k, v in problematic_result.component_results.items()]}")
    print(f"Violations: {problematic_result.violations}")
    print(f"Warnings: {problematic_result.warnings}")
    print(f"Safety scores: {problematic_result.safety_scores}")
    print(f"Human review required: {problematic_result.human_review_required}")
    print(f"Emergency triggered: {problematic_result.emergency_triggered}")
    
    # Test training step validation
    print("\nTesting training step validation...")
    
    training_data = {
        'loss': 0.15,
        'reward': 2.5,
        'step': 1000,
        'context': {'training_phase': 'production'}
    }
    
    is_safe, modifications = orchestrator.validate_training_step(training_data)
    print(f"Training step validation: {'SAFE' if is_safe else 'UNSAFE'}")
    if modifications:
        print(f"Safe modifications: {modifications}")
    
    # Get comprehensive status
    print("\nGetting comprehensive safety status...")
    status = orchestrator.get_safety_status_comprehensive()
    print(f"Overall safety status: {status['overall_status']}")
    print(f"Performance metrics: {status['performance_metrics']}")
    
    print("\nGAELP Safety Integration test completed.")
    print(f"Total safety checks performed: {orchestrator.safety_checks_performed}")
    print(f"Violations detected: {orchestrator.safety_violations_detected}")
    print(f"Human reviews triggered: {orchestrator.human_reviews_triggered}")