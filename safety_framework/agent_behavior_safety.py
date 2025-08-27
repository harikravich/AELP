"""
Agent Behavior Safety Module for GAELP Ad Campaign Safety
Implements action space constraints, behavior monitoring, and intervention mechanisms.
"""

import logging
import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
import json
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


class ActionType(Enum):
    SET_BUDGET = "set_budget"
    ADJUST_BIDDING = "adjust_bidding"
    MODIFY_TARGETING = "modify_targeting"
    CHANGE_CREATIVE = "change_creative"
    PAUSE_CAMPAIGN = "pause_campaign"
    RESUME_CAMPAIGN = "resume_campaign"
    CREATE_CAMPAIGN = "create_campaign"
    DELETE_CAMPAIGN = "delete_campaign"


class BehaviorViolationType(Enum):
    EXCESSIVE_BUDGET = "excessive_budget"
    DISCRIMINATORY_TARGETING = "discriminatory_targeting"
    REPETITIVE_ACTIONS = "repetitive_actions"
    RAPID_CHANGES = "rapid_changes"
    INVALID_PARAMETERS = "invalid_parameters"
    UNETHICAL_TARGETING = "unethical_targeting"
    POLICY_VIOLATION = "policy_violation"
    RESOURCE_ABUSE = "resource_abuse"


class InterventionLevel(Enum):
    WARNING = "warning"
    ACTION_BLOCK = "action_block"
    TEMPORARY_RESTRICTION = "temporary_restriction"
    PERMANENT_RESTRICTION = "permanent_restriction"
    HUMAN_REVIEW = "human_review"


@dataclass
class ActionConstraint:
    """Constraint for agent actions"""
    action_type: ActionType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None
    rate_limit: Optional[int] = None  # Max actions per hour
    requires_approval: bool = False
    ethical_constraints: List[str] = field(default_factory=list)


@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    action_id: str
    agent_id: str
    action_type: ActionType
    parameters: Dict[str, Any]
    timestamp: datetime
    campaign_id: Optional[str] = None
    approved: bool = True
    approval_source: Optional[str] = None


@dataclass
class BehaviorViolation:
    """Detected agent behavior violation"""
    violation_id: str
    agent_id: str
    violation_type: BehaviorViolationType
    description: str
    severity: str
    confidence: float
    actions_involved: List[str]
    timestamp: datetime
    intervention_applied: Optional[InterventionLevel] = None


@dataclass
class EthicalGuideline:
    """Ethical guideline for targeting and behavior"""
    guideline_id: str
    category: str
    rule: str
    rationale: str
    violation_severity: str
    applies_to_actions: List[ActionType]


class ActionConstraintEngine:
    """Enforces constraints on agent actions"""
    
    def __init__(self):
        self.constraints: Dict[ActionType, ActionConstraint] = {}
        self.global_constraints = {
            'max_daily_budget': 10000.0,
            'max_campaign_count': 100,
            'min_bid_amount': 0.01,
            'max_bid_amount': 100.0
        }
        
        # Set up default constraints
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self):
        """Initialize default action constraints"""
        self.constraints = {
            ActionType.SET_BUDGET: ActionConstraint(
                action_type=ActionType.SET_BUDGET,
                min_value=1.0,
                max_value=10000.0,
                rate_limit=10,  # 10 budget changes per hour
                ethical_constraints=['no_excessive_spending']
            ),
            ActionType.ADJUST_BIDDING: ActionConstraint(
                action_type=ActionType.ADJUST_BIDDING,
                min_value=0.01,
                max_value=100.0,
                rate_limit=20,
                ethical_constraints=['fair_bidding']
            ),
            ActionType.MODIFY_TARGETING: ActionConstraint(
                action_type=ActionType.MODIFY_TARGETING,
                rate_limit=5,
                requires_approval=True,
                ethical_constraints=['no_discrimination', 'age_appropriate', 'privacy_compliant']
            ),
            ActionType.CREATE_CAMPAIGN: ActionConstraint(
                action_type=ActionType.CREATE_CAMPAIGN,
                rate_limit=2,
                requires_approval=True,
                ethical_constraints=['content_appropriate', 'legal_compliance']
            ),
            ActionType.DELETE_CAMPAIGN: ActionConstraint(
                action_type=ActionType.DELETE_CAMPAIGN,
                rate_limit=1,
                requires_approval=True,
                ethical_constraints=['data_preservation']
            )
        }
    
    async def validate_action(self, action: AgentAction) -> Tuple[bool, List[str]]:
        """Validate an action against constraints"""
        violations = []
        
        try:
            constraint = self.constraints.get(action.action_type)
            if not constraint:
                return True, []  # No constraints defined
            
            # Check value constraints
            if 'value' in action.parameters:
                value = action.parameters['value']
                
                if constraint.min_value is not None and value < constraint.min_value:
                    violations.append(f"Value {value} below minimum {constraint.min_value}")
                
                if constraint.max_value is not None and value > constraint.max_value:
                    violations.append(f"Value {value} exceeds maximum {constraint.max_value}")
                
                if constraint.allowed_values and value not in constraint.allowed_values:
                    violations.append(f"Value {value} not in allowed values")
            
            # Check global constraints
            if action.action_type == ActionType.SET_BUDGET:
                budget = action.parameters.get('value', 0)
                if budget > self.global_constraints['max_daily_budget']:
                    violations.append(f"Budget {budget} exceeds global daily limit")
            
            # Check targeting constraints
            if action.action_type == ActionType.MODIFY_TARGETING:
                targeting_violations = await self._validate_targeting(action.parameters)
                violations.extend(targeting_violations)
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Action validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def _validate_targeting(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate targeting parameters for ethical compliance"""
        violations = []
        
        try:
            targeting = parameters.get('targeting', {})
            
            # Check for discriminatory targeting
            prohibited_criteria = {
                'race', 'ethnicity', 'religion', 'sexual_orientation',
                'political_affiliation', 'health_conditions', 'financial_status'
            }
            
            for criterion in targeting:
                if criterion.lower() in prohibited_criteria:
                    violations.append(f"Discriminatory targeting criterion: {criterion}")
            
            # Check age targeting
            age_range = targeting.get('age_range', {})
            if age_range:
                min_age = age_range.get('min', 0)
                max_age = age_range.get('max', 100)
                
                if min_age < 13:
                    violations.append("Cannot target users under 13 without special compliance")
                
                if min_age < 18 and 'adult_content' in parameters:
                    violations.append("Cannot target minors with adult content")
            
            # Check location targeting for sensitive areas
            locations = targeting.get('locations', [])
            sensitive_locations = ['hospitals', 'schools', 'religious_sites']
            for location in locations:
                if any(sensitive in location.lower() for sensitive in sensitive_locations):
                    violations.append(f"Sensitive location targeting: {location}")
            
            return violations
            
        except Exception as e:
            logger.error(f"Targeting validation failed: {e}")
            return [f"Targeting validation error: {str(e)}"]
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of all constraints"""
        return {
            'action_constraints': {
                action_type.value: {
                    'min_value': constraint.min_value,
                    'max_value': constraint.max_value,
                    'rate_limit': constraint.rate_limit,
                    'requires_approval': constraint.requires_approval,
                    'ethical_constraints': constraint.ethical_constraints
                }
                for action_type, constraint in self.constraints.items()
            },
            'global_constraints': self.global_constraints
        }


class BehaviorMonitor:
    """Monitors agent behavior for violations and anomalies"""
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.action_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.behavior_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.violations: List[BehaviorViolation] = []
        
        # Behavior analysis parameters
        self.repetition_threshold = 5  # Same action 5 times
        self.rapid_change_threshold = 10  # 10 actions per minute
        self.stuck_behavior_threshold = 20  # Same action 20 times in a row
    
    async def record_action(self, action: AgentAction) -> Optional[BehaviorViolation]:
        """Record agent action and check for violations"""
        try:
            agent_id = action.agent_id
            
            # Store action in history
            self.action_history[agent_id].append(action)
            
            # Update behavior patterns
            await self._update_behavior_patterns(agent_id)
            
            # Check for violations
            violation = await self._detect_behavior_violations(action)
            
            if violation:
                self.violations.append(violation)
                logger.warning(f"Behavior violation detected: {violation.description}")
            
            return violation
            
        except Exception as e:
            logger.error(f"Failed to record action: {e}")
            return None
    
    async def _update_behavior_patterns(self, agent_id: str):
        """Update behavior patterns for an agent"""
        try:
            actions = list(self.action_history[agent_id])
            if len(actions) < 5:
                return
            
            # Calculate action frequency
            recent_actions = [a for a in actions if a.timestamp > datetime.utcnow() - timedelta(minutes=10)]
            action_frequency = len(recent_actions) / 10.0  # Actions per minute
            
            # Calculate action diversity
            action_types = [a.action_type for a in recent_actions]
            unique_actions = len(set(action_types))
            diversity_score = unique_actions / max(len(action_types), 1)
            
            # Calculate repetition patterns
            repetition_score = self._calculate_repetition_score(actions)
            
            # Update patterns
            self.behavior_patterns[agent_id].update({
                'action_frequency': action_frequency,
                'diversity_score': diversity_score,
                'repetition_score': repetition_score,
                'total_actions': len(actions),
                'last_updated': datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Failed to update behavior patterns for {agent_id}: {e}")
    
    def _calculate_repetition_score(self, actions: List[AgentAction]) -> float:
        """Calculate how repetitive the agent's behavior is"""
        if len(actions) < 2:
            return 0.0
        
        # Count consecutive identical actions
        max_consecutive = 0
        current_consecutive = 1
        
        for i in range(1, len(actions)):
            if (actions[i].action_type == actions[i-1].action_type and
                actions[i].parameters == actions[i-1].parameters):
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        
        max_consecutive = max(max_consecutive, current_consecutive)
        return max_consecutive / len(actions)
    
    async def _detect_behavior_violations(self, action: AgentAction) -> Optional[BehaviorViolation]:
        """Detect behavior violations for a specific action"""
        try:
            agent_id = action.agent_id
            actions = list(self.action_history[agent_id])
            
            # Check for excessive repetition
            if len(actions) >= self.repetition_threshold:
                recent_actions = actions[-self.repetition_threshold:]
                if all(a.action_type == action.action_type and 
                      a.parameters == action.parameters 
                      for a in recent_actions):
                    return BehaviorViolation(
                        violation_id=f"repeat_{datetime.utcnow().timestamp()}",
                        agent_id=agent_id,
                        violation_type=BehaviorViolationType.REPETITIVE_ACTIONS,
                        description=f"Agent repeated same action {self.repetition_threshold} times",
                        severity="medium",
                        confidence=0.9,
                        actions_involved=[a.action_id for a in recent_actions],
                        timestamp=datetime.utcnow()
                    )
            
            # Check for rapid changes
            recent_actions = [a for a in actions 
                            if a.timestamp > datetime.utcnow() - timedelta(minutes=1)]
            if len(recent_actions) > self.rapid_change_threshold:
                return BehaviorViolation(
                    violation_id=f"rapid_{datetime.utcnow().timestamp()}",
                    agent_id=agent_id,
                    violation_type=BehaviorViolationType.RAPID_CHANGES,
                    description=f"Agent made {len(recent_actions)} actions in 1 minute",
                    severity="high",
                    confidence=0.8,
                    actions_involved=[a.action_id for a in recent_actions],
                    timestamp=datetime.utcnow()
                )
            
            # Check for stuck behavior
            if len(actions) >= self.stuck_behavior_threshold:
                last_actions = actions[-self.stuck_behavior_threshold:]
                if all(a.action_type == last_actions[0].action_type for a in last_actions):
                    return BehaviorViolation(
                        violation_id=f"stuck_{datetime.utcnow().timestamp()}",
                        agent_id=agent_id,
                        violation_type=BehaviorViolationType.REPETITIVE_ACTIONS,
                        description=f"Agent appears stuck, repeating {action.action_type.value}",
                        severity="critical",
                        confidence=0.95,
                        actions_involved=[a.action_id for a in last_actions],
                        timestamp=datetime.utcnow()
                    )
            
            # Check for discriminatory targeting
            if action.action_type == ActionType.MODIFY_TARGETING:
                discrimination_violation = await self._check_discriminatory_targeting(action)
                if discrimination_violation:
                    return discrimination_violation
            
            return None
            
        except Exception as e:
            logger.error(f"Behavior violation detection failed: {e}")
            return None
    
    async def _check_discriminatory_targeting(self, action: AgentAction) -> Optional[BehaviorViolation]:
        """Check for discriminatory targeting patterns"""
        try:
            targeting = action.parameters.get('targeting', {})
            
            # Check for protected characteristics
            protected_patterns = [
                'race', 'ethnicity', 'religion', 'gender', 'sexual_orientation',
                'age_discrimination', 'disability', 'pregnancy'
            ]
            
            violations = []
            for key, value in targeting.items():
                if any(pattern in key.lower() for pattern in protected_patterns):
                    violations.append(f"Targeting based on protected characteristic: {key}")
            
            if violations:
                return BehaviorViolation(
                    violation_id=f"discrim_{datetime.utcnow().timestamp()}",
                    agent_id=action.agent_id,
                    violation_type=BehaviorViolationType.DISCRIMINATORY_TARGETING,
                    description=f"Discriminatory targeting detected: {', '.join(violations)}",
                    severity="critical",
                    confidence=0.8,
                    actions_involved=[action.action_id],
                    timestamp=datetime.utcnow()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Discriminatory targeting check failed: {e}")
            return None
    
    def get_agent_behavior_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get behavior summary for specific agent"""
        try:
            if agent_id not in self.behavior_patterns:
                return {'no_data': True}
            
            patterns = self.behavior_patterns[agent_id]
            actions = list(self.action_history[agent_id])
            
            agent_violations = [v for v in self.violations if v.agent_id == agent_id]
            
            return {
                'total_actions': len(actions),
                'action_frequency': patterns.get('action_frequency', 0),
                'diversity_score': patterns.get('diversity_score', 0),
                'repetition_score': patterns.get('repetition_score', 0),
                'violations': {
                    'total': len(agent_violations),
                    'critical': len([v for v in agent_violations if v.severity == 'critical']),
                    'recent': len([v for v in agent_violations 
                                 if v.timestamp > datetime.utcnow() - timedelta(hours=24)])
                },
                'action_breakdown': self._get_action_breakdown(actions),
                'last_activity': max(a.timestamp for a in actions) if actions else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get behavior summary for {agent_id}: {e}")
            return {'error': str(e)}
    
    def _get_action_breakdown(self, actions: List[AgentAction]) -> Dict[str, int]:
        """Get breakdown of actions by type"""
        breakdown = {}
        for action in actions:
            action_type = action.action_type.value
            breakdown[action_type] = breakdown.get(action_type, 0) + 1
        return breakdown


class InterventionManager:
    """Manages interventions for behavior violations"""
    
    def __init__(self):
        self.active_interventions: Dict[str, Dict[str, Any]] = {}
        self.intervention_history: List[Dict[str, Any]] = []
        self.escalation_rules = {
            'repetitive_actions': [InterventionLevel.WARNING, InterventionLevel.ACTION_BLOCK],
            'discriminatory_targeting': [InterventionLevel.ACTION_BLOCK, InterventionLevel.HUMAN_REVIEW],
            'rapid_changes': [InterventionLevel.TEMPORARY_RESTRICTION],
            'excessive_budget': [InterventionLevel.ACTION_BLOCK, InterventionLevel.HUMAN_REVIEW]
        }
    
    async def apply_intervention(self, violation: BehaviorViolation) -> InterventionLevel:
        """Apply appropriate intervention for a violation"""
        try:
            agent_id = violation.agent_id
            violation_type = violation.violation_type.value
            
            # Determine intervention level based on violation and history
            intervention_level = await self._determine_intervention_level(violation)
            
            # Apply the intervention
            success = await self._execute_intervention(agent_id, intervention_level, violation)
            
            if success:
                violation.intervention_applied = intervention_level
                
                # Record intervention
                intervention_record = {
                    'agent_id': agent_id,
                    'violation_id': violation.violation_id,
                    'intervention_level': intervention_level,
                    'timestamp': datetime.utcnow(),
                    'reason': violation.description
                }
                self.intervention_history.append(intervention_record)
                
                logger.info(f"Intervention applied: {intervention_level.value} for agent {agent_id}")
                return intervention_level
            else:
                logger.error(f"Failed to apply intervention for agent {agent_id}")
                return InterventionLevel.WARNING
                
        except Exception as e:
            logger.error(f"Intervention application failed: {e}")
            return InterventionLevel.WARNING
    
    async def _determine_intervention_level(self, violation: BehaviorViolation) -> InterventionLevel:
        """Determine appropriate intervention level"""
        try:
            agent_id = violation.agent_id
            violation_type = violation.violation_type.value
            
            # Check agent's violation history
            agent_violations = [
                v for v in self.intervention_history 
                if v['agent_id'] == agent_id and 
                v['timestamp'] > datetime.utcnow() - timedelta(days=7)
            ]
            
            # Base intervention on severity and history
            if violation.severity == 'critical':
                if len(agent_violations) > 2:
                    return InterventionLevel.PERMANENT_RESTRICTION
                elif len(agent_violations) > 0:
                    return InterventionLevel.HUMAN_REVIEW
                else:
                    return InterventionLevel.ACTION_BLOCK
            
            elif violation.severity == 'high':
                if len(agent_violations) > 1:
                    return InterventionLevel.TEMPORARY_RESTRICTION
                else:
                    return InterventionLevel.ACTION_BLOCK
            
            elif violation.severity == 'medium':
                if len(agent_violations) > 2:
                    return InterventionLevel.ACTION_BLOCK
                else:
                    return InterventionLevel.WARNING
            
            else:  # low severity
                return InterventionLevel.WARNING
                
        except Exception as e:
            logger.error(f"Failed to determine intervention level: {e}")
            return InterventionLevel.WARNING
    
    async def _execute_intervention(self, agent_id: str, level: InterventionLevel, 
                                  violation: BehaviorViolation) -> bool:
        """Execute the specified intervention"""
        try:
            if level == InterventionLevel.WARNING:
                # Send warning notification
                logger.warning(f"WARNING issued to agent {agent_id}: {violation.description}")
                return True
            
            elif level == InterventionLevel.ACTION_BLOCK:
                # Block specific action type
                blocked_action = violation.violation_type
                self.active_interventions[agent_id] = {
                    'type': 'action_block',
                    'blocked_actions': [blocked_action],
                    'start_time': datetime.utcnow(),
                    'duration': timedelta(hours=1),
                    'reason': violation.description
                }
                logger.warning(f"ACTION BLOCKED for agent {agent_id}: {blocked_action}")
                return True
            
            elif level == InterventionLevel.TEMPORARY_RESTRICTION:
                # Temporary restriction on all actions
                self.active_interventions[agent_id] = {
                    'type': 'temporary_restriction',
                    'start_time': datetime.utcnow(),
                    'duration': timedelta(hours=24),
                    'reason': violation.description
                }
                logger.warning(f"TEMPORARY RESTRICTION applied to agent {agent_id}")
                return True
            
            elif level == InterventionLevel.PERMANENT_RESTRICTION:
                # Permanent restriction
                self.active_interventions[agent_id] = {
                    'type': 'permanent_restriction',
                    'start_time': datetime.utcnow(),
                    'reason': violation.description
                }
                logger.critical(f"PERMANENT RESTRICTION applied to agent {agent_id}")
                return True
            
            elif level == InterventionLevel.HUMAN_REVIEW:
                # Flag for human review
                self.active_interventions[agent_id] = {
                    'type': 'human_review',
                    'start_time': datetime.utcnow(),
                    'reason': violation.description,
                    'status': 'pending'
                }
                logger.critical(f"HUMAN REVIEW required for agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to execute intervention: {e}")
            return False
    
    async def check_action_allowed(self, agent_id: str, action_type: ActionType) -> Tuple[bool, str]:
        """Check if an action is allowed for an agent"""
        try:
            if agent_id not in self.active_interventions:
                return True, ""
            
            intervention = self.active_interventions[agent_id]
            intervention_type = intervention['type']
            
            # Check if intervention has expired
            if 'duration' in intervention:
                if datetime.utcnow() > intervention['start_time'] + intervention['duration']:
                    del self.active_interventions[agent_id]
                    return True, ""
            
            # Check specific intervention types
            if intervention_type == 'action_block':
                blocked_actions = intervention.get('blocked_actions', [])
                if action_type in blocked_actions:
                    return False, f"Action {action_type.value} is blocked"
            
            elif intervention_type == 'temporary_restriction':
                return False, "Agent is under temporary restriction"
            
            elif intervention_type == 'permanent_restriction':
                return False, "Agent is permanently restricted"
            
            elif intervention_type == 'human_review':
                if intervention.get('status') == 'pending':
                    return False, "Agent is pending human review"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Failed to check action permission: {e}")
            return False, "Permission check failed"
    
    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get intervention system summary"""
        try:
            active_count = len(self.active_interventions)
            
            # Count by intervention type
            by_type = {}
            for intervention in self.active_interventions.values():
                int_type = intervention['type']
                by_type[int_type] = by_type.get(int_type, 0) + 1
            
            return {
                'active_interventions': active_count,
                'by_type': by_type,
                'total_interventions': len(self.intervention_history),
                'recent_interventions': len([
                    i for i in self.intervention_history
                    if i['timestamp'] > datetime.utcnow() - timedelta(hours=24)
                ]),
                'agents_under_review': len([
                    i for i in self.active_interventions.values()
                    if i['type'] == 'human_review' and i.get('status') == 'pending'
                ])
            }
        except Exception as e:
            logger.error(f"Failed to generate intervention summary: {e}")
            return {}


class AgentBehaviorSafetyOrchestrator:
    """Main orchestrator for agent behavior safety"""
    
    def __init__(self):
        self.constraint_engine = ActionConstraintEngine()
        self.behavior_monitor = BehaviorMonitor()
        self.intervention_manager = InterventionManager()
        
        self.ethical_guidelines = self._load_ethical_guidelines()
    
    def _load_ethical_guidelines(self) -> List[EthicalGuideline]:
        """Load ethical guidelines for agent behavior"""
        return [
            EthicalGuideline(
                guideline_id="no_discrimination",
                category="fairness",
                rule="Do not target based on protected characteristics",
                rationale="Prevents discriminatory advertising practices",
                violation_severity="critical",
                applies_to_actions=[ActionType.MODIFY_TARGETING, ActionType.CREATE_CAMPAIGN]
            ),
            EthicalGuideline(
                guideline_id="age_appropriate",
                category="child_safety",
                rule="Ensure age-appropriate targeting and content",
                rationale="Protects minors from inappropriate content",
                violation_severity="high",
                applies_to_actions=[ActionType.MODIFY_TARGETING, ActionType.CHANGE_CREATIVE]
            ),
            EthicalGuideline(
                guideline_id="transparent_pricing",
                category="transparency",
                rule="Avoid deceptive pricing or hidden fees",
                rationale="Maintains trust and regulatory compliance",
                violation_severity="medium",
                applies_to_actions=[ActionType.CHANGE_CREATIVE, ActionType.CREATE_CAMPAIGN]
            ),
            EthicalGuideline(
                guideline_id="privacy_respect",
                category="privacy",
                rule="Respect user privacy and data protection laws",
                rationale="Ensures GDPR/CCPA compliance",
                violation_severity="high",
                applies_to_actions=[ActionType.MODIFY_TARGETING]
            )
        ]
    
    async def validate_and_execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Validate and execute agent action with safety checks"""
        result = {
            'action_allowed': False,
            'action_executed': False,
            'violations': [],
            'interventions': [],
            'warnings': []
        }
        
        try:
            # Check if agent is allowed to perform this action
            allowed, restriction_reason = await self.intervention_manager.check_action_allowed(
                action.agent_id, action.action_type
            )
            
            if not allowed:
                result['warnings'].append(f"Action blocked: {restriction_reason}")
                return result
            
            # Validate action against constraints
            valid, constraint_violations = await self.constraint_engine.validate_action(action)
            
            if not valid:
                result['violations'].extend(constraint_violations)
                result['warnings'].append("Action violates constraints")
                return result
            
            # Record action and check for behavior violations
            behavior_violation = await self.behavior_monitor.record_action(action)
            
            if behavior_violation:
                result['violations'].append(behavior_violation.description)
                
                # Apply intervention
                intervention_level = await self.intervention_manager.apply_intervention(behavior_violation)
                result['interventions'].append(intervention_level.value)
                
                # If critical violation, block the action
                if behavior_violation.severity == 'critical':
                    result['warnings'].append("Action blocked due to critical behavior violation")
                    return result
            
            # Action passes all safety checks
            result['action_allowed'] = True
            result['action_executed'] = True
            
            logger.info(f"Action executed safely: {action.action_type.value} by {action.agent_id}")
            return result
            
        except Exception as e:
            logger.error(f"Action validation and execution failed: {e}")
            result['warnings'].append(f"System error: {str(e)}")
            return result
    
    async def get_agent_safety_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive safety status for an agent"""
        try:
            # Get behavior summary
            behavior_summary = self.behavior_monitor.get_agent_behavior_summary(agent_id)
            
            # Check for active interventions
            allowed, restriction = await self.intervention_manager.check_action_allowed(
                agent_id, ActionType.CREATE_CAMPAIGN  # Test with any action
            )
            
            # Get recent violations
            recent_violations = [
                v for v in self.behavior_monitor.violations
                if v.agent_id == agent_id and 
                v.timestamp > datetime.utcnow() - timedelta(days=7)
            ]
            
            return {
                'agent_id': agent_id,
                'safety_status': 'safe' if allowed and len(recent_violations) == 0 else 'restricted',
                'behavior_summary': behavior_summary,
                'active_restrictions': not allowed,
                'restriction_reason': restriction if not allowed else None,
                'recent_violations': len(recent_violations),
                'ethical_compliance': self._assess_ethical_compliance(agent_id),
                'last_assessment': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get safety status for {agent_id}: {e}")
            return {'error': str(e)}
    
    def _assess_ethical_compliance(self, agent_id: str) -> Dict[str, Any]:
        """Assess agent's compliance with ethical guidelines"""
        try:
            # Get agent's recent violations
            violations = [
                v for v in self.behavior_monitor.violations
                if v.agent_id == agent_id and 
                v.timestamp > datetime.utcnow() - timedelta(days=30)
            ]
            
            # Check compliance with each guideline
            compliance = {}
            for guideline in self.ethical_guidelines:
                guideline_violations = [
                    v for v in violations
                    if guideline.category.lower() in v.description.lower()
                ]
                
                compliance[guideline.guideline_id] = {
                    'compliant': len(guideline_violations) == 0,
                    'violations': len(guideline_violations),
                    'severity': guideline.violation_severity
                }
            
            overall_score = sum(1 for c in compliance.values() if c['compliant']) / len(compliance)
            
            return {
                'overall_score': overall_score,
                'guideline_compliance': compliance,
                'assessment_date': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Ethical compliance assessment failed: {e}")
            return {'error': str(e)}
    
    def get_safety_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive agent behavior safety dashboard"""
        try:
            return {
                'constraints': {
                    'total_constraints': len(self.constraint_engine.constraints),
                    'global_limits': self.constraint_engine.global_constraints
                },
                'behavior_monitoring': {
                    'active_agents': len(self.behavior_monitor.action_history),
                    'total_violations': len(self.behavior_monitor.violations),
                    'critical_violations': len([
                        v for v in self.behavior_monitor.violations
                        if v.severity == 'critical'
                    ]),
                    'recent_violations': len([
                        v for v in self.behavior_monitor.violations
                        if v.timestamp > datetime.utcnow() - timedelta(hours=24)
                    ])
                },
                'interventions': self.intervention_manager.get_intervention_summary(),
                'ethical_guidelines': {
                    'total_guidelines': len(self.ethical_guidelines),
                    'categories': list(set(g.category for g in self.ethical_guidelines))
                },
                'system_health': {
                    'monitoring_active': True,
                    'last_updated': datetime.utcnow(),
                    'safety_systems': {
                        'constraint_engine': 'operational',
                        'behavior_monitor': 'operational',
                        'intervention_manager': 'operational'
                    }
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate safety dashboard: {e}")
            return {'error': str(e)}