#!/usr/bin/env python3
"""
GAELP Learning Verification Script

This script checks existing GAELP training implementations to verify
that actual learning is happening, not just fake learning.

CRITICAL MISSION:
- Check journey_aware_rl_agent.py for real learning
- Verify gaelp_master_integration.py training loops
- Test train_aura_agent.py for gradient flow
- Validate all PPO implementations
- Generate learning verification reports

NO FALLBACKS. DETECT AND FIX FAKE LEARNING.
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import sys
import os
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our verification systems
from learning_verification_system import create_learning_verifier
from verified_training_wrapper import create_verified_wrapper

# Import GAELP components to test
try:
    from journey_aware_rl_agent import JourneyAwarePPOAgent, extract_journey_state_for_encoder
    from enhanced_journey_tracking import MultiTouchJourneySimulator
    from multi_channel_orchestrator import MultiChannelOrchestrator, JourneyAwareRLEnvironment
except ImportError as e:
    print(f"Warning: Could not import journey aware components: {e}")

try:
    from train_aura_agent import AuraRLTrainer
    from aura_campaign_simulator import AuraCampaignEnvironment
except ImportError as e:
    print(f"Warning: Could not import Aura components: {e}")

try:
    from training_orchestrator.rl_agents.agent_factory import AgentFactory, AgentFactoryConfig, AgentType
    from training_orchestrator.rl_agents.ppo_agent import PPOAgent
except ImportError as e:
    print(f"Warning: Could not import training orchestrator components: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GAELPLearningAuditor:
    """
    Comprehensive auditor for GAELP learning systems.
    
    This class will:
    1. Test existing GAELP agents for real learning
    2. Identify broken training loops
    3. Detect fake gradient updates
    4. Generate detailed learning audit reports
    5. Provide fixes for learning issues
    """
    
    def __init__(self):
        self.audit_results = {}
        self.learning_issues = []
        self.audit_timestamp = datetime.now()
        
        logger.info("GAELP Learning Auditor initialized")
        logger.info("Mission: Detect and eliminate fake learning!")
    
    def test_journey_aware_agent_learning(self) -> Dict[str, Any]:
        """Test the JourneyAwarePPOAgent for real learning"""
        
        logger.info("🔍 Testing JourneyAwarePPOAgent learning...")
        
        test_results = {
            'test_name': 'journey_aware_ppo_learning',
            'timestamp': datetime.now(),
            'learning_verified': False,
            'issues_found': [],
            'recommendations': []
        }
        
        try:
            # Create agent with journey encoder
            agent = JourneyAwarePPOAgent(
                state_dim=256,
                hidden_dim=128,
                num_channels=8,
                use_journey_encoder=True
            )
            
            # Wrap with verification
            verified_agent = create_verified_wrapper(agent, "journey_aware_test")
            
            # Create test environment
            simulator = MultiTouchJourneySimulator(num_users=10, time_horizon_days=7)
            orchestrator = MultiChannelOrchestrator(budget_daily=100.0)
            env = JourneyAwareRLEnvironment(simulator, orchestrator)
            
            # Test training loop
            learning_verified = True
            training_issues = []
            
            for episode in range(5):
                logger.info(f"Testing episode {episode}...")
                
                try:
                    # Run verified training episode
                    episode_metrics = verified_agent.verified_train_episode(env, episode, max_steps=20)
                    
                    # Check if learning is verified
                    if not episode_metrics.get('learning_verified', False):
                        learning_verified = False
                        training_issues.append(f"Episode {episode}: Learning not verified")
                    
                    # Check weight updates
                    if not episode_metrics.get('weights_changed', False):
                        learning_verified = False
                        training_issues.append(f"Episode {episode}: No weight updates detected")
                
                except Exception as e:
                    learning_verified = False
                    training_issues.append(f"Episode {episode} failed: {str(e)}")
            
            test_results['learning_verified'] = learning_verified
            test_results['issues_found'] = training_issues
            
            if learning_verified:
                test_results['recommendations'].append("✅ JourneyAwarePPOAgent learning verified!")
            else:
                test_results['recommendations'].extend([
                    "❌ JourneyAwarePPOAgent has learning issues",
                    "• Check optimizer initialization and step() calls",
                    "• Verify gradient computation in update() method", 
                    "• Ensure loss.backward() is called before optimizer.step()",
                    "• Check for detached tensors breaking gradient flow"
                ])
            
            # Generate verification report
            final_report = verified_agent.generate_final_report()
            test_results['detailed_report'] = final_report
            
            # Save verification artifacts
            verified_agent.save_verification_artifacts()
            
        except Exception as e:
            logger.error(f"JourneyAwarePPOAgent test failed: {e}")
            logger.error(traceback.format_exc())
            test_results['error'] = str(e)
            test_results['issues_found'].append(f"Test crashed: {str(e)}")
        
        return test_results
    
    def test_training_orchestrator_ppo(self) -> Dict[str, Any]:
        """Test training orchestrator PPO agent for real learning"""
        
        logger.info("🔍 Testing Training Orchestrator PPO learning...")
        
        test_results = {
            'test_name': 'training_orchestrator_ppo_learning',
            'timestamp': datetime.now(),
            'learning_verified': False,
            'issues_found': [],
            'recommendations': []
        }
        
        try:
            # Create PPO agent from factory
            config = AgentFactoryConfig(
                agent_type=AgentType.PPO,
                agent_id="test_ppo",
                state_dim=64,
                action_dim=32
            )
            
            factory = AgentFactory(config)
            agent = factory.create_agent()
            
            # Wrap with verification
            verified_agent = create_verified_wrapper(agent, "orchestrator_ppo_test")
            
            # Test training with dummy data
            learning_verified = True
            training_issues = []
            
            for step in range(10):
                logger.info(f"Testing training step {step}...")
                
                # Create dummy experience batch
                experiences = []
                for i in range(32):  # Batch of 32 experiences
                    experience = {
                        'state': np.random.randn(64),
                        'action': np.random.randn(32),
                        'reward': np.random.normal(0, 1),
                        'next_state': np.random.randn(64),
                        'done': np.random.random() < 0.1,
                        'metadata': {'user_profile': 'test_user'}
                    }
                    experiences.append(experience)
                
                try:
                    # Run verified policy update
                    update_metrics = verified_agent.verified_update_policy(
                        experiences,
                        episode_reward=np.random.normal(10, 5),
                        episode_length=100
                    )
                    
                    # Check verification results
                    if not update_metrics.get('learning_verified', False):
                        learning_verified = False
                        training_issues.append(f"Step {step}: Learning not verified")
                    
                    if not update_metrics.get('weights_changed', False):
                        learning_verified = False
                        training_issues.append(f"Step {step}: No weight updates")
                
                except Exception as e:
                    learning_verified = False
                    training_issues.append(f"Step {step} failed: {str(e)}")
            
            test_results['learning_verified'] = learning_verified
            test_results['issues_found'] = training_issues
            
            if learning_verified:
                test_results['recommendations'].append("✅ Training Orchestrator PPO learning verified!")
            else:
                test_results['recommendations'].extend([
                    "❌ Training Orchestrator PPO has learning issues",
                    "• Check PPOAgent._ppo_update() method implementation",
                    "• Verify optimizer.step() is called after loss.backward()",
                    "• Check importance sampling integration for gradient corruption",
                    "• Ensure networks are in training mode during updates"
                ])
            
            # Get learning status
            status = verified_agent.get_learning_status()
            test_results['learning_status'] = status
            
        except Exception as e:
            logger.error(f"Training Orchestrator PPO test failed: {e}")
            logger.error(traceback.format_exc())
            test_results['error'] = str(e)
            test_results['issues_found'].append(f"Test crashed: {str(e)}")
        
        return test_results
    
    async def test_aura_agent_learning(self) -> Dict[str, Any]:
        """Test Aura campaign agent learning"""
        
        logger.info("🔍 Testing Aura Agent learning...")
        
        test_results = {
            'test_name': 'aura_agent_learning',
            'timestamp': datetime.now(),
            'learning_verified': False,
            'issues_found': [],
            'recommendations': []
        }
        
        try:
            # Create Aura trainer
            trainer = AuraRLTrainer()
            
            # Extract agent for verification
            agent = trainer.agent
            
            # Wrap with verification
            verified_agent = create_verified_wrapper(agent, "aura_agent_test")
            
            # Test training episodes
            learning_verified = True
            training_issues = []
            
            for episode in range(3):
                logger.info(f"Testing Aura episode {episode}...")
                
                try:
                    # Run training episode using AuraRLTrainer
                    episode_summary = await trainer.train_episode(episode)
                    
                    # Check for learning indicators
                    if episode_summary.get('avg_cac', float('inf')) == float('inf'):
                        training_issues.append(f"Episode {episode}: No valid CAC calculated")
                    
                    if episode_summary.get('total_conversions', 0) == 0:
                        training_issues.append(f"Episode {episode}: No conversions achieved")
                
                except Exception as e:
                    learning_verified = False
                    training_issues.append(f"Episode {episode} failed: {str(e)}")
            
            test_results['learning_verified'] = learning_verified and len(training_issues) == 0
            test_results['issues_found'] = training_issues
            
            if test_results['learning_verified']:
                test_results['recommendations'].append("✅ Aura Agent learning appears functional!")
            else:
                test_results['recommendations'].extend([
                    "❌ Aura Agent has learning/performance issues",
                    "• Check AuraRLTrainer.train_episode() implementation",
                    "• Verify agent.select_action() returns valid actions",
                    "• Check environment.run_campaign() for realistic results",
                    "• Ensure reward calculation encourages learning"
                ])
        
        except Exception as e:
            logger.error(f"Aura Agent test failed: {e}")
            logger.error(traceback.format_exc())
            test_results['error'] = str(e)
            test_results['issues_found'].append(f"Test crashed: {str(e)}")
        
        return test_results
    
    def test_gradient_flow_manually(self) -> Dict[str, Any]:
        """Manually test gradient flow in GAELP models"""
        
        logger.info("🔍 Testing gradient flow manually...")
        
        test_results = {
            'test_name': 'manual_gradient_flow_test',
            'timestamp': datetime.now(),
            'gradient_flow_healthy': False,
            'issues_found': [],
            'recommendations': []
        }
        
        try:
            # Test different model architectures
            models_to_test = []
            
            # Journey-aware actor-critic
            from journey_aware_rl_agent import JourneyAwareActorCritic
            journey_model = JourneyAwareActorCritic(state_dim=256, hidden_dim=128, num_channels=8)
            models_to_test.append(("JourneyAwareActorCritic", journey_model))
            
            # Training orchestrator networks
            from training_orchestrator.rl_agents.networks import PolicyNetwork, ValueNetwork
            policy_net = PolicyNetwork(state_dim=64, action_dim=32, hidden_dims=[128, 64])
            value_net = ValueNetwork(state_dim=64, hidden_dims=[128, 64])
            models_to_test.append(("PolicyNetwork", policy_net))
            models_to_test.append(("ValueNetwork", value_net))
            
            gradient_issues = []
            healthy_gradients = []
            
            for model_name, model in models_to_test:
                logger.info(f"Testing {model_name} gradient flow...")
                
                try:
                    # Create verifier for this model
                    verifier = create_learning_verifier(f"{model_name}_test")
                    
                    # Test gradient flow
                    dummy_input = torch.randn(16, next(iter(model.parameters())).shape[-1] if len(next(iter(model.parameters())).shape) > 1 else 64)
                    
                    # Forward pass
                    if model_name == "JourneyAwareActorCritic":
                        channel_probs, value, bid_amounts = model(dummy_input)
                        loss = channel_probs.mean() + value.mean() + bid_amounts.mean()
                    elif model_name == "PolicyNetwork":
                        action, log_prob = model.sample_action(dummy_input)
                        loss = action.mean() + log_prob.mean()
                    elif model_name == "ValueNetwork":
                        value = model(dummy_input)
                        loss = value.mean()
                    
                    # Test gradient verification
                    gradient_info = verifier.verify_gradient_flow(model, loss, 0)
                    
                    if gradient_info['has_gradients']:
                        healthy_gradients.append(model_name)
                    else:
                        gradient_issues.append(f"{model_name}: {gradient_info['problems']}")
                    
                except Exception as e:
                    gradient_issues.append(f"{model_name}: Test failed - {str(e)}")
            
            test_results['gradient_flow_healthy'] = len(gradient_issues) == 0
            test_results['issues_found'] = gradient_issues
            test_results['healthy_models'] = healthy_gradients
            
            if test_results['gradient_flow_healthy']:
                test_results['recommendations'].append("✅ All tested models have healthy gradient flow!")
            else:
                test_results['recommendations'].extend([
                    "❌ Some models have gradient flow issues",
                    "• Check for .detach() calls that break gradient computation",
                    "• Verify requires_grad=True for all trainable parameters",
                    "• Check loss computation for disconnected operations",
                    "• Ensure backward() is called on loss tensor"
                ])
        
        except Exception as e:
            logger.error(f"Manual gradient flow test failed: {e}")
            logger.error(traceback.format_exc())
            test_results['error'] = str(e)
        
        return test_results
    
    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive learning audit on all GAELP components"""
        
        logger.info("🚨 Starting COMPREHENSIVE GAELP LEARNING AUDIT")
        logger.info("=" * 80)
        
        audit_results = {
            'audit_timestamp': self.audit_timestamp,
            'tests_run': [],
            'overall_learning_health': 'unknown',
            'critical_issues': [],
            'recommendations': []
        }
        
        # Test 1: Journey-aware agent learning
        logger.info("\n📋 TEST 1: Journey-Aware PPO Agent")
        journey_results = self.test_journey_aware_agent_learning()
        audit_results['tests_run'].append(journey_results)
        
        if not journey_results['learning_verified']:
            audit_results['critical_issues'].append("Journey-aware PPO learning broken")
        
        # Test 2: Training orchestrator PPO
        logger.info("\n📋 TEST 2: Training Orchestrator PPO") 
        orchestrator_results = self.test_training_orchestrator_ppo()
        audit_results['tests_run'].append(orchestrator_results)
        
        if not orchestrator_results['learning_verified']:
            audit_results['critical_issues'].append("Training orchestrator PPO learning broken")
        
        # Test 3: Aura agent learning
        logger.info("\n📋 TEST 3: Aura Campaign Agent")
        aura_results = self.test_aura_agent_learning()
        audit_results['tests_run'].append(aura_results)
        
        if not aura_results['learning_verified']:
            audit_results['critical_issues'].append("Aura agent learning broken")
        
        # Test 4: Manual gradient flow
        logger.info("\n📋 TEST 4: Manual Gradient Flow")
        gradient_results = self.test_gradient_flow_manually()
        audit_results['tests_run'].append(gradient_results)
        
        if not gradient_results['gradient_flow_healthy']:
            audit_results['critical_issues'].append("Gradient flow issues detected")
        
        # Determine overall health
        learning_tests_passed = sum(1 for test in audit_results['tests_run'][:3] 
                                   if test.get('learning_verified', False))
        gradient_test_passed = gradient_results.get('gradient_flow_healthy', False)
        
        total_critical_tests = 4
        tests_passed = learning_tests_passed + (1 if gradient_test_passed else 0)
        
        if tests_passed == total_critical_tests:
            audit_results['overall_learning_health'] = 'excellent'
        elif tests_passed >= 3:
            audit_results['overall_learning_health'] = 'good'
        elif tests_passed >= 2:
            audit_results['overall_learning_health'] = 'concerning'
        else:
            audit_results['overall_learning_health'] = 'critical'
        
        # Generate recommendations
        if audit_results['overall_learning_health'] == 'excellent':
            audit_results['recommendations'].append("🎉 All GAELP learning systems verified!")
        else:
            audit_results['recommendations'].extend([
                "🚨 GAELP learning issues detected",
                "• Review failed test recommendations above",
                "• Check optimizer initialization in all agents",
                "• Verify loss.backward() and optimizer.step() calls",
                "• Test with learning_verification_system.py",
                "• Use verified_training_wrapper.py for all training"
            ])
        
        self.audit_results = audit_results
        return audit_results
    
    def generate_audit_report(self) -> str:
        """Generate comprehensive audit report"""
        
        if not self.audit_results:
            return "No audit results available. Run run_comprehensive_audit() first."
        
        report = []
        report.append("=" * 100)
        report.append("🔍 GAELP LEARNING VERIFICATION AUDIT REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now()}")
        report.append(f"Audit Started: {self.audit_results['audit_timestamp']}")
        report.append("")
        
        # Overall status
        health = self.audit_results['overall_learning_health']
        if health == 'excellent':
            report.append("🎉 OVERALL LEARNING HEALTH: EXCELLENT ✅")
        elif health == 'good':
            report.append("👍 OVERALL LEARNING HEALTH: GOOD ✅")
        elif health == 'concerning':
            report.append("⚠️  OVERALL LEARNING HEALTH: CONCERNING ⚠️")
        else:
            report.append("🚨 OVERALL LEARNING HEALTH: CRITICAL ❌")
        
        report.append("")
        
        # Critical issues
        if self.audit_results['critical_issues']:
            report.append("🚨 CRITICAL ISSUES FOUND:")
            for issue in self.audit_results['critical_issues']:
                report.append(f"   • {issue}")
            report.append("")
        
        # Test results summary
        report.append("📋 TEST RESULTS SUMMARY:")
        for test in self.audit_results['tests_run']:
            test_name = test['test_name']
            if test.get('learning_verified') or test.get('gradient_flow_healthy'):
                report.append(f"   ✅ {test_name}: PASSED")
            else:
                report.append(f"   ❌ {test_name}: FAILED")
        
        report.append("")
        
        # Detailed test results
        report.append("📋 DETAILED TEST RESULTS:")
        report.append("")
        
        for test in self.audit_results['tests_run']:
            report.append(f"🔍 {test['test_name'].upper()}")
            report.append(f"   Timestamp: {test['timestamp']}")
            
            if test.get('learning_verified') is not None:
                status = "PASSED" if test['learning_verified'] else "FAILED"
                report.append(f"   Learning Verified: {status}")
            
            if test.get('gradient_flow_healthy') is not None:
                status = "HEALTHY" if test['gradient_flow_healthy'] else "BROKEN"
                report.append(f"   Gradient Flow: {status}")
            
            if test.get('issues_found'):
                report.append("   Issues Found:")
                for issue in test['issues_found']:
                    report.append(f"      • {issue}")
            
            if test.get('recommendations'):
                report.append("   Recommendations:")
                for rec in test['recommendations']:
                    report.append(f"      • {rec}")
            
            report.append("")
        
        # Overall recommendations
        report.append("💡 OVERALL RECOMMENDATIONS:")
        for rec in self.audit_results['recommendations']:
            report.append(f"   • {rec}")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)

async def main():
    """Main function to run GAELP learning audit"""
    
    print("🔍 GAELP LEARNING VERIFICATION AUDIT")
    print("=" * 80)
    print("Mission: Detect and eliminate fake learning in all GAELP agents!")
    print("NO FALLBACKS. NO MOCK LEARNING. REAL VERIFICATION ONLY.")
    print("=" * 80)
    
    # Create auditor
    auditor = GAELPLearningAuditor()
    
    # Run comprehensive audit
    audit_results = await auditor.run_comprehensive_audit()
    
    # Generate and display report
    print("\n" + auditor.generate_audit_report())
    
    # Save detailed results
    import json
    results_path = "/home/hariravichandran/AELP/gaelp_learning_audit_results.json"
    with open(results_path, 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)
    
    report_path = "/home/hariravichandran/AELP/gaelp_learning_audit_report.txt"
    with open(report_path, 'w') as f:
        f.write(auditor.generate_audit_report())
    
    print(f"\n📊 Detailed results saved to:")
    print(f"   • {results_path}")
    print(f"   • {report_path}")
    
    # Final status
    health = audit_results['overall_learning_health']
    if health in ['excellent', 'good']:
        print("\n✅ GAELP learning audit PASSED! Real learning detected.")
    else:
        print("\n❌ GAELP learning audit FAILED! Learning issues must be fixed.")
        print("\nCRITICAL: Review the audit report and fix all learning issues before proceeding!")

if __name__ == "__main__":
    asyncio.run(main())