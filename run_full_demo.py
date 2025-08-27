#!/usr/bin/env python3
"""
Automated GAELP Training Demo - Full System (Mock Personas)
Runs the complete agent training pipeline from simulation to real deployment

NOTE: This is the original demo with mock personas. 
For the new LLM-powered version, use: run_full_demo_llm.py
"""

import asyncio
import json
import time
from typing import Dict, Any
import random

# Import GAELP components
from training_orchestrator import TrainingOrchestrator
from training_orchestrator.config import TrainingOrchestratorConfig, DEVELOPMENT_CONFIG

class MockAgent:
    """Enhanced Mock RL agent with more realistic learning simulation"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.policy_state = {
            "learning_rate": 0.001, 
            "exploration": 0.1,
            "creative_preferences": {"image": 0.33, "video": 0.33, "carousel": 0.34},
            "audience_preferences": {"young_adults": 0.33, "professionals": 0.33, "families": 0.34},
            "bid_preferences": {"cpc": 0.33, "cpm": 0.33, "cpa": 0.34}
        }
        self.performance_history = []
        self.action_rewards = {}  # Track rewards per action type
        
    async def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select campaign action with learned preferences"""
        
        # Use learned preferences (weighted random selection)
        creative_type = self._weighted_choice(self.policy_state["creative_preferences"])
        target_audience = self._weighted_choice(self.policy_state["audience_preferences"])
        bid_strategy = self._weighted_choice(self.policy_state["bid_preferences"])
        
        # Adaptive budget based on performance
        base_budget = 30.0
        if len(self.performance_history) > 5:
            recent_performance = sum(self.performance_history[-5:]) / 5
            if recent_performance > 2.0:
                base_budget *= 1.2  # Increase budget for good performance
            elif recent_performance < 1.0:
                base_budget *= 0.8  # Decrease budget for poor performance
        
        budget = base_budget + random.uniform(-10, 10)
        
        action = {
            "creative_type": creative_type,
            "target_audience": target_audience,
            "budget": max(10.0, min(50.0, budget)),  # Clamp to reasonable range
            "bid_strategy": bid_strategy,
            "bid_amount": random.uniform(1.0, 5.0),
            "audience_size": random.uniform(0.3, 0.8),
            "ab_test_enabled": random.random() > 0.7,
            "ab_test_split": 0.5,
            "action_metadata": {
                "agent_id": self.agent_id,
                "exploration_rate": self.policy_state["exploration"],
                "learning_step": len(self.performance_history)
            }
        }
        
        # Store action for learning
        self.last_action = action
        return action
    
    def _weighted_choice(self, preferences: Dict[str, float]) -> str:
        """Make weighted random choice based on learned preferences"""
        choices = list(preferences.keys())
        weights = list(preferences.values())
        
        # Add exploration noise
        exploration = self.policy_state["exploration"]
        weights = [(1 - exploration) * w + exploration / len(weights) for w in weights]
        
        return random.choices(choices, weights=weights)[0]
    
    async def update_policy(self, reward: float, performance_data: Dict[str, Any]):
        """Update agent policy with more sophisticated learning"""
        self.performance_history.append(reward)
        
        # Track reward for last action
        if hasattr(self, 'last_action'):
            action_signature = (
                self.last_action["creative_type"],
                self.last_action["target_audience"], 
                self.last_action["bid_strategy"]
            )
            
            if action_signature not in self.action_rewards:
                self.action_rewards[action_signature] = []
            self.action_rewards[action_signature].append(reward)
        
        # Update preferences based on performance
        if len(self.performance_history) > 10:
            self._update_preferences()
            
            avg_recent = sum(self.performance_history[-10:]) / 10
            if avg_recent > 2.0:  # Good performance
                self.policy_state["exploration"] *= 0.95  # Explore less
            else:
                self.policy_state["exploration"] *= 1.05  # Explore more
                
            # Keep exploration in reasonable bounds
            self.policy_state["exploration"] = max(0.05, min(0.5, self.policy_state["exploration"]))
                
        print(f"Agent {self.agent_id} updated policy. Exploration: {self.policy_state['exploration']:.3f}, "
              f"Recent ROAS: {reward:.2f}")
    
    def _update_preferences(self):
        """Update action preferences based on observed rewards"""
        learning_rate = self.policy_state["learning_rate"] * 10  # Faster learning for demo
        
        # Update preferences for actions with good rewards
        for action_sig, rewards in self.action_rewards.items():
            if len(rewards) >= 3:  # Need sufficient data
                avg_reward = sum(rewards[-3:]) / 3  # Recent average
                
                creative, audience, bid = action_sig
                
                # Update creative preferences
                if avg_reward > 1.5:  # Above average performance
                    self.policy_state["creative_preferences"][creative] += learning_rate
                elif avg_reward < 1.0:  # Below average performance
                    self.policy_state["creative_preferences"][creative] -= learning_rate * 0.5
                
                # Update audience preferences
                if avg_reward > 1.5:
                    self.policy_state["audience_preferences"][audience] += learning_rate
                elif avg_reward < 1.0:
                    self.policy_state["audience_preferences"][audience] -= learning_rate * 0.5
                
                # Update bid strategy preferences
                if avg_reward > 1.5:
                    self.policy_state["bid_preferences"][bid] += learning_rate
                elif avg_reward < 1.0:
                    self.policy_state["bid_preferences"][bid] -= learning_rate * 0.5
        
        # Normalize preferences to maintain probability distribution
        self._normalize_preferences()
    
    def _normalize_preferences(self):
        """Normalize preference dictionaries to sum to 1"""
        for pref_dict in [self.policy_state["creative_preferences"], 
                         self.policy_state["audience_preferences"],
                         self.policy_state["bid_preferences"]]:
            
            # Ensure all values are positive
            min_val = min(pref_dict.values())
            if min_val <= 0:
                for key in pref_dict:
                    pref_dict[key] -= min_val - 0.01
            
            # Normalize to sum to 1
            total = sum(pref_dict.values())
            for key in pref_dict:
                pref_dict[key] /= total

class MockLLMPersona:
    """Mock LLM persona for simulation environment"""
    
    def __init__(self, persona_config: Dict[str, Any]):
        self.config = persona_config
        self.name = persona_config.get("name", "Anonymous")
        self.demographics = persona_config.get("demographics", {})
        self.interests = persona_config.get("interests", [])
        
    async def respond_to_ad(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user response to ad campaign"""
        # Simulate realistic user engagement based on persona
        base_ctr = 0.02  # 2% baseline CTR
        
        # Adjust based on campaign-persona match
        if campaign["target_audience"] == "young_adults" and self.demographics.get("age_group") == "18-25":
            base_ctr *= 1.5
        if campaign["creative_type"] == "video" and "entertainment" in self.interests:
            base_ctr *= 1.3
            
        # Add some randomness
        ctr = base_ctr * random.uniform(0.5, 2.0)
        
        # Calculate other metrics
        impressions = random.randint(1000, 5000)
        clicks = int(impressions * ctr)
        conversions = int(clicks * random.uniform(0.02, 0.08))  # 2-8% conversion rate
        
        return {
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "ctr": ctr,
            "cost": campaign["budget"],
            "revenue": conversions * random.uniform(20, 80)  # $20-80 per conversion
        }

async def create_mock_environment(environment_type: str):
    """Create simulation or real environment"""
    if environment_type == "simulation":
        # Create diverse LLM personas
        personas = [
            MockLLMPersona({
                "name": "Sarah (Tech Professional)",
                "demographics": {"age_group": "25-35", "income": "high"},
                "interests": ["technology", "productivity", "finance"]
            }),
            MockLLMPersona({
                "name": "Mike (College Student)",
                "demographics": {"age_group": "18-25", "income": "low"},
                "interests": ["entertainment", "gaming", "social"]
            }),
            MockLLMPersona({
                "name": "Jennifer (Working Mom)",
                "demographics": {"age_group": "35-45", "income": "medium"},
                "interests": ["family", "health", "home"]
            }),
            MockLLMPersona({
                "name": "Robert (Retiree)",
                "demographics": {"age_group": "65+", "income": "medium"},
                "interests": ["travel", "health", "hobbies"]
            })
        ]
        return {"type": "simulation", "personas": personas}
    else:
        return {"type": "real", "platform": "meta_ads", "budget_limit": 50}

async def run_training_episode(agent: MockAgent, environment: Dict[str, Any], episode_num: int):
    """Run a single training episode"""
    print(f"\n--- Episode {episode_num} ---")
    
    if environment["type"] == "simulation":
        # Simulation episode
        total_performance = {"revenue": 0, "cost": 0, "conversions": 0}
        
        for persona in environment["personas"]:
            # Agent selects campaign
            observation = {"persona": persona.config, "market_context": {}}
            campaign = await agent.select_action(observation)
            
            # Persona responds to campaign
            response = await persona.respond_to_ad(campaign)
            
            total_performance["revenue"] += response["revenue"]
            total_performance["cost"] += response["cost"]
            total_performance["conversions"] += response["conversions"]
            
        # Calculate ROAS (Return on Ad Spend)
        roas = total_performance["revenue"] / total_performance["cost"] if total_performance["cost"] > 0 else 0
        
        print(f"Simulation Episode: Revenue=${total_performance['revenue']:.2f}, "
              f"Cost=${total_performance['cost']:.2f}, ROAS={roas:.2f}x")
        
        return roas, total_performance
        
    else:
        # Real deployment episode (simulated for demo)
        observation = {"market_context": {"competition": "medium", "seasonality": "normal"}}
        campaign = await agent.select_action(observation)
        
        # Simulate real ad platform response (would be actual API calls)
        performance = {
            "impressions": random.randint(5000, 15000),
            "clicks": random.randint(100, 500),
            "conversions": random.randint(5, 25),
            "cost": campaign["budget"],
            "revenue": random.uniform(50, 200)
        }
        
        roas = performance["revenue"] / performance["cost"]
        
        print(f"Real Deployment: Revenue=${performance['revenue']:.2f}, "
              f"Cost=${performance['cost']:.2f}, ROAS={roas:.2f}x")
        
        return roas, performance

async def run_phase_training(agent: MockAgent, phase_name: str, num_episodes: int):
    """Run training for a specific phase"""
    print(f"\n🚀 Starting {phase_name}")
    print("=" * 60)
    
    # Create appropriate environment for phase
    if "Simulation" in phase_name:
        environment = await create_mock_environment("simulation")
    else:
        environment = await create_mock_environment("real")
    
    phase_performance = []
    
    for episode in range(1, num_episodes + 1):
        reward, performance = await run_training_episode(agent, environment, episode)
        phase_performance.append(reward)
        
        # Update agent policy
        await agent.update_policy(reward, performance)
        
        # Show progress every few episodes
        if episode % 5 == 0 or episode == num_episodes:
            avg_roas = sum(phase_performance[-5:]) / min(5, len(phase_performance))
            print(f"Episode {episode}: Recent avg ROAS = {avg_roas:.2f}x")
        
        # Small delay for realistic viewing
        await asyncio.sleep(0.1)
    
    final_avg = sum(phase_performance) / len(phase_performance)
    print(f"\n✅ {phase_name} Complete! Average ROAS: {final_avg:.2f}x")
    
    return final_avg, phase_performance

async def check_graduation_criteria(performance_history: list, phase: str) -> bool:
    """Check if agent meets criteria to graduate to next phase"""
    if len(performance_history) < 5:
        return False
        
    recent_avg = sum(performance_history[-5:]) / 5
    
    if phase == "simulation":
        return recent_avg > 1.5  # 1.5x ROAS in simulation
    elif phase == "small_budget":
        return recent_avg > 2.0  # 2.0x ROAS in real testing
    else:
        return True

async def main():
    """Main demo function"""
    print("🎯 GAELP Ad Campaign Learning Platform - Live Demo")
    print("=" * 60)
    print("This demo shows an agent learning to optimize ad campaigns")
    print("from scratch through the complete 4-phase training pipeline:")
    print("1. Simulation Training (LLM personas)")
    print("2. Historical Data Validation") 
    print("3. Small Budget Real Testing")
    print("4. Scaled Deployment")
    print("\nPress Ctrl+C at any time to stop the demo.\n")
    
    # Wait a moment for user to read
    await asyncio.sleep(2)
    
    # Create agent
    agent = MockAgent("agent-001")
    print(f"🤖 Created Agent: {agent.agent_id}")
    
    # Initialize orchestrator with development config
    config = DEVELOPMENT_CONFIG
    orchestrator = TrainingOrchestrator(config)
    
    try:
        # Phase 1: Simulation Training
        print("\n" + "="*60)
        print("PHASE 1: SIMULATION TRAINING")
        print("="*60)
        print("Agent learns on LLM personas responding to ad campaigns")
        print("This is safe, fast, and costs no real money.")
        
        sim_avg, sim_performance = await run_phase_training(
            agent, "Simulation Training", 20
        )
        
        if not await check_graduation_criteria(sim_performance, "simulation"):
            print("❌ Agent failed to meet simulation graduation criteria")
            return
        
        print("🎓 Agent graduated from simulation training!")
        
        # Phase 2: Historical Data Validation
        print("\n" + "="*60)
        print("PHASE 2: HISTORICAL DATA VALIDATION")
        print("="*60)
        print("Testing agent on real historical campaign data")
        
        hist_avg, hist_performance = await run_phase_training(
            agent, "Historical Validation", 10
        )
        
        # Phase 3: Small Budget Real Testing
        print("\n" + "="*60)
        print("PHASE 3: SMALL BUDGET REAL TESTING")
        print("="*60)
        print("Deploying agent with $10-50/day budget limits")
        print("Real Facebook/Google Ads with safety controls")
        
        real_avg, real_performance = await run_phase_training(
            agent, "Small Budget Real Testing", 15
        )
        
        if not await check_graduation_criteria(real_performance, "small_budget"):
            print("❌ Agent failed to meet real testing graduation criteria")
            return
            
        print("🎓 Agent graduated to scaled deployment!")
        
        # Phase 4: Scaled Deployment
        print("\n" + "="*60)
        print("PHASE 4: SCALED DEPLOYMENT")
        print("="*60)
        print("Agent managing larger budgets based on proven performance")
        
        scaled_avg, scaled_performance = await run_phase_training(
            agent, "Scaled Deployment", 10
        )
        
        # Final Results
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE - RESULTS SUMMARY")
        print("="*60)
        print(f"Phase 1 (Simulation):      {sim_avg:.2f}x ROAS")
        print(f"Phase 2 (Historical):      {hist_avg:.2f}x ROAS")
        print(f"Phase 3 (Small Budget):    {real_avg:.2f}x ROAS")
        print(f"Phase 4 (Scaled):          {scaled_avg:.2f}x ROAS")
        
        improvement = scaled_avg / sim_avg if sim_avg > 0 else 0
        print(f"\n📈 Total Improvement: {improvement:.1f}x better than initial performance")
        
        if scaled_avg > 3.0:
            print("🏆 EXCELLENT: Agent achieved superhuman performance!")
        elif scaled_avg > 2.0:
            print("✅ SUCCESS: Agent demonstrates strong optimization ability!")
        else:
            print("⚠️  MODERATE: Agent shows improvement but needs more training")
            
        print(f"\n🎯 Agent {agent.agent_id} is now ready for production deployment!")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())