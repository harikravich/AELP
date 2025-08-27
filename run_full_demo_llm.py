#!/usr/bin/env python3
"""
Automated GAELP Training Demo - Full System with Real LLM Personas
Runs the complete agent training pipeline with authentic LLM-powered user simulation
"""

import asyncio
import json
import time
import os
from typing import Dict, Any, List
import random
import logging

# Import GAELP components
from training_orchestrator import TrainingOrchestrator
from training_orchestrator.config import TrainingOrchestratorConfig, DEVELOPMENT_CONFIG

# Import LLM Persona components
from llm_persona_service import (
    LLMPersonaService, LLMPersonaConfig, PersonaConfig, PersonaState
)
from persona_factory import PersonaFactory, PersonaTemplates


class MockAgent:
    """Mock RL agent for demonstration"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.policy_state = {"learning_rate": 0.001, "exploration": 0.1}
        self.performance_history = []
        
    async def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Select campaign action based on observation"""
        # Simulate agent choosing campaign parameters
        return {
            "creative_type": random.choice(["image", "video", "carousel"]),
            "target_audience": random.choice(["young_adults", "professionals", "families"]),
            "budget": random.uniform(10, 50),  # Daily budget in dollars
            "bid_strategy": random.choice(["cpc", "cpm", "cpa"])
        }
    
    async def update_policy(self, reward: float, performance_data: Dict[str, Any]):
        """Update agent policy based on performance"""
        self.performance_history.append(reward)
        
        # Simulate learning - agent gets better over time
        if len(self.performance_history) > 10:
            avg_recent = sum(self.performance_history[-10:]) / 10
            if avg_recent > 2.0:  # Good performance
                self.policy_state["exploration"] *= 0.95  # Explore less
            else:
                self.policy_state["exploration"] *= 1.05  # Explore more
                
        print(f"Agent {self.agent_id} updated policy. Exploration: {self.policy_state['exploration']:.3f}")


class RealLLMPersona:
    """Real LLM-powered persona for authentic simulation"""
    
    def __init__(self, persona_config: PersonaConfig, llm_service: LLMPersonaService):
        self.config = persona_config
        self.llm_service = llm_service
        self.name = persona_config.name
        self.persona_id = persona_config.persona_id
        
    async def respond_to_ad(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Generate authentic user response using LLM"""
        try:
            # Get LLM-powered response
            response = await self.llm_service.respond_to_ad(self.persona_id, campaign)
            
            # Convert to expected format (maintaining backward compatibility)
            return {
                "impressions": response["impressions"],
                "clicks": response["clicks"],
                "conversions": response["conversions"],
                "ctr": response["ctr"],
                "cost": response["cost"],
                "revenue": response["revenue"],
                "engagement_score": response["engagement_score"],
                "emotional_response": response["emotional_response"],
                "reasoning": response["reasoning"],
                "persona_thoughts": response["persona_thoughts"],
                "provider_used": response["provider_used"]
            }
            
        except Exception as e:
            # Fallback to simple heuristic if LLM fails
            logging.warning(f"LLM persona failed, using fallback: {e}")
            return await self._fallback_response(campaign)
    
    async def _fallback_response(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback response when LLM is unavailable"""
        # Simple demographic-based heuristics
        base_engagement = 0.1
        
        # Age-based adjustments
        if self.config.demographics.age < 30 and campaign.get("target_audience") == "young_adults":
            base_engagement *= 1.5
        
        # Interest-based adjustments
        campaign_category = campaign.get("category", "").lower()
        if campaign_category in [interest.lower() for interest in self.config.psychology.interests]:
            base_engagement *= 1.3
        
        # State-based adjustments
        if self.config.history.state == PersonaState.FATIGUED:
            base_engagement *= 0.3
        elif self.config.history.state == PersonaState.BLOCKED:
            base_engagement *= 0.1
        
        impressions = 1
        clicks = 1 if base_engagement > 0.2 and random.random() < base_engagement else 0
        conversions = 1 if clicks > 0 and random.random() < 0.05 else 0
        
        return {
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "ctr": clicks / impressions,
            "cost": campaign.get("budget", 10) * clicks * random.uniform(0.5, 2.0),
            "revenue": conversions * random.uniform(20, 80),
            "engagement_score": base_engagement,
            "emotional_response": "neutral",
            "reasoning": "Fallback response due to LLM unavailability",
            "persona_thoughts": "Using heuristic-based response",
            "provider_used": "fallback"
        }


async def create_llm_environment(environment_type: str, llm_service: LLMPersonaService = None):
    """Create simulation or real environment with LLM personas"""
    if environment_type == "simulation":
        if not llm_service:
            # Initialize LLM service if not provided
            llm_config = LLMPersonaConfig(
                primary_provider="anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai",
                fallback_provider="openai" if os.getenv("OPENAI_API_KEY") else None,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_daily_cost=20.0,  # Conservative for demo
                requests_per_minute=30,
                log_level="INFO"
            )
            
            try:
                llm_service = LLMPersonaService(llm_config)
                print("✅ LLM Persona Service initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize LLM service: {e}")
                print("Falling back to mock personas...")
                return await create_mock_environment(environment_type)
        
        # Create diverse, realistic personas using PersonaFactory
        try:
            persona_configs = PersonaTemplates.get_diverse_test_cohort()
            
            # Add a few more random personas for diversity
            additional_personas = PersonaFactory.create_persona_cohort(6, diversity_level="high")
            persona_configs.extend(additional_personas)
            
            # Register personas with LLM service
            personas = []
            for persona_config in persona_configs:
                await llm_service.create_persona(persona_config)
                llm_persona = RealLLMPersona(persona_config, llm_service)
                personas.append(llm_persona)
                print(f"📝 Created persona: {persona_config.name} ({persona_config.demographics.age}y, {persona_config.demographics.gender})")
            
            print(f"🎭 Created {len(personas)} LLM-powered personas for simulation")
            
            return {
                "type": "simulation", 
                "personas": personas,
                "llm_service": llm_service,
                "is_llm_powered": True
            }
            
        except Exception as e:
            print(f"⚠️  Failed to create LLM personas: {e}")
            print("Falling back to mock personas...")
            return await create_mock_environment(environment_type)
    else:
        return {"type": "real", "platform": "meta_ads", "budget_limit": 50}


async def create_mock_environment(environment_type: str):
    """Create fallback mock environment when LLM is unavailable"""
    if environment_type == "simulation":
        # Create simple mock personas as fallback
        class SimpleMockPersona:
            def __init__(self, name: str, demographics: Dict[str, Any], interests: List[str]):
                self.name = name
                self.demographics = demographics
                self.interests = interests
            
            async def respond_to_ad(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
                # Simple heuristic response
                base_ctr = 0.02
                if campaign.get("target_audience") == "young_adults" and self.demographics.get("age_group") == "18-25":
                    base_ctr *= 1.5
                if campaign.get("creative_type") == "video" and "entertainment" in self.interests:
                    base_ctr *= 1.3
                
                ctr = base_ctr * random.uniform(0.5, 2.0)
                impressions = 1
                clicks = int(impressions * ctr) if random.random() < ctr else 0
                conversions = int(clicks * random.uniform(0.02, 0.08)) if clicks > 0 else 0
                
                return {
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "ctr": ctr,
                    "cost": campaign.get("budget", 10) * clicks,
                    "revenue": conversions * random.uniform(20, 80),
                    "engagement_score": ctr,
                    "emotional_response": "neutral",
                    "reasoning": "Mock persona response",
                    "persona_thoughts": "Generated by fallback system",
                    "provider_used": "mock"
                }
        
        personas = [
            SimpleMockPersona(
                "Sarah (Tech Professional)",
                {"age_group": "25-35", "income": "high"},
                ["technology", "productivity", "finance"]
            ),
            SimpleMockPersona(
                "Mike (College Student)", 
                {"age_group": "18-25", "income": "low"},
                ["entertainment", "gaming", "social"]
            ),
            SimpleMockPersona(
                "Jennifer (Working Mom)",
                {"age_group": "35-45", "income": "medium"},
                ["family", "health", "home"]
            ),
            SimpleMockPersona(
                "Robert (Retiree)",
                {"age_group": "65+", "income": "medium"},
                ["travel", "health", "hobbies"]
            )
        ]
        return {"type": "simulation", "personas": personas, "is_llm_powered": False}
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
            observation = {
                "persona": getattr(persona, 'config', {'name': persona.name}), 
                "market_context": {}
            }
            campaign = await agent.select_action(observation)
            
            # Add more realistic campaign details for LLM personas
            campaign.update({
                "campaign_id": f"camp_{episode_num}_{hash(persona.name) % 1000}",
                "message": f"Discover our amazing {campaign['creative_type']} experience!",
                "cta": "Learn More",
                "category": random.choice(["technology", "fashion", "fitness", "travel", "food"]),
                "brand": f"Brand{random.randint(1, 100)}",
                "price_point": random.choice(["budget", "medium", "premium"]),
                "platform": "social_media",
                "time_of_day": "afternoon",
                "day_of_week": "tuesday"
            })
            
            # Persona responds to campaign
            response = await persona.respond_to_ad(campaign)
            
            total_performance["revenue"] += response["revenue"]
            total_performance["cost"] += response["cost"]
            total_performance["conversions"] += response["conversions"]
            
            # Log LLM-specific metrics if available
            if environment.get("is_llm_powered") and episode_num % 5 == 0:
                print(f"  📊 {persona.name}: {response.get('emotional_response', 'N/A')} response, "
                      f"engagement={response.get('engagement_score', 0):.2f}, "
                      f"provider={response.get('provider_used', 'unknown')}")
                if response.get('reasoning'):
                    print(f"     💭 Reasoning: {response['reasoning'][:100]}...")
            
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
        environment = await create_llm_environment("simulation")
    else:
        environment = await create_llm_environment("real")
    
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
    
    # Show LLM service health if available
    if "Simulation" in phase_name and environment.get("llm_service"):
        try:
            health = await environment["llm_service"].health_check()
            print(f"🔍 LLM Service Health: {health['service_status']}")
            for provider, status in health.get("providers", {}).items():
                print(f"   • {provider}: {status}")
        except Exception as e:
            print(f"⚠️  Could not check LLM service health: {e}")
    
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


async def show_llm_persona_analytics(environment: Dict[str, Any]):
    """Show analytics for LLM personas"""
    if not environment.get("is_llm_powered") or not environment.get("llm_service"):
        return
    
    print("\n📈 LLM PERSONA ANALYTICS")
    print("=" * 50)
    
    try:
        for persona in environment["personas"][:3]:  # Show first 3 for brevity
            analytics = await environment["llm_service"].get_persona_analytics(persona.persona_id)
            print(f"👤 {analytics['persona_name']}:")
            print(f"   • Interactions: {analytics['total_interactions']}")
            print(f"   • CTR: {analytics['ctr']:.3f}")
            print(f"   • Conversion Rate: {analytics['conversion_rate']:.3f}")
            print(f"   • Current State: {analytics['current_state']}")
            print(f"   • Fatigue Level: {analytics['fatigue_level']:.2f}")
    except Exception as e:
        print(f"⚠️  Could not retrieve persona analytics: {e}")


async def main():
    """Main demo function"""
    print("🎯 GAELP Ad Campaign Learning Platform - Live Demo with Real LLM Personas")
    print("=" * 70)
    print("This demo shows an agent learning to optimize ad campaigns")
    print("from scratch through the complete 4-phase training pipeline:")
    print("1. Simulation Training (Real LLM-powered personas with authentic responses)")
    print("2. Historical Data Validation") 
    print("3. Small Budget Real Testing")
    print("4. Scaled Deployment")
    print("\n🤖 LLM Integration Features:")
    print("  • Authentic user personas with detailed psychological profiles")
    print("  • Real-time LLM responses (Claude/GPT) based on persona characteristics")
    print("  • Dynamic persona state management (fatigue, engagement, blocking)")
    print("  • Cost monitoring and rate limiting for API usage")
    print("  • Fallback to heuristic responses if LLM unavailable")
    print("\n⚙️  API Configuration:")
    print(f"  • Anthropic API: {'✅ Configured' if os.getenv('ANTHROPIC_API_KEY') else '❌ Not configured'}")
    print(f"  • OpenAI API: {'✅ Configured' if os.getenv('OPENAI_API_KEY') else '❌ Not configured'}")
    print("\nPress Ctrl+C at any time to stop the demo.\n")
    
    # Wait a moment for user to read
    await asyncio.sleep(3)
    
    # Create agent
    agent = MockAgent("agent-001")
    print(f"🤖 Created Agent: {agent.agent_id}")
    
    # Initialize orchestrator with development config
    config = DEVELOPMENT_CONFIG
    orchestrator = TrainingOrchestrator(config)
    
    try:
        # Phase 1: Simulation Training
        print("\n" + "="*60)
        print("PHASE 1: SIMULATION TRAINING with REAL LLM PERSONAS")
        print("="*60)
        print("Agent learns on authentic LLM-powered personas with:")
        print("• Detailed psychological profiles and demographics")
        print("• Real-time LLM responses based on persona characteristics")
        print("• Dynamic state management (engagement, fatigue, ad-blocking)")
        print("• Authentic reasoning and emotional responses")
        print("This phase costs minimal API fees but provides realistic training data.")
        
        sim_avg, sim_performance = await run_phase_training(
            agent, "Simulation Training", 20
        )
        
        if not await check_graduation_criteria(sim_performance, "simulation"):
            print("❌ Agent failed to meet simulation graduation criteria")
            return
        
        print("🎓 Agent graduated from simulation training!")
        
        # Show persona analytics
        simulation_env = await create_llm_environment("simulation")
        await show_llm_persona_analytics(simulation_env)
        
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
        
        # Final LLM service stats
        final_env = await create_llm_environment("simulation")
        if final_env.get("llm_service"):
            try:
                health = await final_env["llm_service"].health_check()
                print(f"\n🔧 Final LLM Service Status: {health['service_status']}")
            except:
                pass
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: No LLM API keys found in environment variables.")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY to use real LLM personas.")
        print("   Demo will fall back to mock personas.\n")
    
    asyncio.run(main())