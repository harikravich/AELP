#!/usr/bin/env python3
"""
LLM Integration Setup and Configuration for GAELP
Handles setup, testing, and configuration of LLM persona services
"""

import asyncio
import os
import json
import time
from typing import Dict, Any, Optional, List

from llm_persona_service import LLMPersonaService, LLMPersonaConfig
from persona_factory import PersonaFactory, PersonaTemplates


class LLMIntegrationManager:
    """Manages LLM integration setup and configuration"""
    
    def __init__(self):
        self.config: Optional[LLMPersonaConfig] = None
        self.service: Optional[LLMPersonaService] = None
        
    async def setup_integration(self, config_overrides: Dict[str, Any] = None) -> bool:
        """
        Set up LLM integration with automatic configuration
        
        Args:
            config_overrides: Optional configuration overrides
            
        Returns:
            bool: True if setup successful
        """
        
        print("🚀 Setting up GAELP LLM Integration...")
        print("=" * 50)
        
        # Step 1: Check API keys
        api_status = self._check_api_keys()
        if not any(api_status.values()):
            print("❌ No LLM API keys found. Cannot proceed with setup.")
            return False
        
        # Step 2: Create configuration
        self.config = self._create_config(api_status, config_overrides)
        
        # Step 3: Initialize service
        try:
            self.service = LLMPersonaService(self.config)
            print("✅ LLM Persona Service initialized")
        except Exception as e:
            print(f"❌ Failed to initialize LLM service: {e}")
            return False
        
        # Step 4: Health check
        health_ok = await self._health_check()
        if not health_ok:
            print("⚠️  Health check failed")
            return False
        
        # Step 5: Test persona creation
        test_ok = await self._test_persona_creation()
        if not test_ok:
            print("⚠️  Persona creation test failed")
            return False
        
        # Step 6: Test LLM response
        response_ok = await self._test_llm_response()
        if not response_ok:
            print("⚠️  LLM response test failed")
            return False
        
        print("\n🎉 LLM Integration setup complete!")
        return True
    
    def _check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are available"""
        
        api_status = {
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY"))
        }
        
        print("🔑 API Key Status:")
        for provider, available in api_status.items():
            status = "✅ Available" if available else "❌ Not set"
            print(f"   • {provider.title()}: {status}")
        
        return api_status
    
    def _create_config(self, api_status: Dict[str, bool], overrides: Dict[str, Any] = None) -> LLMPersonaConfig:
        """Create LLM configuration based on available APIs"""
        
        # Determine primary and fallback providers
        if api_status["anthropic"]:
            primary = "anthropic"
            fallback = "openai" if api_status["openai"] else None
        elif api_status["openai"]:
            primary = "openai"
            fallback = None
        else:
            raise ValueError("No API keys available")
        
        config = LLMPersonaConfig(
            primary_provider=primary,
            fallback_provider=fallback,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            
            # Conservative defaults for setup
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=2000,
            max_daily_cost=50.0,
            
            # Caching settings
            cache_ttl_seconds=300,
            
            # Monitoring
            enable_monitoring=True,
            log_level="INFO"
        )
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        print(f"⚙️  Configuration created:")
        print(f"   • Primary provider: {config.primary_provider}")
        print(f"   • Fallback provider: {config.fallback_provider}")
        print(f"   • Daily cost limit: ${config.max_daily_cost}")
        
        return config
    
    async def _health_check(self) -> bool:
        """Perform health check on LLM service"""
        
        print("🏥 Performing health check...")
        
        try:
            health = await self.service.health_check()
            
            print(f"   • Service status: {health['service_status']}")
            print(f"   • Redis status: {health['redis_status']}")
            
            for provider, status in health.get("providers", {}).items():
                print(f"   • {provider}: {status}")
            
            return health["service_status"] in ["healthy", "degraded"]
            
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return False
    
    async def _test_persona_creation(self) -> bool:
        """Test persona creation"""
        
        print("👤 Testing persona creation...")
        
        try:
            # Create a simple test persona
            test_persona = PersonaTemplates.tech_early_adopter()
            persona_id = await self.service.create_persona(test_persona)
            
            print(f"   ✅ Created test persona: {test_persona.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Persona creation failed: {e}")
            return False
    
    async def _test_llm_response(self) -> bool:
        """Test LLM response generation"""
        
        print("🤖 Testing LLM response generation...")
        
        try:
            # Create test campaign
            test_campaign = {
                "creative_type": "image",
                "target_audience": "young_adults",
                "budget": 20.0,
                "message": "Discover our amazing new product!",
                "category": "technology",
                "brand": "TestBrand"
            }
            
            # Get personas
            personas = list(self.service.personas.keys())
            if not personas:
                print("   ❌ No personas available for testing")
                return False
            
            # Test response
            test_persona_id = personas[0]
            response = await self.service.respond_to_ad(test_persona_id, test_campaign)
            
            print(f"   ✅ LLM response received:")
            print(f"      • Engagement: {response['engagement_score']:.3f}")
            print(f"      • Emotional response: {response['emotional_response']}")
            print(f"      • Provider: {response['provider_used']}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ LLM response test failed: {e}")
            return False
    
    async def create_test_cohort(self, size: int = 10, diversity: str = "high") -> List[str]:
        """Create a test cohort of personas"""
        
        if not self.service:
            raise ValueError("LLM service not initialized")
        
        print(f"👥 Creating test cohort of {size} personas...")
        
        persona_configs = PersonaFactory.create_persona_cohort(size, diversity)
        persona_ids = []
        
        for i, persona_config in enumerate(persona_configs):
            try:
                persona_id = await self.service.create_persona(persona_config)
                persona_ids.append(persona_id)
                print(f"   {i+1:2d}. {persona_config.name} ({persona_config.demographics.age}y)")
            except Exception as e:
                print(f"   ❌ Failed to create persona {i+1}: {e}")
        
        print(f"✅ Created {len(persona_ids)} personas successfully")
        return persona_ids
    
    async def run_test_campaign(self, campaign_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a test campaign against all personas"""
        
        if not self.service:
            raise ValueError("LLM service not initialized")
        
        # Default test campaign
        if not campaign_config:
            campaign_config = {
                "creative_type": "video",
                "target_audience": "young_adults",
                "budget": 25.0,
                "message": "Experience the future with our innovative product!",
                "category": "technology",
                "brand": "InnovateCorp",
                "price_point": "premium"
            }
        
        print("🎯 Running test campaign...")
        print(f"   Campaign: {campaign_config.get('message', 'Test Campaign')}")
        
        personas = list(self.service.personas.keys())
        results = []
        
        for persona_id in personas:
            try:
                response = await self.service.respond_to_ad(persona_id, campaign_config)
                results.append(response)
                
                persona_name = self.service.personas[persona_id].name
                print(f"   📊 {persona_name}: "
                      f"engagement={response['engagement_score']:.2f}, "
                      f"clicked={response['clicks']}, "
                      f"converted={response['conversions']}")
                
            except Exception as e:
                print(f"   ❌ Failed to get response for persona {persona_id}: {e}")
        
        # Calculate aggregate metrics
        total_impressions = sum(r["impressions"] for r in results)
        total_clicks = sum(r["clicks"] for r in results)
        total_conversions = sum(r["conversions"] for r in results)
        total_cost = sum(r["cost"] for r in results)
        total_revenue = sum(r["revenue"] for r in results)
        
        aggregate_results = {
            "campaign": campaign_config,
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "ctr": total_clicks / max(1, total_impressions),
            "conversion_rate": total_conversions / max(1, total_clicks),
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "roas": total_revenue / max(0.01, total_cost),
            "individual_results": results
        }
        
        print(f"\n📈 Campaign Results:")
        print(f"   • CTR: {aggregate_results['ctr']:.3f}")
        print(f"   • Conversion Rate: {aggregate_results['conversion_rate']:.3f}")
        print(f"   • ROAS: {aggregate_results['roas']:.2f}x")
        
        return aggregate_results
    
    async def get_service_analytics(self) -> Dict[str, Any]:
        """Get comprehensive service analytics"""
        
        if not self.service:
            raise ValueError("LLM service not initialized")
        
        analytics = {
            "service_health": await self.service.health_check(),
            "total_personas": len(self.service.personas),
            "persona_analytics": {}
        }
        
        # Get analytics for each persona
        for persona_id in self.service.personas.keys():
            try:
                persona_analytics = await self.service.get_persona_analytics(persona_id)
                analytics["persona_analytics"][persona_id] = persona_analytics
            except Exception as e:
                print(f"⚠️  Could not get analytics for persona {persona_id}: {e}")
        
        return analytics
    
    def save_configuration(self, filename: str = "llm_config.json"):
        """Save current configuration to file"""
        
        if not self.config:
            raise ValueError("No configuration to save")
        
        config_dict = {
            "primary_provider": self.config.primary_provider,
            "fallback_provider": self.config.fallback_provider,
            "requests_per_minute": self.config.requests_per_minute,
            "requests_per_hour": self.config.requests_per_hour,
            "requests_per_day": self.config.requests_per_day,
            "max_daily_cost": self.config.max_daily_cost,
            "cache_ttl_seconds": self.config.cache_ttl_seconds,
            "log_level": self.config.log_level
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Configuration saved to {filename}")


async def interactive_setup():
    """Interactive setup process"""
    
    print("🎯 GAELP LLM Integration - Interactive Setup")
    print("=" * 50)
    
    manager = LLMIntegrationManager()
    
    # Basic setup
    setup_success = await manager.setup_integration()
    if not setup_success:
        print("❌ Setup failed. Please check your configuration.")
        return
    
    print("\n🎭 Creating test personas...")
    await manager.create_test_cohort(size=5, diversity="high")
    
    print("\n🚀 Running test campaign...")
    results = await manager.run_test_campaign()
    
    print("\n📊 Getting service analytics...")
    analytics = await manager.get_service_analytics()
    
    print(f"\n✅ Setup complete! Service is ready with {analytics['total_personas']} personas")
    
    # Save configuration
    manager.save_configuration()
    
    return manager


async def quick_test():
    """Quick test of LLM integration"""
    
    print("⚡ Quick LLM Integration Test")
    print("=" * 30)
    
    try:
        manager = LLMIntegrationManager()
        success = await manager.setup_integration({
            "max_daily_cost": 5.0,  # Very conservative for testing
            "requests_per_minute": 10
        })
        
        if success:
            # Create one test persona
            persona = PersonaTemplates.tech_early_adopter()
            await manager.service.create_persona(persona)
            
            # Test response
            test_campaign = {
                "message": "Quick test campaign",
                "category": "technology",
                "budget": 10.0
            }
            
            response = await manager.service.respond_to_ad(persona.persona_id, test_campaign)
            
            print(f"✅ Test successful!")
            print(f"   • Engagement: {response['engagement_score']:.3f}")
            print(f"   • Provider: {response['provider_used']}")
            
        else:
            print("❌ Quick test failed")
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(quick_test())
    else:
        asyncio.run(interactive_setup())