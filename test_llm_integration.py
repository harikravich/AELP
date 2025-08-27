#!/usr/bin/env python3
"""
Test script for GAELP LLM Integration
Validates that all components work correctly together
"""

import asyncio
import os
import sys
import json
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_persona_service import LLMPersonaService, LLMPersonaConfig, PersonaState
from persona_factory import PersonaFactory, PersonaTemplates


class LLMIntegrationTester:
    """Comprehensive test suite for LLM integration"""
    
    def __init__(self):
        self.service = None
        self.test_results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        
        print("🧪 GAELP LLM Integration Test Suite")
        print("=" * 40)
        
        tests = [
            ("Environment Check", self.test_environment),
            ("Service Initialization", self.test_service_init),
            ("Persona Creation", self.test_persona_creation),
            ("LLM Response", self.test_llm_response),
            ("State Management", self.test_state_management),
            ("Cost Tracking", self.test_cost_tracking),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                print(f"   {status}: {result.get('message', 'Test completed')}")
            except Exception as e:
                self.test_results[test_name] = {"success": False, "error": str(e)}
                print(f"   ❌ ERROR: {e}")
        
        return self.test_results
    
    async def test_environment(self) -> Dict[str, Any]:
        """Test environment setup"""
        
        checks = {
            "anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "redis_available": True  # Will test connection later
        }
        
        # Test Redis connection
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            checks["redis_available"] = True
        except:
            checks["redis_available"] = False
        
        api_keys_available = checks["anthropic_key"] or checks["openai_key"]
        
        return {
            "success": api_keys_available and checks["redis_available"],
            "message": f"API Keys: {api_keys_available}, Redis: {checks['redis_available']}",
            "details": checks
        }
    
    async def test_service_init(self) -> Dict[str, Any]:
        """Test service initialization"""
        
        try:
            config = LLMPersonaConfig(
                primary_provider="anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai",
                fallback_provider="openai" if os.getenv("OPENAI_API_KEY") else None,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                max_daily_cost=5.0,  # Conservative for testing
                requests_per_minute=10,
                log_level="WARNING"  # Reduce noise
            )
            
            self.service = LLMPersonaService(config)
            
            # Test health check
            health = await self.service.health_check()
            
            return {
                "success": health["service_status"] in ["healthy", "degraded"],
                "message": f"Service status: {health['service_status']}",
                "details": health
            }
            
        except Exception as e:
            return {"success": False, "message": f"Initialization failed: {e}"}
    
    async def test_persona_creation(self) -> Dict[str, Any]:
        """Test persona creation and registration"""
        
        if not self.service:
            return {"success": False, "message": "Service not initialized"}
        
        try:
            # Test different persona types
            personas_created = []
            
            # Template persona
            template_persona = PersonaTemplates.tech_early_adopter()
            persona_id1 = await self.service.create_persona(template_persona)
            personas_created.append(persona_id1)
            
            # Random persona
            random_persona = PersonaFactory.create_random_persona()
            persona_id2 = await self.service.create_persona(random_persona)
            personas_created.append(persona_id2)
            
            # Targeted persona
            targeted_persona = PersonaFactory.create_targeted_persona({
                "age_range": (25, 35),
                "interests": ["technology", "fitness"],
                "income_level": "high"
            })
            persona_id3 = await self.service.create_persona(targeted_persona)
            personas_created.append(persona_id3)
            
            return {
                "success": len(personas_created) == 3,
                "message": f"Created {len(personas_created)} personas",
                "details": {"persona_ids": personas_created}
            }
            
        except Exception as e:
            return {"success": False, "message": f"Persona creation failed: {e}"}
    
    async def test_llm_response(self) -> Dict[str, Any]:
        """Test LLM response generation"""
        
        if not self.service or not self.service.personas:
            return {"success": False, "message": "No personas available"}
        
        try:
            # Get first available persona
            persona_id = list(self.service.personas.keys())[0]
            
            # Create test campaign
            test_campaign = {
                "creative_type": "video",
                "target_audience": "young_adults",
                "budget": 20.0,
                "message": "Revolutionary new tech product!",
                "category": "technology",
                "brand": "TestBrand",
                "campaign_id": "test_001"
            }
            
            # Get LLM response
            response = await self.service.respond_to_ad(persona_id, test_campaign)
            
            # Validate response structure
            required_fields = [
                "impressions", "clicks", "conversions", "ctr", "cost", "revenue",
                "engagement_score", "emotional_response", "reasoning", "provider_used"
            ]
            
            missing_fields = [f for f in required_fields if f not in response]
            
            success = len(missing_fields) == 0 and isinstance(response["engagement_score"], float)
            
            return {
                "success": success,
                "message": f"LLM response generated (provider: {response.get('provider_used', 'unknown')})",
                "details": {
                    "engagement_score": response.get("engagement_score"),
                    "emotional_response": response.get("emotional_response"),
                    "provider_used": response.get("provider_used"),
                    "missing_fields": missing_fields
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"LLM response failed: {e}"}
    
    async def test_state_management(self) -> Dict[str, Any]:
        """Test persona state management"""
        
        if not self.service or not self.service.personas:
            return {"success": False, "message": "No personas available"}
        
        try:
            persona_id = list(self.service.personas.keys())[0]
            persona = self.service.personas[persona_id]
            
            # Record initial state
            initial_state = persona.history.state
            initial_fatigue = persona.history.fatigue_level
            initial_interactions = persona.history.interaction_count
            
            # Simulate multiple ad interactions
            campaign = {
                "creative_type": "image",
                "budget": 10.0,
                "category": "general"
            }
            
            for i in range(3):
                await self.service.respond_to_ad(persona_id, campaign)
            
            # Check state changes
            final_interactions = persona.history.interaction_count
            final_fatigue = persona.history.fatigue_level
            
            state_changed = final_interactions > initial_interactions
            fatigue_increased = final_fatigue >= initial_fatigue
            
            return {
                "success": state_changed and fatigue_increased,
                "message": f"State management working (interactions: {initial_interactions} → {final_interactions})",
                "details": {
                    "initial_state": initial_state.value,
                    "final_state": persona.history.state.value,
                    "fatigue_change": final_fatigue - initial_fatigue,
                    "interaction_change": final_interactions - initial_interactions
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"State management test failed: {e}"}
    
    async def test_cost_tracking(self) -> Dict[str, Any]:
        """Test cost tracking functionality"""
        
        if not self.service:
            return {"success": False, "message": "Service not initialized"}
        
        try:
            # Cost tracking is handled internally
            # We'll test that the service responds to cost limits
            
            # Check current cost tracking components exist
            has_rate_limiter = hasattr(self.service, 'rate_limiter')
            has_cost_tracker = hasattr(self.service, 'cost_tracker')
            
            # Test rate limiting check (should not throw error)
            if self.service.personas:
                persona_id = list(self.service.personas.keys())[0]
                await self.service.rate_limiter.check_limits(persona_id)
                await self.service.cost_tracker.check_limits()
            
            return {
                "success": has_rate_limiter and has_cost_tracker,
                "message": "Cost tracking components available",
                "details": {
                    "rate_limiter": has_rate_limiter,
                    "cost_tracker": has_cost_tracker
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"Cost tracking test failed: {e}"}
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and fallbacks"""
        
        if not self.service:
            return {"success": False, "message": "Service not initialized"}
        
        try:
            # Test with invalid persona ID
            try:
                await self.service.respond_to_ad("invalid_persona_id", {"budget": 10})
                invalid_persona_handled = False
            except Exception:
                invalid_persona_handled = True
            
            # Test service health check
            health = await self.service.health_check()
            health_check_works = "service_status" in health
            
            return {
                "success": invalid_persona_handled and health_check_works,
                "message": "Error handling working",
                "details": {
                    "invalid_persona_handled": invalid_persona_handled,
                    "health_check_works": health_check_works
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error handling test failed: {e}"}
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics"""
        
        if not self.service or not self.service.personas:
            return {"success": False, "message": "No personas available"}
        
        try:
            import time
            
            persona_id = list(self.service.personas.keys())[0]
            campaign = {"budget": 5.0, "category": "test"}
            
            # Time a single request
            start_time = time.time()
            response = await self.service.respond_to_ad(persona_id, campaign)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Performance thresholds
            fast_response = response_time < 10.0  # Under 10 seconds
            has_response = "engagement_score" in response
            
            return {
                "success": fast_response and has_response,
                "message": f"Response time: {response_time:.2f}s",
                "details": {
                    "response_time_seconds": response_time,
                    "under_10s": fast_response,
                    "provider_used": response.get("provider_used", "unknown")
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"Performance test failed: {e}"}
    
    def print_summary(self):
        """Print test summary"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        
        print(f"\n📊 TEST SUMMARY")
        print("=" * 20)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! LLM integration is working correctly.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check the details above.")
        
        return passed_tests == total_tests


async def main():
    """Run the test suite"""
    
    tester = LLMIntegrationTester()
    
    try:
        await tester.run_all_tests()
        all_passed = tester.print_summary()
        
        # Save test results
        with open("test_results.json", "w") as f:
            json.dump(tester.test_results, f, indent=2, default=str)
        
        print(f"\n📁 Test results saved to test_results.json")
        
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)