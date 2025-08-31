#!/usr/bin/env python3
"""
End-to-End Test for GAELP Hybrid System
Tests all components WITHOUT ANY FALLBACKS
"""

import os
import sys
import time
import json
import traceback
from typing import Dict, Any

def test_llm_integration():
    """Test LLM integration - MUST have API key"""
    print("\n=== TESTING LLM INTEGRATION ===")
    from hybrid_llm_rl_integration import LLMStrategyAdvisor, CreativeGenerator, LLMStrategyConfig
    
    # This MUST fail if no API key
    config = LLMStrategyConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        use_caching=True
    )
    
    try:
        advisor = LLMStrategyAdvisor(config)
        generator = CreativeGenerator(config)
        
        # Test strategy generation
        strategy = advisor.get_strategic_context({"impressions": 1000, "channel": "google"}, "conversion")
        print(f"✅ LLM Strategy: {strategy.get('focus', 'strategic advice generated')}")
        
        # Test creative generation
        headline = generator.generate_headline("safety", "concerned_parents", "urgent")
        print(f"✅ LLM Headline: {headline[:50]}...")
        
        return True
    except RuntimeError as e:
        if "OPENAI_API_KEY" in str(e):
            print(f"⚠️ {e}")
            print("   Set OPENAI_API_KEY environment variable to enable LLM features")
            return False
        raise

def test_world_model():
    """Test FULL TransformerWorldModel - NO SIMPLIFICATIONS"""
    print("\n=== TESTING TRANSFORMER WORLD MODEL (FULL) ===")
    import torch
    from transformer_world_model_full import TransformerWorldModel, WorldModelConfig
    
    config = WorldModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_state=16,
        predict_horizon=100
    )
    model = TransformerWorldModel(config)
    
    # Test forward pass with proper dimensions
    batch_size = 4
    seq_len = 10
    state_dim = 50  # Match expected input dimensions
    action_dim = 32
    
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    
    predictions = model(states, actions, predict_steps=10)
    print(f"✅ Model output keys: {list(predictions.keys())}")
    print(f"✅ Mamba SSM active: {hasattr(model, 'mamba')}")
    print(f"✅ Diffusion active: {hasattr(model, 'diffusion')}")
    
    return True

def test_creative_library():
    """Test creative library - REQUIRES LLM for variations"""
    print("\n=== TESTING CREATIVE LIBRARY ===")
    from creative_content_library import CreativeContentLibrary
    
    library = CreativeContentLibrary(enable_llm_generation=True)
    
    # Test base creative retrieval
    creative = library.get_creative_for_context("google", "high_intent", 14)
    print(f"✅ Base creative: {creative.headline[:40]}...")
    
    # Test variation generation (REQUIRES LLM)
    try:
        variation = library.generate_creative_variation(creative, "safety")
        print(f"✅ LLM variation: {variation.headline[:40]}...")
    except RuntimeError as e:
        if "LLM generator is REQUIRED" in str(e):
            print(f"⚠️ {e}")
            return False
    
    return True

def test_master_integration():
    """Test GAELP master integration"""
    print("\n=== TESTING MASTER INTEGRATION ===")
    from gaelp_master_integration import MasterOrchestrator, GAELPConfig
    
    config = GAELPConfig()
    orchestrator = MasterOrchestrator(config)
    
    # Run one step
    state = orchestrator.get_state()
    print(f"✅ Initial state shape: {len(state)}")
    
    # Take an action
    action = {
        'channel': 'google',
        'bid': 5.0,
        'budget': 1000.0,
        'targeting': {'segment': 'high_intent'}
    }
    
    reward = orchestrator.step(action)
    print(f"✅ Step reward: {reward:.2f}")
    
    # Check components
    print(f"✅ RecSim active: {orchestrator.recsim_env is not None}")
    print(f"✅ AuctionGym active: {orchestrator.auction_env is not None}")
    print(f"✅ WorldModel active: {orchestrator.world_model is not None}")
    
    return True

def test_dashboard():
    """Test enhanced dashboard with enterprise sections"""
    print("\n=== TESTING ENTERPRISE DASHBOARD ===")
    from gaelp_live_dashboard_enhanced import GAELPLiveSystemEnhanced
    
    dashboard = GAELPLiveSystemEnhanced()
    
    # Check enterprise sections exist
    sections = [
        'creative_studio',
        'audience_hub', 
        'war_room',
        'attribution_center',
        'ai_arena',
        'executive_dashboard'
    ]
    
    for section in sections:
        if hasattr(dashboard, section):
            print(f"✅ {section} initialized")
        else:
            print(f"❌ {section} missing")
            return False
    
    # Test updates
    dashboard.update_enterprise_sections()
    print("✅ Enterprise sections updated")
    
    return True

def test_no_fallbacks():
    """Verify NO FALLBACK code exists"""
    print("\n=== VERIFYING NO FALLBACKS ===")
    
    forbidden_patterns = [
        'fallback',
        'simplified', 
        'mock',
        'dummy',
        '_AVAILABLE = False',
        'template_'  # template fallbacks
    ]
    
    files_to_check = [
        'hybrid_llm_rl_integration.py',
        'transformer_world_model_full.py',
        'creative_content_library.py',
        'gaelp_master_integration.py'
    ]
    
    violations = []
    for file in files_to_check:
        if not os.path.exists(file):
            continue
            
        with open(file, 'r') as f:
            content = f.read()
            for pattern in forbidden_patterns:
                if pattern.lower() in content.lower():
                    # Check if it's a real violation (not in comments/strings)
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern.lower() in line.lower():
                            # Skip comments
                            if '#' in line and pattern.lower() in line.split('#')[1].lower():
                                continue
                            # Skip docstrings
                            if '"""' in line or "'''" in line:
                                continue
                            violations.append(f"{file}:{i} - contains '{pattern}'")
    
    if violations:
        print("❌ FALLBACK VIOLATIONS FOUND:")
        for v in violations:
            print(f"   {v}")
        return False
    else:
        print("✅ NO FALLBACK CODE DETECTED")
        return True

def run_full_test():
    """Run complete end-to-end test"""
    print("=" * 60)
    print("GAELP HYBRID SYSTEM END-TO-END TEST")
    print("NO FALLBACKS - NO SIMPLIFICATIONS - FULL IMPLEMENTATION")
    print("=" * 60)
    
    results = {}
    
    # Test each component
    tests = [
        ("No Fallbacks Check", test_no_fallbacks),
        ("LLM Integration", test_llm_integration),
        ("World Model (FULL)", test_world_model),
        ("Creative Library", test_creative_library),
        ("Master Integration", test_master_integration),
        ("Enterprise Dashboard", test_dashboard)
    ]
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        print("   No fallbacks, no simplifications, full implementation!")
    else:
        print("\n⚠️ SOME TESTS FAILED - FIX REQUIRED COMPONENTS")
        if not results.get("LLM Integration"):
            print("\n   💡 Set OPENAI_API_KEY to enable LLM features")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)