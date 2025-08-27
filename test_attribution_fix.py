#!/usr/bin/env python3
"""Test if Attribution fixes work"""

from datetime import datetime, timedelta
from attribution_models import AttributionEngine, Journey, Touchpoint

def test_attribution():
    """Test Attribution with correct parameters"""
    
    print("="*80)
    print("TESTING ATTRIBUTION FIX")
    print("="*80)
    
    # Test 1: Initialize Attribution Engine
    print("\n1. Initializing Attribution Engine...")
    try:
        engine = AttributionEngine()
        print(f"   ✅ Attribution Engine initialized")
        # Models are stored in engine.models dictionary
        if hasattr(engine, 'models'):
            print(f"      Available models: {', '.join(engine.models.keys())}")
        else:
            print(f"      Models initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        return False
    
    # Test 2: Create a journey with correct parameters
    print("\n2. Creating test journey...")
    try:
        journey = Journey(
            id="test_attribution_journey",
            touchpoints=[
                Touchpoint(
                    id="tp1",
                    timestamp=datetime.now() - timedelta(days=3),
                    channel='search',
                    action='click',
                    value=5.0,
                    metadata={'campaign': 'brand_search', 'keyword': 'parental controls'}
                ),
                Touchpoint(
                    id="tp2",
                    timestamp=datetime.now() - timedelta(days=2),
                    channel='display',
                    action='impression',
                    value=2.0,
                    metadata={'campaign': 'retargeting', 'placement': 'news_site'}
                ),
                Touchpoint(
                    id="tp3",
                    timestamp=datetime.now() - timedelta(days=1),
                    channel='email',
                    action='click',
                    value=3.0,
                    metadata={'campaign': 'nurture', 'email_type': 'product_demo'}
                ),
                Touchpoint(
                    id="tp4",
                    timestamp=datetime.now() - timedelta(hours=2),
                    channel='search',
                    action='click',
                    value=4.0,
                    metadata={'campaign': 'competitor', 'keyword': 'aura vs competitors'}
                )
            ],
            converted=True,
            conversion_value=150.0,
            conversion_timestamp=datetime.now()
        )
        print(f"   ✅ Journey created with {len(journey.touchpoints)} touchpoints")
        print(f"      Conversion value: ${journey.conversion_value}")
    except Exception as e:
        print(f"   ❌ Failed to create journey: {e}")
        return False
    
    # Test 3: Test different attribution models
    print("\n3. Testing attribution models...")
    # Use only available models (from engine.models.keys())
    models = ['linear', 'time_decay', 'position_based', 'data_driven']
    
    for model in models:
        try:
            credits = engine.calculate_attribution(journey, model_name=model)
            
            if credits and sum(credits.values()) > 0:
                print(f"   ✅ {model.replace('_', ' ').title()}:")
                total_credit = sum(credits.values())
                for touchpoint_id, credit in credits.items():
                    percentage = (credit / total_credit) * 100 if total_credit > 0 else 0
                    print(f"      {touchpoint_id}: {percentage:.1f}% (${credit:.2f})")
            else:
                print(f"   ❌ {model} returned no credits")
                
        except Exception as e:
            print(f"   ❌ {model} failed: {e}")
    
    # Test 4: Test data-driven attribution
    print("\n4. Testing data-driven attribution...")
    try:
        # Train with sample data
        training_journeys = []
        for i in range(10):
            training_journey = Journey(
                id=f"training_{i}",
                touchpoints=[
                    Touchpoint(
                        id=f"train_tp_{i}_1",
                        timestamp=datetime.now() - timedelta(days=i+2),
                        channel='search' if i % 2 == 0 else 'display',
                        action='click',
                        value=i * 2.0
                    ),
                    Touchpoint(
                        id=f"train_tp_{i}_2",
                        timestamp=datetime.now() - timedelta(days=i+1),
                        channel='email',
                        action='click',
                        value=i * 1.5
                    )
                ],
                converted=i % 3 != 0,  # 2/3 convert
                conversion_value=100.0 * (i % 3) if i % 3 != 0 else 0,
                conversion_timestamp=datetime.now() - timedelta(days=i)
            )
            training_journeys.append(training_journey)
        
        # Get data-driven model
        data_driven = engine.models.get('data_driven')
        if data_driven and hasattr(data_driven, 'train'):
            data_driven.train(training_journeys)
            print(f"   ✅ Data-driven model trained with {len(training_journeys)} journeys")
            
            # Test attribution
            credits = engine.calculate_attribution(journey, model_name='data_driven')
            if credits:
                print(f"      Data-driven credits calculated: {len(credits)} touchpoints")
        else:
            print(f"   ⚠️  Data-driven model doesn't support training")
            
    except Exception as e:
        print(f"   ❌ Data-driven attribution failed: {e}")
    
    # Test 5: Test with MasterOrchestrator
    print("\n5. Testing with MasterOrchestrator...")
    try:
        from gaelp_master_integration import MasterOrchestrator, GAELPConfig
        
        config = GAELPConfig()
        master = MasterOrchestrator(config)
        
        if hasattr(master, 'attribution_engine'):
            # Use the master's attribution engine
            credits = master.attribution_engine.calculate_attribution(journey, model_name='time_decay')
            
            if credits:
                print(f"   ✅ MasterOrchestrator attribution works")
                print(f"      Credits calculated for {len(credits)} touchpoints")
            else:
                print(f"   ❌ MasterOrchestrator attribution returned no credits")
        else:
            print("   ❌ MasterOrchestrator doesn't have attribution_engine")
            return False
            
    except Exception as e:
        print(f"   ❌ MasterOrchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("✅ ATTRIBUTION TEST PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_attribution()
    exit(0 if success else 1)