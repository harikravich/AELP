#!/usr/bin/env python3
"""
Debug script to understand Criteo model issues
"""

import logging
import numpy as np
import pandas as pd

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

def debug_criteo_prediction():
    """Debug the CTR prediction process step by step"""
    print("🔍 DEBUGGING CRITEO CTR PREDICTION")
    print("=" * 50)
    
    from criteo_response_model import CriteoUserResponseModel
    
    # Initialize model
    print("1️⃣ Initializing model...")
    model = CriteoUserResponseModel()
    
    # Check if CTR model exists
    print(f"2️⃣ CTR model exists: {model.ctr_model is not None}")
    if model.ctr_model:
        print(f"   Model type: {model.ctr_model.model_type}")
        print(f"   Feature engineer fitted: {model.ctr_model.feature_engineer.is_fitted}")
        print(f"   Training metrics: {model.ctr_model.training_metrics}")
    
    # Test the feature creation
    print("3️⃣ Testing feature creation...")
    ad_content = {
        'category': 'parental_controls',
        'brand': 'gaelp',
        'price': 99.99,
        'creative_quality': 0.8
    }
    
    context = {
        'device': 'mobile',
        'hour': 20,
        'day_of_week': 5,
        'session_duration': 120,
        'page_views': 3,
        'geo_region': 'US',
        'user_segment': 'parents',
        'browser': 'chrome',
        'os': 'windows'
    }
    
    # Create Criteo features
    criteo_features = model._create_criteo_features(ad_content, context)
    print(f"   Created features: {len(criteo_features)} features")
    print(f"   Sample features: {list(criteo_features.keys())[:5]}")
    print(f"   Numerical features: {[k for k,v in criteo_features.items() if k.startswith('num_')][:3]}")
    print(f"   Categorical features: {[k for k,v in criteo_features.items() if k.startswith('cat_')][:3]}")
    
    # Test direct CTR prediction
    print("4️⃣ Testing direct CTR prediction...")
    try:
        ctr = model.predict_ctr(criteo_features)
        print(f"   Direct CTR prediction: {ctr}")
    except Exception as e:
        print(f"   Direct CTR prediction failed: {e}")
    
    # Test full user response
    print("5️⃣ Testing full user response simulation...")
    try:
        response = model.simulate_user_response(
            user_id="debug_user",
            ad_content=ad_content,
            context=context
        )
        print(f"   Full response: {response}")
    except Exception as e:
        print(f"   Full response failed: {e}")
    
    # Test with minimal features (just the base Criteo features)
    print("6️⃣ Testing with minimal feature set...")
    minimal_features = {}
    for i in range(13):
        minimal_features[f'num_{i}'] = np.random.normal(0, 1)
    for i in range(26):
        minimal_features[f'cat_{i}'] = np.random.randint(0, 100)
    
    try:
        minimal_ctr = model.predict_ctr(minimal_features)
        print(f"   Minimal features CTR: {minimal_ctr}")
    except Exception as e:
        print(f"   Minimal features failed: {e}")


def debug_criteo_training():
    """Debug the training process"""
    print("\n🔧 DEBUGGING CRITEO MODEL TRAINING")
    print("=" * 50)
    
    from criteo_response_model import CriteoUserResponseModel, CriteoCTRModel
    import pandas as pd
    
    # Check if we have actual Criteo data
    data_path = "/home/hariravichandran/AELP/data/criteo_processed.csv"
    
    print(f"1️⃣ Checking for Criteo data at {data_path}...")
    import os
    if os.path.exists(data_path):
        print("   ✅ Criteo data file exists")
        try:
            df = pd.read_csv(data_path)
            print(f"   Data shape: {df.shape}")
            print(f"   Columns: {list(df.columns)[:10]}...")
            print(f"   Click rate: {df['click'].mean():.4f}")
        except Exception as e:
            print(f"   ❌ Error reading data: {e}")
    else:
        print("   ⚠️  Criteo data file not found - using synthetic data")
    
    # Test model training directly
    print("2️⃣ Testing direct model training...")
    try:
        # Create synthetic training data
        n_samples = 1000
        X_data = {}
        
        # Create all Criteo features
        for i in range(13):
            X_data[f'num_{i}'] = np.random.normal(0, 1, n_samples)
        for i in range(26):
            X_data[f'cat_{i}'] = np.random.randint(0, 100, n_samples)
        
        X = pd.DataFrame(X_data)
        
        # Create realistic target with some signal
        y = np.random.binomial(1, 0.03 + 0.02 * (X['num_0'] > 0).astype(int), n_samples)
        
        print(f"   Training data shape: {X.shape}")
        print(f"   Target click rate: {y.mean():.4f}")
        
        # Train model
        ctr_model = CriteoCTRModel('gradient_boosting')
        ctr_model.fit(X, pd.Series(y))
        
        print(f"   Training metrics: {ctr_model.training_metrics}")
        
        # Test prediction
        test_row = pd.DataFrame([{f'num_{i}': np.random.normal(0, 1) for i in range(13)} | 
                                {f'cat_{i}': np.random.randint(0, 100) for i in range(26)}])
        pred_ctr = ctr_model.predict_ctr(test_row)[0]
        print(f"   Test prediction: {pred_ctr}")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    debug_criteo_prediction()
    debug_criteo_training()


if __name__ == "__main__":
    main()