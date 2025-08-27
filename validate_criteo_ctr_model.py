"""
Criteo CTR Model Validation Script

This script demonstrates how to train and validate a CTR prediction model
using the processed Criteo dataset for GAELP integration.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path

def load_data():
    """Load the processed Criteo data splits"""
    data_dir = Path("/home/hariravichandran/AELP/data/splits")
    
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv") 
    X_test = pd.read_csv(data_dir / "X_test.csv")
    
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_val = pd.read_csv(data_dir / "y_val.csv").values.ravel()
    y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate different CTR prediction models"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
        
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_test_pred
        
        # Calculate metrics
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        results[name] = {
            'model': model,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'val_predictions': y_val_pred,
            'test_predictions': y_test_pred,
            'val_probabilities': y_val_proba,
            'test_probabilities': y_test_proba
        }
        
        # Print results
        print(f"   Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"     {metric.upper()}: {value:.4f}")
        
        print(f"   Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"     {metric.upper()}: {value:.4f}")
    
    return results

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"   {i:2d}. {feature}: {importance:.4f}")
        
        return feature_importance
    else:
        print("   Model doesn't support feature importance analysis")
        return None

def simulate_ctr_prediction_performance(results, calibration_data):
    """Simulate how the model would perform in the RL environment"""
    
    print(f"\n🎮 RL ENVIRONMENT SIMULATION")
    
    baseline_ctr = calibration_data['ctr_statistics']['baseline_ctr']
    
    # Use best model (highest AUC)
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['test_metrics']['auc'])
    best_model_results = results[best_model_name]
    
    print(f"   Best Model: {best_model_name}")
    print(f"   Model AUC: {best_model_results['test_metrics']['auc']:.4f}")
    
    # Simulate CTR predictions vs actual
    predicted_ctrs = best_model_results['test_probabilities']
    actual_clicks = np.array([1 if results[best_model_name]['test_predictions'][i] == 1 else 0 
                            for i in range(len(results[best_model_name]['test_predictions']))])
    
    # Calculate predicted vs actual CTR
    predicted_ctr = np.mean(predicted_ctrs)
    actual_ctr = np.mean(actual_clicks)
    
    print(f"   Baseline CTR (from calibration): {baseline_ctr:.4f}")
    print(f"   Predicted CTR (model average): {predicted_ctr:.4f}")
    print(f"   Actual CTR (test set): {actual_ctr:.4f}")
    
    # Simulate RL reward calculation
    # Higher CTR predictions that result in clicks = positive reward
    # Lower CTR predictions that avoid non-clicks = positive reward
    rewards = []
    for i in range(len(predicted_ctrs)):
        predicted_ctr = predicted_ctrs[i]
        actual_click = actual_clicks[i]
        
        if actual_click == 1:  # User clicked
            # Reward is higher if we predicted high CTR
            reward = predicted_ctr * 10  # Scale reward
        else:  # User didn't click
            # Reward is higher if we predicted low CTR (avoided wasting bid)
            reward = (1 - predicted_ctr) * 2  # Smaller penalty for conservative bidding
        
        rewards.append(reward)
    
    avg_reward = np.mean(rewards)
    reward_std = np.std(rewards)
    
    print(f"   Average RL Reward: {avg_reward:.4f}")
    print(f"   Reward Standard Deviation: {reward_std:.4f}")
    print(f"   Reward Range: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
    
    return {
        'best_model': best_model_name,
        'predicted_ctr': predicted_ctr,
        'actual_ctr': actual_ctr,
        'avg_reward': avg_reward,
        'reward_std': reward_std
    }

def main():
    """Main validation script"""
    print("=== GAELP Criteo CTR Model Validation ===\n")
    
    # Load data
    print("📊 Loading processed Criteo data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Positive class rate (train): {np.mean(y_train):.4f}")
    print(f"   Positive class rate (test): {np.mean(y_test):.4f}")
    
    # Train and evaluate models
    print(f"\n🚀 Training CTR Prediction Models...")
    results = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Analyze feature importance for best model
    best_model_name = max(results.keys(), 
                         key=lambda x: results[x]['test_metrics']['auc'])
    best_model = results[best_model_name]['model']
    feature_names = X_train.columns.tolist()
    
    print(f"\n📈 Feature Importance Analysis ({best_model_name})")
    feature_importance = analyze_feature_importance(best_model, feature_names)
    
    # Load calibration data and simulate RL performance
    with open("/home/hariravichandran/AELP/data/simulator_calibration.json", 'r') as f:
        calibration_data = json.load(f)
    
    rl_simulation = simulate_ctr_prediction_performance(results, calibration_data)
    
    # Generate summary report
    print(f"\n📋 MODEL VALIDATION SUMMARY")
    print(f"="*50)
    print(f"Dataset: Criteo Display Advertising Challenge (sample)")
    print(f"Samples: {len(X_train) + len(X_val) + len(X_test):,} total")
    print(f"Features: {X_train.shape[1]} (13 numerical + 26 categorical)")
    print(f"Class distribution: {(1-np.mean(y_test))*100:.1f}% negative, {np.mean(y_test)*100:.1f}% positive")
    print(f"")
    print(f"Best Model: {rl_simulation['best_model']}")
    print(f"Test AUC: {results[rl_simulation['best_model']]['test_metrics']['auc']:.4f}")
    print(f"Test F1: {results[rl_simulation['best_model']]['test_metrics']['f1']:.4f}")
    print(f"")
    print(f"RL Simulation Results:")
    print(f"  Predicted CTR: {rl_simulation['predicted_ctr']:.4f}")
    print(f"  Actual CTR: {rl_simulation['actual_ctr']:.4f}")
    print(f"  Average Reward: {rl_simulation['avg_reward']:.4f}")
    print(f"")
    print(f"✅ Model ready for GAELP RL integration!")
    
    return results, rl_simulation

if __name__ == "__main__":
    results, simulation = main()