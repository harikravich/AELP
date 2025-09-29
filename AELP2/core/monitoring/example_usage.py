#!/usr/bin/env python3
"""
Example usage of BigQueryWriter for AELP2 monitoring.

This script demonstrates how to integrate BigQueryWriter into your AELP2 
training pipeline for real-time monitoring and analysis.
"""

import os
import time
import random
from datetime import datetime, timezone
from bq_writer import BigQueryWriter, BigQueryConfig

def simulate_training_episode(episode_id: str, model_version: str) -> dict:
    """Simulate a training episode with realistic metrics."""
    steps = random.randint(800, 2000)
    spend = random.uniform(100, 500)
    revenue = spend * random.uniform(0.8, 1.5)  # 80% to 150% ROAS
    conversions = random.randint(5, 25)
    win_rate = random.uniform(0.4, 0.9)
    avg_cpc = spend / max(steps * win_rate, 1)  # Realistic CPC
    epsilon = max(0.01, random.uniform(0.05, 0.3))  # Exploration rate
    
    return {
        'episode_id': episode_id,
        'steps': steps,
        'spend': round(spend, 2),
        'revenue': round(revenue, 2),
        'conversions': conversions,
        'win_rate': round(win_rate, 3),
        'avg_cpc': round(avg_cpc, 3),
        'epsilon': round(epsilon, 3),
        'model_version': model_version
    }

def simulate_safety_event(episode_id: str) -> dict:
    """Simulate a safety event during training."""
    event_types = ['bid_explosion', 'loss_divergence', 'epsilon_stuck', 'gradient_explosion']
    severities = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    event_type = random.choice(event_types)
    severity = random.choice(severities)
    
    return {
        'episode_id': episode_id,
        'event_type': event_type,
        'severity': severity,
        'metadata': {
            'threshold_exceeded': random.uniform(1.5, 10.0),
            'current_value': random.uniform(10.0, 100.0),
            'detection_method': 'automated_monitoring'
        },
        'action_taken': 'reduced_learning_rate' if severity in ['HIGH', 'CRITICAL'] else 'logged_only'
    }

def simulate_ab_experiment(experiment_id: str) -> dict:
    """Simulate A/B experiment result."""
    variants = ['control', 'treatment_a', 'treatment_b']
    variant = random.choice(variants)
    
    # Treatment variants perform slightly better on average
    performance_multiplier = 1.1 if 'treatment' in variant else 1.0
    
    revenue = random.uniform(50, 200) * performance_multiplier
    spend = revenue * random.uniform(0.7, 0.9)  # Generally profitable
    conversions = random.randint(3, 15)
    
    return {
        'experiment_id': experiment_id,
        'variant': variant,
        'metrics': {
            'revenue': round(revenue, 2),
            'spend': round(spend, 2),
            'conversions': conversions,
            'win_rate': round(random.uniform(0.5, 0.8), 3),
            'profit': round(revenue - spend, 2)
        },
        'user_id': f'user_{random.randint(1000, 9999)}',
        'session_id': f'session_{random.randint(100000, 999999)}'
    }

def main():
    """Main example demonstrating BigQueryWriter usage."""
    
    # Setup environment (in production, these would be set in your deployment)
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'your-gcp-project-id'
    os.environ['BIGQUERY_TRAINING_DATASET'] = 'aelp_training'
    
    print("AELP2 BigQuery Monitoring Example")
    print("=================================")
    
    try:
        # Initialize BigQuery writer
        print("1. Initializing BigQuery writer...")
        with BigQueryWriter() as bq_writer:
            
            # Create tables if they don't exist
            print("2. Creating tables if needed...")
            bq_writer.create_tables_if_not_exist()
            
            print("3. Simulating training episodes with monitoring...")
            
            # Simulate 10 training episodes
            for episode_num in range(1, 11):
                episode_id = f'episode_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{episode_num:03d}'
                model_version = 'qlearning_v2.1'
                
                # Generate episode metrics
                episode_data = simulate_training_episode(episode_id, model_version)
                bq_writer.write_episode_metrics(episode_data)
                
                print(f"   Episode {episode_num}: Profit=${episode_data['revenue'] - episode_data['spend']:.2f}, "
                      f"Win Rate={episode_data['win_rate']:.1%}, Epsilon={episode_data['epsilon']:.3f}")
                
                # Occasionally generate safety events (20% chance)
                if random.random() < 0.2:
                    safety_event = simulate_safety_event(episode_id)
                    bq_writer.write_safety_event(safety_event)
                    print(f"   ⚠️  Safety Event: {safety_event['event_type']} ({safety_event['severity']})")
                
                # Generate A/B experiment data (30% chance)
                if random.random() < 0.3:
                    ab_result = simulate_ab_experiment('bidding_strategy_test_v1')
                    bq_writer.write_ab_result(ab_result)
                    print(f"   🧪 A/B Result: {ab_result['variant']} - Profit=${ab_result['metrics']['profit']:.2f}")
                
                # Small delay to simulate real training time
                time.sleep(0.1)
            
            print("4. Forcing flush of all queued writes...")
            bq_writer.flush_all()
            
            print("5. Example queries you can run:")
            print("   - Check training performance: SELECT * FROM training_episodes ORDER BY timestamp DESC LIMIT 10")
            print("   - Monitor safety events: SELECT * FROM safety_events WHERE severity IN ('HIGH', 'CRITICAL')")
            print("   - A/B experiment results: SELECT experiment_id, variant, AVG(JSON_EXTRACT_SCALAR(metrics, '$.profit')) FROM ab_experiments GROUP BY 1,2")
            
            print("\n✅ Example completed successfully!")
            print("   Check your BigQuery console to see the data.")
            print(f"   Dataset: {bq_writer.config.training_dataset}")
            print("   Tables: training_episodes, safety_events, ab_experiments")
            
    except Exception as e:
        print(f"❌ Error during example execution: {e}")
        print("\nNote: This example requires valid GCP credentials and BigQuery access.")
        print("Set up authentication with: gcloud auth application-default login")

if __name__ == "__main__":
    main()