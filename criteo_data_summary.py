"""
Criteo Dataset Summary and Usage Examples for GAELP

This script provides a comprehensive summary of the processed Criteo dataset
and demonstrates how to use it within the GAELP framework.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_and_summarize_criteo_data():
    """Load and summarize the processed Criteo dataset"""
    
    data_dir = Path("/home/hariravichandran/AELP/data")
    
    print("=== GAELP Criteo Dataset Summary ===\n")
    
    # Load statistics
    with open(data_dir / "criteo_statistics.json", 'r') as f:
        stats = json.load(f)
    
    with open(data_dir / "simulator_calibration.json", 'r') as f:
        calibration = json.load(f)
    
    print("📊 DATASET OVERVIEW")
    print(f"   Total Samples: {stats['total_samples']:,}")
    print(f"   Numerical Features: {stats['num_features']}")
    print(f"   Categorical Features: {stats['cat_features']}")
    print(f"   Click-Through Rate: {stats['click_rate']:.4f} ({stats['click_rate']*100:.2f}%)")
    print(f"   Total Features: {stats['num_features'] + stats['cat_features']}")
    
    print("\n🎯 CTR PREDICTION INSIGHTS")
    ctr_stats = calibration['ctr_statistics']
    print(f"   Baseline CTR: {ctr_stats['baseline_ctr']:.4f}")
    print(f"   CTR Standard Deviation: {ctr_stats['ctr_std']:.4f}")
    print(f"   High CTR Threshold: {ctr_stats['high_ctr_threshold']:.4f}")
    print(f"   Low CTR Threshold: {ctr_stats['low_ctr_threshold']:.4f}")
    
    print("\n🔍 FEATURE IMPORTANCE")
    print("   Top Numerical Features:")
    for i, (feature, importance) in enumerate(calibration['feature_importance']['top_numerical_features'][:3], 1):
        print(f"     {i}. {feature}: {importance:.4f}")
    
    print("   Top Categorical Features:")
    for i, (feature, importance) in enumerate(calibration['feature_importance']['top_categorical_features'][:3], 1):
        print(f"     {i}. {feature}: {importance:.4f}")
    
    # Load data splits info
    splits_dir = data_dir / "splits"
    train_size = len(pd.read_csv(splits_dir / "X_train.csv"))
    val_size = len(pd.read_csv(splits_dir / "X_val.csv"))
    test_size = len(pd.read_csv(splits_dir / "X_test.csv"))
    
    print("\n📈 DATA SPLITS")
    print(f"   Training Set: {train_size:,} samples ({train_size/stats['total_samples']*100:.1f}%)")
    print(f"   Validation Set: {val_size:,} samples ({val_size/stats['total_samples']*100:.1f}%)")
    print(f"   Test Set: {test_size:,} samples ({test_size/stats['total_samples']*100:.1f}%)")
    
    # Simulator calibration parameters
    sim_params = calibration['simulation_parameters']
    print("\n⚙️  SIMULATOR CALIBRATION")
    print(f"   Recommended Episode Length: {sim_params['recommended_episode_length']:,}")
    print(f"   Recommended Batch Size: {sim_params['recommended_batch_size']}")
    print(f"   Noise Level: {sim_params['noise_level']}")
    print(f"   Correlation Strength: {sim_params['correlation_strength']:.4f}")
    
    print("\n📁 GENERATED FILES")
    files = [
        "criteo_processed.csv - Preprocessed dataset with scaled features",
        "criteo_statistics.json - Comprehensive dataset statistics", 
        "simulator_calibration.json - Parameters for RL simulator calibration",
        "splits/X_train.csv - Training features",
        "splits/X_val.csv - Validation features", 
        "splits/X_test.csv - Test features",
        "splits/y_train.csv - Training labels",
        "splits/y_val.csv - Validation labels",
        "splits/y_test.csv - Test labels"
    ]
    
    for file_desc in files:
        print(f"   ✓ {file_desc}")
    
    return stats, calibration

def demonstrate_data_usage():
    """Demonstrate how to use the processed data"""
    
    print("\n\n=== USAGE EXAMPLES ===\n")
    
    data_dir = Path("/home/hariravichandran/AELP/data")
    splits_dir = data_dir / "splits"
    
    print("🔧 LOADING DATA FOR ML TRAINING")
    print("""
# Load training data
import pandas as pd
X_train = pd.read_csv('/home/hariravichandran/AELP/data/splits/X_train.csv')
y_train = pd.read_csv('/home/hariravichandran/AELP/data/splits/y_train.csv')

# Load validation data  
X_val = pd.read_csv('/home/hariravichandran/AELP/data/splits/X_val.csv')
y_val = pd.read_csv('/home/hariravichandran/AELP/data/splits/y_val.csv')

# Train your CTR prediction model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())
""")
    
    print("🎮 INTEGRATING WITH RL SIMULATOR")
    print("""
# Use calibration data for simulator setup
import json
with open('/home/hariravichandran/AELP/data/simulator_calibration.json', 'r') as f:
    calibration = json.load(f)

# Configure RL environment
baseline_ctr = calibration['ctr_statistics']['baseline_ctr']
episode_length = calibration['simulation_parameters']['recommended_episode_length']
batch_size = calibration['simulation_parameters']['recommended_batch_size']

# Use top features for feature importance weighting
top_features = calibration['feature_importance']['top_numerical_features'][:5]
""")
    
    print("💾 BIGQUERY INTEGRATION")
    print("""
# Export for BigQuery (already generated as criteo_training_data.json)
from criteo_data_loader import CriteoDataLoader

loader = CriteoDataLoader()
schema = loader.get_bigquery_schema()  # Get BigQuery table schema
export_file = loader.export_for_bigquery()  # Export as JSONL

# BigQuery table creation SQL
CREATE TABLE `your_project.your_dataset.criteo_training_data` (
  click INTEGER NOT NULL,
  num_0 FLOAT,
  num_1 FLOAT,
  ... # (39 more columns)
);
""")
    
    # Load actual sample data to show
    print("📊 SAMPLE DATA PREVIEW")
    try:
        X_train = pd.read_csv(splits_dir / "X_train.csv")
        y_train = pd.read_csv(splits_dir / "y_train.csv")
        
        print("\nTraining Features (first 3 rows, first 10 columns):")
        print(X_train.iloc[:3, :10])
        
        print(f"\nTraining Labels (first 10):")
        print(y_train.iloc[:10].values.flatten())
        
        print(f"\nFeature Statistics:")
        print(f"   Total Features: {X_train.shape[1]}")
        print(f"   Numerical Features: 13 (scaled to mean=0, std=1)")
        print(f"   Categorical Features: 26 (label encoded)")
        
    except Exception as e:
        print(f"   Error loading sample data: {e}")

def show_bigquery_integration_example():
    """Show how to integrate with BigQuery for GAELP storage"""
    
    print("\n\n=== BIGQUERY INTEGRATION FOR GAELP ===\n")
    
    print("🏗️  BIGQUERY SCHEMA DESIGN")
    print("""
-- Core table for storing training episodes
CREATE TABLE `gaelp.rl_training.criteo_episodes` (
  episode_id STRING NOT NULL,
  agent_id STRING NOT NULL, 
  environment_id STRING NOT NULL,
  created_at TIMESTAMP NOT NULL,
  episode_length INTEGER,
  total_reward FLOAT,
  final_ctr FLOAT,
  convergence_score FLOAT
);

-- Detailed transition data
CREATE TABLE `gaelp.rl_training.criteo_transitions` (
  episode_id STRING NOT NULL,
  step_number INTEGER NOT NULL,
  state_features REPEATED RECORD (
    feature_name STRING,
    feature_value FLOAT
  ),
  action_taken INTEGER,
  reward FLOAT,
  next_state_features REPEATED RECORD (
    feature_name STRING, 
    feature_value FLOAT
  ),
  predicted_ctr FLOAT,
  actual_click INTEGER
) PARTITION BY DATE(PARSE_TIMESTAMP('%Y%m%d', CAST(episode_id AS STRING)))
CLUSTER BY agent_id, environment_id;
""")
    
    print("📈 ANALYTICS QUERIES")
    print("""
-- Agent performance comparison
SELECT 
  agent_id,
  COUNT(*) as episodes,
  AVG(final_ctr) as avg_ctr,
  AVG(total_reward) as avg_reward,
  STDDEV(total_reward) as reward_std
FROM `gaelp.rl_training.criteo_episodes`
WHERE created_at >= DATETIME_SUB(CURRENT_DATETIME(), INTERVAL 7 DAY)
GROUP BY agent_id
ORDER BY avg_reward DESC;

-- Feature importance analysis
WITH feature_correlations AS (
  SELECT 
    sf.feature_name,
    CORR(sf.feature_value, t.reward) as correlation_with_reward,
    COUNT(*) as sample_count
  FROM `gaelp.rl_training.criteo_transitions` t,
  UNNEST(state_features) as sf
  GROUP BY sf.feature_name
)
SELECT * FROM feature_correlations 
ORDER BY ABS(correlation_with_reward) DESC;
""")
    
    print("🔄 STREAMING PIPELINE")
    print("""
# Dataflow pipeline for real-time ingestion
from apache_beam import Pipeline
from apache_beam.io import BigQueryIO

def process_rl_episode(episode_data):
    # Transform episode data to BigQuery format
    return {
        'episode_id': episode_data['id'],
        'agent_id': episode_data['agent_id'],
        'total_reward': episode_data['reward'],
        'final_ctr': episode_data['final_metrics']['ctr']
    }

with Pipeline() as pipeline:
    (pipeline 
     | 'ReadFromPubSub' >> beam.io.ReadFromPubSub(topic='rl-episodes')
     | 'ProcessEpisodes' >> beam.Map(process_rl_episode)
     | 'WriteToBigQuery' >> BigQueryIO.WriteToBigQuery(
         table='gaelp.rl_training.criteo_episodes',
         write_disposition=BigQueryIO.WriteDisposition.WRITE_APPEND
     ))
""")

def main():
    """Main function to run the summary"""
    
    # Load and summarize data
    stats, calibration = load_and_summarize_criteo_data()
    
    # Show usage examples
    demonstrate_data_usage()
    
    # Show BigQuery integration
    show_bigquery_integration_example()
    
    print("\n" + "="*60)
    print("🎉 CRITEO DATASET READY FOR GAELP!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. ✅ Data processed and split into train/val/test")
    print("2. ✅ Feature engineering and normalization complete")
    print("3. ✅ Simulator calibration parameters generated")
    print("4. ✅ BigQuery schema and integration plan ready")
    print("5. 🔄 Integrate with RL training pipeline")
    print("6. 🔄 Set up BigQuery tables and streaming")
    print("7. 🔄 Configure monitoring and analytics dashboards")
    print()
    print("Files ready for use:")
    print("  📊 criteo_data_loader.py - Main data processing class")
    print("  📈 /data/criteo_processed.csv - Processed dataset") 
    print("  📉 /data/splits/ - Train/validation/test splits")
    print("  ⚙️  /data/simulator_calibration.json - RL parameters")
    print("  📋 /data/criteo_statistics.json - Dataset statistics")

if __name__ == "__main__":
    main()