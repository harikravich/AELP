#!/usr/bin/env python3
"""
Fetch real GA4 data for GAELP training
This script uses MCP tools to get actual data from GA4
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_ga4_data(data, filename):
    """Save GA4 data to file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved GA4 data to {filename}")

def discover_segments_from_ga4(user_data):
    """Discover segments from GA4 user data using clustering"""
    if not user_data or not user_data.get('rows'):
        logger.warning("No user data to cluster")
        return {}
    
    # Extract features from GA4 data
    features = []
    user_ids = []
    
    for row in user_data['rows']:
        dims = row.get('dimensionValues', [])
        metrics = row.get('metricValues', [])
        
        if len(metrics) >= 4:
            features.append([
                float(metrics[0].get('value', 0)),  # sessions
                float(metrics[1].get('value', 0)),  # pageViews
                float(metrics[2].get('value', 0)),  # avgSessionDuration
                float(metrics[3].get('value', 0))   # conversions
            ])
            user_ids.append(dims[0].get('value', 'unknown') if dims else 'unknown')
    
    if not features:
        logger.warning("No features extracted from GA4 data")
        return {}
    
    # Normalize and cluster
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    n_clusters = min(5, max(2, len(features) // 10))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_normalized)
    
    # Build segment profiles
    segments = {}
    for i, label in enumerate(cluster_labels):
        cluster_name = f"cluster_{label}"
        if cluster_name not in segments:
            segments[cluster_name] = {
                'users': [],
                'behavioral_metrics': {},
                'discovered_characteristics': {}
            }
        segments[cluster_name]['users'].append(user_ids[i])
    
    # Calculate segment statistics
    for cluster_name, cluster_data in segments.items():
        cluster_indices = [i for i, l in enumerate(cluster_labels) if f"cluster_{l}" == cluster_name]
        if cluster_indices:
            cluster_features = [features[i] for i in cluster_indices]
            segments[cluster_name]['behavioral_metrics'] = {
                'avg_sessions': np.mean([f[0] for f in cluster_features]),
                'avg_page_views': np.mean([f[1] for f in cluster_features]),
                'avg_session_duration': np.mean([f[2] for f in cluster_features]),
                'avg_conversions': np.mean([f[3] for f in cluster_features]),
                'sample_size': len(cluster_indices)
            }
    
    logger.info(f"Discovered {len(segments)} segments from GA4 data")
    return segments

def main():
    """Main function to fetch and process GA4 data"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           FETCHING REAL GA4 DATA FOR GAELP                ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # NOTE: The actual MCP calls will be made by Claude when this script is run
    # This is a placeholder structure that will be filled with real data
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_date} to {end_date}")
    print("\nThis script needs to be run with MCP GA4 tools available.")
    print("The data will be fetched and saved for GAELP training.")
    
    # Placeholder for MCP data
    # When run by Claude, this will be replaced with actual MCP calls
    ga4_data = {
        'user_behavior': {},
        'campaign_performance': {},
        'conversion_events': {},
        'segments': {}
    }
    
    # Save the fetched data
    output_dir = Path("ga4_training_data")
    output_dir.mkdir(exist_ok=True)
    
    save_ga4_data(ga4_data, output_dir / "ga4_raw_data.json")
    
    # Discover segments if we have user data
    if ga4_data.get('user_behavior'):
        segments = discover_segments_from_ga4(ga4_data['user_behavior'])
        save_ga4_data(segments, output_dir / "discovered_segments.json")
    
    print(f"\n✅ Data saved to {output_dir}/")
    print("\nNext steps:")
    print("1. Review discovered segments")
    print("2. Update discovered_patterns.json with real data")
    print("3. Run GAELP training with real patterns")

if __name__ == "__main__":
    main()