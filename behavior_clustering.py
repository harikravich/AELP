#!/usr/bin/env python3
"""
Behavioral Clustering System - Discovers user segments from actual behavior
NO HARDCODED SEGMENTS - Everything emerges from data
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class BehaviorClusteringSystem:
    """Discovers user segments from behavioral patterns - NO HARDCODING"""
    
    def __init__(self):
        self.behavior_vectors = []
        self.cluster_labels = []
        self.cluster_profiles = {}
        self.scaler = StandardScaler()
        self.min_samples_for_clustering = 100
        
        # Track actual behaviors for each discovered cluster
        self.cluster_behaviors = defaultdict(lambda: {
            'page_views': [],
            'time_on_site': [],
            'scroll_depth': [],
            'clicks': [],
            'search_terms': [],
            'referrer_domains': [],
            'device_types': [],
            'hours_active': [],
            'return_visits': 0,
            'conversion_signals': []
        })
        
    def observe_behavior(self, user_id: str, behavior_data: Dict[str, Any]) -> None:
        """Observe actual user behavior - no predetermined categories"""
        
        # Extract behavioral features (not demographic assumptions)
        features = [
            behavior_data.get('time_on_page', 0),
            behavior_data.get('scroll_depth', 0),
            behavior_data.get('num_clicks', 0),
            behavior_data.get('search_query_length', 0),
            behavior_data.get('is_return_visit', 0),
            behavior_data.get('pages_viewed', 1),
            behavior_data.get('video_watch_time', 0),
            behavior_data.get('form_interactions', 0),
            behavior_data.get('hour_of_day', 12),
            behavior_data.get('day_of_week', 3),
            behavior_data.get('referrer_type', 0),  # 0=direct, 1=search, 2=social, 3=email
            behavior_data.get('device_category', 0),  # 0=desktop, 1=mobile, 2=tablet
            behavior_data.get('connection_speed', 1.0),
            behavior_data.get('cart_additions', 0),
            behavior_data.get('price_comparisons', 0),
            behavior_data.get('faq_views', 0),
            behavior_data.get('support_interactions', 0),
            behavior_data.get('download_attempts', 0),
            behavior_data.get('share_clicks', 0),
            behavior_data.get('bookmark_actions', 0)
        ]
        
        self.behavior_vectors.append(features)
        
        # Re-cluster periodically
        if len(self.behavior_vectors) >= self.min_samples_for_clustering and \
           len(self.behavior_vectors) % 50 == 0:
            self._discover_clusters()
    
    def _discover_clusters(self) -> None:
        """Use unsupervised learning to discover natural groupings"""
        
        if len(self.behavior_vectors) < self.min_samples_for_clustering:
            return
            
        # Normalize features
        X = np.array(self.behavior_vectors)
        X_scaled = self.scaler.fit_transform(X)
        
        # Try DBSCAN first for natural density-based clusters
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        
        # If DBSCAN finds too many outliers, fall back to KMeans
        n_outliers = np.sum(labels == -1)
        if n_outliers > len(labels) * 0.3:  # More than 30% outliers
            # Use elbow method to find optimal k
            optimal_k = self._find_optimal_clusters(X_scaled)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
        
        self.cluster_labels = labels
        self._profile_clusters(X, labels)
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        
        inertias = []
        K_range = range(2, min(max_k, len(X) // 10))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (maximum second derivative)
        if len(inertias) < 3:
            return 3  # Default
        
        second_derivatives = np.diff(np.diff(inertias))
        elbow_point = np.argmax(second_derivatives) + 2  # +2 because of double diff
        
        return min(max(elbow_point, 3), 8)  # Between 3 and 8 clusters
    
    def _profile_clusters(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Create profiles for discovered clusters based on actual behavior"""
        
        unique_labels = set(labels) - {-1}  # Exclude outliers
        
        for label in unique_labels:
            mask = labels == label
            cluster_data = X[mask]
            
            # Calculate behavioral statistics
            profile = {
                'size': np.sum(mask),
                'avg_time_on_page': np.mean(cluster_data[:, 0]),
                'avg_scroll_depth': np.mean(cluster_data[:, 1]),
                'avg_clicks': np.mean(cluster_data[:, 2]),
                'avg_pages_viewed': np.mean(cluster_data[:, 5]),
                'return_visit_rate': np.mean(cluster_data[:, 4]),
                'peak_hour': int(np.median(cluster_data[:, 8])),
                'primary_device': int(np.median(cluster_data[:, 11])),
                'engagement_score': self._calculate_engagement_score(cluster_data),
                'conversion_likelihood': self._estimate_conversion_likelihood(cluster_data)
            }
            
            # Generate descriptive name based on behavior
            profile['name'] = self._generate_cluster_name(profile)
            profile['description'] = self._generate_cluster_description(profile)
            
            self.cluster_profiles[label] = profile
    
    def _calculate_engagement_score(self, cluster_data: np.ndarray) -> float:
        """Calculate engagement based on multiple behavioral factors"""
        
        time_score = np.mean(cluster_data[:, 0]) / 300  # Normalize by 5 minutes
        depth_score = np.mean(cluster_data[:, 1])
        click_score = np.mean(cluster_data[:, 2]) / 10  # Normalize by 10 clicks
        page_score = np.mean(cluster_data[:, 5]) / 5  # Normalize by 5 pages
        
        # Weighted combination
        engagement = (time_score * 0.3 + depth_score * 0.2 + 
                     click_score * 0.25 + page_score * 0.25)
        
        return min(engagement, 1.0)
    
    def _estimate_conversion_likelihood(self, cluster_data: np.ndarray) -> float:
        """Estimate conversion likelihood from behavioral signals"""
        
        # High-intent behaviors
        cart_additions = np.mean(cluster_data[:, 13])
        price_comparisons = np.mean(cluster_data[:, 14])
        faq_views = np.mean(cluster_data[:, 15])
        download_attempts = np.mean(cluster_data[:, 17])
        
        # Calculate likelihood
        likelihood = (cart_additions * 0.4 + price_comparisons * 0.2 + 
                     faq_views * 0.2 + download_attempts * 0.2)
        
        return min(likelihood / 5, 1.0)  # Normalize
    
    def _generate_cluster_name(self, profile: Dict) -> str:
        """Generate descriptive name based on actual behavior"""
        
        engagement = profile['engagement_score']
        conversion = profile['conversion_likelihood']
        
        # Name based on behavior, not demographics
        if engagement > 0.7 and conversion > 0.6:
            return f"High_Intent_Engaged_{profile['size']}"
        elif engagement > 0.5 and conversion > 0.4:
            return f"Moderate_Intent_Active_{profile['size']}"
        elif engagement > 0.3:
            return f"Browsing_Exploratory_{profile['size']}"
        elif profile['return_visit_rate'] > 0.5:
            return f"Returning_Low_Engagement_{profile['size']}"
        else:
            return f"Passive_Visitor_{profile['size']}"
    
    def _generate_cluster_description(self, profile: Dict) -> str:
        """Generate human-readable description of cluster behavior"""
        
        desc_parts = []
        
        # Engagement level
        if profile['engagement_score'] > 0.7:
            desc_parts.append("Highly engaged users")
        elif profile['engagement_score'] > 0.4:
            desc_parts.append("Moderately engaged visitors")
        else:
            desc_parts.append("Low engagement browsers")
        
        # Time behavior
        if profile['avg_time_on_page'] > 180:
            desc_parts.append("spending significant time per page")
        
        # Click behavior
        if profile['avg_clicks'] > 5:
            desc_parts.append("actively clicking and exploring")
        
        # Return behavior
        if profile['return_visit_rate'] > 0.3:
            desc_parts.append("frequently returning")
        
        # Conversion likelihood
        if profile['conversion_likelihood'] > 0.5:
            desc_parts.append("showing high purchase intent")
        elif profile['conversion_likelihood'] > 0.2:
            desc_parts.append("showing moderate interest")
        
        return ". ".join(desc_parts) if desc_parts else "General traffic pattern"
    
    def get_cluster_for_behavior(self, behavior_data: Dict[str, Any]) -> Tuple[int, Dict]:
        """Get cluster assignment for new behavior"""
        
        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            return -1, {"name": "Unclustered", "description": "Not enough data for clustering"}
        
        if len(self.behavior_vectors) == 0:
            return -1, {"name": "No Data", "description": "No behavior data collected yet"}
        
        # Extract features
        features = [
            behavior_data.get('time_on_page', 0),
            behavior_data.get('scroll_depth', 0),
            behavior_data.get('num_clicks', 0),
            behavior_data.get('search_query_length', 0),
            behavior_data.get('is_return_visit', 0),
            behavior_data.get('pages_viewed', 1),
            behavior_data.get('video_watch_time', 0),
            behavior_data.get('form_interactions', 0),
            behavior_data.get('hour_of_day', 12),
            behavior_data.get('day_of_week', 3),
            behavior_data.get('referrer_type', 0),
            behavior_data.get('device_category', 0),
            behavior_data.get('connection_speed', 1.0),
            behavior_data.get('cart_additions', 0),
            behavior_data.get('price_comparisons', 0),
            behavior_data.get('faq_views', 0),
            behavior_data.get('support_interactions', 0),
            behavior_data.get('download_attempts', 0),
            behavior_data.get('share_clicks', 0),
            behavior_data.get('bookmark_actions', 0)
        ]
        
        # Find nearest cluster
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Simple nearest neighbor assignment
        min_dist = float('inf')
        best_cluster = -1
        
        # Get the last samples (up to 1000)
        num_samples = min(1000, len(self.behavior_vectors))
        start_idx = max(0, len(self.behavior_vectors) - num_samples)
        
        for i, vec in enumerate(self.behavior_vectors[start_idx:]):
            label_idx = start_idx + i
            if label_idx < len(self.cluster_labels):
                dist = np.linalg.norm(X_scaled[0] - self.scaler.transform([vec])[0])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = self.cluster_labels[label_idx]
        
        if best_cluster in self.cluster_profiles:
            return best_cluster, self.cluster_profiles[best_cluster]
        
        return -1, {"name": "Outlier", "description": "Unusual behavior pattern"}
    
    def get_all_clusters(self) -> Dict[int, Dict]:
        """Get all discovered clusters"""
        return self.cluster_profiles


# Global instance
behavior_clustering = BehaviorClusteringSystem()