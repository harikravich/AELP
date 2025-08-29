"""
Dynamic Segment Discovery System
Discovers user segments from behavioral patterns without hardcoding
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class BehavioralFeatures:
    """Features extracted from user behavior for clustering"""
    # Engagement features
    touchpoint_count: int = 0
    session_duration: float = 0.0
    pages_viewed: int = 0
    scroll_depth: float = 0.0
    video_watch_time: float = 0.0
    
    # Intent features
    search_query_length: int = 0
    form_interactions: int = 0
    download_attempts: int = 0
    comparison_views: int = 0
    pricing_page_views: int = 0
    
    # Urgency features
    time_of_day: int = 0  # Hour
    weekend_activity: bool = False
    late_night_session: bool = False  # 10pm-4am
    rapid_clicks: int = 0  # Clicks within 30 seconds
    
    # Journey features
    days_since_first_touch: float = 0.0
    revisit_count: int = 0
    channel_diversity: int = 1  # Number of different channels used
    device_switches: int = 0
    
    # Content engagement
    blog_reads: int = 0
    feature_page_views: int = 0
    testimonial_views: int = 0
    faq_interactions: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy array for clustering"""
        return np.array([
            self.touchpoint_count,
            self.session_duration / 60,  # Convert to minutes
            self.pages_viewed,
            self.scroll_depth,
            self.video_watch_time / 60,
            self.search_query_length,
            self.form_interactions,
            self.download_attempts,
            self.comparison_views,
            self.pricing_page_views,
            self.time_of_day / 24,  # Normalize
            float(self.weekend_activity),
            float(self.late_night_session),
            self.rapid_clicks,
            self.days_since_first_touch,
            self.revisit_count,
            self.channel_diversity,
            self.device_switches,
            self.blog_reads,
            self.feature_page_views,
            self.testimonial_views,
            self.faq_interactions
        ])

@dataclass
class DiscoveredSegment:
    """A dynamically discovered user segment"""
    segment_id: str
    cluster_center: np.ndarray
    size: int
    behavioral_profile: Dict[str, float]
    conversion_rate: float = 0.0
    avg_conversion_time: float = 0.0
    avg_order_value: float = 0.0
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def get_name(self) -> str:
        """Generate descriptive name based on behavioral profile"""
        # Analyze dominant characteristics
        profile = self.behavioral_profile
        
        # Urgency level
        if profile.get('late_night_session', 0) > 0.5 and profile.get('rapid_clicks', 0) > 3:
            urgency = "urgent"
        elif profile.get('days_since_first_touch', 0) < 1:
            urgency = "immediate"
        else:
            urgency = "patient"
        
        # Research depth
        if profile.get('touchpoint_count', 0) > 10:
            depth = "thorough"
        elif profile.get('comparison_views', 0) > 3:
            depth = "comparing"
        elif profile.get('pages_viewed', 0) < 3:
            depth = "quick"
        else:
            depth = "exploring"
        
        # Intent level
        if profile.get('form_interactions', 0) > 0 or profile.get('pricing_page_views', 0) > 2:
            intent = "high-intent"
        elif profile.get('faq_interactions', 0) > 1:
            intent = "questioning"
        else:
            intent = "browsing"
        
        return f"{urgency}_{depth}_{intent}"
    
    def calculate_similarity(self, features: BehavioralFeatures) -> float:
        """Calculate similarity to this segment"""
        feature_vector = features.to_vector()
        # Euclidean distance normalized by dimensionality
        distance = np.linalg.norm(feature_vector - self.cluster_center)
        max_distance = np.sqrt(len(feature_vector)) * 10  # Approximate max
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, similarity))

class DynamicSegmentDiscovery:
    """System for discovering user segments without hardcoding"""
    
    def __init__(self, min_samples: int = 5, learning_rate: float = 0.01):
        self.segments: Dict[str, DiscoveredSegment] = {}
        self.feature_history: List[BehavioralFeatures] = []
        self.conversion_history: List[Tuple[BehavioralFeatures, bool, float]] = []
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.last_clustering = None
        
    def observe_user(self, features: BehavioralFeatures, 
                     converted: bool = False, 
                     conversion_value: float = 0.0):
        """Observe user behavior and outcomes"""
        self.feature_history.append(features)
        self.conversion_history.append((features, converted, conversion_value))
        
        # Re-cluster periodically
        if len(self.feature_history) >= self.min_samples and \
           len(self.feature_history) % 50 == 0:
            self.discover_segments()
    
    def discover_segments(self):
        """Discover segments using clustering"""
        if len(self.feature_history) < self.min_samples:
            return
        
        # Convert features to matrix
        X = np.array([f.to_vector() for f in self.feature_history])
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Try DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=max(3, self.min_samples // 2))
        labels = dbscan.fit_predict(X_scaled)
        
        # If DBSCAN finds too few clusters, use KMeans
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 3:
            # Use elbow method to find optimal k
            k = min(8, max(3, len(self.feature_history) // 20))
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            cluster_centers = self.scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            # Calculate cluster centers for DBSCAN
            cluster_centers = []
            for label in set(labels):
                if label != -1:
                    mask = labels == label
                    center = np.mean(X[mask], axis=0)
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers)
        
        # Create segment objects
        self.segments = {}
        for i, center in enumerate(cluster_centers):
            # Get all users in this cluster
            mask = labels == i
            cluster_features = [f for j, f in enumerate(self.feature_history) if mask[j]]
            
            # Calculate behavioral profile
            profile = self._calculate_profile(cluster_features)
            
            # Calculate conversion metrics
            cluster_conversions = [(f, c, v) for j, (f, c, v) in enumerate(self.conversion_history) if mask[j]]
            conv_rate = sum(1 for _, c, _ in cluster_conversions if c) / max(1, len(cluster_conversions))
            avg_value = np.mean([v for _, c, v in cluster_conversions if c and v > 0]) if any(c for _, c, _ in cluster_conversions) else 0
            
            segment = DiscoveredSegment(
                segment_id=f"segment_{i}",
                cluster_center=center,
                size=int(np.sum(mask)),
                behavioral_profile=profile,
                conversion_rate=conv_rate,
                avg_order_value=avg_value
            )
            
            self.segments[segment.get_name()] = segment
        
        logger.info(f"Discovered {len(self.segments)} segments: {list(self.segments.keys())}")
        self.last_clustering = datetime.now()
    
    def _calculate_profile(self, features: List[BehavioralFeatures]) -> Dict[str, float]:
        """Calculate average behavioral profile for a segment"""
        if not features:
            return {}
        
        profile = {}
        for attr in ['touchpoint_count', 'session_duration', 'pages_viewed', 
                    'rapid_clicks', 'late_night_session', 'form_interactions',
                    'pricing_page_views', 'comparison_views', 'days_since_first_touch',
                    'faq_interactions']:
            values = [getattr(f, attr, 0) for f in features]
            if isinstance(values[0], bool):
                profile[attr] = sum(values) / len(values)
            else:
                profile[attr] = np.mean(values) if values else 0
        
        return profile
    
    def assign_segment(self, features: BehavioralFeatures) -> Tuple[str, DiscoveredSegment]:
        """Assign user to closest discovered segment"""
        if not self.segments:
            # Bootstrap with initial segment
            self.observe_user(features)
            return "exploring_browsing", self._create_default_segment()
        
        # Find closest segment
        best_segment = None
        best_similarity = -1
        
        for name, segment in self.segments.items():
            similarity = segment.calculate_similarity(features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_segment = segment
        
        # If similarity too low, might be new segment emerging
        if best_similarity < 0.3:
            self.observe_user(features)
            # Trigger re-clustering if enough new observations
            if len(self.feature_history) > len(self.segments) * 20:
                self.discover_segments()
        
        return best_segment.get_name() if best_segment else "unknown", best_segment or self._create_default_segment()
    
    def _create_default_segment(self) -> DiscoveredSegment:
        """Create default segment for bootstrapping"""
        return DiscoveredSegment(
            segment_id="default",
            cluster_center=np.zeros(22),
            size=1,
            behavioral_profile={},
            conversion_rate=0.02
        )
    
    def get_segment_insights(self) -> Dict[str, Any]:
        """Get insights about discovered segments"""
        insights = {
            'total_segments': len(self.segments),
            'total_observations': len(self.feature_history),
            'segments': {}
        }
        
        for name, segment in self.segments.items():
            insights['segments'][name] = {
                'size': segment.size,
                'conversion_rate': round(segment.conversion_rate, 4),
                'avg_order_value': round(segment.avg_order_value, 2),
                'key_behaviors': self._extract_key_behaviors(segment.behavioral_profile),
                'discovered_at': segment.discovered_at.isoformat()
            }
        
        return insights
    
    def _extract_key_behaviors(self, profile: Dict[str, float]) -> List[str]:
        """Extract key behavioral characteristics"""
        behaviors = []
        
        if profile.get('late_night_session', 0) > 0.3:
            behaviors.append('late-night-searcher')
        if profile.get('rapid_clicks', 0) > 3:
            behaviors.append('urgent-need')
        if profile.get('comparison_views', 0) > 2:
            behaviors.append('comparison-shopper')
        if profile.get('touchpoint_count', 0) > 10:
            behaviors.append('thorough-researcher')
        if profile.get('form_interactions', 0) > 0:
            behaviors.append('high-intent')
        if profile.get('pricing_page_views', 0) > 1:
            behaviors.append('price-conscious')
        
        return behaviors

# Global instance
segment_discovery = DynamicSegmentDiscovery()