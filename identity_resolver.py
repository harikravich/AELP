"""
Identity Resolution System for Cross-Device User Tracking

This module implements probabilistic identity resolution to match users across
different devices and sessions. It uses multiple signals including behavioral
patterns, temporal proximity, geographic location, and search patterns to
determine if different device/session identifiers belong to the same user.

Key Features:
- Multi-signal probabilistic matching
- Confidence scoring for identity matches
- Identity graph management and updates
- Journey merging across devices
- Temporal and geographic proximity analysis
- Behavioral fingerprinting
"""

import hashlib
import json
import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import math
import statistics
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchConfidence(Enum):
    """Confidence levels for identity matches"""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # 0.3 - 0.5
    VERY_LOW = "very_low"  # < 0.3


@dataclass
class DeviceSignature:
    """Represents the behavioral and contextual signature of a device/session"""
    device_id: str
    user_agent: str = ""
    screen_resolution: str = ""
    timezone: str = ""
    language: str = ""
    platform: str = ""
    browser: str = ""
    last_seen: Optional[datetime] = None
    
    # Behavioral patterns
    search_patterns: List[str] = field(default_factory=list)
    page_sequences: List[List[str]] = field(default_factory=list)
    session_durations: List[float] = field(default_factory=list)
    click_patterns: List[Dict] = field(default_factory=list)
    time_of_day_usage: List[int] = field(default_factory=list)  # Hours 0-23
    
    # Geographic and temporal data
    ip_addresses: Set[str] = field(default_factory=set)
    geographic_locations: List[Tuple[float, float]] = field(default_factory=list)  # (lat, lon)
    session_timestamps: List[datetime] = field(default_factory=list)
    
    # Derived features
    avg_session_duration: float = 0.0
    primary_usage_hours: Set[int] = field(default_factory=set)
    behavioral_hash: str = ""


@dataclass
class IdentityMatch:
    """Represents a potential identity match between two devices"""
    device_id_1: str
    device_id_2: str
    confidence_score: float
    confidence_level: MatchConfidence
    matching_signals: List[str]
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IdentityCluster:
    """Represents a cluster of devices belonging to the same identity"""
    identity_id: str
    device_ids: Set[str]
    primary_device_id: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    merged_journey: List[Dict] = field(default_factory=list)


class IdentityResolver:
    """
    Main class for cross-device identity resolution.
    
    Implements probabilistic matching using multiple signals:
    - Behavioral patterns (search, navigation, timing)
    - Geographic proximity
    - Temporal patterns
    - Technical fingerprinting
    
    NOW WITH iOS 14.5+ PRIVACY REALISM:
    - 35% maximum match rate to reflect real-world cross-device limitations
    - Probabilistic failures to simulate privacy restrictions
    """
    
    # iOS 14.5+ Reality: Only 35% of cross-device matches succeed
    MAX_MATCH_RATE = 0.35  # Real-world limitation
    
    def __init__(self, 
                 min_confidence_threshold: float = 0.3,
                 high_confidence_threshold: float = 0.8,
                 medium_confidence_threshold: float = 0.5,
                 time_window_hours: int = 24,
                 max_geographic_distance_km: float = 50.0):
        """
        Initialize the identity resolver.
        
        Args:
            min_confidence_threshold: Minimum confidence to consider a match
            high_confidence_threshold: Threshold for high confidence matches
            medium_confidence_threshold: Threshold for medium confidence matches
            time_window_hours: Time window for temporal proximity analysis
            max_geographic_distance_km: Maximum distance for geographic matching
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.time_window_hours = time_window_hours
        self.max_geographic_distance_km = max_geographic_distance_km
        
        # Storage
        self.device_signatures: Dict[str, DeviceSignature] = {}
        self.identity_graph: Dict[str, IdentityCluster] = {}
        self.device_to_identity: Dict[str, str] = {}
        
        # Caching for performance
        self.match_cache: Dict[Tuple[str, str], IdentityMatch] = {}
        self.behavioral_similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Weights for different matching signals
        self.signal_weights = {
            'behavioral_similarity': 0.25,
            'temporal_proximity': 0.20,
            'geographic_proximity': 0.15,
            'search_pattern_similarity': 0.15,
            'session_pattern_similarity': 0.10,
            'technical_similarity': 0.10,
            'usage_time_similarity': 0.05
        }
    
    def add_device_signature(self, signature: DeviceSignature) -> None:
        """Add or update a device signature in the system"""
        # Calculate derived features
        if signature.session_durations:
            signature.avg_session_duration = statistics.mean(signature.session_durations)
        
        if signature.time_of_day_usage:
            # Find primary usage hours (top 3 most common hours)
            hour_counts = defaultdict(int)
            for hour in signature.time_of_day_usage:
                hour_counts[hour] += 1
            
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            signature.primary_usage_hours = set([h[0] for h in sorted_hours[:3]])
        
        # Generate behavioral hash
        signature.behavioral_hash = self._generate_behavioral_hash(signature)
        
        self.device_signatures[signature.device_id] = signature
        logger.info(f"Added device signature for {signature.device_id}")
    
    def resolve(self, device_id: str) -> str:
        """Resolve device ID to canonical user ID"""
        canonical = self.resolve_identity(device_id)
        return canonical if canonical else device_id
    
    def resolve_identity(self, device_id: str, force_recalculate: bool = False) -> Optional[str]:
        """
        Resolve the identity for a given device ID.
        
        NOW WITH iOS 14.5+ REALISM:
        - Only 35% of cross-device matches succeed due to privacy restrictions
        - Even high-confidence matches can fail to simulate real-world limitations
        
        Args:
            device_id: Device ID to resolve
            force_recalculate: Whether to force recalculation of matches
            
        Returns:
            Identity ID if found, None otherwise
        """
        if device_id not in self.device_signatures:
            logger.warning(f"Device {device_id} not found in signatures")
            return None
        
        # Check if already assigned to an identity
        if device_id in self.device_to_identity and not force_recalculate:
            return self.device_to_identity[device_id]
        
        # Find potential matches
        potential_matches = self._find_potential_matches(device_id)
        
        if not potential_matches:
            # Create new identity cluster
            identity_id = self._create_new_identity(device_id)
            return identity_id
        
        # Find best match above threshold
        best_match = max(potential_matches, key=lambda m: m.confidence_score)
        
        if best_match.confidence_score >= self.min_confidence_threshold:
            # iOS 14.5+ PRIVACY LIMITATION: Only 35% of matches succeed
            if random.random() > self.MAX_MATCH_RATE:
                logger.info(f"Cross-device match failed for {device_id} (iOS 14.5+ privacy limitation)")
                # Create new identity instead of matching
                identity_id = self._create_new_identity(device_id)
                return identity_id
            
            # Match succeeded (within the 35% success rate)
            existing_identity_id = self.device_to_identity.get(best_match.device_id_2)
            if existing_identity_id:
                logger.info(f"Cross-device match succeeded for {device_id} -> {existing_identity_id} (within 35% success rate)")
                self._add_device_to_identity(device_id, existing_identity_id, best_match.confidence_score)
                return existing_identity_id
        
        # Create new identity if no good matches
        identity_id = self._create_new_identity(device_id)
        return identity_id
    
    def calculate_match_probability(self, device_id_1: str, device_id_2: str) -> IdentityMatch:
        """
        Calculate the probability that two devices belong to the same user.
        
        Args:
            device_id_1: First device ID
            device_id_2: Second device ID
            
        Returns:
            IdentityMatch object with confidence score and evidence
        """
        # Check cache first
        cache_key = tuple(sorted([device_id_1, device_id_2]))
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]
        
        if device_id_1 not in self.device_signatures or device_id_2 not in self.device_signatures:
            return IdentityMatch(device_id_1, device_id_2, 0.0, MatchConfidence.VERY_LOW, [], {})
        
        sig1 = self.device_signatures[device_id_1]
        sig2 = self.device_signatures[device_id_2]
        
        # Calculate individual signal scores
        signal_scores = {}
        matching_signals = []
        evidence = {}
        
        # Behavioral similarity
        behavioral_score = self._calculate_behavioral_similarity(sig1, sig2)
        signal_scores['behavioral_similarity'] = behavioral_score
        evidence['behavioral_score'] = behavioral_score
        if behavioral_score > 0.6:
            matching_signals.append('behavioral_similarity')
        
        # Temporal proximity
        temporal_score = self._calculate_temporal_proximity(sig1, sig2)
        signal_scores['temporal_proximity'] = temporal_score
        evidence['temporal_score'] = temporal_score
        if temporal_score > 0.5:
            matching_signals.append('temporal_proximity')
        
        # Geographic proximity
        geographic_score = self._calculate_geographic_proximity(sig1, sig2)
        signal_scores['geographic_proximity'] = geographic_score
        evidence['geographic_score'] = geographic_score
        if geographic_score > 0.7:
            matching_signals.append('geographic_proximity')
        
        # Search pattern similarity
        search_score = self._calculate_search_pattern_similarity(sig1, sig2)
        signal_scores['search_pattern_similarity'] = search_score
        evidence['search_score'] = search_score
        if search_score > 0.5:
            matching_signals.append('search_pattern_similarity')
        
        # Session pattern similarity
        session_score = self._calculate_session_pattern_similarity(sig1, sig2)
        signal_scores['session_pattern_similarity'] = session_score
        evidence['session_score'] = session_score
        if session_score > 0.4:
            matching_signals.append('session_pattern_similarity')
        
        # Technical similarity
        tech_score = self._calculate_technical_similarity(sig1, sig2)
        signal_scores['technical_similarity'] = tech_score
        evidence['technical_score'] = tech_score
        if tech_score > 0.3:
            matching_signals.append('technical_similarity')
        
        # Usage time similarity
        usage_time_score = self._calculate_usage_time_similarity(sig1, sig2)
        signal_scores['usage_time_similarity'] = usage_time_score
        evidence['usage_time_score'] = usage_time_score
        if usage_time_score > 0.6:
            matching_signals.append('usage_time_similarity')
        
        # Calculate weighted confidence score
        confidence_score = sum(
            self.signal_weights[signal] * score 
            for signal, score in signal_scores.items()
        )
        
        # Apply bonus for multiple strong signals
        if len(matching_signals) >= 3:
            confidence_score *= 1.1
        elif len(matching_signals) >= 2:
            confidence_score *= 1.05
        
        # Determine confidence level
        if confidence_score >= self.high_confidence_threshold:
            confidence_level = MatchConfidence.HIGH
        elif confidence_score >= self.medium_confidence_threshold:
            confidence_level = MatchConfidence.MEDIUM
        elif confidence_score >= self.min_confidence_threshold:
            confidence_level = MatchConfidence.LOW
        else:
            confidence_level = MatchConfidence.VERY_LOW
        
        match = IdentityMatch(
            device_id_1=device_id_1,
            device_id_2=device_id_2,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            matching_signals=matching_signals,
            evidence=evidence
        )
        
        # Cache the result
        self.match_cache[cache_key] = match
        
        return match
    
    def merge_journeys(self, identity_id: str) -> List[Dict]:
        """
        Merge user journeys across all devices for a given identity.
        
        Args:
            identity_id: Identity ID to merge journeys for
            
        Returns:
            Merged journey as chronologically ordered list of events
        """
        if identity_id not in self.identity_graph:
            return []
        
        cluster = self.identity_graph[identity_id]
        all_events = []
        
        # Collect all events from all devices
        for device_id in cluster.device_ids:
            if device_id in self.device_signatures:
                sig = self.device_signatures[device_id]
                
                # Add session events
                for i, timestamp in enumerate(sig.session_timestamps):
                    event = {
                        'device_id': device_id,
                        'timestamp': timestamp,
                        'event_type': 'session',
                        'duration': sig.session_durations[i] if i < len(sig.session_durations) else 0,
                        'pages': sig.page_sequences[i] if i < len(sig.page_sequences) else []
                    }
                    all_events.append(event)
                
                # Add search events
                for search in sig.search_patterns:
                    # Estimate timestamp based on session patterns
                    if sig.session_timestamps:
                        base_time = sig.session_timestamps[-1]
                    else:
                        base_time = datetime.now()
                    
                    event = {
                        'device_id': device_id,
                        'timestamp': base_time,
                        'event_type': 'search',
                        'query': search
                    }
                    all_events.append(event)
        
        # Sort events chronologically
        merged_journey = sorted(all_events, key=lambda x: x['timestamp'])
        
        # Update the cluster's merged journey
        cluster.merged_journey = merged_journey
        cluster.updated_at = datetime.now()
        
        return merged_journey
    
    def update_identity_graph(self, new_matches: List[IdentityMatch]) -> None:
        """
        Update the identity graph with new matches.
        
        Args:
            new_matches: List of new identity matches to process
        """
        for match in new_matches:
            if match.confidence_score < self.min_confidence_threshold:
                continue
            
            device1 = match.device_id_1
            device2 = match.device_id_2
            
            identity1 = self.device_to_identity.get(device1)
            identity2 = self.device_to_identity.get(device2)
            
            if identity1 and identity2:
                if identity1 != identity2:
                    # Merge two existing identities
                    self._merge_identities(identity1, identity2)
            elif identity1:
                # Add device2 to identity1
                self._add_device_to_identity(device2, identity1, match.confidence_score)
            elif identity2:
                # Add device1 to identity2
                self._add_device_to_identity(device1, identity2, match.confidence_score)
            else:
                # Create new identity with both devices
                identity_id = self._create_new_identity_with_devices([device1, device2])
                self.identity_graph[identity_id].confidence_scores[device1] = match.confidence_score
                self.identity_graph[identity_id].confidence_scores[device2] = match.confidence_score
        
        logger.info(f"Updated identity graph with {len(new_matches)} new matches")
    
    def get_identity_cluster(self, identity_id: str) -> Optional[IdentityCluster]:
        """Get identity cluster by ID"""
        return self.identity_graph.get(identity_id)
    
    def get_device_identity(self, device_id: str) -> Optional[str]:
        """Get identity ID for a device"""
        return self.device_to_identity.get(device_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_devices = len(self.device_signatures)
        total_identities = len(self.identity_graph)
        
        cluster_sizes = [len(cluster.device_ids) for cluster in self.identity_graph.values()]
        avg_cluster_size = statistics.mean(cluster_sizes) if cluster_sizes else 0
        
        high_confidence_matches = sum(
            1 for match in self.match_cache.values() 
            if match.confidence_level == MatchConfidence.HIGH
        )
        
        return {
            'total_devices': total_devices,
            'total_identities': total_identities,
            'average_cluster_size': avg_cluster_size,
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'high_confidence_matches': high_confidence_matches,
            'cache_size': len(self.match_cache)
        }
    
    # Private helper methods
    
    def _find_potential_matches(self, device_id: str) -> List[IdentityMatch]:
        """Find potential identity matches for a device"""
        potential_matches = []
        
        for other_device_id in self.device_signatures:
            if other_device_id == device_id:
                continue
            
            match = self.calculate_match_probability(device_id, other_device_id)
            if match.confidence_score >= self.min_confidence_threshold:
                potential_matches.append(match)
        
        return potential_matches
    
    def _create_new_identity(self, device_id: str) -> str:
        """Create a new identity cluster for a device"""
        identity_id = f"identity_{len(self.identity_graph)}_{int(datetime.now().timestamp())}"
        
        cluster = IdentityCluster(
            identity_id=identity_id,
            device_ids={device_id},
            primary_device_id=device_id,
            confidence_scores={device_id: 1.0}
        )
        
        self.identity_graph[identity_id] = cluster
        self.device_to_identity[device_id] = identity_id
        
        return identity_id
    
    def _create_new_identity_with_devices(self, device_ids: List[str]) -> str:
        """Create a new identity cluster with multiple devices"""
        identity_id = f"identity_{len(self.identity_graph)}_{int(datetime.now().timestamp())}"
        
        cluster = IdentityCluster(
            identity_id=identity_id,
            device_ids=set(device_ids),
            primary_device_id=device_ids[0]
        )
        
        self.identity_graph[identity_id] = cluster
        
        for device_id in device_ids:
            self.device_to_identity[device_id] = identity_id
        
        return identity_id
    
    def _add_device_to_identity(self, device_id: str, identity_id: str, confidence_score: float) -> None:
        """Add a device to an existing identity cluster"""
        if identity_id in self.identity_graph:
            cluster = self.identity_graph[identity_id]
            cluster.device_ids.add(device_id)
            cluster.confidence_scores[device_id] = confidence_score
            cluster.updated_at = datetime.now()
            
            self.device_to_identity[device_id] = identity_id
    
    def _merge_identities(self, identity_id_1: str, identity_id_2: str) -> None:
        """Merge two identity clusters"""
        if identity_id_1 not in self.identity_graph or identity_id_2 not in self.identity_graph:
            return
        
        cluster1 = self.identity_graph[identity_id_1]
        cluster2 = self.identity_graph[identity_id_2]
        
        # Merge into cluster1
        cluster1.device_ids.update(cluster2.device_ids)
        cluster1.confidence_scores.update(cluster2.confidence_scores)
        cluster1.updated_at = datetime.now()
        
        # Update device mappings
        for device_id in cluster2.device_ids:
            self.device_to_identity[device_id] = identity_id_1
        
        # Remove cluster2
        del self.identity_graph[identity_id_2]
    
    def _generate_behavioral_hash(self, signature: DeviceSignature) -> str:
        """Generate a behavioral hash for a device signature"""
        # Combine key behavioral features
        features = {
            'search_patterns': sorted(signature.search_patterns),
            'avg_session_duration': round(signature.avg_session_duration, 2),
            'primary_usage_hours': sorted(list(signature.primary_usage_hours)),
            'platform': signature.platform,
            'timezone': signature.timezone
        }
        
        feature_string = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_string.encode()).hexdigest()
    
    def _calculate_behavioral_similarity(self, sig1: DeviceSignature, sig2: DeviceSignature) -> float:
        """Calculate behavioral similarity between two signatures"""
        cache_key = tuple(sorted([sig1.device_id, sig2.device_id]))
        if cache_key in self.behavioral_similarity_cache:
            return self.behavioral_similarity_cache[cache_key]
        
        similarity = 0.0
        
        # Hash similarity (quick check)
        if sig1.behavioral_hash == sig2.behavioral_hash:
            similarity += 0.3
        
        # Session duration similarity
        if sig1.avg_session_duration > 0 and sig2.avg_session_duration > 0:
            duration_ratio = min(sig1.avg_session_duration, sig2.avg_session_duration) / max(sig1.avg_session_duration, sig2.avg_session_duration)
            similarity += 0.2 * duration_ratio
        
        # Usage time overlap
        if sig1.primary_usage_hours and sig2.primary_usage_hours:
            overlap = len(sig1.primary_usage_hours.intersection(sig2.primary_usage_hours))
            total = len(sig1.primary_usage_hours.union(sig2.primary_usage_hours))
            if total > 0:
                similarity += 0.3 * (overlap / total)
        
        # Page sequence similarity
        if sig1.page_sequences and sig2.page_sequences:
            sequence_similarity = self._calculate_sequence_similarity(sig1.page_sequences, sig2.page_sequences)
            similarity += 0.2 * sequence_similarity
        
        self.behavioral_similarity_cache[cache_key] = similarity
        return min(similarity, 1.0)
    
    def _calculate_temporal_proximity(self, sig1: DeviceSignature, sig2: DeviceSignature) -> float:
        """Calculate temporal proximity between device sessions"""
        if not sig1.session_timestamps or not sig2.session_timestamps:
            return 0.0
        
        # Find closest session times
        min_time_diff = float('inf')
        
        for ts1 in sig1.session_timestamps:
            for ts2 in sig2.session_timestamps:
                time_diff = abs((ts1 - ts2).total_seconds())
                min_time_diff = min(min_time_diff, time_diff)
        
        # Convert to hours and calculate proximity score
        min_hours_diff = min_time_diff / 3600
        
        if min_hours_diff <= 1:
            return 0.9
        elif min_hours_diff <= self.time_window_hours:
            return 0.5 * (1 - min_hours_diff / self.time_window_hours)
        else:
            return 0.0
    
    def _calculate_geographic_proximity(self, sig1: DeviceSignature, sig2: DeviceSignature) -> float:
        """Calculate geographic proximity between devices"""
        if not sig1.geographic_locations or not sig2.geographic_locations:
            return 0.0
        
        min_distance = float('inf')
        
        for loc1 in sig1.geographic_locations:
            for loc2 in sig2.geographic_locations:
                distance = self._haversine_distance(loc1, loc2)
                min_distance = min(min_distance, distance)
        
        if min_distance <= 1.0:  # Same location (within 1km)
            return 0.9
        elif min_distance <= self.max_geographic_distance_km:
            return 0.5 * (1 - min_distance / self.max_geographic_distance_km)
        else:
            return 0.0
    
    def _calculate_search_pattern_similarity(self, sig1: DeviceSignature, sig2: DeviceSignature) -> float:
        """Calculate similarity in search patterns"""
        if not sig1.search_patterns or not sig2.search_patterns:
            return 0.0
        
        # Convert to sets for intersection/union operations
        searches1 = set(sig1.search_patterns)
        searches2 = set(sig2.search_patterns)
        
        if not searches1 or not searches2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(searches1.intersection(searches2))
        union = len(searches1.union(searches2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_session_pattern_similarity(self, sig1: DeviceSignature, sig2: DeviceSignature) -> float:
        """Calculate similarity in session patterns"""
        similarity = 0.0
        
        # Session duration similarity
        if sig1.session_durations and sig2.session_durations:
            avg1 = statistics.mean(sig1.session_durations)
            avg2 = statistics.mean(sig2.session_durations)
            if avg1 > 0 and avg2 > 0:
                duration_ratio = min(avg1, avg2) / max(avg1, avg2)
                similarity += 0.5 * duration_ratio
        
        # Session frequency similarity (sessions per day)
        if sig1.session_timestamps and sig2.session_timestamps:
            days1 = len(set(ts.date() for ts in sig1.session_timestamps))
            days2 = len(set(ts.date() for ts in sig2.session_timestamps))
            
            if days1 > 0 and days2 > 0:
                freq1 = len(sig1.session_timestamps) / days1
                freq2 = len(sig2.session_timestamps) / days2
                freq_ratio = min(freq1, freq2) / max(freq1, freq2)
                similarity += 0.5 * freq_ratio
        
        return min(similarity, 1.0)
    
    def _calculate_technical_similarity(self, sig1: DeviceSignature, sig2: DeviceSignature) -> float:
        """Calculate similarity in technical characteristics"""
        similarity = 0.0
        
        # Platform similarity
        if sig1.platform == sig2.platform:
            similarity += 0.3
        
        # Timezone similarity
        if sig1.timezone == sig2.timezone:
            similarity += 0.3
        
        # Language similarity
        if sig1.language == sig2.language:
            similarity += 0.2
        
        # Browser family similarity
        if sig1.browser and sig2.browser:
            if sig1.browser.split('/')[0] == sig2.browser.split('/')[0]:  # Same browser family
                similarity += 0.2
        
        return min(similarity, 1.0)
    
    def _calculate_usage_time_similarity(self, sig1: DeviceSignature, sig2: DeviceSignature) -> float:
        """Calculate similarity in usage time patterns"""
        if not sig1.time_of_day_usage or not sig2.time_of_day_usage:
            return 0.0
        
        # Create hour usage frequency distributions
        hours1 = defaultdict(int)
        hours2 = defaultdict(int)
        
        for hour in sig1.time_of_day_usage:
            hours1[hour] += 1
            
        for hour in sig2.time_of_day_usage:
            hours2[hour] += 1
        
        # Calculate cosine similarity
        all_hours = set(hours1.keys()).union(set(hours2.keys()))
        
        if not all_hours:
            return 0.0
        
        vec1 = [hours1.get(hour, 0) for hour in sorted(all_hours)]
        vec2 = [hours2.get(hour, 0) for hour in sorted(all_hours)]
        
        return self._cosine_similarity(vec1, vec2)
    
    def _calculate_sequence_similarity(self, sequences1: List[List[str]], sequences2: List[List[str]]) -> float:
        """Calculate similarity between page sequences"""
        if not sequences1 or not sequences2:
            return 0.0
        
        # Find common subsequences
        max_similarity = 0.0
        
        for seq1 in sequences1:
            for seq2 in sequences2:
                similarity = self._longest_common_subsequence_ratio(seq1, seq2)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _haversine_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two geographic points in kilometers"""
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return c * r
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _longest_common_subsequence_ratio(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate LCS ratio between two sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Dynamic programming LCS
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        max_length = max(m, n)
        
        return lcs_length / max_length if max_length > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    # Initialize the resolver
    resolver = IdentityResolver()
    
    # Create sample device signatures
    device1 = DeviceSignature(
        device_id="mobile_lisa_001",
        platform="iOS",
        timezone="America/New_York",
        language="en-US",
        search_patterns=["python tutorial", "machine learning course", "data science jobs"],
        session_durations=[45.0, 32.5, 67.8],
        time_of_day_usage=[9, 10, 14, 15, 20, 21],
        geographic_locations=[(40.7128, -74.0060)],  # NYC
        session_timestamps=[
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(minutes=30)
        ]
    )
    
    device2 = DeviceSignature(
        device_id="desktop_lisa_002",
        platform="Windows",
        timezone="America/New_York",
        language="en-US",
        search_patterns=["python tutorial", "machine learning", "data science career"],
        session_durations=[120.5, 89.3, 156.7],
        time_of_day_usage=[9, 10, 11, 14, 15, 16, 20],
        geographic_locations=[(40.7589, -73.9851)],  # NYC (different location)
        session_timestamps=[
            datetime.now() - timedelta(hours=3),
            datetime.now() - timedelta(hours=2.5),
            datetime.now() - timedelta(hours=1)
        ]
    )
    
    device3 = DeviceSignature(
        device_id="mobile_john_003",
        platform="Android",
        timezone="America/Los_Angeles",
        language="en-US",
        search_patterns=["sports news", "basketball scores", "fitness routine"],
        session_durations=[25.0, 18.5, 31.2],
        time_of_day_usage=[7, 8, 17, 18, 22, 23],
        geographic_locations=[(34.0522, -118.2437)],  # LA
        session_timestamps=[
            datetime.now() - timedelta(hours=4),
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1)
        ]
    )
    
    # Add signatures to resolver
    resolver.add_device_signature(device1)
    resolver.add_device_signature(device2)
    resolver.add_device_signature(device3)
    
    # Test identity resolution
    print("=== Identity Resolution Test ===")
    
    # Resolve identities
    identity1 = resolver.resolve_identity("mobile_lisa_001")
    identity2 = resolver.resolve_identity("desktop_lisa_002")
    identity3 = resolver.resolve_identity("mobile_john_003")
    
    print(f"Mobile Lisa identity: {identity1}")
    print(f"Desktop Lisa identity: {identity2}")
    print(f"Mobile John identity: {identity3}")
    
    # Test match probability calculation
    print("\n=== Match Probability Test ===")
    
    match_lisa = resolver.calculate_match_probability("mobile_lisa_001", "desktop_lisa_002")
    match_cross = resolver.calculate_match_probability("mobile_lisa_001", "mobile_john_003")
    
    print(f"Lisa mobile vs desktop match:")
    print(f"  Confidence: {match_lisa.confidence_score:.3f} ({match_lisa.confidence_level.value})")
    print(f"  Signals: {match_lisa.matching_signals}")
    
    print(f"\nLisa vs John match:")
    print(f"  Confidence: {match_cross.confidence_score:.3f} ({match_cross.confidence_level.value})")
    print(f"  Signals: {match_cross.matching_signals}")
    
    # Test journey merging
    print("\n=== Journey Merging Test ===")
    
    if identity1:
        merged_journey = resolver.merge_journeys(identity1)
        print(f"Merged journey for identity {identity1}:")
        for event in merged_journey[:3]:  # Show first 3 events
            print(f"  {event['timestamp']}: {event['event_type']} on {event['device_id']}")
    
    # Show system statistics
    print("\n=== System Statistics ===")
    stats = resolver.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")