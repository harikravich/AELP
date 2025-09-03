#!/usr/bin/env python3
"""
DYNAMIC SEGMENT INTEGRATION MODULE
Replaces all hardcoded segments with dynamically discovered ones
Provides compatibility layer for existing RL agent and GAELP system

CRITICAL RULES:
- NO hardcoded segment names allowed
- All segments discovered from GA4 data
- Backward compatibility for existing code
- Real-time segment updates
- Validation of segment quality
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from segment_discovery import SegmentDiscoveryEngine, DiscoveredSegment

logger = logging.getLogger(__name__)

@dataclass
class SegmentMapping:
    """Maps discovered segments to behavioral characteristics"""
    segment_id: str
    discovered_name: str
    behavioral_type: str  # e.g., 'high_engagement', 'quick_browser'
    device_preference: str
    conversion_rate: float
    engagement_level: str
    activity_pattern: str
    confidence: float

class DynamicSegmentManager:
    """
    Manages dynamic segments and provides compatibility with existing system
    REPLACES all hardcoded segments like 'concerned_parent', 'crisis_parent', etc.
    """
    
    def __init__(self, discovery_engine: SegmentDiscoveryEngine = None):
        self.discovery_engine = discovery_engine or SegmentDiscoveryEngine()
        self.segment_mappings = {}
        self.legacy_compatibility = {}
        self.last_update = None
        self.update_interval = timedelta(minutes=30)
        
        # FORBIDDEN hardcoded segments - these should never appear
        self.forbidden_segments = [
            'health_conscious', 'budget_conscious', 'premium_focused',
            'concerned_parent', 'proactive_parent', 'crisis_parent',
            'tech_savvy', 'brand_focused', 'performance_driven',
            'researching_parent', 'concerned_parents', 'crisis_parents'
        ]
        
        logger.info("DynamicSegmentManager initialized - NO hardcoded segments allowed")
        self._initialize_segments()
    
    def _initialize_segments(self):
        """Initialize with discovered segments"""
        try:
            segments = self.discovery_engine.discover_segments()
            self._update_segment_mappings(segments)
            logger.info(f"Initialized with {len(segments)} discovered segments")
        except Exception as e:
            logger.error(f"Failed to initialize segments: {e}")
            raise RuntimeError(f"Segment discovery is REQUIRED. No fallback segments allowed. Fix discovery engine: {e}")
    
    def _create_fallback_segments(self):
        """REMOVED - No fallback segments allowed"""
        raise RuntimeError("Fallback segments are not allowed. All segments must be discovered from real data.")
    
    def _update_segment_mappings(self, discovered_segments: Dict[str, DiscoveredSegment]):
        """Update internal mappings from discovered segments"""
        self.segment_mappings = {}
        
        for segment_id, segment in discovered_segments.items():
            # Validate no hardcoded names
            self._validate_segment_name(segment.name)
            
            mapping = SegmentMapping(
                segment_id=segment_id,
                discovered_name=segment.name,
                behavioral_type=self._extract_behavioral_type(segment),
                device_preference=segment.characteristics.get('primary_device', 'mobile'),
                conversion_rate=segment.conversion_rate,
                engagement_level=segment.characteristics.get('engagement_level', 'medium'),
                activity_pattern=segment.characteristics.get('activity_pattern', 'varied'),
                confidence=segment.confidence_score
            )
            
            self.segment_mappings[segment_id] = mapping
        
        self.last_update = datetime.now()
        logger.info(f"Updated {len(self.segment_mappings)} segment mappings")
    
    def _validate_segment_name(self, name: str):
        """Ensure no hardcoded segment names are used"""
        name_lower = name.lower()
        for forbidden in self.forbidden_segments:
            if forbidden.lower() in name_lower:
                raise RuntimeError(f"HARDCODED SEGMENT DETECTED: '{name}' contains forbidden term '{forbidden}'. All segments must be discovered dynamically!")
    
    def _extract_behavioral_type(self, segment: DiscoveredSegment) -> str:
        """Extract behavioral type from segment characteristics"""
        engagement = segment.characteristics.get('engagement_level', 'medium')
        session_style = segment.characteristics.get('session_style', 'moderate_browser')
        
        if engagement == 'high' and session_style == 'deep_explorer':
            return 'high_engagement_researcher'
        elif engagement == 'high':
            return 'high_engagement'
        elif engagement == 'low':
            return 'low_engagement'
        elif session_style == 'quick_visitor':
            return 'quick_browser'
        else:
            return 'medium_engagement'
    
    def should_update_segments(self) -> bool:
        """Check if segments should be updated"""
        if not self.last_update:
            return True
        return datetime.now() - self.last_update > self.update_interval
    
    def update_segments(self, force: bool = False) -> Dict[str, SegmentMapping]:
        """Update segments from discovery engine"""
        if not force and not self.should_update_segments():
            return self.segment_mappings
        
        try:
            logger.info("Updating segments from discovery engine...")
            segments = self.discovery_engine.discover_segments(force_rediscovery=force)
            self._update_segment_mappings(segments)
            return self.segment_mappings
        except Exception as e:
            logger.error(f"Failed to update segments: {e}")
            return self.segment_mappings
    
    def get_all_segments(self) -> Dict[str, SegmentMapping]:
        """Get all discovered segments"""
        self.update_segments()
        return self.segment_mappings
    
    def get_segment_names(self) -> List[str]:
        """Get list of discovered segment IDs - NO hardcoded names"""
        self.update_segments()
        return list(self.segment_mappings.keys())
    
    def get_segment_by_characteristics(self, 
                                     engagement_level: str = None,
                                     device_preference: str = None,
                                     activity_pattern: str = None) -> Optional[str]:
        """Find segment by characteristics - NO hardcoding"""
        self.update_segments()
        
        best_match = None
        best_confidence = 0
        
        for segment_id, mapping in self.segment_mappings.items():
            matches = True
            
            if engagement_level and mapping.engagement_level != engagement_level:
                matches = False
            if device_preference and mapping.device_preference != device_preference:
                matches = False
            if activity_pattern and mapping.activity_pattern != activity_pattern:
                matches = False
            
            if matches and mapping.confidence > best_confidence:
                best_match = segment_id
                best_confidence = mapping.confidence
        
        return best_match
    
    def get_high_conversion_segments(self, min_rate: float = 0.03) -> List[str]:
        """Get segments with high conversion rates"""
        self.update_segments()
        
        high_converting = []
        for segment_id, mapping in self.segment_mappings.items():
            if mapping.conversion_rate >= min_rate:
                high_converting.append(segment_id)
        
        return sorted(high_converting, key=lambda x: self.segment_mappings[x].conversion_rate, reverse=True)
    
    def get_mobile_segments(self) -> List[str]:
        """Get mobile-focused segments"""
        return [sid for sid, mapping in self.get_all_segments().items() 
                if mapping.device_preference == 'mobile']
    
    def get_desktop_segments(self) -> List[str]:
        """Get desktop-focused segments"""
        return [sid for sid, mapping in self.get_all_segments().items() 
                if mapping.device_preference == 'desktop']
    
    def get_segment_conversion_rate(self, segment_id: str) -> float:
        """Get conversion rate for segment"""
        self.update_segments()
        mapping = self.segment_mappings.get(segment_id)
        if not mapping:
            logger.error(f"Segment not found: {segment_id}. Available: {list(self.segment_mappings.keys())}")
            raise RuntimeError(f"Segment '{segment_id}' not found in discovered segments. No fallback values allowed.")
        return mapping.conversion_rate
    
    def get_segment_name(self, segment_id: str) -> str:
        """Get human-readable name for segment"""
        self.update_segments()
        mapping = self.segment_mappings.get(segment_id)
        return mapping.discovered_name if mapping else "Unknown Segment"
    
    def get_legacy_compatible_segments(self) -> Dict[str, str]:
        """
        Provide compatibility mapping for legacy code
        Maps behavioral characteristics to discovered segment IDs
        """
        self.update_segments()
        
        compatible_mapping = {}
        
        # Map by behavioral characteristics, NOT hardcoded names
        high_engagement_segments = [s for s in self.segment_mappings.keys() 
                                  if self.segment_mappings[s].engagement_level == 'high']
        medium_engagement_segments = [s for s in self.segment_mappings.keys() 
                                    if self.segment_mappings[s].engagement_level == 'medium']  
        low_engagement_segments = [s for s in self.segment_mappings.keys() 
                                 if self.segment_mappings[s].engagement_level == 'low']
        
        # Mobile segments
        mobile_segments = self.get_mobile_segments()
        
        # High conversion segments
        high_conversion_segments = self.get_high_conversion_segments()
        
        # Create behavioral mappings
        if high_engagement_segments:
            compatible_mapping['high_engagement'] = high_engagement_segments[0]
        if medium_engagement_segments:
            compatible_mapping['medium_engagement'] = medium_engagement_segments[0]
        if low_engagement_segments:
            compatible_mapping['low_engagement'] = low_engagement_segments[0]
        if mobile_segments:
            compatible_mapping['mobile_focused'] = mobile_segments[0]
        if high_conversion_segments:
            compatible_mapping['high_converting'] = high_conversion_segments[0]
        
        return compatible_mapping
    
    def export_for_rl_agent(self) -> Dict[str, Dict]:
        """Export segments in format expected by RL agent"""
        self.update_segments()
        
        rl_segments = {}
        for segment_id, mapping in self.segment_mappings.items():
            rl_segments[segment_id] = {
                'name': mapping.discovered_name,
                'conversion_rate': mapping.conversion_rate,
                'engagement_level': mapping.engagement_level,
                'device_preference': mapping.device_preference,
                'activity_pattern': mapping.activity_pattern,
                'behavioral_type': mapping.behavioral_type,
                'confidence': mapping.confidence
            }
        
        return rl_segments
    
    def get_segment_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered segments"""
        self.update_segments()
        
        if not self.segment_mappings:
            return {
                'total_segments': 0,
                'avg_conversion_rate': 0.0,
                'min_conversion_rate': 0.0,
                'max_conversion_rate': 0.0,
                'avg_confidence': 0.0,
                'device_distribution': {},
                'engagement_distribution': {},
                'last_updated': None
            }
        
        conversion_rates = [m.conversion_rate for m in self.segment_mappings.values()]
        confidences = [m.confidence for m in self.segment_mappings.values()]
        
        device_dist = defaultdict(int)
        engagement_dist = defaultdict(int)
        
        for mapping in self.segment_mappings.values():
            device_dist[mapping.device_preference] += 1
            engagement_dist[mapping.engagement_level] += 1
        
        return {
            'total_segments': len(self.segment_mappings),
            'avg_conversion_rate': sum(conversion_rates) / len(conversion_rates),
            'min_conversion_rate': min(conversion_rates),
            'max_conversion_rate': max(conversion_rates),
            'avg_confidence': sum(confidences) / len(confidences),
            'device_distribution': dict(device_dist),
            'engagement_distribution': dict(engagement_dist),
            'last_updated': self.last_update.isoformat() if self.last_update else None
        }
    
    def save_segments(self, filename: str = None):
        """Save discovered segments to file"""
        if filename is None:
            filename = f"dynamic_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'segments': {},
            'statistics': self.get_segment_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        for segment_id, mapping in self.segment_mappings.items():
            data['segments'][segment_id] = {
                'id': mapping.segment_id,
                'name': mapping.discovered_name,
                'behavioral_type': mapping.behavioral_type,
                'device_preference': mapping.device_preference,
                'conversion_rate': mapping.conversion_rate,
                'engagement_level': mapping.engagement_level,
                'activity_pattern': mapping.activity_pattern,
                'confidence': mapping.confidence
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.segment_mappings)} dynamic segments to {filename}")


# Global segment manager instance
_segment_manager = None

def get_dynamic_segment_manager() -> DynamicSegmentManager:
    """Get global dynamic segment manager instance"""
    global _segment_manager
    if _segment_manager is None:
        _segment_manager = DynamicSegmentManager()
    return _segment_manager

def get_discovered_segments() -> List[str]:
    """Get list of discovered segment names - NO hardcoding"""
    manager = get_dynamic_segment_manager()
    return manager.get_segment_names()

def get_segment_conversion_rate(segment_id: str) -> float:
    """Get conversion rate for discovered segment"""
    manager = get_dynamic_segment_manager()
    return manager.get_segment_conversion_rate(segment_id)

def get_high_converting_segment() -> Optional[str]:
    """Get highest converting discovered segment"""
    manager = get_dynamic_segment_manager()
    high_conv_segments = manager.get_high_conversion_segments()
    return high_conv_segments[0] if high_conv_segments else None

def get_mobile_segment() -> Optional[str]:
    """Get mobile-focused discovered segment"""
    manager = get_dynamic_segment_manager()
    mobile_segments = manager.get_mobile_segments()
    return mobile_segments[0] if mobile_segments else None

def validate_no_hardcoded_segments(code_or_config: Any):
    """Validate that no hardcoded segments are being used"""
    manager = get_dynamic_segment_manager()
    
    if isinstance(code_or_config, str):
        text = code_or_config.lower()
    elif isinstance(code_or_config, dict):
        text = str(code_or_config).lower()
    else:
        text = str(code_or_config).lower()
    
    for forbidden in manager.forbidden_segments:
        if forbidden.lower() in text:
            raise RuntimeError(f"HARDCODED SEGMENT DETECTED: '{forbidden}' found in code/config. Use dynamic segments only!")


if __name__ == "__main__":
    print("🔬 Dynamic Segment Integration Manager")
    print("="*50)
    
    manager = DynamicSegmentManager()
    
    # Show discovered segments
    segments = manager.get_all_segments()
    print(f"\n✅ Discovered {len(segments)} dynamic segments:")
    
    for segment_id, mapping in segments.items():
        print(f"  {segment_id}: {mapping.discovered_name}")
        print(f"    CVR: {mapping.conversion_rate:.3f}, Device: {mapping.device_preference}")
        print(f"    Engagement: {mapping.engagement_level}, Confidence: {mapping.confidence:.2f}")
    
    # Show statistics
    stats = manager.get_segment_statistics()
    print(f"\n📊 Segment Statistics:")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Avg conversion rate: {stats['avg_conversion_rate']:.3f}")
    print(f"  Device distribution: {stats['device_distribution']}")
    print(f"  Engagement distribution: {stats['engagement_distribution']}")
    
    # Show compatibility mapping
    compat = manager.get_legacy_compatible_segments()
    print(f"\n🔗 Legacy Compatibility Mapping:")
    for behavior_type, segment_id in compat.items():
        print(f"  {behavior_type} -> {segment_id}")
    
    print("\n✅ All segments discovered dynamically - NO hardcoding!")