
# COMPATIBILITY LAYER FOR LEGACY SEGMENT REFERENCES
# DO NOT ADD NEW HARDCODED SEGMENTS HERE

from dynamic_segment_integration import get_dynamic_segment_manager

def get_legacy_segment_mapping():
    """
    Legacy compatibility - maps old hardcoded names to dynamic segments
    WARNING: Do not add new hardcoded segments!
    """
    manager = get_dynamic_segment_manager()
    compat = manager.get_legacy_compatible_segments()
    
    # Map legacy names to discovered segments
    legacy_mapping = {}
    
    # Use behavioral characteristics instead of hardcoded names
    high_conv = manager.get_high_conversion_segments()
    mobile_segs = manager.get_mobile_segments() 
    
    if high_conv:
        legacy_mapping['urgent_need'] = high_conv[0]
    if mobile_segs:
        legacy_mapping['mobile_user'] = mobile_segs[0]
    
    return legacy_mapping

def get_segment_for_behavior(engagement='medium', device='mobile'):
    """Get segment based on behavioral characteristics"""
    manager = get_dynamic_segment_manager()
    return manager.get_segment_by_characteristics(
        engagement_level=engagement,
        device_preference=device
    )
