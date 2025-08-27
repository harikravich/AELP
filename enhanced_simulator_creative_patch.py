"""
Creative Integration Patch for Enhanced Simulator
This module adds creative integration capabilities to the existing enhanced_simulator.py
"""

from creative_integration import get_creative_integration, SimulationContext
import logging

logger = logging.getLogger(__name__)

def enhance_ad_creative_with_selector(ad_creative: dict, context: dict, user_id: str = None) -> dict:
    """
    Enhance basic ad_creative dictionary with rich content from CreativeSelector
    
    Args:
        ad_creative: Basic ad creative (might be empty {})
        context: Simulation context
        user_id: Optional user ID
    
    Returns:
        Enhanced ad_creative with rich content
    """
    
    # If ad_creative is empty or very basic, use CreativeSelector
    if not ad_creative or len(ad_creative) <= 3:
        try:
            creative_integration = get_creative_integration()
            
            # Create simulation context from available data
            sim_context = SimulationContext(
                user_id=user_id or f"enhanced_sim_user_{hash(str(context)) % 10000}",
                persona=_determine_persona_from_context(context),
                channel=context.get('channel', 'search'),
                device_type=context.get('device', 'desktop'),
                time_of_day=_get_time_period(context.get('hour', 14)),
                session_count=context.get('session_count', 1),
                price_sensitivity=context.get('price_sensitivity', 0.5),
                urgency_score=context.get('urgency_score', 0.5),
                technical_level=context.get('technical_level', 0.5),
                conversion_probability=context.get('conversion_probability', 0.05)
            )
            
            # Get rich ad content
            enhanced_creative = creative_integration.get_targeted_ad_content(sim_context)
            
            # Merge with original ad_creative (original takes precedence)
            for key, value in ad_creative.items():
                if value is not None:
                    enhanced_creative[key] = value
            
            logger.debug(f"Enhanced ad creative from {len(ad_creative)} fields to {len(enhanced_creative)} fields")
            
            return enhanced_creative
            
        except Exception as e:
            logger.warning(f"Failed to enhance ad creative with selector: {e}")
            return ad_creative
    else:
        return ad_creative


def _determine_persona_from_context(context: dict) -> str:
    """Determine user persona from simulation context"""
    
    # Look for segment information in context
    segment = context.get('segment', '').lower()
    user_type = context.get('user_type', '').lower()
    
    # Map common segments to personas
    if 'crisis' in segment or 'urgent' in user_type:
        return 'crisis_parent'
    elif 'research' in segment or 'tech' in user_type:
        return 'researcher'
    elif 'price' in segment or 'budget' in user_type:
        return 'price_conscious'
    elif 'parent' in segment or 'concerned' in user_type:
        return 'concerned_parent'
    else:
        # Default based on context hints
        hour = context.get('hour', 14)
        if 20 <= hour <= 23:  # Evening hours
            return 'concerned_parent'  # Parents active in evening
        else:
            return 'researcher'  # Default to researcher


def _get_time_period(hour: int) -> str:
    """Convert hour to time period"""
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'


# Example of how to patch the existing simulate_response method:
def patched_simulate_response(self, ad_creative: dict, context: dict) -> dict:
    """
    Patched version of simulate_response that uses CreativeIntegration
    
    This should replace the simulate_response method in UserBehaviorModel
    """
    
    # Generate unique user ID for this interaction
    user_id = f"user_{self.interaction_count}"
    self.interaction_count += 1
    
    # Enhance ad_creative with CreativeIntegration
    enhanced_ad_creative = enhance_ad_creative_with_selector(
        ad_creative=ad_creative,
        context=context,
        user_id=user_id
    )
    
    # Continue with original simulation logic
    if hasattr(self, 'use_recsim_bridge') and self.use_recsim_bridge:
        return self._simulate_with_recsim_bridge(enhanced_ad_creative, context)
    else:
        return self._simulate_fallback(enhanced_ad_creative, context)


# Monkey patch helper
def apply_creative_integration_patch():
    """
    Apply the creative integration patch to existing UserBehaviorModel instances
    """
    try:
        import enhanced_simulator
        
        # Store original method
        if not hasattr(enhanced_simulator.UserBehaviorModel, '_original_simulate_response'):
            enhanced_simulator.UserBehaviorModel._original_simulate_response = \
                enhanced_simulator.UserBehaviorModel.simulate_response
        
        # Patch the method
        enhanced_simulator.UserBehaviorModel.simulate_response = patched_simulate_response
        
        logger.info("Successfully applied Creative Integration patch to UserBehaviorModel")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply Creative Integration patch: {e}")
        return False