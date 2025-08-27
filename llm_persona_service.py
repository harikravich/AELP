"""
LLM Persona Service for GAELP
Replaces mock personas with real LLM API integration for authentic user simulation
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import random
from enum import Enum

import httpx
import redis
from google.cloud import monitoring_v3


class PersonaState(Enum):
    """Persona behavioral states"""
    FRESH = "fresh"  # Never seen ads before
    ENGAGED = "engaged"  # Actively responding to ads
    FATIGUED = "fatigued"  # Experiencing ad fatigue
    BLOCKED = "blocked"  # Actively avoiding ads
    RECOVERED = "recovered"  # Recovered from fatigue


@dataclass
class PersonaDemographics:
    """Detailed persona demographics"""
    age: int
    gender: str
    income_level: str  # low, medium, high, very_high
    education: str  # high_school, college, graduate, phd
    location: str  # city/region
    employment: str  # student, employed, unemployed, retired
    relationship_status: str  # single, married, divorced, etc.
    household_size: int = 1


@dataclass
class PersonaPsychology:
    """Psychological profile for persona"""
    personality_traits: Dict[str, float] = field(default_factory=dict)  # Big 5 traits
    values: List[str] = field(default_factory=list)  # Core values
    interests: List[str] = field(default_factory=list)  # Hobbies, interests
    shopping_behavior: str = "moderate"  # conservative, moderate, impulsive
    tech_savviness: float = 0.5  # 0-1 scale
    brand_loyalty: float = 0.5  # 0-1 scale
    price_sensitivity: float = 0.5  # 0-1 scale
    social_influence: float = 0.5  # 0-1 scale


@dataclass
class PersonaHistory:
    """Persona interaction history"""
    ads_seen: List[Dict[str, Any]] = field(default_factory=list)
    clicks: List[Dict[str, Any]] = field(default_factory=list)
    purchases: List[Dict[str, Any]] = field(default_factory=list)
    fatigue_level: float = 0.0  # 0-1 scale
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0
    state: PersonaState = PersonaState.FRESH


@dataclass
class PersonaConfig:
    """Complete persona configuration"""
    persona_id: str
    name: str
    demographics: PersonaDemographics
    psychology: PersonaPsychology
    history: PersonaHistory = field(default_factory=PersonaHistory)
    
    def to_prompt_context(self) -> str:
        """Convert persona to LLM prompt context"""
        context = f"""
You are {self.name}, a {self.demographics.age}-year-old {self.demographics.gender} from {self.demographics.location}.

DEMOGRAPHICS:
- Income: {self.demographics.income_level}
- Education: {self.demographics.education}
- Employment: {self.demographics.employment}
- Relationship: {self.demographics.relationship_status}
- Household size: {self.demographics.household_size}

PERSONALITY & PSYCHOLOGY:
- Shopping behavior: {self.psychology.shopping_behavior}
- Tech savviness: {self.psychology.tech_savviness}/1.0
- Brand loyalty: {self.psychology.brand_loyalty}/1.0
- Price sensitivity: {self.psychology.price_sensitivity}/1.0
- Core interests: {', '.join(self.psychology.interests)}
- Values: {', '.join(self.psychology.values)}

CURRENT STATE:
- Ad fatigue level: {self.history.fatigue_level}/1.0
- State: {self.history.state.value}
- Recent ads seen: {len(self.history.ads_seen)} in last 30 days
- Current mood: {self._get_current_mood()}

You respond to advertisements authentically based on your profile. Consider your demographic background, 
psychological traits, current state, and interaction history when deciding how to engage with ads.
"""
        return context.strip()
    
    def _get_current_mood(self) -> str:
        """Determine current mood based on state and history"""
        if self.history.state == PersonaState.FRESH:
            return "curious and open to new products"
        elif self.history.state == PersonaState.ENGAGED:
            return "interested in relevant offers"
        elif self.history.state == PersonaState.FATIGUED:
            return "tired of seeing too many ads"
        elif self.history.state == PersonaState.BLOCKED:
            return "actively avoiding advertisements"
        else:
            return "neutral and selective about ads"


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, persona_context: str) -> Dict[str, Any]:
        """Generate persona response to ad campaign"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if LLM provider is healthy"""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com"
        self.client = httpx.AsyncClient(
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            timeout=30.0
        )
        
    async def generate_response(self, prompt: str, persona_context: str) -> Dict[str, Any]:
        """Generate response using Claude API"""
        
        system_prompt = f"""
{persona_context}

Your task is to respond to an advertisement as this persona would. 
Provide your response in the following JSON format:

{{
    "engagement_score": <0-1 float indicating interest level>,
    "will_click": <boolean>,
    "will_convert": <boolean>,
    "reasoning": "<brief explanation of your decision>",
    "emotional_response": "<positive/neutral/negative>",
    "fatigue_impact": <-0.1 to 0.1 change in fatigue level>,
    "persona_thoughts": "<internal thoughts about the ad>"
}}

Be authentic to your persona. Consider your demographics, psychology, current state, and ad history.
"""
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/messages",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["content"][0]["text"]
            
            # Parse JSON response
            try:
                parsed_response = json.loads(content)
                return parsed_response
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "engagement_score": 0.1,
                    "will_click": False,
                    "will_convert": False,
                    "reasoning": "Failed to parse LLM response",
                    "emotional_response": "neutral",
                    "fatigue_impact": 0.05,
                    "persona_thoughts": content[:200]
                }
                
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Anthropic API health"""
        try:
            response = await self.client.get(f"{self.base_url}/v1/messages")
            return response.status_code in [200, 401]  # 401 is expected without proper auth
        except:
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI GPT API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com"
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
    
    async def generate_response(self, prompt: str, persona_context: str) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        
        system_prompt = f"""
{persona_context}

Respond to advertisements as this persona would. Return only a JSON object with:
- engagement_score (0-1): interest level
- will_click (boolean): likely to click
- will_convert (boolean): likely to purchase
- reasoning (string): brief explanation
- emotional_response (string): positive/neutral/negative
- fatigue_impact (float): -0.1 to 0.1 change in fatigue
- persona_thoughts (string): internal thoughts about the ad
"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return json.loads(content)
            
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except:
            return False


@dataclass
class LLMPersonaConfig:
    """Configuration for LLM persona service"""
    
    # Provider settings
    primary_provider: str = "anthropic"  # anthropic, openai
    fallback_provider: Optional[str] = "openai"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Cost controls
    max_daily_cost: float = 100.0  # USD
    cost_per_1k_tokens: float = 0.01
    
    # Caching
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Persona behavior
    fatigue_recovery_hours: int = 24
    max_ads_per_day: int = 50
    state_transition_probability: float = 0.1
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"


class LLMPersonaService:
    """
    Main service for managing LLM-powered personas
    """
    
    def __init__(self, config: LLMPersonaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Initialize providers
        self.providers = {}
        self._init_providers()
        
        # Initialize caching
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        
        # Rate limiting
        self.rate_limiter = RateLimiter(config)
        
        # Cost tracking
        self.cost_tracker = CostTracker(config)
        
        # Persona management
        self.personas: Dict[str, PersonaConfig] = {}
        
        # Monitoring
        if config.enable_monitoring:
            self.monitoring_client = monitoring_v3.MetricServiceClient()
    
    def _init_providers(self):
        """Initialize LLM providers"""
        if self.config.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(self.config.anthropic_api_key)
        
        if self.config.openai_api_key:
            self.providers["openai"] = OpenAIProvider(self.config.openai_api_key)
        
        if not self.providers:
            raise ValueError("No LLM providers configured")
    
    async def create_persona(self, persona_config: PersonaConfig) -> str:
        """Create and register a new persona"""
        self.personas[persona_config.persona_id] = persona_config
        
        # Cache persona configuration
        await self._cache_persona(persona_config)
        
        self.logger.info(f"Created persona: {persona_config.name} ({persona_config.persona_id})")
        return persona_config.persona_id
    
    async def respond_to_ad(self, persona_id: str, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate authentic persona response to ad campaign
        
        Args:
            persona_id: Unique persona identifier
            campaign: Campaign details (creative, targeting, etc.)
            
        Returns:
            Dict containing engagement metrics and persona response
        """
        
        # Get persona configuration
        persona = self.personas.get(persona_id)
        if not persona:
            raise ValueError(f"Persona {persona_id} not found")
        
        # Check rate limits
        await self.rate_limiter.check_limits(persona_id)
        
        # Check daily cost limits
        await self.cost_tracker.check_limits()
        
        # Check cache for similar recent responses
        cache_key = self._generate_cache_key(persona_id, campaign)
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            self.logger.debug(f"Using cached response for persona {persona_id}")
            return cached_response
        
        # Update persona state before responding
        await self._update_persona_state(persona, campaign)
        
        # Generate ad prompt
        ad_prompt = self._create_ad_prompt(campaign)
        
        # Get LLM response
        llm_response = await self._get_llm_response(persona, ad_prompt)
        
        # Convert to campaign metrics
        campaign_response = await self._convert_to_campaign_metrics(
            persona, campaign, llm_response
        )
        
        # Update persona history
        await self._update_persona_history(persona, campaign, campaign_response)
        
        # Cache response
        await self._cache_response(cache_key, campaign_response)
        
        # Track costs
        await self.cost_tracker.track_request(llm_response.get("tokens_used", 1000))
        
        # Log metrics
        await self._log_interaction_metrics(persona_id, campaign, campaign_response)
        
        return campaign_response
    
    async def _get_llm_response(self, persona: PersonaConfig, ad_prompt: str) -> Dict[str, Any]:
        """Get response from LLM provider with fallback"""
        
        primary_provider = self.providers.get(self.config.primary_provider)
        if not primary_provider:
            raise ValueError(f"Primary provider {self.config.primary_provider} not available")
        
        persona_context = persona.to_prompt_context()
        
        try:
            # Try primary provider
            response = await primary_provider.generate_response(ad_prompt, persona_context)
            response["provider_used"] = self.config.primary_provider
            return response
            
        except Exception as e:
            self.logger.warning(f"Primary provider failed: {e}")
            
            # Try fallback provider
            if self.config.fallback_provider and self.config.fallback_provider in self.providers:
                try:
                    fallback_provider = self.providers[self.config.fallback_provider]
                    response = await fallback_provider.generate_response(ad_prompt, persona_context)
                    response["provider_used"] = self.config.fallback_provider
                    return response
                except Exception as fallback_error:
                    self.logger.error(f"Fallback provider also failed: {fallback_error}")
            
            # If all providers fail, return default response
            return self._get_default_response(persona)
    
    def _create_ad_prompt(self, campaign: Dict[str, Any]) -> str:
        """Create structured prompt for LLM based on campaign"""
        
        prompt = f"""
You are being shown an advertisement with the following details:

CAMPAIGN DETAILS:
- Creative Type: {campaign.get('creative_type', 'image')}
- Target Audience: {campaign.get('target_audience', 'general')}
- Budget: ${campaign.get('budget', 0)}/day
- Bid Strategy: {campaign.get('bid_strategy', 'cpc')}
- Message: {campaign.get('message', 'Check out our product!')}
- Call to Action: {campaign.get('cta', 'Learn More')}
- Product Category: {campaign.get('category', 'general')}
- Brand: {campaign.get('brand', 'Unknown Brand')}
- Price Point: {campaign.get('price_point', 'medium')}

CONTEXT:
- Platform: {campaign.get('platform', 'social_media')}
- Time of Day: {campaign.get('time_of_day', 'afternoon')}
- Day of Week: {campaign.get('day_of_week', 'tuesday')}

Based on your persona profile, how do you respond to this advertisement?
Consider your current state, recent ad exposure, and authentic human behavior.
"""
        return prompt.strip()
    
    async def _convert_to_campaign_metrics(
        self, 
        persona: PersonaConfig, 
        campaign: Dict[str, Any], 
        llm_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert LLM response to campaign performance metrics"""
        
        # Base metrics from LLM response
        engagement_score = max(0.0, min(1.0, llm_response.get("engagement_score", 0.1)))
        will_click = llm_response.get("will_click", False)
        will_convert = llm_response.get("will_convert", False)
        
        # Apply persona-specific adjustments
        engagement_score = self._apply_persona_adjustments(persona, campaign, engagement_score)
        
        # Calculate realistic metrics
        impressions = 1  # This persona saw the ad once
        clicks = 1 if will_click else 0
        conversions = 1 if (will_click and will_convert) else 0
        
        # Calculate costs (simplified)
        cpc = campaign.get("budget", 10) * random.uniform(0.5, 2.0)  # Cost per click
        cost = clicks * cpc if clicks > 0 else 0
        
        # Calculate revenue
        conversion_value = campaign.get("conversion_value", random.uniform(20, 100))
        revenue = conversions * conversion_value
        
        return {
            "persona_id": persona.persona_id,
            "persona_name": persona.name,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "ctr": clicks / impressions,
            "conversion_rate": conversions / max(1, clicks),
            "cost": cost,
            "revenue": revenue,
            "roas": revenue / max(0.01, cost),  # Return on ad spend
            "engagement_score": engagement_score,
            "emotional_response": llm_response.get("emotional_response", "neutral"),
            "reasoning": llm_response.get("reasoning", ""),
            "persona_thoughts": llm_response.get("persona_thoughts", ""),
            "fatigue_impact": llm_response.get("fatigue_impact", 0.0),
            "provider_used": llm_response.get("provider_used", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
    
    def _apply_persona_adjustments(
        self, 
        persona: PersonaConfig, 
        campaign: Dict[str, Any], 
        base_engagement: float
    ) -> float:
        """Apply persona-specific adjustments to engagement"""
        
        adjusted_engagement = base_engagement
        
        # Fatigue adjustment
        fatigue_penalty = persona.history.fatigue_level * 0.5
        adjusted_engagement -= fatigue_penalty
        
        # State adjustment
        if persona.history.state == PersonaState.BLOCKED:
            adjusted_engagement *= 0.1
        elif persona.history.state == PersonaState.FATIGUED:
            adjusted_engagement *= 0.3
        elif persona.history.state == PersonaState.ENGAGED:
            adjusted_engagement *= 1.2
        
        # Demographic matching
        target_audience = campaign.get("target_audience", "")
        if self._matches_target_audience(persona, target_audience):
            adjusted_engagement *= 1.3
        
        # Interest matching
        campaign_category = campaign.get("category", "")
        if campaign_category.lower() in [interest.lower() for interest in persona.psychology.interests]:
            adjusted_engagement *= 1.4
        
        return max(0.0, min(1.0, adjusted_engagement))
    
    def _matches_target_audience(self, persona: PersonaConfig, target_audience: str) -> bool:
        """Check if persona matches target audience"""
        target_lower = target_audience.lower()
        
        if "young" in target_lower and persona.demographics.age < 30:
            return True
        if "professional" in target_lower and persona.demographics.employment == "employed":
            return True
        if "families" in target_lower and persona.demographics.household_size > 1:
            return True
        if "student" in target_lower and persona.demographics.employment == "student":
            return True
        
        return False
    
    async def _update_persona_state(self, persona: PersonaConfig, campaign: Dict[str, Any]):
        """Update persona state based on interaction history"""
        
        current_time = datetime.now()
        
        # Recovery from fatigue
        if (persona.history.last_interaction and 
            current_time - persona.history.last_interaction > timedelta(hours=self.config.fatigue_recovery_hours)):
            persona.history.fatigue_level = max(0.0, persona.history.fatigue_level - 0.2)
            if persona.history.state == PersonaState.FATIGUED:
                persona.history.state = PersonaState.RECOVERED
        
        # Check for state transitions
        ads_today = len([ad for ad in persona.history.ads_seen 
                        if datetime.fromisoformat(ad["timestamp"]).date() == current_time.date()])
        
        if ads_today > self.config.max_ads_per_day:
            persona.history.state = PersonaState.FATIGUED
            persona.history.fatigue_level = min(1.0, persona.history.fatigue_level + 0.3)
        
        # Random state transitions
        if random.random() < self.config.state_transition_probability:
            if persona.history.state == PersonaState.FRESH:
                persona.history.state = PersonaState.ENGAGED
            elif persona.history.state == PersonaState.FATIGUED and random.random() < 0.3:
                persona.history.state = PersonaState.BLOCKED
    
    async def _update_persona_history(
        self, 
        persona: PersonaConfig, 
        campaign: Dict[str, Any], 
        response: Dict[str, Any]
    ):
        """Update persona interaction history"""
        
        interaction = {
            "campaign_id": campaign.get("campaign_id", "unknown"),
            "campaign_type": campaign.get("creative_type", "unknown"),
            "clicked": response["clicks"] > 0,
            "converted": response["conversions"] > 0,
            "engagement_score": response["engagement_score"],
            "timestamp": datetime.now().isoformat()
        }
        
        persona.history.ads_seen.append(interaction)
        persona.history.interaction_count += 1
        persona.history.last_interaction = datetime.now()
        
        if response["clicks"] > 0:
            persona.history.clicks.append(interaction)
        
        if response["conversions"] > 0:
            persona.history.purchases.append(interaction)
        
        # Update fatigue
        persona.history.fatigue_level += response.get("fatigue_impact", 0.01)
        persona.history.fatigue_level = max(0.0, min(1.0, persona.history.fatigue_level))
        
        # Keep history manageable (last 100 interactions)
        if len(persona.history.ads_seen) > 100:
            persona.history.ads_seen = persona.history.ads_seen[-100:]
        
        # Update cached persona
        await self._cache_persona(persona)
    
    def _get_default_response(self, persona: PersonaConfig) -> Dict[str, Any]:
        """Generate default response when LLM fails"""
        
        # Simple heuristic-based response
        base_engagement = 0.1
        
        if persona.history.state == PersonaState.ENGAGED:
            base_engagement = 0.4
        elif persona.history.state == PersonaState.FATIGUED:
            base_engagement = 0.05
        elif persona.history.state == PersonaState.BLOCKED:
            base_engagement = 0.01
        
        will_click = base_engagement > 0.3 and random.random() < base_engagement
        will_convert = will_click and random.random() < 0.1
        
        return {
            "engagement_score": base_engagement,
            "will_click": will_click,
            "will_convert": will_convert,
            "reasoning": "Fallback response due to LLM failure",
            "emotional_response": "neutral",
            "fatigue_impact": 0.02,
            "persona_thoughts": "Default response generated",
            "provider_used": "fallback"
        }
    
    def _generate_cache_key(self, persona_id: str, campaign: Dict[str, Any]) -> str:
        """Generate cache key for response caching"""
        campaign_hash = hash(json.dumps(campaign, sort_keys=True))
        return f"response:{persona_id}:{campaign_hash}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            self.logger.warning(f"Cache retrieval error: {e}")
        return None
    
    async def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response for future use"""
        try:
            self.redis_client.setex(
                cache_key, 
                self.config.cache_ttl_seconds, 
                json.dumps(response)
            )
        except Exception as e:
            self.logger.warning(f"Cache storage error: {e}")
    
    async def _cache_persona(self, persona: PersonaConfig):
        """Cache persona configuration"""
        try:
            persona_data = {
                "config": persona.__dict__,
                "last_updated": datetime.now().isoformat()
            }
            self.redis_client.setex(
                f"persona:{persona.persona_id}",
                86400,  # 24 hours
                json.dumps(persona_data, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Persona cache error: {e}")
    
    async def _log_interaction_metrics(
        self, 
        persona_id: str, 
        campaign: Dict[str, Any], 
        response: Dict[str, Any]
    ):
        """Log interaction metrics for monitoring"""
        
        self.logger.info(
            f"Persona {persona_id} interaction: "
            f"engagement={response['engagement_score']:.3f}, "
            f"clicked={response['clicks']}, "
            f"converted={response['conversions']}, "
            f"provider={response['provider_used']}"
        )
        
        # Additional monitoring metrics can be added here
    
    async def get_persona_analytics(self, persona_id: str) -> Dict[str, Any]:
        """Get analytics for a specific persona"""
        
        persona = self.personas.get(persona_id)
        if not persona:
            raise ValueError(f"Persona {persona_id} not found")
        
        total_ads = len(persona.history.ads_seen)
        total_clicks = len(persona.history.clicks)
        total_conversions = len(persona.history.purchases)
        
        return {
            "persona_id": persona_id,
            "persona_name": persona.name,
            "total_interactions": total_ads,
            "total_clicks": total_clicks,
            "total_conversions": total_conversions,
            "ctr": total_clicks / max(1, total_ads),
            "conversion_rate": total_conversions / max(1, total_clicks),
            "current_state": persona.history.state.value,
            "fatigue_level": persona.history.fatigue_level,
            "last_interaction": persona.history.last_interaction.isoformat() if persona.history.last_interaction else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        
        health_status = {
            "service_status": "healthy",
            "providers": {},
            "redis_status": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        # Check LLM providers
        for name, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                health_status["providers"][name] = "healthy" if is_healthy else "unhealthy"
            except Exception as e:
                health_status["providers"][name] = f"error: {str(e)}"
        
        # Check Redis
        try:
            self.redis_client.ping()
            health_status["redis_status"] = "healthy"
        except Exception as e:
            health_status["redis_status"] = f"error: {str(e)}"
        
        # Overall status
        if any(status != "healthy" for status in health_status["providers"].values()):
            health_status["service_status"] = "degraded"
        
        if health_status["redis_status"] != "healthy":
            health_status["service_status"] = "degraded"
        
        return health_status


class RateLimiter:
    """Rate limiting for LLM API calls"""
    
    def __init__(self, config: LLMPersonaConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
    
    async def check_limits(self, persona_id: str):
        """Check if request is within rate limits"""
        
        current_time = int(time.time())
        
        # Check per-minute limit
        minute_key = f"rate_limit:minute:{current_time // 60}"
        minute_count = self.redis_client.incr(minute_key)
        self.redis_client.expire(minute_key, 60)
        
        if minute_count > self.config.requests_per_minute:
            raise Exception("Rate limit exceeded: requests per minute")
        
        # Check per-hour limit
        hour_key = f"rate_limit:hour:{current_time // 3600}"
        hour_count = self.redis_client.incr(hour_key)
        self.redis_client.expire(hour_key, 3600)
        
        if hour_count > self.config.requests_per_hour:
            raise Exception("Rate limit exceeded: requests per hour")
        
        # Check per-day limit
        day_key = f"rate_limit:day:{current_time // 86400}"
        day_count = self.redis_client.incr(day_key)
        self.redis_client.expire(day_key, 86400)
        
        if day_count > self.config.requests_per_day:
            raise Exception("Rate limit exceeded: requests per day")


class CostTracker:
    """Track and limit LLM API costs"""
    
    def __init__(self, config: LLMPersonaConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
    
    async def check_limits(self):
        """Check if within daily cost limits"""
        
        today_key = f"cost:daily:{datetime.now().strftime('%Y-%m-%d')}"
        daily_cost = float(self.redis_client.get(today_key) or 0)
        
        if daily_cost >= self.config.max_daily_cost:
            raise Exception(f"Daily cost limit exceeded: ${daily_cost:.2f}")
    
    async def track_request(self, tokens_used: int):
        """Track cost of a request"""
        
        cost = (tokens_used / 1000) * self.config.cost_per_1k_tokens
        
        today_key = f"cost:daily:{datetime.now().strftime('%Y-%m-%d')}"
        self.redis_client.incrbyfloat(today_key, cost)
        self.redis_client.expire(today_key, 86400)  # Expire after 24 hours
        
        # Track monthly costs
        month_key = f"cost:monthly:{datetime.now().strftime('%Y-%m')}"
        self.redis_client.incrbyfloat(month_key, cost)
        self.redis_client.expire(month_key, 86400 * 31)  # Expire after 31 days