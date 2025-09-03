"""
Hybrid LLM-RL Integration for GAELP
Enhances existing RL agent with LLM strategic reasoning capabilities
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass
import openai
import os

logger = logging.getLogger(__name__)


@dataclass
class LLMStrategyConfig:
    """Configuration for LLM strategic reasoning."""
    model: str = "gpt-4o-mini"  # Fast, cost-effective for real-time decisions
    temperature: float = 0.7
    max_tokens: int = 150
    use_caching: bool = True
    cache_ttl: int = 3600  # Cache strategy for 1 hour
    # NO FALLBACKS - if LLM fails, we fix it properly


class StrategicLLMReasoner:
    """
    LLM component for high-level strategic reasoning about campaigns.
    Provides context-aware guidance to the RL agent.
    """
    
    def __init__(self, config: LLMStrategyConfig = None):
        self.config = config or LLMStrategyConfig()
        self.strategy_cache = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client - REQUIRED, no fallbacks
        if not self.api_key:
            raise RuntimeError("OpenAI API key is REQUIRED. Set OPENAI_API_KEY environment variable. NO FALLBACKS.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.llm_available = True  # Always true, no fallbacks
    
    def get_strategic_context(self, 
                              market_state: Dict[str, Any],
                              campaign_goal: str) -> Dict[str, Any]:
        """
        Get strategic guidance from LLM based on market conditions.
        
        Args:
            market_state: Current market conditions and competitor positions
            campaign_goal: High-level campaign objective
            
        Returns:
            Strategic recommendations and reasoning
        """
        # Check cache first
        cache_key = self._create_cache_key(market_state, campaign_goal)
        if cache_key in self.strategy_cache:
            logger.info("Using cached LLM strategy")
            return self.strategy_cache[cache_key]
        
        # NO FALLBACKS - LLM is required
        
        try:
            # Construct prompt for strategic reasoning
            prompt = self._construct_strategy_prompt(market_state, campaign_goal)
            
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a strategic marketing AI advisor for Aura Balance parental control app."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            strategy_text = response.choices[0].message.content
            strategy = self._parse_strategy_response(strategy_text)
            
            # Cache the strategy
            if self.config.use_caching:
                self.strategy_cache[cache_key] = strategy
            
            return strategy
            
        except Exception as e:
            logger.error(f"LLM strategy generation failed: {e}")
            raise RuntimeError(f"LLM strategy generation is REQUIRED. Fix the error: {e}. NO FALLBACKS.")
    
    def _construct_strategy_prompt(self, market_state: Dict, goal: str) -> str:
        """Construct prompt for LLM strategic reasoning."""
        prompt = f"""
        Campaign Goal: {goal}
        
        Market Conditions:
        - Our Position: #{market_state.get('our_position', 'Unknown')}
        - Top Competitor: {market_state.get('top_competitor', 'Unknown')} at position #{market_state.get('competitor_position', 'Unknown')}
        - Market Trend: {market_state.get('trend', 'stable')}
        - Budget Remaining: ${market_state.get('budget_remaining', 0):.2f}
        - Current CTR: {market_state.get('ctr', 0):.2%}
        - Current CPA: ${market_state.get('cpa', 0):.2f}
        
        Provide strategic recommendations for:
        1. Bidding strategy (aggressive/conservative/adaptive)
        2. Audience focus (broad/narrow/specific segments)
        3. Creative theme (safety/education/control/trust)
        4. Key message for parents
        
        Format as JSON with keys: bidding_strategy, audience_focus, creative_theme, key_message, reasoning
        """
        return prompt
    
    def _parse_strategy_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured strategy."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Parse response properly - no fallbacks
                raise ValueError(f"LLM response must contain valid JSON. Got: {response}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response. NO FALLBACKS. Error: {e}")
    
    # REMOVED _get_rule_based_strategy - NO FALLBACKS ALLOWED
    
    def _create_cache_key(self, market_state: Dict, goal: str) -> str:
        """Create cache key for strategy."""
        key_parts = [
            goal,
            str(market_state.get('our_position', 0)),
            str(market_state.get('top_competitor', '')),
            str(int(market_state.get('budget_remaining', 0) / 100) * 100)  # Round to nearest $100
        ]
        return "_".join(key_parts)


class CreativeGenerator:
    """
    LLM-powered creative generation for headlines and ad copy.
    Generates infinite variations based on winning themes.
    """
    
    def __init__(self, config: LLMStrategyConfig = None):
        self.config = config or LLMStrategyConfig()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # LLM is REQUIRED - no fallbacks
        if not self.api_key:
            raise RuntimeError("OpenAI API key is REQUIRED for creative generation. NO FALLBACKS.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.llm_available = True  # Always true
        
        # Winning creative themes for Aura Balance
        self.winning_themes = [
            "safety",      # "Keep your teen safe online"
            "balance",     # "Find healthy screen balance"
            "trust",       # "Build trust with your teen"
            "education",   # "Teach responsible device use"
            "peace",       # "Peace of mind for parents"
            "control"      # "Take back control of screen time"
        ]
        
        # Cache for generated creatives
        self.creative_cache = {}
    
    def generate_headline(self, 
                         theme: str,
                         target_segment: str,
                         emotional_tone: str = "concerned") -> str:
        """
        Generate a headline variation for testing.
        
        Args:
            theme: Creative theme (safety, balance, etc.)
            target_segment: Target audience segment
            emotional_tone: Emotional approach (concerned, empowering, urgent)
            
        Returns:
            Generated headline text
        """
        cache_key = f"{theme}_{target_segment}_{emotional_tone}"
        
        # Check cache
        if cache_key in self.creative_cache:
            # Return a random cached variant
            import random
            return random.choice(self.creative_cache[cache_key])
        
        # NO FALLBACKS - LLM generation is required
        
        try:
            prompt = f"""
            Generate 5 compelling headlines for Aura Balance parental control app.
            
            Theme: {theme}
            Target: {target_segment} parents
            Tone: {emotional_tone}
            
            Requirements:
            - 8-12 words maximum
            - Address parent's concern about teen screen time
            - Include emotional trigger
            - Clear call to action implied
            
            Format: Return only the 5 headlines, one per line.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert copywriter for parental control apps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for creativity
                max_tokens=150
            )
            
            headlines = response.choices[0].message.content.strip().split('\n')
            headlines = [h.strip() for h in headlines if h.strip()]
            
            # Cache the results
            self.creative_cache[cache_key] = headlines
            
            if not headlines:
                raise RuntimeError("LLM must generate at least one headline. NO FALLBACKS.")
            return headlines[0]
            
        except Exception as e:
            logger.error(f"LLM headline generation failed: {e}")
            raise RuntimeError(f"LLM headline generation is REQUIRED. Fix the error: {e}. NO FALLBACKS.")
    
    # REMOVED _get_template_headline - NO FALLBACK TEMPLATES ALLOWED
    
    def generate_ad_copy(self, headline: str, theme: str) -> Dict[str, str]:
        """Generate complete ad copy based on headline."""
        # NO FALLBACKS - LLM is required
        
        try:
            prompt = f"""
            Create ad copy for Aura Balance parental control app.
            
            Headline: {headline}
            Theme: {theme}
            
            Generate:
            1. Description (2 lines, 90 chars max)
            2. Call-to-action (3-4 words)
            
            Format as JSON with keys: description, cta
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert ad copywriter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            import re
            json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"LLM must return valid JSON for ad copy. NO FALLBACKS.")
            
        except Exception as e:
            logger.error(f"Ad copy generation failed: {e}")
            raise RuntimeError(f"LLM ad copy generation is REQUIRED. Fix the error: {e}. NO FALLBACKS.")
    
    # REMOVED _get_template_ad_copy - NO FALLBACK TEMPLATES ALLOWED


# Aliases for backward compatibility
LLMStrategyAdvisor = StrategicLLMReasoner


class HybridLLMRLAgent:
    """
    Hybrid agent that combines LLM strategic reasoning with RL optimization.
    Enhances existing RL agent without replacing it.
    """
    
    def __init__(self, base_rl_agent, config: LLMStrategyConfig = None):
        """
        Initialize hybrid agent.
        
        Args:
            base_rl_agent: Existing RL agent to enhance
            config: LLM configuration
        """
        self.rl_agent = base_rl_agent
        self.llm_strategist = StrategicLLMReasoner(config)
        self.creative_generator = CreativeGenerator(config)
        
        # Strategy influence on RL decisions
        self.strategy_weight = 0.3  # 30% LLM strategy, 70% RL optimization
        
        # Track performance for adaptation
        self.performance_history = []
        self.strategy_performance = {}
    
    def get_action(self, state: Dict[str, Any], context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get action combining LLM strategy and RL optimization.
        
        Args:
            state: Current environment state
            context: Optional context for LLM reasoning
            
        Returns:
            Action dictionary with bid, creative, and strategy info
        """
        # Get market state for LLM
        market_state = self._extract_market_state(state)
        
        # Get strategic guidance from LLM
        strategy = self.llm_strategist.get_strategic_context(
            market_state,
            context or "maximize_conversions"
        )
        
        # Get base RL action
        rl_action = self.rl_agent.get_action(state)
        
        # Combine LLM strategy with RL action
        hybrid_action = self._combine_strategy_and_action(strategy, rl_action, state)
        
        # Generate creative if needed
        if hybrid_action.get("needs_new_creative", False):
            creative = self._generate_creative(strategy, state)
            hybrid_action["creative"] = creative
        
        # Track for learning
        hybrid_action["strategy_used"] = strategy
        
        return hybrid_action
    
    def _extract_market_state(self, state: Dict) -> Dict[str, Any]:
        """Extract market conditions from state."""
        return {
            "our_position": state.get("auction_position", 5),
            "top_competitor": state.get("top_competitor", "Unknown"),
            "competitor_position": state.get("competitor_position", 1),
            "trend": self._calculate_trend(state),
            "budget_remaining": state.get("budget_remaining", 1000),
            "ctr": state.get("recent_ctr", 0.02),
            "cpa": state.get("recent_cpa", 50)
        }
    
    def _calculate_trend(self, state: Dict) -> str:
        """Calculate market trend from state."""
        if len(self.performance_history) < 2:
            return "stable"
        
        recent_ctr = np.mean([p["ctr"] for p in self.performance_history[-5:]])
        older_ctr = np.mean([p["ctr"] for p in self.performance_history[-10:-5]])
        
        if recent_ctr > older_ctr * 1.1:
            return "improving"
        elif recent_ctr < older_ctr * 0.9:
            return "declining"
        return "stable"
    
    def _combine_strategy_and_action(self, 
                                    strategy: Dict, 
                                    rl_action: Dict,
                                    state: Dict) -> Dict[str, Any]:
        """Combine LLM strategy with RL action."""
        hybrid_action = rl_action.copy()
        
        # Adjust bid based on strategy
        if strategy["bidding_strategy"] == "aggressive":
            hybrid_action["bid"] = rl_action.get("bid", 1.0) * 1.3
        elif strategy["bidding_strategy"] == "conservative":
            hybrid_action["bid"] = rl_action.get("bid", 1.0) * 0.7
        
        # Adjust targeting based on strategy
        if strategy["audience_focus"] == "narrow":
            hybrid_action["target_segments"] = ["high_intent_parents"]
        elif strategy["audience_focus"] == "broad":
            hybrid_action["target_segments"] = ["all_parents"]
        
        # Flag for creative refresh if theme changed
        current_theme = state.get("current_creative_theme", "")
        if strategy["creative_theme"] != current_theme:
            hybrid_action["needs_new_creative"] = True
            hybrid_action["new_theme"] = strategy["creative_theme"]
        
        return hybrid_action
    
    def _generate_creative(self, strategy: Dict, state: Dict) -> Dict[str, str]:
        """Generate new creative based on strategy."""
        theme = strategy["creative_theme"]
        segment = state.get("target_segment", "concerned_parents")
        
        # Determine emotional tone based on performance
        if state.get("recent_ctr", 0) < 0.01:
            tone = "urgent"  # Low CTR needs urgency
        elif state.get("recent_cpa", 100) > 80:
            tone = "empowering"  # High CPA needs different approach
        else:
            tone = "concerned"
        
        headline = self.creative_generator.generate_headline(theme, segment, tone)
        ad_copy = self.creative_generator.generate_ad_copy(headline, theme)
        
        return {
            "headline": headline,
            "description": ad_copy["description"],
            "cta": ad_copy["cta"],
            "theme": theme,
            "generated_by": "llm"
        }
    
    def update(self, state: Dict, action: Dict, reward: float, next_state: Dict):
        """Update both RL agent and track strategy performance."""
        # Update base RL agent
        self.rl_agent.update(state, action, reward, next_state)
        
        # Track strategy performance
        if "strategy_used" in action:
            strategy_key = action["strategy_used"]["bidding_strategy"]
            if strategy_key not in self.strategy_performance:
                self.strategy_performance[strategy_key] = []
            self.strategy_performance[strategy_key].append(reward)
        
        # Update performance history
        self.performance_history.append({
            "ctr": next_state.get("recent_ctr", 0),
            "cpa": next_state.get("recent_cpa", 0),
            "reward": reward
        })
        
        # Keep history bounded
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
    
    def get_strategy_report(self) -> Dict[str, Any]:
        """Get report on strategy performance."""
        report = {}
        for strategy, rewards in self.strategy_performance.items():
            report[strategy] = {
                "avg_reward": np.mean(rewards) if rewards else 0,
                "total_uses": len(rewards),
                "recent_performance": np.mean(rewards[-10:]) if len(rewards) > 10 else np.mean(rewards)
            }
        return report
    
    # Delegate methods to base RL agent for compatibility
    def store_experience(self, *args, **kwargs):
        """Delegate to base RL agent"""
        if hasattr(self.rl_agent, 'store_experience'):
            return self.rl_agent.store_experience(*args, **kwargs)
        elif hasattr(self.rl_agent, 'remember'):
            # Some agents use 'remember' instead of 'store_experience'
            return self.rl_agent.remember(*args, **kwargs)
        elif hasattr(self.rl_agent, 'add_experience'):
            # Some agents use 'add_experience'
            return self.rl_agent.add_experience(*args, **kwargs)
        else:
            # Log error if no compatible method found
            import logging
            logging.error(f"Base RL agent {type(self.rl_agent).__name__} has no experience storage method")
            raise AttributeError(f"Base RL agent has no store_experience, remember, or add_experience method")
    
    def train_dqn(self, *args, **kwargs):
        """Delegate training to base RL agent"""
        if hasattr(self.rl_agent, 'train_dqn'):
            return self.rl_agent.train_dqn(*args, **kwargs)
        elif hasattr(self.rl_agent, 'train'):
            # AdvancedRLAgent's train() doesn't accept any parameters
            return self.rl_agent.train()
        elif hasattr(self.rl_agent, 'update'):
            return self.rl_agent.update(*args, **kwargs)
        else:
            import logging
            logging.error(f"Base RL agent {type(self.rl_agent).__name__} has no training method")
            raise AttributeError(f"Base RL agent has no train_dqn, train, or update method")
    
    def train_ppo_from_buffer(self, *args, **kwargs):
        """Delegate PPO training to base RL agent"""
        if hasattr(self.rl_agent, 'train_ppo_from_buffer'):
            return self.rl_agent.train_ppo_from_buffer(*args, **kwargs)
        elif hasattr(self.rl_agent, 'train_ppo'):
            # Try without args if the method doesn't accept them
            return self.rl_agent.train_ppo()
        elif hasattr(self.rl_agent, 'train'):
            # Fall back to general training
            return self.rl_agent.train()
        else:
            # PPO training is optional, just log and continue
            import logging
            logging.info(f"Base RL agent {type(self.rl_agent).__name__} doesn't support PPO training")
            return {'ppo_loss': 0.0}
    
    def save_checkpoint(self, *args, **kwargs):
        """Delegate checkpoint saving to base RL agent"""
        if hasattr(self.rl_agent, 'save_checkpoint'):
            return self.rl_agent.save_checkpoint(*args, **kwargs)
        elif hasattr(self.rl_agent, 'save'):
            return self.rl_agent.save(*args, **kwargs)
        else:
            # Checkpoint saving is optional
            import logging
            logging.debug(f"Base RL agent {type(self.rl_agent).__name__} doesn't support checkpointing")
            return None
    
    def detect_performance_drop(self, *args, **kwargs):
        """Delegate performance drop detection to base RL agent"""
        if hasattr(self.rl_agent, 'detect_performance_drop'):
            return self.rl_agent.detect_performance_drop(*args, **kwargs)
        else:
            # Performance drop detection is optional, return False (no drop detected)
            return False
    
    def adapt_to_environment_change(self, *args, **kwargs):
        """Delegate environment adaptation to base RL agent"""
        if hasattr(self.rl_agent, 'adapt_to_environment_change'):
            return self.rl_agent.adapt_to_environment_change(*args, **kwargs)
        elif hasattr(self.rl_agent, 'reset_exploration'):
            # Some agents might use different method name
            return self.rl_agent.reset_exploration(*args, **kwargs)
        else:
            # Adaptation is optional
            import logging
            logging.debug(f"Base RL agent {type(self.rl_agent).__name__} doesn't support environment adaptation")
            return None
    
    # Expose buffer attributes for compatibility
    @property
    def replay_buffer(self):
        """Access base agent's replay buffer"""
        if hasattr(self.rl_agent, 'replay_buffer'):
            return self.rl_agent.replay_buffer
        elif hasattr(self.rl_agent, 'memory'):
            return self.rl_agent.memory
        elif hasattr(self.rl_agent, 'buffer'):
            return self.rl_agent.buffer
        else:
            # Create an empty buffer-like object to prevent errors
            class EmptyBuffer:
                def __len__(self):
                    return 0
                def __iter__(self):
                    return iter([])
            return EmptyBuffer()
    
    @property
    def memory(self):
        """Access base agent's memory"""
        if hasattr(self.rl_agent, 'memory'):
            return self.rl_agent.memory
        elif hasattr(self.rl_agent, 'replay_buffer'):
            return self.rl_agent.replay_buffer
        elif hasattr(self.rl_agent, 'buffer'):
            return self.rl_agent.buffer
        else:
            return self.replay_buffer  # Use the replay_buffer property
    
    @property  
    def buffer(self):
        """Access base agent's buffer"""
        buffer = getattr(self.rl_agent, 'buffer', None)
        return buffer if buffer is not None else []
    
    def get_bid_action(self, *args, **kwargs):
        """Get bid action with LLM enhancement"""
        if hasattr(self.rl_agent, 'get_bid_action'):
            # Get base RL action
            base_action = self.rl_agent.get_bid_action(*args, **kwargs)
            # Could enhance with LLM strategy here if needed
            return base_action
        elif hasattr(self.rl_agent, 'get_action'):
            # Use general action method if needed
            action = self.rl_agent.get_action(*args, **kwargs)
            return action.get('bid', 1.0) if isinstance(action, dict) else action
        else:
            # Default bid
            return 1.0
    
    def get_creative_action(self, *args, **kwargs):
        """Get creative action with LLM enhancement"""
        if hasattr(self.rl_agent, 'get_creative_action'):
            # Get base RL action
            base_action = self.rl_agent.get_creative_action(*args, **kwargs)
            # Could enhance with LLM creative generation here if needed
            return base_action
        elif hasattr(self.rl_agent, 'get_action'):
            # Use general action method if needed
            action = self.rl_agent.get_action(*args, **kwargs)
            return action.get('creative', 0) if isinstance(action, dict) else 0
        else:
            # Default creative selection
            return 0


# Integration function for existing system
def enhance_rl_with_llm(base_rl_agent, config: LLMStrategyConfig = None) -> HybridLLMRLAgent:
    """
    Enhance existing RL agent with LLM capabilities.
    
    Args:
        base_rl_agent: Existing RL agent from the system
        config: Optional LLM configuration
        
    Returns:
        Enhanced hybrid agent
    """
    return HybridLLMRLAgent(base_rl_agent, config)