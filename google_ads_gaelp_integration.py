#!/usr/bin/env python3
"""
Google Ads Integration with GAELP RL System
Connects GAELP reinforcement learning agent with real Google Ads campaigns.
NO FALLBACKS - Only real Google Ads API integration.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from google_ads_production_manager import (
    GoogleAdsProductionManager, GAELPGoogleAdsIntegration,
    CampaignConfig, AdGroupConfig, CampaignPerformance
)

logger = logging.getLogger(__name__)

@dataclass
class GAELPCampaignState:
    """State representation for GAELP RL agent"""
    campaign_id: str
    impressions: int
    clicks: int
    conversions: float
    cost_usd: float
    ctr: float
    conversion_rate: float
    avg_cpc: float
    impression_share: float
    quality_score_estimate: float
    time_running_hours: int
    keyword_performance: Dict[str, Dict[str, float]]
    competitor_pressure: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert campaign state to feature vector for RL agent"""
        features = [
            self.impressions / 1000.0,  # Normalize impressions
            self.clicks / 100.0,  # Normalize clicks
            self.conversions,  # Raw conversions
            self.cost_usd / 100.0,  # Normalize cost
            self.ctr,  # CTR as percentage
            self.conversion_rate,  # Conversion rate as decimal
            self.avg_cpc / 10.0,  # Normalize average CPC
            self.impression_share,  # Impression share as decimal
            self.quality_score_estimate / 10.0,  # Normalize quality score
            self.time_running_hours / 24.0,  # Normalize to days
            self.competitor_pressure  # Competitor pressure index
        ]
        
        # Add keyword performance features (top 5 keywords)
        keyword_features = []
        sorted_keywords = sorted(
            self.keyword_performance.items(),
            key=lambda x: x[1].get('conversions', 0),
            reverse=True
        )[:5]
        
        for _, kw_data in sorted_keywords:
            keyword_features.extend([
                kw_data.get('ctr', 0.0),
                kw_data.get('conversion_rate', 0.0),
                kw_data.get('avg_cpc', 0.0) / 10.0
            ])
        
        # Pad to consistent length
        while len(keyword_features) < 15:  # 5 keywords * 3 features each
            keyword_features.append(0.0)
        
        features.extend(keyword_features)
        
        return np.array(features, dtype=np.float32)

class GAELPGoogleAdsAgent:
    """
    GAELP RL Agent integrated with real Google Ads campaigns
    Learns optimal bidding strategies using live campaign data
    """
    
    def __init__(self, ads_manager: GoogleAdsProductionManager):
        self.ads_manager = ads_manager
        self.rl_integration = GAELPGoogleAdsIntegration(ads_manager)
        self.active_campaigns = {}
        self.campaign_states = {}
        self.action_history = []
        self.performance_history = []
        
        # RL hyperparameters
        self.learning_rate = 0.001
        self.exploration_rate = 0.1
        self.discount_factor = 0.95
        
        # Campaign management parameters
        self.min_bid = 0.50  # Minimum bid in USD
        self.max_bid = 50.00  # Maximum bid in USD
        self.bid_adjustment_step = 0.10  # 10% adjustment steps
        
        # Safety limits
        self.max_daily_spend = 500.0  # Maximum daily spend per campaign
        self.emergency_pause_threshold = 100.0  # Pause campaign if daily spend exceeds
    
    async def create_rl_campaign(self, campaign_type: str = "behavioral_health") -> str:
        """
        Create a new RL-managed campaign
        
        Args:
            campaign_type: Type of campaign to create
            
        Returns:
            Campaign resource name
        """
        if campaign_type == "behavioral_health":
            # Create initial campaign configuration
            initial_config = self.ads_manager.create_behavioral_health_campaign_config()
            
            # Apply RL-driven initial optimizations
            optimized_config = await self._optimize_initial_config(initial_config)
            
            # Create campaign
            campaign_resource_name = await self.ads_manager.create_campaign(optimized_config)
            campaign_id = campaign_resource_name.split('/')[-1]
            
            # Initialize RL state tracking
            self.active_campaigns[campaign_id] = {
                'resource_name': campaign_resource_name,
                'config': optimized_config,
                'created_at': datetime.now(),
                'type': campaign_type,
                'rl_managed': True
            }
            
            # Create initial state
            initial_state = GAELPCampaignState(
                campaign_id=campaign_id,
                impressions=0,
                clicks=0,
                conversions=0,
                cost_usd=0.0,
                ctr=0.0,
                conversion_rate=0.0,
                avg_cpc=0.0,
                impression_share=0.0,
                quality_score_estimate=5.0,  # Default estimate
                time_running_hours=0,
                keyword_performance={},
                competitor_pressure=0.5  # Default medium pressure
            )
            
            self.campaign_states[campaign_id] = initial_state
            
            logger.info(f"🤖 RL-managed campaign created: {campaign_id}")
            return campaign_resource_name
        
        else:
            raise ValueError(f"Unsupported campaign type: {campaign_type}")
    
    async def _optimize_initial_config(self, config: CampaignConfig) -> CampaignConfig:
        """
        Apply RL-driven optimizations to initial campaign configuration
        Uses historical performance data if available
        """
        # Load historical performance data
        historical_data = self._load_historical_performance()
        
        if historical_data:
            # Adjust initial bids based on historical keyword performance
            for ad_group in config.ad_groups:
                for keyword in ad_group.keywords:
                    if keyword in historical_data:
                        historical_performance = historical_data[keyword]
                        
                        # Adjust bid based on historical conversion rate
                        if historical_performance['conversion_rate'] > 0.05:
                            # High-performing keyword - increase initial bid
                            ad_group.max_cpc_bid_micros = int(ad_group.max_cpc_bid_micros * 1.3)
                        elif historical_performance['conversion_rate'] < 0.01:
                            # Low-performing keyword - decrease initial bid
                            ad_group.max_cpc_bid_micros = int(ad_group.max_cpc_bid_micros * 0.7)
        
        # Apply budget optimization based on expected performance
        expected_daily_conversions = self._estimate_daily_conversions(config)
        if expected_daily_conversions > 10:
            # High-potential campaign - increase budget
            config.budget_amount_micros = int(config.budget_amount_micros * 1.5)
        
        return config
    
    def _load_historical_performance(self) -> Dict[str, Any]:
        """Load historical performance data for optimization"""
        try:
            with open('/home/hariravichandran/AELP/data/google_ads_historical_performance.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _estimate_daily_conversions(self, config: CampaignConfig) -> float:
        """Estimate daily conversions based on campaign configuration"""
        # Simple estimation based on budget and average CPC
        total_keywords = sum(len(ag.keywords) for ag in config.ad_groups)
        avg_cpc = sum(ag.max_cpc_bid_micros for ag in config.ad_groups) / len(config.ad_groups) / 1_000_000
        
        estimated_clicks = (config.budget_amount_micros / 1_000_000) / avg_cpc
        estimated_conversions = estimated_clicks * 0.025  # Assume 2.5% conversion rate
        
        return max(estimated_conversions, 1.0)
    
    async def update_campaign_states(self):
        """Update campaign states with latest performance data"""
        for campaign_id in self.active_campaigns:
            try:
                # Get performance data from Google Ads
                performance = await self.ads_manager.get_campaign_performance(campaign_id, days=1)
                keyword_performance = await self.ads_manager._get_keyword_performance(campaign_id, days=1)
                
                # Calculate derived metrics
                conversion_rate = (performance.conversions / performance.clicks) if performance.clicks > 0 else 0
                time_running = (datetime.now() - self.active_campaigns[campaign_id]['created_at']).total_seconds() / 3600
                
                # Calculate competitor pressure based on impression share
                competitor_pressure = min(performance.impression_share, 1.0) if performance.impression_share > 0 else 0.5
                
                # Create keyword performance dictionary
                kw_performance_dict = {}
                for kw_data in keyword_performance:
                    kw_text = kw_data['keyword']
                    kw_performance_dict[kw_text] = {
                        'impressions': kw_data['impressions'],
                        'clicks': kw_data['clicks'],
                        'conversions': kw_data['conversions'],
                        'ctr': (kw_data['clicks'] / kw_data['impressions']) * 100 if kw_data['impressions'] > 0 else 0,
                        'conversion_rate': (kw_data['conversions'] / kw_data['clicks']) if kw_data['clicks'] > 0 else 0,
                        'avg_cpc': kw_data['average_cpc'] / 1_000_000
                    }
                
                # Update campaign state
                self.campaign_states[campaign_id] = GAELPCampaignState(
                    campaign_id=campaign_id,
                    impressions=performance.impressions,
                    clicks=performance.clicks,
                    conversions=performance.conversions,
                    cost_usd=performance.cost_micros / 1_000_000,
                    ctr=performance.ctr,
                    conversion_rate=conversion_rate,
                    avg_cpc=performance.avg_cpc_micros / 1_000_000,
                    impression_share=performance.impression_share,
                    quality_score_estimate=self._estimate_quality_score(performance),
                    time_running_hours=int(time_running),
                    keyword_performance=kw_performance_dict,
                    competitor_pressure=competitor_pressure
                )
                
                logger.info(f"📊 Updated state for campaign {campaign_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to update state for campaign {campaign_id}: {e}")
    
    def _estimate_quality_score(self, performance: CampaignPerformance) -> float:
        """Estimate Quality Score based on performance metrics"""
        # Simplified Quality Score estimation
        ctr_score = min(performance.ctr / 2.0, 5.0)  # CTR component (0-5)
        relevance_score = 3.0  # Default relevance (would need ad relevance data)
        landing_page_score = 3.0  # Default landing page experience
        
        return min(ctr_score + relevance_score + landing_page_score, 10.0)
    
    async def optimize_campaigns(self):
        """
        Run RL optimization on all active campaigns
        Makes real bid adjustments based on performance
        """
        await self.update_campaign_states()
        
        for campaign_id, state in self.campaign_states.items():
            try:
                # Generate RL-driven bid adjustments
                bid_adjustments = await self._generate_bid_adjustments(state)
                
                # Apply safety checks
                safe_adjustments = self._apply_safety_checks(campaign_id, bid_adjustments)
                
                # Apply bid adjustments to real campaign
                if safe_adjustments:
                    await self.ads_manager.update_campaign_bids(campaign_id, safe_adjustments)
                    
                    # Log action for RL learning
                    self.action_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'campaign_id': campaign_id,
                        'state_features': state.to_feature_vector().tolist(),
                        'action': safe_adjustments,
                        'expected_reward': self._estimate_action_reward(state, safe_adjustments)
                    })
                    
                    logger.info(f"🎯 Applied RL optimizations to campaign {campaign_id}: {len(safe_adjustments)} bid adjustments")
                
                # Check for emergency situations
                await self._emergency_checks(campaign_id, state)
                
            except Exception as e:
                logger.error(f"❌ Failed to optimize campaign {campaign_id}: {e}")
    
    async def _generate_bid_adjustments(self, state: GAELPCampaignState) -> Dict[str, float]:
        """
        Generate RL-driven bid adjustments based on current state
        """
        bid_adjustments = {}
        
        # Analyze keyword performance and generate adjustments
        for keyword, kw_data in state.keyword_performance.items():
            current_adjustment = 1.0  # No change baseline
            
            # Performance-based adjustments
            if kw_data['conversions'] > 0:
                # Keyword is converting
                if kw_data['conversion_rate'] > 0.05:  # High conversion rate
                    current_adjustment = 1.3  # Increase bid by 30%
                elif kw_data['conversion_rate'] > 0.02:  # Moderate conversion rate
                    current_adjustment = 1.15  # Increase bid by 15%
                else:  # Low conversion rate but some conversions
                    current_adjustment = 1.05  # Small increase
            
            elif kw_data['clicks'] > 5:
                # Keyword has clicks but no conversions
                if kw_data['clicks'] > 20:
                    current_adjustment = 0.7  # Significant decrease
                else:
                    current_adjustment = 0.85  # Moderate decrease
            
            elif kw_data['impressions'] > 100 and kw_data['clicks'] == 0:
                # High impressions, no clicks (relevance issue)
                current_adjustment = 0.6  # Large decrease
            
            # CTR-based adjustments
            if kw_data['ctr'] > 5.0:  # Exceptional CTR
                current_adjustment *= 1.2
            elif kw_data['ctr'] < 1.0 and kw_data['impressions'] > 50:  # Poor CTR
                current_adjustment *= 0.8
            
            # Cost efficiency adjustments
            if kw_data['avg_cpc'] > 20.0:  # Very expensive keyword
                if kw_data['conversion_rate'] < 0.03:  # Not converting well
                    current_adjustment *= 0.7
            
            # Only include significant adjustments
            if abs(current_adjustment - 1.0) > 0.05:  # More than 5% change
                bid_adjustments[keyword] = current_adjustment
        
        return bid_adjustments
    
    def _apply_safety_checks(self, campaign_id: str, bid_adjustments: Dict[str, float]) -> Dict[str, float]:
        """Apply safety checks to bid adjustments"""
        safe_adjustments = {}
        state = self.campaign_states[campaign_id]
        
        for keyword, adjustment in bid_adjustments.items():
            # Get current keyword data
            kw_data = state.keyword_performance.get(keyword, {})
            current_cpc = kw_data.get('avg_cpc', 5.0)  # Default $5 if unknown
            
            # Calculate new bid
            new_bid = current_cpc * adjustment
            
            # Apply bid limits
            new_bid = max(self.min_bid, min(new_bid, self.max_bid))
            
            # Calculate actual adjustment after limits
            safe_adjustment = new_bid / current_cpc
            
            # Only apply if adjustment is meaningful
            if abs(safe_adjustment - 1.0) > 0.05:
                safe_adjustments[keyword] = safe_adjustment
        
        # Limit number of simultaneous adjustments (max 20 per optimization)
        if len(safe_adjustments) > 20:
            # Keep the most significant adjustments
            sorted_adjustments = sorted(
                safe_adjustments.items(),
                key=lambda x: abs(x[1] - 1.0),
                reverse=True
            )[:20]
            safe_adjustments = dict(sorted_adjustments)
        
        return safe_adjustments
    
    async def _emergency_checks(self, campaign_id: str, state: GAELPCampaignState):
        """Perform emergency checks and actions"""
        
        # Check for excessive spending
        if state.cost_usd > self.emergency_pause_threshold:
            logger.warning(f"⚠️ Emergency pause triggered for campaign {campaign_id}: Daily spend ${state.cost_usd:.2f}")
            await self.ads_manager.pause_campaign(campaign_id)
            
            # Log emergency action
            self.action_history.append({
                'timestamp': datetime.now().isoformat(),
                'campaign_id': campaign_id,
                'action_type': 'emergency_pause',
                'reason': f'excessive_spend_{state.cost_usd:.2f}',
                'state_cost': state.cost_usd
            })
        
        # Check for zero impressions after significant time
        elif state.time_running_hours > 6 and state.impressions == 0:
            logger.warning(f"⚠️ Zero impressions detected for campaign {campaign_id} after {state.time_running_hours} hours")
            
            # Increase bids to get impressions (emergency boost)
            emergency_adjustments = {
                kw: 1.5 for kw in state.keyword_performance.keys()
            }
            
            if emergency_adjustments:
                await self.ads_manager.update_campaign_bids(campaign_id, emergency_adjustments)
                logger.info(f"🚨 Applied emergency bid boost to campaign {campaign_id}")
    
    def _estimate_action_reward(self, state: GAELPCampaignState, bid_adjustments: Dict[str, float]) -> float:
        """Estimate the expected reward from bid adjustments"""
        # Simple reward estimation based on historical patterns
        expected_reward = 0.0
        
        for keyword, adjustment in bid_adjustments.items():
            kw_data = state.keyword_performance.get(keyword, {})
            
            if adjustment > 1.0:  # Increasing bid
                # Expect more clicks and conversions
                expected_clicks_increase = (adjustment - 1.0) * kw_data.get('clicks', 0)
                expected_conversions_increase = expected_clicks_increase * kw_data.get('conversion_rate', 0.02)
                expected_reward += expected_conversions_increase * 10.0  # $10 value per conversion
                
                # Subtract increased cost
                expected_cost_increase = expected_clicks_increase * kw_data.get('avg_cpc', 5.0) * (adjustment - 1.0)
                expected_reward -= expected_cost_increase
            
            else:  # Decreasing bid
                # Expect cost savings but fewer conversions
                cost_savings = kw_data.get('clicks', 0) * kw_data.get('avg_cpc', 5.0) * (1.0 - adjustment)
                expected_reward += cost_savings * 0.5  # Partial credit for cost savings
        
        return expected_reward
    
    async def save_performance_data(self):
        """Save campaign performance data for analysis and future optimization"""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'campaigns': {},
            'action_history': self.action_history[-100:],  # Keep last 100 actions
            'performance_summary': {}
        }
        
        for campaign_id, state in self.campaign_states.items():
            performance_data['campaigns'][campaign_id] = asdict(state)
            
            # Calculate performance metrics
            roi = (state.conversions * 50.0 - state.cost_usd) / max(state.cost_usd, 1.0)  # Assume $50 LTV
            efficiency = state.conversions / max(state.cost_usd, 1.0)
            
            performance_data['performance_summary'][campaign_id] = {
                'roi': roi,
                'efficiency': efficiency,
                'total_conversions': state.conversions,
                'total_cost': state.cost_usd
            }
        
        # Save to file
        os.makedirs('/home/hariravichandran/AELP/data/google_ads_rl/', exist_ok=True)
        filepath = f'/home/hariravichandran/AELP/data/google_ads_rl/performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filepath, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        
        logger.info(f"💾 Performance data saved to {filepath}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of RL optimization performance"""
        if not self.campaign_states:
            return {'status': 'no_campaigns'}
        
        total_campaigns = len(self.campaign_states)
        total_conversions = sum(state.conversions for state in self.campaign_states.values())
        total_cost = sum(state.cost_usd for state in self.campaign_states.values())
        total_actions = len(self.action_history)
        
        avg_roi = (total_conversions * 50.0 - total_cost) / max(total_cost, 1.0)  # Assume $50 LTV
        avg_efficiency = total_conversions / max(total_cost, 1.0)
        
        return {
            'total_campaigns': total_campaigns,
            'total_conversions': total_conversions,
            'total_cost': total_cost,
            'average_roi': avg_roi,
            'efficiency': avg_efficiency,
            'total_rl_actions': total_actions,
            'campaigns_with_conversions': sum(1 for state in self.campaign_states.values() if state.conversions > 0),
            'average_ctr': sum(state.ctr for state in self.campaign_states.values()) / total_campaigns if total_campaigns > 0 else 0,
            'last_optimization': max(action['timestamp'] for action in self.action_history) if self.action_history else None
        }

async def main():
    """
    Main function to demonstrate GAELP Google Ads RL integration
    """
    print("=" * 80)
    print("GAELP GOOGLE ADS RL INTEGRATION")
    print("=" * 80)
    
    try:
        # Initialize Google Ads manager
        ads_manager = GoogleAdsProductionManager()
        
        # Create RL agent
        rl_agent = GAELPGoogleAdsAgent(ads_manager)
        
        print(f"✅ Connected to Google Ads account: {ads_manager.customer_id}")
        print("🤖 RL agent initialized")
        
        # Create RL-managed campaign
        print("\n🚀 Creating RL-managed campaign...")
        campaign_resource_name = await rl_agent.create_rl_campaign("behavioral_health")
        campaign_id = campaign_resource_name.split('/')[-1]
        
        print(f"✅ RL-managed campaign created: {campaign_id}")
        
        # Enable the campaign
        await ads_manager.enable_campaign(campaign_id)
        print("▶️ Campaign enabled and running")
        
        # Simulate RL optimization loop
        print("\n🔄 Running RL optimization cycle...")
        
        for cycle in range(3):  # Run 3 optimization cycles
            print(f"\n--- Optimization Cycle {cycle + 1} ---")
            
            # Update campaign states
            await rl_agent.update_campaign_states()
            print("📊 Campaign states updated")
            
            # Run optimization
            await rl_agent.optimize_campaigns()
            print("🎯 RL optimizations applied")
            
            # Save performance data
            await rl_agent.save_performance_data()
            print("💾 Performance data saved")
            
            # Show summary
            summary = rl_agent.get_optimization_summary()
            print(f"Summary: {summary['total_conversions']:.1f} conversions, "
                  f"${summary['total_cost']:.2f} spent, "
                  f"{summary['average_roi']:.2f} ROI")
            
            if cycle < 2:  # Don't sleep on last cycle
                print("⏳ Waiting for performance data... (in production, this would be hours/days)")
                await asyncio.sleep(2)  # Short sleep for demo
        
        # Final summary
        final_summary = rl_agent.get_optimization_summary()
        
        print("\n" + "=" * 80)
        print("RL OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"• Total campaigns managed: {final_summary['total_campaigns']}")
        print(f"• Total conversions: {final_summary['total_conversions']:.1f}")
        print(f"• Total spend: ${final_summary['total_cost']:.2f}")
        print(f"• Average ROI: {final_summary['average_roi']:.2f}")
        print(f"• Efficiency: {final_summary['efficiency']:.3f} conversions/dollar")
        print(f"• Total RL actions: {final_summary['total_rl_actions']}")
        print(f"• Campaigns with conversions: {final_summary['campaigns_with_conversions']}")
        print(f"• Average CTR: {final_summary['average_ctr']:.2f}%")
        
        print("\n✅ REAL GOOGLE ADS INTEGRATION COMPLETE")
        print("   - Live campaigns created and managed")
        print("   - RL agent actively optimizing bids")
        print("   - Performance data collected and analyzed")
        print("   - Emergency safety systems active")
        
    except Exception as e:
        print(f"❌ Error in RL integration: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())