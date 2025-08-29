#!/usr/bin/env python3
"""
DISPLAY BOT FILTERING SYSTEM
Critical: 85% of 150K display sessions are bots/fraud

This system implements multi-layer bot detection and filtering:
1. Real-time traffic analysis
2. Behavioral pattern detection  
3. Placement quality scoring
4. Automated exclusion management

TARGET: Reduce bot traffic from 85% to <20%
IMPACT: Turn 150K worthless sessions into 120K+ quality sessions
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import re

@dataclass
class TrafficQualityMetrics:
    """Traffic quality indicators for bot detection"""
    bounce_rate: float
    avg_session_duration: float
    pages_per_session: float
    new_user_rate: float
    suspicious_user_agent: bool
    geo_anomaly: bool
    click_pattern_anomaly: bool
    conversion_rate: float

@dataclass 
class PlacementQuality:
    """Quality metrics for display placements"""
    placement_url: str
    sessions: int
    conversions: int
    bounce_rate: float
    avg_duration: float
    bot_score: float
    quality_score: float
    action: str  # 'keep', 'monitor', 'exclude'

class DisplayBotFilter:
    """Advanced bot filtering for Display channel"""
    
    def __init__(self):
        self.bot_detection_rules = self.load_bot_detection_rules()
        self.placement_blacklist = set()
        self.placement_whitelist = set()
        self.suspicious_patterns = self.load_suspicious_patterns()
        
    def load_bot_detection_rules(self) -> Dict:
        """Load bot detection rules and thresholds"""
        return {
            'bounce_rate_threshold': 0.95,  # >95% bounce is suspicious
            'duration_threshold': 2.0,      # <2 seconds is suspicious  
            'pages_threshold': 1.1,         # <1.1 pages is suspicious
            'new_user_threshold': 0.99,     # >99% new users is suspicious
            'conversion_threshold': 0.001,  # <0.1% CVR is suspicious
            'geo_whitelist': {'US', 'CA', 'GB', 'AU', 'IE'},
            'suspicious_user_agents': {
                'bot', 'crawler', 'spider', 'scraper', 
                'automated', 'headless', 'phantom'
            }
        }
    
    def load_suspicious_patterns(self) -> Dict:
        """Load patterns that indicate bot traffic"""
        return {
            'domain_patterns': [
                r'.*\.tk$',  # Free domains
                r'.*\.ml$', 
                r'.*\.ga$',
                r'.*click.*',  # Click networks
                r'.*traffic.*',
                r'.*ads.*network.*'
            ],
            'url_patterns': [
                r'.*bot.*',
                r'.*fake.*',  
                r'.*spam.*',
                r'.*auto.*',
                r'.*generated.*'
            ]
        }
    
    def analyze_traffic_quality(self, traffic_data: Dict) -> TrafficQualityMetrics:
        """Analyze traffic data for bot indicators"""
        
        # Calculate quality metrics
        bounce_rate = traffic_data.get('bounce_rate', 0)
        avg_duration = traffic_data.get('avg_session_duration', 0)
        pages_per_session = traffic_data.get('pages_per_session', 1)
        new_user_rate = traffic_data.get('new_user_rate', 1)
        conversions = traffic_data.get('conversions', 0)
        sessions = traffic_data.get('sessions', 1)
        
        # Check user agent
        user_agent = traffic_data.get('user_agent', '').lower()
        suspicious_ua = any(pattern in user_agent for pattern in self.bot_detection_rules['suspicious_user_agents'])
        
        # Check geography
        country = traffic_data.get('country', '')
        geo_anomaly = country not in self.bot_detection_rules['geo_whitelist']
        
        # Check click patterns (simplified)
        click_pattern_anomaly = self.detect_click_pattern_anomaly(traffic_data)
        
        return TrafficQualityMetrics(
            bounce_rate=bounce_rate,
            avg_session_duration=avg_duration,
            pages_per_session=pages_per_session,
            new_user_rate=new_user_rate,
            suspicious_user_agent=suspicious_ua,
            geo_anomaly=geo_anomaly,
            click_pattern_anomaly=click_pattern_anomaly,
            conversion_rate=conversions / sessions if sessions > 0 else 0
        )
    
    def detect_click_pattern_anomaly(self, traffic_data: Dict) -> bool:
        """Detect suspicious click patterns"""
        # Simplified pattern detection
        ctr = traffic_data.get('ctr', 0)
        
        # Impossibly high or low CTRs
        if ctr > 0.15 or ctr < 0.001:  # >15% or <0.1% CTR
            return True
            
        # Other pattern checks could include:
        # - Click velocity
        # - Time-of-day patterns
        # - Sequential clicking patterns
        
        return False
    
    def calculate_bot_score(self, metrics: TrafficQualityMetrics) -> float:
        """Calculate bot probability score (0-1)"""
        
        rules = self.bot_detection_rules
        score = 0.0
        
        # Bounce rate check (30% weight)
        if metrics.bounce_rate > rules['bounce_rate_threshold']:
            score += 0.30
        
        # Duration check (25% weight)  
        if metrics.avg_session_duration < rules['duration_threshold']:
            score += 0.25
            
        # Pages per session check (15% weight)
        if metrics.pages_per_session < rules['pages_threshold']:
            score += 0.15
            
        # New user rate check (10% weight)
        if metrics.new_user_rate > rules['new_user_threshold']:
            score += 0.10
            
        # Conversion rate check (10% weight)
        if metrics.conversion_rate < rules['conversion_threshold']:
            score += 0.10
            
        # User agent check (5% weight)
        if metrics.suspicious_user_agent:
            score += 0.05
            
        # Geo anomaly check (3% weight)
        if metrics.geo_anomaly:
            score += 0.03
            
        # Click pattern anomaly (2% weight)
        if metrics.click_pattern_anomaly:
            score += 0.02
        
        return min(score, 1.0)  # Cap at 1.0
    
    def analyze_placements(self, placement_data: List[Dict]) -> List[PlacementQuality]:
        """Analyze display placements for quality"""
        
        placement_analysis = []
        
        for placement in placement_data:
            url = placement.get('placement_url', '')
            sessions = placement.get('sessions', 0)
            conversions = placement.get('conversions', 0)
            bounce_rate = placement.get('bounce_rate', 1.0)
            avg_duration = placement.get('avg_session_duration', 0)
            
            # Calculate bot score based on placement metrics
            placement_metrics = TrafficQualityMetrics(
                bounce_rate=bounce_rate,
                avg_session_duration=avg_duration,
                pages_per_session=placement.get('pages_per_session', 1),
                new_user_rate=placement.get('new_user_rate', 1),
                suspicious_user_agent=False,  # Placement-level doesn't have UA
                geo_anomaly=False,            # Placement-level analysis
                click_pattern_anomaly=False,
                conversion_rate=conversions / sessions if sessions > 0 else 0
            )
            
            bot_score = self.calculate_bot_score(placement_metrics)
            
            # Calculate quality score (inverse of bot score + traffic volume bonus)
            traffic_bonus = min(sessions / 1000, 0.2)  # Up to 20% bonus for volume
            quality_score = max(0, (1 - bot_score) + traffic_bonus)
            
            # Determine action
            if bot_score > 0.8:
                action = 'exclude'
            elif bot_score > 0.6:
                action = 'monitor'
            else:
                action = 'keep'
            
            # Check against suspicious patterns
            if self.is_suspicious_domain(url):
                action = 'exclude'
                bot_score = max(bot_score, 0.9)
            
            placement_analysis.append(PlacementQuality(
                placement_url=url,
                sessions=sessions,
                conversions=conversions,
                bounce_rate=bounce_rate,
                avg_duration=avg_duration,
                bot_score=bot_score,
                quality_score=quality_score,
                action=action
            ))
        
        return placement_analysis
    
    def is_suspicious_domain(self, url: str) -> bool:
        """Check if domain matches suspicious patterns"""
        
        for pattern in self.suspicious_patterns['domain_patterns']:
            if re.match(pattern, url, re.IGNORECASE):
                return True
                
        for pattern in self.suspicious_patterns['url_patterns']:
            if re.search(pattern, url, re.IGNORECASE):
                return True
                
        return False
    
    def create_exclusion_lists(self, placement_analysis: List[PlacementQuality]) -> Dict:
        """Create exclusion lists for campaign management"""
        
        exclude_placements = []
        monitor_placements = []
        keep_placements = []
        
        for placement in placement_analysis:
            if placement.action == 'exclude':
                exclude_placements.append({
                    'url': placement.placement_url,
                    'reason': f'Bot score: {placement.bot_score:.2f}',
                    'sessions_saved': placement.sessions
                })
            elif placement.action == 'monitor':
                monitor_placements.append({
                    'url': placement.placement_url,
                    'bot_score': placement.bot_score,
                    'sessions': placement.sessions
                })
            else:
                keep_placements.append({
                    'url': placement.placement_url,
                    'quality_score': placement.quality_score,
                    'sessions': placement.sessions
                })
        
        return {
            'exclude': exclude_placements,
            'monitor': monitor_placements, 
            'keep': keep_placements,
            'summary': {
                'total_analyzed': len(placement_analysis),
                'to_exclude': len(exclude_placements),
                'to_monitor': len(monitor_placements),
                'to_keep': len(keep_placements),
                'sessions_filtered': sum(p['sessions_saved'] for p in exclude_placements)
            }
        }
    
    def run_bot_filtering_analysis(self) -> Dict:
        """Run complete bot filtering analysis"""
        print("\n" + "="*80)
        print("🤖 DISPLAY BOT FILTERING ANALYSIS")
        print("="*80)
        
        # TODO: Replace with real GA4 data
        # For now, raise error to force real data implementation
        raise NotImplementedError("Must use real placement data from GA4, not mock data")
        
        # Analyze placements
        placement_analysis = self.analyze_placements(mock_placement_data)
        
        # Create exclusion lists
        exclusion_lists = self.create_exclusion_lists(placement_analysis)
        
        # Print results
        self.print_filtering_results(exclusion_lists)
        
        # Generate implementation recommendations
        recommendations = self.generate_implementation_recommendations(exclusion_lists)
        
        # Save results
        self.save_filtering_results(exclusion_lists, recommendations)
        
        return {
            'exclusion_lists': exclusion_lists,
            'recommendations': recommendations
        }
    
    def generate_mock_placement_data(self) -> List[Dict]:
        """Generate mock placement data for testing"""
        return [
            {
                'placement_url': 'suspicious-ads-network.com',
                'sessions': 45000,
                'conversions': 0,
                'bounce_rate': 0.99,
                'avg_session_duration': 0.3,
                'pages_per_session': 1.0,
                'new_user_rate': 1.0
            },
            {
                'placement_url': 'fake-traffic-source.tk',
                'sessions': 30000,
                'conversions': 1,
                'bounce_rate': 0.98,
                'avg_session_duration': 0.5,
                'pages_per_session': 1.0,
                'new_user_rate': 0.999
            },
            {
                'placement_url': 'parenting-blog-network.com',
                'sessions': 15000,
                'conversions': 45,
                'bounce_rate': 0.75,
                'avg_session_duration': 120.0,
                'pages_per_session': 2.3,
                'new_user_rate': 0.85
            },
            {
                'placement_url': 'mental-health-resources.org',
                'sessions': 12000,
                'conversions': 72,
                'bounce_rate': 0.65,
                'avg_session_duration': 180.0,
                'pages_per_session': 3.1,
                'new_user_rate': 0.70
            },
            {
                'placement_url': 'bot-generated-content.ml',
                'sessions': 25000,
                'conversions': 2,
                'bounce_rate': 0.97,
                'avg_session_duration': 0.8,
                'pages_per_session': 1.01,
                'new_user_rate': 0.995
            },
            {
                'placement_url': 'family-wellness-magazine.com',
                'sessions': 8000,
                'conversions': 28,
                'bounce_rate': 0.70,
                'avg_session_duration': 95.0,
                'pages_per_session': 2.8,
                'new_user_rate': 0.75
            }
        ]
    
    def print_filtering_results(self, exclusion_lists: Dict):
        """Print bot filtering results"""
        
        summary = exclusion_lists['summary']
        
        print(f"\n📊 PLACEMENT ANALYSIS RESULTS:")
        print(f"   Total placements analyzed: {summary['total_analyzed']}")
        print(f"   To exclude (high bot): {summary['to_exclude']}")
        print(f"   To monitor (medium bot): {summary['to_monitor']}")
        print(f"   To keep (quality): {summary['to_keep']}")
        
        print(f"\n🎯 TRAFFIC IMPACT:")
        print(f"   Sessions to filter out: {summary['sessions_filtered']:,}")
        print(f"   Estimated bot reduction: {summary['sessions_filtered']/150000*100:.1f}%")
        print(f"   Quality sessions remaining: {150000 - summary['sessions_filtered']:,}")
        
        print(f"\n❌ TOP EXCLUSIONS:")
        for exclusion in exclusion_lists['exclude'][:5]:
            print(f"   • {exclusion['url']} - {exclusion['sessions_saved']:,} sessions ({exclusion['reason']})")
        
        print(f"\n👀 PLACEMENTS TO MONITOR:")
        for monitor in exclusion_lists['monitor'][:3]:
            print(f"   • {monitor['url']} - {monitor['sessions']:,} sessions (Bot score: {monitor['bot_score']:.2f})")
        
        print(f"\n✅ QUALITY PLACEMENTS TO KEEP:")
        for keep in exclusion_lists['keep'][:3]:
            print(f"   • {keep['url']} - {keep['sessions']:,} sessions (Quality: {keep['quality_score']:.2f})")
    
    def generate_implementation_recommendations(self, exclusion_lists: Dict) -> List[Dict]:
        """Generate implementation recommendations"""
        
        recommendations = []
        
        # Immediate exclusions
        if exclusion_lists['exclude']:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'action': 'Exclude high-bot placements',
                'details': f"Exclude {len(exclusion_lists['exclude'])} placements immediately",
                'impact': f"Remove {exclusion_lists['summary']['sessions_filtered']:,} bot sessions",
                'implementation': 'Add placement exclusions in Google Ads'
            })
        
        # Monitoring setup
        if exclusion_lists['monitor']:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Set up monitoring alerts',
                'details': f"Monitor {len(exclusion_lists['monitor'])} suspicious placements",
                'impact': 'Prevent future bot traffic accumulation',
                'implementation': 'Weekly placement performance review'
            })
        
        # Quality placement optimization
        if exclusion_lists['keep']:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Optimize quality placements',
                'details': f"Increase bids on {len(exclusion_lists['keep'])} quality placements",
                'impact': 'Improve overall campaign performance',
                'implementation': 'Bid adjustments +20-50% for quality sites'
            })
        
        # Automated rules
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Implement automated bot detection',
            'details': 'Set up automated rules for ongoing bot filtering',
            'impact': 'Continuous protection against new bot sources',
            'implementation': 'Google Ads automated rules + custom scripts'
        })
        
        return recommendations
    
    def save_filtering_results(self, exclusion_lists: Dict, recommendations: List[Dict]):
        """Save filtering results to files"""
        
        # Save exclusion lists
        with open('/home/hariravichandran/AELP/display_bot_exclusions.json', 'w') as f:
            json.dump(exclusion_lists, f, indent=2)
        
        # Save recommendations
        with open('/home/hariravichandran/AELP/display_bot_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Create CSV for easy import to Google Ads
        exclude_df = pd.DataFrame(exclusion_lists['exclude'])
        exclude_df.to_csv('/home/hariravichandran/AELP/display_exclude_placements.csv', index=False)
        
        print(f"\n💾 Files saved:")
        print(f"   • display_bot_exclusions.json - Complete analysis")
        print(f"   • display_bot_recommendations.json - Implementation plan") 
        print(f"   • display_exclude_placements.csv - For Google Ads import")

if __name__ == "__main__":
    bot_filter = DisplayBotFilter()
    results = bot_filter.run_bot_filtering_analysis()
    
    print(f"\n" + "="*80)
    print("🎯 NEXT STEPS - IMPLEMENT BOT FILTERING")
    print("="*80)
    print("1. Upload display_exclude_placements.csv to Google Ads")
    print("2. Set up automated rules for ongoing monitoring") 
    print("3. Review placement performance weekly")
    print("4. Adjust bids for quality placements")
    print("5. Monitor CVR improvement (target: 0.01% → 0.1%+)")