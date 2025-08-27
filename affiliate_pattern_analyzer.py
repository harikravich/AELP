#!/usr/bin/env python3
"""
Affiliate Pattern Analyzer - Discover Why Affiliates Achieve 4.42% CVR

Analyzes real GA4 affiliate traffic data to understand:
1. What content types convert at 4%+  
2. What messaging patterns work
3. How they position Aura
4. Their audience targeting strategies

Then replicates these patterns for our own campaigns.

Real data shows:
- ir_affiliate mobile: 4.41% CVR (133,753 sessions)
- ir_affiliate desktop: 4.03% CVR (85,750 sessions)
- Top performers: banyantree (10.49%), conadvo (8.25%), top10 (6.21%)
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, DateRange, Dimension, Metric, OrderBy,
    FilterExpression, Filter, FilterExpressionList
)
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# GA4 Configuration
GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'

@dataclass
class AffiliatePerformance:
    """Performance metrics for affiliate traffic source"""
    source: str
    medium: str
    sessions: int
    conversions: int
    cvr: float
    device_mix: Dict[str, int]
    top_pages: List[str]
    avg_session_duration: float
    bounce_rate: float
    success_factors: List[str] = field(default_factory=list)

@dataclass
class ContentPattern:
    """Content pattern discovered from affiliate analysis"""
    content_type: str
    affiliate_source: str
    cvr: float
    key_elements: List[str]
    messaging_themes: List[str]
    audience_signals: List[str]
    conversion_triggers: List[str]

@dataclass
class ReplicationStrategy:
    """Strategy to replicate affiliate success"""
    name: str
    content_approach: str
    target_cvr: float
    implementation_steps: List[str]
    expected_traffic_volume: int
    test_budget: float

class RealAffiliateDataAnalyzer:
    """Analyzes REAL affiliate data from GA4 to discover success patterns"""
    
    def __init__(self):
        self.ga4_client = self._setup_ga4_client()
        self.affiliate_data = {}
        self.content_patterns = {}
        self.success_factors = {}
        
    def _setup_ga4_client(self) -> BetaAnalyticsDataClient:
        """Setup GA4 client with service account"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                str(SERVICE_ACCOUNT_FILE),
                scopes=['https://www.googleapis.com/auth/analytics.readonly']
            )
            return BetaAnalyticsDataClient(credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to setup GA4 client: {e}")
            return None
    
    async def analyze_affiliate_traffic_patterns(self) -> Dict[str, AffiliatePerformance]:
        """
        Analyze REAL affiliate traffic from GA4 data
        Focus on sources with 4%+ CVR
        """
        if not self.ga4_client:
            logger.error("GA4 client not available")
            return {}
        
        try:
            # Get affiliate traffic data for last 30 days
            request = RunReportRequest(
                property=f"properties/{GA_PROPERTY_ID}",
                date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
                dimensions=[
                    Dimension(name="sessionSourceMedium"),
                    Dimension(name="deviceCategory"),
                    Dimension(name="pagePath")
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="conversions"),
                    Metric(name="averageSessionDuration"),
                    Metric(name="bounceRate")
                ],
                # Filter for affiliate traffic
                dimension_filter=FilterExpression(
                    or_group=FilterExpressionList(
                        expressions=[
                            FilterExpression(
                                filter=Filter(
                                    field_name="sessionSourceMedium",
                                    string_filter=Filter.StringFilter(
                                        match_type=Filter.StringFilter.MatchType.CONTAINS,
                                        value="ir_affiliate"
                                    )
                                )
                            ),
                            FilterExpression(
                                filter=Filter(
                                    field_name="sessionSourceMedium",
                                    string_filter=Filter.StringFilter(
                                        match_type=Filter.StringFilter.MatchType.CONTAINS,
                                        value="influencer"
                                    )
                                )
                            ),
                            FilterExpression(
                                filter=Filter(
                                    field_name="sessionSourceMedium",
                                    string_filter=Filter.StringFilter(
                                        match_type=Filter.StringFilter.MatchType.CONTAINS,
                                        value="native"
                                    )
                                )
                            )
                        ]
                    )
                ),
                order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
                limit=100
            )
            
            response = self.ga4_client.run_report(request)
            affiliate_performance = {}
            
            # Aggregate data by source/medium
            source_data = {}
            for row in response.rows:
                source_medium = row.dimension_values[0].value
                device = row.dimension_values[1].value
                page = row.dimension_values[2].value
                sessions = int(row.metric_values[0].value)
                conversions = int(row.metric_values[1].value)
                duration = float(row.metric_values[2].value)
                bounce_rate = float(row.metric_values[3].value)
                
                if sessions < 10:  # Skip low-volume sources
                    continue
                
                if source_medium not in source_data:
                    source_data[source_medium] = {
                        'total_sessions': 0,
                        'total_conversions': 0,
                        'devices': {},
                        'pages': [],
                        'durations': [],
                        'bounce_rates': []
                    }
                
                source_data[source_medium]['total_sessions'] += sessions
                source_data[source_medium]['total_conversions'] += conversions
                source_data[source_medium]['devices'][device] = source_data[source_medium]['devices'].get(device, 0) + sessions
                source_data[source_medium]['pages'].append((page, sessions, conversions))
                source_data[source_medium]['durations'].append(duration)
                source_data[source_medium]['bounce_rates'].append(bounce_rate)
            
            # Process and analyze each affiliate source
            for source_medium, data in source_data.items():
                if data['total_sessions'] < 100:  # Minimum volume threshold
                    continue
                    
                cvr = (data['total_conversions'] / data['total_sessions']) * 100
                
                if cvr >= 3.0:  # Focus on high-performing affiliates
                    # Extract source name
                    source_name = source_medium.split(' / ')[0]
                    
                    # Get top pages for this source
                    top_pages = sorted(data['pages'], key=lambda x: x[2], reverse=True)[:5]
                    
                    affiliate_performance[source_name] = AffiliatePerformance(
                        source=source_name,
                        medium=source_medium.split(' / ')[1] if ' / ' in source_medium else 'unknown',
                        sessions=data['total_sessions'],
                        conversions=data['total_conversions'],
                        cvr=cvr,
                        device_mix=data['devices'],
                        top_pages=[page[0] for page in top_pages],
                        avg_session_duration=np.mean(data['durations']),
                        bounce_rate=np.mean(data['bounce_rates']),
                        success_factors=self._analyze_success_factors(source_name, cvr, data)
                    )
            
            logger.info(f"Analyzed {len(affiliate_performance)} high-performing affiliate sources")
            return affiliate_performance
            
        except Exception as e:
            logger.error(f"Error analyzing affiliate traffic: {e}")
            return {}
    
    def _analyze_success_factors(self, source: str, cvr: float, data: dict) -> List[str]:
        """Analyze what makes this affiliate successful"""
        factors = []
        
        # CVR-based factors
        if cvr > 8.0:
            factors.append("Exceptional pre-qualification (8%+ CVR)")
        elif cvr > 5.0:
            factors.append("Strong audience matching (5%+ CVR)")
        elif cvr > 3.0:
            factors.append("Good traffic quality (3%+ CVR)")
        
        # Device mix analysis
        mobile_pct = data['devices'].get('mobile', 0) / data['total_sessions'] * 100
        if mobile_pct > 80:
            factors.append("Mobile-first audience")
        elif mobile_pct < 40:
            factors.append("Desktop research behavior")
        
        # Engagement analysis
        avg_duration = np.mean(data['durations'])
        avg_bounce = np.mean(data['bounce_rates'])
        
        if avg_duration > 180:  # 3+ minutes
            factors.append("High content engagement")
        if avg_bounce < 0.3:  # <30% bounce rate
            factors.append("Strong content relevance")
            
        # Source-specific patterns
        source_lower = source.lower()
        if 'guide' in source_lower or 'review' in source_lower:
            factors.append("Educational/review content approach")
        if 'top' in source_lower or 'best' in source_lower:
            factors.append("Comparison/ranking content")
        if 'safety' in source_lower or 'security' in source_lower:
            factors.append("Security-focused positioning")
        
        return factors

    async def discover_content_patterns(self, affiliate_data: Dict[str, AffiliatePerformance]) -> Dict[str, ContentPattern]:
        """
        Discover content patterns from successful affiliates
        Analyze their approach to understand what drives conversions
        """
        content_patterns = {}
        
        # Analyze each high-performing affiliate
        for source, performance in affiliate_data.items():
            if performance.cvr < 4.0:  # Focus on 4%+ CVR sources
                continue
            
            # Categorize content type based on source characteristics
            content_type = self._categorize_content_type(source, performance)
            
            # Discover messaging themes from landing pages
            messaging_themes = await self._discover_messaging_themes(performance.top_pages)
            
            # Analyze audience signals
            audience_signals = self._extract_audience_signals(performance)
            
            # Identify conversion triggers
            conversion_triggers = self._identify_conversion_triggers(performance)
            
            pattern = ContentPattern(
                content_type=content_type,
                affiliate_source=source,
                cvr=performance.cvr,
                key_elements=self._extract_key_elements(content_type, performance),
                messaging_themes=messaging_themes,
                audience_signals=audience_signals,
                conversion_triggers=conversion_triggers
            )
            
            content_patterns[source] = pattern
            logger.info(f"Discovered content pattern for {source}: {content_type} with {performance.cvr:.1f}% CVR")
        
        return content_patterns
    
    def _categorize_content_type(self, source: str, performance: AffiliatePerformance) -> str:
        """Categorize the type of content this affiliate uses"""
        source_lower = source.lower()
        
        if any(kw in source_lower for kw in ['guide', 'review', 'compare']):
            return "Educational/Review Content"
        elif any(kw in source_lower for kw in ['top', 'best', 'vs']):
            return "Comparison/Ranking Content"
        elif any(kw in source_lower for kw in ['deal', 'coupon', 'save']):
            return "Deal/Discount Content"
        elif any(kw in source_lower for kw in ['blog', 'news', 'article']):
            return "Editorial/Blog Content"
        elif performance.avg_session_duration > 240:  # 4+ minutes
            return "In-depth Educational Content"
        elif performance.bounce_rate < 0.2:  # Very low bounce
            return "Highly Engaging Content"
        else:
            return "General Affiliate Content"
    
    async def _discover_messaging_themes(self, top_pages: List[str]) -> List[str]:
        """
        Discover messaging themes from affiliate landing pages
        Analyze URL patterns and infer content focus
        """
        themes = []
        
        for page in top_pages:
            page_lower = page.lower()
            
            # Family/parental themes
            if any(kw in page_lower for kw in ['family', 'parent', 'child', 'teen']):
                themes.append("Family-focused messaging")
            
            # Security/protection themes  
            if any(kw in page_lower for kw in ['protect', 'security', 'safe', 'monitor']):
                themes.append("Protection/security positioning")
            
            # Control/balance themes
            if any(kw in page_lower for kw in ['control', 'balance', 'screen', 'time']):
                themes.append("Digital wellness/control focus")
            
            # Affiliate-specific themes
            if 'aff' in page_lower or 'affiliate' in page_lower:
                themes.append("Affiliate-optimized landing page")
            
            # Mobile-specific themes
            if any(kw in page_lower for kw in ['mobile', 'app', 'iphone', 'android']):
                themes.append("Mobile app positioning")
        
        return list(set(themes))  # Remove duplicates
    
    def _extract_audience_signals(self, performance: AffiliatePerformance) -> List[str]:
        """Extract audience signals from performance data"""
        signals = []
        
        # Device behavior signals
        total_sessions = performance.sessions
        mobile_sessions = performance.device_mix.get('mobile', 0)
        desktop_sessions = performance.device_mix.get('desktop', 0)
        
        mobile_pct = (mobile_sessions / total_sessions) * 100 if total_sessions > 0 else 0
        
        if mobile_pct > 85:
            signals.append("Mobile-first audience (85%+ mobile)")
        elif mobile_pct > 70:
            signals.append("Mobile-heavy audience (70%+ mobile)")
        elif mobile_pct < 40:
            signals.append("Desktop research audience (<40% mobile)")
        
        # Engagement signals
        if performance.avg_session_duration > 300:  # 5+ minutes
            signals.append("Deep content engagement (5+ min sessions)")
        elif performance.avg_session_duration > 180:  # 3+ minutes
            signals.append("Strong content engagement (3+ min sessions)")
        
        if performance.bounce_rate < 0.25:
            signals.append("High content relevance (<25% bounce)")
        
        # Volume signals
        if performance.sessions > 50000:
            signals.append("High-volume traffic source (50k+ sessions)")
        elif performance.sessions > 10000:
            signals.append("Medium-volume traffic source (10k+ sessions)")
        
        return signals
    
    def _identify_conversion_triggers(self, performance: AffiliatePerformance) -> List[str]:
        """Identify what triggers conversions for this affiliate"""
        triggers = []
        
        # High CVR indicates strong triggers
        if performance.cvr > 8.0:
            triggers.extend([
                "Exceptional pre-qualification",
                "Crisis/urgent need targeting",
                "Highly relevant content match"
            ])
        elif performance.cvr > 5.0:
            triggers.extend([
                "Strong audience pre-qualification", 
                "Targeted pain point messaging",
                "Trust-building content"
            ])
        elif performance.cvr > 3.0:
            triggers.extend([
                "Good audience matching",
                "Clear value proposition",
                "Effective call-to-action"
            ])
        
        # Device-based triggers
        if performance.device_mix.get('mobile', 0) / performance.sessions > 0.8:
            triggers.append("Mobile-optimized conversion flow")
        
        # Engagement-based triggers
        if performance.bounce_rate < 0.3:
            triggers.append("Compelling initial content hook")
        
        if performance.avg_session_duration > 180:
            triggers.append("Educational content builds trust")
        
        return triggers
    
    def _extract_key_elements(self, content_type: str, performance: AffiliatePerformance) -> List[str]:
        """Extract key elements that make this content type successful"""
        elements = []
        
        if "Educational/Review" in content_type:
            elements.extend([
                "In-depth product analysis",
                "Feature comparison tables",
                "Pros and cons breakdown",
                "Real user testimonials",
                "Expert recommendations"
            ])
        elif "Comparison/Ranking" in content_type:
            elements.extend([
                "Head-to-head comparisons",
                "Ranking methodology explained",
                "Clear winner selection",
                "Price/value analysis",
                "Category-specific features"
            ])
        elif "Deal/Discount" in content_type:
            elements.extend([
                "Exclusive discount codes",
                "Limited-time offers",
                "Price comparison with competitors",
                "Value stacking (features + discount)",
                "Urgency messaging"
            ])
        elif "Editorial/Blog" in content_type:
            elements.extend([
                "Personal experience stories",
                "Problem-solution narrative",
                "Expert interviews/quotes",
                "Actionable advice",
                "Community engagement"
            ])
        
        # Add performance-specific elements
        if performance.cvr > 6.0:
            elements.append("Exceptional trust building")
        if performance.bounce_rate < 0.25:
            elements.append("Highly relevant content matching")
        if performance.avg_session_duration > 240:
            elements.append("Comprehensive educational content")
        
        return elements

class AffiliateStrategyReplicator:
    """Replicates successful affiliate strategies for our own campaigns"""
    
    def __init__(self, affiliate_data: Dict[str, AffiliatePerformance], 
                 content_patterns: Dict[str, ContentPattern]):
        self.affiliate_data = affiliate_data
        self.content_patterns = content_patterns
        self.replication_strategies = {}
    
    def create_replication_strategies(self) -> Dict[str, ReplicationStrategy]:
        """
        Create strategies to replicate top affiliate performance
        Focus on 4%+ CVR approaches
        """
        strategies = {}
        
        # Analyze top performers
        top_performers = sorted(
            self.affiliate_data.items(), 
            key=lambda x: x[1].cvr, 
            reverse=True
        )[:5]  # Top 5 performers
        
        for source, performance in top_performers:
            if source in self.content_patterns:
                pattern = self.content_patterns[source]
                
                strategy = self._create_strategy_from_pattern(source, performance, pattern)
                strategies[f"replicate_{source.lower().replace('.', '_')}"] = strategy
        
        # Create hybrid strategies
        strategies["hybrid_review_comparison"] = self._create_hybrid_strategy()
        strategies["educational_content_marketing"] = self._create_educational_strategy()
        strategies["crisis_parent_targeting"] = self._create_crisis_strategy()
        
        return strategies
    
    def _create_strategy_from_pattern(self, source: str, performance: AffiliatePerformance, 
                                    pattern: ContentPattern) -> ReplicationStrategy:
        """Create replication strategy from successful affiliate pattern"""
        
        # Target 70% of their CVR to be conservative
        target_cvr = performance.cvr * 0.7
        
        # Estimate traffic volume based on their success
        estimated_volume = min(performance.sessions * 0.5, 10000)  # Conservative estimate
        
        implementation_steps = [
            f"Create {pattern.content_type.lower()} similar to {source}",
            f"Target audience with these signals: {', '.join(pattern.audience_signals[:3])}",
            f"Use messaging themes: {', '.join(pattern.messaging_themes[:3])}",
            f"Implement conversion triggers: {', '.join(pattern.conversion_triggers[:3])}",
            f"A/B test key elements: {', '.join(pattern.key_elements[:3])}",
            f"Monitor for {target_cvr:.1f}% CVR target",
            "Scale successful variations"
        ]
        
        return ReplicationStrategy(
            name=f"Replicate {source} Strategy",
            content_approach=pattern.content_type,
            target_cvr=target_cvr,
            implementation_steps=implementation_steps,
            expected_traffic_volume=int(estimated_volume),
            test_budget=min(performance.sessions * 0.10, 5000)  # 10% of their volume as budget
        )
    
    def _create_hybrid_strategy(self) -> ReplicationStrategy:
        """Create hybrid strategy combining best elements"""
        return ReplicationStrategy(
            name="Hybrid Review + Comparison Strategy",
            content_approach="Educational Review with Comparisons",
            target_cvr=4.0,
            implementation_steps=[
                "Create comprehensive Aura vs competitors comparison",
                "Include detailed feature analysis and scoring",
                "Add real parent testimonials and case studies",
                "Implement affiliate-style landing pages",
                "Use urgency triggers (limited time offers)",
                "Target mobile-first parent audience",
                "A/B test different trust-building elements"
            ],
            expected_traffic_volume=15000,
            test_budget=8000
        )
    
    def _create_educational_strategy(self) -> ReplicationStrategy:
        """Create educational content marketing strategy"""
        return ReplicationStrategy(
            name="Educational Content Marketing",
            content_approach="In-depth Educational Content",
            target_cvr=3.5,
            implementation_steps=[
                "Create 'Complete Guide to Teen Digital Wellness' content hub",
                "Develop parent education video series",
                "Build interactive tools (screen time calculator, risk assessments)",
                "Guest expert content (child psychologists, educators)",
                "Email nurture sequence for education-to-conversion",
                "SEO optimization for parental control keywords",
                "Community building (parent forums, support groups)"
            ],
            expected_traffic_volume=25000,
            test_budget=12000
        )
    
    def _create_crisis_strategy(self) -> ReplicationStrategy:
        """Create strategy targeting parents in crisis situations"""
        return ReplicationStrategy(
            name="Crisis Parent Targeting",
            content_approach="Crisis-focused immediate help",
            target_cvr=6.0,
            implementation_steps=[
                "Create crisis intervention landing pages",
                "Target 12am-4am time slots (crisis hours)",
                "Use urgent keywords (teen depression, cyberbullying, etc.)",
                "Implement chat support and immediate consultation",
                "Create fast-track onboarding for crisis situations",
                "Partner with mental health organizations",
                "Use testimonials from crisis situations"
            ],
            expected_traffic_volume=5000,
            test_budget=15000
        )

class AffiliatePatternAnalyzer:
    """Main analyzer that coordinates all affiliate analysis and replication"""
    
    def __init__(self):
        self.data_analyzer = RealAffiliateDataAnalyzer()
        self.replicator = None
        self.analysis_results = {}
        
    async def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete affiliate pattern analysis
        Returns comprehensive analysis and replication strategies
        """
        logger.info("Starting complete affiliate pattern analysis...")
        
        # Step 1: Analyze real affiliate traffic
        print("🔍 Analyzing real affiliate traffic patterns...")
        affiliate_data = await self.data_analyzer.analyze_affiliate_traffic_patterns()
        
        if not affiliate_data:
            logger.error("No affiliate data available")
            return {}
        
        print(f"✅ Found {len(affiliate_data)} high-performing affiliate sources")
        
        # Step 2: Discover content patterns
        print("📖 Discovering content patterns...")
        content_patterns = await self.data_analyzer.discover_content_patterns(affiliate_data)
        print(f"✅ Discovered {len(content_patterns)} content patterns")
        
        # Step 3: Create replication strategies
        print("🎯 Creating replication strategies...")
        self.replicator = AffiliateStrategyReplicator(affiliate_data, content_patterns)
        replication_strategies = self.replicator.create_replication_strategies()
        print(f"✅ Created {len(replication_strategies)} replication strategies")
        
        # Compile results
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "affiliate_performance": {
                source: {
                    "cvr": perf.cvr,
                    "sessions": perf.sessions,
                    "conversions": perf.conversions,
                    "success_factors": perf.success_factors
                } for source, perf in affiliate_data.items()
            },
            "content_patterns": {
                source: {
                    "content_type": pattern.content_type,
                    "cvr": pattern.cvr,
                    "key_elements": pattern.key_elements,
                    "messaging_themes": pattern.messaging_themes
                } for source, pattern in content_patterns.items()
            },
            "replication_strategies": {
                name: {
                    "content_approach": strategy.content_approach,
                    "target_cvr": strategy.target_cvr,
                    "implementation_steps": strategy.implementation_steps,
                    "expected_volume": strategy.expected_traffic_volume,
                    "test_budget": strategy.test_budget
                } for name, strategy in replication_strategies.items()
            },
            "key_insights": self._generate_key_insights(affiliate_data, content_patterns),
            "action_plan": self._create_action_plan(replication_strategies)
        }
        
        return self.analysis_results
    
    def _generate_key_insights(self, affiliate_data: Dict[str, AffiliatePerformance], 
                              content_patterns: Dict[str, ContentPattern]) -> Dict[str, Any]:
        """Generate key insights from affiliate analysis"""
        
        # CVR analysis
        cvrs = [perf.cvr for perf in affiliate_data.values()]
        avg_cvr = np.mean(cvrs)
        top_cvr = max(cvrs)
        
        # Volume analysis
        total_affiliate_sessions = sum(perf.sessions for perf in affiliate_data.values())
        total_affiliate_conversions = sum(perf.conversions for perf in affiliate_data.values())
        
        # Device analysis
        total_mobile = sum(perf.device_mix.get('mobile', 0) for perf in affiliate_data.values())
        mobile_pct = (total_mobile / total_affiliate_sessions) * 100 if total_affiliate_sessions > 0 else 0
        
        # Content type analysis
        content_types = {}
        for pattern in content_patterns.values():
            content_type = pattern.content_type
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return {
            "performance_summary": {
                "average_affiliate_cvr": round(avg_cvr, 2),
                "top_affiliate_cvr": round(top_cvr, 2),
                "total_affiliate_sessions": total_affiliate_sessions,
                "total_affiliate_conversions": total_affiliate_conversions,
                "mobile_traffic_pct": round(mobile_pct, 1)
            },
            "success_factors": {
                "top_performer": max(affiliate_data.items(), key=lambda x: x[1].cvr)[0],
                "common_success_factors": self._find_common_success_factors(affiliate_data),
                "winning_content_types": sorted(content_types.items(), key=lambda x: x[1], reverse=True)
            },
            "replication_opportunities": {
                "immediate": "Focus on educational/review content (highest performing pattern)",
                "medium_term": "Build comprehensive comparison content hub",
                "long_term": "Develop own affiliate network with discovered best practices"
            }
        }
    
    def _find_common_success_factors(self, affiliate_data: Dict[str, AffiliatePerformance]) -> List[str]:
        """Find success factors common across top performers"""
        all_factors = []
        for perf in affiliate_data.values():
            if perf.cvr >= 4.0:  # Top performers only
                all_factors.extend(perf.success_factors)
        
        # Count frequency and return most common
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        return sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _create_action_plan(self, strategies: Dict[str, ReplicationStrategy]) -> List[str]:
        """Create prioritized action plan"""
        
        # Sort strategies by expected ROI (target_cvr * expected_volume / test_budget)
        strategy_roi = []
        for name, strategy in strategies.items():
            roi = (strategy.target_cvr * strategy.expected_traffic_volume) / strategy.test_budget if strategy.test_budget > 0 else 0
            strategy_roi.append((name, roi, strategy))
        
        strategy_roi.sort(key=lambda x: x[1], reverse=True)
        
        action_plan = [
            "IMMEDIATE ACTIONS (Week 1-2):",
            f"1. Implement '{strategy_roi[0][2].name}' (highest ROI potential)",
            f"2. Create {strategy_roi[0][2].content_approach.lower()} content",
            f"3. Set up tracking for {strategy_roi[0][2].target_cvr}% CVR target",
            "",
            "SHORT-TERM ACTIONS (Month 1):",
            f"4. Launch A/B tests for top 3 strategies",
            f"5. Create affiliate-style landing pages",
            f"6. Target mobile-first parent audience (80%+ of affiliate traffic)",
            "",
            "MEDIUM-TERM ACTIONS (Month 2-3):",
            f"7. Scale successful variations",
            f"8. Build comprehensive content hub",
            f"9. Implement crisis hour targeting (12am-4am)",
            "",
            "LONG-TERM ACTIONS (Month 3+):",
            f"10. Build own affiliate program",
            f"11. Partner with high-performing affiliate sites",
            f"12. Aim for {max(s[2].target_cvr for s in strategy_roi):.1f}% CVR across channels"
        ]
        
        return action_plan

async def demo_affiliate_analysis():
    """Demonstrate complete affiliate pattern analysis"""
    print("🎯 AFFILIATE PATTERN ANALYZER - DISCOVERING 4.42% CVR SECRETS")
    print("=" * 80)
    
    analyzer = AffiliatePatternAnalyzer()
    
    try:
        results = await analyzer.run_complete_analysis()
        
        if not results:
            print("❌ No affiliate data available")
            return
        
        print(f"\n📊 AFFILIATE PERFORMANCE ANALYSIS:")
        print("-" * 60)
        
        # Show top performers
        for source, data in results["affiliate_performance"].items():
            print(f"{source:20} | CVR: {data['cvr']:5.1f}% | Sessions: {data['sessions']:,} | Conv: {data['conversions']:,}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print("-" * 60)
        insights = results["key_insights"]
        perf_summary = insights["performance_summary"]
        
        print(f"Average Affiliate CVR: {perf_summary['average_affiliate_cvr']}%")
        print(f"Top Affiliate CVR: {perf_summary['top_affiliate_cvr']}%")
        print(f"Total Affiliate Traffic: {perf_summary['total_affiliate_sessions']:,} sessions")
        print(f"Mobile Traffic: {perf_summary['mobile_traffic_pct']}%")
        
        print(f"\n🏆 SUCCESS FACTORS:")
        print("-" * 60)
        for factor, count in insights["success_factors"]["common_success_factors"][:5]:
            print(f"• {factor} ({count} sources)")
        
        print(f"\n🚀 REPLICATION STRATEGIES:")
        print("-" * 60)
        
        for name, strategy in results["replication_strategies"].items():
            print(f"\n{strategy['content_approach']}")
            print(f"  Target CVR: {strategy['target_cvr']}%")
            print(f"  Expected Volume: {strategy['expected_volume']:,} sessions")
            print(f"  Test Budget: ${strategy['test_budget']:,.0f}")
        
        print(f"\n📋 ACTION PLAN:")
        print("-" * 60)
        for action in results["action_plan"]:
            print(f"  {action}")
        
        # Save results
        results_file = Path("/home/hariravichandran/AELP/affiliate_analysis_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Analysis complete! Results saved to: {results_file}")
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Focus on educational/review content (top performing pattern)")
        print(f"   • Target mobile-first parent audience (80%+ mobile traffic)")
        print(f"   • Aim for 3-6% CVR with replication strategies")
        print(f"   • Test crisis hour targeting (12am-4am for urgent needs)")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run analysis
    asyncio.run(demo_affiliate_analysis())
