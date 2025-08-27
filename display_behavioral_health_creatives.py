#!/usr/bin/env python3
"""
BEHAVIORAL HEALTH DISPLAY CREATIVES
Fix the creative relevance from 0% to behavioral health focused

Current problem: Generic "parental controls" messaging
Target: Behavioral health crisis intervention messaging for concerned parents

Key insight from GA4: "Parental Controls App" gets 5.16% CVR
But display ads are probably saying generic things that don't convert.

Need: Crisis-focused, behavioral health messaging that speaks to worried parents
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DisplayCreative:
    """Display creative specification"""
    headline: str
    description: str
    image_concept: str
    cta: str
    emotional_tone: str
    target_audience: str
    messaging_theme: str

@dataclass
class ResponsiveDisplayAd:
    """Google Responsive Display Ad specification"""
    headlines: List[str]
    descriptions: List[str] 
    images: List[str]
    logos: List[str]
    call_to_action: str
    messaging_theme: str

class BehavioralHealthCreativeBuilder:
    """Build behavioral health focused display creatives"""
    
    def __init__(self):
        self.crisis_triggers = [
            "teen depression signs",
            "child behavior changes", 
            "teenage mental health",
            "teen anxiety symptoms",
            "behavioral warning signs",
            "digital wellness concerns",
            "social media impact"
        ]
        
        self.parent_concerns = [
            "Is my teen okay?",
            "Something's changed",
            "I'm worried about",
            "Warning signs I missed",
            "How do I help",
            "Before it's too late",
            "Early intervention"
        ]
        
    def create_crisis_intervention_creatives(self) -> List[DisplayCreative]:
        """Create crisis-focused display creatives"""
        
        creatives = []
        
        # Creative 1: Direct Crisis Question
        creatives.append(DisplayCreative(
            headline="Is Your Teen Really Okay?",
            description="AI detects mood changes before crisis hits. Balance monitors behavioral patterns parents often miss.",
            image_concept="concerned_parent_looking_at_phone",
            cta="Check Teen's Digital Wellness",
            emotional_tone="concerned_urgency", 
            target_audience="worried_parents",
            messaging_theme="crisis_prevention"
        ))
        
        # Creative 2: Behavioral Change Detection
        creatives.append(DisplayCreative(
            headline="Notice Changes in Your Teen?",
            description="Behavioral changes can signal depression or anxiety. Balance helps you understand what's happening.",
            image_concept="teen_behavior_comparison_chart",
            cta="See What Balance Detects",
            emotional_tone="analytical_concern",
            target_audience="observant_parents", 
            messaging_theme="behavioral_analysis"
        ))
        
        # Creative 3: Early Warning System
        creatives.append(DisplayCreative(
            headline="Catch Warning Signs Early",
            description="73% of teens hide mental health struggles. Balance uses AI to identify concerning patterns.",
            image_concept="balance_dashboard_alerts",
            cta="Get Early Warnings",
            emotional_tone="proactive_protection",
            target_audience="preventive_parents",
            messaging_theme="early_detection"
        ))
        
        # Creative 4: Crisis Statistics Hook
        creatives.append(DisplayCreative(
            headline="1 in 5 Teens Has Depression",
            description="Most parents don't know until it's severe. Balance helps identify changes before crisis.",
            image_concept="teen_depression_statistics_visual",
            cta="Protect Your Teen",
            emotional_tone="sobering_reality",
            target_audience="statistics_motivated",
            messaging_theme="awareness_education"
        ))
        
        # Creative 5: Time-Sensitive Intervention
        creatives.append(DisplayCreative(
            headline="Don't Wait for Crisis",
            description="Earlier intervention = better outcomes. Balance monitors teen digital wellness continuously.",
            image_concept="timeline_early_vs_late_intervention",
            cta="Start Monitoring Now", 
            emotional_tone="urgent_prevention",
            target_audience="action_oriented_parents",
            messaging_theme="intervention_timing"
        ))
        
        # Creative 6: AI-Powered Understanding
        creatives.append(DisplayCreative(
            headline="AI That Understands Teen Behavior",
            description="Beyond screen time - Balance analyzes communication patterns, mood indicators, and social changes.",
            image_concept="ai_brain_analyzing_patterns",
            cta="See How It Works",
            emotional_tone="technological_trust",
            target_audience="tech_savvy_parents",
            messaging_theme="ai_capability"
        ))
        
        return creatives
    
    def create_responsive_display_ads(self) -> List[ResponsiveDisplayAd]:
        """Create Google Responsive Display Ads"""
        
        responsive_ads = []
        
        # Responsive Ad 1: Crisis Prevention Focus
        responsive_ads.append(ResponsiveDisplayAd(
            headlines=[
                "Is Your Teen Really Okay?",
                "Notice Changes in Your Teen?", 
                "Catch Warning Signs Early",
                "Don't Wait for Crisis",
                "AI Detects Mood Changes"
            ],
            descriptions=[
                "Balance uses AI to detect depression and anxiety signs before crisis hits.",
                "73% of teens hide mental health struggles. Know when your teen needs help.",
                "Behavioral changes can signal mental health issues. Get early alerts.",
                "Earlier intervention = better outcomes. Monitor teen wellness continuously."
            ],
            images=[
                "concerned_parent_teen_conversation.jpg",
                "balance_dashboard_mental_health_alerts.png", 
                "teen_behavioral_change_timeline.jpg",
                "balance_ai_pattern_detection.png",
                "happy_family_after_help.jpg"
            ],
            logos=["aura_balance_logo.png"],
            call_to_action="Learn More",
            messaging_theme="crisis_prevention"
        ))
        
        # Responsive Ad 2: Behavioral Analysis Focus  
        responsive_ads.append(ResponsiveDisplayAd(
            headlines=[
                "Understand Your Teen's Digital Life",
                "Beyond Screen Time Monitoring", 
                "AI-Powered Behavioral Insights",
                "See What You're Missing",
                "Teen Mental Health Monitoring"
            ],
            descriptions=[
                "Balance analyzes communication patterns, social changes, and mood indicators.",
                "Get insights into your teen's digital wellness and behavioral patterns.",
                "AI technology helps parents understand complex teenage behavior changes.",
                "Comprehensive monitoring goes beyond apps - tracks behavioral health."
            ],
            images=[
                "balance_behavioral_dashboard.png",
                "teen_digital_life_analysis.jpg",
                "parent_understanding_teen_better.jpg", 
                "balance_ai_insights_visualization.png",
                "behavioral_health_monitoring.jpg"
            ],
            logos=["aura_balance_logo.png"],
            call_to_action="Get Free Trial",
            messaging_theme="behavioral_analysis"
        ))
        
        # Responsive Ad 3: Parent Empowerment Focus
        responsive_ads.append(ResponsiveDisplayAd(
            headlines=[
                "Help Your Teen Before Crisis", 
                "Early Intervention Saves Lives",
                "Be the Parent Your Teen Needs",
                "Know When to Step In",
                "Proactive Parent Protection"
            ],
            descriptions=[
                "Balance gives parents the information they need to help their teen thrive.",
                "Don't wait for a crisis. Early support leads to better mental health outcomes.",
                "Equipped parents make the difference. Get the tools to help your teen.",
                "Balance helps you know when your teen needs extra support and care."
            ],
            images=[
                "empowered_parent_helping_teen.jpg",
                "balance_parent_guidance_dashboard.png",
                "successful_teen_parent_relationship.jpg",
                "balance_intervention_timeline.png", 
                "mental_health_support_family.jpg"
            ],
            logos=["aura_balance_logo.png"],
            call_to_action="Start Free Trial",
            messaging_theme="parent_empowerment"
        ))
        
        return responsive_ads
    
    def create_native_ad_concepts(self) -> List[Dict]:
        """Create native ad concepts for editorial placements"""
        
        native_ads = [
            {
                'headline': "73% of Teens Hide Mental Health Struggles - Here's How Parents Can Tell",
                'description': "New AI technology helps parents identify behavioral changes that may indicate depression or anxiety in teenagers.",
                'article_angle': "educational_guide",
                'image_concept': "teen_looking_at_phone_concerned_parent_background",
                'cta': "Read Parent Guide",
                'placement_context': "parenting_articles"
            },
            {
                'headline': "The Warning Signs of Teen Depression Parents Often Miss",
                'description': "Behavioral changes in digital communication can signal mental health issues. Learn what to watch for.",
                'article_angle': "warning_signs_checklist", 
                'image_concept': "parent_child_conversation_checklist",
                'cta': "Download Checklist",
                'placement_context': "mental_health_content"
            },
            {
                'headline': "Why Screen Time Monitoring Isn't Enough for Teen Mental Health",
                'description': "Experts say parents need to look beyond screen time to understand teen behavioral health patterns.",
                'article_angle': "expert_opinion",
                'image_concept': "expert_parent_interview_setup",
                'cta': "Learn More",
                'placement_context': "parenting_advice"
            },
            {
                'headline': "Teen Suicide Rates Rising: How AI Helps Parents Intervene Earlier",
                'description': "Crisis intervention technology gives parents tools to identify concerning behavioral changes.",
                'article_angle': "crisis_prevention_technology",
                'image_concept': "balance_dashboard_crisis_alerts",
                'cta': "See How It Works", 
                'placement_context': "mental_health_news"
            }
        ]
        
        return native_ads
    
    def create_video_ad_concepts(self) -> List[Dict]:
        """Create video ad concepts for YouTube/social"""
        
        video_concepts = [
            {
                'concept': 'day_in_life_behavior_change',
                'duration': 30,
                'narrative': 'Show same teen over 3 months - subtle behavior changes that Balance detects',
                'hook': 'Did you notice what changed?',
                'key_message': 'Balance catches what parents miss',
                'emotional_arc': 'concern → awareness → relief',
                'cta': 'Protect your teen'
            },
            {
                'concept': 'parent_testimonial_crisis_averted',
                'duration': 60, 
                'narrative': 'Real parent shares how Balance helped identify teen depression early',
                'hook': 'I thought she was just being moody',
                'key_message': 'Early detection saves lives',
                'emotional_arc': 'worry → discovery → gratitude',
                'cta': 'Get early warnings'
            },
            {
                'concept': 'ai_explanation_simplified',
                'duration': 45,
                'narrative': 'Animated explanation of how Balance AI detects behavioral patterns',
                'hook': 'How does AI understand teen behavior?',
                'key_message': 'Technology that helps parents help teens',
                'emotional_arc': 'curiosity → understanding → confidence', 
                'cta': 'See how it works'
            }
        ]
        
        return video_concepts
    
    def optimize_for_mobile(self, creative_specs: Dict) -> Dict:
        """Optimize creatives for mobile viewing"""
        
        mobile_optimizations = {
            'headline_length': 30,  # Max characters for mobile
            'description_length': 90,  # Max characters for mobile
            'image_requirements': {
                'aspect_ratios': ['1:1', '1.91:1', '4:5'],
                'min_resolution': '600x600',
                'text_overlay': '<20% of image',
                'mobile_friendly_fonts': True
            },
            'cta_optimization': {
                'thumb_friendly': True,
                'high_contrast': True,
                'action_oriented': True
            }
        }
        
        return mobile_optimizations
    
    def generate_creative_brief(self) -> Dict:
        """Generate complete creative brief"""
        
        print("\n" + "="*80)
        print("🎨 BEHAVIORAL HEALTH DISPLAY CREATIVE BRIEF")
        print("="*80)
        
        crisis_creatives = self.create_crisis_intervention_creatives()
        responsive_ads = self.create_responsive_display_ads()
        native_ads = self.create_native_ad_concepts()
        video_concepts = self.create_video_ad_concepts()
        
        brief = {
            'overview': {
                'objective': 'Replace generic parental controls messaging with behavioral health crisis prevention',
                'target_cvr_improvement': '0.01% → 1.0% (100x improvement)',
                'key_insight': 'Parents respond to specific behavioral health concerns, not generic monitoring',
                'primary_audience': 'Parents concerned about teen mental health'
            },
            'messaging_strategy': {
                'primary_theme': 'Crisis prevention through early behavioral detection',
                'emotional_drivers': ['parental_concern', 'early_intervention', 'prevention'],
                'avoid': ['generic_monitoring', 'surveillance_language', 'parental_control_focus'],
                'emphasize': ['behavioral_health', 'crisis_prevention', 'AI_insights']
            },
            'creative_portfolio': {
                'crisis_intervention_creatives': len(crisis_creatives),
                'responsive_display_ads': len(responsive_ads), 
                'native_ad_concepts': len(native_ads),
                'video_concepts': len(video_concepts)
            },
            'mobile_optimization': self.optimize_for_mobile({}),
            'testing_framework': {
                'a_b_tests': [
                    'Crisis messaging vs. educational messaging',
                    'Statistics-driven vs. emotion-driven headlines',
                    'AI capability vs. parent empowerment focus'
                ],
                'success_metrics': ['CVR', 'CTR', 'engagement_time', 'quality_score']
            }
        }
        
        self.print_creative_analysis(crisis_creatives, responsive_ads, native_ads)
        self.save_creative_brief(brief, crisis_creatives, responsive_ads, native_ads, video_concepts)
        
        return brief
    
    def print_creative_analysis(self, crisis_creatives, responsive_ads, native_ads):
        """Print analysis of created creatives"""
        
        print(f"\n📊 CREATIVE PORTFOLIO ANALYSIS:")
        print(f"   Crisis intervention creatives: {len(crisis_creatives)}")
        print(f"   Responsive display ads: {len(responsive_ads)}")
        print(f"   Native ad concepts: {len(native_ads)}")
        
        print(f"\n🎯 TOP CRISIS INTERVENTION HEADLINES:")
        for creative in crisis_creatives[:3]:
            print(f"   • \"{creative.headline}\" ({creative.messaging_theme})")
        
        print(f"\n📱 RESPONSIVE AD THEMES:")
        for ad in responsive_ads:
            print(f"   • {ad.messaging_theme}: {len(ad.headlines)} headlines, {len(ad.descriptions)} descriptions")
        
        print(f"\n📰 NATIVE AD APPROACHES:")
        for native in native_ads[:2]:
            print(f"   • \"{native['headline']}\" ({native['article_angle']})")
    
    def save_creative_brief(self, brief, crisis_creatives, responsive_ads, native_ads, video_concepts):
        """Save complete creative brief and assets"""
        
        # Complete creative package
        creative_package = {
            'creative_brief': brief,
            'crisis_intervention_creatives': [
                {
                    'headline': c.headline,
                    'description': c.description, 
                    'image_concept': c.image_concept,
                    'cta': c.cta,
                    'emotional_tone': c.emotional_tone,
                    'target_audience': c.target_audience,
                    'messaging_theme': c.messaging_theme
                } for c in crisis_creatives
            ],
            'responsive_display_ads': [
                {
                    'headlines': ad.headlines,
                    'descriptions': ad.descriptions,
                    'images': ad.images,
                    'logos': ad.logos,
                    'call_to_action': ad.call_to_action,
                    'messaging_theme': ad.messaging_theme
                } for ad in responsive_ads
            ],
            'native_ad_concepts': native_ads,
            'video_concepts': video_concepts,
            'created_date': datetime.now().isoformat()
        }
        
        # Save complete package
        with open('/home/hariravichandran/AELP/display_behavioral_health_creatives.json', 'w') as f:
            json.dump(creative_package, f, indent=2)
        
        # Create implementation-ready files
        
        # CSV for Google Ads responsive display ad upload
        rda_data = []
        for i, ad in enumerate(responsive_ads):
            for j, headline in enumerate(ad.headlines):
                for k, desc in enumerate(ad.descriptions):
                    rda_data.append({
                        'Ad_Group': f'Behavioral_Health_RDA_{i+1}',
                        'Headline': headline,
                        'Description': desc,
                        'CTA': ad.call_to_action,
                        'Theme': ad.messaging_theme
                    })
        
        import pandas as pd
        pd.DataFrame(rda_data).to_csv('/home/hariravichandran/AELP/responsive_display_ads_upload.csv', index=False)
        
        print(f"\n💾 Files saved:")
        print(f"   • display_behavioral_health_creatives.json - Complete creative package")
        print(f"   • responsive_display_ads_upload.csv - For Google Ads upload")

if __name__ == "__main__":
    creative_builder = BehavioralHealthCreativeBuilder()
    brief = creative_builder.generate_creative_brief()
    
    print(f"\n" + "="*80)
    print("🎯 NEXT STEPS - IMPLEMENT BEHAVIORAL HEALTH CREATIVES")
    print("="*80)
    print("1. Review and approve creative concepts")
    print("2. Produce images and videos based on concepts") 
    print("3. Upload responsive_display_ads_upload.csv to Google Ads")
    print("4. Set up A/B tests for different messaging themes")
    print("5. Monitor CTR and CVR improvement")
    print("6. Target: CVR improvement from 0.01% to 0.5%+ with new creatives")