"""
Creative Content Analyzer
Analyzes actual ad creative content including headlines, CTAs, images, colors, and emotional tone.
Extracts features that influence creative performance for GAELP RL bidding decisions.

CRITICAL: This module extracts and analyzes ACTUAL creative content, not just IDs.
"""

import json
import re
import colorsys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
import hashlib
import logging

# NLP imports for text analysis
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk_available = True
except ImportError:
    nltk_available = False

# Image analysis imports
try:
    from PIL import Image, ImageDraw
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    image_analysis_available = True
except ImportError:
    image_analysis_available = False

logger = logging.getLogger(__name__)

@dataclass
class ContentFeatures:
    """Extracted features from creative content"""
    # Text features
    headline_length: int = 0
    headline_sentiment: float = 0.0  # -1 to 1
    headline_urgency: float = 0.0    # 0 to 1
    headline_emotion: str = "neutral"  # angry, fear, joy, sadness, surprise, trust
    
    cta_strength: float = 0.0        # 0 to 1 based on action words
    cta_urgency: float = 0.0         # 0 to 1 based on urgency words
    
    description_complexity: float = 0.0  # 0 to 1, based on readability
    description_benefits: int = 0     # Number of benefit mentions
    description_features: int = 0     # Number of feature mentions
    
    # Visual features (from image URL analysis or metadata)
    primary_color: str = "unknown"   # hex color or color name
    color_temperature: str = "neutral"  # warm, cool, neutral
    visual_style: str = "unknown"    # clinical, lifestyle, emotional, comparison
    image_category: str = "unknown"   # people, product, abstract, text
    
    # Content structure
    uses_numbers: bool = False       # Contains specific numbers/stats
    uses_social_proof: bool = False  # Contains testimonials/reviews
    uses_urgency: bool = False       # Contains time-sensitive language
    uses_authority: bool = False     # Contains expert/official references
    
    # Message framing
    message_frame: str = "benefit"   # fear, benefit, social_proof, authority, urgency
    target_pain_point: str = "unknown"  # crisis, prevention, comparison, value
    
    # Performance predictors
    predicted_ctr: float = 0.0       # Predicted based on content features
    predicted_cvr: float = 0.0       # Predicted conversion rate
    fatigue_resistance: float = 0.0  # How resistant to ad fatigue

@dataclass
class CreativeContent:
    """Complete creative content with extracted features"""
    creative_id: str
    headline: str
    description: str
    cta: str
    image_url: str = ""
    
    # Performance data
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    
    # Extracted features
    content_features: ContentFeatures = field(default_factory=ContentFeatures)
    
    # A/B test tracking
    variant_group: str = "control"
    test_name: str = ""
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_ctr(self) -> float:
        return self.clicks / max(1, self.impressions)
    
    def get_cvr(self) -> float:
        return self.conversions / max(1, self.clicks)

class CreativeContentAnalyzer:
    """
    Analyzes actual creative content to extract performance-influencing features
    """
    
    def __init__(self):
        self.creatives: Dict[str, CreativeContent] = {}
        self.feature_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Initialize NLP components
        if nltk_available:
            self._ensure_nltk_data()
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
        else:
            logger.warning("NLTK not available - text analysis will be limited")
            self.sentiment_analyzer = None
            self.stop_words = set()
        
        # Urgency keywords
        self.urgency_words = {
            'now', 'today', 'immediately', 'urgent', 'asap', 'hurry', 'quick',
            'limited', 'expires', 'deadline', 'ending', 'final', 'last chance',
            'act fast', 'don\'t wait', 'while supplies last', 'time-sensitive'
        }
        
        # Authority keywords
        self.authority_words = {
            'expert', 'doctor', 'certified', 'approved', 'recommended',
            'clinical', 'medical', 'scientific', 'research', 'study',
            'proven', 'verified', 'official', 'licensed', 'endorsed'
        }
        
        # CTA strength words
        self.strong_cta_words = {
            'get', 'start', 'try', 'download', 'join', 'discover', 'learn',
            'protect', 'save', 'improve', 'transform', 'achieve', 'unlock'
        }
        
        # Emotion keywords
        self.emotion_keywords = {
            'anger': ['angry', 'furious', 'outraged', 'frustrated', 'mad'],
            'fear': ['scared', 'worried', 'afraid', 'anxious', 'concerned', 'dangerous'],
            'joy': ['happy', 'excited', 'thrilled', 'delighted', 'amazing'],
            'sadness': ['sad', 'depressed', 'disappointed', 'heartbroken'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected'],
            'trust': ['reliable', 'trusted', 'secure', 'safe', 'guaranteed']
        }
        
        # Color mappings for basic color extraction
        self.color_keywords = {
            'red': ['#ff0000', '#ff4444', '#cc0000', 'red', 'crimson'],
            'blue': ['#0000ff', '#4444ff', '#0066cc', 'blue', 'navy'],
            'green': ['#00ff00', '#44ff44', '#00cc66', 'green', 'lime'],
            'orange': ['#ff8800', '#ffaa00', 'orange', 'amber'],
            'purple': ['#8800ff', '#aa44ff', 'purple', 'violet'],
            'yellow': ['#ffff00', '#ffcc00', 'yellow', 'gold'],
            'white': ['#ffffff', '#f5f5f5', 'white', 'cream'],
            'black': ['#000000', '#333333', 'black', 'dark']
        }
    
    def _ensure_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            try:
                nltk.download('vader_lexicon', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK data: {e}")
    
    def analyze_creative(self, creative: CreativeContent) -> ContentFeatures:
        """
        Main method to analyze creative content and extract features
        """
        features = ContentFeatures()
        
        # Analyze text components
        features = self._analyze_headline(creative.headline, features)
        features = self._analyze_description(creative.description, features)
        features = self._analyze_cta(creative.cta, features)
        
        # Analyze visual components
        features = self._analyze_visual_elements(creative.image_url, features)
        
        # Analyze content structure
        features = self._analyze_content_structure(creative, features)
        
        # Predict performance
        features = self._predict_performance(features)
        
        # Store features in creative
        creative.content_features = features
        
        # Update feature performance tracking
        self._update_feature_tracking(creative)
        
        return features
    
    def _analyze_headline(self, headline: str, features: ContentFeatures) -> ContentFeatures:
        """Analyze headline content"""
        if not headline:
            return features
        
        features.headline_length = len(headline)
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(headline.lower())
            features.headline_sentiment = sentiment_scores['compound']
        
        # Urgency detection
        headline_lower = headline.lower()
        urgency_matches = sum(1 for word in self.urgency_words if word in headline_lower)
        features.headline_urgency = min(urgency_matches / 3.0, 1.0)  # Cap at 1.0
        
        # Emotion detection
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in headline_lower)
            emotion_scores[emotion] = score
        
        if emotion_scores:
            features.headline_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # Check for numbers (social proof indicator)
        features.uses_numbers = bool(re.search(r'\d+[%]?', headline))
        
        return features
    
    def _analyze_description(self, description: str, features: ContentFeatures) -> ContentFeatures:
        """Analyze description/body text"""
        if not description:
            return features
        
        desc_lower = description.lower()
        
        # Complexity analysis (based on sentence length and word complexity)
        sentences = description.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        features.description_complexity = min(avg_sentence_length / 20.0, 1.0)
        
        # Benefit vs feature counting
        benefit_words = ['save', 'protect', 'improve', 'reduce', 'increase', 'better', 'easier']
        feature_words = ['includes', 'features', 'has', 'provides', 'offers', 'with']
        
        features.description_benefits = sum(1 for word in benefit_words if word in desc_lower)
        features.description_features = sum(1 for word in feature_words if word in desc_lower)
        
        # Social proof detection
        social_proof_indicators = ['customers', 'users', 'parents', 'reviews', 'rated', 'testimonial']
        features.uses_social_proof = any(indicator in desc_lower for indicator in social_proof_indicators)
        
        # Authority detection
        features.uses_authority = any(word in desc_lower for word in self.authority_words)
        
        return features
    
    def _analyze_cta(self, cta: str, features: ContentFeatures) -> ContentFeatures:
        """Analyze call-to-action strength and urgency"""
        if not cta:
            return features
        
        cta_lower = cta.lower()
        
        # CTA strength based on action words
        strong_words = sum(1 for word in self.strong_cta_words if word in cta_lower)
        features.cta_strength = min(strong_words / 2.0, 1.0)  # Cap at 1.0
        
        # CTA urgency
        urgency_matches = sum(1 for word in self.urgency_words if word in cta_lower)
        features.cta_urgency = min(urgency_matches / 2.0, 1.0)
        
        # Check if CTA uses urgency
        features.uses_urgency = features.cta_urgency > 0 or features.headline_urgency > 0
        
        return features
    
    def _analyze_visual_elements(self, image_url: str, features: ContentFeatures) -> ContentFeatures:
        """Analyze visual elements from image URL or metadata"""
        if not image_url:
            return features
        
        # Extract color information from filename or URL patterns
        url_lower = image_url.lower()
        
        # Basic color detection from URL/filename
        for color_name, color_variations in self.color_keywords.items():
            if any(variation in url_lower for variation in color_variations):
                features.primary_color = color_name
                break
        
        # Color temperature inference
        warm_colors = ['red', 'orange', 'yellow']
        cool_colors = ['blue', 'green', 'purple']
        if features.primary_color in warm_colors:
            features.color_temperature = "warm"
        elif features.primary_color in cool_colors:
            features.color_temperature = "cool"
        
        # Visual style detection from URL patterns
        if any(keyword in url_lower for keyword in ['lifestyle', 'family', 'home']):
            features.visual_style = "lifestyle"
        elif any(keyword in url_lower for keyword in ['clinical', 'medical', 'chart']):
            features.visual_style = "clinical"
        elif any(keyword in url_lower for keyword in ['comparison', 'vs', 'chart']):
            features.visual_style = "comparison"
        elif any(keyword in url_lower for keyword in ['emotional', 'crying', 'worried']):
            features.visual_style = "emotional"
        
        # Image category detection
        if any(keyword in url_lower for keyword in ['people', 'person', 'family', 'teen']):
            features.image_category = "people"
        elif any(keyword in url_lower for keyword in ['product', 'app', 'screen']):
            features.image_category = "product"
        elif any(keyword in url_lower for keyword in ['text', 'quote', 'banner']):
            features.image_category = "text"
        
        return features
    
    def _analyze_content_structure(self, creative: CreativeContent, features: ContentFeatures) -> ContentFeatures:
        """Analyze overall content structure and messaging"""
        all_text = f"{creative.headline} {creative.description} {creative.cta}".lower()
        
        # Message framing analysis
        fear_indicators = ['danger', 'risk', 'threat', 'harm', 'crisis', 'problem']
        benefit_indicators = ['protect', 'improve', 'save', 'better', 'solution']
        authority_indicators = list(self.authority_words)
        social_proof_indicators = ['parents', 'customers', 'users', 'rated']
        urgency_indicators = list(self.urgency_words)
        
        frame_scores = {
            'fear': sum(1 for indicator in fear_indicators if indicator in all_text),
            'benefit': sum(1 for indicator in benefit_indicators if indicator in all_text),
            'authority': sum(1 for indicator in authority_indicators if indicator in all_text),
            'social_proof': sum(1 for indicator in social_proof_indicators if indicator in all_text),
            'urgency': sum(1 for indicator in urgency_indicators if indicator in all_text)
        }
        
        features.message_frame = max(frame_scores, key=frame_scores.get) if frame_scores else "benefit"
        
        # Target pain point analysis
        if any(word in all_text for word in ['crisis', 'emergency', 'help']):
            features.target_pain_point = "crisis"
        elif any(word in all_text for word in ['prevent', 'early', 'detect']):
            features.target_pain_point = "prevention"
        elif any(word in all_text for word in ['compare', 'vs', 'better']):
            features.target_pain_point = "comparison"
        elif any(word in all_text for word in ['free', 'save', 'discount', 'price']):
            features.target_pain_point = "value"
        
        return features
    
    def _predict_performance(self, features: ContentFeatures) -> ContentFeatures:
        """Predict creative performance based on extracted features"""
        
        # CTR prediction based on content features
        ctr_factors = []
        
        # Headline factors
        if 40 <= features.headline_length <= 60:  # Optimal headline length
            ctr_factors.append(0.15)
        if features.headline_urgency > 0.5:
            ctr_factors.append(0.10)
        if features.uses_numbers:
            ctr_factors.append(0.08)
        
        # CTA factors
        if features.cta_strength > 0.5:
            ctr_factors.append(0.12)
        
        # Visual factors
        if features.visual_style == "emotional":
            ctr_factors.append(0.10)
        if features.color_temperature == "warm":
            ctr_factors.append(0.05)
        
        # Message framing factors
        if features.message_frame == "urgency":
            ctr_factors.append(0.08)
        elif features.message_frame == "fear":
            ctr_factors.append(0.06)
        
        base_ctr = 0.02  # 2% base CTR
        features.predicted_ctr = base_ctr + sum(ctr_factors)
        
        # CVR prediction
        cvr_factors = []
        
        if features.description_benefits > features.description_features:
            cvr_factors.append(0.015)
        if features.uses_social_proof:
            cvr_factors.append(0.020)
        if features.uses_authority:
            cvr_factors.append(0.018)
        if features.target_pain_point == "crisis":
            cvr_factors.append(0.025)  # High intent
        
        base_cvr = 0.03  # 3% base CVR
        features.predicted_cvr = base_cvr + sum(cvr_factors)
        
        # Fatigue resistance
        # More diverse content elements = higher fatigue resistance
        diversity_elements = [
            features.headline_emotion != "neutral",
            features.uses_numbers,
            features.uses_social_proof,
            features.uses_authority,
            features.visual_style != "unknown",
            features.primary_color != "unknown"
        ]
        
        features.fatigue_resistance = sum(diversity_elements) / len(diversity_elements)
        
        return features
    
    def _update_feature_tracking(self, creative: CreativeContent):
        """Update performance tracking for content features"""
        features = creative.content_features
        creative_key = creative.creative_id
        
        # Track feature performance if we have actual performance data
        if creative.impressions > 0:
            actual_ctr = creative.get_ctr()
            actual_cvr = creative.get_cvr()
            
            # Update feature performance tracking
            feature_keys = [
                f"headline_length_{min(features.headline_length // 10, 10)}0",
                f"sentiment_{features.headline_sentiment:.1f}",
                f"message_frame_{features.message_frame}",
                f"visual_style_{features.visual_style}",
                f"primary_color_{features.primary_color}",
                f"target_pain_point_{features.target_pain_point}"
            ]
            
            for feature_key in feature_keys:
                if feature_key not in self.feature_performance:
                    self.feature_performance[feature_key] = {
                        'impressions': 0, 'clicks': 0, 'conversions': 0
                    }
                
                perf = self.feature_performance[feature_key]
                perf['impressions'] += creative.impressions
                perf['clicks'] += creative.clicks
                perf['conversions'] += creative.conversions
    
    def add_creative(self, creative_id: str, headline: str, description: str, 
                    cta: str, image_url: str = "") -> CreativeContent:
        """Add new creative and analyze its content"""
        creative = CreativeContent(
            creative_id=creative_id,
            headline=headline,
            description=description,
            cta=cta,
            image_url=image_url
        )
        
        # Analyze content immediately
        self.analyze_creative(creative)
        
        # Store creative
        self.creatives[creative_id] = creative
        
        return creative
    
    def update_creative_performance(self, creative_id: str, impressions: int = 0, 
                                  clicks: int = 0, conversions: int = 0):
        """Update creative performance metrics"""
        if creative_id not in self.creatives:
            logger.warning(f"Creative {creative_id} not found")
            return
        
        creative = self.creatives[creative_id]
        creative.impressions += impressions
        creative.clicks += clicks
        creative.conversions += conversions
        
        # Re-analyze to update feature tracking
        self._update_feature_tracking(creative)
    
    def get_content_insights(self, segment: str = None) -> Dict[str, Any]:
        """Get insights about content performance"""
        insights = {
            'total_creatives': len(self.creatives),
            'top_performing_features': {},
            'content_trends': {},
            'performance_by_features': {}
        }
        
        # Analyze feature performance
        feature_ctr = {}
        feature_cvr = {}
        
        for feature_key, perf in self.feature_performance.items():
            if perf['impressions'] > 10:  # Minimum sample size
                ctr = perf['clicks'] / perf['impressions']
                cvr = perf['conversions'] / max(1, perf['clicks'])
                feature_ctr[feature_key] = ctr
                feature_cvr[feature_key] = cvr
        
        # Top performing features by CTR
        insights['top_performing_features']['ctr'] = sorted(
            feature_ctr.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Top performing features by CVR
        insights['top_performing_features']['cvr'] = sorted(
            feature_cvr.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Content trends
        message_frames = Counter(c.content_features.message_frame for c in self.creatives.values())
        visual_styles = Counter(c.content_features.visual_style for c in self.creatives.values())
        color_usage = Counter(c.content_features.primary_color for c in self.creatives.values())
        
        insights['content_trends'] = {
            'message_frames': dict(message_frames.most_common(5)),
            'visual_styles': dict(visual_styles.most_common(5)),
            'color_usage': dict(color_usage.most_common(5))
        }
        
        return insights
    
    def select_optimal_creative_features(self, target_segment: str, 
                                       target_performance: str = "ctr") -> Dict[str, Any]:
        """Select optimal content features for a target segment and performance metric"""
        
        # Get best performing features
        insights = self.get_content_insights(target_segment)
        top_features = insights['top_performing_features'].get(target_performance, [])
        
        if not top_features:
            return self._get_segment_specific_defaults(target_segment)
        
        # Extract recommendations
        recommendations = {
            'headline_length': 50,  # Default
            'message_frame': 'benefit',
            'visual_style': 'lifestyle',
            'primary_color': 'blue',
            'use_urgency': False,
            'use_social_proof': False,
            'use_authority': False
        }
        
        # Update based on top performing features
        for feature, performance in top_features[:5]:  # Top 5 features
            if 'headline_length' in feature:
                length = int(feature.split('_')[-1].replace('0', ''))
                recommendations['headline_length'] = length * 10
            elif 'message_frame' in feature:
                recommendations['message_frame'] = feature.split('_')[-1]
            elif 'visual_style' in feature:
                recommendations['visual_style'] = feature.split('_')[-1]
            elif 'primary_color' in feature:
                recommendations['primary_color'] = feature.split('_')[-1]
        
        # Add usage recommendations based on high-performing creatives
        high_performers = [c for c in self.creatives.values() 
                          if c.get_ctr() > 0.03 and c.impressions > 50]
        
        if high_performers:
            recommendations['use_urgency'] = sum(c.content_features.uses_urgency 
                                               for c in high_performers) / len(high_performers) > 0.5
            recommendations['use_social_proof'] = sum(c.content_features.uses_social_proof 
                                                    for c in high_performers) / len(high_performers) > 0.5
            recommendations['use_authority'] = sum(c.content_features.uses_authority 
                                                 for c in high_performers) / len(high_performers) > 0.5
        
        return recommendations
    
    def _get_default_recommendations(self) -> Dict[str, Any]:
        """Get default creative feature recommendations"""
        return {
            'headline_length': 50,
            'message_frame': 'benefit',
            'visual_style': 'lifestyle',
            'primary_color': 'blue',
            'use_urgency': False,
            'use_social_proof': True,
            'use_authority': False
        }
    
    def _get_segment_specific_defaults(self, segment: str) -> Dict[str, Any]:
        """Get segment-specific default recommendations when no data available"""
        defaults = {
            'crisis_parents': {
                'headline_length': 45,
                'message_frame': 'urgency',
                'visual_style': 'emotional',
                'primary_color': 'red',
                'use_urgency': True,
                'use_social_proof': False,
                'use_authority': True
            },
            'concerned_parents': {
                'headline_length': 55,
                'message_frame': 'benefit',
                'visual_style': 'lifestyle',
                'primary_color': 'blue',
                'use_urgency': False,
                'use_social_proof': True,
                'use_authority': False
            },
            'researching_parent': {
                'headline_length': 60,
                'message_frame': 'authority',
                'visual_style': 'clinical',
                'primary_color': 'blue',
                'use_urgency': False,
                'use_social_proof': False,
                'use_authority': True
            },
            'proactive_parent': {
                'headline_length': 50,
                'message_frame': 'benefit',
                'visual_style': 'lifestyle',
                'primary_color': 'green',
                'use_urgency': False,
                'use_social_proof': True,
                'use_authority': False
            }
        }
        return defaults.get(segment, self._get_default_recommendations())
    
    def evaluate_creative_content(self, creative_id: str) -> Dict[str, Any]:
        """Evaluate specific creative content and provide improvement suggestions"""
        if creative_id not in self.creatives:
            return {'error': f'Creative {creative_id} not found'}
        
        creative = self.creatives[creative_id]
        features = creative.content_features
        
        evaluation = {
            'creative_id': creative_id,
            'headline': creative.headline,
            'current_performance': {
                'ctr': creative.get_ctr(),
                'cvr': creative.get_cvr(),
                'impressions': creative.impressions
            },
            'content_analysis': {
                'headline_length': features.headline_length,
                'sentiment': features.headline_sentiment,
                'urgency_score': features.headline_urgency,
                'message_frame': features.message_frame,
                'visual_style': features.visual_style
            },
            'predicted_performance': {
                'predicted_ctr': features.predicted_ctr,
                'predicted_cvr': features.predicted_cvr,
                'fatigue_resistance': features.fatigue_resistance
            },
            'improvement_suggestions': []
        }
        
        # Generate improvement suggestions
        suggestions = []
        
        if features.headline_length > 70:
            suggestions.append("Consider shortening headline for better mobile readability")
        elif features.headline_length < 30:
            suggestions.append("Consider lengthening headline to include more compelling details")
        
        if features.headline_sentiment < -0.3:
            suggestions.append("Headline is quite negative - consider more positive framing")
        elif features.headline_sentiment < 0.1:
            suggestions.append("Consider adding more emotional appeal to headline")
        
        if features.cta_strength < 0.5:
            suggestions.append("Use stronger action words in CTA (e.g., 'Get', 'Start', 'Discover')")
        
        if not features.uses_social_proof and features.target_pain_point != "crisis":
            suggestions.append("Consider adding social proof elements (testimonials, user counts)")
        
        if features.description_features > features.description_benefits:
            suggestions.append("Focus more on benefits rather than features in description")
        
        evaluation['improvement_suggestions'] = suggestions
        
        return evaluation
    
    def get_creative_differences(self, creative_id1: str, creative_id2: str) -> Dict[str, Any]:
        """Compare two creatives and identify key differences"""
        if creative_id1 not in self.creatives or creative_id2 not in self.creatives:
            return {'error': 'One or both creatives not found'}
        
        c1 = self.creatives[creative_id1]
        c2 = self.creatives[creative_id2]
        
        differences = {
            'performance_difference': {
                'ctr_diff': c1.get_ctr() - c2.get_ctr(),
                'cvr_diff': c1.get_cvr() - c2.get_cvr()
            },
            'content_differences': {},
            'feature_analysis': {}
        }
        
        # Compare content features
        f1, f2 = c1.content_features, c2.content_features
        
        content_diffs = {
            'headline_length_diff': f1.headline_length - f2.headline_length,
            'sentiment_diff': f1.headline_sentiment - f2.headline_sentiment,
            'urgency_diff': f1.headline_urgency - f2.headline_urgency,
            'different_message_frame': f1.message_frame != f2.message_frame,
            'different_visual_style': f1.visual_style != f2.visual_style,
            'different_color': f1.primary_color != f2.primary_color
        }
        
        differences['content_differences'] = content_diffs
        
        # Identify which creative likely performs better and why
        better_creative = creative_id1 if c1.get_ctr() > c2.get_ctr() else creative_id2
        better_features = f1 if c1.get_ctr() > c2.get_ctr() else f2
        
        differences['analysis'] = {
            'better_performing': better_creative,
            'key_advantages': []
        }
        
        if better_features.uses_urgency:
            differences['analysis']['key_advantages'].append('Uses urgency language')
        if better_features.uses_social_proof:
            differences['analysis']['key_advantages'].append('Includes social proof')
        if better_features.cta_strength > 0.7:
            differences['analysis']['key_advantages'].append('Strong call-to-action')
        
        return differences


# Global instance for use across GAELP system
creative_analyzer = CreativeContentAnalyzer()


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CreativeContentAnalyzer()
    
    print("=== Creative Content Analyzer Demo ===\n")
    
    # Add some example creatives
    creative1 = analyzer.add_creative(
        creative_id="crisis_parent_1",
        headline="Is Your Teen in Crisis? Get Help Now - Trusted by 50,000+ Parents",
        description="AI-powered monitoring detects mood changes before they escalate. Clinical psychologists recommend early detection. Our system spots warning signs you might miss.",
        cta="Start Free Trial",
        image_url="/images/emotional_teen_red.jpg"
    )
    
    creative2 = analyzer.add_creative(
        creative_id="researcher_1", 
        headline="Compare AI Safety Solutions Side-by-Side",
        description="Detailed analysis of GAELP vs competitors. Performance benchmarks included. See how our clinical-grade AI stacks up.",
        cta="View Comparison",
        image_url="/images/clinical_comparison_blue.jpg"
    )
    
    # Simulate some performance data
    analyzer.update_creative_performance("crisis_parent_1", impressions=1000, clicks=45, conversions=8)
    analyzer.update_creative_performance("researcher_1", impressions=800, clicks=28, conversions=12)
    
    # Analyze content
    print("Crisis Parent Creative Analysis:")
    analysis1 = analyzer.evaluate_creative_content("crisis_parent_1")
    print(f"  Headline: {analysis1['headline']}")
    print(f"  CTR: {analysis1['current_performance']['ctr']:.3f}")
    print(f"  Predicted CTR: {analysis1['predicted_performance']['predicted_ctr']:.3f}")
    print(f"  Message Frame: {analysis1['content_analysis']['message_frame']}")
    print(f"  Urgency Score: {analysis1['content_analysis']['urgency_score']:.2f}")
    print(f"  Suggestions: {analysis1['improvement_suggestions'][:2]}")
    print()
    
    print("Researcher Creative Analysis:")
    analysis2 = analyzer.evaluate_creative_content("researcher_1")
    print(f"  Headline: {analysis2['headline']}")
    print(f"  CTR: {analysis2['current_performance']['ctr']:.3f}")
    print(f"  Predicted CTR: {analysis2['predicted_performance']['predicted_ctr']:.3f}")
    print(f"  Message Frame: {analysis2['content_analysis']['message_frame']}")
    print(f"  Visual Style: {analysis2['content_analysis']['visual_style']}")
    print()
    
    # Compare creatives
    print("Creative Comparison:")
    comparison = analyzer.get_creative_differences("crisis_parent_1", "researcher_1")
    print(f"  CTR Difference: {comparison['performance_difference']['ctr_diff']:.3f}")
    print(f"  Better Performer: {comparison['analysis']['better_performing']}")
    print(f"  Key Advantages: {comparison['analysis']['key_advantages']}")
    print()
    
    # Get insights
    print("Content Insights:")
    insights = analyzer.get_content_insights()
    print(f"  Total Creatives: {insights['total_creatives']}")
    print(f"  Top Message Frames: {insights['content_trends']['message_frames']}")
    print(f"  Top Visual Styles: {insights['content_trends']['visual_styles']}")
    
    # Get recommendations
    recommendations = analyzer.select_optimal_creative_features("crisis_parents", "ctr")
    print(f"\nOptimal Features for Crisis Parents:")
    print(f"  Recommended Headline Length: {recommendations['headline_length']}")
    print(f"  Message Frame: {recommendations['message_frame']}")
    print(f"  Visual Style: {recommendations['visual_style']}")
    print(f"  Use Urgency: {recommendations['use_urgency']}")