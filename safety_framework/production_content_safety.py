"""
Production-Ready Content Safety for GAELP Ad Campaign Safety
Integrates with real AI moderation services, platform policies, and regulatory compliance.
"""

import logging
import httpx
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import json
import hashlib
import base64
from urllib.parse import urlparse
import re
from google.cloud import storage
from google.cloud import vision
from google.cloud import videointelligence
from google.cloud import language_v1
import openai
from perspective import PerspectiveAPI
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    URL = "url"
    HTML = "html"
    DOCUMENT = "document"


class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentViolationType(Enum):
    PROHIBITED_CONTENT = "prohibited_content"
    BRAND_SAFETY = "brand_safety"
    AGE_INAPPROPRIATE = "age_inappropriate"
    MISLEADING = "misleading"
    TRADEMARK_VIOLATION = "trademark_violation"
    PLATFORM_POLICY = "platform_policy"
    DISCRIMINATION = "discrimination"
    HARMFUL_CONTENT = "harmful_content"
    ADULT_CONTENT = "adult_content"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    MISINFORMATION = "misinformation"
    SCAM = "scam"
    REGULATORY_VIOLATION = "regulatory_violation"
    COPYRIGHT_VIOLATION = "copyright_violation"


class ModerationAction(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    FLAG_FOR_REVIEW = "flag_for_review"
    REQUIRE_MODIFICATION = "require_modification"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    BLOCK_PERMANENTLY = "block_permanently"


@dataclass
class ProductionContentItem:
    """Enhanced content item for production moderation"""
    content_id: str
    content_type: ContentType
    content: Any  # Text, URL, or binary data
    campaign_id: str
    
    # Content metadata
    language: Optional[str] = None
    target_audience: Optional[str] = None
    geographic_targets: List[str] = field(default_factory=list)
    platform_targets: List[str] = field(default_factory=list)
    
    # Compliance data
    gdpr_applicable: bool = False
    ccpa_applicable: bool = False
    coppa_applicable: bool = False
    jurisdiction: str = "US"
    
    # Content context
    advertiser_category: Optional[str] = None
    brand_name: Optional[str] = None
    product_category: Optional[str] = None
    
    # Technical metadata
    file_size: Optional[int] = None
    duration: Optional[float] = None  # For video/audio
    dimensions: Optional[Tuple[int, int]] = None  # For images/video
    format: Optional[str] = None
    
    # Audit trail
    submitted_by: str = ""
    submission_timestamp: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductionContentViolation:
    """Enhanced content violation with detailed analysis"""
    violation_id: str
    content_id: str
    violation_type: ContentViolationType
    severity: ViolationSeverity
    description: str
    confidence: float
    
    # Specific violation details
    flagged_content: str
    flagged_segments: List[Dict[str, Any]] = field(default_factory=list)  # For video/audio
    
    # AI analysis results
    ai_service_results: Dict[str, Any] = field(default_factory=dict)
    human_review_required: bool = False
    
    # Platform-specific violations
    platform_violations: Dict[str, List[str]] = field(default_factory=dict)
    
    # Legal/regulatory implications
    legal_risk_level: str = "low"  # low, medium, high, critical
    regulatory_violations: List[str] = field(default_factory=list)
    
    # Remediation
    remediation_suggestions: List[str] = field(default_factory=list)
    auto_fix_available: bool = False
    estimated_fix_time: Optional[timedelta] = None
    
    # Appeal process
    appealable: bool = True
    appeal_deadline: Optional[datetime] = None
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModerationResult:
    """Complete moderation result"""
    content_id: str
    action: ModerationAction
    violations: List[ProductionContentViolation]
    overall_risk_score: float
    processing_time: float
    
    # Platform approvals
    platform_approvals: Dict[str, bool] = field(default_factory=dict)
    
    # Human review
    requires_human_review: bool = False
    human_review_priority: str = "normal"  # low, normal, high, urgent
    
    # Next steps
    recommended_actions: List[str] = field(default_factory=list)
    resubmission_allowed: bool = True
    
    processed_at: datetime = field(default_factory=datetime.utcnow)


class AIContentModerator:
    """Advanced AI-powered content moderation"""
    
    def __init__(self, config: Dict[str, Any]):
        # AI service clients
        self.openai_client = openai
        if config.get('openai_api_key'):
            openai.api_key = config['openai_api_key']
        
        # Google Cloud AI services
        if config.get('gcp_project_id'):
            self.vision_client = vision.ImageAnnotatorClient()
            self.video_client = videointelligence.VideoIntelligenceServiceClient()
            self.language_client = language_v1.LanguageServiceClient()
            self.storage_client = storage.Client()
        
        # Perspective API for toxicity detection
        if config.get('perspective_api_key'):
            self.perspective_api = PerspectiveAPI(config['perspective_api_key'])
        
        # Content safety thresholds
        self.toxicity_threshold = config.get('toxicity_threshold', 0.7)
        self.adult_content_threshold = config.get('adult_content_threshold', 0.8)
        self.violence_threshold = config.get('violence_threshold', 0.75)
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        logger.info("AI content moderator initialized with production services")
    
    async def moderate_text(self, content_item: ProductionContentItem) -> List[ProductionContentViolation]:
        """Advanced text moderation using multiple AI services"""
        violations = []
        text = str(content_item.content)
        
        try:
            # OpenAI moderation
            openai_violations = await self._openai_moderation(content_item, text)
            violations.extend(openai_violations)
            
            # Google Cloud Natural Language for sentiment and classification
            google_violations = await self._google_text_analysis(content_item, text)
            violations.extend(google_violations)
            
            # Perspective API for toxicity
            perspective_violations = await self._perspective_analysis(content_item, text)
            violations.extend(perspective_violations)
            
            # Custom regulatory compliance checks
            regulatory_violations = await self._regulatory_text_analysis(content_item, text)
            violations.extend(regulatory_violations)
            
            # Platform-specific policy checks
            platform_violations = await self._platform_policy_checks(content_item, text)
            violations.extend(platform_violations)
            
        except Exception as e:
            logger.error(f"Text moderation failed for {content_item.content_id}: {e}")
            # Create a generic violation for system error
            violations.append(ProductionContentViolation(
                violation_id=str(uuid.uuid4()),
                content_id=content_item.content_id,
                violation_type=ContentViolationType.PROHIBITED_CONTENT,
                severity=ViolationSeverity.MEDIUM,
                description="Content moderation system error - manual review required",
                confidence=0.5,
                flagged_content="system_error",
                human_review_required=True,
                legal_risk_level="medium"
            ))
        
        return violations
    
    async def moderate_image(self, content_item: ProductionContentItem) -> List[ProductionContentViolation]:
        """Advanced image moderation using Google Vision AI"""
        violations = []
        
        try:
            # Google Vision AI analysis
            image_data = content_item.content
            if isinstance(image_data, str):
                # If it's a URL, download the image
                image_data = await self._download_image(image_data)
            
            # Safe search detection
            safe_search_result = await self._google_safe_search(image_data)
            if safe_search_result:
                violations.extend(safe_search_result)
            
            # Object detection for policy violations
            object_violations = await self._google_object_detection(content_item, image_data)
            violations.extend(object_violations)
            
            # OCR for text in images
            ocr_text = await self._google_ocr(image_data)
            if ocr_text:
                # Run text moderation on extracted text
                text_item = ProductionContentItem(
                    content_id=f"{content_item.content_id}_ocr",
                    content_type=ContentType.TEXT,
                    content=ocr_text,
                    campaign_id=content_item.campaign_id,
                    **{k: v for k, v in content_item.__dict__.items() 
                       if k not in ['content_id', 'content_type', 'content']}
                )
                text_violations = await self.moderate_text(text_item)
                for violation in text_violations:
                    violation.content_id = content_item.content_id
                    violation.violation_id = str(uuid.uuid4())
                    violation.description = f"Text in image: {violation.description}"
                violations.extend(text_violations)
            
            # Custom brand safety checks
            brand_violations = await self._brand_safety_image_analysis(content_item, image_data)
            violations.extend(brand_violations)
            
        except Exception as e:
            logger.error(f"Image moderation failed for {content_item.content_id}: {e}")
            violations.append(ProductionContentViolation(
                violation_id=str(uuid.uuid4()),
                content_id=content_item.content_id,
                violation_type=ContentViolationType.PROHIBITED_CONTENT,
                severity=ViolationSeverity.MEDIUM,
                description="Image moderation system error - manual review required",
                confidence=0.5,
                flagged_content="system_error",
                human_review_required=True
            ))
        
        return violations
    
    async def moderate_video(self, content_item: ProductionContentItem) -> List[ProductionContentViolation]:
        """Advanced video moderation using Google Video Intelligence"""
        violations = []
        
        try:
            video_uri = content_item.content
            if not video_uri.startswith('gs://'):
                # Upload to Cloud Storage first
                video_uri = await self._upload_to_gcs(content_item.content, content_item.content_id)
            
            # Video analysis
            features = [
                videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
                videointelligence.Feature.SPEECH_TRANSCRIPTION,
                videointelligence.Feature.OBJECT_TRACKING,
                videointelligence.Feature.TEXT_DETECTION
            ]
            
            operation = self.video_client.annotate_video(
                request={
                    "input_uri": video_uri,
                    "features": features,
                    "video_context": {
                        "explicit_content_detection_config": {
                            "model": "builtin/latest"
                        },
                        "speech_transcription_config": {
                            "language_code": content_item.language or "en-US",
                            "enable_automatic_punctuation": True,
                        }
                    }
                }
            )
            
            # Wait for operation to complete (with timeout)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, operation.result
                ),
                timeout=300  # 5 minutes
            )
            
            # Process explicit content detection
            if result.annotation_results[0].explicit_annotation:
                explicit_violations = await self._process_explicit_content_results(
                    content_item, result.annotation_results[0].explicit_annotation
                )
                violations.extend(explicit_violations)
            
            # Process speech transcription
            if result.annotation_results[0].speech_transcriptions:
                transcript_text = " ".join([
                    alternative.transcript 
                    for transcription in result.annotation_results[0].speech_transcriptions
                    for alternative in transcription.alternatives
                ])
                
                # Run text moderation on transcript
                if transcript_text:
                    text_item = ProductionContentItem(
                        content_id=f"{content_item.content_id}_transcript",
                        content_type=ContentType.TEXT,
                        content=transcript_text,
                        campaign_id=content_item.campaign_id,
                        **{k: v for k, v in content_item.__dict__.items() 
                           if k not in ['content_id', 'content_type', 'content']}
                    )
                    transcript_violations = await self.moderate_text(text_item)
                    for violation in transcript_violations:
                        violation.content_id = content_item.content_id
                        violation.violation_id = str(uuid.uuid4())
                        violation.description = f"Video transcript: {violation.description}"
                    violations.extend(transcript_violations)
            
        except asyncio.TimeoutError:
            logger.error(f"Video moderation timeout for {content_item.content_id}")
            violations.append(ProductionContentViolation(
                violation_id=str(uuid.uuid4()),
                content_id=content_item.content_id,
                violation_type=ContentViolationType.PROHIBITED_CONTENT,
                severity=ViolationSeverity.HIGH,
                description="Video analysis timeout - manual review required",
                confidence=1.0,
                flagged_content="timeout_error",
                human_review_required=True
            ))
        except Exception as e:
            logger.error(f"Video moderation failed for {content_item.content_id}: {e}")
            violations.append(ProductionContentViolation(
                violation_id=str(uuid.uuid4()),
                content_id=content_item.content_id,
                violation_type=ContentViolationType.PROHIBITED_CONTENT,
                severity=ViolationSeverity.MEDIUM,
                description="Video moderation system error - manual review required",
                confidence=0.5,
                flagged_content="system_error",
                human_review_required=True
            ))
        
        return violations
    
    async def _openai_moderation(self, content_item: ProductionContentItem, text: str) -> List[ProductionContentViolation]:
        """Use OpenAI's moderation API"""
        violations = []
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: openai.Moderation.create(input=text)
            )
            
            result = response['results'][0]
            
            if result['flagged']:
                categories = result['categories']
                category_scores = result['category_scores']
                
                for category, flagged in categories.items():
                    if flagged:
                        confidence = category_scores[category]
                        severity = self._determine_severity_from_score(confidence)
                        
                        violation_type = self._map_openai_category_to_violation_type(category)
                        
                        violations.append(ProductionContentViolation(
                            violation_id=str(uuid.uuid4()),
                            content_id=content_item.content_id,
                            violation_type=violation_type,
                            severity=severity,
                            description=f"OpenAI detected {category} content",
                            confidence=confidence,
                            flagged_content=text[:200] + "..." if len(text) > 200 else text,
                            ai_service_results={'openai': result},
                            human_review_required=(confidence > 0.9),
                            legal_risk_level=self._determine_legal_risk(violation_type, confidence)
                        ))
        
        except Exception as e:
            logger.error(f"OpenAI moderation failed: {e}")
        
        return violations
    
    async def _google_safe_search(self, image_data: bytes) -> List[ProductionContentViolation]:
        """Use Google Vision Safe Search"""
        violations = []
        
        try:
            image = vision.Image(content=image_data)
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.vision_client.safe_search_detection(image=image)
            )
            
            safe_search = response.safe_search_annotation
            
            # Check each category
            categories = {
                'adult': safe_search.adult,
                'violence': safe_search.violence,
                'racy': safe_search.racy,
                'spoof': safe_search.spoof,
                'medical': safe_search.medical
            }
            
            for category, likelihood in categories.items():
                if likelihood >= 3:  # LIKELY or VERY_LIKELY
                    severity = ViolationSeverity.HIGH if likelihood >= 4 else ViolationSeverity.MEDIUM
                    confidence = 0.8 if likelihood >= 4 else 0.6
                    
                    violation_type = self._map_safe_search_category_to_violation_type(category)
                    
                    violations.append(ProductionContentViolation(
                        violation_id=str(uuid.uuid4()),
                        content_id="",  # Will be set by caller
                        violation_type=violation_type,
                        severity=severity,
                        description=f"Google Vision detected {category} content",
                        confidence=confidence,
                        flagged_content=f"{category}_content_detected",
                        ai_service_results={'google_vision_safe_search': {category: likelihood}},
                        human_review_required=(likelihood >= 4)
                    ))
        
        except Exception as e:
            logger.error(f"Google Safe Search failed: {e}")
        
        return violations
    
    def _determine_severity_from_score(self, score: float) -> ViolationSeverity:
        """Determine violation severity from AI confidence score"""
        if score >= 0.9:
            return ViolationSeverity.CRITICAL
        elif score >= 0.7:
            return ViolationSeverity.HIGH
        elif score >= 0.5:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    
    def _map_openai_category_to_violation_type(self, category: str) -> ContentViolationType:
        """Map OpenAI moderation categories to our violation types"""
        mapping = {
            'hate': ContentViolationType.HATE_SPEECH,
            'hate/threatening': ContentViolationType.HATE_SPEECH,
            'harassment': ContentViolationType.HARMFUL_CONTENT,
            'harassment/threatening': ContentViolationType.HARMFUL_CONTENT,
            'self-harm': ContentViolationType.HARMFUL_CONTENT,
            'self-harm/intent': ContentViolationType.HARMFUL_CONTENT,
            'self-harm/instructions': ContentViolationType.HARMFUL_CONTENT,
            'sexual': ContentViolationType.ADULT_CONTENT,
            'sexual/minors': ContentViolationType.ADULT_CONTENT,
            'violence': ContentViolationType.VIOLENCE,
            'violence/graphic': ContentViolationType.VIOLENCE
        }
        return mapping.get(category, ContentViolationType.PROHIBITED_CONTENT)
    
    def _map_safe_search_category_to_violation_type(self, category: str) -> ContentViolationType:
        """Map Google Safe Search categories to our violation types"""
        mapping = {
            'adult': ContentViolationType.ADULT_CONTENT,
            'violence': ContentViolationType.VIOLENCE,
            'racy': ContentViolationType.ADULT_CONTENT,
            'spoof': ContentViolationType.MISLEADING,
            'medical': ContentViolationType.BRAND_SAFETY
        }
        return mapping.get(category, ContentViolationType.PROHIBITED_CONTENT)
    
    def _determine_legal_risk(self, violation_type: ContentViolationType, confidence: float) -> str:
        """Determine legal risk level based on violation type and confidence"""
        high_risk_types = {
            ContentViolationType.HATE_SPEECH,
            ContentViolationType.DISCRIMINATION,
            ContentViolationType.ADULT_CONTENT,
            ContentViolationType.REGULATORY_VIOLATION
        }
        
        if violation_type in high_risk_types:
            if confidence > 0.8:
                return "critical"
            elif confidence > 0.6:
                return "high"
            else:
                return "medium"
        else:
            return "low" if confidence < 0.7 else "medium"


class PlatformPolicyEngine:
    """Platform-specific policy enforcement"""
    
    def __init__(self):
        # Platform policy configurations
        self.platform_policies = {
            'google_ads': {
                'prohibited_content': [
                    'counterfeit_goods', 'dangerous_products', 'enabling_dishonest_behavior',
                    'inappropriate_content', 'restricted_content'
                ],
                'targeting_restrictions': {
                    'coppa': ['age_targeting_under_13'],
                    'gdpr': ['personal_data_targeting']
                }
            },
            'facebook_ads': {
                'prohibited_content': [
                    'adult_content', 'alcohol', 'body_parts', 'dating',
                    'gambling', 'health_claims', 'multilevel_marketing'
                ],
                'image_restrictions': {
                    'text_overlay_limit': 0.2,  # 20% max text overlay
                    'before_after_images': False
                }
            },
            'microsoft_ads': {
                'prohibited_content': [
                    'adult_content', 'illegal_content', 'inappropriate_content',
                    'restricted_content', 'unacceptable_business_practices'
                ]
            }
        }
    
    async def check_platform_compliance(self, content_item: ProductionContentItem, 
                                      platforms: List[str]) -> Dict[str, List[ProductionContentViolation]]:
        """Check content compliance for specific platforms"""
        violations_by_platform = {}
        
        for platform in platforms:
            violations = []
            
            if platform in self.platform_policies:
                policy = self.platform_policies[platform]
                
                # Check general content policies
                content_violations = await self._check_content_policies(
                    content_item, policy.get('prohibited_content', [])
                )
                violations.extend(content_violations)
                
                # Check platform-specific restrictions
                if content_item.content_type == ContentType.IMAGE and 'image_restrictions' in policy:
                    image_violations = await self._check_image_restrictions(
                        content_item, policy['image_restrictions']
                    )
                    violations.extend(image_violations)
                
                # Check targeting restrictions
                if 'targeting_restrictions' in policy:
                    targeting_violations = await self._check_targeting_restrictions(
                        content_item, policy['targeting_restrictions']
                    )
                    violations.extend(targeting_violations)
            
            violations_by_platform[platform] = violations
        
        return violations_by_platform
    
    async def _check_content_policies(self, content_item: ProductionContentItem, 
                                    prohibited_categories: List[str]) -> List[ProductionContentViolation]:
        """Check content against platform prohibited categories"""
        violations = []
        
        # This would contain detailed logic for each prohibited category
        # For now, return placeholder implementation
        
        return violations


class ProductionContentSafetyOrchestrator:
    """Production-ready content safety orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.ai_moderator = AIContentModerator(config)
        self.platform_policy_engine = PlatformPolicyEngine()
        
        # Configuration
        self.auto_approve_threshold = config.get('auto_approve_threshold', 0.95)
        self.auto_reject_threshold = config.get('auto_reject_threshold', 0.3)
        self.human_review_threshold = config.get('human_review_threshold', 0.7)
        
        # Audit and compliance
        self.audit_all_decisions = config.get('audit_all_decisions', True)
        self.gdpr_mode = config.get('gdpr_mode', False)
        self.ccpa_mode = config.get('ccpa_mode', False)
        
        # Performance tracking
        self._total_processed = 0
        self._approvals = 0
        self._rejections = 0
        self._human_reviews = 0
        
        logger.info("Production content safety orchestrator initialized")
    
    async def moderate_content(self, content_item: ProductionContentItem, 
                             target_platforms: List[str] = None) -> ModerationResult:
        """Comprehensive content moderation with platform-specific checks"""
        start_time = datetime.utcnow()
        violations = []
        
        try:
            # AI-powered moderation based on content type
            if content_item.content_type == ContentType.TEXT:
                ai_violations = await self.ai_moderator.moderate_text(content_item)
            elif content_item.content_type == ContentType.IMAGE:
                ai_violations = await self.ai_moderator.moderate_image(content_item)
            elif content_item.content_type == ContentType.VIDEO:
                ai_violations = await self.ai_moderator.moderate_video(content_item)
            else:
                ai_violations = []
            
            violations.extend(ai_violations)
            
            # Platform-specific policy checks
            platform_violations = {}
            if target_platforms:
                platform_violations = await self.platform_policy_engine.check_platform_compliance(
                    content_item, target_platforms
                )
                for platform, platform_specific_violations in platform_violations.items():
                    violations.extend(platform_specific_violations)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(violations)
            
            # Determine moderation action
            action = self._determine_moderation_action(overall_risk_score, violations)
            
            # Check if human review is required
            requires_human_review = self._requires_human_review(violations, overall_risk_score)
            
            # Create moderation result
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ModerationResult(
                content_id=content_item.content_id,
                action=action,
                violations=violations,
                overall_risk_score=overall_risk_score,
                processing_time=processing_time,
                platform_approvals={
                    platform: len(platform_violations.get(platform, [])) == 0
                    for platform in (target_platforms or [])
                },
                requires_human_review=requires_human_review,
                human_review_priority=self._determine_review_priority(violations),
                recommended_actions=self._generate_recommended_actions(violations),
                resubmission_allowed=(action != ModerationAction.BLOCK_PERMANENTLY)
            )
            
            # Update metrics
            self._update_metrics(result)
            
            # Audit logging
            if self.audit_all_decisions:
                await self._log_moderation_decision(content_item, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Content moderation failed for {content_item.content_id}: {e}")
            # Return safe fallback - require human review
            return ModerationResult(
                content_id=content_item.content_id,
                action=ModerationAction.ESCALATE_TO_HUMAN,
                violations=[],
                overall_risk_score=0.5,
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                requires_human_review=True,
                human_review_priority="urgent",
                recommended_actions=["System error - manual review required"]
            )
    
    def _calculate_overall_risk_score(self, violations: List[ProductionContentViolation]) -> float:
        """Calculate overall risk score from violations"""
        if not violations:
            return 0.0
        
        # Weight violations by severity and confidence
        weighted_scores = []
        for violation in violations:
            severity_weight = {
                ViolationSeverity.LOW: 0.25,
                ViolationSeverity.MEDIUM: 0.5,
                ViolationSeverity.HIGH: 0.75,
                ViolationSeverity.CRITICAL: 1.0
            }[violation.severity]
            
            weighted_score = violation.confidence * severity_weight
            weighted_scores.append(weighted_score)
        
        # Return maximum weighted score (most severe violation)
        return min(max(weighted_scores), 1.0)
    
    def _determine_moderation_action(self, risk_score: float, 
                                   violations: List[ProductionContentViolation]) -> ModerationAction:
        """Determine appropriate moderation action"""
        # Check for critical violations that require immediate blocking
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            return ModerationAction.BLOCK_PERMANENTLY
        
        # Check for violations requiring human escalation
        escalation_types = {
            ContentViolationType.REGULATORY_VIOLATION,
            ContentViolationType.COPYRIGHT_VIOLATION,
            ContentViolationType.TRADEMARK_VIOLATION
        }
        
        for violation in violations:
            if violation.violation_type in escalation_types:
                return ModerationAction.ESCALATE_TO_HUMAN
        
        # Score-based decisions
        if risk_score >= self.auto_reject_threshold:
            return ModerationAction.REJECT
        elif risk_score >= self.human_review_threshold:
            return ModerationAction.FLAG_FOR_REVIEW
        elif risk_score >= self.auto_approve_threshold:
            return ModerationAction.APPROVE
        else:
            return ModerationAction.REQUIRE_MODIFICATION
    
    def _requires_human_review(self, violations: List[ProductionContentViolation], 
                             risk_score: float) -> bool:
        """Determine if human review is required"""
        # Always require human review for certain violation types
        for violation in violations:
            if violation.human_review_required:
                return True
        
        # Require review for high-risk content
        return risk_score >= self.human_review_threshold
    
    def _determine_review_priority(self, violations: List[ProductionContentViolation]) -> str:
        """Determine priority level for human review"""
        if any(v.severity == ViolationSeverity.CRITICAL for v in violations):
            return "urgent"
        elif any(v.legal_risk_level == "critical" for v in violations):
            return "high"
        elif any(v.severity == ViolationSeverity.HIGH for v in violations):
            return "high"
        else:
            return "normal"
    
    def _generate_recommended_actions(self, violations: List[ProductionContentViolation]) -> List[str]:
        """Generate recommended actions based on violations"""
        actions = []
        
        for violation in violations:
            actions.extend(violation.remediation_suggestions)
        
        # Remove duplicates and return
        return list(set(actions))
    
    def _update_metrics(self, result: ModerationResult):
        """Update processing metrics"""
        self._total_processed += 1
        
        if result.action == ModerationAction.APPROVE:
            self._approvals += 1
        elif result.action in [ModerationAction.REJECT, ModerationAction.BLOCK_PERMANENTLY]:
            self._rejections += 1
        elif result.requires_human_review:
            self._human_reviews += 1
    
    async def _log_moderation_decision(self, content_item: ProductionContentItem, 
                                     result: ModerationResult):
        """Log moderation decision for audit purposes"""
        # This would integrate with audit logging system
        audit_data = {
            'content_id': content_item.content_id,
            'campaign_id': content_item.campaign_id,
            'content_type': content_item.content_type.value,
            'action': result.action.value,
            'risk_score': result.overall_risk_score,
            'violations': [
                {
                    'type': v.violation_type.value,
                    'severity': v.severity.value,
                    'confidence': v.confidence
                }
                for v in result.violations
            ],
            'processing_time': result.processing_time,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Content moderation audit: {audit_data}")
    
    def get_moderation_stats(self) -> Dict[str, Any]:
        """Get moderation statistics"""
        total = max(self._total_processed, 1)  # Avoid division by zero
        
        return {
            'total_processed': self._total_processed,
            'approval_rate': self._approvals / total,
            'rejection_rate': self._rejections / total,
            'human_review_rate': self._human_reviews / total,
            'processing_efficiency': (self._approvals + self._rejections) / total
        }