"""
Content Safety Module for GAELP Ad Campaign Safety
Implements content moderation, brand safety, and compliance checking.
"""

import logging
import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import json
import hashlib
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    URL = "url"


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


@dataclass
class ContentItem:
    """Represents content to be moderated"""
    content_id: str
    content_type: ContentType
    content: Any  # Text, URL, or binary data
    campaign_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContentViolation:
    """Content violation detection result"""
    content_id: str
    violation_type: ContentViolationType
    severity: ViolationSeverity
    description: str
    confidence: float
    flagged_content: str
    remediation_suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ProhibitedContentDetector:
    """Detects prohibited and harmful content"""
    
    def __init__(self):
        # Prohibited keywords and patterns
        self.prohibited_keywords = {
            'illegal': [
                'drugs', 'cocaine', 'heroin', 'marijuana', 'cannabis', 'weed',
                'illegal weapons', 'firearms', 'explosives', 'bomb making',
                'counterfeit', 'fake documents', 'identity theft', 'fraud',
                'money laundering', 'tax evasion', 'piracy', 'torrents'
            ],
            'harmful': [
                'suicide', 'self harm', 'cutting', 'eating disorder',
                'anorexia', 'bulimia', 'violence against', 'hate speech',
                'terrorist', 'extremist', 'nazi', 'white power'
            ],
            'adult': [
                'porn', 'xxx', 'adult content', 'escort', 'prostitution',
                'sexual services', 'strip club', 'adult toys'
            ],
            'misleading': [
                'get rich quick', 'guaranteed income', 'miracle cure',
                'lose weight fast', 'fountain of youth', 'secret government',
                'doctors hate this trick', 'one weird trick'
            ]
        }
        
        # Compile regex patterns
        self.prohibited_patterns = {}
        for category, keywords in self.prohibited_keywords.items():
            pattern = '|'.join(re.escape(keyword) for keyword in keywords)
            self.prohibited_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    async def detect_violations(self, content: ContentItem) -> List[ContentViolation]:
        """Detect content violations in the given content"""
        violations = []
        
        if content.content_type == ContentType.TEXT:
            violations.extend(await self._check_text_content(content))
        elif content.content_type == ContentType.URL:
            violations.extend(await self._check_url_content(content))
        elif content.content_type in [ContentType.IMAGE, ContentType.VIDEO]:
            violations.extend(await self._check_media_content(content))
        
        return violations
    
    async def _check_text_content(self, content: ContentItem) -> List[ContentViolation]:
        """Check text content for violations"""
        violations = []
        text = str(content.content).lower()
        
        for category, pattern in self.prohibited_patterns.items():
            matches = pattern.findall(text)
            if matches:
                severity = self._determine_severity(category)
                violation_type = self._map_category_to_violation_type(category)
                
                violations.append(ContentViolation(
                    content_id=content.content_id,
                    violation_type=violation_type,
                    severity=severity,
                    description=f"Prohibited {category} content detected",
                    confidence=0.9,
                    flagged_content=', '.join(matches[:3]),  # Show first 3 matches
                    remediation_suggestions=[
                        f"Remove references to {category} content",
                        "Revise messaging to comply with platform policies"
                    ]
                ))
        
        # Check for discriminatory language
        discriminatory_violations = await self._check_discriminatory_content(content, text)
        violations.extend(discriminatory_violations)
        
        return violations
    
    async def _check_url_content(self, content: ContentItem) -> List[ContentViolation]:
        """Check URL content for safety issues"""
        violations = []
        url = str(content.content)
        
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Check against known problematic domains
            if self._is_suspicious_domain(domain):
                violations.append(ContentViolation(
                    content_id=content.content_id,
                    violation_type=ContentViolationType.BRAND_SAFETY,
                    severity=ViolationSeverity.MEDIUM,
                    description="Potentially unsafe or low-quality domain",
                    confidence=0.7,
                    flagged_content=domain,
                    remediation_suggestions=[
                        "Use a more reputable domain",
                        "Verify domain safety before including in ads"
                    ]
                ))
            
            # Check for misleading URLs
            if self._is_misleading_url(url):
                violations.append(ContentViolation(
                    content_id=content.content_id,
                    violation_type=ContentViolationType.MISLEADING,
                    severity=ViolationSeverity.HIGH,
                    description="URL appears to be misleading or deceptive",
                    confidence=0.8,
                    flagged_content=url,
                    remediation_suggestions=[
                        "Use clear, descriptive URLs",
                        "Avoid URL shorteners that obscure destination"
                    ]
                ))
        
        except Exception as e:
            logger.error(f"Error checking URL content: {e}")
        
        return violations
    
    async def _check_media_content(self, content: ContentItem) -> List[ContentViolation]:
        """Check image/video content (placeholder for ML-based detection)"""
        violations = []
        
        # This would integrate with image/video analysis services
        # For now, return basic checks based on metadata
        
        metadata = content.metadata
        if 'adult_content_detected' in metadata and metadata['adult_content_detected']:
            violations.append(ContentViolation(
                content_id=content.content_id,
                violation_type=ContentViolationType.AGE_INAPPROPRIATE,
                severity=ViolationSeverity.CRITICAL,
                description="Adult content detected in media",
                confidence=metadata.get('confidence', 0.9),
                flagged_content="Media content",
                remediation_suggestions=[
                    "Remove adult content",
                    "Use age-appropriate imagery"
                ]
            ))
        
        return violations
    
    async def _check_discriminatory_content(self, content: ContentItem, text: str) -> List[ContentViolation]:
        """Check for discriminatory language and bias"""
        violations = []
        
        discriminatory_patterns = [
            r'\b(men|women|boys|girls) (are|can\'t|cannot|shouldn\'t)\b',
            r'\b(old|young|elderly) people (are|can\'t)\b',
            r'\ballahu akbar\b',  # Religious expressions used inappropriately
            r'\b(gay|straight) (people|men|women) (are|can\'t)\b'
        ]
        
        for pattern in discriminatory_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(ContentViolation(
                    content_id=content.content_id,
                    violation_type=ContentViolationType.DISCRIMINATION,
                    severity=ViolationSeverity.HIGH,
                    description="Potentially discriminatory language detected",
                    confidence=0.7,
                    flagged_content="Discriminatory language pattern",
                    remediation_suggestions=[
                        "Use inclusive language",
                        "Avoid stereotypes and generalizations",
                        "Review content for bias"
                    ]
                ))
        
        return violations
    
    def _determine_severity(self, category: str) -> ViolationSeverity:
        """Determine violation severity based on category"""
        severity_map = {
            'illegal': ViolationSeverity.CRITICAL,
            'harmful': ViolationSeverity.CRITICAL,
            'adult': ViolationSeverity.HIGH,
            'misleading': ViolationSeverity.MEDIUM
        }
        return severity_map.get(category, ViolationSeverity.LOW)
    
    def _map_category_to_violation_type(self, category: str) -> ContentViolationType:
        """Map content category to violation type"""
        type_map = {
            'illegal': ContentViolationType.PROHIBITED_CONTENT,
            'harmful': ContentViolationType.HARMFUL_CONTENT,
            'adult': ContentViolationType.AGE_INAPPROPRIATE,
            'misleading': ContentViolationType.MISLEADING
        }
        return type_map.get(category, ContentViolationType.PROHIBITED_CONTENT)
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain is suspicious or low-quality"""
        suspicious_indicators = [
            # Free hosting services
            'blogspot.com', 'wordpress.com', 'wixsite.com',
            # URL shorteners
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl',
            # Suspicious TLDs
            '.tk', '.ml', '.ga', '.cf',
            # Common phishing patterns
            'secure-', '-secure', 'verify-', '-verify'
        ]
        
        return any(indicator in domain for indicator in suspicious_indicators)
    
    def _is_misleading_url(self, url: str) -> bool:
        """Check if URL appears misleading"""
        misleading_patterns = [
            r'bit\.ly/[A-Za-z0-9]+',  # Shortened URLs
            r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+',  # Raw IP addresses
            r'[a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9]+\.com'  # Suspicious hyphenated domains
        ]
        
        return any(re.search(pattern, url) for pattern in misleading_patterns)


class BrandSafetyChecker:
    """Checks content against brand safety guidelines"""
    
    def __init__(self):
        self.unsafe_contexts = {
            'violence': ['war', 'conflict', 'shooting', 'murder', 'death', 'terrorism'],
            'negative_news': ['disaster', 'accident', 'crisis', 'scandal', 'controversy'],
            'inappropriate_language': ['profanity', 'explicit', 'vulgar', 'offensive'],
            'sensitive_topics': ['politics', 'religion', 'controversial', 'divisive']
        }
    
    async def check_brand_safety(self, content: ContentItem, brand_guidelines: Dict[str, Any] = None) -> List[ContentViolation]:
        """Check content against brand safety guidelines"""
        violations = []
        
        if content.content_type == ContentType.TEXT:
            text = str(content.content).lower()
            
            for context, keywords in self.unsafe_contexts.items():
                for keyword in keywords:
                    if keyword in text:
                        # Check if brand allows this context
                        if brand_guidelines and not brand_guidelines.get(f'allow_{context}', False):
                            violations.append(ContentViolation(
                                content_id=content.content_id,
                                violation_type=ContentViolationType.BRAND_SAFETY,
                                severity=ViolationSeverity.MEDIUM,
                                description=f"Content contains {context} which may not align with brand safety",
                                confidence=0.6,
                                flagged_content=keyword,
                                remediation_suggestions=[
                                    f"Avoid {context} in brand messaging",
                                    "Choose more positive, brand-safe content"
                                ]
                            ))
                        break  # Don't duplicate violations for same context
        
        return violations


class AgeAppropriatenessChecker:
    """Checks content for age-appropriate material"""
    
    def __init__(self):
        self.age_restricted_content = [
            'alcohol', 'beer', 'wine', 'liquor', 'drinking',
            'gambling', 'casino', 'betting', 'poker',
            'smoking', 'cigarettes', 'tobacco', 'vaping',
            'mature content', 'adult themes'
        ]
    
    async def check_age_appropriateness(self, content: ContentItem, target_age: int = 13) -> List[ContentViolation]:
        """Check if content is appropriate for target age"""
        violations = []
        
        if content.content_type == ContentType.TEXT:
            text = str(content.content).lower()
            
            for restricted in self.age_restricted_content:
                if restricted in text:
                    violations.append(ContentViolation(
                        content_id=content.content_id,
                        violation_type=ContentViolationType.AGE_INAPPROPRIATE,
                        severity=ViolationSeverity.MEDIUM,
                        description=f"Content may not be appropriate for users under {target_age}",
                        confidence=0.7,
                        flagged_content=restricted,
                        remediation_suggestions=[
                            "Add age restrictions to campaign targeting",
                            "Modify content to be more age-appropriate",
                            "Consider separate campaigns for different age groups"
                        ]
                    ))
        
        return violations


class PlatformPolicyChecker:
    """Checks content against platform-specific policies"""
    
    def __init__(self):
        self.platform_policies = {
            'google_ads': {
                'prohibited': ['cryptocurrency', 'get rich quick', 'miracle cures'],
                'restricted': ['alcohol', 'gambling', 'healthcare'],
                'requirements': ['clear pricing', 'contact information', 'privacy policy']
            },
            'facebook_ads': {
                'prohibited': ['adult content', 'illegal products', 'discriminatory practices'],
                'restricted': ['alcohol', 'dating', 'financial services'],
                'requirements': ['clear call to action', 'accurate representation']
            },
            'microsoft_ads': {
                'prohibited': ['illegal content', 'adult services', 'counterfeit goods'],
                'restricted': ['alcohol', 'gambling', 'political content'],
                'requirements': ['truthful claims', 'clear pricing']
            }
        }
    
    async def check_platform_compliance(self, content: ContentItem, platform: str) -> List[ContentViolation]:
        """Check content compliance with specific platform policies"""
        violations = []
        
        if platform not in self.platform_policies:
            logger.warning(f"Unknown platform: {platform}")
            return violations
        
        policies = self.platform_policies[platform]
        
        if content.content_type == ContentType.TEXT:
            text = str(content.content).lower()
            
            # Check prohibited content
            for prohibited in policies['prohibited']:
                if prohibited in text:
                    violations.append(ContentViolation(
                        content_id=content.content_id,
                        violation_type=ContentViolationType.PLATFORM_POLICY,
                        severity=ViolationSeverity.HIGH,
                        description=f"Content violates {platform} policy: {prohibited} not allowed",
                        confidence=0.8,
                        flagged_content=prohibited,
                        remediation_suggestions=[
                            f"Remove references to {prohibited}",
                            f"Review {platform} advertising policies"
                        ]
                    ))
        
        return violations


class ContentSafetyOrchestrator:
    """Main orchestrator for all content safety checks"""
    
    def __init__(self):
        self.prohibited_detector = ProhibitedContentDetector()
        self.brand_safety_checker = BrandSafetyChecker()
        self.age_checker = AgeAppropriatenessChecker()
        self.platform_checker = PlatformPolicyChecker()
        
        self.violation_history: Dict[str, List[ContentViolation]] = {}
        self.approved_content: Set[str] = set()
        self.rejected_content: Set[str] = set()
    
    async def moderate_content(
        self, 
        content: ContentItem,
        platform: str = None,
        brand_guidelines: Dict[str, Any] = None,
        target_age: int = 13
    ) -> Tuple[bool, List[ContentViolation]]:
        """
        Comprehensive content moderation.
        Returns (is_approved, violations_found)
        """
        all_violations = []
        
        try:
            # Generate content hash for caching
            content_hash = self._generate_content_hash(content)
            
            # Check cache first
            if content_hash in self.approved_content:
                logger.info(f"Content {content.content_id} pre-approved from cache")
                return True, []
            
            if content_hash in self.rejected_content:
                logger.info(f"Content {content.content_id} pre-rejected from cache")
                # Return cached violations if available
                return False, self.violation_history.get(content.content_id, [])
            
            # Run all safety checks in parallel
            tasks = [
                self.prohibited_detector.detect_violations(content),
                self.brand_safety_checker.check_brand_safety(content, brand_guidelines),
                self.age_checker.check_age_appropriateness(content, target_age)
            ]
            
            if platform:
                tasks.append(self.platform_checker.check_platform_compliance(content, platform))
            
            results = await asyncio.gather(*tasks)
            
            # Combine all violations
            for violations in results:
                all_violations.extend(violations)
            
            # Store violations in history
            if all_violations:
                self.violation_history[content.content_id] = all_violations
            
            # Determine approval based on violation severity
            is_approved = self._determine_approval(all_violations)
            
            # Cache result
            if is_approved:
                self.approved_content.add(content_hash)
            else:
                self.rejected_content.add(content_hash)
            
            logger.info(f"Content {content.content_id} moderation complete: "
                       f"{'APPROVED' if is_approved else 'REJECTED'} "
                       f"({len(all_violations)} violations)")
            
            return is_approved, all_violations
            
        except Exception as e:
            logger.error(f"Content moderation failed for {content.content_id}: {e}")
            # Err on the side of caution
            return False, [ContentViolation(
                content_id=content.content_id,
                violation_type=ContentViolationType.HARMFUL_CONTENT,
                severity=ViolationSeverity.CRITICAL,
                description=f"Moderation system error: {str(e)}",
                confidence=1.0,
                flagged_content="System error"
            )]
    
    def _generate_content_hash(self, content: ContentItem) -> str:
        """Generate a hash for content caching"""
        content_str = f"{content.content_type.value}:{str(content.content)}"
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def _determine_approval(self, violations: List[ContentViolation]) -> bool:
        """Determine if content should be approved based on violations"""
        if not violations:
            return True
        
        # Any critical violation = rejection
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            return False
        
        # More than 2 high severity violations = rejection
        high_violations = [v for v in violations if v.severity == ViolationSeverity.HIGH]
        if len(high_violations) > 2:
            return False
        
        # More than 3 medium severity violations = rejection
        medium_violations = [v for v in violations if v.severity == ViolationSeverity.MEDIUM]
        if len(medium_violations) > 3:
            return False
        
        # Otherwise approve (low severity or few violations)
        return True
    
    def get_violation_history(self, content_id: str) -> List[ContentViolation]:
        """Get violation history for specific content"""
        return self.violation_history.get(content_id, [])
    
    def get_moderation_stats(self) -> Dict[str, Any]:
        """Get moderation statistics"""
        total_processed = len(self.approved_content) + len(self.rejected_content)
        
        return {
            "total_processed": total_processed,
            "approved": len(self.approved_content),
            "rejected": len(self.rejected_content),
            "approval_rate": len(self.approved_content) / max(total_processed, 1),
            "total_violations": sum(len(violations) for violations in self.violation_history.values()),
            "violation_types": self._get_violation_type_counts()
        }
    
    def _get_violation_type_counts(self) -> Dict[str, int]:
        """Get counts of different violation types"""
        type_counts = {}
        for violations in self.violation_history.values():
            for violation in violations:
                violation_type = violation.violation_type.value
                type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        return type_counts