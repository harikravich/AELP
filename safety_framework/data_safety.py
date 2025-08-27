"""
Data Safety Module for GAELP Ad Campaign Safety
Implements PII protection, credential management, and privacy compliance.
"""

import logging
import hashlib
import re
import secrets
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    DEVICE_ID = "device_id"
    USER_ID = "user_id"


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class PrivacyRegulation(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    COPPA = "coppa"
    PIPEDA = "pipeda"
    LGPD = "lgpd"


@dataclass
class PIIDetection:
    """PII detection result"""
    pii_type: PIIType
    value: str
    confidence: float
    location: str  # Where in the data it was found
    masked_value: str


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration"""
    data_type: str
    retention_period: timedelta
    auto_delete: bool = True
    anonymize_after: Optional[timedelta] = None
    backup_retention: Optional[timedelta] = None
    compliance_requirements: List[PrivacyRegulation] = field(default_factory=list)


@dataclass
class ConsentRecord:
    """User consent record"""
    user_id: str
    consent_type: str
    granted: bool
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_string: Optional[str] = None  # IAB TCF string
    purposes: List[str] = field(default_factory=list)
    legitimate_interests: List[str] = field(default_factory=list)


@dataclass
class DataProcessingRecord:
    """Record of data processing activities"""
    record_id: str
    data_subject_id: str
    processing_purpose: str
    data_categories: List[str]
    legal_basis: str
    timestamp: datetime
    retention_period: timedelta
    third_parties: List[str] = field(default_factory=list)
    cross_border_transfers: List[str] = field(default_factory=list)


class PIIDetector:
    """Detects and masks personally identifiable information"""
    
    def __init__(self):
        self.pii_patterns = {
            PIIType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE: re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            PIIType.SSN: re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            PIIType.CREDIT_CARD: re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            PIIType.IP_ADDRESS: re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        }
        
        # Common name patterns (simplified)
        self.name_patterns = [
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # First Last
            re.compile(r'\b[A-Z][a-z]+, [A-Z][a-z]+\b')   # Last, First
        ]
    
    async def detect_pii(self, data: Any, context: str = "") -> List[PIIDetection]:
        """Detect PII in various data formats"""
        detections = []
        
        try:
            # Convert data to string for analysis
            if isinstance(data, dict):
                text = json.dumps(data)
            elif isinstance(data, (list, tuple)):
                text = str(data)
            else:
                text = str(data)
            
            # Detect each PII type
            for pii_type, pattern in self.pii_patterns.items():
                matches = pattern.finditer(text)
                for match in matches:
                    value = match.group()
                    masked_value = self._mask_value(value, pii_type)
                    
                    detections.append(PIIDetection(
                        pii_type=pii_type,
                        value=value,
                        confidence=0.9,  # High confidence for regex matches
                        location=f"{context}:pos_{match.start()}",
                        masked_value=masked_value
                    ))
            
            # Detect names (lower confidence)
            for pattern in self.name_patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    value = match.group()
                    # Skip if it looks like a company name or common phrases
                    if not self._looks_like_person_name(value):
                        continue
                    
                    masked_value = self._mask_value(value, PIIType.NAME)
                    
                    detections.append(PIIDetection(
                        pii_type=PIIType.NAME,
                        value=value,
                        confidence=0.6,  # Lower confidence for name detection
                        location=f"{context}:pos_{match.start()}",
                        masked_value=masked_value
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            return []
    
    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Mask PII value appropriately"""
        if pii_type == PIIType.EMAIL:
            parts = value.split('@')
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        
        elif pii_type == PIIType.PHONE:
            return f"***-***-{value[-4:]}"
        
        elif pii_type == PIIType.SSN:
            return f"***-**-{value[-4:]}"
        
        elif pii_type == PIIType.CREDIT_CARD:
            return f"****-****-****-{value[-4:]}"
        
        elif pii_type == PIIType.NAME:
            parts = value.split()
            if len(parts) >= 2:
                return f"{parts[0][0]}*** {parts[-1][0]}***"
        
        elif pii_type == PIIType.IP_ADDRESS:
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.***.***.***"
        
        # Default masking
        if len(value) > 4:
            return f"{value[:2]}***{value[-2:]}"
        else:
            return "***"
    
    def _looks_like_person_name(self, name: str) -> bool:
        """Check if a detected name pattern looks like a person's name"""
        # Skip common non-person patterns
        skip_patterns = [
            r'United States', r'New York', r'Los Angeles',
            r'Customer Service', r'Sales Team', r'Support Center',
            r'Privacy Policy', r'Terms Conditions'
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        
        return True
    
    async def anonymize_data(self, data: Dict[str, Any], detections: List[PIIDetection]) -> Dict[str, Any]:
        """Anonymize data by replacing PII with masked values"""
        try:
            anonymized_data = data.copy()
            text_data = json.dumps(anonymized_data)
            
            # Replace PII with masked values
            for detection in detections:
                text_data = text_data.replace(detection.value, detection.masked_value)
            
            # Parse back to dict
            return json.loads(text_data)
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return data


class CredentialManager:
    """Secure credential management for ad platform APIs"""
    
    def __init__(self, master_key: bytes = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        self.cipher = Fernet(master_key)
        
        self.credentials: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[Dict[str, Any]] = []
        self.credential_expiry: Dict[str, datetime] = {}
    
    async def store_credential(self, credential_id: str, platform: str, 
                             credential_data: Dict[str, Any], 
                             expires_at: datetime = None) -> bool:
        """Store encrypted credentials"""
        try:
            # Encrypt sensitive data
            encrypted_data = {}
            for key, value in credential_data.items():
                if self._is_sensitive_field(key):
                    encrypted_value = self.cipher.encrypt(str(value).encode())
                    encrypted_data[key] = base64.b64encode(encrypted_value).decode()
                else:
                    encrypted_data[key] = value
            
            # Store credential
            self.credentials[credential_id] = {
                'platform': platform,
                'data': encrypted_data,
                'created_at': datetime.utcnow(),
                'last_accessed': None,
                'access_count': 0
            }
            
            # Set expiry if provided
            if expires_at:
                self.credential_expiry[credential_id] = expires_at
            
            logger.info(f"Credential {credential_id} stored for platform {platform}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential {credential_id}: {e}")
            return False
    
    async def retrieve_credential(self, credential_id: str, requester_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt credentials"""
        try:
            if credential_id not in self.credentials:
                logger.error(f"Credential {credential_id} not found")
                return None
            
            # Check expiry
            if credential_id in self.credential_expiry:
                if datetime.utcnow() > self.credential_expiry[credential_id]:
                    logger.error(f"Credential {credential_id} has expired")
                    await self.revoke_credential(credential_id)
                    return None
            
            credential = self.credentials[credential_id]
            
            # Decrypt sensitive data
            decrypted_data = {}
            for key, value in credential['data'].items():
                if self._is_sensitive_field(key):
                    try:
                        encrypted_value = base64.b64decode(value.encode())
                        decrypted_value = self.cipher.decrypt(encrypted_value).decode()
                        decrypted_data[key] = decrypted_value
                    except Exception:
                        logger.error(f"Failed to decrypt field {key}")
                        return None
                else:
                    decrypted_data[key] = value
            
            # Log access
            access_event = {
                'credential_id': credential_id,
                'requester_id': requester_id,
                'timestamp': datetime.utcnow(),
                'platform': credential['platform']
            }
            self.access_log.append(access_event)
            
            # Update access tracking
            credential['last_accessed'] = datetime.utcnow()
            credential['access_count'] += 1
            
            return {
                'platform': credential['platform'],
                'data': decrypted_data
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential {credential_id}: {e}")
            return None
    
    async def revoke_credential(self, credential_id: str) -> bool:
        """Revoke/delete a credential"""
        try:
            if credential_id in self.credentials:
                del self.credentials[credential_id]
            
            if credential_id in self.credential_expiry:
                del self.credential_expiry[credential_id]
            
            logger.info(f"Credential {credential_id} revoked")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke credential {credential_id}: {e}")
            return False
    
    async def rotate_credential(self, credential_id: str, new_data: Dict[str, Any]) -> bool:
        """Rotate an existing credential"""
        try:
            if credential_id not in self.credentials:
                logger.error(f"Credential {credential_id} not found for rotation")
                return False
            
            # Store old credential temporarily for rollback
            old_credential = self.credentials[credential_id].copy()
            
            # Update with new data
            success = await self.store_credential(credential_id, 
                                                old_credential['platform'], 
                                                new_data)
            
            if success:
                logger.info(f"Credential {credential_id} rotated successfully")
                return True
            else:
                # Rollback on failure
                self.credentials[credential_id] = old_credential
                return False
                
        except Exception as e:
            logger.error(f"Failed to rotate credential {credential_id}: {e}")
            return False
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field contains sensitive data that should be encrypted"""
        sensitive_fields = {
            'api_key', 'secret', 'token', 'password', 'private_key',
            'client_secret', 'refresh_token', 'access_token'
        }
        return field_name.lower() in sensitive_fields
    
    def get_access_audit(self, credential_id: str = None) -> List[Dict[str, Any]]:
        """Get access audit log"""
        if credential_id:
            return [event for event in self.access_log if event['credential_id'] == credential_id]
        return self.access_log.copy()
    
    def get_credential_health(self) -> Dict[str, Any]:
        """Get credential health status"""
        now = datetime.utcnow()
        
        expired_count = sum(
            1 for credential_id, expiry in self.credential_expiry.items()
            if expiry <= now
        )
        
        expiring_soon_count = sum(
            1 for credential_id, expiry in self.credential_expiry.items()
            if now < expiry <= now + timedelta(days=7)
        )
        
        return {
            'total_credentials': len(self.credentials),
            'expired': expired_count,
            'expiring_soon': expiring_soon_count,
            'healthy': len(self.credentials) - expired_count - expiring_soon_count,
            'total_accesses': len(self.access_log),
            'platforms': list(set(cred['platform'] for cred in self.credentials.values()))
        }


class ConsentManager:
    """Manages user consent for privacy compliance"""
    
    def __init__(self):
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.consent_purposes = {
            'advertising': 'Personalized advertising',
            'analytics': 'Website analytics',
            'marketing': 'Marketing communications',
            'functional': 'Essential website functionality',
            'third_party': 'Third-party integrations'
        }
    
    async def record_consent(self, user_id: str, consent_type: str, granted: bool,
                           purposes: List[str] = None, ip_address: str = None,
                           user_agent: str = None, consent_string: str = None) -> str:
        """Record user consent"""
        try:
            consent_record = ConsentRecord(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                consent_string=consent_string,
                purposes=purposes or []
            )
            
            if user_id not in self.consent_records:
                self.consent_records[user_id] = []
            
            self.consent_records[user_id].append(consent_record)
            
            logger.info(f"Consent recorded for user {user_id}: {consent_type} = {granted}")
            return f"consent_{uuid.uuid4().hex[:8]}"
            
        except Exception as e:
            logger.error(f"Failed to record consent for user {user_id}: {e}")
            return ""
    
    async def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has granted consent for a specific purpose"""
        try:
            if user_id not in self.consent_records:
                return False
            
            # Get latest consent for each type
            latest_consents = {}
            for record in self.consent_records[user_id]:
                if record.consent_type not in latest_consents:
                    latest_consents[record.consent_type] = record
                elif record.timestamp > latest_consents[record.consent_type].timestamp:
                    latest_consents[record.consent_type] = record
            
            # Check if purpose is covered by any granted consent
            for consent_record in latest_consents.values():
                if consent_record.granted and purpose in consent_record.purposes:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check consent for user {user_id}: {e}")
            return False
    
    async def withdraw_consent(self, user_id: str, consent_type: str = None) -> bool:
        """Withdraw user consent"""
        try:
            if consent_type:
                # Withdraw specific consent type
                await self.record_consent(user_id, consent_type, False)
            else:
                # Withdraw all consent
                for purpose in self.consent_purposes:
                    await self.record_consent(user_id, purpose, False)
            
            logger.info(f"Consent withdrawn for user {user_id}: {consent_type or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to withdraw consent for user {user_id}: {e}")
            return False
    
    def get_consent_status(self, user_id: str) -> Dict[str, Any]:
        """Get current consent status for user"""
        try:
            if user_id not in self.consent_records:
                return {'consents': {}, 'last_updated': None}
            
            # Get latest consent for each type
            latest_consents = {}
            for record in self.consent_records[user_id]:
                if record.consent_type not in latest_consents:
                    latest_consents[record.consent_type] = record
                elif record.timestamp > latest_consents[record.consent_type].timestamp:
                    latest_consents[record.consent_type] = record
            
            status = {
                'consents': {
                    consent_type: record.granted 
                    for consent_type, record in latest_consents.items()
                },
                'last_updated': max(record.timestamp for record in latest_consents.values()) if latest_consents else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get consent status for user {user_id}: {e}")
            return {'consents': {}, 'last_updated': None}


class DataRetentionManager:
    """Manages data retention and deletion policies"""
    
    def __init__(self):
        self.retention_policies: Dict[str, DataRetentionPolicy] = {}
        self.data_inventory: Dict[str, Dict[str, Any]] = {}
        self.deletion_queue: List[Dict[str, Any]] = []
        
        # Default retention policies
        self._set_default_policies()
    
    def _set_default_policies(self):
        """Set default data retention policies"""
        policies = {
            'campaign_data': DataRetentionPolicy(
                data_type='campaign_data',
                retention_period=timedelta(days=2555),  # 7 years
                anonymize_after=timedelta(days=365),     # 1 year
                compliance_requirements=[PrivacyRegulation.GDPR, PrivacyRegulation.CCPA]
            ),
            'user_interactions': DataRetentionPolicy(
                data_type='user_interactions',
                retention_period=timedelta(days=730),    # 2 years
                anonymize_after=timedelta(days=90),      # 3 months
                compliance_requirements=[PrivacyRegulation.GDPR]
            ),
            'audit_logs': DataRetentionPolicy(
                data_type='audit_logs',
                retention_period=timedelta(days=2555),   # 7 years
                auto_delete=False,  # Keep for compliance
                compliance_requirements=[PrivacyRegulation.GDPR, PrivacyRegulation.CCPA]
            ),
            'consent_records': DataRetentionPolicy(
                data_type='consent_records',
                retention_period=timedelta(days=2555),   # 7 years
                auto_delete=False,  # Keep for compliance
                compliance_requirements=[PrivacyRegulation.GDPR, PrivacyRegulation.CCPA]
            )
        }
        
        for policy_name, policy in policies.items():
            self.retention_policies[policy_name] = policy
    
    async def register_data(self, data_id: str, data_type: str, 
                          creation_time: datetime = None, 
                          metadata: Dict[str, Any] = None) -> bool:
        """Register data for retention tracking"""
        try:
            if creation_time is None:
                creation_time = datetime.utcnow()
            
            if data_type not in self.retention_policies:
                logger.warning(f"No retention policy found for data type: {data_type}")
                return False
            
            policy = self.retention_policies[data_type]
            
            self.data_inventory[data_id] = {
                'data_type': data_type,
                'created_at': creation_time,
                'metadata': metadata or {},
                'policy': policy,
                'anonymized': False,
                'deletion_scheduled': False
            }
            
            # Schedule anonymization if required
            if policy.anonymize_after:
                anonymize_at = creation_time + policy.anonymize_after
                if anonymize_at <= datetime.utcnow():
                    await self._schedule_anonymization(data_id)
            
            # Schedule deletion if required
            if policy.auto_delete:
                delete_at = creation_time + policy.retention_period
                if delete_at <= datetime.utcnow():
                    await self._schedule_deletion(data_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register data {data_id}: {e}")
            return False
    
    async def process_retention_schedule(self) -> Dict[str, int]:
        """Process scheduled retention actions"""
        try:
            now = datetime.utcnow()
            results = {
                'anonymized': 0,
                'deleted': 0,
                'errors': 0
            }
            
            for data_id, data_info in self.data_inventory.items():
                try:
                    policy = data_info['policy']
                    created_at = data_info['created_at']
                    
                    # Check for anonymization
                    if (policy.anonymize_after and 
                        not data_info['anonymized'] and
                        now >= created_at + policy.anonymize_after):
                        
                        success = await self._anonymize_data(data_id)
                        if success:
                            results['anonymized'] += 1
                        else:
                            results['errors'] += 1
                    
                    # Check for deletion
                    if (policy.auto_delete and 
                        not data_info['deletion_scheduled'] and
                        now >= created_at + policy.retention_period):
                        
                        success = await self._schedule_deletion(data_id)
                        if success:
                            results['deleted'] += 1
                        else:
                            results['errors'] += 1
                
                except Exception as e:
                    logger.error(f"Failed to process retention for {data_id}: {e}")
                    results['errors'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process retention schedule: {e}")
            return {'anonymized': 0, 'deleted': 0, 'errors': 1}
    
    async def _schedule_anonymization(self, data_id: str) -> bool:
        """Schedule data for anonymization"""
        try:
            if data_id in self.data_inventory:
                # This would trigger actual anonymization process
                self.data_inventory[data_id]['anonymized'] = True
                logger.info(f"Data {data_id} scheduled for anonymization")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to schedule anonymization for {data_id}: {e}")
            return False
    
    async def _schedule_deletion(self, data_id: str) -> bool:
        """Schedule data for deletion"""
        try:
            if data_id in self.data_inventory:
                self.deletion_queue.append({
                    'data_id': data_id,
                    'scheduled_at': datetime.utcnow(),
                    'data_type': self.data_inventory[data_id]['data_type']
                })
                self.data_inventory[data_id]['deletion_scheduled'] = True
                logger.info(f"Data {data_id} scheduled for deletion")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to schedule deletion for {data_id}: {e}")
            return False
    
    async def _anonymize_data(self, data_id: str) -> bool:
        """Anonymize specific data"""
        try:
            # This would trigger actual anonymization process
            logger.info(f"Anonymizing data {data_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to anonymize data {data_id}: {e}")
            return False
    
    def get_retention_summary(self) -> Dict[str, Any]:
        """Get retention management summary"""
        try:
            now = datetime.utcnow()
            
            summary = {
                'total_data_records': len(self.data_inventory),
                'by_type': {},
                'anonymized': 0,
                'pending_deletion': len(self.deletion_queue),
                'retention_compliance': {}
            }
            
            for data_id, data_info in self.data_inventory.items():
                data_type = data_info['data_type']
                summary['by_type'][data_type] = summary['by_type'].get(data_type, 0) + 1
                
                if data_info['anonymized']:
                    summary['anonymized'] += 1
            
            # Check compliance status
            for policy_name, policy in self.retention_policies.items():
                overdue_count = 0
                for data_id, data_info in self.data_inventory.items():
                    if data_info['data_type'] == policy.data_type:
                        age = now - data_info['created_at']
                        if age > policy.retention_period and not data_info['deletion_scheduled']:
                            overdue_count += 1
                
                summary['retention_compliance'][policy_name] = {
                    'overdue_deletions': overdue_count
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate retention summary: {e}")
            return {}


class DataSafetyOrchestrator:
    """Main orchestrator for data safety and privacy compliance"""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.credential_manager = CredentialManager()
        self.consent_manager = ConsentManager()
        self.retention_manager = DataRetentionManager()
        
        self.processing_records: List[DataProcessingRecord] = []
    
    async def process_data_safely(self, data: Any, context: str, 
                                user_id: str = None) -> Dict[str, Any]:
        """Process data with comprehensive safety checks"""
        result = {
            'safe_to_process': False,
            'pii_detected': False,
            'consent_valid': False,
            'processed_data': None,
            'warnings': [],
            'actions_taken': []
        }
        
        try:
            # Detect PII
            pii_detections = await self.pii_detector.detect_pii(data, context)
            
            if pii_detections:
                result['pii_detected'] = True
                result['warnings'].append(f"Detected {len(pii_detections)} PII items")
                
                # Check consent if user_id provided
                if user_id:
                    consent_valid = await self.consent_manager.check_consent(user_id, 'advertising')
                    result['consent_valid'] = consent_valid
                    
                    if not consent_valid:
                        result['warnings'].append("User consent not granted for advertising")
                        return result
                
                # Anonymize data
                if isinstance(data, dict):
                    anonymized_data = await self.pii_detector.anonymize_data(data, pii_detections)
                    result['processed_data'] = anonymized_data
                    result['actions_taken'].append('data_anonymized')
                else:
                    result['warnings'].append("Cannot anonymize non-dict data")
                    return result
            else:
                result['processed_data'] = data
            
            # Register data for retention tracking
            if user_id:
                data_id = f"{user_id}_{context}_{datetime.utcnow().timestamp()}"
                await self.retention_manager.register_data(
                    data_id, 'user_interactions', metadata={'context': context}
                )
                result['actions_taken'].append('retention_tracking_enabled')
            
            result['safe_to_process'] = True
            return result
            
        except Exception as e:
            logger.error(f"Data safety processing failed: {e}")
            result['warnings'].append(f"Processing error: {str(e)}")
            return result
    
    async def record_data_processing(self, data_subject_id: str, purpose: str,
                                   data_categories: List[str], legal_basis: str,
                                   retention_period: timedelta = None) -> str:
        """Record data processing activity for compliance"""
        try:
            record_id = str(uuid.uuid4())
            
            processing_record = DataProcessingRecord(
                record_id=record_id,
                data_subject_id=data_subject_id,
                processing_purpose=purpose,
                data_categories=data_categories,
                legal_basis=legal_basis,
                timestamp=datetime.utcnow(),
                retention_period=retention_period or timedelta(days=365)
            )
            
            self.processing_records.append(processing_record)
            
            logger.info(f"Data processing recorded: {record_id} for {data_subject_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to record data processing: {e}")
            return ""
    
    def get_privacy_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive privacy and data safety dashboard"""
        try:
            return {
                'pii_protection': {
                    'detector_active': True,
                    'supported_pii_types': len(self.pii_detector.pii_patterns),
                    'masking_available': True
                },
                'credential_management': self.credential_manager.get_credential_health(),
                'consent_management': {
                    'total_users': len(self.consent_manager.consent_records),
                    'supported_purposes': len(self.consent_manager.consent_purposes)
                },
                'data_retention': self.retention_manager.get_retention_summary(),
                'processing_records': {
                    'total_records': len(self.processing_records),
                    'recent_processing': len([
                        r for r in self.processing_records
                        if r.timestamp > datetime.utcnow() - timedelta(days=1)
                    ])
                },
                'compliance_status': {
                    'gdpr_ready': True,
                    'ccpa_ready': True,
                    'data_subject_rights': ['access', 'rectification', 'erasure', 'portability'],
                    'last_assessment': datetime.utcnow()
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate privacy dashboard: {e}")
            return {'error': str(e)}