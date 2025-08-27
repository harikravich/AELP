"""
Validation utilities for GAELP testing framework.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import jsonschema
from jsonschema import validate, ValidationError


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 0-1 validation score


class SchemaValidator:
    """JSON schema validation for API requests and responses."""
    
    def __init__(self):
        self.schemas = {
            "agent_config": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "format": "uuid"},
                    "algorithm": {"type": "string", "enum": ["PPO", "A2C", "DQN", "SAC"]},
                    "hyperparameters": {
                        "type": "object",
                        "properties": {
                            "learning_rate": {"type": "number", "minimum": 0.00001, "maximum": 0.1},
                            "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
                            "gamma": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["learning_rate"]
                    },
                    "environment_config": {
                        "type": "object",
                        "properties": {
                            "max_episodes": {"type": "integer", "minimum": 1, "maximum": 100000},
                            "max_steps_per_episode": {"type": "integer", "minimum": 1, "maximum": 10000}
                        }
                    },
                    "safety_config": {
                        "type": "object",
                        "properties": {
                            "max_daily_budget": {"type": "number", "minimum": 0.01, "maximum": 100000},
                            "content_safety_enabled": {"type": "boolean"},
                            "human_approval_required": {"type": "boolean"}
                        }
                    }
                },
                "required": ["algorithm", "hyperparameters"]
            },
            
            "persona_config": {
                "type": "object",
                "properties": {
                    "demographics": {
                        "type": "object",
                        "properties": {
                            "age_range": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 13, "maximum": 100},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "gender": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["male", "female", "non-binary", "all"]},
                                "minItems": 1
                            },
                            "income_range": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 1000000},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "location": {
                                "type": "array",
                                "items": {"type": "string", "pattern": "^[A-Z]{2}$"},
                                "minItems": 1
                            }
                        },
                        "required": ["age_range", "gender"]
                    },
                    "interests": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 50},
                        "minItems": 1,
                        "maxItems": 20
                    },
                    "behavior_patterns": {
                        "type": "object",
                        "properties": {
                            "engagement_likelihood": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "conversion_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "time_to_convert": {"type": "integer", "minimum": 1, "maximum": 8760}
                        }
                    }
                },
                "required": ["demographics", "interests"]
            },
            
            "ad_campaign": {
                "type": "object",
                "properties": {
                    "creative": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string", "minLength": 1, "maxLength": 100},
                            "description": {"type": "string", "minLength": 1, "maxLength": 500},
                            "image_url": {"type": "string", "format": "uri"},
                            "video_url": {"type": "string", "format": "uri"},
                            "call_to_action": {
                                "type": "string",
                                "enum": ["learn_more", "shop_now", "sign_up", "download", "get_quote"]
                            }
                        },
                        "required": ["headline", "description", "call_to_action"]
                    },
                    "targeting": {
                        "type": "object",
                        "properties": {
                            "demographics": {"$ref": "#/definitions/demographics"},
                            "interests": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1
                            },
                            "behavioral": {
                                "type": "object",
                                "properties": {
                                    "purchase_intent": {"type": "string", "enum": ["low", "medium", "high"]},
                                    "device_usage": {
                                        "type": "array",
                                        "items": {"type": "string", "enum": ["mobile", "desktop", "tablet"]},
                                        "minItems": 1
                                    }
                                }
                            }
                        }
                    },
                    "budget": {
                        "type": "object",
                        "properties": {
                            "daily_budget": {"type": "number", "minimum": 0.01, "maximum": 100000},
                            "total_budget": {"type": "number", "minimum": 0.01, "maximum": 1000000},
                            "bid_strategy": {"type": "string", "enum": ["cpc", "cpm", "cpa", "roas"]},
                            "max_bid": {"type": "number", "minimum": 0.01, "maximum": 1000}
                        },
                        "required": ["daily_budget", "bid_strategy"]
                    }
                },
                "required": ["creative", "budget"],
                "definitions": {
                    "demographics": {
                        "type": "object",
                        "properties": {
                            "age_range": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 13, "maximum": 100},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "gender": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["male", "female", "non-binary", "all"]}
                            }
                        }
                    }
                }
            },
            
            "performance_metrics": {
                "type": "object",
                "properties": {
                    "impressions": {"type": "integer", "minimum": 0},
                    "clicks": {"type": "integer", "minimum": 0},
                    "conversions": {"type": "integer", "minimum": 0},
                    "ctr": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "conversion_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "cost_per_click": {"type": "number", "minimum": 0.0},
                    "cost_per_conversion": {"type": "number", "minimum": 0.0},
                    "return_on_ad_spend": {"type": "number", "minimum": 0.0},
                    "total_spend": {"type": "number", "minimum": 0.0},
                    "revenue": {"type": "number", "minimum": 0.0}
                },
                "required": ["impressions", "clicks", "conversions", "total_spend"]
            }
        }
    
    def validate_schema(self, data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """Validate data against a schema."""
        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown schema: {schema_name}"],
                warnings=[],
                score=0.0
            )
        
        schema = self.schemas[schema_name]
        errors = []
        warnings = []
        
        try:
            validate(instance=data, schema=schema)
            is_valid = True
        except ValidationError as e:
            is_valid = False
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            is_valid = False
            errors.append(f"Validation exception: {str(e)}")
        
        # Calculate validation score
        score = 1.0 if is_valid else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=score
        )


class BusinessLogicValidator:
    """Validate business logic constraints."""
    
    def validate_budget_constraints(self, campaign: Dict[str, Any], agent_limits: Dict[str, Any]) -> ValidationResult:
        """Validate budget constraints."""
        errors = []
        warnings = []
        
        budget = campaign.get("budget", {})
        daily_budget = budget.get("daily_budget", 0)
        total_budget = budget.get("total_budget", 0)
        
        # Check basic budget logic
        if total_budget > 0 and daily_budget > total_budget:
            errors.append("Daily budget cannot exceed total budget")
        
        # Check against agent limits
        max_daily = agent_limits.get("max_daily_budget", float('inf'))
        if daily_budget > max_daily:
            errors.append(f"Daily budget ${daily_budget} exceeds limit ${max_daily}")
        
        max_total = agent_limits.get("max_total_budget", float('inf'))
        if total_budget > max_total:
            errors.append(f"Total budget ${total_budget} exceeds limit ${max_total}")
        
        # Warnings for unusual budgets
        if daily_budget < 1.0:
            warnings.append("Very low daily budget may limit campaign effectiveness")
        
        if daily_budget > 1000.0:
            warnings.append("High daily budget - ensure this is intentional")
        
        is_valid = len(errors) == 0
        score = 1.0 if is_valid else (1.0 - len(errors) / 5.0)  # Penalty per error
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0.0, score)
        )
    
    def validate_targeting_constraints(self, targeting: Dict[str, Any]) -> ValidationResult:
        """Validate targeting constraints."""
        errors = []
        warnings = []
        
        demographics = targeting.get("demographics", {})
        
        # Validate age range
        age_range = demographics.get("age_range", [])
        if len(age_range) == 2:
            min_age, max_age = age_range
            if min_age < 13:
                errors.append("Cannot target users under 13 years old")
            if min_age >= max_age:
                errors.append("Age range minimum must be less than maximum")
            if max_age - min_age > 50:
                warnings.append("Very broad age range may reduce targeting effectiveness")
        
        # Validate income range
        income_range = demographics.get("income_range", [])
        if len(income_range) == 2:
            min_income, max_income = income_range
            if min_income < 0:
                errors.append("Income range cannot be negative")
            if max_income < min_income:
                errors.append("Income range maximum must be greater than minimum")
        
        # Check for restricted targeting
        interests = targeting.get("interests", [])
        restricted_keywords = ["gambling", "alcohol", "tobacco", "weapons", "adult"]
        for interest in interests:
            if any(keyword in interest.lower() for keyword in restricted_keywords):
                warnings.append(f"Interest '{interest}' may have platform restrictions")
        
        is_valid = len(errors) == 0
        score = 1.0 if is_valid else (1.0 - len(errors) / 3.0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0.0, score)
        )
    
    def validate_creative_content(self, creative: Dict[str, Any]) -> ValidationResult:
        """Validate creative content."""
        errors = []
        warnings = []
        
        headline = creative.get("headline", "")
        description = creative.get("description", "")
        
        # Basic content validation
        if not headline.strip():
            errors.append("Headline cannot be empty")
        
        if not description.strip():
            errors.append("Description cannot be empty")
        
        # Content quality checks
        if len(headline) > 100:
            errors.append("Headline too long (max 100 characters)")
        
        if len(description) > 500:
            errors.append("Description too long (max 500 characters)")
        
        # Language and content checks
        if headline.isupper():
            warnings.append("All caps headline may appear spammy")
        
        # Check for excessive punctuation
        if headline.count('!') > 2:
            warnings.append("Excessive exclamation marks in headline")
        
        # Check for promotional language
        promotional_words = ["free", "urgent", "limited time", "act now", "guaranteed"]
        promotional_count = sum(1 for word in promotional_words if word in headline.lower())
        if promotional_count > 2:
            warnings.append("High promotional language may affect deliverability")
        
        # URL validation
        image_url = creative.get("image_url", "")
        if image_url and not self._is_valid_url(image_url):
            errors.append("Invalid image URL format")
        
        video_url = creative.get("video_url", "")
        if video_url and not self._is_valid_url(video_url):
            errors.append("Invalid video URL format")
        
        is_valid = len(errors) == 0
        score = 1.0 if is_valid else (1.0 - len(errors) / 4.0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0.0, score)
        )
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None


class PerformanceValidator:
    """Validate performance metrics and training results."""
    
    def validate_training_metrics(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Validate training metrics."""
        errors = []
        warnings = []
        
        episode_rewards = metrics.get("episode_rewards", [])
        policy_loss = metrics.get("policy_loss", [])
        convergence_score = metrics.get("convergence_score", 0)
        
        # Check for valid reward progression
        if not episode_rewards:
            errors.append("No episode rewards data")
        elif len(episode_rewards) < 10:
            warnings.append("Insufficient training data for reliable metrics")
        else:
            # Check for learning progress
            early_rewards = episode_rewards[:len(episode_rewards)//3]
            late_rewards = episode_rewards[-len(episode_rewards)//3:]
            
            early_avg = sum(early_rewards) / len(early_rewards)
            late_avg = sum(late_rewards) / len(late_rewards)
            
            if late_avg <= early_avg:
                warnings.append("No improvement in average rewards over training")
            
            # Check for extreme values
            if any(r < -10 or r > 10 for r in episode_rewards):
                warnings.append("Extreme reward values detected")
        
        # Validate convergence
        if convergence_score < 0 or convergence_score > 1:
            errors.append("Convergence score must be between 0 and 1")
        elif convergence_score < 0.5:
            warnings.append("Low convergence score indicates poor training")
        
        # Check loss trends
        if policy_loss and len(policy_loss) > 10:
            if policy_loss[-1] > policy_loss[0]:
                warnings.append("Policy loss increasing - possible training instability")
        
        is_valid = len(errors) == 0
        score = convergence_score if is_valid else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=score
        )
    
    def validate_campaign_performance(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Validate campaign performance metrics."""
        errors = []
        warnings = []
        
        impressions = metrics.get("impressions", 0)
        clicks = metrics.get("clicks", 0)
        conversions = metrics.get("conversions", 0)
        total_spend = metrics.get("total_spend", 0)
        revenue = metrics.get("revenue", 0)
        
        # Basic consistency checks
        if clicks > impressions:
            errors.append("Clicks cannot exceed impressions")
        
        if conversions > clicks:
            errors.append("Conversions cannot exceed clicks")
        
        # Calculate derived metrics and validate
        ctr = clicks / impressions if impressions > 0 else 0
        conversion_rate = conversions / clicks if clicks > 0 else 0
        roas = revenue / total_spend if total_spend > 0 else 0
        
        # Industry benchmark validation
        if ctr > 0.2:  # 20% CTR is unusually high
            warnings.append("Unusually high click-through rate")
        elif ctr < 0.001:  # 0.1% CTR is very low
            warnings.append("Very low click-through rate")
        
        if conversion_rate > 0.5:  # 50% conversion rate is unusually high
            warnings.append("Unusually high conversion rate")
        
        if roas < 1.0 and total_spend > 0:
            warnings.append("Negative return on ad spend")
        elif roas > 20.0:
            warnings.append("Unusually high return on ad spend")
        
        # Validate provided vs calculated metrics
        provided_ctr = metrics.get("ctr", ctr)
        if abs(provided_ctr - ctr) > 0.001:  # Allow for rounding
            errors.append("Provided CTR doesn't match calculated value")
        
        provided_conversion_rate = metrics.get("conversion_rate", conversion_rate)
        if abs(provided_conversion_rate - conversion_rate) > 0.001:
            errors.append("Provided conversion rate doesn't match calculated value")
        
        is_valid = len(errors) == 0
        
        # Score based on performance quality
        score = 0.0
        if is_valid:
            # Score based on industry benchmarks
            ctr_score = min(1.0, ctr / 0.03)  # 3% CTR as good benchmark
            conversion_score = min(1.0, conversion_rate / 0.05)  # 5% as good benchmark
            roas_score = min(1.0, roas / 3.0)  # 3x ROAS as good benchmark
            
            score = (ctr_score + conversion_score + roas_score) / 3.0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=score
        )


class SafetyValidator:
    """Validate safety and compliance requirements."""
    
    def validate_content_safety(self, content: Dict[str, Any]) -> ValidationResult:
        """Validate content safety."""
        errors = []
        warnings = []
        
        headline = content.get("headline", "")
        description = content.get("description", "")
        
        # Check for prohibited content
        prohibited_patterns = [
            r'\b(?:hate|violence|weapon|drug)\b',
            r'\b(?:discriminat|racist|sexist)\b',
            r'\b(?:illegal|fraud|scam)\b'
        ]
        
        all_text = f"{headline} {description}".lower()
        
        for pattern in prohibited_patterns:
            if re.search(pattern, all_text):
                errors.append(f"Potentially prohibited content detected: {pattern}")
        
        # Check for excessive promotional language
        promotional_patterns = [
            r'\b(?:free|urgent|limited|act now|guaranteed)\b'
        ]
        
        promotional_count = 0
        for pattern in promotional_patterns:
            promotional_count += len(re.findall(pattern, all_text))
        
        if promotional_count > 3:
            warnings.append("High promotional language may affect approval")
        
        # Check for personal information
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email pattern
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, all_text):
                warnings.append("Potential personal information detected")
        
        is_valid = len(errors) == 0
        score = 1.0 if is_valid else (1.0 - len(errors) / 3.0)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=max(0.0, score)
        )
    
    def validate_budget_safety(self, budget_request: Dict[str, Any], current_spend: Dict[str, Any]) -> ValidationResult:
        """Validate budget safety constraints."""
        errors = []
        warnings = []
        
        daily_budget = budget_request.get("daily_budget", 0)
        total_budget = budget_request.get("total_budget", 0)
        
        current_daily = current_spend.get("daily_spent", 0)
        current_total = current_spend.get("total_spent", 0)
        
        # Check for spending limits
        if current_daily + daily_budget > 10000:  # $10k daily limit
            errors.append("Proposed spending exceeds daily safety limit")
        
        if current_total + total_budget > 100000:  # $100k total limit
            errors.append("Proposed spending exceeds total safety limit")
        
        # Check for unusual spending patterns
        if daily_budget > current_daily * 10:
            warnings.append("Large increase in daily budget")
        
        if daily_budget > 0 and current_daily / daily_budget < 0.1:
            warnings.append("Significant budget reduction")
        
        is_valid = len(errors) == 0
        score = 1.0 if is_valid else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=score
        )


class ComprehensiveValidator:
    """Comprehensive validator combining all validation types."""
    
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.business_validator = BusinessLogicValidator()
        self.performance_validator = PerformanceValidator()
        self.safety_validator = SafetyValidator()
    
    def validate_campaign_request(self, campaign: Dict[str, Any], agent_limits: Dict[str, Any]) -> ValidationResult:
        """Comprehensive campaign validation."""
        all_errors = []
        all_warnings = []
        scores = []
        
        # Schema validation
        schema_result = self.schema_validator.validate_schema(campaign, "ad_campaign")
        all_errors.extend(schema_result.errors)
        all_warnings.extend(schema_result.warnings)
        scores.append(schema_result.score)
        
        if schema_result.is_valid:
            # Business logic validation
            budget_result = self.business_validator.validate_budget_constraints(campaign, agent_limits)
            all_errors.extend(budget_result.errors)
            all_warnings.extend(budget_result.warnings)
            scores.append(budget_result.score)
            
            targeting_result = self.business_validator.validate_targeting_constraints(campaign.get("targeting", {}))
            all_errors.extend(targeting_result.errors)
            all_warnings.extend(targeting_result.warnings)
            scores.append(targeting_result.score)
            
            creative_result = self.business_validator.validate_creative_content(campaign.get("creative", {}))
            all_errors.extend(creative_result.errors)
            all_warnings.extend(creative_result.warnings)
            scores.append(creative_result.score)
            
            # Safety validation
            safety_result = self.safety_validator.validate_content_safety(campaign.get("creative", {}))
            all_errors.extend(safety_result.errors)
            all_warnings.extend(safety_result.warnings)
            scores.append(safety_result.score)
        
        is_valid = len(all_errors) == 0
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            score=overall_score
        )
    
    def validate_training_results(self, training_data: Dict[str, Any]) -> ValidationResult:
        """Comprehensive training validation."""
        all_errors = []
        all_warnings = []
        
        # Validate training metrics
        metrics_result = self.performance_validator.validate_training_metrics(training_data)
        all_errors.extend(metrics_result.errors)
        all_warnings.extend(metrics_result.warnings)
        
        # Additional training-specific validations could go here
        
        is_valid = len(all_errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            score=metrics_result.score
        )