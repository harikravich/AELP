"""
Test data generators for GAELP testing framework.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from faker import Faker


@dataclass
class PersonaTemplate:
    """Template for generating personas."""
    age_range: tuple
    genders: List[str]
    income_range: tuple
    locations: List[str]
    interests: List[str]
    engagement_likelihood: float
    conversion_rate: float


class TestDataGenerator:
    """Generate realistic test data for GAELP testing."""
    
    def __init__(self, seed: Optional[int] = None):
        self.fake = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
    
    def generate_persona_config(self, template: Optional[PersonaTemplate] = None) -> Dict[str, Any]:
        """Generate a persona configuration."""
        if not template:
            template = self._get_default_persona_template()
        
        return {
            "demographics": {
                "age_range": list(template.age_range),
                "gender": random.sample(template.genders, random.randint(1, len(template.genders))),
                "income_range": list(template.income_range),
                "location": random.sample(template.locations, random.randint(1, min(3, len(template.locations))))
            },
            "interests": random.sample(template.interests, random.randint(3, len(template.interests))),
            "behavior_patterns": {
                "engagement_likelihood": template.engagement_likelihood + random.uniform(-0.1, 0.1),
                "conversion_rate": template.conversion_rate + random.uniform(-0.02, 0.02),
                "time_to_convert": random.randint(12, 120)  # Hours
            }
        }
    
    def generate_ad_campaign(self, budget_range: tuple = (50.0, 500.0)) -> Dict[str, Any]:
        """Generate an ad campaign configuration."""
        headlines = [
            "Discover Amazing Technology",
            "Transform Your Life Today",
            "Exclusive Limited Time Offer",
            "Revolutionary New Product",
            "Join Thousands of Happy Customers",
            "Unlock Your Potential",
            "Experience the Difference",
            "Premium Quality Guaranteed"
        ]
        
        descriptions = [
            "Experience innovation like never before with our cutting-edge solution",
            "Join the movement that's changing lives around the world",
            "Don't miss out on this incredible opportunity to transform your routine",
            "Discover why millions trust our award-winning products",
            "Get started today and see results in just 30 days",
            "Premium quality meets affordable pricing in this exclusive offer"
        ]
        
        interests_categories = [
            ["fitness", "health", "wellness"],
            ["technology", "gadgets", "innovation"],
            ["fashion", "style", "beauty"],
            ["travel", "adventure", "exploration"],
            ["food", "cooking", "nutrition"],
            ["business", "entrepreneurship", "success"],
            ["education", "learning", "skills"],
            ["entertainment", "music", "movies"]
        ]
        
        daily_budget = random.uniform(*budget_range)
        
        return {
            "creative": {
                "headline": random.choice(headlines),
                "description": random.choice(descriptions),
                "image_url": f"https://cdn.gaelp.dev/ads/{self.fake.uuid4()}.jpg",
                "call_to_action": random.choice(["learn_more", "shop_now", "sign_up", "download"])
            },
            "targeting": {
                "demographics": {
                    "age_range": [random.randint(18, 25), random.randint(45, 65)],
                    "gender": random.sample(["male", "female", "non-binary"], random.randint(1, 2)),
                    "income_range": [random.randint(30000, 50000), random.randint(80000, 150000)]
                },
                "interests": random.choice(interests_categories),
                "behavioral": {
                    "purchase_intent": random.choice(["low", "medium", "high"]),
                    "device_usage": random.sample(["mobile", "desktop", "tablet"], random.randint(1, 3))
                }
            },
            "budget": {
                "daily_budget": daily_budget,
                "total_budget": daily_budget * random.randint(7, 30),  # 1-4 weeks
                "bid_strategy": random.choice(["cpc", "cpm", "cpa", "roas"]),
                "max_bid": round(random.uniform(0.5, 5.0), 2)
            }
        }
    
    def generate_agent_config(self, algorithm: str = "PPO") -> Dict[str, Any]:
        """Generate an agent configuration."""
        hyperparameters = {
            "PPO": {
                "learning_rate": random.choice([0.0001, 0.0003, 0.001]),
                "batch_size": random.choice([32, 64, 128]),
                "n_epochs": random.choice([3, 5, 10]),
                "gamma": random.uniform(0.95, 0.99),
                "gae_lambda": random.uniform(0.9, 0.98),
                "clip_range": random.uniform(0.1, 0.3)
            },
            "A2C": {
                "learning_rate": random.choice([0.0005, 0.001, 0.002]),
                "value_loss_coef": random.uniform(0.25, 0.75),
                "entropy_coef": random.uniform(0.01, 0.1),
                "gamma": random.uniform(0.95, 0.99)
            },
            "DQN": {
                "learning_rate": random.choice([0.0001, 0.0005, 0.001]),
                "buffer_size": random.choice([10000, 50000, 100000]),
                "batch_size": random.choice([32, 64, 128]),
                "target_update_interval": random.choice([1000, 5000, 10000]),
                "epsilon_decay": random.uniform(0.995, 0.9995)
            }
        }
        
        return {
            "agent_id": str(uuid.uuid4()),
            "algorithm": algorithm,
            "hyperparameters": hyperparameters.get(algorithm, hyperparameters["PPO"]),
            "environment_config": {
                "max_episodes": random.randint(500, 2000),
                "max_steps_per_episode": random.randint(50, 200),
                "reward_function": random.choice(["roas_optimized", "balanced", "conservative"])
            },
            "safety_config": {
                "max_daily_budget": random.uniform(100.0, 1000.0),
                "content_safety_enabled": True,
                "human_approval_required": random.choice([True, False])
            }
        }
    
    def generate_training_metrics(self, episodes: int = 100) -> Dict[str, Any]:
        """Generate realistic training metrics."""
        episode_rewards = []
        episode_lengths = []
        policy_losses = []
        value_losses = []
        
        # Simulate learning curve
        base_reward = 0.1
        improvement_rate = 0.01
        noise_level = 0.05
        
        for episode in range(episodes):
            # Gradual improvement with noise
            expected_reward = base_reward + (episode * improvement_rate)
            noise = random.uniform(-noise_level, noise_level)
            reward = max(0, expected_reward + noise)
            episode_rewards.append(reward)
            
            # Episode length typically decreases as agent improves (learns faster)
            base_length = 100
            length_reduction = episode * 0.1
            length = max(10, base_length - length_reduction + random.randint(-10, 10))
            episode_lengths.append(int(length))
            
            # Losses typically decrease over time
            policy_loss = max(0.001, 0.5 * (1 - episode / episodes) + random.uniform(-0.1, 0.1))
            value_loss = max(0.001, 0.8 * (1 - episode / episodes) + random.uniform(-0.15, 0.15))
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "policy_loss": policy_losses,
            "value_loss": value_losses,
            "learning_rate": 0.0003,
            "exploration_rate": max(0.01, 1.0 - (episodes / 1000)),  # Decay over time
            "convergence_score": min(0.95, max(0.1, sum(episode_rewards[-10:]) / 10))  # Last 10 episodes avg
        }
    
    def generate_performance_metrics(self, realistic: bool = True) -> Dict[str, Any]:
        """Generate campaign performance metrics."""
        if realistic:
            # Realistic industry benchmarks
            impressions = random.randint(5000, 50000)
            ctr = random.uniform(0.01, 0.05)  # 1-5% CTR
            clicks = int(impressions * ctr)
            conversion_rate = random.uniform(0.01, 0.1)  # 1-10% conversion rate
            conversions = int(clicks * conversion_rate)
            cost_per_click = random.uniform(0.5, 5.0)
            total_spend = clicks * cost_per_click
            revenue_per_conversion = random.uniform(20.0, 200.0)
            revenue = conversions * revenue_per_conversion
        else:
            # Unrealistic but valid for testing edge cases
            impressions = random.randint(1, 1000000)
            clicks = random.randint(0, impressions)
            conversions = random.randint(0, clicks)
            total_spend = random.uniform(1.0, 10000.0)
            revenue = random.uniform(0.0, 50000.0)
            ctr = clicks / impressions if impressions > 0 else 0
            conversion_rate = conversions / clicks if clicks > 0 else 0
            cost_per_click = total_spend / clicks if clicks > 0 else 0
        
        return {
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "ctr": round(ctr, 4),
            "conversion_rate": round(conversion_rate, 4),
            "cost_per_click": round(cost_per_click, 2),
            "cost_per_conversion": round(total_spend / conversions, 2) if conversions > 0 else 0,
            "return_on_ad_spend": round(revenue / total_spend, 2) if total_spend > 0 else 0,
            "total_spend": round(total_spend, 2),
            "revenue": round(revenue, 2)
        }
    
    def generate_safety_violation(self, violation_type: str = None) -> Dict[str, Any]:
        """Generate a safety violation event."""
        violation_types = [
            "budget_exceeded",
            "inappropriate_content",
            "targeting_violation",
            "policy_breach",
            "suspicious_activity"
        ]
        
        if not violation_type:
            violation_type = random.choice(violation_types)
        
        violation_details = {
            "budget_exceeded": {
                "requested_budget": random.uniform(1000.0, 5000.0),
                "max_allowed": random.uniform(500.0, 1000.0),
                "overage_percentage": random.uniform(0.1, 2.0)
            },
            "inappropriate_content": {
                "content_type": random.choice(["headline", "description", "image"]),
                "safety_score": random.uniform(0.1, 0.4),
                "flagged_categories": random.sample(["hate_speech", "violence", "adult_content"], 2)
            },
            "targeting_violation": {
                "violation_category": random.choice(["age_restriction", "location_restriction", "sensitive_category"]),
                "attempted_targeting": "minors under 18"
            },
            "policy_breach": {
                "policy_type": random.choice(["advertising_standards", "platform_rules", "legal_compliance"]),
                "severity": random.choice(["low", "medium", "high", "critical"])
            },
            "suspicious_activity": {
                "activity_type": random.choice(["unusual_spending_pattern", "rapid_budget_changes", "account_compromise"]),
                "confidence_score": random.uniform(0.7, 0.95)
            }
        }
        
        return {
            "violation_type": violation_type,
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": str(uuid.uuid4()),
            "details": violation_details.get(violation_type, {})
        }
    
    def generate_environment_state(self, environment_type: str = "simulated") -> Dict[str, Any]:
        """Generate environment state data."""
        if environment_type == "simulated":
            return {
                "market_context": {
                    "competition_level": random.uniform(0.3, 0.9),
                    "market_saturation": random.uniform(0.2, 0.8),
                    "seasonal_factor": random.uniform(0.8, 1.3),
                    "economic_indicator": random.uniform(0.9, 1.1)
                },
                "audience_size": random.randint(100000, 10000000),
                "available_budget": random.uniform(1000.0, 50000.0),
                "environment_type": environment_type,
                "noise_level": random.uniform(0.01, 0.1),
                "simulation_parameters": {
                    "user_behavior_model": random.choice(["basic", "advanced", "realistic"]),
                    "market_dynamics": random.choice(["static", "dynamic", "volatile"]),
                    "feedback_delay": random.randint(1, 24)  # Hours
                }
            }
        else:  # real environment
            return {
                "platform": random.choice(["meta_ads", "google_ads", "linkedin_ads"]),
                "account_id": str(uuid.uuid4()),
                "available_budget": random.uniform(500.0, 10000.0),
                "environment_type": environment_type,
                "api_limits": {
                    "requests_per_hour": random.randint(1000, 10000),
                    "daily_budget_limit": random.uniform(1000.0, 50000.0)
                },
                "platform_constraints": {
                    "min_daily_budget": random.uniform(1.0, 50.0),
                    "max_daily_budget": random.uniform(1000.0, 10000.0),
                    "approved_ad_accounts": random.randint(1, 10)
                }
            }
    
    def generate_user_feedback(self, sample_size: int = 1000) -> List[Dict[str, Any]]:
        """Generate simulated user feedback data."""
        feedback_data = []
        
        for _ in range(sample_size):
            response_type = random.choices(
                ["impression", "click", "conversion", "ignore"],
                weights=[70, 20, 5, 5],  # Realistic response distribution
                k=1
            )[0]
            
            engagement_score = {
                "impression": random.uniform(0.1, 0.4),
                "click": random.uniform(0.4, 0.8),
                "conversion": random.uniform(0.8, 1.0),
                "ignore": random.uniform(0.0, 0.2)
            }[response_type]
            
            sentiment = random.choices(
                ["positive", "neutral", "negative"],
                weights=[40, 50, 10],  # Mostly neutral/positive
                k=1
            )[0]
            
            feedback_text = None
            if random.random() < 0.1:  # 10% provide text feedback
                feedback_templates = {
                    "positive": ["Great product!", "Love this!", "Exactly what I needed"],
                    "neutral": ["Looks interesting", "Maybe later", "Not sure yet"],
                    "negative": ["Not for me", "Too expensive", "Seen better"]
                }
                feedback_text = random.choice(feedback_templates[sentiment])
            
            feedback_data.append({
                "user_id": str(uuid.uuid4()),
                "response_type": response_type,
                "engagement_score": round(engagement_score, 3),
                "sentiment": sentiment,
                "feedback_text": feedback_text,
                "timestamp": (datetime.utcnow() - timedelta(
                    minutes=random.randint(0, 1440)  # Last 24 hours
                )).isoformat(),
                "user_demographics": {
                    "age_group": random.choice(["18-24", "25-34", "35-44", "45-54", "55+"]),
                    "gender": random.choice(["male", "female", "non-binary"]),
                    "location": random.choice(["US", "CA", "UK", "AU", "DE"])
                }
            })
        
        return feedback_data
    
    def generate_load_test_data(self, users: int = 100) -> List[Dict[str, Any]]:
        """Generate data for load testing scenarios."""
        load_scenarios = []
        
        for user_id in range(users):
            scenario = {
                "user_id": f"load_test_user_{user_id}",
                "agent_config": self.generate_agent_config(),
                "campaigns": [
                    self.generate_ad_campaign() 
                    for _ in range(random.randint(1, 5))
                ],
                "persona_configs": [
                    self.generate_persona_config()
                    for _ in range(random.randint(1, 3))
                ],
                "request_sequence": [
                    random.choice([
                        "create_agent",
                        "start_training",
                        "check_status",
                        "create_campaign",
                        "get_metrics"
                    ]) for _ in range(random.randint(10, 50))
                ]
            }
            load_scenarios.append(scenario)
        
        return load_scenarios
    
    def _get_default_persona_template(self) -> PersonaTemplate:
        """Get default persona template."""
        return PersonaTemplate(
            age_range=(25, 45),
            genders=["male", "female", "non-binary"],
            income_range=(40000, 100000),
            locations=["US", "CA", "UK", "AU", "DE"],
            interests=["technology", "fitness", "travel", "food", "entertainment", "education"],
            engagement_likelihood=0.3,
            conversion_rate=0.05
        )


class SpecializedDataGenerators:
    """Specialized data generators for specific testing scenarios."""
    
    @staticmethod
    def generate_edge_case_budgets() -> List[Dict[str, Any]]:
        """Generate edge case budget configurations."""
        return [
            {"daily_budget": 0.01, "total_budget": 0.07},  # Minimum budgets
            {"daily_budget": 0.0, "total_budget": 0.0},    # Zero budgets
            {"daily_budget": 999999.0, "total_budget": 9999999.0},  # Maximum budgets
            {"daily_budget": 100.0, "total_budget": 50.0},  # Total < Daily (invalid)
            {"daily_budget": -10.0, "total_budget": 100.0},  # Negative daily
            {"daily_budget": 100.0, "total_budget": -50.0},  # Negative total
        ]
    
    @staticmethod
    def generate_stress_test_configs() -> List[Dict[str, Any]]:
        """Generate configurations for stress testing."""
        return [
            {
                "concurrent_agents": 100,
                "episodes_per_agent": 1000,
                "environment_complexity": "high",
                "data_collection_frequency": "every_step"
            },
            {
                "concurrent_agents": 50,
                "episodes_per_agent": 5000,
                "environment_complexity": "medium",
                "data_collection_frequency": "every_episode"
            },
            {
                "concurrent_agents": 10,
                "episodes_per_agent": 10000,
                "environment_complexity": "maximum",
                "data_collection_frequency": "every_step"
            }
        ]
    
    @staticmethod
    def generate_adversarial_inputs() -> List[Dict[str, Any]]:
        """Generate adversarial inputs for robustness testing."""
        return [
            {
                "campaign": {
                    "creative": {
                        "headline": "A" * 1000,  # Very long headline
                        "description": "B" * 5000  # Very long description
                    }
                }
            },
            {
                "campaign": {
                    "creative": {
                        "headline": "",  # Empty headline
                        "description": ""  # Empty description
                    }
                }
            },
            {
                "campaign": {
                    "creative": {
                        "headline": "Test\x00\x01\x02",  # Control characters
                        "description": "Test\n\r\t"  # Whitespace characters
                    }
                }
            },
            {
                "campaign": {
                    "targeting": {
                        "demographics": {
                            "age_range": [0, 200],  # Invalid age range
                            "income_range": [-1000000, 1000000]  # Invalid income
                        }
                    }
                }
            }
        ]