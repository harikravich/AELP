"""
Persona Factory for GAELP
Generates diverse, realistic user personas for LLM-powered ad testing
"""

import random
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass

from llm_persona_service import PersonaConfig, PersonaDemographics, PersonaPsychology, PersonaHistory, PersonaState


class PersonaFactory:
    """Factory for creating diverse, realistic personas"""
    
    # Demographic data for realistic persona generation
    FIRST_NAMES = {
        "male": ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Christopher",
                "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin"],
        "female": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
                  "Nancy", "Lisa", "Betty", "Helen", "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle"],
        "non-binary": ["Alex", "Jordan", "Casey", "Taylor", "Morgan", "Riley", "Avery", "Sage", "Quinn", "River"]
    }
    
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
                  "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
    
    CITIES = [
        "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ", "Philadelphia, PA",
        "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL",
        "Fort Worth, TX", "Columbus, OH", "Charlotte, NC", "San Francisco, CA", "Indianapolis, IN", "Seattle, WA",
        "Denver, CO", "Washington, DC", "Boston, MA", "El Paso, TX", "Nashville, TN", "Detroit, MI", "Portland, OR"
    ]
    
    EMPLOYMENT_TYPES = {
        "student": ["college student", "graduate student", "high school student"],
        "employed": ["software engineer", "teacher", "nurse", "manager", "consultant", "designer", "analyst",
                    "accountant", "lawyer", "doctor", "sales representative", "marketing specialist"],
        "unemployed": ["job seeker", "between jobs", "freelancer"],
        "retired": ["retiree", "former teacher", "former engineer", "former nurse"]
    }
    
    INTERESTS_BY_AGE = {
        (18, 25): ["gaming", "social media", "music festivals", "travel", "fitness", "fashion", "entertainment",
                  "technology", "nightlife", "sports", "college sports", "memes", "dating apps", "streaming"],
        (26, 35): ["career development", "technology", "travel", "fitness", "cooking", "wine", "networking",
                  "home improvement", "investing", "yoga", "podcasts", "streaming", "outdoor activities"],
        (36, 50): ["family", "home ownership", "investing", "health", "parenting", "education", "travel",
                  "cooking", "wine", "home improvement", "gardening", "book clubs", "fitness", "career"],
        (51, 65): ["health", "travel", "grandchildren", "hobbies", "investing", "home improvement", "gardening",
                  "book clubs", "volunteering", "history", "culture", "wine", "cooking", "fitness"],
        (66, 100): ["health", "grandchildren", "hobbies", "travel", "volunteering", "gardening", "history",
                   "culture", "religion", "community", "family", "reading", "crafts", "nature"]
    }
    
    VALUES_BY_GENERATION = {
        "gen_z": ["authenticity", "social justice", "environmental sustainability", "mental health", "diversity",
                 "work-life balance", "creativity", "technology", "global connectivity"],
        "millennial": ["experiences over possessions", "work-life balance", "social responsibility", "technology",
                      "personal growth", "flexibility", "authenticity", "diversity", "sustainability"],
        "gen_x": ["independence", "self-reliance", "work-life balance", "family", "financial security",
                 "practicality", "skepticism", "diversity", "technology adoption"],
        "boomer": ["hard work", "family values", "financial security", "stability", "respect for authority",
                  "community", "tradition", "patriotism", "face-to-face communication"]
    }
    
    @classmethod
    def create_random_persona(cls) -> PersonaConfig:
        """Create a completely random persona"""
        
        # Basic demographics
        gender = random.choice(["male", "female", "non-binary"])
        age = random.randint(18, 80)
        first_name = random.choice(cls.FIRST_NAMES[gender])
        last_name = random.choice(cls.LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        # Age-based characteristics
        if age < 26:
            generation = "gen_z"
            typical_employment = "student" if age < 23 else "employed"
            income_weights = [0.6, 0.3, 0.1, 0.0]  # low, medium, high, very_high
        elif age < 41:
            generation = "millennial"
            typical_employment = "employed"
            income_weights = [0.2, 0.4, 0.3, 0.1]
        elif age < 57:
            generation = "gen_x"
            typical_employment = "employed"
            income_weights = [0.1, 0.3, 0.4, 0.2]
        else:
            generation = "boomer"
            typical_employment = "retired" if age > 65 else "employed"
            income_weights = [0.1, 0.4, 0.3, 0.2]
        
        # Demographics
        demographics = PersonaDemographics(
            age=age,
            gender=gender,
            income_level=random.choices(
                ["low", "medium", "high", "very_high"],
                weights=income_weights
            )[0],
            education=random.choices(
                ["high_school", "college", "graduate", "phd"],
                weights=[0.3, 0.4, 0.25, 0.05]
            )[0],
            location=random.choice(cls.CITIES),
            employment=cls._choose_employment(age, typical_employment),
            relationship_status=random.choices(
                ["single", "dating", "married", "divorced", "widowed"],
                weights=[0.3, 0.2, 0.4, 0.08, 0.02]
            )[0],
            household_size=random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.15, 0.05])[0]
        )
        
        # Psychology
        psychology = cls._create_psychology(age, generation)
        
        # Create persona
        persona_id = str(uuid.uuid4())
        persona = PersonaConfig(
            persona_id=persona_id,
            name=name,
            demographics=demographics,
            psychology=psychology
        )
        
        return persona
    
    @classmethod
    def create_targeted_persona(cls, target_profile: Dict[str, Any]) -> PersonaConfig:
        """Create a persona matching specific targeting criteria"""
        
        # Extract targeting criteria
        age_range = target_profile.get("age_range", (18, 80))
        gender = target_profile.get("gender", random.choice(["male", "female", "non-binary"]))
        interests = target_profile.get("interests", [])
        income_level = target_profile.get("income_level")
        location = target_profile.get("location")
        
        # Generate age within range
        age = random.randint(age_range[0], age_range[1])
        
        # Generate name
        first_name = random.choice(cls.FIRST_NAMES[gender])
        last_name = random.choice(cls.LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        # Demographics
        demographics = PersonaDemographics(
            age=age,
            gender=gender,
            income_level=income_level or cls._choose_income_for_age(age),
            education=random.choices(
                ["high_school", "college", "graduate", "phd"],
                weights=[0.3, 0.4, 0.25, 0.05]
            )[0],
            location=location or random.choice(cls.CITIES),
            employment=cls._choose_employment_for_age(age),
            relationship_status=random.choices(
                ["single", "dating", "married", "divorced", "widowed"],
                weights=[0.3, 0.2, 0.4, 0.08, 0.02]
            )[0],
            household_size=random.choices([1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.2, 0.15, 0.05])[0]
        )
        
        # Psychology with targeted interests
        generation = cls._get_generation(age)
        psychology = cls._create_psychology(age, generation, target_interests=interests)
        
        # Create persona
        persona_id = str(uuid.uuid4())
        persona = PersonaConfig(
            persona_id=persona_id,
            name=name,
            demographics=demographics,
            psychology=psychology
        )
        
        return persona
    
    @classmethod
    def create_persona_cohort(cls, cohort_size: int, diversity_level: str = "high") -> List[PersonaConfig]:
        """
        Create a cohort of diverse personas
        
        Args:
            cohort_size: Number of personas to create
            diversity_level: "low", "medium", "high" - controls demographic diversity
        """
        
        personas = []
        
        if diversity_level == "low":
            # Similar demographics, some variation in psychology
            base_age = random.randint(25, 45)
            base_gender = random.choice(["male", "female"])
            base_income = random.choice(["medium", "high"])
            
            for _ in range(cohort_size):
                age = base_age + random.randint(-5, 5)
                persona = cls.create_targeted_persona({
                    "age_range": (max(18, age-2), min(80, age+2)),
                    "gender": base_gender if random.random() < 0.8 else random.choice(["male", "female", "non-binary"]),
                    "income_level": base_income if random.random() < 0.7 else random.choice(["low", "medium", "high"])
                })
                personas.append(persona)
        
        elif diversity_level == "medium":
            # Moderate diversity across key dimensions
            age_groups = [(18, 30), (31, 45), (46, 65), (66, 80)]
            genders = ["male", "female", "non-binary"]
            
            for i in range(cohort_size):
                age_range = age_groups[i % len(age_groups)]
                gender = genders[i % len(genders)]
                
                persona = cls.create_targeted_persona({
                    "age_range": age_range,
                    "gender": gender
                })
                personas.append(persona)
        
        else:  # high diversity
            # Maximum diversity
            for _ in range(cohort_size):
                persona = cls.create_random_persona()
                personas.append(persona)
        
        return personas
    
    @classmethod
    def create_campaign_audience(cls, campaign_profile: Dict[str, Any], audience_size: int) -> List[PersonaConfig]:
        """Create an audience specifically for a campaign"""
        
        target_demographics = campaign_profile.get("target_demographics", {})
        product_category = campaign_profile.get("product_category", "general")
        budget_level = campaign_profile.get("budget_level", "medium")
        
        # Determine audience composition based on campaign
        if product_category == "luxury":
            income_distribution = [0.1, 0.2, 0.4, 0.3]  # Skew towards high income
        elif product_category == "budget":
            income_distribution = [0.5, 0.3, 0.15, 0.05]  # Skew towards low income
        else:
            income_distribution = [0.25, 0.35, 0.25, 0.15]  # Balanced
        
        personas = []
        for _ in range(audience_size):
            # Create persona with campaign-specific targeting
            income_level = random.choices(
                ["low", "medium", "high", "very_high"],
                weights=income_distribution
            )[0]
            
            target_profile = {
                **target_demographics,
                "income_level": income_level,
                "interests": cls._get_interests_for_category(product_category)
            }
            
            persona = cls.create_targeted_persona(target_profile)
            personas.append(persona)
        
        return personas
    
    @classmethod
    def _choose_employment(cls, age: int, typical_employment: str) -> str:
        """Choose employment based on age and typical patterns"""
        if age < 22:
            return random.choice(cls.EMPLOYMENT_TYPES["student"])
        elif age > 65:
            return random.choice(cls.EMPLOYMENT_TYPES["retired"])
        elif 22 <= age <= 65:
            if random.random() < 0.85:  # 85% employed
                return random.choice(cls.EMPLOYMENT_TYPES["employed"])
            else:
                return random.choice(cls.EMPLOYMENT_TYPES["unemployed"])
        else:
            return random.choice(cls.EMPLOYMENT_TYPES["employed"])
    
    @classmethod
    def _choose_employment_for_age(cls, age: int) -> str:
        """Choose realistic employment for age"""
        if age < 22:
            return random.choice(cls.EMPLOYMENT_TYPES["student"])
        elif age > 65:
            return random.choice(cls.EMPLOYMENT_TYPES["retired"])
        else:
            return random.choice(cls.EMPLOYMENT_TYPES["employed"])
    
    @classmethod
    def _choose_income_for_age(cls, age: int) -> str:
        """Choose realistic income level for age"""
        if age < 25:
            return random.choices(["low", "medium"], weights=[0.7, 0.3])[0]
        elif age < 35:
            return random.choices(["low", "medium", "high"], weights=[0.3, 0.5, 0.2])[0]
        elif age < 55:
            return random.choices(["medium", "high", "very_high"], weights=[0.4, 0.4, 0.2])[0]
        else:
            return random.choices(["medium", "high"], weights=[0.6, 0.4])[0]
    
    @classmethod
    def _get_generation(cls, age: int) -> str:
        """Determine generation based on age"""
        if age < 26:
            return "gen_z"
        elif age < 41:
            return "millennial"
        elif age < 57:
            return "gen_x"
        else:
            return "boomer"
    
    @classmethod
    def _create_psychology(cls, age: int, generation: str, target_interests: List[str] = None) -> PersonaPsychology:
        """Create psychological profile based on age and generation"""
        
        # Big Five personality traits (0-1 scale)
        personality_traits = {
            "openness": random.uniform(0.2, 0.8),
            "conscientiousness": random.uniform(0.3, 0.9),
            "extraversion": random.uniform(0.2, 0.8),
            "agreeableness": random.uniform(0.3, 0.8),
            "neuroticism": random.uniform(0.1, 0.6)
        }
        
        # Age-based trait adjustments
        if age > 50:
            personality_traits["conscientiousness"] += 0.1
            personality_traits["neuroticism"] -= 0.1
        if age < 30:
            personality_traits["openness"] += 0.1
            personality_traits["extraversion"] += 0.1
        
        # Interests based on age
        age_interests = []
        for age_range, interests in cls.INTERESTS_BY_AGE.items():
            if age_range[0] <= age <= age_range[1]:
                age_interests = interests
                break
        
        # Combine with target interests
        if target_interests:
            final_interests = list(set(target_interests + random.sample(age_interests, min(5, len(age_interests)))))
        else:
            final_interests = random.sample(age_interests, random.randint(3, min(8, len(age_interests))))
        
        # Values based on generation
        generation_values = cls.VALUES_BY_GENERATION.get(generation, [])
        selected_values = random.sample(generation_values, min(5, len(generation_values)))
        
        # Shopping behavior based on personality and demographics
        if personality_traits["conscientiousness"] > 0.7:
            shopping_behavior = "conservative"
        elif personality_traits["openness"] > 0.7 and age < 35:
            shopping_behavior = "impulsive"
        else:
            shopping_behavior = "moderate"
        
        # Tech savviness based on age and openness
        base_tech = max(0.1, 1.0 - (age - 18) / 80)  # Decreases with age
        tech_savviness = min(0.9, base_tech + personality_traits["openness"] * 0.2)
        
        # Other traits
        brand_loyalty = min(0.9, 0.3 + (age / 100) + personality_traits["conscientiousness"] * 0.3)
        price_sensitivity = 0.8 - (cls._income_to_numeric(cls._choose_income_for_age(age)) * 0.2)
        social_influence = personality_traits["extraversion"] * 0.6 + (1.0 - age / 100) * 0.4
        
        return PersonaPsychology(
            personality_traits=personality_traits,
            values=selected_values,
            interests=final_interests,
            shopping_behavior=shopping_behavior,
            tech_savviness=tech_savviness,
            brand_loyalty=brand_loyalty,
            price_sensitivity=max(0.1, min(0.9, price_sensitivity)),
            social_influence=max(0.1, min(0.9, social_influence))
        )
    
    @classmethod
    def _income_to_numeric(cls, income_level: str) -> float:
        """Convert income level to numeric (0-1)"""
        mapping = {"low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 1.0}
        return mapping.get(income_level, 0.5)
    
    @classmethod
    def _get_interests_for_category(cls, category: str) -> List[str]:
        """Get relevant interests for product category"""
        
        category_interests = {
            "technology": ["technology", "gaming", "gadgets", "innovation", "apps"],
            "fashion": ["fashion", "style", "shopping", "beauty", "trends"],
            "fitness": ["fitness", "health", "sports", "nutrition", "wellness"],
            "travel": ["travel", "adventure", "culture", "photography", "exploration"],
            "food": ["cooking", "restaurants", "wine", "food", "culinary"],
            "home": ["home improvement", "gardening", "interior design", "DIY", "family"],
            "finance": ["investing", "personal finance", "business", "career", "education"],
            "entertainment": ["entertainment", "movies", "music", "streaming", "gaming"],
            "automotive": ["cars", "automotive", "transportation", "mechanics", "racing"],
            "luxury": ["luxury goods", "high-end fashion", "fine dining", "travel", "art"]
        }
        
        return category_interests.get(category, ["general interest"])


# Pre-defined persona templates for common use cases
class PersonaTemplates:
    """Pre-defined persona templates for quick testing"""
    
    @staticmethod
    def tech_early_adopter() -> PersonaConfig:
        """Tech-savvy early adopter persona"""
        return PersonaFactory.create_targeted_persona({
            "age_range": (25, 40),
            "gender": random.choice(["male", "female"]),
            "income_level": "high",
            "interests": ["technology", "gadgets", "innovation", "startups", "apps"]
        })
    
    @staticmethod
    def budget_conscious_family() -> PersonaConfig:
        """Budget-conscious family persona"""
        return PersonaFactory.create_targeted_persona({
            "age_range": (30, 45),
            "income_level": "medium",
            "interests": ["family", "savings", "education", "home", "health"]
        })
    
    @staticmethod
    def luxury_consumer() -> PersonaConfig:
        """High-income luxury consumer"""
        return PersonaFactory.create_targeted_persona({
            "age_range": (35, 60),
            "income_level": "very_high",
            "interests": ["luxury goods", "travel", "fine dining", "art", "fashion"]
        })
    
    @staticmethod
    def student() -> PersonaConfig:
        """College student persona"""
        return PersonaFactory.create_targeted_persona({
            "age_range": (18, 24),
            "income_level": "low",
            "interests": ["entertainment", "social media", "gaming", "music", "fashion"]
        })
    
    @staticmethod
    def retiree() -> PersonaConfig:
        """Retired person persona"""
        return PersonaFactory.create_targeted_persona({
            "age_range": (65, 80),
            "income_level": "medium",
            "interests": ["health", "travel", "grandchildren", "hobbies", "community"]
        })
    
    @staticmethod
    def get_diverse_test_cohort() -> List[PersonaConfig]:
        """Get a diverse cohort for testing"""
        return [
            PersonaTemplates.tech_early_adopter(),
            PersonaTemplates.budget_conscious_family(),
            PersonaTemplates.luxury_consumer(),
            PersonaTemplates.student(),
            PersonaTemplates.retiree()
        ]


# Utility functions for persona management
def save_personas_to_file(personas: List[PersonaConfig], filename: str):
    """Save personas to JSON file"""
    import json
    
    persona_data = []
    for persona in personas:
        data = {
            "persona_id": persona.persona_id,
            "name": persona.name,
            "demographics": persona.demographics.__dict__,
            "psychology": persona.psychology.__dict__,
            "history": {
                "ads_seen": persona.history.ads_seen,
                "clicks": persona.history.clicks,
                "purchases": persona.history.purchases,
                "fatigue_level": persona.history.fatigue_level,
                "last_interaction": persona.history.last_interaction.isoformat() if persona.history.last_interaction else None,
                "interaction_count": persona.history.interaction_count,
                "state": persona.history.state.value
            }
        }
        persona_data.append(data)
    
    with open(filename, 'w') as f:
        json.dump(persona_data, f, indent=2)


def load_personas_from_file(filename: str) -> List[PersonaConfig]:
    """Load personas from JSON file"""
    import json
    from datetime import datetime
    
    with open(filename, 'r') as f:
        persona_data = json.load(f)
    
    personas = []
    for data in persona_data:
        demographics = PersonaDemographics(**data["demographics"])
        psychology = PersonaPsychology(**data["psychology"])
        
        history_data = data["history"]
        history = PersonaHistory(
            ads_seen=history_data["ads_seen"],
            clicks=history_data["clicks"],
            purchases=history_data["purchases"],
            fatigue_level=history_data["fatigue_level"],
            last_interaction=datetime.fromisoformat(history_data["last_interaction"]) if history_data["last_interaction"] else None,
            interaction_count=history_data["interaction_count"],
            state=PersonaState(history_data["state"])
        )
        
        persona = PersonaConfig(
            persona_id=data["persona_id"],
            name=data["name"],
            demographics=demographics,
            psychology=psychology,
            history=history
        )
        personas.append(persona)
    
    return personas