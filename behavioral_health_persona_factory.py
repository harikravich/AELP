#!/usr/bin/env python3
"""
Behavioral Health Parent Persona Factory
Generates realistic parents with mental health concerns about their teens
NO HARDCODED SEGMENTS - continuous concern levels and realistic triggers
"""

import random
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

class TriggerEvent(Enum):
    """Realistic trigger events that cause parents to seek help"""
    # Crisis triggers (concern 8-10)
    FOUND_SELF_HARM_CONTENT = "found_self_harm_content"
    DISCOVERED_CONCERNING_SEARCHES = "discovered_concerning_searches"
    CYBERBULLYING_INCIDENT = "cyberbullying_incident"
    SUICIDE_IDEATION_DISCOVERED = "suicide_ideation_discovered"
    EATING_DISORDER_SIGNS = "eating_disorder_signs"
    
    # High concern triggers (concern 6-8)
    GRADES_DROPPING = "grades_dropping"
    SLEEP_DISRUPTION_SEVERE = "sleep_disruption_severe"
    SOCIAL_ISOLATION = "social_isolation_noticed"
    ANGRY_OUTBURSTS = "angry_outbursts_increasing"
    CAUGHT_LYING_ABOUT_ONLINE = "caught_lying_about_online_activity"
    
    # Moderate triggers (concern 4-6)
    TOO_MUCH_SCREEN_TIME = "too_much_screen_time"
    FRIEND_HAD_INCIDENT = "friend_had_incident"
    SCHOOL_COUNSELOR_SUGGESTION = "school_counselor_suggestion"
    NEWS_ARTICLE_READ = "news_article_about_teen_mental_health"
    THERAPIST_MENTIONED = "therapist_mentioned_monitoring"
    
    # Low triggers (concern 2-4)
    GENERAL_WORRY = "general_parental_worry"
    PREVENTION_MINDED = "prevention_minded"
    PEER_RECOMMENDATION = "peer_parent_recommendation"
    SCHOOL_NEWSLETTER = "school_newsletter_mention"

@dataclass
class ParentPersona:
    """A realistic parent with mental health concerns about their teen"""
    # Identity
    persona_id: str
    name: str
    age: int
    gender: str
    location: str
    
    # Socioeconomic
    income_level: float  # 20k-200k+
    education: str
    employment: str
    household_size: int
    number_of_teens: int
    
    # Teen information
    teen_ages: List[int]
    teen_genders: List[str]
    teen_mental_health_history: bool
    
    # Concern dynamics
    base_concern_level: float  # 0-10 baseline anxiety about teens
    current_concern_level: float  # 0-10 current state
    concern_volatility: float  # How much concern fluctuates
    
    # Behavioral traits
    research_thoroughness: float  # 0-1, how deep they research
    price_sensitivity: float  # 0-1, how much price matters
    tech_savviness: float  # 0-1, affects research patterns
    trust_in_experts: float  # 0-1, response to clinical backing
    privacy_concern: float  # 0-1, worry about teen privacy
    spouse_involvement: float  # 0-1, need for partner agreement
    
    # Decision factors
    urgency_decay_rate: float  # How fast crisis feeling fades
    decision_threshold: float  # Concern level needed to purchase
    budget_constraint: float  # Max willing to pay monthly
    
    # Trigger history
    trigger_event: Optional[TriggerEvent] = None
    trigger_timestamp: Optional[datetime] = None
    trigger_intensity: float = 0.0
    
    # Journey state
    journey_stage: str = "unaware"
    days_since_trigger: int = 0
    searches_performed: List[str] = field(default_factory=list)
    ads_seen: List[Dict] = field(default_factory=list)
    competitors_researched: List[str] = field(default_factory=list)
    
    def update_concern(self, hours_passed: float) -> None:
        """Update concern level based on time and decay"""
        if self.trigger_event:
            # Concern decays over time but never below baseline
            days_passed = hours_passed / 24.0
            decay_factor = np.exp(-self.urgency_decay_rate * days_passed)
            crisis_boost = self.trigger_intensity * decay_factor
            self.current_concern_level = min(10, self.base_concern_level + crisis_boost)
            
            # Add some volatility
            noise = np.random.normal(0, self.concern_volatility)
            self.current_concern_level = np.clip(self.current_concern_level + noise, 0, 10)
    
    def generate_search_query(self) -> str:
        """Generate realistic search query based on concern level"""
        concern = self.current_concern_level
        
        if concern >= 9:
            queries = [
                "emergency help teen mental health crisis",
                "child found self harm content what to do",
                "teen suicide warning signs immediate help",
                f"found {self.trigger_event.value.replace('_', ' ')} help",
                "crisis hotline teen mental health",
                "immediate teen psychiatric help"
            ]
        elif concern >= 7:
            queries = [
                "signs of depression in teenagers",
                "teen mental health monitoring apps",
                "how to know if teen is being cyberbullied",
                "parental controls for teen mental health",
                "teen won't talk to me worried",
                "behavioral changes in teenagers warning signs"
            ]
        elif concern >= 5:
            queries = [
                "too much screen time effects on teens",
                "healthy screen time limits teenagers",
                "parental control apps reviewed",
                "teen social media mental health",
                "how to monitor teen online safely",
                "screen time and teen depression"
            ]
        elif concern >= 3:
            queries = [
                "recommended screen time for teenagers",
                "best parental control apps 2024",
                "digital wellness for families",
                "how to talk to teen about online safety",
                "screen time guidelines by age",
                "family digital boundaries"
            ]
        else:
            queries = [
                "parenting teens in digital age",
                "teenage development and technology",
                "family screen time apps",
                "digital parenting tips",
                "teen online safety basics",
                "healthy tech habits for kids"
            ]
        
        # Add personalization based on traits
        query = random.choice(queries)
        
        if self.tech_savviness < 0.3:
            # Less tech-savvy parents use simpler queries
            query = query.replace("apps", "programs").replace("digital", "computer")
        
        if self.trust_in_experts > 0.7:
            # High trust in experts add qualifiers
            if random.random() < 0.5:
                query += " therapist recommended"
        
        return query
    
    def should_click_ad(self, ad_content: Dict[str, Any]) -> bool:
        """Determine if parent would click on specific ad"""
        # Extract ad features
        urgency_level = ad_content.get("urgency_level", 5)
        mentions_clinical = ad_content.get("mentions_clinical", False)
        price_shown = ad_content.get("price_shown", None)
        mentions_crisis = ad_content.get("mentions_crisis", False)
        
        # Base click probability from concern match
        concern_match = 1.0 - abs(self.current_concern_level - urgency_level) / 10.0
        click_prob = concern_match * 0.3  # Max 30% CTR for perfect match
        
        # Modifiers based on persona traits
        if mentions_clinical and self.trust_in_experts > 0.6:
            click_prob *= 1.5
        
        if price_shown and price_shown > self.budget_constraint:
            click_prob *= 0.2  # Still might click to research
        
        if mentions_crisis and self.current_concern_level < 7:
            click_prob *= 0.3  # Crisis messaging scares non-crisis parents
        
        if mentions_crisis and self.current_concern_level >= 8:
            click_prob *= 2.0  # Crisis parents need crisis messaging
        
        # Ad fatigue
        similar_ads = sum(1 for ad in self.ads_seen[-10:] 
                         if ad.get("creative_id") == ad_content.get("creative_id"))
        if similar_ads > 2:
            click_prob *= 0.5 ** similar_ads
        
        return random.random() < min(click_prob, 0.7)  # Cap at 70% CTR
    
    def will_convert(self, touchpoint_count: int, days_since_trigger: int) -> bool:
        """Determine if parent will convert at this point"""
        # No instant conversions - minimum research period
        if touchpoint_count < 3 or days_since_trigger < 1:
            return False
        
        # Crisis parents convert faster
        if self.current_concern_level >= 8 and days_since_trigger >= 1:
            if touchpoint_count >= 5:
                return random.random() < 0.4
        
        # High concern with enough research
        if self.current_concern_level >= 6 and days_since_trigger >= 3:
            if touchpoint_count >= 10 and len(self.competitors_researched) >= 2:
                return random.random() < 0.25
        
        # Moderate concern needs more time
        if self.current_concern_level >= 4 and days_since_trigger >= 7:
            if touchpoint_count >= 15 and len(self.competitors_researched) >= 3:
                # Check if spouse approval needed
                if self.spouse_involvement > 0.7:
                    # Assume spouse discussion happened after a week
                    if days_since_trigger >= 10:
                        return random.random() < 0.15
                else:
                    return random.random() < 0.2
        
        # Low concern rarely converts
        if self.current_concern_level < 4 and days_since_trigger >= 14:
            if touchpoint_count >= 20:
                return random.random() < 0.05
        
        return False


class BehavioralHealthPersonaFactory:
    """Factory for creating realistic parent personas"""
    
    # Realistic distributions based on market research
    TRIGGER_DISTRIBUTION = {
        TriggerEvent.FOUND_SELF_HARM_CONTENT: 0.02,
        TriggerEvent.DISCOVERED_CONCERNING_SEARCHES: 0.03,
        TriggerEvent.CYBERBULLYING_INCIDENT: 0.04,
        TriggerEvent.SUICIDE_IDEATION_DISCOVERED: 0.01,
        TriggerEvent.EATING_DISORDER_SIGNS: 0.02,
        TriggerEvent.GRADES_DROPPING: 0.08,
        TriggerEvent.SLEEP_DISRUPTION_SEVERE: 0.10,
        TriggerEvent.SOCIAL_ISOLATION: 0.06,
        TriggerEvent.ANGRY_OUTBURSTS: 0.05,
        TriggerEvent.CAUGHT_LYING_ABOUT_ONLINE: 0.04,
        TriggerEvent.TOO_MUCH_SCREEN_TIME: 0.15,
        TriggerEvent.FRIEND_HAD_INCIDENT: 0.08,
        TriggerEvent.SCHOOL_COUNSELOR_SUGGESTION: 0.05,
        TriggerEvent.NEWS_ARTICLE_READ: 0.10,
        TriggerEvent.THERAPIST_MENTIONED: 0.03,
        TriggerEvent.GENERAL_WORRY: 0.08,
        TriggerEvent.PREVENTION_MINDED: 0.04,
        TriggerEvent.PEER_RECOMMENDATION: 0.02,
        TriggerEvent.SCHOOL_NEWSLETTER: 0.00  # Remaining probability
    }
    
    # Trigger to concern level mapping
    TRIGGER_CONCERN_MAP = {
        TriggerEvent.FOUND_SELF_HARM_CONTENT: (8, 10),
        TriggerEvent.DISCOVERED_CONCERNING_SEARCHES: (7, 10),
        TriggerEvent.CYBERBULLYING_INCIDENT: (7, 9),
        TriggerEvent.SUICIDE_IDEATION_DISCOVERED: (9, 10),
        TriggerEvent.EATING_DISORDER_SIGNS: (7, 9),
        TriggerEvent.GRADES_DROPPING: (5, 7),
        TriggerEvent.SLEEP_DISRUPTION_SEVERE: (6, 8),
        TriggerEvent.SOCIAL_ISOLATION: (6, 8),
        TriggerEvent.ANGRY_OUTBURSTS: (5, 7),
        TriggerEvent.CAUGHT_LYING_ABOUT_ONLINE: (5, 7),
        TriggerEvent.TOO_MUCH_SCREEN_TIME: (3, 6),
        TriggerEvent.FRIEND_HAD_INCIDENT: (4, 7),
        TriggerEvent.SCHOOL_COUNSELOR_SUGGESTION: (4, 6),
        TriggerEvent.NEWS_ARTICLE_READ: (3, 5),
        TriggerEvent.THERAPIST_MENTIONED: (4, 6),
        TriggerEvent.GENERAL_WORRY: (2, 4),
        TriggerEvent.PREVENTION_MINDED: (1, 3),
        TriggerEvent.PEER_RECOMMENDATION: (2, 4),
        TriggerEvent.SCHOOL_NEWSLETTER: (1, 3)
    }
    
    # US cities with different attitudes toward mental health
    CITIES = [
        ("San Francisco, CA", 0.8),  # High mental health awareness
        ("Portland, OR", 0.75),
        ("Seattle, WA", 0.75),
        ("Austin, TX", 0.7),
        ("Denver, CO", 0.7),
        ("Boston, MA", 0.7),
        ("Los Angeles, CA", 0.65),
        ("New York, NY", 0.65),
        ("Chicago, IL", 0.6),
        ("Atlanta, GA", 0.55),
        ("Phoenix, AZ", 0.5),
        ("Dallas, TX", 0.45),
        ("Houston, TX", 0.45),
        ("Miami, FL", 0.5),
        ("Salt Lake City, UT", 0.4),  # Lower due to stigma
        ("Birmingham, AL", 0.35),
    ]
    
    @classmethod
    def create_parent_with_trigger(cls, trigger_event: Optional[TriggerEvent] = None) -> ParentPersona:
        """Create a parent persona with a specific trigger event"""
        
        # Select trigger if not provided
        if trigger_event is None:
            trigger_event = random.choices(
                list(cls.TRIGGER_DISTRIBUTION.keys()),
                weights=list(cls.TRIGGER_DISTRIBUTION.values())
            )[0]
        
        # Demographics
        age = int(np.random.normal(42, 8))  # Parents of teens typically 35-50
        age = np.clip(age, 32, 65)
        
        gender = random.choices(["female", "male", "non-binary"], weights=[0.65, 0.34, 0.01])[0]
        
        # Location affects mental health awareness
        city, awareness_modifier = random.choice(cls.CITIES)
        
        # Socioeconomic (affects ability to pay)
        income_level = np.random.lognormal(10.8, 0.7)  # Log-normal around 50k
        income_level = np.clip(income_level, 25000, 500000)
        
        education = random.choices(
            ["high_school", "some_college", "college", "graduate"],
            weights=[0.2, 0.3, 0.35, 0.15]
        )[0]
        
        # Family structure
        household_size = random.choices([2, 3, 4, 5, 6], weights=[0.1, 0.25, 0.35, 0.25, 0.05])[0]
        number_of_teens = random.choices([1, 2, 3], weights=[0.6, 0.35, 0.05])[0]
        
        # Teen details (affects concern level)
        teen_ages = [random.randint(13, 17) for _ in range(number_of_teens)]
        teen_genders = [random.choice(["male", "female", "non-binary"]) for _ in range(number_of_teens)]
        teen_mental_health_history = random.random() < 0.3  # 30% have prior issues
        
        # Base concern level (personality trait)
        base_concern = np.random.beta(3, 5) * 10  # Most parents 3-5 baseline
        if teen_mental_health_history:
            base_concern += 2  # Higher baseline if history
        
        # Current concern based on trigger
        concern_range = cls.TRIGGER_CONCERN_MAP[trigger_event]
        trigger_intensity = random.uniform(concern_range[0], concern_range[1])
        current_concern = min(10, base_concern + trigger_intensity)
        
        # Behavioral traits
        research_thoroughness = np.random.beta(4, 2)  # Most parents research thoroughly
        
        # Price sensitivity inversely related to income and concern
        price_sensitivity = 1.0 - (income_level / 200000) * (1.0 - current_concern / 10)
        price_sensitivity = np.clip(price_sensitivity, 0.1, 0.95)
        
        tech_savviness = np.random.beta(3, 3)  # Normal distribution
        if age > 50:
            tech_savviness *= 0.7  # Older parents less tech-savvy
        
        trust_in_experts = awareness_modifier * np.random.beta(5, 2)
        privacy_concern = np.random.beta(3, 2)  # Most parents have some concern
        
        # Spouse involvement higher for expensive decisions
        spouse_involvement = np.random.beta(5, 3) if income_level < 75000 else np.random.beta(3, 3)
        
        # Decision dynamics
        urgency_decay_rate = 0.1 if trigger_intensity >= 8 else 0.3  # Crisis decays slower
        decision_threshold = random.uniform(4, 7)  # Concern level needed to buy
        budget_constraint = min(50, income_level / 2000)  # Roughly 0.5-2.5% of income
        
        # Create persona
        return ParentPersona(
            persona_id=str(uuid.uuid4()),
            name=cls._generate_name(gender),
            age=age,
            gender=gender,
            location=city,
            income_level=income_level,
            education=education,
            employment=cls._generate_employment(age),
            household_size=household_size,
            number_of_teens=number_of_teens,
            teen_ages=teen_ages,
            teen_genders=teen_genders,
            teen_mental_health_history=teen_mental_health_history,
            base_concern_level=base_concern,
            current_concern_level=current_concern,
            concern_volatility=random.uniform(0.1, 0.5),
            trigger_event=trigger_event,
            trigger_timestamp=datetime.now(),
            trigger_intensity=trigger_intensity,
            research_thoroughness=research_thoroughness,
            price_sensitivity=price_sensitivity,
            tech_savviness=tech_savviness,
            trust_in_experts=trust_in_experts,
            privacy_concern=privacy_concern,
            spouse_involvement=spouse_involvement,
            urgency_decay_rate=urgency_decay_rate,
            decision_threshold=decision_threshold,
            budget_constraint=budget_constraint
        )
    
    @classmethod
    def create_market_representative_batch(cls, batch_size: int) -> List[ParentPersona]:
        """Create a batch of parents representing the actual market"""
        parents = []
        
        for _ in range(batch_size):
            # 70% have some trigger, 30% are just browsing
            if random.random() < 0.7:
                parent = cls.create_parent_with_trigger()
            else:
                # Create a low-concern browser
                parent = cls.create_parent_with_trigger(TriggerEvent.GENERAL_WORRY)
                parent.current_concern_level = random.uniform(0, 3)
            
            parents.append(parent)
        
        return parents
    
    @classmethod
    def _generate_name(cls, gender: str) -> str:
        """Generate realistic parent name"""
        first_names = {
            "female": ["Jennifer", "Lisa", "Karen", "Michelle", "Sarah", "Jessica", 
                       "Amanda", "Melissa", "Laura", "Rebecca", "Christina", "Maria"],
            "male": ["Michael", "David", "James", "John", "Robert", "Christopher",
                     "Daniel", "Matthew", "Anthony", "Mark", "Paul", "Steven"],
            "non-binary": ["Alex", "Jordan", "Casey", "Riley", "Avery", "Quinn"]
        }
        
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                     "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Wilson"]
        
        first = random.choice(first_names.get(gender, first_names["non-binary"]))
        last = random.choice(last_names)
        return f"{first} {last}"
    
    @classmethod
    def _generate_employment(cls, age: int) -> str:
        """Generate realistic employment for parent age"""
        if age < 35:
            return random.choice(["teacher", "nurse", "manager", "sales", "tech"])
        elif age < 50:
            return random.choice(["manager", "director", "consultant", "professional", "business owner"])
        else:
            return random.choice(["executive", "senior manager", "consultant", "professional"])


if __name__ == "__main__":
    # Test the factory
    print("Creating 5 realistic parent personas:\n")
    
    for i in range(5):
        parent = BehavioralHealthPersonaFactory.create_parent_with_trigger()
        print(f"Parent {i+1}: {parent.name}")
        print(f"  Trigger: {parent.trigger_event.value}")
        print(f"  Concern Level: {parent.current_concern_level:.1f}/10")
        print(f"  Location: {parent.location}")
        print(f"  Teen Ages: {parent.teen_ages}")
        print(f"  Search Query: {parent.generate_search_query()}")
        print(f"  Budget: ${parent.budget_constraint:.0f}/month max")
        print()