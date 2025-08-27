#!/usr/bin/env python3
"""
Trigger Event System for Behavioral Health Marketing
Models how parents discover problems and how their concern evolves over time
NO HARDCODING - realistic trigger patterns based on actual parent experiences
"""

import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

from behavioral_health_persona_factory import TriggerEvent, ParentPersona

@dataclass
class TriggerContext:
    """Context around a trigger event that affects parent response"""
    time_of_day: int  # 0-23 hour
    day_of_week: int  # 0-6 (Mon-Sun)
    location: str  # Where discovery happened
    discovery_method: str  # How parent found out
    severity_indicators: List[str]  # What made it seem serious
    immediate_action_taken: str  # What parent did right away
    emotional_state: str  # Parent's emotional response
    
class DiscoveryMethod(Enum):
    """How parents typically discover issues"""
    # Direct discovery (high impact)
    CAUGHT_IN_ACT = "caught_in_act"  # Walked in on them
    CHECKED_PHONE = "checked_phone"  # Looked at teen's phone
    BROWSER_HISTORY = "browser_history"  # Checked computer
    OVERHEARD = "overheard_conversation"  # Heard them talking
    
    # Indirect discovery (medium impact)
    SCHOOL_CALLED = "school_called"  # School notified
    FRIEND_PARENT = "friend_parent_told"  # Another parent mentioned
    SIBLING_TOLD = "sibling_reported"  # Brother/sister said something
    BEHAVIOR_CHANGE = "noticed_behavior_change"  # Parent intuition
    
    # External prompt (lower impact)
    NEWS_ARTICLE = "read_news_article"  # Media coverage
    SOCIAL_MEDIA = "saw_social_media_post"  # Facebook parent group
    THERAPIST = "therapist_suggested"  # Professional recommendation
    GENERAL_WORRY = "general_anxiety"  # No specific trigger

@dataclass
class TriggerEvolution:
    """How a trigger event evolves over time"""
    initial_trigger: TriggerEvent
    initial_concern: float
    peak_concern: float
    time_to_peak_hours: float
    decay_rate: float  # How fast concern decreases
    reinforcement_events: List[Tuple[float, str]]  # (hours_after, event)
    resolution_probability: float  # Chance it resolves without intervention

class TriggerEventSystem:
    """System for generating and evolving trigger events"""
    
    # Time patterns for different triggers
    TRIGGER_TIME_PATTERNS = {
        TriggerEvent.FOUND_SELF_HARM_CONTENT: {
            "peak_hours": [22, 23, 0, 1, 2, 3],  # Late night discoveries
            "peak_days": [4, 5, 6],  # Thu-Sat
            "typical_discovery": DiscoveryMethod.CHECKED_PHONE
        },
        TriggerEvent.DISCOVERED_CONCERNING_SEARCHES: {
            "peak_hours": [21, 22, 23, 0, 1],  # Evening/night
            "peak_days": [0, 1, 2, 3, 4],  # Weekdays
            "typical_discovery": DiscoveryMethod.BROWSER_HISTORY
        },
        TriggerEvent.CYBERBULLYING_INCIDENT: {
            "peak_hours": [15, 16, 17, 18, 19, 20],  # After school
            "peak_days": [0, 1, 2, 3, 4],  # School days
            "typical_discovery": DiscoveryMethod.BEHAVIOR_CHANGE
        },
        TriggerEvent.GRADES_DROPPING: {
            "peak_hours": [16, 17, 18, 19],  # Report card time
            "peak_days": [2, 3, 4],  # Mid-week
            "typical_discovery": DiscoveryMethod.SCHOOL_CALLED
        },
        TriggerEvent.SLEEP_DISRUPTION_SEVERE: {
            "peak_hours": [0, 1, 2, 3, 4, 5],  # Middle of night
            "peak_days": [0, 1, 2, 3, 4, 5, 6],  # Any day
            "typical_discovery": DiscoveryMethod.CAUGHT_IN_ACT
        },
        TriggerEvent.TOO_MUCH_SCREEN_TIME: {
            "peak_hours": [20, 21, 22, 23],  # Evening battles
            "peak_days": [0, 1, 2, 3, 4, 5, 6],  # Any day
            "typical_discovery": DiscoveryMethod.BEHAVIOR_CHANGE
        }
    }
    
    # How discovery method affects concern multiplier
    DISCOVERY_IMPACT = {
        DiscoveryMethod.CAUGHT_IN_ACT: 1.5,
        DiscoveryMethod.CHECKED_PHONE: 1.4,
        DiscoveryMethod.BROWSER_HISTORY: 1.3,
        DiscoveryMethod.OVERHEARD: 1.2,
        DiscoveryMethod.SCHOOL_CALLED: 1.3,
        DiscoveryMethod.FRIEND_PARENT: 1.1,
        DiscoveryMethod.SIBLING_TOLD: 1.2,
        DiscoveryMethod.BEHAVIOR_CHANGE: 1.0,
        DiscoveryMethod.NEWS_ARTICLE: 0.8,
        DiscoveryMethod.SOCIAL_MEDIA: 0.9,
        DiscoveryMethod.THERAPIST: 1.1,
        DiscoveryMethod.GENERAL_WORRY: 0.7
    }
    
    @classmethod
    def generate_trigger_event(cls, 
                              current_time: datetime,
                              parent: ParentPersona) -> TriggerContext:
        """Generate a realistic trigger event with context"""
        
        trigger = parent.trigger_event
        hour = current_time.hour
        day = current_time.weekday()
        
        # Get typical pattern for this trigger
        pattern = cls.TRIGGER_TIME_PATTERNS.get(trigger, {})
        peak_hours = pattern.get("peak_hours", list(range(24)))
        typical_discovery = pattern.get("typical_discovery", DiscoveryMethod.GENERAL_WORRY)
        
        # Determine discovery method (can vary from typical)
        if hour in peak_hours:
            # During peak hours, use typical method 70% of time
            if random.random() < 0.7:
                discovery = typical_discovery
            else:
                discovery = random.choice(list(DiscoveryMethod))
        else:
            # Off-peak, more random discovery
            discovery = random.choice(list(DiscoveryMethod))
        
        # Generate context based on trigger and discovery
        context = cls._generate_context(trigger, discovery, hour, day, parent)
        
        return context
    
    @classmethod
    def _generate_context(cls, 
                         trigger: TriggerEvent,
                         discovery: DiscoveryMethod,
                         hour: int,
                         day: int,
                         parent: ParentPersona) -> TriggerContext:
        """Generate detailed context for a trigger event"""
        
        # Location based on time and discovery
        if hour >= 22 or hour <= 5:
            location = "teen's bedroom"
        elif hour >= 6 and hour <= 8:
            location = "breakfast table"
        elif hour >= 15 and hour <= 17:
            location = "after school at home"
        elif hour >= 18 and hour <= 21:
            location = "family room"
        else:
            location = "home"
        
        # Severity indicators based on trigger
        severity_indicators = cls._get_severity_indicators(trigger)
        
        # Immediate action based on parent personality
        if parent.current_concern_level >= 8:
            immediate_action = random.choice([
                "confronted teen immediately",
                "called spouse in panic",
                "searched Google frantically",
                "called therapist emergency line",
                "stayed up all night worrying"
            ])
        elif parent.current_concern_level >= 5:
            immediate_action = random.choice([
                "talked to teen calmly",
                "discussed with spouse",
                "started researching online",
                "made therapy appointment",
                "monitored more closely"
            ])
        else:
            immediate_action = random.choice([
                "made mental note",
                "planned to discuss later",
                "did some casual research",
                "asked friends for advice",
                "waited to see if it continues"
            ])
        
        # Emotional state
        emotional_states = cls._get_emotional_state(parent.current_concern_level)
        
        return TriggerContext(
            time_of_day=hour,
            day_of_week=day,
            location=location,
            discovery_method=discovery.value,
            severity_indicators=severity_indicators,
            immediate_action_taken=immediate_action,
            emotional_state=emotional_states
        )
    
    @classmethod
    def _get_severity_indicators(cls, trigger: TriggerEvent) -> List[str]:
        """Get indicators that make parent think it's serious"""
        
        indicators_map = {
            TriggerEvent.FOUND_SELF_HARM_CONTENT: [
                "multiple searches over weeks",
                "detailed methods researched",
                "community forums joined",
                "images saved",
                "recent timestamps"
            ],
            TriggerEvent.CYBERBULLYING_INCIDENT: [
                "multiple harassers",
                "threats made",
                "public humiliation",
                "going on for months",
                "teen crying frequently"
            ],
            TriggerEvent.SLEEP_DISRUPTION_SEVERE: [
                "up past 3am nightly",
                "falling asleep in class",
                "dark circles under eyes",
                "mood swings",
                "grades affected"
            ],
            TriggerEvent.GRADES_DROPPING: [
                "dropped two letter grades",
                "missing assignments",
                "teacher concerns",
                "skipping classes",
                "gave up activities"
            ],
            TriggerEvent.TOO_MUCH_SCREEN_TIME: [
                "12+ hours daily",
                "violent reactions when limited",
                "lying about usage",
                "sneaking devices at night",
                "no other interests"
            ]
        }
        
        base_indicators = indicators_map.get(trigger, ["concerning behavior noticed"])
        # Return 2-4 random indicators (or all if less than 2)
        if len(base_indicators) <= 2:
            return base_indicators
        num_indicators = random.randint(2, min(4, len(base_indicators)))
        return random.sample(base_indicators, num_indicators)
    
    @classmethod
    def _get_emotional_state(cls, concern_level: float) -> str:
        """Get parent's emotional state based on concern"""
        
        if concern_level >= 9:
            return random.choice(["panicked", "terrified", "desperate", "overwhelmed"])
        elif concern_level >= 7:
            return random.choice(["very worried", "anxious", "scared", "stressed"])
        elif concern_level >= 5:
            return random.choice(["concerned", "worried", "uneasy", "troubled"])
        elif concern_level >= 3:
            return random.choice(["mildly concerned", "watchful", "alert", "cautious"])
        else:
            return random.choice(["curious", "aware", "calm", "observant"])
    
    @classmethod
    def evolve_trigger(cls,
                      parent: ParentPersona,
                      hours_elapsed: float,
                      new_information: Optional[Dict[str, Any]] = None) -> TriggerEvolution:
        """Model how trigger evolves over time"""
        
        if not parent.trigger_event:
            return None
        
        # Base evolution pattern
        evolution = TriggerEvolution(
            initial_trigger=parent.trigger_event,
            initial_concern=parent.trigger_intensity,
            peak_concern=parent.trigger_intensity,
            time_to_peak_hours=0,
            decay_rate=parent.urgency_decay_rate,
            reinforcement_events=[],
            resolution_probability=0.1
        )
        
        # Crisis triggers can escalate
        if parent.trigger_intensity >= 8:
            # 30% chance of escalation in first 24 hours
            if hours_elapsed < 24 and random.random() < 0.3:
                evolution.peak_concern = min(10, parent.trigger_intensity + random.uniform(0.5, 2))
                evolution.time_to_peak_hours = random.uniform(2, 12)
                evolution.reinforcement_events.append(
                    (evolution.time_to_peak_hours, "discovered additional concerning content")
                )
        
        # Moderate triggers might get reinforced
        elif parent.trigger_intensity >= 5:
            # Check for reinforcement events
            if hours_elapsed > 24 and hours_elapsed < 72:
                if random.random() < 0.2:
                    evolution.reinforcement_events.append(
                        (hours_elapsed, "behavior continued despite discussion")
                    )
                    evolution.peak_concern = min(10, parent.current_concern_level + 1)
        
        # Low triggers might resolve
        else:
            if hours_elapsed > 48:
                evolution.resolution_probability = 0.3
                if random.random() < evolution.resolution_probability:
                    evolution.reinforcement_events.append(
                        (hours_elapsed, "teen seemed better after talk")
                    )
        
        # Add new information if provided
        if new_information:
            if new_information.get("talked_to_teen"):
                if random.random() < 0.4:  # 40% chance talk helps
                    evolution.reinforcement_events.append(
                        (hours_elapsed, "productive conversation with teen")
                    )
                    parent.current_concern_level *= 0.8
            
            if new_information.get("found_more_evidence"):
                evolution.reinforcement_events.append(
                    (hours_elapsed, "found additional concerning evidence")
                )
                parent.current_concern_level = min(10, parent.current_concern_level + 2)
            
            if new_information.get("got_professional_opinion"):
                if random.random() < 0.6:  # Professional usually increases urgency
                    evolution.reinforcement_events.append(
                        (hours_elapsed, "therapist recommended immediate intervention")
                    )
                    parent.current_concern_level = min(10, parent.current_concern_level + 1.5)
        
        return evolution
    
    @classmethod
    def get_search_pattern(cls, 
                          parent: ParentPersona,
                          context: TriggerContext,
                          hours_since_trigger: float) -> List[str]:
        """Generate realistic search pattern based on trigger timeline"""
        
        searches = []
        
        # Immediate searches (first 2 hours)
        if hours_since_trigger < 2:
            if parent.current_concern_level >= 8:
                searches.extend([
                    f"what to do if teen {parent.trigger_event.value.replace('_', ' ')}",
                    "emergency teen mental health help",
                    "crisis hotline teenage mental health",
                    f"is {' '.join(context.severity_indicators[:1])} dangerous"
                ])
            else:
                searches.extend([
                    f"teenage {parent.trigger_event.value.replace('_', ' ')} normal",
                    "signs of teen depression",
                    "when to worry about teenager behavior"
                ])
        
        # Research phase (2-24 hours)
        elif hours_since_trigger < 24:
            if parent.tech_savviness > 0.5:
                searches.extend([
                    "best parental monitoring apps 2024",
                    "teen mental health apps clinician recommended",
                    "Bark vs Qustodio vs Aura comparison",
                    "parental controls for teen mental health"
                ])
            else:
                searches.extend([
                    "how to monitor teenager online",
                    "parental control programs",
                    "teen monitoring software reviews"
                ])
        
        # Deep research (24-72 hours)
        elif hours_since_trigger < 72:
            searches.extend([
                "Aura parental controls review",
                "Aura behavioral health features",
                "does Aura detect self harm",
                "Aura vs Bark for mental health",
                "Aura pricing family plan"
            ])
        
        # Comparison shopping (72+ hours)
        else:
            searches.extend([
                "Aura discount code",
                "Aura free trial",
                "cancel Bark switch to Aura",
                "Aura setup guide",
                "Aura customer service"
            ])
        
        return searches


class CrisisSimulator:
    """Simulates crisis discovery and parent response"""
    
    def __init__(self):
        self.active_crises = []
        self.resolved_crises = []
    
    def simulate_day(self, num_parents: int = 1000) -> List[Dict[str, Any]]:
        """Simulate a day of crisis discoveries"""
        
        discoveries = []
        current_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        for hour in range(24):
            current_time = current_time.replace(hour=hour)
            
            # Number of discoveries this hour (follows pattern)
            if hour in [22, 23, 0, 1, 2, 3]:  # Late night peak
                hour_discoveries = int(num_parents * 0.02)  # 2% discover at night
            elif hour in [15, 16, 17, 18, 19, 20]:  # After school peak
                hour_discoveries = int(num_parents * 0.015)  # 1.5% after school
            else:
                hour_discoveries = int(num_parents * 0.005)  # 0.5% other times
            
            for _ in range(hour_discoveries):
                # Create parent with trigger
                from behavioral_health_persona_factory import BehavioralHealthPersonaFactory
                parent = BehavioralHealthPersonaFactory.create_parent_with_trigger()
                
                # Generate trigger context
                context = TriggerEventSystem.generate_trigger_event(current_time, parent)
                
                # Record discovery
                discoveries.append({
                    "timestamp": current_time.isoformat(),
                    "parent_id": parent.persona_id,
                    "parent": parent,
                    "context": context,
                    "initial_concern": parent.current_concern_level,
                    "initial_search": parent.generate_search_query()
                })
        
        return discoveries


if __name__ == "__main__":
    # Test the trigger system
    from behavioral_health_persona_factory import BehavioralHealthPersonaFactory
    
    print("Simulating trigger events and evolution:\n")
    
    # Create a parent with high concern trigger
    parent = BehavioralHealthPersonaFactory.create_parent_with_trigger(
        TriggerEvent.FOUND_SELF_HARM_CONTENT
    )
    
    print(f"Parent: {parent.name}")
    print(f"Trigger: {parent.trigger_event.value}")
    print(f"Initial Concern: {parent.current_concern_level:.1f}")
    
    # Generate trigger context
    context = TriggerEventSystem.generate_trigger_event(datetime.now(), parent)
    print(f"\nDiscovery Context:")
    print(f"  Time: {context.time_of_day}:00")
    print(f"  Location: {context.location}")
    print(f"  How discovered: {context.discovery_method}")
    print(f"  Warning signs: {', '.join(context.severity_indicators)}")
    print(f"  Immediate action: {context.immediate_action_taken}")
    print(f"  Emotional state: {context.emotional_state}")
    
    # Simulate search evolution
    print(f"\nSearch Evolution:")
    for hours in [0.5, 2, 12, 24, 48, 72, 168]:
        searches = TriggerEventSystem.get_search_pattern(parent, context, hours)
        print(f"  After {hours} hours: {searches[0] if searches else 'None'}")
        
        # Update concern
        parent.update_concern(hours)
        print(f"    Concern level: {parent.current_concern_level:.1f}")
    
    # Test crisis simulator
    print("\n\nSimulating 24-hour crisis discovery pattern:")
    simulator = CrisisSimulator()
    discoveries = simulator.simulate_day(num_parents=10000)
    
    # Group by hour
    hourly_counts = {}
    for d in discoveries:
        hour = datetime.fromisoformat(d["timestamp"]).hour
        hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
    
    print(f"Total discoveries: {len(discoveries)}")
    print("Discoveries by hour:")
    for hour in sorted(hourly_counts.keys()):
        bar = "█" * (hourly_counts[hour] // 2)
        print(f"  {hour:02d}:00: {bar} {hourly_counts[hour]}")