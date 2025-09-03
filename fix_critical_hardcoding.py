#!/usr/bin/env python3
"""
CRITICAL HARDCODING FIX SCRIPT

Fixes the most critical hardcoded values in priority files
"""

import re
from pathlib import Path

def fix_fortified_agent():
    """Fix hardcoded values in fortified RL agent"""
    file_path = Path('/home/hariravichandran/AELP/fortified_rl_agent_no_hardcoding.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add import for discovered parameter config
    import_section = """from dynamic_segment_integration import (
    get_discovered_segments,
    get_segment_conversion_rate,
    get_high_converting_segment,
    get_mobile_segment,
    validate_no_hardcoded_segments
)"""
    
    new_import_section = """from dynamic_segment_integration import (
    get_discovered_segments,
    get_segment_conversion_rate,
    get_high_converting_segment,
    get_mobile_segment,
    validate_no_hardcoded_segments
)
from discovered_parameter_config import (
    get_config,
    get_epsilon_params,
    get_learning_rate,
    get_conversion_bonus,
    get_goal_thresholds,
    get_priority_params
)"""
    
    content = content.replace(import_section, new_import_section)
    
    # Fix hardcoded conversion rate default
    content = re.sub(
        r'return 0\.1  # Default achievable conversion rate',
        'return get_conversion_bonus()  # Get from discovered patterns',
        content
    )
    
    # Fix hardcoded goal thresholds
    hardcoded_goals = """        # Dense reward based on distance to goal
        distance = abs(achieved - goal)
        if distance < 0.01:  # Very close to goal
            return 1.0
        elif distance < 0.05:  # Moderately close
            return 0.5
        elif distance < 0.1:  # Somewhat close
            return 0.1
        else:
            return -0.1  # Failed to achieve goal"""
    
    discovered_goals = """        # Dense reward based on distance to goal - thresholds from patterns
        thresholds = get_goal_thresholds()
        distance = abs(achieved - goal)
        if distance < thresholds['close']:  # Very close to goal
            return 1.0
        elif distance < thresholds['medium']:  # Moderately close
            return 0.5
        elif distance < thresholds['far']:  # Somewhat close
            return 0.1
        else:
            return -0.1  # Failed to achieve goal"""
    
    content = content.replace(hardcoded_goals, discovered_goals)
    
    # Fix prioritized replay buffer initialization
    old_init = """    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000):"""
    new_init = """    def __init__(self, capacity, alpha=None, beta_start=None, beta_end=None, beta_frames=None):"""
    content = content.replace(old_init, new_init)
    
    # Fix hardcoded priority parameters
    old_priority_init = """        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Small constant to ensure non-zero priorities
        self.max_priority = 1.0
        
        # Statistics for rare event detection
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0, 'sum': 0.0, 'sum_sq': 0.0}
        self.conversion_count = 0
        self.total_experiences = 0
        
        # Priority decay to prevent old high-priority experiences from dominating
        self.priority_decay = 0.999  # Slight decay per step
        self.decay_step = 0"""
    
    new_priority_init = """        # Get all parameters from discovered patterns
        priority_params = get_priority_params()
        
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha if alpha is not None else priority_params['alpha']
        self.beta_start = beta_start if beta_start is not None else priority_params['beta_start']
        self.beta_end = beta_end if beta_end is not None else priority_params['beta_end']
        self.beta_frames = beta_frames if beta_frames is not None else priority_params['beta_frames']
        self.frame = 1
        self.epsilon = priority_params['epsilon']  # Discovered constant for non-zero priorities
        self.max_priority = 1.0
        
        # Statistics for rare event detection
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0, 'sum': 0.0, 'sum_sq': 0.0}
        self.conversion_count = 0
        self.total_experiences = 0
        
        # Priority decay from discovered patterns
        self.priority_decay = priority_params['priority_decay']
        self.decay_step = 0"""
    
    content = content.replace(old_priority_init, new_priority_init)
    
    # Fix conversion detection threshold
    content = re.sub(
        r'if info\.get\(\'conversion\', False\) or reward > 0\.1:',
        'conversion_threshold = get_conversion_bonus()\n        if info.get(\'conversion\', False) or reward > conversion_threshold:',
        content
    )
    
    # Fix exploration bonus threshold
    content = re.sub(
        r'if info\.get\(\'exploration_bonus\', 0\) > 0\.5:',
        'exploration_params = get_epsilon_params()\n        if info.get(\'exploration_bonus\', 0) > exploration_params[\'exploration_bonus_weight\']:',
        content
    )
    
    # Write back the fixed content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed critical hardcoded values in {file_path.name}")

def fix_fortified_environment():
    """Fix hardcoded values in fortified environment"""
    file_path = Path('/home/hariravichandran/AELP/fortified_environment_no_hardcoding.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add import for discovered parameter config
    if 'from discovered_parameter_config import' not in content:
        # Find the last import line and add after it
        lines = content.split('\n')
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                last_import_idx = i
        
        # Insert new import after last import
        lines.insert(last_import_idx + 1, 'from discovered_parameter_config import get_config, get_epsilon_params')
        content = '\n'.join(lines)
    
    # Fix hardcoded episode length
    content = re.sub(
        r'self\.max_steps = 1000  # Standard episode length',
        'self.max_steps = get_config().get_learning_params().get("episode_length", 1000)  # Discovered episode length',
        content
    )
    
    # Fix hardcoded batch parameters
    content = re.sub(
        r'batch_size = 100  # Standard batch size for efficiency',
        'batch_size = get_config().get_learning_params()["batch_size"]  # Discovered batch size',
        content
    )
    
    content = re.sub(
        r'flush_interval = 5\.0  # Flush every 5 seconds',
        'flush_interval = get_config().get_learning_params().get("flush_interval", 5.0)  # Discovered flush interval',
        content
    )
    
    # Fix hardcoded competition parameters
    content = re.sub(
        r'num_competitors = 6  # Default',
        'num_competitors = get_config().get_learning_params().get("num_competitors", 6)  # Discovered competition level',
        content
    )
    
    content = re.sub(
        r'num_competitors = int\(4 \+ avg_effectiveness \* 8\)  # 4-12 competitors',
        'base_competitors = get_config().get_learning_params().get("base_competitors", 4)\n                num_competitors = int(base_competitors + avg_effectiveness * 8)',
        content
    )
    
    # Fix hardcoded slots
    content = re.sub(
        r'\'num_slots\': 4,  # Standard search results page',
        "'num_slots': get_config().get_learning_params().get('num_slots', 4),  # Discovered slots",
        content
    )
    
    # Fix hardcoded state dimension
    content = re.sub(
        r'state_dim = 45',
        'state_dim = get_config().get_neural_network_params().get("state_dim", 45)  # Discovered state dimension',
        content
    )
    
    # Write back the fixed content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed critical hardcoded values in {file_path.name}")

def add_conversion_threshold_method():
    """Add method to get conversion threshold from patterns"""
    file_path = Path('/home/hariravichandran/AELP/fortified_rl_agent_no_hardcoding.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add method to get conversion threshold
    method_to_add = """
    def _get_conversion_threshold_from_patterns(self) -> float:
        \"\"\"Get conversion threshold from discovered patterns\"\"\"
        return get_conversion_bonus()
"""
    
    # Find the end of the _is_rare_event method and add the new method
    if '_get_conversion_threshold_from_patterns' not in content:
        # Find a good place to insert this method
        if 'def _is_rare_event(self, experience_data):' in content:
            # Add after _is_rare_event method
            method_end = content.find('def _is_rare_event(self, experience_data):')
            # Find the end of this method
            method_start = content.find('\n    def ', method_end + 1)
            if method_start == -1:
                method_start = content.find('\nclass ', method_end + 1)
            
            if method_start != -1:
                content = content[:method_start] + method_to_add + content[method_start:]
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Run all critical fixes"""
    print("🔧 Fixing critical hardcoded values...")
    
    try:
        fix_fortified_agent()
        add_conversion_threshold_method()
        fix_fortified_environment()
        print("\n✅ Critical hardcoding fixes completed!")
        
    except Exception as e:
        print(f"❌ Error during fixes: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)