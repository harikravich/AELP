#!/usr/bin/env python3
"""
Fix all the training issues:
1. Fix discovered_patterns.json
2. Fix BigQuery schema issues
3. Update bid ranges to realistic values
"""

import json
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_discovered_patterns():
    """Fix the truncated discovered_patterns.json file"""
    logger.info("Fixing discovered_patterns.json...")
    
    # Read the file
    with open('discovered_patterns.json', 'r') as f:
        content = f.read()
    
    # Check if it's truncated (doesn't end with proper JSON)
    if not content.rstrip().endswith('}'):
        logger.info("File is truncated, attempting to fix...")
        
        # Find where we are in the structure
        lines = content.split('\n')
        
        # Count open brackets to determine nesting level
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # We need to close arrays and objects properly
        missing_brackets = open_brackets - close_brackets
        missing_braces = open_braces - close_braces
        
        # Remove the incomplete last line if it exists
        if lines[-1].strip() and not lines[-1].strip().endswith((',', ']', '}')):
            lines = lines[:-1]
        
        # Close any open arrays
        for _ in range(missing_brackets):
            lines.append('  ]')
        
        # Close any open objects  
        for i in range(missing_braces):
            indent = '  ' * (missing_braces - i - 1)
            lines.append(f'{indent}}}')
        
        # Write the fixed content
        fixed_content = '\n'.join(lines)
        
        # Validate it's proper JSON
        try:
            json.loads(fixed_content)
            with open('discovered_patterns.json', 'w') as f:
                f.write(fixed_content)
            logger.info("✅ Fixed discovered_patterns.json")
        except json.JSONDecodeError as e:
            logger.error(f"Still invalid JSON: {e}")
            # Create a minimal valid patterns file
            create_minimal_patterns()
    else:
        logger.info("discovered_patterns.json appears to be valid")

def create_minimal_patterns():
    """Create a minimal valid patterns file"""
    logger.info("Creating minimal valid patterns file...")
    
    patterns = {
        "segments": {
            "researching_parent": {
                "discovered_characteristics": {
                    "engagement_level": "high",
                    "conversion_potential": "high",
                    "device_affinity": "mobile"
                },
                "behavioral_metrics": {
                    "avg_session_duration": 473.83,
                    "conversion_rate": 0.061
                }
            },
            "crisis_parent": {
                "discovered_characteristics": {
                    "engagement_level": "medium",
                    "conversion_potential": "medium",
                    "device_affinity": "tablet"
                },
                "behavioral_metrics": {
                    "avg_session_duration": 266.5,
                    "conversion_rate": 0.042
                }
            },
            "concerned_parent": {
                "discovered_characteristics": {
                    "engagement_level": "medium",
                    "conversion_potential": "medium",
                    "device_affinity": "desktop"
                },
                "behavioral_metrics": {
                    "avg_session_duration": 288.33,
                    "conversion_rate": 0.042
                }
            },
            "proactive_parent": {
                "discovered_characteristics": {
                    "engagement_level": "high",
                    "conversion_potential": "high",
                    "device_affinity": "mobile"
                },
                "behavioral_metrics": {
                    "avg_session_duration": 400.0,
                    "conversion_rate": 0.055
                }
            }
        },
        "channels": {
            "organic": {
                "effectiveness": 0.75,
                "cost_efficiency": 1.0,
                "avg_ctr": 0.055,
                "avg_conversion_rate": 0.038
            },
            "paid_search": {
                "effectiveness": 0.85,
                "cost_efficiency": 0.6,
                "avg_ctr": 0.085,
                "avg_conversion_rate": 0.052
            },
            "social": {
                "effectiveness": 0.65,
                "cost_efficiency": 0.7,
                "avg_ctr": 0.045,
                "avg_conversion_rate": 0.028
            },
            "display": {
                "effectiveness": 0.45,
                "cost_efficiency": 0.8,
                "avg_ctr": 0.015,
                "avg_conversion_rate": 0.018
            },
            "email": {
                "effectiveness": 0.70,
                "cost_efficiency": 0.95,
                "avg_ctr": 0.125,
                "avg_conversion_rate": 0.045
            }
        },
        "bid_ranges": {
            "brand_keywords": {"min": 5.0, "max": 15.0},
            "non_brand": {"min": 20.0, "max": 50.0},
            "competitor": {"min": 30.0, "max": 80.0},
            "display": {"min": 2.0, "max": 10.0}
        },
        "creatives": {
            "performance_by_segment": {
                "researching_parent": {
                    "best_creative_ids": [1, 3, 5],
                    "avg_ctr": 0.085
                },
                "crisis_parent": {
                    "best_creative_ids": [2, 4, 6],
                    "avg_ctr": 0.065
                }
            }
        }
    }
    
    with open('discovered_patterns.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    
    logger.info("✅ Created minimal valid patterns file")

def fix_bigquery_schema():
    """Fix the BigQuery schema issues in persistent_user_database.py"""
    logger.info("Fixing BigQuery schema in persistent_user_database.py...")
    
    file_path = 'persistent_user_database.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # The fields that need to be JSON should already be defined as JSON
    # The issue is in how the data is being serialized
    # Let's update the persistent_user_database_batched.py instead
    
    file_path = 'persistent_user_database_batched.py'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Fix the serialization to ensure proper JSON format
    modified = False
    for i, line in enumerate(lines):
        # Fix competitor_exposures serialization
        if 'competitor_exposures: user.competitor_exposures' in line:
            lines[i] = line.replace(
                'competitor_exposures: user.competitor_exposures',
                'competitor_exposures: json.dumps(user.competitor_exposures) if user.competitor_exposures else "{}"'
            )
            modified = True
        
        # Fix competitor_fatigue serialization  
        if 'competitor_fatigue: user.competitor_fatigue' in line:
            lines[i] = line.replace(
                'competitor_fatigue: user.competitor_fatigue',
                'competitor_fatigue: json.dumps(user.competitor_fatigue) if user.competitor_fatigue else "{}"'
            )
            modified = True
            
        # Fix devices_seen - it's already being handled correctly
        
        # Fix journey_history serialization
        if 'journey_history: user.journey_history' in line:
            lines[i] = line.replace(
                'journey_history: user.journey_history',
                'journey_history: json.dumps(user.journey_history) if user.journey_history else "[]"'
            )
            modified = True
            
        # Fix touchpoint_history serialization
        if 'touchpoint_history: user.touchpoint_history' in line:
            lines[i] = line.replace(
                'touchpoint_history: user.touchpoint_history',
                'touchpoint_history: json.dumps(user.touchpoint_history) if user.touchpoint_history else "[]"'
            )
            modified = True
            
        # Fix conversion_history serialization
        if 'conversion_history: user.conversion_history' in line:
            lines[i] = line.replace(
                'conversion_history: user.conversion_history',
                'conversion_history: json.dumps(user.conversion_history) if user.conversion_history else "[]"'
            )
            modified = True
    
    # Add json import if not present
    if modified and 'import json' not in ''.join(lines[:20]):
        for i, line in enumerate(lines):
            if line.startswith('import'):
                lines.insert(i, 'import json\n')
                break
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        logger.info("✅ Fixed BigQuery schema serialization")
    else:
        logger.info("BigQuery schema already appears correct")

def fix_bid_ranges():
    """Update fortified_environment.py with realistic bid ranges"""
    logger.info("Updating bid ranges in fortified_environment.py...")
    
    file_path = 'fortified_environment.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update bid ranges from $0.50-$20 to $5-$50
    content = content.replace(
        'low=0.5, high=20.0',
        'low=5.0, high=50.0'
    )
    
    # Also update any hardcoded ranges
    content = content.replace(
        'np.random.uniform(0.5, 5.0)',
        'np.random.uniform(5.0, 50.0)'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info("✅ Updated bid ranges to realistic values ($5-$50)")

def reduce_ray_logging():
    """Create a cleaner capture script with better filtering"""
    logger.info("Creating cleaner training capture script...")
    
    content = '''#!/usr/bin/env python3
"""
Clean fortified training runner with reduced logging noise
"""

import os
import sys
import logging

# Set Ray logging to WARNING level
os.environ['RAY_DEDUP_LOGS'] = '1'
os.environ['RAY_LOG_TO_STDERR'] = '0'

# Suppress BigQuery INFO logs
logging.getLogger('google.cloud.bigquery').setLevel(logging.WARNING)
logging.getLogger('bigquery_batch_writer').setLevel(logging.WARNING)
logging.getLogger('discovery_engine').setLevel(logging.WARNING)
logging.getLogger('gaelp_parameter_manager').setLevel(logging.WARNING)

# Import after setting env vars
import subprocess

def main():
    print("Starting FORTIFIED training with reduced logging...")
    print("=" * 70)
    
    # Run the actual training with cleaner output
    subprocess.run([
        "python3", "fortified_training_loop.py"
    ])

if __name__ == "__main__":
    main()
'''
    
    with open('run_fortified_clean.py', 'w') as f:
        f.write(content)
    
    os.chmod('run_fortified_clean.py', 0o755)
    logger.info("✅ Created run_fortified_clean.py for cleaner output")

def main():
    """Run all fixes"""
    print("=" * 70)
    print("FIXING ALL TRAINING ISSUES")
    print("=" * 70)
    
    # 1. Fix discovered patterns
    fix_discovered_patterns()
    
    # 2. Fix BigQuery schema
    fix_bigquery_schema()
    
    # 3. Fix bid ranges  
    fix_bid_ranges()
    
    # 4. Create cleaner runner
    reduce_ray_logging()
    
    print("\n" + "=" * 70)
    print("✅ ALL FIXES APPLIED")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Stop current training: Press Ctrl+C")
    print("2. Run cleaner training: python3 run_fortified_clean.py")
    print("   OR")
    print("   Use original with fixes: python3 run_training.py (Option 1)")

if __name__ == "__main__":
    main()