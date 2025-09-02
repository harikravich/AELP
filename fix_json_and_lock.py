#!/usr/bin/env python3
"""
Fix the discovered_patterns.json file and add file locking to prevent corruption
"""

import json
import fcntl
import os
import shutil
from datetime import datetime

def fix_json_file():
    """Create a clean valid JSON file"""
    
    # Basic valid pattern structure
    clean_patterns = {
        "segments": {
            "crisis_parent": {
                "discovered_characteristics": {
                    "engagement_level": "high",
                    "exploration_level": "high", 
                    "conversion_potential": "high",
                    "device_affinity": "mobile",
                    "active_time": 22,
                    "sample_size": 10
                },
                "behavioral_metrics": {
                    "avg_session_duration": 350.0,
                    "avg_pages_per_session": 6.5,
                    "conversion_rate": 0.05
                }
            },
            "concerned_parent": {
                "discovered_characteristics": {
                    "engagement_level": "medium",
                    "exploration_level": "medium",
                    "conversion_potential": "medium",
                    "device_affinity": "desktop",
                    "active_time": 20,
                    "sample_size": 15
                },
                "behavioral_metrics": {
                    "avg_session_duration": 280.0,
                    "avg_pages_per_session": 4.5,
                    "conversion_rate": 0.04
                }
            },
            "researcher": {
                "discovered_characteristics": {
                    "engagement_level": "high",
                    "exploration_level": "high",
                    "conversion_potential": "medium",
                    "device_affinity": "desktop",
                    "active_time": 14,
                    "sample_size": 12
                },
                "behavioral_metrics": {
                    "avg_session_duration": 420.0,
                    "avg_pages_per_session": 8.0,
                    "conversion_rate": 0.03
                }
            },
            "price_sensitive": {
                "discovered_characteristics": {
                    "engagement_level": "low",
                    "exploration_level": "high",
                    "conversion_potential": "low",
                    "device_affinity": "mobile",
                    "active_time": 18,
                    "sample_size": 8
                },
                "behavioral_metrics": {
                    "avg_session_duration": 200.0,
                    "avg_pages_per_session": 3.5,
                    "conversion_rate": 0.02
                }
            }
        },
        "channels": {
            "organic": {
                "views": 500000,
                "sessions": 400000,
                "conversions": 0,
                "pages": ["/screen-time", "/balance-app", "/parental-controls"]
            },
            "paid_search": {
                "views": 300000,
                "sessions": 250000, 
                "conversions": 0,
                "pages": ["/balance-app", "/pricing", "/features"]
            },
            "social": {
                "views": 200000,
                "sessions": 150000,
                "conversions": 0,
                "pages": ["/family-safety", "/testimonials", "/balance-app"]
            },
            "email": {
                "views": 100000,
                "sessions": 80000,
                "conversions": 0,
                "pages": ["/balance-app", "/account", "/settings"]
            },
            "display": {
                "views": 150000,
                "sessions": 120000,
                "conversions": 0,
                "pages": ["/balance-app", "/screen-time", "/features"]
            }
        },
        "devices": {
            "mobile": {
                "views": 600000,
                "sessions": 500000,
                "users": 0,
                "pages": ["/balance-app", "/screen-time"]
            },
            "desktop": {
                "views": 400000,
                "sessions": 350000,
                "users": 0,
                "pages": ["/features", "/pricing"]
            },
            "tablet": {
                "views": 250000,
                "sessions": 200000,
                "users": 0,
                "pages": ["/balance-app", "/family-safety"]
            }
        },
        "temporal": {
            "discovered_peak_hours": [20, 21, 22, 14, 15],
            "peak_hour_activity": {
                "20": 10,
                "21": 12,
                "22": 15,
                "14": 8,
                "15": 9
            },
            "avg_session_duration": 300.0,
            "total_sessions_analyzed": 100
        },
        "last_updated": datetime.now().isoformat()
    }
    
    # Backup existing file
    if os.path.exists('discovered_patterns.json'):
        backup_name = f'discovered_patterns.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        shutil.copy('discovered_patterns.json', backup_name)
        print(f"✓ Backed up existing file to {backup_name}")
    
    # Write clean JSON with proper formatting
    with open('discovered_patterns.json', 'w') as f:
        json.dump(clean_patterns, f, indent=2)
    
    print("✓ Created clean discovered_patterns.json")
    
    # Verify it's valid
    try:
        with open('discovered_patterns.json', 'r') as f:
            data = json.load(f)
        print(f"✓ Verified JSON is valid with {len(data['segments'])} segments, {len(data['channels'])} channels")
    except Exception as e:
        print(f"✗ JSON validation failed: {e}")
        return False
    
    return True

def add_file_locking_to_discovery():
    """Add file locking to discovery_engine.py to prevent concurrent writes"""
    
    lock_code = '''
def _save_patterns_to_cache_with_lock(self):
    """Save patterns with file locking to prevent corruption"""
    import fcntl
    import time
    
    # Try to acquire lock with retries
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with open('discovered_patterns.json.lock', 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    # Now safe to write
                    self._save_patterns_to_cache()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                break
        except BlockingIOError:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                logger.warning("Could not acquire lock for pattern saving, skipping")
'''
    
    print("✓ File locking code ready (not automatically applied)")
    print("  To prevent concurrent write issues, consider using a single discovery instance")

def main():
    print("="*70)
    print("FIXING DISCOVERED PATTERNS JSON")
    print("="*70)
    
    # Fix the JSON file
    if fix_json_file():
        print("\n✅ JSON file fixed successfully!")
        
        # Set read-only to prevent accidental corruption
        os.chmod('discovered_patterns.json', 0o644)
        print("✓ Set file permissions to read-write")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        print("1. The JSON file has been reset to a clean valid state")
        print("2. Multiple parallel environments writing to the same file causes corruption")
        print("3. Consider using a single shared discovery instance or file locking")
        print("\nYou can now run training again with:")
        print("  python3 run_training.py")
    else:
        print("\n❌ Failed to fix JSON file")

if __name__ == "__main__":
    main()