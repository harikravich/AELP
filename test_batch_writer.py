#!/usr/bin/env python3
"""
Test the BigQuery batch writer to ensure it solves quota issues
"""

import time
import logging
from datetime import datetime
from persistent_user_database_batched import BatchedPersistentUserDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_batch_writer():
    """Test batch writing functionality."""
    
    print("=" * 70)
    print("TESTING BIGQUERY BATCH WRITER")
    print("=" * 70)
    
    # Initialize database with batch writer
    db = BatchedPersistentUserDatabase(
        use_batch_writer=True,
        batch_size=50,  # Small batch for testing
        flush_interval=3.0  # Quick flush for testing
    )
    
    print("\n1. Creating multiple users rapidly (would normally exceed quota)...")
    
    # Create many users rapidly (simulating parallel environments)
    users_created = 0
    start_time = time.time()
    
    for i in range(100):
        user_id = f"test_user_{datetime.now().timestamp()}_{i}"
        episode_id = f"test_episode_{i}"
        
        try:
            user, created = db.get_or_create_persistent_user(
                user_id=user_id,
                episode_id=episode_id,
                device_fingerprint={"device_id": f"device_{i}"}
            )
            
            if created:
                users_created += 1
            
            # Simulate rapid updates (would trigger quota errors)
            if i % 10 == 0:
                user.awareness_level += 0.1
                user.fatigue_score += 0.05
                db._update_user_in_database(user)
                
        except Exception as e:
            print(f"Error at user {i}: {e}")
            break
    
    elapsed = time.time() - start_time
    
    print(f"\n2. Created/updated {users_created} users in {elapsed:.2f} seconds")
    print(f"   Rate: {users_created/elapsed:.1f} users/second")
    
    # Check batch stats
    stats = db.get_batch_stats()
    if stats:
        print(f"\n3. Batch Writer Statistics:")
        print(f"   Total writes attempted: {stats['total_writes']}")
        print(f"   Successful writes: {stats['successful_writes']}")
        print(f"   Failed writes: {stats['failed_writes']}")
        print(f"   Batch flushes: {stats['batch_flushes']}")
        print(f"   Quota errors: {stats['quota_errors']}")
    
    print("\n4. Waiting for automatic flush...")
    time.sleep(5)
    
    # Final stats
    final_stats = db.get_batch_stats()
    if final_stats:
        print(f"\n5. Final Statistics:")
        print(f"   Success rate: {final_stats['successful_writes']}/{final_stats['total_writes']} "
              f"({final_stats['successful_writes']/max(1, final_stats['total_writes'])*100:.1f}%)")
        print(f"   Quota errors: {final_stats['quota_errors']}")
    
    # Shutdown
    print("\n6. Shutting down batch writer...")
    db.shutdown()
    
    print("\n" + "=" * 70)
    if stats and stats['quota_errors'] == 0:
        print("✅ SUCCESS: No quota errors with batch writing!")
    else:
        print("⚠️  Some issues detected, check logs")
    print("=" * 70)

if __name__ == "__main__":
    test_batch_writer()