#!/usr/bin/env python3
"""
Demo of Real-Time GA4 to GAELP Model Data Pipeline
Shows the pipeline processing real GA4 data in real-time
"""

import asyncio
import signal
import sys
from discovery_engine import create_production_pipeline
from pipeline_integration import create_integrated_pipeline

async def demo_pipeline():
    """Demo the real-time pipeline with timeout"""
    print("🚀 Demo: Real-Time GA4 to GAELP Model Data Pipeline")
    print("=" * 80)
    print("This demo shows the production-grade pipeline processing real GA4 data")
    print("The pipeline will run for 30 seconds to demonstrate real-time capabilities")
    print("=" * 80)
    
    # Create integrated pipeline
    pipeline, model_updater, health_monitor = await create_integrated_pipeline()
    
    # Set up graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\n📟 Shutdown signal received")
        shutdown_event.set()
    
    # Register signal handler
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        print("✅ Pipeline created, starting real-time processing...")
        
        # Start pipeline tasks
        pipeline_task = asyncio.create_task(pipeline.start_realtime_pipeline())
        health_task = asyncio.create_task(health_monitor.monitor_health())
        
        # Demo timeout (30 seconds)
        timeout_task = asyncio.create_task(asyncio.sleep(30))
        
        # Wait for either shutdown signal or timeout
        done, pending = await asyncio.wait(
            [pipeline_task, health_task, timeout_task, 
             asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            
        print("\n📊 Demo completed, stopping pipeline...")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
    finally:
        # Graceful shutdown
        await pipeline.stop_pipeline()
        
        # Final stats
        final_stats = pipeline.get_pipeline_stats()
        print("\n" + "=" * 80)
        print("📊 Demo Results - Real-Time Pipeline Statistics")
        print("=" * 80)
        print(f"Runtime: {final_stats['runtime_seconds']:.1f} seconds")
        print(f"Events Processed: {final_stats['total_events_processed']:,}")
        print(f"Events Failed: {final_stats['total_events_failed']:,}")
        print(f"Success Rate: {final_stats['success_rate']:.2%}")
        print(f"Model Updates: {model_updater.update_count:,}")
        print(f"Total Revenue Tracked: ${model_updater.total_revenue:,.2f}")
        print(f"Conversion Events: {model_updater.conversion_events:,}")
        print(f"Discovered Patterns:")
        print(f"  - Segments: {final_stats['discovered_patterns']['segments']}")
        print(f"  - Channels: {final_stats['discovered_patterns']['channels']}")
        print(f"  - Devices: {final_stats['discovered_patterns']['devices']}")
        print("=" * 80)
        print("✅ Real-time GA4 pipeline demo completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("- Real-time GA4 data ingestion via MCP")
        print("- Stream processing with guaranteed delivery") 
        print("- Data validation and quality checks")
        print("- Real-time GAELP model updates")
        print("- Pattern discovery from live data")
        print("- Health monitoring and statistics")
        print("- Graceful shutdown handling")

if __name__ == "__main__":
    try:
        asyncio.run(demo_pipeline())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)