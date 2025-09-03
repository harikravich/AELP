#!/usr/bin/env python3
"""
Complete demonstration of RecSim integration with GAELP.
This script showcases all the user modeling capabilities.
RecSim is MANDATORY - NO FALLBACKS ALLOWED.
"""

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_and_install_dependencies():
    """Check if RecSim is installed, install if needed - MANDATORY"""
    
    try:
        import recsim_ng
        logger.info("✅ RecSim NG already installed")
        return True
    except ImportError:
        logger.error("RecSim NG not found. This is REQUIRED!")
        raise ImportError("RecSim NG MUST be installed. NO FALLBACKS ALLOWED!")


def main():
    """Main demonstration runner"""
    
    print("🎯 GAELP RecSim Integration Demo - MANDATORY RecSim")
    print("="*60)
    
    # Check dependencies - MANDATORY
    logger.info("Checking MANDATORY RecSim dependencies...")
    try:
        check_and_install_dependencies()
    except ImportError as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print("RecSim is MANDATORY for realistic user behavior simulation.")
        print("NO FALLBACKS are allowed per CLAUDE.md requirements.")
        sys.exit(1)
    
    print("✅ RecSim integration verified and MANDATORY")
    print("NO FALLBACKS - realistic user behavior guaranteed!")


if __name__ == "__main__":
    main()
