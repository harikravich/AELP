#!/usr/bin/env python3
"""
Complete demonstration of RecSim integration with GAELP.
This script showcases all the user modeling capabilities.
NO FALLBACKS ALLOWED - RecSim is MANDATORY.
"""

import sys
import subprocess
import logging

# NO FALLBACKS - RecSim MUST work
from NO_FALLBACKS import StrictModeEnforcer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_and_install_dependencies():
    """Check if RecSim is installed, install if needed - NO FALLBACKS"""
    
    try:
        import recsim_ng
        logger.info("✓ RecSim NG already installed")
        return True
    except ImportError:
        logger.info("RecSim NG not found. Installing...")
        
        try:
            # Run the installation script
            result = subprocess.run([sys.executable, "install_recsim.py"], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✓ RecSim installation completed")
                return True
            else:
                logger.error(f"Installation script failed: {result.stderr}")
                StrictModeEnforcer.enforce('RECSIM_INSTALLATION', fallback_attempted=True)
                raise RuntimeError("RecSim installation FAILED. Cannot proceed without RecSim. NO FALLBACKS!")
                
        except subprocess.TimeoutExpired:
            logger.error("Installation timed out")
            StrictModeEnforcer.enforce('RECSIM_INSTALLATION', fallback_attempted=True)
            raise RuntimeError("RecSim installation TIMED OUT. Cannot proceed without RecSim. NO FALLBACKS!")
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            StrictModeEnforcer.enforce('RECSIM_INSTALLATION', fallback_attempted=True)
            raise RuntimeError(f"RecSim installation FAILED: {e}. Cannot proceed without RecSim. NO FALLBACKS!")


def run_demonstrations():
    """Run all demonstration scripts"""
    
    demonstrations = [
        {
            'script': 'recsim_user_model.py',
            'description': 'RecSim User Model Test'
        },
        {
            'script': 'enhanced_simulator.py', 
            'description': 'Enhanced Simulator with RecSim Integration'
        },
        {
            'script': 'test_recsim_integration.py',
            'description': 'Comprehensive Integration Test'
        }
    ]
    
    for demo in demonstrations:
        print(f"\n{'='*60}")
        print(f"Running: {demo['description']}")
        print(f"Script: {demo['script']}")
        print('='*60)
        
        try:
            result = subprocess.run([sys.executable, demo['script']], 
                                  timeout=120, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ {demo['description']} completed successfully")
            else:
                logger.error(f"✗ {demo['description']} FAILED")
                raise RuntimeError(f"Demo {demo['script']} FAILED. All components must work. NO FALLBACKS!")
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {demo['script']} timed out")
            raise RuntimeError(f"Demo {demo['script']} TIMED OUT. All components must work. NO FALLBACKS!")
        except FileNotFoundError:
            logger.error(f"✗ {demo['script']} not found")
            raise RuntimeError(f"Demo {demo['script']} NOT FOUND. All components must exist. NO FALLBACKS!")
        except Exception as e:
            logger.error(f"✗ Error running {demo['script']}: {e}")
            raise RuntimeError(f"Demo {demo['script']} FAILED: {e}. All components must work. NO FALLBACKS!")


def show_summary():
    """Show summary of what was implemented"""
    
    print(f"\n{'='*60}")
    print("GAELP RecSim Integration Summary")
    print('='*60)
    
    print("""
🎯 IMPLEMENTED FEATURES:

1. RecSim NG Integration:
   ✓ MANDATORY RecSim NG installation and dependencies
   ✓ Created sophisticated user behavior models
   ✓ Integrated with existing enhanced_simulator.py

2. User Segment Modeling:
   ✓ Impulse Buyer - High conversion, low price sensitivity
   ✓ Researcher - High click rate, very low conversion  
   ✓ Loyal Customer - High conversion for preferred brands
   ✓ Window Shopper - Low engagement, high price sensitivity
   ✓ Price Conscious - Conversion depends heavily on price
   ✓ Brand Loyalist - Strong preference for specific brands

3. Realistic Behavior Patterns:
   ✓ Time-of-day preferences (morning vs evening engagement)
   ✓ Device preferences (mobile vs desktop behavior)
   ✓ Fatigue modeling (users get tired of seeing ads)
   ✓ Interest dynamics (engagement changes over time)
   ✓ Price sensitivity modeling
   ✓ Brand affinity effects

4. Enhanced Simulation Features:
   ✓ Probabilistic modeling with TensorFlow Probability
   ✓ Realistic revenue calculations
   ✓ Time-spent-on-page modeling
   ✓ User state persistence across interactions
   ✓ Comprehensive analytics and reporting

5. Integration with GAELP:
   ✓ Seamless integration with existing auction system
   ✓ MANDATORY RecSim - NO FALLBACKS ALLOWED
   ✓ Enhanced reward calculation based on user responses
   ✓ Real-time user behavior analytics

6. Files Created:
   ✓ recsim_user_model.py - Core user modeling system
   ✓ Enhanced enhanced_simulator.py - Integrated simulation
   ✓ install_recsim.py - Automated installation
   ✓ test_recsim_integration.py - Comprehensive testing
   ✓ Updated requirements.txt - All dependencies

📊 USER SEGMENT CHARACTERISTICS:

• Impulse Buyers: 8% CTR, 15% conversion, low price sensitivity
• Researchers: 12% CTR, 2% conversion, high price sensitivity  
• Loyal Customers: 15% CTR, 25% conversion, brand focused
• Window Shoppers: 5% CTR, 1% conversion, very price sensitive
• Price Conscious: 6% CTR, 8% conversion, extremely price sensitive
• Brand Loyalists: 18% CTR, 30% conversion, brand obsessed

🚀 USAGE:
   python install_recsim.py          # Install dependencies
   python enhanced_simulator.py      # Run enhanced simulation  
   python test_recsim_integration.py # Run comprehensive tests
   python recsim_user_model.py       # Test user models directly

💡 RecSim integration is MANDATORY for realistic user behavior.
   NO FALLBACKS are allowed - system will fail if RecSim unavailable.
   This ensures proper learning and realistic behavior simulation.
""")


def main():
    """Main demonstration runner"""
    
    print("🎯 GAELP RecSim Integration Demo - NO FALLBACKS ALLOWED")
    print("="*60)
    
    # Check and install dependencies - MANDATORY
    logger.info("Step 1: Checking dependencies...")
    if not check_and_install_dependencies():
        # This should never be reached due to exceptions above
        StrictModeEnforcer.enforce('RECSIM_CRITICAL', fallback_attempted=True)
        raise RuntimeError("RecSim installation check FAILED. NO FALLBACKS ALLOWED!")
    
    print(f"\n{'='*60}")
    print("Step 2: Running demonstrations...")
    
    # Run demonstrations - ALL MUST PASS
    run_demonstrations()
    
    # Show summary
    show_summary()
    
    print(f"\n{'='*60}")
    print("🎉 Demo completed! RecSim integration is MANDATORY and working.")
    print("="*60)


if __name__ == "__main__":
    main()