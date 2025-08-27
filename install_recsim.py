#!/usr/bin/env python3
"""
Installation script for RecSim NG and dependencies.
This script will install RecSim NG and test the installation.
"""

import subprocess
import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_package(package_name: str, version: str = None):
    """Install a package using pip"""
    try:
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
            
        logger.info(f"Installing {package_spec}...")
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_spec
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"Successfully installed {package_spec}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_spec}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def test_import(module_name: str, package_name: str = None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        logger.info(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        pkg = package_name or module_name
        logger.warning(f"✗ Failed to import {module_name} from {pkg}: {e}")
        return False


def main():
    """Main installation and testing function"""
    
    logger.info("Installing RecSim NG and dependencies...")
    print("=" * 50)
    
    # Packages to install
    packages = [
        ("tensorflow", "2.13.0"),
        ("dm-tree", "0.1.8"),
        ("recsim-ng", "0.2.0"),
    ]
    
    # Optional packages that might be needed
    optional_packages = [
        ("edward2", None),
        ("tensorflow-probability", "0.21.0"),
    ]
    
    # Install core packages
    success_count = 0
    for package_name, version in packages:
        if install_package(package_name, version):
            success_count += 1
    
    # Install optional packages (don't fail if these don't work)
    for package_name, version in optional_packages:
        install_package(package_name, version)
    
    print("\n" + "=" * 50)
    logger.info("Testing imports...")
    
    # Test imports
    imports_to_test = [
        ("tensorflow", "tensorflow"),
        ("dm_tree", "dm-tree"),
        ("recsim_ng", "recsim-ng"),
        ("recsim_ng.core.value", "recsim-ng"),
        ("recsim_ng.lib.tensorflow.entity", "recsim-ng"),
    ]
    
    import_success_count = 0
    for module_name, package_name in imports_to_test:
        if test_import(module_name, package_name):
            import_success_count += 1
    
    print("\n" + "=" * 50)
    logger.info("Installation Summary:")
    print(f"Packages installed: {success_count}/{len(packages)}")
    print(f"Imports successful: {import_success_count}/{len(imports_to_test)}")
    
    if import_success_count >= 3:  # Core imports working
        logger.info("✓ RecSim NG installation appears successful!")
        
        # Test our user model
        logger.info("Testing RecSim user model...")
        try:
            from recsim_user_model import RecSimUserModel, UserSegment
            
            model = RecSimUserModel()
            logger.info("✓ RecSim user model loaded successfully")
            
            # Quick test
            user_id = "test_user"
            model.generate_user(user_id, UserSegment.IMPULSE_BUYER)
            
            response = model.simulate_ad_response(
                user_id=user_id,
                ad_content={
                    'creative_quality': 0.8,
                    'price_shown': 50.0,
                    'brand_match': 0.7,
                    'relevance_score': 0.8
                },
                context={'hour': 14, 'device': 'mobile'}
            )
            
            logger.info(f"✓ Test simulation successful: clicked={response['clicked']}, converted={response['converted']}")
            
        except Exception as e:
            logger.error(f"✗ Error testing user model: {e}")
    
    else:
        logger.warning("⚠ RecSim NG installation may have issues. Some imports failed.")
        logger.info("The system will fall back to the basic user model.")
    
    print("\nYou can now run:")
    print("  python enhanced_simulator.py")
    print("  python recsim_user_model.py")


if __name__ == "__main__":
    main()