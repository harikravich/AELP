#!/usr/bin/env python3
"""
GAELP Dashboard Launcher
Easy way to start the performance visualization dashboard
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def install_requirements():
    """Install dashboard requirements if needed"""
    requirements_file = Path(__file__).parent / "requirements_dashboard.txt"
    
    print("🔧 Installing dashboard requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    dashboard_file = Path(__file__).parent / "dashboard.py"
    
    print("🚀 Launching GAELP Performance Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔄 Use Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_file),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("🎯 GAELP Performance Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("dashboard.py").exists():
        print("❌ Error: dashboard.py not found in current directory")
        print("Please run this script from the AELP project root")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements_dashboard.txt")
        sys.exit(1)
    
    # Small delay to let user see the installation messages
    time.sleep(2)
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()