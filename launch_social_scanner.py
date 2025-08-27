#!/usr/bin/env python3
"""
Launch script for the Social Media Scanner
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the social media scanner"""
    
    print("🚀 Launching Social Media Scanner - Aura Lead Generation Tool")
    print("=" * 60)
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    scanner_script = script_dir / "social_media_scanner.py"
    
    if not scanner_script.exists():
        print(f"❌ Error: Scanner script not found at {scanner_script}")
        return
    
    print("📱 Starting Teen Social Media Scanner...")
    print("💡 This tool will help parents find hidden teen accounts")
    print("🔒 100% private - no teen data stored")
    print("⚡ Results in under 60 seconds")
    print()
    print("🌐 Opening web interface...")
    print("   Use Ctrl+C to stop the scanner")
    print("=" * 60)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(scanner_script),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "light",
            "--theme.primaryColor", "#4CAF50"
        ]
        
        subprocess.run(cmd, cwd=str(script_dir))
        
    except KeyboardInterrupt:
        print("\n\n✅ Social Media Scanner stopped")
        print("📊 Check scanner_leads.json for captured leads")
    except Exception as e:
        print(f"\n❌ Error launching scanner: {e}")
        print("\nTrying alternative launch method...")
        
        # Alternative launch
        os.system(f"cd {script_dir} && streamlit run social_media_scanner.py")

if __name__ == "__main__":
    main()