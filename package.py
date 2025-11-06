"""
Main script for creating the executable file using PyInstaller.
Run this script to package the trading bot application.
"""

import os
import subprocess
import sys
import platform

def main():
    """
    Main function to create the executable file.
    """
    print("Starting packaging process for Trading Bot...")
    
    # Determine the operating system
    os_name = platform.system()
    print(f"Detected operating system: {os_name}")
    
    if os_name != "Windows":
        print("Note: You are not running on Windows. The .exe file will need to be built on a Windows system.")
        print("This script will still proceed with packaging for demonstration purposes.")
    
    # Make sure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Install required packages if needed
    print("Checking and installing required packages...")
    subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run PyInstaller
    print("Running PyInstaller to create executable...")
    pyinstaller_cmd = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "trading_bot.spec"
    ]
    
    try:
        subprocess.call(pyinstaller_cmd)
        print("PyInstaller completed successfully.")
        
        # Check if the executable was created
        dist_dir = os.path.join(script_dir, "dist")
        exe_path = os.path.join(dist_dir, "TradingBot.exe" if os_name == "Windows" else "TradingBot")
        
        if os.path.exists(dist_dir):
            print(f"Distribution directory created at: {dist_dir}")
            if os.path.exists(exe_path):
                print(f"Executable created successfully at: {exe_path}")
            else:
                print("Executable file not found. Check the 'dist' directory for output files.")
        else:
            print("Distribution directory not created. Check PyInstaller output for errors.")
        
    except Exception as e:
        print(f"Error during packaging: {e}")
        return 1
    
    print("Packaging process completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
