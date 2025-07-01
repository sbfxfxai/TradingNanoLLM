#!/usr/bin/env python3
"""
Startup script for TradingNanoLLM FastAPI backend
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def start_server():
    """Start the FastAPI server"""
    print("Starting TradingNanoLLM API server...")
    try:
        # Start uvicorn server
        subprocess.call([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main startup function"""
    print("🚀 TradingNanoLLM Backend Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found. Please run this script from the backend directory.")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\n" + "=" * 40)
    print("🌐 Starting server at http://localhost:8000")
    print("📚 API docs will be available at http://localhost:8000/docs")
    print("🔧 Health check at http://localhost:8000/health")
    print("=" * 40 + "\n")
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
