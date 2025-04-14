#!/usr/bin/env python
"""
Document Summarizer Application Launcher
----------------------------------------
This script sets up and runs the Document Summarizer Streamlit application.
"""
import os
import argparse
import subprocess
import sys
from dotenv import load_dotenv  # Make sure this import is present

def check_environment():
    """Check if environment variables are set."""
    load_dotenv()  # This loads environment variables from .env file
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY")  # Add OpenRouter API key
    }
    
    available_keys = [key for key, value in api_keys.items() if value]
    
    if not available_keys:
        print("\n⚠️ Warning: No API keys found in environment variables or .env file.")
        print("At least one API key is required for the application to function.")
        print("Please add your API keys to the .env file.\n")
        return False
    
    print("\n✅ Found the following API keys:")
    for key in available_keys:
        print(f"  - {key}")
    
    return True



def install_requirements():
    """Install required packages from requirements.txt."""
    if os.path.exists("requirements.txt"):
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    else:
        print("Error: requirements.txt file not found.")
        return False

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Run Document Summarizer Streamlit App")
    parser.add_argument("--install", action="store_true", help="Install required packages before running")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    parser.add_argument("--check-env", action="store_true", help="Only check environment and exit")
    args = parser.parse_args()
    
    # Check environment if requested
    if args.check_env:
        check_environment()
        return 0
    
    # Install requirements if requested
    if args.install:
        if not install_requirements():
            return 1
    

    
    # Check environment
    check_environment()
    
    # Run the Streamlit app
    print(f"Starting Document Summarizer on port {args.port}...")
    streamlit_args = ["streamlit", "run", "streamlit_app.py", "--server.port", str(args.port)]
    subprocess.call(streamlit_args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())