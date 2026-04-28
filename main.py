#!/usr/bin/env python3
"""Main entry point for LangGraph Multi-Agent Programming Assistant"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Load environment variables from project root
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))

# Set working directory to project root
os.chdir(PROJECT_ROOT)

from src.app import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)