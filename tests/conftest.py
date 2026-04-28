# conftest.py for shared test configuration
import pytest
import sys
import os

# Add the project root to the Python path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# You can add shared fixtures here if needed