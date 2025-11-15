"""
CineMatch V2.1.0 - Test Configuration

Pytest configuration and fixtures for the test suite.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
