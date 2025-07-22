"""
Constants for research experiments.

This module defines common paths and configurations used across all experiments
to ensure consistency and avoid hardcoded workstation-specific paths.
"""

import os
from pathlib import Path

# Get the project root directory (research folder's parent)
RESEARCH_DIR = Path(__file__).parent.parent
PROJECT_ROOT = RESEARCH_DIR.parent

# Define common directories
RESULTS_DIR = str(RESEARCH_DIR / "results")
MODELS_DIR = str(RESEARCH_DIR / "models")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
