'''Common utility functions'''

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

def setup_logging(log_level: str = "INFO"):
    '''Setup logging configuration'''
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/cli-help.log')
        ]
    )

def validate_yaml_structure(data: Dict[str, Any]) -> bool:
    '''Validate YAML structure for command definitions'''
    required_fields = ['tool_name', 'description', 'commands']

    for field in required_fields:
        if field not in data:
            return False

    # Validate commands structure
    for command in data.get('commands', []):
        if not isinstance(command, dict):
            return False
        if 'name' not in command or 'description' not in command:
            return False

    return True

def ensure_directories(paths: List[Path]):
    '''Ensure directories exist'''
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
