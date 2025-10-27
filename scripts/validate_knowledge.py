#!/usr/bin/env python3
'''
Validate YAML knowledge files
'''

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from cli_help.utils import validate_yaml_structure

def main():
    knowledge_dir = Path("knowledge/commands")

    if not knowledge_dir.exists():
        print(f"Knowledge directory not found: {knowledge_dir}")
        sys.exit(1)

    errors = 0

    for yaml_file in knowledge_dir.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            if validate_yaml_structure(data):
                print(f"✓ {yaml_file.name}")
            else:
                print(f"✗ {yaml_file.name} - Invalid structure")
                errors += 1

        except Exception as e:
            print(f"✗ {yaml_file.name} - Error: {e}")
            errors += 1

    if errors > 0:
        print(f"\nFound {errors} errors")
        sys.exit(1)
    else:
        print("\nAll files valid!")

if __name__ == "__main__":
    main()
