#!/usr/bin/env python3
'''
Build knowledge base and embeddings
'''

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_help.config import Config
from cli_help.knowledge_builder import KnowledgeBuilder
from cli_help.utils import setup_logging

def main():
    setup_logging()

    config = Config()
    builder = KnowledgeBuilder(config)

    print("Building knowledge base...")
    try:
        builder.build_embeddings()
        print("Knowledge base built successfully!")
    except Exception as e:
        print(f"Error building knowledge base: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
