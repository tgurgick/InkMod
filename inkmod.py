#!/usr/bin/env python3
"""InkMod CLI entry point."""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import after adding to path
from cli.commands import cli

def main():
    """Main entry point for InkMod."""
    cli()

if __name__ == '__main__':
    main() 