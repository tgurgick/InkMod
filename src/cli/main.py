"""Main entry point for InkMod CLI."""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from .commands import cli

def main():
    """Main entry point."""
    cli()

if __name__ == '__main__':
    main() 