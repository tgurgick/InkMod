#!/usr/bin/env python3
"""Test script for the redline functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.commands import _handle_redline_mode

def test_redline():
    """Test the redline functionality with a sample response."""
    
    sample_response = """Hello there! I hope this email finds you well. I wanted to follow up on our recent conversation about the project timeline. We discussed several key milestones that need to be completed by the end of the month. Please let me know if you have any questions or concerns about the proposed schedule."""
    
    print("Testing redline functionality...")
    print("Sample response:")
    print(sample_response)
    print("\n" + "="*50)
    
    # Note: This would normally be interactive, but for testing we'll just show the structure
    print("Redline mode would start here with numbered sentences:")
    
    import re
    sentences = re.split(r'(?<=[.!?])\s+', sample_response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    for i, sentence in enumerate(sentences, 1):
        print(f"Line {i}: {sentence}")
    
    print("\nCommands available:")
    print("- redline <number> - Edit a specific sentence")
    print("- save - Save all changes and exit")
    print("- quit - Exit without saving")
    print("- show - Show current content")

if __name__ == "__main__":
    test_redline() 