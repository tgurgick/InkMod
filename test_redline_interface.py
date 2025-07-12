#!/usr/bin/env python3
"""Test the simplified redline interface."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_redline_interface():
    """Test the simplified redline interface."""
    
    sample_response = """Hello there! I hope this email finds you well. I wanted to follow up on our recent conversation about the project timeline. We discussed several key milestones that need to be completed by the end of the month. Please let me know if you have any questions or concerns about the proposed schedule."""
    
    print("Testing simplified redline interface...")
    print("Sample response:")
    print(sample_response)
    print("\n" + "="*50)
    
    # Show how the interface would work
    import re
    sentences = re.split(r'(?<=[.!?])\s+', sample_response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print("🔴 Redline Mode")
    print("Generated Content:")
    for i, sentence in enumerate(sentences, 1):
        print(f"Line {i}: {sentence}")
    
    print("\nRedline Commands: <number> | save | quit | show")
    print("\nExample interactions:")
    print("Command: 1")
    print("Editing Line 1:")
    print("Original: Hello there! I hope this email finds you well.")
    print("New version: Hi there! I hope you're doing well.")
    print("✅ Line 1 updated!")
    
    print("\nCommand: 3")
    print("Editing Line 3:")
    print("Original: We discussed several key milestones that need to be completed by the end of the month.")
    print("New version: We outlined key milestones for completion by month-end.")
    print("✅ Line 3 updated!")
    
    print("\nCommand: save")
    print("💾 Redline feedback saved to feedback.json")
    print("📊 Captured 2 revision pairs for training")
    print("✅ Changes saved and feedback captured for training!")

if __name__ == "__main__":
    test_redline_interface() 