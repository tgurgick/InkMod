#!/usr/bin/env python3
"""Example demonstrating the redline functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def demonstrate_redline():
    """Demonstrate the redline functionality with a sample workflow."""
    
    print("🔴 InkMod Redline Feature Demo")
    print("=" * 50)
    
    # Sample generated content
    sample_response = """Hello there! I hope this email finds you well. I wanted to follow up on our recent conversation about the project timeline. We discussed several key milestones that need to be completed by the end of the month. Please let me know if you have any questions or concerns about the proposed schedule."""
    
    print("📝 Generated Content:")
    print(sample_response)
    print("\n" + "=" * 50)
    
    # Show how redline would work
    import re
    sentences = re.split(r'(?<=[.!?])\s+', sample_response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print("🔍 Redline Mode - Sentence Analysis:")
    for i, sentence in enumerate(sentences, 1):
        print(f"Line {i}: {sentence}")
    
    print("\n" + "=" * 50)
    print("💡 Example Redline Commands:")
    print("  redline 1    # Edit the first sentence")
    print("  redline 3    # Edit the third sentence")
    print("  show         # Show current content")
    print("  save         # Save changes and capture feedback")
    print("  quit         # Exit without saving")
    
    print("\n" + "=" * 50)
    print("📊 Feedback Capture:")
    print("When you save changes, the system captures:")
    print("  • Before/after sentence pairs")
    print("  • Vocabulary improvements")
    print("  • Tone adjustments")
    print("  • Phrase preferences")
    print("  • Structural changes")
    
    print("\n" + "=" * 50)
    print("🧠 Model Training:")
    print("Use the captured feedback to train the local model:")
    print("  inkmod apply-feedback --model-path enhanced_style_model.pkl")
    print("  inkmod train-enhanced --incremental  # Apply feedback to training")
    
    print("\n" + "=" * 50)
    print("🎯 Benefits:")
    print("  • Precise feedback on specific sentences")
    print("  • Captures 'this, not that' preferences")
    print("  • Improves model with real user corrections")
    print("  • Maintains style consistency")
    print("  • Enables continuous learning")

if __name__ == "__main__":
    demonstrate_redline() 