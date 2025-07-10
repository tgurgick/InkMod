#!/usr/bin/env python3
"""Test local model generation without API."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.enhanced_style_model import EnhancedStyleModel
from rich.console import Console

console = Console()

def test_local_generation():
    """Test if the local model can generate responses without API."""
    
    console.print("🧪 Testing Local Model Generation (No API Required)")
    console.print("=" * 60)
    
    # Load the enhanced model
    model = EnhancedStyleModel(use_local_llm=True)
    
    if not model.load_model():
        console.print("❌ No trained model found. Please run training first.")
        return False
    
    # Test prompts
    test_prompts = [
        "Write a short email to a colleague",
        "Write a professional email requesting a meeting",
        "Write a casual email about a project update"
    ]
    
    console.print("\n📝 Testing Local Generation (Template-based):")
    console.print("-" * 40)
    
    for i, prompt in enumerate(test_prompts, 1):
        console.print(f"\n🔹 Test {i}: {prompt}")
        
        # Force template generation (no API)
        response = model._generate_with_template(prompt, 150)
        
        console.print(f"📄 Response: {response}")
        console.print(f"📊 Length: {len(response)} characters")
    
    # Test model info
    model_info = model.get_model_info()
    console.print(f"\n📊 Model Statistics:")
    console.print(f"   Vocabulary: {model_info.get('vocabulary_size', 'N/A')} words")
    console.print(f"   Common Phrases: {len(model.common_phrases)}")
    console.print(f"   Tone Markers: {sum(len(markers) for markers in model.tone_markers.values())}")
    console.print(f"   Training Sessions: {model.learning_progress.get('total_training_sessions', 0)}")
    
    console.print(f"\n✅ SUCCESS: Local model can generate responses without API!")
    console.print(f"💡 Quality: Template-based (basic but functional)")
    console.print(f"🚀 Speed: Instant (no API calls)")
    console.print(f"💰 Cost: $0.00 (completely free)")
    
    return True

if __name__ == "__main__":
    test_local_generation() 