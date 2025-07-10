#!/usr/bin/env python3
"""
Simple Model Reader
Read and use the enhanced_style_model.pkl file
"""

import pickle
import sys

def read_model(model_path='enhanced_style_model.pkl'):
    """Read the enhanced model and show its capabilities."""
    
    print("📖 READING ENHANCED MODEL")
    print("=" * 40)
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✅ Loaded {model_path}")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None
    
    return model_data

def show_model_summary(model_data):
    """Show a summary of the model."""
    
    print("\n📊 MODEL SUMMARY")
    print("-" * 30)
    
    # Basic stats
    vocab_size = len(model_data['vocabulary'])
    sentence_patterns = len(model_data['sentence_patterns'])
    paragraph_patterns = len(model_data['paragraph_patterns'])
    common_phrases = len(model_data['common_phrases'])
    
    print(f"Vocabulary size: {vocab_size} words")
    print(f"Sentence patterns: {sentence_patterns}")
    print(f"Paragraph patterns: {paragraph_patterns}")
    print(f"Common phrases: {common_phrases}")
    
    # Tone analysis
    tone_markers = model_data['tone_markers']
    print(f"\nTone markers:")
    for tone, markers in tone_markers.items():
        print(f"  {tone}: {len(markers)} markers")
    
    # Training history
    training_history = model_data.get('training_history', [])
    print(f"\nTraining sessions: {len(training_history)}")
    
    # Performance
    performance_metrics = model_data.get('performance_metrics', [])
    if performance_metrics:
        latest_score = performance_metrics[-1].get('overall_score', 0)
        print(f"Latest performance score: {latest_score:.3f}")

def generate_simple_response(model_data, prompt):
    """Generate a simple response using the model's vocabulary."""
    
    print(f"\n🤖 GENERATING RESPONSE")
    print("-" * 30)
    print(f"Prompt: {prompt}")
    
    # Get vocabulary and common phrases
    vocabulary = model_data['vocabulary']
    common_phrases = model_data['common_phrases']
    tone_markers = model_data['tone_markers']
    
    # Simple generation using learned patterns
    response_parts = []
    
    # Start with a greeting based on tone
    professional_words = tone_markers.get('professional', [])
    if professional_words:
        response_parts.append("Dear [Recipient],")
    else:
        response_parts.append("Hello,")
    
    # Add some common phrases
    if common_phrases:
        import random
        selected_phrases = random.sample(common_phrases[:10], min(3, len(common_phrases[:10])))
        for phrase in selected_phrases:
            response_parts.append(f"{phrase.capitalize()}.")
    
    # Add some vocabulary words
    top_words = [word for word, count in vocabulary.most_common(20)]
    if top_words:
        import random
        selected_words = random.sample(top_words, min(5, len(top_words)))
        sentence = " ".join(selected_words).capitalize() + "."
        response_parts.append(sentence)
    
    # End with a closing
    if professional_words:
        response_parts.append("Best regards,")
    else:
        response_parts.append("Thanks!")
    
    response = "\n\n".join(response_parts)
    print(f"\nGenerated response:")
    print(response)
    
    return response

def main():
    """Main function to explore the model."""
    
    # Read the model
    model_data = read_model()
    if not model_data:
        return
    
    # Show summary
    show_model_summary(model_data)
    
    # Generate a response
    prompt = "Write a professional email about a meeting"
    generate_simple_response(model_data, prompt)
    
    print(f"\n✅ Model exploration complete!")

if __name__ == "__main__":
    main() 