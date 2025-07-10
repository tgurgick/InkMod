#!/usr/bin/env python3
"""
Enhanced Model Explorer
Explore the contents of the enhanced_style_model.pkl file
"""

import pickle
import json
from collections import Counter
from datetime import datetime
import os

def explore_enhanced_model(model_path='enhanced_style_model.pkl'):
    """Explore the enhanced model pickle file."""
    
    print("🔍 ENHANCED MODEL EXPLORER")
    print("=" * 50)
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✅ Successfully loaded {model_path}")
        print(f"📊 File size: {os.path.getsize(model_path)} bytes")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Show overall structure
    print(f"\n📋 MODEL STRUCTURE")
    print("-" * 30)
    for key in model_data.keys():
        data_type = type(model_data[key]).__name__
        if isinstance(model_data[key], (list, dict)):
            size = len(model_data[key])
            print(f"  {key}: {data_type} ({size} items)")
        else:
            print(f"  {key}: {data_type}")
    
    # Explore vocabulary
    print(f"\n📚 VOCABULARY ANALYSIS")
    print("-" * 30)
    vocabulary = model_data['vocabulary']
    print(f"Total unique words: {len(vocabulary)}")
    print(f"Total word occurrences: {sum(vocabulary.values())}")
    
    # Top words
    top_words = vocabulary.most_common(10)
    print(f"\nTop 10 most frequent words:")
    for word, count in top_words:
        print(f"  '{word}': {count} times")
    
    # Word length analysis
    word_lengths = [len(word) for word in vocabulary.keys()]
    avg_length = sum(word_lengths) / len(word_lengths)
    print(f"\nAverage word length: {avg_length:.2f} characters")
    
    # Explore sentence patterns
    print(f"\n📝 SENTENCE PATTERNS")
    print("-" * 30)
    sentence_patterns = model_data['sentence_patterns']
    print(f"Total sentence patterns: {len(sentence_patterns)}")
    
    if sentence_patterns:
        lengths = [p['length'] for p in sentence_patterns]
        avg_length = sum(lengths) / len(lengths)
        print(f"Average sentence length: {avg_length:.1f} words")
        print(f"Min sentence length: {min(lengths)} words")
        print(f"Max sentence length: {max(lengths)} words")
    
    # Explore paragraph patterns
    print(f"\n📄 PARAGRAPH PATTERNS")
    print("-" * 30)
    paragraph_patterns = model_data['paragraph_patterns']
    print(f"Total paragraph patterns: {len(paragraph_patterns)}")
    
    if paragraph_patterns:
        lengths = [p['length'] for p in paragraph_patterns]
        avg_length = sum(lengths) / len(lengths)
        print(f"Average paragraph length: {avg_length:.1f} words")
    
    # Explore tone markers
    print(f"\n🎭 TONE MARKERS")
    print("-" * 30)
    tone_markers = model_data['tone_markers']
    for tone, markers in tone_markers.items():
        print(f"  {tone.capitalize()}: {len(markers)} markers")
        if markers:
            print(f"    Sample: {markers[:3]}")
    
    # Explore common phrases
    print(f"\n💬 COMMON PHRASES")
    print("-" * 30)
    common_phrases = model_data['common_phrases']
    print(f"Total common phrases: {len(common_phrases)}")
    if common_phrases:
        print("Sample phrases:")
        for phrase in common_phrases[:5]:
            print(f"  '{phrase}'")
    
    # Explore training history
    print(f"\n📈 TRAINING HISTORY")
    print("-" * 30)
    training_history = model_data.get('training_history', [])
    print(f"Total training sessions: {len(training_history)}")
    
    for i, session in enumerate(training_history, 1):
        print(f"\nSession {i}:")
        print(f"  Timestamp: {session.get('timestamp', 'Unknown')}")
        print(f"  Samples processed: {session.get('samples_count', 0)}")
        print(f"  Total characters: {session.get('total_characters', 0)}")
        print(f"  Incremental: {session.get('incremental', False)}")
        print(f"  Previous vocab size: {session.get('previous_vocabulary_size', 0)}")
    
    # Explore performance metrics
    print(f"\n📊 PERFORMANCE METRICS")
    print("-" * 30)
    performance_metrics = model_data.get('performance_metrics', [])
    print(f"Total performance records: {len(performance_metrics)}")
    
    if performance_metrics:
        print("\nPerformance progression:")
        for i, metric in enumerate(performance_metrics, 1):
            print(f"  Iteration {i}:")
            print(f"    Overall score: {metric.get('overall_score', 0):.3f}")
            print(f"    Tone score: {metric.get('tone_score', 0):.3f}")
            print(f"    Structure score: {metric.get('structure_score', 0):.3f}")
            print(f"    Cost: ${metric.get('total_cost', 0):.4f}")
    
    # Explore learning progress
    print(f"\n🧠 LEARNING PROGRESS")
    print("-" * 30)
    learning_progress = model_data.get('learning_progress', {})
    
    print(f"Total training sessions: {learning_progress.get('total_training_sessions', 0)}")
    print(f"Total samples processed: {learning_progress.get('total_samples_processed', 0)}")
    
    best_scores = learning_progress.get('best_scores', {})
    print(f"\nBest scores:")
    print(f"  Style: {best_scores.get('style', 0):.3f}")
    print(f"  Tone: {best_scores.get('tone', 0):.3f}")
    print(f"  Structure: {best_scores.get('structure', 0):.3f}")
    
    convergence_indicators = learning_progress.get('convergence_indicators', [])
    print(f"\nConvergence indicators: {len(convergence_indicators)}")
    if convergence_indicators:
        latest = convergence_indicators[-1]
        print(f"  Latest convergence: {latest.get('converged', False)}")
        print(f"  Latest variance: {latest.get('variance', 0):.6f}")
        print(f"  Latest timestamp: {latest.get('timestamp', 'Unknown')}")
    
    # Explore style characteristics
    print(f"\n🎨 STYLE CHARACTERISTICS")
    print("-" * 30)
    style_characteristics = model_data.get('style_characteristics', {})
    for key, value in style_characteristics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Explore LLM config
    print(f"\n🤖 LLM CONFIGURATION")
    print("-" * 30)
    llm_config = model_data.get('llm_config', {})
    for key, value in llm_config.items():
        print(f"  {key}: {value}")
    
    # Show sample training data
    print(f"\n📖 SAMPLE TRAINING DATA")
    print("-" * 30)
    training_samples = model_data.get('training_samples', [])
    print(f"Total training samples: {len(training_samples)}")
    
    if training_samples:
        print("\nSample content (first 200 chars):")
        for i, sample in enumerate(training_samples[:2], 1):
            preview = sample[:200] + "..." if len(sample) > 200 else sample
            print(f"  Sample {i}: {preview}")
    
    print(f"\n✅ Model exploration complete!")

if __name__ == "__main__":
    explore_enhanced_model() 