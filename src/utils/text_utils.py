"""Text processing utilities for InkMod."""

import re
from typing import Dict, List, Tuple
from collections import Counter

def analyze_text_style(text: str) -> Dict[str, any]:
    """Analyze basic text style characteristics."""
    if not text:
        return {}
    
    # Basic statistics
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Count punctuation
    punctuation_counts = Counter(char for char in text if char in '.,!?;:')
    
    # Analyze vocabulary
    word_counts = Counter(word.lower() for word in words)
    unique_words = len(word_counts)
    vocabulary_diversity = unique_words / len(words) if words else 0
    
    # Detect common patterns
    has_contractions = any("'" in word for word in words)
    has_numbers = any(char.isdigit() for char in text)
    has_quotes = text.count('"') > 0 or text.count("'") > 0
    
    return {
        'total_words': len(words),
        'total_sentences': len(sentences),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'avg_word_length': round(avg_word_length, 2),
        'vocabulary_diversity': round(vocabulary_diversity, 3),
        'punctuation_counts': dict(punctuation_counts),
        'has_contractions': has_contractions,
        'has_numbers': has_numbers,
        'has_quotes': has_quotes,
        'most_common_words': word_counts.most_common(10)
    }

def combine_style_samples(samples: Dict[str, str]) -> str:
    """Combine multiple style samples into a single context string."""
    combined = []
    
    for filename, content in samples.items():
        # Add a header to identify the source
        combined.append(f"=== Sample from {filename} ===")
        combined.append(content)
        combined.append("\n")
    
    return "\n".join(combined)

def truncate_text(text: str, max_length: int = 4000) -> str:
    """Truncate text to fit within token limits."""
    if len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundaries
    sentences = re.split(r'([.!?]+)', text)
    truncated = ""
    
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            current_sentence = sentences[i] + sentences[i + 1]
        else:
            current_sentence = sentences[i]
        
        if len(truncated + current_sentence) > max_length:
            break
        truncated += current_sentence
    
    if not truncated:
        # Fallback to simple truncation
        truncated = text[:max_length]
    
    return truncated.strip()

def extract_style_summary(samples: Dict[str, str]) -> str:
    """Extract a concise summary of writing style characteristics."""
    if not samples:
        return ""
    
    # Analyze all samples
    all_analyses = []
    for content in samples.values():
        analysis = analyze_text_style(content)
        if analysis:
            all_analyses.append(analysis)
    
    if not all_analyses:
        return ""
    
    # Calculate averages
    avg_sentence_length = sum(a['avg_sentence_length'] for a in all_analyses) / len(all_analyses)
    avg_word_length = sum(a['avg_word_length'] for a in all_analyses) / len(all_analyses)
    avg_vocabulary_diversity = sum(a['vocabulary_diversity'] for a in all_analyses) / len(all_analyses)
    
    # Count common patterns
    has_contractions = any(a['has_contractions'] for a in all_analyses)
    has_numbers = any(a['has_numbers'] for a in all_analyses)
    has_quotes = any(a['has_quotes'] for a in all_analyses)
    
    # Build summary
    summary_parts = [
        f"Writing style analysis:",
        f"- Average sentence length: {avg_sentence_length:.1f} words",
        f"- Average word length: {avg_word_length:.1f} characters",
        f"- Vocabulary diversity: {avg_vocabulary_diversity:.3f}",
    ]
    
    if has_contractions:
        summary_parts.append("- Uses contractions")
    if has_numbers:
        summary_parts.append("- Includes numbers/data")
    if has_quotes:
        summary_parts.append("- Uses quotes/dialogue")
    
    return "\n".join(summary_parts) 