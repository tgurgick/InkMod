"""Local style model for reinforcement learning with OpenAI comparison."""

import json
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import Counter
import re
from rich.console import Console
from rich.progress import Progress

from core.openai_client import OpenAIClient
from utils.text_utils import extract_style_summary

console = Console()

class LocalStyleModel:
    """Local model that learns writing style patterns."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "local_style_model.pkl"
        self.vocabulary = Counter()
        self.sentence_patterns = []
        self.paragraph_patterns = []
        self.style_characteristics = {}
        self.training_samples = []
        
    def train(self, samples: Dict[str, str]) -> Dict[str, any]:
        """Train the local model on writing samples."""
        
        console.print("🧠 Training local style model...")
        
        # Extract vocabulary and patterns
        for filename, content in samples.items():
            self.training_samples.append(content)
            
            # Extract vocabulary
            words = re.findall(r'\b\w+\b', content.lower())
            self.vocabulary.update(words)
            
            # Extract sentence patterns
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) > 3:  # Only meaningful sentences
                    self.sentence_patterns.append({
                        'length': len(sentence.split()),
                        'structure': self._analyze_sentence_structure(sentence),
                        'tone_indicators': self._extract_tone_indicators(sentence)
                    })
            
            # Extract paragraph patterns
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    self.paragraph_patterns.append({
                        'length': len(para.split()),
                        'sentences': len(re.split(r'[.!?]+', para)),
                        'structure': self._analyze_paragraph_structure(para)
                    })
        
        # Calculate style characteristics
        self.style_characteristics = self._calculate_style_characteristics()
        
        # Save the model
        self.save_model()
        
        return {
            'vocabulary_size': len(self.vocabulary),
            'sentence_patterns': len(self.sentence_patterns),
            'paragraph_patterns': len(self.paragraph_patterns),
            'style_characteristics': self.style_characteristics
        }
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response using the local model."""
        
        # Simple template-based generation
        # In a real implementation, this could be a more sophisticated model
        
        # Extract key words from prompt
        prompt_words = re.findall(r'\b\w+\b', prompt.lower())
        
        # Find most common patterns
        avg_sentence_length = np.mean([p['length'] for p in self.sentence_patterns])
        avg_paragraph_length = np.mean([p['length'] for p in self.paragraph_patterns])
        
        # Generate response using learned patterns
        response = self._generate_from_patterns(prompt, avg_sentence_length, avg_paragraph_length)
        
        return response[:max_length]
    
    def _analyze_sentence_structure(self, sentence: str) -> Dict[str, any]:
        """Analyze the structure of a sentence."""
        words = sentence.split()
        return {
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]),
            'has_contractions': any("'" in word for word in words),
            'has_numbers': any(re.search(r'\d', word) for word in words),
            'punctuation': len(re.findall(r'[,.!?;:]', sentence))
        }
    
    def _extract_tone_indicators(self, sentence: str) -> Dict[str, int]:
        """Extract tone indicators from a sentence."""
        sentence_lower = sentence.lower()
        
        # Simple tone indicators (could be expanded)
        indicators = {
            'formal': len(re.findall(r'\b(regarding|concerning|furthermore|moreover)\b', sentence_lower)),
            'casual': len(re.findall(r'\b(hey|hi|thanks|cool|awesome)\b', sentence_lower)),
            'professional': len(re.findall(r'\b(please|would you|could you|thank you)\b', sentence_lower)),
            'emphatic': len(re.findall(r'\b(really|very|extremely|absolutely)\b', sentence_lower))
        }
        
        return indicators
    
    def _analyze_paragraph_structure(self, paragraph: str) -> Dict[str, any]:
        """Analyze paragraph structure."""
        sentences = re.split(r'[.!?]+', paragraph)
        return {
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()]),
            'has_transitions': len(re.findall(r'\b(however|therefore|meanwhile|furthermore)\b', paragraph.lower()))
        }
    
    def _calculate_style_characteristics(self) -> Dict[str, any]:
        """Calculate overall style characteristics."""
        if not self.sentence_patterns:
            return {}
        
        return {
            'avg_sentence_length': np.mean([p['length'] for p in self.sentence_patterns]),
            'avg_word_length': np.mean([len(word) for word in self.vocabulary.keys()]),
            'vocabulary_diversity': len(self.vocabulary) / sum(self.vocabulary.values()),
            'formal_tone': np.mean([p['tone_indicators']['formal'] for p in self.sentence_patterns]),
            'casual_tone': np.mean([p['tone_indicators']['casual'] for p in self.sentence_patterns]),
            'professional_tone': np.mean([p['tone_indicators']['professional'] for p in self.sentence_patterns])
        }
    
    def _generate_from_patterns(self, prompt: str, avg_sentence_length: float, avg_paragraph_length: float) -> str:
        """Generate response using learned patterns."""
        
        # Simple template-based generation
        # This is a basic implementation - could be enhanced with more sophisticated NLP
        
        # Extract common words from training samples
        common_words = [word for word, count in self.vocabulary.most_common(50)]
        
        # Generate a simple response
        response_parts = []
        
        # Opening
        if self.style_characteristics.get('casual_tone', 0) > 0.1:
            response_parts.append("Hi there,")
        else:
            response_parts.append("Hello,")
        
        # Main content
        sentences_needed = int(avg_paragraph_length / avg_sentence_length)
        for i in range(sentences_needed):
            # Simple sentence generation using common words
            sentence = self._generate_sentence(common_words, avg_sentence_length)
            response_parts.append(sentence)
        
        # Closing
        if self.style_characteristics.get('professional_tone', 0) > 0.1:
            response_parts.append("Best regards,")
        else:
            response_parts.append("Thanks!")
        
        return "\n\n".join(response_parts)
    
    def _generate_sentence(self, common_words: List[str], target_length: float) -> str:
        """Generate a single sentence."""
        # Simple sentence generation
        words = np.random.choice(common_words, size=int(target_length), replace=True)
        sentence = " ".join(words).capitalize() + "."
        return sentence
    
    def save_model(self) -> None:
        """Save the trained model to disk."""
        model_data = {
            'vocabulary': self.vocabulary,
            'sentence_patterns': self.sentence_patterns,
            'paragraph_patterns': self.paragraph_patterns,
            'style_characteristics': self.style_characteristics,
            'training_samples': self.training_samples
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        console.print(f"💾 Local model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load a trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vocabulary = model_data['vocabulary']
            self.sentence_patterns = model_data['sentence_patterns']
            self.paragraph_patterns = model_data['paragraph_patterns']
            self.style_characteristics = model_data['style_characteristics']
            self.training_samples = model_data['training_samples']
            
            console.print(f"📂 Local model loaded from {self.model_path}")
            return True
        except FileNotFoundError:
            console.print(f"⚠️  No saved model found at {self.model_path}")
            return False
        except Exception as e:
            console.print(f"❌ Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the trained model."""
        return {
            'vocabulary_size': len(self.vocabulary),
            'sentence_patterns': len(self.sentence_patterns),
            'paragraph_patterns': len(self.paragraph_patterns),
            'style_characteristics': self.style_characteristics,
            'training_samples': len(self.training_samples)
        } 