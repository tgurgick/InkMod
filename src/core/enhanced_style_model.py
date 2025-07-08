"""Enhanced style model with lightweight LLM integration."""

import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from collections import Counter
import re
from rich.console import Console
from rich.progress import Progress
import requests
import os

from core.openai_client import OpenAIClient
from core.llm_backends import LLMBackendManager, create_backend_manager
from utils.text_utils import combine_style_samples

console = Console()

class EnhancedStyleModel:
    """Enhanced local model with lightweight LLM integration."""
    
    def __init__(self, model_path: Optional[str] = None, use_local_llm: bool = True, backend_name: str = None):
        self.model_path = model_path or "enhanced_style_model.pkl"
        self.use_local_llm = use_local_llm
        self.backend_name = backend_name
        
        # Core style data
        self.vocabulary = Counter()
        self.sentence_patterns = []
        self.paragraph_patterns = []
        self.style_characteristics = {}
        self.training_samples = []
        
        # Enhanced features
        self.style_prompt_template = ""
        self.common_phrases = []
        self.tone_markers = {}
        self.structure_patterns = []
        
        # LLM configuration
        self.llm_config = {
            'model': 'gpt-3.5-turbo',  # Lightweight API model
            'max_tokens': 150,
            'temperature': 0.7
        }
        
        # Initialize LLM backend manager
        self.backend_manager = create_backend_manager()
        if backend_name:
            self.backend_manager.set_backend(backend_name)
        
    def train(self, samples: Dict[str, str]) -> Dict[str, any]:
        """Enhanced training with better pattern extraction."""
        
        console.print("🧠 Training enhanced style model...")
        
        # Security: Validate input samples
        if not samples:
            raise ValueError("No samples provided for training")
        
        # Security: Validate sample content
        for filename, content in samples.items():
            if not isinstance(content, str):
                raise ValueError(f"Invalid content type for {filename}")
            if len(content) > 1000000:  # 1MB limit per sample
                raise ValueError(f"Sample {filename} too large")
        
        # Extract basic patterns (like before)
        for filename, content in samples.items():
            self.training_samples.append(content)
            
            # Extract vocabulary
            words = re.findall(r'\b\w+\b', content.lower())
            self.vocabulary.update(words)
            
            # Extract sentence patterns
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) > 3:
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
        
        # Enhanced pattern extraction
        self._extract_enhanced_patterns(samples)
        
        # Calculate style characteristics
        self.style_characteristics = self._calculate_style_characteristics()
        
        # Create style prompt template
        self.style_prompt_template = self._create_style_prompt_template()
        
        # Save the model
        self.save_model()
        
        # Safe tone markers count
        tone_markers_count = 0
        if isinstance(self.tone_markers, dict):
            tone_markers_count = sum(len(markers) for markers in self.tone_markers.values())
        
        return {
            'vocabulary_size': len(self.vocabulary),
            'sentence_patterns': len(self.sentence_patterns),
            'paragraph_patterns': len(self.paragraph_patterns),
            'style_characteristics': self.style_characteristics,
            'common_phrases': len(self.common_phrases),
            'tone_markers': tone_markers_count
        }
    
    def _extract_enhanced_patterns(self, samples: Dict[str, str]) -> None:
        """Extract more sophisticated patterns."""
        
        # Extract common phrases (2-4 word combinations)
        all_text = ' '.join(samples.values()).lower()
        words = all_text.split()
        
        # Find common 2-4 word phrases
        for length in range(2, 5):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i+length])
                if len(phrase) > length * 2:  # Minimum meaningful phrase
                    self.common_phrases.append(phrase)
        
        # Count phrase frequency
        phrase_counts = Counter(self.common_phrases)
        self.common_phrases = [phrase for phrase, count in phrase_counts.most_common(50)]
        
        # Extract tone markers
        self.tone_markers = {
            'formal': self._extract_formal_markers(samples),
            'casual': self._extract_casual_markers(samples),
            'professional': self._extract_professional_markers(samples),
            'emphatic': self._extract_emphatic_markers(samples)
        }
        
        # Extract structure patterns
        self.structure_patterns = self._extract_structure_patterns(samples)
    
    def _extract_formal_markers(self, samples: Dict[str, str]) -> List[str]:
        """Extract formal language markers."""
        formal_patterns = [
            r'\b(regarding|concerning|furthermore|moreover|nevertheless)\b',
            r'\b(please|would you|could you|thank you)\b',
            r'\b(accordingly|subsequently|consequently)\b'
        ]
        
        markers = []
        all_text = ' '.join(samples.values()).lower()
        
        for pattern in formal_patterns:
            matches = re.findall(pattern, all_text)
            markers.extend(matches)
        
        return list(set(markers))
    
    def _extract_casual_markers(self, samples: Dict[str, str]) -> List[str]:
        """Extract casual language markers."""
        casual_patterns = [
            r'\b(hey|hi|thanks|cool|awesome|great|nice)\b',
            r'\b(yeah|yep|nope|sure|okay)\b',
            r'\b(by the way|btw|asap)\b'
        ]
        
        markers = []
        all_text = ' '.join(samples.values()).lower()
        
        for pattern in casual_patterns:
            matches = re.findall(pattern, all_text)
            markers.extend(matches)
        
        return list(set(markers))
    
    def _extract_professional_markers(self, samples: Dict[str, str]) -> List[str]:
        """Extract professional language markers."""
        professional_patterns = [
            r'\b(please|would you|could you|thank you)\b',
            r'\b(regarding|concerning|with respect to)\b',
            r'\b(accordingly|subsequently|consequently)\b',
            r'\b(best regards|sincerely|yours truly)\b'
        ]
        
        markers = []
        all_text = ' '.join(samples.values()).lower()
        
        for pattern in professional_patterns:
            matches = re.findall(pattern, all_text)
            markers.extend(matches)
        
        return list(set(markers))
    
    def _extract_emphatic_markers(self, samples: Dict[str, str]) -> List[str]:
        """Extract emphatic language markers."""
        emphatic_patterns = [
            r'\b(really|very|extremely|absolutely|definitely)\b',
            r'\b(crucial|essential|important|critical)\b',
            r'\b(clearly|obviously|evidently)\b'
        ]
        
        markers = []
        all_text = ' '.join(samples.values()).lower()
        
        for pattern in emphatic_patterns:
            matches = re.findall(pattern, all_text)
            markers.extend(matches)
        
        return list(set(markers))
    
    def _extract_structure_patterns(self, samples: Dict[str, str]) -> List[Dict[str, any]]:
        """Extract structural patterns."""
        patterns = []
        
        for content in samples.values():
            # Extract opening patterns
            first_sentence = content.split('.')[0] if content.split('.') else ""
            if first_sentence:
                patterns.append({
                    'type': 'opening',
                    'pattern': first_sentence.strip()[:50],
                    'length': len(first_sentence.split())
                })
            
            # Extract closing patterns
            sentences = content.split('.')
            if len(sentences) > 1:
                last_sentence = sentences[-2] if sentences[-1].strip() == "" else sentences[-1]
                if last_sentence.strip():
                    patterns.append({
                        'type': 'closing',
                        'pattern': last_sentence.strip()[:50],
                        'length': len(last_sentence.split())
                    })
        
        return patterns
    
    def _create_style_prompt_template(self) -> str:
        """Create a style prompt template for LLM generation."""
        
        # Analyze dominant tone
        formal_score = len(self.tone_markers['formal'])
        casual_score = len(self.tone_markers['casual'])
        professional_score = len(self.tone_markers['professional'])
        
        if professional_score > formal_score and professional_score > casual_score:
            tone = "professional"
        elif formal_score > casual_score:
            tone = "formal"
        else:
            tone = "casual"
        
        # Create template
        template = f"""You are writing in a {tone} style. Use these characteristics:

Style Characteristics:
- Average sentence length: {self.style_characteristics.get('avg_sentence_length', 15):.1f} words
- Average word length: {self.style_characteristics.get('avg_word_length', 5):.1f} characters
- Tone: {tone}
- Common phrases: {', '.join(self.common_phrases[:10])}

Writing Style:
- Use these tone markers: {', '.join(self.tone_markers[tone][:5])}
- Maintain the average sentence length
- Use vocabulary similar to the training samples
- Follow the structural patterns

Write a response to: {{prompt}}"""
        
        return template
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using enhanced model with lightweight LLM."""
        
        if self.use_local_llm:
            return self._generate_with_lightweight_llm(prompt, max_length)
        else:
            return self._generate_with_template(prompt, max_length)
    
    def _generate_with_lightweight_llm(self, prompt: str, max_length: int) -> str:
        """Generate using lightweight LLM (local backend or API)."""
        
        try:
            # Create enhanced prompt
            enhanced_prompt = self.style_prompt_template.format(prompt=prompt)
            
            if self.use_local_llm and hasattr(self, 'backend_manager'):
                # Try local backend first
                backend = self.backend_manager.get_current_backend()
                if backend and backend.is_loaded:
                    return self.backend_manager.generate(enhanced_prompt, max_length, self.llm_config['temperature'])
            
            # Fallback to OpenAI API
            from core.openai_client import OpenAIClient
            
            # Initialize with lightweight model
            openai_client = OpenAIClient(
                api_key=os.getenv('OPENAI_API_KEY'),
                model=self.llm_config['model']
            )
            
            # Generate response
            result = openai_client.generate_response(
                style_samples={},  # We're using the style template instead
                user_input=enhanced_prompt,
                temperature=self.llm_config['temperature'],
                max_tokens=self.llm_config['max_tokens']
            )
            
            return result['response'][:max_length]
            
        except Exception as e:
            console.print(f"⚠️  LLM generation failed: {e}, falling back to template")
            return self._generate_with_template(prompt, max_length)
    
    def set_backend(self, backend_name: str) -> bool:
        """Set the LLM backend to use."""
        if hasattr(self, 'backend_manager'):
            return self.backend_manager.set_backend(backend_name)
        return False
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        if hasattr(self, 'backend_manager'):
            return self.backend_manager.list_backends()
        return []
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        if hasattr(self, 'backend_manager'):
            return self.backend_manager.get_backend_info()
        return {'error': 'No backend manager available'}
    
    def _generate_with_template(self, prompt: str, max_length: int) -> str:
        """Fallback template-based generation."""
        
        # Use learned patterns for generation
        avg_sentence_length = np.mean([p['length'] for p in self.sentence_patterns])
        avg_paragraph_length = np.mean([p['length'] for p in self.paragraph_patterns])
        
        # Generate response using enhanced patterns
        response_parts = []
        
        # Choose opening based on dominant tone
        dominant_tone = self._get_dominant_tone()
        if dominant_tone == 'casual':
            response_parts.append("Hi there,")
        elif dominant_tone == 'professional':
            response_parts.append("Hello,")
        else:
            response_parts.append("Dear,")
        
        # Generate main content using common phrases
        sentences_needed = int(avg_paragraph_length / avg_sentence_length)
        for i in range(sentences_needed):
            sentence = self._generate_enhanced_sentence(avg_sentence_length)
            response_parts.append(sentence)
        
        # Choose closing based on tone
        if dominant_tone == 'professional':
            response_parts.append("Best regards,")
        elif dominant_tone == 'casual':
            response_parts.append("Thanks!")
        else:
            response_parts.append("Sincerely,")
        
        return "\n\n".join(response_parts)[:max_length]
    
    def _get_dominant_tone(self) -> str:
        """Determine the dominant tone from training data."""
        formal_score = len(self.tone_markers['formal'])
        casual_score = len(self.tone_markers['casual'])
        professional_score = len(self.tone_markers['professional'])
        
        if professional_score > formal_score and professional_score > casual_score:
            return 'professional'
        elif formal_score > casual_score:
            return 'formal'
        else:
            return 'casual'
    
    def _generate_enhanced_sentence(self, target_length: float) -> str:
        """Generate sentence using enhanced patterns."""
        
        # Use common phrases when possible
        if self.common_phrases and np.random.random() < 0.3:
            phrase = np.random.choice(self.common_phrases)
            return phrase.capitalize() + "."
        
        # Use tone-appropriate words
        dominant_tone = self._get_dominant_tone()
        tone_words = self.tone_markers.get(dominant_tone, [])
        
        if tone_words:
            # Mix tone words with common vocabulary
            common_words = [word for word, count in self.vocabulary.most_common(30)]
            all_words = tone_words + common_words
            words = np.random.choice(all_words, size=int(target_length), replace=True)
        else:
            # Fallback to common vocabulary
            common_words = [word for word, count in self.vocabulary.most_common(50)]
            words = np.random.choice(common_words, size=int(target_length), replace=True)
        
        sentence = " ".join(words).capitalize() + "."
        return sentence
    
    def update_from_feedback(self, feedback: List[Dict[str, any]]) -> None:
        """Enhanced model update based on feedback."""
        # Extract suggestions and incorporate them
        all_suggestions = []
        for fb in feedback:
            all_suggestions.extend(fb.get('suggestions', []))
        
        # Update vocabulary with suggested words
        for suggestion in all_suggestions:
            words = suggestion.lower().split()
            self.vocabulary.update(words)
        
        # Update common phrases if new patterns emerge
        for suggestion in all_suggestions:
            # Look for 2-4 word phrases in suggestions
            words = suggestion.lower().split()
            for length in range(2, min(5, len(words) + 1)):
                for i in range(len(words) - length + 1):
                    phrase = ' '.join(words[i:i+length])
                    if len(phrase) > length * 2:
                        self.common_phrases.append(phrase)
        
        # Update tone markers based on feedback
        for suggestion in all_suggestions:
            suggestion_lower = suggestion.lower()
            
            # Check for formal markers
            if any(marker in suggestion_lower for marker in ['formal', 'professional', 'regarding']):
                self.tone_markers['formal'].extend(suggestion_lower.split())
            
            # Check for casual markers
            if any(marker in suggestion_lower for marker in ['casual', 'friendly', 'informal']):
                self.tone_markers['casual'].extend(suggestion_lower.split())
        
        # Remove duplicates
        for tone in self.tone_markers:
            self.tone_markers[tone] = list(set(self.tone_markers[tone]))
        
        # Update style prompt template
        self.style_prompt_template = self._create_style_prompt_template()
        
        # Save updated model
        self.save_model()
    
    # ... existing helper methods from LocalStyleModel ...
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
    
    def save_model(self) -> None:
        """Save the enhanced model to disk."""
        model_data = {
            'vocabulary': self.vocabulary,
            'sentence_patterns': self.sentence_patterns,
            'paragraph_patterns': self.paragraph_patterns,
            'style_characteristics': self.style_characteristics,
            'training_samples': self.training_samples,
            'style_prompt_template': self.style_prompt_template,
            'common_phrases': self.common_phrases,
            'tone_markers': self.tone_markers,
            'structure_patterns': self.structure_patterns,
            'llm_config': self.llm_config
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        console.print(f"💾 Enhanced model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load an enhanced model from disk."""
        try:
            # Security: Validate file size before loading
            file_size = os.path.getsize(self.model_path)
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                console.print(f"❌ Model file too large ({file_size} bytes), potential security risk")
                return False
            
            with open(self.model_path, 'rb') as f:
                # Security: Use restricted unpickler or validate data structure
                model_data = pickle.load(f)
            
            # Security: Validate expected data structure
            required_keys = ['vocabulary', 'sentence_patterns', 'paragraph_patterns', 
                           'style_characteristics', 'training_samples']
            if not all(key in model_data for key in required_keys):
                console.print("❌ Invalid model file structure")
                return False
            
            self.vocabulary = model_data['vocabulary']
            self.sentence_patterns = model_data['sentence_patterns']
            self.paragraph_patterns = model_data['paragraph_patterns']
            self.style_characteristics = model_data['style_characteristics']
            self.training_samples = model_data['training_samples']
            self.style_prompt_template = model_data.get('style_prompt_template', '')
            self.common_phrases = model_data.get('common_phrases', [])
            self.tone_markers = model_data.get('tone_markers', {})
            self.structure_patterns = model_data.get('structure_patterns', [])
            self.llm_config = model_data.get('llm_config', self.llm_config)
            
            console.print(f"📂 Enhanced model loaded from {self.model_path}")
            return True
        except FileNotFoundError:
            console.print(f"⚠️  No saved model found at {self.model_path}")
            return False
        except Exception as e:
            console.print(f"❌ Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the enhanced model."""
        return {
            'vocabulary_size': len(self.vocabulary),
            'sentence_patterns': len(self.sentence_patterns),
            'paragraph_patterns': len(self.paragraph_patterns),
            'style_characteristics': self.style_characteristics,
            'training_samples': len(self.training_samples),
            'common_phrases': len(self.common_phrases),
            'tone_markers': {tone: len(markers) for tone, markers in self.tone_markers.items()},
            'structure_patterns': len(self.structure_patterns),
            'llm_config': self.llm_config
        } 