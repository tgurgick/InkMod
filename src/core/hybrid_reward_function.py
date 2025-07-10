"""Hybrid reward function combining standard NLP metrics with LLM qualitative feedback."""

import re
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
from rich.console import Console

console = Console()

class HybridRewardFunction:
    """Hybrid reward function using standard NLP metrics + LLM qualitative feedback."""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def calculate_reward(
        self, 
        generated_text: str, 
        reference_samples: List[str],
        llm_feedback: str = None
    ) -> Dict[str, Any]:
        """Calculate hybrid reward using standard metrics + LLM feedback."""
        
        # Calculate objective metrics
        objective_scores = self._calculate_objective_metrics(generated_text, reference_samples)
        
        # Parse LLM feedback if provided
        llm_analysis = self._parse_llm_feedback(llm_feedback) if llm_feedback else {}
        
        # Combine for final reward
        combined_reward = self._combine_rewards(objective_scores, llm_analysis)
        
        return combined_reward
    
    def _calculate_objective_metrics(self, generated_text: str, reference_samples: List[str]) -> Dict[str, float]:
        """Calculate standard NLP metrics for style evaluation."""
        
        # Tokenize texts
        generated_tokens = self._tokenize_text(generated_text)
        reference_tokens_list = [self._tokenize_text(sample) for sample in reference_samples]
        
        # Calculate BLEU score for style similarity
        bleu_score = self._calculate_bleu_score(generated_tokens, reference_tokens_list)
        
        # Calculate ROUGE score for vocabulary overlap
        rouge_score = self._calculate_rouge_score(generated_tokens, reference_tokens_list)
        
        # Calculate perplexity-based consistency score
        consistency_score = self._calculate_consistency_score(generated_text, reference_samples)
        
        # Calculate length similarity
        length_score = self._calculate_length_similarity(generated_text, reference_samples)
        
        # Calculate tone similarity
        tone_score = self._calculate_tone_similarity(generated_text, reference_samples)
        
        # Weighted overall score
        overall_score = (
            0.3 * bleu_score +
            0.2 * rouge_score +
            0.2 * consistency_score +
            0.15 * length_score +
            0.15 * tone_score
        )
        
        return {
            'bleu_score': bleu_score,
            'rouge_score': rouge_score,
            'consistency_score': consistency_score,
            'length_score': length_score,
            'tone_score': tone_score,
            'overall_score': overall_score
        }
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _calculate_bleu_score(self, generated_tokens: List[str], reference_tokens_list: List[List[str]]) -> float:
        """Calculate BLEU score for style similarity."""
        try:
            # Use 4-gram BLEU with equal weights
            weights = (0.25, 0.25, 0.25, 0.25)
            bleu_score = sentence_bleu(reference_tokens_list, generated_tokens, weights=weights)
            return min(bleu_score, 1.0)  # Cap at 1.0
        except Exception as e:
            console.print(f"⚠️  BLEU calculation failed: {e}")
            return 0.5  # Default score
    
    def _calculate_rouge_score(self, generated_tokens: List[str], reference_tokens_list: List[List[str]]) -> float:
        """Calculate ROUGE score for vocabulary overlap."""
        try:
            # Extract n-grams from generated text
            generated_unigrams = set(generated_tokens)
            generated_bigrams = set(ngrams(generated_tokens, 2))
            
            # Extract n-grams from reference samples
            reference_unigrams = set()
            reference_bigrams = set()
            
            for ref_tokens in reference_tokens_list:
                reference_unigrams.update(ref_tokens)
                reference_bigrams.update(ngrams(ref_tokens, 2))
            
            # Calculate ROUGE-N scores
            if generated_unigrams:
                rouge_1 = len(generated_unigrams & reference_unigrams) / len(generated_unigrams)
            else:
                rouge_1 = 0.0
                
            if generated_bigrams:
                rouge_2 = len(generated_bigrams & reference_bigrams) / len(generated_bigrams)
            else:
                rouge_2 = 0.0
            
            # Average ROUGE scores
            rouge_score = (rouge_1 + rouge_2) / 2
            return min(rouge_score, 1.0)
            
        except Exception as e:
            console.print(f"⚠️  ROUGE calculation failed: {e}")
            return 0.5  # Default score
    
    def _calculate_consistency_score(self, generated_text: str, reference_samples: List[str]) -> float:
        """Calculate style consistency using vocabulary overlap."""
        try:
            # Extract vocabulary from generated text
            generated_vocab = set(self._tokenize_text(generated_text))
            
            # Extract vocabulary from reference samples
            reference_vocab = set()
            for sample in reference_samples:
                reference_vocab.update(self._tokenize_text(sample))
            
            # Calculate vocabulary overlap
            if generated_vocab:
                overlap_ratio = len(generated_vocab & reference_vocab) / len(generated_vocab)
                return min(overlap_ratio, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            console.print(f"⚠️  Consistency calculation failed: {e}")
            return 0.5  # Default score
    
    def _calculate_length_similarity(self, generated_text: str, reference_samples: List[str]) -> float:
        """Calculate sentence and word length similarity."""
        try:
            # Analyze generated text
            generated_sentences = re.split(r'[.!?]+', generated_text)
            generated_sentences = [s.strip() for s in generated_sentences if s.strip()]
            
            if not generated_sentences:
                return 0.5
            
            generated_avg_sentence_length = np.mean([len(s.split()) for s in generated_sentences])
            generated_avg_word_length = np.mean([len(word) for word in self._tokenize_text(generated_text)])
            
            # Analyze reference samples
            reference_sentence_lengths = []
            reference_word_lengths = []
            
            for sample in reference_samples:
                sentences = re.split(r'[.!?]+', sample)
                sentences = [s.strip() for s in sentences if s.strip()]
                if sentences:
                    reference_sentence_lengths.extend([len(s.split()) for s in sentences])
                    reference_word_lengths.extend([len(word) for word in self._tokenize_text(sample)])
            
            if not reference_sentence_lengths:
                return 0.5
            
            reference_avg_sentence_length = np.mean(reference_sentence_lengths)
            reference_avg_word_length = np.mean(reference_word_lengths)
            
            # Calculate similarity scores
            sentence_similarity = 1.0 / (1.0 + abs(generated_avg_sentence_length - reference_avg_sentence_length))
            word_similarity = 1.0 / (1.0 + abs(generated_avg_word_length - reference_avg_word_length))
            
            return (sentence_similarity + word_similarity) / 2
            
        except Exception as e:
            console.print(f"⚠️  Length similarity calculation failed: {e}")
            return 0.5  # Default score
    
    def _calculate_tone_similarity(self, generated_text: str, reference_samples: List[str]) -> float:
        """Calculate tone similarity using tone markers."""
        try:
            # Define tone markers
            tone_markers = {
                'formal': ['regarding', 'concerning', 'furthermore', 'moreover', 'therefore'],
                'casual': ['hey', 'hi', 'thanks', 'cool', 'awesome', 'great'],
                'professional': ['please', 'would you', 'could you', 'thank you', 'appreciate'],
                'emphatic': ['really', 'very', 'extremely', 'absolutely', 'definitely']
            }
            
            # Count tone markers in generated text
            generated_text_lower = generated_text.lower()
            generated_tone_counts = {}
            for tone, markers in tone_markers.items():
                generated_tone_counts[tone] = sum(generated_text_lower.count(marker) for marker in markers)
            
            # Count tone markers in reference samples
            reference_tone_counts = {}
            for tone, markers in tone_markers.items():
                reference_tone_counts[tone] = 0
                for sample in reference_samples:
                    sample_lower = sample.lower()
                    reference_tone_counts[tone] += sum(sample_lower.count(marker) for marker in markers)
            
            # Calculate tone similarity
            tone_similarities = []
            for tone in tone_markers.keys():
                gen_count = generated_tone_counts.get(tone, 0)
                ref_count = reference_tone_counts.get(tone, 0)
                
                if ref_count > 0:
                    similarity = min(gen_count / ref_count, 1.0) if ref_count > 0 else 0.0
                    tone_similarities.append(similarity)
                else:
                    # If no reference markers, check if generated has none too
                    similarity = 1.0 if gen_count == 0 else 0.5
                    tone_similarities.append(similarity)
            
            return np.mean(tone_similarities) if tone_similarities else 0.5
            
        except Exception as e:
            console.print(f"⚠️  Tone similarity calculation failed: {e}")
            return 0.5  # Default score
    
    def _parse_llm_feedback(self, llm_feedback: str) -> Dict[str, Any]:
        """Parse LLM qualitative feedback into structured format."""
        if not llm_feedback:
            return {}
        
        feedback_analysis = {
            'good_aspects': [],
            'suggestions': [],
            'tone_improvements': [],
            'structure_improvements': [],
            'vocabulary_improvements': []
        }
        
        # Simple parsing of LLM feedback
        lines = llm_feedback.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip().lower()
            
            if 'good' in line or 'positive' in line or 'well' in line:
                current_section = 'good_aspects'
            elif 'suggestion' in line or 'improve' in line or 'better' in line:
                current_section = 'suggestions'
            elif 'tone' in line:
                current_section = 'tone_improvements'
            elif 'structure' in line or 'sentence' in line or 'paragraph' in line:
                current_section = 'structure_improvements'
            elif 'vocabulary' in line or 'word' in line:
                current_section = 'vocabulary_improvements'
            elif line and current_section:
                # Extract words from the line
                words = re.findall(r'\b\w+\b', line)
                if words:
                    feedback_analysis[current_section].extend(words)
        
        return feedback_analysis
    
    def _combine_rewards(self, objective_scores: Dict[str, float], llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine objective scores with LLM analysis."""
        
        combined_reward = {
            'objective_scores': objective_scores,
            'llm_analysis': llm_analysis,
            'overall_score': objective_scores['overall_score'],
            'model_updates': self._create_model_updates(llm_analysis)
        }
        
        return combined_reward
    
    def _create_model_updates(self, llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create model update suggestions from LLM analysis."""
        
        updates = {
            'vocabulary_additions': [],
            'phrase_additions': [],
            'tone_improvements': [],
            'structure_improvements': []
        }
        
        # Extract vocabulary improvements
        if 'vocabulary_improvements' in llm_analysis:
            updates['vocabulary_additions'].extend(llm_analysis['vocabulary_improvements'])
        
        # Extract tone improvements
        if 'tone_improvements' in llm_analysis:
            updates['tone_improvements'].extend(llm_analysis['tone_improvements'])
        
        # Extract structure improvements
        if 'structure_improvements' in llm_analysis:
            updates['structure_improvements'].extend(llm_analysis['structure_improvements'])
        
        # Extract phrases from suggestions
        if 'suggestions' in llm_analysis:
            # Look for 2-3 word phrases in suggestions
            for suggestion in llm_analysis['suggestions']:
                words = suggestion.split()
                for length in range(2, min(4, len(words) + 1)):
                    for i in range(len(words) - length + 1):
                        phrase = ' '.join(words[i:i+length])
                        if len(phrase) > length * 2:  # Minimum length check
                            updates['phrase_additions'].append(phrase)
        
        return updates 