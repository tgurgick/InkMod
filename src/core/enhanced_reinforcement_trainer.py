"""Enhanced reinforcement learning system with lightweight LLM integration."""

import json
import pickle
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.openai_client import OpenAIClient
from core.enhanced_style_model import EnhancedStyleModel
from utils.text_utils import combine_style_samples

console = Console()

class EnhancedReinforcementTrainer:
    """Enhanced reinforcement learning system using lightweight LLMs."""
    
    def __init__(self, openai_client: OpenAIClient, model_path: str = "enhanced_style_model.pkl", backend_name: str = None):
        self.openai_client = openai_client
        self.local_model = EnhancedStyleModel(model_path, use_local_llm=True, backend_name=backend_name)
        self.training_history = []
        self.performance_metrics = []
        self.backend_name = backend_name
        
    def train_with_reinforcement(
        self, 
        samples: Dict[str, str], 
        test_prompts: List[str],
        iterations: int = 5,
        incremental: bool = True
    ) -> Dict[str, any]:
        """Train enhanced local model using OpenAI as teacher with continuous learning."""
        
        console.print("🎯 Starting enhanced reinforcement learning training...")
        
        # Check learning progress
        learning_summary = self.local_model.get_learning_summary()
        if learning_summary['total_sessions'] > 0:
            console.print(f"📈 Previous training sessions: {learning_summary['total_sessions']}")
            console.print(f"📊 Best style score: {learning_summary['best_scores']['style']:.3f}")
            console.print(f"📊 Best tone score: {learning_summary['best_scores']['tone']:.3f}")
            console.print(f"📊 Best structure score: {learning_summary['best_scores']['structure']:.3f}")
        
        # Initial training
        console.print("\n📚 Phase 1: Enhanced local model training")
        initial_stats = self.local_model.train(samples, incremental=incremental)
        console.print(f"✅ Enhanced training complete: {initial_stats['vocabulary_size']} vocabulary items")
        console.print(f"📊 Extracted {initial_stats['common_phrases']} common phrases")
        # Safe tone markers count
        tone_markers_count = 0
        if isinstance(initial_stats['tone_markers'], dict):
            tone_markers_count = sum(len(markers) for markers in initial_stats['tone_markers'].values())
        console.print(f"🎭 Found {tone_markers_count} tone markers")
        
        # Reinforcement learning loop
        for iteration in range(iterations):
            console.print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            
            # Generate responses with enhanced local model
            local_responses = []
            for prompt in test_prompts:
                response = self.local_model.generate_response(prompt)
                local_responses.append({
                    'prompt': prompt,
                    'response': response
                })
            
            # Get OpenAI teacher feedback
            teacher_feedback = self._get_enhanced_teacher_feedback(local_responses, samples)
            
            # Analyze performance
            performance = self._analyze_performance(local_responses, teacher_feedback)
            
            # Add performance metric to continuous learning
            self.local_model.add_performance_metric(performance)
            
            # Update enhanced model based on feedback
            self._update_enhanced_model_from_feedback(teacher_feedback)
            
            console.print(f"📊 Iteration {iteration + 1} performance: {performance['overall_score']:.3f}")
            
            # Check if we should continue training
            if iteration >= 2:  # Need at least 3 iterations to check convergence
                continue_decision = self.local_model.should_continue_training()
                if not continue_decision['continue']:
                    console.print(f"🛑 Training stopped: {continue_decision['reason']}")
                    if 'confidence' in continue_decision:
                        console.print(f"📊 Confidence: {continue_decision['confidence']}")
                    break
        
        # Final evaluation
        final_evaluation = self._evaluate_final_performance(test_prompts, samples)
        
        # Get updated learning summary
        final_learning_summary = self.local_model.get_learning_summary()
        
        return {
            'initial_stats': initial_stats,
            'performance_metrics': self.local_model.performance_metrics,
            'final_evaluation': final_evaluation,
            'training_history': self.local_model.training_history,
            'learning_summary': final_learning_summary,
            'backend_used': self.backend_name,
            'convergence_analysis': self.local_model._get_convergence_status(),
            'performance_trend': self.local_model._get_performance_trend()
        }
    
    def set_backend(self, backend_name: str) -> bool:
        """Set the LLM backend for the local model."""
        if hasattr(self.local_model, 'set_backend'):
            return self.local_model.set_backend(backend_name)
        return False
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        if hasattr(self.local_model, 'get_available_backends'):
            return self.local_model.get_available_backends()
        return []
    
    def get_backend_info(self) -> Dict[str, any]:
        """Get information about the current backend."""
        if hasattr(self.local_model, 'get_backend_info'):
            return self.local_model.get_backend_info()
        return {'error': 'No backend information available'}
    
    def _get_enhanced_teacher_feedback(
        self, 
        local_responses: List[Dict[str, str]], 
        samples: Dict[str, str]
    ) -> List[Dict[str, any]]:
        """Get enhanced feedback from OpenAI teacher."""
        
        feedback = []
        
        for response_data in local_responses:
            prompt = response_data['prompt']
            local_response = response_data['response']
            
            # Create enhanced feedback prompt
            feedback_prompt = self._create_enhanced_feedback_prompt(prompt, local_response, samples)
            
            try:
                # Get OpenAI's analysis
                result = self.openai_client.generate_response(
                    style_samples={},
                    user_input=feedback_prompt,
                    temperature=0.3,
                    max_tokens=400
                )
                
                # Parse enhanced feedback
                parsed_feedback = self._parse_enhanced_teacher_feedback(result['response'])
                parsed_feedback['prompt'] = prompt
                parsed_feedback['local_response'] = local_response
                parsed_feedback['openai_cost'] = result['cost']
                
                feedback.append(parsed_feedback)
                
            except Exception as e:
                console.print(f"⚠️  Failed to get teacher feedback: {e}")
                # Default feedback
                feedback.append({
                    'prompt': prompt,
                    'local_response': local_response,
                    'style_score': 0.5,
                    'suggestions': ['No feedback available'],
                    'tone_analysis': {},
                    'structure_analysis': {},
                    'openai_cost': 0.0
                })
        
        return feedback
    
    def _create_enhanced_feedback_prompt(self, prompt: str, local_response: str, samples: Dict[str, str]) -> str:
        """Create enhanced prompt for OpenAI teacher feedback."""
        
        combined_samples = combine_style_samples(samples)
        
        return f"""You are an expert writing style teacher. Analyze the following response and provide detailed feedback.

Writing Style Samples:
{combined_samples}

Original Prompt: {prompt}

Student Response: {local_response}

Please provide a comprehensive analysis:

1. Style similarity score (0-1) - how well does this match the writing style?
2. Tone analysis - is the tone appropriate and consistent?
3. Structure analysis - does the structure match the style?
4. Specific suggestions for improvement
5. What aspects are good and should be kept

Format your response as:
Score: [0-1]
Tone Analysis: [formal/casual/professional] - [explanation]
Structure Analysis: [sentence length, paragraph structure, etc.]
Good aspects: [list]
Suggestions: [list]"""
    
    def _parse_enhanced_teacher_feedback(self, feedback_text: str) -> Dict[str, any]:
        """Parse enhanced OpenAI feedback response."""
        
        # Enhanced parsing
        lines = feedback_text.split('\n')
        score = 0.5  # Default
        good_aspects = []
        suggestions = []
        tone_analysis = {}
        structure_analysis = {}
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Score:'):
                try:
                    score = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Tone Analysis:'):
                current_section = 'tone'
                tone_text = line.split(':', 1)[1].strip() if ':' in line else ''
                tone_analysis['analysis'] = tone_text
            elif line.startswith('Structure Analysis:'):
                current_section = 'structure'
                structure_text = line.split(':', 1)[1].strip() if ':' in line else ''
                structure_analysis['analysis'] = structure_text
            elif line.startswith('Good aspects:'):
                current_section = 'good'
                good_text = line.split(':', 1)[1].strip() if ':' in line else ''
                good_aspects = [item.strip() for item in good_text.split(',') if item.strip()]
            elif line.startswith('Suggestions:'):
                current_section = 'suggestions'
                suggestions_text = line.split(':', 1)[1].strip() if ':' in line else ''
                suggestions = [item.strip() for item in suggestions_text.split(',') if item.strip()]
            elif current_section and line:
                if current_section == 'tone':
                    tone_analysis['details'] = line
                elif current_section == 'structure':
                    structure_analysis['details'] = line
                elif current_section == 'good':
                    good_aspects.extend([item.strip() for item in line.split(',') if item.strip()])
                elif current_section == 'suggestions':
                    suggestions.extend([item.strip() for item in line.split(',') if item.strip()])
        
        return {
            'style_score': score,
            'good_aspects': good_aspects,
            'suggestions': suggestions,
            'tone_analysis': tone_analysis,
            'structure_analysis': structure_analysis
        }
    
    def _analyze_performance(
        self, 
        local_responses: List[Dict[str, str]], 
        teacher_feedback: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Analyze enhanced performance metrics."""
        
        scores = [feedback['style_score'] for feedback in teacher_feedback]
        
        # Enhanced metrics
        tone_scores = []
        structure_scores = []
        
        for feedback in teacher_feedback:
            # Simple tone scoring based on feedback
            tone_analysis = feedback.get('tone_analysis', {})
            if 'analysis' in tone_analysis:
                if 'appropriate' in tone_analysis['analysis'].lower():
                    tone_scores.append(1.0)
                elif 'inconsistent' in tone_analysis['analysis'].lower():
                    tone_scores.append(0.5)
                else:
                    tone_scores.append(0.7)
            
            # Simple structure scoring
            structure_analysis = feedback.get('structure_analysis', {})
            if 'analysis' in structure_analysis:
                if 'good' in structure_analysis['analysis'].lower():
                    structure_scores.append(1.0)
                elif 'poor' in structure_analysis['analysis'].lower():
                    structure_scores.append(0.3)
                else:
                    structure_scores.append(0.7)
        
        return {
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'overall_score': np.mean(scores),
            'tone_score': np.mean(tone_scores) if tone_scores else 0.5,
            'structure_score': np.mean(structure_scores) if structure_scores else 0.5,
            'total_cost': sum(feedback.get('openai_cost', 0) for feedback in teacher_feedback)
        }
    
    def _update_enhanced_model_from_feedback(self, teacher_feedback: List[Dict[str, any]]) -> None:
        """Update enhanced model based on detailed feedback."""
        
        # Extract all suggestions and tone/structure feedback
        all_suggestions = []
        tone_improvements = []
        structure_improvements = []
        
        for feedback in teacher_feedback:
            all_suggestions.extend(feedback.get('suggestions', []))
            
            # Extract tone improvements
            tone_analysis = feedback.get('tone_analysis', {})
            if 'analysis' in tone_analysis:
                tone_improvements.append(tone_analysis['analysis'])
            
            # Extract structure improvements
            structure_analysis = feedback.get('structure_analysis', {})
            if 'analysis' in structure_analysis:
                structure_improvements.append(structure_analysis['analysis'])
        
        # Update the enhanced model
        self.local_model.update_from_feedback(teacher_feedback)
        
        # Additional enhancements based on tone/structure feedback
        self._enhance_tone_patterns(tone_improvements)
        self._enhance_structure_patterns(structure_improvements)
        
        # Save updated model
        self.local_model.save_model()
        
        # Record training step
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'feedback_count': len(teacher_feedback),
            'avg_score': np.mean([f['style_score'] for f in teacher_feedback]),
            'tone_improvements': len(tone_improvements),
            'structure_improvements': len(structure_improvements)
        })
    
    def _enhance_tone_patterns(self, tone_improvements: List[str]) -> None:
        """Enhance tone patterns based on feedback."""
        
        for improvement in tone_improvements:
            improvement_lower = improvement.lower()
            
            # Extract tone-specific words from feedback
            if 'formal' in improvement_lower:
                formal_words = re.findall(r'\b\w+\b', improvement_lower)
                self.local_model.tone_markers['formal'].extend(formal_words)
            
            if 'casual' in improvement_lower:
                casual_words = re.findall(r'\b\w+\b', improvement_lower)
                self.local_model.tone_markers['casual'].extend(casual_words)
            
            if 'professional' in improvement_lower:
                professional_words = re.findall(r'\b\w+\b', improvement_lower)
                self.local_model.tone_markers['professional'].extend(professional_words)
        
        # Remove duplicates
        for tone in self.local_model.tone_markers:
            self.local_model.tone_markers[tone] = list(set(self.local_model.tone_markers[tone]))
    
    def _enhance_structure_patterns(self, structure_improvements: List[str]) -> None:
        """Enhance structure patterns based on feedback."""
        
        for improvement in structure_improvements:
            improvement_lower = improvement.lower()
            
            # Extract structure-related words
            structure_words = re.findall(r'\b\w+\b', improvement_lower)
            self.local_model.vocabulary.update(structure_words)
            
            # Look for specific structure suggestions
            if 'sentence' in improvement_lower:
                # Extract sentence length suggestions
                length_matches = re.findall(r'(\d+)\s*words?', improvement_lower)
                if length_matches:
                    suggested_length = int(length_matches[0])
                    # Update sentence patterns
                    if self.local_model.sentence_patterns:
                        # Adjust average sentence length
                        current_avg = np.mean([p['length'] for p in self.local_model.sentence_patterns])
                        # Add new pattern with suggested length
                        self.local_model.sentence_patterns.append({
                            'length': suggested_length,
                            'structure': {'word_count': suggested_length},
                            'tone_indicators': {'formal': 0, 'casual': 0, 'professional': 0, 'emphatic': 0}
                        })
    
    def _evaluate_final_performance(
        self, 
        test_prompts: List[str], 
        samples: Dict[str, str]
    ) -> Dict[str, any]:
        """Evaluate final enhanced model performance."""
        
        console.print("\n📊 Final Enhanced Performance Evaluation")
        
        # Generate final responses
        final_responses = []
        for prompt in test_prompts:
            response = self.local_model.generate_response(prompt)
            final_responses.append({
                'prompt': prompt,
                'response': response
            })
        
        # Get final teacher feedback
        final_feedback = self._get_enhanced_teacher_feedback(final_responses, samples)
        final_performance = self._analyze_performance(final_responses, final_feedback)
        
        return {
            'responses': final_responses,
            'feedback': final_feedback,
            'performance': final_performance
        }
    
    def compare_enhanced_models(
        self, 
        test_prompts: List[str], 
        samples: Dict[str, str]
    ) -> Dict[str, any]:
        """Compare enhanced local model vs OpenAI performance."""
        
        console.print("🔍 Comparing Enhanced Local vs OpenAI Models")
        
        results = {
            'enhanced_local': [],
            'openai': [],
            'comparison': {}
        }
        
        total_local_cost = 0
        total_openai_cost = 0
        
        for prompt in test_prompts:
            # Enhanced local model response
            local_response = self.local_model.generate_response(prompt)
            results['enhanced_local'].append({
                'prompt': prompt,
                'response': local_response,
                'cost': 0.0  # Local model has no API cost
            })
            
            # OpenAI response
            try:
                openai_result = self.openai_client.generate_response(
                    style_samples=samples,
                    user_input=prompt,
                    temperature=0.7,
                    max_tokens=200
                )
                
                results['openai'].append({
                    'prompt': prompt,
                    'response': openai_result['response'],
                    'cost': openai_result['cost']
                })
                
                total_openai_cost += openai_result['cost']
                
            except Exception as e:
                console.print(f"⚠️  OpenAI generation failed: {e}")
                results['openai'].append({
                    'prompt': prompt,
                    'response': "Generation failed",
                    'cost': 0.0
                })
        
        # Calculate comparison metrics
        results['comparison'] = {
            'total_local_cost': total_local_cost,
            'total_openai_cost': total_openai_cost,
            'cost_savings': total_openai_cost - total_local_cost,
            'local_responses': len(results['enhanced_local']),
            'openai_responses': len(results['openai'])
        }
        
        return results
    
    def display_enhanced_training_results(self, results: Dict[str, any]) -> None:
        """Display enhanced training results in a nice format."""
        
        console.print("\n" + "="*60)
        console.print("[bold blue]Enhanced Reinforcement Learning Training Results[/bold blue]")
        console.print("="*60)
        
        # Show enhanced performance progression
        if results['performance_metrics']:
            table = Table(title="Enhanced Performance Progression")
            table.add_column("Iteration", style="cyan")
            table.add_column("Style Score", justify="right")
            table.add_column("Tone Score", justify="right")
            table.add_column("Structure Score", justify="right")
            table.add_column("Cost", justify="right")
            
            for i, metrics in enumerate(results['performance_metrics'], 1):
                table.add_row(
                    str(i),
                    f"{metrics['overall_score']:.3f}",
                    f"{metrics.get('tone_score', 0.0):.3f}",
                    f"{metrics.get('structure_score', 0.0):.3f}",
                    f"${metrics['total_cost']:.4f}"
                )
            
            console.print(table)
        
        # Show final evaluation
        if 'final_evaluation' in results:
            final_perf = results['final_evaluation']['performance']
            console.print(f"\n🎯 Final Performance: {final_perf['overall_score']:.3f}")
            console.print(f"🎭 Tone Score: {final_perf.get('tone_score', 0.0):.3f}")
            console.print(f"🏗️  Structure Score: {final_perf.get('structure_score', 0.0):.3f}")
            console.print(f"💰 Total Training Cost: ${sum(m['total_cost'] for m in results['performance_metrics']):.4f}")
        
        # Show enhanced model info
        model_info = self.local_model.get_model_info()
        console.print(f"\n📊 Enhanced Model Statistics:")
        console.print(f"   Vocabulary Size: {model_info['vocabulary_size']}")
        console.print(f"   Common Phrases: {model_info['common_phrases']}")
        console.print(f"   Tone Markers: {model_info['tone_markers']}")
        console.print(f"   Structure Patterns: {model_info['structure_patterns']}")
        console.print(f"   LLM Config: {model_info['llm_config']['model']}")
        
        # Show continuous learning information
        if 'learning_summary' in results:
            learning = results['learning_summary']
            console.print(f"\n📈 Continuous Learning Summary:")
            console.print(f"   Total Sessions: {learning['total_sessions']}")
            console.print(f"   Total Samples: {learning['total_samples']}")
            console.print(f"   Best Style Score: {learning['best_scores']['style']:.3f}")
            console.print(f"   Best Tone Score: {learning['best_scores']['tone']:.3f}")
            console.print(f"   Best Structure Score: {learning['best_scores']['structure']:.3f}")
            
            if 'convergence_analysis' in results:
                conv = results['convergence_analysis']
                console.print(f"   Convergence: {'✅ Yes' if conv['converged'] else '❌ No'} ({conv['confidence']})")
            
            if 'performance_trend' in results:
                trend = results['performance_trend']
                console.print(f"   Trend: {trend['trend']} (rate: {trend['improvement_rate']:.3f})")
    
    def save_enhanced_training_results(self, results: Dict[str, any], filename: str = None) -> None:
        """Save enhanced training results to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_reinforcement_training_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        # Deep convert
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = deep_convert(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        console.print(f"💾 Enhanced training results saved to {filename}") 