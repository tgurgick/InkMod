"""Reinforcement learning system for local style model training."""

import json
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.openai_client import OpenAIClient
from core.local_style_model import LocalStyleModel
from utils.text_utils import combine_style_samples

console = Console()

class ReinforcementTrainer:
    """Reinforcement learning system using OpenAI as teacher."""
    
    def __init__(self, openai_client: OpenAIClient, model_path: str = "local_style_model.pkl"):
        self.openai_client = openai_client
        self.local_model = LocalStyleModel(model_path)
        self.training_history = []
        self.performance_metrics = []
        
    def train_with_reinforcement(
        self, 
        samples: Dict[str, str], 
        test_prompts: List[str],
        iterations: int = 5
    ) -> Dict[str, any]:
        """Train local model using OpenAI as teacher."""
        
        console.print("🎯 Starting reinforcement learning training...")
        
        # Initial training
        console.print("\n📚 Phase 1: Initial local model training")
        initial_stats = self.local_model.train(samples)
        console.print(f"✅ Initial training complete: {initial_stats['vocabulary_size']} vocabulary items")
        
        # Reinforcement learning loop
        for iteration in range(iterations):
            console.print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            
            # Generate responses with local model
            local_responses = []
            for prompt in test_prompts:
                response = self.local_model.generate_response(prompt)
                local_responses.append({
                    'prompt': prompt,
                    'response': response
                })
            
            # Get OpenAI teacher feedback
            teacher_feedback = self._get_teacher_feedback(local_responses, samples)
            
            # Analyze performance
            performance = self._analyze_performance(local_responses, teacher_feedback)
            self.performance_metrics.append(performance)
            
            # Update local model based on feedback
            self._update_model_from_feedback(teacher_feedback)
            
            console.print(f"📊 Iteration {iteration + 1} performance: {performance['overall_score']:.3f}")
        
        # Final evaluation
        final_evaluation = self._evaluate_final_performance(test_prompts, samples)
        
        return {
            'initial_stats': initial_stats,
            'performance_metrics': self.performance_metrics,
            'final_evaluation': final_evaluation,
            'training_history': self.training_history
        }
    
    def _get_teacher_feedback(
        self, 
        local_responses: List[Dict[str, str]], 
        samples: Dict[str, str]
    ) -> List[Dict[str, any]]:
        """Get feedback from OpenAI teacher."""
        
        feedback = []
        
        for response_data in local_responses:
            prompt = response_data['prompt']
            local_response = response_data['response']
            
            # Create feedback prompt for OpenAI
            feedback_prompt = self._create_feedback_prompt(prompt, local_response, samples)
            
            try:
                # Get OpenAI's analysis
                result = self.openai_client.generate_response(
                    style_samples={},
                    user_input=feedback_prompt,
                    temperature=0.3,
                    max_tokens=300
                )
                
                # Parse feedback
                parsed_feedback = self._parse_teacher_feedback(result['response'])
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
                    'openai_cost': 0.0
                })
        
        return feedback
    
    def _create_feedback_prompt(self, prompt: str, local_response: str, samples: Dict[str, str]) -> str:
        """Create prompt for OpenAI teacher feedback."""
        
        combined_samples = combine_style_samples(samples)
        
        return f"""You are a writing style teacher. Analyze the following response and provide feedback.

Writing Style Samples:
{combined_samples}

Original Prompt: {prompt}

Student Response: {local_response}

Please provide:
1. A style similarity score (0-1) - how well does this match the writing style?
2. Specific suggestions for improvement
3. What aspects are good and should be kept

Format your response as:
Score: [0-1]
Good aspects: [list]
Suggestions: [list]"""
    
    def _parse_teacher_feedback(self, feedback_text: str) -> Dict[str, any]:
        """Parse OpenAI's feedback response."""
        
        # Simple parsing - could be enhanced
        lines = feedback_text.split('\n')
        score = 0.5  # Default
        good_aspects = []
        suggestions = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Score:'):
                try:
                    score = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('Good aspects:'):
                good_aspects = [item.strip() for item in line.split(':')[1].split(',')]
            elif line.startswith('Suggestions:'):
                suggestions = [item.strip() for item in line.split(':')[1].split(',')]
        
        return {
            'style_score': score,
            'good_aspects': good_aspects,
            'suggestions': suggestions
        }
    
    def _analyze_performance(
        self, 
        local_responses: List[Dict[str, str]], 
        teacher_feedback: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Analyze performance metrics."""
        
        scores = [feedback['style_score'] for feedback in teacher_feedback]
        
        return {
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'overall_score': np.mean(scores),
            'total_cost': sum(feedback.get('openai_cost', 0) for feedback in teacher_feedback)
        }
    
    def _update_model_from_feedback(self, teacher_feedback: List[Dict[str, any]]) -> None:
        """Update local model based on teacher feedback."""
        
        # Simple update strategy - could be enhanced with more sophisticated learning
        
        # Extract common suggestions
        all_suggestions = []
        for feedback in teacher_feedback:
            all_suggestions.extend(feedback.get('suggestions', []))
        
        # Update vocabulary based on suggestions
        # This is a simplified approach - in practice, you'd want more sophisticated NLP
        for suggestion in all_suggestions:
            words = suggestion.lower().split()
            self.local_model.vocabulary.update(words)
        
        # Save updated model
        self.local_model.save_model()
        
        # Record training step
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'feedback_count': len(teacher_feedback),
            'avg_score': np.mean([f['style_score'] for f in teacher_feedback])
        })
    
    def _evaluate_final_performance(
        self, 
        test_prompts: List[str], 
        samples: Dict[str, str]
    ) -> Dict[str, any]:
        """Evaluate final model performance."""
        
        console.print("\n📊 Final Performance Evaluation")
        
        # Generate final responses
        final_responses = []
        for prompt in test_prompts:
            response = self.local_model.generate_response(prompt)
            final_responses.append({
                'prompt': prompt,
                'response': response
            })
        
        # Get final teacher feedback
        final_feedback = self._get_teacher_feedback(final_responses, samples)
        final_performance = self._analyze_performance(final_responses, final_feedback)
        
        return {
            'responses': final_responses,
            'feedback': final_feedback,
            'performance': final_performance
        }
    
    def compare_models(
        self, 
        test_prompts: List[str], 
        samples: Dict[str, str]
    ) -> Dict[str, any]:
        """Compare local model vs OpenAI performance."""
        
        console.print("🔍 Comparing Local vs OpenAI Models")
        
        results = {
            'local': [],
            'openai': [],
            'comparison': {}
        }
        
        total_local_cost = 0
        total_openai_cost = 0
        
        for prompt in test_prompts:
            # Local model response
            local_response = self.local_model.generate_response(prompt)
            results['local'].append({
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
            'local_responses': len(results['local']),
            'openai_responses': len(results['openai'])
        }
        
        return results
    
    def display_training_results(self, results: Dict[str, any]) -> None:
        """Display training results in a nice format."""
        
        console.print("\n" + "="*60)
        console.print("[bold blue]Reinforcement Learning Training Results[/bold blue]")
        console.print("="*60)
        
        # Show performance progression
        if results['performance_metrics']:
            table = Table(title="Performance Progression")
            table.add_column("Iteration", style="cyan")
            table.add_column("Avg Score", justify="right")
            table.add_column("Min Score", justify="right")
            table.add_column("Max Score", justify="right")
            table.add_column("Cost", justify="right")
            
            for i, metrics in enumerate(results['performance_metrics'], 1):
                table.add_row(
                    str(i),
                    f"{metrics['avg_score']:.3f}",
                    f"{metrics['min_score']:.3f}",
                    f"{metrics['max_score']:.3f}",
                    f"${metrics['total_cost']:.4f}"
                )
            
            console.print(table)
        
        # Show final evaluation
        if 'final_evaluation' in results:
            final_perf = results['final_evaluation']['performance']
            console.print(f"\n🎯 Final Performance: {final_perf['overall_score']:.3f}")
            console.print(f"💰 Total Training Cost: ${sum(m['total_cost'] for m in results['performance_metrics']):.4f}")
        
        # Show model info
        model_info = self.local_model.get_model_info()
        console.print(f"\n📊 Final Model Statistics:")
        console.print(f"   Vocabulary Size: {model_info['vocabulary_size']}")
        console.print(f"   Sentence Patterns: {model_info['sentence_patterns']}")
        console.print(f"   Paragraph Patterns: {model_info['paragraph_patterns']}")
    
    def save_training_results(self, results: Dict[str, any], filename: str = None) -> None:
        """Save training results to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reinforcement_training_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
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
        
        console.print(f"💾 Training results saved to {filename}") 