"""Hybrid reinforcement learning system using standard NLP metrics + LLM qualitative feedback."""

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
from core.hybrid_reward_function import HybridRewardFunction
from utils.text_utils import combine_style_samples

console = Console()

class HybridReinforcementTrainer:
    """Hybrid reinforcement learning system using standard NLP metrics + LLM qualitative feedback."""
    
    def __init__(self, openai_client: OpenAIClient, model_path: str = "enhanced_style_model.pkl", backend_name: str = None):
        self.openai_client = openai_client
        self.local_model = EnhancedStyleModel(model_path, use_local_llm=True, backend_name=backend_name)
        self.reward_function = HybridRewardFunction()
        self.training_history = []
        self.performance_metrics = []
        self.backend_name = backend_name
        
    def train_with_hybrid_reinforcement(
        self, 
        samples: Dict[str, str], 
        test_prompts: List[str],
        iterations: int = 5,
        incremental: bool = True,
        use_llm_feedback: bool = True
    ) -> Dict[str, any]:
        """Train enhanced local model using hybrid reward function."""
        
        console.print("🎯 Starting hybrid reinforcement learning training...")
        console.print("📊 Using standard NLP metrics + LLM qualitative feedback")
        
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
        
        # Convert samples to list for reward function
        reference_samples = list(samples.values())
        
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
            
            # Calculate hybrid rewards
            hybrid_rewards = []
            total_cost = 0.0
            
            for response_data in local_responses:
                prompt = response_data['prompt']
                local_response = response_data['response']
                
                # Get LLM qualitative feedback (optional)
                llm_feedback = None
                if use_llm_feedback:
                    llm_feedback = self._get_llm_qualitative_feedback(prompt, local_response, samples)
                    if llm_feedback:
                        total_cost += llm_feedback.get('cost', 0.0)
                
                # Calculate hybrid reward
                reward = self.reward_function.calculate_reward(
                    generated_text=local_response,
                    reference_samples=reference_samples,
                    llm_feedback=llm_feedback.get('feedback', None) if llm_feedback else None
                )
                
                reward['prompt'] = prompt
                reward['local_response'] = local_response
                hybrid_rewards.append(reward)
            
            # Analyze performance
            performance = self._analyze_hybrid_performance(hybrid_rewards, total_cost)
            
            # Add performance metric to continuous learning
            self.local_model.add_performance_metric(performance)
            
            # Update enhanced model based on hybrid feedback
            self._update_enhanced_model_from_hybrid_feedback(hybrid_rewards)
            
            console.print(f"📊 Iteration {iteration + 1} performance: {performance['overall_score']:.3f}")
            console.print(f"💰 LLM feedback cost: ${total_cost:.4f}")
            
            # Check if we should continue training
            if iteration >= 2:  # Need at least 3 iterations to check convergence
                continue_decision = self.local_model.should_continue_training()
                if not continue_decision['continue']:
                    console.print(f"🛑 Training stopped: {continue_decision['reason']}")
                    if 'confidence' in continue_decision:
                        console.print(f"📊 Confidence: {continue_decision['confidence']}")
                    break
        
        # Final evaluation
        final_evaluation = self._evaluate_final_hybrid_performance(test_prompts, samples)
        
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
            'performance_trend': self.local_model._get_performance_trend(),
            'reward_function_type': 'hybrid'
        }
    
    def _get_llm_qualitative_feedback(self, prompt: str, local_response: str, samples: Dict[str, str]) -> Dict[str, any]:
        """Get qualitative feedback from LLM (no scoring)."""
        
        try:
            # Create qualitative feedback prompt
            feedback_prompt = self._create_qualitative_feedback_prompt(prompt, local_response, samples)
            
            # Get OpenAI's qualitative analysis
            result = self.openai_client.generate_response(
                style_samples={},
                user_input=feedback_prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            return {
                'feedback': result['response'],
                'cost': result.get('cost', 0.0)
            }
            
        except Exception as e:
            console.print(f"⚠️  Failed to get LLM feedback: {e}")
            return {'feedback': None, 'cost': 0.0}
    
    def _create_qualitative_feedback_prompt(self, prompt: str, local_response: str, samples: Dict[str, str]) -> str:
        """Create prompt for qualitative LLM feedback (no scoring)."""
        
        combined_samples = combine_style_samples(samples)
        
        return f"""Analyze this writing style response and provide specific, actionable suggestions.

Reference Style Samples:
{combined_samples}

Original Prompt: {prompt}

Generated Response: {local_response}

Provide detailed qualitative feedback on:
1. What aspects match the style well?
2. What specific vocabulary improvements are needed?
3. How could the tone be improved?
4. What structural changes would help?
5. Any other specific suggestions for improvement?

Focus on concrete, actionable suggestions rather than numerical scores.
Be specific about vocabulary, tone, and structure improvements."""
    
    def _analyze_hybrid_performance(self, hybrid_rewards: List[Dict[str, any]], total_cost: float) -> Dict[str, any]:
        """Analyze performance using hybrid reward metrics."""
        
        # Extract objective scores
        objective_scores = [reward['objective_scores'] for reward in hybrid_rewards]
        
        # Calculate averages
        avg_bleu = np.mean([s['bleu_score'] for s in objective_scores])
        avg_rouge = np.mean([s['rouge_score'] for s in objective_scores])
        avg_consistency = np.mean([s['consistency_score'] for s in objective_scores])
        avg_length = np.mean([s['length_score'] for s in objective_scores])
        avg_tone = np.mean([s['tone_score'] for s in objective_scores])
        avg_overall = np.mean([s['overall_score'] for s in objective_scores])
        
        return {
            'avg_bleu_score': avg_bleu,
            'avg_rouge_score': avg_rouge,
            'avg_consistency_score': avg_consistency,
            'avg_length_score': avg_length,
            'avg_tone_score': avg_tone,
            'overall_score': avg_overall,
            'min_score': np.min([s['overall_score'] for s in objective_scores]),
            'max_score': np.max([s['overall_score'] for s in objective_scores]),
            'std_score': np.std([s['overall_score'] for s in objective_scores]),
            'total_cost': total_cost,
            'reward_type': 'hybrid'
        }
    
    def _update_enhanced_model_from_hybrid_feedback(self, hybrid_rewards: List[Dict[str, any]]) -> None:
        """Update enhanced model based on hybrid feedback."""
        
        # Extract all model updates from hybrid rewards
        all_vocabulary_additions = []
        all_phrase_additions = []
        all_tone_improvements = []
        all_structure_improvements = []
        
        for reward in hybrid_rewards:
            model_updates = reward.get('model_updates', {})
            
            all_vocabulary_additions.extend(model_updates.get('vocabulary_additions', []))
            all_phrase_additions.extend(model_updates.get('phrase_additions', []))
            all_tone_improvements.extend(model_updates.get('tone_improvements', []))
            all_structure_improvements.extend(model_updates.get('structure_improvements', []))
        
        # Update vocabulary
        for word in all_vocabulary_additions:
            self.local_model.vocabulary[word.lower()] += 1
        
        # Update common phrases
        self.local_model.common_phrases.extend(all_phrase_additions)
        
        # Update tone markers
        for tone_word in all_tone_improvements:
            # Determine which tone category it belongs to
            if any(marker in tone_word.lower() for marker in ['formal', 'professional', 'regarding']):
                self.local_model.tone_markers['formal'].append(tone_word.lower())
            elif any(marker in tone_word.lower() for marker in ['casual', 'friendly', 'informal']):
                self.local_model.tone_markers['casual'].append(tone_word.lower())
            elif any(marker in tone_word.lower() for marker in ['please', 'would', 'could', 'thank']):
                self.local_model.tone_markers['professional'].append(tone_word.lower())
        
        # Remove duplicates
        for tone in self.local_model.tone_markers:
            self.local_model.tone_markers[tone] = list(set(self.local_model.tone_markers[tone]))
        
        # Update style prompt template
        self.local_model.style_prompt_template = self.local_model._create_style_prompt_template()
        
        # Save updated model
        self.local_model.save_model()
        
        # Record training step
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'reward_count': len(hybrid_rewards),
            'avg_score': np.mean([r['overall_score'] for r in hybrid_rewards]),
            'vocabulary_additions': len(all_vocabulary_additions),
            'phrase_additions': len(all_phrase_additions),
            'tone_improvements': len(all_tone_improvements),
            'structure_improvements': len(all_structure_improvements)
        })
    
    def _evaluate_final_hybrid_performance(self, test_prompts: List[str], samples: Dict[str, str]) -> Dict[str, any]:
        """Evaluate final hybrid model performance."""
        
        console.print("\n📊 Final Hybrid Performance Evaluation")
        
        # Generate final responses
        final_responses = []
        reference_samples = list(samples.values())
        
        for prompt in test_prompts:
            response = self.local_model.generate_response(prompt)
            final_responses.append({
                'prompt': prompt,
                'response': response
            })
        
        # Calculate final hybrid rewards
        final_rewards = []
        for response_data in final_responses:
            reward = self.reward_function.calculate_reward(
                generated_text=response_data['response'],
                reference_samples=reference_samples
            )
            reward['prompt'] = response_data['prompt']
            reward['local_response'] = response_data['response']
            final_rewards.append(reward)
        
        # Analyze final performance
        final_performance = self._analyze_hybrid_performance(final_rewards, 0.0)
        
        return {
            'final_responses': final_responses,
            'final_rewards': final_rewards,
            'final_performance': final_performance
        }
    
    def display_hybrid_training_results(self, results: Dict[str, any]) -> None:
        """Display hybrid training results."""
        
        console.print("\n" + "="*60)
        console.print("Hybrid Reinforcement Learning Training Results")
        console.print("="*60)
        
        # Performance progression table
        if results['performance_metrics']:
            table = Table(title="Hybrid Performance Progression")
            table.add_column("Iteration", style="cyan")
            table.add_column("BLEU Score", style="green")
            table.add_column("ROUGE Score", style="blue")
            table.add_column("Consistency", style="yellow")
            table.add_column("Length", style="magenta")
            table.add_column("Tone", style="red")
            table.add_column("Overall", style="bold")
            table.add_column("Cost", style="dim")
            
            for i, metric in enumerate(results['performance_metrics']):
                table.add_row(
                    str(i + 1),
                    f"{metric.get('avg_bleu_score', 0):.3f}",
                    f"{metric.get('avg_rouge_score', 0):.3f}",
                    f"{metric.get('avg_consistency_score', 0):.3f}",
                    f"{metric.get('avg_length_score', 0):.3f}",
                    f"{metric.get('avg_tone_score', 0):.3f}",
                    f"{metric.get('overall_score', 0):.3f}",
                    f"${metric.get('total_cost', 0):.4f}"
                )
            
            console.print(table)
        
        # Final evaluation
        if 'final_evaluation' in results:
            final_perf = results['final_evaluation']['final_performance']
            console.print(f"\n🎯 Final Performance: {final_perf['overall_score']:.3f}")
            console.print(f"📊 BLEU Score: {final_perf['avg_bleu_score']:.3f}")
            console.print(f"📊 ROUGE Score: {final_perf['avg_rouge_score']:.3f}")
            console.print(f"📊 Consistency Score: {final_perf['avg_consistency_score']:.3f}")
            console.print(f"📊 Length Score: {final_perf['avg_length_score']:.3f}")
            console.print(f"📊 Tone Score: {final_perf['avg_tone_score']:.3f}")
        
        # Learning summary
        learning_summary = results['learning_summary']
        console.print(f"\n📈 Learning Summary:")
        console.print(f"   Total Sessions: {learning_summary['total_sessions']}")
        console.print(f"   Total Samples: {learning_summary['total_samples']}")
        console.print(f"   Best Style Score: {learning_summary['best_scores']['style']:.3f}")
        console.print(f"   Best Tone Score: {learning_summary['best_scores']['tone']:.3f}")
        console.print(f"   Best Structure Score: {learning_summary['best_scores']['structure']:.3f}")
        
        # Convergence analysis
        convergence = results['convergence_analysis']
        console.print(f"\n🔄 Convergence Status:")
        console.print(f"   Converged: {'✅ Yes' if convergence['converged'] else '❌ No'}")
        console.print(f"   Confidence: {convergence['confidence']}")
        console.print(f"   Variance: {convergence['variance']:.6f}")
    
    def save_hybrid_training_results(self, results: Dict[str, any], filename: str = None) -> None:
        """Save hybrid training results to file."""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hybrid_reinforcement_training_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
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
        
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        # Convert results
        json_results = deep_convert(results)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        console.print(f"💾 Hybrid training results saved to {filename}") 