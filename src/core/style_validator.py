"""Style validation and comparison system for InkMod."""

from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.openai_client import OpenAIClient
from core.ai_style_analyzer import AIStyleAnalyzer
from utils.text_utils import extract_style_summary, combine_style_samples

console = Console()

class StyleValidator:
    """Validates and compares different style analysis methods."""
    
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
        self.ai_analyzer = AIStyleAnalyzer(openai_client)
    
    def compare_analysis_methods(
        self, 
        samples: Dict[str, str], 
        test_input: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, any]:
        """Compare different style analysis methods."""
        
        console.print("🔬 Comparing style analysis methods...")
        
        results = {}
        
        # Method 1: Current rigid analysis
        console.print("\n📊 Method 1: Rigid Metrics Analysis")
        rigid_style = extract_style_summary(samples)
        rigid_output = self._generate_with_style_guide(samples, test_input, rigid_style, temperature, max_tokens)
        results['rigid'] = {
            'style_guide': rigid_style,
            'output': rigid_output['response'],
            'tokens': rigid_output['total_tokens'],
            'cost': rigid_output['cost']
        }
        
        # Method 2: AI-generated style analysis
        console.print("\n🤖 Method 2: AI-Generated Style Analysis")
        ai_analysis = self.ai_analyzer.analyze_style_with_ai(samples)
        ai_style_guide = ai_analysis['ai_analysis']
        ai_output = self._generate_with_style_guide(samples, test_input, ai_style_guide, temperature, max_tokens)
        results['ai'] = {
            'style_guide': ai_style_guide,
            'output': ai_output['response'],
            'tokens': ai_output['total_tokens'],
            'cost': ai_output['cost'],
            'analysis_cost': ai_analysis['cost']
        }
        
        # Method 3: Direct sample comparison (no style guide)
        console.print("\n📝 Method 3: Direct Sample Comparison")
        direct_output = self._generate_with_direct_samples(samples, test_input, temperature, max_tokens)
        results['direct'] = {
            'style_guide': "Direct sample comparison",
            'output': direct_output['response'],
            'tokens': direct_output['total_tokens'],
            'cost': direct_output['cost']
        }
        
        # Calculate similarity scores
        similarity_scores = self._calculate_similarity_scores(results, samples)
        
        return {
            'results': results,
            'similarity_scores': similarity_scores,
            'recommendation': self._get_recommendation(similarity_scores, results)
        }
    
    def _generate_with_style_guide(
        self, 
        samples: Dict[str, str], 
        test_input: str, 
        style_guide: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, any]:
        """Generate content using a specific style guide."""
        
        # Create enhanced prompt with style guide
        enhanced_prompt = f"""Style Guide:
{style_guide}

Writing Samples:
{combine_style_samples(samples)}

Please write a response to: {test_input}

Use the style guide above to ensure your response matches the writing style of the samples."""
        
        return self.openai_client.generate_response(
            style_samples={},  # We're using the style guide instead
            user_input=enhanced_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _generate_with_direct_samples(
        self, 
        samples: Dict[str, str], 
        test_input: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict[str, any]:
        """Generate content using direct sample comparison."""
        
        return self.openai_client.generate_response(
            style_samples=samples,
            user_input=test_input,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _calculate_similarity_scores(
        self, 
        results: Dict[str, any], 
        original_samples: Dict[str, str]
    ) -> Dict[str, float]:
        """Calculate how well each method matches the original style."""
        
        # Simple similarity scoring (could be enhanced with more sophisticated metrics)
        scores = {}
        
        for method, result in results.items():
            output = result['output']
            
            # Calculate basic similarity metrics
            similarity = self._calculate_basic_similarity(output, original_samples)
            scores[method] = similarity
        
        return scores
    
    def _calculate_basic_similarity(self, output: str, samples: Dict[str, str]) -> float:
        """Calculate basic similarity between output and original samples."""
        
        # Combine all original samples
        all_original = ' '.join(samples.values()).lower()
        output_lower = output.lower()
        
        # Simple word overlap similarity
        original_words = set(all_original.split())
        output_words = set(output_lower.split())
        
        if not original_words:
            return 0.0
        
        overlap = len(original_words.intersection(output_words))
        similarity = overlap / len(original_words)
        
        return min(similarity * 2, 1.0)  # Scale and cap at 1.0
    
    def _get_recommendation(
        self, 
        similarity_scores: Dict[str, float], 
        results: Dict[str, any]
    ) -> Dict[str, any]:
        """Get recommendation based on similarity scores and costs."""
        
        # Find best performing method
        best_method = max(similarity_scores.items(), key=lambda x: x[1])
        
        # Calculate total costs
        total_costs = {}
        for method, result in results.items():
            if method == 'ai':
                total_costs[method] = result['cost'] + result['analysis_cost']
            else:
                total_costs[method] = result['cost']
        
        return {
            'best_method': best_method[0],
            'best_score': best_method[1],
            'total_costs': total_costs,
            'recommendation': self._generate_recommendation_text(
                similarity_scores, total_costs, best_method
            )
        }
    
    def _generate_recommendation_text(
        self, 
        scores: Dict[str, float], 
        costs: Dict[str, float], 
        best_method: Tuple[str, float]
    ) -> str:
        """Generate human-readable recommendation."""
        
        method_name = best_method[0]
        score = best_method[1]
        
        if method_name == 'ai':
            return f"AI-generated style analysis performed best (similarity: {score:.2f}) but costs ${costs['ai']:.4f} total."
        elif method_name == 'rigid':
            return f"Rigid metrics analysis performed best (similarity: {score:.2f}) and is cost-effective (${costs['rigid']:.4f})."
        else:
            return f"Direct sample comparison performed best (similarity: {score:.2f}) and is most cost-effective (${costs['direct']:.4f})."
    
    def display_comparison_results(self, comparison_result: Dict[str, any]) -> None:
        """Display comparison results in a nice format."""
        
        results = comparison_result['results']
        scores = comparison_result['similarity_scores']
        recommendation = comparison_result['recommendation']
        
        console.print("\n" + "="*60)
        console.print("[bold blue]Style Analysis Method Comparison[/bold blue]")
        console.print("="*60)
        
        # Create comparison table
        table = Table(title="Method Comparison")
        table.add_column("Method", style="cyan")
        table.add_column("Similarity Score", justify="right")
        table.add_column("Generation Cost", justify="right")
        table.add_column("Total Cost", justify="right")
        table.add_column("Best Method", justify="center")
        
        for method, result in results.items():
            similarity = scores.get(method, 0.0)
            gen_cost = result['cost']
            
            if method == 'ai':
                total_cost = result['cost'] + result['analysis_cost']
            else:
                total_cost = result['cost']
            
            is_best = method == recommendation['best_method']
            best_marker = "⭐" if is_best else ""
            
            table.add_row(
                method.title(),
                f"{similarity:.3f}",
                f"${gen_cost:.4f}",
                f"${total_cost:.4f}",
                best_marker
            )
        
        console.print(table)
        console.print()
        
        # Show recommendation
        console.print(Panel(
            f"[bold]Recommendation:[/bold] {recommendation['recommendation']}",
            title="🎯 Analysis Recommendation",
            border_style="green"
        ))
        
        # Show sample outputs
        console.print("\n[bold]Sample Outputs:[/bold]")
        for method, result in results.items():
            console.print(f"\n[cyan]{method.title()} Method:[/cyan]")
            console.print(result['output'][:200] + "..." if len(result['output']) > 200 else result['output']) 