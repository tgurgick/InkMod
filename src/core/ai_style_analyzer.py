"""AI-powered style analysis for InkMod."""

from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel

from core.openai_client import OpenAIClient
from utils.text_utils import combine_style_samples

console = Console()

class AIStyleAnalyzer:
    """Uses OpenAI to generate nuanced style analysis."""
    
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
    
    def analyze_style_with_ai(self, samples: Dict[str, str]) -> Dict[str, any]:
        """Generate AI-powered style analysis."""
        
        # Combine samples for analysis
        combined_samples = combine_style_samples(samples)
        
        # Create analysis prompt
        analysis_prompt = self._create_analysis_prompt(combined_samples)
        
        try:
            console.print("🤖 Analyzing writing style with AI...")
            
            # Generate style analysis
            result = self.openai_client.generate_response(
                style_samples={},  # Empty for analysis
                user_input=analysis_prompt,
                temperature=0.3,  # Lower temperature for consistent analysis
                max_tokens=800
            )
            
            # Parse the analysis
            style_guide = self._parse_style_analysis(result['response'])
            
            return {
                'ai_analysis': result['response'],
                'style_guide': style_guide,
                'tokens_used': result['total_tokens'],
                'cost': result['cost']
            }
            
        except Exception as e:
            console.print(f"❌ AI style analysis failed: {e}")
            return self._fallback_analysis(samples)
    
    def _create_analysis_prompt(self, samples: str) -> str:
        """Create prompt for AI style analysis."""
        
        return f"""Please analyze the following writing samples and create a comprehensive style guide.

Writing samples:
{samples}

Please provide a detailed analysis including:

1. **Tone and Voice:**
   - Overall tone (formal, casual, academic, etc.)
   - Voice characteristics (authoritative, friendly, professional, etc.)
   - Emotional undertones

2. **Writing Patterns:**
   - Sentence structure preferences
   - Paragraph organization
   - Transitional phrases and connectors
   - Repetition patterns

3. **Language Choices:**
   - Vocabulary level and complexity
   - Technical vs. conversational language
   - Industry-specific terminology
   - Cultural references or idioms

4. **Structural Elements:**
   - Opening and closing patterns
   - Use of lists, bullet points, or formatting
   - Citation or reference styles
   - Length preferences

5. **Unique Characteristics:**
   - Distinctive phrases or expressions
   - Personal writing quirks
   - Consistent formatting choices

Format your response as a clear, actionable style guide that can be used to replicate this writing style. Be specific and provide examples where possible."""

    def _parse_style_analysis(self, analysis: str) -> Dict[str, str]:
        """Parse AI analysis into structured format."""
        
        # Simple parsing - could be enhanced with more sophisticated parsing
        sections = {
            'tone_voice': '',
            'writing_patterns': '',
            'language_choices': '',
            'structural_elements': '',
            'unique_characteristics': ''
        }
        
        # Basic section extraction (could be improved with regex or NLP)
        lines = analysis.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'tone' in line.lower() or 'voice' in line.lower():
                current_section = 'tone_voice'
            elif 'pattern' in line.lower():
                current_section = 'writing_patterns'
            elif 'language' in line.lower() or 'vocabulary' in line.lower():
                current_section = 'language_choices'
            elif 'structural' in line.lower() or 'format' in line.lower():
                current_section = 'structural_elements'
            elif 'unique' in line.lower() or 'characteristic' in line.lower():
                current_section = 'unique_characteristics'
            elif current_section and line:
                sections[current_section] += line + '\n'
        
        return sections
    
    def _fallback_analysis(self, samples: Dict[str, str]) -> Dict[str, any]:
        """Fallback to basic analysis if AI fails."""
        from ..utils.text_utils import extract_style_summary
        
        console.print("⚠️  Falling back to basic style analysis...")
        
        basic_analysis = extract_style_summary(samples)
        
        return {
            'ai_analysis': f"Basic analysis: {basic_analysis}",
            'style_guide': {'basic_metrics': basic_analysis},
            'tokens_used': 0,
            'cost': 0.0
        }
    
    def display_ai_analysis(self, analysis_result: Dict[str, any]) -> None:
        """Display AI-generated style analysis."""
        
        console.print("\n[bold blue]AI Style Analysis[/bold blue]")
        console.print("=" * 50)
        
        # Show AI analysis
        console.print("[bold]AI-Generated Style Guide:[/bold]")
        console.print(analysis_result['ai_analysis'])
        console.print()
        
        # Show usage info
        console.print(f"📊 Analysis tokens: {analysis_result['tokens_used']}")
        console.print(f"💰 Analysis cost: ${analysis_result['cost']:.4f}")
        console.print()
        
        # Show structured guide
        if 'style_guide' in analysis_result and isinstance(analysis_result['style_guide'], dict):
            console.print("[bold]Structured Style Guide:[/bold]")
            for section, content in analysis_result['style_guide'].items():
                if content.strip():
                    console.print(f"\n[cyan]{section.replace('_', ' ').title()}:[/cyan]")
                    console.print(content.strip()) 