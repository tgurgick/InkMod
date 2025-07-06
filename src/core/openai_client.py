"""OpenAI client for InkMod."""

import tiktoken
from typing import Dict, List, Optional
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress

from config.settings import settings

console = Console()

class OpenAIClient:
    """Handles OpenAI API interactions for style mirroring."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.client = OpenAI(api_key=self.api_key)
        self.encoding = tiktoken.encoding_for_model(self.model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on token usage."""
        # Rough cost estimates (these may vary)
        if "gpt-4" in self.model:
            return (prompt_tokens * 0.00003) + (completion_tokens * 0.00006)
        else:  # gpt-3.5-turbo
            return (prompt_tokens * 0.0000015) + (completion_tokens * 0.000002)
    
    def generate_response(
        self,
        style_samples: Dict[str, str],
        user_input: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> Dict[str, any]:
        """Generate a response that matches the provided style samples."""
        
        # Prepare the prompt
        prompt = self._build_prompt(style_samples, user_input, system_prompt)
        
        # Count tokens and estimate cost
        prompt_tokens = self.count_tokens(prompt)
        estimated_completion_tokens = max_tokens
        
        console.info(f"Prompt tokens: {prompt_tokens}")
        console.info(f"Estimated cost: ${self.estimate_cost(prompt_tokens, estimated_completion_tokens):.4f}")
        
        try:
            with Progress() as progress:
                task = progress.add_task("Generating response...", total=None)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(system_prompt)},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                progress.update(task, completed=True)
            
            completion = response.choices[0].message.content
            actual_completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            actual_cost = self.estimate_cost(prompt_tokens, actual_completion_tokens)
            
            return {
                'response': completion,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': actual_completion_tokens,
                'total_tokens': total_tokens,
                'cost': actual_cost,
                'model': self.model
            }
            
        except Exception as e:
            console.print(f"❌ OpenAI API error: {e}")
            raise
    
    def _build_prompt(self, style_samples: Dict[str, str], user_input: str, system_prompt: Optional[str] = None) -> str:
        """Build the prompt for style mirroring."""
        from utils.text_utils import combine_style_samples, extract_style_summary
        
        # Combine style samples
        style_context = combine_style_samples(style_samples)
        
        # Add style analysis
        style_summary = extract_style_summary(style_samples)
        
        # Build the prompt
        prompt_parts = [
            "Below are writing samples that demonstrate a specific writing style:",
            "",
            style_context,
            "",
            f"Style characteristics:",
            style_summary,
            "",
            f"Please write a response to the following request, matching the style of the samples above:",
            "",
            f"Request: {user_input}",
            "",
            "Response:"
        ]
        
        return "\n".join(prompt_parts)
    
    def _get_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """Get the system prompt for the conversation."""
        if custom_prompt:
            return custom_prompt
        
        return """You are a writing assistant that can mirror specific writing styles. 
Your task is to analyze the provided writing samples and generate responses that match the style, tone, vocabulary, and structure of those samples.

Key guidelines:
1. Match the tone (formal, casual, academic, etc.)
2. Use similar vocabulary and word choices
3. Follow similar sentence structure patterns
4. Maintain the same level of formality
5. Use similar punctuation and formatting
6. Avoid adding your own style - strictly mirror the provided samples

Respond only with the requested content, matching the style as closely as possible.""" 