"""Prompt engineering for InkMod."""

from typing import Dict, List, Optional
from rich.console import Console

console = Console()

class PromptEngine:
    """Manages prompt creation and optimization for style mirroring."""
    
    def __init__(self):
        self.prompt_templates = {
            'default': self._default_prompt_template,
            'formal': self._formal_prompt_template,
            'casual': self._casual_prompt_template,
            'academic': self._academic_prompt_template,
            'creative': self._creative_prompt_template
        }
    
    def create_prompt(
        self,
        style_samples: Dict[str, str],
        user_input: str,
        prompt_type: str = 'default',
        custom_instructions: Optional[str] = None
    ) -> str:
        """Create a prompt for style mirroring."""
        
        template = self.prompt_templates.get(prompt_type, self._default_prompt_template)
        
        # Get style context
        style_context = self._build_style_context(style_samples)
        
        # Create the prompt
        prompt = template(
            style_context=style_context,
            user_input=user_input,
            custom_instructions=custom_instructions
        )
        
        return prompt
    
    def _build_style_context(self, style_samples: Dict[str, str]) -> str:
        """Build the style context from samples."""
        from utils.text_utils import combine_style_samples, extract_style_summary
        
        # Combine samples
        combined_samples = combine_style_samples(style_samples)
        
        # Get style summary
        style_summary = extract_style_summary(style_samples)
        
        context_parts = [
            "Writing samples that demonstrate the target style:",
            "",
            combined_samples,
            "",
            "Style characteristics:",
            style_summary
        ]
        
        return "\n".join(context_parts)
    
    def _default_prompt_template(
        self,
        style_context: str,
        user_input: str,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Default prompt template."""
        
        prompt_parts = [
            style_context,
            "",
            "Based on the writing samples above, please write a response to the following request:",
            "",
            f"Request: {user_input}",
            "",
            "Your response should match the style, tone, vocabulary, and structure of the provided samples."
        ]
        
        if custom_instructions:
            prompt_parts.append("")
            prompt_parts.append(f"Additional instructions: {custom_instructions}")
        
        prompt_parts.append("")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def _formal_prompt_template(
        self,
        style_context: str,
        user_input: str,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Formal writing prompt template."""
        
        prompt_parts = [
            style_context,
            "",
            "Please write a formal response to the following request, maintaining professional tone and structure:",
            "",
            f"Request: {user_input}",
            "",
            "Ensure the response is well-structured, uses appropriate formal language, and maintains consistency with the provided writing samples."
        ]
        
        if custom_instructions:
            prompt_parts.append("")
            prompt_parts.append(f"Additional instructions: {custom_instructions}")
        
        prompt_parts.append("")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def _casual_prompt_template(
        self,
        style_context: str,
        user_input: str,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Casual writing prompt template."""
        
        prompt_parts = [
            style_context,
            "",
            "Please write a casual, conversational response to the following request:",
            "",
            f"Request: {user_input}",
            "",
            "Keep the tone relaxed and friendly, using natural language that matches the provided writing samples."
        ]
        
        if custom_instructions:
            prompt_parts.append("")
            prompt_parts.append(f"Additional instructions: {custom_instructions}")
        
        prompt_parts.append("")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def _academic_prompt_template(
        self,
        style_context: str,
        user_input: str,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Academic writing prompt template."""
        
        prompt_parts = [
            style_context,
            "",
            "Please write an academic response to the following request:",
            "",
            f"Request: {user_input}",
            "",
            "Use scholarly language, proper citations if needed, and maintain academic rigor while matching the style of the provided samples."
        ]
        
        if custom_instructions:
            prompt_parts.append("")
            prompt_parts.append(f"Additional instructions: {custom_instructions}")
        
        prompt_parts.append("")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def _creative_prompt_template(
        self,
        style_context: str,
        user_input: str,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Creative writing prompt template."""
        
        prompt_parts = [
            style_context,
            "",
            "Please write a creative response to the following request:",
            "",
            f"Request: {user_input}",
            "",
            "Be imaginative and expressive while maintaining the unique voice and style of the provided writing samples."
        ]
        
        if custom_instructions:
            prompt_parts.append("")
            prompt_parts.append(f"Additional instructions: {custom_instructions}")
        
        prompt_parts.append("")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def optimize_prompt_length(self, prompt: str, max_tokens: int = 4000) -> str:
        """Optimize prompt length to fit within token limits."""
        from utils.text_utils import truncate_text
        
        if len(prompt) > max_tokens * 4:  # Rough character to token conversion
            console.print("⚠️  Prompt is very long, truncating...")
            return truncate_text(prompt, max_tokens * 4)
        
        return prompt
    
    def get_available_templates(self) -> List[str]:
        """Get list of available prompt templates."""
        return list(self.prompt_templates.keys()) 