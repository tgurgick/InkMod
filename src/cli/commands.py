"""CLI commands for InkMod."""

import json
import sys
from typing import Optional, List
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text

from core.style_analyzer import StyleAnalyzer
from core.openai_client import OpenAIClient
from core.prompt_engine import PromptEngine
from utils.file_processor import FileProcessor
from config.settings import settings

console = Console()

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """InkMod - Writing Style Mirror CLI Tool"""
    pass

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--input', '-i', required=True, help='Text to generate a response for')
@click.option('--output-file', '-o', help='Save output to file')
@click.option('--temperature', '-t', default=0.7, help='OpenAI temperature (0.0-1.0)')
@click.option('--max-tokens', '-m', default=1000, help='Maximum tokens for response')
@click.option('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
@click.option('--prompt-type', default='default', help='Prompt template type')
@click.option('--show-analysis', is_flag=True, help='Show detailed style analysis')
@click.option('--edit-mode', is_flag=True, help='Enable interactive editing mode')
def generate(style_folder, input, output_file, temperature, max_tokens, model, prompt_type, show_analysis, edit_mode):
    """Generate a response that matches the provided writing style."""
    
    try:
        # Validate settings
        settings.validate()
        
        # Initialize components
        file_processor = FileProcessor()
        style_analyzer = StyleAnalyzer(file_processor)
        openai_client = OpenAIClient(model=model)
        prompt_engine = PromptEngine()
        
        # Analyze style samples
        console.print(f"📁 Analyzing style samples from: {style_folder}")
        analysis_result = style_analyzer.analyze_style_folder(style_folder)
        
        if show_analysis:
            style_analyzer.display_style_analysis(analysis_result)
        
        # Generate response
        console.print(f"\n🤖 Generating response with {model}...")
        
        result = openai_client.generate_response(
            style_samples=analysis_result['samples'],
            user_input=input,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Display result
        console.print("\n" + "="*50)
        console.print("[bold green]Generated Response:[/bold green]")
        console.print("="*50)
        console.print(result['response'])
        console.print("="*50)
        
        # Show usage info
        console.print(f"\n📊 Usage: {result['total_tokens']} tokens (${result['cost']:.4f})")
        
        # Handle edit mode
        if edit_mode:
            edited_response = _handle_edit_mode(result['response'], input, style_folder)
            if edited_response != result['response']:
                result['response'] = edited_response
                _save_feedback(input, result['response'], edited_response, style_folder)
        
        # Save to file if requested
        if output_file:
            file_processor.save_output(result['response'], output_file)
        
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--input-file', '-i', help='File containing inputs (one per line)')
@click.option('--output-file', '-o', help='Save outputs to file')
@click.option('--temperature', '-t', default=0.7, help='OpenAI temperature (0.0-1.0)')
@click.option('--max-tokens', '-m', default=1000, help='Maximum tokens for response')
@click.option('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
def batch(style_folder, input_file, output_file, temperature, max_tokens, model):
    """Process multiple inputs in batch mode."""
    
    try:
        # Validate settings
        settings.validate()
        
        # Initialize components
        file_processor = FileProcessor()
        style_analyzer = StyleAnalyzer(file_processor)
        openai_client = OpenAIClient(model=model)
        
        # Analyze style samples
        console.print(f"📁 Analyzing style samples from: {style_folder}")
        analysis_result = style_analyzer.analyze_style_folder(style_folder)
        
        # Get inputs
        if input_file:
            inputs = file_processor.read_input_file(input_file)
            if not inputs:
                console.print("No valid inputs found in file")
                return
        else:
            # Interactive input
            inputs = []
            console.print("Enter inputs (one per line, empty line to finish):")
            while True:
                user_input = Prompt.ask("Input")
                if not user_input:
                    break
                inputs.append(user_input)
        
        if not inputs:
            console.print("No inputs provided")
            return
        
        # Process inputs
        results = []
        total_cost = 0
        
        with console.status("Processing batch inputs..."):
            for i, user_input in enumerate(inputs, 1):
                console.print(f"\n[{i}/{len(inputs)}] Processing: {user_input[:50]}...")
                
                result = openai_client.generate_response(
                    style_samples=analysis_result['samples'],
                    user_input=user_input,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                results.append({
                    'input': user_input,
                    'output': result['response'],
                    'tokens': result['total_tokens'],
                    'cost': result['cost']
                })
                
                total_cost += result['cost']
        
        # Display results
        console.print(f"\n✅ Batch processing complete!")
        console.print(f"📊 Total cost: ${total_cost:.4f}")
        
        for i, result in enumerate(results, 1):
            console.print(f"\n--- Result {i} ---")
            console.print(f"Input: {result['input']}")
            console.print(f"Output: {result['output']}")
            console.print(f"Tokens: {result['tokens']} (${result['cost']:.4f})")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Input: {result['input']}\n")
                    f.write(f"Output: {result['output']}\n")
                    f.write("-" * 50 + "\n")
            console.print(f"Results saved to {output_file}")
        
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--temperature', '-t', default=0.7, help='OpenAI temperature (0.0-1.0)')
@click.option('--max-tokens', '-m', default=1000, help='Maximum tokens for response')
@click.option('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
def interactive(style_folder, temperature, max_tokens, model):
    """Start interactive mode for real-time style mirroring."""
    
    try:
        # Validate settings
        settings.validate()
        
        # Initialize components
        file_processor = FileProcessor()
        style_analyzer = StyleAnalyzer(file_processor)
        openai_client = OpenAIClient(model=model)
        
        # Analyze style samples
        console.print(f"📁 Analyzing style samples from: {style_folder}")
        analysis_result = style_analyzer.analyze_style_folder(style_folder)
        
        # Show welcome message
        welcome_text = Text("InkMod Interactive Mode", style="bold blue")
        console.print(Panel(welcome_text, title="Welcome"))
        console.print("Enter your requests and get style-matched responses.")
        console.print("Type 'quit' or 'exit' to end the session.\n")
        
        # Interactive loop
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]Your request[/bold cyan]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                # Generate response
                console.print("🤖 Generating response...")
                result = openai_client.generate_response(
                    style_samples=analysis_result['samples'],
                    user_input=user_input,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Display response
                console.print("\n[bold green]Response:[/bold green]")
                console.print(result['response'])
                console.print(f"\n📊 Tokens: {result['total_tokens']} (${result['cost']:.4f})")
                
                # Ask for feedback
                if Confirm.ask("Would you like to edit this response?"):
                    edited_response = _handle_edit_mode(result['response'], user_input, style_folder)
                    if edited_response != result['response']:
                        _save_feedback(user_input, result['response'], edited_response, style_folder)
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"Error: {e}")
                continue
        
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
def analyze(style_folder):
    """Analyze writing style samples without generating responses."""
    
    try:
        # Initialize components
        file_processor = FileProcessor()
        style_analyzer = StyleAnalyzer(file_processor)
        
        # Analyze style samples
        console.print(f"📁 Analyzing style samples from: {style_folder}")
        analysis_result = style_analyzer.analyze_style_folder(style_folder)
        
        # Display analysis
        style_analyzer.display_style_analysis(analysis_result)
        
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

def _handle_edit_mode(original_response: str, user_input: str, style_folder: str) -> str:
    """Handle interactive editing mode."""
    console.print("\n[bold yellow]Edit Mode[/bold yellow]")
    console.print("Edit the response below. Press Enter twice to finish editing.")
    console.print("Original response:")
    console.print(original_response)
    console.print("\n" + "="*50)
    
    # Get edited response
    edited_lines = []
    console.print("Enter your edited response:")
    
    while True:
        line = input()
        if line == "" and edited_lines and edited_lines[-1] == "":
            edited_lines.pop()  # Remove the last empty line
            break
        edited_lines.append(line)
    
    edited_response = "\n".join(edited_lines).strip()
    
    if edited_response == original_response:
        console.print("No changes made.")
        return original_response
    
    console.print("\n[bold green]Edited Response:[/bold green]")
    console.print(edited_response)
    
    return edited_response

def _save_feedback(original_input: str, generated_response: str, edited_response: str, style_folder: str):
    """Save feedback data for future improvements."""
    feedback_data = {
        'original_input': original_input,
        'generated_response': generated_response,
        'edited_response': edited_response,
        'style_folder': style_folder,
        'timestamp': str(Path().cwd() / 'feedback.json')
    }
    
    try:
        # Load existing feedback
        feedback_file = Path(settings.FEEDBACK_FILE)
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                existing_feedback = json.load(f)
        else:
            existing_feedback = []
        
        # Add new feedback
        existing_feedback.append(feedback_data)
        
        # Save feedback
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(existing_feedback, f, indent=2, ensure_ascii=False)
        
        console.print(f"Feedback saved to {feedback_file}")
        
    except Exception as e:
        console.print(f"Failed to save feedback: {e}")

if __name__ == '__main__':
    cli() 