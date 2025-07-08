"""CLI commands for InkMod."""

import json
import sys
from typing import Optional, List
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text

from core.style_analyzer import StyleAnalyzer
from core.openai_client import OpenAIClient
from core.prompt_engine import PromptEngine
from utils.file_processor import FileProcessor
from config.settings import settings, load_config

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
@click.option('--model', default=None, help='OpenAI model to use')
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
@click.option('--model', default=None, help='OpenAI model to use')
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
@click.option('--model', default=None, help='OpenAI model to use')
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

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--test-input', '-t', required=True, help='Test input to generate content for')
@click.option('--model', '-m', help='OpenAI model to use (defaults to config)')
@click.option('--temperature', '-temp', type=float, default=0.7, help='Temperature for generation')
@click.option('--max-tokens', type=int, default=500, help='Maximum tokens to generate')
def validate(style_folder: str, test_input: str, model: str, temperature: float, max_tokens: int):
    """Compare different style analysis methods and validate their effectiveness."""
    
    try:
        # Initialize components
        config = load_config()
        openai_client = OpenAIClient(
            api_key=config['openai']['api_key'],
            model=model or config['openai']['model']
        )
        
        # Load style samples
        file_processor = FileProcessor()
        samples = file_processor.process_style_folder(style_folder)
        
        if not samples:
            console.print("❌ No style samples found in the specified folder.")
            return
        
        console.print(f"📁 Loaded {len(samples)} style samples from {style_folder}")
        
        # Initialize validator
        from core.style_validator import StyleValidator
        validator = StyleValidator(openai_client)
        
        # Run comparison (pass temperature and max_tokens)
        comparison_result = validator.compare_analysis_methods(samples, test_input, temperature=temperature, max_tokens=max_tokens)
        
        # Display results
        validator.display_comparison_results(comparison_result)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(comparison_result, f, indent=2)
        
        console.print(f"\n💾 Results saved to {results_file}")
        
    except Exception as e:
        console.print(f"❌ Validation failed: {e}")
        if config.get('debug', False):
            console.print_exception()

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--test-prompts', '-t', required=True, help='File containing test prompts (one per line)')
@click.option('--iterations', '-i', type=int, default=5, help='Number of training iterations')
@click.option('--model', '-m', help='OpenAI model to use (defaults to config)')
@click.option('--output-file', '-o', help='Save training results to file')
def train(style_folder: str, test_prompts: str, iterations: int, model: str, output_file: str):
    """Train a local style model using OpenAI as teacher (reinforcement learning)."""
    
    try:
        # Initialize components
        config = load_config()
        openai_client = OpenAIClient(
            api_key=config['openai']['api_key'],
            model=model or config['openai']['model']
        )
        
        # Load style samples
        file_processor = FileProcessor()
        samples = file_processor.process_style_folder(style_folder)
        
        if not samples:
            console.print("❌ No style samples found in the specified folder.")
            return
        
        console.print(f"📁 Loaded {len(samples)} style samples from {style_folder}")
        
        # Load test prompts
        try:
            with open(test_prompts, 'r') as f:
                test_prompt_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            console.print(f"❌ Test prompts file not found: {test_prompts}")
            return
        
        if not test_prompt_list:
            console.print("❌ No test prompts found in file.")
            return
        
        console.print(f"📝 Loaded {len(test_prompt_list)} test prompts")
        
        # Initialize reinforcement trainer
        from core.reinforcement_trainer import ReinforcementTrainer
        trainer = ReinforcementTrainer(openai_client)
        
        # Start training
        results = trainer.train_with_reinforcement(samples, test_prompt_list, iterations)
        
        # Display results
        trainer.display_training_results(results)
        
        # Save results
        if output_file:
            trainer.save_training_results(results, output_file)
        else:
            trainer.save_training_results(results)
        
        # Compare final models
        console.print("\n🔍 Comparing Local vs OpenAI Models...")
        comparison = trainer.compare_models(test_prompt_list, samples)
        
        console.print(f"\n💰 Cost Comparison:")
        console.print(f"   Local Model Cost: ${comparison['comparison']['total_local_cost']:.4f}")
        console.print(f"   OpenAI Cost: ${comparison['comparison']['total_openai_cost']:.4f}")
        console.print(f"   Potential Savings: ${comparison['comparison']['cost_savings']:.4f}")
        
    except Exception as e:
        console.print(f"❌ Training failed: {e}")
        if config.get('debug', False):
            console.print_exception()

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--test-prompts', '-t', required=True, help='File containing test prompts (one per line)')
@click.option('--iterations', '-i', type=int, default=5, help='Number of training iterations')
@click.option('--model', '-m', help='OpenAI model to use (defaults to config)')
@click.option('--output-file', '-o', help='Save training results to file')
@click.option('--use-lightweight-llm', is_flag=True, default=True, help='Use lightweight LLM for generation')
@click.option('--backend', '-b', help='Local LLM backend to use (e.g., llama-7b, gpt4all-j, hf-distilgpt2)')
def train_enhanced(style_folder: str, test_prompts: str, iterations: int, model: str, output_file: str, use_lightweight_llm: bool, backend: str):
    """Train an enhanced local style model using lightweight LLM integration."""
    
    try:
        # Initialize components
        config = load_config()
        openai_client = OpenAIClient(
            api_key=config['openai']['api_key'],
            model=model or config['openai']['model']
        )
        
        # Load style samples
        file_processor = FileProcessor()
        samples = file_processor.process_style_folder(style_folder)
        
        if not samples:
            console.print("❌ No style samples found in the specified folder.")
            return
        
        console.print(f"📁 Loaded {len(samples)} style samples from {style_folder}")
        
        # Load test prompts
        try:
            with open(test_prompts, 'r') as f:
                test_prompt_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            console.print(f"❌ Test prompts file not found: {test_prompts}")
            return
        
        if not test_prompt_list:
            console.print("❌ No test prompts found in file.")
            return
        
        console.print(f"📝 Loaded {len(test_prompt_list)} test prompts")
        
        # Initialize enhanced reinforcement trainer with backend
        from core.enhanced_reinforcement_trainer import EnhancedReinforcementTrainer
        trainer = EnhancedReinforcementTrainer(openai_client, backend_name=backend)
        
        # Set backend if specified
        if backend:
            console.print(f"🔧 Setting backend to: {backend}")
            if not trainer.set_backend(backend):
                console.print(f"⚠️  Backend '{backend}' not available, using default")
                available_backends = trainer.get_available_backends()
                if available_backends:
                    console.print(f"Available backends: {', '.join(available_backends)}")
        
        # Configure lightweight LLM usage
        trainer.local_model.use_local_llm = use_lightweight_llm
        if use_lightweight_llm:
            if backend:
                console.print(f"🤖 Using local LLM backend: {backend}")
            else:
                console.print("🤖 Using lightweight LLM (GPT-3.5-turbo) for generation")
        else:
            console.print("📝 Using template-based generation")
        
        # Start enhanced training
        results = trainer.train_with_reinforcement(samples, test_prompt_list, iterations)
        
        # Display enhanced results
        trainer.display_enhanced_training_results(results)
        
        # Save results
        if output_file:
            trainer.save_enhanced_training_results(results, output_file)
        else:
            trainer.save_enhanced_training_results(results)
        
        # Compare enhanced models
        console.print("\n🔍 Comparing Enhanced Local vs OpenAI Models...")
        comparison = trainer.compare_enhanced_models(test_prompt_list, samples)
        
        console.print(f"\n💰 Enhanced Cost Comparison:")
        console.print(f"   Enhanced Local Model Cost: ${comparison['comparison']['total_local_cost']:.4f}")
        console.print(f"   OpenAI Cost: ${comparison['comparison']['total_openai_cost']:.4f}")
        console.print(f"   Potential Savings: ${comparison['comparison']['cost_savings']:.4f}")
        
        # Show sample outputs
        console.print("\n📄 Sample Enhanced Local Model Outputs:")
        for i, result in enumerate(comparison['enhanced_local'][:3], 1):
            console.print(f"\n{i}. Prompt: {result['prompt'][:50]}...")
            console.print(f"   Response: {result['response'][:100]}...")
        
    except Exception as e:
        console.print(f"❌ Enhanced training failed: {e}")
        if config.get('debug', False):
            console.print_exception()

@cli.command()
@click.option('--list', '-l', is_flag=True, help='List available backends')
@click.option('--info', '-i', help='Get information about a specific backend')
@click.option('--test', '-t', help='Test a specific backend with a sample prompt')
def backends(list: bool, info: str, test: str):
    """Manage local LLM backends."""
    
    try:
        from core.llm_backends import create_backend_manager
        
        # Create backend manager
        manager = create_backend_manager()
        
        if list:
            console.print("🔧 Available Local LLM Backends:")
            console.print("=" * 50)
            
            backends = manager.list_backends()
            if not backends:
                console.print("❌ No backends available")
                return
            
            for backend in backends:
                info = manager.get_backend_info(backend)
                status = "✅ Loaded" if info.get('is_loaded', False) else "⏳ Not loaded"
                console.print(f"  • {backend} - {status}")
            
            console.print("\n💡 To use a backend, specify it with --backend option in train commands")
            console.print("   Example: inkmod train-enhanced --backend llama-7b")
        
        elif info:
            backend_info = manager.get_backend_info(info)
            if 'error' in backend_info:
                console.print(f"❌ {backend_info['error']}")
                return
            
            console.print(f"🔧 Backend Information: {info}")
            console.print("=" * 50)
            console.print(f"Type: {backend_info.get('backend', 'Unknown')}")
            console.print(f"Model Path: {backend_info.get('model_path', 'Unknown')}")
            console.print(f"Status: {'✅ Loaded' if backend_info.get('is_loaded', False) else '⏳ Not loaded'}")
            console.print(f"Config: {backend_info.get('config', {})}")
        
        elif test:
            if not manager.set_backend(test):
                console.print(f"❌ Backend '{test}' not found")
                return
            
            console.print(f"🧪 Testing backend: {test}")
            console.print("=" * 50)
            
            test_prompt = "Write a short email about a meeting tomorrow."
            console.print(f"Test prompt: {test_prompt}")
            
            response = manager.generate(test_prompt, max_tokens=100, temperature=0.7)
            console.print(f"Response: {response}")
        
        else:
            console.print("🔧 Backend Management")
            console.print("Use --list to see available backends")
            console.print("Use --info <backend> to get backend information")
            console.print("Use --test <backend> to test a backend")
    
    except Exception as e:
        console.print(f"❌ Backend management failed: {e}")
        if config.get('debug', False):
            console.print_exception()

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