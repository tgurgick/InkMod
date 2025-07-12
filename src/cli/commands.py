"""CLI commands for InkMod."""

import json
import os
import sys
from typing import Optional, List, Dict
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
@click.option('--incremental/--fresh', default=True, help='Use incremental training (default) or start fresh')
def train_enhanced(style_folder: str, test_prompts: str, iterations: int, model: str, output_file: str, use_lightweight_llm: bool, backend: str, incremental: bool):
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
        results = trainer.train_with_reinforcement(samples, test_prompt_list, iterations, incremental)
        
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
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--test-prompts', '-t', required=True, help='File containing test prompts (one per line)')
@click.option('--iterations', '-i', type=int, default=5, help='Number of training iterations')
@click.option('--model', '-m', help='OpenAI model to use (defaults to config)')
@click.option('--output-file', '-o', help='Save training results to file')
@click.option('--use-lightweight-llm', is_flag=True, help='Use lightweight LLM for generation')
@click.option('--backend', '-b', help='Local LLM backend to use (e.g., llama-7b, gpt4all-j, hf-distilgpt2)')
@click.option('--incremental/--fresh', default=True, help='Use incremental training (default) or start fresh')
@click.option('--no-llm-feedback', is_flag=True, help='Disable LLM qualitative feedback (use only standard metrics)')
def train_hybrid(style_folder: str, test_prompts: str, iterations: int, model: str, output_file: str, use_lightweight_llm: bool, backend: str, incremental: bool, no_llm_feedback: bool):
    """Train an enhanced local style model using hybrid reward function (standard NLP metrics + LLM feedback)."""
    
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
        
        # Initialize hybrid reinforcement trainer
        from core.hybrid_reinforcement_trainer import HybridReinforcementTrainer
        trainer = HybridReinforcementTrainer(openai_client, backend_name=backend)
        
        # Set backend if specified
        if backend:
            trainer.local_model.set_backend(backend)
        
        # Start hybrid training
        results = trainer.train_with_hybrid_reinforcement(
            samples=samples,
            test_prompts=test_prompt_list,
            iterations=iterations,
            incremental=incremental,
            use_llm_feedback=not no_llm_feedback
        )
        
        # Display results
        trainer.display_hybrid_training_results(results)
        
        # Save results
        if output_file:
            trainer.save_hybrid_training_results(results, output_file)
        else:
            trainer.save_hybrid_training_results(results)
        
        console.print("\n✅ Hybrid training complete!")
        
    except Exception as e:
        console.print(f"❌ Hybrid training failed: {e}")
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

@cli.command()
@click.option('--model-path', '-m', default='enhanced_style_model.pkl', help='Path to the trained model')
def learning_progress(model_path: str):
    """Show continuous learning progress and model statistics."""
    
    try:
        from core.enhanced_style_model import EnhancedStyleModel
        
        # Load the model
        model = EnhancedStyleModel(model_path)
        
        if not model.load_model():
            console.print(f"❌ Model not found: {model_path}")
            return
        
        # Get learning summary
        summary = model.get_learning_summary()
        
        console.print("📊 Continuous Learning Progress")
        console.print("=" * 50)
        
        # Basic statistics
        console.print(f"📈 Total Training Sessions: {summary['total_sessions']}")
        console.print(f"📝 Total Samples Processed: {summary['total_samples']}")
        console.print(f"📚 Current Vocabulary Size: {summary['current_vocabulary_size']}")
        console.print(f"📄 Current Samples Count: {summary['current_samples_count']}")
        
        # Best scores
        console.print(f"\n🏆 Best Performance Scores:")
        console.print(f"   Style Score: {summary['best_scores']['style']:.3f}")
        console.print(f"   Tone Score: {summary['best_scores']['tone']:.3f}")
        console.print(f"   Structure Score: {summary['best_scores']['structure']:.3f}")
        
        # Convergence analysis
        convergence = summary['convergence_status']
        console.print(f"\n🔄 Convergence Status:")
        console.print(f"   Converged: {'✅ Yes' if convergence['converged'] else '❌ No'}")
        console.print(f"   Confidence: {convergence['confidence']}")
        if 'variance' in convergence:
            console.print(f"   Variance: {convergence['variance']:.4f}")
        
        # Performance trend
        trend = summary['performance_trend']
        console.print(f"\n📈 Performance Trend:")
        console.print(f"   Trend: {trend['trend']}")
        console.print(f"   Improvement Rate: {trend['improvement_rate']:.3f}")
        
        # Training recommendations
        console.print(f"\n💡 Training Recommendations:")
        if convergence['converged']:
            console.print("   ✅ Model has converged - consider stopping training")
        elif trend['trend'] == 'improving':
            console.print("   🔄 Model is still improving - continue training")
        elif trend['trend'] == 'stable':
            console.print("   ⚖️  Model is stable - try more iterations or new data")
        elif trend['trend'] == 'declining':
            console.print("   ⚠️  Performance declining - review training data")
        else:
            console.print("   📊 Insufficient data - continue training")
        
        # Last improvement
        if summary['last_improvement']:
            console.        print(f"\n🕒 Last Improvement: {summary['last_improvement']}")
        
    except Exception as e:
        console.print(f"❌ Learning progress analysis failed: {e}")
        if config.get('debug', False):
            console.print_exception()

@cli.command()
@click.option('--model-path', '-m', default='enhanced_style_model.pkl', help='Path to the trained model')
@click.option('--detailed/--summary', default=False, help='Show detailed analysis (default: summary)')
@click.option('--export-json', help='Export model info to JSON file')
def explore(model_path: str, detailed: bool, export_json: str):
    """Explore and analyze a trained model in detail."""
    
    try:
        from core.enhanced_style_model import EnhancedStyleModel
        
        # Load the model
        model = EnhancedStyleModel(model_path)
        
        if not model.load_model():
            console.print(f"❌ Model not found: {model_path}")
            return
        
        # Get model info
        model_info = model.get_model_info()
        learning_summary = model.get_learning_summary()
        
        console.print("🔍 MODEL EXPLORER")
        console.print("=" * 50)
        
        # Basic model information
        console.print(f"📁 Model Path: {model_path}")
        console.print(f"📊 File Size: {os.path.getsize(model_path)} bytes")
        
        # Core statistics
        console.print(f"\n📈 CORE STATISTICS")
        console.print("-" * 30)
        console.print(f"Vocabulary Size: {model_info['vocabulary_size']} words")
        console.print(f"Sentence Patterns: {model_info['sentence_patterns']}")
        console.print(f"Paragraph Patterns: {model_info['paragraph_patterns']}")
        console.print(f"Common Phrases: {model_info['common_phrases']}")
        console.print(f"Training Sessions: {model_info['training_sessions']}")
        console.print(f"Performance Records: {model_info['performance_metrics_count']}")
        
        # Tone analysis
        console.print(f"\n🎭 TONE ANALYSIS")
        console.print("-" * 30)
        tone_markers = model_info['tone_markers']
        total_markers = sum(tone_markers.values())
        console.print(f"Total Tone Markers: {total_markers}")
        for tone, count in tone_markers.items():
            percentage = (count / total_markers * 100) if total_markers > 0 else 0
            console.print(f"  {tone.capitalize()}: {count} markers ({percentage:.1f}%)")
        
        # Learning progress
        console.print(f"\n🧠 LEARNING PROGRESS")
        console.print("-" * 30)
        console.print(f"Total Sessions: {learning_summary['total_sessions']}")
        console.print(f"Total Samples: {learning_summary['total_samples']}")
        
        best_scores = learning_summary['best_scores']
        console.print(f"\nBest Performance Scores:")
        console.print(f"  Style: {best_scores['style']:.3f}")
        console.print(f"  Tone: {best_scores['tone']:.3f}")
        console.print(f"  Structure: {best_scores['structure']:.3f}")
        
        # Convergence analysis
        convergence = learning_summary['convergence_status']
        console.print(f"\nConvergence Status:")
        console.print(f"  Converged: {'✅ Yes' if convergence['converged'] else '❌ No'}")
        console.print(f"  Confidence: {convergence['confidence']}")
        if 'variance' in convergence:
            console.print(f"  Variance: {convergence['variance']:.6f}")
        
        # Performance trend
        trend = learning_summary['performance_trend']
        console.print(f"\nPerformance Trend:")
        console.print(f"  Trend: {trend['trend']}")
        console.print(f"  Improvement Rate: {trend['improvement_rate']:.3f}")
        
        if detailed:
            # Detailed analysis
            console.print(f"\n📊 DETAILED ANALYSIS")
            console.print("-" * 30)
            
            # Vocabulary analysis
            vocabulary = model.vocabulary
            top_words = vocabulary.most_common(10)
            console.print(f"\nTop 10 Most Frequent Words:")
            for word, count in top_words:
                console.print(f"  '{word}': {count} times")
            
            # Word length analysis
            word_lengths = [len(word) for word in vocabulary.keys()]
            avg_length = sum(word_lengths) / len(word_lengths)
            console.print(f"\nVocabulary Analysis:")
            console.print(f"  Average word length: {avg_length:.2f} characters")
            console.print(f"  Shortest word: {min(word_lengths)} characters")
            console.print(f"  Longest word: {max(word_lengths)} characters")
            
            # Sentence pattern analysis
            if model.sentence_patterns:
                lengths = [p['length'] for p in model.sentence_patterns]
                avg_length = sum(lengths) / len(lengths)
                console.print(f"\nSentence Pattern Analysis:")
                console.print(f"  Average sentence length: {avg_length:.1f} words")
                console.print(f"  Min sentence length: {min(lengths)} words")
                console.print(f"  Max sentence length: {max(lengths)} words")
            
            # Training history details
            console.print(f"\n📈 TRAINING HISTORY DETAILS")
            console.print("-" * 30)
            for i, session in enumerate(model.training_history, 1):
                console.print(f"\nSession {i}:")
                console.print(f"  Timestamp: {session.get('timestamp', 'Unknown')}")
                console.print(f"  Samples processed: {session.get('samples_count', 0)}")
                console.print(f"  Total characters: {session.get('total_characters', 0)}")
                console.print(f"  Incremental: {session.get('incremental', False)}")
                console.print(f"  Previous vocab size: {session.get('previous_vocabulary_size', 0)}")
            
            # Performance metrics details
            if model.performance_metrics:
                console.print(f"\n📊 PERFORMANCE METRICS DETAILS")
                console.print("-" * 30)
                for i, metric in enumerate(model.performance_metrics, 1):
                    console.print(f"\nIteration {i}:")
                    console.print(f"  Overall score: {metric.get('overall_score', 0):.3f}")
                    console.print(f"  Tone score: {metric.get('tone_score', 0):.3f}")
                    console.print(f"  Structure score: {metric.get('structure_score', 0):.3f}")
                    console.print(f"  Cost: ${metric.get('total_cost', 0):.4f}")
                    if 'timestamp' in metric:
                        console.print(f"  Timestamp: {metric['timestamp']}")
            
            # Style characteristics
            if model.style_characteristics:
                console.print(f"\n🎨 STYLE CHARACTERISTICS")
                console.print("-" * 30)
                for key, value in model.style_characteristics.items():
                    if isinstance(value, float):
                        console.print(f"  {key}: {value:.3f}")
                    else:
                        console.print(f"  {key}: {value}")
            
            # Sample common phrases
            if model.common_phrases:
                console.print(f"\n💬 SAMPLE COMMON PHRASES")
                console.print("-" * 30)
                for phrase in model.common_phrases[:10]:
                    console.print(f"  '{phrase}'")
            
            # Sample training data
            if model.training_samples:
                console.print(f"\n📖 SAMPLE TRAINING DATA")
                console.print("-" * 30)
                for i, sample in enumerate(model.training_samples[:2], 1):
                    preview = sample[:200] + "..." if len(sample) > 200 else sample
                    console.print(f"  Sample {i}: {preview}")
        
        # Export to JSON if requested
        if export_json:
            try:
                import json
                
                # Prepare data for JSON export
                export_data = {
                    'model_info': model_info,
                    'learning_summary': learning_summary,
                    'vocabulary_size': len(model.vocabulary),
                    'top_words': dict(model.vocabulary.most_common(20)),
                    'tone_markers': model.tone_markers,
                    'training_history': model.training_history,
                    'performance_metrics': model.performance_metrics,
                    'style_characteristics': model.style_characteristics,
                    'llm_config': model.llm_config
                }
                
                with open(export_json, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                console.print(f"\n💾 Model data exported to: {export_json}")
                
            except Exception as e:
                console.print(f"❌ Failed to export JSON: {e}")
        
        # Training recommendations
        console.print(f"\n💡 TRAINING RECOMMENDATIONS")
        console.print("-" * 30)
        
        if convergence['converged']:
            console.print("  ✅ Model has converged - consider stopping training")
        elif trend['trend'] == 'improving':
            console.print("  🔄 Model is still improving - continue training")
        elif trend['trend'] == 'stable':
            console.print("  ⚖️  Model is stable - try more iterations or new data")
        elif trend['trend'] == 'declining':
            console.print("  ⚠️  Performance declining - review training data")
        else:
            console.print("  📊 Insufficient data - continue training")
        
        if learning_summary['total_sessions'] < 3:
            console.print("  📈 Consider more training sessions for better performance")
        
        console.print(f"\n✅ Model exploration complete!")
        
    except Exception as e:
        console.print(f"❌ Model exploration failed: {e}")
        if settings.DEBUG:
            console.print_exception()

@cli.command()
@click.option('--style-folder', '-s', required=True, help='Folder containing writing style samples')
@click.option('--input', '-i', required=True, help='Text to generate a response for')
@click.option('--model', default=None, help='OpenAI model to use (defaults to config)')
@click.option('--temperature', '-t', default=0.7, help='OpenAI temperature (0.0-1.0)')
@click.option('--max-tokens', '-m', default=1000, help='Maximum tokens for response')
@click.option('--output-file', '-o', help='Save final output to file')
def redline(style_folder: str, input: str, model: str, temperature: float, max_tokens: int, output_file: str):
    """Generate content and allow sentence-by-sentence redlining with feedback capture."""
    
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
        
        # Generate initial response
        console.print(f"\n🤖 Generating response with {model}...")
        
        result = openai_client.generate_response(
            style_samples=analysis_result['samples'],
            user_input=input,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        original_response = result['response']
        
        # Start redline process
        final_response = _handle_redline_mode(original_response, input, style_folder)
        
        # Display final result
        console.print("\n" + "="*50)
        console.print("[bold green]Final Response:[/bold green]")
        console.print("="*50)
        console.print(final_response)
        console.print("="*50)
        
        # Show usage info
        console.print(f"\n📊 Usage: {result['total_tokens']} tokens (${result['cost']:.4f})")
        
        # Save to file if requested
        if output_file:
            file_processor.save_output(final_response, output_file)
            console.print(f"💾 Final response saved to: {output_file}")
        
    except Exception as e:
        console.print(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--model-path', '-m', default='enhanced_style_model.pkl', help='Path to the trained model')
@click.option('--feedback-file', '-f', default=None, help='Path to feedback file (defaults to settings)')
@click.option('--apply-all', is_flag=True, help='Apply all feedback entries (default: prompt for each)')
def apply_feedback(model_path: str, feedback_file: str, apply_all: bool):
    """Apply redline feedback to the local model for training."""
    
    try:
        # Validate settings
        settings.validate()
        
        # Use default feedback file if not specified
        if not feedback_file:
            feedback_file = settings.FEEDBACK_FILE
        
        feedback_path = Path(feedback_file)
        if not feedback_path.exists():
            console.print(f"❌ Feedback file not found: {feedback_file}")
            return
        
        # Load feedback data
        with open(feedback_path, 'r', encoding='utf-8') as f:
            feedback_data = json.load(f)
        
        if not feedback_data:
            console.print("❌ No feedback data found in file.")
            return
        
        # Filter for redline feedback
        redline_feedback = [f for f in feedback_data if f.get('feedback_type') == 'redline_revisions']
        
        if not redline_feedback:
            console.print("❌ No redline feedback found in file.")
            return
        
        console.print(f"📊 Found {len(redline_feedback)} redline feedback entries")
        
        # Load the local model
        from core.enhanced_style_model import EnhancedStyleModel
        model = EnhancedStyleModel(model_path)
        
        # Process each feedback entry
        applied_count = 0
        for i, feedback in enumerate(redline_feedback, 1):
            console.print(f"\n[bold]Feedback Entry {i}:[/bold]")
            console.print(f"  Original Input: {feedback['original_input'][:100]}...")
            console.print(f"  Revisions: {feedback['total_revisions']} sentence changes")
            
            if not apply_all:
                if not Confirm.ask("Apply this feedback to the model?"):
                    continue
            
            # Extract feedback pairs
            feedback_pairs = feedback.get('feedback_pairs', [])
            
            # Update model with feedback
            model.update_from_redline_feedback(feedback_pairs)
            applied_count += 1
            
            console.print(f"  ✅ Applied {len(feedback_pairs)} revision pairs")
        
        # Save updated model
        model.save_model()
        
        console.print(f"\n✅ Successfully applied {applied_count} feedback entries to model")
        console.print(f"💾 Updated model saved to: {model_path}")
        
        # Show learning progress
        learning_summary = model.get_learning_summary()
        console.print(f"\n📈 Learning Progress:")
        console.print(f"  Total sessions: {learning_summary['total_sessions']}")
        console.print(f"  Best style score: {learning_summary['best_scores']['style']:.3f}")
        console.print(f"  Best tone score: {learning_summary['best_scores']['tone']:.3f}")
        
    except Exception as e:
        console.print(f"❌ Error applying feedback: {e}")
        if settings.DEBUG:
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

def _handle_redline_mode(original_response: str, user_input: str, style_folder: str) -> str:
    """Handle sentence-by-sentence redlining mode with feedback capture."""
    
    import re
    
    # Split response into sentences
    sentences = re.split(r'(?<=[.!?])\s+', original_response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    console.print("\n[bold yellow]🔴 Redline Mode[/bold yellow]")
    console.print("Review the generated content sentence by sentence.")
    console.print("\n" + "="*50)
    
    # Display numbered sentences
    console.print("[bold]Generated Content:[/bold]")
    for i, sentence in enumerate(sentences, 1):
        console.print(f"Line {i}: {sentence}")
    
    # Store feedback pairs
    feedback_pairs = []
    current_sentences = sentences.copy()
    
    while True:
        console.print(f"\n[bold cyan]Redline Commands:[/bold cyan] <number> | save | quit | show | back")
        command = Prompt.ask("Command").strip().lower()
        
        if command == 'quit':
            if Confirm.ask("Exit without saving changes?"):
                console.print("❌ Exiting without saving changes.")
                return original_response
            continue
            
        elif command == 'save':
            if feedback_pairs:
                _save_redline_feedback(user_input, original_response, current_sentences, feedback_pairs, style_folder)
                console.print("✅ Changes saved and feedback captured for training!")
            else:
                console.print("ℹ️  No changes made.")
            break
            
        elif command == 'show':
            console.print("\n[bold]Current Content:[/bold]")
            for i, sentence in enumerate(current_sentences, 1):
                console.print(f"Line {i}: {sentence}")
            continue
            
        elif command == 'back':
            console.print("🔄 Back to main menu")
            continue
            
        elif command.isdigit():
            try:
                line_num = int(command)
                if 1 <= line_num <= len(current_sentences):
                    original_sentence = current_sentences[line_num - 1]
                    console.print(f"\n[bold]Editing Line {line_num}:[/bold]")
                    console.print(f"Original: {original_sentence}")
                    
                    # Get new sentence with back option
                    new_sentence = Prompt.ask("New version (or 'back' to cancel)")
                    
                    if new_sentence.strip().lower() == 'back':
                        console.print("🔄 Cancelled editing - back to main menu")
                        continue
                    elif new_sentence.strip() != original_sentence:
                        # Store feedback pair
                        feedback_pairs.append({
                            'line_number': line_num,
                            'before': original_sentence,
                            'after': new_sentence.strip(),
                            'feedback_type': 'sentence_revision'
                        })
                        
                        # Update current sentences
                        current_sentences[line_num - 1] = new_sentence.strip()
                        
                        console.print(f"✅ Line {line_num} updated!")
                        
                        # Show updated content
                        console.print(f"\n[bold]Updated Line {line_num}:[/bold]")
                        console.print(f"New: {new_sentence.strip()}")
                    else:
                        console.print("ℹ️  No changes made to this line.")
                else:
                    console.print(f"❌ Invalid line number. Valid range: 1-{len(current_sentences)}")
            except ValueError:
                console.print("❌ Invalid line number format.")
            continue
            
        else:
            console.print("❌ Unknown command. Use: <number> | save | quit | show | back")
            continue
    
    # Return final content
    return " ".join(current_sentences)

def _save_redline_feedback(user_input: str, original_response: str, final_sentences: List[str], feedback_pairs: List[Dict], style_folder: str):
    """Save redline feedback for training purposes."""
    
    feedback_data = {
        'original_input': user_input,
        'original_response': original_response,
        'final_response': " ".join(final_sentences),
        'style_folder': style_folder,
        'timestamp': datetime.now().isoformat(),
        'feedback_pairs': feedback_pairs,
        'total_revisions': len(feedback_pairs),
        'feedback_type': 'redline_revisions'
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
        
        console.print(f"💾 Redline feedback saved to {feedback_file}")
        console.print(f"📊 Captured {len(feedback_pairs)} revision pairs for training")
        
        # Show summary of changes
        if feedback_pairs:
            console.print(f"\n[bold]Revision Summary:[/bold]")
            for pair in feedback_pairs:
                console.print(f"  Line {pair['line_number']}: '{pair['before'][:50]}...' → '{pair['after'][:50]}...'")
        
    except Exception as e:
        console.print(f"❌ Failed to save redline feedback: {e}")

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