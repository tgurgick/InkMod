"""Modular LLM backend system for local generation."""

import os
import json
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from rich.console import Console
from pathlib import Path

console = Console()

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        # Security: Validate model path
        self._validate_model_path(model_path)
        self.model_path = model_path
        self.config = config
        self.model = None
        self.is_loaded = False
    
    def _validate_model_path(self, model_path: str) -> None:
        """Validate model path for security."""
        if not model_path:
            raise ValueError("Model path cannot be empty")
        
        # For local files, ensure they're in expected directories
        if not model_path.startswith(('http://', 'https://', 'distilgpt2', 'TinyLlama/')):
            path = Path(model_path)
            if path.exists() and not path.is_relative_to(Path.cwd()):
                raise ValueError(f"Model path {model_path} is outside current working directory")
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass

class LlamaCppBackend(LLMBackend):
    """Llama.cpp backend for local LLM generation."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.llm = None
    
    def load_model(self) -> bool:
        """Load Llama.cpp model."""
        try:
            from llama_cpp import Llama
            
            console.print(f"🦙 Loading Llama.cpp model: {self.model_path}")
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.config.get('context_length', 2048),
                n_threads=self.config.get('threads', 4),
                n_gpu_layers=self.config.get('gpu_layers', 0)
            )
            
            self.is_loaded = True
            console.print("✅ Llama.cpp model loaded successfully")
            return True
            
        except ImportError:
            console.print("❌ llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            return False
        except Exception as e:
            console.print(f"❌ Failed to load Llama.cpp model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text using Llama.cpp."""
        if not self.is_loaded:
            if not self.load_model():
                return "Model not loaded"
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n", "Human:", "Assistant:"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            console.print(f"❌ Llama.cpp generation failed: {e}")
            return "Generation failed"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Llama.cpp model information."""
        return {
            'backend': 'llama.cpp',
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'config': self.config
        }

class GPT4AllBackend(LLMBackend):
    """GPT4All backend for local LLM generation."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.model = None
    
    def load_model(self) -> bool:
        """Load GPT4All model."""
        try:
            from gpt4all import GPT4All
            
            console.print(f"🤖 Loading GPT4All model: {self.model_path}")
            
            self.model = GPT4All(
                model_name=self.model_path,
                model_path=self.config.get('model_path', ''),
                allow_download=self.config.get('allow_download', True)
            )
            
            self.is_loaded = True
            console.print("✅ GPT4All model loaded successfully")
            return True
            
        except ImportError:
            console.print("❌ gpt4all not installed. Install with: pip install gpt4all")
            return False
        except Exception as e:
            console.print(f"❌ Failed to load GPT4All model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text using GPT4All."""
        if not self.is_loaded:
            if not self.load_model():
                return "Model not loaded"
        
        try:
            response = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_k=self.config.get('top_k', 40),
                top_p=self.config.get('top_p', 0.9)
            )
            
            return response.strip()
            
        except Exception as e:
            console.print(f"❌ GPT4All generation failed: {e}")
            return "Generation failed"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get GPT4All model information."""
        return {
            'backend': 'gpt4all',
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'config': self.config
        }

class HuggingFaceBackend(LLMBackend):
    """HuggingFace Transformers backend for local LLM generation."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.tokenizer = None
        self.model = None
    
    def load_model(self) -> bool:
        """Load HuggingFace model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            console.print(f"🤗 Loading HuggingFace model: {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.config.get('use_half_precision', True) else torch.float32,
                device_map=self.config.get('device_map', None)
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            console.print("✅ HuggingFace model loaded successfully")
            return True
            
        except ImportError:
            console.print("❌ transformers not installed. Install with: pip install transformers torch")
            return False
        except Exception as e:
            console.print(f"❌ Failed to load HuggingFace model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text using HuggingFace Transformers."""
        if not self.is_loaded:
            if not self.load_model():
                return "Model not loaded"
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            console.print(f"❌ HuggingFace generation failed: {e}")
            return "Generation failed"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace model information."""
        return {
            'backend': 'huggingface',
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'config': self.config
        }

class LLMBackendManager:
    """Manager for multiple LLM backends."""
    
    def __init__(self):
        self.backends = {}
        self.current_backend = None
    
    def register_backend(self, name: str, backend: LLMBackend) -> None:
        """Register a backend."""
        self.backends[name] = backend
        console.print(f"📝 Registered backend: {name}")
    
    def set_backend(self, name: str) -> bool:
        """Set the current backend."""
        if name not in self.backends:
            console.print(f"❌ Backend '{name}' not found. Available: {list(self.backends.keys())}")
            return False
        
        self.current_backend = name
        console.print(f"✅ Set current backend to: {name}")
        return True
    
    def get_current_backend(self) -> Optional[LLMBackend]:
        """Get the current backend."""
        if self.current_backend is None:
            return None
        return self.backends.get(self.current_backend)
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text using the current backend."""
        backend = self.get_current_backend()
        if backend is None:
            return "No backend selected"
        
        return backend.generate(prompt, max_tokens, temperature)
    
    def list_backends(self) -> List[str]:
        """List available backends."""
        return list(self.backends.keys())
    
    def get_backend_info(self, name: str = None) -> Dict[str, Any]:
        """Get information about a backend."""
        if name is None:
            name = self.current_backend
        
        if name not in self.backends:
            return {'error': f'Backend {name} not found'}
        
        return self.backends[name].get_model_info()

# Predefined model configurations
MODEL_CONFIGS = {
    'llama': {
        'llama-7b': {
            'model_path': 'models/llama-7b.ggmlv3.q4_0.bin',
            'config': {
                'context_length': 2048,
                'threads': 4,
                'gpu_layers': 0
            }
        },
        'llama-13b': {
            'model_path': 'models/llama-13b.ggmlv3.q4_0.bin',
            'config': {
                'context_length': 2048,
                'threads': 8,
                'gpu_layers': 0
            }
        }
    },
    'gpt4all': {
        'gpt4all-j': {
            'model_path': 'ggml-gpt4all-j-v1.3-groovy.bin',
            'config': {
                'allow_download': True,
                'top_k': 40,
                'top_p': 0.9
            }
        },
        'mpt-7b': {
            'model_path': 'mpt-7b-instruct.ggmlv3.q4_0.bin',
            'config': {
                'allow_download': True,
                'top_k': 40,
                'top_p': 0.9
            }
        }
    },
    'huggingface': {
        'distilgpt2': {
            'model_path': 'distilgpt2',
            'config': {
                'use_half_precision': True,
                'device_map': None
            }
        },
        'tiny-llama': {
            'model_path': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'config': {
                'use_half_precision': True,
                'device_map': None
            }
        }
    }
}

def create_backend_manager() -> LLMBackendManager:
    """Create and configure a backend manager with available models."""
    manager = LLMBackendManager()
    
    # Add Llama.cpp backends
    for model_name, config in MODEL_CONFIGS['llama'].items():
        backend = LlamaCppBackend(config['model_path'], config['config'])
        manager.register_backend(f"llama-{model_name}", backend)
    
    # Add GPT4All backends
    for model_name, config in MODEL_CONFIGS['gpt4all'].items():
        backend = GPT4AllBackend(config['model_path'], config['config'])
        manager.register_backend(f"gpt4all-{model_name}", backend)
    
    # Add HuggingFace backends
    for model_name, config in MODEL_CONFIGS['huggingface'].items():
        backend = HuggingFaceBackend(config['model_path'], config['config'])
        manager.register_backend(f"hf-{model_name}", backend)
    
    return manager 