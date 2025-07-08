"""Configuration settings for InkMod."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings and configuration."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    
    # Application Configuration
    DEFAULT_STYLE_FOLDER: str = os.getenv("DEFAULT_STYLE_FOLDER", "./writing-samples")
    FEEDBACK_FILE: str = os.getenv("FEEDBACK_FILE", "./feedback.json")
    
    # Text Processing
    MAX_SAMPLE_SIZE: int = int(os.getenv("MAX_SAMPLE_SIZE", "10000"))  # characters per file
    MAX_TOTAL_SAMPLES: int = int(os.getenv("MAX_TOTAL_SAMPLES", "50000"))  # total characters
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are configured."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Security: Validate API key format
        if cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format")
        
        # Security: Validate model name
        allowed_models = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4']
        if cls.OPENAI_MODEL not in allowed_models:
            raise ValueError(f"Unsupported model: {cls.OPENAI_MODEL}")
        
        return True

# Global settings instance
settings = Settings()

def load_config() -> dict:
    """Load configuration as a dictionary for compatibility."""
    return {
        'openai': {
            'api_key': settings.OPENAI_API_KEY,
            'model': settings.OPENAI_MODEL,
            'max_tokens': settings.OPENAI_MAX_TOKENS,
            'temperature': settings.OPENAI_TEMPERATURE
        },
        'app': {
            'default_style_folder': settings.DEFAULT_STYLE_FOLDER,
            'feedback_file': settings.FEEDBACK_FILE,
            'max_sample_size': settings.MAX_SAMPLE_SIZE,
            'max_total_samples': settings.MAX_TOTAL_SAMPLES
        }
    } 