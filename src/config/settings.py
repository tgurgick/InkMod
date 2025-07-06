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
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
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
        return True

# Global settings instance
settings = Settings() 