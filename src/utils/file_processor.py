"""File processing utilities for InkMod."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console
from rich.progress import Progress

console = Console()

class FileProcessor:
    """Handles file processing for style samples."""
    
    def __init__(self, max_sample_size: int = 10000, max_total_samples: int = 50000):
        self.max_sample_size = max_sample_size
        self.max_total_samples = max_total_samples
    
    def get_text_files(self, folder_path: str) -> List[str]:
        """Get all text files from a folder."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Style folder not found: {folder_path}")
        
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        # Find all text files
        text_extensions = ['*.txt', '*.md', '*.rst']
        text_files = []
        
        for ext in text_extensions:
            text_files.extend(glob.glob(str(folder / ext)))
        
        if not text_files:
            raise ValueError(f"No text files found in {folder_path}")
        
        return sorted(text_files)
    
    def read_text_file(self, file_path: str) -> str:
        """Read and validate a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if len(content) > self.max_sample_size:
                console.print(f"⚠️  File {file_path} is too large ({len(content)} chars), truncating...")
                content = content[:self.max_sample_size]
            
            return content
        except UnicodeDecodeError:
            console.print(f"⚠️  Could not read {file_path} as UTF-8, trying with different encoding...")
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read().strip()
                return content[:self.max_sample_size]
            except Exception as e:
                console.print(f"❌ Failed to read {file_path}: {e}")
                return ""
    
    def process_style_folder(self, folder_path: str) -> Dict[str, str]:
        """Process all text files in a style folder."""
        text_files = self.get_text_files(folder_path)
        samples = {}
        total_chars = 0
        
        with Progress() as progress:
            task = progress.add_task("Processing style samples...", total=len(text_files))
            
            for file_path in text_files:
                content = self.read_text_file(file_path)
                if content:
                    file_name = Path(file_path).name
                    samples[file_name] = content
                    total_chars += len(content)
                    
                    if total_chars > self.max_total_samples:
                        console.warning(f"Reached maximum sample size ({self.max_total_samples} chars), stopping...")
                        break
                
                progress.advance(task)
        
        if not samples:
            raise ValueError(f"No valid text content found in {folder_path}")
        
        console.print(f"✅ Processed {len(samples)} files ({total_chars} total characters)")
        return samples
    
    def save_output(self, content: str, output_file: str) -> None:
        """Save output to a file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            console.print(f"✅ Output saved to {output_file}")
        except Exception as e:
            console.print(f"❌ Failed to save output: {e}")
    
    def read_input_file(self, input_file: str) -> List[str]:
        """Read inputs from a file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            return lines
        except Exception as e:
            console.print(f"❌ Failed to read input file: {e}")
            return [] 