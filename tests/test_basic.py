"""Basic tests for InkMod."""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.text_utils import analyze_text_style, combine_style_samples, extract_style_summary
from utils.file_processor import FileProcessor
from core.style_analyzer import StyleAnalyzer

def test_text_analysis():
    """Test basic text analysis functionality."""
    sample_text = "This is a test sentence. It has multiple sentences. We can analyze it."
    
    analysis = analyze_text_style(sample_text)
    
    assert analysis['total_words'] == 13
    assert analysis['total_sentences'] == 3
    assert analysis['avg_sentence_length'] > 0
    assert analysis['avg_word_length'] > 0

def test_combine_style_samples():
    """Test combining multiple style samples."""
    samples = {
        'file1.txt': 'This is sample one.',
        'file2.txt': 'This is sample two.'
    }
    
    combined = combine_style_samples(samples)
    
    assert 'file1.txt' in combined
    assert 'file2.txt' in combined
    assert 'sample one' in combined
    assert 'sample two' in combined

def test_file_processor():
    """Test file processor functionality."""
    processor = FileProcessor()
    
    # Test with a non-existent folder
    with pytest.raises(FileNotFoundError):
        processor.get_text_files("non_existent_folder")

def test_style_analyzer():
    """Test style analyzer functionality."""
    analyzer = StyleAnalyzer()
    
    # Test validation with empty samples
    assert not analyzer.validate_style_samples({})
    
    # Test validation with minimal samples
    minimal_samples = {'test.txt': 'This is a test.'}
    assert analyzer.validate_style_samples(minimal_samples)

if __name__ == '__main__':
    pytest.main([__file__]) 