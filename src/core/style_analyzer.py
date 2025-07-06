"""Style analysis component for InkMod."""

from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table

from utils.text_utils import analyze_text_style, extract_style_summary
from utils.file_processor import FileProcessor

console = Console()

class StyleAnalyzer:
    """Analyzes writing style from text samples."""
    
    def __init__(self, file_processor: Optional[FileProcessor] = None):
        self.file_processor = file_processor or FileProcessor()
    
    def analyze_style_folder(self, folder_path: str) -> Dict[str, any]:
        """Analyze the writing style from a folder of text files."""
        try:
            # Process the style samples
            samples = self.file_processor.process_style_folder(folder_path)
            
            # Analyze each sample
            analyses = {}
            for filename, content in samples.items():
                analysis = analyze_text_style(content)
                if analysis:
                    analyses[filename] = analysis
            
            # Generate overall style summary
            style_summary = extract_style_summary(samples)
            
            return {
                'samples': samples,
                'analyses': analyses,
                'style_summary': style_summary,
                'total_files': len(samples),
                'total_characters': sum(len(content) for content in samples.values())
            }
            
        except Exception as e:
            console.print(f"❌ Error analyzing style folder: {e}")
            raise
    
    def display_style_analysis(self, analysis_result: Dict[str, any]) -> None:
        """Display the style analysis results."""
        console.print("\n[bold blue]Style Analysis Results[/bold blue]")
        console.print("=" * 50)
        
        # Basic stats
        console.print(f"📁 Files analyzed: {analysis_result['total_files']}")
        console.print(f"📊 Total characters: {analysis_result['total_characters']:,}")
        console.print()
        
        # Style summary
        if analysis_result['style_summary']:
            console.print("[bold]Style Characteristics:[/bold]")
            console.print(analysis_result['style_summary'])
            console.print()
        
        # Detailed analysis table
        if analysis_result['analyses']:
            table = Table(title="Detailed File Analysis")
            table.add_column("File", style="cyan")
            table.add_column("Words", justify="right")
            table.add_column("Sentences", justify="right")
            table.add_column("Avg Sentence Length", justify="right")
            table.add_column("Avg Word Length", justify="right")
            table.add_column("Vocabulary Diversity", justify="right")
            
            for filename, analysis in analysis_result['analyses'].items():
                table.add_row(
                    filename,
                    str(analysis['total_words']),
                    str(analysis['total_sentences']),
                    f"{analysis['avg_sentence_length']:.1f}",
                    f"{analysis['avg_word_length']:.1f}",
                    f"{analysis['vocabulary_diversity']:.3f}"
                )
            
            console.print(table)
            console.print()
        
        # Most common words across all samples
        all_words = []
        for analysis in analysis_result['analyses'].values():
            all_words.extend([word for word, _ in analysis['most_common_words'][:5]])
        
        if all_words:
            from collections import Counter
            word_counts = Counter(all_words)
            console.print("[bold]Most Common Words:[/bold]")
            for word, count in word_counts.most_common(10):
                console.print(f"  • {word}: {count} times")
            console.print()
    
    def validate_style_samples(self, samples: Dict[str, str]) -> bool:
        """Validate that style samples are sufficient for analysis."""
        if not samples:
            console.print("❌ No style samples provided")
            return False
        
        total_chars = sum(len(content) for content in samples.values())
        if total_chars < 100:
            console.print("⚠️  Very little text provided - style analysis may be limited")
            return False
        
        if len(samples) < 2:
            console.print("⚠️  Only one sample provided - consider adding more for better style analysis")
        
        return True
    
    def get_style_context(self, analysis_result: Dict[str, any]) -> str:
        """Get a formatted context string for the style."""
        context_parts = [
            f"Style Analysis:",
            f"- {analysis_result['total_files']} files analyzed",
            f"- {analysis_result['total_characters']:,} total characters",
            "",
            analysis_result['style_summary']
        ]
        
        return "\n".join(context_parts) 