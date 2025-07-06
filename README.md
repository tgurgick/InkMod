# InkMod - Writing Style Mirror CLI Tool

**🚧 Work in Progress - MVP Version** 

InkMod is an open-source CLI tool that analyzes a user's writing style from provided examples and generates responses that match that style using OpenAI's API. This is currently in MVP (Minimum Viable Product) phase with core functionality implemented.

## Features

- **Style Analysis**: Analyze writing samples to understand tone, vocabulary, and structure
- **Style Mirroring**: Generate responses that match the analyzed writing style
- **Interactive Mode**: Real-time style mirroring with immediate feedback
- **Batch Processing**: Process multiple inputs at once
- **Feedback System**: Capture user edits to improve future generations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/inkmod.git
cd inkmod
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

**Note**: You'll need a valid OpenAI API key to use the generation features. The analyze command works without an API key.

## Usage

### Basic Usage

Generate a response in a specific style:
```bash
inkmod --style-folder ./writing-samples --input "Write a professional email response"
```

### Interactive Mode

Start an interactive session:
```bash
inkmod --style-folder ./writing-samples --interactive
```

### Batch Processing

Process multiple inputs from a file:
```bash
inkmod --style-folder ./writing-samples --input-file inputs.txt --output-file outputs.txt
```

### Edit Mode (with feedback)

Generate and edit responses while capturing feedback:
```bash
inkmod --style-folder ./writing-samples --input "Write a blog post" --edit-mode
```

## Writing Samples

Create a folder with text files containing writing samples. The tool will analyze these files to understand the writing style.

Example folder structure:
```
writing-samples/
├── email1.txt
├── email2.txt
├── blog-post1.txt
└── blog-post2.txt
```

## Configuration

You can configure various parameters:

```bash
# Set model parameters
inkmod --style-folder ./samples --input "Write something" --temperature 0.7 --max-tokens 500

# Use different OpenAI model
inkmod --style-folder ./samples --input "Write something" --model gpt-4
```

## Development

### Current Status
- ✅ **MVP Complete**: Core functionality implemented
- 🔄 **Phase 2**: Enhanced feedback system (in progress)
- 📋 **Phase 3**: Advanced features and templates (planned)
- 🌐 **Phase 4**: Web frontend (planned)

### Project Structure
```
inkmod/
├── src/
│   ├── cli/          # CLI commands and interface
│   ├── core/         # Core functionality (OpenAI, style analysis)
│   ├── utils/        # File processing and text utilities
│   └── config/       # Settings and configuration
├── examples/         # Sample writing files for testing
├── tests/           # Test files
├── requirements.txt  # Dependencies
├── setup.py         # Package installation
└── inkmod.py        # Main entry point
```

### Running Tests
```bash
pytest tests/
```

### Development Roadmap
See [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) for detailed development phases and features.

### Contributing

We welcome contributions! Since this is a work in progress, there are many opportunities to help:

1. **Test the MVP**: Try out the current functionality and report issues
2. **Enhance Features**: Work on Phase 2 features (enhanced feedback system)
3. **Add Tests**: Improve test coverage
4. **Documentation**: Help improve docs and examples

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Current Development Priorities

- **Phase 2**: Enhanced feedback system and learning capabilities
- **Testing**: More comprehensive test coverage
- **Documentation**: Better examples and tutorials
- **Performance**: Optimize token usage and response times

## License

MIT License - see LICENSE file for details.

## Roadmap

See [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) for detailed development phases and features. 