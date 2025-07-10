# InkMod - Writing Style Mirror CLI Tool

**🚧 Work in Progress - MVP Version** 

InkMod is an open-source CLI tool that analyzes a user's writing style from provided examples and generates responses that match that style using OpenAI's API. This is currently in MVP (Minimum Viable Product) phase with core functionality implemented.

## 🎯 Performance Scoring

InkMod uses a comprehensive scoring system to evaluate style matching quality:

### **Score Ranges (0.0 to 1.0)**

| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| 0.0 - 0.3   | Poor          | Minimal style matching |
| 0.3 - 0.5   | Fair          | Basic style recognition |
| 0.5 - 0.7   | Good          | Solid style matching |
| 0.7 - 0.8   | Very Good     | Strong style capture |
| 0.8 - 0.9   | Excellent     | Near-perfect matching |
| 0.9 - 1.0   | Outstanding   | Exceptional style replication |

### **Score Components**

- **Style Score**: Overall writing style characteristics (vocabulary, patterns, flow)
- **Tone Score**: Consistency of tone (formal/casual/professional)  
- **Structure Score**: Sentence length, paragraph structure, organization

### **Typical Results**
- **Style Score**: 0.7+ (Good to Excellent)
- **Tone Score**: 0.8+ (Excellent) 
- **Structure Score**: 0.6-0.8 (Good)

**Production-ready performance**: 0.7+ style score with 3-5 training iterations.

📖 **For detailed scoring analysis and training strategies, see [docs/SCORING_AND_RL_GUIDE.md](docs/SCORING_AND_RL_GUIDE.md)**

## Features

- **Style Analysis**: Analyze writing samples to understand tone, vocabulary, and structure
- **Style Mirroring**: Generate responses that match the analyzed writing style
- **Style Validation**: Compare different analysis methods and validate their effectiveness
- **Reinforcement Learning**: Train a local model using OpenAI as teacher
- **Continuous Learning**: Model remembers and improves with each session
- **Multi-Backend LLM Support**: Use Llama.cpp, GPT4All, or HuggingFace for local generation
- **Backend Management**: List, inspect, and test local LLM backends
- **Interactive Mode**: Real-time style mirroring with immediate feedback
- **Batch Processing**: Process multiple inputs at once
- **Feedback System**: Capture user edits to improve future generations
- **AI-Powered Analysis**: Use OpenAI to generate nuanced style guides
- **Model Inspection**: Explore and export learned models with the `explore` command

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tgurgick/InkMod.git
cd InkMod
```

2. Install dependencies and the CLI tool:
```bash
pip install -e .
```

3. Set up your OpenAI API key and model in a `.env` file (or as environment variables):
```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o  # Default model is gpt-4o
```

**Note:**
- The CLI will use the model specified in your `.env` file (`OPENAI_MODEL`).
- If not set, it defaults to `gpt-4o`.
- You can override the model at runtime with the `--model` flag.

## Usage

### Basic Usage

Generate a response in a specific style:
```bash
inkmod generate --style-folder ./writing-samples --input "Write a professional email response"
```

### Style Validation

Compare different style analysis methods:
```bash
inkmod validate --style-folder ./writing-samples --test-input "Write a friendly follow-up email about a missed meeting"
```

### Reinforcement Learning Training

Train a local model using OpenAI as teacher:
```bash
inkmod train --style-folder ./writing-samples --test-prompts test_prompts.txt --iterations 5
```

This creates a local model that learns your writing style and can generate responses without API calls.

#### **Training Iterations Guide**

| Use Case | Recommended Iterations | Expected Performance |
|----------|----------------------|---------------------|
| **Basic Style Matching** | 3-5 iterations | 0.6-0.7 style score |
| **Professional Writing** | 5-8 iterations | 0.7-0.8 style score |
| **High-Quality Content** | 8-12 iterations | 0.8+ style score |
| **Research/Publication** | 12-20 iterations | 0.85+ style score |

**Cost**: ~$0.18 per iteration. Most use cases achieve good results with 3-5 iterations.

### Enhanced Training & Continuous Learning

Train with advanced pattern extraction, multi-backend LLMs, and continuous learning:
```bash
inkmod train-enhanced --style-folder ./writing-samples --test-prompts test_prompts.txt --backend hf-distilgpt2 --iterations 5 --incremental
```
- Use `--incremental` (default) to build on previous learning
- Use `--fresh` to start a new model from scratch

### Model Inspection & Exploration

Explore and analyze your learned model:
```bash
inkmod explore --model-path enhanced_style_model.pkl --detailed
```
- Use `--detailed` for a full breakdown (vocabulary, patterns, history, scores)
- Use `--export-json model_info.json` to export model details for further analysis

📖 **See [docs/MODEL_FILE_GUIDE.md](docs/MODEL_FILE_GUIDE.md) for a full explanation of the model file structure and how to read it.**

### Backend Management

List, inspect, and test local LLM backends:
```bash
inkmod backends --list
inkmod backends --info llama-7b
inkmod backends --test gpt4all-j
```

### Interactive Mode

Start an interactive session:
```bash
inkmod interactive --style-folder ./writing-samples
```

### Batch Processing

Process multiple inputs from a file:
```bash
inkmod batch --style-folder ./writing-samples --input-file inputs.txt --output-file outputs.txt
```

### Edit Mode (with feedback)

Generate and edit responses while capturing feedback:
```bash
inkmod generate --style-folder ./writing-samples --input "Write a blog post" --edit-mode
```

### Check Learning Progress

See your model's learning journey, best scores, and convergence:
```bash
inkmod learning-progress
```

## Sample Output

### Style Validation Results

When you run the validation command, you'll see a comparison of different style analysis methods:

```
============================================================
Style Analysis Method Comparison
============================================================
                            Method Comparison                             
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Method ┃ Similarity Score ┃ Generation Cost ┃ Total Cost ┃ Best Method ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Rigid  │            0.315 │         $0.0254 │    $0.0254 │     ⭐      │
│ Ai     │            0.308 │         $0.0491 │    $0.1203 │             │
│ Direct │            0.234 │         $0.0239 │    $0.0239 │             │
└────────┴──────────────────┴─────────────────┴────────────┴─────────────┘

╭─────────────────────────────────────── 🎯 Analysis Recommendation ────────────────────────────────────────╮
│ Recommendation: Rigid metrics analysis performed best (similarity: 0.32) and is cost-effective ($0.0254). │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Sample Outputs:

Rigid Method:
Hi there,

I hope you're doing well. I wanted to touch base regarding the meeting we missed yesterday. I understand that
things can get hectic, and I just want to make sure we can reschedule at a time...

Ai Method:
Hi [Recipient's Name],

Thanks for your note about the meeting we missed. I completely understand that things can get hectic, and I'm
glad we're able to touch base now.

The team and I are keen to cat...

Direct Method:
Hi there,

I hope you're doing well. I wanted to follow up regarding our meeting that was scheduled for yesterday. I 
understand things can get hectic, and it's easy for appointments to slip through th...
```

This validation helps you understand which method best matches your writing style and provides cost analysis for each approach.

### Reinforcement Learning Training

The training system shows how a local model learns from OpenAI feedback:

```
🎯 Starting reinforcement learning training...

📚 Phase 1: Initial local model training
🧠 Training local style model...
✅ Initial training complete: 266 vocabulary items

🔄 Iteration 1/2
📊 Iteration 1 performance: 0.100

🔄 Iteration 2/2
📊 Iteration 2 performance: 0.140

============================================================
Reinforcement Learning Training Results
============================================================
                  Performance Progression                  
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ Iteration ┃ Avg Score ┃ Min Score ┃ Max Score ┃    Cost ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ 1         │     0.100 │     0.100 │     0.100 │ $0.1461 │
│ 2         │     0.140 │     0.100 │     0.200 │ $0.1446 │
└───────────┴───────────┴───────────┴───────────┴─────────┘

🎯 Final Performance: 0.120
💰 Total Training Cost: $0.2908

💰 Cost Comparison:
   Local Model Cost: $0.0000
   OpenAI Cost: $0.1262
   Potential Savings: $0.1262
```

## Local LLM Backends: Cost, Privacy, and Output Quality

InkMod supports multiple generation modes, each with different tradeoffs for cost, privacy, and output quality. For best results, use a lightweight local LLM backend (such as Llama.cpp, GPT4All, or HuggingFace models) together with your learned style model.

| Mode                | Cost   | Privacy | Output Quality         | Notes                                 |
|---------------------|--------|---------|------------------------|---------------------------------------|
| Template-based      | $0     | Local   | Basic, style-matched   | No LLM, just patterns                 |
| Local LLM backend   | $0     | Local   | High, context-aware    | Best local results, needs LLM         |
| OpenAI API          | $$$    | Cloud   | High, context-aware    | Best quality, but costs money         |

### Key Points
- **Local LLM backend** (Llama.cpp, GPT4All, HuggingFace, etc.) + your learned model = **best results, zero cost, and full privacy**.
- All processing is local—no data leaves your machine.
- You can generate any type of text, and the LLM will adapt to your prompt, using your style as a guide.
- Template-based mode is always available, but is less flexible and creative.
- OpenAI API mode provides the highest quality, but incurs cost and requires an API key.

See the [docs/LOCAL_MODEL_LEARNING.md](docs/LOCAL_MODEL_LEARNING.md) and [docs/SCORING_AND_RL_GUIDE.md](docs/SCORING_AND_RL_GUIDE.md) for more details on model selection and backend setup.

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

You can configure various parameters in your `.env` file or via CLI flags:

```bash
# Set model parameters in .env
OPENAI_MODEL=gpt-4o
OPENAI_MAX_TOKENS=1000
```

## Development

### Current Status
- ✅ **MVP Complete**: Core functionality implemented
- ✅ **Style Validation**: Compare different analysis methods
- ✅ **Reinforcement Learning**: Local model training with OpenAI teacher
- 🔄 **Phase 2**: Enhanced feedback system (in progress)
- 📋 **Phase 3**: Advanced features and templates (planned)
- 🌐 **Phase 4**: Web frontend (planned)

### Project Structure
```
inkmod/
├── src/
│   ├── cli/          # CLI commands and interface
│   ├── core/         # Core functionality (OpenAI, style analysis, validation, local models)
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