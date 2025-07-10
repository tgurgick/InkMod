# Scoring and Reinforcement Learning Guide

## Overview

InkMod uses reinforcement learning to improve writing style imitation. The system generates responses, evaluates them against your original style, and updates the model based on feedback.

## Scoring Systems

### Traditional LLM-Only Scoring

The original system used OpenAI as a "teacher" to evaluate responses:

- **Style Score (0-1)**: How well the response matches your writing style
- **Tone Score (0-1)**: Appropriateness and consistency of tone
- **Structure Score (0-1)**: Sentence/paragraph structure quality

**Limitations:**
- Inconsistent scoring (same input → different scores)
- Subjective evaluation
- High API costs
- Slow evaluation (API latency)

### Hybrid Reward Function (Recommended)

The new hybrid approach combines **standard NLP metrics** with **optional LLM feedback**:

#### **Standard NLP Metrics (Objective)**

1. **BLEU Score (0-1)**
   - Measures n-gram overlap with reference samples
   - Indicates style similarity
   - Fast and reproducible

2. **ROUGE Score (0-1)**
   - Evaluates vocabulary and phrase overlap
   - Measures content similarity
   - Uses unigrams and bigrams

3. **Consistency Score (0-1)**
   - Measures vocabulary overlap with training data
   - Indicates style consistency
   - Based on word frequency analysis

4. **Length Score (0-1)**
   - Compares sentence/paragraph structure similarity
   - Evaluates average sentence length
   - Measures word length patterns

5. **Tone Score (0-1)**
   - Evaluates tone marker usage consistency
   - Compares formal/casual/professional markers
   - Based on predefined tone indicators

#### **Combined Score Calculation**
```python
overall_score = (
    0.3 * bleu_score +      # Style similarity
    0.2 * rouge_score +     # Vocabulary overlap
    0.2 * consistency_score + # Style consistency
    0.15 * length_score +   # Structural similarity
    0.15 * tone_score       # Tone consistency
)
```

#### **LLM Qualitative Feedback (Optional)**
When enabled, provides detailed suggestions:
- Vocabulary improvements
- Tone adjustments
- Structural recommendations
- Style enhancements

## Training Iterations

### Recommended Iterations

| Use Case | Iterations | Reasoning |
|----------|------------|-----------|
| **Initial Training** | 5-10 | Establish baseline patterns |
| **Style Refinement** | 3-5 | Fine-tune existing model |
| **New Writing Samples** | 3-7 | Adapt to new style data |
| **Convergence Testing** | 2-3 | Check if model has converged |

### Convergence Detection

The system automatically detects when training should stop:

- **High Convergence**: Model performance is stable
- **Low Variance**: Consistent scores across iterations
- **No Improvement**: Performance plateaus

### Cost Analysis

#### **Hybrid Approach (Recommended)**
```
Standard Metrics Only: $0.00 (no API calls)
With LLM Feedback: ~$0.20 per iteration (5 prompts)
```

#### **Traditional LLM-Only**
```
Every Evaluation: ~$0.18 per iteration (5 prompts)
```

## Performance Benchmarks

### **Good Performance Scores**
- **Overall Score**: 0.7+ (excellent), 0.5+ (good), 0.3+ (acceptable)
- **BLEU Score**: 0.05+ (good style similarity)
- **ROUGE Score**: 0.4+ (good vocabulary overlap)
- **Consistency Score**: 0.6+ (good style consistency)
- **Length Score**: 0.5+ (good structural similarity)
- **Tone Score**: 0.3+ (good tone consistency)

### **Sample Results**
```
🎯 Final Performance: 0.349
📊 BLEU Score: 0.034      (style similarity)
📊 ROUGE Score: 0.494     (vocabulary overlap)
📊 Consistency Score: 0.695 (style consistency)
📊 Length Score: 0.587    (structural similarity)
📊 Tone Score: 0.086      (tone consistency)
```

## Usage Recommendations

### **For Best Results**
1. **Use hybrid approach**: Standard metrics + optional LLM feedback
2. **Start with 5 iterations**: Establish baseline performance
3. **Monitor convergence**: Let system auto-detect when to stop
4. **Use incremental training**: Build on previous learning
5. **Test with local LLM backends**: For zero-cost generation

### **For Cost Optimization**
1. **Use `--no-llm-feedback`**: Standard metrics only
2. **Use local LLM backends**: Llama.cpp, GPT4All, HuggingFace
3. **Monitor API usage**: Track costs in training results
4. **Batch training**: Run multiple iterations together

### **For Quality Optimization**
1. **Enable LLM feedback**: Get detailed improvement suggestions
2. **Use high-quality samples**: Clean, representative writing samples
3. **Diverse test prompts**: Cover different writing scenarios
4. **Regular retraining**: Update model with new samples

## Troubleshooting

### **Low Scores**
- **Check training samples**: Ensure they're representative of your style
- **Increase iterations**: Allow more training time
- **Review test prompts**: Make sure they're appropriate
- **Enable LLM feedback**: Get specific improvement suggestions

### **High Variance**
- **Check sample quality**: Ensure consistent writing style
- **Reduce iterations**: Model may be overfitting
- **Use incremental training**: Build on stable baseline

### **Convergence Issues**
- **Add more samples**: Increase training data diversity
- **Adjust learning rate**: Modify model update frequency
- **Check for errors**: Review training logs for issues 