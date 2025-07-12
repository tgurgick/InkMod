# 🔴 Redline Feedback Guide

## Overview

The redline feature in InkMod enables **sentence-by-sentence editing** with precise feedback capture for continuous model improvement. This user-driven learning system allows you to correct specific sentences and teach the model your exact preferences.

## 🎯 How Redline Works

### **1. Content Generation**
```bash
inkmod redline --style-folder ./writing-samples --input "Write a professional email"
```

The system generates content using your learned style model and displays it sentence by sentence.

### **2. Sentence-by-Sentence Review**
```
🔴 Redline Mode
Generated Content:
Line 1: Hello there! I hope this email finds you well.
Line 2: I wanted to follow up on our recent conversation about the project timeline.
Line 3: We discussed several key milestones that need to be completed by the end of the month.

Redline Commands: <number> | save | quit | show | back
```

### **3. Precise Editing**
```
Command: 1
Editing Line 1:
Original: Hello there! I hope this email finds you well.
New version (or 'back' to cancel): Hi there! I hope you're doing well.
✅ Line 1 updated!
```

### **4. Feedback Capture**
When you save changes, the system captures:
- **Before/after pairs** for each edited sentence
- **Vocabulary improvements** from your corrections
- **Tone adjustments** based on your preferences
- **Structural changes** from your edits

### **5. Model Learning**
```bash
inkmod apply-feedback --model-path enhanced_style_model.pkl
```

The model learns from your corrections and improves future generations.

## 📊 Feedback Data Structure

### **Feedback Entry Format**
```json
{
  "original_input": "Write a professional email",
  "original_response": "Dear Sir/Madam...",
  "final_response": "Hi there...",
  "style_folder": "./writing-samples",
  "timestamp": "2025-07-12T09:38:14.496140",
  "feedback_pairs": [
    {
      "line_number": 1,
      "before": "Dear Sir/Madam",
      "after": "Hi there",
      "feedback_type": "sentence_revision"
    }
  ],
  "total_revisions": 1,
  "feedback_type": "redline_revisions"
}
```

### **Feedback File Location**
- **Default**: `./feedback.json`
- **Configurable**: Set `FEEDBACK_FILE` in environment variables
- **Format**: JSON array of feedback entries

## 🧠 Learning Process

### **How the Model Learns from Redline Feedback**

#### **1. Vocabulary Update**
- **Process**: Extracts words from your "after" sentences
- **Effect**: Model becomes more likely to use your preferred words
- **Example**: If you change "Dear Sir/Madam" to "Hi there", "Hi" and "there" get higher priority

#### **2. Phrase Extraction**
- **Process**: Identifies 2-4 word phrases from your corrections
- **Effect**: Model learns your preferred multi-word expressions
- **Example**: "hope you're doing well" becomes a common phrase

#### **3. Tone Marker Enhancement**
- **Process**: Analyzes tone indicators in your preferred sentences
- **Effect**: Model better matches your tone preferences
- **Example**: If you prefer casual over formal, casual markers get stronger

#### **4. Sentence Pattern Learning**
- **Process**: Analyzes structure of your corrected sentences
- **Effect**: Model generates sentences with similar structure
- **Example**: If you prefer shorter sentences, model learns this pattern

#### **5. Style Characteristic Updates**
- **Process**: Recalculates overall style metrics
- **Effect**: Global style consistency improves
- **Example**: Average sentence length, vocabulary diversity, tone balance

## 🎯 Benefits of Redline Learning

### **Precision**
- **Targeted Corrections**: Edit specific sentences, not entire responses
- **Granular Feedback**: Each sentence correction teaches the model
- **Contextual Learning**: Model learns in context of your writing style

### **Efficiency**
- **Zero API Cost**: No OpenAI calls required for feedback
- **Immediate Learning**: Model updates instantly with your corrections
- **Transparent Process**: You can see exactly what the model learned

### **User-Driven**
- **Real Preferences**: Model learns from actual user corrections
- **"This, Not That" Learning**: Clear before/after pairs
- **Continuous Improvement**: Each session makes the model better

### **Transparency**
- **Inspectable**: Can view vocabulary, phrases, and patterns learned
- **Trackable**: Learning progress is recorded and viewable
- **Controllable**: You decide what corrections to apply

## 📈 Usage Workflow

### **Complete Redline Workflow**

#### **Step 1: Generate Content**
```bash
inkmod redline --style-folder ./writing-samples --input "Write a professional email"
```

#### **Step 2: Review and Edit**
```
🔴 Redline Mode
Generated Content:
Line 1: Hello there! I hope this email finds you well.
Line 2: I wanted to follow up on our recent conversation about the project timeline.

Redline Commands: <number> | save | quit | show | back
Command: 1
Editing Line 1:
Original: Hello there! I hope this email finds you well.
New version (or 'back' to cancel): Hi there! I hope you're doing well.
✅ Line 1 updated!
```

#### **Step 3: Save Feedback**
```
Command: save
💾 Redline feedback saved to feedback.json
📊 Captured 1 revision pairs for training
✅ Changes saved and feedback captured for training!
```

#### **Step 4: Apply to Model**
```bash
inkmod apply-feedback --model-path enhanced_style_model.pkl
```

#### **Step 5: Check Progress**
```bash
inkmod learning-progress
```

## 🔧 Advanced Usage

### **Multiple Edits in One Session**
```
Command: 1
Editing Line 1: [make edit]

Command: 3
Editing Line 3: [make edit]

Command: save
💾 Redline feedback saved to feedback.json
📊 Captured 2 revision pairs for training
```

### **Cancelling an Edit**
```
Command: 2
Editing Line 2:
Original: I wanted to follow up on our recent conversation about the project timeline.
New version (or 'back' to cancel): back
🔄 Cancelled editing - back to main menu
```

### **Batch Feedback Application**
```bash
# Apply all feedback entries
inkmod apply-feedback --model-path enhanced_style_model.pkl --apply-all

# Apply specific feedback file
inkmod apply-feedback --model-path enhanced_style_model.pkl --feedback-file custom_feedback.json
```

### **Feedback Review**
```bash
# View feedback file
cat feedback.json | jq '.[0].feedback_pairs'

# Check learning progress
inkmod learning-progress
```

## 🎯 Best Practices

### **Effective Redline Editing**

#### **1. Be Specific**
- **Good**: Change "Dear Sir/Madam" to "Hi there"
- **Better**: Change "I hope this email finds you well" to "I hope you're doing well"

#### **2. Be Consistent**
- **Good**: Use similar corrections across multiple sessions
- **Better**: The model learns your consistent preferences

#### **3. Be Comprehensive**
- **Good**: Edit multiple sentences in one session
- **Better**: More feedback = faster learning

#### **4. Be Patient**
- **Good**: Apply feedback regularly
- **Better**: Model improves incrementally over time

### **When to Use Redline**

#### **Perfect For:**
- **Style Refinement**: Fine-tune your writing style
- **Tone Adjustment**: Correct formal/casual preferences
- **Vocabulary Improvement**: Teach preferred words and phrases
- **Structure Learning**: Adjust sentence and paragraph structure

#### **Not Ideal For:**
- **Content Generation**: Use `generate` for one-off content
- **Batch Processing**: Use `batch` for multiple inputs
- **Interactive Sessions**: Use `interactive` for real-time editing

## 🔍 Troubleshooting

### **Common Issues**

#### **No Feedback Captured**
- **Cause**: No changes made or save not completed
- **Solution**: Make edits and use `save` command

#### **Model Not Learning**
- **Cause**: Feedback not applied to model
- **Solution**: Run `inkmod apply-feedback`

#### **Poor Learning Results**
- **Cause**: Inconsistent corrections or insufficient feedback
- **Solution**: Be consistent and provide more feedback

#### **Feedback File Issues**
- **Cause**: Corrupted or invalid feedback file
- **Solution**: Check file format and regenerate if needed

#### **Stuck in Edit Mode**
- **Cause**: Accidentally typed wrong number or got confused
- **Solution**: Type 'back' when prompted for new version to cancel and return to main menu

### **Debug Commands**
```bash
# Check feedback file
cat feedback.json

# Verify model updates
inkmod learning-progress

# Test generation after feedback
inkmod generate --style-folder ./writing-samples --input "Test prompt"
```

## 📊 Performance Metrics

### **Learning Progress Indicators**
- **Total Sessions**: Number of feedback applications
- **Vocabulary Growth**: New words learned from corrections
- **Tone Improvements**: Enhanced tone marker accuracy
- **Pattern Enhancements**: Better sentence structure matching

### **Expected Improvements**
- **After 5-10 corrections**: Noticeable style improvements
- **After 20-30 corrections**: Significant style matching
- **After 50+ corrections**: Highly personalized style model

## 🚀 Integration with Other Features

### **Redline + Enhanced Training**
```bash
# Apply redline feedback
inkmod apply-feedback --model-path enhanced_style_model.pkl

# Continue with enhanced training
inkmod train-enhanced --incremental --style-folder ./writing-samples
```

### **Redline + Hybrid Training**
```bash
# Apply redline feedback
inkmod apply-feedback --model-path enhanced_style_model.pkl

# Use hybrid reward training
inkmod train-hybrid --style-folder ./writing-samples --test-prompts prompts.txt
```

### **Redline + Model Exploration**
```bash
# Apply feedback
inkmod apply-feedback --model-path enhanced_style_model.pkl

# Explore updated model
inkmod explore --model-path enhanced_style_model.pkl --detailed
```

## 🎯 Summary

The redline feature provides a powerful, user-driven approach to model improvement:

- **Precise**: Edit specific sentences with targeted feedback
- **Efficient**: Zero API cost, immediate learning
- **Transparent**: See exactly what the model learns
- **Continuous**: Incremental improvement over time
- **User-Driven**: Real preferences, not AI evaluation

This makes InkMod's learning system uniquely responsive to your specific writing style preferences and corrections. 