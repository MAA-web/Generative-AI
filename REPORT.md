# Transformer Architecture Implementation Report

## Executive Summary

This report documents the implementation and evaluation of Transformer architectures from scratch, focusing on three variants: Encoder-Only (BERT-style), Decoder-Only (GPT-style), and Encoder-Decoder (T5-style). The project applies these models to medical domain tasks including triage classification, text generation, and text simplification.

---

## 1. Design

### 1.1 Architecture Overview

The project implements a complete Transformer architecture from scratch using PyTorch, without relying on pre-built Transformer libraries. The design follows the original "Attention Is All You Need" paper architecture with custom implementations of core components.

### 1.2 Core Building Blocks

#### 1.2.1 Positional Encoding
- **Purpose**: Adds sequence position information to token embeddings
- **Implementation**: Sinusoidal encoding using sin/cos functions
- **Configuration**: `d_model=128`, `max_len=128`
- **Key Feature**: Non-learnable positional encodings that allow the model to understand token order

#### 1.2.2 Multi-Head Attention
- **Purpose**: Enables the model to attend to different representation subspaces simultaneously
- **Implementation**: Scaled dot-product attention with multiple heads
- **Configuration**: `d_model=128`, `n_heads=4` (resulting in `d_k=32` per head)
- **Features**:
  - Query, Key, Value projections
  - Masking support (causal masks for decoders)
  - Softmax normalization with scaling factor `1/√d_k`

#### 1.2.3 Feed-Forward Network
- **Purpose**: Applies point-wise transformations to each position
- **Implementation**: Two linear layers with ReLU activation
- **Configuration**: `d_model=128`, `d_ff=512` (4x expansion)

### 1.3 Model Variants

#### 1.3.1 Encoder-Only (MedicalClassifier)
- **Architecture**: Stack of EncoderBlocks with self-attention
- **Components**:
  - Embedding layer
  - Positional encoding
  - N encoder blocks (N=2) with layer normalization and residual connections
  - Classification head with average pooling
- **Use Case**: Classification tasks (triage zone prediction)

#### 1.3.2 Decoder-Only (MedicalGPT)
- **Architecture**: Stack of masked self-attention blocks
- **Components**:
  - Embedding layer
  - Positional encoding
  - N decoder blocks (N=2) with causal masking
  - Vocabulary projection layer
- **Use Case**: Autoregressive text generation

#### 1.3.3 Encoder-Decoder (MedicalSummarizer)
- **Architecture**: Separate encoder and decoder stacks
- **Components**:
  - Shared embedding layer
  - Encoder: N encoder blocks (N=2)
  - Decoder: N decoder blocks with:
    - Masked self-attention
    - Cross-attention to encoder output
    - Feed-forward network
  - Vocabulary projection layer
- **Use Case**: Sequence-to-sequence tasks (summarization, simplification)

### 1.4 Tokenizer Design

**SimpleTokenizer**:
- **Vocabulary Building**: Creates vocabulary from corpus with special tokens:
  - `<PAD>`: 0 (padding)
  - `<SOS>`: 1 (start of sequence)
  - `<EOS>`: 2 (end of sequence)
  - `<UNK>`: 3 (unknown words)
- **Encoding**: Converts text to fixed-length token sequences (max_len=20-128)
- **Limitations**: Basic word-level tokenization, no subword units

### 1.5 Training Configuration

- **Optimizer**: Adam optimizer
- **Learning Rate**: 0.001 (standard experiments), 0.0005 (GPT on abstracts)
- **Loss Function**: Cross-entropy loss
- **Training Epochs**: 30-50 epochs depending on experiment
- **Device**: CPU (no GPU acceleration)

---

## 2. Datasets

### 2.1 Synthetic Medical Corpus
- **Size**: 8 sentences
- **Content**: Medical phrases for demonstration
- **Examples**:
  - "patient presents with acute headache and nausea"
  - "prescribe ibuprofen for minor muscle pain"
  - "emergency trauma in lower abdomen area"
- **Purpose**: Initial model validation and architecture testing

### 2.2 Triage Questions Dataset (`triage_questions.csv`)
- **Size**: 87 triage questions
- **Features**:
  - Input: `triage_zone` column (text descriptions)
  - Labels: `class` column (triage zones: Black, Green, Red, Yellow)
- **Task**: Multi-class classification (4 classes)
- **Label Distribution**: 
  - Black: 0
  - Green: 1
  - Red: 2
  - Yellow: 3
- **Vocabulary Size**: 54 tokens

### 2.3 Messages Dataset (`messages.csv`)
- **Size**: 2000+ messages (subset used: 2000 rows)
- **Content**: Medical messages/abstracts
- **Usage**: Decoder-Only model training for text generation
- **Preprocessing**: Text sliced to 1000 characters per sample
- **Vocabulary**: Built from first 1000 rows

### 2.4 Medical Text Classification Dataset (`medical_tc_train.csv`)
- **Size**: Limited subset (8 abstracts used in final experiment)
- **Content**: Medical abstracts
- **Usage**: Decoder-Only model training
- **Preprocessing**: Text sliced to 300 characters, limited to 8 samples

### 2.5 Dataset Characteristics

**Strengths**:
- Diverse medical domain coverage
- Real-world triage classification task
- Multiple text lengths for robustness testing

**Limitations**:
- Small dataset sizes (87-2000 samples)
- Limited vocabulary coverage
- No train/validation/test splits
- Potential overfitting risk due to small datasets

---

## 3. Results

### 3.1 Experiment 1: Encoder-Only (Triage Classification)

**Configuration**:
- Model: MedicalClassifier
- Parameters: `vocab_size=54`, `d_model=128`, `n_layers=2`, `n_heads=4`, `n_classes=4`
- Training: 50 epochs, Adam optimizer, lr=0.001

**Results**:
- **Initial Loss**: 1.3533 (Epoch 1)
- **Final Loss**: 0.0008 (Epoch 50)
- **Loss Reduction**: 99.94% reduction
- **Training Progress**:
  - Epoch 11: 0.0560
  - Epoch 21: 0.0045
  - Epoch 31: 0.0016
  - Epoch 41: 0.0010
  - Epoch 50: 0.0008

**Analysis**: 
- Rapid convergence within 11 epochs
- Very low final loss suggests potential overfitting
- Model successfully learns triage classification patterns

### 3.2 Experiment 2: Decoder-Only (Text Generation)

#### 3.2.1 Synthetic Corpus Experiment
**Configuration**:
- Model: MedicalGPT
- Parameters: `vocab_size=48`, `d_model=128`, `n_layers=2`, `n_heads=4`
- Training: 41 epochs, Adam optimizer, lr=0.001
- Dataset: 8 synthetic medical sentences

**Results**:
- **Initial Loss**: 4.2435 (Epoch 0)
- **Final Loss**: 0.0320 (Epoch 40)
- **Loss Reduction**: 99.25% reduction
- **Training Progress**:
  - Epoch 20: 0.2025
  - Epoch 40: 0.0320

#### 3.2.2 Messages Dataset Experiment
**Configuration**:
- Model: MedicalGPT
- Training: 30 epochs, Adam optimizer, lr=0.0005
- Dataset: 2000 messages, sliced to 1000 characters

**Results**:
- **Initial Loss**: 7.5396 (Epoch 0)
- **Final Loss**: 0.4292 (Epoch 20)
- **Loss Reduction**: 94.31% reduction
- **Training Progress**:
  - Epoch 10: 0.8077
  - Epoch 20: 0.4292

#### 3.2.3 Medical Abstracts Experiment
**Configuration**:
- Model: MedicalGPT
- Training: 30 epochs
- Dataset: 8 abstracts, sliced to 300 characters

**Results**:
- **Initial Loss**: 10.0008 (Epoch 0)
- **Final Loss**: 4.1961 (Epoch 20)
- **Loss Reduction**: 58.04% reduction
- **Training Progress**:
  - Epoch 10: 5.3307
  - Epoch 20: 4.1961

**Analysis**:
- Best performance on synthetic corpus (small, controlled vocabulary)
- Moderate performance on messages dataset (larger, more diverse)
- Weakest performance on medical abstracts (very small sample size)
- Loss reduction correlates with dataset size and vocabulary complexity

### 3.3 Experiment 3: Encoder-Decoder (Text Simplification)

**Configuration**:
- Model: MedicalSummarizer
- Parameters: `vocab_size=48`, `d_model=128`, `n_layers=2`, `n_heads=4`
- Training: 41 epochs, Adam optimizer, lr=0.001
- Task: Map full sentence to first 8 words (simplification)

**Results**:
- **Initial Loss**: 3.8569 (Epoch 0)
- **Final Loss**: 0.0246 (Epoch 40)
- **Loss Reduction**: 99.36% reduction
- **Training Progress**:
  - Epoch 20: 0.0869
  - Epoch 40: 0.0246

**Analysis**:
- Excellent convergence on simplification task
- Lower initial loss compared to decoder-only variant
- Successful sequence-to-sequence learning

### 3.4 Overall Performance Summary

| Experiment | Model Type | Initial Loss | Final Loss | Reduction | Status |
|------------|------------|--------------|------------|-----------|--------|
| Triage Classification | Encoder-Only | 1.3533 | 0.0008 | 99.94% | Excellent |
| Generation (Synthetic) | Decoder-Only | 4.2435 | 0.0320 | 99.25% | Excellent |
| Generation (Messages) | Decoder-Only | 7.5396 | 0.4292 | 94.31% | Good |
| Generation (Abstracts) | Decoder-Only | 10.0008 | 4.1961 | 58.04% | Moderate |
| Simplification | Encoder-Decoder | 3.8569 | 0.0246 | 99.36% | Excellent |

---

## 4. Evaluation

### 4.1 Training Metrics

**Loss Curves**:
- All models show consistent loss reduction
- No signs of training instability
- Smooth convergence patterns observed

**Convergence Speed**:
- Encoder-Only: Very fast (converged by epoch 11)
- Decoder-Only: Moderate (converged by epoch 20-40)
- Encoder-Decoder: Fast (converged by epoch 20)

### 4.2 Model Evaluation Limitations

**Missing Evaluation Components**:
1. **No Validation Set**: All experiments train on full dataset without validation split
2. **No Test Set**: No held-out test set for final evaluation
3. **No Quantitative Metrics**: 
   - Classification: No accuracy, precision, recall, F1-score
   - Generation: No BLEU, ROUGE, perplexity metrics
   - No qualitative output examples
4. **No Baseline Comparisons**: No comparison with simple baselines or pre-trained models
5. **No Ablation Studies**: No analysis of individual components' contributions

### 4.3 Potential Overfitting Indicators

1. **Extremely Low Final Losses**: 
   - Encoder-Only: 0.0008 (suspiciously low)
   - Encoder-Decoder: 0.0246 (very low)
   - Suggests possible memorization rather than generalization

2. **Small Dataset Sizes**:
   - Triage: 87 samples
   - Synthetic corpus: 8 samples
   - Medical abstracts: 8 samples

3. **No Regularization**:
   - No dropout layers
   - No weight decay
   - No early stopping

### 4.4 Model Capacity Analysis

**Model Size**:
- Small models: 2 layers, 4 heads, 128 dimensions
- Appropriate for small datasets
- May be insufficient for complex medical language understanding

**Vocabulary Coverage**:
- Limited vocabularies (48-54 tokens)
- May miss important medical terminology
- SimpleTokenizer limitations (word-level, no subword units)

---

## 5. Error Analysis

### 5.1 Implementation Issues

#### 5.1.1 Tensor Reshaping Errors
- **Issue**: Encountered stride errors when using `.view()` on non-contiguous tensors
- **Solution**: Used `.reshape()` or `.clone().detach()` to ensure contiguous tensors
- **Location**: Encoder-Decoder training loop

#### 5.1.2 Mask Handling
- **Issue**: Need to handle both 2D (causal) and 4D masks in attention mechanism
- **Solution**: Broadcasting in masked_fill operation
- **Impact**: Correct causal masking for autoregressive generation

### 5.2 Data Preprocessing Issues

#### 5.2.1 Text Truncation
- **Issue**: Fixed-length sequences may truncate important information
- **Impact**: Loss of context in longer medical texts
- **Mitigation**: Used slicing (300-1000 characters) but may lose important details

#### 5.2.2 Vocabulary Building
- **Issue**: SimpleTokenizer builds vocabulary from limited corpus
- **Impact**: Many medical terms may be mapped to `<UNK>` token
- **Limitation**: No subword tokenization (BPE, WordPiece) for better coverage

### 5.3 Training Issues

#### 5.3.1 Learning Rate
- **Observation**: Different learning rates used (0.001 vs 0.0005)
- **Impact**: May affect convergence speed and final performance
- **Recommendation**: Systematic hyperparameter tuning needed

#### 5.3.2 Batch Processing
- **Issue**: All data processed in single batch (no batching strategy)
- **Impact**: May cause memory issues with larger datasets
- **Limitation**: No mini-batch training for scalability

### 5.4 Model Architecture Issues

#### 5.4.1 Decoder-Only Implementation
- **Issue**: Uses EncoderBlocks instead of true DecoderBlocks
- **Impact**: May not fully implement decoder-only architecture correctly
- **Note**: Causal masking applied, but architecture could be more decoder-specific

#### 5.4.2 No Dropout
- **Issue**: No dropout layers for regularization
- **Impact**: Increased overfitting risk, especially with small datasets
- **Recommendation**: Add dropout to attention and feed-forward layers

### 5.5 Evaluation Gaps

#### 5.5.1 No Error Analysis on Predictions
- **Missing**: Analysis of which classes are confused
- **Missing**: Examples of misclassified triage cases
- **Missing**: Generated text quality assessment

#### 5.5.2 No Out-of-Distribution Testing
- **Missing**: Testing on unseen medical terminology
- **Missing**: Testing on different medical domains
- **Missing**: Robustness to input variations

---

## 6. Lessons Learned

### 6.1 Technical Lessons

#### 6.1.1 Transformer Implementation
- **Key Insight**: Building Transformers from scratch provides deep understanding of attention mechanisms
- **Challenge**: Proper tensor manipulation and masking requires careful attention
- **Benefit**: Full control over architecture modifications

#### 6.1.2 Architecture Variants
- **Encoder-Only**: Best for classification tasks with clear input-output mapping
- **Decoder-Only**: Effective for generation but requires careful causal masking
- **Encoder-Decoder**: Powerful for sequence-to-sequence tasks but more complex

#### 6.1.3 Tokenization
- **Limitation**: Simple word-level tokenization insufficient for medical domain
- **Need**: Subword tokenization (BPE, SentencePiece) for better vocabulary coverage
- **Impact**: `<UNK>` tokens reduce model effectiveness

### 6.2 Data Lessons

#### 6.2.1 Dataset Size
- **Finding**: Small datasets (8-87 samples) can achieve very low training loss
- **Concern**: Low loss may indicate overfitting rather than true learning
- **Recommendation**: Need validation/test splits to assess generalization

#### 6.2.2 Medical Domain Specificity
- **Challenge**: Medical terminology requires specialized vocabulary
- **Observation**: Limited vocabulary (48-54 tokens) may miss important terms
- **Recommendation**: Use medical domain pre-trained tokenizers or larger vocabularies

#### 6.2.3 Data Preprocessing
- **Finding**: Text truncation (slicing) necessary for fixed-length sequences
- **Trade-off**: Balance between sequence length and information retention
- **Recommendation**: Use dynamic padding or longer sequences when possible

### 6.3 Training Lessons

#### 6.3.1 Learning Rate
- **Observation**: Learning rate of 0.001 works well for small models
- **Finding**: Lower learning rate (0.0005) used for larger datasets
- **Recommendation**: Systematic learning rate scheduling and tuning

#### 6.3.2 Convergence
- **Finding**: Models converge quickly (11-40 epochs) on small datasets
- **Concern**: Fast convergence may indicate insufficient training or overfitting
- **Recommendation**: Monitor validation metrics, not just training loss

#### 6.3.3 Regularization
- **Gap**: No dropout or other regularization techniques used
- **Impact**: Models may overfit to training data
- **Recommendation**: Add dropout (0.1-0.3) and weight decay for better generalization

### 6.4 Evaluation Lessons

#### 6.4.1 Metrics Importance
- **Gap**: Only training loss reported, no accuracy or other metrics
- **Impact**: Cannot assess true model performance
- **Recommendation**: Always include task-specific metrics (accuracy, BLEU, etc.)

#### 6.4.2 Validation Strategy
- **Gap**: No train/validation/test splits
- **Impact**: Cannot detect overfitting or assess generalization
- **Recommendation**: Implement proper data splitting (e.g., 70/15/15)

#### 6.4.3 Qualitative Analysis
- **Gap**: No examples of model outputs
- **Impact**: Cannot assess practical utility
- **Recommendation**: Include sample predictions and generated text

### 6.5 Best Practices Identified

1. **Modular Design**: Separating building blocks (PositionalEncoding, MultiHeadAttention, FeedForward) enables easy experimentation
2. **Clear Documentation**: Inline comments help understand complex operations
3. **Visualization**: Loss curves provide immediate feedback on training progress
4. **Flexible Architecture**: Support for different variants (encoder, decoder, encoder-decoder) in one codebase

### 6.6 Areas for Improvement

1. **Evaluation Framework**: Implement comprehensive metrics and validation
2. **Regularization**: Add dropout and weight decay
3. **Tokenization**: Upgrade to subword tokenization
4. **Data Handling**: Implement proper train/validation/test splits
5. **Model Scaling**: Experiment with larger models and more layers
6. **Hyperparameter Tuning**: Systematic search for optimal learning rates, batch sizes
7. **Baseline Comparisons**: Compare against simple baselines and pre-trained models
8. **Error Analysis**: Detailed analysis of failure cases
9. **GPU Acceleration**: Move to GPU for faster training and larger models
10. **Medical Domain Adaptation**: Use medical domain pre-trained embeddings or models

### 6.7 Project Strengths

1. **Educational Value**: Complete from-scratch implementation provides deep understanding
2. **Comprehensive Coverage**: All three major Transformer variants implemented
3. **Real-World Application**: Applied to medical domain tasks
4. **Code Quality**: Well-structured, commented code
5. **Experimentation**: Multiple experiments with different datasets

### 6.8 Project Limitations

1. **Small Scale**: Limited to small datasets and models
2. **Evaluation Gaps**: Missing comprehensive evaluation metrics
3. **Overfitting Risk**: Very low losses suggest potential memorization
4. **Tokenization**: Basic tokenization may limit performance
5. **No Baselines**: Cannot assess relative performance
6. **CPU-Only**: Limited by CPU training speed

---

## 7. Conclusions

This project successfully implements Transformer architectures from scratch and demonstrates their application to medical domain tasks. The models show strong training loss reduction across all experiments, with the Encoder-Only model achieving 99.94% loss reduction on triage classification.

However, the evaluation reveals significant gaps: no validation/test sets, no quantitative metrics beyond training loss, and potential overfitting concerns. The small dataset sizes and very low final losses suggest the models may be memorizing rather than generalizing.

**Key Achievements**:
- Complete Transformer implementation from scratch
- Successful application to medical tasks
- All three major variants (Encoder, Decoder, Encoder-Decoder) working

**Key Recommendations**:
- Implement proper train/validation/test splits
- Add comprehensive evaluation metrics
- Include regularization (dropout, weight decay)
- Upgrade tokenization to subword methods
- Conduct error analysis on predictions
- Compare against baselines

The project provides a solid foundation for understanding Transformers and demonstrates their potenti