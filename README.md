# Transformer Architectures: From Scratch Implementation

A comprehensive implementation of transformer neural network architectures from scratch using PyTorch. This project demonstrates three fundamental transformer variants with practical applications in medical text processing, complete with training visualization.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture Components](#architecture-components)
- [Transformer Variants](#transformer-variants)
- [Project Structure](#project-structure)
- [Technical Flow](#technical-flow)
- [Usage](#usage)
- [Experiments](#experiments)
- [Dependencies](#dependencies)

## 🎯 Overview

This project implements transformer architectures from the ground up, providing educational and practical examples of how modern language models work. The implementation focuses on three key transformer variants:

1. **Encoder-Only (BERT-style)** - Classification tasks
2. **Decoder-Only (GPT-style)** - Autoregressive generation
3. **Encoder-Decoder (T5-style)** - Sequence-to-sequence tasks

All implementations are demonstrated using a medical text corpus for practical, domain-specific applications.

## 🏗️ Architecture Components

### Core Building Blocks (`building_blocks.py`)

#### 1. **Positional Encoding**
```python
class PositionalEncoding(nn.Module)
```
- **Purpose**: Injects positional information into token embeddings since transformers lack inherent sequence awareness
- **Implementation**: Uses sinusoidal encoding with frequencies computed as `exp(-log(10000) * 2i / d_model)`
- **Formula**: 
  - Even dimensions: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
  - Odd dimensions: `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
- **Key Feature**: Pre-computed buffer registered to avoid GPU transfer overhead

#### 2. **Multi-Head Attention**
```python
class MultiHeadAttention(nn.Module)
```
- **Purpose**: Allows the model to attend to different parts of the input simultaneously
- **Architecture**:
  - Splits `d_model` into `n_heads` parallel attention mechanisms
  - Each head has dimension `d_k = d_model // n_heads`
  - Independent Query (Q), Key (K), and Value (V) projections
- **Attention Mechanism**:
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k) × V
  ```
- **Scaled Dot-Product**: Division by `√d_k` prevents softmax saturation
- **Masking Support**: Handles causal masks (decoder) and padding masks

#### 3. **Feed-Forward Network**
```python
class FeedForward(nn.Module)
```
- **Purpose**: Applies point-wise non-linear transformations
- **Architecture**: Two linear layers with ReLU activation
  - Expansion: `d_model → d_ff` (default 512)
  - Contraction: `d_ff → d_model`
- **Design**: Position-wise fully connected network applied independently to each token

## 🔄 Transformer Variants

### 1. Encoder-Only Architecture (BERT-style)

**Class**: `MedicalClassifier`

**Use Case**: Medical text triage/classification (emergency vs. routine)

**Architecture**:
```
Input Tokens → Embedding → Positional Encoding
    ↓
[N × EncoderBlock]
    ↓
Mean Pooling (sequence-level representation)
    ↓
Linear Classifier → Class Probabilities
```

**EncoderBlock Components**:
1. Multi-head self-attention (bidirectional)
2. Residual connection + LayerNorm
3. Feed-forward network
4. Residual connection + LayerNorm

**Key Features**:
- Bidirectional context (full sequence attention)
- Sequence-level classification via mean pooling
- No masking (sees entire sequence)

### 2. Decoder-Only Architecture (GPT-style)

**Class**: `MedicalGPT`

**Use Case**: Autoregressive text generation (next-token prediction)

**Architecture**:
```
Input Tokens → Embedding → Positional Encoding
    ↓
[N × EncoderBlock with Causal Mask]
    ↓
Linear Projection → Vocabulary Logits
```

**Key Features**:
- **Causal Masking**: Lower triangular matrix prevents future token access
  ```python
  mask = torch.tril(torch.ones(seq_len, seq_len))
  ```
- Autoregressive training: Predicts token at position `t` given tokens `[0...t-1]`
- Next-token prediction objective

### 3. Encoder-Decoder Architecture (T5-style)

**Class**: `MedicalSummarizer`

**Use Case**: Sequence-to-sequence tasks (e.g., medical text simplification)

**Architecture**:
```
Source Sequence → Encoder Stack → Encoded Representation
                                            ↓
Target Sequence → Decoder Stack ← Cross-Attention
    ↓
Linear Projection → Vocabulary Logits
```

**EncoderDecoderBlock Components**:
1. **Self-Attention**: Decoder attends to its own positions (with causal mask)
2. **Cross-Attention**: Queries from decoder, Keys/Values from encoder
3. Feed-forward network
4. Layer normalization after each sub-layer

**Key Features**:
- Separate encoder and decoder stacks
- Cross-attention connects encoder output to decoder
- Causal masking on decoder self-attention only
- Bidirectional encoder, unidirectional decoder

## 📁 Project Structure

```
gen-ai/
├── building_blocks.py      # Core transformer components
│   ├── PositionalEncoding
│   ├── MultiHeadAttention
│   └── FeedForward
├── main.py                 # Transformer variants & experiments
│   ├── EncoderBlock
│   ├── MedicalClassifier (Encoder-Only)
│   ├── MedicalGPT (Decoder-Only)
│   ├── EncoderDecoderBlock
│   ├── MedicalSummarizer (Encoder-Decoder)
│   ├── SimpleTokenizer
│   ├── create_graphs()     # Visualization
│   └── run_experiments()   # Training pipeline
├── README.md
└── .gitignore
```

## 🔬 Technical Flow

### Data Processing Pipeline

1. **Tokenization** (`SimpleTokenizer`):
   - Builds vocabulary from corpus
   - Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
   - Word-level tokenization with padding to `max_len=20`

2. **Encoding**:
   - Converts text → token IDs
   - Pads sequences to uniform length
   - Returns `(batch_size, seq_len)` tensor

### Training Flow

#### Experiment 1: Encoder-Only Classification
```python
# Forward Pass
embeddings = embedding(input_ids)           # (B, L, d_model)
pos_encoded = positional_encoding(embeddings) # (B, L, d_model)
encoded = encoder_stack(pos_encoded)        # (B, L, d_model)
pooled = encoded.mean(dim=1)                # (B, d_model)
logits = classifier(pooled)                 # (B, n_classes)

# Loss Calculation
loss = CrossEntropy(logits, labels)
```

#### Experiment 2: Decoder-Only Generation
```python
# Forward Pass
embeddings = embedding(input_ids)
pos_encoded = positional_encoding(embeddings)
mask = causal_mask(seq_len)                 # Lower triangular
decoded = decoder_stack(pos_encoded, mask)  # (B, L, d_model)
logits = lm_head(decoded)                   # (B, L, vocab_size)

# Loss Calculation (next-token prediction)
loss = CrossEntropy(logits[:, :-1], input_ids[:, 1:])
```

#### Experiment 3: Encoder-Decoder Seq2Seq
```python
# Forward Pass
enc_emb = embedding(source_ids)
enc_out = encoder_stack(pos_enc(enc_emb))   # (B, L_src, d_model)

dec_emb = embedding(target_ids)
dec_out = decoder_stack(
    pos_enc(dec_emb),
    enc_out,
    tgt_mask=causal_mask(L_tgt)
)                                            # (B, L_tgt, d_model)

logits = lm_head(dec_out)                   # (B, L_tgt, vocab_size)

# Loss Calculation
loss = CrossEntropy(logits, target_ids)
```

### Optimization

- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Cross-Entropy
- **Training**: 
  - Encoder-Only: 21 epochs
  - Decoder-Only: 41 epochs
  - Encoder-Decoder: 41 epochs

### Visualization

The `create_graphs()` function generates a 2×2 subplot visualization:
1. Individual loss curve for Encoder-Only
2. Individual loss curve for Decoder-Only
3. Individual loss curve for Encoder-Decoder
4. Combined comparison of all three experiments

Graphs are saved as `training_curves.png` (300 DPI) and displayed interactively.

## 🚀 Usage

### Basic Execution

```bash
python main.py
```

This will:
1. Initialize tokenizer from medical corpus
2. Train all three transformer variants
3. Generate and display training loss curves
4. Save visualization to `training_curves.png`

### Code Example

```python
from main import MedicalClassifier, SimpleTokenizer, medical_corpus

# Initialize components
tokenizer = SimpleTokenizer(medical_corpus)
model = MedicalClassifier(
    vocab_size=len(tokenizer.vocab),
    d_model=128,
    n_layers=2,
    n_heads=4,
    n_classes=2
)

# Encode text
tokens = tokenizer.encode("patient presents with chest pain")
# Use model for inference...
```

## 🧪 Experiments

### Experiment 1: Encoder-Only (Triage)
- **Task**: Binary classification (emergency vs. routine)
- **Labels**: Emergency (1) if contains "emergency" or "chest", else routine (0)
- **Output**: Class probabilities for each medical note
- **Application**: Automated medical triage systems

### Experiment 2: Decoder-Only (Generation)
- **Task**: Next-token prediction / language modeling
- **Objective**: Predict the next word given previous context
- **Training**: Autoregressive self-supervision
- **Application**: Medical text generation, clinical note completion

### Experiment 3: Encoder-Decoder (Simplification)
- **Task**: Sequence-to-sequence mapping
- **Specific Task**: Full sentence → First 8 tokens (simplification)
- **Input**: Full medical note
- **Output**: Simplified/summarized version
- **Application**: Medical text simplification, summarization

## 📦 Dependencies

```python
torch>=1.9.0          # Deep learning framework
matplotlib>=3.3.0     # Visualization
```

Install dependencies:
```bash
pip install torch matplotlib
```

## 🔍 Key Design Decisions

1. **From Scratch Implementation**: No reliance on pre-built transformer libraries (e.g., `transformers`) for educational clarity

2. **Residual Connections**: All sub-layers use residual connections with pre-normalization (LayerNorm before sub-layer)

3. **Masking Strategy**:
   - Encoder: No masking (bidirectional)
   - Decoder self-attention: Causal mask (unidirectional)
   - Decoder cross-attention: No mask (full encoder access)

4. **Pooling Strategy**: Encoder-only uses mean pooling for sequence-level representation

5. **Vocabulary**: Simple word-level tokenization; can be extended with BPE/WordPiece

## 📊 Model Specifications

| Model | d_model | n_heads | n_layers | d_ff | Parameters |
|-------|---------|---------|----------|------|------------|
| MedicalClassifier | 128 | 4 | 2 | 512 | ~50K |
| MedicalGPT | 128 | 4 | 2 | 512 | ~80K |
| MedicalSummarizer | 128 | 4 | 2 (enc/dec) | 512 | ~130K |

*Note: Actual parameter counts depend on vocabulary size*

## 🎓 Educational Value

This implementation serves as an excellent learning resource for:
- Understanding transformer architecture fundamentals
- Comparing encoder-only, decoder-only, and encoder-decoder designs
- Learning attention mechanisms and masking strategies
- Practical PyTorch implementation patterns
- Medical NLP applications

## 📝 Notes

- Models are trained on CPU for demonstration purposes
- Medical corpus is small-scale for educational use
- For production use, consider larger datasets, GPU acceleration, and more sophisticated tokenization
- The implementation follows the original "Attention is All You Need" paper architecture closely

---

**Author**: Educational implementation for understanding transformer architectures  
**License**: Educational use
