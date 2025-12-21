from building_blocks import MultiHeadAttention
from building_blocks import FeedForward
from building_blocks import PositionalEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# ==========================================
# 2. TRANSFORMER VARIANTS
# ==========================================

# --- VARIANT 1: ENCODER ONLY (BERT-Style) ---
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class MedicalClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            x = layer(x)
        return self.fc(x.mean(dim=1)) 





# --- VARIANT 2: DECODER ONLY (GPT-Style) ---
class MedicalGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        # Create Causal Mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        x = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc(x)






# --- VARIANT 3: ENCODER-DECODER (T5-Style) ---
class EncoderDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention (with causal mask)
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        # Cross attention (queries from decoder, keys/values from encoder)
        x = self.norm2(x + self.cross_attn(x, enc_output, enc_output, src_mask))
        x = self.norm3(x + self.ff(x))
        return x

class MedicalSummarizer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([EncoderDecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # Encode
        enc_out = self.pos_enc(self.embedding(src))
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # Decode
        tgt_seq_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).unsqueeze(0).unsqueeze(0)
        
        dec_out = self.pos_enc(self.embedding(tgt))
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
        
        return self.fc(dec_out)






# ==========================================
# 3. TOKENIZER & DATASET
# ==========================================

class SimpleTokenizer:
    def __init__(self, corpus):
        self.vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        words = sorted(list(set(" ".join(corpus).split())))
        for i, word in enumerate(words):
            self.vocab[word] = i + 4
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text, max_len=20):
        tokens = [self.vocab.get(w, 3) for w in text.split()]
        tokens = tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

medical_corpus = [
    "patient presents with acute headache and nausea",
    "prescribe ibuprofen for minor muscle pain",
    "emergency trauma in lower abdomen area",
    "routine checkup for diabetic patient management",
    "severe chest pain radiating to left arm",
    "apply bandage to the clean wound",
    "symptoms include high fever and dry cough",
    "administer saline drip for dehydration"
]

# ==========================================
# 4. GRAPH GENERATION
# ==========================================

def create_graphs(epochs_enc, losses_enc, epochs_dec, losses_dec, epochs_seq2seq, losses_seq2seq):
    """Create visualization graphs for all three experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Transformer Variants: Training Loss Curves', fontsize=16, fontweight='bold')
    
    # Graph 1: Encoder-Only (Triage)
    axes[0, 0].plot(epochs_enc, losses_enc, 'b-', linewidth=2, label='Encoder-Only')
    axes[0, 0].set_title('Experiment 1: Encoder-Only (Triage)', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Graph 2: Decoder-Only (Generation)
    axes[0, 1].plot(epochs_dec, losses_dec, 'r-', linewidth=2, label='Decoder-Only')
    axes[0, 1].set_title('Experiment 2: Decoder-Only (Generation)', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Graph 3: Encoder-Decoder (Simplification)
    axes[1, 0].plot(epochs_seq2seq, losses_seq2seq, 'g-', linewidth=2, label='Encoder-Decoder')
    axes[1, 0].set_title('Experiment 3: Encoder-Decoder (Simplification)', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Graph 4: Combined comparison
    axes[1, 1].plot(epochs_enc, losses_enc, 'b-', linewidth=2, label='Encoder-Only', alpha=0.8)
    axes[1, 1].plot(epochs_dec, losses_dec, 'r-', linewidth=2, label='Decoder-Only', alpha=0.8)
    axes[1, 1].plot(epochs_seq2seq, losses_seq2seq, 'g-', linewidth=2, label='Encoder-Decoder', alpha=0.8)
    axes[1, 1].set_title('All Experiments Comparison', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("\nGraphs saved as 'training_curves.png'")
    plt.show()

# ==========================================
# 5. RUNNING EXPERIMENTS
# ==========================================

def run_experiments():
    tokenizer = SimpleTokenizer(medical_corpus)
    vocab_size = len(tokenizer.vocab)
    device = torch.device("cpu")

    print(f"Vocab Size: {vocab_size}")
    
    # Pre-process data
    src_data = torch.stack([torch.tensor(tokenizer.encode(s)) for s in medical_corpus])

    # --- 1. Encoder-Only ---
    print("\n--- Experiment 1: Encoder-Only (Triage) ---")
    model_enc = MedicalClassifier(vocab_size, 128, 2, 4, 2)
    labels = torch.tensor([1 if ("emergency" in s or "chest" in s) else 0 for s in medical_corpus])
    optimizer = torch.optim.Adam(model_enc.parameters(), lr=0.001)
    
    losses_enc = []
    epochs_enc = []

    for epoch in range(21):
        output = model_enc(src_data)
        loss = F.cross_entropy(output, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses_enc.append(loss.item())
        epochs_enc.append(epoch)
        if epoch % 10 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # --- 2. Decoder-Only ---
    print("\n--- Experiment 2: Decoder-Only (Generation) ---")
    model_dec = MedicalGPT(vocab_size, 128, 2, 4)
    optimizer = torch.optim.Adam(model_dec.parameters(), lr=0.001)
    
    losses_dec = []
    epochs_dec = []

    for epoch in range(41):
        output = model_dec(src_data) 
        # Shift targets for next-token prediction
        loss = F.cross_entropy(output[:, :-1, :].reshape(-1, vocab_size), src_data[:, 1:].reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses_dec.append(loss.item())
        epochs_dec.append(epoch)
        if epoch % 20 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # --- 3. Encoder-Decoder ---
    print("\n--- Experiment 3: Encoder-Decoder (Simplification) ---")
    model_seq2seq = MedicalSummarizer(vocab_size, 128, 2, 4)
    optimizer = torch.optim.Adam(model_seq2seq.parameters(), lr=0.001)
    
    # Task: Map full sentence to first 5 words (Simplification)
    # We use .clone().detach() to avoid stride issues
    target_data = src_data[:, :8].clone().detach() 
    
    losses_seq2seq = []
    epochs_seq2seq = []

    for epoch in range(41):
        optimizer.zero_grad()
        output = model_seq2seq(src_data, target_data)
        # Use .reshape instead of .view to avoid the error you saw
        loss = F.cross_entropy(output.reshape(-1, vocab_size), target_data.reshape(-1))
        loss.backward(); optimizer.step()
        losses_seq2seq.append(loss.item())
        epochs_seq2seq.append(epoch)
        if epoch % 20 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("\nAll experiments completed successfully on CPU.")
    
    # Generate graphs
    create_graphs(epochs_enc, losses_enc, epochs_dec, losses_dec, epochs_seq2seq, losses_seq2seq)

if __name__ == "__main__":
    run_experiments()