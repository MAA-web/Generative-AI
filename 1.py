import pandas as pd
import torch

from main import MedicalGPT, SimpleTokenizer

def train_gpt_on_abstracts(csv_path, tokenizer, model):
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    # 2. Select the abstract column and take a small subset (e.g., 500 rows) for CPU
    # We filter out very long abstracts to keep training fast
    abstracts = df['medical_abstract'].str.slice(0, 300).tolist()[:5] 
    
    # 3. Tokenize
    encoded_data = [tokenizer.encode(text, max_len=128) for text in abstracts]
    input_ids = torch.tensor(encoded_data) # Shape: [500, 128]
    
    # 4. Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    model.train()
    
    print(f"Training GPT on {len(abstracts)} abstracts...")
    
    for epoch in range(30):
        optimizer.zero_grad()
        
        # Forward pass
        # model(input_ids) returns [batch, seq_len, vocab_size]
        logits = model(input_ids)
        
        # Shift inputs for Next-Token Prediction
        # If input is [A, B, C, D], logits for [A, B, C] should predict [B, C, D]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("GPT Training Complete.")






    # --- Example Integration ---

# Load CSV to build vocabulary
df = pd.read_csv('medical_tc_train.csv')
all_text = df['medical_abstract'].tolist()

# 1. Initialize Tokenizer with the real medical text
tokenizer = SimpleTokenizer(all_text[:1000]) # Use first 1000 rows to build vocab
vocab_size = len(tokenizer.vocab)

# 2. Initialize the Decoder-Only Model
# d_model=128, layers=2, heads=4
model_dec = MedicalGPT(vocab_size, 128, 2, 4)

# 3. Run Training
train_gpt_on_abstracts('medical_tc_train.csv', tokenizer, model_dec)