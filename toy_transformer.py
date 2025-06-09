import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---- Hyperparameters ----
vocab = {'<pad>': 0, 'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5}
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)
d_model = 8  # Embedding dimension
max_len = 10

# ---- Toy Input ----
sentence = "the cat sat on the mat"
tokens = [vocab[word] for word in sentence.split()]
input_tensor = torch.tensor(tokens).unsqueeze(0)  # shape: [1, seq_len]

# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---- Transformer ----
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embedding(x)                    # [1, seq_len, d_model]
        x = self.pos_encoding(x)                 # add positional info
        attn_output, attn_weights = self.self_attn(x, x, x)  # self-attention
        x = self.fc(attn_output)                 # simple feed-forward
        return x, attn_weights

# ---- Run the Model ----
model = TinyTransformer()
output, attn_weights = model(input_tensor)

print("Output shape:", output.shape)
print("Attention Weights shape:", attn_weights.shape)
print("Attention Weights (rounded):")
print(attn_weights[0].detach().numpy().round(2))
