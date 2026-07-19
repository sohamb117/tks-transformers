import torch.nn as nn
import torch
from transformer.attention_head import MultiHeadedAttention, FeedForwardNetwork
from transformer.positional_encodings import PositionalEncodings

class Transformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.proj = nn.Linear(512, 50_258, device=self.device)
        self.decoder = nn.Sequential(
            nn.Embedding(num_embeddings=50_258, embedding_dim=512, device=self.device),
            PositionalEncodings(512, device=self.device),      
            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512),
                  
            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512),

        )

    def forward(self, X):
        last_token = self.decoder(X)
        output = self.proj(last_token)
        return output
    
    def tokenize_input(self, in_str):
        return self.tkn.encode(in_str)

    def most_likely_token(self, in_str):
        return in_str.index(max(in_str))

    def generate_output(self):
        ctx_encoded = self.tokenize_input(self.ctx)
        for i in range(0, 100):
            forwarded = self.forward(ctx_encoded)
            mlt = self.most_likely_token(forwarded)
            ctx_encoded.append(mlt)
            ctx_encoded = ctx_encoded[1:]
        return self.tkn.decode(ctx_encoded[-100:])
    
