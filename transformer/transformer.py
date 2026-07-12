import torch.nn as nn
import torch
from attention_head import AttentionHeads

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential([
            Tokenizer(),
            PositionalEncoding(),
            AttentionHeads(),
            ResdiualConnections(),
            nn.LayerNorm1d(),
            FeedForwardNetwork(),
            ResdiualConnections(),
            nn.LayerNorm1d(),      
        ])

        #this takes in raw symnols
        self.decoder = (
            Tokenizer(),
            Embedder(),
            PositionalEncoding(),
            MaskedAttentionHeads(),
        )
        #somewhere here i need to feed in encoder inputs
        self.decoder_encoder = nn.Sequential([
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            nn.Linear(),
            nn.Softmax()
        ])
