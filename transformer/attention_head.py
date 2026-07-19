import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_input, d_k, device):
        super().__init__()
        self.device = device
        self.attention_heads = [AttentionHead(d_input, d_k, self.device) for _ in range(n_heads)]
        self.W_o = nn.Linear(n_heads * d_input, d_input, device=self.device)

    def forward(self, X):
        outputs = torch.cat([head(X) for head in self.attention_heads], dim=-1)

        return self.W_o(outputs) + X #residuals

class AttentionHead(nn.Module):
    def __init__(self, d_input, d_out, device):
        super().__init__()
        self.d = d_out
        self.device = device
        self.W_q = nn.Linear(d_input, d_out, device=self.device)
        self.W_k = nn.Linear(d_input, d_out, device=self.device)
        self.W_v = nn.Linear(d_input, d_input, device=self.device)

    def forward(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        matrix = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d, dtype=torch.float32, device=self.device))
        
        mask = torch.full((X.size(1), X.size(1)), float('-inf'), device=self.device)
        mask = torch.triu(mask, diagonal=1)

        matrix = matrix + mask

        matrix = torch.softmax(matrix, dim=1)
        delta_e = torch.matmul(matrix, V)

        new_x = X + delta_e
        return new_x

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_input, device):
        super().__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(d_input, 2048, device=self.device),
            nn.ReLU(),
            nn.Linear(2048, d_input, device=self.device),
        )

    def forward(self, X):
        return self.network(X) + X #resisuals
