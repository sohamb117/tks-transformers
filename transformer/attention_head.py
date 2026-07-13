import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_input, d_k):
        super().__init__()
        self.attention_heads = [AttentionHead(d_input, d_k) for _ in range(n_heads)]
        self.W_o = nn.Linear(n_heads * d_input, d_input)

    def forward(self, X):
        outputs = torch.empty()
        for head in self.attention_heads:
            outputs = torch.vstack(outputs, head(X))

        return self.W_o(outputs) + X #residuals

class AttentionHead(nn.Module):
    def _init_(self, d_input, d_out:int=128):
        super().__init__()
        self.d = d_out
        self.W_q = nn.Linear(d_input, d_out)
        self.W_k = nn.Linear(d_input, d_out)
        self.W_v = nn.Linear(d_input, d_input)

    def forward(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        matrix = torch.matmul(Q, K.T) / torch.sqrt(self.d)
        matrix = nn.Softmax(dim=1)(matrix)
        delta_e = torch.matmul(matrix, V)

        new_x = X + delta_e
        return new_x

class FeedForwardNetwork():
    def __init__(self, d_input):
        self.network = nn.Sequential(
            nn.Linear(d_input, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_input),
        )

    def forward(self, X):
        return self.network(X) + X #resisuals
