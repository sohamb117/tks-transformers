import torch.nn as nn
import torch.nn.functional as F
import torch

class AttentionHeads(nn.Module):
    def __init__(self):
        self.attention_heads = 

    def forward(self):
        pass

class AttentionHead():
    def _init_(self, d_input, d_out:int=128):
        self.W_q = torch.empty()
        self.W_k
        self.W_v

    def forward(self, X):
        Q = torch.matmul(self.W_q, X)
        K = torch.matmul(self.W_k, X)
        V = torch.matmul(self.W_v, X)
        matrix = torch.matmul(Q, K.T) / torch.sqrt(self.d)
        delta_e = torch.matmul(matrix, V)

        new_x = X + delta_e


class ResidualConnections():
    pass

class FeedForwardNetwork():
    pass
