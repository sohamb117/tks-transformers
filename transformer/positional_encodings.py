import torch.nn as nn
import torch

class PositionalEncodings(nn.Module):
    def __init__(self, d_in, device):
        super().__init__()
        self.d = d_in
        self.device = device
    
    def forward(self, X:torch.tensor):
        def pe(pos, i):
            angle = pos / (10000 ** ((i//2 * 2) / 10))
            return torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))

        dim = X.size(1)

        pos = torch.arange(dim, device=self.device).unsqueeze(1)
        i = torch.arange(self.d, device=self.device)

        table = pe(pos, i)
        return table + X


        
