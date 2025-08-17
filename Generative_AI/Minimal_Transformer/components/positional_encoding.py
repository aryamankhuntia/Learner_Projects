import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_length=5000):
        super().__init__()
        pe = torch.zeros(max_length,d_model)
        position=torch.arange(0,max_length,dtype=torch.float).unsqueeze(1)
        denom=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*denom)
        pe[:,1::2]=torch.cos(position*denom)
        
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x=x+self.pe[:,:x.size(1)]
        return x