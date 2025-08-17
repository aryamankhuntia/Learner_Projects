import torch
import torch.nn as nn
import torch.nn.functional as func
import positional_encoding as pe
import transformer_block as tb

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,d_ff,n_layers,max_length=100,dropout=0.1):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,d_model)
        self.pe=pe.PositionalEncoding(d_model,max_length)
        self.layers=nn.ModuleList([
            tb.TransformerBlock(d_model,n_heads,d_ff,dropout) for i in range(n_layers)
        ])
        
    def forward(self,x,mask=None):
        x=self.embed(x)
        x=self.pe(x)
        for layer in self.layers:
            x=layer(x,mask)
        return x