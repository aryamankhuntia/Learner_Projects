import torch
import torch.nn as nn
import torch.func as func
import positional_encoding as pe
import transformer_block as tb

class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,d_ff,n_layers,max_length=100,dropout=0.1):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,d_model)
        self.pe=pe.PositionalEncoding(d_model,max_length)
        self.layers=nn.ModuleList([
            tb.DecoderBlock(d_model,n_heads,d_ff,dropout) for i in range(n_layers)
        ])
        self.fc_out=nn.Linear(d_model,vocab_size)
        
    def forward(self,x,encode_out,source_mask=None,target_mask=None):
        x=self.embed(x)
        x=self.pe(x)
        for layer in self.layers:
            x=layer(x,encode_out,source_mask,target_mask)
            
        return self.fc_out(x)