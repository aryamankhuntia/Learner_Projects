import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from utils import make_source_mask, make_target_mask

class Transformer(nn.Module):
    def __init__(self,source_vocab,target_vocab,d_model=128,n_heads=4,d_ff=256,n_layers=2,max_length=100):
        super().__init__()
        self.encoder = Encoder(source_vocab,d_model,n_heads,d_ff,n_layers,max_length)
        self.decoder = Decoder(target_vocab,d_model,n_heads,d_ff,n_layers,max_length)
    
    def forward(self,source,target):
        source_mask=make_source_mask(source)
        target_mask=make_target_mask(target)
        encode_out=self.encoder(source,source_mask)
        output=self.decoder(target,encode_out,source_mask,target_mask)
        
        return output