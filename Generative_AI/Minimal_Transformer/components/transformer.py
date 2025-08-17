import torch
import torch.nn as nn
import torch.func as func
import encoder
import decoder

class Transformer(nn.Module):
    def __init__(self,source_vocab,target_vocab,d_model=128,n_heads=4,d_ff=256,n_layers=2,max_length=100):
        super().__init__()
        self.encoder = encoder.Encoder(source_vocab,d_model,n_heads,d_ff,n_layers,max_length)
        self.decoder = decoder.Decoder(target_vocab,d_model,n_heads,d_ff,n_layers,max_length)
        
    def make_source_mask(self,source):
        return (source!=0).unsqueeze(1).unsqueeze(2)
    
    def make_target_mask(self,target):
        batch_size,sequence_length=target.shape
        mask=torch.tril(torch.ones(sequence_length,sequence_length)).unsqueeze(0).unsqueeze(0)
        
        return mask.to(target.device)
    
    def forward(self,source,target):
        source_mask=self.make_source_mask(source)
        target_mask=self.make_target_mask(target)
        encode_out=self.encoder(source,source_mask)
        output=self.decoder(target,encode_out,source_mask,target_mask)
        
        return output