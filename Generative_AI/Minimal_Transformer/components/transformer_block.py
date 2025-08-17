import torch
import torch.nn as nn
import torch.nn.functional as func
import attention as attn
import feedforward as ff

class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.attention=attn.MultiHeadAttention(d_model,n_heads)
        self.ff=ff.FeedForward(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,mask=None):
        attention_out=self.attention(x,x,x,mask)
        x=self.norm1(x+self.dropout(attention_out))
        
        ff_out=self.ff(x)
        x=self.norm2(x+self.dropout(ff_out))
        
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.self_attention=attn.MultiHeadAttention(d_model,n_heads)
        self.cross_attention=attn.MultiHeadAttention(d_model,n_heads)
        self.ff=ff.FeedForward(d_model,d_ff,dropout)
        
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,encode_out,source_mask=None,target_mask=None):
        x_dash=self.self_attention(x,x,x,target_mask)
        x=self.norm1(x+self.dropout(x_dash))
        
        x_dash=self.cross_attention(x,encode_out,encode_out,source_mask)
        x=self.norm2(x+self.dropout(x_dash))
        
        x_dash=self.ff(x)
        x=self.norm3(x+self.dropout(x_dash))
        
        return x