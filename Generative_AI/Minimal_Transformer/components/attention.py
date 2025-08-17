import torch
import torch.nn as nn
import torch.nn.functional as func
from .feedforward import FeedForward

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale=d_k**0.5

    def forward(self,Q,K,V,mask=None):
        scores=torch.matmul(Q,K.transpose(-2,-1))/self.scale

        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))

        attention=func.softmax(scores,dim=-1)
        output=torch.matmul(attention,V)
        
        return output, attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
        
    def forward(self,Q,K,V,mask=None):
        batchsize=Q.size(0)
        Q=self.w_q(Q)
        K=self.w_k(K)
        V=self.w_v(V)
        Q=Q.view(batchsize,-1,self.n_heads,self.d_k).transpose(1,2)
        K=K.view(batchsize,-1,self.n_heads,self.d_k).transpose(1,2)
        V=V.view(batchsize,-1,self.n_heads,self.d_k).transpose(1,2)
        
        output,attention=self.attention(Q,K,V,mask)
        output=output.transpose(1,2).contiguous().view(batchsize,-1,self.n_heads*self.d_k)
        
        return self.w_o(output)
    
class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        self.attention=MultiHeadAttention(d_model,n_heads)
        self.ff=FeedForward(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,mask=None):
        attention_out=self.attention(x,x,x,mask)
        x=self.norm1(x+self.dropout(attention_out))
        
        ff_out=self.ff(x)
        x=self.norm2(x+self.dropout(ff_out))
        
        return x

    
    
    

