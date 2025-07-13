import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import inspect

class LayerNornm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(ndim))
        self.bias = nn.parameter(torch.zeros(ndim) if bias else None)

    def forward(self,input):
        return F.layer_norm(input, self.weights.shape , self.weights, self.bias, eps=1e-5)
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd , 3*config.n_embd  ,bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_haed = config.n_haed #12
        self.n_embd = config.n_embd #768
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            
    
            
    def forward(self,x):
        B,T,C = x.size()

        q,k,v = self.c_attn(x).split(self.n_embd,dim=2) #split the 3 dimension  because we did that 3*n+embd
        # q,k,v shape is (B,T,C) where C = n_embd
        # We will transpose the k,q,v from [B, T, n_head,head_dim] to [B, n_head, T , head_dim]
        k = k.view(B,T,self.n_haed,C // self.n_haed).transpose(1,2)
        q = q.view(B,T,self.n_haed,C // self.n_haed).transpose(1,2)
        v = v.view(B,T,self.n_haed,C // self.n_haed).transpose(1,2)

        #Compute attention scores → apply softmax → optionally mask → apply dropout → multiply by values.
        """
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v , 
                                                             attn_mask=None,
                                                             dropout_p=self.dropout if self.training else 0 , 
                                                             is_causal= True)
        
        """
        y = self.scaled_dot_product_attention(q,k,v)
        y =y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(y)
        return y
    


def scaled_dot_product_attention(q,k,v,is_causal=True, dropout_p=0.0,training=False):
    """
    q: query tensor of shape (B, n_head, T, head_dim)
    k.transpose(-2, -1): key tensor of shape (B, n_head, head_dim, T)

    torch.matmul(q, k.transpose(-2, -1)) computes the dot product between queries and keys.
    The result is a tensor of shape (B, n_head, T, T) representing the attention scores.
    """
    matmul = torch.matmul(q, k.transpose(-2, -1)) 
    d_k = k.size(-1) # head_dim
    scaled_attention_logits = matmul / math.sqrt(d_k)
    if is_causal:
        seq_len = q.size(-2) # T
        causal_mask = torch.tril(torch.ones(seq_len,seq_len,device=q.device , dtype=torch.bool)) # (seq_len, seq_len) Is a lower tringular matrix 
        """
        if the mask is True, replace the value with -inf, else keep the original value
        but we need to do this in the ~causal_mask because we want to mask the upper triangular 
        part and masked_fill will fill the True part with -inf 
        (basicly we want to replace the false part with -inf but because of the way masked_fill works, we need to invert the mask)
        """
        scaled_attention_logits = scaled_attention_logits.masked_fill(~causal_mask, float('-inf')) 
    
    attention_weights = F.softmax(scaled_attention_logits, dim=-1) #apply softmax to the last dimension to get attention weights
    if training and dropout_p > 0:
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=training)

    
    output = torch.matmul(attention_weights, v) # (B, n_head, T, head_dim)
    return output 