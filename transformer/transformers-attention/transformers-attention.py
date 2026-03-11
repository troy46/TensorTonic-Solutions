import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = Q.shape[2]
    score = torch.matmul(Q, K.transpose(1,2))/math.sqrt(d_k)
    attention = torch.softmax(score, dim=2)
    return torch.matmul(attention,V)