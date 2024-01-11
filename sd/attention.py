import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: [batch_size, seq_len, dim]

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)

        # [batch_size, seq_len, dim] -> [batch_size, seq_len, dim * 3] -> 3 tensors of shape [batch_size, seq_len, dim]
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # same as applying three different in_projections combined in one big matrix

        # [batch_size, seq_len, dim] -> [batch_size, seq_len, n_heads, dim / n_heads] -> [batch_size, n_heads, seq_len, dim / n_heads]
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # [batch_size, n_heads, seq_len, seq_len]
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask where the upper triangle (above the principle diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=-1)

        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, dim / n_heads] -> [batch_size, num_heads, seq_len, dim / n_heads]
        output = weight @ v

        # [batch_size, num_heads, seq_len, dim / n_heads] -> [batch_size, seq_len, num_heads, dim / num_heads]
        output = output.transpose(1, 2)

        # [batch_size, seq_len, num_heads, dim / num_heads] -> [batch_size, seq_len, dim]
        output.reshape(input_shape)

        # [batch_size, seq_len, dim]
        output = self.out_proj(output)

        return output