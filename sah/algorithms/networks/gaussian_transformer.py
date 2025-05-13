import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """A single Transformer block with multi-head self-attention and a feed-forward network.

    Args:
        embed_dim (int): Dimension of input embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of the hidden layer in the feed-forward network.
        dropout (float): Dropout probability.
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.attn.out_proj.weight.data.zero_()
        self.attn.out_proj.bias.data.zero_()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.linear2.weight.data.zero_()
        self.linear2.bias.data.zero_()
        self.ff = nn.Sequential(self.linear1, nn.GELU(), self.linear2)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        normalized_x = self.norm1(x)
        attn_out, _ = self.attn(
            normalized_x, normalized_x, normalized_x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x


class GaussianTransformer(nn.Module):
    """Transformer that predicts a diagonal Gaussian (mean and per-dimension sigma) for each token.

    Args:
        embed_dim (int): Size of input token embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Hidden dimension of feed-forward layers.
        num_layers (int): Number of Transformer blocks.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.mean_head = nn.Linear(embed_dim, embed_dim)
        nn.init.eye_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.sigma_head = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, embed_dim).
            mask (Tensor, optional): Key padding mask of shape (batch_size, seq_len).

        Returns:
            mu (Tensor): Means of shape (batch_size, seq_len, embed_dim).
            sigma (Tensor): Positive sigmas of shape (batch_size, seq_len, embed_dim).
        """
        for layer in self.layers:
            x = layer(x, mask)

        mu = self.mean_head(x)
        raw_sigma = self.sigma_head(x)
        sigma = self.sigmoid(raw_sigma)
        return mu, sigma
