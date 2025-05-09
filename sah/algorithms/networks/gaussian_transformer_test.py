import torch

from sah.algorithms.networks.gaussian_transformer import GaussianTransformer


def test_gaussian_transformer_mean_identity():
    """Verify that a freshly initialized GaussianTransformer returns the input as the mean
    output."""
    embed_dim = 8
    num_heads = 2
    hidden_dim = 16
    num_layers = 3

    model = GaussianTransformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)

    mu, sigma = model(x)

    assert torch.allclose(mu, x, atol=1e-6), \
        f"Expected mean output to equal input, but got max diff {torch.abs(mu - x).max()}"

    assert torch.all(sigma > 0), "Expected all sigma values to be positive"
