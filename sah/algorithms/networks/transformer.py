
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        pad_token_id: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        pad_mask = None
        if self.pad_token_id is not None:
            pad_mask = input_ids.eq(self.pad_token_id)  # (batch, seq_len)

        causal_mask = self.causal_mask[:seq_len, :seq_len]

        x = self.tok_emb(input_ids)
        x = self.dropout(x)

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )

        logits = self.lm_head(x)       # (batch, seq_len, vocab_size)
        return logits
