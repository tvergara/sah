import hydra_zen
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 20,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        pad_token_id: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

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
        # self.lm_head.weight = self.tok_emb.weight

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape

        pad_mask = None
        if self.pad_token_id is not None:
            pad_mask = input_ids.eq(self.pad_token_id)  # (batch, seq_len)

        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

        x = self.tok_emb(input_ids)
        # x = self.dropout(x)
        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )

        logits = self.lm_head(x)       # (batch, seq_len, vocab_size)
        return logits

    def forward_with_continuous_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, vocab_size = inputs.shape
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

        x = inputs @ self.tok_emb.weight
        x = self.encoder(
            x,
            mask=causal_mask,
        )

        logits = self.lm_head(x)       # (batch, seq_len, vocab_size)
        return logits

    def freeze_up_to(self,  probe_start: int) -> None:
        ordered_layers = [self.tok_emb] + list(self.encoder.layers)

        if not 0 <= probe_start <= len(ordered_layers):
            raise ValueError(f"probe_start={probe_start} out of range (0-{len(ordered_layers)})")

        for idx, submodule in enumerate(ordered_layers):
            trainable = idx >= probe_start          # freeze if index < probe_start
            submodule.requires_grad_(trainable)

@hydra_zen.hydrated_dataclass(
    target=Transformer,
    unsafe_hash=True,
    populate_full_signature=True,
)
class TransformerConfig:
    d_model: int = 50
    dim_feedforward: int = 200
    n_heads: int = 5
    n_layers: int = 5
