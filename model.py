# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from module import ESM1bLayerNorm, RobertaLMHead, TransformerLayer


class ESMDiffusion(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        max_seq_len: int = 2048,
        T: int = 100,
        token_dropout: bool = True,
        use_checkpoint: bool = True,
        padding_idx: int = 0,
        mask_idx: int = 2,
        cls_idx: int = 2,
        eos_idx: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.max_seq_len = max_seq_len
        self.T = T
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.token_dropout = token_dropout
        self.use_checkpoint = use_checkpoint

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        # Positional embeddings
        self.pos_emb = nn.Embedding(self.max_seq_len, self.embed_dim)

        # Domain embeddings for conditioning
        self.domain_emb = nn.Embedding(3, self.embed_dim)

        # Timestep conditioning for diffusion
        self.time_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.input_layer_norm = ESM1bLayerNorm(self.embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.vocab_size,
            weight=self.embed_tokens.weight,
        )

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal timestep embeddings for diffusion."""
        # t: (B,)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        sin = torch.sin(args)
        cos = torch.cos(args)
        emb = torch.cat([sin, cos], dim=-1)  # (B, 2*half)
        if emb.size(1) < self.embed_dim:
            pad = torch.zeros(emb.size(0),
                              self.embed_dim - emb.size(1),
                              device=t.device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb

    def forward(self, x_t, t, domain, positions=None, repr_layers=[], need_head_weights=False,
                seq_boundaries=None, max_seqlen=None):
        """
        Forward pass for diffusion language model.

        Args:
            x_t: Token indices at timestep t (B, L)
            t: Timestep tensor (B,)
            domain: Domain indices for conditioning (B,) or (B, L)
            positions: Optional position indices (B, L)
            repr_layers: List of layer indices to return representations from
            need_head_weights: Whether to return attention weights
            seq_boundaries: List of sequence boundary indices for each batch item (for FlashAttention)
            max_seqlen: Maximum sequence length in batch for FlashAttention
        """
        assert x_t.ndim == 2
        B, L = x_t.shape

        padding_mask = x_t.eq(self.padding_idx)  # B, L

        # Token embeddings
        x = self.embed_scale * self.embed_tokens(x_t)

        # Token dropout (masking strategy) - zero out MASK token embeddings
        # This forces the model to predict from context only, not from a learned mask embedding
        if self.token_dropout and self.training and self.mask_idx is not None:
            x = x.masked_fill((x_t == self.mask_idx).unsqueeze(-1), 0.0)

        # Positional embeddings
        if positions is None:
            pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, L)
        else:
            pos_ids = positions
        pos = self.pos_emb(pos_ids)

        # Timestep conditioning
        t_emb = self.time_mlp(self.get_timestep_embedding(t))

        # Domain conditioning
        d_emb = self.domain_emb(domain) if domain.dim() == 1 else self.domain_emb(domain)

         # Combine embeddings
        if domain.dim() == 1:
            cond = (t_emb + d_emb).unsqueeze(1)
            h = x + pos + cond
        else:
            t_cond = t_emb.unsqueeze(1).expand(-1, L, -1)
            h = x + pos + d_emb + t_cond

        # ✅ ADD THIS NEW LINE:
        h = self.input_layer_norm(h)

        # Apply padding mask
        if padding_mask is not None:
            h = h * (1 - padding_mask.unsqueeze(-1).type_as(h))
        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = h

        if need_head_weights:
            attn_weights = []

        # (B, L, E) => (L, B, E)
        h = h.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        # Transformer layers with gradient checkpointing
        for layer_idx, layer in enumerate(self.layers):
            if self.training and self.use_checkpoint:
                # Note: checkpoint doesn't support keyword args well, so we use a wrapper
                def checkpoint_wrapper(h_in, mask, need_weights):
                    return layer(
                        h_in,
                        self_attn_padding_mask=mask,
                        need_head_weights=need_weights,
                        seq_boundaries=seq_boundaries,
                        max_seqlen=max_seqlen,
                    )
                h, attn = checkpoint(
                    checkpoint_wrapper,
                    h,
                    padding_mask,
                    need_head_weights,
                    use_reentrant=False
                )
            else:
                h, attn = layer(
                    h,
                    self_attn_padding_mask=padding_mask,
                    need_head_weights=need_head_weights,
                    seq_boundaries=seq_boundaries,
                    max_seqlen=max_seqlen,
                )

            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = h.transpose(0, 1)
            if need_head_weights:
                # (H, B, L, L) => (B, H, L, L)
                attn_weights.append(attn.transpose(1, 0))

        h = self.emb_layer_norm_after(h)
        h = h.transpose(0, 1)  # (L, B, E) => (B, L, E)

        # Last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = h

        logits = self.lm_head(h)

        result = {"logits": logits, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x num_layers x H x L x L
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions

        return result
