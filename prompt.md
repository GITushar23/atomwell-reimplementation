# Goal (clear and precise)

You want a **diffusion model** over sequences (proteins / SMILES / DNA) that:

* learns **unconditional generation**
* learns **conditional generation** via motif conditioning
* works with **batched sequence packing + FlashAttention**
* has **no cross-sequence interaction**

✅ This is achieved by **global motif conditioning during training**.

---

# What is Global Motif Conditioning?

Instead of using per-token masks, we:

1. **Extract a motif** (substring) from each sequence
2. **Encode the motif** into a fixed-size embedding vector
3. **Add this embedding globally** to the model (like timestep or domain conditioning)
4. **Train on the full sequence** (no loss masking needed)

### Key Difference from Partial Conditioning:

| Approach | How it works | What model learns | Guarantees |
|----------|-------------|-------------------|------------|
| **Partial conditioning** (cond_mask) | Some tokens stay clean, others noised | "Fill in the gaps around fixed tokens" | Exact motif placement |
| **Global conditioning** (motif embedding) | Motif encoded as vector, added to all positions | "Generate sequences compatible with this motif" | Semantic similarity, soft constraints |

**We use: Global conditioning** ✅

---

# Why Global Conditioning is Better for Your Use Case

✅ **Simpler implementation** - no per-token masking complexity
✅ **More flexible** - model learns semantic relationships, not just exact matches
✅ **Better for properties** - condition on "sequences like those with motif X"
✅ **Variable-length motifs** - works naturally with any substring length
✅ **Train on full sequences** - full cross-entropy loss, no masking

---

# Training Setup (End-to-End)

## STEP 1 — Dataset sequences (with EOS)

```
S0: A G M L K <EOS>          (length 6)
S1: Q W E R <EOS>            (length 5)
S2: P T Y <EOS>              (length 4)
S3: H I K L M N <EOS>        (length 7)
```

---

## STEP 2 — Extract motifs during training

For each sequence, randomly extract a **substring** to condition on:

### S0 — extract prefix
```
Sequence: A G M L K <EOS>
Motif:    A G              (first 2 tokens)
```

### S1 — extract internal motif
```
Sequence: Q W E R <EOS>
Motif:    W E              (middle 2 tokens)
```

### S2 — unconditional (no motif)
```
Sequence: P T Y <EOS>
Motif:    (empty)           (unconditional generation)
```

### S3 — extract suffix
```
Sequence: H I K L M N <EOS>
Motif:    M N              (last 2 tokens before EOS)
```

**Training strategy:**
- Randomly choose motif position (prefix, middle, suffix)
- Randomly choose motif length (1-5 tokens)
- Sometimes use no motif (unconditional, ~20% of time)

---

## STEP 3 — Encode motifs

### Tokenize and embed:

```python
# S0 motif: "AG"
motif_tokens = [tok_A, tok_G]
motif_token_embs = embed_tokens(motif_tokens)  # (2, embed_dim)

# Pool to fixed size
motif_emb = mean_pool(motif_token_embs)  # (embed_dim,)

# Pass through MLP encoder
motif_cond = motif_encoder(motif_emb)  # (embed_dim,)
```

### For batch:

```python
# Shape: (batch_size, embed_dim)
motif_conds = [
    encode_motif("AG"),      # S0
    encode_motif("WE"),      # S1
    encode_motif(None),      # S2 (unconditional)
    encode_motif("MN"),      # S3
]
```

---

## STEP 4 — Batched sequence packing

Pack sequences into slots (capacity = 10 for demo, real = 2048):

```
Lengths:
S3 = 7
S0 = 6
S1 = 5
S2 = 4
```

Packing:
```
Slot 0: [S3]           (7 tokens)
Slot 1: [S0, S2]       (6 + 4 = 10 tokens)
Slot 2: [S1]           (5 tokens)
```

---

## STEP 5 — Flatten for FlashAttention

### Tokens (x₀, clean):

```
x0_flat =
[ H, I, K, L, M, N, <EOS>,    # S3
  A, G, M, L, K, <EOS>,        # S0
  P, T, Y, <EOS>,              # S2
  Q, W, E, R, <EOS> ]          # S1
```

### Motif embeddings (aligned per sequence):

```python
# Each sequence has its own motif embedding
motif_emb_flat = [
    motif_MN,  # for all tokens in S3 (indices 0-6)
    motif_AG,  # for all tokens in S0 (indices 7-12)
    motif_NONE,# for all tokens in S2 (indices 13-16)
    motif_WE,  # for all tokens in S1 (indices 17-21)
]
```

### cu_seqlens (FlashAttention boundaries):

```
cu_seqlens = [0, 7, 13, 17, 22]
max_seqlen = 7
```

✅ No cross-sequence attention

---

## STEP 6 — Forward diffusion

Apply standard discrete diffusion noise:

```python
# Sample timestep
t = sample_timestep()  # e.g., t=50

# Add noise to ALL tokens (no masking)
x_t = add_noise(x0_flat, t)

# Result (all tokens noised):
x_t_flat = [N0, N1, N2, N3, N4, N5, N6,  # S3
            N7, N8, N9, N10, N11, N12,    # S0
            N13, N14, N15, N16,            # S2
            N17, N18, N19, N20, N21]       # S1
```

**Key:** No clean tokens remain. Conditioning comes from the motif embedding vector.

---

## STEP 7 — Model forward (with motif conditioning)

### Model architecture modification:

```python
class ESMDiffusion(nn.Module):
    def __init__(self, ...):
        # Existing components
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.time_mlp = nn.Sequential(...)
        self.domain_emb = nn.Embedding(3, embed_dim)

        # NEW: Motif encoder
        self.motif_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x_t, t, domain, motif_emb=None, cu_seqlens=None, ...):
        # Token embeddings
        x = self.embed_tokens(x_t)  # (total_tokens, embed_dim)

        # Positional embeddings
        pos = self.pos_emb(positions)

        # Timestep conditioning
        t_emb = self.time_mlp(self.get_timestep_embedding(t))

        # Domain conditioning
        d_emb = self.domain_emb(domain)

        # Motif conditioning (NEW!)
        if motif_emb is not None:
            # motif_emb already expanded per sequence
            cond = t_emb + d_emb + motif_emb  # (total_tokens, embed_dim)
        else:
            cond = t_emb + d_emb

        # Combine all embeddings
        h = x + pos + cond

        # FlashAttention Transformer (cu_seqlens prevents cross-attention)
        for layer in self.layers:
            h, attn = layer(h, cu_seqlens=cu_seqlens)

        # Output logits
        logits = self.lm_head(h)
        return logits
```

### Conditioning flow:

```
Input sequence (noised): [N0, N1, N2, ...]
                          ↓
Token embeddings:        [emb(N0), emb(N1), ...]
                          ↓
                          + positional embeddings
                          + timestep embeddings
                          + domain embeddings
                          + motif embeddings  ← GLOBAL CONDITIONING
                          ↓
Transformer (FlashAttention)
                          ↓
Output logits:           [logits0, logits1, ...]
```

---

## STEP 8 — Loss computation

**Full sequence cross-entropy** (no masking needed):

```python
# Flatten logits and targets
logits_flat = logits.view(-1, vocab_size)  # (total_tokens, vocab_size)
targets_flat = x0_flat.view(-1)             # (total_tokens,)

# Standard cross-entropy on ALL tokens
loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=PAD_IDX)
```

✅ Train on entire sequence
✅ Model learns: "Given motif embedding, generate compatible sequences"

---

## STEP 9 — What the model learns

Through training with diverse motif extractions:

| Motif position | What model learns |
|----------------|-------------------|
| Prefix (AG...) | "Generate sequences that start with AG" |
| Middle (...WE...) | "Generate sequences containing WE" |
| Suffix (...MN) | "Generate sequences ending with MN" |
| No motif | "Generate unconditionally" |

**Result:** Model learns **semantic relationships** between motifs and full sequences.

---

# Inference

## Input from user:

```python
motif = "WEL"  # User wants protein containing this motif
domain = "protein"
```

## Encoding:

```python
# Tokenize motif
motif_tokens = tokenize("WEL")  # [W, E, L]

# Embed and encode
motif_token_embs = model.embed_tokens(motif_tokens)
motif_emb = mean_pool(motif_token_embs)
motif_cond = model.motif_encoder(motif_emb)  # (embed_dim,)
```

## Reverse diffusion:

```python
# Start with pure noise (no predefined length)
x_t = random_tokens(max_len=50)  # or sample from length distribution

# Denoise step by step
for t in reversed(range(T)):
    logits = model(x_t, t, domain, motif_emb=motif_cond)
    x_t = sample_prev_step(x_t, logits, t)

    # Stop if EOS is sampled
    if EOS in x_t:
        break

# Final sequence (example)
result = "AGWELKN<EOS>"
```

**Key:** Model generates sequences **semantically related** to motif "WEL", but:
- Exact placement not guaranteed
- May contain motif, may be similar in properties
- Length determined by EOS prediction

---

# Why This Works

### During training:
- Model sees motif "WE" → must generate "QWER<EOS>"
- Model sees motif "AG" → must generate "AGMLK<EOS>"
- Model sees no motif → generates unconditionally

### At inference:
- Motif embedding becomes a **semantic guide**
- Model generates sequences with similar:
  - Composition
  - Structure
  - Properties
  - May include the exact motif (learned pattern)

---

# Summary: Global vs Partial Conditioning

## Global Conditioning (What We Use):
✅ Motif → embedding vector → global signal
✅ Loss on full sequence
✅ Learns semantic compatibility
✅ Flexible, property-based generation

## Partial Conditioning (Alternative):
❌ Per-token clean/noisy masking
❌ Loss only on masked positions
❌ Learns exact inpainting
❌ Guarantees exact motif placement

**For protein/molecule design with motif guidance → Global conditioning is better.**
