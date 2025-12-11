# sample.py

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from bindwell.model import ESMDiffusion
from diffusion import D3PM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("/teamspace/studios/this_studio/data")

CKPT_PATH = "atomwell_epoch_0_step_28200.pt"

T = 100
MAX_LEN = 64
DOMAIN_ID = 0         # 0=SMILES, 1=Protein, 2=DNA
NUM_SAMPLES = 10

vocab = np.load(DATA_ROOT / "vocab.npy", allow_pickle=True).item()
VOCAB_SIZE = max(vocab.values()) + 1
MASK_ID = vocab["[MASK]"]
EOS_ID = vocab["[EOS]"]

inv_vocab = {v: k for k, v in vocab.items()}


def decode(seq):
    out = []
    for i in seq:
        tok_id = int(i)
        tok_name = inv_vocab.get(tok_id, "[UNK]")

        if tok_name in ["[PAD]", "[SEP]", "[EOS]"]:
            break

        if tok_name.startswith("SMI_"):
            out.append(tok_name[4:])
        elif tok_name.startswith("PROT_"):
            out.append(tok_name[5:])
        elif tok_name.startswith("DNA_"):
            out.append(tok_name[4:])
        elif tok_name in ["[SMI]", "[PROT]", "[DNA]"]:
            continue
        else:
            out.append(tok_name)

    return "".join(out)


def is_valid_sequence(seq, domain_row):
    for tok_id, dom_id in zip(seq, domain_row):
        tok_name = inv_vocab.get(int(tok_id), "[UNK]")
        dom_id = int(dom_id)

        if tok_name in ["[PAD]", "[SEP]", "[EOS]", "[MASK]", "[UNK]"]:
            continue

        if dom_id == 0:  # SMILES
            if not (tok_name.startswith("SMI_") or tok_name == "[SMI]"):
                return False
        elif dom_id == 1:  # Protein
            if not (tok_name.startswith("PROT_") or tok_name == "[PROT]"):
                return False
        elif dom_id == 2:  # DNA
            if not (tok_name.startswith("DNA_") or tok_name == "[DNA]"):
                return False
        else:
            return False

    return True


model = ESMDiffusion(
    vocab_size=VOCAB_SIZE,
    num_layers=12,
    embed_dim=1024,
    attention_heads=16,
    max_seq_len=2048,
    T=T,
    token_dropout=False,
    use_checkpoint=False,
    padding_idx=0,
    mask_idx=MASK_ID,
).to(DEVICE)

raw_ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
    state_dict = raw_ckpt["model"]
    print(
        f"Loaded training checkpoint '{CKPT_PATH}' "
        f"(step={raw_ckpt.get('step')}, epoch={raw_ckpt.get('epoch')})"
    )
else:
    state_dict = raw_ckpt
    print(f"Loaded pure state_dict checkpoint '{CKPT_PATH}'")

missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print("Warning: missing keys in state_dict:", missing)
if unexpected:
    print("Warning: unexpected keys in state_dict:", unexpected)

model.eval()

diffusion = D3PM(vocab_size=VOCAB_SIZE, T=T, mask_id=MASK_ID)


@torch.no_grad()
def p_sample_step(model, diffusion, x_t, t_batch, domain):
    B, L = x_t.shape
    t = t_batch[0].item()
    seq_boundaries = [[0, L] for _ in range(B)]
    x_t_minus_1 = diffusion.p_sample(
        model, x_t, t, domain,
        seq_boundaries=seq_boundaries,
        max_seqlen=L
    )
    return x_t_minus_1


print(f"Sampling {NUM_SAMPLES} sequence(s) from domain {DOMAIN_ID}")
print(f"Max length: {MAX_LEN}, Timesteps: {T}")
print(f"Device: {DEVICE}\n")

B = NUM_SAMPLES
domain = torch.full((B, MAX_LEN), DOMAIN_ID, dtype=torch.long, device=DEVICE)
x_t = torch.full((B, MAX_LEN), MASK_ID, dtype=torch.long, device=DEVICE)

snapshots = []
valid_flags_at_t1 = None

print("Running reverse diffusion process...")
for t in range(T, 0, -1):
    if t % 20 == 0 or t == 1:
        print(f"  t = {t}/{T}")

    t_batch = torch.full((B,), t, dtype=torch.long, device=DEVICE)
    x_t = p_sample_step(model, diffusion, x_t, t_batch, domain)

    for b in range(B):
        seq = x_t[b]
        eos_pos = (seq == EOS_ID).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            first_eos = eos_pos[0].item()
            seq[first_eos + 1:] = 0

    if t in [T, 80, 60, 40, 20, 5, 1]:
        snapshots.append((t, x_t.clone().cpu()))

    if t == 1:
        x_cpu = x_t.detach().cpu()
        d_cpu = domain.detach().cpu()
        valid_flags_at_t1 = [
            is_valid_sequence(x_cpu[b], d_cpu[b]) for b in range(B)
        ]

print("\nDone sampling!\n")

print("=" * 70)
print("SAMPLING RESULTS (Proper D3PM Reverse Process)")
print("=" * 70)

for t, seq_batch in snapshots:
    print(f"\n--- Timestep t={t} ---")
    for i in range(B):
        decoded = decode(seq_batch[i])
        if t == 1 and valid_flags_at_t1 is not None:
            if not valid_flags_at_t1[i]:
                continue
            print(f"[sample {i}] (VALID) {decoded}")
        else:
            print(f"[sample {i}] {decoded}")
