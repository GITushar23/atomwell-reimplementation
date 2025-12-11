import os
import random
import time
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataset import make_dataloader
from model import ESMDiffusion
from diffusion import D3PM
from collections import defaultdict  # <-- already here

# DATA_ROOT = Path("/teamspace/lightning_storage/bindwelldata/data")
DATA_ROOT = Path("/teamspace/studios/this_studio/data")
RESUME_CKPT = "atomwell_epoch_0_step_26350.pt"  # e.g. "atomwell_epoch_0_step_1600.pt" or None to start fresh
# RESUME_CKPT = None  # e.g. "atomwell_epoch_0_step_1600.pt" or None to start fresh

# -------------------------
# Seeding utilities
# -------------------------
SEED = 42  # choose whatever you like


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"

# If we are NOT resuming, set seeds now.
# If we ARE resuming, we'll restore RNG state from checkpoint instead.
if not RESUME_CKPT:
    seed_all(SEED)

vocab = np.load(DATA_ROOT / "vocab.npy", allow_pickle=True).item()
VOCAB_SIZE = max(vocab.values()) + 1
MASK_ID = vocab["[MASK]"]
SEP_ID = vocab["[SEP]"]
EOS_ID = vocab["[EOS]"]
T = 100
BATCH_SIZE = 16
GRAD_ACCUM = 2
NUM_WORKERS = 12
NUM_EPOCHS = 10

model = ESMDiffusion(
    vocab_size=VOCAB_SIZE,
    num_layers=12,
    embed_dim=1024,
    attention_heads=16,
    max_seq_len=2048,
    T=T,
    token_dropout=True,
    use_checkpoint=True,
    padding_idx=0,
    mask_idx=MASK_ID,
    eos_idx=EOS_ID,
    # sep_idx=SEP_ID,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.1f}M params | Vocab: {VOCAB_SIZE} tokens | Device: {device}")

diffusion = D3PM(vocab_size=VOCAB_SIZE, T=T, mask_id=MASK_ID)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda')

# history dict for plotting later
train_history = defaultdict(list)

start_step = 0
start_epoch = 0

# track last checkpoint paths so we can keep only the most recent 3
recent_ckpt_paths = []

# -------------------------
# Resume from checkpoint (including RNG state)
# -------------------------
if RESUME_CKPT:
    print(f"Loading checkpoint from {RESUME_CKPT} ...")
    ckpt = torch.load(RESUME_CKPT, map_location=device,weights_only=False )

    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    NEW_LR = 5e-6
    for pg in optimizer.param_groups:
        pg["lr"] = NEW_LR
    start_step = ckpt['step']
    start_epoch = ckpt['epoch']
    train_history = ckpt.get('history', defaultdict(list))

    # Restore RNG states if present; otherwise fall back to seeding
    if 'random_state' in ckpt:
        random.setstate(ckpt['random_state'])
        np.random.set_state(ckpt['np_random_state'])

        # Ensure torch RNG state is a ByteTensor on CPU
        torch_state = ckpt['torch_random_state']
        if not isinstance(torch_state, torch.ByteTensor):
            torch_state = torch_state.cpu() if hasattr(torch_state, 'cpu') else torch.ByteTensor(torch_state)
        torch.set_rng_state(torch_state)

        if device == 'cuda' and ckpt.get('cuda_random_state') is not None:
            cuda_states = ckpt['cuda_random_state']
            # Ensure all CUDA states are ByteTensors on CPU
            if isinstance(cuda_states, list):
                cuda_states = [s.cpu() if hasattr(s, 'cpu') else s for s in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states)
        print("Random number generator states restored from checkpoint.")
    else:
        print("Checkpoint has no RNG state, seeding from SEED instead.")
        seed_all(SEED)

    print(f"Resumed from {RESUME_CKPT} at epoch {start_epoch}, step {start_step}")
else:
    # fresh run with deterministic seeding
    seed_all(SEED)

model.train()
optimizer.zero_grad(set_to_none=True)

global_step = start_step
loss_accum = 0.0
accum_step = 0  # Track gradient accumulation steps

# Throughput tracking variables (do NOT add to train_history)
accum_token_count = 0
accum_start_time = None

def _sync():
    if device == 'cuda':
        torch.cuda.synchronize()

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
    loader = make_dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=120)

    for batch in pbar:
        
        x0 = batch["tokens"].to(device, non_blocking=True)
        domain = batch["domain"].to(device, non_blocking=True)
        pos = batch["positions"].to(device, non_blocking=True)
        seq_boundaries = batch.get("seq_boundaries", None)  # List of lists, stays on CPU
        max_seqlen = batch.get("max_seqlen", None)
        B, L = x0.shape

        # -------------------------
        # Token counts per domain in current batch
        # -------------------------
        with torch.no_grad():
            # Mask out padding tokens
            token_mask = (x0 != 0)  # (B, L)

            if domain.dim() == 1:
                # domain is per-sequence (B,)
                # Expand to token level
                domain_expanded = domain.unsqueeze(1).expand_as(x0)  # (B, L)
            else:
                # domain already per-token (B, L)
                domain_expanded = domain

            # Only count valid (non-padding) tokens
            valid_domains = domain_expanded[token_mask]  # (N_tokens,)

            # Count tokens per domain
            domains, counts = torch.unique(valid_domains, return_counts=True)

            # Move to CPU for logging / printing
            domain_token_counts = {
                int(d.item()): int(c.item())
                for d, c in zip(domains, counts)
            }

            # Build a compact string for tqdm postfix, e.g. "d0:1532 d1:876"
            if len(domain_token_counts) > 0:
                domain_str = " ".join(
                    [f"d{d}:{c}" for d, c in sorted(domain_token_counts.items())]
                )
            else:
                domain_str = "n/a"

            # Count valid tokens in this micro-batch
            num_valid_tokens = int(token_mask.sum().item())

        # Start throughput timer at first micro-batch of accumulation window
        if accum_step == 0:
            _sync()
            accum_start_time = time.perf_counter()
            accum_token_count = 0

        # accumulate token counts for the current accumulation window
        accum_token_count += num_valid_tokens

        t = diffusion.sample_timesteps(B, device)
        x_t = diffusion.q_sample(x0, t)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(
                x_t, t, domain,
                positions=pos,
                seq_boundaries=seq_boundaries,
                max_seqlen=max_seqlen
            )
            logits = output["logits"]
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), x0.view(-1), ignore_index=0)
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        accum_step += 1

        # Clean up to prevent memory leaks
        del output, logits, x_t, t

        # Perform optimizer step every GRAD_ACCUM batches
        if accum_step % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0  # CRITICAL: Reset accumulation counter

            # Increment global step AFTER optimizer update
            global_step += 1

            # convert back to full loss per step
            # Use .item() to detach from computation graph and free memory
            full_loss = loss.item() * GRAD_ACCUM
            loss_accum = full_loss * 0.1 + loss_accum * 0.9

            # Explicitly delete loss tensor to free memory
            del loss

            # log metrics for plotting
            current_lr = optimizer.param_groups[0]["lr"]
            train_history["step"].append(global_step)
            train_history["epoch"].append(epoch)
            train_history["loss"].append(full_loss)
            train_history["smoothed_loss"].append(loss_accum)
            train_history["grad_norm"].append(float(grad_norm))
            train_history["lr"].append(current_lr)

            # Throughput measurement: stop timer and compute tokens/sec
            if accum_start_time is not None:
                _sync()
                elapsed = max(1e-12, time.perf_counter() - accum_start_time)
                tokens_per_sec = accum_token_count / elapsed
            else:
                tokens_per_sec = float("nan")

            # now also show domain_str and throughput in the tqdm bar
            pbar.set_postfix(
                loss=f"{loss_accum:.4f}",
                gnorm=f"{grad_norm:.2f}",
                step=global_step,
                domains=domain_str,
                throughput=f"{tokens_per_sec:.0f} tok/s",
            )

            # Print a short line so it's easy to copy/paste results when you change BATCH_SIZE
            # tqdm.write(f"[Epoch {epoch+1}, Step {global_step}] Throughput: {tokens_per_sec:.0f} tok/s (BATCH_SIZE={BATCH_SIZE}, grad_accum={GRAD_ACCUM})")

            # reset accumulation throughput trackers
            accum_token_count = 0
            accum_start_time = None

            # Save checkpoint every 10 steps, keeping ONLY the latest 3
            if global_step % 50 == 0:
                ckpt_path = Path(f"atomwell_epoch_{epoch}_step_{global_step}.pt")

                # Save RNG state as well so resuming gives same data / sampling behavior
                # Move RNG states to CPU to ensure proper ByteTensor format
                torch_rng = torch.get_rng_state().cpu()
                cuda_rng = None
                if device == 'cuda':
                    cuda_rng = [state.cpu() for state in torch.cuda.get_rng_state_all()]

                ckpt_data = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': global_step,
                    'epoch': epoch,
                    'history': dict(train_history),
                    'random_state': random.getstate(),
                    'np_random_state': np.random.get_state(),
                    'torch_random_state': torch_rng,
                    'cuda_random_state': cuda_rng,
                }

                torch.save(ckpt_data, ckpt_path)

                # add new checkpoint to list
                recent_ckpt_paths.append(ckpt_path)

                # if we now have more than 3, delete the oldest
                if len(recent_ckpt_paths) > 3:
                    old_ckpt = recent_ckpt_paths.pop(0)
                    if old_ckpt.exists():
                        try:
                            old_ckpt.unlink()
                        except Exception as e:
                            tqdm.write(f"Warning: failed to delete old checkpoint {old_ckpt}: {e}")

                tqdm.write(f"[Epoch {epoch+1}, Step {global_step}] Checkpoint saved: {ckpt_path}")

    pbar.close()

# Save final model
torch.save(model.state_dict(), "atomwell.pt")

# save history as npz (easy to load & plot with numpy/matplotlib)
np.savez("train_history.npz", **train_history)

print(f"\nTraining complete after {NUM_EPOCHS} epochs ({global_step} steps).")
print("Final model saved to atomwell.pt")
print("Training history saved to train_history.npz")
