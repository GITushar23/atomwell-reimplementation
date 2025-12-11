import random
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataset import make_dataloader
from model import ESMDiffusion
from diffusion import D3PM
from collections import defaultdict

DATA_ROOT = Path("/teamspace/studios/this_studio/data")
RESUME_CKPT = "atomwell_epoch_0_step_26350.pt"
SEED = 42


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"

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

print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params | Vocab: {VOCAB_SIZE} | Device: {device}")

diffusion = D3PM(vocab_size=VOCAB_SIZE, T=T, mask_id=MASK_ID)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda')
train_history = defaultdict(list)
start_step = 0
start_epoch = 0

if RESUME_CKPT:
    print(f"Loading checkpoint from {RESUME_CKPT}...")
    ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    for pg in optimizer.param_groups:
        pg["lr"] = 5e-6
    start_step = ckpt['step']
    start_epoch = ckpt['epoch']
    train_history = ckpt.get('history', defaultdict(list))

    if 'random_state' in ckpt:
        random.setstate(ckpt['random_state'])
        np.random.set_state(ckpt['np_random_state'])
        torch_state = ckpt['torch_random_state']
        if not isinstance(torch_state, torch.ByteTensor):
            torch_state = torch_state.cpu() if hasattr(torch_state, 'cpu') else torch.ByteTensor(torch_state)
        torch.set_rng_state(torch_state)
        if device == 'cuda' and ckpt.get('cuda_random_state') is not None:
            cuda_states = ckpt['cuda_random_state']
            if isinstance(cuda_states, list):
                cuda_states = [s.cpu() if hasattr(s, 'cpu') else s for s in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states)
        print("RNG states restored.")
    else:
        seed_all(SEED)
    print(f"Resumed from epoch {start_epoch}, step {start_step}")
else:
    seed_all(SEED)

model.train()
optimizer.zero_grad(set_to_none=True)
global_step = start_step
loss_accum = 0.0
accum_step = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
    loader = make_dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=100)

    for batch in pbar:
        x0 = batch["tokens"].to(device, non_blocking=True)
        domain = batch["domain"].to(device, non_blocking=True)
        pos = batch["positions"].to(device, non_blocking=True)
        seq_boundaries = batch.get("seq_boundaries", None)
        max_seqlen = batch.get("max_seqlen", None)
        B, L = x0.shape

        t = diffusion.sample_timesteps(B, device)
        x_t = diffusion.q_sample(x0, t)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(x_t, t, domain, positions=pos, seq_boundaries=seq_boundaries, max_seqlen=max_seqlen)
            logits = output["logits"]
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), x0.view(-1), ignore_index=0) / GRAD_ACCUM

        scaler.scale(loss).backward()
        accum_step += 1
        del output, logits, x_t, t

        if accum_step % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0
            global_step += 1

            full_loss = loss.item() * GRAD_ACCUM
            loss_accum = full_loss * 0.1 + loss_accum * 0.9
            del loss

            train_history["step"].append(global_step)
            train_history["epoch"].append(epoch)
            train_history["loss"].append(full_loss)
            train_history["smoothed_loss"].append(loss_accum)
            train_history["grad_norm"].append(float(grad_norm))
            train_history["lr"].append(optimizer.param_groups[0]["lr"])

            pbar.set_postfix(loss=f"{loss_accum:.4f}", gnorm=f"{grad_norm:.2f}", step=global_step)

            if global_step % 50 == 0:
                ckpt_path = Path(f"atomwell_epoch_{epoch}_step_{global_step}.pt")
                torch_rng = torch.get_rng_state().cpu()
                cuda_rng = [state.cpu() for state in torch.cuda.get_rng_state_all()] if device == 'cuda' else None

                torch.save({
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
                }, ckpt_path)

                tqdm.write(f"Checkpoint saved: {ckpt_path}")

    pbar.close()

torch.save(model.state_dict(), "atomwell.pt")
np.savez("train_history.npz", **train_history)
print(f"\nTraining complete: {NUM_EPOCHS} epochs, {global_step} steps")
