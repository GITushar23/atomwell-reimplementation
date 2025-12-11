import pickle
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader

# DATA_ROOT = Path("/teamspace/lightning_storage/bindwelldata/data")
DATA_ROOT = Path("/teamspace/studios/this_studio/data")


class StreamingPackedDataset(IterableDataset):
    def __init__(self, max_len=2048, mix_domains=True, domain_weights=(0.6, 0.3, 0.1)):
        self.max_len = max_len
        self.mix_domains = mix_domains
        self.domain_weights = domain_weights  # (SMILES, protein, DNA) sampling probabilities
        vocab = np.load(DATA_ROOT / "vocab.npy", allow_pickle=True).item()
        self.sep_id = vocab["[SEP]"]
        self.eos_id = vocab["[EOS]"]

        # Group files by domain for mixed-domain sampling
        self.smiles_files = [str(p) for p in sorted((DATA_ROOT / "chunks/smiles").glob("*.pkl"))]
        self.protein_files = [str(p) for p in sorted((DATA_ROOT / "chunks/proteins").glob("*.pkl"))]
        self.dna_files = [str(p) for p in sorted((DATA_ROOT / "chunks/dna").glob("*.pkl"))]

        # For non-mixed mode, create flat list as before
        self.files = []
        self.files.extend([(f, 0) for f in self.smiles_files])
        self.files.extend([(f, 1) for f in self.protein_files])
        self.files.extend([(f, 2) for f in self.dna_files])

        print(f"Dataset: {len(self.smiles_files)} SMILES files, {len(self.protein_files)} protein files, {len(self.dna_files)} DNA files")
        print(f"Mixed-domain batching: {'ENABLED' if mix_domains else 'DISABLED'}")
        if mix_domains:
            print(f"Domain sampling weights: SMILES={domain_weights[0]:.1%}, Protein={domain_weights[1]:.1%}, DNA={domain_weights[2]:.1%}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        if self.mix_domains:
            # Mixed-domain mode: create iterators for each domain and interleave
            yield from self._iter_mixed_domains(worker_info, worker_id)
        else:
            # Original mode: process files sequentially
            yield from self._iter_sequential(worker_info, worker_id)

    def _iter_sequential(self, worker_info, worker_id):
        """Original sequential iteration - one domain at a time per batch"""
        if worker_info is None:
            files = self.files
        else:
            per_worker = len(self.files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
            files = self.files[start:end]

        # shuffled deterministically based on worker's RNG state (set in worker_init_fn)
        random.shuffle(files)

        current_tok, current_dom, current_pos = [], [], []
        count = 0

        for filepath, domain in files:
            with open(filepath, 'rb') as f:
                seqs = pickle.load(f)

            # deterministic given worker seed
            random.shuffle(seqs)

            for seq in seqs:
                # Append EOS token to each sequence
                seq_with_eos = list(seq) + [self.eos_id]
                seq_len = len(seq_with_eos)

                # Truncate sequences that are too long
                if seq_len > self.max_len:
                    seq_with_eos = seq_with_eos[:self.max_len]
                    seq_len = self.max_len

                if len(current_tok) + seq_len + 1 > self.max_len:
                    if current_tok:
                        yield (current_tok, current_dom, current_pos)
                        count += 1
                    current_tok = seq_with_eos
                    current_dom = [domain] * seq_len
                    current_pos = list(range(seq_len))
                else:
                    if current_tok:
                        current_tok.append(self.sep_id)
                        current_dom.append(domain)
                        current_pos.append(0)
                    current_tok.extend(seq_with_eos)
                    current_dom.extend([domain] * seq_len)
                    current_pos.extend(list(range(seq_len)))

        if current_tok:
            yield (current_tok, current_dom, current_pos)
            count += 1

        print(f"[Worker {worker_id}] Epoch complete: {count} packed sequences")

    def _iter_mixed_domains(self, worker_info, worker_id):
        """Mixed-domain iteration - interleave sequences from all domains using streaming"""
        # Split files per worker
        def split_files_for_worker(file_list):
            if worker_info is None:
                return file_list
            per_worker = len(file_list) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(file_list)
            return file_list[start:end]

        smiles_files = split_files_for_worker(self.smiles_files)
        protein_files = split_files_for_worker(self.protein_files)
        dna_files = split_files_for_worker(self.dna_files)

        # Shuffle each domain's files
        random.shuffle(smiles_files)
        random.shuffle(protein_files)
        random.shuffle(dna_files)

        # Create iterators for each domain that yield (sequence, domain_id)
        def sequence_iterator(file_list, domain_id):
            """Generator that yields sequences from a list of files"""
            for filepath in file_list:
                with open(filepath, 'rb') as f:
                    seqs = pickle.load(f)
                random.shuffle(seqs)
                for seq in seqs:
                    yield (seq, domain_id)

        # Create iterators for each domain
        smiles_iter = sequence_iterator(smiles_files, 0)
        protein_iter = sequence_iterator(protein_files, 1)
        dna_iter = sequence_iterator(dna_files, 2)

        # Weighted random sampling based on domain_weights
        # This is MUCH faster than min-count balancing (3-4s vs 9s per batch)
        iterators = [smiles_iter, protein_iter, dna_iter]
        active_iterators = [0, 1, 2]  # Track which iterators are still active
        domain_token_counts = [0, 0, 0]  # Track tokens per domain for logging

        current_tok, current_dom, current_pos = [], [], []
        count = 0

        while active_iterators:
            # Fast weighted random selection
            # Renormalize weights for active iterators only
            active_weights = [self.domain_weights[i] for i in active_iterators]
            total_weight = sum(active_weights)
            normalized_weights = [w / total_weight for w in active_weights]
            domain_idx = random.choices(active_iterators, weights=normalized_weights, k=1)[0]

            try:
                seq, domain = next(iterators[domain_idx])

                # Append EOS token to each sequence
                seq_with_eos = list(seq) + [self.eos_id]
                seq_len = len(seq_with_eos)

                # Truncate sequences that are too long
                if seq_len > self.max_len:
                    seq_with_eos = seq_with_eos[:self.max_len]
                    seq_len = self.max_len

                # Update token count for this domain
                domain_token_counts[domain] += seq_len

                if len(current_tok) + seq_len + 1 > self.max_len:
                    if current_tok:
                        yield (current_tok, current_dom, current_pos)
                        count += 1
                    current_tok = seq_with_eos
                    current_dom = [domain] * seq_len
                    current_pos = list(range(seq_len))
                else:
                    if current_tok:
                        current_tok.append(self.sep_id)
                        current_dom.append(domain)
                        current_pos.append(0)
                    current_tok.extend(seq_with_eos)
                    current_dom.extend([domain] * seq_len)
                    current_pos.extend(list(range(seq_len)))

            except StopIteration:
                # This iterator is exhausted, remove it
                active_iterators.remove(domain_idx)

        # Yield any remaining packed sequence
        if current_tok:
            yield (current_tok, current_dom, current_pos)
            count += 1

        print(f"[Worker {worker_id}] Epoch complete: {count} packed sequences (mixed-domain)")
        print(f"[Worker {worker_id}] Token distribution - SMILES:{domain_token_counts[0]} Protein:{domain_token_counts[1]} DNA:{domain_token_counts[2]}")
        print(f"[Worker {worker_id}] Sampling weights - SMILES:{self.domain_weights[0]:.1%} Protein:{self.domain_weights[1]:.1%} DNA:{self.domain_weights[2]:.1%}")


def collate_packed(batch):
    L = max(len(t) for t, _, _ in batch)
    B = len(batch)

    tokens = torch.zeros(B, L, dtype=torch.long)
    domains = torch.zeros(B, L, dtype=torch.long)
    positions = torch.zeros(B, L, dtype=torch.long)

    # Track sequence boundaries within each packed sample
    # We need to identify where each sub-sequence starts based on position resets
    seq_boundaries_list = []  # List of lists: one per batch item

    vocab = np.load(DATA_ROOT / "vocab.npy", allow_pickle=True).item()
    sep_id = vocab["[SEP]"]

    for i, (t, d, p) in enumerate(batch):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        domains[i, :len(d)] = torch.tensor(d, dtype=torch.long)
        positions[i, :len(p)] = torch.tensor(p, dtype=torch.long)

        # Find sequence boundaries by detecting position resets
        # When position resets to 0 (and we're not at the start), it's a new sequence
        boundaries = [0]  # Start of first sequence
        for j in range(1, len(p)):
            # Position reset to 0 indicates new sequence start
            # (previous position should be higher, and current is 0)
            if p[j] == 0 and p[j-1] != 0:
                boundaries.append(j)
        boundaries.append(len(t))  # End of last sequence
        seq_boundaries_list.append(boundaries)

    return {
        "tokens": tokens,
        "domain": domains,
        "positions": positions,
        "seq_boundaries": seq_boundaries_list,  # For FlashAttention: list of boundary indices per batch item
        "max_seqlen": L,
    }


def _worker_init_fn(worker_id):
    """
    Make Python's random and NumPy deterministic per worker, using
    the seed that PyTorch assigns to this worker.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    base_seed = worker_info.seed  # unique per worker/process
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))


def make_dataloader(batch_size=16, num_workers=4, mix_domains=True, domain_weights=(0.8, 0.15, 0.05)):
    """
    Args:
        domain_weights: (SMILES, protein, DNA) sampling probabilities.
            Default (0.6, 0.3, 0.1) compensates for sequence length differences:
            - SMILES: shortest sequences, sampled most frequently (60%)
            - Protein: medium sequences (30%)
            - DNA: longest sequences, sampled least (10%)
    """
    return DataLoader(
        StreamingPackedDataset(mix_domains=mix_domains, domain_weights=domain_weights),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_packed,
        persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )
