#!/usr/bin/env python3
import numpy as np
import os
import gc
from multiprocessing import Pool
import random
from pathlib import Path
import pickle

NUM_WORKERS = 32
MAX_LEN = 2048
MIN_SEQ_LEN = 30
SAVE_BATCH_SIZE = 100_000
REPORT_EVERY = 500_000
CACHE_SIZE = 3

SMILES_FILE = "/teamspace/lightning_storage/bindwelldata/data/molecules/SMILES_random_50M.txt"
PROTEIN_FILE = "/teamspace/lightning_storage/bindwelldata/data/proteins/uniref50.fasta"
DNA_FILE = "/teamspace/lightning_storage/bindwelldata/data/nucleic/silva_138_2_parc_10M_plus.fasta"

SMILES_CHUNKS_DIR = "/teamspace/lightning_storage/bindwelldata/data/chunks/smiles"
PROTEIN_CHUNKS_DIR = "/teamspace/lightning_storage/bindwelldata/data/chunks/proteins"
DNA_CHUNKS_DIR = "/teamspace/lightning_storage/bindwelldata/data/chunks/dna"

VOCAB_OUTPUT = "/teamspace/lightning_storage/bindwelldata/data/vocab.npy"

VOCAB = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[SMI]": 3, "[PROT]": 4, "[DNA]": 5}
AA = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
BASES = "ABCDGHKMNRSUVWY"

def get_next_id():
    return max(VOCAB.values()) + 1

def get_file_chunks(filepath, num_chunks):
    file_size = os.path.getsize(filepath)
    chunk_size = file_size // num_chunks
    ranges = []

    with open(filepath, 'rb') as f:
        start = 0
        for i in range(num_chunks):
            if i == num_chunks - 1:
                end = file_size
            else:
                end = min(start + chunk_size, file_size)
                f.seek(end)
                f.readline()
                end = f.tell()

            if end > start:
                ranges.append((start, end))
            start = end

    return ranges

def get_fasta_chunks(filepath, num_chunks):
    file_size = os.path.getsize(filepath)
    chunk_size = file_size // num_chunks
    ranges = []

    with open(filepath, 'rb') as f:
        start = 0
        for i in range(num_chunks):
            if i == num_chunks - 1:
                end = file_size
            else:
                end = min(start + chunk_size, file_size)
                f.seek(end)

                while True:
                    line = f.readline()
                    if not line:
                        end = file_size
                        break
                    if line.startswith(b'>'):
                        end = f.tell() - len(line)
                        break
                    end = f.tell()

            if end > start:
                ranges.append((start, end))
            start = end

    return ranges

def worker_scan_smiles(args):
    filepath, start, end, chunk_id = args
    chars = set()
    try:
        with open(filepath, 'rb') as f:
            f.seek(start)
            data = f.read(end - start)
        for line in data.decode('utf-8', errors='ignore').split('\n'):
            if line.strip():
                chars.update(line.strip())
    except Exception as e:
        print(f"Chunk {chunk_id} scan error: {e}")
    return chars

def worker_tokenize_smiles(args):
    filepath, start, end, smi_map, smi_tok, unk, output_dir, chunk_id = args

    try:
        with open(filepath, 'rb') as f:
            f.seek(start)
            data = f.read(end - start)

        tokens = []
        batch_num = 0
        total_count = 0

        for line in data.decode('utf-8', errors='ignore').split('\n'):
            s = line.strip()
            if s:
                tokens.append([smi_tok] + [smi_map.get(c, unk) for c in s])

                if len(tokens) >= SAVE_BATCH_SIZE:
                    out_file = os.path.join(output_dir, f"chunk_{chunk_id:03d}_batch_{batch_num:03d}.pkl")
                    with open(out_file, 'wb') as f:
                        pickle.dump(tokens, f)
                    total_count += len(tokens)
                    tokens = []
                    batch_num += 1

        if tokens:
            out_file = os.path.join(output_dir, f"chunk_{chunk_id:03d}_batch_{batch_num:03d}.pkl")
            with open(out_file, 'wb') as f:
                pickle.dump(tokens, f)
            total_count += len(tokens)

        return total_count

    except Exception as e:
        print(f"Chunk {chunk_id} error: {e}")
        import traceback
        traceback.print_exc()
        return 0

def worker_tokenize_fasta(args):
    filepath, start, end, vocab_map, mod_tok, unk, min_len, output_dir, chunk_id, uppercase = args

    try:
        with open(filepath, 'rb') as f:
            f.seek(start)
            data = f.read(end - start)

        sequences = []
        batch_num = 0
        total_count = 0
        current_seq = []

        for line in data.decode('utf-8', errors='ignore').split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_seq:
                    s = ''.join(current_seq)
                    if uppercase:
                        s = s.upper()
                    if len(s) >= min_len:
                        sequences.append([mod_tok] + [vocab_map.get(c, unk) for c in s])

                        if len(sequences) >= SAVE_BATCH_SIZE:
                            out_file = os.path.join(output_dir, f"chunk_{chunk_id:03d}_batch_{batch_num:03d}.pkl")
                            with open(out_file, 'wb') as f:
                                pickle.dump(sequences, f)
                            total_count += len(sequences)
                            sequences = []
                            batch_num += 1

                current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            s = ''.join(current_seq)
            if uppercase:
                s = s.upper()
            if len(s) >= min_len:
                sequences.append([mod_tok] + [vocab_map.get(c, unk) for c in s])

        if sequences:
            out_file = os.path.join(output_dir, f"chunk_{chunk_id:03d}_batch_{batch_num:03d}.pkl")
            with open(out_file, 'wb') as f:
                pickle.dump(sequences, f)
            total_count += len(sequences)

        return total_count

    except Exception as e:
        print(f"Chunk {chunk_id} error: {e}")
        import traceback
        traceback.print_exc()
        return 0

def process_smiles(smiles_file, output_dir):
    print("\n" + "="*60)
    print("STEP 1: SMILES")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    file_size = os.path.getsize(smiles_file)
    print(f"File: {smiles_file}")
    print(f"Size: {file_size / 1e9:.2f} GB")

    print("Splitting file...")
    byte_ranges = get_file_chunks(smiles_file, NUM_WORKERS)
    print(f"Created {len(byte_ranges)} chunks")

    print("Scanning for unique characters...")
    args = [(smiles_file, s, e, i) for i, (s, e) in enumerate(byte_ranges)]

    with Pool(NUM_WORKERS) as p:
        char_sets = p.map(worker_scan_smiles, args)

    smiles_chars = set().union(*char_sets)
    del char_sets
    gc.collect()

    print(f"Found {len(smiles_chars)} unique chars")

    next_id = get_next_id()
    for ch in sorted(smiles_chars):
        VOCAB[f"SMI_{ch}"] = next_id
        next_id += 1

    print(f"Vocab size after SMILES: {len(VOCAB)}")

    smi_map = {k[4:]: v for k, v in VOCAB.items() if k.startswith("SMI_")}

    print(f"Tokenizing with {NUM_WORKERS} workers...")
    args = [
        (smiles_file, s, e, smi_map, VOCAB['[SMI]'], VOCAB['[UNK]'], output_dir, i)
        for i, (s, e) in enumerate(byte_ranges)
    ]

    with Pool(NUM_WORKERS) as p:
        counts = list(p.imap_unordered(worker_tokenize_smiles, args))

    total = sum(counts)
    del counts, args, byte_ranges, smi_map
    gc.collect()

    print(f"SMILES complete: {total:,} sequences")
    return total

def process_fasta(fasta_file, output_dir, vocab_map, mod_tok, min_len, uppercase, name):
    print("\n" + "="*60)
    print(f"STEP: {name}")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)
    file_size = os.path.getsize(fasta_file)
    print(f"File: {fasta_file}")
    print(f"Size: {file_size / 1e9:.2f} GB")

    print("Splitting at sequence boundaries...")
    byte_ranges = get_fasta_chunks(fasta_file, NUM_WORKERS)
    print(f"Created {len(byte_ranges)} chunks")

    for i, (s, e) in enumerate(byte_ranges):
        print(f"  Chunk {i}: {(e-s)/1e6:.1f} MB")

    print(f"Tokenizing with {NUM_WORKERS} workers...")
    args = [
        (fasta_file, s, e, vocab_map, mod_tok, VOCAB["[UNK]"], min_len, output_dir, i, uppercase)
        for i, (s, e) in enumerate(byte_ranges)
    ]

    results = []
    with Pool(NUM_WORKERS) as p:
        for count in p.imap_unordered(worker_tokenize_fasta, args):
            results.append(count)
            print(f"  Completed chunk: {count:,} sequences ({len(results)}/{len(args)})")

    total = sum(results)
    del results, args, byte_ranges
    gc.collect()

    print(f"{name} complete: {total:,} sequences")
    return total

def pack_all(chunk_dirs, output_dir):
    print("\n" + "="*60)
    print("PACKING")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    print("Counting sequences...")
    total_seqs = 0
    seq_counts = []
    for d in chunk_dirs:
        count = 0
        for pf in Path(d).glob("*.pkl"):
            with open(pf, 'rb') as f:
                count += len(pickle.load(f))
        seq_counts.append(count)
        total_seqs += count
        print(f"  {d}: {count:,}")

    print(f"Total: {total_seqs:,}")

    print("Creating shuffle indices...")
    indices = []
    for domain, count in enumerate(seq_counts):
        indices.extend([(domain, i) for i in range(count)])

    random.shuffle(indices)
    print(f"Shuffled {len(indices):,} indices")

    print("Building file maps...")
    file_maps = []

    for d in chunk_dirs:
        fmap = []
        idx = 0
        for pf in sorted(Path(d).glob("*.pkl")):
            with open(pf, 'rb') as f:
                n = len(pickle.load(f))
            fmap.append((str(pf), idx, idx + n))
            idx += n
        file_maps.append(fmap)

    print("Packing into 2048-token chunks...")
    SEP_ID = VOCAB["[SEP]"]

    packed_tokens = []
    packed_domains = []
    packed_positions = []

    current_tok, current_dom, current_pos = [], [], []

    cache = [{} for _ in chunk_dirs]

    for i, (domain, seq_idx) in enumerate(indices):
        fmap = file_maps[domain]
        file_path = None
        local_idx = None

        for fp, start, end in fmap:
            if start <= seq_idx < end:
                file_path = fp
                local_idx = seq_idx - start
                break

        if file_path not in cache[domain]:
            if len(cache[domain]) > CACHE_SIZE:
                cache[domain].clear()
                gc.collect()

            with open(file_path, 'rb') as f:
                cache[domain][file_path] = pickle.load(f)

        seq = cache[domain][file_path][local_idx]

        if len(current_tok) + len(seq) + 1 > MAX_LEN:
            if current_tok:
                packed_tokens.append(current_tok)
                packed_domains.append(current_dom)
                packed_positions.append(current_pos)
            current_tok = list(seq)
            current_dom = [domain] * len(seq)
            current_pos = list(range(len(seq)))
        else:
            if current_tok:
                current_tok.append(SEP_ID)
                current_dom.append(domain)
                current_pos.append(0)
            current_tok.extend(seq)
            current_dom.extend([domain] * len(seq))
            current_pos.extend(range(len(seq)))

        if (i + 1) % REPORT_EVERY == 0:
            print(f"  {i+1:,}/{len(indices):,} - {len(packed_tokens):,} chunks")
            for c in cache:
                c.clear()
            gc.collect()

    if current_tok:
        packed_tokens.append(current_tok)
        packed_domains.append(current_dom)
        packed_positions.append(current_pos)

    del indices, file_maps, cache
    gc.collect()

    print(f"\nCreated {len(packed_tokens):,} chunks")

    print("Saving packed data...")

    for name, data in [("packed_tokens", packed_tokens),
                       ("packed_domains", packed_domains),
                       ("packed_positions", packed_positions)]:
        out_path = os.path.join(output_dir, f"{name}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved {name}")

    avg_len = np.mean([len(t) for t in packed_tokens])

    print(f"\n  Avg tokens/chunk: {avg_len:.1f}")
    print(f"  Efficiency: {avg_len/MAX_LEN*100:.1f}%")

    del packed_tokens, packed_domains, packed_positions
    gc.collect()

def main():
    smiles_count = process_smiles(SMILES_FILE, SMILES_CHUNKS_DIR)
    gc.collect()
    print("RAM freed after SMILES\n")

    next_id = get_next_id()
    for aa in AA:
        VOCAB[f"PROT_{aa}"] = next_id
        next_id += 1

    for base in BASES:
        VOCAB[f"DNA_{base}"] = next_id
        next_id += 1

    VOCAB["[MASK]"] = next_id
    next_id += 1
    VOCAB["[EOS]"] = next_id

    print(f"Final vocab size: {len(VOCAB)}")

    os.makedirs(os.path.dirname(VOCAB_OUTPUT), exist_ok=True)
    np.save(VOCAB_OUTPUT, VOCAB)
    print(f"Saved vocab to {VOCAB_OUTPUT}")

    prot_map = {k[5:]: v for k, v in VOCAB.items() if k.startswith("PROT_")}
    protein_count = process_fasta(
        PROTEIN_FILE, PROTEIN_CHUNKS_DIR, prot_map,
        VOCAB["[PROT]"], MIN_SEQ_LEN, uppercase=False, name="PROTEINS"
    )
    del prot_map
    gc.collect()
    print("RAM freed after PROTEINS\n")

    dna_map = {k[4:]: v for k, v in VOCAB.items() if k.startswith("DNA_")}
    dna_count = process_fasta(
        DNA_FILE, DNA_CHUNKS_DIR, dna_map,
        VOCAB["[DNA]"], MIN_SEQ_LEN, uppercase=True, name="DNA"
    )
    del dna_map
    gc.collect()
    print("RAM freed after DNA\n")

    print("\n" + "="*60)
    print("TOKENIZATION COMPLETE")
    print(f"  SMILES:   {smiles_count:,}")
    print(f"  Proteins: {protein_count:,}")
    print(f"  DNA:      {dna_count:,}")
    print(f"  Total:    {smiles_count + protein_count + dna_count:,}")
    print("="*60)

    print("\nALL DONE")

if __name__ == "__main__":
    main()
