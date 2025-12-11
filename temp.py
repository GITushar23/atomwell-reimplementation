import numpy as np
from pathlib import Path

DATA_ROOT = Path("/teamspace/studios/this_studio/data")

vocab = np.load(DATA_ROOT / "vocab.npy", allow_pickle=True).item()
inv_vocab = {v: k for k, v in vocab.items()}

print(f"Vocabulary size: {len(vocab)}\n")

print("ID\tTOKEN")
print("-" * 40)

for idx in sorted(inv_vocab):
    print(f"{idx}\t{inv_vocab[idx]}")



# #!/usr/bin/env python3
# import os
# from pathlib import Path
# from multiprocessing import Pool
# import gc

# NUM_WORKERS = 32

# # protein_file = Path("/teamspace/lightning_storage/bindwelldata/data/proteins/uniref50.fasta")
# protein_file = Path("/teamspace/lightning_storage/bindwelldata/data/nucleic/silva_138_2_parc_10M_plus.fasta")
# # dna_file = Path("/teamspace/lightning_storage/bindwelldata/data/nucleic/silva_138_2_parc_10M_plus.fasta")

# def get_fasta_chunks(filepath, num_chunks):
#     """Split FASTA file at sequence boundaries (aligned to '>')."""
#     file_size = os.path.getsize(filepath)
#     chunk_size = file_size // num_chunks
#     ranges = []

#     with open(filepath, 'rb') as f:
#         start = 0
#         for i in range(num_chunks):
#             if i == num_chunks - 1:
#                 end = file_size
#             else:
#                 end = min(start + chunk_size, file_size)
#                 f.seek(end)

#                 # Move to next header or EOF
#                 while True:
#                     line = f.readline()
#                     if not line:
#                         end = file_size
#                         break
#                     if line.startswith(b'>'):
#                         end = f.tell() - len(line)
#                         break
#                     end = f.tell()

#             if end > start:
#                 ranges.append((start, end))
#             start = end

#     return ranges


# def worker_scan_fasta_chars(args):
#     """Worker: scan a chunk of FASTA for unique sequence characters."""
#     filepath, start, end, chunk_id = args
#     chars = set()
#     try:
#         with open(filepath, 'rb') as f:
#             f.seek(start)
#             data = f.read(end - start)

#         for line in data.decode('utf-8', errors='ignore').split('\n'):
#             line = line.strip()
#             if not line or line.startswith('>'):
#                 continue
#             chars.update(line)
#     except Exception as e:
#         print(f"Chunk {chunk_id} scan error: {e}")
#     return chars


# def main():
#     fasta_path = str(protein_file)
#     print(f"Scanning: {fasta_path}")

#     print("Splitting at sequence boundaries...")
#     byte_ranges = get_fasta_chunks(fasta_path, NUM_WORKERS)
#     print(f"Created {len(byte_ranges)} chunks")

#     args = [(fasta_path, s, e, i) for i, (s, e) in enumerate(byte_ranges)]

#     print(f"Scanning with {NUM_WORKERS} workers...")
#     with Pool(NUM_WORKERS) as p:
#         char_sets = p.map(worker_scan_fasta_chars, args)

#     # Union of all sets
#     all_chars = set().union(*char_sets)
#     del char_sets
#     gc.collect()

#     print("Unique raw characters in protein sequences:")
#     print("".join(sorted(all_chars)))


# if __name__ == "__main__":
#     main()
# import numpy as np
# from pathlib import Path

# DATA_ROOT = Path("/teamspace/studios/this_studio/data")

# vocab = np.load(DATA_ROOT / "vocab.npy", allow_pickle=True).item()

# if "[EOF]" not in vocab:
#     print("No [EOF] token found.")
# else:
#     eof_id = vocab["[EOF]"]
#     del vocab["[EOF]"]
#     vocab["[EOS]"] = eof_id

#     np.save(DATA_ROOT / "vocab.npy", vocab)

#     print(f"Replaced [EOF] → [EOS] at ID {eof_id}")
