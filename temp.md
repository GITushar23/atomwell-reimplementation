Atomwell
Atomwell is our unified, all-atom foundational model that deals with small-molecules, proteins and nucleic acids. It’s a 150M parameter model trained on more than 50 billion tokens of FASTA and SMILES sequences. A diffusion language model, we’ve trained it using the D3PM paradigm, which enables the model to predict “clean” tokens given a timestep and a noised sequence.

This architectural choice of diffusion means that Atomwell is not only very good at predicting properties of small molecules and proteins for downstream tasks (via embeddings), but also excellent at conditional generation of proteins/small-molecules given a motif/part of a sequence.

Atomwell is trained using Flash Attention, with sequence packing (concatenating smaller sequences into one large sequence to prevent wastage from <PAD> tokens). We edited the FA implementation of ESM2 to include support for not only sequence packing, but “batched sequence packing”, where the model is able to batch together sequences after concatenating smaller sequences into one.



Timestep: 100
<MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK> <MASK>

to 

Timestep: 0
C C [NH+] 1 C C [C@H] ( [C@H] ( C 1 ) N C ( = O ) c 2 c c c c n 2 )
