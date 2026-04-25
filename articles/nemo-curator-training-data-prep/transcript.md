# Transcript — A3: nemo-curator-training-data-prep

Provenance for the data-path envelope sweep. Cleaned source material from the 2026-04-25 (afternoon) session that produced this piece.

## Working setup at session start

Carried forward from A1 + A2:
- `nvcr.io/nvidia/nemo:26.04.00` on disk (70 GB, used by A2)
- A2's `evidence/sweep.py` as the harness skeleton
- pgvector, NIM Embed running

Disk delta from this session:
- 300 MB wikitext-103 train parquets (HuggingFace)
- 417 MiB packed.int32.npy (tokenized output)
- 2.6 GB derived container layer (Curator + cuDF + fasttext + langid model)

## Corpus pick

wikitext-103-raw chosen for: ~500 MB raw text (large enough to need real prefetching, small enough to process in <5 min), public license, no auth, well-known benchmark. The S3 mirror at `https://s3.amazonaws.com/research.metamind.io/wikitext/` is dead (returns 467-byte error page); used the HuggingFace `Salesforce/wikitext` mirror instead. Validation and test splits are placeholder (15 bytes each on this mirror) — only train is meaningful here.

## Install friction (4 distinct gotchas)

1. **NeMo container does not ship Curator.** Has to be added.
2. **`pip install nemo-curator` writes to wrong path.** NeMo's container has `/opt/venv/` as its uv venv; default `pip` writes to `/usr/local/lib/python3.12/dist-packages/`. The venv's python searches its own site-packages first, so the install appears to succeed but `import nemo_curator` finds an older version (or nothing). Fix: `/opt/venv/bin/python3 -m pip install …`.
3. **`cosmos-xenna` (Curator's autoscaler backend) needs PuLP < 3.** PuLP 3.x and 4.x removed the `lowBound=` kwarg from `LpVariable.__init__`. Pin `pulp<3`.
4. **RAPIDS cuDF on aarch64 works.** `pip install --extra-index-url=https://pypi.nvidia.com cudf-cu13` succeeded cleanly. Worth knowing.

All four pinned in `evidence/Dockerfile`. Final image: `nemo-curator-spark:1.1`, 73 GB, build wall ~3 min.

## Curator pipeline output

```
input parquets: 2
counting raw inputs ...
  raw docs=1,803,438  raw chars=539,094,213  (4.4s)
... pipeline stages spin up ...
pipeline wall: 82.9s
counting cleaned outputs ...
  cleaned docs=668,856  cleaned chars=509,478,389
  drop ratio: docs 62.9%  chars 5.4%
```

The 62.9% doc-drop is mostly `WordCountFilter(min_words=50)` killing single-line headers (`= = = Section title = = =`) — wikitext parquet has one row per *line*, not one row per article.

## Tokenize + pack output

```
loading cleaned JSONL ...   2 jsonl files, loaded 668,856 docs in 1.4s
exact-dedup ...             668,856 → 660,773  (removed 8,083, 1.2%)  4.6s
loading GPT-2 tokenizer ... vocab_size = 50,257
tokenizing ...
  doc       0/660,773  tokens=    684,877  ( 3,036,077 tok/s, 0.2s)
  ...
  doc 655,360/660,773  tokens=109,131,099  ( 2,734,648 tok/s, 39.9s)
  tokenize wall: 40.0s   total tokens: 109,339,897
concatenating into one packed stream ...
  packed.shape = (109339897,)  size = 417.1 MiB  (0.2s)
writing packed memmap ...   wrote /work/packed.int32.npy  (0.0s)
```

Tokenization throughput: **2.73 M tok/s** on a single CPU process. That's 190× faster than the GB10's training-throughput peak. Data prep finishes well before training has eaten any meaningful fraction of the corpus.

## Data-path sweep results (8 configs)

```
sweep on NVIDIA GB10  torch=2.11.0a0+eb65b36914.nv26.02  te=2.14.0+71bbefbf
corpus packed.int32.npy: 109,339,897 tokens, vocab=50258
[ 1/8] batch= 4 seq=1024 prec=bf16  → tok/s(step)=   13158  tok/s(+data)=   13153  data= 0.12ms  step= 311.3ms  data%=0.04  peak= 7.94GiB  loss  11.05→  7.77  (11s)
[ 2/8] batch= 8 seq=1024 prec=bf16  → tok/s(step)=   13482  tok/s(+data)=   13479  data= 0.14ms  step= 607.6ms  data%=0.02  peak=13.83GiB  loss  11.04→  7.58  (19s)
[ 3/8] batch=16 seq=1024 prec=bf16  → tok/s(step)=   14422  tok/s(+data)=   14420  data= 0.14ms  step=1136.0ms  data%=0.01  peak=25.63GiB  loss  11.04→  7.69  (36s)
[ 4/8] batch= 4 seq=2048 prec=bf16  → tok/s(step)=   13001  tok/s(+data)=   12998  data= 0.14ms  step= 630.1ms  data%=0.02  peak=13.85GiB  loss  10.98→  7.68  (20s)
[ 5/8] batch= 4 seq=1024 prec=fp8  → tok/s(step)=   13921  tok/s(+data)=   13916  data= 0.10ms  step= 294.2ms  data%=0.04  peak= 8.03GiB  loss  10.98→  8.40  (11s)
[ 6/8] batch= 8 seq=1024 prec=fp8  → tok/s(step)=   14394  tok/s(+data)=   14391  data= 0.12ms  step= 569.1ms  data%=0.02  peak=13.46GiB  loss  11.05→  8.37  (18s)
[ 7/8] batch=16 seq=1024 prec=fp8  → tok/s(step)=   14980  tok/s(+data)=   14978  data= 0.15ms  step=1093.7ms  data%=0.01  peak=24.32GiB  loss  11.09→  8.33  (35s)
[ 8/8] batch= 4 seq=2048 prec=fp8  → tok/s(step)=   13740  tok/s(+data)=   13737  data= 0.12ms  step= 596.2ms  data%=0.02  peak=13.52GiB  loss  11.01→  8.28  (19s)
sweep done in 2.8 min (168.0s)
```

Headlines:
- Data overhead: 0.01–0.04 % across all configs.
- Peak real-data throughput: 14,980 tok/s @ b=16/s=1024/fp8 — slightly *above* A2's 14,266 random-token peak.
- A3's step-only throughput is consistently 2–6 % HIGHER than A2's at every matched config (likely: prefetched mmap + non_blocking H2D copy overlaps with previous step's optimizer; A2's `torch.randint` ran on the GPU and contended with the model).

## Findings I'm carrying forward

1. **NeMo container's `/opt/venv` trap is the biggest install friction.** Worth a memory note. Same trap will hit anyone trying to layer libraries onto NeMo / Megatron / TRT-LLM containers that ship a uv venv.
2. **PuLP 3.x broke cosmos-xenna.** Pin `pulp<3` in any container that includes Curator 1.x.
3. **Data path is invisible at sub-ms cost on the GB10.** The agent loop in A4 doesn't need to optimize it.
4. **HuggingFace wikitext parquet is row-per-line, not row-per-article.** `WordCountFilter(min_words=50)` is the right floor.
5. **Tokenization at 2.73 M tok/s on CPU.** Single-process; could be parallelized further, but at this corpus size 40 s is already irrelevant.
6. **GB10's unified memory architecture is what makes the data path free.** No PCIe crossing for the H2D copy. Different story on a discrete-GPU box.
