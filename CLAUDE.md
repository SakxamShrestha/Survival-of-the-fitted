# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this repository is

Two unrelated things share this directory, which is a known wart:

1. **The live work** — a CSCI semester research project on iterated knowledge
   distillation. See `PROJECT.md` for the full record. Files: `PROJECT.md`,
   `csci-proposal-draft.md`, `spike_iterated_distill.py`.
2. **A separate CSCI assignment** — the "Playlist Chaos" Streamlit debugging exercise
   (`app.py`, `playlist_logic.py`, `README.md`, `requirements.txt`). These are the repo's
   git history and its remote. As of the last session they are **deleted from the working
   tree but still present at HEAD**; `git restore .` recovers them.

Do not assume a task about "the project" means the research project without checking which
files the user is pointing at.

## Running the experiment

```
python3 spike_iterated_distill.py --gens 20 --tau 1.0 --bottleneck 4096
```

Requires PyTorch and NumPy. Torch 2.8.0 is installed system-wide; there is no project
virtualenv for the research code. torchvision is **not** installed, and the disk is near
full (about 8 GB free), so avoid large dataset downloads without checking first.

### Device selection — measured, not assumed

| model | CPU | MPS | use |
|---|---|---|---|
| synthetic MLP (92k params, 1500 steps) | **1.61 s** | 2.47 s | **CPU** |
| EMNIST-shaped CNN (212k params, 1500 steps) | 48.2 s | **8.7 s** | **MPS** |

The tiny MLP is slower on MPS because kernel-launch overhead dominates; the convolutional
model is 5.5x faster there. Do not apply one blanket rule to both. `bench_cnn.py`
reproduces the CNN measurement.

Google Colab Pro is available if a scale-up arm is needed. Do not design around it — the
core experiment must stay runnable locally, because its statistical power comes from
running many independent chains cheaply. Reach for Colab only for an explicit scale-up
(e.g. CIFAR-100), never for the main grid.

Key flags: `--tau` (distillation temperature), `--bottleneck` (samples passed between
generations), `--mode` (`soft` / `sampled` / `argmax`), `--seed-kind` (`noise` /
`structured`), `--gens`, `--seed`, `--log` (JSONL output).

## Working rules specific to this project

### Reproduce before you cite

Every number that goes into the proposal or `PROJECT.md` must have been produced by a run
on this machine. Do not import numbers from a subagent, a web source, or a prior
conversation summary without re-running them.

This rule exists because it has already been violated once: a subagent reported that
`--tau 0.5` was a "frozen fixed point at agreement 1.000" and that `--mode argmax` was an
absorbing state. Neither reproduced. Both decay; temperature only orders the rate. A
"phase boundary" framing was written into the proposal on that basis and had to be removed.

If a number cannot currently be reproduced, mark it explicitly as unverified rather than
stating it.

### Verify citations before adding them

Check the DOI or arXiv ID resolves, and check the author list. A prior-art report used in
this project cited "A Tale of Tails" as Dohmatob, Alemohammad & Shumailov while claiming
direct retrieval; the actual authors are Dohmatob, Feng, Yang, Charton, Kempe.

Do not cite unrefereed preprints as authority. Two are known to be real but unsuitable:
arXiv:2601.13100 (contains no empirical work) and arXiv:2608.08572.

### Terminology — four different things, do not blur them

- **Model collapse** — distributional degradation in *generative* chains trained on their
  own output. Tails first, then variance.
- **Neural collapse** — terminal-phase last-layer geometry (simplex ETF) in a *supervised
  classifier*. Requires class labels, balanced classes, and training past zero error.
- **Rank collapse** — loss of effective rank in representations. Spectral, label-free.
- **Mode collapse** — a generator covering few modes of the target. GAN pathology.

Earlier drafts of this project treated neural collapse as the *escape* from model collapse.
They are unrelated and live at different levels. What the pilot actually measures is
closest to rank collapse.

### Metrics

- **Never report CKA alone.** CKA between consecutive generations reaches 0.950 in this
  setup while functional agreement with the founding model is 0.188. It is blind to the
  degradation being studied. Always pair it with effective rank and functional agreement.
- **The CKA-blindness observation is not novel.** Davari et al. (arXiv:2210.16156) already
  documented it, including the random-label arm. Report it as replication.
- **Topographic similarity does not apply here** and was cut deliberately. It requires a
  ground-truth meaning space and a discrete message channel; soft logits are neither. Do
  not reintroduce it.
- **Neural Collapse metrics require a class partition.** They are computed against the
  frozen generation-0 label partition, which is well-defined because generation 0 has
  explicit labels even in the noise arm. NC4 against the model's own argmax is trivially
  near 1 and is not informative.

### Framing

This project is a **measurement testing a theoretical prediction**, not a discovery. An
adversarial prior-art search returned no novelty on any headline claim, and that verdict
was accepted rather than argued with. Do not reintroduce discovery language ("prove that
neural networks have a built-in tendency…", "structure emerges from noise"). Claims must be
scoped to what one architecture on one data distribution can support.

A clean negative result is a valid outcome here and is stated as such in the proposal.

## Style

Written deliverables — the proposal, `PROJECT.md`, report text, commit messages — are
normal academic prose for a human reader. Keep the compressed conversational style out of
files.
