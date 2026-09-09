# Survival of the Fitted

Information loss and code contraction in iterated knowledge distillation.

CSCI semester research project — Sakxam Shrestha.

## The question

Train a network. Throw away its training data, keep only its outputs on a handful of
inputs, and train a fresh network on those. Repeat for suppose twenty-five generations. Nothing in
the chain after generation 0 ever sees a real label it was trained on. 

Two things can be measured at the end.
-  **What the chain computes** — does the last generation agree with the first about anything?
-  **what the chain still says** — which of the fifty output classes are still in use at all? This project measures both, and the
gap between them.

The organising question is whether a distinction from Bayesian iterated-learning theory
transfers to neural chains. Griffiths and Kalish (2007) prove that an iterated-learning
chain converges to the learner's prior, but only for learners that *sample from the
posterior*. A network trained by SGD to a single weight vector is a maximizer, not a
sampler. Whether that distinction governs where a real distillation chain comes to rest is
what the semester is for.

Subsidiary: how do distillation temperature and transmission bottleneck width govern the
rate and route of information loss, and does the bottleneck produce systematic *code
contraction* rather than uniform degradation?

## What this is, and is not

This is a **measurement testing a theoretical prediction**, not a discovery. An adversarial
prior-art search returned no novelty on any headline claim, and that verdict was accepted
rather than argued with. Claims here are scoped to what one architecture on one data
distribution can support, and a clean negative result is treated as a valid outcome.

The project began as "does structure self-organise when a chain of models is seeded on pure
random noise?" That framing was abandoned early. Theory already answers it — Mobahi,
Farajtabar and Bartlett (2020) prove that repeated self-distillation without a ground-truth
anchor collapses the solution's eigenspectrum — and the pilot confirmed it: agreement with
the founding model decays monotonically toward chance in every condition tested. Nothing
self-organises. The noise-seeded chain survives as one controlled arm rather than as the
headline.

## Running it

```
python3 spike_iterated_distill.py --gens 20 --tau 1.0 --bottleneck 4096
python3 spike_iterated_distill.py --gens 20 --tau 1.0 --bottleneck 128
python3 spike_iterated_distill.py --gens 20 --mode argmax
python3 spike_iterated_distill.py --gens 20 --seed-kind structured
```

Roughly 40 seconds per 20-generation chain on a laptop CPU. Requires PyTorch and NumPy; no
GPU needed. Key flags: `--tau` (distillation temperature), `--bottleneck` (samples passed
between generations), `--mode` (`soft` / `sampled` / `argmax`), `--seed-kind` (`noise` /
`structured`), `--gens`, `--seed`, `--log`.

The central experiment is `seed_stability.py`, which runs the same-founder and
cross-founder arms. It currently hardcodes its constants and prints to stdout; giving it a
CLI and JSONL logging is the first task in the plan.

### Device selection is measured, not assumed

| model | CPU | MPS | use |
|---|---|---|---|
| synthetic MLP (92k params, 1500 steps) | **1.61 s** | 2.47 s | **CPU** |
| EMNIST-shaped CNN (212k params, 1500 steps) | 48.2 s | **8.7 s** | **MPS** |

The tiny MLP is slower on MPS because kernel-launch overhead dominates; the convolutional
model is 5.5x faster there. One blanket rule does not fit both. `bench_cnn.py` reproduces
the CNN measurement.

## Design

Inputs are synthetic compositional attribute-value vectors: five attributes of ten values
each, one-hot concatenated to fifty dimensions, with a fixed pool of 4,096 points reused
across all generations so cross-generation comparisons share a reference set. Fifty output
classes.

The founding generation is trained on i.i.d. random labels (the noise arm) or on a
deterministic function of the attributes (the structured arm). Each subsequent generation
is a freshly initialised 50–256–256–50 ReLU MLP of about 92k parameters. It never sees the
original labels; it trains only on the previous generation's soft logits over `B` subsampled
pool points, using a KL loss at temperature `tau`.

A note on what the bottleneck does. The input matrix has rank 46, not 50 — the five
attribute blocks each sum to one, which pins four directions to exactly zero. Every `B` in
the swept grid exceeds 46, so a generic subsample already spans the whole input row space.
**The bottleneck therefore removes no linear information about the input distribution.**
What it restricts is how many (input, target) constraints are available to pin down a
92k-parameter student. This corrects a natural misreading and is verified by
`torch.linalg.matrix_rank` on the real pool.

## Established so far

Every number below was produced by the scripts in this repository, on the machine this work
is being done on. Nothing is imported from a summary or a secondary source.

### Pilot

Seed 0 only, 20 generations, 50 classes. Chance agreement is 0.020; maximum entropy 3.912.

| condition | agreement with gen 0 at g20 | effective rank at g20 | CKA vs previous gen | classes used |
|---|---|---|---|---|
| tau = 0.5, B = 4096 | 0.282 | 52.9 | 0.918 | 50 |
| tau = 1.0, B = 4096 | 0.188 | 41.3 | 0.950 | 50 |
| tau = 2.0, B = 4096 | 0.145 | 36.2 | 0.964 | 49 |
| argmax transmission | 0.689 | 59.5 | 0.769 | 50 |
| tau = 1.0, B = 128 | 0.025 | 27.3 | 0.918 | **19** |

Effective rank at generation 1 is approximately 74 in all conditions.

Three readings. Every condition loses ancestral information monotonically. Temperature
orders the decay rate cleanly but does not create distinct phases — earlier drafts described
"frozen" and "dissolved" regimes with a sharp boundary, which did not reproduce and has been
removed. And the bottleneck is the stronger lever: at `B = 128` agreement reaches chance
within a single generation while the used output alphabet contracts from fifty classes to
nineteen.

### Seed stability of code contraction

`B = 128`, `tau = 1.0`, 25 generations, five runs per arm. Jaccard overlap of the surviving
class sets, against the expected overlap for random subsets of the same size.

| arm | setup | mean survivors | observed Jaccard | random baseline | ratio |
|---|---|---|---|---|---|
| **A** | one shared founder, 5 chains, soft logits | 7.4 of 50 | 0.235 | 0.080 | **2.94x** |
| **B** | 5 different founders, soft logits | 4.4 of 50 | 0.052 | 0.046 | **1.13x** |
| **C** | 5 different founders, *sampled* hard labels | 2.4 of 50 | 0.025 | 0.025 | **1.02x** |

Arm A is systematic; Arms B and C are indistinguishable from chance. The stationary state is
**founder-specific, not architecture-specific**: chains descending from the same generation-0
model land on overlapping surviving classes at roughly three times chance, while chains from
different founders land on unrelated classes.

This falsifies the hypothesis that the chain could be used as an instrument to read out the
architecture's inductive prior. That hypothesis was formulated and tested in the same
session, and the falsification is recorded here because it is the useful part.

Arm C tested the obvious repair and it failed. Replacing deterministic soft-logit
transmission with hard labels *sampled* from the teacher's distribution did not raise
cross-founder overlap; it made contraction more severe. The likely reason matters for the
rest of the project: Griffiths and Kalish require sampling in *hypothesis* space, over the
learner's beliefs. Sampling output labels leaves the learner doing SGD to a single point
estimate on noisier targets. The remaining live test is sampling in weight space, which has
not been tried here.

Note also that the founder's identity persists in the *code* long after it is gone from the
*function*. At `B = 128`, functional agreement with generation 0 reaches chance within one
generation, yet the surviving alphabet still carries founder identity at generation 25.

### On CKA

**CKA is never reported alone in this project.** CKA between consecutive generations climbs
to 0.950 while functional agreement with the founding model sits at 0.188, and CKA against
generation 0 stays flat near 0.65 across the whole chain. It is blind to the degradation
being studied. This is a replication, not a finding — Davari et al. (arXiv:2210.16156)
documented it first, and Ding, Denain and Steinhardt (arXiv:2108.01661) frame the
sensitivity criterion it fails. It is always paired with effective rank and functional
agreement.

## Not yet established

The claim this project intends to defend is broader than what it can currently support. Of
its seven clauses, two are established — path dependence, and the collapse of functional
agreement within one generation. The rest are pending:

- **The controls have not been run.** Eight are specified; zero are complete. Each has a
  competing explanation attached with a citation, and the most dangerous is a shuffled-
  transmission control that could reduce "code contraction" to "frequent classes win by
  frequency."
- **The analytic null does not exist yet.** A ridge-regression chain under the identical
  protocol is a contraction map with an analytically tractable fixed point. Without it, the
  project cannot distinguish its result from regression to the mean.
- **"The code outlives the function" is inferred, not measured.** It rests on survivor-set
  overlap. Permutation-invariant information measures — adjusted mutual information,
  variation of information, adjusted Rand index against the fixed pool — would measure it
  directly.
- **Weight-space posterior sampling is untried.** This is the live hypothesis and the axis on
  which no prior work was found.
- Longer runs (the pilot is 20–25 generations), more founders (n = 5 gives ten
  non-independent pairwise comparisons), and the structured-seed arm.

Twenty-five generations is a horizon, not a proof of non-convergence. Survivor sets are
small, between two and eleven classes, and no class survives in all five runs even in Arm A.
The effect is a partial overlap, not a fixed attractor.

## Repository

| file | what it is |
|---|---|
| `PROJECT.md` | the record of what has been established, with prior-art position and full design |
| `RESEARCH-PLAN.md` | the fourteen-week plan: controls, statistics, the mechanism arm, and the reading list |
| `spike_iterated_distill.py` | runnable pilot; reproduces every number in the pilot table |
| `seed_stability.py` | the central experiment — Arms A and B, and the source of the path-dependence result |
| `sampler_arm.py` | Arm C, sampled hard-label transmission |
| `bench_cnn.py` | CPU/MPS benchmark for the natural-data replication arm |
| `csci-proposal-FINAL.md` | Assignment 02 proposal |
| `CLAUDE.md` | working rules for this repository |

## Working rules

Two rules govern everything in this repository, and both exist because they were violated
once.

**Reproduce before you cite.** Every number in `PROJECT.md`, the proposal, or any report text
must have been produced by a run on this machine. Numbers are not imported from a summary, a
search result, or a prior conversation. A claim that a particular temperature was a "frozen
fixed point" and that argmax transmission was an "absorbing state" was written into the
proposal on that basis; neither reproduced, both decay, and the framing had to be removed.

**Verify citations before adding them.** Check that the DOI or arXiv ID resolves and that the
author list is right. A prior-art report used here cited a paper to the wrong authors while
claiming direct retrieval. Unrefereed preprints are not cited as authority.

## Status

The pilot and the seed-stability experiment are complete and reproduced. The proposal is
written. The plan in `RESEARCH-PLAN.md` runs the controls block first, then two cheap
measurements that cannot come back uninterpretable, then the weight-space sampling arm
behind a preregistered decision gate. Nothing above the controls line should be read as
settled until they have run.
