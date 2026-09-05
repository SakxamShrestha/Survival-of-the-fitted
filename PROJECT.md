# Surviving the Bottleneck

Information loss and code contraction in iterated knowledge distillation.

AI110 semester research project — Sakxam Shrestha.

## Research question

**Does the sampler/maximizer distinction from Bayesian iterated-learning theory govern
where a neural distillation chain comes to rest?**

Griffiths & Kalish (2007) proved that an iterated-learning chain converges to the learner's
prior — but only for learners that *sample from the posterior*. Learners that instead
*maximize* (take a point estimate) are known not to converge to the prior. A neural network
trained by SGD to a single weight vector is a maximizer. The question is whether that
distinction, established for Bayesian agents, predicts the behavior of real neural chains,
and whether making the transmission channel more sampler-like restores convergence.

Subsidiary: how do distillation temperature (tau) and transmission bottleneck width (B)
govern the rate and route of information loss, and does the bottleneck produce systematic
*code contraction* rather than uniform degradation?

## How this differs from the original proposal

The project started as "does structure self-organize when a chain of models is seeded on
pure random noise?" That framing was abandoned for two reasons, both established before
any substantial work was done:

1. **Theory already answers it.** Mobahi, Farajtabar & Bartlett (NeurIPS 2020) prove that
   repeated self-distillation without a ground-truth anchor sparsifies the solution's
   eigenspectrum and collapses it. Griffiths & Kalish (Cognitive Science, 2007) prove that
   iterated-learning chains converge to the learner's prior regardless of founding data.
2. **The pilot confirmed it empirically.** Agreement with the founding model decays
   monotonically toward chance in every condition tested. Nothing self-organizes.

The noise-seeded chain survives as one controlled arm rather than as the headline claim.
The project is now a **measurement** — testing a theoretical prediction in a setting where
it has not been measured — rather than a discovery claim.

## Verified pilot results

All numbers below were produced by `spike_iterated_distill.py` on this machine.
**Seed 0 only, 20 generations, 50 classes, chance agreement = 0.020, max entropy = 3.912.**

| condition | agreement w/ gen 0 @ g20 | effective rank @ g20 | CKA vs prev gen | classes used |
|---|---|---|---|---|
| tau = 0.5, B = 4096 | 0.282 | 52.9 | 0.918 | 50 |
| tau = 1.0, B = 4096 | 0.188 | 41.3 | 0.950 | 50 |
| tau = 2.0, B = 4096 | 0.145 | 36.2 | 0.964 | 49 |
| argmax transmission | 0.689 | 59.5 | 0.769 | 50 |
| tau = 1.0, B = 128 | 0.025 | 27.3 | 0.918 | **19** |

Effective rank at generation 1 is approximately 74 in all conditions.

### What the pilot establishes

- **No structure emerges.** Every condition loses ancestral information monotonically.
- **Temperature orders the decay rate** cleanly, but does not create distinct phases.
  Earlier framings of this project described "frozen" and "dissolved" regimes with a sharp
  boundary between them. That did not reproduce and has been removed.
- **The bottleneck is the stronger lever.** B = 128 drives agreement to chance within one
  generation and contracts the used output alphabet from 50 classes to 19 by generation 20.
  This code contraction is the most interesting effect in the data and is the current
  center of the project.
- **CKA is blind to the degradation.** CKA between consecutive generations climbs to 0.950
  while functional agreement with the founding model sits at 0.188, and CKA against
  generation 0 stays flat near 0.65 across the whole chain.

## Seed-stability of code contraction (`seed_stability.py`)

The central experiment so far. B = 128, tau = 1.0, 25 generations, 5 runs per arm.
Jaccard overlap of the surviving class sets, against the expected overlap for random
subsets of the same size.

| arm | setup | mean survivors | observed Jaccard | random baseline | ratio |
|---|---|---|---|---|---|
| **A** | one shared founder, 5 chains, soft logits | 7.4 of 50 | 0.235 | 0.080 | **2.94x** |
| **B** | 5 different founders, soft logits | 4.4 of 50 | 0.052 | 0.046 | **1.13x** |
| **C** | 5 different founders, *sampled* hard labels | 2.4 of 50 | 0.025 | 0.025 | **1.02x** |

**Arm A is systematic. Arms B and C are indistinguishable from chance.**

### What this establishes

**The stationary state is founder-specific, not architecture-specific.** Chains descending
from the same generation-0 model land on overlapping surviving classes at roughly 3x
chance. Chains descending from different founders land on unrelated classes. Within 25
generations these chains do *not* converge to a common endpoint.

This **falsifies** the hypothesis that the chain can be used as an instrument to read out
the architecture's inductive prior. That hypothesis was formulated and tested in the same
session; it is recorded here because the falsification is the useful part.

It is also what Bayesian iterated-learning theory predicts. Griffiths & Kalish's
convergence-to-the-prior result holds for learners that **sample from the posterior**. An
SGD-trained network taking a point estimate is a **maximizer**, and maximizer chains are
known not to converge to the prior — they retain path dependence on their starting point.
The Arm A / Arm B split is the signature of a maximizing learner.

**Arm C tested the obvious repair and it failed.** Replacing deterministic soft-logit
transmission with hard labels *sampled* from the teacher's output distribution did not
raise cross-founder overlap (1.02x versus Arm B's 1.13x); it made contraction more severe,
dropping mean survivors from 4.4 to 2.4.

The likely reason is a mismatch that matters for the rest of the project: Griffiths &
Kalish require sampling **in hypothesis space** — over the learner's beliefs. Sampling
output labels does not make the learner a posterior sampler; it is still doing SGD to a
single point estimate, now on noisier targets. Sampling in output space is a red herring.
The remaining live test is sampling in **weight space** — SGLD, a deep ensemble, or another
approximate posterior sampler in place of plain SGD. That is the condition the theorem
actually names, and it has not been tried here.

Note also that the founder's identity survives in the *code* long after it is gone from the
*function*: at B = 128 functional agreement with generation 0 reaches chance within a
single generation, yet the surviving alphabet still carries founder identity at generation
25.

### The strongest statement currently supported

Neural distillation chains at a tight transmission bottleneck are **path-dependent**. They
do not converge to a common endpoint under either deterministic soft-logit transmission or
output-sampled transmission. Where a chain comes to rest is a property of its founder, not
of the architecture.

### Caveats

- 25 generations only. Slower mixing could still reach a shared endpoint; Arm B is evidence
  of non-convergence *within this horizon*, not a proof of non-convergence.
- Survivor sets are small (2–11 classes) and n = 5 runs gives 10 pairwise comparisons, so
  the Jaccard estimates are noisy.
- No class survives in all five runs even in Arm A. The effect is a partial overlap, not a
  fixed attractor.

### What is NOT yet verified

- **Weight-space sampling (the live hypothesis).** Replace SGD with SGLD or a deep ensemble
  so the learner approximates a posterior sampler, then rerun the cross-founder arm. If
  overlap rises above the ~1.1x chance level, the sampler/maximizer prediction transfers to
  neural chains; if not, it does not, and the path-dependence result stands on its own.
- 100-generation runs (pilot is 20–25 generations)
- The structured-seed arm
- Whether surviving classes correlate with any property of the input pool
- Larger n. Five runs gives ten pairwise comparisons; the Jaccard estimates are noisy.

## Position against prior art

An adversarial prior-art search returned no novelty on any headline claim. This is
accepted and reflected in the framing above. Verified positions:

- **Closest prior work: Jarvis, Klein, Rosman & Saxe (PNAS 2026, DOI 10.1073/pnas.2509739123)**
  — tracks representational geometry via SVD across iterated-learning generations. Uses
  *deep linear* networks, plain observation rather than knowledge distillation, and no
  temperature parameter. The delta claimed here is: nonlinear MLP, KL-divergence
  distillation with swept tau, swept B, and Neural Collapse instrumentation.
- **Griffiths & Kalish (2007)** — proves analytically that iterated-learning chains
  converge to the learner's prior independent of founding data. This is the theoretical
  prediction the noise-vs-structure arms test empirically.
- **Davari et al. (ICLR 2023, arXiv:2210.16156)** — already documented CKA reporting high
  similarity between a generalizing network, a random-label memorizing network, and an
  untrained network (~70% vs ~10% linear probe accuracy). **The CKA observation in this
  project replicates a known result and must not be presented as novel.**
- **Mobahi et al. (NeurIPS 2020)**, **Borup & Andersen (NeurIPS 2021)** — theoretical
  guarantee of collapse in exactly the anchor-free chaining regime used here.
- **Kirby (2001)**, **Smith, Kirby & Brighton (2003)** — the transmission bottleneck as a
  controlled variable is the founding methodology of iterated learning, not a contribution
  of this project.

Two further citations surfaced by the prior-art search — Flouro & Chadwick
(arXiv:2601.13100) and Du et al. (arXiv:2608.08572) — were confirmed to exist but are
unrefereed preprints, and Flouro & Chadwick contains no empirical work. Neither should be
cited as authority.

## Design

- **Data.** Synthetic compositional attribute-value inputs: 5 attributes x 10 values,
  one-hot concatenated to 50 dimensions, fixed pool of 4,096 points reused across all
  generations so cross-generation comparisons share a reference set. 50 output classes.
- **Founding generation.** Noise arm: i.i.d. random labels (the Zhang et al. 2017
  random-label regime). Structured arm: labels are a deterministic function of the
  attributes.
- **Chain model.** 50–256–256–50 ReLU MLP, ~92k parameters. Each generation is initialized
  from scratch, never sees the original labels, and trains only on the previous
  generation's soft logits over B subsampled pool points using KL loss at temperature tau.
- **Independent variables.** tau in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}; B in {64, 128, 256,
  512, 1024, 4096}; founding seed in {noise, structured}; transmission mode in {soft,
  sampled, argmax}. Five random seeds per cell.
- **Primary metrics.** Functional agreement with generation 0 and n−1; effective rank of
  the penultimate-layer covariance spectrum; output entropy; mean logit norm; surviving
  output alphabet size.
- **Null model.** A ridge-regression chain under the identical protocol, whose fixed point
  is analytically computable. Iterated distillation is a contraction map, so falling rank
  is the boring default; without this null the project cannot distinguish self-organization
  from regression to the mean.

## Compute budget for a natural-data replication arm

Measured with `bench_cnn.py` using random tensors of EMNIST-balanced's exact shape
(28x28x1, 47 classes) — this benchmarks wall-clock cost only, not whether the dynamics
transfer. CNN is conv16-conv32-fc128-fc47, 211,695 parameters, 1500 steps per generation.

| | per generation | 10 founders x 25 gens |
|---|---|---|
| synthetic MLP, CPU | 1.6–2.0 s | ~10 min |
| EMNIST-shaped CNN, CPU | 43–49 s | ~3.4 h |
| **EMNIST-shaped CNN, MPS** | **8.7 s** | **~40 min** |

On MPS the full external-validity arm — cross-founder and same-founder overlap, 10 founders
each at 25 generations — costs roughly **1.3 hours**. Extending it to the full learner-rule
design (10 founders x 3 rules x 100 generations) costs about **7 hours**, i.e. one overnight
run. The natural-data arm is affordable and should be written into the plan.

Levers if it needs to be cheaper: 1500 steps over a 4,096-point pool is ~48 epochs, which
is likely more than needed; halving it roughly halves the cost.

**Google Colab Pro is available as headroom** if a larger arm is wanted — a T4 or A100
would cut the CNN generation cost well below the 8.7 s measured on MPS and would put
CIFAR-100 (100 balanced classes) within reach. Treat it as insurance, not as the plan: the
core design's strength is that it runs locally, and a design that *needs* rented compute
loses the ability to run ten founders per condition on a whim. Local first, Colab only for
a scale-up arm.

**Dataset criteria** (in priority order): known generative factors, balanced classes,
30+ classes so the alphabet has room to contract, and small inputs so many chains are
affordable. EMNIST-balanced (47 classes, balanced by construction) is the closest match to
the synthetic setup's 50 classes. dSprites and Shapes3D are the options with known latent
factors. MNIST and Fashion-MNIST have too few classes to measure contraction. Generic
tabular datasets are unsuitable — unknown factor structure, usually imbalanced, and the
analytic null is lost.

## Files

| file | what it is |
|---|---|
| `ai110-proposal-draft.md` | Assignment 02 proposal. Needs the four prior-art revisions listed above. |
| `spike_iterated_distill.py` | Runnable pilot. Reproduces every number in this document. |

```
python3 spike_iterated_distill.py --gens 20 --tau 1.0 --bottleneck 4096
python3 spike_iterated_distill.py --gens 20 --tau 1.0 --bottleneck 128
python3 spike_iterated_distill.py --gens 20 --mode argmax
python3 spike_iterated_distill.py --gens 20 --seed-kind structured
```

Roughly 40 seconds per 20-generation chain on a laptop CPU. No GPU required.

## Open work

1. Code-contraction seed stability at B = 128 across 5 seeds — does the same subset of
   classes survive?
2. Apply the four prior-art revisions to the proposal (reframe as prediction-test, cite
   Griffiths & Kalish, cite Jarvis et al. with explicit delta, downgrade the CKA claim).
3. Build the ridge-regression null chain.
4. Extend to 100 generations and multiple seeds.
5. Structured-seed arm.
