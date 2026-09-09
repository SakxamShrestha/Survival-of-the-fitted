# Research plan — Surviving the Bottleneck

Semester plan for the remainder of the CSCI project, written 2026-09-04. Read alongside
`PROJECT.md`, which is the record of what has been established; this document is the record
of what is planned and why.

## 1. Where the project stands

The pilot and the seed-stability experiment are complete and reproduced on this machine.
The proposal (`csci-proposal-FINAL.md`) is written. The declared open work is: a
weight-space posterior-sampling arm, a ridge-regression null chain, longer runs, more
founders, and the structured-seed arm.

Three things surfaced in planning that change what should be done next. Each is recorded
here with its verification status, because two of them contradict statements currently in
`PROJECT.md`.

### 1.1 The maximizer framing in `PROJECT.md` is probably too strong

`PROJECT.md` currently states that "maximizer chains are known not to converge to the prior
— they retain path dependence on their starting point," and reads the Arm A / Arm B split
as the signature of that.

The relevant paper is Kirby, Dowman & Griffiths (2007), *PNAS* 104(12), 5241–5245,
"Innateness and culture in the evolution of language" (DOI 10.1073/pnas.0608222104). Its
existence, author list, venue and page range are verified. Its headline result is that
cultural transmission through learners with weak innate biases **magnifies those biases
into strong universals**, and that the magnification gets stronger as the bottleneck
tightens.

If that applies to MAP learners in the way it appears to, the prediction for a maximizer
chain is not "no convergence" but "convergence to a *sharpened* version of the prior,"
with the sharpening controlled by the bottleneck. That is a different claim, and under it
the Arm A / Arm B split is not straightforwardly the predicted signature of maximization.

**Status: the paper is verified; the specific MAP-versus-sampler claim is not.** The PDF
text could not be extracted automatically. This must be read first-hand in week 1 and the
sentence in `PROJECT.md` rewritten to whatever the paper actually supports. It is the
single most likely place for an examiner to land.

### 1.2 A directly competing paper now exists, and it is refereed

**Guo, D., Wu, J., & Yiu, S. M. — "Model Collapse as Cultural Evolution", arXiv:2605.23054,
accepted at CoNLL 2026.** Verified directly: title, three-author list, abstract and the
acceptance note in the comments field.

It frames model collapse through cultural-evolution theory, derives and tests five
predictions by iteratively training LLaMA-2-7B and Mistral-7B over 10 generations in
multiple languages, reports a non-monotonic compositionality trajectory under unfiltered
self-training, and finds evidence for a compression–communication trade-off with
task-grounded filtering sustaining language quality.

This postdates the project's earlier adversarial prior-art search, which is why it did not
appear there. It occupies the "model collapse is iterated learning" framing at LLM scale in
a refereed venue.

The remaining delta is narrower but still real, and must be stated explicitly rather than
left implicit:

- a controlled synthetic task with an exact chance baseline and an analytically computable
  overlap null, where their setting has neither;
- temperature and bottleneck width as swept independent variables;
- the **learner's inference rule** as the manipulated variable, which they do not vary.

The third of these is the strongest and should become the spine of the project (§3).

### 1.3 The input pool has rank 46, not 50 — verified on this machine

Computed directly from `make_pool(seed=0)`:

```
X shape (4096, 50), rank(X) = 46
top eigenvalue of X^T X / n = 0.5016, multiplicity 1
four eigenvalues exactly 0
remaining 45 eigenvalues spread over roughly 0.082 – 0.120
every row of X sums to 5
```

The rank deficiency is structural: with five attribute blocks of ten one-hot values each,
the five block-sum directions are all pinned to the constant 1, leaving four exact null
directions. The all-ones direction carries eigenvalue k/v = 0.5 and is **constant on the
data**, so whether it participates at all depends on whether a model has a bias term.

Two consequences:

1. **Every bottleneck width in the declared grid {64, 128, 256, 512, 1024, 4096} exceeds
   rank(X) = 46.** A generic subsample of B ≥ 46 pool rows spans the entire row space. The
   bottleneck therefore removes *no* linear information about the input distribution; it
   restricts only the number of (input, target) constraints relative to a 92k-parameter
   student. The proposal currently reads as though B limits input coverage. It does not,
   and that should be corrected.
2. Analysis of the ridge null must use the *empirical* pool spectrum, not an idealized
   degenerate one. A population-level derivation gives a 45-fold degenerate bulk, and a
   degenerate bulk would predict pure scalar shrinkage — no shape change, no agreement
   decay, no alphabet contraction. **That prediction does not hold for the actual pool**,
   whose bulk is already spread over roughly a 1.5x range at n = 4096. The null is
   *near*-degenerate, not degenerate, and its predictions must be computed numerically from
   the real `X`, not from the closed form.

## 2. The decision this plan makes

**We must choose what the manipulated variable of the semester is, and the difficulty is
that every cheap arm has a boring explanation an examiner can reach for first.**

The catalogue of boring explanations, all of which currently apply to the existing results:

| result | boring explanation | citation an examiner would use |
|---|---|---|
| alphabet contracts 50 → 19 at B = 128 | minority collapse under extreme class imbalance (~2.6 transmitted examples per class) | Fang, He, Long & Su (2021), *PNAS* 118(43), DOI 10.1073/pnas.2103091118 |
| agreement with generation 0 decays monotonically | per-step distillation slack compounding; a single distillation step already loses a lot | Stanton, Izmailov, Kirichenko, Alemi & Wilson (2021), arXiv:2106.05945 |
| effective rank falls | iterated fitting is a contraction map; regression to the mean | Dohmatob, Feng & Kempe (2024), arXiv:2402.07712 |
| same-founder chains overlap, different-founder chains do not | SGD outcomes are path-dependent on early conditions, a mundane property | Frankle, Dziugaite, Roy & Carbin (2020), arXiv:1912.05671 |
| the chain collapses at all | the protocol replaces data rather than accumulating it, the known-worst case | Gerstgrasser et al. (2024), arXiv:2404.01413 |

Every one of these is cheap to address and none of them has been addressed. That is the
project's actual weakness — not a shortage of experiments, a shortage of controls.

**The pick: the learner's inference rule is the manipulated variable, run as a dose–response
sweep rather than a binary contrast, and gated behind a block of controls that runs first.**

The reason it is the right spine: it is the one axis where no prior work was found. A
literature search returned no paper that places an approximate weight-space posterior
sampler inside an iterated-learning or distillation chain and measures convergence. It is
also the axis the theory actually names, it is the axis Guo et al. do not vary, and — as
described in §4.2 — it can be implemented at roughly 1x compute, which matters when the
design calls for many independent chains.

**The assumption it depends on most:** that a cheap approximate sampler is different enough
from SGD to move the measured quantity at all. If the sampler arm is inert, the result is
uninterpretable — a null could mean "the sampler/maximizer distinction does not transfer"
or "my sampler was not a sampler." Izmailov, Vikram, Hoffman & Wilson (2021, arXiv:2104.14421)
show, using full-batch HMC as ground truth, that SG-MCMC methods and deep ensembles produce
predictive distributions distinctly different from the true posterior, so this is a real
risk rather than a hypothetical one.

**The cheapest thing that would prove the assumption wrong, doable in one afternoon:** train
two students on *identical* transmitted targets, differing only in the sampler's randomness,
and measure functional agreement between them. If they agree at near 1.0, the sampler is
inert and the whole arm is measuring nothing. Run this before any grid.

**What the pick gives up:** the EMNIST natural-data arm moves to the end of the semester and
may not happen. That is the correct trade — external validity for a mechanism that has not
been pinned down is worth little, and the compute budget in `PROJECT.md` shows the EMNIST
arm can be added later in about 1.3 hours if time allows.

### Options considered and not taken

**Transmission protocol as the spine** — sweep an anchor rate r (fraction of targets that
are true generation-0 labels) or compare replacing versus accumulating transmitted data.
Cheap, has a sharp citable prediction, and Gerstgrasser et al. prove accumulation is stable
where replacement is not. Rejected as the *spine* because it is the axis most thoroughly
covered by existing work; kept as a supporting arm (§4.4).

**Rigor-only: nulls and controls as the whole semester.** Defensible and it is what the
project most needs, but a semester whose entire output is "our earlier result survives four
controls" has no new measurement in it. Rejected as the whole, adopted as the first block.

**Do nothing new — scale n and generations on the existing arms.** n = 5 gives ten
non-independent pairwise comparisons and the Jaccard estimates are noisy, so more founders
genuinely helps. But scaling an uncontrolled result buys precision on a number that may not
survive the controls. Rejected; folded into §4.1 as something done *after* the controls.

### Pre-mortem on the pick

It is December and this was a disaster. The most likely reason, written in advance: the
sampler arm needed a proper prior over the weights to be well-defined, the scale of that
prior turned out to be a free parameter with a large effect, and no principled way to set it
was found — so the arm produced a curve whose shape is a function of an arbitrary choice.
The second most likely reason: at B = 128 with 92k parameters, an honest posterior is
prior-dominated and the chain reaches its stationary state in one generation, which
technically confirms the theory but measures nothing.

Both are mitigable and the mitigations are in §4.2. Neither is a reason to pick a different
spine, but both are reasons to run the sampler arm at more than one bottleneck width and to
report the prior scale as a swept variable rather than a fixed constant.

## 3. Corrections to make in week 1, before any new experiment

These cost hours, not days, and each of them changes what the final report says.

1. Rewrite the maximizer sentence in `PROJECT.md` to match what Kirby, Dowman & Griffiths
   actually establish (§1.1). Read the paper first.
2. Add Guo et al. (arXiv:2605.23054) to the prior-art section with an explicit statement of
   the remaining delta (§1.2).
3. Correct the description of the bottleneck in `PROJECT.md` and in the proposal: B does not
   limit input coverage, because every B in the grid exceeds rank(X) = 46 (§1.3).
4. Fix the docstring in `sampler_arm.py`. It calls sampled hard-label transmission a
   "sampler-like learner". That is the exact conflation `PROJECT.md` already corrected in
   prose: sampling in output space does not make the learner a posterior sampler.
5. Delete the stale claims in the `spike_iterated_distill.py` module docstring — it still
   says `--tau 0.5` is a "frozen fixed point" and `--mode argmax` an "absorbing state".
   Neither reproduced, `PROJECT.md` says so, and the docstring is the last place those
   claims survive.
6. Check the CKA attribution. `PROJECT.md` credits Davari et al. (arXiv:2210.16156) with a
   random-label arm and ~70% versus ~10% linear-probe accuracies. The abstract does not
   mention them. Verify against the paper body or soften the claim to what the paper
   supports.

## 4. Workstreams

### 4.1 Controls and nulls — weeks 2–5, run before anything new

Each of these is cheap and each one can invalidate a current headline claim. That is why
they come first.

**C-1. Shuffled transmission.** Permute the teacher's logit rows across the B selected
inputs before training, destroying the input→target correspondence while preserving the
marginal distribution of transmitted logit vectors exactly. One line. If alphabet
contraction survives shuffling, "code contraction" reduces to "frequent classes win by
frequency" and the framing weakens substantially. This is the most dangerous single control
and should be run first.

**C-2. No chaining.** Train G independent students all distilled directly from generation 0
at bottleneck B, and compare student g against chain-generation g. If chain-generation 5
looks like a fresh one-step student, iteration is doing nothing and the iterated framing is
unsupported. Currently absent from the design and close to mandatory.

**C-3. One-step fidelity baseline.** Agreement with generation 0 after a *single*
distillation step at B = 4096. Stanton et al. (arXiv:2106.05945) show substantial
teacher–student discrepancy after one step even with matched capacity and no bottleneck,
attributed to optimization rather than information loss. Whatever that number is, it is the
floor against which all downstream decay must be read.

**C-4. Matched-imbalance control.** At B = 128 over 50 classes the transmitted set averages
about 2.6 examples per class. Fang et al. (2021) characterize minority collapse
analytically in exactly that regime. Either construct a matched-imbalance non-chained
control, or establish that soft-logit transmission places mass on all classes and therefore
evades the mechanism. Without this, the best result in the project has a competing
explanation with a *PNAS* citation attached.

**C-5. Noise-source decomposition.** `seed_stability.run_chain` seeds student
initialization and subsample selection together, so a "chain seed" varies both. Split them:
fixed subsample sequence with varying init isolates optimization noise; fixed init sequence
with varying subsample isolates channel noise. Two lines. This maps directly onto the ridge
null, which has channel noise and no optimization noise.

**C-6. Held-out probe pool.** Every metric is currently computed on the same 4,096-point
pool the chain trains on; at B = 4096 those are training metrics. Generate a second pool
from `make_pool` with a different seed and never train on it. Three lines.

**C-7. Ridge-regression null chain.** Each generation solves ridge regression on the
previous generation's predictions at B subsampled points. The iteration is exactly linear:
W_g = A_g W_{g-1} with A_g = (Σ̂_g + λI/B)⁻¹ Σ̂_g, so W_g is a product of independent random
PSD shrinkage matrices and the unique fixed point is W = 0. What matters is not the scale,
which dies trivially, but the *shape* W_g/‖W_g‖, since every metric except `logit_norm` is
scale-invariant. **Compute its predictions numerically from the real pool, not from the
idealized degenerate spectrum** (§1.3). Compare against the MLP at *matched agreement decay
rate*, tuning λ/B, not at matched B — the parameter counts differ by a factor of 37.

Present this null as an instantiation of an established analytic framework, not as an
invention: iterated linear fitting on previous generations' outputs is exactly the setting
of Dohmatob, Feng & Kempe (arXiv:2402.07712), and is used and extended by Gerstgrasser et
al. (arXiv:2404.01413).

**C-8. Frozen-body / random-features null.** Keep the MLP but freeze `model.body` at random
initialization and train only the head each generation. The chain becomes linear regression
in a fixed 256-dimensional random feature space, `eff_rank(feats)` is pinned by
construction, and any surviving alphabet contraction is purely a readout phenomenon. This
is also the bridge to the sampler arm: with the body frozen, the head is a GLM and its
Laplace posterior is closed-form, so the analytic prediction and the experiment must agree.
That is a correctness check on the sampler implementation that nothing else in the design
provides.

### 4.2 The learner-rule arm — weeks 6–10, the spine

**Design principle: sampler versus maximizer is not binary, it is a posterior temperature.**
Target p(w | D) ∝ p(D | w)^λ · p(w). Large λ is the current maximizer arm; λ = 1 is honest
Bayes. One scalar connects the existing Arm B continuously to the new arm, which turns a
yes/no into a dose–response curve of cross-founder overlap against λ. There is direct
precedent for interpolating between MAP and sampling learners in the iterated-learning
literature (Kirby, Dowman & Griffiths, 2007).

**Two prerequisites that are easy to miss:**

- *There is currently no prior for the chain to converge to.* Every `train()` call passes
  `wd = 0.0`, which is an improper flat prior over 92k weights. "Converges to the learner's
  prior" is undefined in that setting. The arm must specify a proper prior, e.g.
  w ~ N(0, σ²I), and σ is not a nuisance parameter — it *defines* the attractor the theory
  predicts. That yields a free falsification test: two values of σ should produce two
  different attractors, with within-σ cross-founder overlap exceeding across-σ overlap.
- *SGLD, variational inference and anchoring all need a log-likelihood, and KL-to-soft-logits
  is not one.* Sampled hard labels give an honest categorical likelihood with N = B
  observations. This is a second reason to build the sampler arm on top of `sampler_arm.py`
  rather than instead of it.

**Griffiths & Kalish require two sampling steps, not one.** Their chain is a Gibbs sampler
on the joint distribution of hypotheses and data: the producer samples data given a
hypothesis, the learner samples a hypothesis given the data. `sampler_arm.py` already
implements the *production* step correctly. What it lacks is the *learning* step. The
correct experiment is sampled hard labels **and** posterior-sampled weights.

**Method ranking** — faithfulness to what the theory requires, against cost here:

| method | genuine posterior sampler? | cost vs SGD | rough LoC | role |
|---|---|---|---|---|
| anchored / randomized-MAP, K = 1 | yes; exact in the linear-Gaussian case | **1.0x** | ~5 | primary |
| cyclical SG-MCMC | yes; designed for multimodal posteriors | 2–4x | ~30 | second arm |
| preconditioned SGLD | yes, biased | ~1.0x | ~25 | fallback |
| SGHMC | yes, biased | ~1.0x | ~15 | better default than plain SGLD |
| last-layer Laplace | Gaussian centered at the MAP | ~1.05x | ~20 | correctness check against C-8 |
| SWAG | yes, but strictly within-basin | ~1.2x | ~40 | locality axis |
| deep ensemble, unanchored | **no** — K independent maximizers | Kx | ~10 | confound, not a test |
| MC dropout | **no** — fixed mask distribution around one point | 1.0x | ~3 | **negative control** |
| full-batch HMC | gold standard | 20–100x | ~60 | one calibration point only |

**Primary method: anchored ensembling at K = 1 per generation.** Draw w_anchor ~ N(0, σ²I)
at the start of each generation, then minimize NLL + ‖w − w_anchor‖² / (2σ²). It is a
principled weight-space posterior sampler with a stated exactness condition (exact in the
linear-Gaussian case; approximate otherwise, with a published and honestly characterized
failure mode — correctly centered, marginal variance underestimated). It costs nothing
extra, it is about five lines against the existing `train()`, and there is no ambiguity
about what gets transmitted because the trained student *is* the weight sample. Key
citations: Pearce, Leibfried, Brintrup, Zaki & Neely (AISTATS 2020, arXiv:1810.05546 — note
the five-author list, it is commonly miscited as three); exactness result Bardsley, Solonen,
Haario & Laine (*SIAM J. Sci. Comput.* 36(4):A1895–A1910, 2014, DOI 10.1137/140964023).

**Mandatory negative control: MC dropout**, explicitly labelled non-Bayesian. It is
stochastic but not posterior-approximating — the randomness is a fixed mask distribution
around one SGD-found point — so it should *not* raise cross-founder overlap. If it does, the
effect is "noise", not "posterior sampling", and the headline claim collapses. Three lines,
1.0x cost, and it must be in place before the result exists rather than added afterwards.

**Extraction — what does generation n+1 actually train on?** Once the teacher is a
distribution over weights this is a real design question with non-interchangeable answers:

- **(a) one weight sample → one teacher → sampled hard labels.** The only option that
  instantiates the Gibbs chain. This is the arm.
- **(b) posterior-predictive average → sampled hard labels.** Marginalizing the hypothesis
  before producing breaks the Gibbs argument; the chain no longer targets the prior. A
  legitimate *contrast* arm, not the theory's condition.
- **(c) posterior-predictive average → soft logits.** Furthest from the theory; equivalent
  to ensemble distillation. Useful only to show that averaging the channel is not what
  matters.
- **(d) pass weights directly.** Breaks the bottleneck abstraction. Not viable.

The (a)-versus-(b) contrast is itself a result worth reporting: it separates sampling in
hypothesis space from marginalizing over hypotheses, and only (a) is predicted to work.

**Inertness check, before any grid** (this is the falsification test from §2): two students,
identical transmitted targets, different anchor draws. If they agree functionally, σ is too
large and the arm is measuring nothing. Report weight-space distance travelled and the
noise-to-gradient-norm ratio as standing diagnostics.

**Caveat to state explicitly in the report:** AdamW at constant learning rate already has a
stationary distribution approximating a very cold, badly-scaled posterior (Mandt, Hoffman &
Blei, *JMLR* 18, 2017, arXiv:1704.04289). The existing arm should be described as sampling
a very cold posterior, not as a pure maximizer. This is what licenses the λ-sweep framing.
Relatedly, posterior temperature is a live dispute in the literature — Wenzel et al. (ICML
2020, arXiv:2002.02405) report tempered posteriors outperforming T = 1; Izmailov et al.
(arXiv:2104.14421) dispute the explanation. Cite both; the λ-sweep lands directly in that
dispute, which is a strength.

**One further caveat:** weight-space is massively non-identifiable under neuron permutation
and ReLU rescaling. The project's observable — the surviving class alphabet — is a
function-space quantity and is safe, but raw ‖w₁ − w₂‖ is meaningless as evidence and must
not be reported as such.

### 4.3 Statistics and metrics — weeks 3–9, interleaved

**The Jaccard analysis has three defects and they are all fixable without new runs.**
`expected_jaccard(k, C)` computes a ratio of expectations rather than an expectation of the
ratio, which is a Jensen bias of unknown sign at survivor-set sizes of 2–11; it plugs the
*mean* size into a nonlinear formula, which is a second Jensen bias; and the ten pairwise
Jaccards from five runs are not independent, so any standard error or t-test on them is
invalid.

Replacements, in priority order:

1. **Cross-arm permutation test as the primary statistic.** Pool the chains of Arms A and B,
   randomly reassign the same-founder / different-founder labels 10⁴ times, recompute the
   difference in mean pairwise Jaccard. This tests the actual claim and needs no baseline
   formula at all.
2. **Exact resampling null conditioned on the observed survivor-set sizes.** Fixes all three
   defects at once because the null is over the identical statistic with the identical
   dependency structure. About ten lines of NumPy.
3. **Frequency-matched null.** Draw survivors with probability proportional to the founder's
   generation-0 class frequency. If the effect vanishes, "founder identity" reduces to
   "founder class-frequency imbalance" — the boring explanation the current analysis cannot
   exclude.
4. **Report the overlap coefficient alongside Jaccard.** Jaccard is size-dependent and
   survivor sizes differ systematically across arms, so the Arm A versus Arm C comparison is
   partly a size comparison. The overlap coefficient removes that confound.
5. **Raise n to 20 founders per arm.** Five gives essentially no resolution.

**Recast the central claim as a classification problem.** Jaccard overlap of survivor sets
is a weak, size-confounded, small-n statistic for what is really a group-separation
question. Train F founders, run R chains from each, summarize each descendant by its
survivor indicator or mean softmax or feature eigenspectrum, and report leave-one-chain-out
balanced accuracy of founder identification against 1/F chance with a permutation test.
This should become the headline statistic, with Jaccard demoted to a descriptive.

**Add information-theoretic transmission measures. This is the most important metric
addition.** Agreement with generation 0 is permutation-*sensitive*: a descendant that
partitions the pool exactly as the founder does but relabels every class scores at chance.
Mutual information, variation of information and the adjusted Rand index between the two
labelings of the fixed pool are permutation-*invariant*. `PROJECT.md` claims the founder's
identity survives in the code long after it is gone from the function; that claim is
currently supported only by survivor-set overlap, and these metrics measure it directly. If
I(ŷ₀; ŷ_g) stays above its permutation null while agreement sits at chance, that is the
project's cleanest and most defensible result.

*Estimator trap:* plug-in mutual information on a 50×50 table with 4,096 samples is biased
upward by roughly (r−1)(s−1)/2N with r, s the occupied row and column counts. Because the
alphabet contracts, r shrinks with g, so the bias shrinks with g and the raw curve's shape
is partly an artifact. Use adjusted mutual information, or subtract a per-generation
permutation null computed with the same estimator on the same table shape.

**Generalize alphabet size to a family.** `n_classes_used` is a hard threshold in which a
class used once counts as much as one used a thousand times. Report Hill numbers of the
predicted-class distribution: N₀ (the current metric), N₁ = perplexity, N₂ = inverse
Simpson, N∞. This separates *losing* classes from *re-weighting* them, and it makes the
metric set one family indexed by q, since effective rank is already the q = 1 Hill number of
the eigenvalue spectrum.

**Replace CKA with measures that are not blind.** Three cheap ones, all a few lines:
principal angles between top-k feature eigenspaces (weights every retained direction
equally, so a single large direction cannot saturate it); orthogonal Procrustes distance
(invariant to rotation but not to general linear maps, so the anisotropic shrinkage a
contraction chain produces is not washed out); and a closed-form ridge linear probe of the
founder's labels from a descendant's frozen features, which distinguishes "the founder's
structure is gone" from "present but discarded by the descendant's own readout" — the
code-versus-function claim moved to the representation level. Methodological justification:
Ding, Denain & Steinhardt (NeurIPS 2021, arXiv:2108.01661) frame exactly the sensitivity /
specificity criterion needed here. Note that this is a *second* independent prior on CKA's
insensitivity alongside Davari et al., which downgrades the project's planned CKA note from
a warning to a confirmation in a third setting.

**Also worth adding:** eigenvalue decay exponent (OLS slope of log λ_i on log i over a fixed
index window), which is the quantity Mobahi, Farajtabar & Bartlett's spectral-sparsification
claim is actually about and which the project cites but does not measure; stable rank and
participation ratio alongside effective rank, to separate "the tail died" from "one direction
grew"; and the mean symmetric KL between generation-0 and generation-g outputs, at
transmitted points versus held-out points — the gap is the generalization error of the
transmission, which is the most direct available measurement of what the bottleneck loses.

**Document which effective rank is being reported.** The implementation uses p_i ∝ σ_i²
(variance shares); the standard definition uses p_i ∝ σ_i. Both are defensible, they are
different numbers, and the proposal must say which one or the value is not reproducible from
the definition.

**Per-class survival regression — free, and the analysis most likely to change the
conclusion.** Regress each class's survival at generation 25 on predictors already available
in the logs: generation-0 predicted frequency, generation-0 top-1 margin, the founder head's
column norm and column sum, and mean feature norm of assigned points. If founder class
frequency alone explains survival, the founder-identity story reduces to "frequent classes
survive." Costs no new runs.

**On topographic similarity:** not reintroduced, for the reasons already recorded in
`CLAUDE.md`. The compositionality question it would have answered has a legitimate
substitute, because the inputs have known generative factors: per-attribute mutual
information I(attr_j; ŷ_g) for j = 1..5, plus the fraction of ŷ_g explained by an additive
attribute model. This is supervised decodability on a known factor basis, not a correlation
between distances in two spaces, and it must not be described as topographic similarity. It
is the right instrument for the structured-seed arm, where labels depend only on attributes
1 and 2, so the chain can be watched forgetting *which attributes matter*.

### 4.4 Supporting arms — weeks 10–12, as time allows

Ordered by value per line of code.

**S-1. Re-plot everything against transmitted bits rather than B.** Estimate per-generation
throughput as B · (log C − mean teacher entropy) for the label channel, or B × effective
rank of the transmitted logit matrix for the soft channel. If the τ and B curves collapse
onto a single curve when re-plotted this way, that is the single most compelling figure the
project could produce, and it costs nothing beyond re-plotting existing logs. Do this first.

**S-2. Deep-linear arm — one-character change, real strategic payoff.** Delete the ReLUs.
The function class becomes linear but the optimizer's implicit bias remains that of a deep
network, which isolates nonlinearity from depth-induced implicit regularization and
separates both from the ridge null's explicit regularization. Strategically: deep linear
networks are exactly the model class of the project's closest prior work (Jarvis, Klein,
Rosman & Saxe, PNAS 2026), so including this arm makes that paper a *nested special case* of
this design rather than an adjacent competitor. That is the strongest available position
against the nearest neighbour.

**S-3. Anchor rate and accumulation.** Mix a fraction r of true generation-0 labels into
each generation's targets and sweep r ∈ {0, 0.01, 0.05, 0.2, 1}; r = 1 degenerates to
control C-2. Mobahi et al. prove collapse specifically in the anchor-free regime, so r > 0
is precisely the excluded case and mapping the boundary converts a binary result into a
dose–response curve. The cheaper and better-supported variant is *accumulation*: generation
g trains on the union of all previous generations' transmitted pairs rather than only
g−1's. Gerstgrasser et al. show, empirically and in the linear framework, that accumulating
avoids collapse where replacing does not. About four lines, with a strong citable
prediction.

**S-4. Schedule reversal — path dependence with no founder contrast at all.** Same founder,
two temperature or bottleneck schedules with identical arithmetic mean (annealed up versus
annealed down). If the endpoints differ, the chain is path-dependent, full stop — no
Jaccard, no baseline formula, no founder comparison required. A clean self-contained
demonstration of the project's central claim, absent from the current design, very high
value per line.

**S-5. Architecture variation along the chain.** Alternate widths, depths or activations
generation-to-generation. If the stationary state were the architecture's prior, alternating
architectures should destroy survivor overlap; if it is founder-driven, it should survive.
Directly separates the two candidate causes and is nearly free. Also run a monotonically
widening chain to check contraction is not a capacity artifact.

**S-6. Structured-seed arm** with the per-attribute mutual information from §4.3. Already
declared open work; the metric addition is what makes it produce something the noise arm
cannot.

**S-7. Mode connectivity as a geometric test of path dependence.** Align two networks by
activation-matching permutation (Hungarian assignment on cross-correlation of hidden
activations, about thirty lines for a 2×256 MLP), then interpolate and measure the barrier.
Same-founder descendants should lie in a common basin modulo permutation; different-founder
descendants should not. Independent evidence for the same claim with no reliance on argmax
survivor sets. Reference for the alignment algorithms: Ainsworth, Hayase & Srinivasa,
arXiv:2209.04836 — cite honestly, including its stated counterexample to the single-basin
claim. State up front that barriers between independently trained networks are the default
expectation; the informative quantity is barrier(same-founder) minus barrier(different-founder).

**Deliberately not doing:** feature distillation or attention transfer (no analytic handle,
no clean hypothesis); curriculum over bottleneck width (S-4 tests the only sharp hypothesis
more cleanly); sparse autoencoders on the 256-dimensional representation (4,096 points is
badly underpowered); active or entropy-maximizing bottleneck selection (interesting, but the
item most at risk of scope creep — defer past this semester); scaling to bigger models or
natural data before the mechanism arms are done.

### 4.5 External validity — weeks 13–14 if the mechanism arms land

The EMNIST-balanced arm, per the measured budget in `PROJECT.md`: about 8.7 s per generation
on MPS, roughly 1.3 hours for ten founders at 25 generations. Add it only after §4.2 has a
result. If the semester runs short, this is the thing to cut, and cutting it should be stated
as a scope decision rather than left as a gap.

## 5. Fourteen-week schedule

Weeks are counted from the week of 2026-09-07. Reading and doing are paired so the reading
arrives before the code that depends on it.

| week | read | do |
|---|---|---|
| 1 | Griffiths & Kalish (2007) | Corrections §3 items 1, 4, 5, 6. Rewrite the maximizer sentence. |
| 2 | Kirby, Dowman & Griffiths (2007); **Guo et al. arXiv:2605.23054** | Corrections §3 items 2, 3. Write the delta paragraph against Guo et al. |
| 3 | Rafferty, Griffiths & Klein (2014) | **C-1 shuffled transmission**, **C-2 no chaining**. Estimate the mixing-time bound; decide whether 25 generations is defensible. |
| 4 | Mobahi et al. (2020); Borup & Andersen (2021); Pareek, Du & Oh (2024) | **C-7 ridge null** computed numerically from the real pool. **C-8 frozen-body null.** |
| 5 | Papyan, Han & Donoho (2020); Hui, Belkin & Nakkiran (2022) | **C-3 one-step fidelity**, **C-4 matched imbalance**, **C-5 noise decomposition**, **C-6 held-out pool**. Fix the NC metric scope. |
| 6 | Jarvis et al. (2026); Saxe, McClelland & Ganguli (2014) if needed | §4.3 statistics rewrite: permutation tests, resampling null, overlap coefficient. **S-2 deep-linear arm.** |
| 7 | Welling & Teh (2011); Zhang et al. cyclical SG-MCMC (2020) | Implement the sampler arm. **Run the inertness check before anything else.** |
| 8 | Izmailov et al. (2021); Wenzel et al. (2020); Lakshminarayanan et al. (2017) | Sampler-validity diagnostics. **MC dropout negative control.** σ sweep. |
| 9 | Roy & Vetterli; RankMe (2023); re-read Kornblith, Davari; Ding et al. (2021) | λ dose–response sweep, n = 20 founders. Finalize the metric set and the CKA note. |
| 10 | Stanton et al. (2021); Hinton, Vinyals & Dean (2015) | Mutual information / variation of information / adjusted Rand index across all arms. Per-class survival regression. |
| 11 | Fang, He, Long & Su (2021) | **S-1 transmitted-bits re-plot.** **S-3 anchor and accumulation sweep.** |
| 12 | Gerstgrasser et al. (2024); Bertrand et al. (2024); Schaeffer et al. (2025) | **S-4 schedule reversal**, **S-5 architecture variation**, **S-6 structured-seed arm.** Write the scope and limitations section. |
| 13 | Tier-3 skim block | EMNIST arm if the schedule holds. Assemble the bibliography. |
| 14 | Frankle et al. (2020); Navarro et al. (2018); Thompson, Kirby & Smith (2016) | Write the path-dependence defence. Final report. |

The four weeks that change what gets *written*, not just what gets cited, are 2, 3, 5 and 11.

## 6. Reading list

Every entry below was checked by a search agent against arXiv, Crossref or dblp. Three were
additionally verified by hand in this session and are marked. Anything not verified is
marked UNVERIFIED and must be checked before it enters the bibliography.

### Tier 1 — read carefully

1. **Griffiths, T. L., & Kalish, M. L. (2007).** Language evolution by iterated learning with
   Bayesian agents. *Cognitive Science*, 31(3), 441–480. DOI 10.1080/15326900701326576.
   The entire research question rests on this. After reading you should be able to explain
   why the sampler chain is a Gibbs sampler on the joint distribution of hypotheses and data,
   and exactly what the paper says about MAP learners.
2. **Kirby, S., Dowman, M., & Griffiths, T. L. (2007).** Innateness and culture in the
   evolution of language. *PNAS*, 104(12), 5241–5245. DOI 10.1073/pnas.0608222104.
   **Verified by hand.** The most important missing citation. Works out the MAP case with the
   bottleneck as a free variable; establishes that cultural transmission magnifies weak
   biases into strong universals, more strongly as the bottleneck tightens.
3. **Rafferty, A. N., Griffiths, T. L., & Klein, D. (2014).** Analyzing the rate at which
   languages lose the influence of a common ancestor. *Cognitive Science*, 38, 1406–1431.
   DOI 10.1111/cogs.12112. Directly bears on the "25 generations only" caveat. Produce an
   estimate of how many generations Arm B *should* need; if it far exceeds 25, Arm B is a
   statement about the compute budget, not about neural chains.
4. **Mobahi, H., Farajtabar, M., & Bartlett, P. L. (2020).** Self-distillation amplifies
   regularization in Hilbert space. NeurIPS 2020. arXiv:2002.05715. Underwrites the claim
   that falling effective rank is the boring default, and tells you what the ridge null
   should do before you run it.
5. **Papyan, V., Han, X. Y., & Donoho, D. L. (2020).** Prevalence of neural collapse during
   the terminal phase of deep learning training. *PNAS*. DOI 10.1073/pnas.2015509117,
   arXiv:2008.08186. Define NC1–NC4 and its three preconditions; explain why the frozen
   generation-0 partition is the only defensible one to compute them against.
6. **Jarvis, D., Klein, R., Rosman, B., & Saxe, A. M. (2026).** Compositionality and
   systematicity emerge from iterated learning in deep linear networks. *PNAS*, 123(19),
   e2509739123. DOI 10.1073/pnas.2509739123. The nearest neighbour. Identify which of their
   results are consequences of linearity, and defend each of the four claimed deltas.
   Background if the SVD dynamics are unfamiliar: Saxe, McClelland & Ganguli (2014),
   arXiv:1312.6120.
7. **Welling, M., & Teh, Y. W. (2011).** Bayesian learning via stochastic gradient Langevin
   dynamics. ICML 2011, 681–688. Underwrites the sampler arm. Write down the update rule and
   explain the noise-to-step-size relationship. Read together with Izmailov et al. below.

### Tier 2 — method and metric background

8. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network.
   arXiv:1503.02531. The origin of the τ being swept.
9. Borup, K., & Andersen, L. N. (2021). Even your teacher needs guidance. NeurIPS 2021.
   arXiv:2102.13088. Quantifies how much a ground-truth anchor changes the anchor-free
   regime; underwrites S-3.
10. Pareek, D., Du, S. S., & Oh, S. (2024). Understanding the gains from repeated
    self-distillation. arXiv:2407.04600. **Venue UNVERIFIED.** Multi-step self-distillation
    in the linear-regression setting of the ridge null; may hand you the closed form.
11. Roy, O., & Vetterli, M. (2007). The effective rank: a measure of effective
    dimensionality. EUSIPCO 2007, 606–610. **No arXiv ID confirmed — state the formula
    rather than relying on the citation.** Four pages; read to know which definition you are
    reporting.
12. Garrido, Q., Balestriero, R., Najman, L., & LeCun, Y. (2023). RankMe. ICML 2023.
    arXiv:2210.02885. Justifies treating rank as meaningful rather than decorative.
13. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive
    uncertainty estimation using deep ensembles. NeurIPS 2017. arXiv:1612.01474.
14. Wilson, A. G., & Izmailov, P. (2020). Bayesian deep learning and a probabilistic
    perspective of generalization. NeurIPS 2020. arXiv:2002.08791. The argument that deep
    ensembles are approximate Bayesian marginalization — and it is contested.
15. Zhang, R., Li, C., Zhang, J., Chen, C., & Wilson, A. G. (2020). Cyclical stochastic
    gradient MCMC for Bayesian deep learning. ICLR 2020. arXiv:1902.03932. Implement from
    this rather than from Welling & Teh directly; plain SGLD does not mix on a multimodal
    posterior.
16. Pearce, T., Leibfried, F., Brintrup, A., Zaki, M., & Neely, A. (2020). Uncertainty in
    neural networks: approximately Bayesian ensembling. AISTATS 2020. arXiv:1810.05546.
    The primary method. Note the five-author list.
17. Ding, F., Denain, J.-S., & Steinhardt, J. (2021). Grounding representation similarity
    with statistical testing. NeurIPS 2021. arXiv:2108.01661. The sensitivity/specificity
    framework for the metric replacements, and a second prior on CKA's insensitivity.

Re-read in the same block: Kornblith, Norouzi, Lee & Hinton (2019), arXiv:1905.00414;
Davari, Horoi, Natik, Lajoie, Wolf & Belilovsky (2023), arXiv:2210.16156.

### Tier 3 — skim, know it exists

Model collapse: Shumailov, Shumaylov, Zhao, Papernot, Anderson & Gal (2024), *Nature*
631(8022), 755–759, DOI 10.1038/s41586-024-07566-y (a Nature Author Correction
s41586-025-08905-3 exists — UNVERIFIED, check before citing); Alemohammad et al. (2024),
Self-consuming generative models go MAD, ICLR 2024, arXiv:2307.01850; **Dohmatob, E., Feng,
Y., Yang, P., Charton, F., & Kempe, J. (2024)**, A tale of tails, ICML 2024, arXiv:2402.07043
— *this is the author list; it was previously miscited in this project*; Dohmatob, Feng &
Kempe (2024), Model collapse demystified: the case of regression, NeurIPS 2024,
arXiv:2402.07712.

Iterated learning, founding literature: Kirby (2001), DOI 10.1109/4235.918430; Smith, Kirby &
Brighton (2003), DOI 10.1162/106454603322694825; Kirby, Cornish & Smith (2008), DOI
10.1073/pnas.0707835105; Kirby, Tamariz, Cornish & Smith (2015), DOI
10.1016/j.cognition.2015.03.016 — read this one properly if you want to argue about the
compression/expressivity trade-off. Human-subject side: Kalish, Griffiths & Lewandowsky
(2007), DOI 10.3758/BF03194066; Griffiths, Kalish & Lewandowsky (2008), DOI
10.1098/rstb.2008.0146.

Iterated learning applied to neural networks: Ren, Guo, Labeau, Cohen & Kirby (2020), ICLR
2020, arXiv:2002.01365; Lu, Singhal, Strub, Pietquin & Courville (2020), arXiv:2003.12694;
Zheng, Zhang, Kembhavi & Krishna (2024), CVPR 2024, arXiv:2404.02145; Zhu & Griffiths
(2024), arXiv:2406.01860 — iterated learning in context on a frozen LLM, explicitly framed
as MCMC, different level but the same idea; Yamakoshi, Griffiths & Hawkins (2022), Findings
of ACL 2022, arXiv:2202.12226.

Metrics: Klabunde, Schumacher, Strohmaier & Lemmerich (2025), *ACM Computing Surveys* 57(9),
art. 242, arXiv:2305.06329 — use as a lookup table when justifying the metric set; Fort, Hu
& Lakshminarayanan (2019), arXiv:1912.02757.

### The adversarial list — what a hostile examiner cites against you

Ordered by how much damage each does. Read these; do not wait for someone else to bring them
up.

- **Kirby, Dowman & Griffiths (2007).** Attacks the central framing. See §1.1.
- **Rafferty, Griffiths & Klein (2014).** Attacks Arm B. If their mixing bound exceeds 25
  generations, the non-convergence result is about your compute budget.
- **Guo, Wu & Yiu (2026), arXiv:2605.23054, CoNLL 2026.** Attacks novelty of the framing.
  Verified by hand. See §1.2. Refereed, so it counts. Read it in week 2, not week 12.
- **Fang, He, Long & Su (2021), *PNAS* 118(43), DOI 10.1073/pnas.2103091118.** Attacks the
  best result. Minority collapse is the analytically characterized merging of
  underrepresented classes' classifiers, and B = 128 over 50 classes is squarely in that
  regime. Answered by control C-4.
- **Stanton, Izmailov, Kirichenko, Alemi & Wilson (2021), arXiv:2106.05945.** Attacks the
  primary metric. Even a single distillation step with matched capacity and no bottleneck
  leaves a large teacher–student discrepancy, attributed to optimization rather than
  information transfer. Answered by control C-3.
- **Izmailov, Vikram, Hoffman & Wilson (2021), arXiv:2104.14421.** Attacks the sampler arm
  pre-emptively: SG-MCMC and deep ensembles produce predictive distributions distinctly
  different from full-batch HMC. Design the arm so a null can be distinguished from a broken
  sampler, and say so in advance.
- **Wenzel et al. (2020), arXiv:2002.02405.** Attacks the same arm from the other side:
  posterior temperature is a free knob with large effects. Whatever temperature you pick, be
  ready to say why.
- **Gerstgrasser et al. (2024), arXiv:2404.01413**, with Bertrand, Bose, Duplessis,
  Jiralerspong & Gidel, arXiv:2310.00429 (ICLR 2024, venue UNVERIFIED). Attacks relevance:
  collapse is contingent on the protocol, and accumulating data is provably stable where
  replacing is not. Your chain never sees fresh data, the known-worst case. Answered by S-3.
- **Schaeffer, Kazdan, Arulandu & Koyejo (2025), arXiv:2503.03150** (venue UNVERIFIED).
  Counts eight conflicting definitions of model collapse and argues most headline scenarios
  rest on unrealistic assumptions. An ally for the terminology discipline in `CLAUDE.md` and
  a weapon against sloppy phrasing. Companion: Kazdan et al. (2024), arXiv:2410.16713.
- **Frankle, Dziugaite, Roy & Carbin (2020), arXiv:1912.05671.** Attacks the Arm A / Arm B
  contrast: path dependence of SGD is mundane and well documented. Your defence is that the
  founder's influence has vanished from the *function* by generation 1 while surviving in the
  *alphabet* at generation 25. Make that argument load-bearing — it is the best one you have,
  and §4.3's mutual-information measures are what turn it from an assertion into a
  measurement.
- **Ding, Denain & Steinhardt (2021), arXiv:2108.01661.** Attacks the CKA methodological
  note, which is a planned deliverable. Two independent priors, not one; downgrade it from
  "warning worth passing on" to "confirmed in a third setting."
- **Hui, Belkin & Nakkiran (2022), arXiv:2202.08384.** Attacks the NC instrumentation: NC
  occurs on the train set but not the test distribution and its link to generalization is
  unclear.
- **Navarro, Perfors, Kary, Brown & Donkin (2018), DOI 10.1111/cogs.12667**, with Thompson,
  Kirby & Smith (2016), DOI 10.1073/pnas.1523631113, and Rafferty, Griffiths & Ettlinger
  (2013), *Cognition* 129, 70–87. The inference from chain endpoints to priors is already
  known to be fragile. Collectively these say the falsified "read out the architecture's
  prior" hypothesis was *predicted* to fail. Good for you if you cite it, bad if an examiner
  cites it first.
- Borji (2024), arXiv:2410.12954. **Unrefereed — per `CLAUDE.md`, do not cite as authority.**
  Argues the Nature collapse result is a generic consequence of repeatedly refitting a
  distribution to its own samples. Know it exists; it is the sharpest version of "your
  contraction is regression to the mean," which is what the ridge null answers.

Lesser threat: Dohmatob, Feng, Subramonian & Kempe (2024), arXiv:2410.04840, strong model
collapse from ~1% synthetic contamination, non-monotone in model size. ICLR 2025, venue
UNVERIFIED.

## 7. Preregistration

Before the week-9 dose–response sweep runs, commit the following to a timestamped file:

- the primary statistic (founder-identification balanced accuracy, leave-one-chain-out) and
  its permutation test;
- the secondary statistic (cross-arm permutation test on mean pairwise Jaccard);
- the number of founders per arm and the number of generations, fixed in advance;
- the σ and λ grids, fixed in advance;
- the inertness threshold below which the sampler arm is declared broken rather than null;
- the decision rule that distinguishes "the sampler/maximizer distinction does not transfer"
  from "the sampler was not a sampler."

The last item is the one Izmailov et al. force, and writing it after seeing the result is
worthless.

## 8. Logging

No JSONL logs currently exist in the repository; every number so far came from stdout and
would cost a rerun to recover. From now on, every run writes JSONL via the existing `--log`
flag, and `seed_stability.py` and `sampler_arm.py` gain the same. Figures are regenerated
from logs, never from a one-off script. This is cheap insurance and it is currently absent.

## 9. Repository hygiene

The research files are all untracked and the git remote points at the unrelated Playlist
Chaos starter assignment. The research work has no version history at all. Before the
controls block starts, either point this directory at a new remote and commit the research
files, or move the research work to its own repository. Losing `seed_stability.py` would
cost the project its central result.
