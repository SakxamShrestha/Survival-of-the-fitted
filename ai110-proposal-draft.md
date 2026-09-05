# Surviving the Bottleneck: Samplers, Maximizers, and Where Iterated Distillation Comes to Rest

**Sakxam Shrestha — Project Proposal (Assignment 02)**

## 1. Introduction to the Research Topic

As AI-generated content saturates the internet, a growing share of the data used to train
new models is itself model output. The literature calls the resulting degradation *model
collapse*: Shumailov et al. (Nature, 2024) showed that a generative model recursively
trained on its own samples first loses the tails of the data distribution and eventually
loses variance altogether. A separate research tradition argues almost the opposite.
In *iterated learning* models from cognitive science (Kirby, Cornish & Smith, PNAS 2008),
a narrow transmission bottleneck between generations is precisely the pressure that forces
structure to emerge, because only compressible, systematic patterns survive repeated
retransmission through a finite channel. These two literatures make opposite predictions
about the same process, and they are rarely tested against each other in the same
experimental frame. My project asks which outcome a chain of neural networks actually
lands in, and — more importantly — what controls the answer.

The sharpest theoretical statement about where such a chain ends up comes from Griffiths
and Kalish (2007), who modeled iterated learning as a Markov process over hypotheses and
proved that the chain's stationary distribution is exactly the *prior* of the learners —
the founding data is asymptotically erased. That result carries a condition that is easy to
overlook and central to this project: it holds for learners who **sample from the
posterior**. Learners who instead **maximize**, committing to a single best hypothesis, are
known not to converge to the prior; their chains stay path-dependent on where they started.
A neural network trained by stochastic gradient descent to a single weight vector is a
maximizer, not a sampler. Whether the sampler/maximizer distinction — established for
idealized Bayesian agents and human participants — actually governs chains of real neural
networks is, as far as I can determine, untested.

My preliminary pilot supplies the first evidence that it does. I built a 50-class MLP chain
in which generation 0 memorizes random labels over a fixed pool of 4,096 synthetic inputs,
and each later generation is trained from scratch on only its predecessor's soft logits
over a subsample of B pool points. Over 20 generations, agreement with the founding model
decays monotonically in every condition, ordered by distillation temperature: 0.282 at
temperature 0.5, 0.188 at 1.0, and 0.145 at 2.0, against a chance level of 0.020. Nothing
self-organizes. The transmission bottleneck is the stronger lever: at B = 128, functional
agreement reaches chance within a single generation while the model's output alphabet
contracts sharply, from all 50 classes to a handful. I then asked whether that surviving
alphabet is a stable signature or arbitrary die-off, using the overlap (Jaccard index) of
surviving class sets across five independent runs. **Five chains descending from the same
founder overlapped at 2.94 times the random-subset baseline. Five chains descending from
different founders overlapped at 1.13 times baseline — indistinguishable from chance.** The
endpoint is founder-specific, not architecture-specific: within this horizon the chains do
not converge to anything shared, which is precisely the signature of a maximizing learner.
This result also falsified a hypothesis I had formed one step earlier — that the chain
could be used as an instrument to read out the architecture's inductive prior — and the
falsification is what redirected the project onto its current question.

My research question is therefore: **does the sampler/maximizer distinction from Bayesian
iterated-learning theory govern the stationary state of neural distillation chains, and at
what point — if any — does making the learner a posterior sampler restore convergence to a
shared endpoint?** I have already run and eliminated the obvious first answer. Replacing
deterministic soft-logit transmission with hard labels *sampled* from the teacher's output
distribution did not raise cross-founder overlap at all (1.02x baseline, versus 1.13x for
deterministic transmission); it merely made the contraction more severe, cutting mean
survivors from 4.4 classes to 2.4. The likely reason is a distinction the theory is
specific about and I was not: Griffiths and Kalish require sampling in **hypothesis
space**, over the learner's beliefs. Sampling output labels leaves the learner performing
SGD to a single point estimate on noisier targets — it is still a maximizer. The live test
is therefore to sample in **weight space**, replacing plain SGD with stochastic-gradient
Langevin dynamics or a deep ensemble so that each generation approximates a draw from a
posterior rather than its mode, and asking whether cross-founder overlap then rises above
chance. Temperature and bottleneck width enter as the parameters controlling how much noise
and how much compression the channel applies. The noise-seeded founder, which motivated my
original proposal, is now load-bearing rather than decorative: memorized random labels
guarantee that no structure in the founding data contaminates the measurement of where the
chain settles.

I expect to learn whether a well-established result in cognitive science transfers to deep
networks, and if so, at what channel configuration the transfer occurs. I want to be
explicit that the headline phenomena here are not novel discoveries. Repeated distillation
without a ground-truth anchor is proven to collapse (Mobahi, Farajtabar & Bartlett, NeurIPS
2020; Borup & Andersen, NeurIPS 2021); the transmission bottleneck as a controlled variable
is the founding methodology of iterated learning (Kirby, 2001); and Jarvis, Klein, Rosman
and Saxe (PNAS, 2026) have already tracked representational geometry across iterated-learning
generations. My contribution is narrower and I will state it as such: those results are
established for Bayesian agents, for human learners, and for *deep linear* networks whose
priors are analytically derivable. This project tests them where the prior is **not**
analytically known — a nonlinear network trained by SGD under knowledge distillation — and
sweeps the channel parameters that theory says should control the outcome. A clean negative
result is a legitimate and fully reportable outcome; the pilot above already contains one,
and it is what redirected the project.

## 2. Detailed Plan

**Data and task.** The core experiment uses a synthetic compositional attribute-value
dataset rather than a natural image corpus, because it makes every metric well-defined.
Each input is 5 attributes with 10 possible values each, one-hot encoded and concatenated
into a 50-dimensional vector, drawn as a fixed pool of 4,096 points that is reused across
all generations so that cross-generation representational comparisons are computed on a
common reference set. There are 50 output classes. In the **structured-seed** arm, the
generation-0 labels are a deterministic function of the input attributes. In the
**noise-seed** arm, generation 0 is trained to memorize independent uniform random labels,
which reproduces the random-label regime of Zhang et al. (ICLR 2017). A secondary
replication on MNIST will test whether the decay-rate ordering is an artifact of the synthetic
domain.

**Technologies.** Python 3.12, PyTorch for models and training, NumPy/SciPy for the
representational metrics, scikit-learn for clustering agreement, and Matplotlib for
figures. Experiments are configuration-driven (YAML) with per-generation metric logging to
JSONL, so every figure in the report is regenerable from a logged run.

**System architecture.** The chain model is a 50–256–256–50 ReLU MLP (approximately 92,000
parameters). Generation 0 is trained to zero training error on its assigned labels.
Generation n+1 is initialized from scratch, is never shown the original labels, and is
trained only on generation n's soft logits over a subsampled set of B pool points, using a
KL-divergence distillation loss at temperature tau. B is the transmission bottleneck, and
it is the direct operationalization of the bottleneck concept in the project title.

**Independent variables.** The primary variable is the **learner rule**: plain SGD (a
maximizer, already measured), stochastic-gradient Langevin dynamics at several noise
scales, and a deep ensemble whose members are averaged to approximate a posterior draw.
Secondary variables: distillation temperature tau in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
bottleneck width B in {64, 128, 256, 512, 1024, 4096}; founding-seed condition in
{random-label noise, structured labels}; and transmission mode in {soft logits, sampled
hard labels, argmax hard labels}. Chains run 100 generations. The cross-founder design
requires at least ten independent founders per cell rather than five, since the pilot's
five founders yield only ten pairwise comparisons and correspondingly noisy overlap
estimates.

**Dependent variables.** Primary: the **Jaccard overlap of surviving output-class sets
across independently founded chains**, reported against the analytic random-subset baseline
for sets of the observed size. This is the quantity that distinguishes convergence to a
shared endpoint from path-dependent die-off, and it is the measure on which the central
hypothesis turns. Supporting: functional agreement of generation n with generation 0 and
with generation n−1 on a fixed probe set; effective rank of the penultimate-layer
covariance spectrum, computed as exp(−sum p_i log p_i) over the normalized eigenvalues;
surviving alphabet size; output entropy; and mean logit norm. Secondary: linear and RBF
Centered Kernel Alignment, restricted to the one variant that bears on the hypothesis —
(gen n of chain A, gen n of an independently founded chain B). Neural Collapse NC1–NC3 will
be computed against the frozen generation-0 label partition, which is well-defined
precisely because generation 0 has explicit labels even in the noise arm.

**Null model and ablations.** This is the methodological core of the project. Iterated
distillation on softened targets is a contraction map, so low-rank, tidy-looking geometry
is the *boring default outcome*, not evidence of self-organization. I will therefore run
an analytically tractable **ridge-regression chain** under the identical protocol, whose
fixed point can be computed in closed form. If the neural chain's rank and agreement
trajectories match the linear chain's, the phenomenon is regression to the mean and I will
report it as such. Further ablations isolate weight decay (which alone induces low-rank
geometry), fresh initialization versus warm start, and a fixed versus resampled input pool.
The primary metric and its decision thresholds will be committed to a timestamped git
commit before the full grid is run, so the analysis is pre-registered.

**Evaluation of success.** Success is a completed, reproducible measurement with an
explicit null comparison — not confirmation of a particular hypothesis. The primary
decision rule is pre-registered: for each learner rule, cross-founder Jaccard overlap is
compared against the analytic random-subset baseline, and the sampler/maximizer prediction
is supported only if a posterior-sampling learner exceeds baseline by a factor stated in
advance while plain SGD does not. Concretely: (i) cross-founder overlap is reported per
learner rule with confidence intervals over at least ten founders; (ii) the same-founder
versus cross-founder contrast is reported, since the pilot's 2.94x versus 1.13x split is
the effect the project is built on and it needs replication at larger n; (iii) the
code-contraction curve — surviving alphabet size as a function of B — is characterized;
(iv) the noise-seeded and structure-seeded conditions are compared with a stated effect
size; and (v) the neural-versus-linear-null comparison is reported whichever way it comes
out. A finding that no learner rule produces convergence is a complete result and would be
reported as such: it would establish that neural distillation chains are path-dependent in
a way idealized Bayesian chains are not.

**Compute feasibility.** A single generation trains in roughly 2 seconds on a laptop CPU.
The full grid is on the order of 20 CPU-hours and is embarrassingly parallel across cells;
no GPU is required. Feasibility is deliberately not a risk in this design, which is what
makes five seeds and a full ablation set affordable within one semester.

**Deliverables.** (1) An open-source, configuration-driven experimental harness with
per-generation metric logging and the cross-founder overlap analysis; (2) the figure set —
cross-founder overlap by learner rule against the random baseline, the same-founder versus
cross-founder contrast, code-contraction curves as a function of B, and a heatmap of
agreement half-life over the tau × B grid; (3) a written report; and (4) a replication note
on CKA. On that last point I want to be precise about credit: in the pilot, CKA between
consecutive generations rose to 0.950 while functional agreement with the founding model
fell to 0.188, and CKA against generation 0 sat flat near 0.65 while the function it
measures was destroyed. This is not a new finding. Davari et al. (ICLR 2023) already
documented CKA reporting high similarity between a generalizing network, a network
memorizing random labels, and an untrained network. My contribution is only to confirm the
failure mode persists across a long generational chain, and to record that CKA — my own
originally proposed primary metric — should never be reported without an effective-rank or
functional-agreement measure beside it.

## 3. Schedule

- **Weeks 1–3:** Harness and metric implementations; replicate the pilot's same-founder
  versus cross-founder contrast at n = 10 founders. Pre-registration frozen end of week 3.
- **Weeks 4–7:** Learner-rule arm — SGLD at several noise scales and a deep ensemble,
  cross-founder overlap measured against baseline for each.
- **Weeks 8–10:** Ridge-regression null chain; tau x B grid; ablations (weight decay, warm
  start, resampled pool); structured-seed arm.
- **Weeks 11–13:** Analysis, figures, report.

## 4. Record of revisions

This proposal has been revised twice against evidence, and both revisions are recorded
because the reasoning is part of the work.

1. The original question — "does structure self-organize when a chain is seeded on pure
   random noise?" — was dropped after an adversarial prior-art search and a 40-second pilot
   both answered it negatively. Theory (Mobahi et al., 2020; Griffiths & Kalish, 2007)
   already predicted the answer.
2. The replacement question — "can the chain be used as an instrument to read out the
   architecture's inductive prior?" — was falsified by the cross-founder experiment
   reported above. Chains from different founders do not converge.

What survives is a narrower and better-posed question about *why* they fail to converge,
with a specific mechanism named by existing theory and a specific intervention to test it.
