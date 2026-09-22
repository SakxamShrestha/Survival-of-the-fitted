"""Arm C: different founders, SAMPLED hard-label transmission.

Not a sampler-like learner, despite what an earlier version of this docstring said.
Sampling the transmitted *labels* randomizes the channel; the learner is still SGD
descending to a single weight vector, so it remains a maximizer — now fitting noisier
targets. Griffiths & Kalish require sampling in *hypothesis* space, over the learner's
beliefs, which is a different intervention and is not implemented here.

Reported in PROJECT.md as evidence that output-space sampling does not restore
cross-founder convergence: overlap stays at chance while contraction gets more severe.
"""
import numpy as np, torch, torch.nn.functional as F
from spike_iterated_distill import MLP, make_pool, train
from seed_stability import jaccard, expected_jaccard, summarize
import itertools

GENS, TAU, B, N_RUNS, DEV = 25, 1.0, 128, 5, "cpu"

def run_sampled(X, teacher, n_classes, chain_seed):
    torch.manual_seed(chain_seed)
    for _ in range(GENS):
        with torch.no_grad():
            t = teacher(X)
        sel = torch.randperm(X.shape[0])[:B]
        hard = torch.multinomial(F.softmax(t[sel] / TAU, -1), 1).squeeze(-1)
        s = MLP(X.shape[1], n_classes).to(DEV)
        train(s, X[sel], hard, 1500, 256, 1e-3, 0.0, False, 1.0, DEV)
        teacher = s
    with torch.no_grad():
        return set(teacher(X).argmax(-1).unique().tolist())

X, _, C = make_pool(seed=0, kind="noise")
surv = []
for r in range(N_RUNS):
    torch.manual_seed(200 + r)
    rng = np.random.default_rng(200 + r)
    y_r = torch.from_numpy(rng.integers(0, C, size=X.shape[0])).long()
    f = MLP(X.shape[1], C).to(DEV)
    train(f, X, y_r, 3000, 256, 1e-3, 0.0, False, 1.0, DEV)
    surv.append(run_sampled(X, f, C, 300 + r))
summarize(f"Arm C: {N_RUNS} different founders, SAMPLED transmission, {GENS} gens, B={B}", surv, C)
