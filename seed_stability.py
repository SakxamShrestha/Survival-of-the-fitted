"""
Seed-stability of code contraction.

Tests whether the surviving output alphabet under a tight transmission bottleneck
is a stable signature (a readout of the network's prior) or arbitrary die-off
(rank collapse under another name).

Arm A: fixed data pool, fixed generation-0 model, varying chain randomness.
       Do chains descending from the SAME founder land on the same classes?
Arm B: fixed data pool, DIFFERENT generation-0 random labels per run.
       Do different founders converge to the same classes anyway?

Baseline: expected Jaccard overlap if survivors of size k were drawn at random
from C classes.
"""

import itertools
import numpy as np
import torch
import torch.nn.functional as F

from spike_iterated_distill import MLP, make_pool, train

GENS = 25
TAU = 1.0
B = 128
N_RUNS = 5
DEVICE = "cpu"


def run_chain(X, teacher, gens, tau, bottleneck, chain_seed, n_classes):
    """Run a distillation chain from a given founder; return final surviving classes."""
    torch.manual_seed(chain_seed)
    for _ in range(gens):
        with torch.no_grad():
            t_logits = teacher(X)
        sel = torch.randperm(X.shape[0])[:bottleneck]
        student = MLP(X.shape[1], n_classes).to(DEVICE)
        train(student, X[sel], t_logits[sel], 1500, 256, 1e-3, 0.0, True, tau, DEVICE)
        teacher = student
    with torch.no_grad():
        pred = teacher(X).argmax(-1)
    return set(pred.unique().tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def expected_jaccard(k, C):
    """E[Jaccard] for two independent uniform random subsets of size k from C."""
    inter = k * k / C
    return inter / (2 * k - inter)


def summarize(name, survivors, C):
    sizes = [len(s) for s in survivors]
    pairs = list(itertools.combinations(range(len(survivors)), 2))
    obs = [jaccard(survivors[i], survivors[j]) for i, j in pairs]
    k = float(np.mean(sizes))
    exp = expected_jaccard(k, C)
    print(f"\n--- {name} ---")
    for i, s in enumerate(survivors):
        print(f"  run {i}: {len(s):2d} classes  {sorted(s)}")
    print(f"  mean surviving classes : {k:.1f} of {C}")
    print(f"  observed mean Jaccard  : {np.mean(obs):.3f}  (range {min(obs):.3f}-{max(obs):.3f})")
    print(f"  random-subset baseline : {exp:.3f}")
    print(f"  ratio observed/random  : {np.mean(obs)/exp:.2f}x")
    always = set.intersection(*survivors)
    never = set(range(C)) - set.union(*survivors)
    print(f"  survive in ALL runs    : {len(always):2d} {sorted(always)}")
    print(f"  die in ALL runs        : {len(never):2d} {sorted(never)}")
    return np.mean(obs), exp


if __name__ == "__main__":
    # ---- Arm A: same founder, different chain randomness
    torch.manual_seed(0)
    X, y0, C = make_pool(seed=0, kind="noise")
    founder = MLP(X.shape[1], C).to(DEVICE)
    train(founder, X, y0, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        acc = (founder(X).argmax(-1) == y0).float().mean()
    print(f"shared founder memorization accuracy: {acc:.3f}")

    surv_a = [run_chain(X, founder, GENS, TAU, B, 100 + r, C) for r in range(N_RUNS)]
    obs_a, exp_a = summarize(f"Arm A: same founder, {N_RUNS} chains, {GENS} gens, B={B}", surv_a, C)

    # ---- Arm B: different founders (different gen-0 random labels), same pool
    surv_b = []
    for r in range(N_RUNS):
        torch.manual_seed(200 + r)
        rng = np.random.default_rng(200 + r)
        y_r = torch.from_numpy(rng.integers(0, C, size=X.shape[0])).long()
        f_r = MLP(X.shape[1], C).to(DEVICE)
        train(f_r, X, y_r, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
        surv_b.append(run_chain(X, f_r, GENS, TAU, B, 300 + r, C))
    obs_b, exp_b = summarize(f"Arm B: {N_RUNS} different founders, {GENS} gens, B={B}", surv_b, C)

    print("\n=== VERDICT ===")
    for name, obs, exp in [("Arm A", obs_a, exp_a), ("Arm B", obs_b, exp_b)]:
        if obs > 2 * exp:
            v = "SYSTEMATIC - survivors are a stable signature"
        elif obs > 1.3 * exp:
            v = "WEAKLY SYSTEMATIC - above chance but not decisive"
        else:
            v = "ARBITRARY - consistent with random die-off"
        print(f"{name}: {obs:.3f} vs random {exp:.3f} -> {v}")
