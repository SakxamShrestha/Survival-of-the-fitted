"""
Seed-stability of code contraction, logged, at a sample size that supports
interval estimates.

This re-runs the three arms that `seed_stability.py` (A, B) and `sampler_arm.py`
(C) established at n = 5. Those two scripts only print; their numbers were
recorded in PROJECT.md from a terminal session and cannot be re-checked. This
runner writes `logs/seed_stability_n<runs>.json`, so every subsequent claim can
be traced to a file and drift can be detected.

    python3 run_seed_stability.py --runs 5 --arms A     # reproduction gate, ~3.5 min
    python3 run_seed_stability.py --runs 20             # full run, ~42 min

Arms:
  A   one shared founder, soft-logit transmission
  B   a different founder per run, soft-logit transmission
  C   the same founders as B, sampled hard-label transmission

Arms B and C are a paired comparison, so they must descend from identical
founders. In the two original scripts that held only because both files happened
to call `torch.manual_seed(200 + r)` and then build and train the same MLP. Here
each founder is trained once and handed to both chains, which makes the pairing
structural rather than coincidental.

RNG consumption order is deliberately identical to the original scripts so that
runs 0-4 reproduce the published n = 5 values exactly. Do not reorder the calls
inside `train_founder`, `run_chain_soft` or `run_chain_sampled`.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F

from spike_iterated_distill import MLP, make_pool, train
from seed_stability import summarize

GENS = 25
TAU = 1.0
B = 128
DEVICE = "cpu"

REPO = os.path.dirname(os.path.abspath(__file__))

ARM_DESC = {
    "A": "one shared founder, soft-logit transmission",
    "B": "different founders, soft-logit transmission",
    "C": "different founders, sampled hard-label transmission",
}


def param_hash(model):
    """Stable digest of a model's parameters, for asserting a founder is reused
    rather than silently retrained or mutated between arms."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def train_founder(X, y, n_classes, seed):
    """Generation 0: memorize the assigned labels.

    Replicates the founder construction in both original scripts: seed torch,
    then build the MLP (the first consumer of torch RNG), then train. Label
    generation is deliberately done by the caller because it draws from NumPy
    and consumes no torch RNG in either script.
    """
    torch.manual_seed(seed)
    founder = MLP(X.shape[1], n_classes).to(DEVICE)
    train(founder, X, y, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    return founder


def run_chain_soft(X, teacher, n_classes, chain_seed):
    """Arms A and B: soft-logit transmission, as `seed_stability.run_chain`.

    Adds the per-generation alphabet size that the original discarded. The extra
    forward pass consumes no RNG, so the chain is unchanged.
    """
    torch.manual_seed(chain_seed)
    sizes = []
    for _ in range(GENS):
        with torch.no_grad():
            t_logits = teacher(X)
        sel = torch.randperm(X.shape[0])[:B]
        student = MLP(X.shape[1], n_classes).to(DEVICE)
        train(student, X[sel], t_logits[sel], 1500, 256, 1e-3, 0.0, True, TAU, DEVICE)
        teacher = student
        with torch.no_grad():
            sizes.append(int(teacher(X).argmax(-1).unique().numel()))
    with torch.no_grad():
        survivors = sorted(teacher(X).argmax(-1).unique().tolist())
    return sizes, survivors


def run_chain_sampled(X, teacher, n_classes, chain_seed):
    """Arm C: hard labels sampled from the teacher, as `sampler_arm.run_sampled`.

    `torch.multinomial` draws from the torch RNG, so its position between the
    `randperm` and the student's construction is load-bearing for reproduction.
    """
    torch.manual_seed(chain_seed)
    sizes = []
    for _ in range(GENS):
        with torch.no_grad():
            t_logits = teacher(X)
        sel = torch.randperm(X.shape[0])[:B]
        hard = torch.multinomial(F.softmax(t_logits[sel] / TAU, -1), 1).squeeze(-1)
        student = MLP(X.shape[1], n_classes).to(DEVICE)
        train(student, X[sel], hard, 1500, 256, 1e-3, 0.0, False, 1.0, DEVICE)
        teacher = student
        with torch.no_grad():
            sizes.append(int(teacher(X).argmax(-1).unique().numel()))
    with torch.no_grad():
        survivors = sorted(teacher(X).argmax(-1).unique().tolist())
    return sizes, survivors


def founder_accuracy(founder, X, y):
    with torch.no_grad():
        return float((founder(X).argmax(-1) == y).float().mean())


def record(run, founder_seed, chain_seed, founder_acc, sizes, survivors):
    return {"run": run, "founder_seed": founder_seed, "chain_seed": chain_seed,
            "founder_acc": founder_acc, "sizes": sizes, "survivors": survivors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--arms", default="ABC", help="subset of ABC")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = [a for a in "ABC" if a in args.arms.upper()]
    if not arms:
        ap.error("--arms must name at least one of A, B, C")

    t0 = time.time()
    X, y0, C = make_pool(seed=0, kind="noise")
    X, y0 = X.to(DEVICE), y0.to(DEVICE)
    results = {a: [] for a in arms}

    # ---- Arm A: one shared founder, chain randomness is the only difference.
    if "A" in arms:
        founder = train_founder(X, y0, C, seed=0)
        acc = founder_accuracy(founder, X, y0)
        before = param_hash(founder)
        print(f"Arm A shared founder (seed 0): memorization accuracy {acc:.3f}")
        for r in range(args.runs):
            sizes, survivors = run_chain_soft(X, founder, C, 100 + r)
            results["A"].append(record(r, 0, 100 + r, acc, sizes, survivors))
            print(f"  A run {r:2d}  chain seed {100+r}  "
                  f"{len(survivors):2d} of {C} survive   [{time.time()-t0:5.0f}s]")
        # A chain trains fresh students, so the shared founder must come out of
        # every run untouched. If it did not, later runs would not be comparable.
        assert param_hash(founder) == before, "Arm A founder mutated during a chain"

    # ---- Arms B and C: a different founder per run, shared between the arms.
    if "B" in arms or "C" in arms:
        for r in range(args.runs):
            rng = np.random.default_rng(200 + r)
            y_r = torch.from_numpy(rng.integers(0, C, size=X.shape[0])).long().to(DEVICE)
            founder_r = train_founder(X, y_r, C, seed=200 + r)
            acc_r = founder_accuracy(founder_r, X, y_r)
            before_r = param_hash(founder_r)

            if "B" in arms:
                sizes, survivors = run_chain_soft(X, founder_r, C, 300 + r)
                results["B"].append(record(r, 200 + r, 300 + r, acc_r, sizes, survivors))
                print(f"  B run {r:2d}  founder {200+r}  "
                      f"{len(survivors):2d} of {C} survive   [{time.time()-t0:5.0f}s]")

            # The paired design depends on Arm C starting from exactly the
            # founder Arm B started from, not a retrained equivalent.
            assert param_hash(founder_r) == before_r, \
                f"founder {200+r} changed between arms B and C"

            if "C" in arms:
                sizes, survivors = run_chain_sampled(X, founder_r, C, 300 + r)
                results["C"].append(record(r, 200 + r, 300 + r, acc_r, sizes, survivors))
                print(f"  C run {r:2d}  founder {200+r}  "
                      f"{len(survivors):2d} of {C} survive   [{time.time()-t0:5.0f}s]")

    # ---- Summary in the format the published n = 5 numbers were reported in,
    # so the reproduction gate is a direct comparison.
    for a in arms:
        summarize(f"Arm {a}: {ARM_DESC[a]} ({args.runs} runs, {GENS} gens, B={B})",
                  [set(run["survivors"]) for run in results[a]], C)

    out = args.out or os.path.join(REPO, "logs", f"seed_stability_n{args.runs}.json")
    payload = {
        "gens": GENS, "tau": TAU, "bottleneck": B, "n_runs": args.runs,
        "n_classes": C, "device": DEVICE, "steps_founder": 3000, "steps_gen": 1500,
        "torch_version": torch.__version__, "created": date.today().isoformat(),
        "elapsed_sec": round(time.time() - t0, 1),
        "arms": {a: {"description": ARM_DESC[a], "runs": results[a]} for a in arms},
    }
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
