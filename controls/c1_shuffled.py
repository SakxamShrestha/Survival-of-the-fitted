"""
C-1: shuffled transmission.

Permutes the teacher's logits across the B selected inputs before each student
trains. The same set of logit vectors is handed over; only the pairing with
inputs is destroyed. If the output alphabet still contracts, contraction tells
us nothing about the learned mapping.

Both arms share a founder within a seed, so shuffling is the only difference.

    python3 controls/c1_shuffled.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

# Run as `python3 controls/c1_shuffled.py` from the repository root: Python puts
# this file's directory on sys.path, not the root, so add the root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

GENS = 25
TAU = 1.0
B = 128
SEEDS = (0, 1, 2)
DEVICE = "cpu"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_chain(X, teacher, n_classes, chain_seed, shuffle):
    """Run one chain; return per-generation alphabet size and final survivors."""
    torch.manual_seed(chain_seed)
    sizes = []
    for _ in range(GENS):
        with torch.no_grad():
            t_logits = teacher(X)
        sel = torch.randperm(X.shape[0])[:B]
        Xs, Ts = X[sel], t_logits[sel]
        if shuffle:
            # Permute targets against inputs. The multiset of transmitted logit
            # vectors is unchanged; only the pairing is destroyed.
            Ts = Ts[torch.randperm(Ts.shape[0])]
        student = MLP(X.shape[1], n_classes).to(DEVICE)
        train(student, Xs, Ts, 1500, 256, 1e-3, 0.0, True, TAU, DEVICE)
        teacher = student
        with torch.no_grad():
            sizes.append(int(teacher(X).argmax(-1).unique().numel()))
    with torch.no_grad():
        survivors = sorted(teacher(X).argmax(-1).unique().tolist())
    return sizes, survivors


def main():
    t0 = time.time()
    results = {"intact": [], "shuffled": []}
    for seed in SEEDS:
        torch.manual_seed(seed)
        X, y0, C = make_pool(seed=seed, kind="noise")
        founder = MLP(X.shape[1], C).to(DEVICE)
        train(founder, X, y0, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
        with torch.no_grad():
            acc = float((founder(X).argmax(-1) == y0).float().mean())
        print(f"\nseed {seed}: founder memorization accuracy {acc:.3f}")

        for arm, shuffle in (("intact", False), ("shuffled", True)):
            sizes, survivors = run_chain(X, founder, C, 100 + seed, shuffle)
            results[arm].append({"seed": seed, "sizes": sizes,
                                 "final": sizes[-1], "survivors": survivors})
            print(f"  {arm:9s} -> {sizes[-1]:2d} of {C} classes survive "
                  f"(trajectory {sizes[0]} -> {sizes[len(sizes)//2]} -> {sizes[-1]})")

    intact = [r["final"] for r in results["intact"]]
    shuffled = [r["final"] for r in results["shuffled"]]
    print(f"\n=== C-1 VERDICT  ({time.time()-t0:.0f}s, {len(SEEDS)} seeds, "
          f"B={B}, tau={TAU}, {GENS} gens) ===")
    print(f"intact transmission   : {np.mean(intact):.1f} classes  {intact}")
    print(f"shuffled transmission : {np.mean(shuffled):.1f} classes  {shuffled}")
    if np.mean(shuffled) <= np.mean(intact) * 1.25:
        print("Contraction SURVIVES shuffling. The effect does not depend on the")
        print("input-to-target mapping, so 'code contraction' must be stated as an")
        print("alphabet-frequency effect rather than a mapping-preserving one.")
    else:
        print("Contraction WEAKENS under shuffling. The effect depends on the")
        print("input-to-target correspondence being preserved.")

    out = os.path.join(REPO, "logs", "c1_shuffled.json")
    with open(out, "w") as handle:
        json.dump({"gens": GENS, "tau": TAU, "bottleneck": B,
                   "seeds": list(SEEDS), "results": results}, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
