"""
C-8: frozen-body / random-features null, and the Laplace correctness bridge.

The preregistered rule is gates/thresholds.json's "C-8" entry, committed
2026-10-20 at c873fcb and amended twice before this file existed. It carries two
separable things.

GATE A, the null question. Freeze model.body at random initialization, train
only the head each generation, and ask whether the alphabet still contracts.
Compared against C-1's intact arm, 7.50 with a 95 percent interval of
[5.65, 9.25] over 20 chains, which is measured on the training pool - so gate A
is measured there too.

TOLERANCE, the correctness question, and NOT a hypothesis test. With the body
frozen the head is a GLM whose Laplace posterior is closed form, so the analytic
prediction and the experiment must agree. A reproduction check that fails is a
bug, not a finding.

Two transmission modes, because the two jobs need different ones, and this is
the amendment this file forced:

  - Gate A transmits SOFT logits and trains with KL, matching C-1's intact arm,
    because that is the arm it is compared against.
  - The tolerance transmits HARD labels, because a Laplace posterior needs a
    categorical likelihood and the KL objective's optimum is not the multinomial
    MAP. Comparing a KL-trained head against a multinomial posterior would
    measure the soft-versus-hard objective mismatch, which is a fourth cause of
    disagreement on top of the three the rule names. RESEARCH-PLAN section 4.2
    makes the same point for S-1: KL-to-soft-logits is not a log-likelihood,
    which is why the sampler arm is built on sampled hard labels.

KNOWN LIMITATION OF THE TOLERANCE, found by smoke test on 2026-10-20 and NOT
yet resolved. At bottleneck 128 in a 256-dimensional frozen feature space the
transmitted batch can determine at most 128 of 256 feature directions;
analysis/laplace_glm reports prior_only_directions = 128, which is that, by
construction. In those directions the Laplace mean is exactly the prior mean,
zero, while a head trained by 1500 SGD steps from random initialization leaves
them near their initialization. The two estimators therefore differ in half the
space for a reason that has nothing to do with the sampler bridge.

Matching the objectives does not fix it. With weight decay set to alpha/N the
measured disagreement moved from 0.016357 to 0.016113 against a seed-spread
baseline of about 0.001 - a 2 percent change, so the objective mismatch was not
the driver. A finite number of SGD steps does not drive data-free directions to
the prior mean.

Until this is resolved the tolerance must NOT be read as a go/no-go for S-1.
The candidate fixes, none of them chosen here: compare inside the data-spanned
subspace only, which is the comparison that actually tests the bridge; run the
tolerance at a bottleneck at least as large as the feature dimension; or train
the head to convergence, which tests whether SGD finds the MAP and is a
different question. Each needs a rule change before it runs.

Gate A is EXPECTED TO REFUSE on the founder-comparability precondition. The
frozen-body founder reaches about 0.1135 on the training pool against the full
MLP's 0.977294921875; a 256-dimensional random feature space cannot fit 4096
random labels over 50 classes. The precondition requires the frozen founder to
land inside the range C-1's own twenty founders span. It does not. The
tolerance is unaffected by that, which is the reason to run this control
anyway: it is S-1's only correctness check.

    python3 controls/c8_frozen_body.py | tee logs/c8_frozen_body.txt
    python3 controls/c8_frozen_body.py --sanity-only    # writes no log
    python3 controls/c8_frozen_body.py --chains 3 --gens 5   # smoke, writes no log
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train, effective_rank
from analysis import laplace_glm as lg

# Imported rather than copied so the interval arithmetic and the bootstrap
# cannot drift from the arm this control is compared against. c1_shuffled_n20
# guards its runner behind __main__, so importing it runs no experiment.
from controls.c1_shuffled_n20 import (
    BOOT_SEED, N_BOOT, bootstrap_ci_mean, intervals_overlap,
)

GENS = 25
TAU = 1.0
BOTTLENECK = 128
STEPS_GEN = 1500
STEPS_FOUNDER = 3_000
DEVICE = "cpu"

# Same pools and founders as C-1's intact arm, and the same chain seeds, so the
# frozen-body arm sees the identical transmitted subsamples and only the learner
# differs. C-1 reuses one chain seed across its intact and shuffled arms for the
# same reason.
POOL_SEEDS = tuple(range(20))
CHAIN_SEED_OFFSET = 600

# Fresh block for the tolerance's seed-spread baseline. Checked clean against
# every seed constant in the repository: 0-19 and 600-619 (this file and C-1),
# 100-119 / 200-219 / 300-319 (seed stability arms), 400-404 (C-4 family),
# 1000-1019 (C-3 students), 9000 (probe pool), and the small permutation seeds.
SPREAD_SEED_OFFSET = 2_000

PRIOR_PRECISION = 1.0        # alpha > 0 is required; the posterior is improper at 0
PROBE_SEED = 9_000
N_CLASSES_EXPECTED = 50
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C1_LOG = os.path.join(REPO, "logs", "c1_shuffled_n20.json")

NOTE = (
    "One shared frozen body per chain: the body is initialized once, never trained, and the "
    "head is reinitialized and retrained from scratch each generation. That is the reading "
    "RESEARCH-PLAN section 4.1 forces by calling the feature space FIXED; under a fresh body "
    "per generation the feature space drifts and the Laplace comparison is ill-posed. Pools, "
    "founders and chain seeds are C-1's, so the frozen arm sees the same transmitted "
    "subsamples as the intact arm it is compared against. Gate A transmits soft logits to "
    "match that arm; the tolerance transmits hard labels because a Laplace posterior needs a "
    "categorical likelihood."
)


# ------------------------------------------------------------ reference arm

def c1_intact_interval(path):
    """C-1's intact-arm mean and 95 percent interval, recomputed from its log.

    Recomputed rather than read from a summary field, so that this control and
    C-1 are using the same bootstrap over the same per-chain values.
    """
    with open(path) as handle:
        ref = json.load(handle)
    finals = [int(run["final"]) for run in ref["results"]["intact"]]
    if len(finals) != len(POOL_SEEDS):
        raise ValueError(
            f"{os.path.relpath(path, REPO)} intact arm has {len(finals)} chains, "
            f"expected {len(POOL_SEEDS)}"
        )
    rng = np.random.default_rng(BOOT_SEED)
    lo, hi = bootstrap_ci_mean(finals, rng)
    return float(np.mean(finals)), (lo, hi), finals, ref


# ------------------------------------------------------------ frozen chain

def new_frozen_model(d_in, n_classes, init_seed):
    """An MLP whose body is frozen at initialization and never trains."""
    torch.manual_seed(init_seed)
    model = MLP(d_in, n_classes).to(DEVICE)
    model.body.requires_grad_(False)
    return model


def reinit_head(model, init_seed):
    """Fresh head on the SAME frozen body: a new student in a fixed feature space."""
    torch.manual_seed(init_seed)
    h = model.head.in_features
    out = model.head.out_features
    model.head = nn.Linear(h, out).to(DEVICE)
    return model


def train_head(model, X, targets, soft, steps=STEPS_GEN, wd=0.0):
    """Head-only training.

    Both halves are required. requires_grad_(False) alone still hands the body's
    tensors to AdamW, which with a non-zero weight decay would keep shrinking
    them; passing only the head's parameters alone would leave the body
    accumulating gradients.

    `wd` defaults to 0.0, which is what every published arm uses and what gate
    A's chain must use to stay comparable to C-1. The tolerance path overrides
    it: see matched_weight_decay.
    """
    return train(model, X, targets, steps, 256, 1e-3, wd, soft, TAU, DEVICE,
                 params=model.head.parameters())


def matched_weight_decay(n_observations, prior_precision=PRIOR_PRECISION):
    """The decay that puts SGD on the same objective as the Laplace posterior.

    analysis/laplace_glm.negative_log_posterior is a SUM of per-observation
    negative log likelihood plus 0.5 * alpha * ||w||^2, while train() minimizes
    MEAN cross-entropy. Dividing through by N, the L2 coefficient that matches
    is alpha / N.

    A residual mismatch remains and is reported rather than hidden: AdamW's
    weight decay is decoupled, applied directly to the weights rather than
    through the adaptive preconditioner, so it is not identical to an L2 term
    inside the loss. It is the closest correspondence this optimizer admits.
    Any disagreement attributable to that difference is a FOURTH cause on top of
    the three the committed rule names, and the tolerance cannot separate it.
    """
    return float(prior_precision) / float(n_observations)


def build_frozen_founder(pool_seed):
    """Generation 0 of a frozen-body chain, on that chain's own pool."""
    X, y, n_classes = make_pool(seed=pool_seed, kind="noise")
    X, y = X.to(DEVICE), y.to(DEVICE)
    model = new_frozen_model(X.shape[1], n_classes, pool_seed)
    train_head(model, X, y, soft=False, steps=STEPS_FOUNDER)
    with torch.no_grad():
        acc = float((model(X).argmax(-1) == y).float().mean())
        feats = model(X, return_feats=True)[1].clone()
    return model, X, y, n_classes, acc, feats


def run_frozen_chain(model, X, n_classes, chain_seed, gens, soft):
    """One frozen-body chain. Returns per-generation alphabet sizes and a record
    of the last generation's transmitted batch, which the tolerance needs.

    The body object is carried through unchanged; only the head is replaced. The
    features are asserted bitwise identical every generation, which is a
    freeze-leak check and not a measurement - with a fixed pool and a frozen body
    they are constant by construction.
    """
    torch.manual_seed(chain_seed)
    with torch.no_grad():
        feats0 = model(X, return_feats=True)[1].clone()

    sizes, leaked = [], 0
    last = None
    for g in range(gens):
        with torch.no_grad():
            t_logits = model(X)
        sel = torch.randperm(X.shape[0])[:BOTTLENECK]
        Xs = X[sel]
        if soft:
            targets = t_logits[sel]
        else:
            targets = t_logits[sel].argmax(-1)

        reinit_head(model, chain_seed * 1000 + g)
        train_head(model, Xs, targets, soft=soft)

        with torch.no_grad():
            feats = model(X, return_feats=True)[1]
            if not torch.equal(feats, feats0):
                leaked += 1
            sizes.append(int(model(X).argmax(-1).unique().numel()))
        last = (sel.clone(), Xs.clone(), targets.clone())

    return sizes, leaked, last, float(effective_rank(feats0))


# ------------------------------------------------------------ tolerance

def disagreement(a, b):
    """Fraction of points where two predicted-class vectors differ."""
    return float((np.asarray(a) != np.asarray(b)).mean())


def sgd_seed_spread(model, X, Xs, targets, n_pairs, base_seed):
    """SGD's own run-to-run predicted-class disagreement on identical data.

    This is the tolerance's baseline and the committed rule requires it to be
    measured and recorded BEFORE the Laplace comparison is read. Two heads are
    trained on the same transmitted batch in the same frozen feature space,
    differing only in initialization seed, and their predictions are compared on
    the full pool.
    """
    out = []
    for i in range(n_pairs):
        preds = []
        for which in (0, 1):
            reinit_head(model, base_seed + 2 * i + which)
            train_head(model, Xs, targets, soft=False,
                       wd=matched_weight_decay(len(targets)))
            with torch.no_grad():
                preds.append(model(X).argmax(-1).cpu().numpy())
        out.append(disagreement(preds[0], preds[1]))
    return out


def laplace_vs_sgd(model, X, Xs, targets, n_classes, sgd_seed):
    """Predicted-class disagreement between an SGD-trained head and the
    closed-form Laplace posterior mean for the same GLM on the same data.
    """
    with torch.no_grad():
        feats_batch = model(Xs, return_feats=True)[1].cpu().numpy().astype(np.float64)
        feats_pool = model(X, return_feats=True)[1].cpu().numpy().astype(np.float64)

    posterior = lg.fit_laplace(
        feats_batch, targets.cpu().numpy(), n_classes, prior_precision=PRIOR_PRECISION)
    if posterior.reason is not None:
        return None, posterior

    reinit_head(model, sgd_seed)
    train_head(model, Xs, targets, soft=False,
               wd=matched_weight_decay(len(targets)))
    with torch.no_grad():
        sgd_pred = model(X).argmax(-1).cpu().numpy()

    # Use the module's own predictor rather than a local matmul, so the weight
    # layout cannot be guessed wrong here and drift from how the posterior is
    # actually parameterized.
    analytic_pred = lg.predict_classes(posterior, feats_pool)
    return disagreement(sgd_pred, analytic_pred), posterior


# ------------------------------------------------------------ reporting

def print_rule(rule):
    print(f"C-8 preregistered rule, committed {rule['committed']}:")
    print(f"  question {rule['question']}")
    gate = rule["gates"]["A"]
    print(f"  gate A   {gate['condition']}")
    print(f"           fails if {gate['fails_if']}")
    print("  tolerance: a reproduction check, no null and no p-value; baseline is SGD's own")
    print("             seed-to-seed disagreement, measured before the comparison is read")
    print()


def main():
    ap = argparse.ArgumentParser(description="C-8: frozen-body null and Laplace bridge")
    ap.add_argument("--sanity-only", action="store_true",
                    help="precondition and reproduction checks only; writes no log")
    ap.add_argument("--chains", type=int, default=len(POOL_SEEDS),
                    help="smoke-test override; any value below the committed 20 writes no log")
    ap.add_argument("--gens", type=int, default=GENS,
                    help="smoke-test override; any value below the committed 25 writes no log")
    args = ap.parse_args()

    smoke = args.chains != len(POOL_SEEDS) or args.gens != GENS
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-8"]

    if args.sanity_only or smoke:
        print(f"C-8 {'smoke test' if smoke else 'harness verification'}: no gate verdict is "
              f"taken and no log is written.\n")
    else:
        print_rule(rule)
    print(f"{NOTE}\n")

    c1_mean, c1_ci, c1_finals, c1_ref = c1_intact_interval(C1_LOG)
    print(f"comparison arm, C-1 intact, recomputed from its log: mean {c1_mean:.2f}, "
          f"95% [{c1_ci[0]:.2f}, {c1_ci[1]:.2f}] over {len(c1_finals)} chains")
    c1_founder_accs = [float(r["founder_acc"]) for r in c1_ref["results"]["intact"]]
    c1_range = (min(c1_founder_accs), max(c1_founder_accs))
    print(f"its founders span [{c1_range[0]:.12f}, {c1_range[1]:.12f}] -- the precondition's "
          f"range\n")

    chains = []
    for i, pool_seed in enumerate(POOL_SEEDS[:args.chains]):
        model, X, y, n_classes, acc, _ = build_frozen_founder(pool_seed)
        sizes, leaked, last, eff = run_frozen_chain(
            model, X, n_classes, CHAIN_SEED_OFFSET + pool_seed, args.gens, soft=True)
        chains.append({
            "pool_seed": int(pool_seed),
            "chain_seed": int(CHAIN_SEED_OFFSET + pool_seed),
            "founder_acc": acc,
            "sizes": sizes,
            "final": int(sizes[-1]),
            "gen1": int(sizes[0]),
            "feature_leaks": int(leaked),
            "eff_rank_frozen_feats": eff,
        })
        print(f"  chain {pool_seed:2d}  founder {acc:.6f}  gen1 {sizes[0]:2d}  "
              f"gen{args.gens} {sizes[-1]:2d}  leaks {leaked}  "
              f"eff_rank {eff:.3f}   [{time.time()-t0:5.0f}s]")

    finals = [c["final"] for c in chains]
    gen1s = [c["gen1"] for c in chains]
    founder_accs = [c["founder_acc"] for c in chains]
    total_leaks = sum(c["feature_leaks"] for c in chains)

    rng = np.random.default_rng(BOOT_SEED)
    frozen_ci = bootstrap_ci_mean(finals, rng)
    rng1 = np.random.default_rng(BOOT_SEED)
    gen1_ci = bootstrap_ci_mean(gen1s, rng1)

    print(f"\nfrozen-body arm: mean {np.mean(finals):.2f}, "
          f"95% [{frozen_ci[0]:.2f}, {frozen_ci[1]:.2f}]")
    print(f"  generation 1:  mean {np.mean(gen1s):.2f}, "
          f"95% [{gen1_ci[0]:.2f}, {gen1_ci[1]:.2f}]")

    # ---- precondition, read before gate A
    inside = [c1_range[0] <= a <= c1_range[1] for a in founder_accs]
    precondition_ok = all(inside)
    print(f"\nprecondition, founder comparability: {sum(inside)} of {len(inside)} frozen "
          f"founders inside C-1's founder range")
    print(f"  frozen founders span [{min(founder_accs):.12f}, {max(founder_accs):.12f}]")
    print(f"  -> {'satisfied' if precondition_ok else 'NOT SATISFIED'}")

    # ---- headroom floor 1
    saturated = intervals_overlap(gen1_ci, c1_ci)
    if saturated:
        print("  headroom floor 1: generation-1 alphabet already overlaps C-1's interval, "
              "so the comparison had not moved before the chain ran")

    if not precondition_ok:
        gate_a = {"verdict": "NOT EVALUABLE", "pass": None, "headroom_ok": False,
                  "not_evaluable_reason": (
                      "the frozen-body founders do not fall inside the range C-1's own twenty "
                      "founders span, so the two arms do not start from comparable models")}
        print("\ngate A: NOT EVALUABLE - " + gate_a["not_evaluable_reason"])
    elif saturated:
        gate_a = {"verdict": "NOT EVALUABLE", "pass": None, "headroom_ok": False,
                  "not_evaluable_reason": (
                      "the generation-1 alphabet already overlaps the comparison interval")}
        print("\ngate A: NOT EVALUABLE - " + gate_a["not_evaluable_reason"])
    else:
        overlap = intervals_overlap(frozen_ci, c1_ci)
        gate_a = {"verdict": "PASSED" if overlap else "FAILED", "pass": bool(overlap),
                  "headroom_ok": True, "not_evaluable_reason": None}
        print(f"\ngate A: intervals {'overlap' if overlap else 'do NOT overlap'} "
              f"-> {gate_a['verdict']}")

    # ---- tolerance, on a hard-label chain
    print(f"\ntolerance, hard-label transmission  [{time.time()-t0:5.0f}s]")
    model, X, y, n_classes, acc, _ = build_frozen_founder(POOL_SEEDS[0])
    _, _, last, _ = run_frozen_chain(
        model, X, n_classes, CHAIN_SEED_OFFSET, args.gens, soft=False)
    _, Xs, targets = last

    spread = sgd_seed_spread(model, X, Xs, targets,
                             n_pairs=min(args.chains, 20), base_seed=SPREAD_SEED_OFFSET)
    rng2 = np.random.default_rng(BOOT_SEED)
    spread_ci = bootstrap_ci_mean(spread, rng2)
    print(f"  SGD seed-to-seed disagreement, {len(spread)} pairs: "
          f"mean {np.mean(spread):.6f}, 95% [{spread_ci[0]:.6f}, {spread_ci[1]:.6f}]")

    obs, posterior = laplace_vs_sgd(model, X, Xs, targets, n_classes,
                                    sgd_seed=SPREAD_SEED_OFFSET + 500)
    if obs is None:
        tolerance = {"verdict": "NOT EVALUABLE", "observed": None,
                     "not_evaluable_reason": posterior.reason}
        print(f"  -> NOT EVALUABLE: {posterior.reason}")
    else:
        within = spread_ci[0] <= obs <= spread_ci[1]
        tolerance = {
            "verdict": "WITHIN" if within else "OUTSIDE", "observed": obs,
            "baseline_mean": float(np.mean(spread)),
            "baseline_ci95": [spread_ci[0], spread_ci[1]],
            "n_pairs": len(spread), "converged": bool(posterior.converged),
            "n_iter": int(posterior.n_iter),
            "prior_only_directions": int(posterior.prior_only_directions),
            "not_evaluable_reason": None,
        }
        print(f"  Laplace vs SGD disagreement: {obs:.6f}")
        print(f"  posterior converged={posterior.converged} n_iter={posterior.n_iter} "
              f"prior_only_directions={posterior.prior_only_directions}")
        print(f"  -> {tolerance['verdict']} the baseline interval"
              f"{' : S-1 may proceed' if within else ' : S-1 must NOT proceed'}")

    print(f"\nsanity: feature leaks across all chains and generations: {total_leaks} "
          f"(must be 0)")

    if args.sanity_only or smoke:
        print(f"\n=== C-8 {'SMOKE' if smoke else 'HARNESS'} COMPLETE  "
              f"({time.time()-t0:.0f}s) === no log written")
        return

    out = os.path.join(REPO, "logs", "c8_frozen_body.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-8",
            "gate_a": gate_a, "tolerance": tolerance,
            "precondition_founder_comparability": {
                "c1_founder_range": list(c1_range),
                "frozen_founder_range": [min(founder_accs), max(founder_accs)],
                "n_inside": int(sum(inside)), "satisfied": bool(precondition_ok),
            },
            "frozen_mean": float(np.mean(finals)), "frozen_ci95": list(frozen_ci),
            "gen1_mean": float(np.mean(gen1s)), "gen1_ci95": list(gen1_ci),
            "c1_intact_mean": c1_mean, "c1_intact_ci95": list(c1_ci),
            "headroom_gen1_saturated": bool(saturated),
            "feature_leaks_total": int(total_leaks),
            "sgd_seed_spread": spread,
            "chains": chains,
            "gens": args.gens, "tau": TAU, "bottleneck": BOTTLENECK,
            "steps_gen": STEPS_GEN, "steps_founder": STEPS_FOUNDER,
            "pool_seeds": list(POOL_SEEDS), "chain_seed_offset": CHAIN_SEED_OFFSET,
            "spread_seed_offset": SPREAD_SEED_OFFSET,
            "prior_precision": PRIOR_PRECISION,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "device": DEVICE, "reference_log": os.path.relpath(C1_LOG, REPO),
            "note": NOTE, "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
        }, handle, indent=1)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
