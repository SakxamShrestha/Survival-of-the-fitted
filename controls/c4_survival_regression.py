"""
C-4S: what predicts how long a class survives -- transmission, or the founder?

The preregistered rule is gates/thresholds.json's "C-4S" entry, committed
2026-10-15 together with C-4F and before either ran. The question it asks is
not C-4F's. C-4F asks whether C-4T's timing effect is confounded with founder
class frequency. C-4S asks what the operative variable actually is, across
the six predictors RESEARCH-PLAN.md section 4.3 names, and its stake is
stated there plainly: if founder class frequency alone explains survival,
then the founder-identity claim in PROJECT.md reduces to "frequent classes
survive."

Two gates carry decisions and four predictors are reported as descriptives
with bootstrap intervals and no gate. That split is in the committed rule and
is deliberate -- six gates at 0.05 would be a multiple-comparisons problem,
and the two that carry decisions are the two whose outcomes change what the
report claims. A descriptive is not promoted to a gate here, whatever it
turns out to say.

    gate A  pooled Spearman(founder generation-0 predicted class frequency,
            death generation). One-sided, because the boring explanation makes
            a signed prediction: classes the founder predicts more often
            should survive longer, not merely differently.

    gate B  pooled partial Spearman(L2 norm of the founder head's weight row,
            death generation | founder frequency), both residualised on
            frequency by rank regression. Two-sided, because no theory in this
            project predicts the sign of a head-norm effect and committing to
            a direction would be a guess dressed as a hypothesis.

The outcome is death generation and not a binary survival indicator at
generation 25, for the reason the committed rule gives: C-4 established that
37 to 48 of the 50 classes are already dead by then, so an endpoint indicator
is the saturated comparison that made C-4's gate B unreadable. Censoring is
real but light in rank terms -- two to thirteen classes per chain are
censored at 26 -- which is why both gates are rank-based and why no
coefficient from a least-squares fit is used as a gate statistic.

Headroom, per the committed clause. Gate A reports its null mean, standard
deviation and range. Gate B additionally reports the rank variance each
residual retains, and is declared NOT EVALUABLE if either falls below one
percent of its unresidualised rank variance, because a partial correlation
computed on a residual with nothing left in it is the error C-4's gate B made
in a different coordinate system. The check is in analysis/rank_stats.py and
is enforced before any p-value is read.

No chains are trained. Death generations and generation-1 transmitted counts
are read unmodified from logs/c4_timing.json. The founder is rebuilt from
train_seed 0 to obtain the five founder-side predictors, none of which any
existing log records.

    python3 controls/c4_survival_regression.py | tee logs/c4_survival_regression.txt
    python3 controls/c4_survival_regression.py --sanity-only   # writes no log
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Run as `python3 controls/c4_survival_regression.py` from the repository root:
# Python puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import rank_stats as rs

# Read-only helpers, imported rather than copied so that the founder
# construction, the reference-log handling and the headroom reporting cannot
# drift from C-4F's. Both modules guard their runners behind __main__, so
# importing them executes no experiment.
from controls.c4_frequency_null import (
    CHAIN_SEED0,
    DEVICE,
    EXPECTED_FOUNDER_ACC,
    GENS,
    N_CHAINS,
    N_PERM,
    PROBE_POOL_SIZE,
    PROBE_SEED,
    REPO,
    STAT_TOLERANCE,
    TRAIN_SEED,
    build_founder,
    combine_verdict,
    finite_or_none,
    founder_class_frequency,
    load_reference,
    print_gate,
    summarise,
)
from controls.c4_timing import HAVE_SCIPY

PERM_SEED = 17          # distinct from c4_class_mass.py (7), c4_timing.py (11), C-4F (13)
BOOT_SEED = 19
N_BOOT = 10_000
RETAINED_FLOOR = rs.DEFAULT_RETAINED_FLOOR   # the committed one percent
REFERENCE_LOG = os.path.join(REPO, "logs", "c4_timing.json")

NOTE = (
    "No chains are trained. Death generations and generation-1 effective transmitted "
    "counts are read unmodified from logs/c4_timing.json; that file is read and never "
    "rewritten. The founder is rebuilt from train_seed 0 to obtain five per-class "
    "predictors on the held-out probe pool at probe_seed 9000 - predicted frequency, mean "
    "top-1 margin, head weight-row L2 norm, head weight-row sum, and mean feature norm - "
    "none of which any existing log records. The founder is one model, so its five "
    "predictors are identical across the five chains and are repeated when pooling; only "
    "the death generations differ per chain."
)

GATE_A_LABEL = "pooled Spearman(founder frequency, death gen)"
GATE_B_LABEL = "pooled partial Spearman(head row L2 norm, death gen | founder frequency)"


# ------------------------------------------------------------ founder predictors

def founder_predictors(founder, Xpr, n_classes):
    """The six per-class predictors the committed rule names, as a dict.

    Five are founder-side and computed here; the sixth, the generation-1
    effective transmitted count, is per-chain and is read from C-4T's log by
    the caller. Every founder-side predictor is defined over the probe points
    the founder assigns to class k by argmax, so a class the founder never
    predicts has no points and its margin and feature norm are undefined.
    Those are recorded as NaN rather than zero: zero is a measured value and
    would place such a class at the bottom of a rank rather than outside it.
    """
    with torch.no_grad():
        logits, feats = founder(Xpr, return_feats=True)
        pred = logits.argmax(-1)
        top2 = logits.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        feat_norm = feats.norm(dim=-1).cpu().numpy()
        head = founder.head.weight.detach().cpu().numpy()

    assigned = pred.cpu().numpy()
    mean_margin = np.full(n_classes, np.nan)
    mean_feat_norm = np.full(n_classes, np.nan)
    for k in range(n_classes):
        points = assigned == k
        if points.any():
            mean_margin[k] = float(margin[points].mean())
            mean_feat_norm[k] = float(feat_norm[points].mean())

    return {
        "founder_frequency": founder_class_frequency(founder, Xpr, n_classes).astype(float),
        "head_row_l2": np.linalg.norm(head, axis=1),
        "head_row_sum": head.sum(axis=1),
        "mean_top1_margin": mean_margin,
        "mean_feature_norm": mean_feat_norm,
    }


def pool(per_class, n_chains):
    """Repeat a per-class founder predictor once per chain, to match pooled death."""
    return np.tile(np.asarray(per_class, dtype=float), n_chains)


# ------------------------------------------------------------ gates

def frequency_gate(freq_pooled, death_by_chain, rng):
    """Gate A: does founder frequency alone predict death generation?

    One-sided on a positive correlation. The null permutes each chain's death
    generations among its own classes, which is C-4T's gate A scheme carried
    over unchanged for comparability, as the committed derivation requires.
    """
    pooled_death = np.concatenate(death_by_chain)
    observed = rs.spearman(freq_pooled, pooled_death)

    null = np.empty(N_PERM)
    for i in range(N_PERM):
        shuffled = np.concatenate([rng.permutation(d) for d in death_by_chain])
        null[i] = rs.spearman(freq_pooled, shuffled)

    result = summarise(observed, null, "high", None, None, None)
    result["retained_a"] = None
    result["retained_b"] = None
    return result


def head_geometry_gate(head_pooled, freq_pooled, death_by_chain, rng):
    """Gate B: does head geometry predict survival beyond founder frequency?

    Two-sided on a partial correlation with frequency projected out of both
    variables. The null permutes the head-norm values within each chain's
    block, leaving the frequency covariate and the death generations in place,
    so the only thing destroyed is the pairing the gate is about.

    The residualisation is done once on the pooled real data and the
    permutation is applied to the residual, which is what the committed metric
    describes: the statistic is a correlation between two residual vectors, and
    the null is over their pairing.
    """
    pooled_death = np.concatenate(death_by_chain)
    observed = rs.partial_spearman(
        head_pooled, pooled_death, freq_pooled, retained_floor=RETAINED_FLOOR)

    if observed.reason is not None:
        result = summarise(float("nan"), np.full(N_PERM, np.nan), "two", None, None, None)
        result["not_evaluable_reason"] = observed.reason
        result["headroom_ok"] = False
        result["pass"] = None
        result["retained_a"] = observed.retained_a
        result["retained_b"] = observed.retained_b
        return result

    resid_head, _ = rs.residualise_on_ranks(head_pooled, freq_pooled)
    resid_death, _ = rs.residualise_on_ranks(pooled_death, freq_pooled)
    blocks = np.split(resid_head, len(death_by_chain))

    null = np.empty(N_PERM)
    for i in range(N_PERM):
        shuffled = np.concatenate([rng.permutation(b) for b in blocks])
        null[i] = float(np.corrcoef(shuffled, resid_death)[0, 1])

    result = summarise(observed.rho, null, "two", None, None, None)
    result["retained_a"] = observed.retained_a
    result["retained_b"] = observed.retained_b
    return result


# ------------------------------------------------------------ descriptives

def bootstrap_spearman(predictor_per_class, death_by_chain, rng, n_boot=N_BOOT):
    """Pooled Spearman with a percentile interval, resampling classes.

    Classes are the resampling unit, not pooled points: the same class appears
    once per chain and its five death generations are not independent of each
    other, so resampling points would treat one class as five observations and
    report an interval narrower than the data supports.

    A predictor with NaN entries - a class the founder never predicts has no
    margin and no feature norm - is handled by dropping those classes from both
    the point estimate and every resample, and the count dropped is reported
    beside the interval rather than absorbed silently.
    """
    predictor = np.asarray(predictor_per_class, dtype=float)
    n_classes = predictor.size
    keep = np.flatnonzero(np.isfinite(predictor))

    def pooled(class_idx):
        pred = np.concatenate([predictor[class_idx] for _ in death_by_chain])
        death = np.concatenate([d[class_idx] for d in death_by_chain])
        return rs.spearman(pred, death)

    observed = pooled(keep)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = pooled(rng.choice(keep, size=keep.size, replace=True))

    finite = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan))
    return {
        "observed": finite_or_none(observed),
        "ci95_lo": finite_or_none(lo),
        "ci95_hi": finite_or_none(hi),
        "n_boot": int(n_boot),
        "n_classes_used": int(keep.size),
        "n_classes_dropped": int(n_classes - keep.size),
        "finite_draws": int(finite.size),
    }


# ------------------------------------------------------------ sanity

def check_sanity(acc, predictors, ref, death, n_classes):
    """The reproduction checks the committed rule requires.

    The founder must be C-4T's founder, its predicted frequencies must account
    for every probe point, and the per-chain mean death generations already in
    C-4T's log must come back out under this script's data handling.
    """
    freq = predictors["founder_frequency"]
    detail = [
        {
            "check": "founder memorization accuracy reproduces logs/c4_timing.json",
            "expected": EXPECTED_FOUNDER_ACC, "got": acc,
            "match": acc == EXPECTED_FOUNDER_ACC,
        },
        {
            "check": "founder predicted frequencies account for every probe point",
            "expected": PROBE_POOL_SIZE, "got": int(freq.sum()),
            "match": int(freq.sum()) == PROBE_POOL_SIZE,
        },
        {
            "check": "every founder-side predictor has one entry per class",
            "expected": n_classes,
            "got": sorted({int(v.size) for v in predictors.values()}),
            "match": all(v.size == n_classes for v in predictors.values()),
        },
    ]

    for chain, d in zip(ref["chains"], death):
        expected = float(np.mean(chain["death_gen"]))
        detail.append({
            "check": f"chain {chain['chain_seed']} mean death generation reproduces",
            "expected": expected, "got": float(np.mean(d)),
            "match": bool(abs(float(np.mean(d)) - expected) < STAT_TOLERANCE),
        })

    return all(d["match"] for d in detail), detail


# ------------------------------------------------------------ reporting

def print_rule(rule):
    print(f"C-4S preregistered rule, committed {rule['committed']}:")
    print(f"  question {rule['question']}")
    for name in ("A", "B"):
        gate = rule["gates"][name]
        print(f"  gate {name}  {gate['condition']}")
        print(f"          fails if {gate['fails_if']}")
    print()


def print_retained(result):
    """Gate B's headroom, in the terms the committed clause asks for."""
    if result.get("retained_a") is None:
        return
    print(f"          residual rank variance retained: head norm "
          f"{result['retained_a']:.4f}, death generation {result['retained_b']:.4f} "
          f"(committed floor {RETAINED_FLOOR})")


def print_descriptive(name, stats):
    dropped = ""
    if stats["n_classes_dropped"]:
        dropped = (f", {stats['n_classes_dropped']} class(es) dropped as undefined; "
                   f"{stats['n_classes_used']} used")
    print(f"  {name:<22} rho = {stats['observed']:+.4f}  "
          f"95% [{stats['ci95_lo']:+.4f}, {stats['ci95_hi']:+.4f}]{dropped}")


def main():
    parser = argparse.ArgumentParser(
        description="C-4S: does the founder predict how long a class survives?")
    parser.add_argument(
        "--sanity-only", action="store_true",
        help="rebuild the founder, reproduce C-4T's per-chain death generations and "
             "report gate B's residual headroom, without reading or evaluating either "
             "gate; writes no log")
    args = parser.parse_args()

    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-4S"]

    backend = "scipy.stats.spearmanr" if HAVE_SCIPY else "numpy rank + Pearson"
    if args.sanity_only:
        print("C-4S harness verification (--sanity-only). No gate is read and no log is "
              "written.\n")
    else:
        print_rule(rule)
    print(f"{NOTE}\n")
    print(f"rank backend: analysis/rank_stats.py (numpy); C-4T backend was {backend}\n")

    ref, death, mass, _ = load_reference(REFERENCE_LOG)
    founder, _, Xpr, n_classes, acc = build_founder()
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.12f}  "
          f"({time.time()-t0:.1f}s)")

    predictors = founder_predictors(founder, Xpr, n_classes)
    predictors["transmitted_count_g1"] = np.mean(mass, axis=0)

    sanity_ok, sanity_detail = check_sanity(acc, predictors, ref, death, n_classes)
    print("\nsanity:")
    for d in sanity_detail:
        print(f"  {d['check']}: expected {d['expected']}, got {d['got']} -> "
              f"{'match' if d['match'] else 'MISMATCH'}")

    freq_pooled = pool(predictors["founder_frequency"], len(death))
    head_pooled = pool(predictors["head_row_l2"], len(death))
    probe = rs.partial_spearman(
        head_pooled, np.concatenate(death), freq_pooled, retained_floor=RETAINED_FLOOR)
    print(f"\ngate B residual headroom: head norm {probe.retained_a:.4f}, death "
          f"generation {probe.retained_b:.4f} of unresidualised rank variance "
          f"(committed floor {RETAINED_FLOOR})")
    if probe.reason is not None:
        print(f"  -> NOT EVALUABLE: {probe.reason}")

    if not sanity_ok:
        print("\n=== C-4S MISCONFIGURED ===")
        print("The founder or C-4T's per-chain death generations do not reproduce, so the")
        print("predictors do not belong to the founder whose chains produced these death")
        print("generations. Gates were not read.")
        return

    if args.sanity_only:
        print(f"\n=== C-4S HARNESS OK  ({time.time()-t0:.0f}s) ===")
        print("The founder reproduces, every predictor is defined, and gate B's residuals")
        print("are reported above. Commit the C-4S preregistration before running without")
        print("--sanity-only; the commit order is the evidence that the rule was written")
        print("before the result existed.")
        return

    rng = np.random.default_rng(PERM_SEED)
    gate_a = frequency_gate(freq_pooled, death, rng)
    gate_b = head_geometry_gate(head_pooled, freq_pooled, death, rng)

    boot_rng = np.random.default_rng(BOOT_SEED)
    descriptive_names = ("mean_top1_margin", "head_row_sum", "mean_feature_norm",
                         "transmitted_count_g1")
    descriptives = {
        name: bootstrap_spearman(predictors[name], death, boot_rng)
        for name in descriptive_names
    }

    print(f"\n=== C-4S VERDICT  ({time.time()-t0:.0f}s) ===")
    print_gate("gate A", GATE_A_LABEL, gate_a, "")
    print_gate("gate B", GATE_B_LABEL, gate_b, "")
    print_retained(gate_b)

    print("\nDescriptives, no gate attached - the committed rule reports these with")
    print("intervals and takes no decision from them (six gates at 0.05 would be a")
    print("multiple-comparisons problem). Spearman against death generation:")
    for name in descriptive_names:
        print_descriptive(name, descriptives[name])

    verdict = combine_verdict(rule, {"gate A": gate_a, "gate B": gate_b})

    out = os.path.join(REPO, "logs", "c4_survival_regression.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-4S", "verdict": verdict,
            "sanity_ok": bool(sanity_ok), "sanity_detail": sanity_detail,
            "gate_a_frequency": gate_a, "gate_b_head_geometry": gate_b,
            "descriptives": descriptives,
            "predictors_per_class": {k: [float(x) for x in v]
                                     for k, v in predictors.items()},
            "retained_floor": RETAINED_FLOOR,
            "n_perm": N_PERM, "perm_seed": PERM_SEED,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "gens": GENS, "n_chains": N_CHAINS, "chain_seed0": CHAIN_SEED0,
            "train_seed": TRAIN_SEED, "probe_seed": PROBE_SEED,
            "n_classes": n_classes, "founder_acc": acc, "device": DEVICE,
            "reference_log": os.path.relpath(REFERENCE_LOG, REPO),
            "note": NOTE,
            "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
        }, handle, indent=1)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
