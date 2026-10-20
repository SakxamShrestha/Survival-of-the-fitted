"""
C-4O: does the same-founder survivor-set overlap survive a frequency-matched null?

The preregistered rule is gates/thresholds.json's "C-4O" entry, committed
2026-10-20, before this file was written. C-4S made this control a prerequisite
for any further founder-identity statement rather than a refinement of the
statistics, and README.md and PROJECT.md were qualified the same day to say the
2.12x overlap must not be presented as evidence of founder identity until this
has run. This control lifts or confirms that qualification.

    gate A  mean pairwise Jaccard over Arm A's twenty survivor sets - the
            statistic the published 2.12x figure is computed from - against a
            null drawn at the observed set sizes with class probability
            proportional to the founder's own generation-0 class frequency.

    gate B  the same comparison on the mean pairwise overlap coefficient,
            which is insensitive to survivor-set size where Jaccard is not.

Two bounds on the conclusion, both written into the rule before the run.

The sampler preserves weight *ordering*, not exact per-class inclusion
probability: draw_survivors_weighted draws without replacement with
renormalization, and its docstring states that no without-replacement scheme
can make inclusion exactly proportional while drawing exactly k distinct
classes. A verdict here is therefore about an ordering-matched frequency null.

The weights are measured on the TRAINING pool, make_pool(seed=0, kind="noise"),
and not on the held-out probe pool C-4F and C-4S weight on. That is not an
inconsistency between the controls. run_seed_stability.py defines each chain's
survivor set as the classes its final teacher predicts on that training pool,
so each control measures its covariate on the pool its own outcome is defined
on. A probe-pool vector here would be a weighting from one pool applied to sets
from another - the splice withdrawn from figs/fig2_cka_blindness.png on the
same day this rule was committed.

Three headroom floors, per the committed clause. The third is specific to this
control and is the one most likely to fire: if weighting the null by founder
frequency does not move its mean by at least one of its own standard
deviations, this is the uniform null under another name and no verdict comes
out of it.

No chains are trained. Arm A's survivor sets are read unmodified from
logs/seed_stability_n20.json.

    python3 controls/c4o_overlap_null.py | tee logs/c4o_overlap_null.txt
    python3 controls/c4o_overlap_null.py --sanity-only   # writes no log
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# Run as `python3 controls/c4o_overlap_null.py` from the repository root:
# Python puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import overlap_stats as ov

# Read-only helpers, imported rather than copied so the founder construction and
# the headroom reporting cannot drift from C-4F's and C-4S's. Both modules guard
# their runners behind __main__, so importing them executes no experiment.
from controls.c4_frequency_null import (
    EXPECTED_FOUNDER_ACC,
    REPO,
    TRAIN_SEED,
    build_founder,
    combine_verdict,
    finite_or_none,
    founder_class_frequency,
    print_gate,
    summarise,
)

N_NULL = 10_000
NULL_SEED = 23          # distinct from c4_timing (11), C-4F (13), C-4S (17/19)
ARM = "A"
EXPECTED_N_RUNS = 20
EXPECTED_FOUNDER_SEED = 0
EXPECTED_CHAIN_SEEDS = list(range(100, 120))

SD_FLOOR = 0.004        # a tenth of the observed statistic's own interval half-width
MIN_DISTINCT_SETS = 1_000
DISTINCT_CAP = 1e12
REPRO_TOLERANCE = 5e-4  # "to three decimal places", per the committed sanity check
EXPECTED_JACCARD = 0.193
EXPECTED_UNIFORM_NULL = 0.091

REFERENCE_LOG = os.path.join(REPO, "logs", "seed_stability_n20.json")

NOTE = (
    "No chains are trained. Arm A's twenty survivor sets are read unmodified from "
    "logs/seed_stability_n20.json; that file is read and never rewritten. The founder is "
    "rebuilt from founder_seed 0 at 3000 steps, which both that log and logs/c4_timing.json "
    "record at memorization accuracy 0.977294921875. Class weights are the founder's "
    "generation-0 predicted class frequency on the TRAINING pool, make_pool(seed=0), because "
    "that is the pool run_seed_stability.py defines its survivor sets on. The "
    "size-conditioned uniform null is recomputed here beside the frequency-matched one, so "
    "any difference is attributable to the weighting and not to the data handling."
)


# ------------------------------------------------------------ reference data

def load_arm(path):
    """Arm A's twenty survivor sets, with the seeds the committed rule requires."""
    with open(path) as handle:
        ref = json.load(handle)

    runs = ref["arms"][ARM]["runs"]
    if len(runs) != EXPECTED_N_RUNS:
        raise ValueError(
            f"arm {ARM} has {len(runs)} runs, expected {EXPECTED_N_RUNS}; this is not the "
            "n = 20 run whose null C-4O replaces"
        )
    founder_seeds = {r["founder_seed"] for r in runs}
    if founder_seeds != {EXPECTED_FOUNDER_SEED}:
        raise ValueError(
            f"arm {ARM} founder seeds are {sorted(founder_seeds)}, expected a single shared "
            f"founder at seed {EXPECTED_FOUNDER_SEED}"
        )
    chain_seeds = [r["chain_seed"] for r in runs]
    if chain_seeds != EXPECTED_CHAIN_SEEDS:
        raise ValueError(
            f"arm {ARM} chain seeds are {chain_seeds}, expected "
            f"{EXPECTED_CHAIN_SEEDS[0]} to {EXPECTED_CHAIN_SEEDS[-1]}"
        )

    sets = [set(r["survivors"]) for r in runs]
    return ref, sets, [float(r["founder_acc"]) for r in runs]


def distinct_weighted_sets(sets, weights):
    """How many distinct survivor-set assignments the weighting leaves available.

    The product over runs of "choose k_i from the classes carrying positive
    weight". A class with zero weight can never be drawn, so a concentrated
    weighting shrinks this sharply; that is the degeneracy the committed floor
    exists to catch. Capped for reporting, because the product stops being
    informative far below the cap.
    """
    n_positive = int(np.count_nonzero(np.asarray(weights, dtype=float) > 0))
    total = 1.0
    per_run = []
    for s in sets:
        k = len(s)
        count = float(math.comb(n_positive, k)) if k <= n_positive else 0.0
        per_run.append(count)
        total = min(total * count, DISTINCT_CAP)
    return total, per_run, n_positive


# ------------------------------------------------------------ gates

def overlap_gate(sets, weights, n_classes, statistic_fn, rng_seed):
    """One gate: frequency-matched null, with the uniform null beside it.

    Both nulls are built from the same primitive over the same statistic with
    the same dependency structure, and each gets its own generator seeded
    identically, so the only difference between them is the weighting.
    """
    matched = ov.frequency_matched_null(
        sets, weights, np.random.default_rng(rng_seed), n_null=N_NULL,
        statistic_fn=statistic_fn, alternative="greater")
    uniform = ov.size_conditioned_null(
        sets, n_classes, np.random.default_rng(rng_seed), n_null=N_NULL,
        statistic_fn=statistic_fn, alternative="greater")

    distinct, per_run, n_positive = distinct_weighted_sets(sets, weights)
    result = summarise(matched.observed, matched.null_samples, "high",
                       SD_FLOOR, distinct, MIN_DISTINCT_SETS)

    matched_mean, matched_headroom = ov.null_headroom(
        matched.null_samples, matched.observed, "high")
    uniform_mean, uniform_headroom = ov.null_headroom(
        uniform.null_samples, uniform.observed, "high")

    # The third committed floor, specific to this control: a weighting that
    # does not move the null's mean by at least one of its own standard
    # deviations has not produced a frequency-matched null, whatever it is
    # called, and no verdict is taken from it.
    separation = abs(matched_mean - uniform_mean)
    matched_sd = result["null_sd"] or 0.0
    if separation < matched_sd:
        reason = (
            f"the frequency-matched null's mean {matched_mean:.4f} is within one of its own "
            f"standard deviations ({matched_sd:.4f}) of the uniform null's mean "
            f"{uniform_mean:.4f}; separation {separation:.4f}. Weighting by founder "
            "frequency did not move the null, so this is the uniform null under another name"
        )
        prior = result["not_evaluable_reason"]
        result["not_evaluable_reason"] = f"{prior}; {reason}" if prior else reason
        result["headroom_ok"] = False
        result["pass"] = None

    result.update({
        "null_headroom_to_bound": finite_or_none(matched_headroom),
        "uniform_null_mean": finite_or_none(uniform_mean),
        "uniform_null_headroom_to_bound": finite_or_none(uniform_headroom),
        "uniform_null_sd": finite_or_none(np.std(uniform.null_samples)),
        "uniform_p": finite_or_none(uniform.p_value),
        "null_separation": finite_or_none(separation),
        "separation_floor": finite_or_none(matched_sd),
        "distinct_per_run": [finite_or_none(v) for v in per_run],
        "classes_with_positive_weight": n_positive,
    })
    return result


# ------------------------------------------------------------ sanity

def check_sanity(acc, logged_accs, sets, n_classes, jaccard_obs, uniform_mean):
    """The reproduction checks the committed rule requires."""
    detail = [
        {
            "check": "founder memorization accuracy reproduces both logs",
            "expected": EXPECTED_FOUNDER_ACC, "got": acc,
            "match": acc == EXPECTED_FOUNDER_ACC,
        },
        {
            "check": "every Arm A run records that same founder accuracy",
            "expected": EXPECTED_FOUNDER_ACC, "got": sorted(set(logged_accs)),
            "match": set(logged_accs) == {EXPECTED_FOUNDER_ACC},
        },
        {
            "check": "observed mean pairwise Jaccard reproduces seed_stability_stats.py",
            "expected": EXPECTED_JACCARD, "got": round(jaccard_obs, 4),
            "match": abs(jaccard_obs - EXPECTED_JACCARD) < REPRO_TOLERANCE,
        },
        {
            "check": "size-conditioned uniform null mean reproduces its 0.091",
            "expected": EXPECTED_UNIFORM_NULL, "got": round(uniform_mean, 4),
            "match": abs(uniform_mean - EXPECTED_UNIFORM_NULL) < REPRO_TOLERANCE,
        },
        {
            "check": "every survivor class index is inside the class range",
            "expected": f"0 to {n_classes - 1}",
            "got": f"{min(min(s) for s in sets)} to {max(max(s) for s in sets)}",
            "match": all(0 <= c < n_classes for s in sets for c in s),
        },
    ]
    return all(d["match"] for d in detail), detail


# ------------------------------------------------------------ reporting

def print_rule(rule):
    print(f"C-4O preregistered rule, committed {rule['committed']}:")
    print(f"  question {rule['question']}")
    for name in ("A", "B"):
        gate = rule["gates"][name]
        print(f"  gate {name}  {gate['condition']}")
        print(f"          fails if {gate['fails_if']}")
    print()


def print_comparison(name, result):
    print(f"          frequency-matched null mean {result['null_mean']:.4f} "
          f"(sd {result['null_sd']:.4f}), uniform null mean "
          f"{result['uniform_null_mean']:.4f} (sd {result['uniform_null_sd']:.4f}), "
          f"separation {result['null_separation']:.4f} against a floor of "
          f"{result['separation_floor']:.4f}")
    print(f"          against the uniform null the same statistic gives "
          f"p = {result['uniform_p']:.4f}; distinct assignments "
          f"{result['distinct_assignments']:.0f} over "
          f"{result['classes_with_positive_weight']} classes with positive weight")


def main():
    parser = argparse.ArgumentParser(
        description="C-4O: does same-founder overlap survive a frequency-matched null?")
    parser.add_argument(
        "--sanity-only", action="store_true",
        help="rebuild the founder, reproduce the published statistic and the uniform null, "
             "and report the weighting's separation, without reading either gate; writes "
             "no log")
    args = parser.parse_args()

    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-4O"]

    if args.sanity_only:
        print("C-4O harness verification (--sanity-only). No gate is read and no log is "
              "written.\n")
    else:
        print_rule(rule)
    print(f"{NOTE}\n")

    ref, sets, logged_accs = load_arm(REFERENCE_LOG)
    founder, X, _, n_classes, acc = build_founder()
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.12f}  "
          f"({time.time()-t0:.1f}s)")

    weights = founder_class_frequency(founder, X, n_classes).astype(float)
    print(f"founder training-pool class frequency: min {int(weights.min())}, "
          f"max {int(weights.max())}, sum {int(weights.sum())}, "
          f"{int(np.count_nonzero(weights))} of {n_classes} classes predicted at all")
    print(f"Arm A survivor-set sizes: {[len(s) for s in sets]}")

    jaccard_obs = ov.mean_pairwise_jaccard(sets)
    uniform_probe = ov.size_conditioned_null(
        sets, n_classes, np.random.default_rng(NULL_SEED), n_null=N_NULL,
        statistic_fn=ov.mean_pairwise_jaccard, alternative="greater")
    uniform_mean = float(np.mean(uniform_probe.null_samples))

    sanity_ok, sanity_detail = check_sanity(
        acc, logged_accs, sets, n_classes, jaccard_obs, uniform_mean)
    print("\nsanity:")
    for d in sanity_detail:
        print(f"  {d['check']}: expected {d['expected']}, got {d['got']} -> "
              f"{'match' if d['match'] else 'MISMATCH'}")

    if not sanity_ok:
        print("\n=== C-4O MISCONFIGURED ===")
        print("The founder or the published statistic does not reproduce, so the")
        print("frequency-matched null would be compared against a different quantity than")
        print("the figure it is meant to test. Gates were not read.")
        return

    if args.sanity_only:
        distinct, _, n_positive = distinct_weighted_sets(sets, weights)
        print(f"\ndistinct weighted assignments {distinct:.0f} over {n_positive} classes "
              f"with positive weight (committed floor {MIN_DISTINCT_SETS})")
        print(f"\n=== C-4O HARNESS OK  ({time.time()-t0:.0f}s) ===")
        print("The founder reproduces, the published Jaccard and uniform null come back out")
        print("of the same log, and the weighting is reported above. The C-4O")
        print("preregistration is already committed; run without --sanity-only to read the")
        print("gates.")
        return

    gate_a = overlap_gate(sets, weights, n_classes, ov.mean_pairwise_jaccard, NULL_SEED)
    gate_b = overlap_gate(sets, weights, n_classes, ov.mean_pairwise_overlap, NULL_SEED + 1)

    print(f"\n=== C-4O VERDICT  ({time.time()-t0:.0f}s) ===")
    print_gate("gate A", "mean pairwise Jaccard, frequency-matched null", gate_a, "")
    print_comparison("gate A", gate_a)
    print_gate("gate B", "mean pairwise overlap coefficient, frequency-matched null",
               gate_b, "")
    print_comparison("gate B", gate_b)

    verdict = combine_verdict(rule, {"gate A": gate_a, "gate B": gate_b})

    out = os.path.join(REPO, "logs", "c4o_overlap_null.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-4O", "verdict": verdict,
            "sanity_ok": bool(sanity_ok), "sanity_detail": sanity_detail,
            "gate_a_jaccard": gate_a, "gate_b_overlap_coefficient": gate_b,
            "founder_training_pool_frequency": [int(v) for v in weights],
            "survivor_sets": [sorted(s) for s in sets],
            "arm": ARM, "n_runs": EXPECTED_N_RUNS,
            "n_null": N_NULL, "null_seed": NULL_SEED,
            "sd_floor": SD_FLOOR, "min_distinct_sets": MIN_DISTINCT_SETS,
            "n_classes": n_classes, "founder_acc": acc,
            "train_seed": TRAIN_SEED,
            "gens": ref["gens"], "tau": ref["tau"], "bottleneck": ref["bottleneck"],
            "reference_log": os.path.relpath(REFERENCE_LOG, REPO),
            "note": NOTE,
            "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
        }, handle, indent=1)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
