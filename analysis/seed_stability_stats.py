"""Interval estimates for the seed-stability arms.

Post-processing only. Reads the JSON written by ``run_seed_stability.py`` and
computes everything from the saved survivor sets, so the statistics can be
corrected and re-run without repeating the chains.

    python3 analysis/seed_stability_stats.py
    python3 analysis/seed_stability_stats.py --log logs/seed_stability_n20.json --first 5

Three things are computed beyond the point estimates the original scripts
printed.

**Bootstrap interval.** Resampling is over *runs*, not over pairs. The C(n, 2)
pairwise Jaccards are not independent — each run appears in n - 1 of them — so an
interval taken over pairs would be far too narrow. Resampling runs with
replacement duplicates them, and a duplicated run paired against itself has
Jaccard 1.0, which would bias the mean upward; pairs whose two slots carry the
same original run index are therefore skipped.

**Empirical null.** ``seed_stability.expected_jaccard`` evaluates a closed form at
the *mean* survivor count. Observed set sizes are heterogeneous (roughly 2-11
classes), and Jaccard is not linear in set size, so the closed form is an
approximation. The empirical null redraws random subsets at each run's actual
observed size and is the one to trust if the two disagree.

**Permutation test, Arm A versus Arm B.** The null being tested is precisely
*"the arm label carries no information about survivor-set overlap"*. It is not a
general test of founder specificity: Arm A's runs share a founder by
construction, so they are not exchangeable with Arm B's runs in the strict sense
a permutation test assumes. Read it as a calibrated statement about this
contrast, not as a p-value for the mechanism.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

# Run as `python3 analysis/seed_stability_stats.py` from the repository root:
# Python puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seed_stability import jaccard, expected_jaccard

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_BOOT = 10000
N_PERM = 10000
N_NULL = 10000


def mean_pairwise(sets):
    """Mean Jaccard over all distinct pairs."""
    pairs = list(itertools.combinations(range(len(sets)), 2))
    return float(np.mean([jaccard(sets[i], sets[j]) for i, j in pairs]))


def bootstrap_ci(sets, rng, n_boot=N_BOOT, alpha=0.05):
    """Percentile interval, resampling runs with replacement.

    Pairs drawn from the same original run are skipped: a run against itself has
    Jaccard 1.0 and is an artefact of resampling, not data.
    """
    n = len(sets)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals = [jaccard(sets[idx[i]], sets[idx[j]])
                for i, j in itertools.combinations(range(n), 2)
                if idx[i] != idx[j]]
        if vals:
            stats.append(np.mean(vals))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def empirical_null(sizes, n_classes, rng, n_null=N_NULL):
    """Mean pairwise Jaccard for random subsets matching the observed sizes."""
    stats = []
    for _ in range(n_null):
        draws = [set(rng.choice(n_classes, size=k, replace=False)) for k in sizes]
        stats.append(mean_pairwise(draws))
    return float(np.mean(stats)), float(np.percentile(stats, 97.5))


def permutation_test(sets_a, sets_b, rng, n_perm=N_PERM):
    """Two-sided test on the difference in mean pairwise Jaccard."""
    observed = mean_pairwise(sets_a) - mean_pairwise(sets_b)
    pooled = list(sets_a) + list(sets_b)
    na = len(sets_a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pooled))
        a = [pooled[i] for i in perm[:na]]
        b = [pooled[i] for i in perm[na:]]
        if abs(mean_pairwise(a) - mean_pairwise(b)) >= abs(observed):
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(REPO, "logs", "seed_stability_n20.json"))
    ap.add_argument("--first", type=int, default=0,
                    help="use only the first N runs per arm (for subset checks)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.log) as handle:
        data = json.load(handle)
    C = data["n_classes"]
    rng = np.random.default_rng(args.seed)

    print(f"{os.path.relpath(args.log, REPO)}  |  {data['gens']} gens, "
          f"tau={data['tau']}, B={data['bottleneck']}, {data['n_runs']} runs, "
          f"{C} classes, torch {data['torch_version']}")
    if args.first:
        print(f"restricted to the first {args.first} runs per arm")

    survivors = {}
    for arm, block in data["arms"].items():
        runs = block["runs"][: args.first] if args.first else block["runs"]
        survivors[arm] = [set(r["survivors"]) for r in runs]

    print(f"\n{'arm':<4}{'n':>4}{'mean k':>8}{'Jaccard':>10}{'95% CI':>18}"
          f"{'analytic':>10}{'empirical':>11}{'ratio':>8}")
    for arm in sorted(survivors):
        sets = survivors[arm]
        sizes = [len(s) for s in sets]
        obs = mean_pairwise(sets)
        lo, hi = bootstrap_ci(sets, rng)
        analytic = expected_jaccard(float(np.mean(sizes)), C)
        emp, emp_hi = empirical_null(sizes, C, rng)
        print(f"{arm:<4}{len(sets):>4}{np.mean(sizes):>8.1f}{obs:>10.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{analytic:>10.3f}{emp:>11.3f}"
              f"{obs/emp:>7.2f}x")
        print(f"     {data['arms'][arm]['description']}"
              f"   (null 97.5th pct {emp_hi:.3f}, "
              f"{'above' if obs > emp_hi else 'INSIDE'} the null)")

    if "A" in survivors and "B" in survivors:
        diff, p = permutation_test(survivors["A"], survivors["B"], rng)
        print(f"\nArm A vs Arm B: difference in mean Jaccard {diff:+.3f}, "
              f"permutation p = {p:.4f} ({N_PERM} permutations)")
        print("null: the arm label carries no information about survivor-set overlap")

    if "B" in survivors and "C" in survivors:
        diff, p = permutation_test(survivors["B"], survivors["C"], rng)
        print(f"Arm B vs Arm C: difference in mean Jaccard {diff:+.3f}, "
              f"permutation p = {p:.4f}")


if __name__ == "__main__":
    main()
