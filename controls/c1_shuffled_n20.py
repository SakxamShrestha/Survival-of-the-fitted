"""
C-1: shuffled transmission, replication at n = 20.

controls/c1_shuffled.py permutes the teacher's logits across the B selected
inputs before each student trains, destroying the input-to-target pairing
while leaving the multiset of transmitted logit vectors unchanged. If the
output alphabet still contracts under shuffling, contraction tells us nothing
about the learned input-to-output mapping. Both arms share a founder within a
replicate, so shuffling is the only difference between them.

That script ran at three seeds per arm on 2026-09-14 and reported 11.3
surviving classes intact against 1.3 shuffled -- "roughly a factor of nine" --
judged against an inline `1.25` constant that was never preregistered.
analysis/consistency_checks.py flags exactly that (check 5, "controls without
gates"). This script is the replication at n = 20 the gate in
gates/thresholds.json under "C-1" was written for, committed 2026-09-29,
before these twenty chains per arm existed. It retires the inline 1.25
constant; the decision rule now lives in the committed JSON and is read from
it at runtime, so the threshold cannot drift from the committed record.

    python3 controls/c1_shuffled_n20.py

Two choices the preregistration did not fix:

  - Which pool/founder seeds the twenty replicates use. c1_shuffled.py's own
    design is not "one shared founder, many chains" the way C-2/C-3/C-4 are:
    each of its three seeds builds its OWN pool and its OWN founder, and only
    the chain shares that founder across the intact/shuffled pair. That
    experimental logic is preserved unchanged here. Seeds 0-19 continue the
    original script's numbering (0, 1, 2) rather than opening a new
    namespace -- pool/founder seeds are not the thing the preregistration
    worries about colliding, only per-generation CHAIN seeds are (see
    SEED_NOTE below), because a chain-seed collision is what could make two
    supposedly independent chains secretly share randomness.

  - Chain-seed block. Reused for both arms within a replicate, exactly as
    c1_shuffled.py does, so generation 1's random B-selection is identical
    between the intact and shuffled arm of a replicate and shuffling is the
    only difference until its extra randperm() call diverges the two RNG
    streams. See SEED_NOTE for the block chosen and why it is clean.

The preregistration's sanity_check is informational, not a gate: the first
three replicates here share IDENTICAL founders with the 2026-09-14 run (same
pool seed, same torch.manual_seed sequence for the founder), but not identical
CHAIN seeds, so they are not expected to reproduce 11.3/1.3 exactly. What
would be a red flag is any of those three replicates' intact-arm counts
landing far outside the old per-seed range of 9 to 13 -- that would mean the
harness changed, not that the finding changed. This is printed as a
comparison and does not affect the verdict.
"""

import json
import os
import re
import sys
import time

import numpy as np
import torch

# Run as `python3 controls/c1_shuffled_n20.py` from the repository root:
# Python puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

GENS = 25
TAU = 1.0
BOTTLENECK = 128            # the C-1 condition: same tight bottleneck as C-4
DEVICE = "cpu"               # measured faster than MPS for this MLP (1.61 s vs 2.47 s)

# Each replicate gets its own pool AND its own founder -- c1_shuffled.py's own
# design, preserved unchanged. Seeds 0-19 continue that script's numbering
# (0, 1, 2) rather than opening a new namespace.
POOL_SEEDS = tuple(range(20))

# Chain seed = CHAIN_SEED_OFFSET + pool seed, reused for BOTH the intact and
# shuffled arm within a replicate, exactly as c1_shuffled.py's `100 + seed`
# was reused for both arms.
#
# Clean block: every torch.manual_seed(...) call and every *_SEED* constant in
# this repository was grepped on 2026-09-29. Chain seeds already on record are
# 100-119 (seed_stability.py / run_seed_stability.py Arm A), 300-319 (the same
# two files' Arm B/C, which also reuses 300-304 in the standalone
# sampler_arm.py), and 400-404 (controls/c4_class_mass.py). Founder seeds
# 200-219 are also on record for the same files' Arm B/C founders. 600-619
# does not intersect any of these, nor STUDENT_SEED0 = 1000-1019
# (controls/c3_one_step.py) or the small fixed seeds (PERM_SEED = 7,
# PROBE_SEED = 9000) used elsewhere.
CHAIN_SEED_OFFSET = 600

N_BOOT = 10_000              # matches analysis/seed_stability_stats.py's N_BOOT
BOOT_SEED = 0                # matches that script's own default --seed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED_NOTE = (
    "Pool/founder seeds 0-19 continue controls/c1_shuffled.py's own numbering "
    "(0, 1, 2); each replicate builds its own pool and its own founder by that "
    "script's design, not a single shared founder the way C-2/C-3/C-4 do. "
    "Chain seeds are 600-619 (CHAIN_SEED_OFFSET + pool seed), reused for both "
    "arms within a replicate exactly as the n = 3 script reused 100 + seed. "
    "600-619 does not collide with any chain seed on record: 100-119 and "
    "300-319 (seed_stability.py / run_seed_stability.py), 200-219 (the same "
    "files' founder seeds), or 400-404 (controls/c4_class_mass.py)."
)

# Quoted from gates/thresholds.json's own "background" and "sanity_check"
# fields for C-1, not independently re-derived. Used only for the informational
# comparison the sanity_check calls for; it is reported, not gated.
OLD_N3_INTACT_MEAN = 11.3
OLD_N3_SHUFFLED_MEAN = 1.3
OLD_N3_INTACT_RANGE = (9, 13)


def gate_a_threshold_from_rule(rule):
    """C-1's gate A threshold is written in prose ("at least 3.0"), not a
    dedicated numeric field the way C-3's top-level "threshold" key is. Pull
    the number out of both `condition` and `fails_if` and require them to
    agree, so a future hand-edit that changes one sentence but not the other
    is caught here rather than silently parsed past.
    """
    cond = rule["gates"]["A"]["condition"]
    fails = rule["gates"]["A"]["fails_if"]
    cond_val = float(re.search(r"\d+\.\d+", cond).group())
    fails_val = float(re.search(r"\d+\.\d+", fails).group())
    assert cond_val == fails_val, (
        f"gate A threshold disagrees between condition ({cond_val}) and "
        f"fails_if ({fails_val}) in gates/thresholds.json; fix the JSON "
        "before trusting this run"
    )
    return cond_val


def run_chain(X, teacher, n_classes, chain_seed, shuffle):
    """One 25-generation chain; identical logic to controls/c1_shuffled.py's
    run_chain. Returns the per-generation alphabet-size trajectory and the
    final surviving classes, both measured on the full pool X exactly as the
    n = 3 pilot measured them (not a held-out probe pool -- that is not part
    of what this replication changes).
    """
    torch.manual_seed(chain_seed)
    sizes = []
    for _ in range(GENS):
        with torch.no_grad():
            t_logits = teacher(X)
        sel = torch.randperm(X.shape[0])[:BOTTLENECK]
        Xs, Ts = X[sel], t_logits[sel]
        if shuffle:
            # Permute targets against inputs. The multiset of transmitted
            # logit vectors is unchanged; only the pairing is destroyed.
            Ts = Ts[torch.randperm(Ts.shape[0])]
        student = MLP(X.shape[1], n_classes).to(DEVICE)
        train(student, Xs, Ts, 1500, 256, 1e-3, 0.0, True, TAU, DEVICE)
        teacher = student
        with torch.no_grad():
            sizes.append(int(teacher(X).argmax(-1).unique().numel()))
    with torch.no_grad():
        survivors = sorted(teacher(X).argmax(-1).unique().tolist())
    return sizes, survivors


def bootstrap_ci_mean(values, rng, n_boot=N_BOOT, alpha=0.05):
    """95% percentile interval on the arm mean, resampling CHAINS with
    replacement -- analysis/seed_stability_stats.py's bootstrap convention
    (resample over runs, not over pairs), applied to a scalar per-chain metric
    instead of a pairwise Jaccard. That script skips resampled pairs where
    both slots land on the same original run, because Jaccard(x, x) = 1 is an
    artefact of a *pairwise* statistic; there is no analogous artefact here,
    since a resampled chain just contributes its own surviving-class count to
    the resampled mean.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def intervals_overlap(a, b):
    """Whether two closed intervals [lo, hi] share any point."""
    return not (a[0] > b[1] or b[0] > a[1])


def main():
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-1"]
    gate_a_threshold = gate_a_threshold_from_rule(rule)

    print(f"C-1 preregistered rule, committed {rule['committed']}:")
    print(f"  metric   {rule['metric']}")
    print(f"  gate A   {rule['gates']['A']['condition']}")
    print(f"           fails if {rule['gates']['A']['fails_if']}  "
          f"(threshold parsed from JSON: {gate_a_threshold})")
    print(f"  gate B   {rule['gates']['B']['condition']}")
    print(f"           fails if {rule['gates']['B']['fails_if']}")
    print(f"\n{SEED_NOTE}\n")

    header = (f"{'seed':>4}  {'chain_seed':>10}  {'founder_acc':>11}  "
              f"{'intact':>6}  {'shuffled':>8}  {'elapsed':>8}")
    print(header)
    print("-" * len(header))

    results = {"intact": [], "shuffled": []}
    for seed in POOL_SEEDS:
        torch.manual_seed(seed)
        X, y0, C = make_pool(seed=seed, kind="noise")
        founder = MLP(X.shape[1], C).to(DEVICE)
        train(founder, X, y0, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
        with torch.no_grad():
            acc = float((founder(X).argmax(-1) == y0).float().mean())

        chain_seed = CHAIN_SEED_OFFSET + seed
        row_finals = {}
        for arm, shuffle in (("intact", False), ("shuffled", True)):
            sizes, survivors = run_chain(X, founder, C, chain_seed, shuffle)
            results[arm].append({"seed": seed, "chain_seed": chain_seed,
                                  "founder_acc": acc, "sizes": sizes,
                                  "final": sizes[-1], "survivors": survivors})
            row_finals[arm] = sizes[-1]

        print(f"{seed:>4}  {chain_seed:>10}  {acc:>11.3f}  "
              f"{row_finals['intact']:>6}  {row_finals['shuffled']:>8}  "
              f"{time.time()-t0:>7.0f}s")

    rng = np.random.default_rng(BOOT_SEED)
    intact = [r["final"] for r in results["intact"]]
    shuffled = [r["final"] for r in results["shuffled"]]
    intact_mean, shuffled_mean = float(np.mean(intact)), float(np.mean(shuffled))
    intact_ci = bootstrap_ci_mean(intact, rng)
    shuffled_ci = bootstrap_ci_mean(shuffled, rng)

    print(f"\n=== C-1 VERDICT  ({time.time()-t0:.0f}s, {len(POOL_SEEDS)} chains/arm, "
          f"BOTTLENECK={BOTTLENECK}, tau={TAU}, {GENS} gens) ===")
    print(f"intact transmission   : mean {intact_mean:.2f}  "
          f"95% CI [{intact_ci[0]:.2f}, {intact_ci[1]:.2f}]  {intact}")
    print(f"shuffled transmission : mean {shuffled_mean:.2f}  "
          f"95% CI [{shuffled_ci[0]:.2f}, {shuffled_ci[1]:.2f}]  {shuffled}")

    # Informational only -- see module docstring and the JSON's sanity_check.
    # Not gated: the first three replicates share founders with the
    # 2026-09-14 run but not chain seeds, so exact reproduction of 11.3/1.3 is
    # not expected.
    old_seed_vals = [r["final"] for r in results["intact"] if r["seed"] in (0, 1, 2)]
    lo, hi = OLD_N3_INTACT_RANGE
    within = [lo <= v <= hi for v in old_seed_vals]
    print(f"\nsanity comparison (not gated): replicates 0-2 share founders with "
          f"the 2026-09-14 run (means {OLD_N3_INTACT_MEAN}/{OLD_N3_SHUFFLED_MEAN}) "
          f"but not chain seeds, so exact reproduction is not expected.")
    print(f"  new intact finals for seeds 0-2: {old_seed_vals}  vs old per-seed "
          f"range [{lo}, {hi}] -> {['within' if w else 'outside' for w in within]}")

    ratio = intact_mean / shuffled_mean if shuffled_mean else float("inf")
    gate_a_pass = ratio >= gate_a_threshold
    overlap = intervals_overlap(intact_ci, shuffled_ci)
    gate_b_pass = not overlap

    print(f"\ngate A: ratio {ratio:.2f}x vs threshold {gate_a_threshold:.2f}x -> "
          f"{'PASSED' if gate_a_pass else 'FAILED'}")
    print(f"gate B: intact CI [{intact_ci[0]:.2f}, {intact_ci[1]:.2f}] vs shuffled CI "
          f"[{shuffled_ci[0]:.2f}, {shuffled_ci[1]:.2f}] "
          f"-> {'overlap' if overlap else 'no overlap'} -> "
          f"{'PASSED' if gate_b_pass else 'FAILED'}")

    if gate_a_pass and gate_b_pass:
        verdict = "PASSED"
        for line in rule["verdict_if_passed"]:
            print(line)
    else:
        verdict = "FAILED"
        for line in rule["verdict_if_failed"]:
            print(line)

    out = os.path.join(REPO, "logs", "c1_shuffled_n20.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-1", "verdict": verdict,
            "gate_a_pass": bool(gate_a_pass), "gate_a_ratio": ratio,
            "gate_a_threshold": gate_a_threshold,
            "gate_b_pass": bool(gate_b_pass), "gate_b_overlap": bool(overlap),
            "intact_mean": intact_mean, "intact_ci95": list(intact_ci),
            "intact_finals": intact,
            "shuffled_mean": shuffled_mean, "shuffled_ci95": list(shuffled_ci),
            "shuffled_finals": shuffled,
            "old_n3_reference": {
                "intact_mean": OLD_N3_INTACT_MEAN,
                "shuffled_mean": OLD_N3_SHUFFLED_MEAN,
                "intact_per_seed_range": list(OLD_N3_INTACT_RANGE),
                "comparable_new_seeds": [0, 1, 2],
                "comparable_new_intact_finals": old_seed_vals,
                "note": "informational only, not gated; see module docstring",
            },
            "gens": GENS, "tau": TAU, "bottleneck": BOTTLENECK,
            "n_chains_per_arm": len(POOL_SEEDS),
            "pool_seeds": list(POOL_SEEDS), "chain_seed_offset": CHAIN_SEED_OFFSET,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "device": DEVICE,
            "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
            "seed_note": SEED_NOTE,
            "results": results,
        }, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
