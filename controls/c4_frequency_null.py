"""
C-4F: is C-4T's timing effect a founder-frequency effect?

C-4T (2026-09-29) failed both its gates. Pooled Spearman between a class's
effective transmitted count at generation 1 and its death generation was 0.363
at p = 1.0e-4, and the classes starved at generation 1 died 5.51 generations
earlier than the rest at p = 1.0e-4. Minority collapse (Fang, He, Long and Su,
PNAS 2021) is therefore on record as an unexcluded explanation for alphabet
contraction.

Both of C-4T's nulls are uniform over classes, and neither knows anything about
the founder. A class the founder itself rarely predicts contributes little
softmax mass to any transmitted set, so it is starved at generation 1 close to
by construction, and it may die early because it was already marginal in
generation 0 rather than because of the mechanism Fang et al. describe. Under a
uniform null, "starved" and "founder-rare" are not separable.

This control re-runs C-4T's two statistics against nulls matched on the
founder's generation-0 predicted class frequency, and reports them beside
C-4T's own uniform nulls recomputed from the same log. Nothing else changes:
same statistic, same data, same permutation count, same one-sided direction.
What changes is what the null is allowed to know.

Both outcomes cost the project something, and gates/thresholds.json records that
before this ran. If the effect survives matching, minority collapse stands and
the constructed matched-imbalance control at depth is owed. If it does not, the
timing effect is a founder-frequency effect, which is the "frequent classes
survive" account that RESEARCH-PLAN.md section 4.3 names as the boring rival to
the founder-identity claim. C-4S measures that account directly.

No chains are trained. Death generations, generation-1 transmitted mass and the
generation-1 starved sets are read unmodified from logs/c4_timing.json, which
controls/c4_timing.py produced on this machine; that file is read and never
rewritten. The only new computation is the founder's generation-0 predicted
class frequency on the held-out probe pool, which no existing log records and
which requires rebuilding the founder from train_seed 0.

    python3 controls/c4_frequency_null.py --sanity-only
    python3 controls/c4_frequency_null.py | tee logs/c4_frequency_null.txt

--sanity-only rebuilds the founder, reproduces C-4T's two observed statistics
from its log, and reports the headroom each matched null will have, without
reading or evaluating either gate. It writes no log. It exists so the harness
can be verified before the preregistration is committed, since reading a gate
before its rule is in the git history is the one thing the preregistration
procedure is there to prevent.

Decision rules are preregistered in gates/thresholds.json under "C-4F" and are
read from that file at runtime rather than restated here, so the thresholds
cannot drift from the committed record.

Direction is fixed in the preregistration: a significant effect is what FAILS a
gate. This control cannot manufacture support for the headline claim, only move
which explanation the project owes an answer to.

One application note, recorded here rather than left implicit. The committed
headroom rule names two floors: a null standard deviation of 1.0 generation and
1000 distinct achievable fake starved sets. The first is dimensioned in
generations and is applied to gate A, whose statistic is a difference of mean
death generations. Gate B's statistic is a rank correlation and has no committed
floor on that scale, so for gate B the null standard deviation is reported
without a floor and only the distinct-assignment floor is enforced. This is an
application of the rule as written to a statistic of different dimension, not a
relaxation of it, and the log records which floor was enforced on which gate.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# Run as `python3 controls/c4_frequency_null.py` from the repository root:
# Python puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

# Read-only helpers, imported rather than copied so that the rank statistic
# cannot drift from C-4T's. controls/c4_timing.py guards its runner behind
# __main__, so importing it executes no experiment.
from controls.c4_timing import HAVE_SCIPY, spearman_corr

TAU = 1.0
BOTTLENECK = 128            # the C-4 condition: the contraction result's bottleneck
GENS = 25
N_CHAINS = 5
STARVED_CUTOFF = 1.0        # one effective example; the unit Fang et al. use, as in C-4
N_PERM = 10_000             # carried over from C-4 and C-4T unchanged
N_STRATA = 5                # five strata of ten classes; derivation in the committed rule
DEVICE = "cpu"              # measured faster than MPS for this MLP (1.61 s vs 2.47 s)
TRAIN_SEED = 0
PROBE_SEED = 9_000
CHAIN_SEED0 = 400
PERM_SEED = 13              # distinct from c4_class_mass.py (7) and c4_timing.py (11)

FOUNDER_STEPS = 3_000       # identical to c4_timing.py's founder
EXPECTED_FOUNDER_ACC = 0.977294921875   # recorded in logs/c4_timing.json
PROBE_POOL_SIZE = 4_096

SD_FLOOR_GENERATIONS = 1.0  # death generation is integer-valued; see the committed rule
MIN_DISTINCT_SETS = 1_000
DISTINCT_CAP = 1e12         # reporting cap; the product stops being informative far below
STAT_TOLERANCE = 1e-9       # reproducing a float already in the log, not measuring one

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_LOG = os.path.join(REPO, "logs", "c4_timing.json")

NOTE = (
    "No chains are trained. Death generations, generation-1 transmitted mass and the "
    "generation-1 starved sets are read unmodified from logs/c4_timing.json; that file is "
    "read and never rewritten. The founder is rebuilt from train_seed 0 solely to obtain "
    "generation-0 predicted class frequency on the held-out probe pool at probe_seed 9000, "
    "the covariate the matching needs. Both matched nulls are reported beside C-4T's own "
    "uniform nulls recomputed from the same log, so any difference is attributable to the "
    "null and not to the data handling."
)


# ------------------------------------------------------------ reference data

def load_reference(path):
    """C-4T's log: per-chain death generations, generation-1 mass and starved sets.

    Requires all five chains at the expected seeds. A log with fewer chains, or
    different seeds, is not the experiment whose nulls this control replaces,
    and failing here is cheaper than discovering it inside a verdict.
    """
    with open(path) as handle:
        ref = json.load(handle)

    expected = [CHAIN_SEED0 + c for c in range(N_CHAINS)]
    got = [c["chain_seed"] for c in ref["chains"]]
    if got != expected:
        raise ValueError(
            f"{os.path.relpath(path, REPO)} has chain seeds {got}, expected {expected}; "
            "this is not the C-4T run whose nulls C-4F replaces"
        )

    death = [np.asarray(c["death_gen"], dtype=int) for c in ref["chains"]]
    mass = [np.asarray(c["mass_g1"], dtype=float) for c in ref["chains"]]
    masks = []
    for chain, d in zip(ref["chains"], death):
        mask = np.zeros(len(d), dtype=bool)
        if chain["starved_g1"]:
            mask[chain["starved_g1"]] = True
        masks.append(mask)
    return ref, death, mass, masks


# ------------------------------------------------------------ founder covariate

def build_founder():
    """Rebuild C-4T's founder and return it with the pools and its accuracy.

    The RNG sequence - make_pool at TRAIN_SEED, make_pool at PROBE_SEED,
    manual_seed(TRAIN_SEED), MLP init, 3000 training steps - is c4_timing.main's
    sequence. Any divergence shows up as a memorization accuracy that is not
    0.977294921875, which the sanity check treats as disqualifying.
    """
    X, y, n_classes = make_pool(seed=TRAIN_SEED, kind="noise")
    Xpr, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    X, y, Xpr = X.to(DEVICE), y.to(DEVICE), Xpr.to(DEVICE)

    torch.manual_seed(TRAIN_SEED)
    founder = MLP(X.shape[1], n_classes).to(DEVICE)
    train(founder, X, y, FOUNDER_STEPS, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        acc = float((founder(X).argmax(-1) == y).float().mean())
    return founder, X, Xpr, n_classes, acc


def founder_class_frequency(founder, Xpr, n_classes):
    """Generation-0 predicted class frequency: probe points the founder argmaxes to k."""
    with torch.no_grad():
        pred = founder(Xpr).argmax(-1)
    counts = np.bincount(pred.cpu().numpy(), minlength=n_classes)
    return counts.astype(int)


def frequency_strata(freq, n_strata):
    """Stratum index per class, from a rank on founder frequency.

    Ties are broken by class index via a stable sort, so the partition is a
    function of the founder alone and is reproducible. Returns a new array;
    `freq` is not modified.
    """
    n_classes = len(freq)
    if n_classes % n_strata != 0:
        raise ValueError(
            f"{n_classes} classes do not divide into {n_strata} equal strata; the "
            "committed rule specifies five strata of ten"
        )
    order = np.argsort(freq, kind="stable")
    strata = np.empty(n_classes, dtype=int)
    per = n_classes // n_strata
    for position, class_index in enumerate(order):
        strata[class_index] = min(position // per, n_strata - 1)
    return strata


# ------------------------------------------------------------ matched null

def stratum_profile(mask, strata, n_strata):
    """How many of a chain's starved classes fall in each founder-frequency stratum."""
    return np.array([int(np.sum(mask & (strata == s))) for s in range(n_strata)], dtype=int)


def draw_matched_mask(profile, members_by_stratum, n_classes, rng):
    """A fake starved set with the same stratum profile as the real one.

    Draws `profile[s]` classes without replacement from stratum s, for every s.
    The result carries the real starved set's founder-frequency composition
    exactly, which is the point: what the null shuffles is which particular
    classes within a frequency band were starved, not how frequency-skewed the
    starved set is.
    """
    mask = np.zeros(n_classes, dtype=bool)
    for stratum, count in enumerate(profile):
        if count == 0:
            continue
        picked = rng.choice(members_by_stratum[stratum], size=int(count), replace=False)
        mask[picked] = True
    return mask


def distinct_matched_sets(profiles, members_by_stratum):
    """How many distinct fake starved sets the matching leaves available.

    The product over chains of the product over strata of C(stratum size,
    profile count). This is the committed headroom diagnostic: stratified
    matching constrains the draw, and a starved set that exhausts its strata
    leaves a null that is nearly deterministic. Capped for reporting.
    """
    total = 1.0
    per_chain = []
    for profile in profiles:
        chain_total = 1.0
        for stratum, count in enumerate(profile):
            chain_total *= math.comb(len(members_by_stratum[stratum]), int(count))
        per_chain.append(float(min(chain_total, DISTINCT_CAP)))
        total = min(total * chain_total, DISTINCT_CAP)
    return float(total), per_chain


# ------------------------------------------------------------ statistics

def finite_or_none(value):
    """A float for the log, or None when the value is not finite.

    json.dump writes a bare NaN, which is not valid JSON and which a reader
    quietly turns back into a float. A field that could not be measured is
    recorded as null instead, so a degenerate run cannot be mistaken for a
    measured one.
    """
    number = float(value)
    return number if math.isfinite(number) else None


def diff_stat(death_by_chain, masks):
    """Mean death generation of the masked classes minus that of the rest, pooled.

    The statistic C-4T's gate B used. Returns NaN when either side is empty
    across every chain, which the callers treat as not evaluable, not as zero.
    """
    starved, other = [], []
    for death, mask in zip(death_by_chain, masks):
        starved.extend(death[mask])
        other.extend(death[~mask])
    if not starved or not other:
        return float("nan")
    return float(np.mean(starved) - np.mean(other))


def pooled_spearman(mass_by_chain, death_by_chain):
    """Spearman rho between generation-1 effective count and death generation, pooled.

    The statistic C-4T's gate A used.
    """
    mass = np.concatenate(mass_by_chain)
    death = np.concatenate(death_by_chain)
    if np.std(mass) == 0 or np.std(death) == 0:
        return float("nan")
    return spearman_corr(mass, death)


def summarise(observed, null, extreme_side, sd_floor, distinct, distinct_floor):
    """Gate result plus the headroom the committed rule requires beside it.

    `extreme_side` is "low" when the hypothesis predicts a smaller statistic
    (starved classes dying earlier), "high" when it predicts a larger one (a
    positive correlation), and "two" when no direction is predicted, in which
    case the extreme count is taken on the absolute value. C-4F uses only the
    one-sided forms; "two" exists for C-4S's gate B, which the committed rule
    makes two-sided because no theory in this project predicts the sign of a
    head-norm effect. A gate whose null cannot clear its floors is reported
    NOT EVALUABLE and contributes no verdict: a null without room is the defect
    that made C-4's gate B and C-2R's gates meaningless, and it is checked here
    rather than discovered afterwards.

    A non-finite statistic is refused for the reason
    controls/c4_timing.py:211-215 records: a NaN compares as False against every
    permutation, so the extreme count comes out at zero and the p-value lands at
    its own minimum, 1/(1 + n_perm). An undefined correlation would therefore be
    reported as the most significant result the test can produce. Gate A happens
    to be shielded by its standard-deviation floor, which fires on the same
    degenerate input; gate B's distinct-assignment floor is structural and
    depends only on the stratum sizes, so it cannot catch this and the check
    below is gate B's only protection. This is not reachable on the real
    logs/c4_timing.json data, where both pooled statistics are non-constant and
    within-stratum permutation preserves the variance exactly. It is guarded
    because a latent path that converts "not measurable" into "p = 1e-4" is the
    single worst failure this harness could have.
    """
    null = np.asarray(null, dtype=float)
    null_sd = float(np.std(null))

    reasons = []
    if not np.isfinite(observed):
        reasons.append(
            "the observed statistic is not finite, so its comparison against the null is "
            "undefined"
        )
    non_finite = int(np.sum(~np.isfinite(null)))
    if non_finite:
        reasons.append(
            f"{non_finite} of {len(null)} null replicates are not finite, so the null "
            "distribution is undefined"
        )
    if sd_floor is not None and null_sd < sd_floor:
        reasons.append(
            f"null standard deviation {null_sd:.4f} is below the committed floor of "
            f"{sd_floor} generation"
        )
    if distinct is not None and distinct < distinct_floor:
        reasons.append(
            f"only {distinct:.0f} distinct fake starved sets are achievable, below the "
            f"committed floor of {distinct_floor}"
        )

    if reasons:
        p = None
    else:
        if extreme_side == "low":
            extreme = int(np.sum(null <= observed))
        elif extreme_side == "two":
            extreme = int(np.sum(np.abs(null) >= abs(observed)))
        else:
            extreme = int(np.sum(null >= observed))
        p = float((1 + extreme) / (1 + len(null)))

    return {
        "observed": finite_or_none(observed),
        "null_mean": finite_or_none(np.mean(null)),
        "null_sd": finite_or_none(null_sd),
        "null_min": finite_or_none(np.min(null)),
        "null_max": finite_or_none(np.max(null)),
        "p": p,
        "n_perm": int(len(null)),
        "distinct_assignments": None if distinct is None else float(distinct),
        "sd_floor_enforced": sd_floor,
        "headroom_ok": not reasons,
        "not_evaluable_reason": None if not reasons else "; ".join(reasons),
        "pass": None if reasons else bool(p >= 0.05),
    }


# ------------------------------------------------------------ gates

def matched_starved_gate(death, masks, strata, members, n_classes, rng):
    """Gate A: starved-versus-rest timing against a frequency-matched null."""
    profiles = [stratum_profile(m, strata, len(members)) for m in masks]
    distinct, per_chain = distinct_matched_sets(profiles, members)

    observed = diff_stat(death, masks)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        fake = [draw_matched_mask(p, members, n_classes, rng) for p in profiles]
        for real, drawn in zip(profiles, fake):
            drawn_profile = stratum_profile(drawn, strata, len(members))
            if not np.array_equal(real, drawn_profile):
                raise AssertionError(
                    "matched draw did not reproduce the real stratum profile: "
                    f"{drawn_profile.tolist()} against {real.tolist()}"
                )
        null[i] = diff_stat(death, fake)

    result = summarise(observed, null, "low", SD_FLOOR_GENERATIONS,
                       distinct, MIN_DISTINCT_SETS)
    result["stratum_profiles"] = [p.tolist() for p in profiles]
    result["distinct_per_chain"] = per_chain
    return result


def uniform_starved_gate(death, masks, n_classes, rng):
    """C-4T's gate B recomputed: the same statistic against its uniform null.

    Reported for comparison, not for a verdict. Its purpose is to show that any
    difference C-4F reports comes from the null rather than from the data
    handling, so its observed value must equal the one in C-4T's log.
    """
    sizes = [int(m.sum()) for m in masks]
    observed = diff_stat(death, masks)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        fake = []
        for size in sizes:
            mask = np.zeros(n_classes, dtype=bool)
            if size > 0:
                mask[rng.choice(n_classes, size=size, replace=False)] = True
            fake.append(mask)
        null[i] = diff_stat(death, fake)
    return summarise(observed, null, "low", None, None, MIN_DISTINCT_SETS)


def stratified_spearman_gate(mass, death, strata, members, rng):
    """Gate B: the dose-response correlation against a within-stratum null.

    Permuting death generations only within founder-frequency strata holds the
    founder-frequency composition of the comparison fixed, so what survives is
    whatever transmitted mass explains beyond founder frequency.
    """
    observed = pooled_spearman(mass, death)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        shuffled = []
        for d in death:
            new = d.copy()
            for stratum_members in members:
                new[stratum_members] = rng.permutation(d[stratum_members])
            shuffled.append(new)
        null[i] = pooled_spearman(mass, shuffled)

    per_chain = math.prod(math.factorial(len(m)) for m in members)
    distinct = float(min(per_chain ** len(death), DISTINCT_CAP))
    return summarise(observed, null, "high", None, distinct, MIN_DISTINCT_SETS)


def uniform_spearman_gate(mass, death, rng):
    """C-4T's gate A recomputed against its uniform null, for comparison only."""
    observed = pooled_spearman(mass, death)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = pooled_spearman(mass, [rng.permutation(d) for d in death])
    return summarise(observed, null, "high", None, None, MIN_DISTINCT_SETS)


# ------------------------------------------------------------ sanity

def check_sanity(acc, freq, ref, death, mass, n_classes):
    """The reproduction checks the committed rule requires.

    The founder must be C-4T's founder, the probe frequencies must account for
    every probe point, and C-4T's two observed statistics must come back out of
    its own log under this script's data handling.
    """
    detail = [
        {
            "check": "founder memorization accuracy reproduces logs/c4_timing.json",
            "expected": EXPECTED_FOUNDER_ACC,
            "got": acc,
            "match": acc == EXPECTED_FOUNDER_ACC,
        },
        {
            "check": "founder predicted frequencies account for every probe point",
            "expected": PROBE_POOL_SIZE,
            "got": int(freq.sum()),
            "match": int(freq.sum()) == PROBE_POOL_SIZE,
        },
    ]

    starved_masks = [
        np.isin(np.arange(n_classes), chain["starved_g1"]) for chain in ref["chains"]
    ]
    for key, value, label in (
        ("gate_a_observed_rho", pooled_spearman(mass, death),
         "C-4T gate A observed rho recomputed from its log"),
        ("gate_b_observed_diff", diff_stat(death, starved_masks),
         "C-4T gate B observed difference recomputed from its log"),
    ):
        expected = ref[key]
        detail.append({
            "check": label,
            "expected": expected,
            "got": float(value),
            "match": bool(abs(float(value) - expected) < STAT_TOLERANCE),
        })

    return all(d["match"] for d in detail), detail


# ------------------------------------------------------------ reporting

def print_rule(rule):
    print(f"C-4F preregistered rule, committed {rule['committed']}:")
    print(f"  metric   {rule['metric']}")
    for name in ("A", "B"):
        gate = rule["gates"][name]
        print(f"  gate {name}   {gate['condition']}")
        print(f"           fails if {gate['fails_if']}")
    print()


def print_gate(name, label, result, units):
    if not result["headroom_ok"]:
        print(f"{name}: NOT EVALUABLE - {result['not_evaluable_reason']}")
        return
    verdict = "PASSED" if result["pass"] else "FAILED"
    print(f"{name}: {label} = {result['observed']:.4f}{units}  "
          f"(null mean {result['null_mean']:.4f}, sd {result['null_sd']:.4f}, "
          f"range [{result['null_min']:.4f}, {result['null_max']:.4f}]), "
          f"p = {result['p']:.4f} -> {verdict}")


def combine_verdict(rule, gates_by_name):
    """Apply the committed `combination` clause and print its consequence.

    The clause, not this function, decides what happens when only one gate is
    readable: both evaluable gives PASSED or FAILED, exactly one gives PARTIAL
    carrying only that gate's consequence, and neither gives NOT EVALUABLE. The
    policy lives in gates/thresholds.json so that it is in the git history
    alongside the thresholds rather than chosen here after the fact.
    """
    evaluable = {name: g for name, g in gates_by_name.items() if g["headroom_ok"]}
    refused = [name for name in gates_by_name if name not in evaluable]

    if not evaluable:
        print("\nNeither gate had the headroom the committed rule requires. This control")
        print("returns no verdict and nothing should be concluded from it.")
        return "NOT EVALUABLE"

    failed = [name for name, g in evaluable.items() if not g["pass"]]
    outcome = "verdict_if_failed" if failed else "verdict_if_passed"

    if refused:
        carried = ", ".join(sorted(evaluable))
        print(f"\nPARTIAL: {carried} was readable and {', '.join(refused)} was not.")
        print("The consequence below is the one that gate carries, and it is the whole of")
        print("what this control establishes. It is not a pass or a failure of the control.")
        for line in rule[outcome]:
            print(line)
        return f"PARTIAL ({carried} only, {outcome.removeprefix('verdict_if_')})"

    for line in rule[outcome]:
        print(line)
    return "FAILED" if failed else "PASSED"


def report_setup(args, rule, backend):
    if args.sanity_only:
        print("C-4F harness verification (--sanity-only). No gate is read and no log is "
              "written.\n")
    else:
        print_rule(rule)
    print(f"{NOTE}\n")
    print(f"Spearman backend: {backend}\n")


def main():
    parser = argparse.ArgumentParser(
        description="C-4F: is C-4T's timing effect a founder-frequency effect?")
    parser.add_argument(
        "--sanity-only", action="store_true",
        help="rebuild the founder, reproduce C-4T's statistics and report the matched "
             "nulls' headroom, without reading or evaluating either gate; writes no log")
    args = parser.parse_args()

    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-4F"]

    backend = "scipy.stats.spearmanr" if HAVE_SCIPY else "numpy rank + Pearson"
    report_setup(args, rule, backend)

    ref, death, mass, masks = load_reference(REFERENCE_LOG)
    founder, _, Xpr, n_classes, acc = build_founder()
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.12f}  "
          f"({time.time()-t0:.1f}s)")

    freq = founder_class_frequency(founder, Xpr, n_classes)
    strata = frequency_strata(freq, N_STRATA)
    members = [np.flatnonzero(strata == s) for s in range(N_STRATA)]
    print(f"founder frequency strata (rarest first), probe points per stratum: "
          f"{[int(freq[m].sum()) for m in members]}")

    sanity_ok, sanity_detail = check_sanity(acc, freq, ref, death, mass, n_classes)
    print("\nsanity:")
    for d in sanity_detail:
        mark = "match" if d["match"] else "MISMATCH"
        print(f"  {d['check']}: expected {d['expected']}, got {d['got']} -> {mark}")

    profiles = [stratum_profile(m, strata, N_STRATA) for m in masks]
    distinct, per_chain = distinct_matched_sets(profiles, members)
    print(f"\nmatched-null headroom: stratum profiles {[p.tolist() for p in profiles]}")
    print(f"  distinct fake starved sets per chain {[f'{v:.0f}' for v in per_chain]}, "
          f"product {distinct:.0f} (committed floor {MIN_DISTINCT_SETS})")

    if not sanity_ok:
        print("\n=== C-4F MISCONFIGURED ===")
        print("The founder or the reference statistics do not reproduce. The frequency")
        print("covariate does not belong to the founder whose chains produced these death")
        print("generations, so the matching would be meaningless. Gates were not read.")
        return

    if args.sanity_only:
        print(f"\n=== C-4F HARNESS OK  ({time.time()-t0:.0f}s) ===")
        print("The founder reproduces, C-4T's two observed statistics come back out of its")
        print("log under this script's data handling, and the matched null has the headroom")
        print("the committed rule requires. Commit the C-4F preregistration before running")
        print("without --sanity-only; the commit order is the evidence that the rule was")
        print("written before the result existed.")
        return

    rng = np.random.default_rng(PERM_SEED)
    gate_a = matched_starved_gate(death, masks, strata, members, n_classes, rng)
    gate_b = stratified_spearman_gate(mass, death, strata, members, rng)
    uniform_b = uniform_starved_gate(death, masks, n_classes, rng)
    uniform_a = uniform_spearman_gate(mass, death, rng)

    print(f"\n=== C-4F VERDICT  ({time.time()-t0:.0f}s) ===")
    print_gate("gate A", "starved-minus-other mean death gen, frequency-matched null",
               gate_a, " generations")
    print_gate("gate B", "pooled Spearman, within-stratum null", gate_b, "")
    print("\nC-4T's own nulls recomputed on the same data, for comparison only:")
    print_gate("  C-4T gate B", "starved-minus-other mean death gen, uniform null",
               uniform_b, " generations")
    print_gate("  C-4T gate A", "pooled Spearman, uniform null", uniform_a, "")

    verdict = combine_verdict(rule, {"gate A": gate_a, "gate B": gate_b})

    out = os.path.join(REPO, "logs", "c4_frequency_null.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-4F", "verdict": verdict,
            "sanity_ok": bool(sanity_ok), "sanity_detail": sanity_detail,
            "spearman_backend": backend,
            "gate_a_matched": gate_a, "gate_b_stratified": gate_b,
            "c4t_gate_b_uniform_recomputed": uniform_b,
            "c4t_gate_a_uniform_recomputed": uniform_a,
            "founder_class_frequency": [int(v) for v in freq],
            "strata": [int(v) for v in strata],
            "starved_cutoff": STARVED_CUTOFF, "n_perm": N_PERM, "perm_seed": PERM_SEED,
            "n_strata": N_STRATA, "sd_floor_generations": SD_FLOOR_GENERATIONS,
            "min_distinct_sets": MIN_DISTINCT_SETS,
            "tau": TAU, "bottleneck": BOTTLENECK, "gens": GENS, "n_chains": N_CHAINS,
            "chain_seed0": CHAIN_SEED0, "train_seed": TRAIN_SEED,
            "probe_seed": PROBE_SEED, "n_classes": n_classes, "founder_acc": acc,
            "device": DEVICE,
            "reference_log": os.path.relpath(REFERENCE_LOG, REPO),
            "note": NOTE,
            "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
        }, handle, indent=1)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
