"""
C-4T: does a class that receives less transmitted signal die sooner?

C-4 (2026-09-29) established that soft-logit transmission starves classes: 26 of
them fall below one effective example at the first transmission, pooled over
five chains. Its gate B was supposed to say whether that starvation matters. It
could not: gate B compared the classes starved at generation 1 against the
classes absent at generation 25, but by generation 25 between 37 and 48 of 50
classes are already absent, so a randomly chosen set of three to seven classes
already overlaps the dead set at a null of 0.880. The observed overlap of 0.971
against that null gave p = 0.1004 with 0.12 of headroom - a gate with no
resolution, not a passing result. The defect is in the endpoint comparison, not
in the data.

This control replaces the endpoint with a timing measurement. Each class gets a
death generation: the first generation at which it is absent from the student's
predictions on the held-out probe pool and remains absent through generation 25
(classes still predicted at generation 25 are censored at 26). Minority collapse
(Fang, He, Long and Su, PNAS 2021) predicts that classes fed less signal die
earlier. Gate A tests that directly, across all fifty classes rather than only
the starved handful. Gate B restricts the same test to the starved set C-4
already identified.

The chains are seed-identical to controls/c4_class_mass.py - same founder, same
bottleneck, same seeds 400-404 - so the re-run must reproduce C-4's generation-1
starved counts exactly (4, 7, 5, 7, 3). What is new is recording, at every
generation, WHICH classes the student predicts on the probe pool. c4_class_mass.py
never logged that, so per-class death generations could not be recovered from its
output; a re-run was required to get them.

    python3 controls/c4_timing.py | tee logs/c4_timing.txt

Decision rules are preregistered in gates/thresholds.json under "C-4T", committed
before this ran. They are read from that file at runtime rather than restated
here, so the thresholds cannot drift from the committed record.

Direction is fixed in the preregistration: a significant POSITIVE correlation
between generation-1 signal and death generation (gate A), or starved classes
dying significantly EARLIER than the rest (gate B), is what FAILS this control.
A null result is what clears minority collapse as the mechanism; this gate
cannot manufacture support for the project's headline result, only weaken it or
leave it unchanged.

controls/c4_class_mass.py is not edited or imported for its data: it stays
byte-identical to the version that produced logs/c4_class_mass.json, and this
script only reads that JSON file, once, to check that its own re-run reproduces
it.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Run as `python3 controls/c4_timing.py` from the repository root: Python puts
# this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

# scipy is checked for, not assumed: use scipy.stats.spearmanr when it is
# installed, and fall back to a numpy rank transform (average ranks for ties,
# matching scipy.stats.rankdata's default) plus Pearson correlation on the
# ranks when it is not. Pearson-on-ranks is the definition of Spearman's rho,
# so the fallback is not an approximation, just a slower way to the same number.
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

TAU = 1.0
BOTTLENECK = 128            # the C-4 condition: the contraction result's bottleneck
GENS = 25
N_CHAINS = 5
STARVED_CUTOFF = 1.0        # one effective example; the unit Fang et al. use, same as C-4
N_PERM = 10_000
DEVICE = "cpu"              # measured faster than MPS for this MLP (1.61 s vs 2.47 s)
TRAIN_SEED = 0
PROBE_SEED = 9_000          # identical to c2_no_chaining.py, c3_one_step.py, c4_class_mass.py
CHAIN_SEED0 = 400           # identical to c4_class_mass.py; these chains ARE C-4's chains
PERM_SEED = 11              # distinct from c4_class_mass.py's PERM_SEED (7); a different test

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTE = (
    "Chains, founder, bottleneck and seeds (400-404) are identical to "
    "controls/c4_class_mass.py, which is not edited or re-imported for data. The only "
    "addition is recording, per generation, which classes the student predicts on the "
    "held-out probe pool - the set c4_class_mass.py never logged. Death generation is "
    "the first generation a class is absent from that set and remains absent through "
    "generation 25; still-predicted-at-25 classes are censored at 26. This control has "
    "no argmax arm: C-4's argmax chain was a sanity check on the effective-count metric, "
    "which is not recomputed here."
)


def transmitted_mass(teacher_logits):
    """Effective per-class count in one transmitted soft-target matrix.

    Identical to c4_class_mass.transmitted_mass's soft branch: the column sum of
    the teacher's softmax at the transmission temperature. This control has no
    argmax arm, so there is no mode argument.
    """
    probs = F.softmax(teacher_logits / TAU, dim=-1)
    return probs.sum(0)


def run_chain(X, Xpr, founder, n_classes, chain_seed):
    """One 25-generation soft chain, recording per-generation mass and the set
    of classes the student predicts on the held-out probe pool.

    The random-number sequence this consumes - torch.manual_seed(chain_seed),
    then per generation a randperm draw, an MLP initialization and a training
    loop - is exactly c4_class_mass.run_chain's soft-mode sequence. Nothing here
    that would perturb that sequence (dropout, extra RNG draws) is added; the
    training-pool forward pass c4_class_mass.py also computes is dropped because
    it consumes no RNG and this control does not need it, so dropping it cannot
    change the starved-class reproduction the sanity check depends on.
    """
    torch.manual_seed(chain_seed)
    teacher = founder
    rows = []
    for gen in range(1, GENS + 1):
        sel = torch.randperm(X.shape[0])[:BOTTLENECK]
        with torch.no_grad():
            t_logits = teacher(X[sel])
        mass = transmitted_mass(t_logits)

        student = MLP(X.shape[1], n_classes).to(DEVICE)
        train(student, X[sel], t_logits, 1500, 256, 1e-3, 0.0, True, TAU, DEVICE)

        with torch.no_grad():
            pred_pr = student(Xpr).argmax(-1)

        rows.append({
            "gen": gen,
            "mass": [float(v) for v in mass],
            "n_starved": int((mass < STARVED_CUTOFF).sum()),
            "starved": sorted(int(k) for k in (mass < STARVED_CUTOFF).nonzero().flatten()),
            "probe_classes": sorted(int(k) for k in pred_pr.unique()),
        })
        teacher = student
    return rows


def death_generations(rows, n_classes, gens):
    """d_k per class: first generation absent AND remaining absent through `gens`.

    Equivalently, if L is the last generation (1-indexed) a class is predicted
    on the probe pool, d_k = L + 1, unless L = gens, in which case the class is
    still alive at the end and is censored at gens + 1. A class never predicted
    has L = 0, giving d_k = 1.
    """
    presence = [set(r["probe_classes"]) for r in rows]
    death = np.empty(n_classes, dtype=int)
    for k in range(n_classes):
        last_seen = 0
        for g in range(1, gens + 1):
            if k in presence[g - 1]:
                last_seen = g
        death[k] = gens + 1 if last_seen == gens else last_seen + 1
    return death


def _rankdata_avg(x):
    """Average ranks (ties share the mean rank), matching scipy.stats.rankdata's
    default 'average' method. Used only when scipy is not installed.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def spearman_corr(a, b):
    """Spearman rank correlation between two 1-D arrays."""
    if HAVE_SCIPY:
        rho, _ = _scipy_spearmanr(a, b)
        return float(rho)
    ra, rb = _rankdata_avg(a), _rankdata_avg(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_gate(mass_g1_by_chain, death_by_chain, rng):
    """Gate A: pooled Spearman(gen-1 effective count, death generation).

    Null permutes each chain's death-generation array among its own 50 classes
    (class labels shuffled within chain, per the preregistration), then pools
    and recomputes. One-sided: extreme means a large POSITIVE correlation.

    Returns (None, None, None, reason) if the correlation is undefined - a
    pooled array with zero variance (e.g. every class censored at the same
    generation, which can only happen if death generation barely varies).
    scipy returns nan for a constant input, and the nan then compares as
    False against every permutation, which would otherwise manufacture a
    spurious near-zero p instead of reporting "not evaluable". This cannot
    happen on the real 25-generation, 5-chain run - C-4 already established
    26 starved classes and 37-48 of 50 dead by generation 25, so death
    generation has substantial spread - but it can happen on a heavily
    truncated smoke test, which is exactly where this was caught.
    """
    all_mass = np.concatenate(mass_g1_by_chain)
    all_death = np.concatenate(death_by_chain)
    if np.std(all_mass) == 0 or np.std(all_death) == 0:
        return None, None, None, "pooled mass or death generation has zero variance"

    observed = spearman_corr(all_mass, all_death)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        shuffled = [rng.permutation(d) for d in death_by_chain]
        null[i] = spearman_corr(all_mass, np.concatenate(shuffled))
    p = float((1 + int((null >= observed).sum())) / (1 + N_PERM))
    return float(observed), float(null.mean()), p, None


def starved_timing_gate(death_by_chain, starved_mask_by_chain, rng):
    """Gate B: mean death gen of starved classes minus mean death gen of the rest,
    pooled across chains.

    Null reassigns which classes count as 'starved' within each chain, holding
    each chain's starved-set size fixed (same scheme as c4_class_mass.py's gate
    B). One-sided: extreme means the starved set dies EARLIER, i.e. a large
    negative difference.
    """
    n_classes = len(starved_mask_by_chain[0])

    def diff_stat(masks):
        starved, other = [], []
        for d, m in zip(death_by_chain, masks):
            starved.extend(d[m])
            other.extend(d[~m])
        return float(np.mean(starved) - np.mean(other))

    observed = diff_stat(starved_mask_by_chain)
    sizes = [int(m.sum()) for m in starved_mask_by_chain]

    null = np.empty(N_PERM)
    for i in range(N_PERM):
        fake_masks = []
        for size in sizes:
            mask = np.zeros(n_classes, dtype=bool)
            if size > 0:
                idx = rng.choice(n_classes, size=size, replace=False)
                mask[idx] = True
            fake_masks.append(mask)
        null[i] = diff_stat(fake_masks)
    p = float((1 + int((null <= observed).sum())) / (1 + N_PERM))
    return float(observed), float(null.mean()), p


def check_reproduction(chains, ref_path):
    """Do this run's generation-1 starved sets reproduce logs/c4_class_mass.json?

    Compares by chain seed, not by position, and requires every reference chain
    to be present and matched - a run that produces fewer chains than the
    reference (or different seeds) has not reproduced "4, 7, 5, 7, 3" and fails
    here rather than passing on a partial check.
    """
    with open(ref_path) as handle:
        ref = json.load(handle)
    ref_by_seed = {c["chain_seed"]: sorted(c["starved_g1"]) for c in ref["chains"]}
    got_by_seed = {c["chain_seed"]: sorted(c["starved_g1"]) for c in chains}

    ok = True
    detail = []
    for seed in sorted(set(ref_by_seed) | set(got_by_seed)):
        want = ref_by_seed.get(seed)
        got = got_by_seed.get(seed)
        match = want is not None and got is not None and want == got
        ok = ok and match
        detail.append({
            "chain_seed": seed,
            "expected_starved_g1": want,
            "got_starved_g1": got,
            "match": bool(match),
        })
    return ok, detail


def main():
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-4T"]

    print(f"C-4T preregistered rule, committed {rule['committed']}:")
    print(f"  metric   {rule['metric']}")
    print(f"  gate A   {rule['gates']['A']['condition']}")
    print(f"           fails if {rule['gates']['A']['fails_if']}")
    print(f"  gate B   {rule['gates']['B']['condition']}")
    print(f"           fails if {rule['gates']['B']['fails_if']}")
    print(f"\n{NOTE}\n")
    backend = "scipy.stats.spearmanr" if HAVE_SCIPY else "numpy rank + Pearson (scipy not installed)"
    print(f"Spearman backend: {backend}\n")

    X, y, C = make_pool(seed=TRAIN_SEED, kind="noise")
    Xpr, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    X, y, Xpr = X.to(DEVICE), y.to(DEVICE), Xpr.to(DEVICE)

    torch.manual_seed(TRAIN_SEED)
    founder = MLP(X.shape[1], C).to(DEVICE)
    train(founder, X, y, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        acc = float((founder(X).argmax(-1) == y).float().mean())
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.3f}  "
          f"({time.time()-t0:.1f}s)\n")

    header = (f"{'chain':>5}  {'starved g1':>10}  {f'alive g{GENS}':>9}  "
              f"{'mean death':>10}  {'censored':>8}")
    print(header)
    print("-" * len(header))

    chains, mass_g1_by_chain, death_by_chain, starved_mask_by_chain = [], [], [], []
    for c in range(N_CHAINS):
        seed = CHAIN_SEED0 + c
        rows = run_chain(X, Xpr, founder, C, seed)
        mass_g1 = np.array(rows[0]["mass"])
        starved_g1 = rows[0]["starved"]
        death = death_generations(rows, C, GENS)
        alive_g25 = len(rows[-1]["probe_classes"])
        censored = int((death == GENS + 1).sum())

        mask = np.zeros(C, dtype=bool)
        if starved_g1:
            mask[starved_g1] = True

        mass_g1_by_chain.append(mass_g1)
        death_by_chain.append(death)
        starved_mask_by_chain.append(mask)

        chains.append({
            "chain_seed": seed,
            "starved_g1": starved_g1,
            "n_starved_g1": len(starved_g1),
            "mass_g1": [float(v) for v in mass_g1],
            "death_gen": [int(v) for v in death],
            "alive_probe_g25": rows[-1]["probe_classes"],
        })

        print(f"{seed:>5}  {len(starved_g1):>10}  {alive_g25:>9}  "
              f"{float(death.mean()):>10.2f}  {censored:>8}")

    ref_path = os.path.join(REPO, "logs", "c4_class_mass.json")
    sanity_ok, sanity_detail = check_reproduction(chains, ref_path)

    print(f"\n=== C-4T VERDICT  ({time.time()-t0:.0f}s) ===")
    print(f"sanity: generation-1 starved sets vs {os.path.relpath(ref_path, REPO)} -> "
          f"{'OK' if sanity_ok else 'FAILED'}")
    for d in sanity_detail:
        mark = "match" if d["match"] else "MISMATCH"
        print(f"  seed {d['chain_seed']}: expected {d['expected_starved_g1']}, "
              f"got {d['got_starved_g1']} -> {mark}")

    gate_a_pass = gate_b_pass = None
    gate_a_obs = gate_a_null = gate_a_p = gate_a_why = None
    gate_b_obs = gate_b_null = gate_b_p = None

    if not sanity_ok:
        verdict = "MISCONFIGURED"
        print("\nThe generation-1 starved sets do not reproduce logs/c4_class_mass.json in")
        print("full. This re-run is not the same experiment C-4 ran, so its timing numbers")
        print("cannot be attached to C-4's result. Gates were not read. Fix the chain")
        print("configuration (seeds, chain count, founder, bottleneck) before rerunning.")
    else:
        rng = np.random.default_rng(PERM_SEED)
        gate_a_obs, gate_a_null, gate_a_p, gate_a_why = spearman_gate(
            mass_g1_by_chain, death_by_chain, rng)
        gate_a_pass = None if gate_a_p is None else gate_a_p >= 0.05

        gate_b_obs, gate_b_null, gate_b_p = starved_timing_gate(
            death_by_chain, starved_mask_by_chain, rng)
        gate_b_pass = gate_b_p >= 0.05

        if gate_a_pass is None:
            print(f"\ngate A: not evaluable ({gate_a_why}) -> NOT EVALUABLE")
        else:
            print(f"\ngate A: Spearman rho = {gate_a_obs:.4f} (null mean {gate_a_null:.4f}), "
                  f"p = {gate_a_p:.4f} -> {'PASSED' if gate_a_pass else 'FAILED'}")
        print(f"gate B: starved-minus-other mean death gen = {gate_b_obs:.3f} "
              f"(null mean {gate_b_null:.3f}), p = {gate_b_p:.4f} -> "
              f"{'PASSED' if gate_b_pass else 'FAILED'}")

        if gate_a_pass is None:
            verdict = "INCONCLUSIVE"
            print("Gate A could not be read, so this control returns no verdict on the")
            print("minority-collapse explanation. Nothing further should be concluded")
            print("from this run.")
        elif gate_a_pass and gate_b_pass:
            verdict = "PASSED"
            for line in rule["verdict_if_passed"]:
                print(line)
        else:
            verdict = "FAILED"
            for line in rule["verdict_if_failed"]:
                print(line)

    out = os.path.join(REPO, "logs", "c4_timing.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-4T", "verdict": verdict,
            "sanity_ok": bool(sanity_ok),
            "sanity_detail": sanity_detail,
            "spearman_backend": backend,
            "gate_a_pass": gate_a_pass,
            "gate_a_observed_rho": gate_a_obs,
            "gate_a_null_mean": gate_a_null,
            "gate_a_p": gate_a_p,
            "gate_a_not_evaluable_reason": gate_a_why,
            "gate_b_pass": gate_b_pass,
            "gate_b_observed_diff": gate_b_obs,
            "gate_b_null_mean": gate_b_null,
            "gate_b_p": gate_b_p,
            "starved_cutoff": STARVED_CUTOFF, "n_perm": N_PERM, "perm_seed": PERM_SEED,
            "tau": TAU, "bottleneck": BOTTLENECK, "gens": GENS, "n_chains": N_CHAINS,
            "chain_seed0": CHAIN_SEED0, "train_seed": TRAIN_SEED, "probe_seed": PROBE_SEED,
            "n_classes": C, "founder_acc": acc, "device": DEVICE,
            "reference_log": os.path.relpath(ref_path, REPO),
            "note": NOTE,
            "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
            "chains": chains,
        }, handle, indent=1)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
