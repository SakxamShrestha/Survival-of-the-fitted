"""
C-4: does soft-logit transmission starve classes?

At B = 128 over 50 classes the transmitted set carries about 2.56 examples per
class. Fang, He, Long and Su (PNAS 2021) characterize minority collapse
analytically in exactly that regime, which makes it a competing explanation for
this project's best result — alphabet contraction under a tight bottleneck —
and it is a competing explanation with a PNAS citation attached.

Their analysis is written in per-class example counts. A hard-label channel has
those; a soft-logit channel does not, because every input contributes some
probability to every class. The corresponding quantity is the effective count:

    m_k(g) = sum over the B transmitted inputs of P(class k | input)

where P is the teacher's softmax at the transmission temperature. If m_k stays
above the unit Fang et al.'s theory is written in, the soft channel is not in
their regime and the mechanism is evaded. If it does not, the starvation is real
and the question becomes whether it predicts anything.

RESEARCH-PLAN section 4.1 allows C-4 to be answered two ways: construct a
matched-imbalance control, or establish that soft-logit transmission places mass
on all classes. This script judges the second route only. It can close the
question or declare the first route mandatory. It cannot substitute for it, and
a passing verdict here is not a constructed control.

    python3 controls/c4_class_mass.py | tee logs/c4_class_mass.txt

Decision rules are preregistered in gates/thresholds.json under "C-4", committed
before this ran. They are read from that file at runtime rather than restated
here, so the thresholds cannot drift from the committed record.

Three choices the preregistration did not fix:

  - Which pool defines a dead class. The probe pool, following C-2, C-3 and C-6:
    C-6 established that training-pool alphabet counts overstate what survives.
    Both pools are reported.
  - How many chains. Five, which is what gate B's permutation test pools over.
    This is a screening control, not an estimate: it decides whether the
    expensive constructed control is required, and five chains is enough to
    decide that.
  - Chain seeds. 400-404, chosen not to collide with seed_stability.py (100-119,
    200-219) or sampler_arm.py (300-304), so the chains here are independent of
    every chain already on record.

The argmax arm at the end is the sanity check, not a result. A hard channel
either selects a class or does not, so it must show classes at an effective
count of exactly zero. If it does not, the measurement is wrong.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Run as `python3 controls/c4_class_mass.py` from the repository root: Python
# puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

TAU = 1.0
BOTTLENECK = 128            # the condition the contraction result is stated at
GENS = 25
N_CHAINS = 5
STARVED_CUTOFF = 1.0        # one effective example; the unit Fang et al. use
N_PERM = 10_000
DEVICE = "cpu"              # measured faster than MPS for this MLP (1.61 s vs 2.47 s)
TRAIN_SEED = 0
PROBE_SEED = 9_000          # identical to c2_no_chaining.py, c3_one_step.py, c6_probe_pool.py
CHAIN_SEED0 = 400
PERM_SEED = 7

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POOL_NOTE = (
    "A class counts as dead if generation 25 never predicts it on the held-out probe "
    "pool. C-6 showed training-pool alphabet counts overstate what survives (6 of 50 "
    "against 9 of 50 at this bottleneck), so the probe pool is the one gate B reads. "
    "Both are reported."
)

SEED_NOTE = (
    "Chain seeds 400-404 do not collide with seed_stability.py (100-119, 200-219) or "
    "sampler_arm.py (300-304), so these five chains are independent of every chain "
    "already on record. The founder is seed 0, the same founder C-2, C-3 and C-6 used."
)


def transmitted_mass(teacher_logits, mode):
    """Effective per-class count in one transmitted target matrix.

    For the soft channel this is the column sum of the teacher's softmax. For the
    argmax channel the transmitted target is a hard label, so the same expression
    over one-hot rows reduces to an integer count of selected examples — which is
    what makes the two arms directly comparable rather than two different metrics.
    """
    if mode == "soft":
        probs = F.softmax(teacher_logits / TAU, dim=-1)
    else:
        hard = teacher_logits.argmax(-1)
        probs = F.one_hot(hard, teacher_logits.shape[1]).float()
    return probs.sum(0)


def run_chain(X, Xpr, founder, n_classes, chain_seed, mode):
    """One 25-generation chain, recording the transmitted mass at every step."""
    torch.manual_seed(chain_seed)
    teacher = founder
    rows = []
    for gen in range(1, GENS + 1):
        sel = torch.randperm(X.shape[0])[:BOTTLENECK]
        with torch.no_grad():
            t_logits = teacher(X[sel])
        mass = transmitted_mass(t_logits, mode)

        student = MLP(X.shape[1], n_classes).to(DEVICE)
        if mode == "soft":
            train(student, X[sel], t_logits, 1500, 256, 1e-3, 0.0, True, TAU, DEVICE)
        else:
            train(student, X[sel], t_logits.argmax(-1), 1500, 256, 1e-3, 0.0,
                  False, TAU, DEVICE)

        with torch.no_grad():
            pred_tr = student(X).argmax(-1)
            pred_pr = student(Xpr).argmax(-1)

        rows.append({
            "gen": gen,
            "mass": [float(v) for v in mass],
            "min_mass": float(mass.min()),
            "max_mass": float(mass.max()),
            "n_starved": int((mass < STARVED_CUTOFF).sum()),
            "starved": sorted(int(k) for k in (mass < STARVED_CUTOFF).nonzero().flatten()),
            "n_classes_used_train": int(pred_tr.unique().numel()),
            "n_classes_used_probe": int(pred_pr.unique().numel()),
        })
        teacher = student

    alive_probe = sorted(int(k) for k in pred_pr.unique())
    alive_train = sorted(int(k) for k in pred_tr.unique())
    return rows, alive_probe, alive_train


def overlap_coefficient(a, b):
    """|a and b| / min(|a|, |b|). Undefined, and returned as None, on an empty set.

    Jaccard is not used here: the starved set and the dead set have systematically
    different sizes, and RESEARCH-PLAN section 4.3 records size dependence as a
    known defect of Jaccard in this project.
    """
    if not a or not b:
        return None
    return len(set(a) & set(b)) / min(len(a), len(b))


def permutation_p(starved_sets, dead_sets, n_classes, rng):
    """Does starvation at generation 1 predict death at generation 25?

    The null reassigns WHICH classes are starved, holding each chain's starved-set
    size fixed, so it tests the identity of the starved classes rather than how
    many there are. Returns the observed pooled overlap, the null distribution's
    mean, and a one-sided p.
    """
    pairs = [(s, d) for s, d in zip(starved_sets, dead_sets)
             if overlap_coefficient(s, d) is not None]
    if not pairs:
        # Name which side is empty. The two cases mean opposite things: no starved
        # classes is gate A passing, no dead classes is a chain that never
        # contracted, and neither is evidence about the other.
        if not any(starved_sets):
            reason = "no chain had a starved class at generation 1"
        elif not any(dead_sets):
            reason = "no chain lost a class by generation 25"
        else:
            reason = "no chain had both a starved and a dead set"
        return None, None, None, reason

    observed = float(np.mean([overlap_coefficient(s, d) for s, d in pairs]))
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        vals = []
        for s, d in pairs:
            fake = rng.choice(n_classes, size=len(s), replace=False)
            vals.append(overlap_coefficient(list(fake), d))
        null[i] = np.mean(vals)
    p = float((1 + int((null >= observed).sum())) / (1 + N_PERM))
    return observed, float(null.mean()), p, None


def main():
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-4"]

    print(f"C-4 preregistered rule, committed {rule['committed']}:")
    print(f"  metric   {rule['metric']}")
    print(f"  gate A   {rule['gates']['A']['condition']}")
    print(f"           fails if {rule['gates']['A']['fails_if']}")
    print(f"  gate B   {rule['gates']['B']['condition']}")
    print(f"           fails if {rule['gates']['B']['fails_if']}")
    print(f"\n{POOL_NOTE}\n{SEED_NOTE}\n")

    X, y, C = make_pool(seed=TRAIN_SEED, kind="noise")
    Xpr, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    X, y, Xpr = X.to(DEVICE), y.to(DEVICE), Xpr.to(DEVICE)

    torch.manual_seed(TRAIN_SEED)
    founder = MLP(X.shape[1], C).to(DEVICE)
    train(founder, X, y, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        acc = float((founder(X).argmax(-1) == y).float().mean())
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.3f}  "
          f"({time.time()-t0:.1f}s)")
    print(f"uniform transmission would give {BOTTLENECK}/{C} = {BOTTLENECK/C:.2f} "
          f"effective examples per class\n")

    header = (f"{'chain':>5}  {'starved g1':>10}  {'min mass g1':>11}  "
              f"{'min mass g25':>12}  {'alphabet g25':>12}")
    print(header)
    print("-" * len(header))

    chains, starved_sets, dead_sets = [], [], []
    for c in range(N_CHAINS):
        rows, alive_probe, alive_train = run_chain(
            X, Xpr, founder, C, CHAIN_SEED0 + c, "soft")
        dead_probe = sorted(set(range(C)) - set(alive_probe))
        starved_g1 = rows[0]["starved"]
        starved_sets.append(starved_g1)
        dead_sets.append(dead_probe)
        chains.append({
            "chain_seed": CHAIN_SEED0 + c, "rows": rows,
            "alive_probe_g25": alive_probe, "alive_train_g25": alive_train,
            "dead_probe_g25": dead_probe,
            "starved_g1": starved_g1,
            "overlap_starved_dead": overlap_coefficient(starved_g1, dead_probe),
        })
        print(f"{CHAIN_SEED0+c:>5}  {len(starved_g1):>10}  {rows[0]['min_mass']:>11.4f}  "
              f"{rows[-1]['min_mass']:>12.4f}  {len(alive_probe):>12}")

    # Sanity: the hard channel must starve classes outright, or the metric is wrong.
    print(f"\nsanity arm: argmax transmission, one chain  ({time.time()-t0:.0f}s so far)")
    arg_rows, arg_alive, _ = run_chain(X, Xpr, founder, C, CHAIN_SEED0, "argmax")
    arg_starved_g1 = arg_rows[0]["n_starved"]
    arg_zero_g1 = sum(1 for v in arg_rows[0]["mass"] if v == 0.0)
    sanity_ok = arg_zero_g1 > 0
    print(f"  classes at exactly zero effective count at g1: {arg_zero_g1}")
    print(f"  classes below {STARVED_CUTOFF} at g1: {arg_starved_g1}")

    total_starved_g1 = sum(len(s) for s in starved_sets)
    gate_a_pass = total_starved_g1 == 0

    rng = np.random.default_rng(PERM_SEED)
    observed, null_mean, p, why = permutation_p(starved_sets, dead_sets, C, rng)
    if p is None:
        # Not evaluable is not the same as passed. A gate that could not be read
        # decides nothing, and recording it as a pass would be exactly the
        # after-the-fact softening the preregistration exists to prevent.
        gate_b_pass, gate_b_note = None, f"not evaluable: {why}"
    else:
        gate_b_pass = p >= 0.05
        gate_b_note = f"pooled overlap {observed:.3f} vs null mean {null_mean:.3f}, p = {p:.4f}"

    print(f"\n=== C-4 VERDICT  ({time.time()-t0:.0f}s) ===")
    print(f"sanity: argmax arm shows {arg_zero_g1} classes at exactly zero -> "
          f"{'OK' if sanity_ok else 'FAILED'}")

    if not sanity_ok:
        verdict = "MISCONFIGURED"
        print("A hard channel cannot feed every class at B = 128 over 50 classes, so a")
        print("count of zero starved classes here means the measurement is wrong. Fix it")
        print("before reading anything in the soft arm.")
    else:
        gate_b_label = {True: "PASSED", False: "FAILED", None: "NOT EVALUABLE"}[gate_b_pass]
        print(f"gate A: {total_starved_g1} starved classes at generation 1 across "
              f"{N_CHAINS} chains -> {'PASSED' if gate_a_pass else 'FAILED'}")
        print(f"gate B: {gate_b_note} -> {gate_b_label}")
        if gate_b_pass is None:
            verdict = "INCONCLUSIVE"
            print("Gate B could not be read, so this control returns no verdict on the")
            print("minority-collapse explanation. Gate A's result stands on its own and")
            print("nothing further should be concluded from this run.")
        elif gate_a_pass and gate_b_pass:
            verdict = "PASSED"
            for line in rule["verdict_if_passed"]:
                print(line)
        else:
            verdict = "FAILED"
            for line in rule["verdict_if_failed"]:
                print(line)

    out = os.path.join(REPO, "logs", "c4_class_mass.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-4", "verdict": verdict,
            "route": "evasion check only; not the constructed matched-imbalance control",
            "sanity_ok": bool(sanity_ok),
            "argmax_zero_classes_g1": arg_zero_g1,
            "argmax_starved_g1": arg_starved_g1,
            "gate_a_pass": bool(gate_a_pass),
            "gate_a_starved_total_g1": total_starved_g1,
            "gate_b_pass": bool(gate_b_pass),
            "gate_b_observed_overlap": observed,
            "gate_b_null_mean": null_mean,
            "gate_b_p": p,
            "starved_cutoff": STARVED_CUTOFF, "n_perm": N_PERM, "perm_seed": PERM_SEED,
            "tau": TAU, "bottleneck": BOTTLENECK, "gens": GENS, "n_chains": N_CHAINS,
            "chain_seed0": CHAIN_SEED0, "train_seed": TRAIN_SEED, "probe_seed": PROBE_SEED,
            "n_classes": C, "founder_acc": acc, "device": DEVICE,
            "gated_on_pool": "probe", "pool_note": POOL_NOTE, "seed_note": SEED_NOTE,
            "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
            "chains": chains,
            "argmax_rows": arg_rows,
            "argmax_alive_probe_g25": arg_alive,
        }, handle, indent=1)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
