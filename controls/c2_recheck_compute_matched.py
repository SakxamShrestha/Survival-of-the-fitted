"""
C-2R: no chaining, re-checked against a compute-matched reference.

C-2 compared 25-generation chains against one-step students trained for 1500
steps, the budget of each individual chain student. C-3 then showed that
one-step agreement is an optimization quantity rather than an information
limit: it rises monotonically with training budget, from 0.142 at 375 steps to
0.306 at 12000. That leaves one serious objection to C-2 unanswered. The chain
spends 25 x 1500 = 37500 optimizer steps in total. A critic can say the chain
does not lose information at all -- it accumulates optimization error, and a
single student given the chain's whole compute budget would look just as far
from the founder. This control tests exactly that, by giving a one-step
student the chain's entire compute budget in a single step and asking whether
the chain still escapes the resulting reference.

The chains are NOT re-run. This reuses the five chains per condition already
logged in logs/c2_no_chaining.json -- the same chains C-2 judged -- and reads
their per-generation trajectories out of that file. Only the reference is new:
a second population of one-step students, trained for 37500 steps instead of
1500, built with c2_no_chaining.py's one-step construction, probe-pool
handling and escape-generation logic (falling below the reference's 2.5th
percentile) left UNCHANGED so the two controls are directly comparable.

    python3 controls/c2_recheck_compute_matched.py

Decision rules are preregistered in gates/thresholds.json under "C-2R",
committed before this ran. They are read from that file at runtime rather than
restated here, so the thresholds cannot drift from the committed record.

Two choices the preregistration left to this script:

  - Which pool the gates read. Gate A and gate B name the held-out probe pool
    explicitly, as C-2 and C-3 do; both pools are still reported.
  - Which seeds the students use. The same 20 seeds (1000-1019) are used for
    both the 1500-step sanity arm and the 37500-step gate arm, and for both
    B = 128 and B = 4096, the way c3_one_step.py pairs its budget sweep. Only
    the step count and bottleneck differ between arms; initialization and data
    order do not.

A 37500-step student costs 25 times a standard one, and this control trains
80 of them (2 bottlenecks x 2 budgets x 20 students), so it is expensive
enough that the budget sweep in c3_one_step.py -- 6 budgets x 20 students, all
at B = 4096 -- is the closer cost comparison, not c2_no_chaining.py itself.
"""

import json
import os
import sys
import time

import numpy as np
import torch

# Run as `python3 controls/c2_recheck_compute_matched.py` from the repository
# root: Python puts this file's directory on sys.path, not the root, so add
# the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

TAU = 1.0
GENS = 25                     # chain length judged; matches c2_no_chaining.py exactly
BOTTLENECKS = (128, 4096)     # C-2's two conditions; gate A reads B=4096, gate B reads B=128
SANITY_BUDGET = 1500          # reproduces C-2's own one-step reference budget
GATE_BUDGET = 37500           # 25 gens x 1500 steps/gen = the chain's total optimizer-step budget
N_STUDENTS = 20                # C-3's n, not C-2's 50: a 37500-step student costs 25x a standard one
DEVICE = "cpu"                 # measured faster than MPS for this MLP (CLAUDE.md device table)
TRAIN_SEED = 0                  # identical to C-2, C-3, C-6
PROBE_SEED = 9_000               # identical to C-2, C-3, C-6
STUDENT_SEED0 = 1000               # identical to c2_no_chaining.py and c3_one_step.py: seeds
                                     # 1000-1019 pair BOTH budgets and BOTH bottlenecks, so only
                                     # steps and B differ between any two runs of the same seed
C3_GATE_BUDGET_FALLBACK_MEAN = 0.306  # C-3's 12000-step probe mean at B=4096; used only if
                                        # logs/c3_one_step.json cannot be read (see load_c3_reference)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

assert GATE_BUDGET == GENS * SANITY_BUDGET, (
    "37500 is derived, not chosen: it is 25 generations times the 1500 steps each "
    "chain student trains for. If this assertion fails, GENS or SANITY_BUDGET drifted."
)

POOL_NOTE = (
    "Gates are evaluated on the held-out probe pool, which the C-2R metric names "
    "explicitly for both gate A and gate B, the same choice C-2 and C-3 made. Both "
    "pools are reported."
)

SEED_NOTE = (
    "The same 20 student seeds (1000-1019) are used for the 1500-step sanity arm and "
    "the 37500-step gate arm, and for both B = 128 and B = 4096, so initialization and "
    "data order are identical everywhere and only the step count and bottleneck differ. "
    "These are the same seeds c2_no_chaining.py and c3_one_step.py used, so the "
    "1500-step arm at each B is an exact re-run of 20 of C-2's own reference students."
)


def measure(model, Xtr, Xpr, base_tr, base_pr):
    """Functional agreement and alphabet size, on both pools.

    Copied unchanged from controls/c2_no_chaining.py.
    """
    with torch.no_grad():
        ptr = model(Xtr).argmax(-1)
        ppr = model(Xpr).argmax(-1)
    return {
        "train": {"agree_gen0": float((ptr == base_tr).float().mean()),
                  "n_classes_used": int(ptr.unique().numel())},
        "probe": {"agree_gen0": float((ppr == base_pr).float().mean()),
                  "n_classes_used": int(ppr.unique().numel())},
    }


def one_step(Xtr, t_logits, bottleneck, n_classes, seed, steps):
    """A student distilled once, directly from generation 0, for `steps` steps.

    Structurally identical to c2_no_chaining.one_step: same seeding, same
    selection of `bottleneck` inputs, same student construction, same train()
    call. `bottleneck` and `steps` are both parameters here (c2_no_chaining.py
    fixes steps at 1500; c3_one_step.py fixes bottleneck at 4096) because this
    control needs the full B x budget grid at once.
    """
    torch.manual_seed(seed)
    sel = torch.randperm(Xtr.shape[0])[:bottleneck]
    student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    train(student, Xtr[sel], t_logits[sel], steps, 256, 1e-3, 0.0, True, TAU, DEVICE)
    return student


def first_escape(values, floor):
    """Lowest generation index (1-based) whose value falls below the floor.

    Copied unchanged from controls/c2_no_chaining.py.
    """
    for i, v in enumerate(values, start=1):
        if v < floor:
            return i
    return None


def load_c2_chains(c2_log):
    """Pull the already-logged chain trajectories out of c2_no_chaining.json.

    The chains are not re-run; the C-2R entry says so explicitly ("this judges
    the same chains C-2 judged, and only the reference is new"). Returns
    (series, problems). `series[B]` holds the 25-generation mean_agree_by_gen
    and mean_classes_by_gen lists for that bottleneck when both are present
    and the right length; `problems` lists exactly what is missing or
    malformed, so a caller can report it and stop rather than guess.
    """
    series, problems = {}, []
    conditions = c2_log.get("conditions", {})
    for B in BOTTLENECKS:
        cond = conditions.get(str(B))
        if cond is None:
            problems.append(f'conditions["{B}"] is missing from logs/c2_no_chaining.json')
            continue
        chains = cond.get("chains", {})
        agree = chains.get("mean_agree_by_gen")
        classes = chains.get("mean_classes_by_gen")
        if not isinstance(agree, list) or len(agree) != GENS:
            got = "missing" if agree is None else f"length {len(agree)}, expected {GENS}"
            problems.append(f'conditions["{B}"].chains.mean_agree_by_gen is {got}')
        if not isinstance(classes, list) or len(classes) != GENS:
            got = "missing" if classes is None else f"length {len(classes)}, expected {GENS}"
            problems.append(f'conditions["{B}"].chains.mean_classes_by_gen is {got}')
        series[B] = {
            "agree": agree, "classes": classes,
            "c2_gate_a_escape": cond.get("gate_a_escape"),
            "c2_gate_b_escape": cond.get("gate_b_escape"),
            "c2_reference_agree_ci95": cond.get("reference", {}).get("agree_ci95"),
        }
    return series, problems


def load_c3_reference():
    """C-3's 12000-step probe-pool mean at B=4096, for the secondary check only.

    Read from logs/c3_one_step.json when available so the number cannot drift
    from that control's own record. Falls back to the value named in the C-2R
    sanity_check text (0.306) when the file is unavailable -- e.g. under the
    smoke test, which does not stage logs/c3_one_step.json into the scratch
    REPO.
    """
    path = os.path.join(REPO, "logs", "c3_one_step.json")
    try:
        with open(path) as handle:
            c3 = json.load(handle)
        return float(c3["conditions"]["12000"]["probe_agree_mean"]), "logs/c3_one_step.json"
    except (FileNotFoundError, KeyError, ValueError):
        return (C3_GATE_BUDGET_FALLBACK_MEAN,
                "gates/thresholds.json C-2R sanity_check (logs/c3_one_step.json unavailable)")


def main():
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-2R"]

    print(f"C-2R preregistered rule, committed {rule['committed']}:")
    print(f"  question    {rule['question']}")
    print(f"  reference   {rule['reference_distribution']}")
    for key in ("sanity", "A", "B"):
        gate = rule["gates"][key]
        print(f"  gate {key:<7} {gate['condition']}")
        print(f"           fails if {gate['fails_if']}")
    print(f"\n{POOL_NOTE}\n{SEED_NOTE}\n")
    print("secondary check (reported, not gated):")
    for line in rule["sanity_check"]:
        print(f"  {line}")
    print()

    # ---- read the already-logged chains before spending any compute
    c2_log_path = os.path.join(REPO, "logs", "c2_no_chaining.json")
    with open(c2_log_path) as handle:
        c2_log = json.load(handle)
    chain_series, problems = load_c2_chains(c2_log)
    if problems:
        print("STOP: required chain data is not available in logs/c2_no_chaining.json:")
        for p in problems:
            print(f"  - {p}")
        print(f"  top-level keys present: {sorted(c2_log.keys())}")
        print(f"  conditions present: {sorted(c2_log.get('conditions', {}).keys())}")
        print("No training was run and no log was written.")
        sys.exit(1)

    # ---- founder and pools, identical construction to C-2, C-3, C-6
    Xtr, ytr, C = make_pool(seed=TRAIN_SEED, kind="noise")
    Xpr, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    Xtr, ytr, Xpr = Xtr.to(DEVICE), ytr.to(DEVICE), Xpr.to(DEVICE)

    mismatches = []
    if c2_log.get("train_seed") != TRAIN_SEED:
        mismatches.append(f"c2 train_seed={c2_log.get('train_seed')} != {TRAIN_SEED}")
    if c2_log.get("probe_seed") != PROBE_SEED:
        mismatches.append(f"c2 probe_seed={c2_log.get('probe_seed')} != {PROBE_SEED}")
    if c2_log.get("n_classes") != C:
        mismatches.append(f"c2 n_classes={c2_log.get('n_classes')} != {C}")
    if c2_log.get("gens") != GENS:
        mismatches.append(f"c2 gens={c2_log.get('gens')} != {GENS}")
    if mismatches:
        print("STOP: logs/c2_no_chaining.json was not produced with this control's "
              "founder/pool/chain-length configuration:")
        for m in mismatches:
            print(f"  - {m}")
        print("The chains read from that file would not be comparable to the reference "
              "built here. No training was run and no log was written.")
        sys.exit(1)

    torch.manual_seed(TRAIN_SEED)
    founder = MLP(Xtr.shape[1], C).to(DEVICE)
    train(founder, Xtr, ytr, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        base_tr = founder(Xtr).argmax(-1)
        base_pr = founder(Xpr).argmax(-1)
        founder_logits = founder(Xtr)
        acc = float((base_tr == ytr).float().mean())
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.3f}  "
          f"({time.time()-t0:.1f}s)\n")

    # ---- two reference arms (1500-step sanity, 37500-step gate) at each B
    header = (f"{'B':>5}  {'arm':>7}  {'steps':>6}  {'n':>3}  "
              f"{'agree mean':>10}  {'agree sd':>9}  {'agree p2.5':>10}  "
              f"{'cls mean':>8}  {'cls sd':>7}  {'cls p2.5':>8}  {'elapsed':>8}")
    print(header)
    print("-" * len(header))

    arm_stats = {}
    for B in BOTTLENECKS:
        arm_stats[B] = {}
        for arm_name, steps in (("sanity", SANITY_BUDGET), ("gate", GATE_BUDGET)):
            runs = [measure(one_step(Xtr, founder_logits, B, C, STUDENT_SEED0 + r, steps),
                             Xtr, Xpr, base_tr, base_pr)
                    for r in range(N_STUDENTS)]

            pr_agree = np.array([m["probe"]["agree_gen0"] for m in runs])
            pr_cls = np.array([m["probe"]["n_classes_used"] for m in runs])
            tr_agree = np.array([m["train"]["agree_gen0"] for m in runs])
            tr_cls = np.array([m["train"]["n_classes_used"] for m in runs])
            lo_a, hi_a = np.percentile(pr_agree, [2.5, 97.5])
            lo_c, hi_c = np.percentile(pr_cls, [2.5, 97.5])

            stats = {
                "steps": steps, "n": N_STUDENTS,
                "probe_agree_mean": float(pr_agree.mean()),
                "probe_agree_sd": float(pr_agree.std(ddof=1)),
                "probe_agree_ci95": [float(lo_a), float(hi_a)],
                "probe_classes_mean": float(pr_cls.mean()),
                "probe_classes_sd": float(pr_cls.std(ddof=1)),
                "probe_classes_ci95": [float(lo_c), float(hi_c)],
                "train_agree_mean": float(tr_agree.mean()),
                "train_classes_mean": float(tr_cls.mean()),
                "probe_agree_runs": [float(v) for v in pr_agree],
                "probe_classes_runs": [int(v) for v in pr_cls],
            }
            arm_stats[B][arm_name] = stats
            print(f"{B:>5}  {arm_name:>7}  {steps:>6}  {N_STUDENTS:>3}  "
                  f"{stats['probe_agree_mean']:>10.3f}  {stats['probe_agree_sd']:>9.4f}  "
                  f"{lo_a:>10.3f}  {stats['probe_classes_mean']:>8.1f}  "
                  f"{stats['probe_classes_sd']:>7.2f}  {lo_c:>8.1f}  "
                  f"{time.time()-t0:>7.0f}s")

    # ---- sanity gate: does the 1500-step arm reproduce C-2's own means?
    ref_ci_4096 = tuple(c2_log["conditions"]["4096"]["reference"]["agree_ci95"])
    ref_ci_128 = tuple(c2_log["conditions"]["128"]["reference"]["agree_ci95"])
    sanity_4096 = arm_stats[4096]["sanity"]["probe_agree_mean"]
    sanity_128 = arm_stats[128]["sanity"]["probe_agree_mean"]
    sanity_ok_4096 = ref_ci_4096[0] <= sanity_4096 <= ref_ci_4096[1]
    sanity_ok_128 = ref_ci_128[0] <= sanity_128 <= ref_ci_128[1]
    sanity_ok = sanity_ok_4096 and sanity_ok_128

    print(f"\nsanity: 1500-step arm at B=4096 mean {sanity_4096:.3f} vs C-2 reference "
          f"[{ref_ci_4096[0]:.3f}, {ref_ci_4096[1]:.3f}] (nominally [0.220, 0.243]) -> "
          f"{'OK' if sanity_ok_4096 else 'FAILED'}")
    print(f"sanity: 1500-step arm at B=128  mean {sanity_128:.3f} vs C-2 reference "
          f"[{ref_ci_128[0]:.3f}, {ref_ci_128[1]:.3f}] (nominally [0.047, 0.064]) -> "
          f"{'OK' if sanity_ok_128 else 'FAILED'}")

    # ---- gate A (B=4096, function) and gate B (B=128, alphabet), against the
    # 37500-step reference's 2.5th percentile, C-2's own escape criterion
    floor_a = arm_stats[4096]["gate"]["probe_agree_ci95"][0]
    floor_b = arm_stats[128]["gate"]["probe_classes_ci95"][0]
    chain_agree_4096 = chain_series[4096]["agree"]
    chain_classes_128 = chain_series[128]["classes"]
    esc_a = first_escape(chain_agree_4096, floor_a)
    esc_b = first_escape(chain_classes_128, floor_b)

    # supplementary, unofficial escape checks at the other metric/B pairing,
    # reported for context only -- not part of the gated verdict
    floor_a_128 = arm_stats[128]["gate"]["probe_agree_ci95"][0]
    floor_b_4096 = arm_stats[4096]["gate"]["probe_classes_ci95"][0]
    esc_a_128 = first_escape(chain_series[128]["agree"], floor_a_128)
    esc_b_4096 = first_escape(chain_series[4096]["classes"], floor_b_4096)

    print(f"\n=== C-2R VERDICT  ({time.time()-t0:.0f}s) ===")
    print(f"sanity: {'OK' if sanity_ok else 'FAILED'}")

    if not sanity_ok:
        verdict = "MISCONFIGURED"
        print(rule["gates"]["sanity"]["meaning_if_failed"])
    else:
        gate_a_escaped = esc_a is not None
        gate_b_escaped = esc_b is not None
        print(f"gate A (B=4096, function): chain agreement escapes the 37500-step "
              f"reference's 2.5th percentile ({floor_a:.3f}) at generation "
              f"{esc_a if gate_a_escaped else 'NEVER'}")
        if not gate_a_escaped:
            print(f"  {rule['gates']['A']['meaning_if_failed']}")
        print(f"gate B (B=128, alphabet): chain alphabet escapes the 37500-step "
              f"reference's 2.5th percentile ({floor_b:.1f}) at generation "
              f"{esc_b if gate_b_escaped else 'NEVER'}")
        if not gate_b_escaped:
            print(f"  {rule['gates']['B']['meaning_if_failed']}")

        print(f"\n(supplementary, not gated: agreement at B=128 escapes at generation "
              f"{esc_a_128 if esc_a_128 else 'NEVER'}; alphabet at B=4096 escapes at "
              f"generation {esc_b_4096 if esc_b_4096 else 'NEVER'})")

        if gate_a_escaped and gate_b_escaped:
            verdict = "PASSED"
            print()
            for line in rule["verdict_if_passed"]:
                print(line)
        else:
            verdict = "FAILED"
            print()
            for line in rule["verdict_if_failed"]:
                print(line)

    # ---- secondary check, reported but not gated
    c3_mean, c3_source = load_c3_reference()
    gate_4096_mean = arm_stats[4096]["gate"]["probe_agree_mean"]
    secondary_exceeds = gate_4096_mean > c3_mean
    print(f"\nsecondary (not gated): 37500-step reference mean at B=4096 "
          f"{gate_4096_mean:.3f} vs C-3's 12000-step mean {c3_mean:.3f} ({c3_source}) -> "
          f"{'exceeds' if secondary_exceeds else 'does NOT exceed'}")

    out = os.path.join(REPO, "logs", "c2_recheck_compute_matched.json")
    with open(out, "w") as handle:
        json.dump({
            "control": "C-2R", "verdict": verdict,
            "sanity_ok": bool(sanity_ok),
            "sanity_ok_4096": bool(sanity_ok_4096), "sanity_ok_128": bool(sanity_ok_128),
            "sanity_arm_steps": SANITY_BUDGET,
            "sanity_arm_mean_4096": sanity_4096, "sanity_arm_mean_128": sanity_128,
            "c2_reference_ci95_4096": list(ref_ci_4096),
            "c2_reference_ci95_128": list(ref_ci_128),
            "gate_budget": GATE_BUDGET,
            "gate_a_bottleneck": 4096, "gate_a_floor": float(floor_a),
            "gate_a_escape": esc_a,
            "gate_b_bottleneck": 128, "gate_b_floor": float(floor_b),
            "gate_b_escape": esc_b,
            "supplementary_agree_escape_b128": esc_a_128,
            "supplementary_classes_escape_b4096": esc_b_4096,
            "c2_original_gate_a_escape_4096": chain_series[4096]["c2_gate_a_escape"],
            "c2_original_gate_b_escape_128": chain_series[128]["c2_gate_b_escape"],
            "secondary_check": {
                "description": rule["sanity_check"],
                "gate_arm_mean_b4096": gate_4096_mean,
                "c3_12000step_mean_b4096": c3_mean,
                "c3_source": c3_source,
                "exceeds": bool(secondary_exceeds),
            },
            "n_students": N_STUDENTS, "student_seed0": STUDENT_SEED0,
            "train_seed": TRAIN_SEED, "probe_seed": PROBE_SEED,
            "n_classes": C, "founder_acc": acc, "device": DEVICE, "tau": TAU, "gens": GENS,
            "gated_on_pool": "probe", "pool_note": POOL_NOTE, "seed_note": SEED_NOTE,
            "c2_log_source": os.path.relpath(c2_log_path, REPO),
            "c2_log_note": "chains are read from this file, not re-run; only the "
                           "reference (both arms, both bottlenecks) is computed here",
            "torch_version": torch.__version__,
            "elapsed_sec": round(time.time() - t0, 1),
            "conditions": {
                str(B): {
                    "bottleneck": B,
                    "sanity": arm_stats[B]["sanity"],
                    "gate": arm_stats[B]["gate"],
                    "chain_mean_agree_by_gen": chain_series[B]["agree"],
                    "chain_mean_classes_by_gen": chain_series[B]["classes"],
                }
                for B in BOTTLENECKS
            },
        }, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
