"""
C-10: does iteration destroy anything on natural data that one step does not?

GATED. The decision rule is gates/thresholds.json entry "C-10", committed before
this file wrote any log, and ci/check_gate_ordering.py verifies that ordering
from the git history rather than from anything asserted here.

This is C-2 and C-2R transported to a real dataset. C-9 ran fifteen chains on
sklearn's load_digits and is descriptive by construction: every one of its
chains is a chain, so nothing in that arm is a one-step student and there is no
reference against which a chain could have failed to escape. Without one, C-9's
fall to 6.600 surviving classes at B = 37 cannot be told apart from the claim
that a 37-sample bottleneck over ten classes is lossy, which is a statement
about information capacity and not about iteration. On synthetic data C-2 is
exactly what excludes that reading.

Three gates, all on the held-out probe pool:

    A   agree_gen0 against a STEP-MATCHED reference (1500 steps, the budget each
        chain student gets)
    B   agree_gen0 against a COMPUTE-MATCHED reference (37500 steps = 25 x 1500,
        the chain's entire compute budget spent in one step)
    C   accuracy against the TRUE digit labels, read against both references

Gate C is why a real dataset is worth spending. On the synthetic arm's random
generation-0 labels, probe-pool true-label accuracy is chance by construction
and carries no information; here it is the difference between a chain that
drifts from its founder and a chain that loses the task.

A fourth gate on n_classes_used was drafted and deliberately dropped; see
gate_D_dropped in the rule. The statistic is still recorded at every generation
and reported as description, and no verdict may be attached to it later.

Both bottleneck conditions are gated and neither is privileged. The
correspondence between the synthetic headline condition and this pool is
genuinely unresolved -- 128 is 3.1% of 4096 and 10.7% of 1198 -- and C-9 shows
the two readings disagree. Choosing one now, with those numbers in hand, would
be choosing after the fact. A disagreement between the two conditions is the
outcome SPLIT BY CORRESPONDENCE and is reported as two results, never averaged.

Pools, founder, scaling and metric definitions are C-9's, imported from its
runner rather than reimplemented, which is what makes the comparison like for
like.

    /usr/bin/python3 controls/c10_digits_iteration.py

Target interpreter is /usr/bin/python3, which carries torch 2.8.0.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import sklearn
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, train
from controls.c9_digits_descriptive import (
    BATCH, DEVICE, GENS, LR, STEPS_GEN, TAU, WD,
    Tee, accuracy, build_founder, conditions, load_pools, log_paths, metrics,
    reference,
)

CONTROL = "C-10"

# Chains and references. n = 20 is C-3's budget-sweep n, reused by C-2R because a
# 37500-step student costs 25 standard ones, and it is C-1's and C-8b's gated n.
# C-9's n = 5 is a descriptive arm's n and is not carried here.
N_CHAINS = 20
N_REF = 20
STEPS_COMPUTE = GENS * STEPS_GEN        # 37500; C-2R's derivation, not a chosen number

# Seed namespaces, none of which collide with anything on record: 0-19, 100-119,
# 200-219 and 300-319 (seed stability), 400-404 (C-4), 600-619 (C-1, C-8, C-8b),
# 900-904 (C-9's own chains, excluded from every gate here and not re-run).
CHAIN_SEED_OFFSET = 910
REF_STEP_SEED_OFFSET = 9_300
REF_COMPUTE_SEED_OFFSET = 9_400

ESCAPE_PCT = 2.5                        # the one-sided form of the repository's 95% intervals
CENTRAL = (2.5, 97.5)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO, "logs", "c10_digits_iteration.json")
C9_JSON = os.path.join(REPO, "logs", "c9_digits_descriptive.json")

NOT_EVALUABLE = "NOT EVALUABLE"
PASSED, FAILED = "PASSED", "FAILED"


# ------------------------------------------------------------ preconditions

def founder_identity(rebuilt_acc):
    """The founder here must reproduce C-9's founder training-pool accuracy.

    Read from C-9's log rather than written down, so it is a configuration fact
    and not a threshold chosen in this file. The probe accuracy is returned
    beside it and is deliberately NOT part of the check: on real data
    memorisation and generalisation are two quantities, and conflating them is
    the error the held-out split exists to prevent.
    """
    with open(C9_JSON, encoding="utf-8") as handle:
        c9 = json.load(handle)
    expected = c9["founder_accuracy"]
    ok = rebuilt_acc["train"] == expected["train"]
    return {
        "passed": bool(ok),
        "expected_train": expected["train"], "rebuilt_train": rebuilt_acc["train"],
        "expected_probe_reported_not_checked": expected["probe"],
        "rebuilt_probe_reported_not_checked": rebuilt_acc["probe"],
        "note": ("Training-pool accuracy is the identity check. Probe-pool accuracy is a "
                 "different quantity and is reported beside it, never as the same one."),
    }


# ------------------------------------------------------------ students

def one_step(Xtr, founder_logits, bottleneck, n_classes, steps, seed):
    """A student distilled once, directly from generation 0, at `steps` steps.

    Each student draws its own random B-subset, so the reference carries the
    same bottleneck-selection variation the chains carry. A reference without it
    would be narrower than the thing it is compared against, for a reason having
    nothing to do with iteration.
    """
    torch.manual_seed(seed)
    sel = torch.randperm(Xtr.shape[0])[:bottleneck]
    student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    train(student, Xtr[sel], founder_logits[sel], steps, BATCH, LR, WD, True, TAU, DEVICE)
    return student


def observe(model, pools, refs):
    """Every gated and descriptive statistic for one model, on both pools."""
    Xtr, ytr, Xpr, ypr = pools
    out = {}
    for pool, X, y in (("train", Xtr, ytr), ("probe", Xpr, ypr)):
        row = metrics(model, X, refs[pool]["pred"], refs[pool]["feats"])
        with torch.no_grad():
            row["acc_true"] = accuracy(model(X).argmax(-1), y)
        out[pool] = row
    return out


def chain(cond, seed, founder, pools, refs, n_classes):
    """One chain of GENS generations. C-9's run_chain loop, with every
    generation observed on both pools."""
    Xtr = pools[0]
    torch.manual_seed(seed)
    teacher, traj = founder, []
    for _ in range(GENS):
        with torch.no_grad():
            t_logits = teacher(Xtr)
        sel = torch.randperm(Xtr.shape[0])[: cond["B"]]
        student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
        train(student, Xtr[sel], t_logits[sel], STEPS_GEN, BATCH, LR, WD, True, TAU, DEVICE)
        teacher = student
        traj.append(observe(teacher, pools, refs))
    return traj


def reference_arm(Xtr, founder_logits, cond, n_classes, steps, seed_offset, pools, refs):
    """N_REF one-step students at a fixed step budget, each observed like a chain
    generation."""
    return [observe(one_step(Xtr, founder_logits, cond["B"], n_classes, steps,
                             seed_offset + r), pools, refs)
            for r in range(N_REF)]


# ------------------------------------------------------------ gate machinery

def spread(rows, key):
    """A reference arm's distribution of one probe-pool statistic."""
    values = np.array([row["probe"][key] for row in rows], dtype=float)
    lo, hi = np.percentile(values, CENTRAL)
    return {"n": int(values.size), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "min": float(values.min()), "max": float(values.max()),
            "p2_5": float(lo), "p97_5": float(hi),
            "degenerate": bool(np.ptp(values) == 0.0),
            "values": [float(v) for v in values]}


def chain_means(chains, key):
    """The across-chain mean trajectory, which is what C-2 compares against the
    reference percentile. Reused unchanged so the two controls are comparable."""
    return [float(np.mean([c[g]["probe"][key] for c in chains])) for g in range(GENS)]


def first_escape(values, floor):
    """Lowest 1-based generation whose value falls below the floor."""
    for i, value in enumerate(values, start=1):
        if value < floor:
            return i
    return None


def read_gate(trajectory, ref, n_chains, sanity_ok):
    """One gate, with the three headroom floors applied before any verdict.

    Each floor declares NOT EVALUABLE rather than passed: a statistic that
    cannot fail has not passed. The floors are checked in the order
    incompleteness, degeneracy, saturation, because a truncated arm makes the
    other two questions meaningless.
    """
    if not sanity_ok:
        return {"verdict": NOT_EVALUABLE,
                "reason": "the sanity gate failed: the one-step reference is misconfigured "
                          "and nothing else in this run can be read"}
    if ref["n"] < N_REF or n_chains < N_CHAINS:
        return {"verdict": NOT_EVALUABLE, "floor": "3 reference incompleteness",
                "reason": f"{ref['n']} of {N_REF} reference students and {n_chains} of "
                          f"{N_CHAINS} chains completed; a percentile on a truncated "
                          f"reference is a percentile of a different distribution"}
    if ref["degenerate"]:
        return {"verdict": NOT_EVALUABLE, "floor": "2 degenerate reference",
                "reason": "every reference student returned the same value, so the 2.5th "
                          "percentile is a point mass and 'falls below it' is a question "
                          "about float equality rather than about the chain"}
    if trajectory[0] < ref["p2_5"]:
        return {"verdict": NOT_EVALUABLE, "floor": "1 saturation at generation 1",
                "reason": f"the chain mean at generation 1 ({trajectory[0]:.4f}) is already "
                          f"below the floor ({ref['p2_5']:.4f}); an escape present before "
                          f"the chain runs is not evidence about the chain",
                "gen1": trajectory[0], "floor_value": ref["p2_5"]}
    escape = first_escape(trajectory, ref["p2_5"])
    return {"verdict": PASSED if escape else FAILED, "escape_gen": escape,
            "floor_value": ref["p2_5"], "gen1": trajectory[0],
            "gen_last": trajectory[-1], "trajectory": trajectory}


def condition_verdict(gates):
    """PARTIAL whenever at least one and not all evaluable gates were read.

    C-4F's definition: it labels evidence that is real and incomplete, and it is
    not reported as a pass or as a failure.
    """
    read = sorted(k for k, g in gates.items() if g["verdict"] in (PASSED, FAILED))
    refused = sorted(k for k in gates if k not in read)
    if not read:
        return NOT_EVALUABLE
    if refused:
        return f"PARTIAL ({', '.join(read)} read; {', '.join(refused)} refused)"
    return PASSED if all(gates[k]["verdict"] == PASSED for k in read) else FAILED


# ------------------------------------------------------------ one condition

def run_condition(cond, founder, founder_logits, pools, refs, n_classes, t0):
    Xtr = pools[0]
    print(f"\n--- {cond['name']}  B = {cond['B']}  ({cond['match']}) ---", flush=True)

    step_ref = reference_arm(Xtr, founder_logits, cond, n_classes, STEPS_GEN,
                             REF_STEP_SEED_OFFSET, pools, refs)
    print(f"  step-matched reference done ({N_REF} students x {STEPS_GEN} steps)  "
          f"{time.time() - t0:.0f}s", flush=True)
    comp_ref = reference_arm(Xtr, founder_logits, cond, n_classes, STEPS_COMPUTE,
                             REF_COMPUTE_SEED_OFFSET, pools, refs)
    print(f"  compute-matched reference done ({N_REF} students x {STEPS_COMPUTE} steps)  "
          f"{time.time() - t0:.0f}s", flush=True)

    chains = []
    for i in range(N_CHAINS):
        seed = CHAIN_SEED_OFFSET + i
        traj = chain(cond, seed, founder, pools, refs, n_classes)
        chains.append(traj)
        last = traj[-1]["probe"]
        print(f"  chain seed {seed}  gen{GENS} probe agree {last['agree_gen0']:.3f}  "
              f"acc_true {last['acc_true']:.3f}  classes {last['n_classes_used']}  "
              f"{time.time() - t0:.0f}s", flush=True)

    spreads = {
        "step_agree": spread(step_ref, "agree_gen0"),
        "compute_agree": spread(comp_ref, "agree_gen0"),
        "step_acc": spread(step_ref, "acc_true"),
        "compute_acc": spread(comp_ref, "acc_true"),
        "step_classes": spread(step_ref, "n_classes_used"),
    }
    agree_traj = chain_means(chains, "agree_gen0")
    acc_traj = chain_means(chains, "acc_true")
    classes_traj = chain_means(chains, "n_classes_used")

    sanity_lo, sanity_hi = spreads["step_agree"]["p2_5"], spreads["step_agree"]["p97_5"]
    sanity_ok = sanity_lo <= agree_traj[0] <= sanity_hi

    gates = {
        "A": read_gate(agree_traj, spreads["step_agree"], len(chains), sanity_ok),
        "B": read_gate(agree_traj, spreads["compute_agree"], len(chains), sanity_ok),
        "C_vs_step": read_gate(acc_traj, spreads["step_acc"], len(chains), sanity_ok),
        "C_vs_compute": read_gate(acc_traj, spreads["compute_acc"], len(chains), sanity_ok),
    }
    return {
        "condition": cond["name"], "B": cond["B"], "match": cond["match"],
        "sanity": {"passed": bool(sanity_ok), "gen1_mean": agree_traj[0],
                   "central_95": [sanity_lo, sanity_hi]},
        "reference_spreads": spreads,
        "chain_means": {"agree_gen0": agree_traj, "acc_true": acc_traj,
                        "n_classes_used_DESCRIPTIVE": classes_traj},
        "gates": gates,
        "verdict": condition_verdict(gates),
        "chain_seeds": [CHAIN_SEED_OFFSET + i for i in range(N_CHAINS)],
        "chains": chains,
    }


# ------------------------------------------------------------ reporting

def report_gates(res):
    print(f"\n{res['condition']}  B = {res['B']}")
    s = res["sanity"]
    print(f"  sanity: chain gen 1 mean agreement {s['gen1_mean']:.4f} in "
          f"[{s['central_95'][0]:.4f}, {s['central_95'][1]:.4f}] -> "
          f"{'OK' if s['passed'] else 'FAILED'}")
    labels = {"A": "agree_gen0 vs step-matched", "B": "agree_gen0 vs compute-matched",
              "C_vs_step": "acc_true   vs step-matched",
              "C_vs_compute": "acc_true   vs compute-matched"}
    for key, gate in res["gates"].items():
        line = f"  gate {key:<13} {labels[key]:<32} {gate['verdict']}"
        if gate["verdict"] in (PASSED, FAILED):
            where = f"escapes at gen {gate['escape_gen']}" if gate["escape_gen"] else "NEVER escapes"
            line += f"  ({where}; floor {gate['floor_value']:.4f}, " \
                    f"gen1 {gate['gen1']:.4f}, gen{GENS} {gate['gen_last']:.4f})"
        else:
            line += f"  [{gate.get('floor', 'sanity')}]"
        print(line)
    print(f"  condition verdict: {res['verdict']}")
    cls = res["chain_means"]["n_classes_used_DESCRIPTIVE"]
    print(f"  DESCRIPTIVE, no gate: surviving alphabet gen1 {cls[0]:.2f} -> "
          f"gen{GENS} {cls[-1]:.2f}; one-step reference mean "
          f"{res['reference_spreads']['step_classes']['mean']:.2f}. "
          f"No verdict attaches to this and none may be added later.")


def report_combination(results):
    print("\n=== combination across bottleneck conditions ===")
    verdicts = {r["condition"]: r["verdict"] for r in results}
    for name, verdict in verdicts.items():
        print(f"  {name:<22} {verdict}")
    if len(set(verdicts.values())) == 1:
        print("  The two conditions agree; the result is stated once with both named.")
    else:
        print("  SPLIT BY CORRESPONDENCE. The two readings of the synthetic bottleneck give\n"
              "  different verdicts. Both are stated side by side with their bottlenecks\n"
              "  named, never averaged, never resolved, and never reported with one omitted.")


# ------------------------------------------------------------ main

def run(out_json, out_txt):
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json"), encoding="utf-8") as handle:
        rule = json.load(handle)["C-10"]
    print(f"=== {CONTROL}: {rule['name']} ===")
    print(f"preregistered gates, committed {rule['committed']}:")
    for key, gate in rule["gates"].items():
        print(f"  {key:<7} {gate['condition']}")
    print()

    Xtr, ytr, Xpr, ypr, n_classes = load_pools()
    pools = (Xtr, ytr, Xpr, ypr)
    founder = build_founder(Xtr, ytr, n_classes)
    refs = {"train": reference(founder, Xtr), "probe": reference(founder, Xpr)}
    founder_acc = {"train": accuracy(refs["train"]["pred"], ytr),
                   "probe": accuracy(refs["probe"]["pred"], ypr)}

    identity = founder_identity(founder_acc)
    print(f"precondition_founder_identity: training-pool accuracy "
          f"{identity['rebuilt_train']:.4f} against C-9's {identity['expected_train']:.4f} "
          f"-> {'PASS' if identity['passed'] else 'FAIL'}")
    print(f"  probe-pool accuracy {identity['rebuilt_probe_reported_not_checked']:.4f} "
          f"reported beside it, not checked: a different quantity.")
    if not identity["passed"]:
        print("\nThe harness has drifted from the arm whose pools this control shares.\n"
              "Every gate is NOT EVALUABLE and no chain is run.")
        payload = {"control": CONTROL, "verdict": NOT_EVALUABLE,
                   "precondition_founder_identity": identity,
                   "elapsed_sec": round(time.time() - t0, 1)}
        with open(out_json, "x") as handle:
            json.dump(payload, handle, indent=2)
        return

    with torch.no_grad():
        founder_logits = founder(Xtr)

    gated = [c for c in conditions(int(Xtr.shape[0])) if not c["name"].endswith("full_pool")]
    print(f"\ngated conditions: {', '.join(c['name'] for c in gated)}  "
          f"(both, neither privileged)")
    print("full-pool condition not re-run here; C-9's descriptive numbers stand.")

    results = [run_condition(c, founder, founder_logits, pools, refs, n_classes, t0)
               for c in gated]

    print("\n=== gate readings ===")
    for res in results:
        report_gates(res)
    report_combination(results)

    elapsed = round(time.time() - t0, 1)
    payload = {
        "control": CONTROL, "status": "GATED",
        "rule_committed": rule["committed"],
        "precondition_founder_identity": identity,
        "dataset": "sklearn.datasets.load_digits (offline, ships with scikit-learn)",
        "pools": {"n_train": int(Xtr.shape[0]), "n_probe": int(Xpr.shape[0]),
                  "n_features": int(Xtr.shape[1]), "n_classes": n_classes},
        "founder_accuracy": founder_acc,
        "gens": GENS, "tau": TAU, "mode": "soft", "device": DEVICE,
        "steps_gen": STEPS_GEN, "steps_compute_matched": STEPS_COMPUTE,
        "batch": BATCH, "lr": LR, "weight_decay": WD,
        "n_chains": N_CHAINS, "n_reference": N_REF,
        "chain_seed_offset": CHAIN_SEED_OFFSET,
        "ref_seed_offsets": {"step_matched": REF_STEP_SEED_OFFSET,
                             "compute_matched": REF_COMPUTE_SEED_OFFSET},
        "excluded_seen_chain_seeds": "900-904 (C-9's own chains); not re-run, not gated",
        "escape_percentile": ESCAPE_PCT,
        "torch_version": torch.__version__, "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "elapsed_sec": elapsed, "results": results,
    }
    with open(out_json, "x") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwall {elapsed}s; wrote {os.path.relpath(out_json, REPO)} "
          f"and {os.path.relpath(out_txt, REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=OUT_JSON,
                    help="JSON log path; the TXT sits beside it. Never overwritten.")
    out_json, out_txt = log_paths(ap.parse_args().out)
    with open(out_txt, "x") as handle:
        saved = sys.stdout
        sys.stdout = Tee(saved, handle)
        try:
            run(out_json, out_txt)
        finally:
            sys.stdout = saved


if __name__ == "__main__":
    main()
