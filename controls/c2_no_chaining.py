"""
C-2: no chaining.

Does iteration do anything that a single distillation step does not? Nothing in
this project has ever tested it. Every result so far compares chain generation g
against generation 0, which measures cumulative loss but says nothing about
whether the accumulation matters: if chain-generation 5 looks like a fresh
one-step student, "iterated distillation" is the wrong description of the whole
design.

The reference distribution is a population of students distilled directly from
generation 0 at the same bottleneck. The chain must leave that distribution on
either functional agreement or alphabet size for iteration to be doing anything.

    python3 controls/c2_no_chaining.py

Decision rules are preregistered in gates/thresholds.json under "C-2", committed
before this ran. They are read from that file rather than restated here, so the
thresholds cannot drift from the committed record.

One choice the preregistration did not fix: which pool the gates read. It does
not say, and C-6 has since shown that training-pool metrics mislead badly, so the
gates are evaluated on the held-out probe pool and both pools are reported. That
is a strengthening consistent with the rule as written, but it was not in the
rule, and it is recorded in the log rather than left implicit.
"""

import json
import os
import sys
import time

import numpy as np
import torch

# Run as `python3 controls/c2_no_chaining.py` from the repository root: Python
# puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

GENS = 25
TAU = 1.0
N_REF = 50                  # one-step students forming the reference distribution
N_CHAINS = 5
DEVICE = "cpu"
TRAIN_SEED = 0
PROBE_SEED = 9_000          # identical to controls/c6_probe_pool.py
BOTTLENECKS = (128, 4096)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POOL_NOTE = (
    "Gates are evaluated on the held-out probe pool. The preregistered C-2 entry does "
    "not specify a pool; C-6 established that training-pool metrics are misleading "
    "(argmax 0.969 train vs 0.217 probe at generation 1), so the probe pool was chosen "
    "after C-6 and before C-2 ran. Both pools are reported."
)


def measure(model, Xtr, Xpr, base_tr, base_pr):
    """Functional agreement and alphabet size, on both pools."""
    with torch.no_grad():
        ptr = model(Xtr).argmax(-1)
        ppr = model(Xpr).argmax(-1)
    return {
        "train": {"agree_gen0": float((ptr == base_tr).float().mean()),
                  "n_classes_used": int(ptr.unique().numel())},
        "probe": {"agree_gen0": float((ppr == base_pr).float().mean()),
                  "n_classes_used": int(ppr.unique().numel())},
    }


def one_step(Xtr, t_logits, bottleneck, n_classes, seed):
    """A student distilled once, directly from generation 0."""
    torch.manual_seed(seed)
    sel = torch.randperm(Xtr.shape[0])[:bottleneck]
    student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    train(student, Xtr[sel], t_logits[sel], 1500, 256, 1e-3, 0.0, True, TAU, DEVICE)
    return student


def run_chain(Xtr, Xpr, founder, bottleneck, n_classes, seed, base_tr, base_pr):
    """A chain of GENS generations; returns the per-generation trajectory."""
    torch.manual_seed(seed)
    teacher, traj = founder, []
    for _ in range(GENS):
        with torch.no_grad():
            t_logits = teacher(Xtr)
        sel = torch.randperm(Xtr.shape[0])[:bottleneck]
        student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
        train(student, Xtr[sel], t_logits[sel], 1500, 256, 1e-3, 0.0, True, TAU, DEVICE)
        teacher = student
        traj.append(measure(teacher, Xtr, Xpr, base_tr, base_pr))
    return traj


def first_escape(values, floor):
    """Lowest generation index (1-based) whose value falls below the floor."""
    for i, v in enumerate(values, start=1):
        if v < floor:
            return i
    return None


def main():
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-2"]
    print(f"C-2 preregistered gates, committed {rule['committed']}:")
    for key, gate in rule["gates"].items():
        print(f"  {key:<7} {gate['condition']}")
    print(f"\n{POOL_NOTE}\n")

    Xtr, ytr, C = make_pool(seed=TRAIN_SEED, kind="noise")
    Xpr, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    Xtr, ytr, Xpr = Xtr.to(DEVICE), ytr.to(DEVICE), Xpr.to(DEVICE)

    torch.manual_seed(TRAIN_SEED)
    founder = MLP(Xtr.shape[1], C).to(DEVICE)
    train(founder, Xtr, ytr, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        base_tr = founder(Xtr).argmax(-1)
        base_pr = founder(Xpr).argmax(-1)
        founder_logits = founder(Xtr)
        acc = float((base_tr == ytr).float().mean())
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.3f}\n")

    results, verdicts = {}, {}
    for B in BOTTLENECKS:
        # ---- reference: N_REF students, each one distillation step from gen 0
        ref = [measure(one_step(Xtr, founder_logits, B, C, 1000 + r),
                       Xtr, Xpr, base_tr, base_pr) for r in range(N_REF)]
        ref_agree = np.array([m["probe"]["agree_gen0"] for m in ref])
        ref_classes = np.array([m["probe"]["n_classes_used"] for m in ref])
        lo_a, hi_a = np.percentile(ref_agree, [2.5, 97.5])
        lo_c = np.percentile(ref_classes, 2.5)
        print(f"B={B}: one-step reference over {N_REF} students (probe pool)")
        print(f"  agree_gen0     mean {ref_agree.mean():.3f}  "
              f"central 95% [{lo_a:.3f}, {hi_a:.3f}]")
        print(f"  n_classes_used mean {ref_classes.mean():.1f}  2.5th pct {lo_c:.1f}")

        # ---- chains
        chains = [run_chain(Xtr, Xpr, founder, B, C, 2000 + r, base_tr, base_pr)
                  for r in range(N_CHAINS)]
        mean_agree = [float(np.mean([c[g]["probe"]["agree_gen0"] for c in chains]))
                      for g in range(GENS)]
        mean_classes = [float(np.mean([c[g]["probe"]["n_classes_used"] for c in chains]))
                        for g in range(GENS)]

        sanity_ok = lo_a <= mean_agree[0] <= hi_a
        esc_a = first_escape(mean_agree, lo_a)
        esc_c = first_escape(mean_classes, lo_c)
        print(f"  chain gen 1 agreement {mean_agree[0]:.3f} -> sanity "
              f"{'OK' if sanity_ok else 'FAILED (gen 1 outside the one-step reference)'}")
        print(f"  gate A: agreement escapes at generation "
              f"{esc_a if esc_a else 'NEVER'}")
        print(f"  gate B: alphabet escapes at generation "
              f"{esc_c if esc_c else 'NEVER'}")
        print(f"  gen 25: agreement {mean_agree[-1]:.3f}, "
              f"alphabet {mean_classes[-1]:.1f}\n")

        verdicts[B] = {"sanity": bool(sanity_ok), "gate_a_escape": esc_a,
                       "gate_b_escape": esc_c}
        results[B] = {
            "bottleneck": B, "founder_acc": acc,
            "reference": {"n": N_REF, "agree_mean": float(ref_agree.mean()),
                          "agree_ci95": [float(lo_a), float(hi_a)],
                          "classes_mean": float(ref_classes.mean()),
                          "classes_p2_5": float(lo_c), "runs": ref},
            "chains": {"n": N_CHAINS, "mean_agree_by_gen": mean_agree,
                       "mean_classes_by_gen": mean_classes, "runs": chains},
            **verdicts[B],
        }

    # ---- overall verdict, per the preregistered rule
    print(f"=== C-2 VERDICT  ({time.time()-t0:.0f}s) ===")
    if not all(v["sanity"] for v in verdicts.values()):
        verdict = "MISCONFIGURED"
        print("Chain generation 1 fell outside the one-step reference. Generation 1 IS a")
        print("one-step student by construction, so the comparison is broken. Fix it before")
        print("reading anything else here.")
    else:
        any_a = any(v["gate_a_escape"] for v in verdicts.values())
        any_b = any(v["gate_b_escape"] for v in verdicts.values())
        if any_a or any_b:
            verdict = "PASSED"
            print(f"gate A (function) {'escaped' if any_a else 'did NOT escape'}; "
                  f"gate B (alphabet) {'escaped' if any_b else 'did NOT escape'}.")
            print("The chain leaves the one-step reference, so iteration is doing something")
            print("a single distillation step does not. The iterated framing is supported.")
            if not any_b:
                print("\nNote: alphabet contraction never escaped. One-step students contract")
                print("as much as chains do, so contraction is not an iterated effect and")
                print("C-4's minority-collapse concern (Fang et al. 2021) applies directly.")
        else:
            verdict = "FAILED"
            print("Neither gate escaped at either bottleneck.")
            for line in rule["verdict_if_failed"]:
                print(line)

    out = os.path.join(REPO, "logs", "c2_no_chaining.json")
    with open(out, "w") as handle:
        json.dump({"control": "C-2", "verdict": verdict, "gens": GENS, "tau": TAU,
                   "n_reference": N_REF, "n_chains": N_CHAINS,
                   "train_seed": TRAIN_SEED, "probe_seed": PROBE_SEED,
                   "n_classes": C, "gated_on_pool": "probe", "pool_note": POOL_NOTE,
                   "torch_version": torch.__version__,
                   "elapsed_sec": round(time.time() - t0, 1),
                   "conditions": {str(B): results[B] for B in BOTTLENECKS}},
                  handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
