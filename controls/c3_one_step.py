"""
C-3: one-step fidelity baseline.

A single distillation step from generation 0 at B = 4096 already loses most of
the founder's function: C-2 measured agreement of 0.230 on held-out inputs over
50 one-step students trained for the standard 1500 steps. That number is the
floor against which every downstream decay figure in this project is read.

What C-2 did not test is whether the floor MOVES with the training budget.
Stanton et al. attribute one-step teacher-student discrepancy to optimization
rather than to information loss. If that holds here, the floor is a property of
how long we train rather than of what the bottleneck destroys, and every decay
number has been read against a budget-dependent baseline without anyone saying
so.

This sweeps the one-step student's training budget from 375 to 12000 steps
(0.25x to 8x the standard 1500) and measures agreement with the founder at each
budget. The founder, the bottleneck and the probe pool are identical to C-2 and
C-6, so the three controls are directly comparable.

    python3 controls/c3_one_step.py

Decision rules are preregistered in gates/thresholds.json under "C-3", committed
before this ran. They are read from that file at runtime rather than restated
here, so the thresholds cannot drift from the committed record.

Two choices the preregistration did not fix:

  - Which pool the gate reads. The C-3 entry names the held-out probe pool
    explicitly in its metric, so this follows it; both pools are still reported.
  - Which seeds the students use. They are the first 20 of C-2's 50 reference
    seeds, and the same 20 are reused at every budget. That makes the budgets
    paired (identical initialization and data order, only the step count
    differs) and makes the 1500-step arm an exact re-run of 20 of C-2's own
    students, which is what the sanity check needs.

The student's final training loss is also recorded, as a full-batch value
computed after training rather than a noisy last-minibatch number. It is what
separates "still optimizing" from "converged": an agreement floor that holds
while the loss is still falling is a different claim from one that holds after
the loss has bottomed out.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Run as `python3 controls/c3_one_step.py` from the repository root: Python
# puts this file's directory on sys.path, not the root, so add the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train

TAU = 1.0
BOTTLENECK = 4096           # the C-3 condition: the full pool, so no subsampling
BUDGETS = (375, 750, 1500, 3000, 6000, 12000)
STANDARD_BUDGET = 1500
GATE_BUDGET = 6000          # 4x, the multiplier named in the preregistered rule
N_STUDENTS = 20
DEVICE = "cpu"              # measured faster than MPS for this MLP (1.61 s vs 2.47 s)
TRAIN_SEED = 0
PROBE_SEED = 9_000          # identical to controls/c6_probe_pool.py and c2_no_chaining.py
STUDENT_SEED0 = 1000        # identical to c2_no_chaining.py, so the 1500 arm re-runs C-2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POOL_NOTE = (
    "The gate is evaluated on the held-out probe pool, which the preregistered C-3 "
    "metric names explicitly. Both pools are reported. Training-pool agreement at "
    "B = 4096 is a memorization metric by construction, since the student trains on "
    "every point being measured."
)

SEED_NOTE = (
    "The same 20 student seeds (1000-1019) are used at every budget, so the budgets "
    "are paired: initialization and data order are identical and only the step count "
    "differs. These are the first 20 of the 50 seeds c2_no_chaining.py used for its "
    "B = 4096 reference, so the 1500-step arm is an exact re-run of 20 of C-2's own "
    "students rather than an independent redraw."
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


def final_loss(model, X, targets):
    """Full-batch distillation loss after training, same objective train() uses."""
    with torch.no_grad():
        loss = F.kl_div(
            F.log_softmax(model(X) / TAU, dim=-1),
            F.log_softmax(targets / TAU, dim=-1),
            log_target=True, reduction="batchmean",
        ) * (TAU ** 2)
    return float(loss)


def one_step(Xtr, t_logits, n_classes, seed, steps):
    """A student distilled once, directly from generation 0, for `steps` steps.

    Structurally identical to c2_no_chaining.one_step; the step count is the
    only thing this control varies.
    """
    torch.manual_seed(seed)
    sel = torch.randperm(Xtr.shape[0])[:BOTTLENECK]
    student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    train(student, Xtr[sel], t_logits[sel], steps, 256, 1e-3, 0.0, True, TAU, DEVICE)
    return student, final_loss(student, Xtr[sel], t_logits[sel])


def main():
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-3"]
    ref_lo, ref_hi = rule["reference_interval"]
    threshold = rule["threshold"]

    print(f"C-3 preregistered rule, committed {rule['committed']}:")
    print(f"  metric     {rule['metric']}")
    print(f"  reference  [{ref_lo:.3f}, {ref_hi:.3f}]  ({rule['reference_provenance']})")
    print(f"  fails if   {rule['fails_if']}")
    print(f"\n{POOL_NOTE}\n{SEED_NOTE}\n")

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
    print(f"founder (seed {TRAIN_SEED}) memorization accuracy {acc:.3f}  "
          f"({time.time()-t0:.1f}s)\n")

    header = (f"{'steps':>6}  {'probe agree':>12}  {'95% CI':>16}  "
              f"{'train agree':>12}  {'probe cls':>9}  {'loss':>9}")
    print(header)
    print("-" * len(header))

    conditions = {}
    for steps in BUDGETS:
        runs, losses = [], []
        for r in range(N_STUDENTS):
            student, loss = one_step(Xtr, founder_logits, C, STUDENT_SEED0 + r, steps)
            runs.append(measure(student, Xtr, Xpr, base_tr, base_pr))
            losses.append(loss)

        pr_agree = np.array([m["probe"]["agree_gen0"] for m in runs])
        tr_agree = np.array([m["train"]["agree_gen0"] for m in runs])
        pr_cls = np.array([m["probe"]["n_classes_used"] for m in runs])
        tr_cls = np.array([m["train"]["n_classes_used"] for m in runs])
        loss_arr = np.array(losses)
        lo, hi = np.percentile(pr_agree, [2.5, 97.5])

        print(f"{steps:>6}  {pr_agree.mean():>12.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"{tr_agree.mean():>12.3f}  {pr_cls.mean():>9.1f}  {loss_arr.mean():>9.4f}")

        conditions[str(steps)] = {
            "steps": steps, "n": N_STUDENTS,
            "probe_agree_mean": float(pr_agree.mean()),
            "probe_agree_sd": float(pr_agree.std(ddof=1)),
            "probe_agree_ci95": [float(lo), float(hi)],
            "train_agree_mean": float(tr_agree.mean()),
            "probe_classes_mean": float(pr_cls.mean()),
            "train_classes_mean": float(tr_cls.mean()),
            "final_train_loss_mean": float(loss_arr.mean()),
            "final_train_loss_sd": float(loss_arr.std(ddof=1)),
            "probe_agree_runs": [float(v) for v in pr_agree],
            "train_agree_runs": [float(v) for v in tr_agree],
            "final_train_loss_runs": [float(v) for v in loss_arr],
        }

    std = conditions[str(STANDARD_BUDGET)]["probe_agree_mean"]
    gate = conditions[str(GATE_BUDGET)]["probe_agree_mean"]
    sanity_ok = ref_lo <= std <= ref_hi

    print(f"\n=== C-3 VERDICT  ({time.time()-t0:.0f}s) ===")
    print(f"sanity: {STANDARD_BUDGET}-step arm mean {std:.3f} vs C-2 reference interval "
          f"[{ref_lo:.3f}, {ref_hi:.3f}] -> {'OK' if sanity_ok else 'FAILED'}")

    if not sanity_ok:
        verdict = "MISCONFIGURED"
        print("The standard-budget arm does not reproduce the C-2 reference. This sweep is")
        print("not measuring what C-2 measured, so nothing else in it can be read. Fix the")
        print("configuration before drawing any conclusion about the budget.")
    else:
        print(f"main:   {GATE_BUDGET}-step arm mean {gate:.3f} vs threshold {threshold:.3f}")
        if gate > threshold:
            verdict = "FAILED"
            print("The one-step floor MOVED with the training budget.")
            for line in rule["verdict_if_failed"]:
                print(line)
        else:
            verdict = "PASSED"
            print("The one-step floor did not move at 4x the training budget.")
            for line in rule["verdict_if_passed"]:
                print(line)

    out = os.path.join(REPO, "logs", "c3_one_step.json")
    with open(out, "w") as handle:
        json.dump({"control": "C-3", "verdict": verdict,
                   "sanity_ok": bool(sanity_ok),
                   "sanity_arm_steps": STANDARD_BUDGET,
                   "sanity_arm_mean": std,
                   "reference_interval": [ref_lo, ref_hi],
                   "gate_arm_steps": GATE_BUDGET, "gate_arm_mean": gate,
                   "threshold": threshold,
                   "tau": TAU, "bottleneck": BOTTLENECK, "budgets": list(BUDGETS),
                   "n_students": N_STUDENTS, "student_seed0": STUDENT_SEED0,
                   "train_seed": TRAIN_SEED, "probe_seed": PROBE_SEED,
                   "n_classes": C, "founder_acc": acc, "device": DEVICE,
                   "gated_on_pool": "probe", "pool_note": POOL_NOTE,
                   "seed_note": SEED_NOTE,
                   "torch_version": torch.__version__,
                   "elapsed_sec": round(time.time() - t0, 1),
                   "conditions": conditions},
                  handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
