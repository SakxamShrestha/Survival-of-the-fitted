"""
C-6: held-out probe pool.

Every metric in this project has been computed on the same 4,096-point pool the
chain trains on. At B = 4096 the students train on all of it, so the
tau0.5/1.0/2.0_b4096 rows of the pilot table are training metrics by
construction, and three of them are cited as evidence in the public README.

This rebuilds each published pilot condition and reports every metric twice: on
the training pool, and on a second pool drawn from the same distribution that no
model ever trains on. The gap between them is the size of the error.

Metrics transfer to the probe pool cleanly because `agree_gen0` compares the
student's predictions against the FOUNDER'S predictions, not against generation
0's labels. Only the founder's memorization accuracy is specific to the training
pool, and it is reported for the training pool alone.

    python3 controls/c6_probe_pool.py

Decision rule is preregistered in gates/thresholds.json under "C-6" and was
committed before this ran.
"""

import json
import os
import sys
import time

import torch

# Run as `python3 controls/c6_probe_pool.py` from the repository root: Python puts
# this file's directory on sys.path, not the root, so add the root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train, effective_rank, linear_cka

GENS = 25
DEVICE = "cpu"
TRAIN_SEED = 0
PROBE_SEED = 9_000          # any seed that is not TRAIN_SEED; fixed for reproducibility

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The six published pilot conditions.
CONDITIONS = [
    {"name": "tau0.5_b4096", "tau": 0.5, "B": 4096, "mode": "soft"},
    {"name": "tau1.0_b4096", "tau": 1.0, "B": 4096, "mode": "soft"},
    {"name": "tau2.0_b4096", "tau": 2.0, "B": 4096, "mode": "soft"},
    {"name": "tau1.0_b512",  "tau": 1.0, "B": 512,  "mode": "soft"},
    {"name": "tau1.0_b128",  "tau": 1.0, "B": 128,  "mode": "soft"},
    {"name": "argmax",       "tau": 1.0, "B": 4096, "mode": "argmax"},
]


def metrics(model, X, base_pred, base_feats):
    """Every chain metric, evaluated on whichever pool X is."""
    with torch.no_grad():
        logits, feats = model(X, return_feats=True)
    pred = logits.argmax(-1)
    probs = torch.softmax(logits, -1)
    return {
        "agree_gen0": float((pred == base_pred).float().mean()),
        "entropy": float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()),
        "logit_norm": float(logits.norm(dim=-1).mean()),
        "n_classes_used": int(pred.unique().numel()),
        "eff_rank": effective_rank(feats),
        "cka_gen0": linear_cka(feats, base_feats),
    }


def run_condition(cond, Xtr, ytr, Xpr, n_classes):
    """One chain, with every generation measured on both pools."""
    torch.manual_seed(TRAIN_SEED)
    founder = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    train(founder, Xtr, ytr, 3000, 256, 1e-3, 0.0, False, 1.0, DEVICE)

    with torch.no_grad():
        tr_pred, tr_feats = (lambda o: (o[0].argmax(-1), o[1]))(founder(Xtr, return_feats=True))
        pr_pred, pr_feats = (lambda o: (o[0].argmax(-1), o[1]))(founder(Xpr, return_feats=True))
        founder_acc = float((tr_pred == ytr).float().mean())

    teacher = founder
    rows = []
    for g in range(1, GENS + 1):
        with torch.no_grad():
            t_logits = teacher(Xtr)
        sel = torch.randperm(Xtr.shape[0])[: cond["B"]]
        student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
        if cond["mode"] == "soft":
            train(student, Xtr[sel], t_logits[sel], 1500, 256, 1e-3, 0.0,
                  True, cond["tau"], DEVICE)
        else:
            hard = t_logits[sel].argmax(-1)
            train(student, Xtr[sel], hard, 1500, 256, 1e-3, 0.0, False, 1.0, DEVICE)
        teacher = student
        rows.append({
            "gen": g,
            "train": metrics(student, Xtr, tr_pred, tr_feats),
            "probe": metrics(student, Xpr, pr_pred, pr_feats),
        })
    return founder_acc, rows


def main():
    t0 = time.time()
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        rule = json.load(handle)["C-6"]
    threshold = rule["threshold"]
    print(f"C-6 preregistered threshold: {rule['metric']} > {threshold}")
    print(f"committed {rule['committed']}\n")

    Xtr, ytr, C = make_pool(seed=TRAIN_SEED, kind="noise")
    Xpr, _, C_probe = make_pool(seed=PROBE_SEED, kind="noise")
    assert C == C_probe, "probe pool must have the same class count"
    Xtr, ytr, Xpr = Xtr.to(DEVICE), ytr.to(DEVICE), Xpr.to(DEVICE)

    results, breaches = [], []
    print(f"{'condition':<15}{'train':>9}{'probe':>9}{'gap':>9}   "
          f"{'classes tr/pr':>14}{'effrank tr/pr':>16}")
    for cond in CONDITIONS:
        acc, rows = run_condition(cond, Xtr, ytr, Xpr, C)
        last = rows[-1]
        gap = abs(last["probe"]["agree_gen0"] - last["train"]["agree_gen0"])
        breached = gap > threshold
        if breached:
            breaches.append(cond["name"])
        results.append({"condition": cond["name"], **{k: cond[k] for k in ("tau", "B", "mode")},
                        "founder_acc": acc, "gap_agree_gen0_g25": gap,
                        "exceeds_threshold": breached, "rows": rows})
        tr, pr = last["train"], last["probe"]
        classes = f"{tr['n_classes_used']}/{pr['n_classes_used']}"
        ranks = f"{tr['eff_rank']:.1f}/{pr['eff_rank']:.1f}"
        flag = "   BREACH" if breached else ""
        print(f"{cond['name']:<15}{tr['agree_gen0']:>9.3f}{pr['agree_gen0']:>9.3f}"
              f"{gap:>9.3f}{classes:>14}{ranks:>16}{flag}")

    # ---- sanity check, per the preregistered rule: tight bottlenecks train on a
    # small fraction of the pool, so their two pools should already agree closely.
    tight = next(r for r in results if r["condition"] == "tau1.0_b128")
    sanity_ok = tight["gap_agree_gen0_g25"] <= threshold

    print(f"\n=== C-6 VERDICT  ({time.time()-t0:.0f}s, {GENS} gens, "
          f"probe pool seed {PROBE_SEED}) ===")
    print(f"sanity (B=128 gap {tight['gap_agree_gen0_g25']:.3f} <= {threshold}): "
          f"{'OK' if sanity_ok else 'FAILED - probe pool is built wrong'}")
    if not sanity_ok:
        verdict = "MISCONFIGURED"
        print("The tight-bottleneck condition trains on 128 of 4,096 points, so its two")
        print("pools should already agree. They do not. Fix the probe pool before reading")
        print("anything else here.")
    elif breaches:
        verdict = "FAILED"
        print(f"threshold exceeded in: {', '.join(breaches)}")
        for line in rule["verdict_if_failed"]:
            print(line)
    else:
        verdict = "PASSED"
        print("No condition exceeds the threshold. The published metrics survive being")
        print("recomputed off the training pool, and the pilot table stands as written.")

    out = os.path.join(REPO, "logs", "c6_probe_pool.json")
    with open(out, "w") as handle:
        json.dump({"control": "C-6", "verdict": verdict, "threshold": threshold,
                   "gens": GENS, "train_seed": TRAIN_SEED, "probe_seed": PROBE_SEED,
                   "n_classes": C, "torch_version": torch.__version__,
                   "elapsed_sec": round(time.time() - t0, 1),
                   "breaches": breaches, "results": results}, handle, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
