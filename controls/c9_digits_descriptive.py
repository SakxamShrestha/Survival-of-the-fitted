"""
C-9: the same chain on a real dataset. STRICTLY DESCRIPTIVE, UNGATED.

Does the chain behaviour this project measured on synthetic random-label data
appear at all on natural data? That is the whole question, and the answer is
description and can never be anything else, which is the cost of not having
preregistered it. There is no gate, no threshold, no permutation null and no
verdict here, and none may be added later: a decision rule written once the
numbers are in hand is not a decision rule. gates/thresholds.json has no "C-9"
entry and must not acquire one. The precedent is the frozen-founder-body arm in
progress-report-4.md section 2.8, which carries exactly this status. So this
arm can show whether a behaviour is visible outside the synthetic pool, and it
cannot support a claim that one condition differs from another: nothing fixed
in advance says what a difference would have had to look like.

The data is sklearn's `load_digits`: 1,797 samples, 64 integer-valued pixel
features, 10 classes, shipped inside the package and read from disk with no
network access. Features are divided by 16.0 and cast to float32, labels to
int64. Nothing is downloaded; no fetch_openml, no torchvision.

The sample is split stratified into a training pool and a held-out probe pool,
and every metric is reported on both. The split is load-bearing in a way it is
not on random labels: a founder trained on real digits generalises as well as
memorises, so training-pool accuracy (memorisation) and probe-pool accuracy
(generalisation) are two different quantities. Conflating them is the error the
split exists to avoid, so both are printed, labelled, before any chain runs.

Three bottleneck conditions, because the correspondence to the synthetic
headline condition is genuinely unresolved. That condition is B = 128 from a
4,096-point pool, 3.1% per generation; the digits training pool is far smaller,
so B = 128 on it is a much larger fraction of what the teacher could have
transmitted, and matching the absolute number and matching the fraction are
different experiments. Both run here, plus the no-bottleneck limit: B = 128
(absolute match), B = 3.1% of the pool (ratio match, derived from 128 / 4096 at
runtime), and B = the whole pool. NEITHER match is privileged over the other;
which is right this arm does not settle, and the third condition is the
reference point both are read against.

Five chain seeds per condition, fifteen chains, so intervals can be reported
rather than single runs. Metric definitions are reused unchanged: `metrics()`
returns exactly the keys controls/c6_probe_pool.py returns, by the same calls,
which is what makes the comparison like for like. Do not redefine them here.

    /usr/bin/python3 controls/c9_digits_descriptive.py

Target interpreter is /usr/bin/python3, which carries torch 2.8.0. The log
records the torch, python and sklearn versions; the runner never overwrites one.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Run as `/usr/bin/python3 controls/c9_digits_descriptive.py` from the repository
# root: Python puts this file's directory on sys.path, not the root, so add the
# root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, train, effective_rank, linear_cka
from controls.c1_shuffled_n20 import BOOT_SEED, bootstrap_ci_mean

CONTROL = "C-9"

# Chain and training settings, all matching controls/c6_probe_pool.py.
GENS = 25
DEVICE = "cpu"
TAU = 1.0
STEPS_FOUNDER = 3_000
STEPS_GEN = 1_500
BATCH = 256
LR = 1e-3
WD = 0.0

# Seeds, named rather than inline: the split seed defines the two pools that
# everything else is read on.
SPLIT_SEED = 9_100          # the stratified training/probe split; fixed
FOUNDER_SEED = 0            # the single shared founder
CHAIN_SEED_OFFSET = 900     # chain seeds 900..904, a namespace no other arm uses
N_CHAINS = 5

# load_digits features are integers in [0, 16]; dividing by the maximum puts
# them in [0, 1] without fitting anything to the data.
DIGITS_FEATURE_MAX = 16.0
PROBE_FRACTION = 1.0 / 3.0
EXPECTED_SAMPLES, EXPECTED_FEATURES, EXPECTED_CLASSES = 1797, 64, 10

# The synthetic headline condition, used only to derive the matched bottlenecks.
SYNTH_BOTTLENECK = 128
SYNTH_POOL = 4096

POOLS = ("train", "probe")
SUMMARY_KEYS = ("agree_gen0", "eff_rank", "n_classes_used", "entropy")

NOTES = {   # carried into the JSON so the log states its own status
    "status": ("No gate, threshold, permutation null or verdict exists for this arm and "
               "none may be added after the fact. gates/thresholds.json has no C-9 entry. "
               "Precedent: progress-report-4.md section 2.8."),
    "bottleneck": ("Three conditions because the correspondence to the synthetic B = 128 "
                   "of 4096 is unresolved: absolute match, ratio match, no bottleneck. "
                   "Neither match is privileged."),
    "founder_acc": ("train = memorisation on the pool it trained on; probe = "
                    "generalisation to held-out samples. Two different quantities."),
    "interval": ("ci95_descriptive is C-1's bootstrap_ci_mean over one condition's "
                 "chains. It describes those chains; it is not a test."),
    "metrics": ("metrics() returns controls/c6_probe_pool.py's keys by the same calls. "
                "acc_true_labels is extra and sits outside metrics()."),
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO, "logs", "c9_digits_descriptive.json")


# ------------------------------------------------------------ log safety

class Tee:
    """Mirror stdout into the .txt log, as controls/s1_sampler_arm.py does."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def log_paths(out_json):
    """JSON and TXT paths; refuses to overwrite either. Most runners here
    replace their log in place, but a log is sometimes the only record of a
    cited number, so this one stops and leaves the decision to the operator."""
    js = os.path.abspath(out_json)
    tx = os.path.splitext(js)[0] + ".txt"
    for path in (js, tx):
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; refusing to overwrite a log. Pass --out "
                             f"with a new path, or move the existing log aside.")
    return js, tx


# ------------------------------------------------------------ data

def load_pools():
    """load_digits, scaled and split stratified into training and probe pools.

    Shape, feature range and class coverage are checked before anything trains:
    a mis-scaled or class-incomplete pool yields numbers that look ordinary.
    """
    data = load_digits()
    X = (data.data / DIGITS_FEATURE_MAX).astype(np.float32)
    y = data.target.astype(np.int64)
    n_classes = int(y.max()) + 1
    if X.shape != (EXPECTED_SAMPLES, EXPECTED_FEATURES) or n_classes != EXPECTED_CLASSES:
        raise SystemExit(f"load_digits gave shape {X.shape} with {n_classes} classes, "
                         f"expected {(EXPECTED_SAMPLES, EXPECTED_FEATURES)} with "
                         f"{EXPECTED_CLASSES}")
    if X.min() < 0.0 or X.max() > 1.0:
        raise SystemExit(f"scaled features fall outside [0, 1]: [{X.min()}, {X.max()}]")
    Xtr, Xpr, ytr, ypr = train_test_split(
        X, y, test_size=PROBE_FRACTION, random_state=SPLIT_SEED, stratify=y)
    for name, labels in (("training", ytr), ("probe", ypr)):
        present = int(np.unique(labels).size)
        if present != n_classes:
            raise SystemExit(f"{name} pool covers {present} of {n_classes} classes; "
                             f"the stratified split is not doing its job")
    tensors = [torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)
               for a in (Xtr, ytr, Xpr, ypr)]
    return (*tensors, n_classes)


def conditions(n_train):
    """The three bottlenecks, the ratio match derived from the fraction the
    synthetic condition transmits rather than hardcoded. Neither it nor the
    absolute match is the privileged correspondence."""
    if SYNTH_BOTTLENECK > n_train:
        raise SystemExit(f"B = {SYNTH_BOTTLENECK} exceeds the training pool "
                         f"({n_train}); the absolute match is not runnable")
    ratio = SYNTH_BOTTLENECK / SYNTH_POOL
    b_ratio = max(1, round(n_train * ratio))
    return [
        {"name": "b128_absolute_match", "B": SYNTH_BOTTLENECK,
         "match": "absolute: the synthetic headline condition's own B"},
        {"name": f"b{b_ratio}_ratio_match", "B": b_ratio,
         "match": f"ratio: {ratio:.4f} of this pool, as 128/4096 is of 4096"},
        {"name": f"b{n_train}_full_pool", "B": n_train, "match": "no bottleneck"},
    ]


def metrics(model, X, base_pred, base_feats):
    """Every chain metric, evaluated on whichever pool X is. Same keys and same
    calls as controls/c6_probe_pool.py's metrics(), reused unchanged."""
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


def interval(values):
    """Descriptive spread over the chains of one condition. C-1's
    bootstrap_ci_mean with a fresh rng at BOOT_SEED, so the interval is
    computed as every other interval here is. At n = 5 it describes these five
    chains and nothing more, so the raw values, sd and range sit beside it."""
    arr = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci_mean(list(arr), np.random.default_rng(BOOT_SEED))
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "ci95_descriptive": [lo, hi], "n": int(arr.size),
            "values": [float(v) for v in arr]}


# ------------------------------------------------------- chains and reporting

def build_founder(Xtr, ytr, n_classes):
    """One founder, shared by all fifteen chains, trained on the training pool."""
    torch.manual_seed(FOUNDER_SEED)
    founder = MLP(Xtr.shape[1], n_classes).to(DEVICE)
    train(founder, Xtr, ytr, STEPS_FOUNDER, BATCH, LR, WD, False, 1.0, DEVICE)
    return founder


def reference(founder, X):
    """The founder's predictions and features on one pool, frozen for the run."""
    with torch.no_grad():
        logits, feats = founder(X, return_feats=True)
    return {"pred": logits.argmax(-1), "feats": feats}


def accuracy(pred, y):
    return float((pred == y).float().mean())


def run_chain(cond, chain_seed, founder, pools, refs, n_classes):
    """One chain, every generation measured on both pools. The loop is
    controls/c6_probe_pool.py's run_condition loop: teacher forward over the
    whole training pool, a random B-subset, a fresh student on soft targets at
    TAU, student becomes teacher."""
    Xtr, ytr, Xpr, ypr = pools
    torch.manual_seed(chain_seed)
    teacher = founder
    rows = []
    for g in range(1, GENS + 1):
        with torch.no_grad():
            t_logits = teacher(Xtr)
        sel = torch.randperm(Xtr.shape[0])[: cond["B"]]
        student = MLP(Xtr.shape[1], n_classes).to(DEVICE)
        train(student, Xtr[sel], t_logits[sel], STEPS_GEN, BATCH, LR, WD,
              True, TAU, DEVICE)
        teacher = student
        with torch.no_grad():
            tr_pred, pr_pred = student(Xtr).argmax(-1), student(Xpr).argmax(-1)
        rows.append({
            "gen": g,
            "train": metrics(student, Xtr, refs["train"]["pred"], refs["train"]["feats"]),
            "probe": metrics(student, Xpr, refs["probe"]["pred"], refs["probe"]["feats"]),
            # Real labels exist here and do not in the synthetic arm, so true-label
            # accuracy is recorded too. It sits outside metrics() so that the keys
            # shared with the synthetic arm stay identical.
            "acc_true_labels": {"train": accuracy(tr_pred, ytr),
                                "probe": accuracy(pr_pred, ypr)},
        })
    return rows


def summarise(chains):
    """Generation-GENS spread across the five chains of one condition."""
    last = [c["rows"][-1] for c in chains]
    out = {"gen": GENS}
    for pool in POOLS:
        out[pool] = {key: interval([row[pool][key] for row in last])
                     for key in SUMMARY_KEYS}
    out["acc_true_labels"] = {
        pool: interval([row["acc_true_labels"][pool] for row in last])
        for pool in POOLS}
    return out


def fmt(iv):
    sd = "  na" if iv["sd"] is None else f"{iv['sd']:.3f}"
    return f"{iv['mean']:.3f} sd {sd} [{iv['min']:.3f}, {iv['max']:.3f}] (n={iv['n']})"


def report_founder(sizes, acc):
    print(f"=== {CONTROL}: sklearn load_digits, DESCRIPTIVE AND UNGATED ===\n"
          "No gate, no threshold, no permutation null, no verdict. This arm can show "
          "whether a behaviour is visible on real data and nothing more.\n\n"
          f"training pool : {sizes['n_train']} samples (split seed {SPLIT_SEED}, stratified)\n"
          f"probe pool    : {sizes['n_probe']} samples held out, no model trains on it\n"
          f"features {sizes['n_features']}, classes {sizes['n_classes']}, "
          f"scaled by 1/{DIGITS_FEATURE_MAX}\n\n"
          "FOUNDER ACCURACY, the two pools reported separately:\n"
          f"  training pool (MEMORISATION)   : {acc['train']:.4f}\n"
          f"  probe pool    (GENERALISATION) : {acc['probe']:.4f}\n"
          "  Two different quantities on real data. Neither may be quoted as the other, "
          "and neither is a chain result.\n")


def report_conditions(results):
    print(f"\n=== generation {GENS}, {N_CHAINS} chains per condition "
          f"(description, not a comparison) ===")
    for res in results:
        print(f"\n{res['condition']}  B = {res['B']}  ({res['match']})")
        for pool in POOLS:
            for key in ("agree_gen0", "eff_rank", "n_classes_used"):
                print(f"  {pool:<6} {key:<14} {fmt(res['summary'][pool][key])}")
            print(f"  {pool:<6} {'acc_true':<14} "
                  f"{fmt(res['summary']['acc_true_labels'][pool])}")
    print("\nThe three conditions are not tested against one another. The absolute match "
          "and the ratio match\nare two unresolved readings of the same synthetic "
          "condition, and neither is privileged.")


def run(out_json, out_txt):
    t0 = time.time()
    Xtr, ytr, Xpr, ypr, n_classes = load_pools()
    pools = (Xtr, ytr, Xpr, ypr)
    sizes = {"n_train": int(Xtr.shape[0]), "n_probe": int(Xpr.shape[0]),
             "n_features": int(Xtr.shape[1]), "n_classes": n_classes}

    founder = build_founder(Xtr, ytr, n_classes)
    refs = {"train": reference(founder, Xtr), "probe": reference(founder, Xpr)}
    founder_acc = {"train": accuracy(refs["train"]["pred"], ytr),
                   "probe": accuracy(refs["probe"]["pred"], ypr)}
    report_founder(sizes, founder_acc)

    chain_seeds = [CHAIN_SEED_OFFSET + i for i in range(N_CHAINS)]
    results = []
    for cond in conditions(sizes["n_train"]):
        chains = []
        for seed in chain_seeds:
            rows = run_chain(cond, seed, founder, pools, refs, n_classes)
            chains.append({"chain_seed": seed, "rows": rows})
            tr, pr = rows[-1]["train"], rows[-1]["probe"]
            print(f"{cond['name']:<22} seed {seed}  "
                  f"agree tr/pr {tr['agree_gen0']:.3f}/{pr['agree_gen0']:.3f}  "
                  f"classes tr/pr {tr['n_classes_used']}/{pr['n_classes_used']}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        results.append({"condition": cond["name"], "B": cond["B"],
                        "match": cond["match"], "tau": TAU, "mode": "soft",
                        "summary": summarise(chains), "chains": chains})

    report_conditions(results)
    elapsed = round(time.time() - t0, 1)
    payload = {
        "control": CONTROL, "status": "DESCRIPTIVE AND UNGATED",
        "status_note": NOTES["status"], "bottleneck_note": NOTES["bottleneck"],
        "founder_accuracy_note": NOTES["founder_acc"], "interval_note": NOTES["interval"],
        "metrics_note": NOTES["metrics"], "pools": sizes, "founder_accuracy": founder_acc,
        "dataset": "sklearn.datasets.load_digits (offline, ships with scikit-learn)",
        "split": {"seed": SPLIT_SEED, "probe_fraction": PROBE_FRACTION, "stratified": True},
        "gens": GENS, "tau": TAU, "mode": "soft", "device": DEVICE,
        "steps_founder": STEPS_FOUNDER, "steps_gen": STEPS_GEN, "batch": BATCH,
        "lr": LR, "weight_decay": WD, "founder_seed": FOUNDER_SEED,
        "chain_seeds": chain_seeds, "n_chains_per_condition": N_CHAINS,
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
