"""
C-8b: frozen random-body learner from a shared, trained founder.

The preregistered rule is gates/thresholds.json's "C-8b" entry, committed at
4c9bbc2 before this file existed. It is read at runtime and printed; no
constant below may differ from it, and the ones that can be checked against it
are asserted.

Design, as the rule fixes it:

  - Generation 0 is C-1's OWN founder for each pool seed: torch.manual_seed(seed),
    make_pool(seed, "noise"), the full MLP, 3000 steps. Its memorisation accuracy
    must equal the value logged in logs/c1_shuffled_n20.json by float equality,
    or gate A is NOT EVALUABLE.
  - Students are a head reinitialised and retrained each generation on ONE
    frozen random body per chain, built and trained with controls/c8_frozen_body's
    own new_frozen_model / reinit_head / train_head (imported, not copied). The
    body is seeded by the pool seed as C-8 seeds it. Because the founder is also
    built after torch.manual_seed(pool_seed), that random body is bit-for-bit
    the founder's body at INITIALISATION, before its 3000 steps. This is C-8's
    construction carried over, not a choice made here, and it is disclosed.
  - Soft transmission at tau 1.0, B = 128, 25 generations, chain seed =
    600 + pool seed, the C-1 intact arm's settings.
  - The chain loop differs from c8_frozen_body.run_frozen_chain in one respect
    only: at generation 1 the teacher is the shared founder rather than the
    frozen model itself. Everything after the teacher's forward pass (randperm
    selection, reinit_head seeding chain_seed * 1000 + g, head-only training)
    is the same sequence of calls. That is why it could not be reused directly.
  - Gate A is read on the 17 chains with seeds 603-619 only. Chains 600-602
    were seen in a scratch spike before the rule was committed; they are run
    and reported separately and never enter the gate.
  - Every interval is C-1's bootstrap_ci_mean (imported) with a FRESH
    np.random.default_rng(BOOT_SEED) per interval, the way C-1 computed its
    intact interval and C-8's c1_intact_interval recomputes it. The shuffled
    floor uses the shuffled_ci95 C-1 logged, which C-1 drew from the same rng
    continued after its intact interval; the script re-derives both of C-1's
    logged intervals from its per-chain finals and asserts they match before
    using anything.

    python3 controls/c8b_shared_founder.py | tee logs/c8b_shared_founder.txt
    python3 controls/c8b_shared_founder.py --seeds 3 --gens 2   # smoke, no log
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_iterated_distill import MLP, make_pool, train, effective_rank
from controls.c1_shuffled_n20 import (
    BOOT_SEED, N_BOOT, bootstrap_ci_mean, intervals_overlap,
)
from controls.c8_frozen_body import new_frozen_model, reinit_head, train_head

CONTROL = "C-8b"
GENS = 25
TAU = 1.0
BOTTLENECK = 128
STEPS_FOUNDER = 3_000
STEPS_GEN = 1_500
DEVICE = "cpu"
N_CHAINS_RUN = 20
N_CHAINS_GATED = 17

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C1_LOG = os.path.join(REPO, "logs", "c1_shuffled_n20.json")
RULES = os.path.join(REPO, "gates", "thresholds.json")
OUT_JSON = os.path.join(REPO, "logs", "c8b_shared_founder.json")


# ------------------------------------------------------------ pure logic

def parse_seed_range(text):
    """'603-619' -> (603, ..., 619), inclusive at both ends."""
    lo, hi = (int(part) for part in text.split("-"))
    if hi < lo:
        raise ValueError(f"seed range {text!r} is reversed")
    return tuple(range(lo, hi + 1))


def chain_to_pool_seed(c1_log):
    """Map chain seed -> pool seed from C-1's own log, validating consistency.

    Every intact record must satisfy chain_seed == chain_seed_offset + seed,
    and the set of record seeds must equal the log's pool_seeds.
    """
    offset = int(c1_log["chain_seed_offset"])
    records = c1_log["results"]["intact"]
    mapping = {}
    for rec in records:
        if int(rec["chain_seed"]) != offset + int(rec["seed"]):
            raise ValueError(
                f"C-1 record seed {rec['seed']} has chain seed {rec['chain_seed']}, "
                f"expected {offset + int(rec['seed'])}")
        mapping[int(rec["chain_seed"])] = int(rec["seed"])
    if sorted(mapping.values()) != sorted(int(s) for s in c1_log["pool_seeds"]):
        raise ValueError("C-1 pool_seeds do not match the seeds in its intact results")
    return mapping


def select_chains(records, chain_seeds):
    """Records for exactly `chain_seeds`, in that order; error if any is absent."""
    by_seed = {int(r["chain_seed"]): r for r in records}
    missing = [s for s in chain_seeds if s not in by_seed]
    if missing:
        raise ValueError(f"chain seeds absent from records: {missing}")
    return [by_seed[s] for s in chain_seeds]


def check_founder_identity(observed, logged):
    """Float-equality check of founder accuracy, per pool seed in `logged`."""
    mismatches = [
        {"seed": seed, "logged": want, "observed": observed.get(seed)}
        for seed, want in sorted(logged.items())
        if observed.get(seed) is None or observed[seed] != want
    ]
    return not mismatches, mismatches


def raw_two_sided(frozen, intact):
    """deeper / shallower / overlap, by two-sided interval overlap."""
    if frozen[1] < intact[0]:
        return "deeper"
    if frozen[0] > intact[1]:
        return "shallower"
    return "overlap"


def classify_outcome(frozen25, intact25, frozen1, shuffled, founder_ok):
    """Gate A reading with the precondition and headroom floors 1 and 2.

    Order: founder identity, then floor 1 (frozen gen-1 interval overlaps intact
    gen-25 interval), then the two-sided reading, then floor 2 (frozen gen-25
    interval overlaps C-1's shuffled interval: only 'deeper' may be stated,
    worded 'to the floor'). Under floor 2 a non-'deeper' reading is returned as
    NOT EVALUABLE with the raw reading kept for the record; the rule says only
    'deeper' may be stated and does not name what else is said.
    """
    raw = raw_two_sided(frozen25, intact25)
    result = {"raw_outcome": raw, "evaluable": True, "floor_fired": None,
              "outcome": raw}
    if not founder_ok:
        return dict(result, evaluable=False, outcome="NOT EVALUABLE",
                    floor_fired="precondition_founder_identity")
    if intervals_overlap(frozen1, intact25):
        return dict(result, evaluable=False, outcome="NOT EVALUABLE",
                    floor_fired="headroom_1_saturation_at_gen1")
    if intervals_overlap(frozen25, shuffled):
        if raw == "deeper":
            return dict(result, outcome="deeper (to the floor)",
                        floor_fired="headroom_2_shuffled_floor")
        return dict(result, evaluable=False, outcome="NOT EVALUABLE",
                    floor_fired="headroom_2_shuffled_floor")
    return result


def interval(values):
    """C-1's bootstrap with a fresh rng at BOOT_SEED; returns mean and CI."""
    rng = np.random.default_rng(BOOT_SEED)
    lo, hi = bootstrap_ci_mean(list(values), rng)
    return {"mean": float(np.mean(values)), "ci95": [lo, hi], "n": len(values)}


# ------------------------------------------------------------ training

def build_c1_founder(seed):
    """C-1's founder recipe, step for step as controls/c1_shuffled_n20.main."""
    torch.manual_seed(seed)
    X, y0, C = make_pool(seed=seed, kind="noise")
    founder = MLP(X.shape[1], C).to(DEVICE)
    train(founder, X, y0, STEPS_FOUNDER, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        acc = float((founder(X).argmax(-1) == y0).float().mean())
    return X, y0, C, founder, acc


def body_snapshot(model):
    return {k: v.detach().clone() for k, v in model.body.state_dict().items()}


def body_unchanged(model, snap):
    state = model.body.state_dict()
    return all(torch.equal(state[k], v) for k, v in snap.items())


def run_shared_founder_chain(founder, student, X, chain_seed, gens):
    """Head-only chain on `student`'s frozen body, generation-1 teacher = founder.

    Mirrors c8_frozen_body.run_frozen_chain call for call except the teacher of
    generation 1. Records per-generation alphabet sizes on the training pool,
    generation-1 agreement with the founder, and two freeze-leak checks per
    generation (features bitwise equal; body state_dict unchanged).
    """
    torch.manual_seed(chain_seed)
    with torch.no_grad():
        feats0 = student(X, return_feats=True)[1].clone()
        founder_pred = founder(X).argmax(-1)
    snap = body_snapshot(student)
    sizes, feat_leaks, body_changes, agree1 = [], 0, 0, None
    for g in range(gens):
        with torch.no_grad():
            t_logits = (founder if g == 0 else student)(X)
        sel = torch.randperm(X.shape[0])[:BOTTLENECK]
        Xs, targets = X[sel], t_logits[sel]
        reinit_head(student, chain_seed * 1000 + g)
        train_head(student, Xs, targets, soft=True, steps=STEPS_GEN)
        with torch.no_grad():
            feats, pred = student(X, return_feats=True)[1], student(X).argmax(-1)
        feat_leaks += int(not torch.equal(feats, feats0))
        body_changes += int(not body_unchanged(student, snap))
        sizes.append(int(pred.unique().numel()))
        if g == 0:
            agree1 = float((pred == founder_pred).float().mean())
    return {"sizes": sizes, "final": sizes[-1], "agree_gen1": agree1,
            "feature_leaks": feat_leaks, "body_changes": body_changes,
            "body_effective_rank": float(effective_rank(feats0))}


def intact_gen1(founder, X, C, chain_seed):
    """Generation 1 of C-1's intact run_chain, recomputed for the fidelity
    covariate only. Same calls as c1_shuffled_n20.run_chain's first iteration;
    validated against C-1's logged sizes[0] by the caller."""
    torch.manual_seed(chain_seed)
    with torch.no_grad():
        t_logits = founder(X)
        founder_pred = t_logits.argmax(-1)
    sel = torch.randperm(X.shape[0])[:BOTTLENECK]
    student = MLP(X.shape[1], C).to(DEVICE)
    train(student, X[sel], t_logits[sel], STEPS_GEN, 256, 1e-3, 0.0, True, TAU, DEVICE)
    with torch.no_grad():
        pred = student(X).argmax(-1)
    return int(pred.unique().numel()), float((pred == founder_pred).float().mean())


def run_seed(pool_seed, chain_seed, gens):
    X, _y, C, founder, acc = build_c1_founder(pool_seed)
    random_student = new_frozen_model(X.shape[1], C, pool_seed)
    frozen = run_shared_founder_chain(founder, random_student, X, chain_seed, gens)
    founder_body = copy.deepcopy(founder)
    founder_body.body.requires_grad_(False)
    companion = run_shared_founder_chain(founder, founder_body, X, chain_seed, gens)
    i_size, i_agree = intact_gen1(founder, X, C, chain_seed)
    return {"seed": pool_seed, "chain_seed": chain_seed, "founder_acc": acc,
            "frozen_random_body": frozen, "companion_founder_body": companion,
            "intact_gen1": {"size": i_size, "agree_gen1": i_agree}}


# ------------------------------------------------------------ analysis

def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def check_rule_constants(rule):
    fixed = rule["fixed_in_advance"]
    assert fixed["gens"] == GENS and fixed["bottleneck"] == BOTTLENECK
    assert fixed["tau"] == TAU
    assert fixed["n_chains_run"] == N_CHAINS_RUN
    assert fixed["n_chains_gated"] == N_CHAINS_GATED
    gated = parse_seed_range(fixed["gated_chain_seeds"])
    seen = parse_seed_range(fixed["disclosed_seen_chain_seeds"])
    assert len(gated) == N_CHAINS_GATED and not set(gated) & set(seen)
    return gated, seen


def verify_c1_intervals(c1):
    """Re-derive C-1's logged intact and shuffled intervals with its own
    bootstrap (one rng, intact then shuffled) and require exact equality."""
    rng = np.random.default_rng(int(c1["boot_seed"]))
    assert int(c1["n_boot"]) == N_BOOT
    i_ci = list(bootstrap_ci_mean([r["final"] for r in c1["results"]["intact"]], rng))
    s_ci = list(bootstrap_ci_mean([r["final"] for r in c1["results"]["shuffled"]], rng))
    return {"intact_ci95_rederived": i_ci, "intact_ci95_logged": c1["intact_ci95"],
            "shuffled_ci95_rederived": s_ci, "shuffled_ci95_logged": c1["shuffled_ci95"],
            "match": i_ci == c1["intact_ci95"] and s_ci == c1["shuffled_ci95"]}


def arm_summary(runs, key):
    finals = [r[key]["final"] for r in runs]
    gen1 = [r[key]["sizes"][0] for r in runs]
    mean_over = [float(np.mean(r[key]["sizes"])) for r in runs]
    agree = [r[key]["agree_gen1"] for r in runs]
    return {"gen25": interval(finals), "gen1": interval(gen1),
            "mean_alphabet_gens_1_25": interval(mean_over),
            "agree_gen1": interval(agree), "finals": finals}


def analyse(runs, c1, gated, seen, founder_ok, founder_mismatches):
    by_chain = {r["chain_seed"]: r for r in runs}
    missing = [s for s in gated if s not in by_chain]
    if missing:
        raise ValueError(f"gated chains never ran: {missing}")
    g_runs = [by_chain[s] for s in gated]
    c1_g = select_chains(c1["results"]["intact"], [r["chain_seed"] for r in g_runs])
    intact25 = interval([r["final"] for r in c1_g])
    intact_mean_over = interval([float(np.mean(r["sizes"])) for r in c1_g])
    gen1_matches = [r["intact_gen1"]["size"] == c["sizes"][0]
                    for r, c in zip(g_runs, select_chains(
                        c1["results"]["intact"], [r["chain_seed"] for r in g_runs]))]
    all_gen1_ok = all(
        r["intact_gen1"]["size"] == select_chains(c1["results"]["intact"],
                                                  [r["chain_seed"]])[0]["sizes"][0]
        for r in runs)
    intact_agree = (interval([r["intact_gen1"]["agree_gen1"] for r in g_runs])
                    if all(gen1_matches) else "DISCARDED: gen-1 alphabets differ from C-1's log")
    frozen = arm_summary(g_runs, "frozen_random_body")
    companion = arm_summary(g_runs, "companion_founder_body")
    shuffled = list(c1["shuffled_ci95"])
    shuffled_17 = interval(
        [r["final"] for r in select_chains(c1["results"]["shuffled"],
                                           [r["chain_seed"] for r in g_runs])])
    gate = classify_outcome(frozen["gen25"]["ci95"], intact25["ci95"],
                            frozen["gen1"]["ci95"], shuffled, founder_ok)
    disclosed = [{"chain_seed": r["chain_seed"],
                  "frozen_final": r["frozen_random_body"]["final"],
                  "companion_final": r["companion_founder_body"]["final"],
                  "c1_intact_final": select_chains(c1["results"]["intact"],
                                                   [r["chain_seed"]])[0]["final"],
                  "frozen_agree_gen1": r["frozen_random_body"]["agree_gen1"],
                  "intact_agree_gen1": r["intact_gen1"]["agree_gen1"]}
                 for r in runs if r["chain_seed"] in seen]
    return {
        "gate_a": dict(gate, frozen_gen25=frozen["gen25"], intact_gen25=intact25,
                       frozen_gen1=frozen["gen1"],
                       shuffled_ci95_c1_logged_n20=shuffled,
                       shuffled_17_descriptive_not_used=shuffled_17),
        "precondition_founder_identity": {"ok": founder_ok,
                                          "mismatches": founder_mismatches},
        "frozen_random_body": frozen,
        "intact_17": {"gen25": intact25, "mean_alphabet_gens_1_25": intact_mean_over,
                      "agree_gen1": intact_agree, "gen1_alphabets_match_c1": gen1_matches},
        "intact_gen1_all_20_match_c1": all_gen1_ok,
        "descriptive_companion_founder_body_NOT_GATED": companion,
        "disclosed_chains_600_602_EXCLUDED": disclosed,
    }


def fmt(iv):
    return f"{iv['mean']:.3f} [{iv['ci95'][0]:.3f}, {iv['ci95'][1]:.3f}] (n={iv['n']})"


def report(res, sanity):
    g = res["gate_a"]
    print("\n=== C-8b GATE A (17 gated chains, 603-619, training pool) ===")
    print(f"founder identity precondition : {'PASS' if res['precondition_founder_identity']['ok'] else 'FAIL'}")
    print(f"frozen  gen-25 : {fmt(g['frozen_gen25'])}")
    print(f"intact  gen-25 : {fmt(g['intact_gen25'])}   (C-1 log, same 17 seeds)")
    print(f"frozen  gen-1  : {fmt(g['frozen_gen1'])}   (floor 1)")
    print(f"C-1 shuffled CI (logged, n=20): {g['shuffled_ci95_c1_logged_n20']}   (floor 2)")
    print(f"raw two-sided reading: {g['raw_outcome']}")
    print(f"OUTCOME: {g['outcome']}   floor fired: {g['floor_fired']}")
    fr, it = res["frozen_random_body"], res["intact_17"]
    print("\nmean alphabet over gens 1-25:")
    print(f"  frozen {fmt(fr['mean_alphabet_gens_1_25'])}")
    print(f"  intact {fmt(it['mean_alphabet_gens_1_25'])}")
    print("fidelity covariate (gen-1 agreement with founder, not gated):")
    print(f"  frozen {fmt(fr['agree_gen1'])}")
    ia = it["agree_gen1"]
    print(f"  intact {fmt(ia) if isinstance(ia, dict) else ia}")
    c = res["descriptive_companion_founder_body_NOT_GATED"]
    print("\ndescriptive companion, founder's own body frozen (NOT GATED):")
    print(f"  gen-25 {fmt(c['gen25'])}   gen-1 {fmt(c['gen1'])}")
    print(f"  mean over gens {fmt(c['mean_alphabet_gens_1_25'])}   agree gen-1 {fmt(c['agree_gen1'])}")
    print("\ndisclosed chains 600-602 (EXCLUDED from the gate):")
    for d in res["disclosed_chains_600_602_EXCLUDED"]:
        print(f"  {d}")
    print(f"\nsanity: {json.dumps(sanity)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="smoke: first N pool seeds, no log")
    ap.add_argument("--gens", type=int, default=GENS)
    ap.add_argument("--out", default=OUT_JSON, help="JSON log path; never overwritten")
    args = ap.parse_args()
    smoke = args.seeds is not None or args.gens != GENS
    out_json = os.path.abspath(args.out)
    if not smoke and os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite a log")

    t0 = time.time()
    c1_hash_before = sha256(C1_LOG)
    with open(RULES) as handle:
        rule = json.load(handle)[CONTROL]
    gated, seen = check_rule_constants(rule)
    with open(C1_LOG) as handle:
        c1 = json.load(handle)
    c1_check = verify_c1_intervals(c1)
    if not c1_check["match"]:
        raise SystemExit(f"C-1's logged intervals do not re-derive: {c1_check}")
    assert c1["gens"] == GENS and c1["bottleneck"] == BOTTLENECK and c1["tau"] == TAU
    # torch 2.5.1 and 2.8.0 diverge after 1,500 steps; C-1's gen-1 sizes
    # reproduce only under the version that produced them.
    if not smoke and torch.__version__ != c1["torch_version"]:
        raise SystemExit(f"torch {torch.__version__} differs from C-1's {c1['torch_version']}; "
                         f"run under the interpreter C-1 used")
    mapping = chain_to_pool_seed(c1)
    chain_seeds = sorted(mapping)[: args.seeds] if args.seeds else sorted(mapping)
    assert smoke or len(chain_seeds) == N_CHAINS_RUN

    print(f"C-8b rule committed {rule['committed']}: {rule['question']}")
    print(f"C-1 interval re-derivation: {c1_check}")
    runs = []
    for cs in chain_seeds:
        r = run_seed(mapping[cs], cs, args.gens)
        runs.append(r)
        print(f"seed {r['seed']:>2} chain {cs}  founder_acc {r['founder_acc']!r}  "
              f"frozen {r['frozen_random_body']['sizes'][0]}->{r['frozen_random_body']['final']}  "
              f"companion {r['companion_founder_body']['sizes'][0]}->{r['companion_founder_body']['final']}  "
              f"intact_g1 {r['intact_gen1']['size']}  {time.time()-t0:.0f}s", flush=True)

    logged_acc = {int(r["seed"]): r["founder_acc"] for r in c1["results"]["intact"]
                  if int(r["chain_seed"]) in chain_seeds}
    founder_ok, mism = check_founder_identity({r["seed"]: r["founder_acc"] for r in runs},
                                              logged_acc)
    if smoke:
        print(f"smoke run, no log written. founder identity: {founder_ok} {mism}")
        return
    res = analyse(runs, c1, gated, seen, founder_ok, mism)
    sanity = {
        "feature_leaks_total": sum(r[k]["feature_leaks"] for r in runs
                                   for k in ("frozen_random_body", "companion_founder_body")),
        "body_state_changes_total": sum(r[k]["body_changes"] for r in runs
                                        for k in ("frozen_random_body", "companion_founder_body")),
        "c1_log_sha256_unchanged": sha256(C1_LOG) == c1_hash_before,
        "c1_intervals_rederived": c1_check["match"],
    }
    report(res, sanity)
    elapsed = round(time.time() - t0, 1)
    with open(out_json, "x") as handle:
        json.dump({"control": CONTROL, "rule_committed": rule["committed"],
                   **res, "sanity": sanity, "c1_interval_check": c1_check,
                   "gens": GENS, "tau": TAU, "bottleneck": BOTTLENECK,
                   "gated_chain_seeds": list(gated), "disclosed_chain_seeds": list(seen),
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "bootstrap_note": "fresh default_rng(BOOT_SEED) per interval; floor 2 uses C-1's logged shuffled_ci95",
                   "random_body_note": "C-8's new_frozen_model(d_in, C, pool_seed): identical to the founder's body at initialisation",
                   "device": DEVICE, "torch_version": torch.__version__,
                   "torch_num_threads": torch.get_num_threads(),
                   "elapsed_sec": elapsed, "runs": runs}, handle, indent=2)
    print(f"\nwall {elapsed}s; wrote {os.path.relpath(out_json, REPO)}")


if __name__ == "__main__":
    main()
