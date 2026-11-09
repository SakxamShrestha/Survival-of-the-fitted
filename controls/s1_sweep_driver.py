"""
S-1 sweep driver: runs one arm's cells in the committed order, one process each.

Every design choice here is read from gates/thresholds.json entry "S-1" and none
is made locally:

  cell order      design_amendment_2026_11_08.cell_coverage - lambda 100, 10, 1,
                  and within each lambda 4 x sigma_match before sigma_match.
  founders        fixed_in_advance.n_founders_per_arm, pool seeds 0 .. n - 1.
  chain seeds     recording_amendment_2026_11_08.chain_seeds - 7700 + 2 s and
                  7701 + 2 s for the founder at pool seed s.
  dropout rate    design_amendment_2026_11_08.negative_control_rate_and_budget.

It launches controls/s1_sampler_arm.py chain once per (cell, founder) with a
distinct --tag, sequentially, and never retries a cell: a chain whose log already
exists is SKIPPED, because open_logs refuses to overwrite and a half-written log
is better reported than replaced.

    /usr/bin/python3 controls/s1_sweep_driver.py --arm dropout
    /usr/bin/python3 controls/s1_sweep_driver.py --arm sampler --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs")
RULE = os.path.join(REPO, "gates", "thresholds.json")
RUNNER = os.path.join(REPO, "controls", "s1_sampler_arm.py")
INTERPRETER = "/usr/bin/python3"      # torch 2.8.0; see CLAUDE.md

CHAIN_SEED_BASE = 7_700               # recording_amendment_2026_11_08.chain_seeds
CHAINS_PER_FOUNDER = 2                # design_amendment_2026_11_08.chains_per_founder


def load_rule():
    with open(RULE) as handle:
        return json.load(handle)["S-1"]


def committed_cells(rule):
    """(lambda, sigma multiplier) pairs in the order cell_coverage fixes."""
    lambdas = sorted((float(v) for v in rule["fixed_in_advance"]["lambda_grid"]), reverse=True)
    text = rule["fixed_in_advance"]["sigma_grid"]
    if "sigma_match" not in text or "4 x sigma_match" not in text:
        raise SystemExit(f"sigma_grid no longer reads as committed: {text!r}")
    order = rule["design_amendment_2026_11_08"]["cell_coverage"]["rule"]
    joined = " ".join(order)
    if "lambda 100 then 10 then 1" not in joined or "4 x sigma_match before sigma_match" not in joined:
        raise SystemExit("cell_coverage no longer reads as the committed order")
    return [(lam, mult) for lam in lambdas for mult in (4.0, 1.0)]


def chain_seeds(pool_seed, rule):
    """The committed formula, checked against the clause's own text."""
    text = rule["recording_amendment_2026_11_08"]["chain_seeds"]["rule"]
    if "7700 + 2 s" not in text or "7701 + 2 s" not in text:
        raise SystemExit(f"chain_seeds clause no longer reads as committed: {text!r}")
    return [CHAIN_SEED_BASE + CHAINS_PER_FOUNDER * pool_seed + i
            for i in range(CHAINS_PER_FOUNDER)]


def dropout_rate(rule):
    return float(rule["design_amendment_2026_11_08"]
                 ["negative_control_rate_and_budget"]["dropout_rate"])


def tag(arm, lam, mult, pool_seed):
    return f"{arm}_lam{lam:g}_sig{mult:g}_pool{pool_seed}"


def log_paths(arm, lam, mult, pool_seed):
    stem = os.path.join(LOGS, f"s1_chain_{tag(arm, lam, mult, pool_seed)}")
    return stem + ".json", stem + ".txt"


def invocation(arm, lam, mult, pool_seed, rule):
    cmd = [INTERPRETER, RUNNER, "chain", "--arm", arm,
           "--lam", f"{lam:g}", "--sigma-mult", f"{mult:g}",
           "--pool-seed", str(pool_seed),
           "--chain-seeds", *[str(s) for s in chain_seeds(pool_seed, rule)]]
    if arm == "dropout":
        cmd += ["--dropout-p", f"{dropout_rate(rule):g}"]
    return cmd + ["--tag", tag(arm, lam, mult, pool_seed)]


def plan(arm, rule, n_founders):
    return [(lam, mult, s)
            for lam, mult in committed_cells(rule)
            for s in range(n_founders)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arm", choices=["sampler", "dropout"], required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the invocations in order and exit")
    ap.add_argument("--only-cell", default=None,
                    help="restrict to one cell, as 'lam,mult' (e.g. '100,4.0')")
    args = ap.parse_args(argv)

    rule = load_rule()
    n_founders = int(rule["fixed_in_advance"]["n_founders_per_arm"])
    cells = plan(args.arm, rule, n_founders)
    if args.only_cell:
        want = tuple(float(v) for v in args.only_cell.split(","))
        cells = [c for c in cells if (c[0], c[1]) == want]
        if not cells:
            raise SystemExit(f"--only-cell {args.only_cell} is not on the committed grid")

    started = time.time()
    print(f"=== S-1 sweep driver: arm {args.arm}, {len(cells)} invocations, "
          f"{CHAINS_PER_FOUNDER} chains each ===", flush=True)
    done = skipped = failed = 0
    for lam, mult, pool_seed in cells:
        cmd = invocation(args.arm, lam, mult, pool_seed, rule)
        js, tx = log_paths(args.arm, lam, mult, pool_seed)
        label = f"lambda {lam:g} sigma x{mult:g} pool {pool_seed}"
        if os.path.exists(js) or os.path.exists(tx):
            print(f"SKIP  {label}: log exists ({os.path.basename(js)})", flush=True)
            skipped += 1
            continue
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True)
            continue
        t0 = time.time()
        print(f"RUN   {label}", flush=True)
        result = subprocess.run(cmd, cwd=REPO)
        elapsed = time.time() - t0
        if result.returncode == 0:
            done += 1
            print(f"OK    {label}  ({elapsed:.0f}s)", flush=True)
        else:
            failed += 1
            print(f"FAIL  {label}  exit {result.returncode} after {elapsed:.0f}s; "
                  f"continuing, the cell is reported unrun", flush=True)
    print(f"=== driver done: {done} run, {skipped} skipped, {failed} failed, "
          f"{time.time() - started:.0f}s total ===", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
