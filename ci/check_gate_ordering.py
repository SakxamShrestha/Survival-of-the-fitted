#!/usr/bin/env python3
"""Was every control's decision rule committed before the run it judges?

This is the one mechanical check on the project's central methodological
claim. gates/thresholds.json's own _README states the argument: the same author
writes the experiment and its kill condition, so no file permission makes that
honest, and what a commit buys is that moving a threshold becomes a visible diff
instead of a silent edit. The evidence for "the rule predates the result" is
therefore the order of two commits, and until now nothing verified that order.
It held because the author and the assistant remembered to do it in two steps.

The check, per control:

  1. Find the earliest commit in which gates/thresholds.json contains an entry
     for that control.
  2. Find the commit that first added that control's log under logs/.
  3. Require the first to be a strict ancestor of the second.

Strict matters. Equality is the exact defect this project already recorded
against C-1, whose 1.25 cutoff was an inline constant in
controls/c1_shuffled.py committed in the same commit as the control itself.
A gate landing in the same commit as its result is a gate written with the
result in view, whatever the file says.

Controls are matched to logs by what the log itself declares, not by a
filename table here: every control log written by this project carries a
top-level "control" field (logs/c4_timing.json says "C-4T"). A table in this
file would be a second place for the mapping to drift.

Two legitimate exemptions, both read from the data rather than hardcoded:

  is_replication_gate   C-1's entry says so explicitly. Its n = 3 result was
                        already published when the gate was written, and the
                        entry says as much in its own background field: what is
                        preregistered is the rule for judging the n = 20
                        replication. The gate must still precede the log it
                        actually judges, so the check is not skipped - it is
                        applied to the replication log and the exemption is
                        recorded in the output.

  no log yet            A gate preregistered for a run that has not happened
                        has nothing to order against. Reported as pending,
                        which is not a pass.

Exit codes: 0 all ordered correctly, 1 at least one violation, 2 the check
could not run.

    python3 ci/check_gate_ordering.py
    python3 ci/check_gate_ordering.py --quiet     # exit code only
"""

import glob
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS = os.path.join("gates", "thresholds.json")
LOGS_GLOB = os.path.join(REPO, "logs", "*.json")
SHORT_SHA = 7


def git_lines(*args):
    """Run a read-only git command, returning its stdout lines.

    Raises on failure rather than returning empty, because an empty result and
    a broken invocation mean opposite things here: the first is "no such
    commit", the second is "this check is not working" and must not be
    reported as a pass.
    """
    result = subprocess.run(
        ("git", "-C", REPO) + args,
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed with {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return [line for line in result.stdout.splitlines() if line]


def logs_by_control():
    """Map each control id to every log file that declares it.

    A control may write several logs: S-1 writes one per precondition stage,
    and C-8b was rerun under the torch version C-1 used. No log is chosen over
    another; the gate must precede each of them, so every one is checked.
    """
    found = {}
    for path in sorted(glob.glob(LOGS_GLOB)):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        control = payload.get("control")
        if not control:
            continue
        found.setdefault(control, []).append(os.path.relpath(path, REPO))
    return found


def gate_entry_commit(control):
    """The earliest commit whose gates/thresholds.json contains `control`.

    Walks the file's history oldest-first and parses each revision. Revisions
    that predate the file, or in which it does not parse, are skipped: a
    broken intermediate revision is not evidence about ordering either way,
    and ci/check_json_validity.py is what fails on unparseable JSON at HEAD.
    """
    for sha in git_lines("log", "--reverse", "--format=%H", "--", THRESHOLDS):
        blob = subprocess.run(
            ("git", "-C", REPO, "show", f"{sha}:{THRESHOLDS}"),
            capture_output=True, text=True, check=False,
        )
        if blob.returncode != 0:
            continue
        try:
            if control in json.loads(blob.stdout):
                return sha
        except json.JSONDecodeError:
            continue
    return None


def log_add_commit(relpath):
    """The commit that first added `relpath`, or None if it is not committed."""
    adds = git_lines("log", "--reverse", "--diff-filter=A", "--format=%H",
                     "--", relpath)
    return adds[0] if adds else None


def is_strict_ancestor(earlier, later):
    """Is `earlier` a genuine ancestor of `later`, and not the same commit?"""
    if earlier == later:
        return False
    result = subprocess.run(
        ("git", "-C", REPO, "merge-base", "--is-ancestor", earlier, later),
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


def classify(control, entry, log):
    """One control's ordering status. Returns a new row dict."""
    gate_sha = gate_entry_commit(control)
    log_sha = log_add_commit(log)

    if gate_sha is None:
        return {
            "control": control, "status": "VIOLATION", "log": log,
            "detail": "no committed revision of gates/thresholds.json contains this "
                      "entry, but its log is committed",
        }
    if log_sha is None:
        return {
            "control": control, "status": "uncommitted", "log": log,
            "detail": "the log is in the working tree but not committed; ordering is "
                      "established at commit time",
        }

    if is_strict_ancestor(gate_sha, log_sha):
        detail = f"gate {gate_sha[:SHORT_SHA]} precedes log {log_sha[:SHORT_SHA]}"
        if entry.get("is_replication_gate", False):
            detail += ("; flagged is_replication_gate, so it judges this replication "
                       "and not the earlier published run")
        return {"control": control, "status": "ok", "log": log, "detail": detail}

    if gate_sha == log_sha:
        reason = ("they are the same commit, which is a rule written with the result "
                  "in view")
    else:
        reason = ("the log was committed first, so the threshold could have been "
                  "chosen after seeing it")
    return {
        "control": control, "status": "VIOLATION", "log": log,
        "detail": f"gate {gate_sha[:SHORT_SHA]} does not precede log "
                  f"{log_sha[:SHORT_SHA]} - {reason}",
    }


def audit():
    """One row per control in gates/thresholds.json, plus unmapped logs."""
    with open(os.path.join(REPO, THRESHOLDS), encoding="utf-8") as handle:
        gates = json.load(handle)

    logs = logs_by_control()
    rows = []

    for control, entry in gates.items():
        if control.startswith("_"):
            continue
        control_logs = logs.get(control)
        if not control_logs:
            rows.append({
                "control": control, "status": "pending", "log": None,
                "detail": "preregistered; no log declares this control yet",
            })
            continue
        rows.extend(classify(control, entry, log) for log in control_logs)

    mapped = {path for paths in logs.values() for path in paths}
    unmapped = sorted(
        os.path.relpath(path, REPO) for path in glob.glob(LOGS_GLOB)
        if os.path.relpath(path, REPO) not in mapped
    )
    return rows, unmapped


def main():
    quiet = "--quiet" in sys.argv

    try:
        rows, unmapped = audit()
    except Exception as exc:                    # noqa: BLE001 - reported, not hidden
        print(f"gate-ordering check could not run: {exc}", file=sys.stderr)
        return 2

    violations = [row for row in rows if row["status"] == "VIOLATION"]

    if not quiet:
        print("=== gate ordering: was each rule committed before its run? ===\n")
        for row in sorted(rows, key=lambda r: r["control"]):
            where = f"  [{row['log']}]" if row["log"] else ""
            print(f"  {row['control']:6} {row['status']:12} {row['detail']}{where}")
        if unmapped:
            print("\nlogs declaring no control id, not checked:")
            for path in unmapped:
                print(f"  {path}")
        n_controls = len({row["control"] for row in rows})
        print(f"\n{len(violations)} violation(s) across {len(rows)} log(s) of "
              f"{n_controls} preregistered control(s).")
        if violations:
            print("A violation means the evidence for preregistration is absent for "
                  "that control, not that its result is wrong. It cannot be repaired "
                  "by editing history: rewriting those commits destroys the same "
                  "evidence it would be trying to restore.")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
