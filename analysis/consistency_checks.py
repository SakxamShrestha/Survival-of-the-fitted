#!/usr/bin/env python3
"""Consistency checks over this project's deliverables.

Every check here was derived from a defect that actually occurred in this repository.
None of them is a general style rule, and none of them uses a model. They are
deterministic comparisons between what the deliverables say and what the logs, the
gate file and the retraction ledger record, in the same spirit as
.claude/hooks/publish-fence.py: a narrow mechanical backstop for a rule that was
previously written in prose and enforced by nothing.

The checks:

  1. Pool splice. A metric measured only on the training pool must not be presented
     alongside a held-out figure as though both came from the same data. Derived from
     the consecutive-generation CKA of 0.959 being quoted next to a held-out agreement
     of 0.130, when no held-out consecutive-generation CKA exists in this repository.

  2. Baseline-scoped ratios. Each arm of the seed-stability sweep yields an analytic
     ratio, printed in the log, and an empirical-null ratio, quoted in the deliverables.
     CLAUDE.md requires that disagreeing sources be reported side by side. Derived from
     Arm B's analytic ratio never appearing in any deliverable while Arm A's and Arm C's
     both do.

  3. Retracted attributions. A citation claim downgraded to unverified in PROJECT.md
     must not survive as a live assertion elsewhere. Derived from the Davari random-label
     attribution and the dropped ICLR 2023 venue persisting after their retraction.

  4. Superseded values. A number replaced by a later measurement must not be stated as
     current fact without a marker saying it was superseded. Derived from the 2.94x and
     1.13x seed-stability figures surviving the n = 20 rerun in both proposal files.

  5. Controls without gates. Every control recorded as run must have a preregistered
     entry in gates/thresholds.json. Derived from C-1, whose 1.25 cutoff is an inline
     constant in controls/c1_shuffled.py committed in the same commit as the control
     itself, while PROJECT.md states that decision rules are preregistered for all eight.

  6. Banned citations. The two preprints CLAUDE.md rules unsuitable must not appear in
     a deliverable. An exact two-item blocklist, previously enforced only by an agent
     remembering a sentence.

A check reports; it does not edit, and it does not decide what a hit means. Several
findings below are expected to be judgement calls for the author rather than errors.

Usage:

    python3 analysis/consistency_checks.py            # report, exit 1 if anything found
    python3 analysis/consistency_checks.py --quiet    # exit code only

Exit codes: 0 clean, 1 findings, 2 the checker itself could not run.
"""

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_PATH = os.path.join(REPO, "gates", "retractions.json")
THRESHOLDS_PATH = os.path.join(REPO, "gates", "thresholds.json")


# ---------------------------------------------------------------- text helpers


def read(relpath):
    """Return a file's text, or None if it is absent."""
    path = os.path.join(REPO, relpath)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def line_of(text, index):
    """1-indexed line number of a character offset."""
    return text.count("\n", 0, index) + 1


LIST_ITEM = re.compile(r"^(?:[-*+]\s|\d+\.\s)")


def blocks(text):
    """Split into claim-sized blocks, keeping each one's starting line.

    Blocks rather than lines because a claim and the marker that qualifies it are
    usually in the same paragraph, and a superseding note usually sits in the one
    immediately after.

    A blank line ends a block, and so does the start of a top-level list item. The
    list rule matters: PROJECT.md's "What the pilot establishes" section is one
    unbroken bullet list in which a withdrawn temperature claim sits directly above
    a live CKA claim. Treated as a single block, the withdrawal marker on the first
    bullet suppresses a genuine finding on the second.
    """
    out = []
    current, start = [], None
    offset = 0

    def flush():
        if current:
            out.append({
                "text": "".join(current),
                "start": start,
                "line": line_of(text, start),
            })

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            flush()
            current, start = [], None
        elif LIST_ITEM.match(line) and current:
            flush()
            current, start = [line], offset
        else:
            if start is None:
                start = offset
            current.append(line)
        offset += len(line)

    flush()
    return out


def block_index(block_list, offset):
    """Index of the block containing a character offset."""
    for i, block in enumerate(block_list):
        if block["start"] <= offset < block["start"] + len(block["text"]):
            return i
    return max(0, len(block_list) - 1)


def window(block_list, i, radius):
    """Concatenated text of block i together with its neighbours."""
    lo = max(0, i - radius)
    hi = min(len(block_list), i + radius + 1)
    return "".join(b["text"] for b in block_list[lo:hi])


def number_pattern(value):
    """Match a decimal as a whole number, so 2.94 does not match inside 12.945."""
    return re.compile(r"(?<![\d.])" + re.escape(value) + r"(?![\d])")


def contains_any(haystack, needles):
    """Substring test, insensitive to case and to line wrapping.

    Whitespace is collapsed because a phrase that matters to a check is routinely
    split across a line break in these files: the video script wraps "neither of
    them trained on" mid-phrase, and an unnormalized search silently misses it.
    """
    low = re.sub(r"\s+", " ", haystack).lower()
    return any(re.sub(r"\s+", " ", n).lower() in low for n in needles)


def first_trigger(block_text, triggers):
    """Offset of the earliest trigger inside a block, or None."""
    low = block_text.lower()
    hits = [low.find(t.lower()) for t in triggers]
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else None


def json_keys(obj, acc=None):
    """Every key name appearing anywhere in a nested JSON structure."""
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for key, value in obj.items():
            acc.add(key)
            json_keys(value, acc)
    elif isinstance(obj, list):
        for item in obj:
            json_keys(item, acc)
    return acc


def finding(check, path, line, message):
    return {"check": check, "path": path, "line": line, "message": message}


# ---------------------------------------------------------------------- checks


def check_pool_splice(ledger):
    """A training-pool-only metric quoted in a held-out context."""
    found = []
    heldout = ledger["heldout_markers"]

    for spec in ledger["pool_scoped_fields"]:
        field = spec["field"]
        probe_log = spec["verify_absent_from"]
        raw = read(probe_log)

        if raw is None:
            found.append(finding(
                "pool-splice", probe_log, 0,
                f"cannot verify that {field} is absent: file not found. "
                "The check cannot run and the claim it guards is unchecked.",
            ))
            continue

        # Verify against the log rather than trusting the ledger, so the check
        # retires itself the day the field is genuinely measured held-out.
        if field in json_keys(json.loads(raw)):
            found.append(finding(
                "pool-splice", probe_log, 0,
                f"{field} now appears in {probe_log}. The held-out measurement this "
                f"check assumed missing may exist; retire the ledger entry for {field} "
                "after confirming what it measures.",
            ))
            continue

        triggers = list(spec["prose_aliases"]) + list(spec["training_pool_only_values"])

        # Frozen deliverables are submitted, dated artifacts and are not
        # retroactively corrected, which is the ledger's stated reason for the
        # same exemption in check_superseded_values. A pool splice is a
        # measurement belief held at submission time, so it falls under that
        # rule; a citation does not, which is why the citation checks still
        # apply to frozen files. Honouring the exemption here was missing, and
        # the gap only became visible when midterm-report.md was added to the
        # deliverables list on 2026-10-20.
        frozen = set(ledger["frozen_deliverables"]["files"])

        for relpath in ledger["deliverables"]:
            if relpath in frozen:
                continue
            text = read(relpath)
            if text is None:
                continue
            block_list = blocks(text)
            for i, block in enumerate(block_list):
                offset = first_trigger(block["text"], triggers)
                if offset is None:
                    continue
                context = window(block_list, i, 2)
                if not contains_any(context, heldout):
                    continue
                # A claim already marked superseded is a record of what was once
                # believed, not an assertion. The pilot table in PROJECT.md is the
                # case this exempts.
                if contains_any(context, ledger["supersede_markers"]):
                    continue
                found.append(finding(
                    "pool-splice", relpath,
                    line_of(text, block["start"] + offset),
                    f"{field} is measured only on the {', '.join(spec['measured_on'])} "
                    f"and is quoted here within two paragraphs of held-out language. "
                    f"No held-out value for {field} exists in this repository.",
                ))
    return found


def check_baseline_ratios(ledger):
    """An arm's analytic ratio that never reaches a deliverable."""
    found = []
    spec = ledger["baseline_scoped_ratios"]
    log_path = spec["source_log"]
    log = read(log_path)

    if log is None:
        return [finding("baseline-ratio", log_path, 0,
                        "seed-stability log not found; the check cannot run.")]

    corpus = {}
    for relpath in ledger["deliverables"]:
        text = read(relpath)
        if text is not None:
            corpus[relpath] = text

    arm = None
    for match in re.finditer(
        r"---\s*(Arm\s+\w+)[^\n]*---|ratio observed/random\s*:\s*([\d.]+)x", log
    ):
        if match.group(1):
            arm = match.group(1)
            continue

        ratio = match.group(2)
        line = line_of(log, match.start())
        pattern = number_pattern(ratio)
        cited_in = [p for p, t in corpus.items() if pattern.search(t)]

        if not cited_in:
            found.append(finding(
                "baseline-ratio", log_path, line,
                f"{arm or 'an arm'} has an analytic ratio of {ratio}x that appears in no "
                "deliverable. The empirical-null ratio is quoted instead, and the two are "
                "measured against different nulls. CLAUDE.md requires both side by side.",
            ))
    return found


def check_retracted_attributions(ledger):
    """A retracted or unverified attribution still asserted somewhere."""
    found = []
    for entry in ledger["retracted_attributions"]:
        exempt = set(entry.get("exempt", []))
        for relpath in ledger["deliverables"]:
            if relpath in exempt:
                continue
            text = read(relpath)
            if text is None:
                continue
            block_list = blocks(text)
            for pattern in entry["patterns"]:
                for match in re.finditer(re.escape(pattern), text, re.IGNORECASE):
                    i = block_index(block_list, match.start())
                    # Text that already calls the claim unverified, or instructs
                    # someone to verify it, is the correction rather than the
                    # assertion. RESEARCH-PLAN.md's open-work item is that case.
                    if contains_any(window(block_list, i, 1),
                                    ledger["retraction_markers"]):
                        continue
                    found.append(finding(
                        "retracted-attribution", relpath, line_of(text, match.start()),
                        f"\"{pattern}\" was retracted on {entry['retracted']} and the "
                        f"correction is recorded in {entry['recorded_in']}, but the claim "
                        "is still asserted here.",
                    ))
    return found


def check_superseded_values(ledger):
    """A superseded number stated without a marker saying so."""
    found = []
    markers = ledger["supersede_markers"]
    frozen = set(ledger["frozen_deliverables"]["files"])

    for entry in ledger["superseded_values"]:
        value = entry["value"]
        pattern = number_pattern(value)
        for relpath in ledger["deliverables"]:
            if relpath in frozen:
                continue
            text = read(relpath)
            if text is None:
                continue
            block_list = blocks(text)
            for i, block in enumerate(block_list):
                match = pattern.search(block["text"])
                if not match:
                    continue
                # Radius 2: a superseding note often sits one block past a short
                # trailing remark on the table it supersedes.
                if contains_any(window(block_list, i, 2), markers):
                    continue
                found.append(finding(
                    "superseded-value", relpath,
                    line_of(text, block["start"] + match.start()),
                    f"{value} was superseded by {entry['replaced_by']} on "
                    f"{entry['superseded']} and is stated here with no marker. "
                    f"The correction is recorded in {entry['recorded_in']}.",
                ))
    return found


def check_controls_have_gates(ledger):
    """A control recorded as run with no preregistered threshold."""
    found = []
    text = read("PROJECT.md")
    raw = read("gates/thresholds.json")

    if text is None or raw is None:
        return [finding("control-gate", "gates/thresholds.json", 0,
                        "PROJECT.md or gates/thresholds.json not found; cannot run.")]

    preregistered = {k for k in json.loads(raw) if not k.startswith("_")}

    for match in re.finditer(
        r"^\|\s*(C-\d+)\b[^|]*\|\s*([^|]*?)\s*\|", text, re.MULTILINE
    ):
        control, status = match.group(1), match.group(2).strip().lower()
        if status != "run":
            continue
        if control not in preregistered:
            found.append(finding(
                "control-gate", "PROJECT.md", line_of(text, match.start()),
                f"{control} is recorded as run but has no entry in gates/thresholds.json. "
                "PROJECT.md states that decision rules are preregistered and committed "
                "before the control runs.",
            ))
    return found


def check_banned_citations(ledger):
    """A preprint ruled unsuitable appearing in a deliverable."""
    found = []
    for entry in ledger["banned_citations"]:
        exempt = set(entry.get("exempt", []))
        for relpath in ledger["deliverables"]:
            if relpath in exempt:
                continue
            text = read(relpath)
            if text is None:
                continue
            for match in re.finditer(re.escape(entry["id"]), text, re.IGNORECASE):
                found.append(finding(
                    "banned-citation", relpath, line_of(text, match.start()),
                    f"{entry['id']} is ruled unsuitable as authority ({entry['reason']}) "
                    f"in {entry['recorded_in']}.",
                ))
    return found


CHECKS = [
    ("1  pool splice", check_pool_splice),
    ("2  baseline-scoped ratios", check_baseline_ratios),
    ("3  retracted attributions", check_retracted_attributions),
    ("4  superseded values", check_superseded_values),
    ("5  controls without gates", check_controls_have_gates),
    ("6  banned citations", check_banned_citations),
]


def main():
    quiet = "--quiet" in sys.argv

    try:
        with open(LEDGER_PATH, encoding="utf-8") as fh:
            ledger = json.load(fh)
    except Exception as exc:
        print(f"cannot read {LEDGER_PATH}: {exc}", file=sys.stderr)
        return 2

    total = 0
    for name, check in CHECKS:
        try:
            found = check(ledger)
        except Exception as exc:
            print(f"check {name} failed to run: {exc}", file=sys.stderr)
            return 2

        total += len(found)
        if quiet:
            continue

        print(f"\n=== check {name} — {len(found)} finding(s) ===")
        for item in sorted(found, key=lambda f: (f["path"], f["line"])):
            print(f"  {item['path']}:{item['line']}: {item['message']}")

    if not quiet:
        print(f"\n{total} finding(s) across {len(CHECKS)} checks.")
        if total:
            print("A finding is a discrepancy to resolve, not a verdict. Each one names "
                  "the file that records the correction it rests on.")

    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
