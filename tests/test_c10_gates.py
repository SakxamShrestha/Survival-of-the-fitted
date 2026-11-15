"""
Tests for controls/c10_digits_iteration.py's gate machinery.

Synthetic data only. Every trajectory and reference spread here is
hand-constructed for the test; none is read from logs/, and no statistic from
any real C-10 run appears. Arrange-Act-Assert throughout; test names state the
behaviour under test.

    /usr/bin/python3 -m pytest tests/test_c10_gates.py -v

Why these particular cases. C-10 reads four gate positions against two
reference arms on a real dataset, and every refusal it can issue is a headroom
floor that must fire BEFORE a verdict is taken. A floor that silently failed to
fire would turn a statistic that cannot fail into a reported pass, which is the
exact defect C-4's gate B recorded and C-8's floor 3 was written to prevent. So
the tests assert the refusals, their ordering, and the wording of the PARTIAL
combination, rather than any numeric outcome:

  - a falling trajectory escapes at the first generation below the floor, and
    one that never falls is FAILED rather than refused;
  - floor 1 catches an escape that already exists at generation 1, before any
    chaining has happened;
  - floor 2 catches a reference with zero spread, where "falls below the 2.5th
    percentile" is a question about float equality;
  - floor 3 catches a truncated reference or chain arm, and is checked FIRST,
    because a percentile on a short arm makes the other two questions
    meaningless;
  - a failed sanity gate refuses every gate whatever the trajectory does;
  - PARTIAL names both the gates that were read and the ones refused, because
    C-4F's definition requires a PARTIAL to say which evidence carried it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controls.c10_digits_iteration import (  # noqa: E402
    FAILED, NOT_EVALUABLE, PASSED, condition_verdict, first_escape, read_gate,
)

HEALTHY_REF = {"n": 20, "p2_5": 0.50, "p97_5": 0.90, "degenerate": False}
N_CHAINS_OK = 20


# ----------------------------------------------------------------- escapes

def test_escape_is_the_first_generation_below_the_floor():
    # Arrange
    trajectory = [0.80, 0.70, 0.60, 0.45, 0.30]
    # Act
    gate = read_gate(trajectory, HEALTHY_REF, N_CHAINS_OK, sanity_ok=True)
    # Assert
    assert gate["verdict"] == PASSED
    assert gate["escape_gen"] == 4


def test_trajectory_that_never_falls_below_the_floor_is_failed_not_refused():
    trajectory = [0.80, 0.78, 0.76, 0.74, 0.72]

    gate = read_gate(trajectory, HEALTHY_REF, N_CHAINS_OK, sanity_ok=True)

    assert gate["verdict"] == FAILED
    assert gate["escape_gen"] is None


def test_first_escape_returns_none_when_nothing_falls():
    assert first_escape([0.9, 0.8], 0.5) is None
    assert first_escape([0.9, 0.4, 0.3], 0.5) == 2


# ------------------------------------------------------------- the floors

def test_floor_1_refuses_an_escape_that_exists_before_any_chaining():
    already_escaped = [0.40, 0.35, 0.30]

    gate = read_gate(already_escaped, HEALTHY_REF, N_CHAINS_OK, sanity_ok=True)

    assert gate["verdict"] == NOT_EVALUABLE
    assert gate["floor"] == "1 saturation at generation 1"


def test_floor_2_refuses_a_reference_with_zero_spread():
    point_mass = {**HEALTHY_REF, "degenerate": True}

    gate = read_gate([0.80, 0.40], point_mass, N_CHAINS_OK, sanity_ok=True)

    assert gate["verdict"] == NOT_EVALUABLE
    assert gate["floor"] == "2 degenerate reference"


def test_floor_3_refuses_a_truncated_reference_arm():
    short_ref = {**HEALTHY_REF, "n": 19}

    gate = read_gate([0.80, 0.40], short_ref, N_CHAINS_OK, sanity_ok=True)

    assert gate["verdict"] == NOT_EVALUABLE
    assert gate["floor"] == "3 reference incompleteness"


def test_floor_3_refuses_a_truncated_chain_arm():
    gate = read_gate([0.80, 0.40], HEALTHY_REF, 19, sanity_ok=True)

    assert gate["verdict"] == NOT_EVALUABLE
    assert gate["floor"] == "3 reference incompleteness"


def test_incompleteness_is_checked_before_saturation():
    # A short arm AND an escape at generation 1: the truncation must win, because
    # a percentile on five students is a percentile of a different distribution
    # and the saturation question cannot be asked of it.
    short_and_saturated = {**HEALTHY_REF, "n": 5}

    gate = read_gate([0.40, 0.30], short_and_saturated, N_CHAINS_OK, sanity_ok=True)

    assert gate["floor"] == "3 reference incompleteness"


def test_failed_sanity_refuses_every_gate_whatever_the_trajectory_does():
    clearly_escaping = [0.80, 0.40, 0.20]

    gate = read_gate(clearly_escaping, HEALTHY_REF, N_CHAINS_OK, sanity_ok=False)

    assert gate["verdict"] == NOT_EVALUABLE
    assert "sanity gate failed" in gate["reason"]


# -------------------------------------------------------- the combination

@pytest.mark.parametrize("gates, expected", [
    ({"A": {"verdict": PASSED}, "B": {"verdict": PASSED}}, PASSED),
    ({"A": {"verdict": PASSED}, "B": {"verdict": FAILED}}, FAILED),
    ({"A": {"verdict": FAILED}, "B": {"verdict": FAILED}}, FAILED),
    ({"A": {"verdict": NOT_EVALUABLE}, "B": {"verdict": NOT_EVALUABLE}}, NOT_EVALUABLE),
])
def test_condition_verdict_when_every_gate_resolves_the_same_way(gates, expected):
    assert condition_verdict(gates) == expected


def test_partial_names_both_the_gates_read_and_the_gates_refused():
    gates = {"A": {"verdict": PASSED},
             "B": {"verdict": NOT_EVALUABLE},
             "C_vs_step": {"verdict": FAILED}}

    verdict = condition_verdict(gates)

    assert verdict == "PARTIAL (A, C_vs_step read; B refused)"


def test_partial_is_not_reported_as_a_pass_or_a_failure():
    gates = {"A": {"verdict": PASSED}, "B": {"verdict": NOT_EVALUABLE}}

    verdict = condition_verdict(gates)

    assert verdict.startswith("PARTIAL")
    assert verdict not in (PASSED, FAILED, NOT_EVALUABLE)
