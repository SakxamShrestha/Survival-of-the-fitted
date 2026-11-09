"""
Tests for controls/s1_aggregate.py, the S-1 verdict aggregator.

Synthetic data only. Every gate reading, p-value and chain log here is
hand-constructed for the test; none is read from logs/, and no gate statistic
is computed by the siblings that own them (analysis/founder_id.py,
analysis/sigma_attractor.py). Those are injected as stubs, so this file passes
whether or not they exist yet.

The Holm-Bonferroni cases below are hand-computed in the test's own comments
before being asserted, because every verdict in the arm passes through that
correction.

    /usr/bin/python3 -m pytest tests/test_s1_aggregate.py -v
"""

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from controls import s1_aggregate as agg


# ------------------------------------------------------------------ fixtures

def reading(name, p, observed=0.5, evaluable=True, refusals=()):
    """A gate reading with no statistic behind it."""
    return agg.GateReading(name=name, observed=observed, p_value=p,
                           evaluable=evaluable, refusals=tuple(refusals),
                           detail={"stub": True})


def refused(name, kind="headroom_1_degenerate_alphabet", reason="stub refusal"):
    return agg.GateReading(name=name, observed=None, p_value=None,
                           evaluable=False, refusals=((kind, reason),),
                           detail={"stub": True})


def full_coverage():
    return agg.coverage_from_cell_keys(agg.COMMITTED_CELLS, n_founders={c: 20 for c in agg.COMMITTED_CELLS})


def partial_coverage():
    """lambda 100 at both sigmas complete; nothing else run."""
    done = (agg.COMMITTED_CELLS[0], agg.COMMITTED_CELLS[1])
    return agg.coverage_from_cell_keys(done, n_founders={c: 20 for c in done})


def combine(gates, exactness="PASS", inertness="PASS", ggn="PASS",
            cov=None, mc=False, **kw):
    return agg.combine(exactness=exactness, inertness=inertness, ggn=ggn,
                       gate_readings=gates,
                       coverage_info=cov if cov is not None else full_coverage(),
                       mc_dropout_reproduced=mc, **kw)


def chain_payload(lam, mult, pool_seed, chain_seeds=(600, 601), gens=25,
                  probe_final=(9, 11), survivors=((1, 2, 3), (2, 3, 4)),
                  probe_survivors=None, arm="sampler"):
    """A minimal logs/s1_chain_<tag>.json payload in the runner's shape."""
    chains = []
    for i, cs in enumerate(chain_seeds):
        chain = {"arm": arm, "lambda": lam, "sigma_multiplier": mult,
                 "chain_seed": cs,
                 "sizes": [40] * (gens - 1) + [len(survivors[i])],
                 "probe_sizes": [40] * (gens - 1) + [probe_final[i]],
                 "survivors": list(survivors[i]),
                 "final": len(survivors[i]), "diagnostics": [], "gen_times": [],
                 "prior_sds": {}, "p_drop": None}
        if probe_survivors is not None:
            chain["probe_survivors"] = list(probe_survivors[i])
        chains.append(chain)
    return {"control": "S-1", "stage": "chain", "arm": arm, "lambda": lam,
            "sigma_multiplier": mult, "pool_seed": pool_seed,
            "founder_acc": 0.977294921875,
            "founder_acc_departs_by_more_than_0.01": False,
            "gens": gens, "bottleneck": 128, "chains": chains,
            "optimizer": {}, "torch_version": "2.8.0", "python_version": "3.9.6"}


def write_cell(tmp_path, tag, payload):
    path = tmp_path / f"s1_chain_{tag}.json"
    path.write_text(json.dumps(payload))
    return str(path)


# ================================================== Holm-Bonferroni, by hand

def test_holm_three_gates_rejects_the_two_below_their_own_thresholds():
    # m = 3, alpha = 0.05. Thresholds: 0.05/3 = 0.0166667, 0.05/2 = 0.025, 0.05/1 = 0.05.
    # 0.001 < 0.0166667 reject; 0.02 < 0.025 reject; 0.4 > 0.05 stop.
    # Adjusted: 3*0.001 = 0.003; max(0.003, 2*0.02 = 0.04) = 0.04; max(0.04, 1*0.4) = 0.4.
    out = agg.holm_bonferroni({"A": 0.001, "B": 0.02, "C": 0.4})
    assert out["m"] == 3
    assert out["adjusted"] == pytest.approx({"A": 0.003, "B": 0.04, "C": 0.4})
    assert out["rejected"] == {"A": True, "B": True, "C": False}


def test_holm_enforces_monotonicity_so_a_smaller_later_p_is_not_rejected_alone():
    # m = 3, p = 0.02, 0.03, 0.04. Step-down stops at once: 0.02 > 0.05/3.
    # Unadjusted-style reasoning would wrongly reject C: 1*0.04 = 0.04 < 0.05.
    # Running max: 3*0.02 = 0.06; max(0.06, 2*0.03 = 0.06) = 0.06; max(0.06, 0.04) = 0.06.
    out = agg.holm_bonferroni({"A": 0.02, "B": 0.03, "C": 0.04})
    assert out["adjusted"] == pytest.approx({"A": 0.06, "B": 0.06, "C": 0.06})
    assert out["rejected"] == {"A": False, "B": False, "C": False}


def test_holm_gives_tied_p_values_one_shared_adjusted_value():
    # m = 3, all p = 0.01. Thresholds 0.0166667, 0.025, 0.05: all three rejected.
    # Adjusted: 0.03; max(0.03, 0.02) = 0.03; max(0.03, 0.01) = 0.03.
    out = agg.holm_bonferroni({"A": 0.01, "B": 0.01, "C": 0.01})
    assert out["adjusted"] == pytest.approx({"A": 0.03, "B": 0.03, "C": 0.03})
    assert all(out["rejected"].values())


def test_holm_ties_above_the_first_threshold_reject_nothing():
    # m = 3, p = 0.02, 0.02, 0.9. 0.02 > 0.05/3, so the step-down stops immediately.
    # Adjusted: 0.06; max(0.06, 2*0.02 = 0.04) = 0.06; max(0.06, 0.9) = 0.9.
    out = agg.holm_bonferroni({"A": 0.02, "B": 0.02, "C": 0.9})
    assert out["adjusted"] == pytest.approx({"A": 0.06, "B": 0.06, "C": 0.9})
    assert not any(out["rejected"].values())


def test_holm_at_family_size_two_rejects_what_family_size_three_would_not():
    # The same p = 0.02 is rejected at m = 2 (0.02 < 0.05/2 = 0.025) and not at
    # m = 3 (0.02 > 0.05/3). Excluding a NOT EVALUABLE gate changes the verdict.
    # m = 2 adjusted: 2*0.02 = 0.04; max(0.04, 1*0.3) = 0.3.
    two = agg.holm_bonferroni({"A": 0.02, "B": 0.3})
    assert two["m"] == 2
    assert two["adjusted"] == pytest.approx({"A": 0.04, "B": 0.3})
    assert two["rejected"] == {"A": True, "B": False}
    three = agg.holm_bonferroni({"A": 0.02, "B": 0.3, "C": 0.4})
    assert three["rejected"]["A"] is False


def test_holm_at_family_size_one_is_an_uncorrected_comparison():
    assert agg.holm_bonferroni({"A": 0.04})["adjusted"] == pytest.approx({"A": 0.04})
    assert agg.holm_bonferroni({"A": 0.04})["rejected"] == {"A": True}
    assert agg.holm_bonferroni({"A": 0.06})["rejected"] == {"A": False}


def test_holm_clamps_adjusted_values_at_one():
    # m = 3, p = 0.5, 0.6, 0.7 -> 1.5, 1.2, 0.7 all clamped and monotonic -> 1.0.
    out = agg.holm_bonferroni({"A": 0.5, "B": 0.6, "C": 0.7})
    assert out["adjusted"] == pytest.approx({"A": 1.0, "B": 1.0, "C": 1.0})


def test_holm_rejects_strictly_below_alpha_and_not_at_it():
    # The committed rule reads "p < 0.05 after Holm-Bonferroni", strictly.
    # m = 2, p = 0.025 -> adjusted exactly 0.05, which is not below alpha.
    out = agg.holm_bonferroni({"A": 0.025, "B": 0.9})
    assert out["adjusted"]["A"] == 0.05
    assert out["rejected"]["A"] is False


def test_holm_step_down_and_adjusted_p_agree_on_every_case_here():
    cases = [{"A": 0.001, "B": 0.02, "C": 0.4}, {"A": 0.02, "B": 0.03, "C": 0.04},
             {"A": 0.01, "B": 0.01, "C": 0.01}, {"A": 0.5, "B": 0.6, "C": 0.7},
             {"A": 0.0001, "B": 0.0002, "C": 0.0003}, {"A": 0.02, "B": 0.3}]
    for case in cases:
        out = agg.holm_bonferroni(case)
        assert out["rejected"] == out["rejected_by_step_down"], case


def test_holm_on_an_empty_family_is_an_empty_result_and_not_an_error():
    out = agg.holm_bonferroni({})
    assert out["m"] == 0 and out["adjusted"] == {} and out["rejected"] == {}


def test_holm_refuses_a_p_value_outside_the_unit_interval():
    with pytest.raises(ValueError, match="p-value"):
        agg.holm_bonferroni({"A": 1.5})
    with pytest.raises(ValueError, match="p-value"):
        agg.holm_bonferroni({"A": -0.1})


def test_holm_refuses_a_non_positive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        agg.holm_bonferroni({"A": 0.01}, alpha=0.0)


# ====================================================== the combination rule

def test_all_three_gates_passing_gives_PASSES():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.5), "C": reading("C", 0.6)})
    assert out["verdict"] == agg.PASSES
    assert out["family"]["m"] == 3
    assert [g["verdict"] for g in out["gates"].values()] == [agg.GATE_PASSES] * 3


def test_one_gate_failing_gives_FAILS():
    out = combine({"A": reading("A", 0.001), "B": reading("B", 0.5), "C": reading("C", 0.6)})
    assert out["verdict"] == agg.FAILS
    assert out["gates"]["A"]["verdict"] == agg.GATE_FAILS
    assert out["gates"]["B"]["verdict"] == agg.GATE_PASSES


def test_a_gate_only_fails_after_the_correction_not_before():
    # Raw p = 0.02 is below 0.05 but its Holm-adjusted value at m = 3 is 0.06.
    out = combine({"A": reading("A", 0.02), "B": reading("B", 0.03), "C": reading("C", 0.04)})
    assert out["verdict"] == agg.PASSES
    assert out["gates"]["A"]["p_value"] == 0.02
    assert out["gates"]["A"]["adjusted_p_value"] == pytest.approx(0.06)


def test_one_evaluable_and_two_refused_gives_PARTIAL_naming_both_sides():
    out = combine({"A": reading("A", 0.4), "B": refused("B", "headroom_3_null_scaling_regime"),
                   "C": refused("C", "headroom_3_null_scaling_regime")})
    assert out["verdict"] == agg.PARTIAL
    assert out["readable_gates"] == ["A"]
    assert out["refused_gates"] == ["B", "C"]
    assert out["family"]["m"] == 1
    assert out["gates"]["B"]["verdict"] == agg.GATE_NOT_EVALUABLE
    assert out["gates"]["B"]["adjusted_p_value"] is None


def test_a_refused_gate_is_excluded_from_the_correction_family():
    # A at p = 0.02 with C refused: family size two, so 0.02 < 0.05/2 and A fails.
    out = combine({"A": reading("A", 0.02), "B": reading("B", 0.3), "C": refused("C")})
    assert out["family"]["m"] == 2
    assert out["family"]["excluded"] == ["C"]
    assert out["gates"]["A"]["adjusted_p_value"] == pytest.approx(0.04)
    assert out["gates"]["A"]["verdict"] == agg.GATE_FAILS
    assert out["verdict"] == agg.PARTIAL


def test_no_evaluable_gate_gives_NOT_EVALUABLE():
    out = combine({"A": refused("A"), "B": refused("B"), "C": refused("C")})
    assert out["verdict"] == agg.NOT_EVALUABLE
    assert out["family"]["m"] == 0
    assert out["readable_gates"] == []


def test_a_gate_missing_from_the_readings_is_refused_rather_than_dropped():
    # A PARTIAL must name which gates were readable, so a gate cannot go
    # missing from both the readable and the refused list.
    with pytest.raises(ValueError, match="every gate"):
        combine({"A": reading("A", 0.4), "B": reading("B", 0.4)})


def test_floors_that_fired_are_listed_with_their_gate_and_reason():
    out = combine({"A": refused("A", "headroom_1_degenerate_alphabet", "mean probe alphabet 3.2 < 4.0"),
                   "B": reading("B", 0.4), "C": reading("C", 0.4)})
    assert out["floors_fired"] == [{"gate": "A",
                                    "floor": "headroom_1_degenerate_alphabet",
                                    "reason": "mean probe alphabet 3.2 < 4.0"}]


# ===================================================== preconditions, BROKEN

def test_a_failed_exactness_precondition_gives_BROKEN_and_reads_no_gate():
    out = combine({"A": reading("A", 0.001), "B": reading("B", 0.001), "C": reading("C", 0.001)},
                  exactness="BROKEN")
    assert out["verdict"] == agg.BROKEN
    assert out["gates"] == {}
    assert out["family"]["m"] == 0
    assert "precondition_sampler_exactness" in out["broken_by"]


def test_a_failed_inertness_precondition_gives_BROKEN_and_reads_no_gate():
    out = combine({"A": reading("A", 0.4)}, inertness="BROKEN")
    assert out["verdict"] == agg.BROKEN
    assert out["gates"] == {}
    assert "precondition_inertness" in out["broken_by"]


def test_a_BROKEN_verdict_makes_no_transfer_statement():
    out = combine({"A": reading("A", 0.4)}, exactness="BROKEN")
    assert out["transfer_statement_permitted"] is False
    assert "BROKEN" in agg.render_report(out)
    assert "no statement is made about whether the sampler/maximizer distinction transfers" \
        in agg.render_report(out)


def test_reading_dependent_inertness_is_not_broken_and_the_gates_are_read():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)},
                  inertness="READING-DEPENDENT")
    assert out["verdict"] == agg.PASSES
    assert any("READING-DEPENDENT" in w for w in out["warnings"])


def test_an_unrun_ggn_precondition_still_reads_the_gates_and_stamps_the_wording():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)},
                  ggn="UNRUN")
    assert out["verdict"] == agg.PASSES
    assert out["required_wording"] == agg.GGN_UNRUN_WORDING
    assert agg.GGN_UNRUN_WORDING in agg.render_report(out)


def test_a_ggn_precondition_that_ran_and_failed_is_still_BROKEN():
    out = combine({"A": reading("A", 0.4)}, ggn="BROKEN")
    assert out["verdict"] == agg.BROKEN
    assert "precondition_categorical_ggn_exactness" in out["broken_by"]


def test_a_passed_ggn_precondition_stamps_no_wording():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)})
    assert out["required_wording"] is None


def test_an_unrun_exactness_precondition_stops_rather_than_guessing():
    with pytest.raises(agg.RuleSilent, match="precondition_sampler_exactness"):
        combine({"A": reading("A", 0.4)}, exactness="UNRUN")


def test_an_unrun_inertness_precondition_stops_rather_than_guessing():
    with pytest.raises(agg.RuleSilent, match="precondition_inertness"):
        combine({"A": reading("A", 0.4)}, inertness="UNRUN")


def test_an_unknown_precondition_verdict_string_is_refused():
    with pytest.raises(ValueError, match="verdict"):
        combine({"A": reading("A", 0.4)}, inertness="probably fine")


def test_the_inertness_verdict_may_be_given_as_the_runner_payload():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)},
                  inertness={"verdict": "PASS", "cells": []})
    assert out["verdict"] == agg.PASSES


# ========================================================= MC-dropout control

def test_mc_dropout_reproduction_withdraws_a_PASSES():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)}, mc=True)
    assert out["verdict"] == agg.WITHDRAWN
    assert out["withdrawn"] is True
    assert out["verdict_before_withdrawal"] == agg.PASSES
    assert out["mc_dropout_reproduced"] is True


def test_mc_dropout_reproduction_withdraws_a_FAILS_and_a_PARTIAL_too():
    fails = combine({"A": reading("A", 0.0001), "B": reading("B", 0.4), "C": reading("C", 0.4)}, mc=True)
    assert fails["verdict"] == agg.WITHDRAWN and fails["verdict_before_withdrawal"] == agg.FAILS
    part = combine({"A": reading("A", 0.4), "B": refused("B"), "C": refused("C")}, mc=True)
    assert part["verdict"] == agg.WITHDRAWN and part["verdict_before_withdrawal"] == agg.PARTIAL


def test_mc_dropout_reproduction_does_not_overturn_BROKEN():
    out = combine({"A": reading("A", 0.4)}, exactness="BROKEN", mc=True)
    assert out["verdict"] == agg.BROKEN
    assert out["mc_dropout_reproduced"] is True
    assert out["withdrawn"] is False


def test_no_mc_dropout_reproduction_leaves_the_verdict_alone():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)}, mc=False)
    assert out["verdict"] == agg.PASSES and out["withdrawn"] is False


# ============================================================ cell coverage

def test_partial_coverage_sets_coverage_limited_without_changing_the_family():
    full = combine({"A": reading("A", 0.02), "B": reading("B", 0.3), "C": reading("C", 0.4)})
    limited = combine({"A": reading("A", 0.02), "B": reading("B", 0.3), "C": reading("C", 0.4)},
                      cov=partial_coverage())
    assert full["coverage_limited"] is False
    assert limited["coverage_limited"] is True
    assert limited["family"]["m"] == full["family"]["m"] == 3
    assert limited["gates"]["A"]["adjusted_p_value"] == full["gates"]["A"]["adjusted_p_value"]
    assert limited["verdict"] == full["verdict"]


def test_coverage_limited_names_the_cells_completed_and_the_cells_not_run():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)},
                  cov=partial_coverage())
    assert out["coverage"]["completed"] == [[100.0, 4.0], [100.0, 1.0]]
    assert out["coverage"]["incomplete"] == [
        {"cell": [10.0, 4.0], "n_founders": 0}, {"cell": [10.0, 1.0], "n_founders": 0},
        {"cell": [1.0, 4.0], "n_founders": 0}, {"cell": [1.0, 1.0], "n_founders": 0}]
    assert any("COVERAGE-LIMITED" in w for w in out["warnings"])
    assert "COVERAGE-LIMITED" in agg.render_report(out)


def test_the_committed_cell_order_is_lambda_100_10_1_and_4x_sigma_before_sigma_match():
    assert agg.COMMITTED_CELLS == ((100.0, 4.0), (100.0, 1.0), (10.0, 4.0),
                                   (10.0, 1.0), (1.0, 4.0), (1.0, 1.0))


def test_coverage_reports_a_cell_with_too_few_founders_as_incomplete():
    cov = agg.coverage_from_cell_keys(agg.COMMITTED_CELLS[:1], n_founders={agg.COMMITTED_CELLS[0]: 12})
    assert cov["completed"] == []
    assert cov["incomplete"][0] == {"cell": [100.0, 4.0], "n_founders": 12}
    assert cov["coverage_limited"] is True


def test_coverage_refuses_a_cell_that_is_not_on_the_committed_grid():
    with pytest.raises(ValueError, match="committed grid"):
        agg.coverage_from_cell_keys(((5.0, 1.0),), n_founders={(5.0, 1.0): 20})


# ================================================ gate C is not interchangeable

def test_gate_c_passing_beside_a_and_b_triggers_the_not_the_theorys_prediction_warning():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)})
    assert out["verdict"] == agg.PASSES
    assert any("not the theory's prediction" in w for w in out["warnings"])
    assert "not the theory's prediction" in agg.render_report(out)


def test_gate_c_failing_beside_passing_a_and_b_raises_no_such_warning():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.0001)})
    assert out["verdict"] == agg.FAILS
    assert not any("not the theory's prediction" in w for w in out["warnings"])


def test_every_partial_verdict_says_gate_c_is_not_interchangeable():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": refused("C")})
    assert out["verdict"] == agg.PARTIAL
    assert any("not interchangeable" in w for w in out["warnings"])
    assert any("the founder was forgotten" in w for w in out["warnings"])


def test_a_partial_without_gate_c_refused_still_says_it():
    out = combine({"A": refused("A"), "B": reading("B", 0.4), "C": reading("C", 0.4)})
    assert out["verdict"] == agg.PARTIAL
    assert any("not interchangeable" in w for w in out["warnings"])


# ============================================================= consequences

def test_the_consequence_carried_for_a_gate_follows_its_verdict():
    cons = {"A": {"passed": "A-passed text", "failed": "A-failed text"},
            "B": {"passed": "B-passed text", "failed": "B-failed text"},
            "C": {"passed": "C-passed text", "failed": "C-failed text"}}
    out = combine({"A": reading("A", 0.0001), "B": reading("B", 0.9), "C": refused("C")},
                  consequences=cons)
    assert out["consequences"]["A"] == "A-failed text"
    assert out["consequences"]["B"] == "B-passed text"
    assert "C" not in out["consequences"]


def test_consequences_can_be_read_from_the_committed_rule_file():
    cons = agg.consequences_from_rule()
    assert set(cons) == {"A", "B", "C"}
    assert cons["C"]["passed"].startswith("sigma does not define an attractor")


# ======================================================== gate result shapes

def test_a_gate_result_dict_in_the_sibling_contract_shape_normalises():
    out = agg.normalise_gate_result("A", {"observed": 0.31, "p_value": 0.02, "null_mean": 0.05,
                                          "null_sd": 0.01, "null_min": 0.0, "null_max": 0.2,
                                          "n_perm": 10000, "seed": 0, "chance_level": 0.05,
                                          "n_chains": 40, "n_founders": 20})
    assert out.name == "A" and out.observed == 0.31 and out.p_value == 0.02
    assert out.evaluable is True and out.detail["null_mean"] == 0.05


def test_a_NullTestResult_from_overlap_stats_normalises_for_gate_b():
    import numpy as np
    from analysis import overlap_stats as ov
    res = ov.cross_arm_permutation_test([{1, 2}, {2, 3}, {1, 3}], [{4, 5}, {5, 6}, {4, 6}],
                                        np.random.default_rng(0), n_perm=50,
                                        alternative="greater")
    out = agg.normalise_gate_result("B", res)
    assert out.p_value == res.p_value and out.observed == pytest.approx(res.observed)
    assert out.detail["null_mean"] == pytest.approx(float(res.null_samples.mean()))
    assert out.detail["n_perm"] == 50


def test_normalising_a_result_without_a_p_value_is_refused():
    with pytest.raises(ValueError, match="p_value"):
        agg.normalise_gate_result("A", {"observed": 0.3})


def test_a_gate_reading_cannot_be_evaluable_and_refused_at_once():
    with pytest.raises(ValueError, match="refus"):
        agg.GateReading(name="A", observed=0.1, p_value=0.2, evaluable=True,
                        refusals=(("floor", "why"),), detail={})


def test_an_evaluable_gate_reading_needs_a_p_value():
    with pytest.raises(ValueError, match="p_value"):
        agg.GateReading(name="A", observed=0.1, p_value=None, evaluable=True,
                        refusals=(), detail={})


# =================================================== floors, through stubs
#
# Floor convention: the predicates return (fires, reason) and True means THE
# FLOOR FIRED, so the gate it guards is NOT EVALUABLE. A pass-through stub is
# therefore (False, ...) and a refusal is (True, ...).

SETS = [{1, 2}, {2, 3}, {1, 3}, {3, 4}]
FOUNDERS = [0, 0, 1, 1]
SIGMAS = [1.0, 1.0, 4.0, 4.0]


def gate_result(p, observed=0.3, **extra):
    """A gate result in the sibling contract's shape."""
    return dict({"observed": observed, "p_value": p, "null_mean": 0.05,
                 "null_sd": 0.01, "null_min": 0.0, "null_max": 0.2,
                 "n_perm": 10000, "seed": 0, "chance_level": 0.05,
                 "n_chains": 4, "n_founders": 2}, **extra)


def read_gates(**kw):
    """read_gates with every stub defaulted to "computed, no floor fired"."""
    defaults = dict(
        survivor_sets=SETS, founder_labels=FOUNDERS, sigma_labels=SIGMAS,
        mean_alphabet_gen25=9.0,
        gate_a=lambda *a, **k: gate_result(0.4),
        gate_b=lambda *a, **k: gate_result(0.4, observed=0.1),
        gate_c=lambda *a, **k: gate_result(0.4, observed=0.1),
        floor_1=lambda mean: (False, f"mean surviving alphabet {mean} is at or above 4.0"),
        floor_2=lambda *a: (False, "the null sits at the chance level"),
        floor_3=lambda *a: (False, "both nulls are in the same scaling regime"),
        scaling_regime=lambda sets: {"n": len(sets), "zeta_1": 0.01, "regime": "1/sqrt(n)"},
    )
    return agg.read_gates(**dict(defaults, **kw))


def test_floor_1_firing_makes_gate_a_not_evaluable():
    gates = read_gates(
        mean_alphabet_gen25=3.2,
        gate_a=lambda *a, **k: pytest.fail("gate A must not be computed under floor 1"),
        floor_1=lambda mean: (True, f"NOT EVALUABLE (headroom floor 1: mean surviving "
                                    f"alphabet {mean} is below 4.0)"))
    assert gates["A"].evaluable is False
    assert gates["A"].refusals[0][0] == "headroom_1_degenerate_alphabet"
    assert "below 4.0" in gates["A"].refusals[0][1]
    assert gates["B"].evaluable is True and gates["C"].evaluable is True


def test_floor_2_firing_makes_gate_a_not_evaluable():
    gates = read_gates(
        gate_a=lambda *a, **k: gate_result(0.01, null_mean=0.09),
        floor_2=lambda *a: (True, "NOT EVALUABLE (headroom floor 2: null mean 0.09 is "
                                  "more than one of its own sds from 0.05)"))
    assert gates["A"].evaluable is False
    assert gates["A"].refusals[0][0] == "headroom_2_chance_level"
    assert gates["A"].detail["raw_p_value"] == 0.01


def test_floor_3_firing_makes_gates_b_and_c_not_evaluable_but_not_gate_a():
    gates = read_gates(
        floor_3=lambda a, b: (True, f"NOT EVALUABLE (headroom floor 3: {a['regime']} "
                                    f"against {b['regime']})"),
        scaling_regime=lambda sets: {"n": len(sets), "zeta_1": 0.0, "regime": "1/n"})
    assert gates["A"].evaluable is True
    for name in ("B", "C"):
        assert gates[name].evaluable is False
        assert gates[name].refusals[0][0] == "headroom_3_null_scaling_regime"


def test_a_floor_that_fires_removes_its_gate_from_the_holm_family():
    # The behaviour a polarity inversion would silently break: gate A at a raw
    # p of 0.0001 must not reach a verdict at all once floor 1 has fired, and
    # the family must shrink to B and C, which changes their adjusted values.
    gates = read_gates(
        mean_alphabet_gen25=3.2,
        gate_a=lambda *a, **k: gate_result(0.0001),
        floor_1=lambda mean: (True, f"NOT EVALUABLE (headroom floor 1: {mean} < 4.0)"),
        gate_b=lambda *a, **k: gate_result(0.02, observed=0.31),
        gate_c=lambda *a, **k: gate_result(0.3, observed=0.05))
    out = combine(gates)
    assert out["gates"]["A"]["verdict"] == agg.GATE_NOT_EVALUABLE
    assert out["gates"]["A"]["p_value"] is None
    assert out["gates"]["A"]["adjusted_p_value"] is None
    assert out["family"]["m"] == 2 and out["family"]["members"] == ["B", "C"]
    assert out["family"]["excluded"] == ["A"]
    # m = 2: 2 * 0.02 = 0.04 < 0.05, so B fails where at m = 3 it would not.
    assert out["gates"]["B"]["adjusted_p_value"] == pytest.approx(0.04)
    assert out["gates"]["B"]["verdict"] == agg.GATE_FAILS
    assert out["verdict"] == agg.PARTIAL
    assert out["floors_fired"][0]["floor"] == "headroom_1_degenerate_alphabet"


def test_a_floor_reporting_no_fire_while_its_reason_says_not_evaluable_is_refused():
    # Catches a future re-inversion at runtime instead of in a verdict.
    with pytest.raises(ValueError, match="polarity"):
        read_gates(floor_1=lambda mean: (False, "NOT EVALUABLE (headroom floor 1)"))


def test_gate_c_is_not_evaluable_when_the_completed_coverage_holds_one_sigma():
    gates = read_gates(
        sigma_labels=[1.0, 1.0, 1.0, 1.0],
        gate_c=lambda *a, **k: pytest.fail("gate C must not be computed on one sigma"))
    assert gates["C"].evaluable is False
    assert gates["C"].refusals[0][0] == "across_sigma_contrast_undefined"
    assert gates["A"].evaluable is True and gates["B"].evaluable is True


def test_read_gates_passes_the_committed_n_perm_and_seed_to_every_gate():
    seen = {}

    def spy(key):
        def fn(*args, **kwargs):
            seen[key] = kwargs
            return gate_result(0.4)
        return fn

    read_gates(gate_a=spy("A"), gate_b=spy("B"), gate_c=spy("C"))
    for key in ("A", "B", "C"):
        assert seen[key]["n_perm"] == 10000 and seen[key]["seed"] == 0


def test_gate_b_has_no_default_because_its_arms_cannot_be_assembled_from_the_rule():
    with pytest.raises(agg.RuleSilent, match="same-founder"):
        read_gates(gate_b=None)


def test_a_floor_returning_something_other_than_a_bool_and_a_string_is_refused():
    with pytest.raises(ValueError, match="floor"):
        read_gates(floor_1=lambda mean: "nope")


def test_read_gates_refuses_labels_that_do_not_line_up_with_the_survivor_sets():
    with pytest.raises(ValueError, match="same length"):
        read_gates(founder_labels=[0, 0, 1])


# ============================================================= chain logs

def test_a_chain_log_is_read_into_a_cell_with_its_arm_lambda_sigma_and_pool_seed(tmp_path):
    write_cell(tmp_path, "l100_s4_p0", chain_payload(100.0, 4.0, 0))
    cells = agg.load_cells(str(tmp_path))
    assert len(cells) == 1
    cell = cells[0]
    assert (cell.arm, cell.lam, cell.sigma_mult, cell.pool_seed) == ("sampler", 100.0, 4.0, 0)
    assert len(cell.chains) == 2
    assert cell.chains[0].survivors_training == frozenset({1, 2, 3})
    assert cell.chains[0].probe_sizes[-1] == 9


def test_load_cells_selects_one_arm_at_a_time(tmp_path):
    write_cell(tmp_path, "samp", chain_payload(100.0, 4.0, 0, arm="sampler"))
    write_cell(tmp_path, "drop", chain_payload(100.0, 4.0, 0, arm="dropout"))
    assert len(agg.load_cells(str(tmp_path), arm="sampler")) == 1
    assert agg.load_cells(str(tmp_path), arm="dropout")[0].arm == "dropout"


def test_load_cells_ignores_logs_that_are_not_chain_logs(tmp_path):
    (tmp_path / "s1_precondition_inertness.json").write_text(json.dumps({"stage": "x"}))
    write_cell(tmp_path, "l100_s4_p0", chain_payload(100.0, 4.0, 0))
    assert len(agg.load_cells(str(tmp_path))) == 1


def test_a_chain_log_off_the_committed_lambda_grid_is_refused(tmp_path):
    write_cell(tmp_path, "bad", chain_payload(50.0, 4.0, 0))
    with pytest.raises(ValueError, match="committed grid"):
        agg.load_cells(str(tmp_path))


def test_a_chain_log_with_other_than_two_chains_per_founder_is_refused(tmp_path):
    write_cell(tmp_path, "bad", chain_payload(100.0, 4.0, 0, chain_seeds=(600,),
                                              probe_final=(9,), survivors=((1, 2),)))
    with pytest.raises(ValueError, match="R = 2"):
        agg.load_cells(str(tmp_path))


def test_a_duplicate_founder_in_one_cell_is_refused(tmp_path):
    write_cell(tmp_path, "a", chain_payload(100.0, 4.0, 0))
    write_cell(tmp_path, "b", chain_payload(100.0, 4.0, 0))
    with pytest.raises(ValueError, match="pool seed 0"):
        agg.load_cells(str(tmp_path))


def test_floor_1_input_is_the_probe_pool_alphabet_at_generation_25(tmp_path):
    write_cell(tmp_path, "a", chain_payload(100.0, 4.0, 0, probe_final=(9, 11)))
    write_cell(tmp_path, "b", chain_payload(100.0, 4.0, 1, probe_final=(3, 5)))
    cells = agg.load_cells(str(tmp_path))
    assert agg.mean_alphabet_gen25(cells) == pytest.approx((9 + 11 + 3 + 5) / 4)


def test_asking_for_probe_pool_survivor_sets_that_the_runner_never_wrote_is_refused(tmp_path):
    write_cell(tmp_path, "a", chain_payload(100.0, 4.0, 0))
    cells = agg.load_cells(str(tmp_path))
    with pytest.raises(agg.ProbeSurvivorsMissing, match="probe_survivors"):
        agg.survivor_sets(cells, pool=agg.PROBE_POOL)


def test_probe_pool_survivor_sets_are_used_when_the_runner_does_write_them(tmp_path):
    write_cell(tmp_path, "a", chain_payload(100.0, 4.0, 0, probe_survivors=((7, 8), (8, 9))))
    cells = agg.load_cells(str(tmp_path))
    sets, founders, sigmas = agg.survivor_sets(cells, pool=agg.PROBE_POOL)
    assert sets == [frozenset({7, 8}), frozenset({8, 9})]
    assert founders == [0, 0] and sigmas == [4.0, 4.0]


def test_training_pool_survivor_sets_must_be_asked_for_by_name(tmp_path):
    write_cell(tmp_path, "a", chain_payload(100.0, 4.0, 0))
    cells = agg.load_cells(str(tmp_path))
    sets, founders, sigmas = agg.survivor_sets(cells, pool=agg.TRAINING_POOL)
    assert sets == [frozenset({1, 2, 3}), frozenset({2, 3, 4})]
    with pytest.raises(ValueError, match="pool"):
        agg.survivor_sets(cells, pool="whichever")


def test_coverage_is_computed_from_loaded_cells(tmp_path):
    for seed in range(20):
        write_cell(tmp_path, f"l100_s4_p{seed}", chain_payload(100.0, 4.0, seed))
    for seed in range(3):
        write_cell(tmp_path, f"l100_s1_p{seed}", chain_payload(100.0, 1.0, seed))
    cov = agg.coverage(agg.load_cells(str(tmp_path)))
    assert cov["completed"] == [[100.0, 4.0]]
    assert {"cell": [100.0, 1.0], "n_founders": 3} in cov["incomplete"]
    assert cov["coverage_limited"] is True


# ====================================================== inertness re-reading

def inertness_payload(verdict, fixed, full):
    return {"stage": "precondition_inertness", "verdict": verdict,
            "cells": [{"lambda": 100.0, "sigma_multiplier": 4.0,
                       "fixed_map": {"exceeds_maximizer_non_overlapping": f},
                       "per_draw_map": {"exceeds_maximizer_non_overlapping": u}}
                      for f, u in zip(fixed, full)]}


def test_inertness_is_reread_from_its_cells_under_the_amendments_definition():
    out = agg.read_inertness(inertness_payload("READING-DEPENDENT", [True, True], [True, False]))
    assert out["verdict"] == "READING-DEPENDENT"
    assert out["verdict_logged"] == "READING-DEPENDENT"
    assert out["agrees_with_log"] is True
    assert out["failing_cells_verdict_bearing"] == []


def test_a_verdict_bearing_failure_is_BROKEN_even_where_the_runner_said_reading_dependent():
    # The runner emits READING-DEPENDENT when the two readings merely differ; the
    # amendment defines it only as verdict-bearing passed and per-draw-MAP did not.
    out = agg.read_inertness(inertness_payload("READING-DEPENDENT", [False, True], [True, True]))
    assert out["verdict"] == "BROKEN"
    assert out["verdict_logged"] == "READING-DEPENDENT"
    assert out["agrees_with_log"] is False
    assert out["failing_cells_verdict_bearing"] == [[100.0, 4.0]]


def test_inertness_passing_on_every_cell_under_both_readings_is_PASS():
    out = agg.read_inertness(inertness_payload("PASS", [True, True], [True, True]))
    assert out["verdict"] == "PASS" and out["agrees_with_log"] is True


def test_a_disagreement_between_log_and_cells_is_reported_in_the_verdict(tmp_path):
    inert = agg.read_inertness(inertness_payload("READING-DEPENDENT", [False, True], [True, True]))
    out = combine({"A": reading("A", 0.4)}, inertness=inert)
    assert out["verdict"] == agg.BROKEN
    assert any("disagrees with its own cells" in w for w in out["warnings"])


def test_inertness_without_cells_falls_back_to_the_logged_verdict():
    out = agg.read_inertness({"verdict": "PASS"})
    assert out["verdict"] == "PASS" and out["verdict_from_cells"] is None


# ================================================================== report

def test_the_report_names_the_control_the_verdict_and_the_gate_lines():
    out = combine({"A": reading("A", 0.4, observed=0.062),
                   "B": reading("B", 0.0001, observed=0.31),
                   "C": refused("C", "headroom_3_null_scaling_regime", "different regimes")})
    text = agg.render_report(out)
    assert text.startswith("S-1 rule committed 2026-10-20")
    assert "=== S-1 GATES ===" in text
    assert "gate A" in text and "gate B" in text and "gate C" in text
    assert "NOT EVALUABLE" in text
    assert "floor fired: headroom_3_null_scaling_regime" in text
    assert "headroom_3_null_scaling_regime: different regimes" in text
    assert f"OUTCOME: {agg.PARTIAL}" in text


def test_the_report_names_the_holm_family_size():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": refused("C")})
    assert "Holm-Bonferroni family: 2 evaluable gates (A, B); excluded: C" in agg.render_report(out)


def test_the_report_of_a_withdrawn_verdict_names_what_was_withdrawn():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)}, mc=True)
    text = agg.render_report(out)
    assert f"OUTCOME: {agg.WITHDRAWN}" in text
    assert "MC dropout reproduced" in text
    assert agg.PASSES in text


def test_the_report_lists_the_cells_completed_and_not_run():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)},
                  cov=partial_coverage())
    text = agg.render_report(out)
    assert "cells completed  : lambda 100 sigma x4, lambda 100 sigma x1" in text
    assert "cells not run    : lambda 10 sigma x4, lambda 10 sigma x1, " \
           "lambda 1 sigma x4, lambda 1 sigma x1" in text


def test_the_report_is_plain_text_without_a_trailing_blank_run():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.4), "C": reading("C", 0.4)})
    text = agg.render_report(out)
    assert not text.endswith("\n\n")
    assert all(not line.rstrip() or line == line.rstrip() for line in text.splitlines())


# ============================================================== entry point

def test_aggregate_reads_the_completed_cells_only_and_renders_its_report(tmp_path):
    # lambda 100 at both sigmas, all 20 founders; lambda 10 x4 with 2 founders.
    for mult in (4.0, 1.0):
        for seed in range(20):
            write_cell(tmp_path, f"l100_s{mult:g}_p{seed}",
                       chain_payload(100.0, mult, seed,
                                     probe_survivors=((seed, seed + 1), (seed + 1, seed + 2))))
    for seed in range(2):
        write_cell(tmp_path, f"l10_s4_p{seed}",
                   chain_payload(10.0, 4.0, seed, probe_survivors=((1, 2), (2, 3))))

    seen = {}

    def spy(key):
        def fn(*args, **kwargs):
            seen[key] = {"n_sets": len(args[0]), "founders": sorted(set(args[1]))}
            return gate_result(0.4)
        return fn

    verdict, text = agg.aggregate(
        str(tmp_path), exactness="PASS", inertness="PASS", ggn="UNRUN",
        gate_a=spy("A"), gate_b=spy("B"), gate_c=spy("C"),
        floor_1=lambda mean: (False, "alphabet at or above 4.0"),
        floor_2=lambda *a: (False, "null at chance"),
        floor_3=lambda *a: (False, "same regime"),
        scaling_regime=lambda sets: {"n": len(sets), "zeta_1": 0.01, "regime": "1/sqrt(n)"})

    # 2 completed cells x 20 founders x R = 2 chains; the lambda 10 cell is excluded.
    assert seen["A"]["n_sets"] == 80
    assert seen["A"]["founders"] == list(range(20))
    assert verdict["coverage"]["completed"] == [[100.0, 4.0], [100.0, 1.0]]
    assert verdict["coverage_limited"] is True
    assert verdict["verdict"] == agg.PASSES
    assert verdict["required_wording"] == agg.GGN_UNRUN_WORDING
    assert "OUTCOME: PASSES" in text


def test_aggregate_on_a_logs_directory_with_no_completed_cell_is_NOT_EVALUABLE(tmp_path):
    write_cell(tmp_path, "l100_s4_p0", chain_payload(100.0, 4.0, 0))
    verdict, text = agg.aggregate(str(tmp_path), exactness="PASS", inertness="PASS",
                                  ggn="PASS", gate_b=lambda *a, **k: gate_result(0.4))
    assert verdict["verdict"] == agg.NOT_EVALUABLE
    assert verdict["gates"] == {}
    assert "no gate read" in text


def test_the_verdict_round_trips_through_json():
    out = combine({"A": reading("A", 0.4), "B": reading("B", 0.0001), "C": refused("C")},
                  cov=partial_coverage(), ggn="UNRUN")
    assert json.loads(json.dumps(out))["verdict"] == agg.PARTIAL
