"""The S-1 sweep driver reads its plan from the rule and invents nothing.

These tests exist because the driver's only job is to turn committed text into
240 shell invocations; a silent disagreement between the two is exactly the
failure the rule's amendments were written to prevent.
"""

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from controls import s1_sweep_driver as drv  # noqa: E402


@pytest.fixture(scope="module")
def rule():
    return drv.load_rule()


def test_cell_order_is_lambda_descending_and_four_sigma_first(rule):
    # Arrange / Act
    cells = drv.committed_cells(rule)

    # Assert
    assert cells == [(100.0, 4.0), (100.0, 1.0),
                     (10.0, 4.0), (10.0, 1.0),
                     (1.0, 4.0), (1.0, 1.0)]


def test_chain_seeds_follow_the_committed_formula(rule):
    assert drv.chain_seeds(0, rule) == [7700, 7701]
    assert drv.chain_seeds(19, rule) == [7738, 7739]


def test_every_founder_gets_two_distinct_unshared_chain_seeds(rule):
    n = int(rule["fixed_in_advance"]["n_founders_per_arm"])
    seeds = [s for pool in range(n) for s in drv.chain_seeds(pool, rule)]

    assert len(seeds) == 2 * n
    assert len(set(seeds)) == len(seeds)


def test_chain_seeds_do_not_collide_with_a_seed_already_spent(rule):
    from controls import s1_sampler_arm as arm

    spent = {arm.SEED_TIMING_CHAIN, arm.SEED_INERT_BATCH, arm.SEED_INERT_MAXIMIZER,
             arm.SEED_INERT_MAP_INIT, arm.SEED_INERT_MAP_INITS, arm.SEED_INERT_DRAWS,
             arm.SEED_EXACT_DRAWS, arm.SEED_GGN_MAP_INIT, arm.SEED_GGN_DRAWS,
             arm.SEED_GGN_DIRECTIONS, arm.PROBE_SEED}
    n = int(rule["fixed_in_advance"]["n_founders_per_arm"])
    seeds = {s for pool in range(n) for s in drv.chain_seeds(pool, rule)}

    assert seeds.isdisjoint(spent)


def test_plan_is_one_invocation_per_cell_and_founder(rule):
    n = int(rule["fixed_in_advance"]["n_founders_per_arm"])
    plan = drv.plan("sampler", rule, n)

    assert len(plan) == 6 * n
    assert len(set(plan)) == len(plan)
    assert plan[0] == (100.0, 4.0, 0)
    assert plan[-1] == (1.0, 1.0, n - 1)


def test_every_tag_in_the_plan_is_distinct(rule):
    """12 of the 14 runners clobber their logs; --tag is the only guard."""
    n = int(rule["fixed_in_advance"]["n_founders_per_arm"])
    tags = [drv.tag(arm, lam, mult, pool)
            for arm in ("sampler", "dropout")
            for lam, mult, pool in drv.plan(arm, rule, n)]

    assert len(set(tags)) == len(tags) == 2 * 6 * n


def test_dropout_invocation_carries_the_committed_rate(rule):
    cmd = drv.invocation("dropout", 100.0, 4.0, 3, rule)

    assert "--dropout-p" in cmd
    assert cmd[cmd.index("--dropout-p") + 1] == "0.1"
    assert cmd[cmd.index("--tag") + 1] == "dropout_lam100_sig4_pool3"
    assert cmd[cmd.index("--chain-seeds") + 1:cmd.index("--chain-seeds") + 3] == ["7706", "7707"]


def test_sampler_invocation_passes_no_dropout_rate(rule):
    cmd = drv.invocation("sampler", 1.0, 1.0, 19, rule)

    assert "--dropout-p" not in cmd
    assert cmd[cmd.index("--tag") + 1] == "sampler_lam1_sig1_pool19"


def test_invocation_uses_the_torch_280_interpreter(rule):
    assert drv.invocation("sampler", 10.0, 4.0, 0, rule)[0] == "/usr/bin/python3"


def test_cell_order_refuses_if_the_committed_text_changes(rule):
    broken = json.loads(json.dumps(rule))
    broken["design_amendment_2026_11_08"]["cell_coverage"]["rule"] = ["lambda 1 then 10 then 100"]

    with pytest.raises(SystemExit):
        drv.committed_cells(broken)


def test_chain_seeds_refuse_if_the_committed_formula_changes(rule):
    broken = json.loads(json.dumps(rule))
    broken["recording_amendment_2026_11_08"]["chain_seeds"]["rule"] = "pick any two seeds"

    with pytest.raises(SystemExit):
        drv.chain_seeds(0, broken)
