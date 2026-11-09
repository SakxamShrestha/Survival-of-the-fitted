"""run_chain records the pool the gates are read on.

design_amendment_2026_11_08.pool_for_gates_and_floor_1 puts gates A, B and C on
the held-out probe pool, and gates A and B are statistics of survivor SETS. The
runner recorded that pool as sizes only, so a swept chain could not be gated on
the pool the rule names. These tests pin the recorded shape with the learning
step stubbed out, so they cost no training.
"""

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from controls import s1_aggregate as agg  # noqa: E402
from controls import s1_sampler_arm as arm  # noqa: E402
from spike_iterated_distill import MLP  # noqa: E402

SIGMA_MATCH = {"body.0.weight": 0.0817, "body.2.weight": 0.036, "head.weight": 0.0361}


@pytest.fixture
def stubbed_chain(monkeypatch):
    """run_chain with no learning: the student is the teacher, untouched."""
    def student(*args, **kwargs):
        return stubbed_chain.model, {"max_abs_grad": 0.0}

    torch.manual_seed(0)
    stubbed_chain.model = MLP(6, 5, h=8)
    monkeypatch.setattr(arm, "sampler_student", student)
    monkeypatch.setattr(arm, "dropout_student", student)
    return stubbed_chain.model


def run(arm_name, model, Xp):
    torch.manual_seed(0)
    X = torch.randn(40, 6)
    return arm.run_chain(arm_name, model, X, 5, 100.0, 4.0, 7700, 2, 8,
                         SIGMA_MATCH, p_drop=0.1 if arm_name == "dropout" else None, Xp=Xp)


def test_chain_records_the_probe_pools_survivor_set(stubbed_chain):
    Xp = torch.randn(30, 6)

    result = run("sampler", stubbed_chain, Xp)

    assert result["probe_survivors"] == sorted(
        int(c) for c in stubbed_chain(Xp).argmax(-1).unique().tolist())
    assert len(result["probe_survivors"]) == result["probe_sizes"][-1]


def test_training_pool_survivors_are_still_recorded(stubbed_chain):
    result = run("sampler", stubbed_chain, torch.randn(30, 6))

    assert result["survivors"] == sorted(set(result["survivors"]))
    assert len(result["survivors"]) == result["sizes"][-1]


def test_probe_survivors_is_none_when_no_probe_pool_is_passed(stubbed_chain):
    result = run("sampler", stubbed_chain, None)

    assert result["probe_survivors"] is None
    assert result["probe_sizes"] == []


def test_the_dropout_arm_records_the_same_keys(stubbed_chain):
    result = run("dropout", stubbed_chain, torch.randn(30, 6))

    assert result["probe_survivors"] is not None
    assert result["p_drop"] == 0.1


def test_the_aggregator_can_read_a_chain_this_shape_on_the_probe_pool(stubbed_chain, tmp_path):
    """The gate side of the contract, not just the runner side."""
    Xp = torch.randn(30, 6)
    chains = [run("sampler", stubbed_chain, Xp) for _ in range(agg.CHAINS_PER_FOUNDER)]
    for i, chain in enumerate(chains):
        chain["chain_seed"] = 7700 + i
    payload = {"control": "S-1", "stage": "chain", "arm": "sampler", "lambda": 100.0,
               "sigma_multiplier": 4.0, "pool_seed": 0, "founder_acc": 0.977294921875,
               "gens": 2, "bottleneck": 8, "chains": chains}
    import json
    (tmp_path / "s1_chain_sampler_lam100_sig4_pool0.json").write_text(json.dumps(payload))

    cells = agg.load_cells(str(tmp_path), arm="sampler")
    sets, founders, sigmas = agg.survivor_sets(cells, agg.PROBE_POOL)

    assert founders == [0, 0] and sigmas == [4.0, 4.0]
    assert all(s == frozenset(chains[0]["probe_survivors"]) for s in sets)
