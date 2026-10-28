"""
Tests for controls/c8b_shared_founder.py's pure decision logic.

Synthetic data only: every log fragment and interval here is hand-constructed
for the test, none is read from logs/. The training loop itself is not unit
tested here; its correctness checks (founder identity, gen-1 alphabet equality,
freeze-leak checks) run inside the control and are reported in its log.

    python3 -m pytest tests/test_c8b_shared_founder.py -v
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from controls import c8b_shared_founder as c8b


def fake_c1_log(n=20, offset=600):
    """A minimal C-1 log with the fields the mapping reads."""
    runs = [{"seed": s, "chain_seed": offset + s, "founder_acc": 0.9 + s / 1000,
             "sizes": [49] * 24 + [s % 5 + 1], "final": s % 5 + 1}
            for s in range(n)]
    return {"pool_seeds": list(range(n)), "chain_seed_offset": offset,
            "results": {"intact": runs, "shuffled": runs}}


# ------------------------------------------------------------ seed range

def test_parse_seed_range_is_inclusive_of_both_ends():
    # Act
    seeds = c8b.parse_seed_range("603-619")

    # Assert
    assert seeds == tuple(range(603, 620))
    assert len(seeds) == 17


def test_parse_seed_range_rejects_reversed_range():
    with pytest.raises(ValueError):
        c8b.parse_seed_range("619-603")


# ------------------------------------------------------------ seed mapping

def test_chain_to_pool_maps_each_chain_seed_to_its_pool_seed():
    # Arrange
    log = fake_c1_log()

    # Act
    mapping = c8b.chain_to_pool_seed(log)

    # Assert
    assert mapping[600] == 0
    assert mapping[603] == 3
    assert mapping[619] == 19
    assert len(mapping) == 20


def test_chain_to_pool_rejects_a_record_inconsistent_with_the_offset():
    # Arrange: one record whose chain seed is not offset + pool seed.
    log = fake_c1_log()
    log["results"]["intact"][5] = dict(log["results"]["intact"][5], chain_seed=999)

    # Act / Assert
    with pytest.raises(ValueError):
        c8b.chain_to_pool_seed(log)


def test_chain_to_pool_rejects_pool_seeds_missing_from_results():
    # Arrange
    log = fake_c1_log()
    log = dict(log, pool_seeds=list(range(21)))

    # Act / Assert
    with pytest.raises(ValueError):
        c8b.chain_to_pool_seed(log)


# ------------------------------------------------------------ subset

def test_gated_subset_selects_seventeen_chains_and_excludes_disclosed():
    # Arrange
    records = fake_c1_log()["results"]["intact"]
    gated = c8b.parse_seed_range("603-619")

    # Act
    subset = c8b.select_chains(records, gated)

    # Assert
    chain_seeds = [r["chain_seed"] for r in subset]
    assert chain_seeds == list(range(603, 620))
    assert not {600, 601, 602} & set(chain_seeds)


def test_gated_subset_preserves_order_of_the_requested_seeds():
    # Arrange
    records = list(reversed(fake_c1_log()["results"]["intact"]))

    # Act
    subset = c8b.select_chains(records, (603, 604, 605))

    # Assert
    assert [r["chain_seed"] for r in subset] == [603, 604, 605]


def test_gated_subset_raises_when_a_requested_seed_is_absent():
    # Arrange
    records = fake_c1_log()["results"]["intact"][:10]

    # Act / Assert
    with pytest.raises(ValueError):
        c8b.select_chains(records, c8b.parse_seed_range("603-619"))


# ------------------------------------------------------------ founder identity

def test_founder_identity_passes_on_exact_float_equality():
    # Arrange
    logged = {0: 0.977294921875, 1: 0.979736328125}
    observed = {0: 0.977294921875, 1: 0.979736328125}

    # Act
    ok, mismatches = c8b.check_founder_identity(observed, logged)

    # Assert
    assert ok is True
    assert mismatches == []


def test_founder_identity_fails_on_a_one_ulp_difference():
    # Arrange
    logged = {0: 0.977294921875}
    observed = {0: 0.977294921875 + 2 ** -40}

    # Act
    ok, mismatches = c8b.check_founder_identity(observed, logged)

    # Assert
    assert ok is False
    assert mismatches[0]["seed"] == 0


def test_founder_identity_fails_when_a_seed_was_not_observed():
    # Act
    ok, mismatches = c8b.check_founder_identity({0: 0.5}, {0: 0.5, 1: 0.6})

    # Assert
    assert ok is False
    assert mismatches[0]["seed"] == 1


# ------------------------------------------------------------ classifier

INTACT25 = (5.0, 9.0)
SHUFFLED = (1.2, 1.7)
FROZEN1_CLEAR = (20.0, 30.0)     # does not overlap INTACT25: floor 1 silent


def test_classifier_returns_deeper_when_frozen_entirely_below_intact():
    result = c8b.classify_outcome((2.0, 3.0), INTACT25, FROZEN1_CLEAR, SHUFFLED, True)
    assert result["outcome"] == "deeper"
    assert result["evaluable"] is True
    assert result["floor_fired"] is None


def test_classifier_returns_shallower_when_frozen_entirely_above_intact():
    result = c8b.classify_outcome((10.0, 12.0), INTACT25, FROZEN1_CLEAR, SHUFFLED, True)
    assert result["outcome"] == "shallower"
    assert result["evaluable"] is True


def test_classifier_returns_overlap_when_intervals_share_a_point():
    result = c8b.classify_outcome((4.0, 5.0), INTACT25, FROZEN1_CLEAR, SHUFFLED, True)
    assert result["outcome"] == "overlap"
    assert result["evaluable"] is True


def test_classifier_not_evaluable_when_founder_identity_fails():
    result = c8b.classify_outcome((2.0, 3.0), INTACT25, FROZEN1_CLEAR, SHUFFLED, False)
    assert result["evaluable"] is False
    assert result["outcome"] == "NOT EVALUABLE"
    assert result["floor_fired"] == "precondition_founder_identity"


def test_classifier_floor1_fires_when_gen1_already_overlaps_intact_gen25():
    # Arrange: frozen gen-1 interval [8, 10] touches intact gen-25 [5, 9].
    result = c8b.classify_outcome((2.0, 3.0), INTACT25, (8.0, 10.0), SHUFFLED, True)

    # Assert
    assert result["evaluable"] is False
    assert result["outcome"] == "NOT EVALUABLE"
    assert result["floor_fired"] == "headroom_1_saturation_at_gen1"


def test_classifier_floor1_takes_precedence_over_floor2():
    result = c8b.classify_outcome((1.0, 1.5), INTACT25, (1.0, 6.0), SHUFFLED, True)
    assert result["floor_fired"] == "headroom_1_saturation_at_gen1"


def test_classifier_floor2_words_deeper_as_to_the_floor():
    # Arrange: frozen [1.0, 1.5] is below intact and overlaps shuffled [1.2, 1.7].
    result = c8b.classify_outcome((1.0, 1.5), INTACT25, FROZEN1_CLEAR, SHUFFLED, True)

    # Assert
    assert result["evaluable"] is True
    assert result["outcome"] == "deeper (to the floor)"
    assert result["floor_fired"] == "headroom_2_shuffled_floor"


def test_classifier_floor2_blocks_a_non_deeper_reading():
    # Arrange: frozen [1.0, 6.0] overlaps both intact and shuffled; only
    # 'deeper' may be stated under floor 2, so 'overlap' is not read.
    result = c8b.classify_outcome((1.0, 6.0), INTACT25, FROZEN1_CLEAR, SHUFFLED, True)

    # Assert
    assert result["evaluable"] is False
    assert result["outcome"] == "NOT EVALUABLE"
    assert result["raw_outcome"] == "overlap"
    assert result["floor_fired"] == "headroom_2_shuffled_floor"


def test_classifier_touching_endpoints_count_as_overlap():
    result = c8b.classify_outcome((3.0, 5.0), INTACT25, FROZEN1_CLEAR, SHUFFLED, True)
    assert result["outcome"] == "overlap"
