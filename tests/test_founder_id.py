"""
Tests for analysis/founder_id.py, the estimator behind S-1 gate A.

Synthetic data only: every survivor set here is hand-constructed for the test
or drawn from a seeded generator inside the test. None is read from ``logs/``
or from any other project output, so nothing in this file can turn into a
cited number by accident. Arrange-Act-Assert throughout; test names state the
behaviour under test.

Two tests carry more weight than the rest and are worth naming here. The
unequal-class-size test is the one that checks the "averaged per class"
clause of the committed metric is doing work rather than decorating it: it
builds a case in which a plain micro accuracy is inflated to well above
chance by a single oversized class while the balanced statistic sits exactly
at chance. The null-mean test is the one that checks the permutation null
lands where headroom floor 2 requires it to -- floor 2 refuses gate A when
the null's mean is more than one of its own standard deviations away from the
statistic's chance level, so if the null were mis-built, floor 2 would refuse
every run of the gate and the refusal would look like a property of the data.

Run from the repository root with the interpreter that has torch 2.8.0, which
is also the one the rest of the project's tests use:

    /usr/bin/python3 -m pytest tests/test_founder_id.py -v
"""

import os
import sys

import numpy as np
import pytest

# Mirrors tests/test_overlap_stats.py: pytest's rootdir is not automatically
# on sys.path, so add the repository root before importing the module.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import founder_id as fid
from analysis import overlap_stats as ov


# --------------------------------------------------------------------------
# Helpers local to the tests
# --------------------------------------------------------------------------

def _independent_sets(n_founders=20, chains_per_founder=2, n_classes=50,
                      set_size=5, seed=12345):
    """Survivor sets drawn independently of the founder label.

    Every set is a uniform random subset of the same alphabet at the same
    size, so the founder label carries no information about the set by
    construction. This is the data on which balanced accuracy must sit near
    chance and the permutation null must sit at chance.
    """
    rng = np.random.default_rng(seed)
    n_chains = n_founders * chains_per_founder
    sets = [set(rng.choice(n_classes, size=set_size, replace=False).tolist())
            for _ in range(n_chains)]
    labels = [f for f in range(n_founders) for _ in range(chains_per_founder)]
    return sets, labels


def _separable_sets(n_founders=20, chains_per_founder=2):
    """Survivor sets that identify their founder perfectly.

    All chains of founder f share one set, and the sets of different founders
    are disjoint, so the leave-one-chain-out rule scores 1.0 against the
    held-out chain's own founder and 0.0 against every other.
    """
    sets, labels = [], []
    for f in range(n_founders):
        for _ in range(chains_per_founder):
            sets.append({10 * f, 10 * f + 1, 10 * f + 2})
            labels.append(f)
    return sets, labels


def _micro_accuracy(survivor_sets, founder_labels):
    """Plain (micro) leave-one-chain-out accuracy: correct / total.

    Written here rather than in the module because the module deliberately
    does not expose it -- the committed metric is the per-class average, and
    a micro accuracy exists in this file only to demonstrate what the per-class
    average is protecting against.
    """
    predictions = fid.predict_founders_loo(survivor_sets, founder_labels)
    truth = np.asarray(founder_labels)
    return float(np.mean(predictions == truth))


def _reference_balanced_accuracy(survivor_sets, founder_labels):
    """Deliberately naive reference for the vectorised implementation.

    Plain Python loops, one held-out chain at a time, no precomputed
    similarity matrix and no one-hot arithmetic. It exists so that the
    module's vectorised path -- which it needs in order to run 10000
    permutations in reasonable time -- can be checked against an
    implementation that is slow but reads directly off the committed
    sentence defining the metric.
    """
    sets = [set(s) for s in survivor_sets]
    labels = list(founder_labels)
    founders = sorted(set(labels))

    correct = {f: 0 for f in founders}
    total = {f: 0 for f in founders}

    for i in range(len(sets)):
        scores = []
        for f in founders:
            train = [j for j in range(len(sets)) if j != i and labels[j] == f]
            scores.append(sum(ov.jaccard(sets[i], sets[j]) for j in train)
                          / len(train))
        best = max(scores)
        # Ties go to the smallest founder id, matching the module.
        predicted = founders[scores.index(best)]
        total[labels[i]] += 1
        if predicted == labels[i]:
            correct[labels[i]] += 1

    return float(np.mean([correct[f] / total[f] for f in founders]))


# --------------------------------------------------------------------------
# balanced_accuracy_loo
# --------------------------------------------------------------------------

def test_perfectly_separable_survivor_sets_give_balanced_accuracy_one():
    # Arrange: 20 founders, 2 chains each, founders mutually disjoint.
    sets, labels = _separable_sets()

    # Act
    result = fid.balanced_accuracy_loo(sets, labels)

    # Assert
    assert result == 1.0


def test_survivor_sets_independent_of_founder_give_accuracy_near_chance():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    result = fid.balanced_accuracy_loo(sets, labels)

    # Assert: chance is 1/20 = 0.05 at 20 founders. The tolerance is wide
    # because a single realisation of 40 chains is noisy; the point of this
    # test is that the statistic does not sit near 1.0, not that it hits
    # 0.05 exactly. The null mean is what is checked tightly, further down.
    assert result == pytest.approx(0.05, abs=0.15)
    assert result < 0.5


def test_balanced_accuracy_is_not_inflated_by_one_oversized_class():
    # Arrange: four founders. Founder 0 has 10 chains, founders 1, 2 and 3
    # have 2 each, and every one of the 16 survivor sets is identical, so no
    # chain carries any information about its founder at all. Every founder
    # therefore scores identically for every held-out chain, the tie rule
    # sends every prediction to founder 0, and founder 0's 10 chains are all
    # "correct" purely because it is the largest class.
    sets = [{0, 1, 2}] * 16
    labels = [0] * 10 + [1, 1, 2, 2, 3, 3]

    # Act
    balanced = fid.balanced_accuracy_loo(sets, labels)
    micro = _micro_accuracy(sets, labels)

    # Assert: the plain accuracy is inflated to 10/16 by the large class,
    # which is more than twice the 1/4 chance level and would read as a
    # strong founder signal. The per-class average is exactly chance, which
    # is the correct reading of data that carries no founder information.
    assert micro == pytest.approx(10 / 16)
    assert balanced == pytest.approx(0.25)
    assert balanced < micro


def test_balanced_accuracy_is_insensitive_to_survivor_set_cardinality():
    # Arrange: founders are perfectly separable, but their survivor sets
    # differ enormously in cardinality -- founder 0's chains hold 2 classes
    # and founder 2's hold 40. A statistic that let set size leak in would
    # not return the same value as the equal-cardinality case.
    sets = [
        {0, 1}, {0, 1},
        set(range(100, 110)), set(range(100, 110)),
        set(range(200, 240)), set(range(200, 240)),
    ]
    labels = [0, 0, 1, 1, 2, 2]

    # Act
    result = fid.balanced_accuracy_loo(sets, labels)

    # Assert
    assert result == 1.0


def test_balanced_accuracy_matches_a_naive_reference_implementation():
    # Arrange: eight founders at R = 3, each with a three-class signature
    # buried under four classes of noise drawn from the same 16-class
    # alphabet. The founders are therefore partly but not fully
    # distinguishable, so agreement between the two paths is not trivially
    # satisfied the way it would be on separable or on degenerate data --
    # both of which are tested separately above.
    rng = np.random.default_rng(7)
    sets, labels = [], []
    for f in range(8):
        core = set(rng.choice(16, size=3, replace=False).tolist())
        for _ in range(3):
            noise = set(rng.choice(16, size=4, replace=False).tolist())
            sets.append(core | noise)
            labels.append(f)

    # Act
    vectorised = fid.balanced_accuracy_loo(sets, labels)
    reference = _reference_balanced_accuracy(sets, labels)

    # Assert
    assert vectorised == pytest.approx(reference)
    assert 0.0 < vectorised < 1.0


def test_balanced_accuracy_accepts_sequences_as_well_as_sets():
    # Arrange: the same data as the separable case, but passed as lists with
    # duplicate entries, which set semantics must absorb.
    sets = [[0, 0, 1], [1, 0], [10, 11, 11], [11, 10]]
    labels = [0, 0, 1, 1]

    # Act
    result = fid.balanced_accuracy_loo(sets, labels)

    # Assert
    assert result == 1.0


def test_balanced_accuracy_does_not_mutate_its_inputs():
    # Arrange
    sets = [{0, 1}, {0, 1}, {5, 6}, {5, 6}]
    labels = [0, 0, 1, 1]
    sets_before = [set(s) for s in sets]
    labels_before = list(labels)

    # Act
    fid.balanced_accuracy_loo(sets, labels)

    # Assert
    assert sets == sets_before
    assert labels == labels_before


def test_balanced_accuracy_refuses_a_founder_with_only_one_chain():
    # Arrange: R = 1 for founder 2. Leaving out that founder's only chain
    # leaves its class with no training chain, so its per-class accuracy is
    # undefined and the statistic cannot be formed.
    sets = [{0, 1}, {0, 1}, {5, 6}, {5, 6}, {9}]
    labels = [0, 0, 1, 1, 2]

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        fid.balanced_accuracy_loo(sets, labels)
    message = str(excinfo.value)
    assert "2" in message
    assert "two chains" in message


def test_balanced_accuracy_refuses_a_single_founder():
    # Arrange: a one-way identification has no chance level and no decision
    # to make.
    sets = [{0, 1}, {0, 1}]
    labels = [0, 0]

    # Act / Assert
    with pytest.raises(ValueError):
        fid.balanced_accuracy_loo(sets, labels)


def test_balanced_accuracy_refuses_mismatched_input_lengths():
    # Arrange
    sets = [{0, 1}, {0, 1}, {5, 6}, {5, 6}]
    labels = [0, 0, 1]

    # Act / Assert
    with pytest.raises(ValueError):
        fid.balanced_accuracy_loo(sets, labels)


def test_balanced_accuracy_refuses_empty_input():
    # Act / Assert
    with pytest.raises(ValueError):
        fid.balanced_accuracy_loo([], [])


# --------------------------------------------------------------------------
# predict_founders_loo -- the tie rule
# --------------------------------------------------------------------------

def test_ties_are_broken_toward_the_smallest_founder_id():
    # Arrange: every set identical, so every founder scores identically for
    # every held-out chain and every decision is a tie. Labels are
    # deliberately not in sorted order to confirm the rule is over founder
    # ids and not over order of appearance.
    sets = [{3, 4}] * 6
    labels = [7, 7, 2, 2, 5, 5]

    # Act
    predictions = fid.predict_founders_loo(sets, labels)

    # Assert
    assert list(predictions) == [2] * 6


def test_an_all_tied_prediction_lands_exactly_at_chance_when_classes_are_equal():
    # Arrange: the tie rule sends every prediction to one founder. With equal
    # class sizes that founder scores a per-class accuracy of 1.0 and every
    # other scores 0.0, so the per-class average is exactly 1/n_founders.
    # This is why ties go to a fixed founder rather than being counted as
    # abstentions, which would place a fully degenerate run *below* chance
    # and make headroom floor 2 fire on data that is simply uninformative.
    n_founders = 20
    sets = [{1, 2}] * (2 * n_founders)
    labels = [f for f in range(n_founders) for _ in range(2)]

    # Act
    result = fid.balanced_accuracy_loo(sets, labels)

    # Assert
    assert result == pytest.approx(1.0 / n_founders)


def test_a_tied_block_donates_exactly_one_over_n_founders_to_the_statistic():
    # Arrange: four founders at R = 2. Founders 0, 1 and 2 share one common
    # survivor set, so all six of their chains tie and all six predictions go
    # to founder 0. Founder 3's two chains share a set disjoint from the
    # rest, so founder 3 is genuinely identifiable. Exactly one founder of
    # four carries real signal.
    sets = [{0, 1}] * 6 + [{50, 51}] * 2
    labels = [0, 0, 1, 1, 2, 2, 3, 3]

    # Act
    result = fid.balanced_accuracy_loo(sets, labels)

    # Assert: founder 3 contributes 1/4 from genuine signal and founder 0
    # contributes 1/4 purely because the ties fell to it, so the statistic
    # reads 0.5 where only a quarter of the founders are identifiable. This
    # pins the upward bias the module docstring records: a tied block always
    # donates 1/n_founders on top of whatever signal exists. It is the
    # mechanism behind the measured departure of the permutation null from
    # 0.05, and it is why headroom floor 2 must be read knowing the tie rule.
    assert result == pytest.approx(0.5)


# --------------------------------------------------------------------------
# gate_a_test
# --------------------------------------------------------------------------

def test_gate_a_test_returns_exactly_the_agreed_keys():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=200, seed=0)

    # Assert: the aggregator in controls/ reads this dict by key, so the key
    # set is part of the interface and not an implementation detail.
    assert set(result) == {
        "observed", "null_mean", "null_sd", "null_min", "null_max",
        "p_value", "n_perm", "seed", "chance_level", "n_chains",
        "n_founders",
    }


def test_gate_a_test_observed_equals_the_statistic_computed_directly():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=200, seed=0)

    # Assert
    assert result["observed"] == fid.balanced_accuracy_loo(sets, labels)


def test_gate_a_test_reports_the_design_it_was_given():
    # Arrange: the design_amendment_2026_11_08 shape, R = 2 over 20 founders.
    sets, labels = _independent_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=200, seed=3)

    # Assert
    assert result["n_chains"] == 40
    assert result["n_founders"] == 20
    assert result["chance_level"] == pytest.approx(0.05)
    assert result["n_perm"] == 200
    assert result["seed"] == 3


def test_gate_a_test_is_significant_on_perfectly_separable_data():
    # Arrange
    sets, labels = _separable_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=500, seed=0)

    # Assert: the observed statistic is the attainable maximum, so no
    # permutation can beat it and the p-value is the smallest the +1/+1
    # estimate can return.
    assert result["observed"] == 1.0
    assert result["p_value"] == pytest.approx(1.0 / 501.0)
    assert result["p_value"] < 0.05


def test_gate_a_test_is_not_significant_when_sets_are_independent_of_founder():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=2000, seed=0)

    # Assert: gate A fails if p < 0.05, so independent data must not reach
    # that. 0.05 is the committed threshold, used here as the committed
    # threshold and not as a tolerance.
    assert result["p_value"] > 0.05


def test_gate_a_permutation_null_mean_sits_at_the_chance_level():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=2000, seed=0)
    fires, _ = fid.floor_2_chance_level(
        result["null_mean"], result["null_sd"], result["chance_level"]
    )

    # Assert: this is headroom floor 2 applied to its own null. A null that
    # does not sit within one of its own standard deviations of 0.05 is
    # measuring something other than the founder-label permutation, and the
    # floor would refuse the gate on every run.
    assert result["null_mean"] == pytest.approx(0.05, abs=result["null_sd"])
    assert fires is False


def test_gate_a_test_null_range_brackets_its_own_mean():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=500, seed=1)

    # Assert
    assert result["null_min"] <= result["null_mean"] <= result["null_max"]
    assert result["null_sd"] > 0.0


def test_gate_a_test_is_deterministic_for_a_fixed_seed():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    first = fid.gate_a_test(sets, labels, n_perm=400, seed=11)
    second = fid.gate_a_test(sets, labels, n_perm=400, seed=11)

    # Assert: every reported quantity, not only the p-value, so that a run
    # recorded in a log can be reproduced in full from the seed.
    assert first == second


def test_gate_a_test_null_depends_on_the_seed():
    # Arrange
    sets, labels = _independent_sets()

    # Act
    first = fid.gate_a_test(sets, labels, n_perm=400, seed=11)
    second = fid.gate_a_test(sets, labels, n_perm=400, seed=12)

    # Assert: the observed statistic is seed-free, the null is not.
    assert first["observed"] == second["observed"]
    assert first["null_mean"] != second["null_mean"]


def test_gate_a_test_p_value_is_never_zero_and_never_exceeds_one():
    # Arrange
    sets, labels = _separable_sets()

    # Act
    result = fid.gate_a_test(sets, labels, n_perm=100, seed=0)

    # Assert: the +1/+1 estimate is bounded below by 1/(n_perm + 1), which
    # is the conservative convention this repository already uses.
    assert 1.0 / 101.0 <= result["p_value"] <= 1.0


def test_gate_a_test_refuses_a_non_positive_permutation_count():
    # Arrange
    sets, labels = _separable_sets(n_founders=3)

    # Act / Assert
    with pytest.raises(ValueError):
        fid.gate_a_test(sets, labels, n_perm=0, seed=0)


def test_gate_a_test_refuses_r_equal_to_one():
    # Arrange
    sets = [{0, 1}, {0, 1}, {5, 6}, {5, 6}, {9}]
    labels = [0, 0, 1, 1, 2]

    # Act / Assert: the refusal must come from the same rule the statistic
    # uses, so that a gate cannot be read at R = 1 by going through the test
    # wrapper instead of the statistic.
    with pytest.raises(ValueError):
        fid.gate_a_test(sets, labels, n_perm=10, seed=0)


# --------------------------------------------------------------------------
# headroom floor 1 -- degenerate alphabet
# --------------------------------------------------------------------------

def test_floor_1_fires_below_four_surviving_classes():
    # Act
    fires, reason = fid.floor_1_degenerate_alphabet(3.9)

    # Assert
    assert fires is True
    assert "NOT EVALUABLE" in reason
    assert "4.0" in reason
    assert "3.9" in reason


def test_floor_1_does_not_fire_exactly_at_four_surviving_classes():
    # Act: the committed wording is "below 4.0 classes", so 4.0 itself is
    # inside the evaluable regime.
    fires, reason = fid.floor_1_degenerate_alphabet(4.0)

    # Assert
    assert fires is False
    assert "NOT EVALUABLE" not in reason


def test_floor_1_does_not_fire_above_four_surviving_classes():
    # Act
    fires, _ = fid.floor_1_degenerate_alphabet(4.4)

    # Assert: 4.4 is the maximizer arm's measured value at this bottleneck,
    # which is the regime the floor was calibrated just beneath.
    assert fires is False


def test_floor_1_refuses_a_negative_alphabet():
    # Act / Assert: a mean count of surviving classes cannot be negative, and
    # a negative input means the caller computed something else.
    with pytest.raises(ValueError):
        fid.floor_1_degenerate_alphabet(-1.0)


def test_floor_1_refuses_a_non_finite_alphabet():
    # Act / Assert
    with pytest.raises(ValueError):
        fid.floor_1_degenerate_alphabet(float("nan"))


# --------------------------------------------------------------------------
# headroom floor 2 -- chance-level separation
# --------------------------------------------------------------------------

def test_floor_2_does_not_fire_when_the_null_sits_on_the_chance_level():
    # Act
    fires, reason = fid.floor_2_chance_level(0.05, 0.01, 0.05)

    # Assert
    assert fires is False
    assert "NOT EVALUABLE" not in reason


def test_floor_2_fires_when_the_null_mean_is_more_than_one_sd_from_chance():
    # Act
    fires, reason = fid.floor_2_chance_level(0.09, 0.01, 0.05)

    # Assert
    assert fires is True
    assert "NOT EVALUABLE" in reason


def test_floor_2_fires_when_the_null_mean_is_below_chance_by_more_than_one_sd():
    # Act: the committed wording is "differs from 0.05 by more than one of
    # its own standard deviations", which is two-sided in the departure even
    # though gate A's p-value is one-sided.
    fires, _ = fid.floor_2_chance_level(0.01, 0.005, 0.05)

    # Assert
    assert fires is True


def test_floor_2_does_not_fire_exactly_one_sd_from_chance():
    # Arrange: exactly representable binary fractions, so that the boundary
    # is tested at the boundary rather than one float ulp away from it. The
    # committed comparison is a strict "more than", so a departure of exactly
    # one standard deviation leaves the gate evaluable.
    chance, null_sd, null_mean = 0.5, 0.25, 0.75

    # Act
    fires, _ = fid.floor_2_chance_level(null_mean, null_sd, chance)

    # Assert
    assert fires is False


def test_floor_2_fires_on_a_zero_standard_deviation_null_away_from_chance():
    # Act: a null with no spread that sits off chance is the clearest case
    # of a null measuring the wrong thing.
    fires, _ = fid.floor_2_chance_level(0.2, 0.0, 0.05)

    # Assert
    assert fires is True


def test_floor_2_refuses_a_negative_standard_deviation():
    # Act / Assert
    with pytest.raises(ValueError):
        fid.floor_2_chance_level(0.05, -0.01, 0.05)


def test_floor_2_refuses_a_non_finite_null_mean():
    # Act / Assert
    with pytest.raises(ValueError):
        fid.floor_2_chance_level(float("inf"), 0.01, 0.05)


def test_floor_2_default_chance_level_is_the_twenty_founder_value():
    # Act: the committed floor names 0.05 explicitly for 20 founders, so the
    # default must be that value and not something derived at call time.
    fires, reason = fid.floor_2_chance_level(0.05, 0.01)

    # Assert
    assert fires is False
    assert "0.05" in reason


# --------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------

def test_committed_constants_match_the_rule_text():
    # Assert: these are read out of gates/thresholds.json key S-1 and must
    # not drift. A test on a constant looks redundant until someone edits
    # the constant.
    assert fid.MIN_MEAN_ALPHABET_GEN25 == 4.0
    assert fid.N_FOUNDERS_COMMITTED == 20
    assert fid.CHANCE_LEVEL_20_FOUNDERS == 0.05
    assert fid.DEFAULT_N_PERM == 10_000
