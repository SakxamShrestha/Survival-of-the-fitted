"""
Tests for analysis/overlap_stats.py.

Synthetic data only: every survivor set here is hand-constructed for the
test, none is read from logs/ or any other project output. Arrange-Act-Assert
throughout; test names state the behaviour under test.

Run from the repository root:

    python3 -m pytest tests/test_overlap_stats.py -v
"""

import dataclasses
import os
import sys

import numpy as np
import pytest

# Mirrors analysis/seed_stability_stats.py's sys.path pattern: pytest's
# rootdir is not automatically on sys.path, so add the repository root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import overlap_stats as ov


# --------------------------------------------------------------------------
# jaccard
# --------------------------------------------------------------------------

def test_jaccard_of_two_empty_sets_is_defined_as_one():
    # Arrange
    a, b = set(), set()

    # Act
    result = ov.jaccard(a, b)

    # Assert: the empty set is the same as the empty set, by this module's
    # explicit convention (matching scipy.spatial.distance.jaccard's
    # all-zero-vector case).
    assert result == 1.0


def test_jaccard_of_one_empty_and_one_nonempty_set_is_zero():
    # Arrange
    a, b = set(), {1, 2, 3}

    # Act
    result = ov.jaccard(a, b)

    # Assert
    assert result == 0.0


def test_jaccard_of_disjoint_sets_is_zero():
    # Arrange
    a, b = {1, 2}, {3, 4}

    # Act
    result = ov.jaccard(a, b)

    # Assert
    assert result == 0.0


def test_jaccard_known_value_computed_by_hand():
    # Arrange: intersection {2, 3} has size 2, union {1, 2, 3, 4} has size 4.
    a, b = {1, 2, 3}, {2, 3, 4}

    # Act
    result = ov.jaccard(a, b)

    # Assert
    assert result == pytest.approx(2 / 4)


def test_jaccard_of_identical_single_class_sets_is_one():
    # Arrange
    a, b = {7}, {7}

    # Act
    result = ov.jaccard(a, b)

    # Assert
    assert result == 1.0


def test_jaccard_accepts_array_like_input_not_just_python_sets():
    # Arrange: a has a duplicate entry, which set semantics must absorb.
    a, b = np.array([1, 2, 2, 3]), [2, 3, 4]

    # Act
    result = ov.jaccard(a, b)

    # Assert: same as the {1, 2, 3} vs {2, 3, 4} case above.
    assert result == pytest.approx(2 / 4)


# --------------------------------------------------------------------------
# overlap_coefficient
# --------------------------------------------------------------------------

def test_overlap_coefficient_of_two_empty_sets_is_defined_as_one():
    # Arrange
    a, b = set(), set()

    # Act
    result = ov.overlap_coefficient(a, b)

    # Assert
    assert result == 1.0


def test_overlap_coefficient_of_one_empty_and_one_nonempty_set_is_zero():
    # Arrange: this is the division-by-zero edge case overlap_coefficient
    # must handle explicitly that jaccard does not (min(|a|, |b|) = 0 here
    # even though only one set, not both, is empty).
    a, b = set(), {1, 2, 3}

    # Act
    result = ov.overlap_coefficient(a, b)

    # Assert
    assert result == 0.0


def test_overlap_coefficient_known_value_computed_by_hand():
    # Arrange: intersection {2, 3} has size 2, min(|a|, |b|) = min(3, 3) = 3.
    a, b = {1, 2, 3}, {2, 3, 4}

    # Act
    result = ov.overlap_coefficient(a, b)

    # Assert
    assert result == pytest.approx(2 / 3)


def test_overlap_coefficient_of_identical_single_class_sets_is_one():
    # Arrange
    a, b = {7}, {7}

    # Act
    result = ov.overlap_coefficient(a, b)

    # Assert
    assert result == 1.0


def test_overlap_coefficient_is_insensitive_to_set_size_where_jaccard_is_not():
    # Arrange: a is a strict subset of b, and b is five times larger than a.
    # This is the size confound RESEARCH-PLAN.md 4.3 item 4 names: Jaccard
    # penalizes the size difference even though a overlaps b completely.
    a, b = {1, 2}, set(range(10))

    # Act
    jaccard_value = ov.jaccard(a, b)
    overlap_value = ov.overlap_coefficient(a, b)

    # Assert: the disagreement is the point. Jaccard is dragged down by the
    # size mismatch; the overlap coefficient reports the complete containment.
    assert jaccard_value == pytest.approx(0.2)
    assert overlap_value == 1.0
    assert overlap_value != jaccard_value


# --------------------------------------------------------------------------
# mean_pairwise_jaccard / mean_pairwise_overlap
# --------------------------------------------------------------------------

def test_mean_pairwise_jaccard_known_value_over_three_sets():
    # Arrange: pairwise Jaccards are 0.5, 2/3 and 0.25 (verified by hand
    # against the single-pair formula above).
    sets = [{1, 2, 3}, {2, 3, 4}, {1, 2}]

    # Act
    result = ov.mean_pairwise_jaccard(sets)

    # Assert
    assert result == pytest.approx((0.5 + 2 / 3 + 0.25) / 3)


def test_mean_pairwise_overlap_known_value_over_three_sets():
    # Arrange: pairwise overlap coefficients are 2/3, 1.0 and 0.5.
    sets = [{1, 2, 3}, {2, 3, 4}, {1, 2}]

    # Act
    result = ov.mean_pairwise_overlap(sets)

    # Assert
    assert result == pytest.approx((2 / 3 + 1.0 + 0.5) / 3)


def test_mean_pairwise_jaccard_raises_on_fewer_than_two_sets():
    # Arrange
    sets = [{1, 2, 3}]

    # Act / Assert
    with pytest.raises(ValueError):
        ov.mean_pairwise_jaccard(sets)


def test_mean_pairwise_overlap_raises_on_empty_list_of_sets():
    # Arrange
    sets = []

    # Act / Assert
    with pytest.raises(ValueError):
        ov.mean_pairwise_overlap(sets)


# --------------------------------------------------------------------------
# cross_arm_permutation_test (the primary statistic)
# --------------------------------------------------------------------------

def test_cross_arm_permutation_test_is_deterministic_given_seeded_rng():
    # Arrange
    sets_a = [{1, 2, 3}, {2, 3, 4}, {1, 2, 5}]
    sets_b = [{10, 11}, {10, 12}, {11, 13}]

    # Act
    first = ov.cross_arm_permutation_test(
        sets_a, sets_b, np.random.default_rng(42), n_perm=300)
    second = ov.cross_arm_permutation_test(
        sets_a, sets_b, np.random.default_rng(42), n_perm=300)

    # Assert
    assert first.observed == second.observed
    assert np.array_equal(first.null_samples, second.null_samples)
    assert first.p_value == second.p_value


def test_cross_arm_permutation_test_raises_when_an_arm_has_fewer_than_two_chains():
    # Arrange
    sets_a = [{1, 2}]
    sets_b = [{3, 4}, {5, 6}]

    # Act / Assert
    with pytest.raises(ValueError):
        ov.cross_arm_permutation_test(sets_a, sets_b, np.random.default_rng(0))


def test_cross_arm_permutation_test_raises_on_invalid_alternative():
    # Arrange
    sets_a = [{1, 2}, {1, 3}]
    sets_b = [{4, 5}, {4, 6}]

    # Act / Assert
    with pytest.raises(ValueError):
        ov.cross_arm_permutation_test(
            sets_a, sets_b, np.random.default_rng(0), alternative="sideways")


def test_cross_arm_permutation_test_can_run_on_the_overlap_coefficient():
    # Arrange: RESEARCH-PLAN.md 4.3 item 4 asks for the overlap coefficient
    # to be reported alongside Jaccard, so the primary test must accept it
    # as a drop-in statistic_fn.
    sets_a = [{1, 2, 3}, {2, 3, 4}, {1, 2, 5}]
    sets_b = [{10, 11}, {10, 12}, {11, 13}]

    # Act
    result = ov.cross_arm_permutation_test(
        sets_a, sets_b, np.random.default_rng(7), n_perm=200,
        statistic_fn=ov.mean_pairwise_overlap)

    # Assert: ran to completion and produced a well-formed result.
    assert result.null_samples.shape == (200,)
    assert 0.0 <= result.p_value <= 1.0


def test_cross_arm_permutation_test_p_values_are_roughly_uniform_under_the_null():
    # Arrange: construct a scenario where the null hypothesis is exactly
    # true by construction -- every chain in both arms is an independent
    # uniform random survivor set of the same size from the same universe,
    # so "arm label" carries no information about overlap. The rng seed is
    # fixed, so this is a deterministic fixture, not a fresh gamble on every
    # run: with n_perm = 1000 the smallest p a single replicate could report
    # is 1 / 1001, so a p below 0.01 would mean several of the most extreme
    # permutations landed below the observed value -- a real calibration
    # signature, not resolution noise.
    n_classes = 30
    set_size = 6
    n_chains_per_arm = 6
    n_replicates = 20
    n_perm = 1000
    master_rng = np.random.default_rng(0)

    p_values = []
    for _ in range(n_replicates):
        sets_a = [
            set(master_rng.choice(n_classes, size=set_size, replace=False))
            for _ in range(n_chains_per_arm)
        ]
        sets_b = [
            set(master_rng.choice(n_classes, size=set_size, replace=False))
            for _ in range(n_chains_per_arm)
        ]
        result = ov.cross_arm_permutation_test(sets_a, sets_b, master_rng, n_perm=n_perm)
        p_values.append(result.p_value)

    # Act: nothing further to do; p_values is the thing under test.

    # Assert, loosely: under a true null, p < 0.01 has about a 1% chance per
    # replicate, so requiring none of a modest number of replicates to land
    # there is a calibration check, not a strict uniformity test. Because the
    # seed above is fixed, this assertion's outcome does not vary run to run.
    assert all(p >= 0.01 for p in p_values), p_values


# --------------------------------------------------------------------------
# draw_survivors_weighted
# --------------------------------------------------------------------------

def test_draw_survivors_weighted_respects_the_requested_sizes():
    # Arrange
    weights = np.ones(20)

    # Act
    draws = ov.draw_survivors_weighted([0, 1, 3, 5], weights, np.random.default_rng(0))

    # Assert
    assert [len(d) for d in draws] == [0, 1, 3, 5]


def test_draw_survivors_weighted_never_draws_a_zero_weight_class():
    # Arrange: only classes 0-4 have nonzero weight, out of 20.
    weights = np.zeros(20)
    weights[:5] = 1.0
    rng = np.random.default_rng(0)

    # Act: draw repeatedly so an accidental pass is implausible.
    forbidden = set(range(5, 20))
    seen = set()
    for _ in range(200):
        (draw,) = ov.draw_survivors_weighted([3], weights, rng)
        seen.update(draw.tolist())

    # Assert
    assert seen.isdisjoint(forbidden)


def test_draw_survivors_weighted_draws_without_replacement_within_a_set():
    # Arrange
    weights = np.ones(10)

    # Act
    (draw,) = ov.draw_survivors_weighted([10], weights, np.random.default_rng(3))

    # Assert: requesting all 10 classes from 10 must return each exactly once.
    assert sorted(draw.tolist()) == list(range(10))


def test_draw_survivors_weighted_is_deterministic_given_seeded_rng():
    # Arrange
    weights = np.array([3.0, 1.0, 1.0, 0.0, 2.0])

    # Act
    first = ov.draw_survivors_weighted([2, 3], weights, np.random.default_rng(11))
    second = ov.draw_survivors_weighted([2, 3], weights, np.random.default_rng(11))

    # Assert
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_draw_survivors_weighted_raises_when_size_exceeds_available_support():
    # Arrange: only 2 classes have nonzero weight.
    weights = np.array([1.0, 1.0, 0.0, 0.0])

    # Act / Assert
    with pytest.raises(ValueError):
        ov.draw_survivors_weighted([3], weights, np.random.default_rng(0))


def test_draw_survivors_weighted_raises_on_negative_weights():
    # Arrange
    weights = np.array([1.0, -1.0, 2.0])

    # Act / Assert
    with pytest.raises(ValueError):
        ov.draw_survivors_weighted([1], weights, np.random.default_rng(0))


def test_draw_survivors_weighted_raises_when_all_weights_are_zero():
    # Arrange
    weights = np.zeros(5)

    # Act / Assert
    with pytest.raises(ValueError):
        ov.draw_survivors_weighted([1], weights, np.random.default_rng(0))


def test_draw_survivors_weighted_does_not_mutate_its_inputs():
    # Arrange
    sizes = [2, 3]
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    sizes_copy, weights_copy = list(sizes), weights.copy()

    # Act
    ov.draw_survivors_weighted(sizes, weights, np.random.default_rng(0))

    # Assert
    assert sizes == sizes_copy
    assert np.array_equal(weights, weights_copy)


# --------------------------------------------------------------------------
# size_conditioned_null
# --------------------------------------------------------------------------

def test_size_conditioned_null_is_deterministic_given_seeded_rng():
    # Arrange
    sets = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5, 6}]

    # Act
    first = ov.size_conditioned_null(
        sets, n_classes=20, rng=np.random.default_rng(5), n_null=200)
    second = ov.size_conditioned_null(
        sets, n_classes=20, rng=np.random.default_rng(5), n_null=200)

    # Assert
    assert first.observed == second.observed
    assert np.array_equal(first.null_samples, second.null_samples)
    assert first.p_value == second.p_value


def test_size_conditioned_null_raises_on_fewer_than_two_sets():
    # Arrange
    sets = [{1, 2, 3}]

    # Act / Assert
    with pytest.raises(ValueError):
        ov.size_conditioned_null(sets, n_classes=20, rng=np.random.default_rng(0))


def test_size_conditioned_null_raises_on_non_positive_n_classes():
    # Arrange
    sets = [{1, 2}, {2, 3}]

    # Act / Assert
    with pytest.raises(ValueError):
        ov.size_conditioned_null(sets, n_classes=0, rng=np.random.default_rng(0))


# --------------------------------------------------------------------------
# frequency_matched_null
# --------------------------------------------------------------------------

def test_frequency_matched_null_matches_size_conditioned_null_under_uniform_weights():
    # Arrange: frequency_matched_null with uniform weights is, by
    # construction, the same draw as size_conditioned_null's unweighted
    # resampling -- this pins the two nulls to each other so they cannot
    # drift apart under a future edit to only one of them.
    n_classes = 20
    sets = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5, 6}]
    seed = 55

    # Act
    size_result = ov.size_conditioned_null(
        sets, n_classes, np.random.default_rng(seed), n_null=150)
    freq_result = ov.frequency_matched_null(
        sets, np.ones(n_classes), np.random.default_rng(seed), n_null=150)

    # Assert
    assert size_result.observed == freq_result.observed
    assert np.array_equal(size_result.null_samples, freq_result.null_samples)
    assert size_result.p_value == freq_result.p_value


def test_frequency_matched_null_is_deterministic_given_seeded_rng():
    # Arrange
    sets = [{1, 2, 3}, {2, 3, 4}, {1, 5}]
    weights = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0])

    # Act
    first = ov.frequency_matched_null(
        sets, weights, np.random.default_rng(9), n_null=150)
    second = ov.frequency_matched_null(
        sets, weights, np.random.default_rng(9), n_null=150)

    # Assert
    assert first.observed == second.observed
    assert np.array_equal(first.null_samples, second.null_samples)
    assert first.p_value == second.p_value


def test_frequency_matched_null_raises_on_fewer_than_two_sets():
    # Arrange
    sets = [{1, 2}]
    weights = np.ones(10)

    # Act / Assert
    with pytest.raises(ValueError):
        ov.frequency_matched_null(sets, weights, np.random.default_rng(0))


# --------------------------------------------------------------------------
# null_headroom
# --------------------------------------------------------------------------

def test_null_headroom_reports_both_the_null_mean_and_the_gap_to_the_bound():
    # Arrange: a null distribution already sitting close to the upper bound
    # of 1.0, in the shape of the C-4 gate B failure this function exists to
    # catch (synthetic numbers, not the project's real figures).
    null_samples = np.full(1000, 0.75)
    observed = 0.95

    # Act
    null_mean, headroom = ov.null_headroom(null_samples, observed, extreme_side="high")

    # Assert
    assert null_mean == pytest.approx(0.75)
    assert headroom == pytest.approx(0.25)


def test_null_headroom_handles_the_low_extreme_side_with_a_custom_bound():
    # Arrange: a statistic with a natural range outside [0, 1] (e.g. a raw
    # difference in generation counts), where the default bound would be
    # wrong and an explicit bound is required.
    null_samples = np.full(500, 8.0)
    observed = 3.0

    # Act
    null_mean, headroom = ov.null_headroom(
        null_samples, observed, extreme_side="low", bound=1.0)

    # Assert
    assert null_mean == pytest.approx(8.0)
    assert headroom == pytest.approx(7.0)


def test_null_headroom_default_bound_is_zero_for_the_low_side():
    # Arrange: no explicit bound, relying on the [0, 1]-bounded default this
    # module's own statistics (Jaccard, overlap coefficient) satisfy.
    null_samples = np.full(500, 0.3)
    observed = 0.1

    # Act
    null_mean, headroom = ov.null_headroom(null_samples, observed, extreme_side="low")

    # Assert
    assert null_mean == pytest.approx(0.3)
    assert headroom == pytest.approx(0.3)


def test_null_headroom_raises_on_invalid_extreme_side():
    # Arrange
    null_samples = np.full(100, 0.5)

    # Act / Assert
    with pytest.raises(ValueError):
        ov.null_headroom(null_samples, 0.6, extreme_side="sideways")


def test_null_headroom_raises_on_empty_null_samples():
    # Arrange
    null_samples = np.array([])

    # Act / Assert
    with pytest.raises(ValueError):
        ov.null_headroom(null_samples, 0.5, extreme_side="high")


def test_null_headroom_raises_when_observed_falls_below_the_lower_bound():
    # Arrange: the low-side mirror of the bound check below. A statistic under
    # its theoretical minimum means the statistic or the bound is wrong, and
    # silently returning a negative headroom would let a gate report room it
    # does not have.
    null_samples = np.full(100, 0.5)

    # Act / Assert
    with pytest.raises(ValueError):
        ov.null_headroom(null_samples, observed=-0.5, extreme_side="low")


def test_null_headroom_raises_when_observed_exceeds_the_upper_bound():
    # Arrange: observed above the theoretical maximum means the statistic or
    # bound is wrong, and this must fail loudly rather than silently return
    # a negative headroom.
    null_samples = np.full(100, 0.5)

    # Act / Assert
    with pytest.raises(ValueError):
        ov.null_headroom(null_samples, observed=1.5, extreme_side="high")


# --------------------------------------------------------------------------
# NullTestResult
# --------------------------------------------------------------------------

def test_null_test_result_fields_cannot_be_reassigned():
    # Arrange
    result = ov.NullTestResult(
        observed=0.5, null_samples=np.array([0.1, 0.2]), p_value=0.3)

    # Act / Assert: immutable by construction (coding-style.md: never mutate,
    # always return a new copy), so reassignment must fail, not silently work.
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.observed = 0.9


def test_null_test_result_null_samples_cannot_be_mutated_in_place():
    # Arrange: frozen=True blocks rebinding the attribute but not writing
    # through the array it holds, so the array itself must be read-only.
    # Otherwise a caller could edit a null distribution after its p-value had
    # been read off it and the two would silently disagree.
    result = ov.NullTestResult(
        observed=0.5, null_samples=np.array([0.1, 0.2]), p_value=0.3)

    # Act / Assert
    with pytest.raises(ValueError):
        result.null_samples[0] = 99.0


def test_resampling_null_returns_a_read_only_null_distribution():
    # Arrange
    sets = [{0, 1, 2}, {1, 2, 3}, {2, 3, 4}]
    rng = np.random.default_rng(0)

    # Act
    result = ov.size_conditioned_null(sets, n_classes=10, rng=rng, n_null=50)

    # Assert: the guarantee has to hold on results the module builds itself,
    # not only on a hand-constructed one.
    with pytest.raises(ValueError):
        result.null_samples[0] = 99.0
