"""
Tests for analysis/rank_stats.py.

Synthetic data only: every array here is hand-constructed for the test, none
is read from logs/ or any other project output. Arrange-Act-Assert
throughout; test names state the behaviour under test.

The degenerate cases carry most of the weight. C-4S's gate B is a partial
correlation, and a partial correlation computed on a residual with no
variance left is the error C-4's gate B made in a different coordinate
system: a statistic that cannot move, reported as though it had moved. Every
"not evaluable" path below exists to make that failure visible instead of
silent.

Run from the repository root:

    python3 -m pytest tests/test_rank_stats.py -v
"""

import os
import sys

import numpy as np
import pytest

# Mirrors tests/test_overlap_stats.py's sys.path pattern: pytest's rootdir is
# not automatically on sys.path, so add the repository root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import rank_stats as rs


# --------------------------------------------------------------------------
# average_ranks
# --------------------------------------------------------------------------

def test_average_ranks_of_distinct_values_are_one_based_positions():
    # Arrange
    x = [30.0, 10.0, 20.0]

    # Act
    result = rs.average_ranks(x)

    # Assert
    assert list(result) == [3.0, 1.0, 2.0]


def test_average_ranks_splits_a_tie_between_the_positions_it_spans():
    # Arrange: two values tied for positions 2 and 3
    x = [1.0, 5.0, 5.0, 9.0]

    # Act
    result = rs.average_ranks(x)

    # Assert
    assert list(result) == [1.0, 2.5, 2.5, 4.0]


def test_average_ranks_of_all_equal_values_are_all_the_mean_rank():
    # Arrange
    x = [7.0, 7.0, 7.0, 7.0]

    # Act
    result = rs.average_ranks(x)

    # Assert
    assert list(result) == [2.5, 2.5, 2.5, 2.5]


def test_average_ranks_of_a_single_value_is_rank_one():
    # Arrange
    x = [42.0]

    # Act
    result = rs.average_ranks(x)

    # Assert
    assert list(result) == [1.0]


def test_average_ranks_does_not_mutate_its_input():
    # Arrange
    x = np.array([3.0, 1.0, 2.0])
    before = x.copy()

    # Act
    rs.average_ranks(x)

    # Assert
    assert np.array_equal(x, before)


def test_average_ranks_rejects_an_empty_array():
    # Arrange
    x = []

    # Act / Assert
    with pytest.raises(ValueError, match="empty"):
        rs.average_ranks(x)


# --------------------------------------------------------------------------
# spearman
# --------------------------------------------------------------------------

def test_spearman_of_a_strictly_increasing_relation_is_one():
    # Arrange
    a = [1.0, 2.0, 3.0, 4.0]
    b = [10.0, 20.0, 30.0, 40.0]

    # Act
    result = rs.spearman(a, b)

    # Assert
    assert result == pytest.approx(1.0)


def test_spearman_of_a_strictly_decreasing_relation_is_minus_one():
    # Arrange
    a = [1.0, 2.0, 3.0, 4.0]
    b = [40.0, 30.0, 20.0, 10.0]

    # Act
    result = rs.spearman(a, b)

    # Assert
    assert result == pytest.approx(-1.0)


def test_spearman_is_one_for_a_monotone_nonlinear_relation():
    # Arrange: Pearson would be below 1 here; Spearman sees only order.
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [1.0, 4.0, 9.0, 16.0, 625.0]

    # Act
    result = rs.spearman(a, b)

    # Assert
    assert result == pytest.approx(1.0)


def test_spearman_is_nan_when_one_input_is_constant():
    # Arrange: a constant has no ranks to correlate, so the statistic is
    # undefined rather than zero. Returning 0.0 here would be read as "no
    # relation measured" when nothing was measurable at all.
    a = [5.0, 5.0, 5.0, 5.0]
    b = [1.0, 2.0, 3.0, 4.0]

    # Act
    result = rs.spearman(a, b)

    # Assert
    assert np.isnan(result)


def test_spearman_rejects_inputs_of_different_lengths():
    # Arrange
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0]

    # Act / Assert
    with pytest.raises(ValueError, match="same length"):
        rs.spearman(a, b)


# --------------------------------------------------------------------------
# residualise_on_ranks
# --------------------------------------------------------------------------

def test_residualising_an_unrelated_variable_retains_nearly_all_rank_variance():
    # Arrange: y's order is deliberately scrambled relative to x's.
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [4.0, 1.0, 6.0, 2.0, 5.0, 3.0]

    # Act
    residual, retained = rs.residualise_on_ranks(y, x)

    # Assert
    assert retained > 0.9
    assert len(residual) == len(y)


def test_residualising_a_variable_on_itself_leaves_no_rank_variance():
    # Arrange
    x = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Act
    residual, retained = rs.residualise_on_ranks(x, x)

    # Assert: this is the case that must be caught, not reported as a
    # correlation of zero.
    assert retained == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(residual, 0.0, atol=1e-12)


def test_residualising_a_monotone_transform_leaves_no_rank_variance():
    # Arrange: a different scale, identical order.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 40.0, 600.0, 8000.0, 100000.0]

    # Act
    _, retained = rs.residualise_on_ranks(y, x)

    # Assert
    assert retained == pytest.approx(0.0, abs=1e-12)


def test_residualise_on_ranks_retains_everything_when_the_control_is_constant():
    # Arrange: nothing can be projected out, so the residual is the centred
    # rank vector itself and all of its variance survives.
    y = [1.0, 2.0, 3.0, 4.0]
    x = [7.0, 7.0, 7.0, 7.0]

    # Act
    residual, retained = rs.residualise_on_ranks(y, x)

    # Assert
    assert retained == pytest.approx(1.0)
    assert np.std(residual) > 0


def test_residualise_on_ranks_does_not_mutate_its_inputs():
    # Arrange
    y = np.array([4.0, 1.0, 6.0, 2.0])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_before, x_before = y.copy(), x.copy()

    # Act
    rs.residualise_on_ranks(y, x)

    # Assert
    assert np.array_equal(y, y_before)
    assert np.array_equal(x, x_before)


# --------------------------------------------------------------------------
# partial_spearman
# --------------------------------------------------------------------------

def test_partial_spearman_with_a_constant_control_equals_plain_spearman():
    # Arrange: a control with no order projects nothing out, so the partial
    # statistic must reduce to the unconditional one. This is an identity,
    # not an approximation, and it pins the whole construction down.
    a = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    b = [2.0, 7.0, 1.0, 8.0, 3.0, 9.0, 4.0, 10.0]
    control = [7.0] * 8

    # Act
    result = rs.partial_spearman(a, b, control)

    # Assert
    assert result.reason is None
    assert result.rho == pytest.approx(rs.spearman(a, b))


def test_partial_spearman_is_one_when_both_carry_the_same_component_beyond_the_control():
    # Arrange: a and b differ from the control by the identical scrambling
    # term, so their residual orders coincide exactly.
    control = np.arange(1.0, 21.0)
    shared = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5,
                       8.0, 7.0, 1.2, 9.5, 4.5, 2.5, 6.5, 8.5, 7.5, 0.5])
    a = control + 10.0 * shared
    b = control + 10.0 * shared

    # Act
    result = rs.partial_spearman(a, b, control)

    # Assert
    assert result.reason is None
    assert result.rho == pytest.approx(1.0)


def test_partial_spearman_is_minus_one_when_the_components_beyond_the_control_are_opposed():
    # Arrange: the same construction with the sign of the extra term flipped
    # in b. Gate B is two-sided precisely because no theory here predicts
    # this sign, so the negative case has to be measurable.
    control = np.arange(1.0, 21.0)
    shared = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5,
                       8.0, 7.0, 1.2, 9.5, 4.5, 2.5, 6.5, 8.5, 7.5, 0.5])
    a = control + 10.0 * shared
    b = control - 10.0 * shared

    # Act
    result = rs.partial_spearman(a, b, control)

    # Assert: strongly negative rather than exactly -1. The control
    # contributes to both rank orders, so flipping the sign of the extra term
    # makes the two residuals near-mirrors of each other and not exact ones.
    # The +1 case above is exact because there a and b are identical.
    assert result.reason is None
    assert result.rho < -0.9


def test_partial_spearman_is_small_when_the_components_beyond_the_control_are_independent():
    # Arrange: both variables are driven by the control plus a scrambling
    # term large enough to move the ranks, but the two terms are drawn
    # independently. Fixed seed, so this is a deterministic assertion about
    # one sample rather than a claim about the estimator's expectation; the
    # bound is loose because n = 20.
    rng = np.random.default_rng(0)
    control = np.arange(1.0, 21.0)
    a = control + 10.0 * rng.normal(size=20)
    b = control + 10.0 * rng.normal(size=20)

    # Act
    result = rs.partial_spearman(a, b, control)

    # Assert
    assert result.reason is None
    assert abs(result.rho) < 0.5


def test_partial_spearman_is_not_evaluable_when_a_residual_has_no_variance():
    # Arrange: `a` is the control under a monotone transform, so residualising
    # it leaves nothing. A partial correlation here is computed on nothing.
    control = np.arange(1.0, 21.0)
    a = control * 3.0
    b = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5,
                  8.0, 7.0, 1.2, 9.5, 4.5, 2.5, 6.5, 8.5, 7.5, 0.5])

    # Act
    result = rs.partial_spearman(a, b, control, retained_floor=0.01)

    # Assert
    assert result.reason is not None
    assert np.isnan(result.rho)
    assert result.retained_a == pytest.approx(0.0, abs=1e-12)


def test_partial_spearman_reports_retained_variance_for_both_variables():
    # Arrange
    control = np.arange(1.0, 13.0)
    a = np.array([5.0, 2.0, 9.0, 1.0, 7.0, 3.0, 11.0, 4.0, 8.0, 6.0, 12.0, 10.0])
    b = np.array([2.0, 8.0, 1.0, 11.0, 4.0, 9.0, 3.0, 12.0, 5.0, 10.0, 6.0, 7.0])

    # Act
    result = rs.partial_spearman(a, b, control)

    # Assert: both are reported, so a reader can see how much room the gate
    # had rather than inferring it from the p-value.
    assert 0.0 < result.retained_a <= 1.0
    assert 0.0 < result.retained_b <= 1.0


def test_partial_spearman_result_is_immutable():
    # Arrange
    control = np.arange(1.0, 11.0)
    a = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5, 3.5])
    b = np.array([2.0, 7.0, 1.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 6.0])
    result = rs.partial_spearman(a, b, control)

    # Act / Assert
    with pytest.raises(Exception):
        result.rho = 0.0


def test_partial_spearman_rejects_inputs_of_different_lengths():
    # Arrange
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    control = [1.0, 2.0]

    # Act / Assert
    with pytest.raises(ValueError, match="same length"):
        rs.partial_spearman(a, b, control)
