"""
Tests for analysis/anchored_sampler_check.py.

Synthetic data only: every array here is constructed inside the test, none is
read from logs/ or any other project output. Arrange-Act-Assert throughout;
test names state the behaviour under test.

The algebraic identities carry most of the weight, and they are tested
separately from the sampling. ``summarise_draws`` is a Monte-Carlo comparison
and could in principle pass a broken sampler by luck; the two covariance
identities below cannot, because they are checked against closed forms with no
randomness in them at all. If the identity tests pass and the sampling tests
fail, the defect is in the draw loop. If the identity tests fail, the module's
central claim -- that the anchor-only variant's shortfall is algebra and not
noise -- is the thing that broke.

Draw counts are kept small where a test is about a code path rather than about
a number, so the whole file runs in a few seconds.

Run from the repository root:

    python3 -m pytest tests/test_anchored_sampler_check.py -v
"""

import os
import sys

import numpy as np
import pytest

# Mirrors tests/test_rank_stats.py's sys.path pattern: pytest's rootdir is not
# automatically on sys.path, so add the repository root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import anchored_sampler_check as asc


def _problem(n_samples=40, n_features=5, noise_sd=0.5, prior_sd=1.3, seed=0):
    """A linear-Gaussian problem and its exact posterior."""
    rng = np.random.default_rng(seed)
    design = asc.make_design(n_samples, n_features, rng)
    w_true = prior_sd * rng.standard_normal(n_features)
    targets = asc.simulate_targets(design, w_true, noise_sd, rng)
    posterior = asc.exact_posterior(design, targets, noise_sd, prior_sd)
    return design, targets, posterior


# --------------------------------------------------------------------------
# exact_posterior
# --------------------------------------------------------------------------

def test_exact_posterior_with_no_data_returns_the_prior():
    # Arrange
    design = np.zeros((0, 4))
    targets = np.zeros(0)

    # Act
    posterior = asc.exact_posterior(design, targets, noise_sd=0.5, prior_sd=1.3)

    # Assert
    assert np.allclose(posterior.mean, np.zeros(4))
    assert np.allclose(posterior.cov, 1.3 ** 2 * np.eye(4))


def test_exact_posterior_refuses_a_flat_prior():
    # Arrange
    design, targets, _ = _problem()

    # Act / Assert
    with pytest.raises(ValueError, match="prior_sd must be positive"):
        asc.exact_posterior(design, targets, noise_sd=0.5, prior_sd=0.0)


def test_exact_posterior_refuses_zero_noise():
    # Arrange
    design, targets, _ = _problem()

    # Act / Assert
    with pytest.raises(ValueError, match="noise_sd must be positive"):
        asc.exact_posterior(design, targets, noise_sd=0.0, prior_sd=1.3)


# --------------------------------------------------------------------------
# The two covariance identities, checked without sampling
# --------------------------------------------------------------------------

def test_anchor_only_covariance_equals_sigma_times_prior_precision_times_sigma():
    # Arrange
    design, _, posterior = _problem()
    prior_sd = 1.3
    expected = posterior.cov @ posterior.cov / prior_sd ** 2

    # Act
    actual = asc.anchor_only_cov(design, noise_sd=0.5, prior_sd=prior_sd)

    # Assert
    assert np.allclose(actual, expected)


def test_anchor_only_covariance_equals_the_posterior_only_with_no_data():
    # Arrange
    empty = np.zeros((0, 4))
    with_data, _, posterior_with_data = _problem(n_features=4)

    # Act
    no_data_cov = asc.anchor_only_cov(empty, noise_sd=0.5, prior_sd=1.3)
    with_data_cov = asc.anchor_only_cov(with_data, noise_sd=0.5, prior_sd=1.3)

    # Assert
    assert np.allclose(no_data_cov, 1.3 ** 2 * np.eye(4))
    assert not np.allclose(with_data_cov, posterior_with_data.cov)


def test_anchor_only_sd_ratio_falls_monotonically_with_sample_size():
    # Arrange / Act
    table = asc.degradation_with_sample_size(
        sizes=(0, 8, 32, 128), n_features=5, seed=3
    )
    ratios = [ratio for _, ratio in table]

    # Assert
    assert ratios[0] == pytest.approx(1.0)
    assert all(later < earlier for earlier, later in zip(ratios, ratios[1:]))


# --------------------------------------------------------------------------
# draw_variant
# --------------------------------------------------------------------------

def test_draw_variant_rejects_an_unknown_variant():
    # Arrange
    design, targets, _ = _problem()

    # Act / Assert
    with pytest.raises(ValueError, match="unknown variant"):
        asc.draw_variant(
            "anchor_and_a_prayer", design, targets, 0.5, 1.3,
            np.random.default_rng(0), n_draws=2,
        )


def test_draw_variant_does_not_modify_its_inputs():
    # Arrange
    design, targets, _ = _problem()
    design_before, targets_before = design.copy(), targets.copy()

    # Act
    asc.draw_variant(
        "randomize_then_optimize", design, targets, 0.5, 1.3,
        np.random.default_rng(0), n_draws=5,
    )

    # Assert
    assert np.array_equal(design, design_before)
    assert np.array_equal(targets, targets_before)


def test_anchor_only_draws_are_distinct_from_each_other():
    # The point of the module: an inert-looking sampler is not what fails here.
    # Anchor-only draws genuinely differ, which is why no check that merely
    # asks whether draws differ can reject it.
    # Arrange / Act
    design, targets, _ = _problem()
    draws = asc.draw_variant(
        "anchor_only", design, targets, 0.5, 1.3,
        np.random.default_rng(0), n_draws=50,
    )

    # Assert
    assert np.all(draws.std(axis=0) > 0.0)
    assert len(np.unique(draws.round(12), axis=0)) == 50


# --------------------------------------------------------------------------
# check_linear_gaussian_exactness -- the precondition itself
# --------------------------------------------------------------------------

def test_randomize_then_optimize_passes_the_exactness_check():
    # Act
    check = asc.check_linear_gaussian_exactness(
        variant="randomize_then_optimize", seed=0, n_draws=8000
    )

    # Assert
    assert check.passed
    assert check.reason is None
    assert check.trace_ratio == pytest.approx(1.0, abs=asc.TRACE_RATIO_TOLERANCE)


def test_anchor_only_fails_the_exactness_check_on_its_width_not_its_centre():
    # Act
    check = asc.check_linear_gaussian_exactness(
        variant="anchor_only", seed=0, n_draws=8000
    )

    # Assert
    assert not check.passed
    assert "marginal standard deviations are wrong" in check.reason
    assert check.max_mean_dev_sigmas < asc.MEAN_TOLERANCE_SIGMAS
    assert check.sd_ratio_max < 0.2


def test_anchor_only_fails_on_every_seed_tested():
    # A single seed would leave the verdict resting on one design matrix.
    # Act
    checks = [
        asc.check_linear_gaussian_exactness(
            variant="anchor_only", seed=seed, n_draws=4000
        )
        for seed in (0, 1, 2)
    ]

    # Assert
    assert all(not check.passed for check in checks)


def test_a_deliberately_shrunk_sampler_is_rejected():
    # Guards the tolerance itself: a sampler only ten percent too narrow must
    # still be refused, or the check would pass variants that are merely less
    # wrong than anchor-only.
    # Arrange
    design, targets, posterior = _problem()
    draws = asc.draw_variant(
        "randomize_then_optimize", design, targets, 0.5, 1.3,
        np.random.default_rng(0), n_draws=8000,
    )
    shrunk = posterior.mean + 0.90 * (draws - posterior.mean)

    # Act
    check = asc.summarise_draws("shrunk_by_ten_percent", shrunk, posterior, seed=0)

    # Assert
    assert not check.passed
    assert "marginal standard deviations are wrong" in check.reason


def test_an_offset_sampler_is_rejected_on_its_centre():
    # Arrange
    design, targets, posterior = _problem()
    draws = asc.draw_variant(
        "randomize_then_optimize", design, targets, 0.5, 1.3,
        np.random.default_rng(0), n_draws=8000,
    )
    offset = draws + 10.0 * posterior.marginal_sd

    # Act
    check = asc.summarise_draws("offset_by_ten_sd", offset, posterior, seed=0)

    # Assert
    assert not check.passed
    assert "not centred on the posterior mean" in check.reason
