"""
Tests for analysis/laplace_glm.py.

Synthetic data only: every feature matrix and label vector here is
hand-constructed for the test, none is read from logs/ or any other project
output. Arrange-Act-Assert throughout; test names state the behaviour under
test.

Why these particular cases. This module is the analytic half of C-8's
correctness bridge: with the MLP body frozen the head is a GLM, so its Laplace
posterior is closed-form and must agree with what the chain actually produces.
A silently wrong posterior here would make every C-8 and S-1 correctness
conclusion wrong, and nothing downstream would reveal it -- the experiment
would simply agree with a wrong analytic prediction, or disagree with it for a
reason no one could attribute. So the tests assert provable identities rather
than fitted expectations:

  - K = 1 makes the likelihood constant, so the MAP is the prior's mean, zero,
    and the posterior covariance is exactly the prior covariance.
  - Two classes force an antisymmetric MAP, because the likelihood sees only
    the difference of the two weight columns while the prior penalises both.
  - The all-columns-equal direction is an exact eigenvector of the Hessian with
    eigenvalue exactly the prior precision, which pins the Hessian
    construction against the softmax's shift invariance.
  - Zero prior precision makes that eigenvalue zero, so the posterior does not
    exist. Every existing arm in this project passes wd = 0.0, so this is the
    degenerate case the repository is most likely to hand the module.

Run from the repository root:

    python3 -m pytest tests/test_laplace_glm.py -v
"""

import os
import sys

import numpy as np
import pytest

# Mirrors tests/test_rank_stats.py's sys.path pattern: pytest's rootdir is not
# automatically on sys.path, so add the repository root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import laplace_glm as lg


# --------------------------------------------------------------------------
# single-class problems, where the answer is analytic
# --------------------------------------------------------------------------

def test_single_class_problem_has_its_map_exactly_at_the_prior_mean():
    # Arrange: with one class the softmax is identically 1, so the
    # log-likelihood is identically 0 and only the prior has any say.
    features = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    labels = np.array([0, 0, 0])

    # Act
    posterior = lg.fit_laplace(features, labels, n_classes=1, prior_precision=2.0)

    # Assert
    assert posterior.reason is None
    assert np.allclose(posterior.mean, 0.0, atol=1e-10)


def test_single_class_posterior_covariance_is_exactly_the_prior_covariance():
    # Arrange: the likelihood contributes a zero Hessian when K = 1, because
    # diag(p) - p p^T is 1 - 1 = 0, so H is the prior term alone.
    features = np.array([[1.0, 2.0], [3.0, -1.0]])
    labels = np.array([0, 0])
    alpha = 4.0

    # Act
    posterior = lg.fit_laplace(
        features, labels, n_classes=1, prior_precision=alpha, with_covariance=True)

    # Assert
    assert posterior.covariance is not None
    assert np.allclose(posterior.covariance, np.eye(2) / alpha, atol=1e-10)


# --------------------------------------------------------------------------
# two-class problems, where the MAP's structure is provable
# --------------------------------------------------------------------------

def test_two_class_map_is_antisymmetric_across_its_columns():
    # Arrange: the likelihood depends only on the difference of the two weight
    # columns, while the prior penalises the sum of their squared norms. For a
    # fixed difference d the penalty is minimised at +d/2 and -d/2, so the MAP
    # must satisfy W[:, 0] = -W[:, 1] exactly.
    features = np.array([[1.0], [-1.0], [2.0], [-2.0]])
    labels = np.array([0, 1, 0, 1])

    # Act
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)

    # Assert
    assert posterior.reason is None
    assert np.allclose(posterior.mean[:, 0], -posterior.mean[:, 1], atol=1e-8)


def test_two_class_map_points_the_separating_direction_the_right_way():
    # Arrange: class 0 sits at positive x, class 1 at negative x, so the
    # weight column for class 0 must end up positive. Direction is provable;
    # magnitude depends on the prior and is not asserted.
    features = np.array([[1.0], [2.0], [-1.0], [-2.0]])
    labels = np.array([0, 0, 1, 1])

    # Act
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=0.5)

    # Assert
    assert posterior.reason is None
    assert posterior.mean[0, 0] > 0.0
    assert posterior.mean[0, 1] < 0.0


def test_convergence_is_reported_when_the_objective_stops_improving():
    # Arrange: a problem whose max-abs gradient cannot reach grad_tol, which at
    # C-8's real shape is every problem -- the gradient bottoms out near 1e-6
    # on float64 while the objective stops moving. The fit must report
    # converged on the objective criterion rather than running to max_iter and
    # reporting failure at an optimum it had already found.
    rng = np.random.default_rng(5)
    features = rng.normal(size=(60, 8))
    labels = rng.integers(0, 4, size=60)

    # Act
    posterior = lg.fit_laplace(
        features, labels, n_classes=4, prior_precision=1.0,
        max_iter=200, grad_tol=0.0)

    # Assert
    assert posterior.reason is None
    assert posterior.converged
    assert posterior.n_iter < 200


def test_a_strong_prior_drives_the_map_toward_zero():
    # Arrange: identical data, two prior precisions four orders apart. The
    # MAP under the tighter prior must be the smaller of the two.
    features = np.array([[1.0], [2.0], [-1.0], [-2.0]])
    labels = np.array([0, 0, 1, 1])

    # Act
    loose = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1e-2)
    tight = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1e4)

    # Assert
    assert np.linalg.norm(tight.mean) < np.linalg.norm(loose.mean)
    assert np.allclose(tight.mean, 0.0, atol=1e-3)


# --------------------------------------------------------------------------
# the Hessian, pinned against the softmax's shift invariance
# --------------------------------------------------------------------------

def test_uniform_shift_direction_is_an_eigenvector_of_the_hessian():
    # Arrange: adding the same vector to every class column leaves all softmax
    # outputs unchanged, so the likelihood Hessian annihilates that direction
    # and only the prior term survives. H v must equal alpha * v exactly.
    rng = np.random.default_rng(0)
    features = rng.normal(size=(7, 3))
    weights = rng.normal(size=(3, 4))
    alpha = 1.5
    shift = np.tile(rng.normal(size=(3, 1)), (1, 4))

    # Act
    product = lg.hessian_vector_product(features, weights, alpha, shift)

    # Assert
    assert np.allclose(product, alpha * shift, atol=1e-10)


def test_dense_hessian_agrees_with_the_hessian_vector_product():
    # Arrange: the dense matrix and the matrix-free product are two
    # implementations of one object, and the chain will only ever use the
    # second, so they are checked against each other.
    rng = np.random.default_rng(1)
    features = rng.normal(size=(9, 3))
    weights = rng.normal(size=(3, 2))
    alpha = 0.7
    v = rng.normal(size=(3, 2))

    # Act
    dense = lg.dense_hessian(features, weights, alpha)
    from_dense = (dense @ v.reshape(-1, order="F")).reshape(3, 2, order="F")
    matrix_free = lg.hessian_vector_product(features, weights, alpha, v)

    # Assert
    assert np.allclose(from_dense, matrix_free, atol=1e-10)


def test_dense_hessian_is_symmetric():
    # Arrange
    rng = np.random.default_rng(2)
    features = rng.normal(size=(6, 3))
    weights = rng.normal(size=(3, 3))

    # Act
    dense = lg.dense_hessian(features, weights, 1.0)

    # Assert
    assert np.allclose(dense, dense.T, atol=1e-10)


def test_dense_hessian_refuses_to_allocate_beyond_its_size_cap():
    # Arrange: D * K = 200 * 50 = 10000, so the dense matrix would be 10000 by
    # 10000. Refusing beats silently allocating it, which at C-8's real shape
    # (D = 256, K = 50) would be over a gigabyte.
    features = np.zeros((3, 200))
    weights = np.zeros((200, 50))

    # Act / Assert
    with pytest.raises(ValueError, match="max_dim"):
        lg.dense_hessian(features, weights, 1.0, max_dim=4096)


# --------------------------------------------------------------------------
# derivatives against finite differences
#
# The structural tests above constrain the Hessian's null space and the MAP's
# symmetry, but a sign error or a factor of two in the gradient would satisfy
# all of them. These two compare the analytic derivatives against the
# objective they are supposed to differentiate.
# --------------------------------------------------------------------------

def test_gradient_agrees_with_finite_differences_of_the_objective():
    # Arrange
    rng = np.random.default_rng(3)
    features = rng.normal(size=(8, 3))
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    weights = rng.normal(size=(3, 3)) * 0.4
    alpha = 0.9
    step = 1e-6

    # Act
    analytic = lg.gradient(features, labels, weights, alpha)
    numeric = np.zeros_like(analytic)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            bump = np.zeros_like(weights)
            bump[i, j] = step
            high = lg.negative_log_posterior(features, labels, weights + bump, alpha)
            low = lg.negative_log_posterior(features, labels, weights - bump, alpha)
            numeric[i, j] = (high - low) / (2 * step)

    # Assert
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_hessian_vector_product_agrees_with_finite_differences_of_the_gradient():
    # Arrange
    rng = np.random.default_rng(4)
    features = rng.normal(size=(8, 3))
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    weights = rng.normal(size=(3, 3)) * 0.4
    alpha = 0.9
    v = rng.normal(size=(3, 3))
    step = 1e-6

    # Act
    analytic = lg.hessian_vector_product(features, weights, alpha, v)
    high = lg.gradient(features, labels, weights + step * v, alpha)
    low = lg.gradient(features, labels, weights - step * v, alpha)
    numeric = (high - low) / (2 * step)

    # Assert
    assert np.allclose(analytic, numeric, atol=1e-6)


# --------------------------------------------------------------------------
# refusals: a posterior that cannot be formed must not be returned as one
# --------------------------------------------------------------------------

def test_zero_prior_precision_is_refused():
    # Arrange: this is the repository's default. train() passes wd = 0.0 into
    # AdamW, an improper flat prior, under which the shift direction above has
    # eigenvalue exactly zero and no posterior exists.
    features = np.array([[1.0], [-1.0]])
    labels = np.array([0, 1])

    # Act
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=0.0)

    # Assert
    assert posterior.reason is not None
    assert "prior" in posterior.reason
    assert np.all(np.isnan(posterior.mean))
    assert posterior.covariance is None


def test_negative_prior_precision_is_refused():
    # Arrange
    features = np.array([[1.0], [-1.0]])
    labels = np.array([0, 1])

    # Act
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=-1.0)

    # Assert
    assert posterior.reason is not None
    assert np.all(np.isnan(posterior.mean))


def test_all_zero_features_are_refused_for_having_no_data_content():
    # Arrange: a rank-zero feature matrix leaves the posterior equal to the
    # prior in every direction, so an agreement check against it would compare
    # the prior with itself and pass whatever the chain did.
    features = np.zeros((4, 3))
    labels = np.array([0, 1, 0, 1])

    # Act
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)

    # Assert
    assert posterior.reason is not None
    assert "rank" in posterior.reason
    assert np.all(np.isnan(posterior.mean))


def test_a_duplicated_feature_column_is_reported_not_refused():
    # Arrange: rank deficiency short of rank zero does NOT make the posterior
    # improper -- the prior term keeps the Hessian positive definite. What it
    # does is leave some directions determined by the prior alone, and a
    # reader needs to see how many. Refusing here would reject a healthy
    # input, which in this project is the same error as passing a gate that
    # cannot see.
    features = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]])
    labels = np.array([0, 0, 1, 1])

    # Act
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)

    # Assert
    assert posterior.reason is None
    assert posterior.data_rank == 1
    assert posterior.prior_only_directions == 1


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def test_mismatched_feature_and_label_counts_are_rejected():
    # Arrange
    features = np.array([[1.0], [2.0], [3.0]])
    labels = np.array([0, 1])

    # Act / Assert
    with pytest.raises(ValueError, match="same length"):
        lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)


def test_a_label_outside_the_class_range_is_rejected():
    # Arrange
    features = np.array([[1.0], [2.0]])
    labels = np.array([0, 5])

    # Act / Assert
    with pytest.raises(ValueError, match="class range"):
        lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)


def test_an_empty_feature_matrix_is_rejected():
    # Arrange
    features = np.zeros((0, 3))
    labels = np.zeros(0, dtype=int)

    # Act / Assert
    with pytest.raises(ValueError, match="empty"):
        lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------

def test_predicted_probabilities_sum_to_one_on_every_row():
    # Arrange
    features = np.array([[1.0], [2.0], [-1.0]])
    labels = np.array([0, 0, 1])
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)

    # Act
    proba = lg.predict_proba(posterior, features)

    # Assert
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_predicted_classes_recover_separable_labels():
    # Arrange: cleanly separable data under a weak prior, so the plug-in
    # prediction at the posterior mean must reproduce the labels.
    features = np.array([[3.0], [2.0], [-2.0], [-3.0]])
    labels = np.array([0, 0, 1, 1])
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1e-3)

    # Act
    predicted = lg.predict_classes(posterior, features)

    # Assert
    assert list(predicted) == list(labels)


def test_predicting_from_a_refused_posterior_is_rejected():
    # Arrange
    features = np.array([[1.0], [-1.0]])
    labels = np.array([0, 1])
    refused = lg.fit_laplace(features, labels, n_classes=2, prior_precision=0.0)

    # Act / Assert
    with pytest.raises(ValueError, match="not evaluable"):
        lg.predict_classes(refused, features)


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------

def test_the_posterior_result_cannot_be_rebound():
    # Arrange
    features = np.array([[1.0], [-1.0]])
    labels = np.array([0, 1])
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)

    # Act / Assert
    with pytest.raises(Exception):
        posterior.mean = np.zeros((1, 2))


def test_the_held_mean_array_is_read_only():
    # Arrange: frozen=True blocks rebinding the attribute but not writing
    # through the array it points at, the same hazard NullTestResult guards.
    features = np.array([[1.0], [-1.0]])
    labels = np.array([0, 1])
    posterior = lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)

    # Act / Assert
    with pytest.raises(ValueError):
        posterior.mean[0, 0] = 99.0


def test_fitting_does_not_mutate_its_inputs():
    # Arrange
    features = np.array([[1.0, 2.0], [-1.0, 0.5]])
    labels = np.array([0, 1])
    features_before, labels_before = features.copy(), labels.copy()

    # Act
    lg.fit_laplace(features, labels, n_classes=2, prior_precision=1.0)

    # Assert
    assert np.array_equal(features, features_before)
    assert np.array_equal(labels, labels_before)
