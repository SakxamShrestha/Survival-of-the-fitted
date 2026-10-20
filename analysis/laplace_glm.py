"""
Closed-form Laplace posterior for a multinomial GLM on fixed features.

This is the analytic half of control C-8's correctness bridge. With the MLP's
body frozen, ``spike_iterated_distill.MLP``'s head is a multinomial GLM over a
fixed 256-dimensional random feature space, so its Laplace posterior is
closed-form and the analytic prediction and the experiment MUST agree.
RESEARCH-PLAN.md section 4.1 calls that "a correctness check on the sampler
implementation that nothing else in the design provides", which is why this
module is built and tested on its own, before any chain imports it: a silently
wrong posterior here would not fail anywhere downstream. The chain would simply
agree with a wrong prediction, or disagree with it for a reason nobody could
attribute.

**The prior is not optional, and that is the point.** The softmax is
over-parameterised: adding the same vector to every class's weight column
leaves every predicted probability unchanged. Writing the Hessian as H and that
all-columns-equal direction as v, the likelihood term annihilates v exactly, so

    H v = alpha * v

where alpha is the prior precision. At alpha = 0 that eigenvalue is zero, the
Hessian is singular with a null space of dimension D, and no posterior exists.
Every existing arm in this repository calls ``train()`` with ``wd = 0.0``
(spike_iterated_distill.py:100), an improper flat prior, so this is the
degenerate input the project is most likely to hand this module. It is refused
rather than answered, and ``gates/thresholds.json``'s S-1 entry requires a
non-zero weight decay in every sampler run for the same reason.

NumPy only, matching ``analysis/overlap_stats.py``. Torch is not used: the
Hessian-vector product for this model is four lines of array algebra, so
autograd would add a dependency and hide the structure that the eigenvector
identity above is checked against.

**Size.** The dense Hessian is (D*K) by (D*K). At C-8's real shape, D = 256 and
K = 50, that is 12800 by 12800 in float64 -- about 1.3 GB. Forming it silently
would be a worse failure than refusing to, so ``dense_hessian`` has a cap and
``fit_laplace`` finds the MAP matrix-free, by Newton-CG over the
Hessian-vector product, and forms the covariance only when explicitly asked.
The posterior MEAN is all C-8's agreement check needs; the covariance is for
S-1, when it samples.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# D*K beyond which dense_hessian refuses to allocate. 4096 squared in float64
# is 134 MB, which is a reasonable ceiling for an interactive check; C-8's real
# shape is far above it and is expected to use the matrix-free path.
DEFAULT_MAX_HESSIAN_DIM = 4096

# Newton-CG settings. The objective is strictly convex for prior_precision > 0,
# so these control accuracy rather than whether it converges at all.
DEFAULT_MAX_ITER = 100
DEFAULT_GRAD_TOL = 1e-10
# Relative objective-improvement floor. The gradient tolerance above is
# unreachable at scale: at C-8's shape (D = 256, K = 50, N = 4096) the
# max-abs gradient bottoms out around 1e-6, the float64 noise floor for a
# 12800-parameter sum over 4096 terms, while the objective stops changing
# entirely. Without a second criterion the fit reports converged=False after
# any number of iterations, which would land in C-8's log and read as a
# failure when the optimum had in fact been reached. Measured at three
# iteration budgets: identical objective at 25, 100 and 300.
DEFAULT_OBJ_RTOL = 1e-12
_CG_MAX_ITER = 250
_CG_TOL = 1e-12
_BACKTRACK_SHRINK = 0.5
_BACKTRACK_MAX = 40


@dataclass(frozen=True)
class LaplacePosterior:
    """A Laplace posterior over a multinomial GLM's weights, or a refusal.

    mean                  -- (D, K) MAP weights, the posterior's mean under the
                             Laplace approximation. All-nan exactly when
                             `reason` is set, following the convention in
                             analysis/rank_stats.py: a value that could not be
                             measured is not reported as a number.
    covariance            -- (D*K, D*K) inverse Hessian at the MAP in
                             column-major (Fortran) vec order, or None when not
                             requested or not formable. C-8 does not need it;
                             S-1 does, when it samples.
    prior_precision       -- alpha, the L2 prior precision actually used.
    data_rank             -- rank of the feature matrix.
    prior_only_directions -- D - data_rank: how many weight directions the data
                             says nothing about, so the posterior there is the
                             prior. Reported rather than refused, because a
                             rank-deficient feature matrix still has a proper
                             posterior; what it loses is data content, and a
                             reader has to be able to see how much.
    n_iter / converged    -- Newton-CG diagnostics. A non-converged fit is
                             reported as such rather than silently returned.
    reason                -- why the posterior could not be formed, or None.
    """

    mean: np.ndarray
    covariance: Optional[np.ndarray]
    prior_precision: float
    data_rank: int
    prior_only_directions: int
    n_iter: int
    converged: bool
    reason: Optional[str]

    def __post_init__(self):
        """Make the held arrays read-only.

        `frozen=True` blocks rebinding an attribute but not writing through the
        array it points at, the same hazard overlap_stats.NullTestResult
        guards. A caller editing the MAP in place after a comparison had been
        read off it would leave the two silently disagreeing.
        """
        self.mean.flags.writeable = False
        if self.covariance is not None:
            self.covariance.flags.writeable = False


# ------------------------------------------------------------ internals

def _softmax_rows(z):
    """Row-wise softmax, shifted by each row's maximum for stability."""
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _validate(features, labels, n_classes):
    """Return fresh float/int copies, rejecting malformed input.

    These raise rather than returning a refusal: a length mismatch is a caller
    bug, not a degenerate measurement, and the two should not be reported in
    the same channel.
    """
    phi = np.asarray(features, dtype=float)
    if phi.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {phi.shape}")
    if phi.size == 0:
        raise ValueError("features is empty; a posterior needs at least one observation")

    y = np.asarray(labels).ravel()
    if y.size != phi.shape[0]:
        raise ValueError(
            f"features and labels must be the same length; got {phi.shape[0]} and {y.size}"
        )
    if n_classes < 1:
        raise ValueError(f"n_classes must be at least 1, got {n_classes}")
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(
            f"labels {y.min()} to {y.max()} fall outside the class range "
            f"0 to {n_classes - 1}"
        )
    return phi, y.astype(int)


def _one_hot(labels, n_classes):
    out = np.zeros((labels.size, n_classes), dtype=float)
    out[np.arange(labels.size), labels] = 1.0
    return out


# ------------------------------------------------------------ objective

def negative_log_posterior(features, labels, weights, prior_precision):
    """Negative log posterior: cross-entropy plus the L2 prior term."""
    phi = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    y = np.asarray(labels).ravel().astype(int)

    logits = phi @ w
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    chosen = shifted[np.arange(y.size), y]
    nll = float(-(chosen - log_norm).sum())
    return nll + 0.5 * float(prior_precision) * float(w.ravel() @ w.ravel())


def gradient(features, labels, weights, prior_precision):
    """d(negative log posterior)/dW, shape (D, K)."""
    phi = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    y = np.asarray(labels).ravel().astype(int)

    probs = _softmax_rows(phi @ w)
    return phi.T @ (probs - _one_hot(y, w.shape[1])) + float(prior_precision) * w


def hessian_vector_product(features, weights, prior_precision, v):
    """H @ v without forming H, for v shaped (D, K).

    Derivation. The likelihood block for classes (k, k') is
    Phi^T diag(P[:,k] delta_kk' - P[:,k] P[:,k']) Phi, so for U = Phi V the
    per-row action is P*U - P * rowsum(P*U), which is four array operations and
    costs O(N*D*K) rather than O((D*K)^2).

    The all-columns-equal direction is annihilated exactly by the likelihood
    term, because rowsum(P * U) equals U's shared column when P's rows sum to
    one, leaving only the prior term. That identity is what
    tests/test_laplace_glm.py pins.
    """
    phi = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    direction = np.asarray(v, dtype=float).reshape(w.shape)

    probs = _softmax_rows(phi @ w)
    u = phi @ direction
    weighted = probs * u
    inner = weighted - probs * weighted.sum(axis=1, keepdims=True)
    return phi.T @ inner + float(prior_precision) * direction


def dense_hessian(features, weights, prior_precision, max_dim=DEFAULT_MAX_HESSIAN_DIM):
    """The full (D*K, D*K) Hessian in column-major vec order.

    Index (d, k) sits at row d + k*D, so a caller reshapes with order="F".
    Refuses above `max_dim` rather than allocating: see this module's docstring
    on why 1.3 GB is a worse outcome than an exception.
    """
    phi = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    n_features, n_classes = w.shape
    size = n_features * n_classes

    if size > max_dim:
        raise ValueError(
            f"dense Hessian would be {size} by {size}; above max_dim={max_dim}. "
            "Use hessian_vector_product, which is matrix-free."
        )

    probs = _softmax_rows(phi @ w)
    out = np.zeros((size, size), dtype=float)
    for k in range(n_classes):
        for k2 in range(n_classes):
            delta = 1.0 if k == k2 else 0.0
            scale = probs[:, k] * delta - probs[:, k] * probs[:, k2]
            block = phi.T @ (phi * scale[:, None])
            if k == k2:
                block = block + float(prior_precision) * np.eye(n_features)
            out[k * n_features:(k + 1) * n_features,
                k2 * n_features:(k2 + 1) * n_features] = block
    return out


# ------------------------------------------------------------ the fit

def _conjugate_gradient(hvp, rhs):
    """Solve H p = rhs for p, with H available only as a product."""
    p = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    rs_old = float(residual.ravel() @ residual.ravel())

    for _ in range(_CG_MAX_ITER):
        if rs_old <= _CG_TOL:
            break
        hd = hvp(direction)
        denominator = float(direction.ravel() @ hd.ravel())
        if denominator <= 0.0:
            break
        step = rs_old / denominator
        p = p + step * direction
        residual = residual - step * hd
        rs_new = float(residual.ravel() @ residual.ravel())
        direction = residual + (rs_new / rs_old) * direction
        rs_old = rs_new
    return p


def fit_laplace(features, labels, n_classes, prior_precision,
                with_covariance=False, max_iter=DEFAULT_MAX_ITER,
                grad_tol=DEFAULT_GRAD_TOL, obj_rtol=DEFAULT_OBJ_RTOL,
                max_dim=DEFAULT_MAX_HESSIAN_DIM):
    """MAP fit plus the Laplace posterior around it, or a refusal.

    Two refusals, both returned as a `reason` rather than raised, because both
    are properties of the data and the prior rather than caller errors:

      - prior_precision <= 0, under which the posterior does not exist at all
        (see the module docstring's eigenvector identity);
      - a rank-zero feature matrix, under which the posterior equals the prior
        in every direction, so an agreement check against it would compare the
        prior with itself and pass whatever the chain did.

    Rank deficiency short of rank zero is reported in `prior_only_directions`
    and NOT refused. The prior term keeps the Hessian positive definite, so the
    posterior is proper; refusing it would reject a healthy input, which in this
    project is the same class of error as passing a gate that cannot see.
    """
    phi, y = _validate(features, labels, n_classes)
    n_features = phi.shape[1]
    alpha = float(prior_precision)
    nan_mean = np.full((n_features, n_classes), np.nan)

    if alpha <= 0.0:
        return LaplacePosterior(
            mean=nan_mean, covariance=None, prior_precision=alpha,
            data_rank=int(np.linalg.matrix_rank(phi)),
            prior_only_directions=n_features - int(np.linalg.matrix_rank(phi)),
            n_iter=0, converged=False,
            reason=(
                f"prior precision {alpha} is not positive, so the softmax's shift "
                "direction has eigenvalue zero and the posterior is improper; "
                "train()'s wd = 0.0 is exactly this case"
            ),
        )

    data_rank = int(np.linalg.matrix_rank(phi))
    if data_rank == 0:
        return LaplacePosterior(
            mean=nan_mean, covariance=None, prior_precision=alpha,
            data_rank=0, prior_only_directions=n_features,
            n_iter=0, converged=False,
            reason=(
                "the feature matrix has rank 0, so the posterior equals the prior in "
                "every direction and carries no data content to compare against"
            ),
        )

    weights = np.zeros((n_features, n_classes), dtype=float)
    objective = negative_log_posterior(phi, y, weights, alpha)
    converged = False
    used = 0

    for used in range(1, max_iter + 1):
        grad = gradient(phi, y, weights, alpha)
        if float(np.abs(grad).max()) < grad_tol:
            converged = True
            used -= 1
            break

        step_dir = _conjugate_gradient(
            lambda v: hessian_vector_product(phi, weights, alpha, v), -grad)
        if float(grad.ravel() @ step_dir.ravel()) >= 0.0:
            step_dir = -grad  # CG returned no descent direction; fall back

        scale = 1.0
        for _ in range(_BACKTRACK_MAX):
            trial = weights + scale * step_dir
            trial_objective = negative_log_posterior(phi, y, trial, alpha)
            if trial_objective <= objective:
                break
            scale *= _BACKTRACK_SHRINK
        else:
            converged = True  # no step improves the objective; at the optimum
            break

        # A step that does not strictly improve the objective means the
        # optimum has been reached to working precision. Accepting equality as
        # progress is what previously let this loop run to max_iter reporting
        # converged=False while the objective had stopped moving.
        improvement = objective - trial_objective
        weights, objective = trial, trial_objective
        if improvement <= obj_rtol * max(1.0, abs(objective)):
            converged = True
            break
    else:
        converged = float(np.abs(gradient(phi, y, weights, alpha)).max()) < grad_tol

    covariance = None
    if with_covariance:
        hessian = dense_hessian(phi, weights, alpha, max_dim=max_dim)
        covariance = np.linalg.inv(hessian)

    return LaplacePosterior(
        mean=weights, covariance=covariance, prior_precision=alpha,
        data_rank=data_rank, prior_only_directions=n_features - data_rank,
        n_iter=used, converged=converged, reason=None,
    )


# ------------------------------------------------------------ prediction

def _usable(posterior):
    if posterior.reason is not None:
        raise ValueError(
            f"posterior is not evaluable and carries no mean to predict from: "
            f"{posterior.reason}"
        )


def predict_logits(posterior, features):
    """Logits at the posterior mean. Plug-in, which is what C-8 compares."""
    _usable(posterior)
    return np.asarray(features, dtype=float) @ posterior.mean


def predict_proba(posterior, features):
    """Class probabilities at the posterior mean."""
    return _softmax_rows(predict_logits(posterior, features))


def predict_classes(posterior, features):
    """Argmax class at the posterior mean, the quantity C-8's agreement uses."""
    return predict_logits(posterior, features).argmax(axis=1)
