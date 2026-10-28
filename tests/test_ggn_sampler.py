"""
Tests for analysis/ggn_sampler.py, the S-1 draw function.

Synthetic data only; nothing is read from logs/. Arrange-Act-Assert throughout.

What is pinned, and why each is an identity rather than a fitted expectation:

  - The Gaussian branch of the draw objective is algebraically randomize-then-
    optimize with targets y + s * xi / sqrt(lambda) (gates/thresholds.json S-1,
    categorical_sampler_amendment.method). For a linear model the minimiser is
    closed form, so a draw from the iterative path must equal the closed-form
    RTO solution for the SAME anchor and the SAME xi.
  - The categorical GGN objective is bounded below because p_hat - y and
    B^{1/2} xi are orthogonal to the all-ones vector, which is B's null space.
    Equivalently the objective is invariant to adding a constant to every
    logit of a row.
  - B_i = diag(p) - p p^T is PSD and B^{1/2} squares back to B.
  - Anchors are drawn per parameter tensor at that tensor's layer sd, with each
    bias taking its own layer's weight sd.
  - The MAP pass minimises lambda * sum CE + sum_l ||w_l||^2 / (2 sigma_l^2):
    sum reduction, explicit L2, and no optimizer weight decay.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import ggn_sampler as gs  # noqa: E402

DT = torch.float64


# ------------------------------------------------------------ helpers

def _linear_gaussian_problem(seed=0, n=30, d=4, noise_sd=0.5, prior_sd=1.3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = X @ (prior_sd * rng.standard_normal(d)) + noise_sd * rng.standard_normal(n)
    return torch.tensor(X, dtype=DT), torch.tensor(y, dtype=DT)


def _closed_form_rto(X, y, theta0, xi, lam, noise_sd, prior_sd):
    """argmin lam ||y + s xi / sqrt(lam) - X w||^2 / (2 s^2) + ||w - theta0||^2 / (2 p^2)."""
    d = X.shape[1]
    prec = lam * X.T @ X / noise_sd ** 2 + torch.eye(d, dtype=DT) / prior_sd ** 2
    out = []
    for s in range(theta0.shape[0]):
        y_t = y + noise_sd * xi[s, :, 0] / math.sqrt(lam)
        rhs = lam * X.T @ y_t / noise_sd ** 2 + theta0[s] / prior_sd ** 2
        out.append(torch.linalg.solve(prec, rhs))
    return torch.stack(out)


# ------------------------------------------------------------ prior sds

def test_bias_takes_its_own_layers_weight_sd():
    # Arrange
    names = ["body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias",
             "head.weight", "head.bias"]
    match = {"body.0.weight": 0.0817, "body.2.weight": 0.036, "head.weight": 0.0361}

    # Act
    sds = gs.layer_prior_sds(names, match, multiplier=1.0)

    # Assert
    assert sds["body.0.bias"] == pytest.approx(0.0817)
    assert sds["body.2.bias"] == pytest.approx(0.036)
    assert sds["head.bias"] == pytest.approx(0.0361)
    assert sds["head.weight"] == pytest.approx(0.0361)


def test_multiplier_scales_every_layer():
    match = {"body.0.weight": 0.0817, "body.2.weight": 0.036, "head.weight": 0.0361}
    sds = gs.layer_prior_sds(["body.0.weight", "head.bias"], match, multiplier=4.0)
    assert sds["body.0.weight"] == pytest.approx(4 * 0.0817)
    assert sds["head.bias"] == pytest.approx(4 * 0.0361)


def test_layer_without_a_measured_sd_is_refused():
    with pytest.raises(KeyError):
        gs.layer_prior_sds(["mystery.weight"], {"head.weight": 0.0361}, 1.0)


def test_non_positive_prior_sd_is_refused():
    # A zero or negative sd is the improper prior S-1's sanity_check forbids.
    with pytest.raises(ValueError):
        gs.layer_prior_sds(["head.weight"], {"head.weight": 0.0361}, multiplier=0.0)


# ------------------------------------------------------------ anchors

def test_anchor_draws_have_the_per_layer_sd_including_bias():
    # Arrange
    sds = {"a.weight": 0.08, "a.bias": 0.08, "b.weight": 0.5, "b.bias": 0.5}
    shapes = {"a.weight": (40, 30), "a.bias": (40,), "b.weight": (20, 40), "b.bias": (20,)}
    gen = torch.Generator().manual_seed(0)

    # Act
    theta0, xi = gs.draw_noise(sds, shapes, n_draws=500, n_rows=7, n_classes=3,
                               generator=gen)

    # Assert
    for name, sd in sds.items():
        assert theta0[name].shape == (500,) + shapes[name]
        assert float(theta0[name].std()) == pytest.approx(sd, rel=0.05)
        assert float(theta0[name].mean()) == pytest.approx(0.0, abs=3 * sd / math.sqrt(theta0[name].numel()) + 1e-12)
    assert xi.shape == (500, 7, 3)
    assert float(xi.std()) == pytest.approx(1.0, rel=0.05)


def test_draw_noise_is_reproducible_from_the_generator():
    sds = {"w": 1.0}
    shapes = {"w": (3,)}
    a = gs.draw_noise(sds, shapes, 4, 5, 2, torch.Generator().manual_seed(9))
    b = gs.draw_noise(sds, shapes, 4, 5, 2, torch.Generator().manual_seed(9))
    assert torch.equal(a[0]["w"], b[0]["w"]) and torch.equal(a[1], b[1])


# ------------------------------------------------------------ curvature

def test_categorical_curvature_is_psd_and_its_root_squares_back():
    # Arrange
    gen = torch.Generator().manual_seed(1)
    p = F.softmax(3.0 * torch.randn(16, 6, generator=gen, dtype=DT), -1)

    # Act
    B, B_half = gs.categorical_curvature(p)

    # Assert
    assert torch.allclose(B, B.transpose(-1, -2))
    assert float(torch.linalg.eigvalsh(B).min()) > -1e-12
    assert torch.allclose(B_half @ B_half, B, atol=1e-10)
    assert torch.allclose(B_half, B_half.transpose(-1, -2), atol=1e-12)


def test_ones_vector_is_in_the_null_space_of_b_and_its_root():
    gen = torch.Generator().manual_seed(2)
    p = F.softmax(torch.randn(10, 5, generator=gen, dtype=DT), -1)
    B, B_half = gs.categorical_curvature(p)
    ones = torch.ones(5, dtype=DT)
    assert float((B @ ones).abs().max()) < 1e-12
    assert float((B_half @ ones).abs().max()) < 1e-7


def test_expansion_terms_are_orthogonal_to_ones():
    # The two linear terms in the GGN objective must have no component along
    # the logit-shift direction, otherwise the objective is unbounded below.
    gen = torch.Generator().manual_seed(3)
    f_hat = torch.randn(12, 4, generator=gen, dtype=DT)
    labels = torch.randint(0, 4, (12,), generator=gen)
    lik = gs.CategoricalLikelihood(labels, 4)
    exp = lik.expansion(f_hat)
    xi = torch.randn(1, 12, 4, generator=gen, dtype=DT)

    assert float(exp.residual.sum(-1).abs().max()) < 1e-12
    noise = torch.einsum("nkj,snj->snk", exp.curvature_sqrt, xi)
    assert float(noise.sum(-1).abs().max()) < 1e-7


def test_ggn_objective_is_invariant_to_a_per_row_logit_shift():
    # Arrange: the loss in logit space only, no prior.
    gen = torch.Generator().manual_seed(4)
    f_hat = torch.randn(9, 5, generator=gen, dtype=DT)
    labels = torch.randint(0, 5, (9,), generator=gen)
    exp = gs.CategoricalLikelihood(labels, 5).expansion(f_hat)
    xi = torch.randn(1, 9, 5, generator=gen, dtype=DT)
    f = torch.randn(1, 9, 5, generator=gen, dtype=DT)
    shift = 1e3 * torch.randn(1, 9, 1, generator=gen, dtype=DT)

    # Act
    a = gs.ggn_logit_loss(f, exp, xi, lam=10.0)
    b = gs.ggn_logit_loss(f + shift, exp, xi, lam=10.0)

    # Assert
    assert float(a) == pytest.approx(float(b), rel=1e-9, abs=1e-6)


def test_ggn_objective_is_bounded_below_along_random_rays():
    # A quadratic with PSD curvature and linear terms in its range attains a
    # finite minimum; far along any ray the loss must not go to -infinity.
    gen = torch.Generator().manual_seed(5)
    f_hat = torch.randn(6, 4, generator=gen, dtype=DT)
    labels = torch.randint(0, 4, (6,), generator=gen)
    exp = gs.CategoricalLikelihood(labels, 4).expansion(f_hat)
    xi = torch.randn(1, 6, 4, generator=gen, dtype=DT)
    for _ in range(20):
        ray = torch.randn(1, 6, 4, generator=gen, dtype=DT)
        far = [float(gs.ggn_logit_loss(f_hat[None] + t * ray, exp, xi, 1.0))
               for t in (1e2, 1e3, 1e4)]
        assert far[2] >= far[1] - 1e-6 or far[2] > 0


# ------------------------------------------------------------ Gaussian branch

def test_gaussian_expansion_is_rto_curvature():
    X, y = _linear_gaussian_problem()
    lik = gs.GaussianLikelihood(y, noise_sd=0.5)
    f_hat = torch.zeros(X.shape[0], 1, dtype=DT)
    exp = lik.expansion(f_hat)
    assert torch.allclose(exp.curvature, torch.full((X.shape[0], 1, 1), 4.0, dtype=DT))
    assert torch.allclose(exp.curvature_sqrt, torch.full((X.shape[0], 1, 1), 2.0, dtype=DT))
    assert torch.allclose(exp.residual[:, 0], (f_hat[:, 0] - y) / 0.25)


@pytest.mark.parametrize("lam", [1.0, 10.0, 100.0])
def test_gaussian_branch_draw_equals_closed_form_rto(lam):
    # Arrange
    X, y = _linear_gaussian_problem(seed=11)
    noise_sd, prior_sd = 0.5, 1.3
    lik = gs.GaussianLikelihood(y, noise_sd)
    sds = {"w": prior_sd}
    init = {"w": torch.zeros(1, X.shape[1], dtype=DT)}
    gen = torch.Generator().manual_seed(0)

    # Act
    mp = gs.map_pass(gs.linear_gaussian_forward, init, X, lik, lam, sds)
    theta0, xi = gs.draw_noise(sds, {"w": (X.shape[1],)}, n_draws=6,
                               n_rows=X.shape[0], n_classes=1, generator=gen)
    dr = gs.draw_pass(gs.linear_gaussian_forward, X, mp.expansion, lam, sds, theta0, xi)

    # Assert
    expected = _closed_form_rto(X, y, theta0["w"], xi, lam, noise_sd, prior_sd)
    assert torch.allclose(dr.params["w"], expected, atol=1e-7)


def test_gaussian_map_is_the_tempered_posterior_mean():
    X, y = _linear_gaussian_problem(seed=12)
    lam, s, p = 10.0, 0.5, 1.3
    mp = gs.map_pass(gs.linear_gaussian_forward,
                     {"w": torch.zeros(1, X.shape[1], dtype=DT)},
                     X, gs.GaussianLikelihood(y, s), lam, {"w": p})
    prec = lam * X.T @ X / s ** 2 + torch.eye(X.shape[1], dtype=DT) / p ** 2
    mean = torch.linalg.solve(prec, lam * X.T @ y / s ** 2)
    assert torch.allclose(mp.params["w"][0], mean, atol=1e-8)


# ------------------------------------------------------------ MAP objective

def test_map_objective_is_lambda_times_sum_ce_plus_explicit_l2():
    # Arrange
    gen = torch.Generator().manual_seed(6)
    feats = torch.randn(10, 3, generator=gen, dtype=DT)
    labels = torch.randint(0, 4, (10,), generator=gen)
    params = {"head.weight": torch.randn(1, 4, 3, generator=gen, dtype=DT),
              "head.bias": torch.randn(1, 4, generator=gen, dtype=DT)}
    sds = {"head.weight": 0.7, "head.bias": 0.7}
    lam = 3.0

    # Act
    got = gs.map_objective(gs.linear_head_forward, params, feats,
                           gs.CategoricalLikelihood(labels, 4), lam, sds)

    # Assert
    logits = feats @ params["head.weight"][0].T + params["head.bias"][0]
    ce_sum = F.cross_entropy(logits, labels, reduction="sum")
    l2 = sum((v ** 2).sum() / (2 * 0.7 ** 2) for v in params.values())
    assert float(got) == pytest.approx(float(lam * ce_sum + l2), rel=1e-12)


def test_map_pass_reaches_stationarity_of_its_own_objective():
    gen = torch.Generator().manual_seed(7)
    feats = torch.randn(20, 3, generator=gen, dtype=DT)
    labels = torch.randint(0, 4, (20,), generator=gen)
    sds = {"head.weight": 1.0, "head.bias": 1.0}
    init = {"head.weight": torch.zeros(1, 4, 3, dtype=DT),
            "head.bias": torch.zeros(1, 4, dtype=DT)}
    mp = gs.map_pass(gs.linear_head_forward, init, feats,
                     gs.CategoricalLikelihood(labels, 4), 5.0, sds)
    assert mp.diagnostics["max_abs_grad"] < 1e-6
    assert mp.diagnostics["optimizer"] == "adam_cosine+lbfgs"
    assert mp.diagnostics["weight_decay"] == 0.0
    assert mp.diagnostics["reduction"] == "sum"


def test_map_pass_does_not_mutate_its_initial_params():
    X, y = _linear_gaussian_problem(seed=13)
    init = {"w": torch.zeros(1, X.shape[1], dtype=DT)}
    before = init["w"].clone()
    gs.map_pass(gs.linear_gaussian_forward, init, X, gs.GaussianLikelihood(y, 0.5),
                1.0, {"w": 1.3})
    assert torch.equal(init["w"], before)


# ------------------------------------------------------------ head GLM exactness

def test_head_draws_are_map_plus_inverse_hessian_noise_for_a_glm():
    # For a linear head the GGN quadratic is exact, so the draw has the closed
    # form w = w_map + H^-1 (sqrt(lam) J^T B^{1/2} xi + A theta0), H = lam J^T B J + A.
    gen = torch.Generator().manual_seed(8)
    n, g, k, lam, sd = 15, 3, 3, 4.0, 0.9
    feats = torch.randn(n, g, generator=gen, dtype=DT)
    labels = torch.randint(0, k, (n,), generator=gen)
    sds = {"head.weight": sd, "head.bias": sd}
    init = {"head.weight": torch.zeros(1, k, g, dtype=DT), "head.bias": torch.zeros(1, k, dtype=DT)}
    lik = gs.CategoricalLikelihood(labels, k)
    mp = gs.map_pass(gs.linear_head_forward, init, feats, lik, lam, sds)
    theta0, xi = gs.draw_noise(sds, {"head.weight": (k, g), "head.bias": (k,)}, 3, n, k, gen)
    dr = gs.draw_pass(gs.linear_head_forward, feats, mp.expansion, lam, sds, theta0, xi)

    phi = torch.cat([feats, torch.ones(n, 1, dtype=DT)], 1)          # (n, g+1)
    # vec order: index (k, d) -> k*(g+1) + d ; J maps vec(W_aug) to logits (n, k)
    J = torch.zeros(n * k, k * (g + 1), dtype=DT)
    for i in range(n):
        for c in range(k):
            J[i * k + c, c * (g + 1):(c + 1) * (g + 1)] = phi[i]
    Bblk = torch.block_diag(*mp.expansion.curvature)
    Rblk = torch.block_diag(*mp.expansion.curvature_sqrt)
    H = lam * J.T @ Bblk @ J + torch.eye(k * (g + 1), dtype=DT) / sd ** 2
    w_map = torch.cat([mp.params["head.weight"][0], mp.params["head.bias"][0][:, None]], 1).reshape(-1)
    for s in range(3):
        th = torch.cat([theta0["head.weight"][s], theta0["head.bias"][s][:, None]], 1).reshape(-1)
        rhs = math.sqrt(lam) * J.T @ Rblk @ xi[s].reshape(-1) + th / sd ** 2
        expect = w_map + torch.linalg.solve(H, rhs)
        got = torch.cat([dr.params["head.weight"][s], dr.params["head.bias"][s][:, None]], 1).reshape(-1)
        assert torch.allclose(got, expect, atol=1e-6)


# ------------------------------------------------------------ MLP plumbing

def test_functional_mlp_matches_the_module():
    from spike_iterated_distill import MLP
    torch.manual_seed(0)
    model = MLP(50, 7)
    X = torch.rand(9, 50)
    params = gs.params_from_module(model)
    got = gs.mlp_forward(params, X.to(DT))[0]
    want = model.double()(X.to(DT))
    assert torch.allclose(got, want, atol=1e-12)


def test_module_round_trip_preserves_parameters():
    from spike_iterated_distill import MLP
    torch.manual_seed(1)
    model = MLP(50, 7)
    params = gs.params_from_module(model)
    back = gs.module_from_params(params, 50, 7)
    for name, p in back.named_parameters():
        assert torch.equal(p.detach(), params[name][0])


def test_mlp_draw_runs_and_moves_away_from_the_map():
    from spike_iterated_distill import MLP
    torch.manual_seed(2)
    X = torch.rand(16, 50, dtype=DT)
    labels = torch.randint(0, 5, (16,))
    match = {"body.0.weight": 0.0817, "body.2.weight": 0.036, "head.weight": 0.0361}
    init = gs.params_from_module(MLP(50, 5))
    sds = gs.layer_prior_sds(list(init), match, 4.0)
    lik = gs.CategoricalLikelihood(labels, 5)
    mp = gs.map_pass(gs.mlp_forward, init, X, lik, 10.0, sds,
                     opt=gs.OptConfig(adam_steps=200, max_iter=50))
    gen = torch.Generator().manual_seed(0)
    shapes = {k: tuple(v.shape[1:]) for k, v in mp.params.items()}
    theta0, xi = gs.draw_noise(sds, shapes, 1, 16, 5, gen)
    dr = gs.draw_pass(gs.mlp_forward, X, mp.expansion, 10.0, sds, theta0, xi,
                      opt=gs.OptConfig(adam_steps=200, max_iter=50))
    assert math.isfinite(dr.diagnostics["final_loss"])
    moved = sum(float((dr.params[k] - mp.params[k]).abs().max()) for k in sds)
    assert moved > 0


def test_per_draw_expansion_equals_separate_shared_expansions():
    # Batched MAPs: the loss with a per-draw expansion is the sum of the losses
    # each draw would have under its own shared expansion.
    gen = torch.Generator().manual_seed(9)
    labels = torch.randint(0, 4, (7,), generator=gen)
    lik = gs.CategoricalLikelihood(labels, 4)
    f_hat = torch.randn(3, 7, 4, generator=gen, dtype=DT)
    f = torch.randn(3, 7, 4, generator=gen, dtype=DT)
    xi = torch.randn(3, 7, 4, generator=gen, dtype=DT)
    batched = gs.ggn_logit_loss(f, lik.expansion(f_hat), xi, 5.0)
    separate = sum(gs.ggn_logit_loss(f[s:s + 1], lik.expansion(f_hat[s]), xi[s:s + 1], 5.0)
                   for s in range(3))
    assert float(batched) == pytest.approx(float(separate), rel=1e-12)


def test_batched_map_pass_matches_single_map_passes():
    X, y = _linear_gaussian_problem(seed=14)
    lik = gs.GaussianLikelihood(y, 0.5)
    init = {"w": torch.randn(3, X.shape[1], generator=torch.Generator().manual_seed(0), dtype=DT)}
    batched = gs.map_pass(gs.linear_gaussian_forward, init, X, lik, 2.0, {"w": 1.3})
    for s in range(3):
        one = gs.map_pass(gs.linear_gaussian_forward, {"w": init["w"][s:s + 1]}, X, lik, 2.0, {"w": 1.3})
        assert torch.allclose(batched.params["w"][s], one.params["w"][0], atol=1e-7)
    assert batched.f_hat.shape == (3, X.shape[0], 1)
