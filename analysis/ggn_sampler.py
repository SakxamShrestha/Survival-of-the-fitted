"""
The S-1 weight-draw step: a MAP pass and a GGN draw pass.

This module implements exactly the method committed in gates/thresholds.json,
entry "S-1", field categorical_sampler_amendment.method (2026-10-27). It changes
no constant of that rule; every tunable here is an optimizer setting, which the
rule leaves to the implementation and requires only to converge.

Per generation, on the transmitted batch (x_i, y_i), i = 1..B:

  (1) MAP pass. Minimise  lambda * sum_i CE(f(x_i; w), y_i)
                          + sum_l ||w_l||^2 / (2 sigma_l^2),
      full batch, reduction "sum", with no optimizer weight decay: the prior is
      an explicit term in the loss. Record f_hat_i and p_hat_i.

  (2) Draw pass. Draw theta0_l ~ N(0, sigma_l^2 I) for every parameter tensor
      and xi_i ~ N(0, I_K), initialise at theta0 and minimise
         lambda * sum_i [ 1/2 d_i^T B_i d_i + (p_hat_i - y_i)^T d_i ]
         - sqrt(lambda) * sum_i xi_i^T B_i^{1/2} f_i
         + sum_l ||w_l - theta0_l||^2 / (2 sigma_l^2),
      with d_i = f_i - f_hat_i and B_i = diag(p_hat_i) - p_hat_i p_hat_i^T.

The network stays nonlinear; only the likelihood is replaced by its GGN
quadratic in logit space around the MAP. Bias parameters take the prior sd of
their own layer's weights (the amendment closes that gap; layer_prior_sds
implements it).

**One draw function, three call sites.** The amendment requires that the SAME
draw function be run through the linear-Gaussian precondition. That is arranged
by writing the draw pass against an abstract expansion (f_hat, B, B^{1/2}, and
the residual, which is the gradient of the negative log-likelihood in logit
space at f_hat) and an abstract functional forward. The Gaussian likelihood
supplies B_i = I / s^2, B_i^{1/2} = I / s and residual (f_hat - y) / s^2, under
which the objective is algebraically randomize-then-optimize with targets
y + s * xi / sqrt(lambda); the categorical likelihood supplies the softmax
GGN. draw_pass and map_pass are the same code in every case; only the
likelihood object and the forward differ.

**Optimizer.** Full-batch cosine-decayed Adam followed by an L-BFGS polish with
a strong-Wolfe line search, in float64, with ONE configuration for both passes
and at every call site, so that the inner-optimizer convergence
the linear-Gaussian precondition certifies is the convergence of the optimizer
the chain actually uses. AdamW at a fixed step count would not be certified by
that check. Convergence is reported (max |grad|) and never assumed.

**Vectorisation.** Every parameter tensor carries a leading draw dimension S,
and the objective is the sum of S independent objectives, so S draws at a fixed
MAP are optimised in one call. For the chain S = 1.

No function here mutates its arguments.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

DTYPE = torch.float64

# Optimizer defaults. One configuration, used by both passes at every call site
# (linear-Gaussian precondition, head-GLM precondition, chain). Phase 1 is
# full-batch Adam with a cosine-decayed learning rate, which does the heavy
# lifting on the nonconvex MLP; phase 2 is an L-BFGS polish (strong Wolfe, so
# it only accepts loss-decreasing steps), which drives the convex call sites to
# tolerance_grad. Measured on 2026-10-27 (scratch, not in logs/): on the MLP
# neither optimizer alone reaches a gradient tolerance at lambda = 100 or at
# 4 x sigma_match, lambda = 10, and cosine Adam reached the lower loss.
DEFAULT_ADAM_STEPS = 3_000
DEFAULT_ADAM_LR = 1e-3
DEFAULT_MAX_ITER = 500
DEFAULT_TOL_GRAD = 1e-9
DEFAULT_TOL_CHANGE = 1e-15
DEFAULT_HISTORY = 20
DEFAULT_MAX_ROUNDS = 2


@dataclass(frozen=True)
class OptConfig:
    """Settings shared by the MAP pass and the draw pass."""

    adam_steps: int = DEFAULT_ADAM_STEPS
    adam_lr: float = DEFAULT_ADAM_LR
    max_iter: int = DEFAULT_MAX_ITER
    tolerance_grad: float = DEFAULT_TOL_GRAD
    tolerance_change: float = DEFAULT_TOL_CHANGE
    history_size: int = DEFAULT_HISTORY
    max_rounds: int = DEFAULT_MAX_ROUNDS


DEFAULT_OPT = OptConfig()


# ------------------------------------------------------------ prior

def layer_prior_sds(param_names, sigma_match_weights, multiplier):
    """Per-tensor prior sd: each tensor takes its layer's measured weight sd.

    ``sigma_match_weights`` maps "<layer>.weight" to that layer's measured
    initialization sd (S-1's sigma_match_measured). A "<layer>.bias" tensor
    takes "<layer>.weight"'s sd, as the amendment requires. ``multiplier`` is
    1 for sigma_match and 4 for 4 x sigma_match.
    """
    if not multiplier > 0:
        raise ValueError(
            f"prior sd multiplier must be positive, got {multiplier}; a zero sd "
            "is the improper prior S-1's sanity_check forbids")
    out = {}
    for name in param_names:
        layer = name.rsplit(".", 1)[0] if "." in name else name
        key = f"{layer}.weight" if "." in name else name
        if key not in sigma_match_weights:
            raise KeyError(f"no measured prior sd for {name!r} (looked up {key!r})")
        sd = float(sigma_match_weights[key]) * float(multiplier)
        if not sd > 0:
            raise ValueError(f"prior sd for {name!r} is {sd}; must be positive")
        out[name] = sd
    return out


def prior_term(params, sds, center=None):
    """sum_l ||w_l - c_l||^2 / (2 sd_l^2), summed over the draw dimension too."""
    total = 0.0
    for name, value in params.items():
        diff = value if center is None else value - center[name]
        total = total + (diff ** 2).sum() / (2.0 * sds[name] ** 2)
    return total


# ------------------------------------------------------------ curvature

def categorical_curvature(probs):
    """B = diag(p) - p p^T per row and its symmetric PSD square root.

    ``probs`` is (..., K). The root is taken by eigendecomposition with negative
    rounding-level eigenvalues clamped to zero, so B^{1/2} is symmetric, PSD and
    annihilates the all-ones vector to rounding.
    """
    p = probs.to(DTYPE)
    B = torch.diag_embed(p) - p[..., :, None] * p[..., None, :]
    B = 0.5 * (B + B.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(B)
    root = evecs @ torch.diag_embed(evals.clamp_min(0.0).sqrt()) @ evecs.transpose(-1, -2)
    # Project onto the complement of the all-ones vector. B 1 = 0 exactly, so
    # P B P = B and (P R P)^2 = P R^2 P = B: this removes only eigh's rounding
    # leak along 1, which otherwise makes the noise term see a logit shift.
    k = p.shape[-1]
    proj = torch.eye(k, dtype=DTYPE) - torch.full((k, k), 1.0 / k, dtype=DTYPE)
    root = proj @ root @ proj
    root = 0.5 * (root + root.transpose(-1, -2))
    return B, root


@dataclass(frozen=True)
class GGNExpansion:
    """The likelihood's quadratic expansion in logit space around f_hat.

    f_hat          (N, K)     logits at the MAP
    curvature      (N, K, K)  B_i
    curvature_sqrt (N, K, K)  B_i^{1/2}
    residual       (N, K)     d NLL / d f at f_hat: p_hat - y (categorical),
                              (f_hat - y) / s^2 (Gaussian)
    """

    f_hat: torch.Tensor
    curvature: torch.Tensor
    curvature_sqrt: torch.Tensor
    residual: torch.Tensor


class CategoricalLikelihood:
    """Hard labels under a softmax: an honest categorical likelihood."""

    kind = "categorical"

    def __init__(self, labels, n_classes):
        self.labels = torch.as_tensor(labels).long().reshape(-1)
        self.n_classes = int(n_classes)
        if self.labels.numel() and (int(self.labels.min()) < 0
                                    or int(self.labels.max()) >= self.n_classes):
            raise ValueError("labels fall outside the class range")

    def nll_sum(self, f):
        s = f.shape[0]
        return F.cross_entropy(f.reshape(-1, self.n_classes),
                               self.labels.repeat(s), reduction="sum")

    def expansion(self, f_hat):
        f_hat = f_hat.detach().to(DTYPE)
        p = F.softmax(f_hat, -1)
        B, root = categorical_curvature(p)
        onehot = F.one_hot(self.labels, self.n_classes).to(DTYPE)
        return GGNExpansion(f_hat=f_hat, curvature=B, curvature_sqrt=root,
                            residual=p - onehot)


class GaussianLikelihood:
    """y = f + eps, eps ~ N(0, s^2). K = 1; the GGN expansion is exact."""

    kind = "gaussian"

    def __init__(self, targets, noise_sd):
        if not noise_sd > 0:
            raise ValueError(f"noise_sd must be positive, got {noise_sd}")
        self.targets = torch.as_tensor(targets, dtype=DTYPE).reshape(-1)
        self.noise_sd = float(noise_sd)
        self.n_classes = 1

    def nll_sum(self, f):
        return 0.5 * ((f[..., 0] - self.targets) ** 2).sum() / self.noise_sd ** 2

    def expansion(self, f_hat):
        f_hat = f_hat.detach().to(DTYPE)
        lead = tuple(f_hat.shape[:-1])
        prec = 1.0 / self.noise_sd ** 2
        return GGNExpansion(
            f_hat=f_hat,
            curvature=torch.full(lead + (1, 1), prec, dtype=DTYPE),
            curvature_sqrt=torch.full(lead + (1, 1), 1.0 / self.noise_sd, dtype=DTYPE),
            residual=(f_hat - self.targets[:, None]) * prec,
        )


# ------------------------------------------------------------ objectives

def ggn_logit_loss(f, expansion, xi, lam):
    """The draw objective's likelihood part, summed over draws and rows.

    lam * sum_i [1/2 d^T B d + r^T d] - sqrt(lam) * sum_i xi^T B^{1/2} f.
    ``f`` and ``xi`` are (S, N, K). The expansion is either shared by all S
    draws (f_hat (N, K)) or one per draw (f_hat (S, N, K)), the latter when
    each draw has its own MAP.
    """
    if expansion.f_hat.dim() == 2:
        d = f - expansion.f_hat[None]
        quad = 0.5 * torch.einsum("snk,nkj,snj->", d, expansion.curvature, d)
        lin = (expansion.residual[None] * d).sum()
        noise = (torch.einsum("nkj,snj->snk", expansion.curvature_sqrt, xi) * f).sum()
    else:
        d = f - expansion.f_hat
        quad = 0.5 * torch.einsum("snk,snkj,snj->", d, expansion.curvature, d)
        lin = (expansion.residual * d).sum()
        noise = (torch.einsum("snkj,snj->snk", expansion.curvature_sqrt, xi) * f).sum()
    return lam * (quad + lin) - math.sqrt(lam) * noise


def map_objective(forward, params, X, likelihood, lam, sds):
    """lam * sum NLL + explicit L2 prior centred at zero."""
    return lam * likelihood.nll_sum(forward(params, X)) + prior_term(params, sds)


def draw_objective(forward, params, X, expansion, lam, sds, theta0, xi):
    """The amendment's draw objective, anchored at theta0."""
    return (ggn_logit_loss(forward(params, X), expansion, xi, lam)
            + prior_term(params, sds, center=theta0))


# ------------------------------------------------------------ optimizer

def _minimise(loss_fn, init, opt):
    """Full-batch L-BFGS from ``init`` (not modified). Returns (params, diag)."""
    leaves = {k: v.detach().clone().to(DTYPE).requires_grad_(True) for k, v in init.items()}
    plist = list(leaves.values())
    started = time.time()
    adam_loss = None
    if opt.adam_steps > 0:
        adam = torch.optim.Adam(plist, lr=opt.adam_lr, weight_decay=0.0)
        for step in range(opt.adam_steps):
            for group in adam.param_groups:
                group["lr"] = opt.adam_lr * 0.5 * (1.0 + math.cos(math.pi * step / opt.adam_steps))
            loss = loss_fn(leaves)
            adam.zero_grad(set_to_none=True)
            loss.backward()
            adam.step()
        adam_loss = float(loss.detach())
    total_iter, rounds = 0, 0
    max_grad = float("inf")
    for rounds in range(1, opt.max_rounds + 1):
        lbfgs = torch.optim.LBFGS(
            plist, lr=1.0, max_iter=opt.max_iter, max_eval=int(opt.max_iter * 1.25),
            tolerance_grad=opt.tolerance_grad, tolerance_change=opt.tolerance_change,
            history_size=opt.history_size, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad(set_to_none=False)
            loss = loss_fn(leaves)
            loss.backward()
            return loss

        lbfgs.step(closure)
        total_iter += int(lbfgs.state[lbfgs._params[0]].get("n_iter", 0))
        final = closure()
        max_grad = max(float(p.grad.abs().max()) for p in plist)
        if max_grad <= opt.tolerance_grad:
            break
    out = {k: v.detach().clone() for k, v in leaves.items()}
    diag = {
        "optimizer": "adam_cosine+lbfgs", "line_search": "strong_wolfe",
        "adam_steps": int(opt.adam_steps), "adam_lr": float(opt.adam_lr),
        "loss_after_adam": adam_loss,
        "weight_decay": 0.0, "reduction": "sum", "dtype": str(DTYPE),
        "lbfgs_iter": int(total_iter), "rounds": int(rounds),
        "final_loss": float(final.detach()), "max_abs_grad": float(max_grad),
        "converged": bool(max_grad <= opt.tolerance_grad),
        "tolerance_grad": opt.tolerance_grad,
        "elapsed_sec": round(time.time() - started, 3),
    }
    return out, diag


# ------------------------------------------------------------ the two passes

@dataclass(frozen=True)
class MapResult:
    params: Dict[str, torch.Tensor]
    f_hat: torch.Tensor
    expansion: GGNExpansion
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DrawResult:
    params: Dict[str, torch.Tensor]
    diagnostics: dict = field(default_factory=dict)


def map_pass(forward, init_params, X, likelihood, lam, sds, opt=DEFAULT_OPT):
    """Pass (1). ``init_params`` carries a leading dimension S.

    S = 1 is one MAP (f_hat (N, K), shared expansion). S > 1 fits S independent
    MAPs from S initialisations in one call (f_hat (S, N, K), one expansion per
    MAP), which is how the MAP-to-MAP spread is measured.
    """
    if not lam > 0:
        raise ValueError(f"lambda must be positive, got {lam}")
    Xd = X.to(DTYPE)
    params, diag = _minimise(
        lambda p: map_objective(forward, p, Xd, likelihood, lam, sds), init_params, opt)
    with torch.no_grad():
        f_hat = forward(params, Xd)
        f_hat = f_hat[0] if f_hat.shape[0] == 1 else f_hat
    diag = {**diag, "lambda": float(lam)}
    return MapResult(params=params, f_hat=f_hat, expansion=likelihood.expansion(f_hat),
                     diagnostics=diag)


def draw_noise(sds, shapes, n_draws, n_rows, n_classes, generator):
    """theta0_l ~ N(0, sd_l^2 I) per tensor (in ``shapes`` order), then xi ~ N(0, I_K)."""
    theta0 = {
        name: sds[name] * torch.randn((n_draws,) + tuple(shape), generator=generator,
                                      dtype=DTYPE)
        for name, shape in shapes.items()
    }
    xi = torch.randn((n_draws, n_rows, n_classes), generator=generator, dtype=DTYPE)
    return theta0, xi


def draw_pass(forward, X, expansion, lam, sds, theta0, xi, opt=DEFAULT_OPT):
    """Pass (2): initialise at theta0 and minimise the GGN draw objective."""
    if not lam > 0:
        raise ValueError(f"lambda must be positive, got {lam}")
    Xd = X.to(DTYPE)
    params, diag = _minimise(
        lambda p: draw_objective(forward, p, Xd, expansion, lam, sds, theta0, xi),
        theta0, opt)
    return DrawResult(params=params, diagnostics={**diag, "lambda": float(lam)})


def sample(forward, map_result, X, lam, sds, n_draws, generator, opt=DEFAULT_OPT):
    """Draw ``n_draws`` weight samples at a fixed MAP. Returns (DrawResult, theta0, xi)."""
    if map_result.f_hat.dim() != 2:
        raise ValueError("sample() draws at ONE fixed MAP; use draw_pass for per-draw MAPs")
    shapes = {k: tuple(v.shape[1:]) for k, v in map_result.params.items()}
    k = map_result.f_hat.shape[-1]
    theta0, xi = draw_noise(sds, shapes, n_draws, X.shape[0], k, generator)
    return draw_pass(forward, X, map_result.expansion, lam, sds, theta0, xi, opt), theta0, xi


# ------------------------------------------------------------ forwards

def linear_gaussian_forward(params, X):
    """f = X w, K = 1. params['w'] is (S, D)."""
    return torch.einsum("nd,sd->sn", X, params["w"])[..., None]


def linear_head_forward(params, feats):
    """The MLP's head on fixed features: feats @ W^T + b, leading draw dim."""
    return (torch.einsum("ng,skg->snk", feats, params["head.weight"])
            + params["head.bias"][:, None, :])


def mlp_forward(params, X, masks=None):
    """spike_iterated_distill.MLP as a function of (S, ...) parameters.

    ``masks``, when given, maps "h1"/"h2" to (S, h) or (S, N, h) multipliers on
    the two hidden ReLU layers; used only by the MC-dropout negative control.
    """
    h1 = F.relu(torch.einsum("nd,shd->snh", X, params["body.0.weight"])
                + params["body.0.bias"][:, None, :])
    if masks is not None:
        h1 = h1 * (masks["h1"][:, None, :] if masks["h1"].dim() == 2 else masks["h1"])
    h2 = F.relu(torch.einsum("snh,sgh->sng", h1, params["body.2.weight"])
                + params["body.2.bias"][:, None, :])
    if masks is not None:
        h2 = h2 * (masks["h2"][:, None, :] if masks["h2"].dim() == 2 else masks["h2"])
    return (torch.einsum("sng,skg->snk", h2, params["head.weight"])
            + params["head.bias"][:, None, :])


def params_from_module(module):
    """A module's parameters as float64 tensors with a leading draw dim of 1."""
    return {n: p.detach().to(DTYPE).clone()[None] for n, p in module.named_parameters()}


def module_from_params(params, d_in, n_classes, index=0):
    """Draw ``index`` as a float64 spike_iterated_distill.MLP."""
    from spike_iterated_distill import MLP
    model = MLP(d_in, n_classes).to(DTYPE)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(params[name][index])
    return model


def select_draw(params, index):
    """One draw's parameters, keeping a leading dim of 1."""
    return {k: v[index:index + 1].clone() for k, v in params.items()}
