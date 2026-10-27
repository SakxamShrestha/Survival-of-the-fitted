"""
Linear-Gaussian exactness check for the anchored (randomized-MAP) sampler.

This module exists because the primary method of ``gates/thresholds.json``'s
S-1 entry -- "anchored / randomized-MAP at K = 1" -- has two plausible
implementations that differ by three lines of code, agree on their mean, and
differ by more than an order of magnitude in their spread. One of them is a
posterior sampler and the other is not.

    variant A, anchor only.  Draw an anchor w_a ~ N(0, sigma_p^2 I) and
        minimize  ||y - X w||^2 / (2 sigma_n^2)  +  ||w - w_a||^2 / (2 sigma_p^2).

    variant B, randomize-then-optimize.  Draw the anchor AND perturb the
        targets, y~ = y + e with e ~ N(0, sigma_n^2 I), then minimize
        ||y~ - X w||^2 / (2 sigma_n^2)  +  ||w - w_a||^2 / (2 sigma_p^2).

Writing the posterior precision as A = X'X / sigma_n^2 + I / sigma_p^2 and the
posterior covariance as Sigma = A^-1, both variants are unbiased for the
posterior mean, and their covariances are

    variant A:  Sigma (I / sigma_p^2) Sigma
    variant B:  Sigma A Sigma  =  Sigma

so variant B reproduces the posterior covariance identically and variant A
reproduces it only when Sigma = sigma_p^2 I, which is to say only when there is
no data. This is algebra and not a sampling artefact: variant A's shortfall
grows with N and does not shrink with the number of draws. RESEARCH-PLAN.md
section 4.2 stated variant A's formula while citing variant B's exactness
result, which is why this check is worth having as code rather than as a
remark.

**What this buys S-1.** S-1's inertness precondition, its MC-dropout negative
control and its gate C all test whether the sampler's draws DIFFER. None of
them tests whether the draws differ by the RIGHT AMOUNT. Variant A passes all
three -- its draws genuinely are distinct, it is not MC dropout, and the prior
still enters through the anchor, so sigma-dependent attractors remain
plausible -- while being roughly sixteen to nineteen times too narrow at the
shape measured below. An arm built on variant A would therefore reach a verdict
intact, and a null result would read as "the sampler/maximizer distinction does
not transfer" when the truth was "the sampler was not a sampler". That is
verbatim the distinction RESEARCH-PLAN.md section 7 requires be settled in
advance and calls worthless to write afterwards.

**Scope, stated plainly.** The draws here are produced by solving the normal
equations, not by running SGD, because the quantity under test is the
estimator's target distribution and not an optimizer's convergence. A sampler
that passes this check can still be wrong in the real arm for reasons this
check cannot see -- an incompletely converged inner optimization, a
misconfigured weight decay, a per-layer sigma mismatch. Passing is necessary
and not sufficient. What a failure here means is unambiguous, which is the
useful direction for a precondition.

NumPy only, matching ``analysis/overlap_stats.py`` and
``analysis/laplace_glm.py``. No torch, no frozen body, no GLM and no chain: the
check is seconds of array algebra, by design, so that there is no cost argument
for skipping it. No function here mutates its arguments.

Run the recorded comparison with::

    python3 analysis/anchored_sampler_check.py | tee logs/anchored_sampler_exactness.txt
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Shape of the recorded comparison. D is small enough that the dense exact
# posterior is free, and N is comfortably above D so that the data term
# dominates the prior term -- which is the regime the real arm runs in and the
# regime in which variant A fails worst.
DEFAULT_N = 60
DEFAULT_D = 8
DEFAULT_NOISE_SD = 0.5
DEFAULT_PRIOR_SD = 1.3

# Draws per variant. The relative standard error of an estimated standard
# deviation is 1 / sqrt(2 * draws), which is 0.5 percent here, so the five
# percent tolerance below sits ten Monte-Carlo standard errors away from a
# correct sampler and cannot be tripped by noise.
DEFAULT_DRAWS = 20_000

# Tolerances for the pass/fail rule. Both are deliberately loose relative to
# the failure they exist to catch: variant A comes in near 0.05 on the sd ratio
# and near 0.003 on the trace ratio, so a 20x margin on either side of 1.0
# would still separate the two variants. They are tight relative to Monte-Carlo
# noise, per the note on DEFAULT_DRAWS.
SD_RATIO_TOLERANCE = 0.05
TRACE_RATIO_TOLERANCE = 0.05

# Centring tolerance, in standard errors of the exact posterior's own marginal
# standard deviation. Five is chosen for the multiplicity across D coordinates:
# a two-sided five-sigma coordinate-wise bound has a family-wise false-alarm
# rate below 1e-5 at D = 8. Using the EXACT posterior's sd as the denominator
# rather than the sampler's own is conservative in a narrow sampler's favour,
# which is intended -- variant A should pass the centring test, because being
# correctly centred is exactly what makes it dangerous.
MEAN_TOLERANCE_SIGMAS = 5.0

# Sample sizes for the degradation table. N = 0 is included because it is the
# one case in which variant A is exact, and seeing 1.0 there is what identifies
# the mechanism as "the data term breaks it" rather than "the implementation is
# buggy".
DEGRADATION_SIZES = (0, 4, 8, 16, 64, 256)


@dataclass(frozen=True)
class ExactPosterior:
    """The closed-form Gaussian posterior for the linear-Gaussian model."""

    mean: np.ndarray
    cov: np.ndarray

    @property
    def marginal_sd(self):
        return np.sqrt(np.diag(self.cov))


@dataclass(frozen=True)
class VariantCheck:
    """One variant's agreement with the exact posterior.

    ``reason`` is set exactly when ``passed`` is False, and names which of the
    three comparisons failed. ``sd_ratio_min`` and ``sd_ratio_max`` are the
    extremes over coordinates of empirical marginal sd divided by exact
    marginal sd; a range rather than a mean, because a sampler can be correct
    in some directions and collapsed in others and a mean would hide it.
    """

    variant: str
    max_mean_dev: float
    max_mean_dev_sigmas: float
    sd_ratio_min: float
    sd_ratio_max: float
    trace_ratio: float
    n_draws: int
    seed: int
    passed: bool
    reason: Optional[str]


def make_design(n_samples, n_features, rng):
    """A standard-normal design matrix and a true weight vector.

    The true weights are drawn from the prior, so the test is not dependent on
    the data being well explained; the posterior is correct whatever w_true is,
    and drawing it from the prior simply keeps the simulated y on scale.
    """
    design = rng.standard_normal((n_samples, n_features))
    return design


def simulate_targets(design, w_true, noise_sd, rng):
    """Observations from the linear-Gaussian likelihood. ``design`` is unchanged."""
    return design @ w_true + noise_sd * rng.standard_normal(design.shape[0])


def exact_posterior(design, targets, noise_sd, prior_sd):
    """Closed-form posterior for y = X w + eps, eps ~ N(0, noise_sd^2 I),
    w ~ N(0, prior_sd^2 I).

    Returns an ``ExactPosterior``. The precision is strictly positive definite
    for any prior_sd > 0, including N = 0, so no case here is degenerate --
    unlike the flat-prior input ``analysis/laplace_glm.py`` refuses.
    """
    _validate_scales(noise_sd, prior_sd)
    n_features = design.shape[1]
    precision = (
        design.T @ design / noise_sd ** 2
        + np.eye(n_features) / prior_sd ** 2
    )
    cov = np.linalg.inv(precision)
    mean = cov @ (design.T @ targets) / noise_sd ** 2
    return ExactPosterior(mean=mean, cov=cov)


def _validate_scales(noise_sd, prior_sd):
    """Reject the two scales that make the posterior undefined."""
    if not noise_sd > 0:
        raise ValueError(f"noise_sd must be positive; got {noise_sd}")
    if not prior_sd > 0:
        raise ValueError(
            f"prior_sd must be positive; got {prior_sd}. A flat prior has no "
            "proper posterior and no anchor distribution to draw from -- the "
            "same degenerate input S-1's sanity_check clause forbids."
        )


def _solve_penalised(precision, design, targets, anchor, noise_sd, prior_sd):
    """Exact minimizer of the anchored penalised least-squares objective."""
    rhs = design.T @ targets / noise_sd ** 2 + anchor / prior_sd ** 2
    return np.linalg.solve(precision, rhs)


def draw_variant(variant, design, targets, noise_sd, prior_sd, rng, n_draws):
    """``n_draws`` by D array of weight draws from one variant.

    ``variant`` is "anchor_only" (RESEARCH-PLAN.md's original formula) or
    "randomize_then_optimize" (the corrected one). Neither argument array is
    modified.
    """
    if variant not in ("anchor_only", "randomize_then_optimize"):
        raise ValueError(f"unknown variant {variant!r}")

    _validate_scales(noise_sd, prior_sd)
    n_samples, n_features = design.shape
    precision = (
        design.T @ design / noise_sd ** 2 + np.eye(n_features) / prior_sd ** 2
    )
    perturb = variant == "randomize_then_optimize"

    draws = [
        _solve_penalised(
            precision, design,
            targets + noise_sd * rng.standard_normal(n_samples) if perturb else targets,
            prior_sd * rng.standard_normal(n_features),
            noise_sd, prior_sd,
        )
        for _ in range(n_draws)
    ]
    return np.asarray(draws)


def summarise_draws(variant, draws, posterior, seed):
    """Compare a variant's draws against the exact posterior.

    This is the function S-1's precondition reads. The three comparisons are
    the posterior mean, every marginal standard deviation, and the total
    variance; a variant passes only if all three agree within the tolerances
    fixed at the top of this module.
    """
    n_draws = draws.shape[0]
    mean_dev = np.abs(draws.mean(axis=0) - posterior.mean)
    standard_errors = posterior.marginal_sd / np.sqrt(n_draws)
    sd_ratios = draws.std(axis=0, ddof=1) / posterior.marginal_sd
    trace_ratio = float(
        np.sum(draws.var(axis=0, ddof=1)) / np.trace(posterior.cov)
    )

    max_sigmas = float(np.max(mean_dev / standard_errors))
    reason = _first_failure(
        max_sigmas, float(np.min(sd_ratios)), float(np.max(sd_ratios)), trace_ratio
    )
    return VariantCheck(
        variant=variant,
        max_mean_dev=float(np.max(mean_dev)),
        max_mean_dev_sigmas=max_sigmas,
        sd_ratio_min=float(np.min(sd_ratios)),
        sd_ratio_max=float(np.max(sd_ratios)),
        trace_ratio=trace_ratio,
        n_draws=int(n_draws),
        seed=int(seed),
        passed=reason is None,
        reason=reason,
    )


def _first_failure(max_sigmas, sd_min, sd_max, trace_ratio):
    """The first comparison that fell outside tolerance, as a sentence, or None."""
    if max_sigmas > MEAN_TOLERANCE_SIGMAS:
        return (
            f"the sampler is not centred on the posterior mean: worst coordinate "
            f"is {max_sigmas:.1f} standard errors out, tolerance "
            f"{MEAN_TOLERANCE_SIGMAS:.0f}"
        )
    if abs(sd_min - 1.0) > SD_RATIO_TOLERANCE or abs(sd_max - 1.0) > SD_RATIO_TOLERANCE:
        narrow = 1.0 / sd_min if sd_min > 0 else float("inf")
        return (
            f"marginal standard deviations are wrong: ratio to the exact "
            f"posterior spans {sd_min:.4f} to {sd_max:.4f}, tolerance "
            f"1 +/- {SD_RATIO_TOLERANCE:.2f}. The narrowest coordinate is "
            f"{narrow:.1f}x too narrow"
        )
    if abs(trace_ratio - 1.0) > TRACE_RATIO_TOLERANCE:
        return (
            f"total posterior variance is wrong: trace ratio {trace_ratio:.4f}, "
            f"tolerance 1 +/- {TRACE_RATIO_TOLERANCE:.2f}"
        )
    return None


def check_linear_gaussian_exactness(
    variant="randomize_then_optimize", seed=0, n_samples=DEFAULT_N,
    n_features=DEFAULT_D, noise_sd=DEFAULT_NOISE_SD, prior_sd=DEFAULT_PRIOR_SD,
    n_draws=DEFAULT_DRAWS,
):
    """The reusable precondition: does this variant sample its own stated target?

    The anchored sampler's claim to be a posterior sampler rests on a stated
    exactness condition -- exact in the linear-Gaussian case -- so this tests
    that condition directly rather than testing a proxy for it. Returns a
    ``VariantCheck``; read ``passed`` for the verdict and ``reason`` for why
    not.
    """
    rng = np.random.default_rng(seed)
    design = make_design(n_samples, n_features, rng)
    w_true = prior_sd * rng.standard_normal(n_features)
    targets = simulate_targets(design, w_true, noise_sd, rng)
    posterior = exact_posterior(design, targets, noise_sd, prior_sd)
    draws = draw_variant(
        variant, design, targets, noise_sd, prior_sd, rng, n_draws
    )
    return summarise_draws(variant, draws, posterior, seed)


def anchor_only_cov(design, noise_sd, prior_sd):
    """Closed-form covariance of variant A's draws: Sigma (I / sigma_p^2) Sigma.

    Analytic rather than sampled, so the degradation table below carries no
    Monte-Carlo error and the claim "this is algebra, not sampling noise" is
    checkable in one line.
    """
    _validate_scales(noise_sd, prior_sd)
    n_features = design.shape[1]
    cov = np.linalg.inv(
        design.T @ design / noise_sd ** 2 + np.eye(n_features) / prior_sd ** 2
    )
    return cov @ cov / prior_sd ** 2


def degradation_with_sample_size(
    sizes=DEGRADATION_SIZES, n_features=DEFAULT_D, noise_sd=DEFAULT_NOISE_SD,
    prior_sd=DEFAULT_PRIOR_SD, seed=0,
):
    """Variant A's analytic sd ratio as a function of N.

    Returns a tuple of ``(n_samples, mean_sd_ratio)`` pairs. The ratio is 1.0
    at N = 0 and falls monotonically as the data term grows, which locates the
    defect in the missing target perturbation rather than anywhere else.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n_samples in sizes:
        design = make_design(n_samples, n_features, rng)
        posterior_cov = np.linalg.inv(
            design.T @ design / noise_sd ** 2 + np.eye(n_features) / prior_sd ** 2
        )
        ratio = np.sqrt(
            np.diag(anchor_only_cov(design, noise_sd, prior_sd))
            / np.diag(posterior_cov)
        )
        rows.append((int(n_samples), float(np.mean(ratio))))
    return tuple(rows)


def _print_check(check):
    """One variant's result, as the log records it."""
    verdict = "PASS" if check.passed else "FAIL"
    print(f"  {check.variant:<26} seed {check.seed}  ->  {verdict}")
    print(f"    max abs(mean - mu)     {check.max_mean_dev:.6f} "
          f"({check.max_mean_dev_sigmas:.2f} standard errors)")
    print(f"    marginal sd ratio      {check.sd_ratio_min:.4f} to "
          f"{check.sd_ratio_max:.4f}")
    print(f"    trace ratio            {check.trace_ratio:.4f}")
    if check.reason is not None:
        print(f"    reason                 {check.reason}")


def main(argv=None):
    """Record the comparison both variants are judged by."""
    import argparse
    import json
    import os
    import sys
    import time

    parser = argparse.ArgumentParser(
        description="Linear-Gaussian exactness check for the anchored sampler."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                        help="seeds to run; at least two, so that a result is "
                             "not one draw of the design matrix")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--noise-sd", type=float, default=DEFAULT_NOISE_SD)
    parser.add_argument("--prior-sd", type=float, default=DEFAULT_PRIOR_SD)
    parser.add_argument("--no-log", action="store_true",
                        help="print the comparison without writing the JSON log")
    args = parser.parse_args(argv)

    if len(args.seeds) < 2:
        parser.error("at least two seeds are required")

    started = time.time()
    print("=== anchored sampler: linear-Gaussian exactness check ===")
    print(f"model   y = X w + eps,  eps ~ N(0, {args.noise_sd}^2 I),  "
          f"w ~ N(0, {args.prior_sd}^2 I)")
    print(f"shape   N = {args.n}, D = {args.d}, {args.draws} draws per variant")
    print("target  the closed-form posterior; both variants are compared to it")
    print()

    results = []
    for variant in ("anchor_only", "randomize_then_optimize"):
        label = ("RESEARCH-PLAN.md's original formula"
                 if variant == "anchor_only" else "the corrected formula")
        print(f"{variant}  -- {label}")
        for seed in args.seeds:
            check = check_linear_gaussian_exactness(
                variant=variant, seed=seed, n_samples=args.n,
                n_features=args.d, noise_sd=args.noise_sd,
                prior_sd=args.prior_sd, n_draws=args.draws,
            )
            _print_check(check)
            results.append(check)
        print()

    table = degradation_with_sample_size(
        n_features=args.d, noise_sd=args.noise_sd, prior_sd=args.prior_sd
    )
    print("anchor_only, analytic mean marginal sd ratio by sample size")
    print("(no sampling involved; the shortfall is algebraic and does not")
    print(" shrink with more draws)")
    for n_samples, ratio in table:
        print(f"    N = {n_samples:<4} sd ratio {ratio:.4f}")
    print()

    anchor_failed = all(not c.passed for c in results if c.variant == "anchor_only")
    rto_passed = all(
        c.passed for c in results if c.variant == "randomize_then_optimize"
    )
    print(f"=== VERDICT  ({time.time() - started:.1f}s) ===")
    if anchor_failed and rto_passed:
        print("anchor-only FAILS the exactness condition it is cited as satisfying,")
        print("on every seed. Randomize-then-optimize PASSES on every seed. The two")
        print("differ by three lines of code and agree on the posterior mean, so no")
        print("check that only asks whether draws differ can tell them apart.")
    else:
        print("UNEXPECTED: the algebra in this module's docstring predicts anchor-only")
        print("fails and randomize-then-optimize passes. One of them did not. Do not")
        print("read this run as a verdict; find out why first.")

    if args.no_log:
        return 0

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "logs", "anchored_sampler_exactness.json")
    with open(out, "w") as handle:
        json.dump({
            "check": "anchored sampler linear-Gaussian exactness",
            "purpose": (
                "RESEARCH-PLAN.md section 4.2 stated the anchor-only formula while "
                "citing randomize-then-optimize's exactness result. This records "
                "which of the two samples the posterior it claims to."
            ),
            "model": {
                "n_samples": args.n, "n_features": args.d,
                "noise_sd": args.noise_sd, "prior_sd": args.prior_sd,
            },
            "n_draws": args.draws,
            "seeds": list(args.seeds),
            "tolerances": {
                "sd_ratio": SD_RATIO_TOLERANCE,
                "trace_ratio": TRACE_RATIO_TOLERANCE,
                "mean_sigmas": MEAN_TOLERANCE_SIGMAS,
            },
            "results": [vars(c) for c in results],
            "anchor_only_sd_ratio_by_n": [
                {"n_samples": n, "mean_sd_ratio": r} for n, r in table
            ],
            "anchor_only_fails_every_seed": bool(anchor_failed),
            "randomize_then_optimize_passes_every_seed": bool(rto_passed),
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "elapsed_sec": round(time.time() - started, 2),
        }, handle, indent=1)
    print(f"\nwrote {os.path.relpath(out, repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
