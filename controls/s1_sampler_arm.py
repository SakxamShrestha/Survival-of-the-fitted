"""
S-1: the learner-rule arm, built stage by stage in the committed order of work.

The rule is gates/thresholds.json entry "S-1" (committed 2026-10-20, amended
2026-10-23 and 2026-10-27). This file implements it and changes nothing in it.
The draw step is analysis/ggn_sampler.py; every stage below calls that module's
map_pass / draw_pass, so the preconditions test the code the chain runs.

Subcommands, in the committed order. Each writes logs/s1_<stage>.{json,txt} and
refuses to overwrite an existing log.

  exactness   precondition_sampler_exactness: linear-Gaussian, N = 60, D = 8,
              noise sd 0.5, prior sd 1.3, 20000 draws, seeds 0, 1, 2, through
              the Gaussian branch of the SAME draw function.
  ggn         precondition_categorical_ggn_exactness: C-8's frozen body, pool
              seed 0, generation-1 hard-label batch, sigma 0.0361, lambda in
              {1, 10, 100}, 200 directions, 2000 draws.
  inertness   precondition_inertness: sampler draw-to-draw vs maximizer
              seed-to-seed functional disagreement on the probe pool
              (probe_seed 9000), n = 20 per side, non-overlapping 95% intervals.
  timing      one full 25-generation chain for the sampler and one for MC
              dropout, timed. Not a gate.
  chain       run named chains of either arm for the sweep. BUILT, NOT RUN by
              this file's author: the parent launches it. Its design arguments
              that the rule does not fix are REQUIRED flags with no default,
              so no interpretation is baked in silently.

    python3 controls/s1_sampler_arm.py exactness
    python3 controls/s1_sampler_arm.py ggn
    python3 controls/s1_sampler_arm.py inertness
    python3 controls/s1_sampler_arm.py timing --dropout-p P
"""

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spike_iterated_distill import MLP, make_pool, train  # noqa: E402
from analysis import ggn_sampler as gs  # noqa: E402
from analysis import anchored_sampler_check as asc  # noqa: E402
from analysis import laplace_glm as lg  # noqa: E402
from controls.c1_shuffled_n20 import BOOT_SEED, N_BOOT, bootstrap_ci_mean, intervals_overlap  # noqa: E402

LOGS = os.path.join(REPO, "logs")
DEVICE = "cpu"
TAU = 1.0
PROBE_SEED = 9_000
FOUNDER_STEPS = 3_000
MAXIMIZER_STEPS = 1_500
ESTABLISHED_FOUNDER_ACC = 0.977294921875

# Seed block 7000-7999, checked clean against every seed constant in the
# repository (0-19, 100-119, 200-219, 300-319, 400-404, 600-619, 1000-1019,
# 2000-3539, 9000).
SEED_EXACT_DRAWS = 7_010          # + seed
SEED_GGN_MAP_INIT = 7_100          # + lambda index
SEED_GGN_DRAWS = 7_110             # + lambda index
SEED_GGN_DIRECTIONS = 7_120
SEED_INERT_BATCH = 7_200
SEED_INERT_MAXIMIZER = 7_300       # + 2 i + {0, 1}
SEED_INERT_MAP_INIT = 7_400        # fixed-MAP reading
SEED_INERT_MAP_INITS = 7_410       # per-draw-MAP reading, + j
SEED_INERT_DRAWS = 7_450           # + cell index
SEED_TIMING_CHAIN = 7_600

N_INERT = 20
GGN_N_DRAWS = 2_000
GGN_CHUNK = 250
GGN_N_RANDOM = 100
GGN_N_EIGEN = 100
GGN_SIGMA = 0.0361
GGN_VAR_TOL = 0.10
GGN_MEAN_SE = 5.0


def load_rule():
    with open(os.path.join(REPO, "gates", "thresholds.json")) as handle:
        return json.load(handle)["S-1"]


def sigma_multipliers(rule):
    """sigma_grid is prose; read the two multipliers it names and refuse otherwise."""
    text = rule["fixed_in_advance"]["sigma_grid"]
    if "sigma_match" not in text or "4 x sigma_match" not in text:
        raise ValueError(f"sigma_grid no longer reads as committed: {text!r}")
    return (1.0, 4.0)


# ------------------------------------------------------------ logging

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def open_logs(stage):
    """Paths for logs/s1_<stage>.{json,txt}; refuses to overwrite either."""
    js = os.path.join(LOGS, f"s1_{stage}.json")
    tx = os.path.join(LOGS, f"s1_{stage}.txt")
    for path in (js, tx):
        if os.path.exists(path):
            raise SystemExit(f"refusing to overwrite {os.path.relpath(path, REPO)}")
    return js, tx


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump({"control": "S-1", **payload,
                   "torch_version": torch.__version__,
                   "python_version": sys.version.split()[0]}, handle, indent=1)
    print(f"wrote {os.path.relpath(path, REPO)}")


def run_logged(stage, fn, args):
    js, tx = open_logs(stage)
    with open(tx, "w") as handle:
        old = sys.stdout
        sys.stdout = Tee(old, handle)
        try:
            payload = fn(args)
            write_json(js, payload)
        finally:
            sys.stdout = old


def opt_record(opt):
    return {k: getattr(opt, k) for k in opt.__dataclass_fields__}


# ------------------------------------------------------------ shared pieces

def build_founder(pool_seed):
    """C-1's founder recipe: torch.manual_seed(s), make_pool(s, noise), MLP, 3000 steps."""
    torch.manual_seed(pool_seed)
    X, y, n_classes = make_pool(seed=pool_seed, kind="noise")
    founder = MLP(X.shape[1], n_classes).to(DEVICE)
    train(founder, X, y, FOUNDER_STEPS, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    with torch.no_grad():
        acc = float((founder(X).argmax(-1) == y).float().mean())
    return founder, X, y, n_classes, acc


def sampled_batch(teacher_logits_fn, X, bottleneck, generator):
    """sampler_arm.py's production step: subsample, then sampled hard labels at tau."""
    with torch.no_grad():
        t = teacher_logits_fn(X)
    sel = torch.randperm(X.shape[0], generator=generator)[:bottleneck]
    probs = F.softmax(t[sel].to(torch.float64) / TAU, -1)
    hard = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return sel, hard


def fresh_mlp_params(d_in, n_classes, seed, n=1):
    """n independent default-initialised MLPs as stacked float64 params.

    Uses torch.random.fork_rng so the caller's global RNG stream is untouched.
    """
    stacks = []
    with torch.random.fork_rng():
        for j in range(n):
            torch.manual_seed(seed + j)
            stacks.append(gs.params_from_module(MLP(d_in, n_classes)))
    return {k: torch.cat([s[k] for s in stacks]) for k in stacks[0]}


def argmax_disagreement(a, b):
    return float((a != b).double().mean())


def ci(values):
    rng = np.random.default_rng(BOOT_SEED)
    lo, hi = bootstrap_ci_mean(list(values), rng)
    return [float(lo), float(hi)]


# ============================================================ stage 1: exactness

EXACT_SEEDS = (0, 1, 2)
EXACT_LAMBDAS_SUPPLEMENTARY = (10.0, 100.0)


def stage_exactness(args):
    rule = load_rule()
    started = time.time()
    n, d, s, p, draws = (asc.DEFAULT_N, asc.DEFAULT_D, asc.DEFAULT_NOISE_SD,
                         asc.DEFAULT_PRIOR_SD, asc.DEFAULT_DRAWS)
    assert (n, d, s, p, draws) == (60, 8, 0.5, 1.3, 20000), "committed shape drifted"
    opt = gs.DEFAULT_OPT
    print("=== S-1 precondition_sampler_exactness, through analysis/ggn_sampler ===")
    print(f"linear-Gaussian N={n} D={d} noise sd {s} prior sd {p}, {draws} draws, seeds {EXACT_SEEDS}")
    print("path: GaussianLikelihood -> map_pass -> sample -> draw_pass (the chain's functions)")
    print(f"optimizer: {opt_record(opt)}")
    print("tolerances (analysis/anchored_sampler_check): mean <= 5 SE of exact sd, every sd ratio "
          "1 +/- 0.05, trace ratio 1 +/- 0.05\n")
    rows = []
    for seed in EXACT_SEEDS:
        rng = np.random.default_rng(seed)
        design = asc.make_design(n, d, rng)
        w_true = p * rng.standard_normal(d)
        targets = asc.simulate_targets(design, w_true, s, rng)
        X = torch.tensor(design, dtype=gs.DTYPE)
        lik = gs.GaussianLikelihood(targets, s)
        for lam in (1.0,) + EXACT_LAMBDAS_SUPPLEMENTARY:
            t0 = time.time()
            posterior = asc.exact_posterior(design, targets, s / math.sqrt(lam), p)
            mp = gs.map_pass(gs.linear_gaussian_forward,
                             {"w": torch.zeros(1, d, dtype=gs.DTYPE)}, X, lik, lam, {"w": p}, opt)
            gen = torch.Generator().manual_seed(SEED_EXACT_DRAWS + seed)
            dr, _, _ = gs.sample(gs.linear_gaussian_forward, mp, X, lam, {"w": p}, draws, gen, opt)
            W = dr.params["w"].numpy()
            check = asc.summarise_draws(f"ggn_gaussian_branch_lambda{lam:g}", W, posterior, seed)
            map_dev = float(np.max(np.abs(mp.params["w"][0].numpy() - posterior.mean)
                                   / posterior.marginal_sd))
            draw_norm = float(np.linalg.norm(W @ design.T, axis=1).mean())
            map_norm = float(np.linalg.norm(design @ mp.params["w"][0].numpy()))
            row = {**vars(check), "lambda": lam,
                   "verdict_bearing": lam == 1.0,
                   "map_minus_exact_mean_in_sd": map_dev,
                   "map_diag": mp.diagnostics, "draw_diag": dr.diagnostics,
                   "headroom": {"draw_output_norm_mean": draw_norm, "map_output_norm": map_norm,
                                "draw_marginal_sd_mean": float(W.std(0, ddof=1).mean()),
                                "map_to_map_spread": "0 by convexity (unique MAP); not sampled"},
                   "elapsed_sec": round(time.time() - t0, 2)}
            rows.append(row)
            tag = "VERDICT" if lam == 1.0 else "supplementary"
            print(f"seed {seed} lambda {lam:>5g} [{tag}] -> {'PASS' if check.passed else 'FAIL'}  "
                  f"mean {check.max_mean_dev_sigmas:.2f} SE  sd ratio {check.sd_ratio_min:.4f}"
                  f"-{check.sd_ratio_max:.4f}  trace {check.trace_ratio:.4f}  "
                  f"MAP-vs-exact {map_dev:.1e} sd  draw |grad| {dr.diagnostics['max_abs_grad']:.1e}"
                  f"  ({row['elapsed_sec']}s)")
            if check.reason:
                print(f"    reason: {check.reason}")
    verdict_rows = [r for r in rows if r["verdict_bearing"]]
    passed = all(r["passed"] for r in verdict_rows)
    supp_ok = all(r["passed"] for r in rows if not r["verdict_bearing"])
    verdict = "PASS" if passed else "BROKEN"
    print(f"\n=== precondition_sampler_exactness: {verdict} "
          f"({sum(r['passed'] for r in verdict_rows)} of {len(verdict_rows)} seeds) ===")
    print(f"supplementary tempered lambda 10, 100 (not in the committed shape, no verdict): "
          f"{'all pass' if supp_ok else 'NOT all pass'}")
    print("scope: certifies the Gaussian branch of the shared draw/MAP functions and their "
          "optimizer on a convex problem; not B^{1/2}, softmax curvature or the MLP.")
    return {"stage": "precondition_sampler_exactness", "verdict": verdict,
            "committed_shape": {"n": n, "d": d, "noise_sd": s, "prior_sd": p, "draws": draws,
                                "seeds": list(EXACT_SEEDS), "lambda": 1.0},
            "tolerances": {"mean_se": asc.MEAN_TOLERANCE_SIGMAS, "sd_ratio": asc.SD_RATIO_TOLERANCE,
                           "trace_ratio": asc.TRACE_RATIO_TOLERANCE},
            "same_function_path": "analysis/ggn_sampler.map_pass + sample/draw_pass with "
                                  "GaussianLikelihood and linear_gaussian_forward",
            "optimizer": opt_record(opt), "draw_seed_base": SEED_EXACT_DRAWS,
            "supplementary_tempered_all_pass": bool(supp_ok),
            "rule_committed": rule["committed"], "results": rows,
            "elapsed_sec": round(time.time() - started, 1)}


# ============================================================ stage 2: GGN head

def head_glm_layout(params):
    """(S, K, G) weight and (S, K) bias -> (S, G + 1, K), matching augment_features."""
    return torch.cat([params["head.weight"].transpose(1, 2), params["head.bias"][:, None, :]], 1)


def ggn_directions(phi, mean, alpha_eff, n_classes, rng):
    """100 random unit vectors and the 100 leading Hessian eigenvectors, (200, G+1, K).

    SciPy cannot be imported in this environment (built against NumPy 1.x, NumPy
    2 installed), so eigsh is replaced by an exact route. The fit's Hessian is
    alpha_eff I + M^T M with M the (N K) x (G+1)K matrix whose (i, k) row is
    phi_i (x) R_i[k, :], R_i = B_i^{1/2} at the fit's MAP. Its leading
    eigenvectors are M^T u / sqrt(s) for the leading eigenpairs (s, u) of the
    Gram matrix M M^T, whose entries are (phi_i . phi_j) (R_i R_j^T)[k, l].
    Each returned eigenvector is then checked against laplace_glm's own
    Hessian-vector product (the residual is logged).
    """
    P = lg._softmax_rows(phi @ mean)
    _, R = gs.categorical_curvature(torch.tensor(P, dtype=gs.DTYPE))
    Phi = torch.tensor(phi, dtype=gs.DTYPE)
    n = Phi.shape[0]
    gram = (Phi @ Phi.T)[:, None, :, None] * torch.einsum("ikm,jlm->ikjl", R, R)
    gram = gram.reshape(n * n_classes, n * n_classes)
    s, U = torch.linalg.eigh(0.5 * (gram + gram.T))
    s, U = s.flip(0)[:GGN_N_EIGEN], U.flip(1)[:, :GGN_N_EIGEN]
    V = torch.einsum("id,ikm,ikj->jdm", Phi, R, U.reshape(n, n_classes, -1)) / s.sqrt()[:, None, None]
    V = V / V.reshape(V.shape[0], -1).norm(dim=1)[:, None, None]
    evals = (s + alpha_eff).numpy()
    evecs = V.numpy()
    resid = max(float(np.linalg.norm(lg.hessian_vector_product(phi, mean, alpha_eff, v) - mu * v))
                / mu for v, mu in zip(evecs, evals))
    rand = rng.standard_normal((GGN_N_RANDOM,) + phi.shape[1:] + (n_classes,))
    rand /= np.linalg.norm(rand.reshape(GGN_N_RANDOM, -1), axis=1)[:, None, None]
    dirs = np.concatenate([rand, evecs])
    return dirs, evals, resid


def exact_directional_variance(phi, mean, alpha_eff, dirs, lam, tol=1e-13, max_iter=5000):
    """u^T H^-1 u / lam, H^-1 u by conjugate gradients on laplace_glm's HVP."""
    out = []
    for u in dirs:
        x = np.zeros_like(u)
        r = u.copy()
        d = r.copy()
        rs = float((r * r).sum())
        for _ in range(max_iter):
            hd = lg.hessian_vector_product(phi, mean, alpha_eff, d)
            step = rs / float((d * hd).sum())
            x = x + step * d
            r = r - step * hd
            rs_new = float((r * r).sum())
            if math.sqrt(rs_new) < tol:
                break
            d = r + (rs_new / rs) * d
            rs = rs_new
        else:
            raise RuntimeError("CG did not converge")
        out.append(float((u * x).sum()) / lam)
    return np.asarray(out)


def stage_ggn(args):
    from controls.c8_frozen_body import (CHAIN_SEED_OFFSET, POOL_SEEDS, augment_features,
                                         build_frozen_founder, run_frozen_chain)
    rule = load_rule()
    started = time.time()
    lambdas = [float(v) for v in rule["fixed_in_advance"]["lambda_grid"]]
    opt = gs.DEFAULT_OPT
    print("=== S-1 precondition_categorical_ggn_exactness ===")
    pool_seed = POOL_SEEDS[0]
    model, X, _, n_classes, founder_acc, _ = build_frozen_founder(pool_seed)
    _, _, captured, _, _ = run_frozen_chain(model, X, n_classes, CHAIN_SEED_OFFSET + pool_seed,
                                            1, soft=False)
    sel, Xs, targets = captured[1]
    with torch.no_grad():
        feats = model.body(Xs).to(gs.DTYPE)
        Xp, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
        feats_probe = model.body(Xp).to(gs.DTYPE)
    phi = augment_features(feats.numpy())
    labels = targets.cpu().numpy()
    n_batch_classes = int(np.unique(labels).size)
    print(f"C-8 frozen body, pool seed {pool_seed}, chain seed {CHAIN_SEED_OFFSET + pool_seed}: "
          f"frozen founder acc {founder_acc:.6f}; generation-1 hard-label batch of "
          f"{len(labels)} with {n_batch_classes} of {n_classes} classes")
    alpha = 1.0 / GGN_SIGMA ** 2
    sds = {"head.weight": GGN_SIGMA, "head.bias": GGN_SIGMA}
    print(f"sigma {GGN_SIGMA} on weight and bias -> alpha {alpha:.3f}; "
          f"target N(fit_laplace MAP at alpha/lambda, H^-1/lambda)")
    print(f"tolerance: every directional variance ratio 1 +/- {GGN_VAR_TOL}, mean within "
          f"{GGN_MEAN_SE} SE, {GGN_N_DRAWS} draws, {GGN_N_RANDOM} random + {GGN_N_EIGEN} "
          f"leading-Hessian directions\noptimizer: {opt_record(opt)}\n")
    lik = gs.CategoricalLikelihood(targets, n_classes)
    rng_dirs = np.random.default_rng(SEED_GGN_DIRECTIONS)
    per_lambda = []
    for li, lam in enumerate(lambdas):
        t0 = time.time()
        post = lg.fit_laplace(phi, labels, n_classes, prior_precision=alpha / lam)
        if post.reason is not None:
            raise SystemExit(f"fit_laplace refused: {post.reason}")
        mean = np.asarray(post.mean)
        dirs, evals, eig_resid = ggn_directions(phi, mean, alpha / lam, n_classes, rng_dirs)
        var_exact = exact_directional_variance(phi, mean, alpha / lam, dirs, lam)
        eig_check = float(np.max(np.abs(var_exact[GGN_N_RANDOM:] * lam * evals - 1.0)))
        # the sampler: MAP pass then draws, restricted to the head
        init_mod = nn.Linear(feats.shape[1], n_classes)
        with torch.random.fork_rng():
            torch.manual_seed(SEED_GGN_MAP_INIT + li)
            init_mod.reset_parameters()
        init = {"head.weight": init_mod.weight.detach().to(gs.DTYPE)[None],
                "head.bias": init_mod.bias.detach().to(gs.DTYPE)[None]}
        mp = gs.map_pass(gs.linear_head_forward, init, feats, lik, lam, sds, opt)
        w_map = head_glm_layout(mp.params)[0].numpy()
        map_vs_fit = float(np.abs(w_map - mean).max())
        gen = torch.Generator().manual_seed(SEED_GGN_DRAWS + li)
        projections, draw_diags, norms, probe_preds = [], [], [], []
        dirs_t = torch.tensor(dirs, dtype=gs.DTYPE)
        for start in range(0, GGN_N_DRAWS, GGN_CHUNK):
            dr, _, _ = gs.sample(gs.linear_head_forward, mp, feats, lam, sds, GGN_CHUNK, gen, opt)
            Wg = head_glm_layout(dr.params)
            projections.append(torch.einsum("sdk,jdk->sj", Wg, dirs_t).numpy())
            draw_diags.append(dr.diagnostics)
            with torch.no_grad():
                fb = gs.linear_head_forward(dr.params, feats)
                norms.append(fb.norm(dim=-1).mean(-1).numpy())
                probe_preds.append(gs.linear_head_forward(dr.params, feats_probe).argmax(-1))
            print(f"  lambda {lam:g}: draws {start + GGN_CHUNK}/{GGN_N_DRAWS}  draw |grad| "
                  f"{dr.diagnostics['max_abs_grad']:.1e}  [{time.time() - t0:.0f}s]", flush=True)
        proj = np.concatenate(projections)
        exact_mean = np.einsum("dk,jdk->j", mean, dirs)
        ratios = proj.var(0, ddof=1) / var_exact
        mean_se = np.abs(proj.mean(0) - exact_mean) / np.sqrt(var_exact / proj.shape[0])
        ratio_ok = bool(np.all(np.abs(ratios - 1.0) <= GGN_VAR_TOL))
        mean_ok = bool(np.all(mean_se <= GGN_MEAN_SE))
        preds = torch.cat(probe_preds)
        pair_dis = [argmax_disagreement(preds[2 * i], preds[2 * i + 1]) for i in range(min(100, preds.shape[0] // 2))]
        with torch.no_grad():
            map_norm = float(mp.f_hat.norm(dim=-1).mean())
            map_probe = gs.linear_head_forward(mp.params, feats_probe)[0].argmax(-1)
        draw_vs_map = float(np.mean([argmax_disagreement(p, map_probe) for p in preds[:200]]))
        res = {
            "lambda": lam, "passed": ratio_ok and mean_ok,
            "var_ratio_ok": ratio_ok, "mean_ok": mean_ok,
            "var_ratio_random": [float(ratios[:GGN_N_RANDOM].min()), float(ratios[:GGN_N_RANDOM].max())],
            "var_ratio_eigen": [float(ratios[GGN_N_RANDOM:].min()), float(ratios[GGN_N_RANDOM:].max())],
            "mean_se_max_random": float(mean_se[:GGN_N_RANDOM].max()),
            "mean_se_max_eigen": float(mean_se[GGN_N_RANDOM:].max()),
            "n_ratio_outside": int(np.sum(np.abs(ratios - 1.0) > GGN_VAR_TOL)),
            "n_mean_outside": int(np.sum(mean_se > GGN_MEAN_SE)),
            "hessian_eigen_range": [float(evals.min()), float(evals.max())],
            "eigvec_cg_vs_eigenvalue_max_rel": eig_check,
            "eigvec_hvp_residual_max_rel": eig_resid,
            "fit_laplace": {"converged": bool(post.converged), "n_iter": int(post.n_iter),
                            "data_rank": int(post.data_rank),
                            "prior_only_directions": int(post.prior_only_directions)},
            "sampler_map_vs_fit_laplace_max_abs": map_vs_fit,
            "map_diag": mp.diagnostics,
            "draw_max_abs_grad_worst": float(max(d["max_abs_grad"] for d in draw_diags)),
            "headroom": {
                "map_to_map_spread": "0 by strict convexity of the GLM objective; sampler MAP "
                                     "matches fit_laplace to sampler_map_vs_fit_laplace_max_abs",
                "draw_to_draw_probe_disagreement_mean": float(np.mean(pair_dis)),
                "draw_vs_map_probe_disagreement_mean": draw_vs_map,
                "draw_batch_logit_norm_mean": float(np.mean(np.concatenate(norms))),
                "map_batch_logit_norm": map_norm,
                "map_mean_max_prob": float(F.softmax(mp.f_hat, -1).max(-1).values.mean()),
            },
            "elapsed_sec": round(time.time() - t0, 1),
        }
        per_lambda.append(res)
        print(f"lambda {lam:>5g} -> {'PASS' if res['passed'] else 'FAIL'}  var ratio random "
              f"{res['var_ratio_random'][0]:.4f}-{res['var_ratio_random'][1]:.4f}  eigen "
              f"{res['var_ratio_eigen'][0]:.4f}-{res['var_ratio_eigen'][1]:.4f}  mean max "
              f"{max(res['mean_se_max_random'], res['mean_se_max_eigen']):.2f} SE  "
              f"MAP-vs-fit {map_vs_fit:.1e}  ({res['elapsed_sec']}s)")
        print(f"    headroom: draw-to-draw probe disagreement {res['headroom']['draw_to_draw_probe_disagreement_mean']:.4f}, "
              f"draw-vs-MAP {draw_vs_map:.4f}, logit norm draw {res['headroom']['draw_batch_logit_norm_mean']:.4f} "
              f"vs MAP {map_norm:.4f}, MAP mean max-prob {res['headroom']['map_mean_max_prob']:.4f}")
    verdict = "PASS" if all(r["passed"] for r in per_lambda) else "BROKEN"
    print(f"\n=== precondition_categorical_ggn_exactness: {verdict} ===")
    return {"stage": "precondition_categorical_ggn_exactness", "verdict": verdict,
            "pool_seed": int(pool_seed), "chain_seed": int(CHAIN_SEED_OFFSET + pool_seed),
            "frozen_founder_acc": founder_acc, "batch_size": int(len(labels)),
            "batch_classes_present": n_batch_classes, "sigma": GGN_SIGMA, "alpha": alpha,
            "lambda_grid": lambdas, "n_draws": GGN_N_DRAWS, "chunk": GGN_CHUNK,
            "n_random": GGN_N_RANDOM, "n_eigen": GGN_N_EIGEN,
            "var_tol": GGN_VAR_TOL, "mean_se_tol": GGN_MEAN_SE,
            "seeds": {"map_init": SEED_GGN_MAP_INIT, "draws": SEED_GGN_DRAWS,
                      "directions": SEED_GGN_DIRECTIONS},
            "optimizer": opt_record(opt), "results": per_lambda,
            "elapsed_sec": round(time.time() - started, 1)}


# ============================================================ stage 3: inertness

def maximizer_student(Xs, hard, n_classes, seed):
    """sampler_arm.py's learner, the lambda -> infinity reference: train() unchanged."""
    torch.manual_seed(seed)
    s = MLP(Xs.shape[1], n_classes).to(DEVICE)
    train(s, Xs, hard, MAXIMIZER_STEPS, 256, 1e-3, 0.0, False, 1.0, DEVICE)
    return s


def stage_inertness(args):
    rule = load_rule()
    started = time.time()
    lambdas = [float(v) for v in rule["fixed_in_advance"]["lambda_grid"]]
    mults = sigma_multipliers(rule)
    match = rule["fixed_in_advance"]["sigma_match_measured"]
    B = int(rule["fixed_in_advance"]["bottleneck"])
    opt = gs.DEFAULT_OPT
    print("=== S-1 precondition_inertness ===")
    founder, X, _, n_classes, acc = build_founder(0)
    print(f"founder: C-1 recipe, pool seed 0, memorization {acc!r} "
          f"(established {ESTABLISHED_FOUNDER_ACC!r}, |diff| {abs(acc - ESTABLISHED_FOUNDER_ACC):.2e})")
    gen = torch.Generator().manual_seed(SEED_INERT_BATCH)
    sel, hard = sampled_batch(founder, X, B, gen)
    Xs, Xs64 = X[sel], X[sel].to(gs.DTYPE)
    Xp, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    Xp64 = Xp.to(gs.DTYPE)
    print(f"transmitted batch: B = {B}, sampled hard labels at tau {TAU}, "
          f"{int(hard.unique().numel())} classes; probe pool seed {PROBE_SEED}\n")

    # maximizer side, measured first
    t0 = time.time()
    max_dis = []
    for i in range(N_INERT):
        preds = []
        for which in (0, 1):
            s = maximizer_student(Xs, hard, n_classes, SEED_INERT_MAXIMIZER + 2 * i + which)
            with torch.no_grad():
                preds.append(s(Xp).argmax(-1))
        max_dis.append(argmax_disagreement(preds[0], preds[1]))
    max_ci = ci(max_dis)
    print(f"maximizer seed-to-seed (train(), {MAXIMIZER_STEPS} steps, wd 0): mean "
          f"{np.mean(max_dis):.4f}, 95% [{max_ci[0]:.4f}, {max_ci[1]:.4f}], n = {N_INERT} "
          f"pairs  ({time.time() - t0:.0f}s)\n")

    lik = gs.CategoricalLikelihood(hard, n_classes)
    cells = []
    for mi, mult in enumerate(mults):
        for li, lam in enumerate(lambdas):
            ci_idx = mi * len(lambdas) + li
            t0 = time.time()
            init = fresh_mlp_params(X.shape[1], n_classes, SEED_INERT_MAP_INIT)
            sds = gs.layer_prior_sds(list(init), match, mult)
            # reading (i): fixed MAP, 2n draws differing only in anchor and xi
            mp = gs.map_pass(gs.mlp_forward, init, Xs64, lik, lam, sds, opt)
            g = torch.Generator().manual_seed(SEED_INERT_DRAWS + ci_idx)
            dr, _, _ = gs.sample(gs.mlp_forward, mp, Xs64, lam, sds, 2 * N_INERT, g, opt)
            with torch.no_grad():
                fp = gs.mlp_forward(dr.params, Xp64)
                map_p = gs.mlp_forward(mp.params, Xp64)[0]
            pr = fp.argmax(-1)
            fixed = [argmax_disagreement(pr[2 * i], pr[2 * i + 1]) for i in range(N_INERT)]
            # reading (ii): each draw from its own MAP (MAP init seed + anchor + xi)
            inits = fresh_mlp_params(X.shape[1], n_classes, SEED_INERT_MAP_INITS, n=2 * N_INERT)
            mps = gs.map_pass(gs.mlp_forward, inits, Xs64, lik, lam, sds, opt)
            g2 = torch.Generator().manual_seed(SEED_INERT_DRAWS + 100 + ci_idx)
            shapes = {k: tuple(v.shape[1:]) for k, v in mps.params.items()}
            th, xi = gs.draw_noise(sds, shapes, 2 * N_INERT, B, n_classes, g2)
            dr2 = gs.draw_pass(gs.mlp_forward, Xs64, mps.expansion, lam, sds, th, xi, opt)
            with torch.no_grad():
                pr2 = gs.mlp_forward(dr2.params, Xp64).argmax(-1)
                mapsp = gs.mlp_forward(mps.params, Xp64)
            full = [argmax_disagreement(pr2[2 * i], pr2[2 * i + 1]) for i in range(N_INERT)]
            map_map = [argmax_disagreement(mapsp[2 * i].argmax(-1), mapsp[2 * i + 1].argmax(-1))
                       for i in range(N_INERT)]
            ci_fixed, ci_full, ci_mm = ci(fixed), ci(full), ci(map_map)
            pass_fixed = ci_fixed[0] > max_ci[1]
            pass_full = ci_full[0] > max_ci[1]
            cell = {
                "lambda": lam, "sigma_multiplier": mult, "prior_sds": sds,
                "prior_precision_per_tensor": {k: 1.0 / v ** 2 for k, v in sds.items()},
                "fixed_map": {"values": fixed, "mean": float(np.mean(fixed)), "ci95": ci_fixed,
                              "exceeds_maximizer_non_overlapping": bool(pass_fixed)},
                "per_draw_map": {"values": full, "mean": float(np.mean(full)), "ci95": ci_full,
                                 "exceeds_maximizer_non_overlapping": bool(pass_full)},
                "headroom": {
                    "map_to_map_probe_disagreement": {"values": map_map, "mean": float(np.mean(map_map)),
                                                      "ci95": ci_mm},
                    "draw_to_draw_fixed_map_mean": float(np.mean(fixed)),
                    "draw_vs_map_probe_disagreement_mean": float(np.mean(
                        [argmax_disagreement(p, map_p.argmax(-1)) for p in pr])),
                    "draw_probe_logit_norm_mean": float(fp.norm(dim=-1).mean()),
                    "map_probe_logit_norm": float(map_p.norm(dim=-1).mean()),
                    "map_mean_max_prob_batch": float(F.softmax(mp.f_hat, -1).max(-1).values.mean()),
                    "map_probe_alphabet": int(map_p.argmax(-1).unique().numel()),
                    "draw_probe_alphabet_mean": float(np.mean([int(p.unique().numel()) for p in pr])),
                },
                "map_diag": mp.diagnostics, "draw_diag": dr.diagnostics,
                "per_draw_map_diag": mps.diagnostics, "per_draw_draw_diag": dr2.diagnostics,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            cells.append(cell)
            h = cell["headroom"]
            print(f"lambda {lam:>5g} sigma x{mult:g}: fixed-MAP draw-to-draw {np.mean(fixed):.4f} "
                  f"[{ci_fixed[0]:.4f}, {ci_fixed[1]:.4f}] -> {'EXCEEDS' if pass_fixed else 'does NOT exceed'};"
                  f" per-draw-MAP {np.mean(full):.4f} [{ci_full[0]:.4f}, {ci_full[1]:.4f}] -> "
                  f"{'EXCEEDS' if pass_full else 'does NOT exceed'}  ({cell['elapsed_sec']}s)", flush=True)
            print(f"    headroom: MAP-to-MAP {h['map_to_map_probe_disagreement']['mean']:.4f}, "
                  f"draw-vs-MAP {h['draw_vs_map_probe_disagreement_mean']:.4f}, probe logit norm draw "
                  f"{h['draw_probe_logit_norm_mean']:.3f} vs MAP {h['map_probe_logit_norm']:.3f}, MAP "
                  f"max-prob {h['map_mean_max_prob_batch']:.3f}, probe alphabet MAP {h['map_probe_alphabet']} "
                  f"draws {h['draw_probe_alphabet_mean']:.1f}; |grad| MAP "
                  f"{mp.diagnostics['max_abs_grad']:.1e} draw {dr.diagnostics['max_abs_grad']:.1e}",
                  flush=True)
    all_fixed = all(c["fixed_map"]["exceeds_maximizer_non_overlapping"] for c in cells)
    all_full = all(c["per_draw_map"]["exceeds_maximizer_non_overlapping"] for c in cells)
    verdict = ("PASS" if all_fixed and all_full else
               "BROKEN" if not all_fixed and not all_full else "READING-DEPENDENT")
    print(f"\n=== precondition_inertness: {verdict} "
          f"(fixed-MAP reading: {sum(c['fixed_map']['exceeds_maximizer_non_overlapping'] for c in cells)}"
          f"/{len(cells)} cells; per-draw-MAP reading: "
          f"{sum(c['per_draw_map']['exceeds_maximizer_non_overlapping'] for c in cells)}/{len(cells)}) ===")
    print("Reminder from the rule: MC dropout is expected to pass this check too; a pass is not "
          "evidence that the sampler is genuine.")
    return {"stage": "precondition_inertness", "verdict": verdict,
            "reading_note": ("The rule does not say whether 'the sampler's randomness' includes the "
                             "MAP pass's initialisation, nor whether the check is per (lambda, sigma) "
                             "cell. Both readings are run on every cell and reported; the verdict is "
                             "PASS only if every cell passes under both."),
            "founder_acc": acc, "founder_acc_established": ESTABLISHED_FOUNDER_ACC,
            "bottleneck": B, "tau": TAU, "probe_seed": PROBE_SEED, "n_per_side": N_INERT,
            "batch_classes": int(hard.unique().numel()),
            "maximizer": {"values": max_dis, "mean": float(np.mean(max_dis)), "ci95": max_ci,
                          "recipe": "train(MLP, Xs, hard, 1500, 256, 1e-3, wd=0.0), as sampler_arm.py"},
            "cells": cells, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "seeds": {"batch": SEED_INERT_BATCH, "maximizer": SEED_INERT_MAXIMIZER,
                      "map_init": SEED_INERT_MAP_INIT, "map_inits": SEED_INERT_MAP_INITS,
                      "draws": SEED_INERT_DRAWS},
            "optimizer": opt_record(opt), "elapsed_sec": round(time.time() - started, 1)}


# ============================================================ chains (sweep + negative control)

class Teacher:
    """A float64 functional MLP, optionally with a fixed per-unit dropout mask."""

    def __init__(self, params, masks=None):
        self.params, self.masks = params, masks

    def __call__(self, X):
        with torch.no_grad():
            return gs.mlp_forward(self.params, X.to(gs.DTYPE), self.masks)[0]


def sampler_student(Xs64, hard, n_classes, lam, sds, init_seed, generator, opt):
    lik = gs.CategoricalLikelihood(hard, n_classes)
    init = fresh_mlp_params(Xs64.shape[1], n_classes, init_seed)
    mp = gs.map_pass(gs.mlp_forward, init, Xs64, lik, lam, sds, opt)
    dr, _, _ = gs.sample(gs.mlp_forward, mp, Xs64, lam, sds, 1, generator, opt)
    with torch.no_grad():
        f_draw = gs.mlp_forward(dr.params, Xs64)[0]
    diag = {"map_loss": mp.diagnostics["final_loss"], "map_grad": mp.diagnostics["max_abs_grad"],
            "draw_loss": dr.diagnostics["final_loss"], "draw_grad": dr.diagnostics["max_abs_grad"],
            "map_logit_norm": float(mp.f_hat.norm(dim=-1).mean()),
            "draw_logit_norm": float(f_draw.norm(dim=-1).mean()),
            "draw_vs_map_batch_argmax_agree": float((f_draw.argmax(-1) == mp.f_hat.argmax(-1)).double().mean()),
            "map_mean_max_prob": float(F.softmax(mp.f_hat, -1).max(-1).values.mean())}
    return Teacher(dr.params), diag


def dropout_student(Xs64, hard, n_classes, lam, sds, init_seed, generator, opt, p_drop):
    """MC-dropout negative control: the same lambda/sigma objective trained with
    dropout, then ONE fixed per-unit Bernoulli mask frozen into the student.

    Interpretation, not fixed by the rule (see the report): dropout rate p_drop,
    per-example masks during training, inverted scaling, cosine Adam for
    opt.adam_steps steps with no L-BFGS polish (the dropout objective is
    stochastic), and a per-unit mask shared by all inputs as the 'draw'.
    """
    params = {k: v.detach().clone().requires_grad_(True)
              for k, v in fresh_mlp_params(Xs64.shape[1], n_classes, init_seed).items()}
    adam = torch.optim.Adam(params.values(), lr=opt.adam_lr, weight_decay=0.0)
    keep = 1.0 - p_drop
    h = params["body.0.bias"].shape[-1]
    n = Xs64.shape[0]
    for step in range(opt.adam_steps):
        for group in adam.param_groups:
            group["lr"] = opt.adam_lr * 0.5 * (1.0 + math.cos(math.pi * step / opt.adam_steps))
        masks = {k: (torch.rand(1, n, h, generator=generator, dtype=gs.DTYPE) < keep).to(gs.DTYPE) / keep
                 for k in ("h1", "h2")}
        f = gs.mlp_forward(params, Xs64, masks)
        loss = lam * F.cross_entropy(f[0], hard, reduction="sum") + gs.prior_term(params, sds)
        adam.zero_grad(set_to_none=True)
        loss.backward()
        adam.step()
    fixed = {k: (torch.rand(1, h, generator=generator, dtype=gs.DTYPE) < keep).to(gs.DTYPE) / keep
             for k in ("h1", "h2")}
    out = {k: v.detach().clone() for k, v in params.items()}
    with torch.no_grad():
        f_mean = gs.mlp_forward(out, Xs64)[0]
        f_draw = gs.mlp_forward(out, Xs64, fixed)[0]
    diag = {"train_loss_last": float(loss.detach()), "p_drop": p_drop,
            "map_logit_norm": float(f_mean.norm(dim=-1).mean()),
            "draw_logit_norm": float(f_draw.norm(dim=-1).mean()),
            "draw_vs_map_batch_argmax_agree": float((f_draw.argmax(-1) == f_mean.argmax(-1)).double().mean())}
    return Teacher(out, fixed), diag


def run_chain(arm, founder, X, n_classes, lam, mult, chain_seed, gens, bottleneck, match,
              opt=None, p_drop=None, Xp=None):
    """One chain: production step (sampled hard labels) then the arm's learning step."""
    opt = gs.DEFAULT_OPT if opt is None else opt
    names = [n for n, _ in founder.named_parameters()]
    sds = gs.layer_prior_sds(names, match, mult)
    gen = torch.Generator().manual_seed(chain_seed)          # production stream
    gdraw = torch.Generator().manual_seed(chain_seed + 500_000)  # learner stream
    teacher = founder
    sizes, probe_sizes, diags, gen_times = [], [], [], []
    X64 = X.to(gs.DTYPE)
    for g in range(gens):
        t0 = time.time()
        sel, hard = sampled_batch(teacher, X, bottleneck, gen)
        init_seed = chain_seed * 1000 + g
        if arm == "sampler":
            teacher, diag = sampler_student(X64[sel], hard, n_classes, lam, sds, init_seed, gdraw, opt)
        elif arm == "dropout":
            teacher, diag = dropout_student(X64[sel], hard, n_classes, lam, sds, init_seed, gdraw,
                                            opt, p_drop)
        else:
            raise ValueError(arm)
        pred = teacher(X).argmax(-1)
        sizes.append(int(pred.unique().numel()))
        if Xp is not None:
            probe_sizes.append(int(teacher(Xp).argmax(-1).unique().numel()))
        diags.append({**diag, "batch_classes": int(hard.unique().numel())})
        gen_times.append(round(time.time() - t0, 2))
        print(f"    gen {g + 1:2d}  alphabet {sizes[-1]:2d}  ({gen_times[-1]:.1f}s)", flush=True)
    survivors = sorted(int(c) for c in teacher(X).argmax(-1).unique().tolist())
    # The gates and headroom floor 1 are read on the probe pool
    # (design_amendment_2026_11_08.pool_for_gates_and_floor_1), so the probe pool's
    # survivor SET is recorded and not only its size.
    probe_survivors = (None if Xp is None else
                       sorted(int(c) for c in teacher(Xp).argmax(-1).unique().tolist()))
    return {"arm": arm, "lambda": lam, "sigma_multiplier": mult, "chain_seed": chain_seed,
            "sizes": sizes, "probe_sizes": probe_sizes, "survivors": survivors,
            "probe_survivors": probe_survivors,
            "final": sizes[-1], "diagnostics": diags, "gen_times": gen_times,
            "prior_sds": sds, "p_drop": p_drop}


def stage_timing(args):
    rule = load_rule()
    started = time.time()
    fixed = rule["fixed_in_advance"]
    gens, B = int(fixed["gens"]), int(fixed["bottleneck"])
    match = fixed["sigma_match_measured"]
    lam, mult = float(args.lam), float(args.sigma_mult)
    founder, X, _, n_classes, acc = build_founder(0)
    Xp, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    print(f"=== S-1 timing: one {gens}-generation chain per arm, lambda {lam:g}, sigma x{mult:g} ===")
    print(f"founder pool seed 0 acc {acc!r}; torch threads {torch.get_num_threads()}")
    print("NOT a gate and NOT a sweep cell: one chain per arm, for the wall-time projection only.\n")
    out = {}
    for arm in ("sampler", "dropout"):
        t0 = time.time()
        print(f"  {arm}:")
        res = run_chain(arm, founder, X, n_classes, lam, mult, SEED_TIMING_CHAIN, gens, B, match,
                        p_drop=args.dropout_p if arm == "dropout" else None, Xp=Xp)
        res["wall_sec"] = round(time.time() - t0, 1)
        out[arm] = res
        print(f"  {arm}: {res['wall_sec']} s for {gens} generations; final alphabet {res['final']}\n")
    return {"stage": "timing", "lambda": lam, "sigma_multiplier": mult, "gens": gens,
            "bottleneck": B, "founder_acc": acc, "chain_seed": SEED_TIMING_CHAIN,
            "dropout_p_used_for_timing_only": args.dropout_p,
            "torch_threads": torch.get_num_threads(), "chains": out,
            "optimizer": opt_record(gs.DEFAULT_OPT), "elapsed_sec": round(time.time() - started, 1)}


def stage_chain(args):
    """Sweep/negative-control worker. Every design choice the rule leaves open is a
    required flag; this function chooses none of them."""
    rule = load_rule()
    fixed = rule["fixed_in_advance"]
    gens, B = int(fixed["gens"]), int(fixed["bottleneck"])
    if float(args.lam) not in [float(v) for v in fixed["lambda_grid"]]:
        raise SystemExit(f"lambda {args.lam} is not on the committed grid {fixed['lambda_grid']}")
    if float(args.sigma_mult) not in sigma_multipliers(rule):
        raise SystemExit(f"sigma multiplier {args.sigma_mult} is not on the committed grid")
    if args.arm == "dropout" and args.dropout_p is None:
        raise SystemExit("--dropout-p is required for the dropout arm; the rule does not fix it")
    founder, X, _, n_classes, acc = build_founder(args.pool_seed)
    Xp, _, _ = make_pool(seed=PROBE_SEED, kind="noise")
    flag = abs(acc - ESTABLISHED_FOUNDER_ACC) > 0.01
    print(f"founder pool seed {args.pool_seed}: acc {acc!r}"
          f"{'  REPORTED: departs from 0.977294921875 by more than 0.01' if flag else ''}")
    chains = []
    for cs in args.chain_seeds:
        print(f"  chain seed {cs}")
        chains.append(run_chain(args.arm, founder, X, n_classes, float(args.lam),
                                float(args.sigma_mult), cs, gens, B,
                                fixed["sigma_match_measured"], p_drop=args.dropout_p, Xp=Xp))
    return {"stage": "chain", "arm": args.arm, "lambda": float(args.lam),
            "sigma_multiplier": float(args.sigma_mult), "pool_seed": args.pool_seed,
            "founder_acc": acc, "founder_acc_departs_by_more_than_0.01": bool(flag),
            "gens": gens, "bottleneck": B, "chains": chains,
            "optimizer": opt_record(gs.DEFAULT_OPT)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("exactness")
    sub.add_parser("ggn")
    sub.add_parser("inertness")
    t = sub.add_parser("timing")
    t.add_argument("--dropout-p", type=float, required=True)
    t.add_argument("--lam", type=float, default=100.0)
    t.add_argument("--sigma-mult", type=float, default=1.0)
    c = sub.add_parser("chain")
    c.add_argument("--arm", choices=["sampler", "dropout"], required=True)
    c.add_argument("--lam", type=float, required=True)
    c.add_argument("--sigma-mult", type=float, required=True)
    c.add_argument("--pool-seed", type=int, required=True)
    c.add_argument("--chain-seeds", type=int, nargs="+", required=True)
    c.add_argument("--dropout-p", type=float, default=None)
    c.add_argument("--tag", type=str, required=True,
                   help="log name suffix: writes logs/s1_chain_<tag>.{json,txt}")
    args = ap.parse_args(argv)
    if args.stage == "chain":
        run_logged(f"chain_{args.tag}", stage_chain, args)
    else:
        fn = {"exactness": stage_exactness, "ggn": stage_ggn,
              "inertness": stage_inertness, "timing": stage_timing}[args.stage]
        stage = {"exactness": "precondition_exactness", "ggn": "precondition_ggn",
                 "inertness": "precondition_inertness", "timing": "timing"}[args.stage]
        run_logged(stage, fn, args)


if __name__ == "__main__":
    main()
