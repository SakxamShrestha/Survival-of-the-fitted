"""Figures for the iterated-distillation chains.

Reads the JSONL logs written by ``spike_iterated_distill.py --log`` and renders
the figure set used in the written report. Every figure is regenerated from a
saved run rather than from a one-off script, so the report can be rebuilt from
``logs/`` alone.

    python3 analysis/plot_chain.py

Writes PNGs to ``figs/`` at 200 dpi.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs")
FIGS = os.path.join(REPO, "figs")

N_CLASSES = 50
CHANCE = 1.0 / N_CLASSES

# Categorical hues, assigned in this fixed order and never cycled. Validated for
# colour-vision deficiency separation before use.
CAT = ["#2547C7", "#C1440E", "#1E7A4F", "#7B2D8E"]
# Temperature is an ordered variable, so it gets a sequential ramp rather than
# categorical hues: light means low tau.
SEQ = {"0.5": "#A8BCEB", "1.0": "#5478D4", "2.0": "#16306F"}

INK = "#1C1A17"
MUTED = "#6B6459"
GRID = "#DDD8CE"


def load(name):
    path = os.path.join(LOGS, name)
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def col(rows, key):
    return [r["gen"] for r in rows], [r[key] for r in rows]


# ---------------------------------------------------------------- C-6 probe data
#
# `load` above reads the per-generation JSONL written by
# `spike_iterated_distill.py --log`, whose metrics are computed on the pool the
# chain trains on. C-6 (`controls/c6_probe_pool.py`) re-ran every condition
# measuring on a held-out pool as well, and writes a single nested JSON instead,
# so it gets its own reader rather than bending `load` to two shapes.
#
# Figures that plot agreement read the probe values. C-6 established that the
# training-pool numbers overstate faithfulness — by 0.753 at generation 1 in the
# argmax condition — so they are not what a claim should rest on.

def load_c6():
    with open(os.path.join(LOGS, "c6_probe_pool.json")) as handle:
        data = json.load(handle)
    return {r["condition"]: r for r in data["results"]}


def c6_col(cond, pool, key):
    """Generations and one metric for one condition, on the chosen pool."""
    rows = cond["rows"]
    return [r["gen"] for r in rows], [r[pool][key] for r in rows]


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=12)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, REPO)}")


def fig_train_vs_probe():
    """C-6: what the training pool hid.

    Replaces the earlier temperature-sweep figure, which plotted training-pool
    agreement and showed a clean monotone ordering in tau. That ordering does not
    exist on held-out inputs. Dashed traces are the training pool, solid are the
    probe pool; the gap between them is the measurement error, not a result.
    """
    c6 = load_c6()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Left: the temperature ordering, and its disappearance.
    for tau in ("0.5", "1.0", "2.0"):
        cond = c6[f"tau{tau}_b4096"]
        x, tr = c6_col(cond, "train", "agree_gen0")
        _, pr = c6_col(cond, "probe", "agree_gen0")
        a.plot(x, tr, color=SEQ[tau], linewidth=1.4, linestyle=(0, (4, 3)), alpha=.85)
        a.plot(x, pr, color=SEQ[tau], linewidth=2.2, marker="o", markersize=3.5,
               label=f"tau = {tau}")
    a.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    a.annotate(f"chance = {CHANCE:.3f}", (0.4, CHANCE), textcoords="offset points",
               xytext=(0, 7), fontsize=9, color=MUTED)
    # The three probe traces land within 0.013 of each other, which is closer than
    # chance agreement itself — so they are labelled as one cluster rather than
    # three values that would simply overprint.
    a.annotate("all three within 0.013,\ncloser than chance", (25, 0.137),
               textcoords="offset points", xytext=(-96, 46), fontsize=9,
               color=INK, ha="left",
               arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    style(a, "generation", "functional agreement with generation 0",
          "Temperature orders decay only on training data")
    a.set_ylim(-0.03, 1.02)
    a.legend(frameon=False, fontsize=9.5, labelcolor=INK)

    # Right: argmax, where the gap is present before any chaining happens.
    cond = c6["argmax"]
    x, tr = c6_col(cond, "train", "agree_gen0")
    _, pr = c6_col(cond, "probe", "agree_gen0")
    b.fill_between(x, pr, tr, color=CAT[1], alpha=.13, linewidth=0)
    b.plot(x, tr, color=CAT[1], linewidth=1.6, linestyle=(0, (4, 3)),
           label="training pool")
    b.plot(x, pr, color=CAT[1], linewidth=2.2, marker="o", markersize=3.5,
           label="held-out probe pool")
    for series, value in ((tr, tr[-1]), (pr, pr[-1])):
        b.annotate(f"{value:.3f}", (x[-1], value), textcoords="offset points",
                   xytext=(7, 0), fontsize=9.5, color=CAT[1], va="center",
                   weight="bold")
    b.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    b.annotate(f"gap {tr[0]-pr[0]:.3f}\nat generation 1", (1, (tr[0] + pr[0]) / 2),
               textcoords="offset points", xytext=(22, -6), fontsize=9,
               color=INK, ha="left",
               arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    style(b, "generation", "functional agreement with generation 0",
          "Argmax preserves the memorized table, not the function")
    b.set_ylim(-0.03, 1.02)
    b.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="center right")

    for ax in (a, b):
        ax.set_xlim(0.2, 27.5)
    fig.tight_layout()
    save(fig, "fig1_train_vs_probe.png")


def fig_cka_blindness():
    """CKA climbs while functional agreement collapses. Both are similarity
    measures on the same 0-1 scale, so they share one axis."""
    rows = load("tau1.0_b4096.jsonl")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # Every curve carries the pool it was measured on in its own label. An
    # earlier version of this figure plotted cka_prev against the held-out
    # agreement alone, which presented two pools as one observation: cka_prev
    # is recorded only on the training pool, so the comparison the headline
    # claim rests on has to be the training-pool pair. The held-out agreement
    # is kept as a third curve because it is the honest generalization figure
    # and the blindness holds against either, but it is no longer the only
    # agreement curve on the axes.
    x, cka = col(rows, "cka_prev")
    _, agree_train = col(rows, "agree_gen0")
    _, agree_probe = c6_col(load_c6()["tau1.0_b4096"], "probe", "agree_gen0")
    ax.plot(x, cka, color=CAT[1], linewidth=2, marker="s", markersize=4,
            label="CKA vs previous generation (training pool)")
    ax.plot(x, agree_train, color=CAT[0], linewidth=2, marker="o", markersize=4,
            label="agreement with generation 0 (training pool)")
    ax.plot(x, agree_probe, color=CAT[0], linewidth=2, marker="o", markersize=3,
            linestyle=(0, (5, 2)), alpha=0.75,
            label="agreement with generation 0 (held-out)")
    for series, color, offset in ((cka, CAT[1], 0), (agree_train, CAT[0], 5),
                                  (agree_probe, CAT[0], -8)):
        ax.annotate(f"{series[-1]:.3f}", (x[-1], series[-1]),
                    textcoords="offset points", xytext=(7, offset), fontsize=9.5,
                    color=color, va="center", weight="bold")
    ax.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    style(ax, "generation", "similarity (all measures, 0-1)",
          "CKA reports near-identity exactly where the models have stopped agreeing")
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlim(0.2, 27.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="center right")
    save(fig, "fig2_cka_blindness.png")


def fig_contraction():
    """Alphabet size and effective rank have different units, so they get
    separate panels rather than a second y-axis."""
    tight, wide = load("tau1.0_b128.jsonl"), load("tau1.0_b4096.jsonl")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for rows, color, label in ((wide, CAT[0], "B = 4096"), (tight, CAT[1], "B = 128")):
        x, y = col(rows, "n_classes_used")
        a.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=label)
        a.annotate(f"{y[-1]:d}", (x[-1], y[-1]), textcoords="offset points",
                   xytext=(7, 0), fontsize=9.5, color=color, va="center", weight="bold")
    a.set_ylim(0, 55)
    style(a, "generation", "output classes still in use (of 50)",
          "A tight bottleneck contracts the output alphabet")
    a.legend(frameon=False, fontsize=9.5, labelcolor=INK)

    for rows, color, label in ((wide, CAT[0], "B = 4096"), (tight, CAT[1], "B = 128")):
        x, y = col(rows, "eff_rank")
        b.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=label)
        b.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), textcoords="offset points",
                   xytext=(7, 0), fontsize=9.5, color=color, va="center", weight="bold")
    # At B = 128 effective rank bottoms out around generation 10 and then climbs
    # while the alphabet is still collapsing, so the two measures dissociate.
    # The title must not claim they fall together -- they do not.
    b.annotate("rank recovers while\nthe alphabet collapses", (10, 23.4),
               textcoords="offset points", xytext=(18, -34), fontsize=9,
               color=CAT[1], ha="left",
               arrowprops=dict(arrowstyle="->", color=CAT[1], lw=1.1))
    style(b, "generation", "effective rank of penultimate-layer covariance",
          "But effective rank does not: at B = 128 it recovers after generation 10")
    b.legend(frameon=False, fontsize=9.5, labelcolor=INK)

    for ax in (a, b):
        ax.set_xlim(0.2, 27.5)
    fig.tight_layout()
    save(fig, "fig3_code_contraction.png")


def fig_bottleneck():
    """Agreement decay against bottleneck width, on held-out inputs.

    Unlike the temperature ordering, this one survives C-6: the probe-pool
    ordering 0.025 / 0.058 / 0.130 matches the training pool. Regenerated on
    probe data for consistency with the other agreement figures, not because the
    conclusion changed.
    """
    c6 = load_c6()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = (("tau1.0_b4096", CAT[0], "B = 4096"),
              ("tau1.0_b512", CAT[2], "B = 512"),
              ("tau1.0_b128", CAT[1], "B = 128"))
    for name, color, label in series:
        x, y = c6_col(c6[name], "probe", "agree_gen0")
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(7, 0), fontsize=9, color=color, va="center")
    ax.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"chance = {CHANCE:.3f}", (0.4, CHANCE), textcoords="offset points",
                xytext=(0, 7), fontsize=9, color=MUTED)
    style(ax, "generation", "functional agreement with generation 0 (held-out)",
          "On held-out inputs the bottleneck is the only lever that moves the outcome")
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlim(0.2, 27.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK)
    save(fig, "fig4_bottleneck.png")




def fig_c1():
    """C-1: intact versus shuffled transmission, 20 seeds per arm."""
    with open(os.path.join(LOGS, "c1_shuffled_n20.json")) as handle:
        data = json.load(handle)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    arms = (("intact", CAT[0], "intact transmission"),
            ("shuffled", CAT[1], "shuffled transmission"))
    for arm, color, label in arms:
        for i, run in enumerate(data["results"][arm]):
            gens = range(1, len(run["sizes"]) + 1)
            ax.plot(gens, run["sizes"], color=color, linewidth=1.8, alpha=0.85,
                    label=label if i == 0 else None)
        finals = [r["final"] for r in data["results"][arm]]
        mean = sum(finals) / len(finals)
        ax.annotate(f"mean {mean:.1f}", (len(data['results'][arm][0]['sizes']), mean),
                    textcoords="offset points", xytext=(9, 0), fontsize=9.5,
                    color=color, va="center", weight="bold")
    style(ax, "generation", "output classes still in use (of 50)",
          "C-1: destroying the input-to-target pairing makes contraction worse, not better")
    ax.set_ylim(0, 55)
    ax.set_xlim(0.5, 29)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK)
    save(fig, "fig5_c1_shuffled_control.png")


def fig_c3_budget():
    """C-3: the one-step fidelity floor moves with the training budget.

    Reads the sweep written by ``controls/c3_one_step.py``. The preregistered
    ceiling is the upper bound of C-2's 1,500-step reference interval, committed
    before the sweep ran; the measurement crosses it at four times the budget, so
    the "floor" measures optimization rather than what the bottleneck destroys.
    """
    with open(os.path.join(LOGS, "c3_one_step.json")) as handle:
        data = json.load(handle)
    budgets = data["budgets"]
    conds = data["conditions"]
    means = [conds[str(b)]["probe_agree_mean"] for b in budgets]
    lo = [conds[str(b)]["probe_agree_ci95"][0] for b in budgets]
    hi = [conds[str(b)]["probe_agree_ci95"][1] for b in budgets]
    threshold = data["threshold"]
    ref = data["reference_interval"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(ref[0], ref[1], color=CAT[0], alpha=0.10)
    ax.annotate("C-2 reference interval", (budgets[0], ref[1]),
                textcoords="offset points", xytext=(3, -13), fontsize=9, color=CAT[0])
    ax.axhline(threshold, color=CAT[1], linewidth=1.3, linestyle=(0, (4, 3)))
    ax.annotate(f"preregistered ceiling = {threshold:.3f}", (budgets[-1], threshold),
                textcoords="offset points", xytext=(-4, -14), fontsize=9,
                color=CAT[1], ha="right")
    ax.fill_between(budgets, lo, hi, color=CAT[0], alpha=0.22, linewidth=0)
    ax.plot(budgets, means, color=CAT[0], linewidth=2, marker="o", markersize=4.5)
    for b, m in zip(budgets, means):
        # 1,500 is the sanity arm and sits on the preregistered ceiling, so its
        # label goes below the marker to keep the dashed line readable.
        dy = -18 if b == 1500 else 9
        ax.annotate(f"{m:.3f}", (b, m), textcoords="offset points", xytext=(0, dy),
                    fontsize=9, color=CAT[0], ha="center")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b:,}" for b in budgets])
    ax.minorticks_off()
    style(ax, "training steps per student (same 20 seeds at every budget)",
          "agreement after one distillation step (held-out)",
          "C-3: the one-step floor is not a floor — it moves with the training budget")
    ax.set_ylim(0.10, 0.35)
    save(fig, "fig6_c3_budget.png")


def fig_c4o_null_separation():
    """C-4O: the two nulls sit on top of each other, so neither gate is readable.

    The point of this figure is a negative one and it is the clearest thing the
    period produced. The observed overlap sits far to the right of both nulls,
    which is why a p-value of 1e-4 was available against either. But the
    frequency-matched null and the uniform null differ by 0.0009 when the
    matched null's own standard deviation is 0.0052, so the weighting that was
    supposed to make the comparison informative did not move it at all. Reading
    a verdict off the right-hand gap would have been reading the uniform null
    under another name.
    """
    with open(os.path.join(LOGS, "c4o_overlap_null.json")) as handle:
        data = json.load(handle)
    gate = data["gate_a_jaccard"]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    lo = min(gate["null_min"], gate["uniform_null_mean"] - 4 * gate["uniform_null_sd"])
    hi = max(gate["null_max"], gate["observed"]) + 0.01
    bins = [lo + (hi - lo) * i / 70.0 for i in range(71)]

    # The null distributions are not stored in the log - only their summaries -
    # so the curves are the Gaussian the reported mean and sd describe. That is
    # an approximation and is labelled as one; the floor comparison this figure
    # exists to show depends only on those two summaries.
    import math
    for mean, sd, color, label in (
        (gate["uniform_null_mean"], gate["uniform_null_sd"], MUTED,
         f"uniform null (mean {gate['uniform_null_mean']:.4f})"),
        (gate["null_mean"], gate["null_sd"], CAT[1],
         f"frequency-matched null (mean {gate['null_mean']:.4f})"),
    ):
        y = [math.exp(-0.5 * ((b - mean) / sd) ** 2) for b in bins]
        ax.plot(bins, y, color=color, linewidth=2, label=label)
        ax.fill_between(bins, y, color=color, alpha=0.12)

    ax.axvline(gate["observed"], color=CAT[0], linewidth=2.2)
    ax.annotate(f"observed {gate['observed']:.4f}",
                (gate["observed"], 0.92), textcoords="offset points", xytext=(-9, 0),
                fontsize=10, color=CAT[0], ha="right", weight="bold")
    ax.annotate(
        f"the two nulls differ by {gate['null_separation']:.4f},\nagainst a floor of "
        f"{gate['separation_floor']:.4f} — so this\ncomparison measures nothing",
        (gate["null_mean"], 0.55), textcoords="offset points", xytext=(26, 0),
        fontsize=9.5, color=INK, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.1,
                        shrinkA=2, shrinkB=6))

    ax.set_ylim(0, 1.22)
    ax.set_yticks([])
    style(ax, "mean pairwise Jaccard among Arm A's twenty survivor sets",
          "null density (normal fit to the reported mean and sd)",
          "C-4O: the matched null never left the uniform null, so no verdict was taken")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper left")
    save(fig, "fig7_c4o_null_separation.png")


def fig_sampler_exactness():
    """S-1 precondition: the plan's sampler was correctly centred and far too narrow.

    Left, every seed's range of marginal standard-deviation ratios against the
    exact linear-Gaussian posterior, on a log axis because the anchor-only
    variant sits a factor of twenty below the band. Right, the analytic ratio
    for the anchor-only variant as the number of observations grows: it is
    exact only with no data, so more draws cannot rescue it. Both panels read
    logs/anchored_sampler_exactness.json and nothing else.
    """
    with open(os.path.join(LOGS, "anchored_sampler_exactness.json")) as handle:
        data = json.load(handle)
    tol = data["tolerances"]["sd_ratio"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    variants = (("anchor_only", CAT[1], "anchor only (plan's formula)"),
                ("randomize_then_optimize", CAT[0], "randomize-then-optimize"))
    for offset, (name, color, label) in enumerate(variants):
        rows = [r for r in data["results"] if r["variant"] == name]
        for i, r in enumerate(rows):
            x = r["seed"] + (offset - 0.5) * 0.28
            a.plot([x, x], [r["sd_ratio_min"], r["sd_ratio_max"]],
                   color=color, linewidth=6, solid_capstyle="butt",
                   label=label if i == 0 else None)
    a.axhspan(1 - tol, 1 + tol, color=CAT[2], alpha=0.15, linewidth=0)
    a.annotate(f"pass band 1 \u00b1 {tol}", (-0.35, 1 + tol), fontsize=9,
               color=CAT[2], ha="left", va="bottom")
    a.set_yscale("log")
    a.set_ylim(0.02, 2.0)
    a.set_xlim(-0.4, 2.4)
    a.set_xticks([r["seed"] for r in data["results"] if r["variant"] == "anchor_only"])
    style(a, "seed", "sd ratio to exact posterior, min to max",
          "Both centred correctly; only one has the right width")
    a.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right")

    series = data["anchor_only_sd_ratio_by_n"]
    b.plot([r["n_samples"] for r in series], [r["mean_sd_ratio"] for r in series],
           color=CAT[1], linewidth=2, marker="o")
    b.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    b.set_xscale("symlog", linthresh=4)
    b.set_xlim(0, 300)
    b.set_ylim(0, 1.1)
    style(b, "observations N (analytic, no sampling)", "anchor-only mean sd ratio",
          "Exact only when there is no data")
    save(fig, "fig8_sampler_exactness.png")


if __name__ == "__main__":
    fig_train_vs_probe()
    fig_cka_blindness()
    fig_contraction()
    fig_bottleneck()
    fig_c1()
    fig_c3_budget()
    fig_c4o_null_separation()
    fig_sampler_exactness()
