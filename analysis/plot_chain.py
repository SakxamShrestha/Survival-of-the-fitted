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


def fig_decay():
    """Agreement with generation 0, swept over distillation temperature."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tau in ("0.5", "1.0", "2.0"):
        rows = load(f"tau{tau}_b4096.jsonl")
        x, y = col(rows, "agree_gen0")
        ax.plot(x, y, color=SEQ[tau], linewidth=2, marker="o", markersize=4,
                label=f"tau = {tau}")
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(7, 0), fontsize=9, color=SEQ[tau], va="center")
    ax.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"chance = {CHANCE:.3f}", (0.4, CHANCE), textcoords="offset points",
                xytext=(0, 7), fontsize=9, color=MUTED)
    style(ax, "generation", "functional agreement with generation 0",
          "Ancestral information decays monotonically at every temperature")
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlim(0.2, 27.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK)
    save(fig, "fig1_decay_by_temperature.png")


def fig_cka_blindness():
    """CKA climbs while functional agreement collapses. Both are similarity
    measures on the same 0-1 scale, so they share one axis."""
    rows = load("tau1.0_b4096.jsonl")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x, agree = col(rows, "agree_gen0")
    _, cka = col(rows, "cka_prev")
    ax.plot(x, cka, color=CAT[1], linewidth=2, marker="s", markersize=4,
            label="CKA vs previous generation")
    ax.plot(x, agree, color=CAT[0], linewidth=2, marker="o", markersize=4,
            label="functional agreement with generation 0")
    ax.annotate(f"{cka[-1]:.3f}", (x[-1], cka[-1]), textcoords="offset points",
                xytext=(7, 0), fontsize=9.5, color=CAT[1], va="center", weight="bold")
    ax.annotate(f"{agree[-1]:.3f}", (x[-1], agree[-1]), textcoords="offset points",
                xytext=(7, 0), fontsize=9.5, color=CAT[0], va="center", weight="bold")
    ax.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    style(ax, "generation", "similarity (both measures, 0-1)",
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
    """Agreement decay against bottleneck width at fixed temperature."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = (("tau1.0_b4096.jsonl", CAT[0], "B = 4096"),
              ("tau1.0_b512.jsonl", CAT[2], "B = 512"),
              ("tau1.0_b128.jsonl", CAT[1], "B = 128"))
    for name, color, label in series:
        rows = load(name)
        x, y = col(rows, "agree_gen0")
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(7, 0), fontsize=9, color=color, va="center")
    ax.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"chance = {CHANCE:.3f}", (0.4, CHANCE), textcoords="offset points",
                xytext=(0, 7), fontsize=9, color=MUTED)
    style(ax, "generation", "functional agreement with generation 0",
          "The bottleneck is a stronger lever than temperature")
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlim(0.2, 27.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK)
    save(fig, "fig4_bottleneck.png")




def fig_c1():
    """C-1: intact versus shuffled transmission, 3 seeds per arm."""
    with open(os.path.join(LOGS, "c1_shuffled.json")) as handle:
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


if __name__ == "__main__":
    fig_decay()
    fig_cka_blindness()
    fig_contraction()
    fig_bottleneck()
    fig_c1()
