"""
Rank-based statistics shared by the C-4 family of controls.

Two call sites need the same primitives, which is why they live here rather
than in either control. ``controls/c4_timing.py`` computes a pooled Spearman
correlation between transmitted signal and death generation (C-4T's gate A),
and ``controls/c4_survival_regression.py`` computes that same correlation
against a founder covariate plus a partial correlation with founder class
frequency projected out (C-4S's gates A and B).

The partial correlation is the reason this module reports more than a
coefficient. ``gates/thresholds.json``'s C-4S headroom clause requires the
rank variance each residual retains to be reported beside every p-value, and
declares a gate NOT EVALUABLE when either residual keeps less than one
percent of its unresidualised rank variance. A partial correlation computed
on a residual with no variance left is not a small effect; it is no
measurement at all, reported in the shape of one. That is the defect C-4's
gate B had in a different coordinate system -- a statistic pinned at its own
ceiling, read as a near miss -- and ``partial_spearman`` refuses it by
construction rather than leaving it to a reader to notice.

NumPy only, matching ``analysis/overlap_stats.py``. No function here requires
scipy, and no function mutates its arguments.

Definition used for the partial correlation: both variables' average ranks
are regressed on the control's average ranks by ordinary least squares, and
the Pearson correlation of the two residual vectors is returned. This is the
standard partial Spearman coefficient. Re-ranking the residuals before
correlating them would be a different statistic, and not the one the
committed rule names.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# C-4S's headroom clause: a residual keeping less than one percent of its
# unresidualised rank variance leaves a partial correlation computed on
# nothing. Default rather than a hardcoded constant at the call site, so the
# controls and the tests agree on the committed figure.
DEFAULT_RETAINED_FLOOR = 0.01


@dataclass(frozen=True)
class PartialResult:
    """A partial Spearman coefficient with the room its residuals had.

    ``rho`` is nan exactly when ``reason`` is set. ``retained_a`` and
    ``retained_b`` are each the residual's rank variance as a fraction of
    its unresidualised rank variance, and are reported whether or not the
    gate is evaluable: a reader needs to see how much room the statistic had
    before reading its value.
    """

    rho: float
    retained_a: float
    retained_b: float
    reason: Optional[str]


def _as_1d(x, name):
    """Return a fresh float array, rejecting an empty one."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty; ranks are undefined on no observations")
    return arr


def average_ranks(x):
    """One-based average ranks; tied values share the mean of their positions.

    Matches ``scipy.stats.rankdata``'s default "average" method, and the
    behaviour of ``controls/c4_timing.py``'s ``_rankdata_avg``. Returns a new
    array; ``x`` is not modified.
    """
    values = _as_1d(x, "input")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=float)

    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + 1 + stop + 1) / 2.0
        start = stop + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation, or nan when either input is constant.

    A constant input has no order to correlate, so the coefficient is
    undefined. Returning nan rather than 0.0 is deliberate: zero reads as "a
    relation was measured and found absent", which is a different claim from
    "nothing was measurable".
    """
    ra, rb = _ranks_of_pair(a, b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _ranks_of_pair(a, b):
    """Average ranks of two equal-length inputs."""
    va, vb = _as_1d(a, "first input"), _as_1d(b, "second input")
    if va.size != vb.size:
        raise ValueError(
            f"inputs must be the same length; got {va.size} and {vb.size}"
        )
    return average_ranks(va), average_ranks(vb)


def residualise_on_ranks(y, control):
    """Project the control's rank order out of ``y``'s rank order.

    Returns ``(residual, retained)``, where ``residual`` is the centred rank
    vector of ``y`` with its least-squares fit on the centred rank vector of
    ``control`` removed, and ``retained`` is the residual's variance as a
    fraction of the centred rank variance it started with.

    Two degenerate inputs, both of which occur in practice:

    - A constant control projects nothing out. The residual is the centred
      rank vector itself and ``retained`` is 1.0.
    - A constant ``y`` has no rank variance to retain, so ``retained`` is
      0.0 rather than an undefined ratio.

    Neither argument is modified.
    """
    ry, rc = _ranks_of_pair(y, control)
    centred_y = ry - ry.mean()
    centred_c = rc - rc.mean()

    total = float(centred_y @ centred_y)
    if total == 0.0:
        return centred_y, 0.0

    denominator = float(centred_c @ centred_c)
    if denominator == 0.0:
        return centred_y, 1.0

    beta = float(centred_c @ centred_y) / denominator
    residual = centred_y - beta * centred_c
    retained = float(residual @ residual) / total
    return residual, max(retained, 0.0)


def partial_spearman(a, b, control, retained_floor=DEFAULT_RETAINED_FLOOR):
    """Spearman correlation between ``a`` and ``b`` with ``control`` removed.

    Both variables are residualised on the control by rank regression and the
    Pearson correlation of the residuals is returned. The result carries the
    rank variance each residual retained, and is NOT EVALUABLE -- ``rho`` nan
    and ``reason`` set -- when either retains less than ``retained_floor`` of
    it, per C-4S's committed headroom clause.
    """
    residual_a, retained_a = residualise_on_ranks(a, control)
    residual_b, retained_b = residualise_on_ranks(b, control)

    starved = _name_starved_residual(retained_a, retained_b, retained_floor)
    if starved is not None:
        return PartialResult(float("nan"), retained_a, retained_b, starved)

    if np.std(residual_a) == 0 or np.std(residual_b) == 0:
        return PartialResult(
            float("nan"), retained_a, retained_b,
            "a residual has zero variance, so the partial correlation is undefined",
        )

    rho = float(np.corrcoef(residual_a, residual_b)[0, 1])
    return PartialResult(rho, retained_a, retained_b, None)


def _name_starved_residual(retained_a, retained_b, floor):
    """Which residual fell below the floor, as a sentence, or None."""
    below = [
        name for name, value in (("first", retained_a), ("second", retained_b))
        if value < floor
    ]
    if not below:
        return None
    which = " and ".join(below)
    return (
        f"the {which} residual retains less than {floor:.0%} of its "
        "unresidualised rank variance, so the control already explains its "
        "order and a partial correlation would be computed on nothing"
    )
