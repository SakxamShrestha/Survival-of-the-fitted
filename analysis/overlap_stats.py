"""
Survivor-set overlap statistics for the seed-stability arms.

RESEARCH-PLAN.md section 4.3 identifies three defects in
``seed_stability.expected_jaccard(k, C)``: it computes a ratio of expectations
rather than an expectation of the ratio (a Jensen bias of unknown sign), it
plugs the *mean* survivor-set size into a nonlinear formula (a second Jensen
bias), and the pairwise Jaccards computed from a handful of runs are not
independent, so no textbook standard error applies to them. This module is
the first four of the five replacements that section proposes:

    1. ``cross_arm_permutation_test`` -- the primary statistic.
    2. ``size_conditioned_null`` -- exact resampling null, fixes all three
       defects at once because it is over the identical statistic with the
       identical dependency structure as the data.
    3. ``frequency_matched_null`` -- the same resampling, weighted by a
       founder's generation-0 class frequency, built on the general primitive
       ``draw_survivors_weighted``.
    4. ``overlap_coefficient`` alongside ``jaccard``, because survivor-set
       sizes differ systematically across arms and Jaccard is size-sensitive
       where the overlap coefficient is not.

(The fifth replacement, raising n to 20 founders per arm, is a run-count
decision and has no code here.)

``expected_jaccard`` itself is not edited, and nothing here imports it: its
call sites are a decision for whoever is publishing numbers derived from it,
not for this module.

NumPy only. No function in this file requires scipy.

**Every resampling or permutation null in this file returns the raw null
distribution, not just a p-value**, so it can be handed to ``null_headroom``,
which reports how much room the null had to move before its p-value means
anything. Two preregistered gates in this project already shipped without
that check: C-4's gate B compared classes starved at generation 1 against
classes dead by generation 25, but by generation 25, 37-48 of 50 classes are
already dead, so a random 3-7 class subset already overlaps the dead set at
a null mean of 0.880 -- against an upper bound of 1.0, 0.12 of headroom. The
observed overlap of 0.971 gave p = 0.1004: a gate with almost no room to
detect anything, not a near-miss. C-2R's gates fired at generation 1, before
any chaining existed -- the same root failure in a different shape: the
comparison had not yet had the chance to move. ``null_headroom``
operationalizes the first failure mode; the second is why "not evaluable"
must remain an acceptable verdict for a gate regardless of headroom.
"""

import itertools
from dataclasses import dataclass

import numpy as np

# The RESEARCH-PLAN.md 4.3 cross-arm permutation test specifies 10**4
# reassignments; used as the default repeat count for every Monte Carlo null
# in this module so the defaults are consistent with each other.
DEFAULT_N_PERM = 10_000
DEFAULT_N_NULL = 10_000

# Floating-point slack for the observed-vs-bound sanity check in
# null_headroom. Not a modeling choice -- just enough to absorb rounding.
_BOUND_EPSILON = 1e-9

HIGH = "high"
LOW = "low"
_EXTREME_SIDES = (HIGH, LOW)
_ALTERNATIVES = ("two-sided", "greater", "less")


@dataclass(frozen=True)
class NullTestResult:
    """Result of a permutation or resampling null test.

    observed      -- the statistic computed on the real data.
    null_samples  -- the full null distribution. Kept, not discarded once the
                      p-value is read off, because null_headroom needs it:
                      a p-value alone cannot tell you whether the null was
                      already pinned near the most extreme value the
                      statistic could take.
    p_value       -- computed with +1/+1 smoothing (one more "extreme or
                      equal" draw than observed, one more than the
                      denominator) so a result is never reported as exactly
                      zero, matching the convention already used elsewhere
                      in this repository (controls/c4_timing.py,
                      analysis/seed_stability_stats.py).
    """

    observed: float
    null_samples: np.ndarray
    p_value: float

    def __post_init__(self):
        """Make the held null distribution read-only.

        `frozen=True` blocks rebinding the attribute but not writing through
        the array it points at, so without this a caller could edit a null
        distribution in place after its p-value had been read off it and the
        two would silently disagree. The flag is set on the stored array
        rather than on a copy, so the caller's array is not quietly
        duplicated; every caller in this module passes a freshly built array.
        """
        self.null_samples.flags.writeable = False


def jaccard(a, b):
    """Jaccard similarity |a n b| / |a u b| of two survivor sets.

    Empty-set edge case, explicit rather than a max(denominator, 1) trick:
    both empty -> 1.0, the same set (matching scipy.spatial.distance.jaccard's
    convention for two all-zero vectors: distance 0, similarity 1). Exactly
    one empty needs no special case; the formula is already 0 / |nonempty|.
    """
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def overlap_coefficient(a, b):
    """Overlap coefficient |a n b| / min(|a|, |b|) of two survivor sets.

    Unlike Jaccard, this divides by zero whenever *either* set is empty, not
    only when both are. Two explicit cases, same "empty equals empty"
    convention as jaccard(): both empty -> 1.0 (the same set); exactly one
    empty -> 0.0 (a real zero, not an undefined 0/0 -- a set with no
    survivors cannot overlap one that has some).
    """
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    smaller = min(len(a), len(b))
    if smaller == 0:
        return 0.0
    return len(a & b) / smaller


def _mean_pairwise(sets, pair_fn):
    """Shared core of mean_pairwise_jaccard and mean_pairwise_overlap.

    The mean is over all C(n, 2) distinct pairs, which are not independent
    -- every set appears in n - 1 of them, the dependency RESEARCH-PLAN.md
    4.3 names in expected_jaccard. No standard error is attempted here; use
    cross_arm_permutation_test or size_conditioned_null, whose nulls are
    built from this identical statistic, for a p-value.
    """
    sets = list(sets)
    if len(sets) < 2:
        raise ValueError(
            f"need at least two survivor sets to form a pair, got {len(sets)}"
        )
    pairs = itertools.combinations(range(len(sets)), 2)
    values = [pair_fn(sets[i], sets[j]) for i, j in pairs]
    return float(np.mean(values))


def mean_pairwise_jaccard(sets):
    """Mean Jaccard similarity over all distinct pairs of survivor sets."""
    return _mean_pairwise(sets, jaccard)


def mean_pairwise_overlap(sets):
    """Mean overlap coefficient over all distinct pairs of survivor sets."""
    return _mean_pairwise(sets, overlap_coefficient)


def _p_value(null_samples, observed, alternative):
    """Tail probability of `observed` against `null_samples`, +1/+1 smoothed."""
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}"
        )
    n = null_samples.shape[0]
    if alternative == "two-sided":
        extreme = np.abs(null_samples) >= abs(observed)
    elif alternative == "greater":
        extreme = null_samples >= observed
    else:  # "less"
        extreme = null_samples <= observed
    return float((1 + int(extreme.sum())) / (1 + n))


def cross_arm_permutation_test(
    sets_a,
    sets_b,
    rng,
    n_perm=DEFAULT_N_PERM,
    statistic_fn=mean_pairwise_jaccard,
    alternative="two-sided",
):
    """PRIMARY statistic (RESEARCH-PLAN.md 4.3 item 1).

    Pools the chains of arm A and arm B, randomly reassigns which pooled
    chain counts as "arm A" and which as "arm B" (preserving group sizes)
    `n_perm` times, and recomputes the difference in `statistic_fn` each
    time. The null is exactly "the arm label carries no information about
    survivor-set overlap" -- no baseline formula, no distributional
    assumption about the statistic, needed.

    `statistic_fn` defaults to mean_pairwise_jaccard; pass
    mean_pairwise_overlap to run the same test on the overlap coefficient
    (4.3 item 4 calls for reporting both). Returns a NullTestResult whose
    `observed` is statistic_fn(sets_a) - statistic_fn(sets_b) and whose
    `null_samples` are that difference recomputed under each relabeling.
    """
    if n_perm < 1:
        raise ValueError(f"n_perm must be positive, got {n_perm}")
    sets_a, sets_b = list(sets_a), list(sets_b)
    if len(sets_a) < 2 or len(sets_b) < 2:
        raise ValueError("each arm needs at least two chains for a pairwise statistic")

    observed = statistic_fn(sets_a) - statistic_fn(sets_b)
    pooled = sets_a + sets_b
    n_a, n_total = len(sets_a), len(pooled)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(n_total)
        relabeled_a = [pooled[j] for j in perm[:n_a]]
        relabeled_b = [pooled[j] for j in perm[n_a:]]
        null[i] = statistic_fn(relabeled_a) - statistic_fn(relabeled_b)

    p_value = _p_value(null, observed, alternative)
    return NullTestResult(observed=float(observed), null_samples=null, p_value=p_value)


def draw_survivors_weighted(sizes, weights, rng):
    """Draw survivor sets of given sizes, sampled without replacement within
    each set, weighted by `weights`.

    What "weighted" means here, stated precisely because it is easy to
    overclaim. The first class in a set is drawn with probability exactly
    proportional to `weights`; each subsequent class is drawn from the
    remaining classes with the weights renormalized, which is what numpy's
    Generator.choice(replace=False, p=...) does. For a set of size k > 1 the
    resulting per-class *inclusion* probability is therefore not proportional
    to its weight, and no without-replacement scheme can make it so while
    also drawing exactly k distinct classes. The ordering is preserved -
    heavier classes are included more often - which is all the nulls built on
    this function require. A null that needs exact inclusion probabilities
    needs a different sampler, and must not assume this one provides them.

    General-purpose primitive, deliberately agnostic about where `weights`
    comes from: pass a uniform array for an unweighted null
    (size_conditioned_null) or a founder's generation-0 class-frequency
    vector for a frequency-matched null (frequency_matched_null, built
    entirely on this function), or any other weighting a future control
    script needs. No assumption beyond "a non-negative vector over
    n_classes classes."

    sizes/weights/rng: iterable of non-negative draw sizes; a 1-D
    non-negative array of length n_classes, not required pre-normalized;
    a numpy.random.Generator. Returns a new list of 1-D int arrays, one per
    entry in `sizes`. Neither `sizes` nor `weights` is mutated.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got shape {weights.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    n_classes = weights.shape[0]
    support = int(np.count_nonzero(weights))
    probs = weights / total

    drawn = []
    for k in sizes:
        if k < 0:
            raise ValueError(f"survivor-set size must be non-negative, got {k}")
        if k > support:
            raise ValueError(
                f"cannot draw {k} classes without replacement from only "
                f"{support} classes with nonzero weight"
            )
        if k == 0:
            drawn.append(np.array([], dtype=int))
            continue
        idx = rng.choice(n_classes, size=k, replace=False, p=probs)
        drawn.append(np.asarray(idx, dtype=int))
    return drawn


def _resampling_null(sets, weights, rng, n_null, statistic_fn, alternative):
    """Shared core of size_conditioned_null and frequency_matched_null.

    Both draw synthetic survivor sets matching the *observed* sizes and
    differ only in whether classes are drawn uniformly or weighted; factored
    here so the two null constructions, which must stay the identical
    statistic over the identical dependency structure, cannot drift apart
    from each other by an edit to only one of them.
    """
    sets = list(sets)
    if len(sets) < 2:
        raise ValueError("need at least two survivor sets for a pairwise statistic")
    if n_null < 1:
        raise ValueError(f"n_null must be positive, got {n_null}")

    sizes = [len(s) for s in sets]
    observed = statistic_fn(sets)

    null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        draws = draw_survivors_weighted(sizes, weights, rng)
        null[i] = statistic_fn(draws)

    p_value = _p_value(null, observed, alternative)
    return NullTestResult(observed=float(observed), null_samples=null, p_value=p_value)


def size_conditioned_null(
    sets,
    n_classes,
    rng,
    n_null=DEFAULT_N_NULL,
    statistic_fn=mean_pairwise_jaccard,
    alternative="greater",
):
    """Exact resampling null, conditioned on the OBSERVED survivor-set sizes
    (RESEARCH-PLAN.md 4.3 item 2).

    Fixes all three expected_jaccard(k, C) defects at once: each replicate
    computes the real statistic and those are averaged, so this is an
    expectation of the ratio, not a ratio of expectations; it uses every
    run's actual size, never a mean plugged into a formula; and each
    replicate draws a full set of synthetic survivor sets jointly, so the
    same dependency structure as the real data (each synthetic set in n - 1
    pairs) is reproduced instead of assumed away.

    `statistic_fn` defaults to mean_pairwise_jaccard; pass
    mean_pairwise_overlap for the overlap coefficient's null instead.
    `alternative` defaults to "greater" (is the observed overlap higher than
    chance alone would produce at these sizes); pass "two-sided" or "less"
    for a different direction.
    """
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}")
    uniform_weights = np.ones(n_classes, dtype=float)
    return _resampling_null(sets, uniform_weights, rng, n_null, statistic_fn, alternative)


def frequency_matched_null(
    sets,
    weights,
    rng,
    n_null=DEFAULT_N_NULL,
    statistic_fn=mean_pairwise_jaccard,
    alternative="greater",
):
    """Frequency-matched null (RESEARCH-PLAN.md 4.3 item 3).

    Identical to size_conditioned_null except survivors are drawn with
    probability proportional to `weights` rather than uniformly -- typically
    a founder's generation-0 class frequency. If the cross-arm effect
    vanishes against this null but not against size_conditioned_null's
    uniform one, "founder identity" reduces to "founder class-frequency
    imbalance." Built entirely on draw_survivors_weighted; no sampling
    logic of its own.
    """
    return _resampling_null(sets, weights, rng, n_null, statistic_fn, alternative)


def null_headroom(null_samples, observed, extreme_side, bound=None):
    """The null's mean, and how much room the statistic had left to move.

    READ THIS BEFORE SKIPPING IT: see this module's docstring for the C-4 /
    C-2R history behind it. A p-value says how surprising the observed
    statistic is relative to the null; it says nothing about whether the
    null was already pinned against the most extreme value the statistic
    could take, in which case no p-value from that comparison is
    trustworthy, however small. Every gate against this module's statistics
    must call this and report headroom next to its p-value, not instead of
    it -- a small p-value next to near-zero headroom is not a result.

    null_samples/observed/extreme_side: the null distribution (e.g.
    NullTestResult.null_samples); the statistic's observed value, used only
    to sanity-check it against `bound`, NOT to compute headroom, which
    depends only on the null and the bound; "high" if a large statistic is
    the extreme of interest, "low" if a small one is.

    bound -- the most extreme value the statistic could possibly take in
    that direction. Defaults to 1.0 for "high" and 0.0 for "low", correct
    for every statistic this module produces (Jaccard and the overlap
    coefficient are bounded to [0, 1]). A statistic with a different
    natural range (e.g. a raw difference in death generations, as in
    controls/c4_timing.py's gate B) MUST pass its own bound -- the default
    is this module's constant, not a universal one.

    Returns (null_mean, headroom); headroom is the non-negative distance
    from the null's mean to `bound`.
    """
    if extreme_side not in _EXTREME_SIDES:
        raise ValueError(
            f"extreme_side must be one of {_EXTREME_SIDES}, got {extreme_side!r}"
        )
    null_samples = np.asarray(null_samples, dtype=float)
    if null_samples.size == 0:
        raise ValueError("null_samples is empty; headroom is not evaluable")

    if bound is None:
        bound = 1.0 if extreme_side == HIGH else 0.0

    if extreme_side == HIGH and observed > bound + _BOUND_EPSILON:
        raise ValueError(
            f"observed ({observed}) exceeds the stated upper bound ({bound}); "
            "the statistic or the bound passed to null_headroom is wrong"
        )
    if extreme_side == LOW and observed < bound - _BOUND_EPSILON:
        raise ValueError(
            f"observed ({observed}) is below the stated lower bound ({bound}); "
            "the statistic or the bound passed to null_headroom is wrong"
        )

    null_mean = float(np.mean(null_samples))
    headroom = abs(bound - null_mean)
    return null_mean, headroom
