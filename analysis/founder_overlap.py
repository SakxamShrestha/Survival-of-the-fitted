"""
Same-founder versus different-founder survivor-set overlap: S-1 gate B.

What this file implements, and nothing more
-------------------------------------------
``gates/thresholds.json`` key ``S-1`` states gate B's metric as a "cross-arm
permutation test on mean pairwise Jaccard of survivor sets, same-founder arm
against different-founder arm, computed by analysis/overlap_stats.py". That
sentence names a function that cannot compute it, for the reason recorded in
``design_amendment_2026_11_08.gate_b_arm_assembly``:
``overlap_stats.cross_arm_permutation_test(sets_a, sets_b, ...)`` permutes
which *set* belongs to which list, while same-founder versus
different-founder is a property of a *pair*. With 20 founders at R = 2 there
are 40 survivor sets, and every one of them sits in one same-founder pair and
38 different-founder pairs at once, so no partition of the 40 sets into two
lists separates the two kinds of pair.

That amendment, committed before this file existed, settles the assembly. It
is implemented here verbatim and is the whole contract of this module:

  * **statistic** -- mean Jaccard over the same-founder pairs minus mean
    Jaccard over the different-founder pairs, both taken over the same
    survivor sets. ``same_minus_different``.
  * **null** -- permute the founder label across chains, 10000 permutations,
    holding the survivor sets fixed; which pairs count as same-founder
    follows from the permuted labels. ``gate_b_test``.
  * **tail** -- one-sided for same-founder overlap EXCEEDING
    different-founder overlap.
  * **p-value** -- the conservative permutation estimate
    ``(1 + #{null >= observed}) / (1 + n_perm)``, matching gate A and
    ``analysis/overlap_stats``.

Both pair counts are returned beside the statistic, which the amendment
requires: at R = 2 the arms are 20 against 760, so the same-founder mean is
the noisier side by a wide margin and the observed difference is dominated by
its variance. That is a property of R = 2 and not of the assembly, and the
permutation null carries the same imbalance, so the p-value stays valid. A
reader who cannot see the imbalance cannot judge it, so it is reported rather
than described.

This module does **not** apply Holm-Bonferroni, does not decide the arm's
verdict, and does not read or write any log. The correction is across gates A,
B and C, so it belongs to ``controls/s1_aggregate.py``; the p-value returned
here is raw. Nothing here imports torch: the statistic is set arithmetic and
label permutation, so the module is importable and testable with no model, no
chain and no GPU. NumPy only, no scipy.

What is imported rather than rewritten, and why
-----------------------------------------------
Two quantities are taken from sibling modules on purpose.

``analysis.overlap_stats.jaccard`` is the project's single Jaccard, with its
empty-set convention already settled and tested there (both sets empty is
1.0). Gate A imports the same function for the same reason: gates A, B and C
must not drift apart on what overlap means, and the amendment's stated reason
for having both A and B is that they reach one conclusion "by a second
route", not by a second metric. The empty-set convention is inherited, not
re-decided: if the alphabet collapses to nothing, every pair reads 1.0 and
this statistic reads exactly 0.0. That is a real consequence worth knowing
when reading a null result, and headroom floor 1 is what guards against the
case arising.

``analysis.sigma_attractor.first_order_variance`` is the zeta_1 estimator
required by ``floor_3_has_no_referent_here``. That clause records floor 3's
refusal half as NOT APPLICABLE here -- both gate B and gate C use a single
null, so "the two nulls being compared" has no referent -- while keeping its
reporting half: each gate reports, for the null it uses, the number of sets n
and an estimate of the first-order variance beside its p-value. The imported
estimator carries a leading-order bias correction and a documented
false-non-degenerate rate of roughly one trial in fifteen; a second
implementation here would carry neither property on the record. It is
returned unclipped, so a negative value is possible and is read as "zero,
within the estimator's noise" -- see that function's own docstring.

The two decisions this file had to make, stated rather than buried
------------------------------------------------------------------
**Which sets zeta_1 is taken over.** Gate B has exactly one null and it holds
every survivor set fixed, so n is the number of survivor sets and zeta_1 is
the estimator applied to all of them. The alternative reading -- one zeta_1
per arm -- is not available: ``first_order_variance`` is a function of a
collection of *sets*, and the two arms are collections of *pairs* over the
same sets, which is the same mismatch that made the named cross-arm function
unusable. This is the only reading the estimator admits, but it is a reading,
so it is written down here.

**What happens to an unbalanced design.** The amendment fixes R = 2 for every
founder and every cell, and says nothing about a design where founders differ
in chain count. This module refuses exactly when the statistic is undefined:
when there is no same-founder pair, or no different-founder pair. An
unbalanced design in between is computed, and the returned pair counts report
the shape it actually had. The reason for the weaker refusal is that the
amendment's own justification for R = 2 is the absence of same-founder pairs
at R = 1 -- "at R = 1 there are no same-founder pairs, and the named
implementation refuses" -- which is a statement about the statistic being
undefined, not about balance. Gate A, whose leave-one-chain-out statistic
genuinely needs R >= 2 for every founder, refuses an unbalanced design in
``analysis/founder_id.py``, so the aggregator still sees such a design
refused; it is refused by the gate that cannot compute it rather than by the
gate that can.

Why the Jaccard matrix is computed once
---------------------------------------
Permuting the founder label does not touch the survivor sets, so the
expensive part of the statistic is invariant under the null: only which cells
of a fixed pairwise matrix count as same-founder changes. The pairwise
Jaccards are therefore computed once as a flat array over the C(n, 2)
unordered pairs, and each permutation is label arithmetic over that array.
This is the same pattern, for the same reason, as
``analysis/founder_id._similarity_matrix``. At the committed 40 sets and
10000 permutations the whole test runs in well under a second.
"""

import os
import sys

import numpy as np

# analysis/ is not a package with an __init__.py, and Python puts this file's
# directory on sys.path rather than the repository root, so the root is added
# here in order to import the project's single Jaccard and its single zeta_1
# estimator. Same pattern, and the same reason, as analysis/founder_id.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.overlap_stats import jaccard
from analysis.sigma_attractor import first_order_variance

# From gates/thresholds.json key S-1, fixed_in_advance.n_perm, restated by the
# gate B amendment as "10000 permutations". Named rather than inlined so that
# a call site reading the committed count is obvious.
DEFAULT_N_PERM = 10_000

# From S-1 fixed_in_advance.n_founders_per_arm. Not enforced -- the statistic
# is defined at any number of founders above one -- but recorded so a reader
# can see what the committed pair counts of 20 and 760 are counted from.
N_FOUNDERS_COMMITTED = 20

# From design_amendment_2026_11_08.chains_per_founder.
CHAINS_PER_FOUNDER_COMMITTED = 2

# The committed tail: one-sided for same-founder overlap EXCEEDING
# different-founder overlap. Upper tail, never lower, never two-sided.
_P_VALUE_SIDE = "greater"


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _validate_design(survivor_sets, founder_labels):
    """Check the inputs form a design on which gate B's statistic exists.

    Returns ``(sets, label_index)`` where ``sets`` is a list of Python sets
    and ``label_index`` maps each chain to its founder's position in the
    sorted array of distinct founder ids. Positions rather than raw ids,
    because the statistic depends only on which chains share a founder, so
    reducing the ids to positions makes that independence structural instead
    of incidental.

    Validation is at the boundary and fails fast, for the reason
    ``analysis/founder_id._validate_design`` gives: every downstream quantity
    is meaningless if the design is malformed, and a malformed design is far
    more likely to arrive as a silently wrong number than as an exception.

    The two conditions specific to this gate are that at least one
    same-founder pair and at least one different-founder pair exist, since
    the statistic is a difference of two means and neither mean is defined
    over an empty arm.
    """
    sets = [set(s) for s in survivor_sets]
    labels = np.asarray(list(founder_labels))

    if len(sets) != labels.shape[0]:
        raise ValueError(
            "survivor_sets and founder_labels must be the same length; got "
            "{} survivor sets and {} labels".format(len(sets), labels.shape[0])
        )
    if not sets:
        raise ValueError(
            "survivor_sets is empty; there are no pairs and no statistic"
        )
    if labels.ndim != 1:
        raise ValueError(
            "founder_labels must be one label per chain, a 1-D sequence; got "
            "shape {}".format(labels.shape)
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "founder_labels must be integer founder ids; got dtype {}".format(
                labels.dtype
            )
        )

    founders, label_index, class_counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )

    if founders.shape[0] < 2:
        raise ValueError(
            "gate B needs at least two founders; got {}. With one founder "
            "every pair is a same-founder pair, there are no "
            "different-founder pairs, and the difference of the two arm "
            "means is undefined.".format(founders.shape[0])
        )
    if not np.any(class_counts >= 2):
        raise ValueError(
            "gate B needs at least one founder with two or more chains; "
            "every founder here has one, so there are no same-founder pairs "
            "and the same-founder arm mean is undefined. S-1 "
            "design_amendment_2026_11_08.chains_per_founder fixes R = 2 for "
            "exactly this reason: 'at R = 1 there are no same-founder "
            "pairs'."
        )

    return sets, label_index


def _validate_n_perm(n_perm):
    """Check the permutation count is a positive whole number."""
    if int(n_perm) != n_perm or n_perm < 1:
        raise ValueError(
            "n_perm must be a positive integer number of permutations; got "
            "{!r}".format(n_perm)
        )
    return int(n_perm)


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------

def _pair_index(n):
    """The row and column indices of every unordered pair i < j.

    Returned as two integer arrays so that a permuted label vector can be
    turned into a same-founder mask for all C(n, 2) pairs at once, with no
    Python-level loop inside the permutation loop.
    """
    rows, cols = np.triu_indices(n, k=1)
    return rows, cols


def _pair_jaccards(sets, rows, cols):
    """Jaccard of every unordered pair, in the order ``_pair_index`` gives.

    Computed once per call and reused across every permutation; this is the
    whole reason 10000 permutations are affordable. The project's single
    Jaccard is used pair by pair rather than reimplemented over indicator
    matrices, because the survivor sets are small and correctness across the
    three gates matters more here than a constant factor.
    """
    return np.array(
        [jaccard(sets[i], sets[j]) for i, j in zip(rows, cols)],
        dtype=np.float64,
    )


def _arm_means(pair_values, same_mask):
    """The two arm means, same-founder first.

    Raises if either arm is empty. Under the committed null that cannot
    happen -- permuting the label multiset preserves every founder's chain
    count, so the number of same-founder pairs is the same on every draw as
    on the observed data -- but the check is made rather than assumed,
    because a silently empty arm would arrive as a NaN difference and a NaN
    compares False against everything, which would read as "no null draw was
    as extreme as the observed" and so as a significant gate.
    """
    n_same = int(np.count_nonzero(same_mask))
    n_different = pair_values.shape[0] - n_same
    if n_same == 0 or n_different == 0:
        raise ValueError(
            "a founder relabelling left one arm empty ({} same-founder "
            "pairs, {} different-founder pairs); the difference of the two "
            "arm means is undefined".format(n_same, n_different)
        )
    same_total = float(pair_values[same_mask].sum())
    different_total = float(pair_values.sum()) - same_total
    return same_total / n_same, different_total / n_different


def same_minus_different(survivor_sets, founder_labels):
    """S-1 gate B's statistic, as committed.

    Mean Jaccard over the same-founder pairs minus mean Jaccard over the
    different-founder pairs, both taken over the same survivor sets. At the
    committed design -- 20 founders at R = 2, 40 survivor sets -- that is a
    mean over 20 pairs minus a mean over 760.

    survivor_sets  -- a sequence of N sets, or of any sequences of int class
                      ids; duplicates are absorbed by set semantics. The
                      empty-set convention is whatever
                      ``analysis/overlap_stats.jaccard`` says it is and is
                      not re-decided here.
    founder_labels -- a sequence of N ints, the founder each chain descended
                      from. Only the partition they induce matters; renaming
                      the founders cannot move the statistic.

    Raises ValueError if the two inputs disagree in length, if fewer than two
    founders are present, if no founder has two or more chains, or if the
    labels are not integers. Returns a float in [-1, 1].
    """
    sets, label_index = _validate_design(survivor_sets, founder_labels)
    rows, cols = _pair_index(len(sets))
    pair_values = _pair_jaccards(sets, rows, cols)
    same_mask = label_index[rows] == label_index[cols]
    same_mean, different_mean = _arm_means(pair_values, same_mask)
    return float(same_mean - different_mean)


# --------------------------------------------------------------------------
# Gate B's permutation test
# --------------------------------------------------------------------------

def gate_b_test(survivor_sets, founder_labels, n_perm=DEFAULT_N_PERM, seed=0):
    """S-1 gate B: one-sided permutation test on same-founder overlap.

    The null permutes the founder label across chains, holding the survivor
    sets fixed, which is the null the committed amendment names and the same
    permutation gate A uses. Because it is a permutation of the existing
    label multiset, every founder keeps its chain count, the number of
    same-founder pairs is identical on every draw, and the arms carry the
    same 20-against-760 imbalance in the null as in the observed data -- which
    is why the amendment can say the p-value remains valid despite it.

    The label multiset is permuted in its sorted order rather than in the
    order the chains happened to be listed in. The draws are uniform either
    way; sorting first means the null distribution is a function of the
    survivor sets and the founders' chain counts alone, so two designs over
    the same sets with the same chain counts produce the same null at the
    same seed. That makes a logged null reproducible from the design summary
    rather than only from the full label vector.

    Returns a dict with exactly these keys:

        observed         -- same_minus_different on the real labels.
        null_mean        -- mean of the permutation distribution. Its
                            expectation is exactly zero whatever the data,
                            because under a uniformly random relabelling
                            every unordered pair is equally likely to fall in
                            the same-founder arm, so both arm means share one
                            expectation.
        null_sd          -- sample standard deviation (ddof=1), matching
                            gate A.
        null_min         -- smallest value the null took.
        null_max         -- largest value the null took. Together with
                            null_min this is the "range" the S-1 headroom
                            block requires beside every p-value.
        p_value          -- the one-sided upper-tail estimate, uncorrected.
        n_perm, seed     -- echoed so a logged result is reproducible from
                            the log alone.
        n_sets           -- N, the number of survivor sets the null holds
                            fixed. This is floor 3's n.
        n_same_pairs     -- the size of the same-founder arm.
        n_different_pairs -- the size of the different-founder arm. Both
                            counts are reported because the amendment
                            requires it: the arms are 20 against 760 at
                            R = 2 and a non-significant gate B must not be
                            read as evidence of absence.
        zeta_1           -- the first-order variance estimate for this null,
                            from analysis/sigma_attractor. This is floor 3's
                            surviving reporting requirement. Unclipped, so it
                            may be negative.

    Floor 3's refusal half is NOT APPLICABLE to this gate and is not computed
    here: gate B uses one null, so "the two nulls being compared" has no
    referent. The aggregator records that, with its reason, and never records
    it as passed.

    Holm-Bonferroni is NOT applied: the correction is across gates A, B and
    C, so it is the aggregator's job. The returned p_value is raw.
    """
    n_perm = _validate_n_perm(n_perm)

    sets, label_index = _validate_design(survivor_sets, founder_labels)
    n_sets = len(sets)
    rows, cols = _pair_index(n_sets)
    pair_values = _pair_jaccards(sets, rows, cols)

    same_mask = label_index[rows] == label_index[cols]
    same_mean, different_mean = _arm_means(pair_values, same_mask)
    observed = float(same_mean - different_mean)
    n_same_pairs = int(np.count_nonzero(same_mask))

    sorted_labels = np.sort(label_index)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        permuted = rng.permutation(sorted_labels)
        permuted_same = permuted[rows] == permuted[cols]
        permuted_same_mean, permuted_different_mean = _arm_means(
            pair_values, permuted_same
        )
        null[k] = permuted_same_mean - permuted_different_mean

    # The committed estimate: (1 + #{null >= observed}) / (1 + n_perm). The
    # +1 in the numerator is the observed statistic itself, which under the
    # null is one more draw from the same distribution; the +1 in the
    # denominator counts it in the total. It can never return exactly zero.
    # The comparison is one-sided and upward, which is the committed tail.
    at_least_as_extreme = int(np.count_nonzero(null >= observed))
    p_value = (1.0 + at_least_as_extreme) / (1.0 + n_perm)

    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)) if n_perm > 1 else 0.0,
        "null_min": float(np.min(null)),
        "null_max": float(np.max(null)),
        "p_value": p_value,
        "n_perm": n_perm,
        "seed": seed,
        "n_sets": int(n_sets),
        "n_same_pairs": n_same_pairs,
        "n_different_pairs": int(pair_values.shape[0] - n_same_pairs),
        "zeta_1": float(first_order_variance(sets)),
    }
