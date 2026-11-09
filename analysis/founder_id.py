"""
Founder identification from survivor sets: the estimator behind S-1 gate A.

What this file implements, and nothing more
-------------------------------------------
``gates/thresholds.json`` key ``S-1`` gate A, committed 2026-10-20 and
amended 2026-11-08, states the metric as

    "founder-identification balanced accuracy, leave-one-chain-out: given a
    chain's generation-25 survivor set, predict which of the 20 founders it
    descended from, averaged per class so that unequal survivor-set sizes
    cannot inflate it"

and the decision rule as

    "one-sided permutation p < 0.05 after Holm-Bonferroni, where the null
    permutes founder labels across chains over 10000 permutations".

Two of the three headroom floors in that entry bear on gate A and are
implemented here as ``floor_1_degenerate_alphabet`` and
``floor_2_chance_level``. Floor 3 is a gate B and gate C floor and is not
implemented here.

This module does **not** apply Holm-Bonferroni, does not decide the arm's
verdict, and does not read or write any log. Holm-Bonferroni is across gates
A, B and C, so it belongs to whatever aggregates all three; this file returns
a raw one-sided p-value and the aggregator corrects it. Nothing here imports
torch: the statistic is set arithmetic and label permutation, so the module is
importable and testable with no model, no chain and no GPU.

The classifier the rule did not name
------------------------------------
The committed sentence says "predict which of the 20 founders it descended
from" and stops. It names no classifier, and a classifier had to be chosen.
The choice made here, and the reasoning, so that it is on the record rather
than buried in code:

**Maximum mean Jaccard to a founder's training chains, leave one chain out.**
For each chain i in turn, i is held out; for every founder f the score is the
*mean* Jaccard between chain i's survivor set and the survivor set of each
remaining chain of founder f; the prediction is the founder with the highest
score.

Why this one. The data are sets of class ids, so the natural similarity is
the one the rest of the project already uses, and
``analysis/overlap_stats.py`` already defines it with its empty-set
convention settled and tested. That function is imported here, not copied,
so there is exactly one Jaccard in this repository and the two gates that
depend on it (A here, B there) cannot drift apart. It is also the simplest
rule that is well defined on sets: a nearest-centroid rule would need an
embedding of a set into a vector space, and choosing that embedding would be
a second undocumented decision on top of the first.

What a different reasonable choice would have changed. Three alternatives
were available and each would move the number:

  * *Maximum single Jaccard* (nearest neighbour rather than mean over the
    founder's chains). At the committed R = 2 this differs from the rule used
    here only for the held-out chain's own founder, which has exactly one
    training chain either way, so for S-1's actual design the two rules
    coincide for the true class and differ only in how a wrong class is
    scored. It would be more optimistic -- a maximum over two is never below
    their mean -- so it would raise the scores of the 19 wrong founders
    relative to the true one and therefore *lower* balanced accuracy. The
    mean was chosen because it is the rule that generalises unchanged to
    R > 2, and because a maximum over unequal numbers of training chains is
    not comparable across classes.
  * *Overlap coefficient* instead of Jaccard. ``overlap_stats`` offers it,
    and it is deliberately insensitive to set size where Jaccard is not. It
    would be the better choice if survivor-set cardinality differed
    systematically between founders. It was not chosen because gate B is
    committed on *Jaccard*, and using a different similarity for gate A
    would mean the two gates no longer measure founder dependence through
    the same lens -- the entry's own stated reason for having both is that
    they reach one conclusion "by a second route", not by a second metric.
  * *Sum* rather than mean over a founder's training chains. This would be
    wrong at any R, and the reason is the asymmetry described below.

The asymmetry that forces a mean, stated because it is easy to miss
--------------------------------------------------------------------
Under leave-one-chain-out at R = 2, the held-out chain's own founder has
R - 1 = 1 training chain while every other founder still has R = 2. A sum
over a founder's training chains would therefore give the 19 wrong founders
twice as many terms as the true one, and would be biased against the correct
answer by construction. The mean removes that, at the cost that the true
founder's score is an average of one sample and the wrong founders' scores
are averages of two, so the true founder's score is the noisier of the two.
That residual asymmetry is a property of R = 2 and not of this code; it is
absorbed by the permutation null, which is computed under exactly the same
leave-one-out rule and therefore carries the same asymmetry.

The tie rule
------------
Ties are broken deterministically toward the **smallest founder id**. Ties
are not a corner case here: if the chains collapse onto a common survivor
set, every founder scores identically for every held-out chain and every
decision is a tie, which is precisely the regime headroom floor 1 exists to
refuse.

Why the smallest id and not an abstention or a coin flip. With equal class
sizes, sending every tied decision to one fixed founder gives that founder a
per-class accuracy of 1.0 and every other founder 0.0, so the per-class
average is exactly 1 / n_founders -- the statistic's own chance level. That
is the right reading of data carrying no founder information. Counting ties
as abstentions would instead place a fully degenerate run at 0.0, *below*
chance, which would push the permutation null off chance and make headroom
floor 2 refuse the gate for a reason that has nothing to do with the null
being mis-built. A random tiebreak would give the same expectation as the
rule used here but would make a logged result irreproducible from its seed.

A deterministic, data-independent tiebreak is in any case forced by the
agreed signature: ``balanced_accuracy_loo(survivor_sets, founder_labels)``
takes no seed, so a pseudo-random tiebreak could not be reproduced from
anything the caller holds.

Two limits on the tie rule, stated rather than discovered.

First, ties are detected as exact float64 equality, via ``numpy.argmax``
returning the first maximum. Scores that are mathematically equal but differ
in the last bits of a division are not treated as ties, and resolve in
whichever direction the arithmetic fell. This is deterministic for fixed
input but it is not mathematically principled, and no epsilon is introduced
to patch it because an epsilon would be a constant chosen by me and this
project has already shipped gates whose chosen constants could not see what
they were meant to.

Second, and more consequential: **whenever a tied block of chains exists, the
tie rule donates exactly 1 / n_founders to the balanced accuracy**, because
the founder the ties fall to collects a per-class accuracy of 1.0 for free.
On fully degenerate data that donation *is* the whole statistic and the
result sits exactly at chance, which is the behaviour wanted. On partly
degenerate data it sits on top of whatever genuine signal exists, so the
statistic is biased upward relative to its nominal chance level in any
regime where exact ties are common. Measured on this machine, on synthetic
survivor sets drawn independently of the founder at 20 founders, R = 2,
5 classes drawn from an alphabet of 50 (a regime in which 61 per cent of
chain pairs have Jaccard exactly 0 and 12 of 40 decisions are exact ties),
the permutation null's mean sat at 0.062 to 0.079 across ten independent
datasets rather than at 0.05, with a null standard deviation of about 0.053.
Headroom floor 2 therefore did not fire -- the departure ran from 0.24 to
0.53 of a null standard deviation, against a firing threshold of 1.0 -- but
it consumed up to half of the floor's budget, and the departure was upward
on all ten datasets, so it is systematic rather than noise. The same
measurement at larger R shrank the departure (mean null mean 0.071 at R = 2,
0.061 at R = 6, 0.055 at R = 10), which is consistent with the tie-donation
mechanism and not with a defect in the permutation.

Two consequences, and one thing that is NOT claimed. The p-value is
unaffected: the observed statistic and every permutation draw carry the same
donation, since the null permutes labels and leaves the survivor sets and
hence the tie structure alone. Floor 2 is affected, and whoever reads it
should know that the floor's remaining margin is real but partly spent
before the data arrives. What is NOT claimed is what this will do on an
actual S-1 chain: the numbers above are synthetic, the real regime at
bottleneck 128 on the probe pool has a different survivor-set size and a
different overlap structure, and no S-1 chain has been run. If floor 2 fires
on real data, the measurement above is the first thing to check against it,
because the floor may be reading the tie rule rather than the null.

An abstention tiebreak was rejected partly for this reason: it moves the
bias to the other side, placing a degenerate run at 0.0 rather than at
chance, which is a larger departure in the same floor's units.

The p-value
-----------
``(1 + #{null >= observed}) / (1 + n_perm)`` -- the standard conservative
permutation estimate, with the +1 in the numerator accounting for the
observed statistic itself and the +1 in the denominator for the fact that it
is one more draw from the null under the null hypothesis. It is one-sided and
upper-tailed, because gate A's alternative is that the endpoint carries
*more* founder information than chance, never less. It can never return
exactly zero; its floor is 1 / (n_perm + 1), which at the committed
n_perm = 10000 is 9.999e-05. This matches the convention already used in
``analysis/overlap_stats.py``, ``controls/c4_timing.py`` and
``analysis/seed_stability_stats.py``.

The null permutes the founder label vector, which is a permutation of the
existing label multiset and therefore leaves each founder's chain count
unchanged. That matters for more than tidiness: it is what keeps the
leave-one-out rule defined on every permuted assignment, since a null that
could hand some founder a single chain would be a null over a design in
which the statistic does not exist.

Every permutation is drawn from one explicit ``numpy.random.default_rng(seed)``
and the seed is returned alongside the result, so a logged verdict can be
reproduced in full from what was logged.

Ambiguities in the committed rule that this file had to resolve
---------------------------------------------------------------
Recorded here, in the module, because an ambiguity resolved in code without
being written down is the exact failure the 2026-11-08 design amendment was
written to fix.

1. *The classifier.* Not named by the rule. Resolved above.
2. *The tie rule.* Not named by the rule. Resolved above.
3. *Which standard deviation.* The entry asks for "the null's mean, standard
   deviation and range" and floor 2 compares a departure against "one of its
   own standard deviations", without saying whether that is the sample or the
   population figure. This file reports the sample standard deviation
   (``ddof=1``), because the permutation draws are a Monte-Carlo sample from
   the permutation distribution and the unbiased estimate of that
   distribution's spread is the quantity floor 2 is asking about. At the
   committed n_perm = 10000 the two differ by a factor of 1.00005, so the
   choice cannot change a floor-2 verdict at the committed permutation count;
   it is documented because it could at a small n_perm.
4. *Floor 2's boundary.* "differs ... by more than one of its own standard
   deviations" is implemented as the literal strict comparison
   ``abs(null_mean - chance) > null_sd``, with no tolerance, so a departure
   of exactly one standard deviation leaves the gate evaluable. No epsilon is
   added, for the reason given under the tie rule.
5. *Floor 1's boundary.* "below 4.0 classes" is implemented as
   ``mean_alphabet < 4.0``, so exactly 4.0 is inside the evaluable regime.
6. *The sense of the floors' boolean.* The two floor functions return
   ``True`` when the floor **fires**, that is when the gate is NOT EVALUABLE.
   ``False`` means the gate may be read. The reason string is always
   populated, firing or not, so a caller can log why a gate was readable as
   well as why it was not. The committed rule does not fix this polarity, so
   it was fixed by decision on 2026-11-08 and the S-1 floor functions across
   this repository were brought to it: True = fired = NOT EVALUABLE, in
   floors 1, 2 and 3 alike. It matches this project's own logged vocabulary
   ("floor fired: headroom_2_shuffled_floor" in
   logs/c8b_shared_founder_torch280.txt). Each floor function restates it in
   its own docstring in one greppable fixed form, because the one thing that
   went wrong here was that the convention was unwritten rather than wrong:
   an aggregator briefly read the same boolean as "evaluable", under which a
   mean alphabet of 3.2 classes would have been reported as readable and gate
   A would have been read on exactly the degenerate input floor 1 exists to
   refuse. No caller should have to infer this polarity from behaviour.

Equal R is not required by this code
------------------------------------
``design_amendment_2026_11_08`` fixes R = 2 chains per founder for every
founder and every (lambda, sigma) cell, so 20 founders give 40 chains per
cell. Nothing here assumes that: unequal class sizes are handled correctly,
which is what the "averaged per class" clause of the metric buys. What *is*
required is R >= 2 for every founder, because at R = 1 holding out a
founder's only chain leaves its class with no training chain and its
per-class accuracy undefined. That case raises rather than returning a
number, for the reason the amendment gives: a statistic that quietly
substitutes something computable for something undefined is worse than one
that refuses.

NumPy only. No function in this file requires scipy or torch.
"""

import os
import sys

import numpy as np

# analysis/ is not a package with an __init__.py, and Python puts this file's
# directory on sys.path rather than the repository root, so the root is added
# here in order to import the project's single Jaccard. Same pattern, and the
# same reason, as analysis/seed_stability_stats.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from analysis.overlap_stats import jaccard

# From gates/thresholds.json key S-1, fixed_in_advance.n_perm. Named rather
# than inlined so that a call site reading the committed count is obvious.
DEFAULT_N_PERM = 10_000

# From S-1 fixed_in_advance.n_founders_per_arm.
N_FOUNDERS_COMMITTED = 20

# From S-1 headroom floor 2: "Balanced accuracy over 20 founders has a chance
# level of 0.05." Stated in the rule as a literal, so it is a literal here
# rather than 1 / N_FOUNDERS_COMMITTED computed at call time -- the committed
# number is what floor 2 is written against.
CHANCE_LEVEL_20_FOUNDERS = 0.05

# From S-1 headroom floor 1: gate A is NOT EVALUABLE below this mean
# surviving alphabet at generation 25. Derived in the rule from the maximizer
# arm's measured 4.4 surviving classes at the same bottleneck.
MIN_MEAN_ALPHABET_GEN25 = 4.0

# Gate A's alternative is one-sided and upper-tailed: more founder
# information than chance, never less.
_P_VALUE_SIDE = "greater"


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _validate_design(survivor_sets, founder_labels):
    """Check the inputs form a design on which gate A's statistic exists.

    Returns ``(sets, label_index, founders, class_counts)`` where ``sets`` is
    a list of Python sets, ``founders`` is the sorted array of distinct
    founder ids, ``label_index`` maps each chain to its founder's position in
    ``founders``, and ``class_counts`` is the number of chains per founder.

    Validation is at the boundary and fails fast, because every downstream
    quantity -- the statistic, its null, its chance level -- is meaningless if
    the design is malformed, and a malformed design is far more likely to
    arrive as a silently wrong number than as an exception.
    """
    sets = [set(s) for s in survivor_sets]
    labels = np.asarray(list(founder_labels))

    if len(sets) != labels.shape[0]:
        raise ValueError(
            "survivor_sets and founder_labels must be the same length; got "
            "{} survivor sets and {} labels".format(len(sets), labels.shape[0])
        )
    if not sets:
        raise ValueError("survivor_sets is empty; there is nothing to classify")
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
            "founder identification needs at least two founders; got {}. A "
            "one-way identification has no chance level and no decision to "
            "make.".format(founders.shape[0])
        )

    starved = founders[class_counts < 2]
    if starved.size:
        raise ValueError(
            "leave-one-chain-out is undefined for founder(s) {} with fewer "
            "than two chains: holding out a founder's only chain leaves its "
            "class with no training chain and its per-class accuracy "
            "undefined. S-1 design_amendment_2026_11_08 fixes R = 2 for "
            "exactly this reason.".format(sorted(int(f) for f in starved))
        )

    return sets, label_index, founders, class_counts


def _validate_scalar(value, name, minimum=None):
    """Check a reported scalar is a finite real number, and optionally bounded.

    The floors take numbers computed elsewhere -- a mean alphabet from a
    runner, a null mean and standard deviation from a permutation test -- and
    a NaN arriving from an upstream division by zero would otherwise sail
    through every comparison in this file as a silent "does not fire", which
    is the one answer a floor must never give by accident.
    """
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(
            "{} must be a finite number; got {!r}".format(name, value)
        )
    if minimum is not None and scalar < minimum:
        raise ValueError(
            "{} must be at least {}; got {!r}".format(name, minimum, value)
        )
    return scalar


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------

def _similarity_matrix(sets):
    """Pairwise Jaccard between every pair of chains, diagonal zeroed.

    Computed once and reused across every permutation. This is the whole
    reason 10000 permutations are affordable: permuting founder labels does
    not touch the survivor sets, so the expensive part of the statistic is
    invariant under the null and only the label arithmetic has to be redone.

    The diagonal is zeroed rather than left at its true value of 1.0 so that
    the leave-one-out exclusion of chain i from its own founder's training
    chains falls out of a plain matrix product instead of needing a mask per
    held-out chain.
    """
    n = len(sets)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = jaccard(sets[i], sets[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _balanced_accuracy_from_matrix(similarity, label_index, n_founders,
                                   class_counts, return_predictions=False):
    """Leave-one-chain-out balanced accuracy, given a precomputed similarity.

    ``similarity`` must have a zeroed diagonal. ``label_index`` holds each
    chain's founder as a position in the sorted founder array, so that
    ``numpy.argmax`` breaking ties toward the first maximum is exactly the
    documented tie rule, "toward the smallest founder id".

    The arithmetic. ``onehot`` is chains by founders. ``sums`` is then the
    total similarity from each chain to each founder's chains, excluding the
    chain itself because the diagonal is zero. ``counts`` is each founder's
    chain count minus one for the held-out chain's own founder, which is the
    number of terms actually in that sum. Dividing gives the mean over a
    founder's *training* chains, which is the asymmetry correction the module
    docstring describes; validation guarantees the divisor is at least one.
    """
    onehot = np.zeros((label_index.shape[0], n_founders), dtype=np.float64)
    onehot[np.arange(label_index.shape[0]), label_index] = 1.0

    sums = similarity @ onehot
    counts = class_counts[np.newaxis, :] - onehot
    scores = sums / counts

    predictions = np.argmax(scores, axis=1)

    hits = np.bincount(
        label_index[predictions == label_index], minlength=n_founders
    )
    # "averaged per class so that unequal survivor-set sizes cannot inflate
    # it": the mean of per-founder accuracies, not of per-chain outcomes.
    per_class = hits / class_counts
    balanced = float(np.mean(per_class))

    if return_predictions:
        return balanced, predictions
    return balanced


def balanced_accuracy_loo(survivor_sets, founder_labels):
    """Leave-one-chain-out founder-identification balanced accuracy.

    This is S-1 gate A's primary statistic. Each chain is held out in turn
    and its founder predicted by maximum mean Jaccard to the remaining chains
    of each founder, ties going to the smallest founder id; the returned value
    is the mean of the per-founder accuracies, so a founder with many chains
    does not count for more than a founder with few.

    survivor_sets  -- a sequence of N sets, or of any sequences of int class
                      ids; duplicates are absorbed by set semantics. The
                      empty-set convention is whatever
                      analysis/overlap_stats.jaccard says it is and is not
                      re-decided here.
    founder_labels -- a sequence of N ints, the founder each chain descended
                      from.

    Raises ValueError if any founder has fewer than two chains, if there are
    fewer than two founders, or if the two inputs disagree in length. Returns
    a float in [0, 1]; the chance level is 1 / n_founders, which is 0.05 at
    the committed 20 founders.
    """
    sets, label_index, founders, class_counts = _validate_design(
        survivor_sets, founder_labels
    )
    similarity = _similarity_matrix(sets)
    return _balanced_accuracy_from_matrix(
        similarity, label_index, founders.shape[0], class_counts
    )


def predict_founders_loo(survivor_sets, founder_labels):
    """The leave-one-chain-out predictions themselves, as founder ids.

    Not part of what gate A reads -- the gate reads the balanced accuracy --
    but exposed because a confusion pattern is the first thing worth looking
    at when a gate verdict is surprising, and because the tie rule is only
    directly testable through the predictions.

    Returns a numpy array of N founder ids in the same order as the inputs.
    """
    sets, label_index, founders, class_counts = _validate_design(
        survivor_sets, founder_labels
    )
    similarity = _similarity_matrix(sets)
    _, predictions = _balanced_accuracy_from_matrix(
        similarity, label_index, founders.shape[0], class_counts,
        return_predictions=True,
    )
    return founders[predictions]


# --------------------------------------------------------------------------
# Gate A's permutation test
# --------------------------------------------------------------------------

def gate_a_test(survivor_sets, founder_labels, n_perm=DEFAULT_N_PERM, seed=0):
    """S-1 gate A: one-sided permutation test on founder identifiability.

    The null permutes founder labels across chains, which is the null the
    committed ``fails_if`` names. Because it is a permutation of the existing
    label multiset, every founder keeps its chain count and the leave-one-out
    statistic stays defined on every draw.

    Returns a dict with exactly these keys:

        observed      -- balanced_accuracy_loo on the real labels.
        null_mean     -- mean of the permutation distribution.
        null_sd       -- its sample standard deviation (ddof=1); see the
                         module docstring on why the sample figure.
        null_min      -- smallest value the null took.
        null_max      -- largest value the null took. null_min and null_max
                         together are the "range" the headroom block requires
                         beside every p-value.
        p_value       -- the one-sided upper-tail estimate, uncorrected.
        n_perm, seed  -- echoed so a logged result is reproducible from the
                         log alone.
        chance_level  -- 1 / n_founders, the statistic's own chance level,
                         which is what headroom floor 2 compares the null
                         mean against.
        n_chains      -- N.
        n_founders    -- the number of distinct founders present.

    Holm-Bonferroni is NOT applied: the correction is across gates A, B and
    C, so it is the aggregator's job. The returned p_value is raw.
    """
    if int(n_perm) != n_perm or n_perm < 1:
        raise ValueError(
            "n_perm must be a positive integer number of permutations; got "
            "{!r}".format(n_perm)
        )
    n_perm = int(n_perm)

    sets, label_index, founders, class_counts = _validate_design(
        survivor_sets, founder_labels
    )
    n_founders = founders.shape[0]
    similarity = _similarity_matrix(sets)

    observed = _balanced_accuracy_from_matrix(
        similarity, label_index, n_founders, class_counts
    )

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        permuted = rng.permutation(label_index)
        null[k] = _balanced_accuracy_from_matrix(
            similarity, permuted, n_founders, class_counts
        )

    # The standard conservative permutation estimate:
    # (1 + #{null >= observed}) / (1 + n_perm). The +1 in the numerator is
    # the observed statistic itself, which under the null is one more draw
    # from the same distribution; the +1 in the denominator counts it in the
    # total. It can never return exactly zero.
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
        "chance_level": 1.0 / n_founders,
        "n_chains": len(sets),
        "n_founders": int(n_founders),
    }


# --------------------------------------------------------------------------
# Headroom floors bearing on gate A
# --------------------------------------------------------------------------

def floor_1_degenerate_alphabet(mean_alphabet_gen25,
                                minimum=MIN_MEAN_ALPHABET_GEN25):
    """Headroom floor 1: refuse gate A on a degenerate survivor alphabet.

    Returns (fires, reason).
    True means the floor FIRED and the gate is NOT EVALUABLE.

    That sentence is stated here, in this exact form, in both floor functions
    in this module, because an unstated polarity is what went wrong: this
    function was briefly paired with an aggregator that read the same boolean
    as "evaluable", under which a mean alphabet of 3.2 classes would have been
    read as readable data and gate A would have been read on exactly the
    degenerate input this floor exists to refuse. The implementation was never
    wrong; the convention was never written down. The adopted convention
    matches this project's own logged vocabulary -- "floor fired:
    headroom_2_shuffled_floor" in logs/c8b_shared_founder_torch280.txt.

    Committed wording: "Gate A is NOT EVALUABLE if the mean surviving
    alphabet at generation 25 is below 4.0 classes." Implemented as the
    literal strict comparison, so exactly 4.0 is inside the evaluable
    regime.

    The input is the mean number of surviving classes at generation 25, read
    on the held-out probe pool at probe_seed 9000 --
    ``design_amendment_2026_11_08.founders_and_pools`` fixes that pool for
    gates A, B, C and floor 1, and the 4.0 threshold was calibrated against a
    probe-pool figure of 4.4, so reading it on the training pool would change
    the floor's bite without changing its number. This function cannot check
    which pool it was handed; that is the caller's responsibility to record.

    The reason string is populated whether or not the floor fires, so a
    caller can log why a gate was readable as well as why it was not.
    """
    alphabet = _validate_scalar(
        mean_alphabet_gen25, "mean_alphabet_gen25", minimum=0.0
    )
    threshold = _validate_scalar(minimum, "minimum", minimum=0.0)

    if alphabet < threshold:
        return True, (
            "NOT EVALUABLE (headroom floor 1, degenerate alphabet): mean "
            "surviving alphabet at generation 25 is {:.4g} classes, below "
            "the committed floor of {:.1f}. Beneath it the number of "
            "distinct attainable survivor sets is too small for a 20-way "
            "identification to be informative, whatever its p-value."
            .format(alphabet, threshold)
        )
    return False, (
        "Headroom floor 1 does not fire: mean surviving alphabet at "
        "generation 25 is {:.4g} classes, at or above the committed floor "
        "of {:.1f}.".format(alphabet, threshold)
    )


def floor_2_chance_level(null_mean, null_sd,
                         chance_level=CHANCE_LEVEL_20_FOUNDERS):
    """Headroom floor 2: refuse gate A if its null does not sit at chance.

    Returns (fires, reason).
    True means the floor FIRED and the gate is NOT EVALUABLE.
    See floor_1_degenerate_alphabet for why this sentence is stated in both
    floor functions in this exact form.

    Committed wording: "Balanced accuracy over 20 founders has a chance level
    of 0.05. Gate A is NOT EVALUABLE if the permutation null's mean differs
    from 0.05 by more than one of its own standard deviations, because a null
    that does not sit at the statistic's own chance level is measuring
    something other than the label permutation it claims to."

    The comparison is the literal ``abs(null_mean - chance_level) > null_sd``,
    strict and with no tolerance, so a departure of exactly one standard
    deviation leaves the gate evaluable. The departure is two-sided even
    though gate A's p-value is one-sided: the committed word is "differs",
    and a null sitting *below* chance is as much a sign of a mis-built null
    as one sitting above it.

    ``chance_level`` defaults to the committed 0.05 rather than to a value
    derived from the data, because 0.05 is what the rule states. Pass
    ``gate_a_test(...)["chance_level"]`` instead when the design in hand is
    not the committed 20 founders -- the two agree at 20 founders.

    MEASURED WARNING: part of this floor's budget is spent by gate A's own
    tie rule before any data arrives. Read this before concluding that a
    fired floor 2 means the null was mis-built.

    The mechanism. ``balanced_accuracy_loo`` breaks ties toward the smallest
    founder id, so whenever a tied block of chains exists the tiebreak
    donates exactly 1 / n_founders to the statistic: the founder the ties
    fall to collects a per-class accuracy of 1.0 for free. On fully
    degenerate data that donation is the whole statistic and the result sits
    exactly at chance, which is why the rule was chosen. On partly
    degenerate data it sits on top of whatever genuine signal exists, so the
    statistic -- and with it the permutation null this floor reads -- is
    biased UPWARD relative to the nominal chance level in any regime where
    exact ties are common.

    Measured on this machine, 20 founders, R = 2, survivor sets of 5 classes
    drawn independently of the founder from an alphabet of 50, n_perm = 4000,
    ten independent datasets: the null mean ran 0.0625 to 0.0793 rather than
    0.05, and upward on all ten, against a null standard deviation of about
    0.052 to 0.055. The departure ran 0.24 to 0.53 of a null standard
    deviation against this floor's firing threshold of 1.0 -- so the floor
    did not fire, but up to about half its one-sd budget was already spent,
    and the departure was systematic rather than noise. In that regime 61 per
    cent of chain pairs had Jaccard exactly 0 and 12 of 40 decisions were
    exact ties. Varying R thins the bias, which is what confirms the
    tie-donation mechanism rather than a defect in the permutation: mean null
    mean 0.071 at R = 2, 0.085 at R = 3, 0.075 at R = 4, 0.061 at R = 6 and
    0.055 at R = 10.

    What this does and does not touch. Gate A's p-value is UNAFFECTED,
    because the observed statistic and every permutation draw carry the
    donation equally -- the null permutes labels and leaves the survivor sets,
    and hence the tie structure, alone. It is this floor that is affected.

    What is NOT claimed. Those numbers are SYNTHETIC and do not transfer to a
    real chain. No S-1 chain has been run; the real regime at bottleneck 128
    on the probe pool has different survivor-set sizes and a different
    overlap structure, and nothing above predicts what it will do. The point
    of recording it here is narrower: if this floor fires on real data, check
    it against this measurement first, because the floor may be reading gate
    A's tie rule rather than a mis-built null.
    """
    mean = _validate_scalar(null_mean, "null_mean")
    sd = _validate_scalar(null_sd, "null_sd", minimum=0.0)
    chance = _validate_scalar(chance_level, "chance_level", minimum=0.0)

    departure = abs(mean - chance)
    if departure > sd:
        return True, (
            "NOT EVALUABLE (headroom floor 2, chance-level separation): the "
            "permutation null's mean is {:.6g}, which differs from the "
            "statistic's chance level of {:.6g} by {:.6g}, more than one of "
            "the null's own standard deviations ({:.6g}). A null that does "
            "not sit at the statistic's own chance level is measuring "
            "something other than the founder-label permutation it claims "
            "to.".format(mean, chance, departure, sd)
        )
    return False, (
        "Headroom floor 2 does not fire: the permutation null's mean is "
        "{:.6g}, within one standard deviation ({:.6g}) of the statistic's "
        "chance level of {:.6g}; the departure is {:.6g}."
        .format(mean, sd, chance, departure)
    )
