"""
S-1 gate C and headroom floor 3: is the prior scale an attractor, and is the
comparison even in one scaling regime?

Both things in this module are implementations of text already committed in
``gates/thresholds.json`` under the key ``S-1``. Nothing here chooses a
threshold, a grid or a statistic; where the committed text left a choice to
the code, the choice is named in a docstring below and listed in this module's
"Ambiguities resolved here" section rather than made silently. That section
exists because the 2026-11-08 amendment to S-1 was written precisely to stop
the runner from deciding what the rule had not, and a statistic file is as
capable of deciding silently as a runner is.

Gate C, verbatim from the committed entry:

    metric:    "within-sigma cross-founder mean pairwise Jaccard minus
                across-sigma cross-founder mean pairwise Jaccard"
    condition: "the prior scale does not define an attractor: chains at the
                same sigma are no more alike than chains at different sigma"
    fails_if:  "one-sided permutation p < 0.05 after Holm-Bonferroni for
                within-sigma exceeding across-sigma, permuting the sigma label
                over 10000 permutations"

The load-bearing word is **cross-founder**. Pairs are taken between chains of
DIFFERENT founders, so same-founder pairs are excluded from both terms. That
exclusion is what makes the statistic a statement about sigma instead of a
statement about founder identity: gate A and gate B already ask whether the
founder is recoverable, and if same-founder pairs were left in, their high
mutual similarity would enter the within-sigma term whenever two chains of one
founder happened to share a sigma and the across-sigma term whenever they did
not. The gate would then fire on founder dependence that gates A and B are the
proper instruments for, and the committed distinction in the combination block
-- "A and B ask whether the founder is forgotten; C asks whether the prior is
found" -- would collapse. ``tests/test_sigma_attractor.py`` contains a fixture
whose statistic has the opposite SIGN under the two readings, so the exclusion
is pinned by a test that no tolerance can satisfy by accident.

Headroom floor 3, verbatim from the same entry:

    "The mean pairwise Jaccard is a U-statistic of order two, so its null
    standard deviation scales as 1/sqrt(n) when the first-order projection has
    positive variance and as 1/n when it does not. A uniform-over-classes null
    is degenerate in exactly that sense under class-exchangeability at fixed
    set sizes, so comparing a non-degenerate null against a degenerate one
    compares two different scaling regimes. Gate B and gate C must therefore
    report, for each null they use, the number of sets n and an estimate of
    that first-order variance, and are NOT EVALUABLE if the two nulls being
    compared fall in different regimes."

Floor 3 is the direct consequence of C-4O, whose separation floor was blind to
exactly this, and the committed text says in so many words that C-4O's refusal
is not evidence either way about S-1.

Multiplicity is NOT applied here. The committed correction is
Holm-Bonferroni across gates A, B and C, which is a decision about three
p-values at once; ``gate_c_test`` returns gate C's raw one-sided p-value and
the caller -- ``controls/s1_aggregate.py`` -- applies Holm across the three.
A function holding one p-value cannot correct for the other two.

NumPy only. Nothing here imports torch, reads a log, or runs a chain.

Ambiguities resolved here, each one a place where the committed rule does not
reach and the code therefore had to:

1. **The permutation is unstratified.** "Permuting the sigma label" is read as
   a free permutation of the sigma labels over all chains with the founder
   labels held fixed. See ``_permuted_statistics``.
2. **Jaccard's empty-set convention is inherited, not redefined.** This module
   calls ``analysis.overlap_stats.jaccard``, under which two empty survivor
   sets have similarity 1.0. See ``_pairwise_jaccard_matrix``.
3. **An empty within-sigma or across-sigma term is an error, not a NaN.** See
   ``within_minus_across`` and ``_permuted_statistics``.
4. **"Positive variance" is decided by which term dominates the U-statistic
   variance, not by a significance test.** See ``scaling_regime``, whose
   docstring derives the criterion and states what a different one would
   change.
5. **The first-order variance estimate is reported unclipped and may be
   negative.** See ``first_order_variance``.
"""

import itertools

import numpy as np

from analysis.overlap_stats import jaccard

# The committed entry fixes n_perm at 10000 for every S-1 gate
# (fixed_in_advance.n_perm), so the default here is the committed value and not
# a convenience: a caller who omits it gets the preregistered test.
DEFAULT_N_PERM = 10_000

# The two scaling regimes floor 3 names. "non-degenerate" is the 1/sqrt(n)
# regime, in which the first-order projection has positive variance;
# "degenerate" is the 1/n regime, in which it does not.
NON_DEGENERATE = "non-degenerate"
DEGENERATE = "degenerate"
_REGIMES = (NON_DEGENERATE, DEGENERATE)

# The minimum number of sets for which the first-order variance estimator is
# defined. Its bias correction divides by n - 2, and at n = 2 there is a single
# pair and so no between-set variation to estimate from at all.
_MIN_SETS_FOR_ZETA_1 = 3


# --------------------------------------------------------------------------
# Pair bookkeeping
# --------------------------------------------------------------------------

def _validated_labels(survivor_sets, founder_labels, sigma_labels):
    """Coerce the three parallel inputs and check they describe one collection.

    Validated at the boundary rather than trusted, because the three arrays
    arrive from a runner that builds them in three separate loops and a silent
    off-by-one between them would misattribute chains to founders without ever
    raising. The founder and sigma labels are kept as object arrays of whatever
    the caller passed, so that string sigma names like "4x_sigma_match" and
    integer codes both work; only equality between labels is ever used.
    """
    sets = list(survivor_sets)
    founders = np.asarray(list(founder_labels), dtype=object)
    sigmas = np.asarray(list(sigma_labels), dtype=object)

    if not (len(sets) == founders.shape[0] == sigmas.shape[0]):
        raise ValueError(
            "survivor_sets, founder_labels and sigma_labels must be the same "
            f"length, got {len(sets)}, {founders.shape[0]} and "
            f"{sigmas.shape[0]}"
        )
    if len(sets) < 2:
        raise ValueError(
            f"need at least two chains to form a pair, got {len(sets)}"
        )
    return sets, founders, sigmas


def _pairwise_jaccard_matrix(sets):
    """The full n-by-n matrix of pairwise Jaccard similarities.

    Computed once and reused across every permutation. This is what makes the
    committed 10000 permutations cheap: permuting the sigma label changes which
    pairs fall in which term but never changes any pair's similarity, so the
    expensive part of the statistic is permutation-invariant and belongs
    outside the loop.

    ``analysis.overlap_stats.jaccard`` is called rather than reimplemented, so
    that gate C and gate B -- which the committed entry routes through
    ``analysis/overlap_stats.py`` by name -- cannot drift apart on the
    empty-set convention. Under that convention two empty survivor sets have
    similarity 1.0, which matters here because a chain can in principle end a
    generation with no surviving classes; inheriting the convention keeps that
    case defined the same way in both gates instead of twice.
    """
    n = len(sets)
    matrix = np.zeros((n, n), dtype=float)
    for i, j in itertools.combinations(range(n), 2):
        value = jaccard(sets[i], sets[j])
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def _cross_founder_pairs(founders):
    """Row and column indices of every pair of chains with DIFFERENT founders.

    Returned as two parallel index arrays so that the within/across split can
    be recomputed per permutation with one vectorized comparison. Because the
    founder labels are held fixed by the committed null, this index set is
    computed once and is the same for the observed statistic and for every
    permuted one -- which is the mechanical reason the permutation cannot
    disturb the cross-founder pairing structure.
    """
    n = founders.shape[0]
    pairs = [
        (i, j)
        for i, j in itertools.combinations(range(n), 2)
        if founders[i] != founders[j]
    ]
    if not pairs:
        raise ValueError(
            "no cross-founder pairs: every chain has the same founder, so the "
            "cross-founder mean pairwise Jaccard gate C names does not exist"
        )
    pi = np.fromiter((i for i, _ in pairs), dtype=int, count=len(pairs))
    pj = np.fromiter((j for _, j in pairs), dtype=int, count=len(pairs))
    return pi, pj


def _split_means(values, same_sigma):
    """Mean of `values` where `same_sigma`, and mean where not.

    Both terms are required to be non-empty. An empty term is raised rather
    than returned as a NaN because the committed metric is a difference of two
    means: if one of them has no pairs, gate C has not produced a small
    statistic, it has produced no statistic, and a NaN propagating into a
    p-value comparison would read as a pass. This is the third ambiguity in
    the module docstring: the rule says nothing about the case, so the code
    refuses it loudly instead of choosing a value for it.
    """
    n_within = int(same_sigma.sum())
    n_across = int(same_sigma.shape[0] - n_within)
    if n_within == 0:
        raise ValueError(
            "no within-sigma cross-founder pairs: every cross-founder pair "
            "spans two different sigma values, so gate C's first term is empty"
        )
    if n_across == 0:
        raise ValueError(
            "no across-sigma cross-founder pairs: every cross-founder pair "
            "shares a sigma value, so gate C's second term is empty"
        )
    weights = same_sigma.astype(float)
    within_sum = float(values @ weights)
    total = float(values.sum())
    within_mean = within_sum / n_within
    across_mean = (total - within_sum) / n_across
    return within_mean, across_mean, n_within, n_across


# --------------------------------------------------------------------------
# Gate C
# --------------------------------------------------------------------------

def within_minus_across(survivor_sets, founder_labels, sigma_labels):
    """Gate C's committed statistic, on cross-founder pairs only.

    The within-sigma cross-founder mean pairwise Jaccard minus the across-sigma
    cross-founder mean pairwise Jaccard. A pair counts only if the two chains
    have different founders; among those, it counts toward the first term if
    the two chains share a sigma value and toward the second if they do not.

    Positive means chains at the same prior scale are more alike than chains at
    different prior scales, which is the direction gate C fires on: under the
    committed reading a significant positive value means sigma defines an
    attractor and gate C FAILS, which the entry records as positive evidence
    that the sampler is converging to a prior rather than wandering. A value at
    or below zero is the gate PASSING, and the entry is explicit that passing
    here is the outcome that most threatens the arm's interpretation.

    survivor_sets/founder_labels/sigma_labels: three parallel sequences of the
    same length, one entry per chain. A survivor set is any iterable of class
    labels and is treated as a set, so duplicates and ordering are absorbed.
    Labels are compared only for equality, so integers and strings both work.
    """
    sets, founders, sigmas = _validated_labels(
        survivor_sets, founder_labels, sigma_labels
    )
    matrix = _pairwise_jaccard_matrix(sets)
    pi, pj = _cross_founder_pairs(founders)
    values = matrix[pi, pj]
    same_sigma = sigmas[pi] == sigmas[pj]
    within_mean, across_mean, _, _ = _split_means(values, same_sigma)
    return float(within_mean - across_mean)


def _permuted_statistics(values, pi, pj, sigmas, n_perm, rng):
    """The committed null: permute the sigma label, hold the founders fixed.

    This is the first ambiguity in the module docstring. The committed text
    says "permuting the sigma label over 10000 permutations" and says nothing
    about stratification, so the free permutation it literally names is what is
    implemented: the multiset of sigma labels is shuffled over all chains, and
    the founder labels are not touched.

    Two consequences, stated here rather than left to be discovered.

    First, the cross-founder pairing structure is preserved by construction and
    not by care. ``pi`` and ``pj`` are derived from the founder labels, which
    the permutation never moves, so every permuted statistic is computed over
    the identical set of cross-founder pairs as the observed one. Only the
    within/across partition of that fixed set changes. A permutation that
    moved founders as well would be testing a different hypothesis -- that is
    gate A's null, not gate C's.

    Second, a free permutation does not preserve the design's balance. S-1 runs
    R = 2 chains per founder per cell, so each founder contributes two chains
    at each of the two sigma values; a free permutation can leave a founder
    with three chains at one sigma and one at the other. That is a statement
    about the null's conditioning, not about its validity: under gate C's null
    hypothesis the sigma labels carry no information about the survivor sets,
    so they are exchangeable across chains and the free permutation is a valid
    null. A within-founder stratified permutation would be a different,
    conditional null that also tests the same hypothesis, and it would be a
    defensible choice -- but it is not the one the committed text names, and
    choosing it here would be the code deciding what the rule left open. If the
    stratified null is wanted, it should be written into the rule under a new
    key first.
    """
    sigmas = np.asarray(sigmas, dtype=object)
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        permuted = rng.permutation(sigmas)
        same_sigma = permuted[pi] == permuted[pj]
        try:
            within_mean, across_mean, _, _ = _split_means(values, same_sigma)
        except ValueError as exc:
            raise ValueError(
                f"permutation {k} left one of gate C's two terms empty, so the "
                "null is not computable on this design: " + str(exc)
            ) from exc
        null[k] = within_mean - across_mean
    return null


def gate_c_test(survivor_sets, founder_labels, sigma_labels,
                n_perm=DEFAULT_N_PERM, seed=0):
    """Gate C's one-sided permutation test, with its null's headroom beside it.

    Returns a dict with exactly these keys:

        observed        gate C's statistic on the real labels.
        null_mean       mean of the permutation null.
        null_sd         standard deviation of the permutation null, ddof = 0:
                        a description of the n_perm draws actually generated,
                        not an inference about a population behind them.
        null_min        smallest null draw.
        null_max        largest null draw.
        p_value         the one-sided p-value, see below.
        n_perm          the number of permutations used.
        seed            the seed the null was generated from.
        n_sets          the number of chains.
        n_within_pairs  cross-founder pairs sharing a sigma value.
        n_across_pairs  cross-founder pairs spanning two sigma values.

    The null's mean, standard deviation and range are returned because the
    committed headroom block requires them beside every p-value, and because
    ``analysis/overlap_stats.py``'s own docstring records two gates in this
    project that shipped as p-values with no room to move. ``n_within_pairs``
    and ``n_across_pairs`` are returned for the same reason one layer down: a
    gate C computed over eight pairs against eight pairs is a different object
    from one computed over three thousand, and the aggregator cannot tell
    which it is holding from a p-value.

    The p-value is the conservative permutation estimate

        p = (1 + #{null >= observed}) / (1 + n_perm)

    one-sided for within-sigma EXCEEDING across-sigma, which is the direction
    the committed ``fails_if`` names. The +1 on both numerator and denominator
    counts the observed labelling as one of the equally likely labellings under
    the null: it keeps the estimate from ever reading as exactly zero, it is
    valid rather than merely cautious, and it is the convention already used in
    ``analysis/overlap_stats.py``, ``controls/c4_timing.py`` and
    ``analysis/seed_stability_stats.py``. At the committed n_perm = 10000 the
    smallest attainable p-value is 1 / 10001, comfortably below the 0.05 the
    gate is read against even after Holm-Bonferroni over three gates.

    Holm-Bonferroni is NOT applied here. The committed correction runs across
    gates A, B and C, so it belongs to whatever holds all three p-values; this
    function returns gate C's raw one-sided p-value for that caller to correct.

    seed is threaded through an explicit ``numpy.random.default_rng(seed)`` and
    returned in the result, so that a logged verdict carries the seed that
    produced its null and the null can be regenerated from the log alone.
    """
    if n_perm < 1:
        raise ValueError(f"n_perm must be positive, got {n_perm}")

    sets, founders, sigmas = _validated_labels(
        survivor_sets, founder_labels, sigma_labels
    )
    matrix = _pairwise_jaccard_matrix(sets)
    pi, pj = _cross_founder_pairs(founders)
    values = matrix[pi, pj]

    same_sigma = sigmas[pi] == sigmas[pj]
    within_mean, across_mean, n_within, n_across = _split_means(values, same_sigma)
    observed = float(within_mean - across_mean)

    rng = np.random.default_rng(seed)
    null = _permuted_statistics(values, pi, pj, sigmas, n_perm, rng)

    # One-sided for within-sigma exceeding across-sigma: a null draw is
    # "extreme or equal" when it is at least as large as the observed
    # statistic. The +1 / +1 smoothing is described in this docstring.
    n_at_least_observed = int(np.count_nonzero(null >= observed))
    p_value = (1 + n_at_least_observed) / (1 + n_perm)

    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "null_min": float(np.min(null)),
        "null_max": float(np.max(null)),
        "p_value": float(p_value),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "n_sets": len(sets),
        "n_within_pairs": int(n_within),
        "n_across_pairs": int(n_across),
    }


# --------------------------------------------------------------------------
# Headroom floor 3
# --------------------------------------------------------------------------

def _variance_components(sets):
    """Estimates of zeta_1 and zeta_2 for the mean pairwise Jaccard.

    Write the statistic as a U-statistic of order two,

        U = (1 / C(n, 2)) * sum_{i < j} h(X_i, X_j),   h = Jaccard,

    and let theta = E h. Its Hoeffding decomposition has two variance
    components:

        h_1(x) = E[h(x, X)]        the first-order (Hajek) projection
        zeta_1 = Var(h_1(X))       what floor 3 calls the first-order variance
        zeta_2 = Var(h(X_1, X_2))  the full kernel variance

    and the exact variance of U is

        Var(U) = [ 4 (n - 2) zeta_1 + 2 zeta_2 ] / [ n (n - 1) ].

    That formula is the whole content of floor 3's "1/sqrt(n) versus 1/n": when
    zeta_1 > 0 the first term dominates and Var(U) ~ 4 zeta_1 / n, so the
    standard deviation falls as 1/sqrt(n); when zeta_1 = 0 the first term
    vanishes identically and Var(U) ~ 2 zeta_2 / n^2, so it falls as 1/n.

    zeta_2 is estimated by the sample variance of the C(n, 2) pairwise Jaccards.

    zeta_1 needs a bias correction, which is the reason this is a helper and
    not one line. The natural plug-in is the sample variance of the per-set
    averages

        g_i = (1 / (n - 1)) * sum_{j != i} h(X_i, X_j),

    but g_i is a noisy estimate of h_1(X_i) rather than h_1(X_i) itself.
    Conditioning on X_i, the n - 1 kernel values entering g_i are independent,
    so

        Var(g_i) = zeta_1 + (zeta_2 - zeta_1) / (n - 1),

    and the uncorrected sample variance of the g_i therefore overstates zeta_1
    by roughly zeta_2 / (n - 1) -- which is exactly the amount that would make
    a truly degenerate null look non-degenerate, since under degeneracy that
    term is the only thing there is. Inverting the relation gives the estimator
    used here,

        zeta_1_hat = [ (n - 1) * S_g^2 - zeta_2_hat ] / (n - 2),

    where S_g^2 is the sample variance of the g_i with ddof = 1. This is a
    leading-order correction: the g_i are mutually dependent, so S_g^2 is not
    exactly unbiased for Var(g_i), and the correction removes the O(1/n) bias
    term without claiming to remove the O(1/n^2) ones. That is sufficient for
    a regime classification, which only needs to know which of two terms
    dominates, and it is not sufficient for quoting zeta_1 as a measured
    quantity. It is not quoted as one anywhere.
    """
    sets = list(sets)
    n = len(sets)
    if n < _MIN_SETS_FOR_ZETA_1:
        raise ValueError(
            "the first-order variance estimate needs at least three survivor "
            f"sets, got {n}: its bias correction divides by n - 2, and at "
            "n = 2 there is one pair and no between-set variation to estimate"
        )

    matrix = _pairwise_jaccard_matrix(sets)
    upper = np.triu_indices(n, k=1)
    pair_values = matrix[upper]

    # Row sums of the symmetric matrix have a zero diagonal, so dividing by
    # n - 1 gives each set's mean Jaccard against all the others.
    per_set_means = matrix.sum(axis=1) / (n - 1)

    zeta_2 = float(np.var(pair_values, ddof=1))
    s_g_squared = float(np.var(per_set_means, ddof=1))
    zeta_1 = ((n - 1) * s_g_squared - zeta_2) / (n - 2)
    return float(zeta_1), zeta_2, n


def first_order_variance(survivor_sets):
    """The zeta_1 estimate for the mean pairwise Jaccard over these sets.

    This is the quantity floor 3 requires be reported for every null a gate
    uses, beside that null's number of sets.

    The estimate is returned UNCLIPPED and can be negative. That is deliberate
    and is the fifth ambiguity named in the module docstring. zeta_1 is a
    variance and so is non-negative in truth, but its bias-corrected estimator
    is a difference of two positive quantities and fluctuates around zero when
    the truth is zero -- going negative about half the time. Clipping at zero
    would throw away exactly the signal floor 3 is asking for, because a
    negative estimate is the clean finite-sample signature of a degenerate
    null, and it would turn "degenerate" into "indistinguishable from a null
    with a vanishingly small positive zeta_1". A reported negative value is
    read as "zero, within the estimator's noise", which is what it is.

    See ``_variance_components`` for the derivation of the estimator and for
    the order of the bias it does and does not remove.
    """
    zeta_1, _, _ = _variance_components(survivor_sets)
    return zeta_1


def second_order_variance(survivor_sets):
    """The zeta_2 estimate: the sample variance of the pairwise Jaccards.

    Not named in the committed rule and not part of the interface the
    aggregator is written against, but exported rather than kept private
    because it is the denominator of the regime criterion in
    ``scaling_regime``. A classifier whose comparison scale cannot be inspected
    from outside is a threshold hidden in a function, and floor 3 exists
    because a floor's bite was invisible once already.
    """
    _, zeta_2, _ = _variance_components(survivor_sets)
    return zeta_2


def first_order_variance_share(survivor_sets):
    """The fraction of Var(U) carried by the first-order term, in [0, 1].

    Using the exact variance decomposition in ``_variance_components``,

        share = 4 (n - 2) zeta_1 / [ 4 (n - 2) zeta_1 + 2 zeta_2 ]
              = 2 (n - 2) zeta_1 / [ 2 (n - 2) zeta_1 + zeta_2 ].

    This is the single number ``scaling_regime`` thresholds, exported so that
    the margin by which a null was classified can be reported rather than only
    the verdict. Clamped to [0, 1] for reporting, since a negative zeta_1
    estimate would otherwise give a negative share; the classification itself
    uses the unclamped comparison, so the clamp cannot change a verdict.
    """
    zeta_1, zeta_2, n = _variance_components(survivor_sets)
    first = 2 * (n - 2) * zeta_1
    denominator = first + zeta_2
    if denominator <= 0:
        return 0.0
    return float(min(max(first / denominator, 0.0), 1.0))


def scaling_regime(survivor_sets):
    """Classify a null into floor 3's 1/sqrt(n) or 1/n regime.

    Returns a dict with exactly these keys:

        n        the number of sets in the null.
        zeta_1   the first-order variance estimate, unclipped.
        regime   "non-degenerate" or "degenerate".

    n and zeta_1 are the two things floor 3 requires be reported for each null;
    regime is the classification the floor's refusal is read off.

    MEASURED LIMITATION, 2026-11-08. This classifier misclassifies a genuinely
    degenerate null as non-degenerate at a rate of roughly one trial in
    fifteen, and that rate does not reliably fall as n grows. Measured by
    drawing exactly the case the rule names as degenerate -- uniform over 50
    classes at fixed set size, class-exchangeable -- and counting how often
    the regime came back "non-degenerate", 200 trials per cell:

        n = 10, size  5 : 7.0%        n = 10, size 12 : 3.5%
        n = 20, size  5 : 8.5%        n = 20, size 12 : 7.0%
        n = 40, size  5 : 8.0%        n = 40, size 12 : 7.5%
        n = 80, size  5 : 6.0%        n = 80, size 12 : 2.0%

    The direction of the error is the unsafe one. A false "non-degenerate"
    makes two nulls look like they share a regime when one of them does not,
    so floor 3 permits a comparison it was written to refuse; the opposite
    error would merely cost an evaluable gate. This is NOT the bias in
    zeta_1_hat, which the estimator corrects to leading order: it is the
    residual variance of the corrected estimate against the criterion's
    threshold, and no bias correction removes it.

    Consequences, stated rather than left to inference. This rate is tolerable
    for a classification used once per gate and reported beside its inputs, and
    it is NOT tolerable as a silent precondition. Floor 3 therefore reports n
    and zeta_1 alongside the regime, as the rule requires, so a reader can see
    how close a call was rather than only its verdict. And if floor 3 ever
    refuses a real gate, this measurement is the first thing to check it
    against: at these rates a single refusal is weak evidence on its own.

    Synthetic, and not transferable. These are uniform draws over a 50-class
    alphabet at fixed cardinality. Real S-1 survivor sets have a different size
    distribution and real between-set structure, no S-1 chain has been run, and
    nothing here predicts the rate on real data.

    THE CRITERION, and why it is this one. Floor 3 says the null's standard
    deviation scales as 1/sqrt(n) "when the first-order projection has positive
    variance" and as 1/n "when it does not". A bare ``zeta_1_hat > 0`` test
    cannot implement that: the estimate is a continuous random quantity and is
    never exactly zero, so the test would classify roughly half of all truly
    degenerate nulls as non-degenerate and would be answering a question about
    rounding. A criterion is needed and the rule does not state one, so this is
    the fourth ambiguity named in the module docstring.

    The criterion used is:

        non-degenerate  iff  zeta_2_hat > 0  and  2 (n - 2) zeta_1_hat >= zeta_2_hat

    which is exactly the statement that the first-order term of the exact
    U-statistic variance is at least as large as the second-order term, since

        Var(U) = [ 4 (n - 2) zeta_1 + 2 zeta_2 ] / [ n (n - 1) ].

    Equivalently, ``first_order_variance_share`` is at least one half.

    JUSTIFICATION. The two regimes floor 3 names are the two terms of that
    decomposition, and the point where they contribute equally is the only
    place the rule's own text locates a boundary. The criterion therefore
    introduces no constant that was not already in the rule: the threshold is
    the equality of the two named terms, and n enters through the (n - 2)
    factor rather than through a sample-size rule of thumb. It also fails in
    the right direction on the case the rule names explicitly. For a
    uniform-over-classes null at fixed set sizes, zeta_1 is exactly zero by
    class-exchangeability, S_g^2 is of order zeta_2 / (n - 1), and the
    corrected zeta_1_hat is of order zeta_2 / n^(3/2); then
    2 (n - 2) zeta_1_hat is of order zeta_2 / sqrt(n), which is below zeta_2
    for any n worth running. For a null with genuine between-set structure
    zeta_1 is a fixed fraction of zeta_2 and 2 (n - 2) zeta_1 exceeds zeta_2
    already at modest n. S-1 reads gate C on 20 founders times 2 chains times
    2 sigma values, so n = 80 and 2 (n - 2) = 156: both separations are wide
    at the size this gate is actually run at.

    WHAT A DIFFERENT THRESHOLD WOULD CHANGE. Requiring a share above one half
    -- say 0.9, i.e. 18 (n - 2) zeta_1_hat >= zeta_2_hat -- makes the
    classifier readier to call a null degenerate, so it makes floor 3 readier
    to refuse gate B and gate C as NOT EVALUABLE: conservative in the floor's
    own direction, at the cost of refusing comparisons that were fine.
    Requiring a share above 0.1 does the reverse and lets a weakly degenerate
    null be compared against a non-degenerate one, which is the C-4O failure
    floor 3 was written to prevent. The verdicts that would move are only those
    whose share lands between the old threshold and the new one; a null whose
    share is near 0 or near 1 is classified the same way by any threshold in
    the open interval, and both of the cases in this project's tests are of
    that kind. Because the share is exported by
    ``first_order_variance_share``, a reader can see which case a given null
    was, rather than having to trust that it was not a near-miss.

    The zero-variance case is classified degenerate. If every pairwise Jaccard
    is identical then zeta_2_hat is zero, the kernel is constant, zeta_1 is
    zero in truth, and the comparison 0 >= 0 would otherwise return
    "non-degenerate" on a null that has no first-order variation at all. The
    explicit ``zeta_2_hat > 0`` conjunct is what keeps that case on the correct
    side, and such a null has no usable scaling behaviour in either regime.
    """
    zeta_1, zeta_2, n = _variance_components(survivor_sets)
    is_non_degenerate = zeta_2 > 0 and 2 * (n - 2) * zeta_1 >= zeta_2
    return {
        "n": n,
        "zeta_1": zeta_1,
        "regime": NON_DEGENERATE if is_non_degenerate else DEGENERATE,
    }


def floor_3_regime_mismatch(regime_a, regime_b):
    """Does headroom floor 3 fire on a gate comparing these two nulls?

    **True means the floor fired and the gate is NOT EVALUABLE.** That is the
    polarity every floor predicate in this project uses -- see
    ``logs/c8b_shared_founder_torch280.txt``, which records a fired floor as
    "floor fired: headroom_2_shuffled_floor" -- so a bare boolean read off any
    of them means the same thing without the reader having to recall which
    function they are holding.

    Returns ``(fires, reason)``:

        (True,  reason)  the two nulls are in DIFFERENT scaling regimes, so
                         floor 3 fires and the gate is NOT EVALUABLE.
        (False, reason)  the two nulls are in the SAME regime, so floor 3 does
                         not fire and the gate may be read.

    The reason names both regimes in either case, so that a verdict can be
    written up from the return value alone rather than from the inputs.

    The committed text is that gate B and gate C "are NOT EVALUABLE if the two
    nulls being compared fall in different regimes", so a mismatch is a refusal
    to read the gate at all and not a caveat attached to its p-value.

    Note what the committed text does NOT say. A matched pair of DEGENERATE
    nulls does NOT fire the floor: floor 3 is about comparing across two
    different scaling regimes, not about degeneracy being disqualifying in
    itself. Two degenerate nulls share a 1/n scaling and are comparable to each
    other. Reading that clause as "degenerate nulls may not be used" would be a
    stricter rule than the one committed, and tightening a preregistered floor
    in code is the same category of error as loosening it.

    An unrecognised regime name raises instead of being treated as either a
    match or a mismatch, since the only two regimes the rule defines are the
    ones named here and anything else is a caller bug that would otherwise
    decide a gate silently.
    """
    for name, value in (("regime_a", regime_a), ("regime_b", regime_b)):
        if value not in _REGIMES:
            raise ValueError(
                f"{name} must be one of the two regime names floor 3 defines, "
                f"{_REGIMES}, got {value!r}"
            )

    if regime_a == regime_b:
        return False, (
            f"Headroom floor 3 does not fire: the first null is {regime_a} and "
            f"the second is {regime_b}, so the two null standard deviations "
            "scale the same way in n and the comparison is within one regime. "
            "The gate may be read."
        )
    return True, (
        f"NOT EVALUABLE under headroom floor 3: the first null is {regime_a} "
        f"and the second is {regime_b}, so their null standard deviations "
        "scale differently in n (1/sqrt(n) for non-degenerate, 1/n for "
        "degenerate) and the comparison spans two scaling regimes."
    )


def floor_3_regime_match(regime_a, regime_b):
    """DEPRECATED spelling of the floor 3 predicate, with INVERTED polarity.

    **True means the two nulls match and the gate MAY be read** -- the opposite
    of ``floor_3_regime_mismatch``, which is the one to call. Kept as a thin
    alias rather than deleted because a caller may already import this name,
    and a predicate that is silently renamed and silently inverted is the exact
    hazard the rename was made to remove: both spellings resolve to one
    implementation here, so they cannot drift apart.
    """
    fires, reason = floor_3_regime_mismatch(regime_a, regime_b)
    return not fires, reason
