"""
Tests for analysis/sigma_attractor.py -- S-1 gate C and headroom floor 3.

Synthetic data only: every survivor set here is hand-constructed for the test
or drawn from a seeded ``numpy.random.default_rng``. Nothing is read from
``logs/``, from any control's output, or from any prior conversation; no number
asserted below is imported from anywhere. Arrange-Act-Assert throughout, and
test names state the behaviour under test.

Two of these tests are the reason the file exists in this shape rather than a
shorter one, and they are marked in the sections below:

  * ``test_permutation_null_is_flat_when_only_same_founder_pairs_carry_signal``
    is the direct check that the permutation preserves the cross-founder
    pairing structure. If the implementation leaked same-founder pairs into
    either term, the null in that fixture would acquire a non-zero spread,
    because permuting sigma changes which same-founder pairs count as
    within-sigma. The null being exactly flat is therefore evidence about the
    permutation's structure and not merely about the observed statistic.

  * ``test_same_founder_pairs_are_excluded_even_when_including_them_flips_the_sign``
    builds a fixture whose all-pairs statistic has the opposite sign to its
    cross-founder statistic, so an implementation that forgot the
    "cross-founder" clause of the committed metric cannot pass by accident.

Run from the repository root:

    /usr/bin/python3 -m pytest tests/test_sigma_attractor.py -v
"""

import itertools
import os
import sys

import numpy as np
import pytest

# Mirrors tests/test_overlap_stats.py's sys.path pattern: pytest's rootdir is
# not automatically on sys.path, so add the repository root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import overlap_stats as ov
from analysis import sigma_attractor as sa


# --------------------------------------------------------------------------
# Fixtures, built as module-level helpers so that the construction of each is
# visible beside the test that reasons about it.
# --------------------------------------------------------------------------

# Class-label blocks used by the hand-computed fixtures. Named rather than
# inlined so that a reader can see that the blocks are disjoint, which is what
# makes the intersection sizes below checkable by eye.
_BLOCK = 20


def _sigma_driven_chains(n_founders=4, chains_per_cell=2):
    """Chains whose survivor sets are determined by sigma, not by founder.

    Every chain at sigma 0 draws its survivors from classes 0-19 and every
    chain at sigma 1 from classes 20-39, so two chains at the same sigma
    overlap in eight of ten classes however different their founders, and two
    chains at different sigma are disjoint. This is the case gate C is written
    to detect: the prior scale defines an attractor, so gate C must FAIL
    (one-sided p < 0.05 for within-sigma exceeding across-sigma).

    The founder enters only through a single distinguishing class per founder,
    so that the fixture is not the fully degenerate "all chains in a cell are
    the identical set" case while still being hand-computable: the
    within-sigma cross-founder Jaccard is 8 / 10 = 0.8 for every such pair and
    the across-sigma Jaccard is 0.
    """
    sets, founders, sigmas = [], [], []
    for sigma in (0, 1):
        base = list(range(_BLOCK * sigma, _BLOCK * sigma + 8))
        for founder in range(n_founders):
            survivors = base + [_BLOCK * sigma + 8 + founder]
            for _ in range(chains_per_cell):
                sets.append(list(survivors))
                founders.append(founder)
                sigmas.append(sigma)
    return sets, founders, sigmas


def _sign_flip_chains():
    """Chains whose cross-founder and all-pairs statistics have opposite signs.

    Four size-ten survivor sets over classes 0-33, two chains per
    (founder, sigma) cell, eight chains in all. Every intersection is readable
    off the literals:

        A0 = 0..9        A1 = 10..19
        B0 = {10,11,12} u 20..26        B1 = {0,1,2} u 27..33

        |A0 n B0| = 0   |A1 n B1| = 0      (cross-founder, within-sigma)
        |A0 n B1| = 3   |A1 n B0| = 3      (cross-founder, across-sigma)
        |A0 n A1| = 0   |B0 n B1| = 0      (same-founder, across-sigma)

    All sets have size ten, so an intersection of m gives a Jaccard of
    m / (20 - m): 0 for m = 0 and 3/17 for m = 3.

    Cross-founder only, which is what the committed metric asks for:
    within-sigma mean 0, across-sigma mean 3/17, statistic -3/17 < 0.

    All pairs, which is what a reading that dropped "cross-founder" would
    compute: the four same-founder same-sigma pairs are identical sets at
    Jaccard 1.0 and dominate the within-sigma term, giving 4/12 = 1/3 against
    an across-sigma term of (8 * 3/17) / 16 = 3/34, a statistic of about
    +0.245 > 0.

    The sign therefore flips, and only the cross-founder reading is negative.
    """
    a0 = list(range(0, 10))
    a1 = list(range(10, 20))
    b0 = [10, 11, 12] + list(range(20, 27))
    b1 = [0, 1, 2] + list(range(27, 34))

    sets = [a0, a0, a1, a1, b0, b0, b1, b1]
    founders = [0, 0, 0, 0, 1, 1, 1, 1]
    sigmas = [0, 0, 1, 1, 0, 0, 1, 1]
    return sets, founders, sigmas


def _founder_only_chains(n_founders=4, chains_per_cell=2):
    """Chains in which the founder carries everything and sigma carries nothing.

    All four of a founder's chains hold that founder's identical set, and the
    four founders' sets are built to have the SAME pairwise Jaccard as each
    other: each is two shared classes {0, 1} plus eight classes unique to the
    founder, so every cross-founder pair intersects in exactly 2 of a union of
    18 and sits at 1/9, while every same-founder pair sits at 1.0.

    The cross-founder within-sigma and across-sigma means are therefore both
    exactly 1/9 under the real labelling AND under every permutation of the
    sigma label, so the whole null is flat at zero. The shared classes make the
    constant nonzero rather than zero, so a statistic that dropped one of the
    two terms could not match the flat null by arithmetic accident.

    Four founders rather than two, so that no permutation of the sigma label
    can leave one of gate C's terms empty: each sigma group holds eight chains
    and each founder only four, so a sigma group can never consist of a single
    founder. The two-founder case genuinely can degenerate that way and is
    pinned separately in
    ``test_gate_c_refuses_a_design_whose_permutations_can_empty_a_term``.
    """
    sets, founders, sigmas = [], [], []
    for founder in range(n_founders):
        survivors = [0, 1] + list(range(10 + 8 * founder, 18 + 8 * founder))
        for sigma in (0, 1):
            for _ in range(chains_per_cell):
                sets.append(list(survivors))
                founders.append(founder)
                sigmas.append(sigma)
    return sets, founders, sigmas


def _anti_sigma_chains():
    """Chains at the same sigma are LESS alike than chains at different sigma.

    Gate C is one-sided for within-sigma exceeding across-sigma, so a fixture
    with a strongly negative statistic is needed to show the test does not fire
    on the wrong tail. Four founders, two sigma values, two chains per cell,
    sixteen chains. A chain's survivor set is block X = 0..9 when
    (founder + sigma) is even and block Y = 10..19 when it is odd, and the two
    blocks are disjoint, so every pairwise Jaccard is exactly 1.0 or 0.0.

    Hand-computed. Of the six unordered founder pairs, two are same-parity
    ({0,2} and {1,3}) and four are different-parity. A cross-founder pair at
    the SAME sigma lands on the same block exactly when the two founders have
    the same parity, so the within-sigma mean is 2/6 = 1/3; a cross-founder
    pair at DIFFERENT sigma lands on the same block exactly when the founders
    have different parity, so the across-sigma mean is 4/6 = 2/3. The statistic
    is 1/3 - 2/3 = -1/3.

    Pair counts, also hand-computed: C(16, 2) = 120 pairs in all, of which
    4 * C(4, 2) = 24 are same-founder and excluded, leaving 96 cross-founder
    pairs split 48 within-sigma and 48 across-sigma.
    """
    sets, founders, sigmas = [], [], []
    for founder in range(4):
        for sigma in (0, 1):
            block = 0 if (founder + sigma) % 2 == 0 else 1
            survivors = list(range(10 * block, 10 * block + 10))
            for _ in range(2):
                sets.append(list(survivors))
                founders.append(founder)
                sigmas.append(sigma)
    return sets, founders, sigmas


def _uniform_equal_size_sets(n_sets, n_classes, size, seed):
    """The degenerate null the committed rule names explicitly.

    Survivor sets drawn uniformly over classes at a FIXED common size. Under
    class-exchangeability at a fixed size every set has the same conditional
    mean Jaccard against an independent draw, so the first-order (Hajek)
    projection of the mean pairwise Jaccard is constant and its variance
    zeta_1 is exactly zero. Drawn through overlap_stats.draw_survivors_weighted
    with uniform weights so that this is the same sampler
    overlap_stats.size_conditioned_null uses, not a re-implementation of it.
    """
    rng = np.random.default_rng(seed)
    weights = np.ones(n_classes, dtype=float)
    return ov.draw_survivors_weighted([size] * n_sets, weights, rng)


def _clustered_sets(n_classes=50, cluster=10, outliers=10, seed=0):
    """A null with real between-set structure, so zeta_1 is positive.

    Half the sets are the identical "core" set and half are independent random
    subsets of the remaining classes. A core set sits at Jaccard 1.0 against
    every other core set and 0 against every outlier, while an outlier sits at
    0 against every core set and near 0 against the other outliers. The
    conditional mean Jaccard therefore differs sharply between the two kinds of
    set, which is exactly a positive first-order projection variance: the
    non-degenerate regime.
    """
    rng = np.random.default_rng(seed)
    core = list(range(cluster))
    sets = [list(core) for _ in range(cluster)]
    remaining = np.arange(cluster, n_classes)
    for _ in range(outliers):
        sets.append(sorted(rng.choice(remaining, size=cluster, replace=False).tolist()))
    return sets


def _all_pairs_within_minus_across(sets, founder_labels, sigma_labels):
    """The statistic a reading that DROPPED "cross-founder" would compute.

    Used only by the sign-flip test, as the thing the real implementation must
    not agree with. Deliberately written here in the test rather than exported
    from the module, because it is not a quantity the committed rule asks for
    and nothing in the project should be able to call it by accident.
    """
    within, across = [], []
    for i, j in itertools.combinations(range(len(sets)), 2):
        value = ov.jaccard(sets[i], sets[j])
        if sigma_labels[i] == sigma_labels[j]:
            within.append(value)
        else:
            across.append(value)
    return float(np.mean(within)) - float(np.mean(across))


# --------------------------------------------------------------------------
# within_minus_across -- the committed gate C statistic
# --------------------------------------------------------------------------

def test_within_minus_across_is_positive_when_sigma_drives_the_survivor_sets():
    # Arrange
    sets, founders, sigmas = _sigma_driven_chains()

    # Act
    observed = sa.within_minus_across(sets, founders, sigmas)

    # Assert: every cross-founder within-sigma pair is 8/10 = 0.8 and every
    # cross-founder across-sigma pair is disjoint, so the statistic is 0.8.
    assert observed == pytest.approx(0.8)


def test_within_minus_across_hand_computed_on_the_sign_flip_fixture():
    # Arrange
    sets, founders, sigmas = _sign_flip_chains()

    # Act
    observed = sa.within_minus_across(sets, founders, sigmas)

    # Assert: 0 - 3/17, computed in the fixture's docstring.
    assert observed == pytest.approx(0.0 - 3 / 17)


def test_same_founder_pairs_are_excluded_even_when_including_them_flips_the_sign():
    # Arrange: a fixture built so that the two readings disagree in SIGN, not
    # merely in magnitude, so no tolerance choice can hide a leak.
    sets, founders, sigmas = _sign_flip_chains()

    # Act
    cross_founder = sa.within_minus_across(sets, founders, sigmas)
    all_pairs = _all_pairs_within_minus_across(sets, founders, sigmas)

    # Assert: the committed metric is the negative one; the all-pairs reading
    # is positive. Both exact values are hand-computed in the fixture.
    assert cross_founder == pytest.approx(-3 / 17)
    assert all_pairs == pytest.approx(1 / 3 - 3 / 34)
    assert cross_founder < 0 < all_pairs


def test_within_minus_across_is_zero_when_only_the_founder_carries_signal():
    # Arrange
    sets, founders, sigmas = _founder_only_chains()

    # Act
    observed = sa.within_minus_across(sets, founders, sigmas)

    # Assert: both cross-founder terms are 1/9, so the difference is exactly
    # zero however large the same-founder similarity is.
    assert observed == pytest.approx(0.0)


def test_within_minus_across_rejects_mismatched_label_lengths():
    # Arrange
    sets, founders, sigmas = _sigma_driven_chains()

    # Act / Assert
    with pytest.raises(ValueError, match="same length"):
        sa.within_minus_across(sets, founders[:-1], sigmas)


def test_within_minus_across_rejects_a_single_sigma_value():
    # Arrange: with one sigma value there are no across-sigma pairs at all, so
    # the committed difference has no second term and is not a number.
    sets, founders, sigmas = _sigma_driven_chains()
    one_sigma = [0] * len(sigmas)

    # Act / Assert
    with pytest.raises(ValueError, match="across-sigma"):
        sa.within_minus_across(sets, founders, one_sigma)


def test_within_minus_across_rejects_a_single_founder():
    # Arrange: with one founder there are no cross-founder pairs, which is the
    # whole content of the metric.
    sets, founders, sigmas = _sigma_driven_chains()
    one_founder = [0] * len(founders)

    # Act / Assert
    with pytest.raises(ValueError, match="cross-founder"):
        sa.within_minus_across(sets, one_founder, sigmas)


# --------------------------------------------------------------------------
# gate_c_test -- the permutation test the committed rule specifies
# --------------------------------------------------------------------------

def test_gate_c_returns_exactly_the_committed_keys():
    # Arrange
    sets, founders, sigmas = _sigma_driven_chains()

    # Act
    result = sa.gate_c_test(sets, founders, sigmas, n_perm=200, seed=0)

    # Assert: the aggregator is written against this key set, so an extra or
    # missing key is a breaking change and is pinned here.
    assert set(result) == {
        "observed", "null_mean", "null_sd", "null_min", "null_max",
        "p_value", "n_perm", "seed", "n_sets", "n_within_pairs",
        "n_across_pairs",
    }


def test_gate_c_is_significant_when_same_sigma_chains_really_are_more_alike():
    # Arrange
    sets, founders, sigmas = _sigma_driven_chains()

    # Act
    result = sa.gate_c_test(sets, founders, sigmas, n_perm=1000, seed=0)

    # Assert: the smallest p-value the conservative estimate can return at
    # n_perm = 1000 is 1 / 1001. Gate C FAILS here, which is the correct
    # reading of this fixture: sigma defines an attractor in it.
    assert result["observed"] == pytest.approx(0.8)
    assert result["p_value"] == pytest.approx(1 / 1001)
    assert result["p_value"] < 0.05


def test_gate_c_is_not_significant_when_the_sets_are_independent_of_sigma():
    # Arrange / Act: ten independent fixtures in which sigma is assigned at
    # random over sets drawn from one common distribution. Several seeds rather
    # than one, because a single seed would be a 5-percent coin flip and
    # picking the seed that passed would be seed-shopping.
    p_values, observed = [], []
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        sets = _uniform_equal_size_sets(n_sets=24, n_classes=50, size=10, seed=seed)
        founders = [i // 2 for i in range(24)]
        sigmas = rng.permutation([0] * 12 + [1] * 12).tolist()
        result = sa.gate_c_test(sets, founders, sigmas, n_perm=500, seed=seed)
        p_values.append(result["p_value"])
        observed.append(result["observed"])

    # Assert: the statistic sits near zero on average, and the gate does not
    # fire. One of ten crossing 0.05 would be within chance, so the assertion
    # is on the count, not on every single draw.
    assert abs(float(np.mean(observed))) < 0.02
    assert sum(p > 0.05 for p in p_values) >= 8


def test_permutation_null_is_flat_when_only_same_founder_pairs_carry_signal():
    # Arrange: the direct test that the permutation preserves the cross-founder
    # pairing structure. See this module's docstring.
    sets, founders, sigmas = _founder_only_chains()

    # Act
    result = sa.gate_c_test(sets, founders, sigmas, n_perm=300, seed=0)

    # Assert: holding the founders fixed and permuting sigma can never change
    # a cross-founder mean here, so the entire null collapses onto zero. A
    # leak of same-founder pairs into either term would give this a non-zero
    # spread, because a permutation can move a same-founder pair from the
    # within-sigma term to the across-sigma term.
    assert result["observed"] == pytest.approx(0.0)
    assert result["null_sd"] == pytest.approx(0.0)
    assert result["null_min"] == pytest.approx(0.0)
    assert result["null_max"] == pytest.approx(0.0)


def test_permutation_null_has_spread_when_sigma_is_informative():
    # Arrange: the complement of the test above, so that a flat null cannot be
    # mistaken for a correct implementation that never permutes anything.
    sets, founders, sigmas = _sigma_driven_chains()

    # Act
    result = sa.gate_c_test(sets, founders, sigmas, n_perm=300, seed=0)

    # Assert
    assert result["null_sd"] > 0.0
    assert result["null_min"] < result["null_max"]


def test_gate_c_pair_counts_cover_every_cross_founder_pair_and_no_other():
    # Arrange
    sets, founders, sigmas = _anti_sigma_chains()
    n = len(sets)
    expected_cross_founder = sum(
        1 for i, j in itertools.combinations(range(n), 2)
        if founders[i] != founders[j]
    )

    # Act
    result = sa.gate_c_test(sets, founders, sigmas, n_perm=50, seed=0)

    # Assert: the two reported counts partition the cross-founder pairs
    # exactly, and fall short of all C(n, 2) pairs by the 24 same-founder
    # pairs the metric excludes. Counts hand-computed in the fixture.
    assert result["n_sets"] == 16
    assert expected_cross_founder == 96
    assert result["n_within_pairs"] + result["n_across_pairs"] == expected_cross_founder
    assert expected_cross_founder < n * (n - 1) // 2
    assert result["n_within_pairs"] == 48
    assert result["n_across_pairs"] == 48


def test_gate_c_refuses_a_design_whose_permutations_can_empty_a_term():
    # Arrange: eight chains over two founders, four chains each, four chains at
    # each sigma. A free permutation of the sigma label -- which is the null the
    # committed rule names -- can put all four of founder 0's chains at one
    # sigma and all four of founder 1's at the other, and then every
    # cross-founder pair spans two sigma values and gate C's within-sigma term
    # has no pairs at all.
    sets, founders, sigmas = _sign_flip_chains()

    # Act / Assert: refused loudly rather than returned as a NaN that would
    # read as a passing gate downstream. This cannot arise on the real S-1
    # design, where 20 founders contribute 2 chains each per cell so a sigma
    # group of 40 chains can never be a single founder; it is pinned here
    # because the committed rule does not cover the case and a smaller pilot
    # design would hit it.
    with pytest.raises(ValueError, match="left one of gate C's two terms empty"):
        sa.gate_c_test(sets, founders, sigmas, n_perm=1000, seed=0)


def test_gate_c_p_value_uses_the_conservative_plus_one_estimate():
    # Arrange: the founder-only fixture makes every null draw exactly equal to
    # the observed statistic, so every one of them counts as "null >= observed"
    # and the numerator and denominator are both n_perm + 1.
    sets, founders, sigmas = _founder_only_chains()
    n_perm = 128

    # Act
    result = sa.gate_c_test(sets, founders, sigmas, n_perm=n_perm, seed=0)

    # Assert: (1 + n_perm) / (1 + n_perm) = 1.0, the largest value the
    # estimate can take, and the one that correctly says "no evidence".
    assert result["p_value"] == pytest.approx(1.0)
    assert result["n_perm"] == n_perm


def test_gate_c_p_value_is_one_sided_for_within_exceeding_across():
    # Arrange: a strongly NEGATIVE observed statistic -- chains at the same
    # sigma are LESS alike than chains at different sigma. The committed gate
    # only fires for within-sigma EXCEEDING across-sigma, so this must give a
    # large p-value, not a small one.
    sets, founders, sigmas = _anti_sigma_chains()

    # Act
    result = sa.gate_c_test(sets, founders, sigmas, n_perm=1000, seed=0)

    # Assert: the statistic is the hand-computed 1/3 - 2/3, and almost every
    # permutation beats it, so the gate does not fire. A two-sided test would
    # have fired here, which is what this test exists to forbid.
    assert result["observed"] == pytest.approx(-1 / 3)
    assert result["p_value"] > 0.5


def test_gate_c_is_deterministic_under_a_fixed_seed():
    # Arrange
    sets, founders, sigmas = _sigma_driven_chains()

    # Act
    first = sa.gate_c_test(sets, founders, sigmas, n_perm=400, seed=7)
    second = sa.gate_c_test(sets, founders, sigmas, n_perm=400, seed=7)

    # Assert
    assert first == second
    assert first["seed"] == 7


def test_gate_c_null_differs_across_seeds():
    # Arrange: determinism must not be achieved by ignoring the seed.
    sets, founders, sigmas = _sigma_driven_chains()

    # Act
    first = sa.gate_c_test(sets, founders, sigmas, n_perm=400, seed=0)
    second = sa.gate_c_test(sets, founders, sigmas, n_perm=400, seed=1)

    # Assert
    assert first["null_mean"] != second["null_mean"]


def test_gate_c_rejects_a_non_positive_permutation_count():
    # Arrange
    sets, founders, sigmas = _sigma_driven_chains()

    # Act / Assert
    with pytest.raises(ValueError, match="n_perm"):
        sa.gate_c_test(sets, founders, sigmas, n_perm=0, seed=0)


def test_gate_c_defaults_to_the_committed_ten_thousand_permutations():
    # Arrange / Act: the committed rule fixes 10000 permutations, so the
    # default must not need to be supplied by a caller to be correct.
    sets, founders, sigmas = _founder_only_chains()
    result = sa.gate_c_test(sets, founders, sigmas)

    # Assert: run on the flat-null fixture, which is the cheapest one, so this
    # test pins the default without paying for an expensive null.
    assert result["n_perm"] == 10_000
    assert result["seed"] == 0


# --------------------------------------------------------------------------
# first_order_variance and scaling_regime -- headroom floor 3
# --------------------------------------------------------------------------

def test_uniform_over_classes_null_at_fixed_sizes_is_classified_degenerate():
    # Arrange / Act: the case the committed rule names explicitly. Five seeds,
    # because a classifier that happened to work on one draw would not be
    # evidence about the regime.
    regimes = []
    for seed in range(5):
        sets = _uniform_equal_size_sets(n_sets=60, n_classes=50, size=10, seed=seed)
        regimes.append(sa.scaling_regime(sets)["regime"])

    # Assert
    assert regimes == ["degenerate"] * 5


def test_uniform_null_first_order_variance_estimate_sits_near_zero():
    # Arrange: the sharper statement behind the classification. zeta_1 is
    # exactly zero for this null, so the estimate must be small compared with
    # the second-order variance it is being weighed against -- which is what
    # the regime criterion compares.
    sets = _uniform_equal_size_sets(n_sets=60, n_classes=50, size=10, seed=0)
    n = len(sets)

    # Act
    zeta_1 = sa.first_order_variance(sets)
    zeta_2 = sa.second_order_variance(sets)

    # Assert: the criterion's own scale. 2 * (n - 2) * zeta_1 is the
    # first-order contribution to the U-statistic variance and zeta_2 the
    # second-order one; for a truly degenerate null the former must be the
    # smaller.
    assert zeta_2 > 0
    assert 2 * (n - 2) * zeta_1 < zeta_2


def test_a_null_with_between_set_structure_is_classified_non_degenerate():
    # Arrange / Act
    regimes = []
    for seed in range(5):
        sets = _clustered_sets(seed=seed)
        regimes.append(sa.scaling_regime(sets)["regime"])

    # Assert
    assert regimes == ["non-degenerate"] * 5


def test_clustered_null_first_order_variance_is_clearly_positive():
    # Arrange
    sets = _clustered_sets(seed=0)

    # Act
    zeta_1 = sa.first_order_variance(sets)

    # Assert: half the sets sit at mean Jaccard near 0.47 against the pool and
    # half near 0, so the first-order projection has real spread.
    assert zeta_1 > 0.01


def test_scaling_regime_returns_exactly_the_committed_keys():
    # Arrange
    sets = _clustered_sets(seed=0)

    # Act
    result = sa.scaling_regime(sets)

    # Assert: floor 3 requires n and an estimate of the first-order variance
    # to be reported for every null, and the aggregator reads these three keys.
    assert set(result) == {"n", "zeta_1", "regime"}
    assert result["n"] == len(sets)
    assert result["zeta_1"] == pytest.approx(sa.first_order_variance(sets))


def test_identical_sets_give_a_degenerate_regime_rather_than_a_division():
    # Arrange: every pairwise Jaccard is 1.0, so both variance components are
    # zero. zeta_1 = 0 is degenerate by definition, and the classifier must
    # say so rather than fall through a zero comparison to "non-degenerate".
    sets = [list(range(10)) for _ in range(12)]

    # Act
    result = sa.scaling_regime(sets)

    # Assert
    assert result["zeta_1"] == pytest.approx(0.0)
    assert result["regime"] == "degenerate"


def test_first_order_variance_needs_at_least_three_sets():
    # Arrange: the bias correction divides by n - 2, and with two sets there is
    # one pair and no between-set variation to estimate from at all.
    sets = [list(range(10)), list(range(5, 15))]

    # Act / Assert
    with pytest.raises(ValueError, match="at least three"):
        sa.first_order_variance(sets)


def test_first_order_variance_is_deterministic_and_order_invariant():
    # Arrange: a U-statistic kernel is symmetric and the estimator is a
    # function of the multiset of sets, so relabelling the sets must not move
    # it.
    sets = _clustered_sets(seed=3)
    shuffled = list(reversed(sets))

    # Act
    original = sa.first_order_variance(sets)
    reordered = sa.first_order_variance(shuffled)

    # Assert
    assert original == pytest.approx(reordered)


# --------------------------------------------------------------------------
# floor_3_regime_mismatch -- the NOT EVALUABLE decision
#
# Polarity, which is the whole reason these tests are explicit about `is True`
# and `is False` rather than truthiness: True means THE FLOOR FIRED and the
# gate is NOT EVALUABLE. The deprecated `floor_3_regime_match` spelling has the
# opposite polarity and is tested separately below.
# --------------------------------------------------------------------------

def test_floor_3_does_not_fire_on_two_non_degenerate_nulls():
    # Arrange / Act
    fires, reason = sa.floor_3_regime_mismatch("non-degenerate", "non-degenerate")

    # Assert
    assert fires is False
    assert "non-degenerate" in reason


def test_floor_3_does_not_fire_on_two_degenerate_nulls():
    # Arrange / Act: floor 3 is about a MISMATCH of regimes, not about
    # degeneracy itself, so two degenerate nulls are comparable to each other
    # and the floor must not fire. Reading the committed clause as "degenerate
    # nulls may not be used" would tighten a preregistered floor.
    fires, reason = sa.floor_3_regime_mismatch("degenerate", "degenerate")

    # Assert
    assert fires is False
    assert "degenerate" in reason


def test_floor_3_fires_on_a_mismatched_pair_and_names_both_regimes():
    # Arrange / Act
    fires, reason = sa.floor_3_regime_mismatch("non-degenerate", "degenerate")

    # Assert: the refusal has to say which null was which, or the gate's
    # NOT EVALUABLE verdict cannot be written up from it.
    assert fires is True
    assert "non-degenerate" in reason
    assert "degenerate" in reason
    assert "NOT EVALUABLE" in reason


def test_floor_3_names_both_regimes_even_when_it_does_not_fire():
    # Arrange / Act: a verdict written from the return value alone must be able
    # to state what the two regimes were in the permitting case too, not only
    # in the refusing one.
    _, reason = sa.floor_3_regime_mismatch("degenerate", "degenerate")

    # Assert
    assert reason.count("degenerate") >= 2
    assert "may be read" in reason


def test_floor_3_firing_is_symmetric_in_its_two_arguments():
    # Arrange / Act
    forward, _ = sa.floor_3_regime_mismatch("degenerate", "non-degenerate")
    backward, _ = sa.floor_3_regime_mismatch("non-degenerate", "degenerate")

    # Assert
    assert forward is True
    assert backward is True


def test_floor_3_rejects_a_regime_name_it_does_not_define():
    # Arrange: the only two regimes the committed rule names are the 1/sqrt(n)
    # and 1/n ones. Anything else is a caller bug and must not be silently
    # treated as a match or a mismatch.
    # Act / Assert
    with pytest.raises(ValueError, match="regime"):
        sa.floor_3_regime_mismatch("non-degenerate", "unknown")


def test_deprecated_regime_match_alias_is_the_exact_inverse():
    # Arrange: the alias exists so that a sibling importing either spelling
    # gets a correct answer. What must never happen is the two spellings
    # disagreeing about anything except the sign of the boolean, so the
    # equivalence is pinned over every combination rather than asserted once.
    for a in ("degenerate", "non-degenerate"):
        for b in ("degenerate", "non-degenerate"):
            # Act
            fires, fires_reason = sa.floor_3_regime_mismatch(a, b)
            matched, matched_reason = sa.floor_3_regime_match(a, b)

            # Assert
            assert matched is (not fires)
            assert matched_reason == fires_reason


def test_deprecated_regime_match_alias_still_validates_regime_names():
    # Arrange / Act / Assert: the alias must not be a hole in the validation.
    with pytest.raises(ValueError, match="regime"):
        sa.floor_3_regime_match("unknown", "degenerate")
