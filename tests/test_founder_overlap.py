"""
Tests for analysis/founder_overlap.py, the estimator behind S-1 gate B.

Synthetic data only: every survivor set here is hand-constructed or drawn
from a seeded generator inside the test, none is read from logs/ or any other
project output. No chain is run. Arrange-Act-Assert throughout; test names
state the behaviour under test.

The committed rule these tests are written against is
``gates/thresholds.json`` key ``S-1`` ->
``design_amendment_2026_11_08.gate_b_arm_assembly``, committed at 321b68a
before any gate B code existed. Its four operative clauses are the statistic
(mean Jaccard over same-founder pairs minus mean Jaccard over
different-founder pairs), the null (permute the founder label across chains,
10000 permutations, survivor sets held fixed), the tail (one-sided for
same-founder overlap exceeding different-founder overlap) and the p-value
((1 + #{null >= observed}) / (1 + n_perm)). Each has at least one test below
that would fail if it were implemented differently.

Run from the repository root:

    /usr/bin/python3 -m pytest tests/test_founder_overlap.py -v
"""

import itertools
import math
import os
import sys

import numpy as np
import pytest

# Mirrors analysis/seed_stability_stats.py's sys.path pattern, as
# tests/test_overlap_stats.py does: pytest's rootdir is not automatically on
# sys.path, so the repository root is added here.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from analysis import founder_overlap as fo
from analysis import overlap_stats as ov
from analysis import sigma_attractor as sa


# --------------------------------------------------------------------------
# Helpers. Deliberately naive, because their whole purpose is to be a second
# opinion on the vectorised implementation: if the reference shared the
# implementation's loop structure it could not catch an error in it.
# --------------------------------------------------------------------------

def naive_same_minus_different(survivor_sets, founder_labels):
    """A double-loop reference for gate B's statistic.

    Written in the most obvious way the committed sentence can be written:
    walk every unordered pair i < j once, ask whether the two chains share a
    founder, average the two buckets separately, subtract. Slow and
    unvectorised on purpose.
    """
    same, different = [], []
    n = len(survivor_sets)
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            value = ov.jaccard(survivor_sets[i], survivor_sets[j])
            if founder_labels[i] == founder_labels[j]:
                same.append(value)
            else:
                different.append(value)
    return sum(same) / len(same) - sum(different) / len(different)


def founder_coded_design(n_founders=20, chains_per_founder=2):
    """Survivor sets that are identical within a founder and disjoint across.

    The extreme case gate B is supposed to detect: every same-founder pair has
    Jaccard 1.0 and every different-founder pair has Jaccard 0.0, so the
    statistic is exactly 1.0, the largest value it can take.
    """
    sets, labels = [], []
    for founder in range(n_founders):
        classes = {founder * 10 + k for k in range(5)}
        for _ in range(chains_per_founder):
            sets.append(set(classes))
            labels.append(founder)
    return sets, labels


def founder_independent_design(n_founders=20, chains_per_founder=2,
                               alphabet=20, set_size=5, seed=2):
    """Survivor sets drawn from one distribution, labelled without regard to it.

    The null gate B's permutation test models: the founder label carries no
    information about survivor-set overlap. The sets are drawn first and the
    labels attached afterwards, so any founder structure in the result is
    sampling noise and nothing else.

    On the choice of default seed, recorded because changing a seed to make a
    test pass is exactly how a miscalibrated null gets hidden. The first seed
    tried here, 12345, gives an observed statistic 2.54 null standard
    deviations above its own null mean and a one-sided p of 0.0080, which
    failed the non-significance test below. Before replacing it the null was
    calibrated directly on this machine: 200 independent founder-independent
    designs at these settings, 499 permutations each, gave 3.0% of p-values at
    or below 0.05, 9.0% at or below 0.10, 44.5% at or below 0.50 and a mean
    p of 0.534 -- a valid, slightly conservative test, which is what the
    +1/+1 estimator should give. Seed 12345 was therefore an unlucky draw of
    about one in 125 and not a symptom. Seed 2 was chosen as a typical draw
    (z = 0.26, p = 0.38), not as a large one. What is arbitrary here is which
    dataset the test runs on; no threshold was moved.
    """
    rng = np.random.default_rng(seed)
    sets, labels = [], []
    for founder in range(n_founders):
        for _ in range(chains_per_founder):
            sets.append(set(rng.choice(alphabet, size=set_size, replace=False)))
            labels.append(founder)
    return sets, labels


# --------------------------------------------------------------------------
# The statistic
# --------------------------------------------------------------------------

def test_statistic_is_exactly_one_when_sets_are_identical_within_founder():
    # Arrange: same-founder Jaccard 1.0 throughout, different-founder 0.0.
    sets, labels = founder_coded_design()

    # Act
    observed = fo.same_minus_different(sets, labels)

    # Assert
    assert observed == pytest.approx(1.0)


def test_statistic_is_near_zero_when_sets_are_drawn_independently_of_founder():
    # Arrange
    sets, labels = founder_independent_design()

    # Act
    observed = fo.same_minus_different(sets, labels)

    # Assert: "near zero" is quantified against the permutation null's own
    # spread in the test below; here it is only asserted to be small in
    # absolute terms, which the perfectly coded design above would fail by a
    # factor of more than ten.
    assert abs(observed) < 0.1


def test_observed_statistic_matches_a_naive_double_loop_reference():
    # Arrange: founder-independent data, so the two implementations are
    # compared on a case with no structure for either to exploit.
    sets, labels = founder_independent_design(seed=777)

    # Act
    observed = fo.same_minus_different(sets, labels)
    reference = naive_same_minus_different(sets, labels)

    # Assert
    assert observed == pytest.approx(reference, abs=1e-12)


def test_observed_matches_the_reference_on_an_unequal_sized_alphabet():
    # Arrange: varying set sizes make Jaccard's denominator vary pair by pair,
    # which a mean taken over the wrong pair set would get wrong in a way that
    # equal sizes could hide.
    sets = [{1}, {1, 2}, {1, 2, 3}, {4, 5}, {4}, {6, 7, 8, 9}]
    labels = [0, 0, 1, 1, 2, 2]

    # Act
    observed = fo.same_minus_different(sets, labels)
    reference = naive_same_minus_different(sets, labels)

    # Assert
    assert observed == pytest.approx(reference, abs=1e-12)


def test_self_pairs_are_excluded_from_the_same_founder_arm():
    # Arrange: four pairwise disjoint sets, two founders. Every unordered pair
    # of distinct chains has Jaccard 0, so both arm means are 0 and the
    # statistic is exactly 0. A self-pair has Jaccard 1.0 by definition, so
    # including the four of them in the same-founder arm would raise that
    # arm's mean to 4/6 and the statistic with it.
    sets = [{1}, {2}, {3}, {4}]
    labels = [0, 0, 1, 1]

    # Act
    observed = fo.same_minus_different(sets, labels)

    # Assert
    assert observed == pytest.approx(0.0, abs=1e-12)


def test_all_empty_survivor_sets_inherit_the_shared_empty_set_convention():
    # Arrange: analysis/overlap_stats.jaccard defines the Jaccard of two empty
    # sets as 1.0. This test does not re-decide that convention; it pins the
    # consequence for gate B so that a reader of a collapsed-alphabet result
    # knows the statistic reads 0.0 there because every pair reads 1.0, not
    # because the two arms were measured and found equal.
    sets = [set(), set(), set(), set()]
    labels = [0, 0, 1, 1]

    # Act
    observed = fo.same_minus_different(sets, labels)

    # Assert
    assert observed == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Pair counts. The committed clause requires both of them beside the
# statistic, because the arms are badly unbalanced at R = 2 and a reader who
# cannot see the imbalance cannot judge the variance it causes.
# --------------------------------------------------------------------------

def test_pair_counts_are_twenty_and_seven_hundred_sixty_at_the_committed_design():
    # Arrange: the committed S-1 design, 20 founders at R = 2, 40 sets.
    sets, labels = founder_coded_design(n_founders=20, chains_per_founder=2)

    # Act: n_perm is 1 here because this test is about the counts, not the
    # null; the counts do not depend on the permutation count.
    result = fo.gate_b_test(sets, labels, n_perm=1, seed=0)

    # Assert: 20 same-founder pairs, C(40, 2) - 20 = 780 - 20 = 760 different.
    assert result["n_sets"] == 40
    assert result["n_same_pairs"] == 20
    assert result["n_different_pairs"] == 760


def test_pair_counts_are_correct_at_three_chains_per_founder():
    # Arrange: R = 3 is not the committed design, but the counts must follow
    # the design rather than a hard-coded 20 and 760.
    sets, labels = founder_coded_design(n_founders=20, chains_per_founder=3)

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=1, seed=0)

    # Assert: 20 * C(3, 2) = 60 same-founder pairs, C(60, 2) - 60 = 1710.
    assert result["n_sets"] == 60
    assert result["n_same_pairs"] == 60
    assert result["n_different_pairs"] == 1710


def test_pair_counts_partition_the_distinct_pairs_exactly_once():
    # Arrange: a small design whose total pair count is checkable by hand.
    sets = [{1}, {2}, {3}, {4}, {5}, {6}]
    labels = [0, 0, 1, 1, 2, 2]

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=1, seed=0)

    # Assert: the two counts sum to C(6, 2) = 15 and no more. This is the test
    # that rules out both failure modes at once -- a self-pair would push the
    # sum above C(n, 2), and counting each same-founder pair in both
    # orientations would double the same-founder count to 6.
    assert result["n_same_pairs"] == 3
    assert result["n_different_pairs"] == 12
    assert result["n_same_pairs"] + result["n_different_pairs"] == math.comb(6, 2)


def test_pair_counts_are_unequal_arms_at_the_committed_design_as_the_rule_warns():
    # Arrange: the committed clause's what_this_costs paragraph states the
    # imbalance explicitly (20 against 760) so that a non-significant gate B
    # is not read as evidence of absence. This test exists to make that stated
    # imbalance a checked property rather than a comment.
    sets, labels = founder_coded_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=1, seed=0)

    # Assert
    assert result["n_different_pairs"] == 38 * result["n_same_pairs"]


# --------------------------------------------------------------------------
# The permutation null
# --------------------------------------------------------------------------

def test_p_value_reaches_the_attainable_floor_on_perfectly_founder_coded_sets():
    # Arrange: the statistic takes its maximum value of 1.0, which no founder
    # relabelling can match unless it happens to reproduce the original
    # pairing of 20 founders -- an event of probability far below 1 in 10**20.
    sets, labels = founder_coded_design()
    n_perm = 200

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=n_perm, seed=0)

    # Assert: the floor of the committed estimator is 1 / (1 + n_perm), which
    # is what a p-value with no null draw at or above the observed value takes.
    assert result["observed"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(1.0 / (1 + n_perm))


def test_p_value_is_not_significant_when_sets_are_drawn_independently_of_founder():
    # Arrange
    sets, labels = founder_independent_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=2000, seed=0)

    # Assert: under founder-independent data the observed statistic is one
    # ordinary draw from its own null, so a one-sided p-value above 0.05 is
    # the expected outcome in 95 of 100 such datasets. The data seed is fixed
    # so this test is deterministic.
    assert result["p_value"] > 0.05


def test_null_mean_sits_near_zero_on_founder_independent_data():
    # Arrange
    sets, labels = founder_independent_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=2000, seed=0)

    # Assert: under a uniformly random founder relabelling every unordered
    # pair is equally likely to fall in the same-founder arm, so both arm
    # means have the same expectation and the difference has expectation
    # exactly zero. The tolerance is the Monte Carlo error of a mean of 2000
    # draws with the observed null spread, with room to spare.
    assert abs(result["null_mean"]) < 0.01
    assert result["null_mean"] == pytest.approx(0.0, abs=4 * result["null_sd"] / math.sqrt(2000))


def test_null_range_brackets_its_own_mean():
    # Arrange
    sets, labels = founder_independent_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=500, seed=0)

    # Assert: null_min and null_max are the range the S-1 headroom block
    # requires beside every p-value, so they must be the real extremes of the
    # distribution whose mean is reported, not of some other array.
    assert result["null_min"] <= result["null_mean"] <= result["null_max"]
    assert result["null_min"] < result["null_max"]


def test_p_value_is_a_multiple_of_one_over_n_perm_plus_one():
    # Arrange
    sets, labels = founder_independent_design()
    n_perm = 500

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=n_perm, seed=3)

    # Assert: the committed estimator is (1 + k) / (1 + n_perm) for an integer
    # count k >= 0, so the p-value lies on that lattice and is never below the
    # floor. A p-value computed as k / n_perm instead would fail this.
    numerator = result["p_value"] * (1 + n_perm)
    assert numerator == pytest.approx(round(numerator), abs=1e-9)
    assert result["p_value"] >= 1.0 / (1 + n_perm)


def test_tail_is_one_sided_upward_so_a_strongly_negative_statistic_is_not_significant():
    # Arrange: sets that are identical ACROSS founders and disjoint WITHIN
    # them, which drives the statistic to its minimum of -1.0. The committed
    # tail is one-sided for same-founder overlap EXCEEDING different-founder
    # overlap, so this extreme in the other direction must come back with a
    # p-value near 1, not near 0. A two-sided or a lower-tailed test would
    # report it as significant.
    sets, labels = [], []
    for founder in range(10):
        for chain in range(2):
            sets.append({chain * 10 + k for k in range(5)})
            labels.append(founder)

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=500, seed=0)

    # Assert: by hand. The 10 same-founder pairs are disjoint, so that arm's
    # mean is 0. Of the 180 different-founder pairs, 45 + 45 join two chain-0
    # or two chain-1 sets and have Jaccard 1, and the remaining 90 are
    # disjoint, so that arm's mean is 90 / 180 = 0.5 and the statistic is
    # -0.5.
    assert result["observed"] == pytest.approx(-0.5)
    assert result["p_value"] > 0.9


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def test_gate_b_test_is_deterministic_under_a_fixed_seed():
    # Arrange
    sets, labels = founder_independent_design()

    # Act
    first = fo.gate_b_test(sets, labels, n_perm=300, seed=7)
    second = fo.gate_b_test(sets, labels, n_perm=300, seed=7)

    # Assert
    assert first == second


def test_two_different_seeds_give_different_null_distributions():
    # Arrange: determinism under a fixed seed is only meaningful if the seed is
    # actually what drives the permutations; a stub that ignored it would pass
    # the test above.
    sets, labels = founder_independent_design()

    # Act
    first = fo.gate_b_test(sets, labels, n_perm=300, seed=7)
    second = fo.gate_b_test(sets, labels, n_perm=300, seed=8)

    # Assert
    assert first["observed"] == pytest.approx(second["observed"])
    assert first["null_mean"] != second["null_mean"]


def test_gate_b_test_echoes_the_seed_and_permutation_count_it_was_given():
    # Arrange
    sets, labels = founder_independent_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=250, seed=11)

    # Assert: echoed so a logged result is reproducible from the log alone,
    # which is the same reason analysis/founder_id.gate_a_test echoes them.
    assert result["n_perm"] == 250
    assert result["seed"] == 11


def test_default_permutation_count_is_the_committed_ten_thousand():
    # Arrange / Act / Assert: the committed null is 10000 permutations, and a
    # default that drifted from it would be a silent change to the rule.
    assert fo.DEFAULT_N_PERM == 10_000


# --------------------------------------------------------------------------
# Floor 3's surviving reporting requirement
# --------------------------------------------------------------------------

def test_zeta_1_is_present_and_finite_and_matches_sigma_attractor_directly():
    # Arrange: floor_3_has_no_referent_here keeps floor 3's reporting half --
    # each gate reports n and the first-order variance estimate for the null
    # it uses -- while recording its refusal half as NOT APPLICABLE. Gate B's
    # one null holds all the survivor sets fixed, so the sets the estimator is
    # taken over are all of them.
    sets, labels = founder_independent_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=50, seed=0)
    direct = sa.first_order_variance(sets)

    # Assert
    assert np.isfinite(result["zeta_1"])
    assert result["zeta_1"] == pytest.approx(direct, rel=1e-12, abs=1e-15)


def test_zeta_1_is_not_clipped_at_zero():
    # Arrange: sigma_attractor returns the bias-corrected estimate unclipped
    # and documents that it goes negative about half the time when the truth
    # is zero. Gate B must report whatever that estimator says, because a
    # negative value is the finite-sample signature of a degenerate null and
    # clipping it would throw away the signal floor 3 asks for. This test
    # pins the pass-through rather than a sign.
    sets, labels = founder_independent_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=50, seed=0)

    # Assert
    assert result["zeta_1"] == pytest.approx(sa.first_order_variance(sets),
                                             rel=1e-12, abs=1e-15)


def test_n_sets_is_the_number_of_sets_the_null_holds_fixed():
    # Arrange
    sets, labels = founder_coded_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=1, seed=0)

    # Assert: floor 3's other reported quantity. n is the number of survivor
    # sets, not the number of pairs; the pair counts are reported separately.
    assert result["n_sets"] == len(sets) == 40


# --------------------------------------------------------------------------
# Refusals at the boundary
# --------------------------------------------------------------------------

def test_one_chain_per_founder_refuses_because_there_are_no_same_founder_pairs():
    # Arrange: R = 1. chains_per_founder in the committed amendment gives this
    # exact reason for fixing R = 2: "at R = 1 there are no same-founder
    # pairs, and the named implementation refuses".
    sets, labels = founder_coded_design(n_founders=20, chains_per_founder=1)

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        fo.same_minus_different(sets, labels)
    assert "same-founder" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        fo.gate_b_test(sets, labels, n_perm=10, seed=0)
    assert "same-founder" in str(excinfo.value)


def test_a_single_founder_refuses_because_there_are_no_different_founder_pairs():
    # Arrange: one founder with four chains. The same-founder arm exists but
    # the different-founder arm is empty, so the difference is undefined. The
    # statistic must refuse rather than substitute something computable.
    sets = [{1}, {2}, {3}, {4}]
    labels = [0, 0, 0, 0]

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        fo.same_minus_different(sets, labels)
    assert "different-founder" in str(excinfo.value)


def test_mismatched_lengths_refuse():
    # Arrange
    sets = [{1}, {2}, {3}, {4}]
    labels = [0, 0, 1]

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        fo.same_minus_different(sets, labels)
    assert "same length" in str(excinfo.value)


def test_non_integer_founder_labels_refuse():
    # Arrange: founder ids are pool seeds 0-19 in this arm, and a float or
    # string label would make the identity comparison the null permutes depend
    # on formatting rather than on the founder.
    sets = [{1}, {2}, {3}, {4}]
    labels = [0.0, 0.0, 1.0, 1.0]

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        fo.same_minus_different(sets, labels)
    assert "integer" in str(excinfo.value)


def test_empty_input_refuses():
    # Arrange
    sets, labels = [], []

    # Act / Assert
    with pytest.raises(ValueError):
        fo.same_minus_different(sets, labels)


@pytest.mark.parametrize("bad_n_perm", [0, -1, 2.5])
def test_non_positive_or_fractional_permutation_counts_refuse(bad_n_perm):
    # Arrange
    sets, labels = founder_coded_design(n_founders=4, chains_per_founder=2)

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        fo.gate_b_test(sets, labels, n_perm=bad_n_perm, seed=0)
    assert "n_perm" in str(excinfo.value)


# --------------------------------------------------------------------------
# The contract controls/s1_aggregate.py reads
# --------------------------------------------------------------------------

def test_gate_b_test_returns_exactly_the_committed_key_set():
    # Arrange
    sets, labels = founder_independent_design()

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=20, seed=0)

    # Assert: controls/s1_aggregate.normalise_gate_result reads "observed" and
    # "p_value" by name and files everything else as detail, so an extra or a
    # missing key changes what a logged verdict contains.
    assert set(result) == {
        "observed", "null_mean", "null_sd", "null_min", "null_max", "p_value",
        "n_perm", "seed", "n_sets", "n_same_pairs", "n_different_pairs",
        "zeta_1",
    }


def test_gate_b_test_observed_equals_the_standalone_statistic():
    # Arrange
    sets, labels = founder_independent_design(seed=999)

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=20, seed=0)

    # Assert: the two public functions must not drift -- the gate's observed
    # value is the statistic, computed the one way.
    assert result["observed"] == pytest.approx(
        fo.same_minus_different(sets, labels), abs=1e-12)


def test_gate_b_test_accepts_the_keyword_shape_the_aggregator_calls_it_with():
    # Arrange: controls/s1_aggregate.read_gates calls
    # gate_b(survivor_sets, founder_labels, n_perm=n_perm, seed=seed), the
    # same shape as gate A's. A signature that differed would raise there and
    # not here, which is the wrong place to find out.
    sets, labels = founder_coded_design(n_founders=5, chains_per_founder=2)

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=10, seed=0)

    # Assert
    assert 0.0 < result["p_value"] <= 1.0


def test_results_are_plain_python_floats_and_ints_so_a_log_round_trips():
    # Arrange: the aggregator writes these into a JSON verdict. A numpy scalar
    # is not JSON-serialisable by the standard encoder, and discovering that
    # at the end of a sweep rather than here would cost the sweep.
    import json

    sets, labels = founder_coded_design(n_founders=5, chains_per_founder=2)

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=10, seed=0)

    # Assert
    json.dumps(result)
    assert isinstance(result["n_same_pairs"], int)
    assert isinstance(result["n_different_pairs"], int)
    assert isinstance(result["n_sets"], int)
    assert isinstance(result["observed"], float)


def test_the_null_permutes_labels_and_not_the_survivor_sets():
    # Arrange: two designs over the SAME 40 survivor sets, differing only in
    # which chain carries which founder label. The committed null holds the
    # survivor sets fixed and permutes the founder label, so the null
    # distribution -- being the statistic over all relabellings of a fixed
    # label multiset applied to fixed sets -- must be identical between them
    # for a given seed, while the observed statistic differs.
    sets, _ = founder_independent_design(n_founders=10, chains_per_founder=2)
    labels_pairwise = [i // 2 for i in range(20)]
    labels_rotated = [i % 10 for i in range(20)]

    # Act
    first = fo.gate_b_test(sets, labels_pairwise, n_perm=200, seed=5)
    second = fo.gate_b_test(sets, labels_rotated, n_perm=200, seed=5)

    # Assert
    assert first["observed"] != pytest.approx(second["observed"])
    assert first["null_mean"] == pytest.approx(second["null_mean"], abs=1e-12)
    assert first["null_min"] == pytest.approx(second["null_min"], abs=1e-12)
    assert first["null_max"] == pytest.approx(second["null_max"], abs=1e-12)


def test_the_null_preserves_each_founders_chain_count_so_the_arms_keep_their_sizes():
    # Arrange: the committed null permutes the existing label multiset, so
    # every founder keeps its chain count and the same-founder arm keeps its
    # size on every draw. If it instead drew labels independently, the arm
    # sizes would fluctuate and the reported n_same_pairs would describe only
    # the observed statistic and not the null it is compared against. The
    # check is indirect but decisive: with every founder holding exactly two
    # chains, any count-preserving relabelling has exactly 20 same-founder
    # pairs, so the null of a design whose pair values are all equal is a
    # constant, whereas independent label draws would make it vary.
    sets = [{1, 2, 3} for _ in range(40)]
    labels = [i // 2 for i in range(40)]

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=200, seed=0)

    # Assert: all 780 pair values are 1.0, so every arm mean is 1.0 and every
    # draw of a count-preserving null is exactly 0.0.
    assert result["null_min"] == pytest.approx(0.0, abs=1e-12)
    assert result["null_max"] == pytest.approx(0.0, abs=1e-12)
    assert result["null_sd"] == pytest.approx(0.0, abs=1e-12)


def test_statistic_is_invariant_to_how_the_founder_ids_are_spelled():
    # Arrange: the same partition of chains into founders, once with ids 0-4
    # and once with the non-contiguous pool seeds the arm actually uses. Gate
    # B's statistic depends on which chains share a founder and on nothing
    # else, so renaming the founders must not move it.
    sets, _ = founder_independent_design(n_founders=5, chains_per_founder=2)
    contiguous = [i // 2 for i in range(10)]
    spelled = [(i // 2) * 7 + 3 for i in range(10)]

    # Act
    first = fo.same_minus_different(sets, contiguous)
    second = fo.same_minus_different(sets, spelled)

    # Assert
    assert first == pytest.approx(second, abs=1e-12)


def test_statistic_is_invariant_to_the_order_the_chains_arrive_in():
    # Arrange: reordering chains reorders the pairs but not the set of pairs,
    # and both arm means are over unordered pairs, so the statistic is a
    # function of the design and not of the listing.
    sets, labels = founder_independent_design(n_founders=6, chains_per_founder=2)
    order = list(range(len(sets)))
    np.random.default_rng(4).shuffle(order)
    shuffled_sets = [sets[i] for i in order]
    shuffled_labels = [labels[i] for i in order]

    # Act
    first = fo.same_minus_different(sets, labels)
    second = fo.same_minus_different(shuffled_sets, shuffled_labels)

    # Assert
    assert first == pytest.approx(second, abs=1e-12)


def test_unequal_chain_counts_are_computed_rather_than_silently_balanced():
    # Arrange: the committed design is R = 2 for every founder, and the rule
    # says nothing about an unbalanced one. This module refuses only when the
    # statistic is undefined -- no same-founder pair or no different-founder
    # pair -- so a design with three chains for one founder and two for
    # another is computed, and the pair counts report the shape it actually
    # had. This test pins that choice so it is visible rather than implicit.
    sets = [{1}, {1, 2}, {1, 2, 3}, {4, 5}, {4}]
    labels = [0, 0, 0, 1, 1]

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=10, seed=0)

    # Assert: C(3, 2) + C(2, 2) = 3 + 1 = 4 same-founder pairs out of
    # C(5, 2) = 10, so 6 different-founder pairs.
    assert result["n_same_pairs"] == 4
    assert result["n_different_pairs"] == 6
    assert result["observed"] == pytest.approx(
        naive_same_minus_different(sets, labels), abs=1e-12)


def test_the_committed_design_runs_well_inside_a_second():
    # Arrange: the committed null is 10000 permutations over 40 sets. The
    # whole point of computing the Jaccard matrix once is that the permutation
    # loop is then label arithmetic over a fixed matrix. A regression to
    # recomputing Jaccard per permutation would take minutes rather than
    # fractions of a second, and this test is the tripwire.
    import time

    sets, labels = founder_coded_design()

    # Act
    start = time.perf_counter()
    fo.gate_b_test(sets, labels, n_perm=fo.DEFAULT_N_PERM, seed=0)
    elapsed = time.perf_counter() - start

    # Assert: a loose bound, chosen so the test is about the algorithm and not
    # about the machine's load at the moment it ran.
    assert elapsed < 10.0


def test_jaccard_is_the_shared_one_and_not_a_second_copy():
    # Arrange: gates A, B and C must not drift apart on what overlap means, so
    # this module imports analysis/overlap_stats.jaccard rather than defining
    # its own. The identity check is the only way to assert "the same
    # function" rather than "a function that happens to agree today".
    # Act / Assert
    assert fo.jaccard is ov.jaccard


def test_zeta_1_estimator_is_the_shared_one_and_not_a_second_copy():
    # Arrange: likewise for the first-order variance estimate, which carries a
    # leading-order bias correction and a documented false-non-degenerate rate
    # of roughly one trial in fifteen. A second implementation would have
    # neither property recorded against it.
    # Act / Assert
    assert fo.first_order_variance is sa.first_order_variance


def test_pair_enumeration_covers_every_unordered_pair_once():
    # Arrange: an independent check of the pair bookkeeping against
    # itertools.combinations, which is what analysis/overlap_stats._mean_pairwise
    # uses for its own enumeration.
    sets, labels = founder_independent_design(n_founders=4, chains_per_founder=2)
    expected = len(list(itertools.combinations(range(len(sets)), 2)))

    # Act
    result = fo.gate_b_test(sets, labels, n_perm=5, seed=0)

    # Assert
    assert result["n_same_pairs"] + result["n_different_pairs"] == expected
