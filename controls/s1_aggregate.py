"""
S-1 verdict aggregator: reads the per-cell chain logs and combines the three
gates into the arm's verdict under the committed rule.

The rule is gates/thresholds.json entry "S-1" (committed 2026-10-20, amended
2026-10-23, 2026-10-27 and 2026-11-08). This file implements the combination
block and the multiplicity correction; it changes nothing in the rule and
computes no gate statistic of its own. The statistics live where the rule
names them:

  gate A, floors 1 and 2   analysis/founder_id.py
  gate B                   analysis/overlap_stats.cross_arm_permutation_test
  gate C, floor 3          analysis/sigma_attractor.py

Everything here is pure numpy and stdlib. Nothing in this module runs a
chain, writes under logs/, or reads the gate statistics' own logs.

What the rule fixes and this file therefore does not choose:

  alpha 0.05, strictly ("fails_if: one-sided permutation p < 0.05 after
  Holm-Bonferroni"), n_perm 10000, 20 founders, R = 2 chains per founder,
  chance level 0.05, the six cells and their order, and that the correction
  family is gates A, B and C within the completed coverage and is not
  extended across cells.

Four places where the rule is silent and this file stops rather than guesses
are marked RULE SILENT below and raise RuleSilent.

    /usr/bin/python3 -m pytest tests/test_s1_aggregate.py -v
"""

import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE_PATH = os.path.join(REPO, "gates", "thresholds.json")

RULE_COMMITTED = "2026-10-20"
RULE_NAME = ("the learner-rule arm: posterior-sampled weights at temperature "
             "lambda")

# ---- fixed by the rule; none of these is a choice made here -----------------
ALPHA = 0.05
N_PERM = 10_000
PERM_SEED = 0
N_FOUNDERS = 20
CHAINS_PER_FOUNDER = 2          # design_amendment_2026_11_08.chains_per_founder
CHANCE_LEVEL = 0.05             # balanced accuracy over 20 founders
LAMBDA_GRID = (100.0, 10.0, 1.0)
SIGMA_MULTIPLIERS = (4.0, 1.0)  # 4 x sigma_match before sigma_match
# cell_coverage: "lambda 100, 10, 1, and within each lambda 4x sigma_match
# before sigma_match"
COMMITTED_CELLS = tuple((lam, mult) for lam in LAMBDA_GRID
                        for mult in SIGMA_MULTIPLIERS)

GATE_NAMES = ("A", "B", "C")

# ---- verdict vocabulary -----------------------------------------------------
PASSES = "PASSES"
FAILS = "FAILS"
PARTIAL = "PARTIAL"
NOT_EVALUABLE = "NOT EVALUABLE"
BROKEN = "BROKEN"
WITHDRAWN = "WITHDRAWN"

GATE_PASSES = PASSES
GATE_FAILS = FAILS
GATE_NOT_EVALUABLE = NOT_EVALUABLE

PRECONDITION_VERDICTS = ("PASS", "BROKEN", "UNRUN")
INERTNESS_VERDICTS = PRECONDITION_VERDICTS + ("READING-DEPENDENT",)

EXACTNESS = "precondition_sampler_exactness"
GGN = "precondition_categorical_ggn_exactness"
INERTNESS = "precondition_inertness"

# status_of_an_unrun_precondition.required_wording, verbatim
GGN_UNRUN_WORDING = ("under a MAP-centred GGN posterior sampler whose "
                     "head-level exactness check has not been run")

NO_TRANSFER_STATEMENT = ("no statement is made about whether the "
                         "sampler/maximizer distinction transfers")

# combination block, third paragraph
GATE_C_NOT_INTERCHANGEABLE = (
    "Gate C is not interchangeable with A and B: A and B ask whether the "
    "founder is forgotten, C asks whether the prior is found.")
GATE_C_WITHOUT = (
    "Gate C is not readable, so the claim this coverage supports is 'the "
    "founder was forgotten' and not 'the prior was found'.")
ALL_THREE_PASS_WARNING = (
    "Gates A, B and C all pass: the chain forgot its founder without "
    "converging to the prior. That is not the theory's prediction and must "
    "not be reported as though it were.")

PROBE_POOL = "probe"
TRAINING_POOL = "training"
POOLS = (PROBE_POOL, TRAINING_POOL)

FLOOR_1 = "headroom_1_degenerate_alphabet"
FLOOR_2 = "headroom_2_chance_level"
FLOOR_3 = "headroom_3_null_scaling_regime"
SINGLE_SIGMA = "across_sigma_contrast_undefined"


class RuleSilent(Exception):
    """The rule and the task do not cover the situation; say so, do not guess."""


class ProbeSurvivorsMissing(Exception):
    """The gates are read on the probe pool and the log has no probe survivor sets."""


# ============================================================ multiplicity

def holm_bonferroni(p_values, alpha=ALPHA):
    """Holm-Bonferroni over the EVALUABLE gates only.

    `p_values` maps gate name to its raw one-sided p-value. A gate that is NOT
    EVALUABLE is excluded by the caller and never appears here, so the family
    size m is the number of evaluable gates and is not always 3. That is not a
    detail: at m = 3 a raw p of 0.02 is not rejected (0.02 > 0.05/3) and at
    m = 2 it is (0.02 < 0.05/2), so excluding a refused gate can turn a gate's
    verdict from PASSES into FAILS.

    Step-down: sort ascending and compare the k-th smallest (0-indexed)
    against alpha / (m - k), rejecting until the first comparison that fails
    and stopping there. The adjusted values are the running maximum of
    min(1, (m - k) * p_k), which enforces monotonicity; without that running
    maximum a large-m gate with a small raw p could be rejected after an
    earlier gate had already stopped the step-down.

    The comparison is strictly `<`, because the committed text reads
    "one-sided permutation p < 0.05 after Holm-Bonferroni".

    Returns m, the ascending order, the adjusted values, the step-down
    thresholds, and two rejection maps that must agree.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    for name, p in p_values.items():
        if p is None or not np.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"gate {name}: p-value must be in [0, 1], got {p!r}")

    order = sorted(p_values, key=lambda name: (p_values[name], name))
    m = len(order)
    adjusted, thresholds, running = {}, {}, 0.0
    for k, name in enumerate(order):
        running = max(running, min(1.0, (m - k) * p_values[name]))
        adjusted[name] = running
        thresholds[name] = alpha / (m - k)

    step_down, still_rejecting = {}, True
    for name in order:
        still_rejecting = still_rejecting and p_values[name] < thresholds[name]
        step_down[name] = still_rejecting

    return {"m": m, "alpha": alpha, "order": order,
            "adjusted": adjusted, "thresholds": thresholds,
            "rejected": {name: adjusted[name] < alpha for name in order},
            "rejected_by_step_down": step_down}


# ================================================================== readings

class GateReading:
    """One gate's reading: either a statistic, or a refusal with its reason.

    Immutable. `refusals` is a tuple of (floor-or-reason name, explanation)
    pairs; a reading is evaluable exactly when it has none.
    """

    __slots__ = ("name", "observed", "p_value", "evaluable", "refusals", "detail")

    def __init__(self, name, observed, p_value, evaluable, refusals, detail):
        if name not in GATE_NAMES:
            raise ValueError(f"gate name must be one of {GATE_NAMES}, got {name!r}")
        refusals = tuple(tuple(r) for r in refusals)
        if evaluable and refusals:
            raise ValueError(f"gate {name}: evaluable reading carries refusals {refusals}")
        if not evaluable and not refusals:
            raise ValueError(f"gate {name}: a refused reading must name its reason")
        if evaluable and p_value is None:
            raise ValueError(f"gate {name}: an evaluable reading needs a p_value")
        if p_value is not None and (not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0):
            raise ValueError(f"gate {name}: p_value must be in [0, 1], got {p_value!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "evaluable", bool(evaluable))
        object.__setattr__(self, "refusals", refusals)
        object.__setattr__(self, "detail", dict(detail))

    def __setattr__(self, *_):
        raise AttributeError("GateReading is immutable")

    def refused(self, kind, reason):
        """A copy of this reading refused for `kind`, keeping what was computed."""
        return GateReading(name=self.name, observed=self.observed, p_value=None,
                           evaluable=False, refusals=self.refusals + ((kind, reason),),
                           detail=dict(self.detail, raw_p_value=self.p_value))


def _null_summary(null_samples):
    null = np.asarray(null_samples, dtype=float)
    return {"null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_min": float(null.min()), "null_max": float(null.max()),
            "n_perm": int(null.size)}


def normalise_gate_result(name, result):
    """A gate result into a GateReading.

    Accepts either the sibling contract's dict (observed, p_value, null_mean,
    null_sd, null_min, null_max, p_value, n_perm, seed, chance_level, ...) or
    an object carrying `observed`, `p_value` and `null_samples`, which is what
    analysis/overlap_stats.cross_arm_permutation_test returns for gate B.
    """
    if hasattr(result, "null_samples"):
        detail = _null_summary(result.null_samples)
        observed, p_value = float(result.observed), float(result.p_value)
    elif isinstance(result, dict):
        if "p_value" not in result:
            raise ValueError(f"gate {name}: result has no p_value: {sorted(result)}")
        detail = {k: v for k, v in result.items() if k not in ("observed", "p_value")}
        observed = result.get("observed")
        p_value = float(result["p_value"])
    elif isinstance(result, GateReading):
        return result
    else:
        raise ValueError(f"gate {name}: cannot read a result of type {type(result).__name__}")
    return GateReading(name=name, observed=observed, p_value=p_value,
                       evaluable=True, refusals=(), detail=detail)


# =================================================================== floors

def _floor(fn, name, *args):
    """Call a floor and read its (fires, reason) pair.

    CONVENTION: the bool is FIRES. True means the floor fired and the gate it
    guards is NOT EVALUABLE. This matches this repository's logged vocabulary
    ("floor fired: headroom_2_shuffled_floor" in
    logs/c8b_shared_founder_torch280.txt) and the sibling modules
    analysis/founder_id.py and analysis/sigma_attractor.py.

    The polarity is the one thing in this module that cannot be allowed to
    drift: read the wrong way round, a floor that refuses a degenerate
    alphabet of 3.2 classes would instead admit gate A's p-value as a verdict
    on exactly the data the floor exists to refuse. So a floor that reports it
    did NOT fire while its reason reads "NOT EVALUABLE" is treated as a
    re-inversion and raises here, rather than becoming a verdict.
    """
    out = fn(*args)
    if not (isinstance(out, tuple) and len(out) == 2
            and isinstance(out[0], (bool, np.bool_)) and isinstance(out[1], str)):
        raise ValueError(f"floor {name} must return (bool fires, str reason), got {out!r}")
    fires, reason = bool(out[0]), out[1]
    if not fires and NOT_EVALUABLE in reason:
        raise ValueError(
            f"floor {name} reports fires=False while its reason reads "
            f"{reason!r}. The convention is fires=True for a fired floor; a "
            "reason naming NOT EVALUABLE beside a False is a polarity "
            "inversion and is refused rather than read as a pass.")
    return fires, reason


# ============================================= defaults for the gate callables

def _default_gate_a(survivor_sets, founder_labels, *, n_perm=N_PERM, seed=PERM_SEED):
    from analysis.founder_id import gate_a_test
    return gate_a_test(survivor_sets, founder_labels, n_perm=n_perm, seed=seed)


def _default_gate_c(survivor_sets, founder_labels, sigma_labels, *,
                    n_perm=N_PERM, seed=PERM_SEED):
    from analysis.sigma_attractor import gate_c_test
    return gate_c_test(survivor_sets, founder_labels, sigma_labels,
                       n_perm=n_perm, seed=seed)


def _default_gate_b(*_args, **_kwargs):
    """RULE SILENT. Gate B's arms cannot be assembled without a decision.

    The rule's gate B is "mean pairwise Jaccard of survivor sets, same-founder
    arm against different-founder arm, computed by analysis/overlap_stats.py",
    but that module's cross_arm_permutation_test(sets_a, sets_b, rng, ...)
    permutes SET labels and returns
    mean_pairwise_jaccard(sets_a) - mean_pairwise_jaccard(sets_b). A
    same-founder versus different-founder contrast is a statement about PAIRS,
    and with 20 founders x 2 chains in one list of 40 sets there is no split
    into two lists of sets that separates same-founder pairs from
    different-founder ones. Several assemblies are possible (one test per
    founder at n = 2 per arm; a pair-label permutation that module does not
    implement; a different statistic) and the rule names none of them.

    So gate B is injected, not defaulted. Hand read_gates a callable with the
    same (survivor_sets, founder_labels, n_perm=, seed=) shape as gate A that
    performs the assembly, and normalise_gate_result will take either a dict
    or the NullTestResult it returns.
    """
    raise RuleSilent(
        "gate B has no default: the rule's same-founder versus "
        "different-founder contrast is a pair-level statement and "
        "analysis/overlap_stats.cross_arm_permutation_test permutes set "
        "labels, so the assembly is undetermined. Pass gate_b= explicitly.")


def _default_floor_1(mean_alphabet_gen25):
    from analysis.founder_id import floor_1_degenerate_alphabet
    return floor_1_degenerate_alphabet(mean_alphabet_gen25)


def _default_floor_2(null_mean, null_sd, chance_level):
    from analysis.founder_id import floor_2_chance_level
    return floor_2_chance_level(null_mean, null_sd, chance_level)


def _default_floor_3(regime_a, regime_b):
    """RULE SILENT. Floor 3 has no default, for the same reason gate B has none.

    The clause reads: "Gate B and gate C must therefore report, for each null
    they use, the number of sets n and an estimate of that first-order
    variance, and are NOT EVALUABLE if the two nulls being compared fall in
    different regimes." It was written after C-4O, where a frequency-matched
    null was set against a uniform one -- a comparison between TWO nulls.

    Neither gate here is shaped that way. Gate C uses one null, the sigma-label
    permutation. Gate B as the rule words it uses one null, the arm-label
    permutation, and its arms are themselves undetermined (see
    _default_gate_b). So "the two nulls being compared" has no referent in
    either gate, and which two collections of sets floor 3 should take the
    regime of is not fixed by the committed text.

    This matters more than an unimplemented default. Until 2026-11-08 this
    function was wired as scaling_regime(survivor_sets) against
    scaling_regime(survivor_sets) -- the same collection twice, so the two
    regimes were always equal and the floor could never fire. A floor that
    cannot fire has not passed, which is the exact pathology this project has
    recorded three times already (C-4 gate B, both C-2R gates, the C-8
    tolerance). Raising is the honest state; a vacuous pass is not.

    Inject floor_3= and scaling_regime= once an amendment says which two
    collections the regimes are taken over.
    """
    raise RuleSilent(
        "floor 3 has no default: the committed clause compares 'the two nulls "
        "being compared', and neither gate B nor gate C as worded uses two "
        "nulls, so the two collections whose regimes are compared are "
        "undetermined. Pass floor_3= explicitly. Note that the previous wiring "
        "compared survivor_sets with itself, which could never fire.")


def _default_scaling_regime(survivor_sets):
    from analysis.sigma_attractor import scaling_regime
    return scaling_regime(survivor_sets)


def read_gates(*, survivor_sets, founder_labels, sigma_labels, mean_alphabet_gen25,
               gate_a=None, gate_b=None, gate_c=None,
               floor_1=None, floor_2=None, floor_3=None, scaling_regime=None,
               n_perm=N_PERM, seed=PERM_SEED, chance_level=CHANCE_LEVEL):
    """The three gate readings, with the headroom floors applied.

    Floor 1 is read before gate A, since a refused gate needs no statistic.
    Floor 2 needs gate A's own null, so it is read after it. Floor 3 compares
    the two nulls gates B and C use and refuses both when they fall in
    different scaling regimes. Gate C additionally needs two sigma values to
    contrast; on a single-sigma coverage it is NOT EVALUABLE and is not
    computed.
    """
    gate_a = _default_gate_a if gate_a is None else gate_a
    gate_b = _default_gate_b if gate_b is None else gate_b
    gate_c = _default_gate_c if gate_c is None else gate_c
    floor_1 = _default_floor_1 if floor_1 is None else floor_1
    floor_2 = _default_floor_2 if floor_2 is None else floor_2
    floor_3 = _default_floor_3 if floor_3 is None else floor_3
    scaling_regime = _default_scaling_regime if scaling_regime is None else scaling_regime

    if not (len(survivor_sets) == len(founder_labels) == len(sigma_labels)):
        raise ValueError(
            f"survivor_sets ({len(survivor_sets)}), founder_labels "
            f"({len(founder_labels)}) and sigma_labels ({len(sigma_labels)}) "
            "must be the same length")

    gates = {}

    # ---- gate A: floor 1, then the statistic, then floor 2
    fires_1, why_1 = _floor(floor_1, FLOOR_1, mean_alphabet_gen25)
    if fires_1:
        gates["A"] = GateReading(name="A", observed=None, p_value=None, evaluable=False,
                                 refusals=((FLOOR_1, why_1),),
                                 detail={"mean_alphabet_gen25": mean_alphabet_gen25})
    else:
        reading = normalise_gate_result(
            "A", gate_a(survivor_sets, founder_labels, n_perm=n_perm, seed=seed))
        reading.detail.setdefault("floor_1_reason", why_1)
        fires_2, why_2 = _floor(floor_2, FLOOR_2,
                                reading.detail["null_mean"], reading.detail["null_sd"],
                                reading.detail.get("chance_level", chance_level))
        gates["A"] = reading.refused(FLOOR_2, why_2) if fires_2 else reading

    # ---- gates B and C: the statistics, then floor 3 across their two nulls
    gates["B"] = normalise_gate_result(
        "B", gate_b(survivor_sets, founder_labels, n_perm=n_perm, seed=seed))

    distinct_sigma = sorted(set(float(s) for s in sigma_labels))
    if len(distinct_sigma) < 2:
        gates["C"] = GateReading(
            name="C", observed=None, p_value=None, evaluable=False,
            refusals=((SINGLE_SIGMA,
                       "the completed coverage holds one sigma multiplier "
                       f"({distinct_sigma}); the within-sigma minus across-sigma "
                       "contrast is undefined"),),
            detail={"sigma_multipliers": distinct_sigma})
    else:
        gates["C"] = normalise_gate_result(
            "C", gate_c(survivor_sets, founder_labels, sigma_labels,
                        n_perm=n_perm, seed=seed))

    regime_b = scaling_regime(survivor_sets)
    regime_c = scaling_regime(survivor_sets)
    fires_3, why_3 = _floor(floor_3, FLOOR_3, regime_b, regime_c)
    if fires_3:
        for name in ("B", "C"):
            if gates[name].evaluable:
                gates[name] = gates[name].refused(FLOOR_3, why_3)
    for name in ("B", "C"):
        gates[name].detail.setdefault("scaling_regime", regime_b)

    return {name: gates[name] for name in GATE_NAMES}


# ============================================================== chain logs

class ChainLog:
    """One chain inside a cell log. Immutable."""

    __slots__ = ("chain_seed", "sizes", "probe_sizes", "survivors_training",
                 "probe_survivors")

    def __init__(self, chain_seed, sizes, probe_sizes, survivors_training, probe_survivors):
        object.__setattr__(self, "chain_seed", int(chain_seed))
        object.__setattr__(self, "sizes", tuple(int(v) for v in sizes))
        object.__setattr__(self, "probe_sizes", tuple(int(v) for v in probe_sizes))
        object.__setattr__(self, "survivors_training", frozenset(survivors_training))
        object.__setattr__(self, "probe_survivors",
                           None if probe_survivors is None else frozenset(probe_survivors))

    def __setattr__(self, *_):
        raise AttributeError("ChainLog is immutable")


class CellLog:
    """One logs/s1_chain_<tag>.json: one founder of one (lambda, sigma) cell."""

    __slots__ = ("path", "arm", "lam", "sigma_mult", "pool_seed", "founder_acc",
                 "founder_acc_flag", "gens", "bottleneck", "chains")

    def __init__(self, path, arm, lam, sigma_mult, pool_seed, founder_acc,
                 founder_acc_flag, gens, bottleneck, chains):
        for key, value in (("path", path), ("arm", arm), ("lam", float(lam)),
                           ("sigma_mult", float(sigma_mult)), ("pool_seed", int(pool_seed)),
                           ("founder_acc", founder_acc),
                           ("founder_acc_flag", bool(founder_acc_flag)),
                           ("gens", int(gens)), ("bottleneck", int(bottleneck)),
                           ("chains", tuple(chains))):
            object.__setattr__(self, key, value)

    def __setattr__(self, *_):
        raise AttributeError("CellLog is immutable")

    @property
    def cell(self):
        return (self.lam, self.sigma_mult)


def _require(payload, key, path):
    if key not in payload:
        raise ValueError(f"{os.path.basename(path)}: chain log has no {key!r}")
    return payload[key]


def load_chain_log(path):
    """One chain log, validated against what the rule fixes in advance."""
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("stage") != "chain":
        raise ValueError(f"{os.path.basename(path)}: stage is "
                         f"{payload.get('stage')!r}, not 'chain'")
    lam = float(_require(payload, "lambda", path))
    mult = float(_require(payload, "sigma_multiplier", path))
    if (lam, mult) not in COMMITTED_CELLS:
        raise ValueError(f"{os.path.basename(path)}: cell lambda {lam:g} sigma x{mult:g} "
                         f"is not on the committed grid {list(COMMITTED_CELLS)}")
    raw = _require(payload, "chains", path)
    if len(raw) != CHAINS_PER_FOUNDER:
        raise ValueError(f"{os.path.basename(path)}: {len(raw)} chains, but the rule "
                         f"commits R = 2 chains per founder exactly")
    chains = [ChainLog(chain_seed=_require(c, "chain_seed", path),
                       sizes=_require(c, "sizes", path),
                       probe_sizes=_require(c, "probe_sizes", path),
                       survivors_training=_require(c, "survivors", path),
                       probe_survivors=c.get("probe_survivors"))
              for c in raw]
    return CellLog(path=path, arm=_require(payload, "arm", path), lam=lam, sigma_mult=mult,
                   pool_seed=_require(payload, "pool_seed", path),
                   founder_acc=payload.get("founder_acc"),
                   founder_acc_flag=payload.get(
                       "founder_acc_departs_by_more_than_0.01", False),
                   gens=_require(payload, "gens", path),
                   bottleneck=_require(payload, "bottleneck", path), chains=chains)


def load_cells(logs_dir, arm="sampler"):
    """Every logs/s1_chain_*.json for one arm, in a deterministic order.

    The cell and the founder are read from the payload, never from the file
    name: --tag is free text and the runner does not constrain it.
    """
    cells = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "s1_chain_*.json"))):
        cell = load_chain_log(path)
        if cell.arm == arm:
            cells.append(cell)
    seen = {}
    for cell in cells:
        key = (cell.cell, cell.pool_seed)
        if key in seen:
            raise ValueError(
                f"cell lambda {cell.lam:g} sigma x{cell.sigma_mult:g} has two logs for "
                f"pool seed {cell.pool_seed}: {os.path.basename(seen[key])} and "
                f"{os.path.basename(cell.path)}")
        seen[key] = cell.path
    return tuple(sorted(cells, key=lambda c: (COMMITTED_CELLS.index(c.cell), c.pool_seed)))


def mean_alphabet_gen25(cells):
    """Floor 1's input: the mean surviving alphabet at generation 25.

    Read from `probe_sizes`, the PROBE pool, because
    design_amendment_2026_11_08.founders_and_pools puts the gates and floor 1
    on the probe pool at probe_seed 9000 and floor 1's threshold of 4.0 was
    derived from a probe-pool number (4.4). The training-pool counterpart at
    this bottleneck is 8.8 and would change the floor's bite without changing
    its number.
    """
    finals = [chain.probe_sizes[-1] for cell in cells for chain in cell.chains]
    if not finals:
        raise ValueError("no chains: the mean generation-25 alphabet is not defined")
    return float(np.mean(finals))


def survivor_sets(cells, pool):
    """(survivor sets, founder labels, sigma labels) for the gate statistics.

    `pool` must be named explicitly. CLAUDE.md forbids quoting one pool's
    number against another's, and the two pools' alphabets differ by about a
    factor of two at this bottleneck, so there is no default.
    """
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    sets, founders, sigmas = [], [], []
    for cell in cells:
        for chain in cell.chains:
            if pool == PROBE_POOL:
                if chain.probe_survivors is None:
                    raise ProbeSurvivorsMissing(
                        f"{os.path.basename(cell.path)} chain {chain.chain_seed} has no "
                        "'probe_survivors'. The gates are read on the probe pool "
                        "(design_amendment_2026_11_08.pool_for_gates_and_floor_1) but "
                        "controls/s1_sampler_arm.run_chain records survivor SETS on the "
                        "training pool only and the probe pool as sizes. The probe-pool "
                        "survivor sets have to be recorded before any gate can be read "
                        "on the pool the rule names.")
                sets.append(chain.probe_survivors)
            else:
                sets.append(chain.survivors_training)
            founders.append(cell.pool_seed)
            sigmas.append(cell.sigma_mult)
    return sets, founders, sigmas


# ================================================================= coverage

def _cell_name(cell):
    return f"lambda {cell[0]:g} sigma x{cell[1]:g}"


def coverage_from_cell_keys(completed_keys, n_founders):
    """Coverage against the six committed cells, in the committed order.

    A cell counts as completed only with all 20 founders. Incomplete and
    unrun cells are both named, as cell_coverage requires, and either sets
    COVERAGE-LIMITED.
    """
    counts = {}
    for key in list(completed_keys) + list(n_founders):
        cell = (float(key[0]), float(key[1]))
        if cell not in COMMITTED_CELLS:
            raise ValueError(f"cell {_cell_name(cell)} is not on the committed grid "
                             f"{list(COMMITTED_CELLS)}")
        counts[cell] = int(n_founders.get(key, n_founders.get(cell, 0)))
    completed, incomplete = [], []
    for cell in COMMITTED_CELLS:
        found = counts.get(cell, 0)
        if found == N_FOUNDERS:
            completed.append([cell[0], cell[1]])
        else:
            incomplete.append({"cell": [cell[0], cell[1]], "n_founders": found})
    return {"completed": completed, "incomplete": incomplete,
            "n_founders_required": N_FOUNDERS,
            "n_cells_committed": len(COMMITTED_CELLS),
            "coverage_limited": bool(incomplete)}


def coverage(cells):
    """Coverage from loaded cell logs."""
    counts = {}
    for cell in cells:
        counts[cell.cell] = counts.get(cell.cell, 0) + 1
    return coverage_from_cell_keys(
        [cell for cell, found in counts.items() if found == N_FOUNDERS], counts)


# ============================================================== preconditions

def read_inertness(payload):
    """The inertness verdict, re-derived from its own cells where they are there.

    inertness_scope fixes the verdict-bearing reading as the fixed-MAP one and
    defines READING-DEPENDENT as "the verdict-bearing reading passed on every
    cell and the per-draw-MAP reading did not". controls/s1_sampler_arm.py
    emits READING-DEPENDENT whenever the two readings merely differ, which also
    covers the opposite case - verdict-bearing failing somewhere while
    per-draw-MAP passes - and the amendment makes that BROKEN. Both readings
    are reported; the disagreement is named, not resolved silently.
    """
    logged = payload.get("verdict") if isinstance(payload, dict) else payload
    cells = payload.get("cells") if isinstance(payload, dict) else None
    if not cells:
        return {"verdict": logged, "verdict_logged": logged,
                "verdict_from_cells": None, "agrees_with_log": True,
                "failing_cells_verdict_bearing": [], "failing_cells_per_draw_map": []}

    def failing(reading):
        return [[float(c["lambda"]), float(c["sigma_multiplier"])] for c in cells
                if not c[reading]["exceeds_maximizer_non_overlapping"]]

    fixed_bad, full_bad = failing("fixed_map"), failing("per_draw_map")
    if fixed_bad:
        derived = "BROKEN"
    elif full_bad:
        derived = "READING-DEPENDENT"
    else:
        derived = "PASS"
    return {"verdict": derived, "verdict_logged": logged, "verdict_from_cells": derived,
            "agrees_with_log": derived == logged,
            "failing_cells_verdict_bearing": fixed_bad,
            "failing_cells_per_draw_map": full_bad}


def _precondition_verdict(value, name, allowed):
    verdict = value.get("verdict") if isinstance(value, dict) else value
    if verdict not in allowed:
        raise ValueError(f"{name}: verdict must be one of {allowed}, got {verdict!r}")
    return verdict


# =============================================================== combination

def consequences_from_rule(rule_path=RULE_PATH):
    """Each gate's committed meaning_if_passed / meaning_if_failed text.

    Read-only. A PARTIAL verdict must carry "only their consequences", so the
    consequence text comes from the committed file rather than being restated.
    """
    with open(rule_path) as handle:
        gates = json.load(handle)["S-1"]["gates"]
    return {name: {"passed": gates[name]["meaning_if_passed"],
                   "failed": gates[name]["meaning_if_failed"]}
            for name in GATE_NAMES}


def _broken(broken_by, preconditions, coverage_info, mc_dropout_reproduced,
            warnings, inertness_detail):
    return {"control": "S-1", "rule_committed": RULE_COMMITTED,
            "verdict": BROKEN, "verdict_before_withdrawal": None,
            "alpha": ALPHA, "preconditions": preconditions,
            "inertness_detail": inertness_detail,
            "broken_by": list(broken_by), "transfer_statement_permitted": False,
            "gates": {}, "readable_gates": [], "refused_gates": [],
            "family": {"m": 0, "alpha": ALPHA, "members": [], "excluded": [],
                       "adjusted": {}, "rejected": {}, "thresholds": {}},
            "floors_fired": [], "consequences": {},
            "coverage": coverage_info,
            "coverage_limited": bool(coverage_info["coverage_limited"]),
            "mc_dropout_reproduced": bool(mc_dropout_reproduced),
            "withdrawn": False, "required_wording": None,
            "warnings": warnings}


def combine(*, exactness, inertness, ggn, gate_readings, coverage_info,
            mc_dropout_reproduced=False, alpha=ALPHA, consequences=None):
    """The combination block: preconditions, then the gates under Holm-Bonferroni.

    Order, from the committed text: the exactness precondition is read first,
    then the inertness precondition; if either fails the arm is BROKEN, no gate
    is read, and no statement is made about whether the sampler/maximizer
    distinction transfers. A categorical-GGN precondition that RAN and FAILED
    is BROKEN too; one that is UNRUN is not a failed one, and the gates are
    read with the required wording stamped on them.
    """
    inertness_detail = inertness if isinstance(inertness, dict) else None
    preconditions = {
        EXACTNESS: _precondition_verdict(exactness, EXACTNESS, PRECONDITION_VERDICTS),
        GGN: _precondition_verdict(ggn, GGN, PRECONDITION_VERDICTS),
        INERTNESS: _precondition_verdict(inertness, INERTNESS, INERTNESS_VERDICTS),
    }
    warnings = []

    # RULE SILENT. status_of_an_unrun_precondition covers the categorical-GGN
    # check only. Nothing in the entry says what an unrun exactness or
    # inertness precondition licenses, and both are named as read before any
    # gate, so this stops instead of inventing a verdict.
    for name in (EXACTNESS, INERTNESS):
        if preconditions[name] == "UNRUN":
            raise RuleSilent(
                f"{name} has no verdict. status_of_an_unrun_precondition decides the "
                f"unrun case for {GGN} only; the rule does not say whether a gate may be "
                f"read while {name} is unrun, and it is named as read before any gate. "
                "Needed: either a run of that precondition, or a decision filed in "
                "gates/thresholds.json before any gate is read.")

    if inertness_detail and not inertness_detail.get("agrees_with_log", True):
        warnings.append(
            f"{INERTNESS} disagrees with its own cells: the log reads "
            f"{inertness_detail['verdict_logged']!r} and the cells give "
            f"{inertness_detail['verdict_from_cells']!r} under "
            "inertness_scope.reading_dependent. Both are reported; neither is dropped.")

    broken_by = [name for name in (EXACTNESS, INERTNESS, GGN)
                 if preconditions[name] == "BROKEN"]
    if broken_by:
        warnings.append(
            f"{', '.join(broken_by)} failed: the arm is BROKEN, no gate is read, and "
            f"{NO_TRANSFER_STATEMENT}. A BROKEN verdict is reported as BROKEN; it is "
            "not reported as a null result.")
        if mc_dropout_reproduced:
            # RULE SILENT, resolved and disclosed: "If MC dropout reproduces the
            # finding, every verdict is withdrawn regardless of the above" cannot
            # withdraw a BROKEN arm, because no gate was read and there is no
            # finding to reproduce. BROKEN is kept, as it is the more specific
            # non-claim ("the sampler was not a sampler"), and the reproduction is
            # recorded beside it.
            warnings.append(
                "MC dropout reproduced the finding, recorded but not applied: the arm is "
                "BROKEN, no gate was read, and there is no verdict to withdraw.")
        return _broken(broken_by, preconditions, coverage_info, mc_dropout_reproduced,
                       warnings, inertness_detail)

    if preconditions[INERTNESS] == "READING-DEPENDENT":
        warnings.append(
            f"{INERTNESS} is READING-DEPENDENT: the verdict-bearing fixed-MAP reading "
            "passed on every cell and the per-draw-MAP reading did not. Per "
            "inertness_scope this is NOT BROKEN; the gates are read and the difference "
            "between the two readings is disclosed.")

    required_wording = GGN_UNRUN_WORDING if preconditions[GGN] == "UNRUN" else None
    if required_wording:
        warnings.append(
            f"{GGN} has not been run. Every verdict below holds "
            f'"{GGN_UNRUN_WORDING}", and the absence is named with the result.')

    # ---- the gates, and Holm-Bonferroni over the evaluable ones
    #
    # All three or none. A gate missing from the mapping would otherwise appear
    # in neither the readable nor the refused list, and a verdict that silently
    # drops a gate is the one shape a PARTIAL must never take: the rule
    # requires it to name which gates were readable. An empty mapping is the
    # legitimate "no completed cell, nothing to read" case.
    if gate_readings and set(gate_readings) != set(GATE_NAMES):
        raise ValueError(
            f"gate_readings must hold every gate {GATE_NAMES} or be empty, got "
            f"{sorted(gate_readings)}. A refused gate is a reading with its floor "
            "named, not an absent key.")
    readings = {name: normalise_gate_result(name, gate_readings[name])
                for name in GATE_NAMES if name in gate_readings}
    readable = [name for name in GATE_NAMES
                if name in readings and readings[name].evaluable]
    refused = [name for name in GATE_NAMES
               if name in readings and not readings[name].evaluable]
    holm = holm_bonferroni({name: readings[name].p_value for name in readable},
                           alpha=alpha)

    gates, floors_fired, carried = {}, [], {}
    for name in GATE_NAMES:
        if name not in readings:
            continue
        reading = readings[name]
        if reading.evaluable:
            verdict = GATE_FAILS if holm["rejected"][name] else GATE_PASSES
        else:
            verdict = GATE_NOT_EVALUABLE
        gates[name] = {
            "verdict": verdict, "observed": reading.observed,
            "p_value": reading.p_value,
            "adjusted_p_value": holm["adjusted"].get(name),
            "holm_threshold": holm["thresholds"].get(name),
            "refusals": [{"kind": kind, "reason": reason}
                         for kind, reason in reading.refusals],
            "detail": reading.detail,
        }
        for kind, reason in reading.refusals:
            floors_fired.append({"gate": name, "floor": kind, "reason": reason})
        if consequences and verdict in (GATE_PASSES, GATE_FAILS):
            carried[name] = consequences[name][
                "passed" if verdict == GATE_PASSES else "failed"]

    verdicts = [gates[name]["verdict"] for name in GATE_NAMES if name in gates]
    n_eval = len(readable)
    if n_eval == len(GATE_NAMES):
        verdict = FAILS if GATE_FAILS in verdicts else PASSES
    elif n_eval == 0:
        verdict = NOT_EVALUABLE
    else:
        verdict = PARTIAL

    if verdict == PARTIAL:
        warnings.append(
            f"PARTIAL on gates {', '.join(readable)}; refused: "
            f"{', '.join(refused) or 'none'}. Only the readable gates' consequences "
            f"are carried. {GATE_C_NOT_INTERCHANGEABLE}")
        if "C" in refused and all(gates.get(n, {}).get("verdict") == GATE_PASSES
                                  for n in ("A", "B")):
            warnings.append(GATE_C_WITHOUT)
    if verdict == PASSES:
        warnings.append(ALL_THREE_PASS_WARNING)
    if coverage_info["coverage_limited"]:
        not_run = [c["cell"] for c in coverage_info["incomplete"]]
        warnings.append(
            f"COVERAGE-LIMITED: {len(coverage_info['completed'])} of "
            f"{coverage_info['n_cells_committed']} committed cells completed. The "
            f"verdict is read on those cells only; not completed: "
            f"{', '.join(_cell_name(c) for c in not_run)}. Holm-Bonferroni stays "
            "across gates A, B and C within the completed coverage and is not "
            "extended across cells.")

    out = {"control": "S-1", "rule_committed": RULE_COMMITTED,
           "verdict": verdict, "verdict_before_withdrawal": None,
           "alpha": alpha, "preconditions": preconditions,
           "inertness_detail": inertness_detail,
           "broken_by": [], "transfer_statement_permitted": verdict != NOT_EVALUABLE,
           "gates": gates, "readable_gates": readable, "refused_gates": refused,
           "family": {"m": holm["m"], "alpha": alpha, "members": readable,
                      "excluded": refused, "adjusted": holm["adjusted"],
                      "rejected": holm["rejected"], "thresholds": holm["thresholds"]},
           "floors_fired": floors_fired, "consequences": carried,
           "coverage": coverage_info,
           "coverage_limited": bool(coverage_info["coverage_limited"]),
           "mc_dropout_reproduced": bool(mc_dropout_reproduced),
           "withdrawn": False, "required_wording": required_wording,
           "warnings": warnings}

    if mc_dropout_reproduced:
        out["warnings"] = warnings + [
            "MC dropout reproduced the arm's result, so the result is attributable to "
            "injected noise rather than to posterior sampling: every verdict above is "
            f"withdrawn regardless of its p-value. Withdrawn verdict: {verdict}."]
        out["verdict_before_withdrawal"] = verdict
        out["verdict"] = WITHDRAWN
        out["withdrawn"] = True
        out["transfer_statement_permitted"] = False
    return out


# ================================================================== report

_LABEL = 17


def _line(label, value):
    return f"{label.ljust(_LABEL)}: {value}"


def _fmt(value, spec="{:.4f}"):
    return "n/a" if value is None else spec.format(value)


def render_report(verdict):
    """The verdict as a logs/*.txt verdict block, in this repository's format."""
    cov = verdict["coverage"]
    out = [f"S-1 rule committed {verdict['rule_committed']}: {RULE_NAME}", ""]

    out.append("preconditions:")
    for name, value in verdict["preconditions"].items():
        out.append(f"  {name.ljust(42)}: {value}")
    if verdict["required_wording"]:
        out.append(f'  required wording: every verdict below holds '
                   f'"{verdict["required_wording"]}"')
    detail = verdict.get("inertness_detail")
    if detail and not detail.get("agrees_with_log", True):
        out.append(f"  inertness log says {detail['verdict_logged']!r}; its cells give "
                   f"{detail['verdict_from_cells']!r} (both reported)")
    out.append("")

    done = ", ".join(_cell_name(c) for c in cov["completed"]) or "none"
    not_run = ", ".join(
        _cell_name(c["cell"]) + (f" ({c['n_founders']} of {cov['n_founders_required']} founders)"
                                 if c["n_founders"] else "")
        for c in cov["incomplete"]) or "none"
    out.append(_line("cells completed", done))
    out.append(_line("cells not run", not_run))
    n_done, n_all = len(cov["completed"]), cov["n_cells_committed"]
    out.append(_line("coverage", f"COVERAGE-LIMITED ({n_done} of {n_all} cells)"
                     if verdict["coverage_limited"] else f"complete ({n_all} of {n_all} cells)"))
    out.append("")

    out.append("=== S-1 GATES ===")
    if not verdict["gates"]:
        out.append("no gate read")
    else:
        family = verdict["family"]
        out.append(f"Holm-Bonferroni family: {family['m']} evaluable gates "
                   f"({', '.join(family['members']) or 'none'}); excluded: "
                   f"{', '.join(family['excluded']) or 'none'}")
        for name, gate in verdict["gates"].items():
            if gate["verdict"] == GATE_NOT_EVALUABLE:
                kinds = ", ".join(r["kind"] for r in gate["refusals"])
                out.append(f"gate {name}  {GATE_NOT_EVALUABLE}   floor fired: {kinds}")
                for r in gate["refusals"]:
                    # Verbatim, because the floor's own words are the record.
                    out.append(f"          {r['kind']}: {r['reason']}")
            else:
                out.append(
                    f"gate {name}  observed {_fmt(gate['observed'])}  "
                    f"p {_fmt(gate['p_value'], '{:.4g}')}  "
                    f"adjusted {_fmt(gate['adjusted_p_value'], '{:.4g}')}  "
                    f"(alpha/{round(ALPHA / gate['holm_threshold'])})  {gate['verdict']}")
    out.append("")
    out.append(f"OUTCOME: {verdict['verdict']}")
    if verdict["withdrawn"]:
        out.append(f"MC dropout reproduced the finding; withdrawn verdict: "
                   f"{verdict['verdict_before_withdrawal']}")
    if verdict["broken_by"]:
        out.append(f"broken by: {', '.join(verdict['broken_by'])}")

    if verdict["consequences"]:
        out.append("")
        out.append("consequences carried:")
        for name, text in verdict["consequences"].items():
            out.append(f"  gate {name}: {text}")
    if verdict["warnings"]:
        out.append("")
        out.append("notes:")
        for note in verdict["warnings"]:
            out.append(f"  - {note}")
    return "\n".join(out)


# =============================================================== entry point

def aggregate(logs_dir, *, exactness, inertness, ggn, gate_b,
              mc_dropout_reproduced=False, arm="sampler", pool=PROBE_POOL,
              gate_a=None, gate_c=None, floor_1=None, floor_2=None, floor_3=None,
              scaling_regime=None, consequences=None, alpha=ALPHA):
    """Load the chain logs of one arm, read the gates, combine, render.

    Returns (verdict dict, report text). `gate_b` has no default; see
    _default_gate_b for why. The gates are read on the completed cells only.
    """
    cells = load_cells(logs_dir, arm=arm)
    cov = coverage(cells)
    completed = {tuple(c) for c in cov["completed"]}
    used = tuple(cell for cell in cells if cell.cell in completed)

    gates = {}
    if used:
        sets, founders, sigmas = survivor_sets(used, pool=pool)
        gates = read_gates(survivor_sets=sets, founder_labels=founders,
                           sigma_labels=sigmas,
                           mean_alphabet_gen25=mean_alphabet_gen25(used),
                           gate_a=gate_a, gate_b=gate_b, gate_c=gate_c,
                           floor_1=floor_1, floor_2=floor_2, floor_3=floor_3,
                           scaling_regime=scaling_regime)
    result = combine(exactness=exactness, inertness=inertness, ggn=ggn,
                     gate_readings=gates, coverage_info=cov,
                     mc_dropout_reproduced=mc_dropout_reproduced, alpha=alpha,
                     consequences=consequences)
    return result, render_report(result)
