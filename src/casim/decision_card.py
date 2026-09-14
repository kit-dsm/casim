"""Formal decision-configuration model for CASIM.

DecisionCard describes projection only. ProjectionSpec owns parameter
consumption, runtime facts, and trigger requirements per projection
primitive. ConditionSpec owns condition semantics shared by validation,
runtime evaluation, and explanation. Ranking lives in SolutionRanker;
commitment fields in SchedulingCommitmentPolicy; problem structure in
TAXONOMY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from casim.pipelines.taxonomy import solution_kind


# ── parameter validators ──

def _opt_pos_int(v) -> bool:
    """``limit``: positive integer or None; bool and float are invalid.

    Runtime casts to ``int`` (``items[: int(limit)]``); a float would
    silently truncate and a bool would coerce to 0/1.
    """
    return v is None or (isinstance(v, int) and not isinstance(v, bool) and v > 0)


def _opt_nonneg_num(v) -> bool:
    """``due_horizon_s``: non-negative int/float or None; bool is invalid.

    Runtime casts to ``float`` (``float(now) + float(due_horizon_s)``),
    so both int and float are accepted.
    """
    return v is None or (
        isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
    )


def _nonneg_num(v) -> bool:
    """``congestion_penalty``: non-negative int/float; bool is invalid.

    Runtime casts to ``float`` (``float(cfg.get("congestion_penalty", 0.0))``);
    not optional because the active-tour projection always threads it.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def _pos_int(v) -> bool:
    """Condition count threshold: positive integer; bool/float/None invalid.

    Runtime casts to ``int`` (``actual < int(value)``); a float would
    silently truncate, None would raise ``TypeError``, and a bool would
    coerce to 0/1.
    """
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


PARAMETER_SPECS = {
    "limit": _opt_pos_int,
    "due_horizon_s": _opt_nonneg_num,
    "congestion_penalty": _nonneg_num,
}

# ── projection specs ──

@dataclass(frozen=True)
class ProjectionSpec:
    parameters: frozenset[str] = frozenset()
    facts: frozenset[str] = frozenset()
    fixed_cardinalities: Mapping[str, int] = field(default_factory=dict)
    trigger_capabilities: frozenset[str] = frozenset()

PROJECTION_SPECS: dict[tuple[str, str], ProjectionSpec] = {
    ("orders", "buffered"): ProjectionSpec(parameters=frozenset({"limit"}), facts=frozenset({"orders.count"})),
    ("batches", "buffered"): ProjectionSpec(parameters=frozenset({"limit", "due_horizon_s"}), facts=frozenset({"batches.count"})),
    ("batches", "buffered_and_replannable"): ProjectionSpec(parameters=frozenset({"limit", "due_horizon_s"}), facts=frozenset({"batches.count"})),
    ("resources", "all"): ProjectionSpec(facts=frozenset({"resources.count"})),
    ("resources", "nonactive"): ProjectionSpec(facts=frozenset({"resources.count"})),
    ("resources", "available"): ProjectionSpec(facts=frozenset({"resources.count"})),
    ("resources", "dispatchable"): ProjectionSpec(facts=frozenset({"resources.count"})),
    ("active_tour", "residual"): ProjectionSpec(
        parameters=frozenset({"congestion_penalty"}),
        facts=frozenset({"batches.count", "resources.count"}),
        fixed_cardinalities={"batches.count": 1, "resources.count": 1},
        trigger_capabilities=frozenset({"picker_id", "tour_id", "route_version", "resumes_execution"}),
    ),
}

_BASE_FACTS = frozenset({"dynamic.is_break", "dynamic.n_staged_pallets"})
_QUANTITIES = ("orders", "batches", "resources", "active_tour")

# ── condition specs ──

@dataclass(frozen=True)
class ConditionSpec:
    fact: str
    operator: str
    validate_value: Callable[[object], bool]
    drain_presence: bool = False

    def render(self, key, value) -> str:
        return key if self.operator == "false" else f"{key} {self.operator} {value}"

CONDITION_SPECS: dict[str, ConditionSpec] = {
    "orders": ConditionSpec("orders.count", ">=", _pos_int, drain_presence=True),
    "batches": ConditionSpec("batches.count", ">=", _pos_int, drain_presence=True),
    "pickers": ConditionSpec("resources.count", ">=", _pos_int),
    "not_on_break": ConditionSpec("dynamic.is_break", "false", lambda v: v is not None),
    "dock_capacity": ConditionSpec("dynamic.n_staged_pallets", "<=", _pos_int),
}

@dataclass(frozen=True)
class CompiledCondition:
    key: str
    spec: ConditionSpec
    value: object

# ── DecisionCard ──

@dataclass(frozen=True)
class DecisionCard:
    """What operational state is projected for a binding.  Uses the same
    source vocabulary as StateAdapter.  problem_class and replanning are
    the registry key, not card fields."""
    orders: str | None = None
    batches: str | None = None
    resources: str | None = None
    resource_scope: str | None = None
    active_tour: str | None = None

    def _spec(self, q):
        s = getattr(self, q)
        return PROJECTION_SPECS[(q, s)] if s is not None else None

    def exposure_dict(self, *, limit, due_horizon_s, congestion_penalty) -> dict:
        params = {"limit": limit, "due_horizon_s": due_horizon_s, "congestion_penalty": congestion_penalty}
        result = {}
        for q in _QUANTITIES:
            s = getattr(self, q)
            if s is None:
                continue
            spec = PROJECTION_SPECS[(q, s)]
            cfg = {"source": s}
            for p in spec.parameters:
                cfg[p] = params[p]
            if q == "resources" and self.resource_scope is not None:
                cfg["scope"] = self.resource_scope
            result[q] = cfg
        return result

    def facts(self):
        f = set(_BASE_FACTS)
        for q in _QUANTITIES:
            spec = self._spec(q)
            if spec:
                f |= spec.facts
        return frozenset(f)

    def required_trigger_capabilities(self):
        c = set()
        for q in _QUANTITIES:
            spec = self._spec(q)
            if spec:
                c |= spec.trigger_capabilities
        return frozenset(c)

    def fixed_cardinalities(self):
        card = {}
        for q in _QUANTITIES:
            spec = self._spec(q)
            if spec:
                card.update(spec.fixed_cardinalities)
        return card

    def parameters(self):
        p = set()
        for q in _QUANTITIES:
            spec = self._spec(q)
            if spec:
                p |= spec.parameters
        return frozenset(p)


DECISION_CARDS: dict[tuple[str, str], DecisionCard] = {
    ("OBP", "none"): DecisionCard(orders="buffered", resources="all"),
    ("ORSP", "none"): DecisionCard(batches="buffered", resources="nonactive"),
    ("ORSP", "unstarted"): DecisionCard(batches="buffered_and_replannable", resources="available"),
    ("OBRSP", "none"): DecisionCard(orders="buffered", resources="dispatchable", resource_scope="trigger_if_present"),
    ("OBRP", "none"): DecisionCard(orders="buffered", resources="dispatchable", resource_scope="trigger_if_present"),
    ("ORP", "active"): DecisionCard(active_tour="residual"),
    ("OBRP", "active"): DecisionCard(orders="buffered", active_tour="residual"),
}

DECISION_FIELDS = frozenset({"problem_class", "replanning", "triggers", "requires",
                             "due_horizon_s", "limit", "congestion_penalty", "solver", "commit"})
COSY_SOLVER_FIELDS = frozenset({"type", "objective", "repo", "executor"})

# ── runtime condition evaluation (shared with SimulationEngine) ──

def _fact_value(fact, snapshot):
    d = snapshot.dynamic_warehouse_info
    if fact == "orders.count": return len(snapshot.orders.orders)
    if fact == "batches.count": return len(d.buffered_batches)
    if fact == "resources.count": return len(snapshot.resources.resources)
    if fact == "dynamic.is_break": return d.is_break
    if fact == "dynamic.n_staged_pallets": return d.n_staged_pallets
    raise KeyError(fact)

def evaluate_conditions(conditions, snapshot, *, drain: bool) -> bool:
    """Runtime evaluation driven by ConditionSpec.  Preserves drain behavior."""
    if not conditions:
        return True
    if isinstance(conditions, dict):
        conditions = tuple(CompiledCondition(k, CONDITION_SPECS[k], v) for k, v in conditions.items())
    for cond in conditions:
        spec, value = cond.spec, cond.value
        if drain and spec.drain_presence:
            if spec.fact == "orders.count" and not snapshot.orders.orders: return False
            if spec.fact == "batches.count" and not snapshot.dynamic_warehouse_info.buffered_batches: return False
            continue
        actual = _fact_value(spec.fact, snapshot)
        if spec.operator == ">=" and actual < int(value): return False
        if spec.operator == "<=" and actual > int(value): return False
        if spec.operator == "false" and value and actual: return False
    return True

# ── DecisionCompatibilityMapper ──

class DecisionCompatibilityMapper:
    """Orchestrate V1-V8 against the semantic owners.  Owns no semantics."""

    def __init__(self, cards=None):
        self.cards = cards or DECISION_CARDS

    def resolve(self, problem_class, replanning) -> DecisionCard:
        card = self.cards.get((problem_class, replanning))
        if card is None:
            raise ValueError(f"Unsupported decision binding (problem_class={problem_class!r}, "
                              f"replanning={replanning!r}). Supported: {sorted(self.cards)}")
        return card

    def validate(self, problem_class, replanning, problem_cfg, *, trigger_classes) -> DecisionCard:
        card = self.resolve(problem_class, replanning)
        label = f"{problem_class}/{replanning}"
        self._check_fields(problem_cfg, label)
        self._check_params(card, problem_cfg, label)
        self._check_conditions(card, problem_cfg, label)
        self._check_triggers(card, trigger_classes, label)
        self._check_solver(problem_cfg, label)
        self._check_commit(problem_cfg, label)
        return card

    @staticmethod
    def _keys(cfg):
        return set(cfg.keys() if hasattr(cfg, "keys") else cfg) if cfg is not None else set()

    def _check_fields(self, cfg, label):
        unk = self._keys(cfg) - DECISION_FIELDS
        if unk:
            raise ValueError(f"Decision {label} has unknown field(s): {sorted(unk)}; "
                              f"known: {sorted(DECISION_FIELDS)}")

    def _check_params(self, card, cfg, label):
        allowed = card.parameters()
        for name in ("due_horizon_s", "limit", "congestion_penalty"):
            val = cfg.get(name)
            if val is None and name == "congestion_penalty":
                continue
            if val is not None and name not in allowed:
                raise ValueError(f"Decision {label} does not consume parameter {name!r}")
            if val is not None and not PARAMETER_SPECS[name](val):
                raise ValueError(f"Decision {label} has invalid value for {name!r}: {val!r}")

    def _check_conditions(self, card, cfg, label):
        req = cfg.get("requires")
        if req is None:
            return
        facts, cardinals = card.facts(), card.fixed_cardinalities()
        for key in self._keys(req):
            spec = CONDITION_SPECS.get(key)
            if spec is None:
                raise ValueError(f"Decision {label} has unknown requires key {key!r}; "
                                  f"known: {sorted(CONDITION_SPECS)}")
            if spec.fact not in facts:
                raise ValueError(f"Decision {label} cannot require {key!r}: "
                                  f"fact {spec.fact!r} not available")
            val = req[key]
            if not spec.validate_value(val):
                raise ValueError(f"Decision {label} has invalid value for {key!r}: {val!r}")
            if spec.fact in cardinals and spec.operator == ">=" and int(val) > cardinals[spec.fact]:
                raise ValueError(f"Decision {label} condition {key!r} is impossible: "
                                  f"{spec.fact} is always {cardinals[spec.fact]}")

    def _check_triggers(self, card, trigger_classes, label):
        required = card.required_trigger_capabilities()
        if not required:
            return
        if not trigger_classes:
            raise ValueError(f"Decision {label} requires trigger capabilities "
                              f"{sorted(required)} but no trigger is configured")
        for cls in trigger_classes:
            caps = frozenset(getattr(cls, "decision_capabilities", frozenset()))
            missing = required - caps
            if missing:
                raise ValueError(f"Decision {label} requires {sorted(required)} but "
                                  f"{cls.__name__} provides {sorted(caps)} (missing: {sorted(missing)})")

    def _check_solver(self, cfg, label):
        sc = cfg.get("solver")
        if sc is None:
            return
        from casim.pipelines.solution_ranker import SolutionRanker
        st = str(sc.get("type"))
        if st == "cosy":
            unk = self._keys(sc) - COSY_SOLVER_FIELDS
            if unk:
                raise ValueError(f"Decision {label} cosy solver has unknown field(s): {sorted(unk)}; "
                                  f"known: {sorted(COSY_SOLVER_FIELDS)}")
            pc = label.split("/")[0]
            objs = SolutionRanker.supported_objectives(pc)
            configured = sc.get("objective")
            if not objs:
                if configured is not None:
                    raise ValueError(f"Decision {label} ({solution_kind(pc)}) is not ranked "
                                      f"by objective; solver.objective={configured!r} is ignored")
            else:
                resolved = str(configured if configured is not None else "distance")
                if resolved not in objs:
                    raise ValueError(f"Decision {label} does not support ranking objective "
                                      f"{resolved!r}; supported: {sorted(objs)}")
        elif st != "direct":
            raise ValueError(f"Decision {label} has unknown solver type: {st!r}")

    def _check_commit(self, cfg, label):
        from casim.decision_engine import SchedulingCommitmentPolicy
        cc = cfg.get("commit")
        if cc is None:
            return
        pc = label.split("/")[0]
        if solution_kind(pc) != "SchedulingSolution":
            raise ValueError(f"Decision {label} ({solution_kind(pc)}) cannot use a "
                              f"SchedulingCommitmentPolicy; only scheduling decisions")
        unk = self._keys(cc) - set(SchedulingCommitmentPolicy.FIELD_SPECS)
        if unk:
            raise ValueError(f"Decision {label} commit has unknown field(s): {sorted(unk)}; "
                              f"known: {sorted(SchedulingCommitmentPolicy.FIELD_SPECS)}")
        for fn, validator in SchedulingCommitmentPolicy.FIELD_SPECS.items():
            val = cc.get(fn)
            if val is not None and not validator(val):
                raise ValueError(f"Decision {label} commit has invalid value for {fn!r}: {val!r}")
