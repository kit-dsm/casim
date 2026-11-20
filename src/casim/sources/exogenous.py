"""Exogenous rules: what the world does to a scenario, declared in its configuration.

A source says which orders exist and what the warehouse is (W_0 and the
order arrivals). The exogenous rules of ``scenario.yaml`` say what happens to
it that no decision controls: when people work, when trucks leave, which
orders a disruption moves, adds or makes urgent, and how long work really
takes. They are a named mapping, so a variant adds a rule by naming it::

    exogenous:
      roster: {kind: roster, days: ["2026-04-13"], shift: ["05:30", "14:30"],
               pickers: 17, breaks: [["08:30", "09:00"]]}
      due: {kind: dispatch_slots, hours: [10, 12, 14, 16], staging_minutes: 30,
            attribute: customer}
      truck: {kind: due_change, orders: {list: S1}, at: "2026-04-13 07:30",
              due: "2026-04-13 10:00"}

``ExogenousSource`` wraps any source that has an ``epoch`` (the date its time
zero is the midnight of) and applies the rules in a fixed order of phases,
within a phase in the order of their names in the mapping:

1. orders: ``order_move`` (selected orders arrive at another time, with a
   new due date) and ``order_injection`` (copies of a template order arrive);
2. due dates: ``dispatch_slots`` (orders without a due date get one from a
   slot rule on an attribute) and ``due_change`` (a ``TruckDisruption`` sets
   the due date of selected orders at a time);
3. presence: ``roster`` (pickers per day arrive at the shift start, take the
   breaks and leave at the shift end; each departure waits for the running
   tour);
4. times: ``process_times`` (the realized line and setup times, drawn from a
   fit of empirical times, and the pickers' planning means).

Files a rule names are read from the source's data folder (``data_dir`` of the
source, else the scenario's ``data/``). Orders are selected by
``{list: <name>}`` (``lists.csv``, ``list,order_id``) or ``{ids: [...]}``, optionally only those released on a
``day``. Every order a rule moves, adds or changes is marked
(``affected_order_ids``). The order stream closes after the last arrival.
Rules only add or rewrite source events; they never touch the engine.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from ware_ops_algos.domain_models import (
    Article, ClassPickTime, Location, Order, OrderPosition, RecordedLine, RecordedPickingPlan,
    RecordedTour,
)

from casim.domain_objects.process_truth import drawn
from casim.events.base_events import Event
from casim.events.operational_events import (
    OrderArrival, OrderStreamClosed, PickerArrival, PickerDeparture, TruckDisruption,
)

DAY_S = 86400.0
PHASES = (("order_move", "order_injection"), ("dispatch_slots", "due_change"),
          ("roster",), ("process_times",))
FIELDS = {
    "order_move": ({"orders", "at"}, {"due"}),
    "order_injection": ({"template", "copies", "first_id", "at"}, {"due", "attributes"}),
    "dispatch_slots": ({"hours", "staging_minutes", "attribute"}, set()),
    "due_change": ({"orders", "at", "due"}, {"only_earlier"}),
    "roster": ({"days", "shift", "pickers"}, {"breaks"}),
    "process_times": ({"fit", "seed"}, {"first_setup_s"}),
}


def validate(rules: Mapping[str, Mapping]) -> None:
    """Each rule names a known kind and exactly its fields; raise otherwise."""
    for name, rule in rules.items():
        kind = rule.get("kind") if isinstance(rule, Mapping) else None
        if kind not in FIELDS:
            raise ValueError(f"exogenous {name}: unknown kind {kind!r}; allowed: {sorted(FIELDS)}")
        required, optional = FIELDS[kind]
        given = set(rule) - {"kind"}
        if required - given:
            raise ValueError(f"exogenous {name} ({kind}) lacks {sorted(required - given)}")
        if given - required - optional:
            raise ValueError(f"exogenous {name} ({kind}) has unknown {sorted(given - required - optional)}")


def clock(text) -> float:
    """Seconds after midnight of ``HH:MM``."""
    hours, minutes = str(text).split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0


def slot(value, count: int) -> int:
    """A neutral, reproducible slot index of an attribute value (SHA-256)."""
    digest = sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


class ExogenousSource:
    """A source with the exogenous rules of the configuration applied."""

    def __init__(self, base, rules: Mapping[str, Mapping], data_dir: Path):
        validate(rules)
        if getattr(base, "epoch", None) is None:
            raise ValueError("exogenous rules need a source with an epoch")
        # Files a rule names (lists, templates, fits) lie with the source's tables.
        self.base, self.rules = base, dict(rules)
        self.data_dir = Path(getattr(base, "data_dir", data_dir))
        self.epoch = pd.Timestamp(base.epoch)
        self.order_attributes = dict(getattr(base, "order_attributes", {}))
        self.affected_order_ids = frozenset(getattr(base, "affected_order_ids", ()))
        self._events: list[Event] = []

    def at(self, when) -> float:
        return (pd.Timestamp(when) - self.epoch).total_seconds()

    def day_at(self, day, time) -> float:
        return self.at(day) + clock(time)

    # Selection -----------------------------------------------------------

    def _selected(self, selector: Mapping, arrivals: dict[int, OrderArrival]) -> list[int]:
        if "list" in selector:
            lists = pd.read_csv(self.data_dir / "lists.csv")
            ids = set(lists.loc[lists["list"] == selector["list"], "order_id"].astype(int))
        else:
            ids = {int(i) for i in selector["ids"]}
        chosen = [i for i in sorted(ids) if i in arrivals]
        if "day" in selector:
            low = self.at(selector["day"])
            chosen = [i for i in chosen if low <= arrivals[i].time < low + DAY_S]
        return chosen

    # Rules ---------------------------------------------------------------

    def _order_move(self, rule, arrivals):
        time, chosen = self.at(rule["at"]), self._selected(rule["orders"], arrivals)
        for order_id in chosen:
            order = arrivals[order_id].order
            order.order_date = time
            if "due" in rule:
                order.due_date = self.at(rule["due"])
            arrivals[order_id] = OrderArrival(time, order)
        return set(chosen)

    def _order_injection(self, rule, arrivals):
        template = pd.read_csv(self.data_dir / "templates" / f"{rule['template']}.csv")
        time = self.at(rule["at"])
        ids = [int(rule["first_id"]) + k for k in range(int(rule["copies"]))]
        if set(ids) & set(arrivals):
            raise ValueError(f"injected order ids {ids} are taken")
        storage, articles = self.domain.storage, self.domain.articles
        stored = {(loc.article_id, (loc.x, loc.y)) for loc in storage.locations}
        known = {a.article_id for a in articles.articles}
        lines = [(int(r.article), int(r.quantity), (float(r.x), float(r.y)))
                 for r in template.itertuples()]
        for article, _, node in lines:
            if (article, node) not in stored:
                stored.add((article, node))
                storage.locations.append(Location(x=node[0], y=node[1], article_id=article,
                                                  amount=10 ** 9))
            if article not in known:
                known.add(article)
                articles.articles.append(Article(article_id=article))
        storage.build_article_location_mapping()
        plan = self.domain.information.processes.get("recorded_plan")
        tours = list(plan.tours) if plan is not None else []
        for order_id in ids:
            order = Order(order_id=order_id, order_date=time,
                          due_date=self.at(rule["due"]) if "due" in rule else None)
            order.order_positions.extend(OrderPosition(order_number=order_id, article_id=a,
                                                       amount=q) for a, q, _ in lines)
            arrivals[order_id] = OrderArrival(time, order)
            self.order_attributes[order_id] = dict(rule.get("attributes", {}))
            tours.append(RecordedTour(tour_id=max((t.tour_id for t in tours), default=-1) + 1,
                                      picker_id=-1, lines=tuple(
                                          RecordedLine(order_id, a, node) for a, _, node in lines)))
        if plan is not None:
            self.domain.information.processes["recorded_plan"] = RecordedPickingPlan(tuple(tours))
        return set(ids)

    def _dispatch_slots(self, rule, arrivals):
        hours, staging = list(rule["hours"]), float(rule["staging_minutes"]) * 60
        for order_id, arrival in arrivals.items():
            order = arrival.order
            if order.due_date is None:
                midnight = np.floor(arrival.time / DAY_S) * DAY_S
                value = self.order_attributes.get(order_id, {}).get(rule["attribute"])
                order.due_date = midnight + hours[slot(value, len(hours))] * 3600.0 - staging
        return set()

    def _due_change(self, rule, arrivals):
        due, only_earlier = self.at(rule["due"]), rule.get("only_earlier", True)
        chosen = [i for i in self._selected(rule["orders"], arrivals)
                  if not only_earlier or arrivals[i].order.due_date is None
                  or arrivals[i].order.due_date > due]
        self._events.append(TruckDisruption(self.at(rule["at"]), chosen, due))
        return set(chosen)

    def _roster(self, rule, arrivals):
        days = list(rule["days"])
        counts = rule["pickers"]
        counts = [int(counts)] * len(days) if isinstance(counts, int) else [int(c) for c in counts]
        if len(counts) != len(days):
            raise ValueError("roster pickers: one number, or one per day")
        if max(counts) > len(self.domain.resources.resources):
            raise ValueError(f"roster needs {max(counts)} pickers; the source has "
                             f"{len(self.domain.resources.resources)}")
        start, end = (clock(t) for t in rule["shift"])
        breaks = [(clock(a), clock(b)) for a, b in rule.get("breaks", ())]
        if not start < end or any(not start <= a < b <= end for a, b in breaks):
            raise ValueError("roster: the shift must end after it starts, breaks lie inside it")
        for day, count in zip(days, counts):
            midnight = self.at(day)
            for picker in range(count):
                self._events.append(PickerArrival(midnight + start, picker))
                self._events += [PickerDeparture(midnight + a, picker, duration=b - a)
                                 for a, b in breaks]
                self._events.append(PickerDeparture(midnight + end, picker))
        return set()

    def _process_times(self, rule, arrivals):
        fit = json.loads((self.data_dir / rule["fit"]).read_text())
        pickers = [p.id for p in self.domain.resources.resources]
        self.domain.truth = drawn(fit, int(rule["seed"]), int(self.epoch.strftime("%Y%m%d")),
                                  [a.order for a in arrivals.values()], pickers,
                                  float(rule.get("first_setup_s", np.mean(fit["setup_s"]))))
        means = tuple(float(np.mean(pool)) for pool in fit["line_s"])
        for picker in self.domain.resources.resources:
            picker.tour_setup_time = float(np.mean(fit["setup_s"]))
            picker.time_per_pick = float(np.mean(means))
            picker.pick_time_model = ClassPickTime(tuple(fit["quantity_bounds"]), means)
        return set()

    # Source protocol -----------------------------------------------------

    def initial_state(self):
        self.domain = self.base.initial_state()
        events = list(self.base.events())
        arrivals = {e.order.order_id: e for e in events if isinstance(e, OrderArrival)}
        self._events = [e for e in events if not isinstance(e, (OrderArrival, OrderStreamClosed))]
        affected = set(self.affected_order_ids)
        for phase in PHASES:
            for rule in self.rules.values():
                if rule["kind"] in phase:
                    affected |= getattr(self, f"_{rule['kind']}")(rule, arrivals)
        self.affected_order_ids = frozenset(affected)
        self._events += arrivals.values()
        if arrivals:
            self._events.append(OrderStreamClosed(max(a.time for a in arrivals.values())))
        self._events.sort(key=lambda event: event.time)
        return self.domain

    def events(self) -> Iterator[Event]:
        return iter(self._events)
