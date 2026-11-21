"""The IJPE week: the canonical order stream with the world around it.

``casim.sources.order_stream`` reads the orders and lines; this source adds
what the case says happens to them and no decision controls, each from one
parameter of ``source`` in ``scenario.yaml``:

- ``disruptions``: named disruptions, applied first in the order of their
  names; variants add them. ``type`` is one of
  - ``order_move``: the selected orders arrive at ``at`` (optionally due at
    ``due``), and no longer at their release;
  - ``order_injection``: ``copies`` copies of ``templates/<template>.csv``
    arrive at ``at`` with ids from ``first_id`` (optionally ``due`` and
    ``attributes`` such as the store);
  - ``due_change``: at ``at`` a ``TruckDisruption`` makes the selected orders
    due at ``due``; with ``only_earlier`` (default) an order already due earlier
    keeps its deadline;
- ``due_dates``: an order without a due date gets its store's slot of the
  synthetic dispatch plan (docs/ijpe/departure_fallback.md, ``store_slot``)
  minus the staging time, on its day;
- ``roster``: per day, ``pickers`` pickers arrive at the shift start, take the
  breaks and leave at the shift end (each departure waits for the running tour);
- ``times``: the realized line and setup times drawn from the fit with ``seed``,
  and the pickers plan with the means.

Orders are selected by ``{list: <name>}`` (``lists.csv``) or ``{ids: [...]}``,
optionally only those released on a ``day``. Every order a disruption moves,
adds or changes is marked affected. Files lie with the canonical tables.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from ware_ops_algos.domain_models import (
    Article, Location, Order, OrderPosition, RecordedLine, RecordedPickingPlan, RecordedTour,
)

from casim.domain_objects.sim_domain import SimWarehouseDomain
from casim.events.base_events import Event
from casim.events.operational_events import (
    OrderArrival, OrderStreamClosed, PickerArrival, PickerDeparture, TruckDisruption,
)
from casim.sources.order_stream import OrderStreamSource
from scenarios.ijpe.source import apply_fitted_times, store_slot

DAY_S = 86400.0
DISRUPTIONS = ("order_move", "order_injection", "due_change")


def clock(text) -> float:
    """Seconds after midnight of ``HH:MM``."""
    hours, minutes = str(text).split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0


class ScenarioSource:
    def __init__(self, data_dir: Path, *, epoch: str, tables: str, pickers: Mapping,
                 roster: Mapping, due_dates: Mapping, times: Mapping,
                 disruptions: Mapping | None = None):
        self.stream = OrderStreamSource(data_dir, epoch=epoch, pickers=pickers, tables=tables)
        self.data_dir, self.epoch = self.stream.data_dir, self.stream.epoch
        self.roster, self.due_dates, self.times = dict(roster), dict(due_dates), dict(times)
        self.disruptions = dict(disruptions or {})
        for name, rule in self.disruptions.items():
            if rule.get("type") not in DISRUPTIONS:
                raise ValueError(f"disruption {name}: type must be one of {DISRUPTIONS}")
        self.order_attributes = dict(self.stream.order_attributes)
        self.affected_order_ids: frozenset[int] = frozenset()
        self._events: list[Event] = []

    def at(self, when) -> float:
        return (pd.Timestamp(when) - self.epoch).total_seconds()

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

    def _order_move(self, rule, arrivals) -> set[int]:
        time, chosen = self.at(rule["at"]), self._selected(rule["orders"], arrivals)
        for order_id in chosen:
            order = arrivals[order_id].order
            order.order_date = time
            if "due" in rule:
                order.due_date = self.at(rule["due"])
            arrivals[order_id] = OrderArrival(time, order)
        return set(chosen)

    def _order_injection(self, rule, arrivals) -> set[int]:
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
        tours = list(self.domain.information.processes["recorded_plan"].tours)
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
        self.domain.information.processes["recorded_plan"] = RecordedPickingPlan(tuple(tours))
        return set(ids)

    def _due_change(self, rule, arrivals) -> set[int]:
        due, only_earlier = self.at(rule["due"]), rule.get("only_earlier", True)
        chosen = [i for i in self._selected(rule["orders"], arrivals)
                  if not only_earlier or arrivals[i].order.due_date is None
                  or arrivals[i].order.due_date > due]
        self._events.append(TruckDisruption(self.at(rule["at"]), chosen, due))
        return set(chosen)

    def initial_state(self) -> SimWarehouseDomain:
        self.domain = self.stream.initial_state()
        arrivals = {e.order.order_id: e for e in self.stream.events()
                    if isinstance(e, OrderArrival)}
        self._events = []
        affected: set[int] = set()
        # Orders first: moved and injected orders get their slot like any other.
        for rule in self.disruptions.values():
            if rule["type"] != "due_change":
                affected |= getattr(self, f"_{rule['type']}")(rule, arrivals)

        # Due dates: the store's slot minus staging, on the order's day.
        hours = list(self.due_dates["hours"])
        staging = float(self.due_dates["staging_minutes"]) * 60
        for order_id, arrival in arrivals.items():
            if arrival.order.due_date is None:
                midnight = np.floor(arrival.time / DAY_S) * DAY_S
                store = self.order_attributes.get(order_id, {}).get(self.due_dates["attribute"])
                arrival.order.due_date = (midnight + hours[store_slot(store, len(hours))] * 3600.0
                                          - staging)
        for rule in self.disruptions.values():
            if rule["type"] == "due_change":
                affected |= self._due_change(rule, arrivals)

        # Presence: the roster.
        days, counts = list(self.roster["days"]), self.roster["pickers"]
        counts = [int(counts)] * len(days) if isinstance(counts, int) else [int(c) for c in counts]
        if len(counts) != len(days) or max(counts) > len(self.domain.resources.resources):
            raise ValueError("roster pickers: one number or one per day, "
                             "at most source.pickers.count")
        start, end = (clock(t) for t in self.roster["shift"])
        breaks = [(clock(a), clock(b)) for a, b in self.roster.get("breaks", ())]
        if not start < end or any(not start <= a < b <= end for a, b in breaks):
            raise ValueError("roster: the shift must end after it starts, breaks lie inside it")
        for day, count in zip(days, counts):
            midnight = self.at(day)
            for picker in range(count):
                self._events.append(PickerArrival(midnight + start, picker))
                self._events += [PickerDeparture(midnight + a, picker, duration=b - a)
                                 for a, b in breaks]
                self._events.append(PickerDeparture(midnight + end, picker))

        # Times: drawn from the fit; the pickers plan with the means.
        fit = json.loads((self.data_dir / self.times["fit"]).read_text())
        apply_fitted_times(self.domain, fit, int(self.times["seed"]),
                           int(self.epoch.strftime("%Y%m%d")),
                           [a.order for a in arrivals.values()],
                           [p.id for p in self.domain.resources.resources],
                           float(self.times.get("first_setup_s", np.mean(fit["setup_s"]))))
        means = [float(np.mean(pool)) for pool in fit["line_s"]]
        for picker in self.domain.resources.resources:
            picker.time_per_pick = float(np.mean(means))

        self.affected_order_ids = frozenset(affected)
        self._events += arrivals.values()
        self._events.append(OrderStreamClosed(max(a.time for a in arrivals.values())))
        self._events.sort(key=lambda event: event.time)
        return self.domain

    def events(self) -> Iterator[Event]:
        return iter(self._events)
