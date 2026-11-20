"""A stream of orders from canonical tables: the data a scenario simulates.

A folder of plain tables, written once by a conversion script from whatever
the raw data is (a WMS export, a generator); by default the scenario's
``data/``, or ``tables``, a folder relative to it (several scenarios can share
one data set):

- ``orders.csv``: ``order_id``, ``release`` (date and time, ``YYYY-MM-DD HH:MM[:SS]``),
  optionally ``due`` (same format); every further column is an attribute of
  the order (for example ``customer``), readable by exogenous rules
  (``casim.sources.exogenous``);
- ``lines.csv``: ``order_id``, ``article``, ``quantity``, ``x``, ``y``: one row per
  order line in pick order, at the pick node (``x``, ``y``) of the layout;
- the layout, a grid drawing read by ``casim.sources.grid_layout``;
- optionally ``lists.csv`` (``list``, ``order_id``: named sets of orders) and
  ``templates/<name>.csv`` (``article``, ``quantity``, ``x``, ``y``: order lines
  without an order), which exogenous rules use.

Further columns of ``lines.csv`` (for example the recorded picker and times
of a replay) are left to whoever needs them.

Time is seconds from midnight of ``epoch``. Each order arrives at its release
(``OrderArrival``); the stream closes with the last release. The initial
state holds layout, storage (every article at the nodes its lines name),
articles and ``pickers`` identical pickers, and no order: nothing is known
before it arrives. Pickers are present only through presence events, for
example an exogenous ``roster``. The planner information ``recorded_plan``
holds one tour per order with its lines at their nodes, so that the item
assignment takes each line where the data says it is picked (picker -1: no
recorded picker).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd
from ware_ops_algos.domain_models import (
    Article, Articles, ArticleType, DimensionType, InformationType, Location, ManualPicker, Order,
    OrderPosition, OrdersDomain, OrderType, PickCart, PlannerInformation, RecordedLine,
    RecordedPickingPlan, RecordedTour, Resources, ResourceType, StorageLocations, StorageType,
    WarehouseInfo, WarehouseInfoType,
)

from casim.domain_objects.sim_domain import SimWarehouseDomain, initial_dynamic_info
from casim.events.base_events import Event
from casim.events.operational_events import OrderArrival, OrderStreamClosed
from casim.sources.grid_layout import GridLayout

ORDER_COLUMNS = ("order_id", "release")
LINE_COLUMNS = ("order_id", "article", "quantity", "x", "y")


def seconds(epoch: pd.Timestamp, when) -> float:
    """Seconds from ``epoch`` to a date and time."""
    return (pd.Timestamp(when) - epoch).total_seconds()


def read_lines(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in LINE_COLUMNS if c != "order_id" and c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} lacks columns {missing}")
    return frame


class OrderStreamSource:
    """Orders from ``orders.csv`` and ``lines.csv``, released at their time."""

    def __init__(self, data_dir: Path, *, epoch: str, pickers: Mapping, tables: str = ".",
                 layout: str = "layout.csv"):
        """``pickers``: ``count``, ``speed`` (layout units per second),
        ``time_per_pick`` and ``tour_setup_time`` (seconds; the pickers' time model,
        which an exogenous ``process_times`` may replace)."""
        self.data_dir = (Path(data_dir) / tables).resolve()
        self.epoch = pd.Timestamp(epoch)
        self.layout, _ = GridLayout().build(self.data_dir / layout)
        self.picker_spec = dict(pickers)
        orders = pd.read_csv(self.data_dir / "orders.csv")
        lines = read_lines(self.data_dir / "lines.csv")
        missing = [c for c in ORDER_COLUMNS if c not in orders.columns]
        if missing:
            raise ValueError(f"orders.csv lacks columns {missing}")
        if orders["order_id"].duplicated().any():
            raise ValueError("orders.csv names an order twice")
        unknown = set(lines["order_id"]) - set(orders["order_id"])
        if unknown:
            raise ValueError(f"lines.csv names {len(unknown)} orders not in orders.csv")
        attributes = [c for c in orders.columns if c not in (*ORDER_COLUMNS, "due")]
        self.order_attributes = {int(row.order_id): {c: getattr(row, c) for c in attributes}
                                 for row in orders.itertuples()}
        by_order = {int(order): [(int(r.article), int(r.quantity), (float(r.x), float(r.y)))
                                 for r in group.itertuples()]
                    for order, group in lines.groupby("order_id", sort=False)}
        self.orders: list[Order] = []
        for row in orders.sort_values(["release", "order_id"]).itertuples():
            due = getattr(row, "due", None)
            order = Order(order_id=int(row.order_id), order_date=seconds(self.epoch, row.release),
                          due_date=None if due is None or pd.isna(due) else seconds(self.epoch, due))
            order.order_positions.extend(OrderPosition(order_number=order.order_id, article_id=a,
                                                       amount=q)
                                         for a, q, _ in by_order.get(order.order_id, ()))
            if not order.order_positions:
                raise ValueError(f"order {order.order_id} has no line")
            self.orders.append(order)
        self.nodes = {int(order): [node for _, _, node in positions]
                      for order, positions in by_order.items()}
        self.affected_order_ids: frozenset[int] = frozenset()

    def initial_state(self) -> SimWarehouseDomain:
        stored = sorted({(p.article_id, node) for order in self.orders
                         for p, node in zip(order.order_positions, self.nodes[order.order_id])})
        storage = StorageLocations(tpe=StorageType.DEDICATED, locations=[
            Location(x=node[0], y=node[1], article_id=article, amount=10 ** 9)
            for article, node in stored])
        storage.build_article_location_mapping()
        spec = self.picker_spec
        cart = PickCart(n_dimension=1, capacities=[1], dimensions=[DimensionType.ORDERS],
                        n_boxes=1, box_can_mix_orders=False)
        pickers = [ManualPicker(
            id=picker, tpe=ResourceType.HUMAN, capacity=1, speed=float(spec["speed"]),
            time_per_pick=float(spec["time_per_pick"]),
            tour_setup_time=float(spec["tour_setup_time"]), pick_cart=cart,
            available=True, occupied=False, current_location=self.layout.layout_network.start_node)
            for picker in range(int(spec["count"]))]
        plan = RecordedPickingPlan(tours=tuple(
            RecordedTour(tour_id=index, picker_id=-1, lines=tuple(
                RecordedLine(order.order_id, p.article_id, node)
                for p, node in zip(order.order_positions, self.nodes[order.order_id])))
            for index, order in enumerate(self.orders)))
        return SimWarehouseDomain(
            problem_class="OBRSP", objective="mean_order_completion_time", layout=self.layout,
            articles=Articles(ArticleType.STANDARD, [Article(article_id=article)
                                                     for article in sorted({a for a, _ in stored})]),
            orders=OrdersDomain(OrderType.STANDARD, []),
            resources=Resources(ResourceType.HUMAN, pickers), storage=storage,
            information=PlannerInformation(InformationType.PROCESS_INFORMATION,
                                           {"recorded_plan": plan}),
            dynamic_warehouse_info=initial_dynamic_info(),
            warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.ONLINE),
        )

    def events(self) -> Iterator[Event]:
        events: list[Event] = [OrderArrival(order.order_date, order) for order in self.orders]
        events.append(OrderStreamClosed(max(order.order_date for order in self.orders)))
        return iter(events)
