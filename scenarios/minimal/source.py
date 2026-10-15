"""The minimal scenario's source: orders written out in ``scenario.yaml``.

Each order has an arrival, a due date and its picks as (aisle, location); each
distinct location stores one article. One picker with an order-bin cart
arrives at ``shift_start_s``; the stream closes at the last arrival.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ware_ops_algos.domain_models import (
    Article, Articles, ArticleType, DimensionType, Location, Order, OrderPosition,
    OrdersDomain, OrderType, PickCart, Resource, Resources, ResourceType,
    StorageLocations, StorageType, WarehouseInfo, WarehouseInfoType,
)

from casim.domain_objects.sim_domain import SimWarehouseDomain, initial_dynamic_info
from casim.events.base_events import Event
from casim.sources.events import shift_start_release
from casim.sources.layout import conventional_layout


class ScenarioSource:
    def __init__(self, data_dir: Path, *, orders: list[dict], shift_start_s: float,
                 capacity_orders: int, n_aisles: int, n_pick_locations: int,
                 depot_aisle: int, walking_speed: float, pick_time_s: float):
        self.orders = orders
        self.shift_start_s = float(shift_start_s)
        self.capacity_orders = int(capacity_orders)
        self.n_aisles = int(n_aisles)
        self.n_pick_locations = int(n_pick_locations)
        self.depot_aisle = int(depot_aisle)
        self.walking_speed = float(walking_speed)
        self.pick_time_s = float(pick_time_s)
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        layout = conventional_layout(
            n_aisles=self.n_aisles, n_pick_locations=self.n_pick_locations,
            start_connection_point=(self.depot_aisle, 0),
            end_connection_point=(self.depot_aisle, 0))
        article_of: dict[tuple[int, int], int] = {}
        orders = []
        for order_id, spec in enumerate(self.orders):
            amounts: dict[int, int] = {}
            for aisle, location in spec["picks"]:
                article = article_of.setdefault((int(aisle), int(location)), len(article_of))
                amounts[article] = amounts.get(article, 0) + 1
            orders.append(Order(
                order_id=order_id, order_date=float(spec["arrival_s"]),
                due_date=float(spec["due_s"]),
                order_positions=[OrderPosition(order_number=order_id, article_id=article,
                                               amount=amount)
                                 for article, amount in amounts.items()]))
        demand: dict[int, int] = {}
        for order in orders:
            for position in order.order_positions:
                demand[position.article_id] = demand.get(position.article_id, 0) + position.amount
        storage = StorageLocations(
            tpe=StorageType.DEDICATED,
            locations=[Location(x=x, y=y, article_id=article, amount=demand[article])
                       for (x, y), article in article_of.items()])
        storage.build_article_location_mapping()
        cart = PickCart(n_dimension=1, capacities=[1], dimensions=[DimensionType.ORDERS],
                        n_boxes=self.capacity_orders, box_can_mix_orders=False)
        picker = Resource(
            id=0, capacity=self.capacity_orders, speed=self.walking_speed,
            time_per_pick=self.pick_time_s, tour_setup_time=0.0, pick_cart=cart,
            available=True, occupied=False,
            current_location=layout.graph_data.start_location)
        self._domain = SimWarehouseDomain(
            problem_class="OBRSP", objective="mean_order_completion_time",
            layout=layout,
            articles=Articles(ArticleType.STANDARD,
                              [Article(article_id=article, weight=1.0) for article in demand]),
            orders=OrdersDomain(OrderType.STANDARD, orders),
            resources=Resources(ResourceType.HUMAN, [picker]),
            storage=storage,
            dynamic_warehouse_info=initial_dynamic_info(),
            warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.ONLINE),
        )
        return self._domain

    def events(self) -> Iterator[Event]:
        return shift_start_release(self._domain, self.shift_start_s)
