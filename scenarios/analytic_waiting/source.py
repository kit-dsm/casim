"""The analytical waiting example's source: one-line orders on a generated layout.

Order ``i`` arrives at ``arrival_times_s[i]`` with one pick at
``pick_locations[i]`` (or at location 1 of ``aisles[i]``). One picker with a
four-bin cart arrives with the first order; the stream closes at the last
arrival. The forecast, ``information``, is the data card's process
information and never enters the event stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from casim.events.base_events import Event
from casim.sources.events import first_arrival_release
from casim.sources.layout import conventional_layout
from casim.domain_objects.sim_domain import SimWarehouseDomain, initial_dynamic_info
from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    DimensionType,
    Location,
    Order,
    OrderPosition,
    OrdersDomain,
    OrderType,
    PickCart,
    Resource,
    Resources,
    ResourceType,
    StorageLocations,
    StorageType,
    WarehouseInfoType,
    WarehouseInfo,
    ExponentialSingleLineUniformLocationOrderStream,
    InformationType,
    PlannerInformation,
)



class ScenarioSource:
    def __init__(self, data_dir: Path, *, arrival_times_s: list[float], pick_time_s: float,
                 information: dict, aisles: list[int] | None = None,
                 pick_locations: list[list[int]] | None = None):
        self.arrival_times_s = arrival_times_s
        self.pick_time_s = float(pick_time_s)
        self.information = information
        self.aisles = aisles
        self.pick_locations = pick_locations
        self._domain: SimWarehouseDomain | None = None

    def initial_state(self) -> SimWarehouseDomain:
        arrivals = [float(value) for value in self.arrival_times_s]
        pick_nodes = ([tuple(int(value) for value in node) for node in self.pick_locations]
                      if self.pick_locations is not None else
                      [(int(aisle), 1) for aisle in self.aisles])
        if len(arrivals) != len(pick_nodes):
            raise ValueError("Each realized order needs one pick location")

        layout = conventional_layout()
        articles = Articles(
            ArticleType.STANDARD,
            [
                Article(article_id=index, weight=1.0)
                for index in range(len(arrivals))
            ],
        )
        storage = StorageLocations(
            tpe=StorageType.DEDICATED,
            locations=[
                Location(
                    x=node[0],
                    y=node[1],
                    article_id=index,
                    amount=1,
                )
                for index, node in enumerate(pick_nodes)
            ],
        )
        storage.build_article_location_mapping()
        orders = OrdersDomain(
            OrderType.STANDARD,
            [
                Order(
                    order_id=index,
                    order_date=arrival,
                    due_date=None,
                    order_positions=[
                        OrderPosition(
                            order_number=index,
                            article_id=index,
                            amount=1,
                        )
                    ],
                )
                for index, arrival in enumerate(arrivals)
            ],
        )
        cart = PickCart(
            n_dimension=1,
            capacities=[1],
            dimensions=[DimensionType.ORDERS],
            n_boxes=4,
            box_can_mix_orders=False,
        )
        resources = Resources(
            ResourceType.HUMAN,
            [
                Resource(
                    id=0,
                    capacity=4,
                    speed=1.0,
                    time_per_pick=self.pick_time_s,
                    tour_setup_time=0.0,
                    pick_cart=cart,
                    available=True,
                    occupied=False,
                    current_location=layout.graph_data.start_location,
                )
            ],
        )
        processes = {}
        for obj in self.information["objects"]:
            features = {feature["name"]: feature["value"] for feature in obj["features"]}
            if features["type"] != ExponentialSingleLineUniformLocationOrderStream.representation:
                raise ValueError(f"Unsupported process information: {features['type']}")
            processes[obj["name"]] = ExponentialSingleLineUniformLocationOrderStream(
                features["mean_interarrival_time_s"]
            )
        information = PlannerInformation(tpe=InformationType.PROCESS_INFORMATION, processes=processes)
        warehouse_info = WarehouseInfo(tpe=WarehouseInfoType.ONLINE)
        self._domain = SimWarehouseDomain(
            problem_class="OBRSP",
            objective="mean_order_completion_time",
            layout=layout,
            articles=articles,
            orders=orders,
            resources=resources,
            storage=storage,
            dynamic_warehouse_info=initial_dynamic_info(),
            warehouse_info=warehouse_info,
            information=information,
        )
        return self._domain

    def events(self) -> Iterator[Event]:
        return first_arrival_release(self._domain)
