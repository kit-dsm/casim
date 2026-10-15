from __future__ import annotations

from pathlib import Path

from casim.data_loader import DataLoader
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



class StochasticWaitingDataLoader(DataLoader):
    """Build the small single-line system assumed by the analytical model."""

    def __init__(
        self,
        instances_dir: str | Path,
        cfg,
    ):
        super().__init__(Path(instances_dir).resolve())
        self.cfg = cfg

    def load(
        self,
        **kwargs,
    ) -> SimWarehouseDomain:
        arrivals = [float(value) for value in self.cfg.simulation.arrival_times_s]
        pick_nodes = ([tuple(int(value) for value in node)
                       for node in self.cfg.simulation.pick_locations]
                      if "pick_locations" in self.cfg.simulation else
                      [(int(aisle), 1) for aisle in self.cfg.simulation.aisles])
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
                    time_per_pick=float(self.cfg.simulation.pick_time_s),
                    tour_setup_time=0.0,
                    pick_cart=cart,
                    available=True,
                    occupied=False,
                    current_location=layout.graph_data.start_location,
                )
            ],
        )
        processes = {}
        for obj in self.cfg.data_card.information.objects:
            features = {feature.name: feature.value for feature in obj.features}
            if features["type"] != ExponentialSingleLineUniformLocationOrderStream.representation:
                raise ValueError(f"Unsupported process information: {features['type']}")
            processes[obj.name] = ExponentialSingleLineUniformLocationOrderStream(
                features["mean_interarrival_time_s"]
            )
        information = PlannerInformation(tpe=InformationType.PROCESS_INFORMATION, processes=processes)
        warehouse_info = WarehouseInfo(tpe=WarehouseInfoType.ONLINE)
        return SimWarehouseDomain(
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
