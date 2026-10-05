from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import (
    Article,
    Articles,
    ArticleType,
    DimensionType,
    LayoutData,
    LayoutNetwork,
    LayoutParameters,
    LayoutType,
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
    ShelfStorageGraphGenerator,
    ExponentialSingleLineUniformLocationOrderStream,
    InformationType,
    PlannerInformation,
)

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain


class StochasticWaitingDataLoader(DataLoader):
    """Build the small single-line system assumed by the analytical model."""

    def __init__(
        self,
        instances_dir: str | Path,
        cfg,
    ):
        super().__init__(Path(instances_dir).resolve())
        self.cfg = cfg

    @staticmethod
    def _layout(
        n_aisles: int = 8,
        n_pick_locations: int = 17,
        *,
        start=(-1, -1),
        end=(-2, -1),
        start_connection_point=None,
        end_connection_point=None,
    ) -> LayoutData:
        left_centre = n_aisles // 2
        right_centre = left_centre + 1
        start_connection_point = start_connection_point or (left_centre, 0)
        end_connection_point = end_connection_point or (right_centre, 0)
        params = LayoutParameters(
            n_aisles=n_aisles,
            n_pick_locations=n_pick_locations,
            n_blocks=1,
            dist_top_to_pick_location=0.0,
            dist_bottom_to_pick_location=0.0,
            dist_pick_locations=1.0,
            dist_aisle=1.0,
            dist_start=0.0,
            dist_end=0.0,
            start_location=start,
            end_location=end,
            start_connection_point=start_connection_point,
            end_connection_point=end_connection_point,
            depot_location="front_center",
        )
        generator = ShelfStorageGraphGenerator(
            n_aisles=n_aisles,
            n_pick_locations=n_pick_locations,
            dist_aisle=1.0,
            dist_pick_locations=1.0,
            dist_aisle_location=0.0,
            dist_start=0.0,
            dist_end=0.0,
            start_location=start,
            end_location=end,
            start_connection_point=start_connection_point,
            end_connection_point=end_connection_point,
        )
        generator.populate_graph()
        graph = generator.G
        graph.nodes[start]["pos"] = start
        graph.nodes[end]["pos"] = end
        nodes = list(graph.nodes)
        adjacency = nx.to_scipy_sparse_array(
            graph,
            nodelist=nodes,
            weight="weight",
            dtype=float,
        )
        distances, predecessors = floyd_warshall(
            adjacency,
            directed=False,
            return_predecessors=True,
        )
        network = LayoutNetwork(
            graph=graph,
            distance_matrix=pd.DataFrame(
                distances,
                index=nodes,
                columns=nodes,
            ),
            predecessor_matrix=predecessors,
            closest_node_to_start=start_connection_point,
            min_aisle_position=0,
            max_aisle_position=n_pick_locations + 1,
            start_node=start,
            end_node=end,
            node_list=nodes,
        )
        return LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=params,
            layout_network=network,
        )

    def load(
        self,
        **kwargs,
    ) -> SimWarehouseDomain:
        problems = self.cfg.engines.simulation_engine.problems
        order_condition = "casim.simulation_engine.conditions.NbrOrdersCondition"
        active_condition = "casim.simulation_engine.conditions.ActiveTourReadyCondition"
        if set(problems) != set(self.cfg.engines.decision_engine.problems):
            raise ValueError("Decision and simulation engine problems disagree")
        if list(problems.OBRP.triggers) != ["ActiveTourOpportunity"]:
            raise ValueError("Active-tour decisions must use ActiveTourOpportunity")
        if list(problems.OBRSPW.triggers) != ["WaitingOpportunity"]:
            raise ValueError("Waiting decisions must use WaitingOpportunity")
        if not any(c.get("_target_") == active_condition for c in problems.OBRP.conditions):
            raise ValueError("Active-tour decisions require ActiveTourReadyCondition")
        admission = "casim.pipelines.subproblems.admission.RemainingRouteAdmissionNode"
        if admission not in self.cfg.insertion_repo.components or any(
            name.startswith("casim.pipelines.subproblems.batching.")
            for name in self.cfg.insertion_repo.components
        ):
            raise ValueError("Active-tour admission requires the admission node without a batching node")
        if "casim.pipelines.subproblems.picker_routing.SShape" not in self.cfg.insertion_repo.components:
            raise ValueError("Active-tour admission requires SShape routing")
        if not any(
            condition.get("_target_") == order_condition
            and int(condition.threshold) == 3
            and condition.get("allow_when_done", False)
            for condition in problems.OBRSPW.conditions
        ):
            raise ValueError("Analytical waiting requires OBRSPW to start at q-1 = 3 orders and flush on stream closure")
        arrival_times_s = self.cfg.simulation.arrival_times_s
        aisles = self.cfg.simulation.aisles
        arrivals = [float(value) for value in arrival_times_s]
        aisle_values = [int(value) for value in aisles]
        if len(arrivals) != len(aisle_values) or not arrivals:
            raise ValueError("arrival_times_s and aisles must have equal length")
        if arrivals != sorted(arrivals):
            raise ValueError("arrival_times_s must be nondecreasing")
        if any(not 1 <= aisle <= 8 for aisle in aisle_values):
            raise ValueError("aisles must be between one and eight")

        layout = self._layout()
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
                    x=aisle,
                    y=1,
                    article_id=index,
                    amount=1,
                )
                for index, aisle in enumerate(aisle_values)
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
                    time_per_pick=float(self.cfg.simulation.get("pick_time_s", 0.0)),
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
            if obj.name in processes:
                raise ValueError(f"Duplicate process information: {obj.name}")
            processes[obj.name] = ExponentialSingleLineUniformLocationOrderStream(
                features["mean_interarrival_time_s"]
            )
        information = PlannerInformation(tpe=InformationType.PROCESS_INFORMATION, processes=processes)
        warehouse_info = WarehouseInfo(tpe=WarehouseInfoType.ONLINE)
        return SimWarehouseDomain(
            problem_class="OBRSPW",
            objective="makespan",
            layout=layout,
            articles=articles,
            orders=orders,
            resources=resources,
            storage=storage,
            dynamic_warehouse_info=DynamicInfo(tpe=WarehouseInfoType.ONLINE, time=0.0),
            warehouse_info=warehouse_info,
            information=information,
        )
