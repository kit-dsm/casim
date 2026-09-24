"""Explicit, small study input for the supported Walking vs. Waiting rules."""

from pathlib import Path

from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import (
    Article, Articles, ArticleType, DimensionType, Location, Order, OrderPosition,
    OrdersDomain, OrderType, PickCart, Resource, Resources, ResourceType,
    StorageLocations, StorageType, WarehouseInfo, WarehouseInfoType,
)

from casim.domain_objects.sim_domain import DynamicInfo, SimWarehouseDomain
from scenarios.scenario_stochastic_waiting.loader import StochasticWaitingDataLoader


class WalkOrWaitDataLoader(DataLoader):
    """Load an explicit event stream; no random instance generation or hidden fallback."""

    def __init__(self, instances_dir: str | Path, cfg):
        super().__init__(Path(instances_dir).resolve())
        self.cfg = cfg

    def load(self, **kwargs) -> SimWarehouseDomain:
        if kwargs:
            raise ValueError("This scenario accepts explicit simulation orders only; published JSON needs its layout converter")
        cfg = self.cfg
        sim = cfg.simulation
        capacity = int(sim.capacity_orders)
        wait_k = int(cfg.policy.wait_k)
        if wait_k < 0 or wait_k > capacity or wait_k > 4:
            raise ValueError("Supported paper wait thresholds are 0..min(capacity, 4)")
        expected_policy = (
            "casim.pipelines.subproblems.waiting.StartImmediatelyNode" if wait_k == 0 else
            f"casim.pipelines.subproblems.waiting.WaitFor{('One', 'Two', 'Three', 'Four')[wait_k - 1]}"
        )
        if cfg.policy.component != expected_policy:
            raise ValueError("wait_k and the configured waiting component disagree")
        if capacity != 4:
            raise ValueError("The configured paper example currently supports four cart bins only")
        if int(sim.n_pickers) != 1:
            raise ValueError("Paper scenario currently supports one picker; multi-picker assignment is not ported")
        if cfg.policy.component not in cfg.cosy_repo.components:
            raise ValueError("Waiting policy and CoSy repo disagree")
        if cfg.routing.component not in cfg.cosy_repo.components:
            raise ValueError("Routing policy and CoSy repo disagree")
        if set(cfg.engines.decision_engine.problems) != set(cfg.engines.simulation_engine.problems):
            raise ValueError("Decision and simulation engine problem configurations disagree")
        intervention = "OBRP" in cfg.engines.simulation_engine.problems
        if intervention:
            problems = cfg.engines.simulation_engine.problems
            if list(problems.OBRP.triggers) != ["ActiveTourOpportunity"]:
                raise ValueError("Paper admission must use ActiveTourOpportunity")
            if list(problems.OBRSPW.triggers) != ["WaitingOpportunity"]:
                raise ValueError("Paper waiting must use WaitingOpportunity")
            if not any(c.get("_target_") == "casim.simulation_engine.conditions.ActiveTourReadyCondition"
                       for c in problems.OBRP.conditions):
                raise ValueError("Paper admission requires ActiveTourReadyCondition")
        if intervention and cfg.routing.name != "s_shape":
            raise ValueError("Active-route replacement supports S-Shape only; select paper_no_intervention for other routing")
        if wait_k == 0 and not intervention:
            raise ValueError("Paper wait-0 needs active-tour admission; select paper_intervention")
        if wait_k == 0 and cfg.routing.name != "s_shape":
            raise ValueError("Empty aisle patrol is defined for S-Shape routing only")
        if intervention:
            admission = "casim.pipelines.subproblems.batching.RemainingRouteAdmissionNode"
            configured_batchers = [name for name in cfg.insertion_repo.components
                                   if name.startswith("casim.pipelines.subproblems.batching.")]
            if configured_batchers != [admission]:
                raise ValueError("Paper intervention needs only RemainingRouteAdmissionNode in its configured CoSy repo")
        if sim.source != "explicit":
            raise ValueError("Only explicit demonstration orders are loadable; published paper layouts are not imported")

        n_aisles = int(sim.n_aisles)
        n_locations = int(sim.n_pick_locations)
        depot_aisle = int(sim.depot_aisle)
        if n_aisles < 1 or n_locations < 1 or not 1 <= depot_aisle <= n_aisles:
            raise ValueError("Invalid paper layout dimensions or depot aisle")
        layout = StochasticWaitingDataLoader._layout(
            n_aisles=n_aisles,
            n_pick_locations=n_locations,
            start_connection_point=(depot_aisle, 0),
            end_connection_point=(depot_aisle, 0),
        )

        specs = list(sim.orders)
        if not specs:
            raise ValueError("Paper scenario requires at least one explicit order")
        arrivals = [float(spec.arrival_s) for spec in specs]
        if arrivals != sorted(arrivals) or any(time < 0 for time in arrivals):
            raise ValueError("Order arrivals must be nonnegative and sorted")
        if float(sim.shift_start_s) < 0 or float(sim.shift_start_s) > arrivals[0]:
            raise ValueError("Shift start must be nonnegative and no later than the first order")
        if intervention and len(set(arrivals)) != len(arrivals):
            raise ValueError("Simultaneous arrivals lack a defined sequential intervention order")

        article_by_position = {}
        demand_by_article = {}
        orders = []
        for order_id, spec in enumerate(specs):
            if not spec.picks:
                raise ValueError(f"Order {order_id} has no picks")
            positions = []
            for raw_position in spec.picks:
                aisle, location = map(int, raw_position)
                if not (1 <= aisle <= n_aisles and 1 <= location <= n_locations):
                    raise ValueError(f"Pick {(aisle, location)} is outside the configured layout")
                point = (aisle, location)
                article_id = article_by_position.setdefault(point, len(article_by_position))
                demand_by_article[article_id] = demand_by_article.get(article_id, 0) + 1
                positions.append(OrderPosition(
                    order_number=order_id, article_id=article_id, amount=1,
                ))
            orders.append(Order(
                order_id=order_id,
                order_date=arrivals[order_id],
                due_date=float(spec.due_s),
                order_positions=positions,
            ))

        articles = Articles(
            ArticleType.STANDARD,
            [Article(article_id=article_id, weight=1.0)
             for article_id in demand_by_article],
        )
        storage = StorageLocations(
            tpe=StorageType.DEDICATED,
            locations=[Location(x=point[0], y=point[1], article_id=article_id,
                                amount=demand_by_article[article_id])
                       for point, article_id in article_by_position.items()],
        )
        storage.build_article_location_mapping()
        cart = PickCart(
            n_dimension=1,
            capacities=[1],
            dimensions=[DimensionType.ORDERS],
            n_boxes=capacity,
            box_can_mix_orders=False,
        )
        picker = Resource(
            id=0, capacity=capacity, speed=float(sim.walking_speed),
            time_per_pick=float(sim.pick_time_s), tour_setup_time=0.0,
            pick_cart=cart, available=True, occupied=False,
            current_location=layout.graph_data.start_location,
        )
        if picker.speed <= 0 or picker.time_per_pick < 0:
            raise ValueError("Walking speed must be positive and pick time nonnegative")
        return SimWarehouseDomain(
            problem_class="OBRSPW",
            objective="mean_order_completion_time",
            layout=layout,
            articles=articles,
            orders=OrdersDomain(OrderType.STANDARD, orders),
            resources=Resources(ResourceType.HUMAN, [picker]),
            storage=storage,
            dynamic_warehouse_info=DynamicInfo(tpe=WarehouseInfoType.ONLINE, time=0.0),
            warehouse_info=WarehouseInfo(tpe=WarehouseInfoType.ONLINE),
        )
