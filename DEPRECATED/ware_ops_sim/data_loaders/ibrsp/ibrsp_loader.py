from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import floyd_warshall
from ware_ops_algos.algorithms import (GreedyItemAssignment, OrderNrFifoBatching, ExactTSPRoutingDistance,
                                       SShapeRouting, LargestGapRouting, MidpointRouting, ReturnRouting,
                                       NearestNeighbourhoodRouting, EDDScheduling, SchedulingInput, WarehouseOrder,
                                       PickList, ExactTSPBatchingAndRoutingDistance, ExactCombinedBatchingRouting)
from ware_ops_algos.data_loaders import DataLoader
from ware_ops_algos.domain_models import BaseWarehouseDomain, Order, Location, Article, PickCart, DimensionType, \
    ResourceType, Resource, Resources, OrdersDomain, OrderType, Articles, ArticleType, StorageLocations, StorageType, \
    LayoutData, LayoutType, LayoutNetwork, LayoutParameters, WarehouseInfoType, WarehouseInfo
from ware_ops_algos.utils.io_helpers import load_pickle, dump_pickle
from ware_ops_algos.utils.visualization import plot_route, plot_picker_gantt


class IBRSPLoader(DataLoader):
    def __init__(self,
                 instances_dir: str | Path,
                 cache_dir: str = None,
                 cfg=None):
        super().__init__(instances_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cfg = cfg
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, filepath: str, use_cache: bool = True) -> BaseWarehouseDomain:
        filepath = Path(filepath)
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        # Check cache
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{filepath.stem}_domain.pkl"
            self.cache_path = cache_path
            if cache_path.exists():
                return load_pickle(str(cache_path))

        # Parse and build
        parsed = self._parse(str(filepath))
        domain = self._build(parsed)
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f"{filepath.stem}_domain.pkl"
            dump_pickle(str(cache_path), domain)

        return domain

    def _parse(self, filepath: str) -> dict[str, Any]:
        lines = self._load_text(filepath, encoding="utf-8")

        # State for parsing
        line_idx = 0
        header = {}
        articles = []
        sku_entries = []
        order_entries = []
        arcs = []
        locations = []
        shortest_paths = {}
        vertices_coords = {}
        departing_depot = None
        arrival_depot = None

        def next_line():
            nonlocal line_idx
            if line_idx < len(lines):
                line = lines[line_idx]
                line_idx += 1
                return line
            return None

        def peek_line():
            return lines[line_idx] if line_idx < len(lines) else None

        def skip_to_prefix(prefix):
            nonlocal line_idx
            while line_idx < len(lines) and not lines[line_idx].startswith(prefix):
                line_idx += 1
            if line_idx < len(lines):
                return next_line()
            return None

        def next_data_line():
            """Skip comment lines starting with '//'."""
            line = next_line()
            while line is not None and line.startswith("//"):
                line = next_line()
            return line

        # === HEADER ===
        skip_to_prefix("//NbLocations")
        header["NbLocations"] = int(next_data_line())
        header["NbProducts"] = int(next_data_line())
        header["NbPickers"] = int(next_data_line().split()[0])
        header["CapaPicker"] = int(next_data_line().split()[0])
        header["TimeToTravelOneDistanceUnit"] = int(next_data_line().split()[0])
        header["SetupTime"] = int(next_data_line().split()[0])
        header["PickTime"] = int(next_data_line().split()[0])

        # === PRODUCTS ===
        skip_to_prefix("//Products")
        for _ in range(header["NbProducts"]):
            parts = next_data_line().split()
            article_id = int(parts[0])
            location = int(parts[1])
            volume = float(parts[2])
            articles.append(Article(article_id=article_id, volume=volume))
            sku_entries.append((article_id, location))

        # === ORDERS ===
        skip_to_prefix("//Orders")
        skip_to_prefix("//NbOrders")
        header["NbOrders"] = int(next_data_line())
        for _ in range(header["NbOrders"]):
            parts = next_data_line().split()
            order_number = int(parts[0])
            due_date = int(parts[1])  # ignored
            tardiness_penalty = int(parts[2])
            nb_prod = int(parts[3])
            idx = 4
            positions = []
            for _ in range(nb_prod):
                article_id = int(parts[idx])
                amount = int(parts[idx + 1])
                positions.append({"article_id": article_id, "amount": amount})
                idx += 2
            order_entries.append(Order.from_dict(order_number, {"order_positions": positions, "due_date": due_date, "order_date": 0}))

        # === GRAPH ===
        skip_to_prefix("//Graph")
        header["NbVerticesIntersections"] = int(next_data_line())
        header["DepartingDepot"] = int(next_data_line())
        header["ArrivalDepot"] = int(next_data_line())
        departing_depot = header["DepartingDepot"]
        arrival_depot = header["ArrivalDepot"]

        # === ARCS ===
        skip_to_prefix("//Arcs")
        next_line()  # skip header line
        while True:
            line = peek_line()
            if line is None or line.startswith("//LocStart"):
                break
            parts = next_line().split()
            if len(parts) >= 3:
                arcs.append((int(parts[0]), int(parts[1]), float(parts[2])))

        # === SHORTEST PATHS ===
        skip_to_prefix("//LocStart")
        while True:
            line = peek_line()
            if line is None or line.startswith("//Location"):
                break
            parts = next_line().split()
            if len(parts) >= 3:
                shortest_paths[(int(parts[0]), int(parts[1]))] = float(parts[2])

        # === VERTICES + COORDINATES ===
        skip_to_prefix("//Location")
        while (line := next_line()) is not None:
            if line.startswith("//"):
                continue
            parts = line.split()
            try:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                label = parts[3].strip('"')

                # Determine node type
                match label:
                    case "depot":
                        if idx == departing_depot:
                            node_type = "start_node"
                        elif idx == arrival_depot:
                            node_type = "end_node"
                            x += 1  # Adjust arrival depot position
                        else:
                            node_type = "depot_node"
                    case "product":
                        node_type = "pick_node"
                    case "intersection":
                        node_type = "intersection"
                    case _:
                        raise ValueError(f"Unknown node type: {label}")

                vertices_coords[idx] = (x, y, node_type)

            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing vertex line: {line} — {e}")

        aisle_x_positions = sorted(set(
            x for x, y, node_type in vertices_coords.values()
            if node_type == "pick_node"
        ))

        # For some reason the intersection coordinates are not centered with the aisles coordinates.
        # Therefore we "snap" each intersection node to the nearest aisle x
        def snap_to_nearest_aisle(x, aisle_x_positions):
            return min(aisle_x_positions, key=lambda ax: abs(ax - x))

        snapped_coords = {}
        for idx, (x, y, node_type) in vertices_coords.items():
            if node_type == "intersection":
                snapped_x = snap_to_nearest_aisle(x, aisle_x_positions)
                snapped_coords[idx] = (snapped_x, y, node_type)
            else:
                snapped_coords[idx] = (x, y, node_type)

        # Build locations from SKU entries
        for article_id, loc in sku_entries:
            x, y, _ = vertices_coords.get(loc, (0, 0, ""))
            locations.append(Location(x=x, y=y, article_id=article_id, amount=1000))

        return {
            "header": header,
            "articles": articles,
            "locations": locations,
            "orders": order_entries,
            "arcs": arcs,
            "shortest_paths": shortest_paths,
            "vertices_coords": snapped_coords
        }

    def _build(self, parsed: dict[str, Any]) -> BaseWarehouseDomain:
        from ware_ops_algos.generators import (
            ExplicitGraphGenerator,
            distance_matrix_generator_from_shortest_paths
        )

        header = parsed["header"]
        articles = parsed["articles"]
        location_entries = parsed["locations"]
        order_entries = parsed["orders"]
        arcs = parsed["arcs"]
        shortest_paths = parsed["shortest_paths"]
        vertices_coords = parsed["vertices_coords"]

        # Build graph from explicit vertices and arcs
        graph_generator = ExplicitGraphGenerator(vertices_coords, arcs)
        graph_generator.populate_graph()
        graph = graph_generator.G

        # Identify depot nodes
        depot_idx = header["DepartingDepot"]
        end_idx = header["ArrivalDepot"]
        start_node = vertices_coords[depot_idx][:2]
        end_node = vertices_coords[end_idx][:2]

        # Convert shortest paths from vertex indices to coordinate tuples
        shortest_paths_coords = {}
        for (start_idx, end_idx), distance in shortest_paths.items():
            x_start, y_start = vertices_coords[start_idx][:2]
            x_end, y_end = vertices_coords[end_idx][:2]
            shortest_paths_coords[((x_start, y_start), (x_end, y_end))] = distance

        # Build distance matrix
        dima = distance_matrix_generator_from_shortest_paths(graph, shortest_paths_coords)

        # Compute predecessor matrix for path reconstruction
        nodes = list(graph.nodes())
        A = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight='weight', dtype=float)
        _, predecessors = floyd_warshall(A, directed=False, return_predecessors=True)

        # Extract layout dimensions
        intersection_nodes = [
            (x, y) for x, y, node_type in vertices_coords.values()
            if node_type == "intersection"
        ]
        min_aisle_pos = min(y for x, y in intersection_nodes) if intersection_nodes else 0
        max_aisle_pos = max(y for x, y, _ in vertices_coords.values())
        n_aisles = int(max(x for x, y, _ in vertices_coords.values()))

        # Find closest node to start (excluding start/end nodes)
        closest_node_to_start = (
            dima[start_node]
            .drop(labels=[start_node, end_node])
            .idxmin()
        )

        # Layout parameters (distances are encoded in graph, so set to 0)
        layout_params = LayoutParameters(
            n_aisles=n_aisles,
            n_pick_locations=max_aisle_pos,
            dist_top_to_pick_location=0,
            dist_bottom_to_pick_location=0,
            dist_start=0,
            dist_end=0,
            dist_pick_locations=0,
            dist_aisle=0,
            n_blocks=2,
            start_location=start_node,
            end_location=end_node,
        )

        layout_network = LayoutNetwork(
            graph=graph,
            distance_matrix=dima,
            predecessor_matrix=predecessors,
            closest_node_to_start=closest_node_to_start,
            min_aisle_position=min_aisle_pos,
            max_aisle_position=max_aisle_pos,
            start_node=start_node,
            end_node=end_node,
            node_list=nodes
        )

        layout = LayoutData(
            tpe=LayoutType.CONVENTIONAL,
            graph_data=layout_params,
            layout_network=layout_network,
        )

        # Storage
        storage = StorageLocations(tpe=StorageType.DEDICATED, locations=location_entries)
        storage.build_article_location_mapping()

        # Articles
        articles_obj = Articles(tpe=ArticleType.STANDARD, articles=articles)

        # Orders
        orders = OrdersDomain(tpe=OrderType.STANDARD, orders=order_entries)

        # Resources

        # capacity = header["CapaPicker"]
        # pick_cart = PickCart(n_dimension=header["NbDimensionsCapacity"],
        #                      n_boxes=1,
        #                      capacities=[capacity],
        #                      dimensions=[DimensionType.ITEMS])
        # pick_cart = PickCart(
        #     n_dimension=1,
        #     capacities=[capacity],
        #     dimensions=[DimensionType.ORDERLINES],
        #     n_boxes=1,
        #     box_can_mix_orders=True
        # )

        capacity = header["CapaPicker"]
        n_pickers = header["NbPickers"]
        picker_speed = 1 / header["TimeToTravelOneDistanceUnit"]
        time_per_pick = header["PickTime"]
        pick_cart = PickCart(
            n_dimension=1,
            capacities=[capacity],
            dimensions=[DimensionType.ORDERLINES],
            n_boxes=1,
            box_can_mix_orders=True
        )

        resources = Resources(
            tpe=ResourceType.HUMAN,
            resources=[Resource(id=i,
                                capacity=capacity,
                                speed=picker_speed,
                                time_per_pick=time_per_pick,
                                pick_cart=pick_cart,
                                tour_setup_time=header["SetupTime"]) for i in range(n_pickers)]
        )

        warehouse_info = WarehouseInfo(tpe=WarehouseInfoType.OFFLINE)

        return BaseWarehouseDomain(
            problem_class="OBSRP",
            objective="Distance",
            layout=layout,
            articles=articles_obj,
            orders=orders,
            resources=resources,
            storage=storage,
            warehouse_info=warehouse_info
        )


def render_graph(G, plot=True, out_name=False, dpi=150, font_size=6, node_size=80,
                 node_color='lightblue', highlight_nodes=None, highlight_color='red',
                 show_edge_labels=False, figsize=(20, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.get_node_attributes(G, 'pos')
    colors = (
        [highlight_color if node in highlight_nodes else node_color for node in G.nodes()]
        if highlight_nodes else node_color
    )
    nx.draw(G, pos=pos, ax=ax, with_labels=True, node_color=colors,
            font_size=font_size, node_size=node_size,
            arrows=True,                  # show direction
            connectionstyle="arc3,rad=0.1",  # slight curve to separate overlapping edges
            edge_color='gray',
            alpha=0.7,
            width=0.5)
    if show_edge_labels:
        weight = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=weight, font_size=font_size - 1)
    plt.tight_layout()
    if out_name:
        plt.savefig(out_name, dpi=dpi, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close(fig)

def latest_order_arrival(orders: list[WarehouseOrder]) -> float:
    if any(o.order_date is not None for o in orders):
        arrivals = [o.order_date for o in orders]
        return max(arrivals) if arrivals else 0.0
    else:
        return 0.0

def first_due_date(orders: list[WarehouseOrder]) -> float:
    if any(o.order_date is not None for o in orders):
        due_dates = [o.order_date for o in orders]
        return min(due_dates) if due_dates else float("inf")
    else:
        return 0.0

def build_pick_lists(orders: list[WarehouseOrder]):
    # build pick lists
    pick_positions = []
    for order in orders:
        for pos in order.pick_positions:
            pick_positions.append(pos)

    pick_list = PickList(
        pick_positions=pick_positions,
        release=latest_order_arrival(orders),
        earliest_due_date=first_due_date(orders),
        orders=orders
    )
    return pick_list


def _evaluate_due_dates(assignments, orders: OrdersDomain):
    order_by_id = {o.order_id: o for o in orders.orders}
    records = []
    for ass in assignments:
        end_time = ass.end_time
        for on in ass.route.pick_list.order_numbers:
            o = order_by_id.get(on)
            if o is None:
                continue
            if o.due_date is None:
                continue  # skip if no due date
            arrival_time = o.order_date
            start_time = ass.start_time
            due_ts = o.due_date  # .timestamp()
            lateness = end_time - due_ts
            records.append({
                "order_number": on,
                "arrival_time": arrival_time,
                "start_time": start_time,
                "batch_idx": ass.batch_idx,
                "picker_id": ass.picker_id,
                "completion_time": end_time,
                "due_date": o.due_date,
                "lateness": lateness,
                "tardiness": max(0, lateness),
                "on_time": end_time <= due_ts,
            })
    return pd.DataFrame(records)


def main():
    from pathlib import Path
    from typing import Tuple

    def find_project_root() -> Path:
        """Find project root by looking for a marker file."""
        current = Path().resolve()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():  # or setup.py, .git, etc.
                return parent
        raise FileNotFoundError("Could not find project root")

    PROJECT_ROOT = find_project_root()

    DATA_DIR = PROJECT_ROOT / "data"

    instances_base = DATA_DIR / "instances"
    cache_base = DATA_DIR / "instances" / "caches"
    order_list_online = instances_base / "KrisLargeData" / "instances_100_1.txt"

    loader_online = IBRSPLoader(
        instances_dir=DATA_DIR / "KrisLargeData",
        cache_dir=DATA_DIR / "caches" / "KrisLargeData",
    )

    domain = loader_online.load(order_list_online, use_cache=False)

    orders = domain.orders
    layout = domain.layout
    resources = domain.resources
    articles = domain.articles
    storage_locations = domain.storage

    layout_network = layout.layout_network
    graph_data = layout.graph_data
    graph = layout_network.graph
    graph_params = layout.graph_data
    dima = layout_network.distance_matrix
    render_graph(graph)
    selector = GreedyItemAssignment(storage_locations)
    ia_sol = selector.solve(orders.orders)
    orders.orders = ia_sol.resolved_orders

    pick_list_combined = []
    for order in ia_sol.resolved_orders:
        for pos in order.pick_positions:
            pick_list_combined.append(pos)

    router = ExactCombinedBatchingRouting(
        start_node=layout_network.start_node,
        end_node=layout_network.end_node,
        distance_matrix=layout_network.distance_matrix,
        predecessor_matrix=layout_network.predecessor_matrix,
        picker=resources.resources,
        gen_tour=True,
        gen_item_sequence=True,
        big_m=1000,
        time_limit=60,
        node_list=layout_network.node_list,
        node_to_idx={node: idx for idx, node in enumerate(list(graph.nodes))},
        idx_to_node={idx: node for idx, node in enumerate(list(graph.nodes))},
    )

    sol = router.solve(pick_list_combined)
    dist = 0
    print(sol)
    for r in sol.routes:
        print(r.distance)
        dist += r.distance
    print("Total CBR", dist)
    batcher = OrderNrFifoBatching(
        pick_cart=resources.resources[0].pick_cart,
        articles=articles
    )

    batching_sol = batcher.solve(orders.orders)

    print(batching_sol.execution_time)

    # Build pick list from batches
    batches = batching_sol.batches
    pick_lists = []
    for batch in batches:
        pick_list = []
        print(len(batch.orders))
        for order in batch.orders:
            for pos in order.pick_positions:
                pick_list.append(pos)
        pick_lists.append(pick_list)

    pls = []
    for batch in batches:
        pl = build_pick_lists(batch.orders)
        pls.append(pl)
    batching_sol.pick_lists = pls
    pl = []
    o_ids = [53]
    for order in orders.orders:
        if order.order_id in o_ids:
            print(len(order.pick_positions))
            for pos in order.pick_positions:
                pl.append(pos)

    heuristic_router = NearestNeighbourhoodRouting(
        start_node=layout_network.start_node,
        end_node=layout_network.end_node,
        closest_node_to_start=layout_network.closest_node_to_start,
        min_aisle_position=layout_network.min_aisle_position,
        max_aisle_position=layout_network.max_aisle_position,
        distance_matrix=layout_network.distance_matrix,
        predecessor_matrix=layout_network.predecessor_matrix,
        picker=resources.resources,
        gen_tour=True,
        gen_item_sequence=True,
        node_list=layout_network.node_list,
        node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
        idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))}
    )

    tsp_routing = ExactTSPRoutingDistance(
        distance_matrix=layout_network.distance_matrix,
        predecessor_matrix=layout_network.predecessor_matrix,
        big_m=1000,
        start_node=layout_network.start_node,
        end_node=layout_network.end_node,
        closest_node_to_start=layout_network.closest_node_to_start,
        min_aisle_position=layout_network.min_aisle_position,
        max_aisle_position=layout_network.max_aisle_position,
        picker=resources.resources,
        gen_tour=True,  # we dont have predecessor matrix yet
        gen_item_sequence=True,
        set_time_limit=120,
        node_list=layout_network.node_list,
        node_to_idx={node: idx for idx, node in enumerate(list(layout_network.graph.nodes))},
        idx_to_node={idx: node for idx, node in enumerate(list(layout_network.graph.nodes))},
    )

    # import cProfile
    # import pstats
    #
    # with cProfile.Profile() as pr:
    #     sol = heuristic_router.solve(pl)
    # stats = pstats.Stats(pr)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)
    #
    # print(sol.route.distance)
    # plot_route(graph, sol.route.route)
    # print(sol.route.item_sequence)
    # print(len(sol.route.item_sequence))
    #
    # sol = tsp_routing.solve(pl)
    # print(sol.route.distance)
    # plot_route(graph, sol.route.route)
    # print(sol.route.item_sequence)
    # print(len(sol.route.item_sequence))

    total_dist = 0
    routing_solutions = []
    for pl in pls:
        sol = heuristic_router.solve(pl.pick_positions)
        total_dist += sol.route.distance
        print(sol.route.distance)
        # plot_route(graph, sol.route.route)
        routing_solutions.append(sol)
        sol.route.pick_list = pl
        heuristic_router.reset_parameters()
    print("Total Sequential", total_dist)

    routes = []
    for route in routing_solutions:
        routes.append(route.route)
    sequencing_inpt = SchedulingInput(routes=routes,
                                      orders=orders,
                                      resources=resources)
    scheduler = EDDScheduling()
    sequencing_solution = scheduler.solve(sequencing_inpt)
    dd_eval = _evaluate_due_dates(sequencing_solution.jobs, orders)
    print()

if __name__ == "__main__":
    main()