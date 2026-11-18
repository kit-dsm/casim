"""Aggregate checks of the IJPE WMS trace against CASIM's anchored facts.

Run after ``casim run scenarios/ijpe sweep=anchored --output-dir <directory>``:

    python -m scripts.ijpe.trace_diagnosis <directory>

Reads confidential lines in memory and prints only counts and fit statistics.
It does not construct a simulation or write any order, line, or picker record.
The pallet-overhead score is an explanatory counterfactual, not a calibrated
or physically verified CASIM time model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from casim.sources.grid_layout import GridLayout
from casim.sources.recorded import RecordedRow, recorded_tours
from scripts.ijpe import SCENARIO
from scripts.ijpe.replay import CALIBRATION_DAYS, RUNS, matched_lines, recorded_day, runs


def _motion_fit(frame: pd.DataFrame) -> dict[str, float]:
    design = np.column_stack([np.ones(len(frame)), frame.quantity,
                              frame.walk_distance / 1000])
    base, per_unit, seconds_per_m = np.linalg.lstsq(
        design, frame.recorded_s, rcond=None)[0]
    return {"base_s": float(base), "per_unit_s": float(per_unit),
            "speed_m_s": float(1 / seconds_per_m), "lines": len(frame)}


def analyze(root: Path) -> dict:
    layout, locations = GridLayout().build(SCENARIO / "data" / "layout.csv")
    nodes = GridLayout.pick_nodes(locations)
    cross_edges = [data["weight"] / 1000 for _, _, data in
                   layout.layout_network.graph.edges(data=True)
                   if data.get("edge_type") == "cross_aisle"]
    if set(cross_edges) != {15.0}:
        raise ValueError("layout cross-aisle edge differs from the measured 15 m")
    samples = []
    order_errors = []
    order_pallets = []
    order_tour_sizes = []
    sequence_mismatches = 0
    tour_count = 0
    orders_shorter_than_travel = 0
    orders_shorter_than_interior_travel = 0
    single_tour_orders_shorter_than_interior_travel = 0
    single_tour_route_distance_pairs = []
    picker_day_route_distance_pairs = []
    end_order_reordered_lines = 0
    same_order_end_sorted_links = 0
    same_order_end_sorted_start_at_previous_end = 0
    return_to_open_order_links = 0
    return_to_open_order_start_at_own_previous_end = 0
    return_to_open_order_start_before_global_previous_end = 0
    return_to_open_order_own_match_and_global_overlap = 0
    return_to_open_order_overlap_hours = 0.0
    end_gap_samples = []
    for day, setting, directory in runs(root):
        if setting not in {"null", "default"}:
            continue
        picks = pd.read_csv(directory / "facts" / "picks.csv")
        observed = recorded_day(day, nodes)
        for _, picker_lines in observed.groupby("picker"):
            by_start = picker_lines.sort_values(["start_s", "row"])
            by_end = picker_lines.sort_values(["end_s", "row"])
            def path_length(group):
                path = list(group.node)
                return sum(float(layout.layout_network.distance_matrix.at[a, b])
                           for a, b in zip(path, path[1:]))
            picker_day_route_distance_pairs.append((path_length(by_start), path_length(by_end)))
            end_order_reordered_lines += int(sum(
                by_start.row.to_numpy() != by_end.row.to_numpy()))
            previous_by_order = by_end.groupby("order").end_s.shift()
            previous_global_end = by_end.end_s.shift()
            nodes_by_end = list(by_end.node)
            graph_legs = [float(layout.layout_network.distance_matrix.at[a, b])
                          for a, b in zip(nodes_by_end, nodes_by_end[1:])]
            end_gap_samples.append(pd.DataFrame({
                "gap_s": by_end.end_s.iloc[1:].to_numpy() -
                         by_end.end_s.iloc[:-1].to_numpy(),
                "walk_mm": graph_legs,
                "order_change": by_end.order.iloc[1:].to_numpy() !=
                                by_end.order.iloc[:-1].to_numpy(),
            }))
            return_to_order = (previous_by_order.notna() &
                               by_end.order.ne(by_end.order.shift()))
            return_to_open_order_links += int(return_to_order.sum())
            return_to_open_order_start_at_own_previous_end += int((
                return_to_order &
                (by_end.start_s - previous_by_order).abs().le(1)).sum())
            return_to_open_order_start_before_global_previous_end += int((
                return_to_order & (by_end.start_s < previous_global_end)).sum())
            return_to_open_order_own_match_and_global_overlap += int((
                return_to_order &
                (by_end.start_s - previous_by_order).abs().le(1) &
                (by_end.start_s < previous_global_end)).sum())
            return_to_open_order_overlap_hours += float((
                (previous_global_end - by_end.start_s).where(
                    return_to_order &
                    (by_end.start_s - previous_by_order).abs().le(1) &
                    (by_end.start_s < previous_global_end), 0)).sum() / 3600)
            linked = previous_by_order.notna()
            same_order_end_sorted_links += int(linked.sum())
            same_order_end_sorted_start_at_previous_end += int((
                (by_end.start_s - previous_by_order).abs().le(1) & linked).sum())
        ordered = observed.sort_values(["start_s", "row"])
        picker_ids = {picker: index for index, picker in
                      enumerate(ordered.picker.unique())}
        rows = [RecordedRow(picker_ids[row.picker], int(row.order),
                            int(row.article), int(row.quantity), row.node,
                            row.start_s, row.end_s)
                for row in ordered.itertuples()]
        recorded = recorded_tours(rows)
        expected = {tour_index + 1: (picker,
                     [(row.order, row.article, row.node[0], row.node[1])
                      for row in tour])
                    for tour_index, (picker, tour) in enumerate(recorded)}
        tour_count += len(expected)
        distance = layout.layout_network.distance_matrix
        for _, tour in recorded:
            previous_node = layout.layout_network.start_node
            legs = []
            for row in tour:
                legs.append(float(distance.at[previous_node, row.node]))
                previous_node = row.node
            if len({row.order for row in tour}) == 1:
                by_position = observed[observed.order == tour[0].order].sort_values(
                    ["position", "start_s", "row"])
                positioned_nodes = list(by_position.node)
                positioned_distance = sum(float(distance.at[a, b]) for a, b in
                                          zip(positioned_nodes, positioned_nodes[1:]))
                single_tour_route_distance_pairs.append((sum(legs[1:]),
                                                         positioned_distance))
            for order in {row.order for row in tour}:
                positions = [index for index, row in enumerate(tour)
                             if row.order == order]
                observed_span = (max(tour[index].end_s for index in positions) -
                                 min(tour[index].start_s for index in positions))
                minimum_walk_s = sum(legs[positions[0]:positions[-1] + 1]) / 2306
                orders_shorter_than_travel += observed_span < minimum_walk_s
                interior_walk_s = sum(legs[positions[0] + 1:positions[-1] + 1]) / 2306
                orders_shorter_than_interior_travel += observed_span < interior_walk_s
                if len({row.order for row in tour}) == 1:
                    single_tour_orders_shorter_than_interior_travel += (
                        observed_span < interior_walk_s)
        actual = {int(tour_id): (int(group.picker_id.iloc[0]),
                   list(group[["order_id", "article_id", "node_x", "node_y"]]
                        .itertuples(index=False, name=None)))
                  for tour_id, group in picks.sort_values("pick_start").groupby(
                      "tour_id", sort=False)}
        sequence_mismatches += sum(expected.get(key) != actual.get(key)
                                   for key in expected.keys() | actual.keys())
        lines = matched_lines(observed, picks).sort_values(
            ["picker", "tour_id", "pick_start"]).copy()
        tour = lines.groupby(["picker", "tour_id"])
        lines["first_line"] = tour.cumcount() == 0
        lines["previous_order"] = tour.order.shift()
        lines["previous_position"] = tour.position.shift()
        lines["previous_aisle"] = tour.aisle.shift()
        lines["previous_end"] = tour.end_s.shift()
        lines["previous_node_x"] = tour.node_x.shift()
        lines["previous_node_y"] = tour.node_y.shift()
        lines["day"] = day
        samples.append(lines)
        tour_order_counts = lines.groupby("tour_id").order.nunique()
        for _, group in lines.groupby("order"):
            error = ((group.pick_end.max() - group.walk_start.min()) -
                     (group.end_s.max() - group.start_s.min())) / 60
            order_errors.append((day, error))
            order_pallets.append((day, group.pallet.nunique(), error))
            order_tour_sizes.append((int(tour_order_counts.loc[group.tour_id.iloc[0]]), error))
    if not samples:
        raise ValueError(f"no anchored facts in {root}")
    if sequence_mismatches:
        raise ValueError(f"recorded replay differs in {sequence_mismatches} tour sequences")
    frame = pd.concat(samples, ignore_index=True)
    interior = frame[~frame.first_line]
    ordinary = interior[
        (interior.order == interior.previous_order) &
        (interior.aisle == interior.previous_aisle) &
        (interior.start_s >= interior.previous_end) &
        interior.recorded_s.between(4, 59)]
    adjacent = interior[
        (interior.order == interior.previous_order) &
        (interior.aisle == interior.previous_aisle) &
        (interior.start_s >= interior.previous_end) &
        interior.walk_distance.between(799, 801)]
    short = interior[interior.recorded_s <= 1]
    short_changed_node = short[
        (short.node_x != short.previous_node_x) |
        (short.node_y != short.previous_node_y)]
    training = ordinary[ordinary.day.isin(CALIBRATION_DAYS)]
    validation = ordinary[~ordinary.day.isin(CALIBRATION_DAYS)]
    fit = _motion_fit(training)
    fixed_design = np.column_stack([np.ones(len(training)), training.quantity])
    fixed_base, fixed_per_unit = np.linalg.lstsq(
        fixed_design, training.recorded_s - training.walk_distance / 2306,
        rcond=None)[0]
    prediction = (fit["base_s"] + fit["per_unit_s"] * validation.quantity +
                  validation.walk_distance / 1000 / fit["speed_m_s"])
    submitted = 19 + validation.walk_distance / 2306
    fixed_prediction = (fixed_base + fixed_per_unit * validation.quantity +
                        validation.walk_distance / 2306)
    leave_one_out = [_motion_fit(ordinary[ordinary.day != day])["speed_m_s"]
                     for day in sorted(ordinary.day.unique())]
    change_both = interior[(interior.order != interior.previous_order) &
                           (interior.aisle != interior.previous_aisle)]
    transitions = {}
    for order_change in (False, True):
        for aisle_change in (False, True):
            subset = interior[((interior.order != interior.previous_order) == order_change) &
                              ((interior.aisle != interior.previous_aisle) == aisle_change)]
            label = ("new_order" if order_change else "same_order") + "_" + (
                "new_aisle" if aisle_change else "same_aisle")
            transitions[label] = {
                "lines": len(subset),
                "start_before_previous_end": int((subset.start_s < subset.previous_end).sum()),
                "end_before_previous_end": int((subset.end_s < subset.previous_end).sum()),
                "median_recorded_s": float(subset.recorded_s.median()),
                "median_graph_walk_m": float(subset.walk_distance.median() / 1000),
                "travel_only_infeasible_2_306_m_s": int((
                    subset.recorded_s < subset.walk_distance / 2306).sum()),
                "travel_only_infeasible_4_m_s": int((
                    subset.recorded_s < subset.walk_distance / 4000).sum()),
                "position_decreases": int((subset.position < subset.previous_position).sum()),
                "infeasible_4_m_s_with_position_decrease": int((
                    (subset.recorded_s < subset.walk_distance / 4000) &
                    (subset.position < subset.previous_position)).sum()),
            }
    errors = np.asarray([error for _, error in order_errors])
    by_day_rmse = {day: float(np.sqrt(np.mean(np.asarray([
        error for sample_day, error in order_errors if sample_day == day]) ** 2)))
        for day in sorted({day for day, _ in order_errors})}
    single_tour_errors = np.asarray([error for size, error in order_tour_sizes if size == 1])
    multi_tour_errors = np.asarray([error for size, error in order_tour_sizes if size > 1])
    route_distances = np.asarray(single_tour_route_distance_pairs)
    picker_route_distances = np.asarray(picker_day_route_distance_pairs)
    end_gaps = pd.concat(end_gap_samples, ignore_index=True)
    forward_errors = np.asarray([error for day, error in order_errors
                                 if day in {"2026-04-16", "2026-04-17"}])
    saturday_errors = np.asarray([error for day, error in order_errors
                                  if day == "2026-04-11"])
    forward_with_pallet_overhead = np.asarray([
        error + 215 * (pallets - 1) / 60
        for day, pallets, error in order_pallets
        if day in {"2026-04-16", "2026-04-17"}])
    forward_lines = validation[validation.day.isin(["2026-04-16", "2026-04-17"])]
    saturday_lines = validation[validation.day == "2026-04-11"]
    def line_mae(part, fitted):
        estimate = (fit["base_s"] + fit["per_unit_s"] * part.quantity +
                    part.walk_distance / 1000 / fit["speed_m_s"] if fitted
                    else 19 + part.walk_distance / 2306)
        return float(np.mean(abs(estimate - part.recorded_s)))
    return {
        "days": len(samples), "lines": len(frame), "orders": len(errors),
        "tours": tour_count,
        "recorded_tour_sequence_mismatches": sequence_mismatches,
        "end_order_reordered_lines": end_order_reordered_lines,
        "start_order_picker_day_interior_km": float(picker_route_distances[:, 0].sum() / 1e6),
        "end_order_picker_day_interior_km": float(picker_route_distances[:, 1].sum() / 1e6),
        "same_order_end_sorted_links": same_order_end_sorted_links,
        "same_order_start_at_previous_completion_within_1s": (
            same_order_end_sorted_start_at_previous_end),
        "return_to_open_order_links": return_to_open_order_links,
        "return_to_open_order_start_at_own_previous_end": (
            return_to_open_order_start_at_own_previous_end),
        "return_to_open_order_start_before_global_previous_end": (
            return_to_open_order_start_before_global_previous_end),
        "return_to_open_order_own_match_and_global_overlap": (
            return_to_open_order_own_match_and_global_overlap),
        "return_to_open_order_overlap_hours": return_to_open_order_overlap_hours,
        "completion_order_global_gaps": len(end_gaps),
        "completion_order_global_gaps_at_most_1s": int((end_gaps.gap_s <= 1).sum()),
        "completion_order_global_gaps_shorter_than_graph_walk_at_2_306_m_s": int((
            end_gaps.gap_s < end_gaps.walk_mm / 2306).sum()),
        "completion_order_short_gap_and_infeasible_at_2_306_m_s": int((
            (end_gaps.gap_s <= 1) &
            (end_gaps.gap_s < end_gaps.walk_mm / 2306)).sum()),
        "completion_order_global_gaps_shorter_than_graph_walk_at_4_m_s": int((
            end_gaps.gap_s < end_gaps.walk_mm / 4000).sum()),
        "completion_order_same_order_gap_median_s": float(
            end_gaps.loc[~end_gaps.order_change, "gap_s"].median()),
        "completion_order_switch_gap_median_s": float(
            end_gaps.loc[end_gaps.order_change, "gap_s"].median()),
        "order_span_rmse_min": float(np.sqrt(np.mean(errors ** 2))),
        "order_span_rmse_by_day_min": by_day_rmse,
        "single_order_tour_orders": len(single_tour_errors),
        "single_order_tour_rmse_min": float(np.sqrt(np.mean(single_tour_errors ** 2))),
        "single_tour_timestamp_sequence_interior_km": float(route_distances[:, 0].sum() / 1e6),
        "single_tour_position_sequence_interior_km": float(route_distances[:, 1].sum() / 1e6),
        "single_tours_position_route_changes_distance_over_10pct": int((
            abs(route_distances[:, 1] - route_distances[:, 0]) >
            .1 * np.maximum(route_distances[:, 0], 1)).sum()),
        "multi_order_tour_orders": len(multi_tour_errors),
        "multi_order_tour_rmse_min": float(np.sqrt(np.mean(multi_tour_errors ** 2))),
        "multi_order_tour_squared_error_share": float(np.sum(multi_tour_errors ** 2) /
                                                        np.sum(errors ** 2)),
        "forward_thu_fri_order_span_rmse_min": float(np.sqrt(np.mean(forward_errors ** 2))),
        "earlier_sat_order_span_rmse_min": float(np.sqrt(np.mean(saturday_errors ** 2))),
        "orders_with_multiple_pallets": sum(pallets > 1 for _, pallets, _ in order_pallets),
        "diagnostic_forward_rmse_with_215_s_per_extra_pallet_min": float(
            np.sqrt(np.mean(forward_with_pallet_overhead ** 2))),
        "cross_aisle_edge_m": float(cross_edges[0]),
        "interior_lines": len(interior),
        "travel_only_infeasible_2_306_m_s": int((interior.recorded_s <
                                                  interior.walk_distance / 2306).sum()),
        "travel_only_infeasible_4_m_s": int((interior.recorded_s <
                                              interior.walk_distance / 4000).sum()),
        "orders_shorter_than_travel_with_depot_origin_zero_pick_time": orders_shorter_than_travel,
        "orders_shorter_than_interior_travel_zero_pick_time": orders_shorter_than_interior_travel,
        "single_tour_orders_shorter_than_interior_travel_zero_pick_time": (
            single_tour_orders_shorter_than_interior_travel),
        "adjacent_0_8m_lines": len(adjacent),
        "adjacent_0_8m_recorded_median_s": float(adjacent.recorded_s.median()),
        "adjacent_0_8m_recorded_at_most_1s": int((adjacent.recorded_s <= 1).sum()),
        "interior_lines_at_most_1s": len(short),
        "at_most_1s_after_node_change": len(short_changed_node),
        "order_and_aisle_changes": len(change_both),
        "overlapping_order_and_aisle_changes": int((change_both.start_s <
                                                     change_both.previous_end).sum()),
        "order_and_aisle_changes_with_end_before_previous_end": int(
            (change_both.end_s < change_both.previous_end).sum()),
        "transition_consistency": transitions,
        # This subset is selected partly by the observed outcome. Its speed
        # estimate is diagnostic and must not replace the scenario default.
        "ordinary_motion_fit": fit,
        "leave_one_day_out_speed_m_s_min": float(min(leave_one_out)),
        "leave_one_day_out_speed_m_s_max": float(max(leave_one_out)),
        "heldout_ordinary_lines": len(validation),
        "heldout_fit_line_mae_s": float(np.mean(abs(prediction - validation.recorded_s))),
        "fixed_speed_2_306_m_s_pick_fit": {
            "base_s": float(fixed_base), "per_unit_s": float(fixed_per_unit),
            "heldout_line_mae_s": float(np.mean(abs(fixed_prediction - validation.recorded_s))),
        },
        "heldout_submitted_line_mae_s": float(np.mean(abs(submitted - validation.recorded_s))),
        "forward_thu_fri_ordinary_lines": len(forward_lines),
        "forward_thu_fri_fit_line_mae_s": line_mae(forward_lines, True),
        "forward_thu_fri_submitted_line_mae_s": line_mae(forward_lines, False),
        "earlier_sat_ordinary_lines": len(saturday_lines),
        "earlier_sat_fit_line_mae_s": line_mae(saturday_lines, True),
        "earlier_sat_submitted_line_mae_s": line_mae(saturday_lines, False),
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    print(json.dumps(analyze(root), indent=2))
