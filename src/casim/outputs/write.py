"""Write the outputs of a finished run; every file has exactly one writer here.

- ``kpis.json``: the KPI summary (``casim.outputs.kpis.summary``);
- ``tracker.json``: time series for plots (``casim.outputs.kpis.timeline``);
- ``decisions.json``: the number of decisions, chosen pipelines and the log;
- ``service.json``: order service and picker presence, and ``stranded``, the
  committed tours left with a picker who is away at the end;
- ``trace.json``: the digest of the event trace;
- ``facts/tours.csv``, ``facts/orders.csv``, ``facts/picks.csv``: the
  realised facts (``casim.outputs.facts``).
"""

from __future__ import annotations

import json
from pathlib import Path

from ware_ops_algos.planning.lifecycle import TourStates

from casim.outputs.facts import FactsRecorder
from casim.outputs.kpis import jsonable, summary, timeline
from casim.outputs.service import service_measures
from casim.outputs.trace import TraceRecorder


def _json(path: Path, data) -> None:
    path.write_text(json.dumps(jsonable(data), indent=2), encoding="utf-8")


def write_outputs(sim, engine, output_dir: Path, recorder: TraceRecorder,
                  facts: FactsRecorder) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / "kpis.json", summary(sim.state))
    _json(output_dir / "tracker.json", timeline(sim.state.tracker))
    log = engine.decision_tracker.command_decisions
    _json(output_dir / "decisions.json", {
        "decisions": engine.decision_tracker.num_decisions,
        "pipeline_counts": dict(engine.decision_tracker.pipeline_counts),
        "decision_log": log,
        "solve_seconds": sum(row["solve_seconds"] for row in log),
        "conversion_seconds": sum(row["conversion_seconds"] for row in log),
        "apply_seconds": sum(row["apply_seconds"] for row in log),
    })
    _json(output_dir / "trace.json", {"trace_digest": recorder.digest(),
                                      "event_count": len(recorder.trace)})
    facts.write(sim, output_dir)
    service = service_measures(facts.orders(sim), sim.state.tracker)
    service["stranded"] = stranded_work(sim)
    if service["stranded"]["tours"]:
        print(f"warning: {service['stranded']['tours']} committed tours "
              f"({service['stranded']['orders']} orders) never started: their pickers "
              "were away at the end (work pushed into the queue of a picker who left).")
    _json(output_dir / "service.json", service)


def stranded_work(sim) -> dict:
    """Committed tours left in the queue of a picker who is away at the end."""
    resources = sim.state.resource_manager
    tours = [tour for tour in sim.state.tour_manager.get_all_tours()
             if tour.status in (TourStates.SCHEDULED, TourStates.PENDING)
             and tour.assigned_resource is not None
             and not resources.is_present(tour.assigned_resource)]
    return {"tours": len(tours), "orders": sum(len(tour.order_numbers) for tour in tours)}
