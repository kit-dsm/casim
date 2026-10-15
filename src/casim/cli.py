"""Inspect the same decision configuration used by experiments."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from ware_ops_algos.algorithms.algorithm_cards import load_packaged_algo_cards
from ware_ops_algos.planning.lifecycle import (
    Accepts, OrderStates, PHYSICAL, ResourceStates, TOUR_TRANSITIONS, TourStates,
)

from casim.config.schema import DecisionDefinitions
from casim.config.data_card import load_and_flatten_data_card
from casim.run import SCENARIO_FILE, compose_scenario, run, sweep_overrides
from casim.scenario_check import check_all
from casim.setup import resolve


def catalog() -> dict:
    cards = load_packaged_algo_cards()
    return {
        "lifecycle": {
            "orders": [item.value for item in OrderStates],
            "tours": [item.value for item in TourStates],
            "resources": [item.value for item in ResourceStates],
            "accepts": [item.value for item in Accepts],
            "transitions": {status.value: [next_status.value for next_status in options]
                            for status, options in TOUR_TRANSITIONS.items()},
            "physical": {status.value: level.value for status, level in PHYSICAL.items()},
        },
        "algorithms": sorted(
            ({"name": card.algo_name, "problem_type": card.problem_type,
              "configuration": card.configuration} for card in cards),
            key=lambda card: card["name"],
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="casim")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema")
    commands.add_parser("catalog")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("scenario_dir", type=Path, nargs="?", default=Path("."))
    run_parser.add_argument("overrides", nargs="*")
    run_parser.add_argument("--audit", action="store_true")
    run_parser.add_argument("--output-dir", type=Path)
    check = commands.add_parser("check")
    check.add_argument("config_dir", type=Path, nargs="?", default=Path("."),
                       help="a scenario directory, or a legacy config directory")
    check.add_argument("arguments", nargs="*",
                       help="overrides; for a legacy config directory, its config name first")
    check.add_argument("--json", action="store_true", dest="as_json")
    check.add_argument("--all", action="store_true", dest="check_all")
    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(DecisionDefinitions.model_json_schema(), indent=2))
        return 0
    if args.command == "catalog":
        print(json.dumps(catalog(), indent=2))
        return 0
    if args.command == "run":
        base = compose_scenario(args.scenario_dir, args.overrides, args.output_dir)
        combinations = sweep_overrides(base)
        if not combinations:
            base.experiment.audit = args.audit
            result = run(base)
            print(f"outputs: {result.output_dir}; trace digest {result.trace_digest} "
                  f"({result.event_count} events)")
            return 0
        failures = 0
        for combination in combinations:
            output_dir = Path(base.experiment.output_dir) / ",".join(
                item.replace("/", "_") for item in combination)
            cfg = compose_scenario(args.scenario_dir, [*args.overrides, *combination],
                                   output_dir)
            cfg.experiment.audit = args.audit
            # A failed combination is a result of the sweep; the others still run.
            try:
                result = run(cfg)
            except ValueError as error:
                failures += 1
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "error.txt").write_text(str(error), encoding="utf-8")
                print(f"failed: {output_dir}: {error}")
                continue
            print(f"outputs: {result.output_dir}; trace digest {result.trace_digest} "
                  f"({result.event_count} events)")
        print(f"{len(combinations) - failures} of {len(combinations)} runs completed")
        return 1 if failures else 0
    if args.check_all:
        findings = check_all(Path.cwd())
        for finding in findings:
            print(finding)
        return 1 if findings else 0

    if (args.config_dir / SCENARIO_FILE).exists():
        cfg = compose_scenario(args.config_dir, args.arguments)
    else:
        with initialize_config_dir(config_dir=str(args.config_dir.resolve()),
                                   version_base="1.3"):
            cfg = compose(config_name=args.arguments[0], overrides=args.arguments[1:],
                          return_hydra_config=True)
            cfg.hydra.runtime.output_dir = str(Path.cwd() / "tmp_output" / "check")
            HydraConfig.instance().set_config(cfg)
    data_card = load_and_flatten_data_card(cfg.data_card)
    if args.as_json:
        with redirect_stdout(StringIO()):
            try:
                setups = resolve(cfg, data_card)
            except Exception as error:
                print(json.dumps({"error": str(error)}), file=sys.__stdout__)
                return 1
    else:
        setups = resolve(cfg, data_card)
    findings = {
        name: {
            "pipelines": setup.pipeline_count,
            "problem_class": setup.problem_class,
            "decision_features": setup.decision_features,
            "exclusions": setup.exclusions,
            "write_set": setup.write_set,
            "precedence": setup.config.precedence,
            "triggers": [event.__name__ for event in setup.trigger_types],
            "every_s": setup.period_s,
            "conditions": [type(condition).__name__
                           for condition in setup.conditions],
            "state_information": {
                "orders": setup.config.state_information.orders,
                "pick_lists": setup.config.state_information.pick_lists,
                "tours": sorted(s.value for s in setup.config.state_information.tours),
                "resources": sorted(s.value for s in setup.config.state_information.resources),
            },
            "scope": {"freeze_from": setup.config.scope.freeze_from.value,
                      "frozen_accepts": setup.config.scope.frozen_accepts.value},
            "subproblems": setup.subproblems,
            "commitment_policy": (type(setup.commitment_policy).__name__
                                  if setup.commitment_policy else None),
        }
        for name, setup in setups.items()
    }
    if args.as_json:
        findings = {"decisions": findings}
    if args.as_json:
        print(json.dumps(findings, indent=2))
    else:
        for name, finding in findings.items():
            print(f"{name}: {finding['problem_class']}, "
                  f"{finding['pipelines']} pipeline(s), "
                  f"precedence {finding['precedence']}")
            print(f"  triggers: {finding['triggers']}; "
                  f"every_s: {finding['every_s']}; "
                  f"conditions: {finding['conditions']}")
            print(f"  state_information: {finding['state_information']}; "
                  f"scope: {finding['scope']}")
            print(f"  features: {finding['decision_features']}; "
                  f"writes: {finding['write_set']}")
            print(f"  subproblems: {finding['subproblems']}; "
                  f"commitment_policy: {finding['commitment_policy']}")
            for excluded in finding['exclusions']:
                print(f"  excluded {excluded['algorithm']}: "
                      + "; ".join(excluded['reasons']))
    return 0
