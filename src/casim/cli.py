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
from ware_ops_algos.domain_models.lifecycle import (
    Accepts, OrderStates, PickListStates, ResourceStates, TourStates,
)

from casim.config.schema import DecisionDefinitions
from casim.setup import resolve


def catalog() -> dict:
    cards = load_packaged_algo_cards()
    return {
        "lifecycle": {
            "orders": [item.value for item in OrderStates],
            "pick_lists": [item.value for item in PickListStates],
            "tours": [item.value for item in TourStates],
            "resources": [item.value for item in ResourceStates],
            "accepts": [item.value for item in Accepts],
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
    check = commands.add_parser("check")
    check.add_argument("config_dir", type=Path)
    check.add_argument("config_name")
    check.add_argument("overrides", nargs="*")
    check.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(DecisionDefinitions.model_json_schema(), indent=2))
        return 0
    if args.command == "catalog":
        print(json.dumps(catalog(), indent=2))
        return 0

    from scenarios.experiment_commons import load_and_flatten_data_card

    with initialize_config_dir(config_dir=str(args.config_dir.resolve()),
                               version_base="1.3"):
        cfg = compose(config_name=args.config_name, overrides=args.overrides,
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
            "selection": setup.config.selection.model_dump(mode="json", exclude_defaults=True),
            "scope": setup.config.scope.model_dump(mode="json"),
            "waiting": setup.waiting.model_dump(mode="json", exclude_defaults=True),
            "commitment": setup.commitment.model_dump(mode="json", exclude_defaults=True),
            "release_gates": setup.config.release_gates.model_dump(mode="json", exclude_defaults=True),
        }
        for name, setup in setups.items()
    }
    if args.as_json:
        findings = {"on_violation": cfg.get("on_violation", "reject"),
                    "decisions": findings}
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
            print(f"  selection: {finding['selection']}; "
                  f"scope: {finding['scope']}")
            print(f"  features: {finding['decision_features']}; "
                  f"writes: {finding['write_set']}")
            print(f"  waiting: {finding['waiting']}; "
                  f"commitment: {finding['commitment']}; "
                  f"release gates: {finding['release_gates']}")
            for excluded in finding['exclusions']:
                print(f"  excluded {excluded['algorithm']}: "
                      + "; ".join(excluded['reasons']))
    return 0
