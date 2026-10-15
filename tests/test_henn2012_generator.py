"""L2 generator facts that the two published targets share."""

from pathlib import Path

from scenarios.scenario_henn2012.generator import generate_instance


def test_henn_generator_shares_orders_across_capacity(tmp_path: Path):
    base = dict(storage="C", n_orders=30, replication=0, arrival_mode="conditioned")
    forty_five = generate_instance(tmp_path, capacity=45, **base)
    seventy_five = generate_instance(tmp_path, capacity=75, **base)
    root = tmp_path / "HennOOBP"
    def order_lines(name):
        return (root / name).read_text(encoding="ascii").replace("-45-", "-X-").replace(
            "-75-", "-X-")
    assert order_lines(forty_five) == order_lines(seventy_five)
    assert len([line for line in (root / forty_five).read_text().splitlines()
                if line.startswith("Order ")]) == 30


def test_henn_generator_arrival_modes_differ(tmp_path: Path):
    base = dict(storage="C", n_orders=30, replication=0)
    conditioned = generate_instance(tmp_path, capacity=45, arrival_mode="conditioned", **base)
    truncated = generate_instance(tmp_path, capacity=45, arrival_mode="truncated", **base)
    root = tmp_path / "HennOOBP"
    conditioned_arrivals = (root / conditioned).with_name(
        "Arrival_Time_1s-30-45-0.csv").read_text().splitlines()
    truncated_arrivals = (root / truncated).with_name(
        "Arrival_Time_1s-30-45-0.csv").read_text().splitlines()
    assert conditioned_arrivals != truncated_arrivals
