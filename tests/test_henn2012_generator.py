"""L2 generator facts that the two published targets share."""

from pathlib import Path

from scenarios.scenario_henn2012.generator import generate_instance


def test_henn_generator_shares_orders_across_capacity_and_arrival_variant(tmp_path: Path):
    base = dict(storage="C", n_orders=30, replication=0)
    forty_five = generate_instance(tmp_path, capacity=45, **base)
    seventy_five = generate_instance(tmp_path, capacity=75, **base)
    printed = generate_instance(tmp_path, capacity=45, printed_rate=True, **base)
    root = tmp_path / "HennOOBP"
    def order_lines(name):
        return (root / name).read_text(encoding="ascii").replace("-45-", "-X-").replace(
            "-75-", "-X-")
    assert order_lines(forty_five) == order_lines(seventy_five) == order_lines(printed)
    assert len([line for line in (root / forty_five).read_text().splitlines()
                if line.startswith("Order ")]) == 30
    stated_arrivals = (root / forty_five).with_name(
        "Arrival_Time_1s-30-45-0.csv").read_text().splitlines()
    printed_arrivals = (root / printed).with_name(
        "Arrival_Time_1s-30-45-0.csv").read_text().splitlines()
    assert stated_arrivals != printed_arrivals
