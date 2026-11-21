"""Fit an order generator on the canonical IJPE tables and generate weeks.

The partner allows a generated or fitted order stream to be published, not
the recorded one (plan, P5 release). Four generators are fitted on
``scenarios/ijpe/data/canonical`` and write the same canonical format, so a
generated week runs as ``source.tables`` of ``scenarios/ijpe_week``:

- ``G0``, the logic of the submitted generator (``generate_profile`` and
  ``generate_n_orders``): per day the recorded number of orders; per order a
  line count from the empirical distribution, articles by pick frequency over
  the whole area without replacement, a quantity from the article's recorded
  quantities. The submitted generator had no stores; here each order draws a
  store by frequency.
- ``G1``, zone-aware: an order is a zone order of a store-day. Per day the
  recorded number of store-days, stores drawn by frequency without
  replacement; per store-day the set of zones it orders from (its pattern) by
  frequency; per zone order a line count from that zone's distribution and
  articles by pick frequency within the zone.
- ``G2``, ``G1`` fitted per day: the store-day patterns and the zone order
  sizes of each day, so each day's load matches the recorded day; a zone
  order draws only articles stored in the zone's aisles.
- ``G3``, ``G2`` with stores and articles as in the data: each store-day is a
  profile (its zone orders and their line counts) resampled from the same
  day, the draw whose total lines come closest to the recorded day; a zone
  order includes each article with a probability proportional to its share of
  the zone's lines (systematic sampling), so article frequencies match; each
  line takes a quantile of its article's recorded quantities through a normal
  score with a per-store effect (one fitted share of variance between stores),
  so stores that order large quantities do so on every line and day.

Zones are found in the data: the two most frequent disjoint sets of aisles
that orders cover; every aisle belongs to the zone whose set holds it, and an
order to the zone with most of its lines. Articles keep their pick position;
article, store and order identifiers are new. Generated tables go to
``scenarios/ijpe/data/generated/<generator>_<seed>/`` (ignored with the
confidential data until the partner approves a release); only aggregates are
printed.

    python -m scripts.ijpe.order_generator --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "scenarios" / "ijpe" / "data" / "canonical"
OUT = ROOT / "scenarios" / "ijpe" / "data" / "generated"
RELEASE = "05:30"
NORMAL = NormalDist()
DRAWS = 200  # store-day profile samples per day; the one closest to the day's lines is kept


def load(tables: Path = CANONICAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(tables / "orders.csv")
    lines = pd.read_csv(tables / "lines.csv", usecols=["order_id", "article", "quantity", "x", "y"])
    return orders, lines.merge(orders[["order_id", "day", "customer"]], on="order_id")


def aisle_of(lines: pd.DataFrame) -> pd.Series:
    """Aisle number from the rack face: two neighbouring faces form an aisle."""
    faces = sorted(lines["x"].round(-2).unique())
    rank = {face: i // 2 + 1 for i, face in enumerate(faces)}
    return lines["x"].round(-2).map(rank)


def zone_of(lines: pd.DataFrame) -> dict[int, int]:
    """Aisle to zone: the two most frequent disjoint aisle sets of orders."""
    sets = lines.groupby("order_id")["aisle"].agg(lambda s: frozenset(s))
    ranked = [s for s, _ in Counter(sets).most_common() if len(s) > 1]
    first = ranked[0]
    second = next(s for s in ranked if not s & first)
    zones = {a: 0 for a in first} | {a: 1 for a in second}
    for aisle in lines["aisle"].unique():
        if aisle not in zones:
            near = min(zones, key=lambda a: abs(a - aisle))
            zones[aisle] = zones[near]
    return zones


def fit(orders: pd.DataFrame, lines: pd.DataFrame) -> dict:
    lines = lines.assign(aisle=aisle_of(lines))
    zones = zone_of(lines)
    lines["zone"] = lines["aisle"].map(zones)
    order_zone = lines.groupby("order_id")["zone"].agg(lambda s: s.value_counts().idxmax())
    lines["order_zone"] = lines["order_id"].map(order_zone)
    articles = lines.groupby("article").agg(x=("x", "first"), y=("y", "first"), picks=("x", "size"))
    store_days = orders.assign(zone=orders["order_id"].map(order_zone)).groupby(["day", "customer"])
    return {
        "days": sorted(orders["day"].unique()),
        "orders_per_day": orders.groupby("day").size().to_dict(),
        "store_days_per_day": orders.groupby("day")["customer"].nunique().to_dict(),
        "store_freq": orders["customer"].value_counts(normalize=True),
        "articles": articles,
        "quantities": lines.groupby("article")["quantity"].apply(np.array).to_dict(),
        "lines_per_order": lines.groupby("order_id").size().to_numpy(),
        "zone_lines_per_order": {
            z: lines[lines["order_zone"] == z].groupby("order_id").size().to_numpy()
            for z in sorted(set(order_zone))},
        "zone_articles": {
            z: lines[lines["order_zone"] == z]["article"].value_counts()
            for z in sorted(set(order_zone))},
        "patterns": store_days["zone"].agg(lambda s: tuple(sorted(s))).value_counts(normalize=True),
        "day_patterns": {
            day: group.groupby("customer")["zone"].agg(lambda s: tuple(sorted(s)))
            .value_counts(normalize=True)
            for day, group in orders.assign(zone=orders["order_id"].map(order_zone)).groupby("day")},
        "day_zone_lines_per_order": {
            (day, z): group.groupby("order_id").size().to_numpy()
            for (day, z), group in lines.groupby(["day", "order_zone"])},
        "zone_pool": {
            z: lines[lines["zone"] == z]["article"].value_counts() for z in sorted(set(zones.values()))},
        "day_profiles": {
            day: [tuple(sorted(p)) for p in group.groupby(["customer"])[["order_zone", "n"]]
                  .apply(lambda g: list(zip(g["order_zone"], g["n"])))]
            for day, group in lines.groupby(["day", "customer", "order_id"], as_index=False)
            .agg(order_zone=("order_zone", "first"), n=("article", "size")).groupby("day")},
        "day_lines": lines.groupby("day").size().to_dict(),
        "sorted_quantities": lines.groupby("article")["quantity"].apply(np.sort).to_dict(),
        "store_share": store_share(lines),
    }


def store_share(lines: pd.DataFrame) -> float:
    """Share of the variance of quantity normal scores that lies between stores.

    A line's score is the normal quantile of its quantity's mid-rank among its
    article's recorded quantities; the between-store variance is the variance of
    the store means less their sampling variance (one-way random effects).
    """
    rank = lines.groupby("article")["quantity"].rank(method="average")
    count = lines.groupby("article")["quantity"].transform("size")
    score = ((rank - 0.5) / count).clip(1e-6, 1 - 1e-6).map(NORMAL.inv_cdf)
    by_store = score.groupby(lines["customer"])
    within = by_store.var().mean()
    between = by_store.mean().var() - (by_store.var() / by_store.size()).mean()
    return float(np.clip(between / (between + within), 0.0, 1.0))


def _systematic(rng, pool: pd.Series, n: int) -> np.ndarray:
    """n distinct articles, each included with probability n times its share."""
    ids, share = pool.index.to_numpy(), pool.to_numpy(float) / pool.sum()
    n = min(n, len(ids))
    order = rng.permutation(len(ids))
    ids, weight = ids[order], n * share[order]
    capped = np.zeros(len(ids), bool)
    while (weight > 1).any():  # an article cannot be in an order twice
        capped |= weight >= 1
        weight[capped] = 1.0
        rest = ~capped
        weight[rest] *= (n - capped.sum()) / weight[rest].sum()
    points = rng.uniform() + np.arange(n)
    return ids[np.searchsorted(np.cumsum(weight), points, side="right").clip(max=len(ids) - 1)]


def _articles(rng, popularity: pd.Series, n: int) -> np.ndarray:
    probs = popularity.to_numpy(float) / popularity.sum()
    n = min(n, len(probs))
    return popularity.index.to_numpy()[rng.choice(len(probs), size=n, replace=False, p=probs)]


def generate(cal: dict, generator: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stores = cal["store_freq"]
    store_name = {s: f"S{i:03d}" for i, s in enumerate(stores.index, 1)}
    popularity = cal["articles"]["picks"]
    position = dict(zip(cal["articles"].index, zip(cal["articles"]["x"], cal["articles"]["y"])))
    order_rows, line_rows = [], []

    def add_order(day, store, n_lines, pool):
        order_id = 9_000_000 + len(order_rows)  # the source reads integer order ids
        order_rows.append((order_id, day, f"{day} {RELEASE}", store_name[store]))
        for article in _articles(rng, pool, n_lines):
            quantity = int(rng.choice(cal["quantities"][article]))
            line_rows.append((order_id, article, quantity, *position[article]))

    store_effect = {}

    def g3_order(day, store, zone, n_lines):
        order_id = 9_000_000 + len(order_rows)
        order_rows.append((order_id, day, f"{day} {RELEASE}", store_name[store]))
        effect = store_effect.setdefault(store, rng.normal(0.0, np.sqrt(cal["store_share"])))
        for article in _systematic(rng, cal["zone_pool"][zone], n_lines):
            score = effect + rng.normal(0.0, np.sqrt(1.0 - cal["store_share"]))
            values = cal["sorted_quantities"][article]
            quantity = int(values[min(int(NORMAL.cdf(score) * len(values)), len(values) - 1)])
            line_rows.append((order_id, article, quantity, *position[article]))

    for day in cal["days"]:
        if generator == "G3":
            profiles, target = cal["day_profiles"][day], cal["day_lines"][day]
            m = cal["store_days_per_day"][day]
            draws = [rng.integers(len(profiles), size=m) for _ in range(DRAWS)]
            totals = [sum(n for i in d for _, n in profiles[i]) for d in draws]
            chosen = draws[int(np.argmin(np.abs(np.array(totals) - target)))]
            present = rng.choice(stores.index, size=m, replace=False, p=stores.to_numpy())
            for store, i in zip(present, chosen):
                for zone, n_lines in profiles[i]:
                    g3_order(day, store, zone, int(n_lines))
            continue
        if generator == "G0":
            for store in rng.choice(stores.index, size=cal["orders_per_day"][day], p=stores.to_numpy()):
                add_order(day, store, int(rng.choice(cal["lines_per_order"])), popularity)
            continue
        present = rng.choice(stores.index, size=cal["store_days_per_day"][day], replace=False,
                             p=stores.to_numpy())
        patterns = cal["patterns"] if generator == "G1" else cal["day_patterns"][day]
        for store in present:
            for zone in patterns.index[rng.choice(len(patterns), p=patterns.to_numpy())]:
                if generator == "G1":
                    sizes, pool = cal["zone_lines_per_order"][zone], cal["zone_articles"][zone]
                else:
                    sizes, pool = cal["day_zone_lines_per_order"][day, zone], cal["zone_pool"][zone]
                add_order(day, store, int(rng.choice(sizes)), pool)
    orders = pd.DataFrame(order_rows, columns=["order_id", "day", "release", "customer"])
    lines = pd.DataFrame(line_rows, columns=["order_id", "article", "quantity", "x", "y"])
    # New integer article ids by pick frequency (the source reads integers).
    article_name = {a: i for i, a in enumerate(popularity.sort_values(ascending=False).index, 1)}
    return orders, lines.assign(article=lines["article"].map(article_name))


def write(orders: pd.DataFrame, lines: pd.DataFrame, name: str) -> Path:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    orders.to_csv(out / "orders.csv", index=False)
    lines.to_csv(out / "lines.csv", index=False)
    shutil.copy2(CANONICAL / "layout.csv", out / "layout.csv")
    return out


def statistics(orders: pd.DataFrame, lines: pd.DataFrame) -> dict:
    """Aggregates of a week that matter to the model."""
    lines = lines.merge(orders[["order_id", "day", "customer"]], on="order_id", suffixes=("", "_o"))
    lines = lines.assign(aisle=aisle_of(lines))
    per_order = lines.groupby("order_id").agg(n=("article", "size"), aisles=("aisle", "nunique"),
                                              faces=("x", "nunique"))
    store_day = lines.groupby(["day", "customer"]).size().sort_values(ascending=False)
    picks = lines["article"].value_counts()
    share = lines["aisle"].value_counts(normalize=True).sort_index()
    return {
        "orders": len(orders),
        "lines": len(lines),
        "store-days": orders.groupby("day")["customer"].nunique().sum(),
        "lines/order p10": per_order["n"].quantile(.1),
        "lines/order p50": per_order["n"].median(),
        "lines/order p90": per_order["n"].quantile(.9),
        "aisles/order mean": per_order["aisles"].mean(),
        "orders with <= 2 aisles": (per_order["aisles"] <= 2).mean(),
        "orders with >= 6 aisles": (per_order["aisles"] >= 6).mean(),
        "rack faces/order p50": per_order["faces"].median(),
        "share of lines, aisles 7-8": share.loc[share.index >= 7].sum(),
        "share of lines, top 50 articles": picks.head(50).sum() / len(lines),
        "articles picked": lines["article"].nunique(),
        "quantity/line p50": lines["quantity"].median(),
        "quantity/line p90": lines["quantity"].quantile(.9),
        "orders/store-day mean": orders.groupby(["day", "customer"]).size().mean(),
        "lines/store-day p90": store_day.quantile(.9),
        "share of lines, top 10 % store-days": store_day.head(max(1, len(store_day) // 10)).sum()
        / len(lines),
    }


def main(seeds: list[int]) -> None:
    orders, lines = load()
    cal = fit(orders, lines)
    table = {"recorded": statistics(orders, lines.drop(columns=["day", "customer"]))}
    for generator in ("G0", "G1", "G2", "G3"):
        rows = []
        for seed in seeds:
            g_orders, g_lines = generate(cal, generator, seed)
            write(g_orders, g_lines, f"{generator}_{seed}")
            rows.append(statistics(g_orders, g_lines))
        table[f"{generator} (mean of {len(seeds)})"] = pd.DataFrame(rows).mean()
    print(pd.DataFrame(table).to_string(float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    main(parser.parse_args().seeds)
