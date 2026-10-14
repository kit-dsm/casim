"""Choose the L1 pipelines and the stratified L1 subset from CASOP results.

Run from the ware_ops_pipes root:
    PYTHONPATH=experiments/evaluation uv run python <casim>/docs/literature_benchmarks/select_l1_subset.py l1_subset.csv 4
The second argument is the CASIM overhead per run in seconds. One row per
instance and pipeline; `smoke` marks the 3 runs per set and pipeline of the
smoke stage. Pipelines that need an integrated stage (RatliffRosenthalNF,
CombinedBatchingRoutingAssigning) are replaced by the closest expressible one.
"""
import importlib.util, re, sys
import numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("cmp", "experiments/evaluation/03_compare_vbs_to_references.py")
cmp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmp)
SEED, OVERHEAD = 20261014, float(sys.argv[2])
PIPES = {
    "SPRP": ["GIA+SingleOrderBatching+RR", "GIA+SingleOrderBatching+SShape"],
    "SPRP-SS": ["MinMinIA+SingleOrderBatching+RR", "MinMinIA+SingleOrderBatching+SShape"],
    "BahceciOencan": ["GIA+LSBatchingRROrderNrFiFo+RR", "GIA+LSBatchingSShapeFiFoOrderNr+SShape"],
    "HennWaescher": ["GIA+LSBatchingRROrderNrFiFo+RR", "GIA+SavingsSShape+SShape"],
    "MuterOencanWG": ["GIA+LSBatchingRROrderNrFiFo+RR", "GIA+SavingsSShape+SShape"],
    "Foodmart": ["GIA+LSOrdNrNN+NN", "GIA+LSBatchingSShapeFiFoOrderNr+SShape"],
    "Kris": ["GIA+DueDate+NN+EDD"],
}
CAP: dict[tuple[str, str], int] = {}
SMOKE = 3
PER_STRATUM = {"SPRP": 1, "SPRP-SS": 1, "BahceciOencan": 1, "HennWaescher": 1,
               "MuterOencanWG": 2, "Foodmart": None, "Kris": None}
def stratum(name):
    base = re.sub(r"(_u|_cb|_small|_large)$", "", name)
    tail = re.search(r"(_u|_cb|_small|_large)$", name)
    base = re.sub(r"[-_]\d+$", "", base)
    return base + (tail.group(1) if tail else "")
casop = cmp.load_casop()
refs = pd.concat([cmp.load_distance_refs(), cmp.load_kris_refs()], ignore_index=True)
rng = np.random.default_rng(SEED)
chosen, summary = [], []
for name, pipes in PIPES.items():
    x = casop[casop.instance_set.eq(name) & casop.strategy.isin(pipes)]
    r = refs[refs.instance_set.eq(name) & refs.instance_name.isin(x.instance_name)].copy()
    r["stratum"] = r.instance_name.map(stratum)
    k = PER_STRATUM[name]
    picks = []
    for s, g in r.groupby("stratum", sort=True):
        g = g.sort_values("instance_name")
        if k is None or len(g) <= k:
            picks.append(g); continue
        pool = g[g.reference_is_opt] if g.reference_is_opt.sum() >= k else g
        picks.append(pool.iloc[sorted(rng.choice(len(pool), size=k, replace=False))])
    p = pd.concat(picks)
    sub_rng = np.random.default_rng([SEED, len(summary)])
    runs, cpu = 0, 0.0
    for pipe in pipes:
        q = p.sort_values("instance_name")
        cap = CAP.get((name, pipe))
        if cap is not None:
            q = q.iloc[sorted(sub_rng.choice(len(q), size=cap, replace=False))]
        smoke = set(q.instance_name.iloc[sorted(sub_rng.choice(len(q), size=min(SMOKE, len(q)), replace=False))])
        runs += len(q)
        cpu += x[x.strategy.eq(pipe) & x.instance_name.isin(q.instance_name)].total_cpu_time.sum()
        for _, row in q.iterrows():
            chosen.append((name, pipe, row.instance_name, row.stratum, row.instance_name in smoke,
                           row.reference_value, bool(row.reference_is_opt), row.reference_source))
    seconds = cpu + runs * OVERHEAD
    summary.append((name, len(r), int(r.reference_is_opt.sum()), r.stratum.nunique(), len(p),
                    int(p.reference_is_opt.sum()), runs, round(cpu / 60, 1), round(seconds / 3600, 2)))
pd.DataFrame(chosen, columns=["instance_set", "pipeline", "instance_name", "stratum", "smoke", "reference_value", "reference_is_opt", "reference_source"]).to_csv(sys.argv[1], index=False)
s = pd.DataFrame(summary, columns=["set", "with_reference", "proven_opt", "strata", "subset", "subset_opt", "runs", "casop_cpu_min", "expected_h"])
print(s.to_string(index=False)); print("total expected h", round(s.expected_h.sum(), 2))
