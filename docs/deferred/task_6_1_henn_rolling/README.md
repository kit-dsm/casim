# Henn rolling horizon, deferred to Task 6.1

These three files were the rolling-horizon configuration of the Henn scenario
(`scenarios/scenario_henn_online/config/` until Phase S). The decision file
declares `problem_class: rolling`, which `casim check` rejects, so it never
ran (`henn__rolling` is `not_runnable` in `docs/baseline.md`). Phase S requires
every file of a scenario's groups to compose and validate, so they wait here
for Task 6.1, which ports them as the design document's rolling-horizon
strategy with `problem_class: OBRSP`. The data card used the synthetic due
date `due_offset_s: 5400`, now the source parameter `source.due_offset_s` of
`hessler_irnich`.
