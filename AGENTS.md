## Implementation style

Use an Occam's-razor approach.

- Prefer extending existing abstractions over introducing parallel ones.
- Prefer concrete code for the current requirement over speculative generality.
- Do not create an interface, abstract base class, protocol, factory, registry,
  adapter, wrapper, or result hierarchy unless the current task demonstrably
  requires it.
- One implementation is not evidence for an abstraction.
- Do not add extensibility for hypothetical future strategies.
- Prefer small functions and existing data structures over helper classes.
- Prefer local scenario-specific code over changes to generic CASIM packages.
- Move functionality into `src/casim` only when it is already needed in at
  least two concrete places.
- Avoid classes that only store data when an existing type, tuple, dictionary,
  or small dataclass already represents the concept adequately.
- Avoid forwarding wrappers that do not enforce additional behavior.
- Do not rename or reorganize unrelated code.
- Preserve the current control flow unless changing it is necessary to satisfy
  a tested requirement.
- Every new file and class must have a concrete current caller.

Before adding an abstraction, state:

1. the concrete problem it solves;
2. why the existing implementation cannot solve it;
3. the current callers that need it;
4. the simpler alternative considered;
5. why that alternative is insufficient.

If this justification is weak, use the simpler alternative.

Before adding a heuristic, policy, solver, or algorithm:

1. Search `ware_ops_algos`, `src/casim`, and the existing scenarios.
2. Identify the closest existing abstraction and configuration path.
3. State why extending that implementation cannot satisfy the requirement.
4. Identify every concrete current caller of the new implementation.
5. Do not add scenario-local helpers that duplicate established decision logic.
6. Do not introduce a new heuristic unless the user explicitly requested it,
   or its absence and concrete necessity have first been reported to the user.

## Scenario conventions

Start from `casim.setup.build_runtime(cfg)` where it naturally fits. Keep
simulation/decision control flow explicit in the experiment file so a reader
can follow `build -> reset -> run -> decide -> step -> report` without
learning infrastructure.

Scenario code should primarily contain study-specific inputs, hooks/events,
decision deviations, and reporting. Reuse small primitive helpers
(`casim.io_helpers.dump_json`, `casim.io_helpers.dump_jsonl`,
`casim.events.operational_events.add_orders_hook`) for genuinely generic
operations.

Do not introduce scenario base classes, runners, registries, callback
frameworks, or generic abstractions merely to eliminate a few repeated lines.
Prefer a small amount of obvious duplication over an abstraction that hides
the program's execution.

Refactoring must preserve behavior, performance, determinism, and research
semantics. Characterize behavior before simplifying code that affects
objectives, normalization, observations, algorithms, or learning.
