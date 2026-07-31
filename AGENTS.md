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