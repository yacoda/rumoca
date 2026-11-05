# `casadi-dae`

## Use case

Use this target to bring a Rumoca DAE into CasADi in residual form, for hosts
that discretize and optimize it themselves, such as collocation in RTC-Tools.
Unlike `casadi-ode`, it needs no explicit form: the equations stay implicit, and
there is no structural analysis or index reduction.

## Contract

- Readiness 1: exercised by `examples/rtc_tools_bridge`, not yet in CI.
- Input: checked DAE IR.
- Output: a Python module with a `build()` returning a dictionary of `time`,
  `variables` (name, role, scalar type, causality, MX symbol, derivative symbol
  for a state, and binding, start, min, max, nominal and fixed as present),
  `equations` and `initial_equations` (residuals as MX), and `delays`.
- Conditionals with symbolic guards become `if_else`, also in functions.
- A subscript or array size that reads a parameter is evaluated from the
  parameter bindings, so it is fixed at translation.

## Unsupported

Runtime events, clocks, records, and `for` statements in functions invoke a
span-bearing `unsupported-feature:casadi-dae-*` failure. Arrays of more than two
dimensions raise `NotImplementedError` in the generated module.

## Run-time parameters

By default, Rumoca fixes an ordinary parameter at translation when it guards
branches that differ in their unknowns (SPEC_0040 DAE-C22), and reports it as
`WD001`. A host that sets parameters at run time must then not rely on them.
Setting `RUMOCA_RETAIN_PARAMETER_BRANCHES` keeps such a guard as a run-time
branch. This is experimental, and is only meant for hosts of the residuals: the
branches may then have different structure.

## Example

```python
data = model_casadi_dae.build()
residual = ca.Function(
    "residual",
    [data["time"], *[v["symbol"] for v in data["variables"]]],
    [ca.vertcat(*data["equations"])],
)
```

## Verification

`examples/rtc_tools_bridge/compare_models.py` compares the resulting models with those
of pymoca, on variables, bounds, nominals, starts, parameters, and residuals at
random points, with every Boolean parameter flipped.
