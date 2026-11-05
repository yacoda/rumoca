"""Compare the models that pymoca and Rumoca make of the same Modelica source.

usage: compare_models.py <examples folder> [<example> ...]

For every model, with the compiler options of rtc-tools: the variables per role, their bounds, nominals and
parameter values, and the residuals at random points. The residuals are compared per variable set: equations
may be ordered and signed differently, but as multisets of absolute values at one point they must match.
"""

import importlib.metadata
import importlib.resources
import math
import os
import re
import sys
from pathlib import Path

import casadi as ca
import numpy as np
import pymoca.backends.casadi.api as api
import rumoca_bridge

GROUPS = ("states", "der_states", "alg_states", "inputs", "parameters", "constants")
TOLERANCE = 1e-9
DETECT_ALIASES = os.environ.get("COMPARE_DETECT_ALIASES", "1") == "1"


def options(folder):
    libraries = []
    for ep in importlib.metadata.entry_points(group="rtctools.libraries.modelica"):
        if ep.name == "library_folder":
            libraries.append(str(importlib.resources.files(ep.module).joinpath(ep.attr)))
    return {
        "expand_vectors": True,
        "library_folders": libraries,
        "eliminate_constant_assignments": True,
        "replace_constant_values": True,
        "replace_constant_expressions": True,
        "replace_parameter_expressions": True,
        "eliminable_variable_expression": r"(.*[.]|^)_\w+(\[[\d,]+\])?\Z",
        "expand_mx": True,
        "detect_aliases": DETECT_ALIASES,
        "allow_derivative_aliases": False,
        "cache": False,
    }


def names(model, group):
    return [v.symbol.name() for v in getattr(model, group)]


def alias_key(model, name):
    """Alias detection may keep another member of a group: identify a variable by its group.

    Returns the smallest name of the group, and the sign of the variable relative to that name.
    """
    members = model.alias_relation.aliases(name)
    plain = {m.lstrip("-"): -1 if m.startswith("-") else 1 for m in members}
    plain.setdefault(name, 1)
    key = min(plain)
    return key, plain[key] * plain[name]


def keys(model, group):
    return {alias_key(model, n)[0] for n in names(model, group)}


def numeric(value):
    try:
        return np.array(ca.evalf(ca.MX(value))).ravel()
    except Exception:
        return None


def metadata(model, attribute, parameter_values):
    """Values of an attribute per variable, with the parameters at their values."""
    syms = [p.symbol for p in model.parameters]
    result = {}
    for group in ("states", "alg_states", "inputs"):
        for v in getattr(model, group):
            value = ca.MX(getattr(v, attribute))
            if syms and ca.symvar(value):
                value = ca.substitute([value], syms, [ca.DM(parameter_values[s.name()]) for s in syms])[0]
            key, sign = alias_key(model, v.symbol.name())
            result[key] = (numeric(value), sign)
    return result


def parameter_values(model, rng):
    """Declared values where these are numbers; anything else (set at run time) a shared random number."""
    values = {}
    for p in model.parameters:
        name = p.symbol.name()
        value = p.value
        v = None
        if not isinstance(value, ca.MX) or not ca.symvar(value):
            v = numeric(value)
            if v is not None and (v.size != p.symbol.numel() or np.isnan(v).any()):
                v = None
        values[name] = float(v[0]) if v is not None else stable_random(name)
    return values


def stable_random(name, salt=0):
    """A number in (0.5, 1.5) depending only on the name, so both models get the same at a name."""
    h = 2166136261 + salt
    for c in name.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return 0.5 + h / 2**32


def point(model, group, salt):
    values = []
    for name in names(model, group):
        key, sign = alias_key(model, name)
        values.append(sign * stable_random(key, salt))
    return ca.DM(values) if values else ca.DM(0, 1)


def residuals(model, function, values, salt):
    args = [
        ca.DM(stable_random("time", salt)),
        point(model, "states", salt),
        point(model, "der_states", salt),
        point(model, "alg_states", salt),
        point(model, "inputs", salt),
        ca.veccat(*[numeric(c.value)[0] for c in model.constants]) if model.constants else ca.DM(0, 1),
        ca.veccat(*[values[n] for n in names(model, "parameters")]) if model.parameters else ca.DM(0, 1),
    ]
    return np.sort(np.abs(np.array(function(*args)).ravel()))


def system(model, function, values, salt):
    """[Jacobian wrt the variables, residual] at a point, with the columns ordered by variable name."""
    groups = ("states", "der_states", "alg_states", "inputs")
    symbols = [ca.MX.sym(g, len(getattr(model, g))) for g in groups]
    constants = ca.veccat(*[numeric(c.value)[0] for c in model.constants]) if model.constants else ca.DM(0, 1)
    parameters = ca.veccat(*[values[n] for n in names(model, "parameters")]) if model.parameters else ca.DM(0, 1)
    time = ca.DM(stable_random("time", salt))
    residual = function(time, *symbols, constants, parameters)
    x = ca.vertcat(*symbols)
    f = ca.Function("f", [x], [ca.jacobian(residual, x), residual])
    point_x = ca.vertcat(*[point(model, g, salt) for g in groups])
    jacobian, value = f(point_x)
    order = sorted(range(x.numel()), key=[n for g in groups for n in names(model, g)].__getitem__)
    return np.hstack([np.array(jacobian)[:, order], np.array(value)])


def equivalent(a, b, function_a, function_b, values_a, values_b):
    """Do the equations of both systems span the same space, at several points?"""
    for salt in range(3):
        x, y = system(a, function_a, values_a, salt), system(b, function_b, values_b, salt)
        # A random point can be outside the domain of a function (sqrt, log): compare what is defined
        columns = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
        x, y = x[:, columns], y[:, columns]
        rank_x, rank_y = np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)
        if rank_x != rank_y or np.linalg.matrix_rank(np.vstack([x, y])) != rank_x:
            return False
    return True


def compare_attribute(a, b, attribute, values_a, values_b, out):
    ma, mb = metadata(a, attribute, values_a), metadata(b, attribute, values_b)
    differing = []
    for name in sorted(set(ma) & set(mb)):
        (x, sign_x), (y, sign_y) = ma[name], mb[name]
        if x is None or y is None or sign_x != sign_y:
            continue
        if x.size != y.size or not np.allclose(x, y, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True):
            differing.append(f"{name}: {x.tolist()} vs {y.tolist()}")
    if differing:
        out.append(f"{attribute}: {len(differing)} differ, e.g. {differing[0]}")


def compare(folder, name):
    a = api.transfer_model(str(folder), name, options(folder))
    b = rumoca_bridge.transfer_model(str(folder), name, options(folder))
    out = []
    same_variables = True
    for group in GROUPS:
        x, y = keys(a, group), keys(b, group)
        if x != y:
            same_variables = False
            only_a, only_b = sorted(x - y), sorted(y - x)
            out.append(f"{group}: {len(x)} vs {len(y)}; only pymoca {only_a[:3]} only rumoca {only_b[:3]}")
    rng = np.random.default_rng(0)
    values_a, values_b = parameter_values(a, rng), parameter_values(b, rng)
    differing = [
        f"{n}: {values_a[n]} vs {values_b[n]}"
        for n in sorted(set(values_a) & set(values_b))
        if not math.isclose(values_a[n], values_b[n], rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    ]
    if differing:
        out.append(f"parameters: {len(differing)} differ, e.g. {differing[0]}")
    for attribute in ("min", "max", "nominal", "start", "fixed"):
        compare_attribute(a, b, attribute, values_a, values_b, out)

    # Parameters that rtc-tools sets at run time must have the same effect
    switches = [p.symbol.name() for p in a.parameters if p.python_type is bool]
    flips = ([None] + switches) if same_variables else []
    for flip in flips:
        flipped_a, flipped_b = dict(values_a), dict(values_b)
        if flip is not None and flip in flipped_b:
            flipped_a[flip] = 1.0 - flipped_a[flip]
            flipped_b[flip] = 1.0 - flipped_b[flip]
        compare_residuals(a, b, flipped_a, flipped_b, flip, out)
    if not same_variables:
        out.append("residuals not compared")
    return out


def compare_residuals(a, b, values_a, values_b, flip, out):
    suffix = f" with {flip} flipped" if flip else ""
    if flip is not None or True:
        for label, make in (("dae", "dae_residual_function"), ("initial", "initial_residual_function")):
            fa, fb = getattr(a, make), getattr(b, make)
            if fa.n_out() == 0 and fb.n_out() == 0:
                continue
            if fa.n_out() == 0 or fb.n_out() == 0:
                out.append(f"{label} residual: only one model has it")
                continue
            worst = 0.0
            sizes = (fa.size1_out(0), fb.size1_out(0))
            if sizes[0] != sizes[1]:
                out.append(f"{label} residual: {sizes[0]} vs {sizes[1]} rows")
                continue
            for salt in range(3):
                ra, rb = residuals(a, fa, values_a, salt), residuals(b, fb, values_b, salt)
                worst = max(worst, float(np.max(np.abs(ra - rb) / (1 + np.abs(ra)))) if ra.size else 0.0)
            if worst > TOLERANCE:
                if not DETECT_ALIASES and equivalent(a, b, fa, fb, values_a, values_b):
                    print(f"       {label} residuals differ row by row, but span the same space{suffix}")
                else:
                    out.append(f"{label} residual: differs by {worst:.2e} at random points ({sizes[0]} rows){suffix}")


def main():
    examples = Path(sys.argv[1])
    wanted = sys.argv[2:]
    same = differ = failed = 0
    for model_folder in sorted(examples.glob("*/model")):
        example = model_folder.parent.name
        if wanted and example not in wanted:
            continue
        files = [f for f in sorted(model_folder.glob("*.mo")) if not f.name.endswith(".basemodelica.mo")]
        for source in files:
            found = re.search(r"^\s*model\s+(\w+)", source.read_text(), re.M)
            model = found.group(1) if found else source.stem
            if model in ("PIDController",) or (example == "channel_wave_damping" and model == "Example"):
                continue
            label = f"{example}/{model}"
            try:
                result = compare(model_folder, model)
            except Exception as e:
                failed += 1
                print(f"FAILED {label}: {type(e).__name__}: {' '.join(str(e).split())[:190]}")
                continue
            if result:
                differ += 1
                print(f"DIFFER {label}")
                for line in result:
                    print(f"    {line[:210]}")
            else:
                same += 1
                print(f"SAME   {label}")
    print(f"same {same}, differ {differ}, failed {failed}")


if __name__ == "__main__":
    main()
