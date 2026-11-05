"""Experiment: Rumoca as a frontend for rtc-tools, in place of pymoca's parser and flattener.

`transfer_model` has the signature of `pymoca.backends.casadi.api.transfer_model`. It compiles the model
with Rumoca's casadi-dae target and fills a pymoca Model from that, so that rtc-tools itself and pymoca's
`simplify` (vector expansion, alias detection, elimination) are the same on both routes. Any numerical
difference is thus due to the frontend.

Environment:
    RUMOCA              the rumoca executable (default: rumoca on the PATH)
    RUMOCA_TEMPLATE     casadi_dae.py.jinja
    RUMOCA_SOURCE_ROOTS extra source roots, separated by ":", for the Modelica Standard Library
"""

import importlib.util
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import casadi as ca
from pymoca.backends.casadi.model import DelayArgument, Model, StringVariable, Variable, _DefaultValue

logger = logging.getLogger("rtctools")

PYTHON_TYPES = {"real": float, "integer": int, "boolean": bool, "enumeration": int}
UNDERDETERMINED = "ED001"


def _plain_name(name):
    """A Base Modelica name is quoted: 'a.b[1]' is the variable that a hierarchical model calls a.b[1]."""
    return name[1:-1] if name.startswith("'") and name.endswith("'") else name


def _model_file(model_folder, model_name, basemodelica):
    folder = Path(model_folder)
    if basemodelica:
        return folder / f"{model_name}.basemodelica.mo"
    candidate = folder / f"{model_name}.mo"
    if candidate.is_file():
        return candidate
    # As pymoca does, also find a model in a file of another name
    pattern = re.compile(rf"^\s*(?:partial\s+)?(?:model|class|block)\s+{re.escape(model_name)}\b", re.M)
    for path in sorted(folder.glob("*.mo")):
        if not path.name.endswith(".basemodelica.mo") and pattern.search(path.read_text()):
            return path
    raise FileNotFoundError(f"no model {model_name} in {model_folder}")


def _source_roots(compiler_options):
    roots = []
    for folder in compiler_options.get("library_folders", []):
        # rtc-tools-standard-library registers an old Modelica package; the MSL comes from RUMOCA_SOURCE_ROOTS
        if (Path(folder) / "Modelica").exists() or (Path(folder) / "package.mo").exists():
            continue
        roots.append(str(folder))
    roots += [r for r in os.environ.get("RUMOCA_SOURCE_ROOTS", "").split(":") if r]
    return roots


def _compile(path, model, roots, output):
    command = [
        os.environ.get("RUMOCA", "rumoca"),
        "compile",
        str(path),
        "--model",
        model,
        "--target",
        os.environ["RUMOCA_TEMPLATE"],
        "--phase",
        "dae",
        "-o",
        str(output),
    ]
    for root in roots:
        command += ["--source-root", root]
    # rtc-tools sets parameters at run time, so a parameter guard must stay a run-time branch
    env = dict(os.environ, RUMOCA_RETAIN_PARAMETER_BRANCHES="1")
    return subprocess.run(command, capture_output=True, text=True, env=env)


def _as_partial(path, model_name, folder):
    """Copy a model folder with the model declared partial.

    rtc-tools models may be under-determined by design, with the optimizer or Python code providing what
    is missing. Rumoca only allows that for a partial model (MLS 4.7), and pymoca does not check it.
    """
    copy = Path(folder) / "partial"
    shutil.copytree(path.parent, copy)
    target = copy / path.name
    source = target.read_text()
    quoted = re.escape(model_name)
    source, count = re.subn(rf"^(\s*)((?:model|class|block)\s+'?{quoted}'?(?:\s|$))", r"\1partial \2", source, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError(f"could not find the declaration of {model_name} in {path}")
    target.write_text(source)
    return target


def _load_module(path):
    spec = importlib.util.spec_from_file_location("rumoca_casadi_dae", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modelica_shape(name, symbol, dimensions):
    """The shape format of pymoca: one entry per dotted part, which for Rumoca is the last that is an array."""
    parts = _split_name(name)
    shape = [(None,)] * len(parts)
    if dimensions:
        shape[-1] = tuple(dimensions)
    return tuple(shape)


def _split_name(name):
    return re.match(r"(?:der\()*(.*?)\)*$", name).group(1).split(".")


def _value(x, python_type=float):
    """Literal attribute values as plain numbers, like pymoca."""
    if isinstance(x, ca.MX):
        return x
    if isinstance(x, ca.DM):
        return python_type(x) if x.is_scalar() else x
    return python_type(x) if not isinstance(x, bool) else x


def _fill(data):
    model = Model()
    model.time = data["time"]

    def variable(entry, symbol, attributes=True):
        python_type = PYTHON_TYPES[entry["scalar"]]
        shape = symbol.shape
        dimensions = () if symbol.is_scalar() else ((shape[0],) if shape[1] == 1 else shape)
        name = _plain_name(symbol.name())
        renamed = ca.MX.sym(name, *shape) if name != symbol.name() else symbol
        renamed._modelica_shape = _modelica_shape(name, symbol, dimensions)
        v = Variable(renamed, python_type)
        if attributes:
            for key, attr in (("min", "min"), ("max", "max"), ("nominal", "nominal"), ("binding", "value")):
                if key in entry:
                    setattr(v, attr, _value(entry[key], python_type if key != "nominal" else float))
            start = entry.get("start")
            # Rumoca makes the default start of 0 explicit, which pymoca's alias handling tells apart
            if start is not None and not (not isinstance(start, ca.MX) and float(ca.DM(start)[0]) == 0.0 and ca.DM(start).is_scalar()):
                v.start = _value(start, python_type)
            elif start is not None:
                v.start = _DefaultValue(0)
            if entry["fixed"] is not None:
                v.fixed = entry["fixed"]
        return v, renamed

    symbols_old, symbols_new = [], []
    for entry in data["variables"]:
        role = entry["role"]
        if entry["scalar"] == "string":
            s = StringVariable(_plain_name(entry["name"]))
            (model.string_constants if role == "constant" else model.string_parameters).append(s)
            continue
        symbol = entry["symbol"]
        if symbol.is_empty():
            continue
        v, new = variable(entry, symbol)
        symbols_old.append(symbol)
        symbols_new.append(new)
        if role == "state":
            model.states.append(v)
            der, new_der = variable(entry, entry["derivative"], attributes=False)
            model.der_states.append(der)
            symbols_old.append(entry["derivative"])
            symbols_new.append(new_der)
        elif role in ("algebraic", "output", "discrete_real", "discrete_value"):
            model.alg_states.append(v)
        elif role == "input":
            model.inputs.append(v)
        elif role == "parameter":
            model.parameters.append(v)
        elif role == "constant":
            model.constants.append(v)
        else:
            raise NotImplementedError(f"variable role {role} of {entry['name']}")
        if entry["causality"] == "output":
            model.outputs.append(_plain_name(entry["name"]))

    def rename(expressions):
        expressions = [ca.vec(ca.MX(e)) for e in expressions]
        expressions = [e for e in expressions if not e.is_empty()]
        return ca.substitute(expressions, symbols_old, symbols_new) if expressions else []

    for i, delay in enumerate(data["delays"]):
        symbol = delay["symbol"]
        name = f"_pymoca_delay_{i}"
        new = ca.MX.sym(name, *symbol.shape)
        new._modelica_shape = symbol.shape
        symbols_old.append(symbol)
        symbols_new.append(new)
        model.delay_states.append(name)
        model.inputs.append(Variable(new))

    for delay in data["delays"]:
        [expression, duration] = ca.substitute(
            [ca.MX(delay["expression"]), ca.MX(delay["duration"])], symbols_old, symbols_new
        )
        model.delay_arguments.append(DelayArgument(expression, duration))

    model.equations = rename(data["equations"])
    model.initial_equations = rename(data["initial_equations"])

    # Attributes may refer to other variables
    def rename_attributes(v):
        for attr in ("value", "start", "min", "max", "nominal"):
            value = getattr(v, attr)
            if isinstance(value, ca.MX):
                setattr(v, attr, ca.substitute([value], symbols_old, symbols_new)[0])

    for group in (model.states, model.alg_states, model.inputs, model.parameters, model.constants):
        for v in group:
            rename_attributes(v)
    return model


def transfer_model(model_folder, model_name, compiler_options=None):
    compiler_options = compiler_options or {}
    basemodelica = compiler_options.get("basemodelica", False)
    path = _model_file(model_folder, model_name, basemodelica)
    roots = [] if basemodelica else _source_roots(compiler_options)
    qualified = f"'{model_name}'.'{model_name}'" if basemodelica else model_name

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "model_casadi_dae.py"
        result = _compile(path, qualified, roots, output)
        if result.returncode != 0 and UNDERDETERMINED in result.stderr + result.stdout:
            logger.info(f"{model_name} is under-determined: compiling it as a partial model.")
            result = _compile(_as_partial(path, model_name, folder), qualified, roots, output)
        if result.returncode != 0:
            raise RuntimeError(f"rumoca failed on {model_name}:\n{result.stderr}{result.stdout}")
        # A parameter fixed at translation ignores a value set at run time. Array sizes are expected to be.
        for line in (result.stderr + result.stdout).splitlines():
            if "WD001" in line and "array dimension" not in line and "for-equation range" not in line:
                logger.warning(f"{model_name}: {line}")
        module = _load_module(output)
    model = _fill(module.build())
    model.simplify(compiler_options)
    return model
