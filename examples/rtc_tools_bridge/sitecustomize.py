"""Experiment: choose the rtc-tools frontend per run, without changing rtc-tools.

RTC_FRONTEND=rumoca   use rumoca_bridge.transfer_model instead of pymoca's
RTC_BASEMODELICA=1    compile the *.basemodelica.mo file of the model folder
RTC_SOLVER_LOG=<file> append the solver status of every solve to a file
"""

import os

if os.environ.get("RTC_FRONTEND") == "rumoca" or os.environ.get("RTC_BASEMODELICA") == "1" or os.environ.get("RTC_SOLVER_LOG"):
    import rtctools.optimization.modelica_mixin as _mm
    import rtctools.simulation.simulation_problem as _sp

    if os.environ.get("RTC_BASEMODELICA") == "1":
        for _cls in (_mm.ModelicaMixin, _sp.SimulationProblem):

            def _patched(self, _orig=_cls.compiler_options):
                options = _orig(self)
                options["basemodelica"] = True
                return options

            _cls.compiler_options = _patched

    if os.environ.get("RTC_FRONTEND") == "rumoca":
        import pymoca.backends.casadi.api as _api
        import rumoca_bridge

        _api.transfer_model = rumoca_bridge.transfer_model

    if os.environ.get("RTC_SOLVER_LOG"):
        import rtctools.optimization.optimization_problem as _op

        _orig_optimize = _op.OptimizationProblem.optimize

        def _logged(self, *args, **kwargs):
            ok = _orig_optimize(self, *args, **kwargs)
            try:
                status = self.solver_stats.get("return_status")
            except Exception:
                status = "?"
            with open(os.environ["RTC_SOLVER_LOG"], "a") as f:
                f.write(f"{os.getcwd().split('/examples/')[-1]} {type(self).__name__} {status}\n")
            return ok

        _op.OptimizationProblem.optimize = _logged
