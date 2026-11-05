# RTC-Tools bridge (experiment)

Rumoca as a frontend for [RTC-Tools](https://github.com/rtc-tools/rtc-tools), in place of pymoca's parser and
flattener, with RTC-Tools unchanged and pymoca's `simplify` still used.

- `rumoca_bridge.py`: `transfer_model` compiles with the `casadi-dae` target and fills a pymoca `Model`.
- `sitecustomize.py`: on the `PYTHONPATH`, selects the frontend per run (`RTC_FRONTEND=rumoca`).
- `compare_models.py`: compares the models of pymoca and Rumoca for every model of an examples folder.

Environment: `RUMOCA`, `RUMOCA_TEMPLATE`, and `RUMOCA_SOURCE_ROOTS` for the Modelica Standard Library.
