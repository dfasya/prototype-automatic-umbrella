"""Self-contained umbrella runtime: sensor data loading, real-time inference,
and a simulated Shelly 2PM Gen4 (fake_shelly.py) for testing without hardware.

Run from the ``deployment/`` directory, e.g. ``python -m modules.inference``.
Needs numpy and pandas; pythermalcomfort only when UTCI is not given.
"""
