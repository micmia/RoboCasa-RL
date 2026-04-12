"""
Compatibility shim for older scripts.

Historically, training scripts imported a top-level `env` module that did not
exist in this repo. We now keep `env/` as a thin wrapper while the real,
maintained implementation lives under `robocasa_rl/`.
"""

from env.custom_pnp_counter_to_cab import MyPnPCounterToCab

__all__ = ["MyPnPCounterToCab"]

