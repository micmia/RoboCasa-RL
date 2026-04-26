"""
Compatibility shim for older scripts.

Historically, training scripts imported a top-level `env` module that did not
exist in this repo. We now keep `env/` as a thin wrapper while the real,
maintained implementation lives under `robocasa_rl/`.
"""

from env.custom_pnp_counter_to_cab import MyPnPCounterToCab
from env.custom_pnp_apple_to_bowl import MyPnPAppleToBowl

__all__ = ["MyPnPCounterToCab", "MyPnPAppleToBowl"]

