"""Explicit diagnostic raw model cold C/N copy entry; no hot replacement."""
import os


def create(built):
    if os.environ.get("P06_STEP_COLD_COPY") != "cn":
        raise ValueError("raw copy factory requires explicit pinned cn cold choice")
    from libero.gpu.host.adapters.environment.raw_factory import create as original_create
    return original_create(built)
