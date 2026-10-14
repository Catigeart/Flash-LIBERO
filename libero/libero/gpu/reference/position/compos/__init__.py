"""MuJoCo 2.3.7 center-of-mass position dynamics on CUDA."""

from .kernel import ComposModel, launch

__all__ = ["ComposModel", "launch"]
