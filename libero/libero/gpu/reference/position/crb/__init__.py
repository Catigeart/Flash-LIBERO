"""GPU composite rigid body inertia and sparse mass matrix."""

from .kernel import CRBModel, launch

__all__ = ["CRBModel", "launch"]
