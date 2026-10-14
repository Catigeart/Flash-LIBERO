"""GPU center-of-mass velocity recurrence."""

from .kernel import ComVelModel, launch

__all__ = ["ComVelModel", "launch"]
