"""CUDA Newton gradient and candidate direction for the frozen dense model."""

from .kernel import NewtonDirectionWorkspace

__all__ = ['NewtonDirectionWorkspace']
