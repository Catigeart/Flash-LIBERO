"""LIBERO package, preserving the upstream ``libero.libero`` public API.

The development tree keeps GPU sources under ``libero/libero/gpu``. A normal
package search path makes them importable as ``libero.gpu`` without aliases or
an import finder. Wheels physically install that same source as ``libero/gpu``.
"""
from pathlib import Path as _Path

_gpu_source_parent = _Path(__file__).resolve().parent / "libero"
if (_gpu_source_parent / "gpu" / "__init__.py").is_file():
    __path__.append(str(_gpu_source_parent))
del _gpu_source_parent, _Path
