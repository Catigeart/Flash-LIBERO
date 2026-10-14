"""GPU arithmetic policy: permit FMA while preserving IEEE error handling."""
if __name__ != "libero.gpu":
    raise ImportError("GPU modules use the canonical libero.gpu namespace")

import warp as _wp

_wp.set_module_options({"fuse_fp": True, "fast_math": False})
