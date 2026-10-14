"""One-time static preparation for the fixed 43-DOF inertia-box fluid model.

The box expression is MuJoCo 2.3.7 engine_passive.c:189-197.  The fixed
topology hashes come from the archived task_model.mjb (SHA256
3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42).
Mass and inertia may differ for the documented local single-body probes, so
their exact bytes are bound to each prepared instance instead of pinned here.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np


PREPARE_VERSION = "n08-static-box-mujoco-2.3.7-v1"
_MINVAL = 1.0e-15
_TOPOLOGY_SHA256 = {
    "body_rootid": "df271de92a2e6e9782eaed57ef00820880168b5761ab153bfa2b28cf9af7e0dd",
    "body_parentid": "8d18b41c7e02b2def07e3c35852487ddd6e10d4f7e4d6ecfcbedd7c1103a4690",
    "body_dofnum": "71beb793d979c86c1de61266643a4a4df400230df830cf95aed21129d4328e76",
    "body_dofadr": "0c4901bfb9cac58bf2e64a19b80dd06c0ec5da06d565a9a7f771197d48fcbd33",
    "dof_parentid": "a57334b5f8e08a96647a0de0fc27eaac9e1fbfa1c76f007f5308360f2f4fd76a",
}
_MASS_INERTIA_SHA256 = {
    # Original task_model.mjb fluid-relevant static values.
    ("56a9bb3f3b1fb7573633c784f38cf883b32db2ad04029f72dbed3c429a6bf32f",
     "eb9b8a8bb9d5f194d540f552eb9708698c6dc4a72d48c23c55bcdff8a745f09d"),
    # Named local diagnostic domain: only moving body 7 retains mass/inertia.
    ("e7912aae067600095f2d7ff32311e24267d3e516e952cfcab287e1bd590f8ef8",
     "01a0cc8be82caf1d3e08282c688b48843f9cbe38eeba539d08b00031f5eb4acb"),
}


@dataclass(frozen=True)
class PreparedFluidModel:
    rootid: np.ndarray
    mass: np.ndarray
    box: np.ndarray
    ancestor: np.ndarray
    wind: np.ndarray
    opt: np.ndarray
    passive_disabled: np.int32
    model_identity_sha256: str
    prepare_version: str = PREPARE_VERSION


def _array(model, name, shape, dtype):
    if not hasattr(model, name):
        raise ValueError(f"missing static model field: {name}")
    arr = np.asarray(getattr(model, name))
    if arr.shape != shape or arr.dtype != dtype:
        raise TypeError(f"{name} requires {dtype} {shape}; got {arr.dtype} {arr.shape}")
    if dtype == np.float64 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains nonfinite values")
    return np.array(arr, dtype=dtype, copy=True, order="C")


def _integer(model, name, expected=None):
    raw = np.asarray(getattr(model, name))
    if raw.shape != () or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError(f"{name} requires an integer scalar")
    value = int(raw)
    if value < -(1 << 31) or value >= (1 << 31):
        raise ValueError(f"{name} outside int32 range")
    if expected is not None and value != expected:
        raise ValueError(f"unsupported {name}: {value}")
    return value


def _digest_fields(fields):
    digest = hashlib.sha256()
    for name, arr in fields.items():
        arr = np.ascontiguousarray(arr)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(arr.dtype).encode("ascii") + b"\0")
        digest.update(repr(arr.shape).encode("ascii") + b"\0")
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _derive_box(mass, inertia):
    """Evaluate the source's separate binary64 operations and sqrt order."""
    box = np.zeros((38, 3), np.float64)
    for body in range(1, 38):
        m = float(mass[body])
        if m < _MINVAL:
            continue
        i0, i1, i2 = map(float, inertia[body])
        terms = (i1 + i2 - i0, i0 + i2 - i1, i0 + i1 - i2)
        for axis, term in enumerate(terms):
            if not math.isfinite(term):
                raise ValueError("inertia-box intermediate is nonfinite")
            radicand = max(_MINVAL, term) / m * 6.0
            if not math.isfinite(radicand):
                raise ValueError("inertia-box radicand is nonfinite")
            box[body, axis] = math.sqrt(radicand)
    return box


def prepare_model(model):
    """Validate and bind all static fluid inputs before any device upload.

    A supplied ``box`` must match the value re-derived from the supplied
    mass/inertia byte for byte.  If omitted, the derived value is used.
    """
    for name, expected in (("nbody", 38), ("nv", 43), ("nq", 48),
                           ("ntendon", 0), ("nplugin", 0)):
        _integer(model, name, expected)
    topology = {}
    for name, shape in (("body_rootid", (38,)), ("body_parentid", (38,)),
                        ("body_dofnum", (38,)), ("body_dofadr", (38,)),
                        ("dof_parentid", (43,))):
        arr = _array(model, name, shape, np.int32)
        if hashlib.sha256(arr.tobytes()).hexdigest() != _TOPOLOGY_SHA256[name]:
            raise ValueError(f"{name} differs from fixed 43-DOF model")
        topology[name] = arr

    mass = _array(model, "body_mass", (38,), np.float64)
    inertia = _array(model, "body_inertia", (38, 3), np.float64)
    if np.any(mass < 0.0) or np.any(inertia < 0.0):
        raise ValueError("negative mass or inertia")
    pair = (hashlib.sha256(mass.tobytes()).hexdigest(),
            hashlib.sha256(inertia.tobytes()).hexdigest())
    if pair not in _MASS_INERTIA_SHA256:
        raise ValueError("mass/inertia differ from fixed MJB and named single-body probe")
    gravcomp = _array(model, "body_gravcomp", (38,), np.float64)
    if np.any(gravcomp != 0.0):
        raise ValueError("unsupported body_gravcomp")
    geom_fluid = np.asarray(getattr(model, "geom_fluid"))
    if geom_fluid.dtype != np.float64 or geom_fluid.shape != (270, 12):
        raise TypeError("geom_fluid requires float64 (270, 12)")
    if not np.all(np.isfinite(geom_fluid)):
        raise ValueError("geom_fluid contains nonfinite values")
    if np.any(geom_fluid != 0.0):
        raise ValueError("ellipsoid fluid branch unsupported")

    expected_box = _derive_box(mass, inertia)
    if hasattr(model, "box"):
        supplied_box = _array(model, "box", (38, 3), np.float64)
        if not np.array_equal(supplied_box.view(np.uint64), expected_box.view(np.uint64)):
            raise ValueError("box bytes differ from MuJoCo mass/inertia derivation")
    wind = _array(model, "opt_wind", (3,), np.float64)
    viscosity = _array(model, "opt_viscosity", (), np.float64)
    density = _array(model, "opt_density", (), np.float64)
    flags = _integer(model, "opt_disableflags")
    opt = np.array([viscosity, density], np.float64)

    mask = np.zeros((38, 43), np.int32)
    for original_body in range(1, 38):
        body = original_body
        while body and topology["body_dofnum"][body] == 0:
            body = int(topology["body_parentid"][body])
        if body:
            j = int(topology["body_dofadr"][body] + topology["body_dofnum"][body] - 1)
            while j >= 0:
                if mask[original_body, j]:
                    raise ValueError("cyclic DOF ancestry")
                mask[original_body, j] = 1
                j = int(topology["dof_parentid"][j])

    identity = _digest_fields({**topology, "body_mass": mass,
                               "body_inertia": inertia, "box": expected_box,
                               "body_gravcomp": gravcomp,
                               "geom_fluid": geom_fluid,
                               "opt_wind": wind, "opt": opt,
                               "opt_disableflags": np.int32(flags)})
    return PreparedFluidModel(topology["body_rootid"], mass, expected_box,
                              mask, wind, opt, np.int32(bool(flags & (1 << 5))),
                              identity)
