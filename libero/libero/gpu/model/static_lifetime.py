"""Exact static source observations across compiles, without fresh MJB identity.

This module never constructs MjData or saves a model. It is deliberately not a
ModelSpec producer: current_full_mjb_sha256 is always None. Every observation
reads all actual public native semantic fields again; an old blob SHA is only
baseline provenance. Dynamic state restoration is the caller's separate proof.
"""
import copy
import hashlib
import json
from pathlib import Path
import uuid
import mujoco
import numpy as np
from .factory import _array_digest


class SemanticObservationRejected(ValueError):
    """Caller must retain rejection and fall back to full identity evidence."""


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _typed(value):
    arr = np.asarray(value)
    if arr.dtype.hasobject or arr.dtype.kind not in "biufcSU":
        raise SemanticObservationRejected("unsupported typed semantic value")
    arr = np.ascontiguousarray(arr)
    result = {"dtype": np.asarray(value).dtype.str, "shape": list(np.asarray(value).shape),
              "raw_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    # Arrays retain exact source bytes through digests; scalar JSON values are
    # descriptive only (dtype is Python binding value dtype, not guessed C ABI).
    if np.asarray(value).ndim == 0:
        item = np.asarray(value).item()
        if isinstance(item, bytes): item = {"bytes_hex": item.hex()}
        if isinstance(item, complex): item = {"real": item.real, "imag": item.imag}
        result["value"] = item
        result["dtype_scope"] = "python-binding-value"
    return result


def _text(value):
    if isinstance(value, bytes): return {"bytes_hex": value.hex()}
    if isinstance(value, str): return {"utf8_hex": value.encode("utf-8").hex()}
    if isinstance(value, tuple): return {"tuple": [_text(v) for v in value]}
    raise SemanticObservationRejected("unsupported text semantic value")


def _group(obj, *, nested=False):
    out = {}
    visual_groups = {"global_": "Global", "headlight": "Headlight", "map": "Map",
                     "quality": "Quality", "rgba": "Rgba", "scale": "Scale"}
    for name in dir(obj):
        if name.startswith("_"): continue
        value = getattr(obj, name)
        if callable(value): continue
        if isinstance(value, (np.ndarray, int, float, complex, np.generic)):
            out[name] = {"kind": "typed", **_typed(value)}
        elif isinstance(value, (bytes, str, tuple)):
            out[name] = {"kind": "text", "content": _text(value)}
        elif nested and visual_groups.get(name) == type(value).__name__:
            out[name] = {"kind": "group", "fields": _group(value)}
        else:
            raise SemanticObservationRejected("unknown public field: " + name + ":" + type(value).__name__)
    return out


def capture_semantic_observation(model):
    """Read actual MuJoCo 2.3.7 fields, never serialize a fresh MJB."""
    if not isinstance(model, mujoco.MjModel) or mujoco.__version__ != "2.3.7":
        raise SemanticObservationRejected("unqualified model binding/version")
    arrays, scalar_text = {}, {}
    for name in dir(model):
        if name.startswith("_"): continue
        value = getattr(model, name)
        if callable(value): continue
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise SemanticObservationRejected("object array has unstable identity: " + name)
            arrays[name] = value
        elif name in ("opt", "stat", "vis"):
            expected = {"opt": "MjOption", "stat": "MjStatistic", "vis": "MjVisual"}[name]
            if type(value).__name__ != expected:
                raise SemanticObservationRejected("unknown model group: " + name)
        elif isinstance(value, (int, float, complex, np.generic)):
            scalar_text[name] = {"kind": "typed", **_typed(value)}
        elif isinstance(value, (str, bytes, tuple)):
            scalar_text[name] = {"kind": "text", "content": _text(value)}
        else:
            raise SemanticObservationRejected("unknown model field: " + name + ":" + type(value).__name__)
    inventory = {n: {"shape": list(a.shape), "dtype": a.dtype.str} for n, a in sorted(arrays.items())}
    observation = {"schema": "native-static-semantic-observation-no-mjb-v1",
        "identity_kind": "source-qualified-semantic-parameter-observation",
        "current_full_mjb_sha256": None, "model_sha256": None,
        "model_blob_identity_computed": False, "native_version": mujoco.mj_versionString(),
        "binding_version": mujoco.__version__, "life_id": str(uuid.uuid4()),
        "native_owner_python_id": id(model), "static_array_inventory": inventory,
        "array_sha256": {n: _array_digest({n: a}) for n, a in sorted(arrays.items())},
        "static_arrays_sha256": _array_digest(arrays), "scalar_text_fields": scalar_text,
        "options": _group(model.opt), "stat": _group(model.stat), "visual": _group(model.vis, nested=True),
        "scope": "static source observation only; no MJB/ModelSpec/GPU or dynamic state identity"}
    observation["semantic_sha256"] = _json_hash({k:v for k,v in observation.items()
        if k not in ("life_id", "native_owner_python_id", "scope")})
    return observation


def alpha_assignment_orbit(baseline_scalar, *, max_states=16):
    """Exact reachable stored-dtype states of the original ``1 - alpha`` writer.

    Python / NumPy arithmetic promotion and assignment rounding are reproduced,
    without an epsilon or a guessed numeric interval. Nonclosing or nonfinite
    orbits fail closed so the caller can collect a full-model fallback instead.
    """
    scalar = np.asarray(baseline_scalar)
    if scalar.ndim != 0 or scalar.dtype.kind != "f" or not np.isfinite(scalar):
        raise SemanticObservationRejected("unsupported alpha orbit scalar")
    dtype = scalar.dtype
    current = scalar[()]
    seen, states = {}, []
    for transition in range(max_states + 1):
        key = np.asarray(current).tobytes().hex()
        if key in seen:
            return {"schema": "stored-dtype-one-minus-alpha-orbit-v1",
                    "dtype": dtype.str, "numpy_version": np.__version__,
                    "writer": "1 - stored_numpy_scalar; assign to original dtype",
                    "states": states, "cycle_start": seen[key],
                    "cycle_length": len(states) - seen[key],
                    "closure_raw_hex": key, "max_states": max_states}
        if transition == max_states:
            raise SemanticObservationRejected("alpha orbit did not close; full identity fallback required")
        if not np.isfinite(current):
            raise SemanticObservationRejected("nonfinite alpha orbit")
        seen[key] = len(states)
        states.append({"raw_hex": key, "value": current.item()})
        assigned = np.empty((), dtype=dtype)
        assigned[()] = 1 - current
        current = assigned[()]
    raise AssertionError("unreachable orbit state")


class StaticSemanticBaseline:
    @classmethod
    def capture(cls, native_model, *, baseline_model_metadata, baseline_mjb_path,
                fixture_body_ids, visual_site_ids, source_policy):
        obj = cls()
        obj.model = native_model  # Retain owner; Python id alone is never a seal.
        obj.policy = copy.deepcopy(source_policy)
        if not obj.policy.get("task_identity") or not obj.policy.get("source_pins"):
            raise SemanticObservationRejected("exact task and source pins required")
        for pin in obj.policy["source_pins"]:
            if _hash_file(pin["path"]) != pin["sha256"]:
                raise SemanticObservationRejected("source policy pin changed")
        obj.full_metadata = copy.deepcopy(baseline_model_metadata)
        obj.mjb_path = str(baseline_mjb_path)
        if _hash_file(obj.mjb_path) != obj.full_metadata["model_sha256"]:
            raise SemanticObservationRejected("baseline actual MJB SHA differs")
        obj.observation = capture_semantic_observation(native_model)
        blob_model = mujoco.MjModel.from_binary_path(obj.mjb_path)
        blob_observation = capture_semantic_observation(blob_model)
        if blob_observation["semantic_sha256"] != obj.observation["semantic_sha256"]:
            raise SemanticObservationRejected("baseline actual native owner differs from saved blob")
        del blob_model
        actual_options = {}
        for name in dir(native_model.opt):
            if name.startswith("_"): continue
            value = getattr(native_model.opt, name)
            if isinstance(value, np.ndarray): actual_options[name] = value.tolist()
            elif isinstance(value, (int, float, np.integer, np.floating)):
                actual_options[name] = value.item() if isinstance(value, np.generic) else value
        if actual_options != obj.full_metadata["options"]:
            raise SemanticObservationRejected("baseline native options differ from metadata")
        for field in ("static_array_inventory", "array_sha256", "static_arrays_sha256"):
            if obj.observation[field] != obj.full_metadata[field]:
                raise SemanticObservationRejected("baseline actual native fields differ: " + field)
        obj.fixtures = frozenset(map(int, fixture_body_ids))
        obj.sites = frozenset(map(int, visual_site_ids))
        if any(i < 0 or i >= native_model.nbody for i in obj.fixtures):
            raise SemanticObservationRejected("fixture ID out of range")
        if any(i < 0 or i >= native_model.nsite for i in obj.sites):
            raise SemanticObservationRejected("visual ID out of range")
        obj.leaves = {n: np.array(getattr(native_model,n),copy=True) for n in ("body_pos", "body_quat", "site_rgba")}
        obj.alpha_orbits = {}
        for site, mode in obj.policy.get("visual_alpha_rules", {}).items():
            try: index = int(site)
            except (TypeError, ValueError):
                raise SemanticObservationRejected("visual rule ID invalid")
            if str(index) != site or index not in obj.sites:
                raise SemanticObservationRejected("visual rule outside authorized sites")
            if mode == "task_one_minus_alpha_orbit":
                obj.alpha_orbits[index] = alpha_assignment_orbit(obj.leaves["site_rgba"][index,3])
            elif mode not in ("hide_positive_alpha", "binary_task_visualization"):
                raise SemanticObservationRejected("unknown visual alpha rule")
        obj.record = {"baseline_mjb_path":obj.mjb_path,
            "baseline_mjb_sha256":obj.full_metadata["model_sha256"],
            "baseline_observation":copy.deepcopy(obj.observation),
            "fixture_body_ids":sorted(obj.fixtures), "visual_site_ids":sorted(obj.sites),
            "visual_alpha_orbits":copy.deepcopy(obj.alpha_orbits),
            "source_policy":copy.deepcopy(obj.policy), "scope":obj.observation["scope"]}
        return obj

    def observe(self, native_model, *, task_identity, official_state_sha256):
        if task_identity != self.policy["task_identity"]:
            raise SemanticObservationRejected("official task differs from baseline")
        if not isinstance(official_state_sha256,str) or len(official_state_sha256)!=64 or any(c not in "0123456789abcdef" for c in official_state_sha256):
            raise SemanticObservationRejected("official state source pin required")
        current = capture_semantic_observation(native_model)
        for field in ("static_array_inventory", "scalar_text_fields", "options", "stat", "visual", "native_version", "binding_version"):
            if current[field] != self.observation[field]:
                raise SemanticObservationRejected("immutable semantic field changed: " + field)
        leaves = {"body_pos", "body_quat", "site_rgba"}
        if set(current["array_sha256"]) != set(self.observation["array_sha256"]):
            raise SemanticObservationRejected("native field schema changed")
        changes = []
        for name, digest in current["array_sha256"].items():
            if digest == self.observation["array_sha256"][name]: continue
            if name not in leaves:
                raise SemanticObservationRejected("immutable array changed: " + name)
            old = self.leaves[name]; new = np.asarray(getattr(native_model,name))
            if not np.isfinite(new).all():
                raise SemanticObservationRejected("nonfinite mutable leaf")
            # Byte equality includes signed zero: no unlisted row may change.
            equal = old.view(np.uint8).reshape(old.shape[0],-1) == new.view(np.uint8).reshape(new.shape[0],-1)
            rows = np.flatnonzero(~np.all(equal,axis=1)).tolist()
            if name in ("body_pos", "body_quat"):
                if set(rows)-self.fixtures:
                    raise SemanticObservationRejected("nonfixture body leaf changed")
                if name == "body_quat" and not np.allclose(np.sum(new[rows]*new[rows],axis=1),1.,rtol=0,atol=1e-12):
                    raise SemanticObservationRejected("fixture quaternion not normalized")
            else:
                if set(rows)-self.sites or not np.array_equal(old[:,:3].view(np.uint8),new[:,:3].view(np.uint8)):
                    raise SemanticObservationRejected("unauthorized visual row or RGB changed")
                modes = self.policy.get("visual_alpha_rules", {})
                for i in rows:
                    mode = modes.get(str(i), "hide_positive_alpha")
                    x,y=float(old[i,3]),float(new[i,3])
                    if mode == "hide_positive_alpha": ok = x > 0. and y == -x
                    elif mode == "binary_task_visualization": ok = x in (0.,1.) and y in (0.,1.)
                    elif mode == "task_one_minus_alpha_orbit":
                        key = np.asarray(new[i,3]).tobytes().hex()
                        ok = key in {state["raw_hex"] for state in self.alpha_orbits[i]["states"]}
                    else: raise SemanticObservationRejected("unknown visual alpha rule")
                    if not ok: raise SemanticObservationRejected("visual alpha source rule violated")
            changes.append({"field":name,"changed_rows":rows,"baseline_field_sha256":self.observation["array_sha256"][name],
                "current_field_sha256":digest,"baseline_rows":old[rows].tolist(),"current_rows":new[rows].tolist()})
        return {"schema":"source-qualified-semantic-parameter-comparison-v1",
            "current_full_mjb_sha256":None,"baseline_mjb_sha256":self.full_metadata["model_sha256"],
            "current_observation":current,"baseline_life_id":self.observation["life_id"],
            "same_native_owner":native_model is self.model,"task_identity":task_identity,
            "official_state_sha256":official_state_sha256,"official_state_pin_scope":"caller-provided source vector pin; dynamic restoration not proved here",
            "immutable_fields_equal":True,"changed_leaves":changes,"source_policy":copy.deepcopy(self.policy),
            "visual_alpha_orbits":copy.deepcopy(self.alpha_orbits),
            "source_semantic_comparison_passed":True,"kernel_license_granted":False,"scope":current["scope"]}