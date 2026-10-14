"""Pure typed numerical comparator; no CUDA or native oracle import.

Raw evidence hashes remain bit exact. Floating numerical equality includes
signed zero. A nonzero difference passes only inside a supplied, pre-output
field-specific source interval. NaN payloads are permitted only on held words.
"""
from __future__ import annotations
import hashlib
import pathlib
import numpy as np


def hash_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        if len(z.files) != len(set(z.files)):
            raise ValueError("duplicate typed NPZ member")
        return {k: np.array(z[k], copy=True) for k in z.files}


def compare_arrays(expected, actual, *, bounds=None, write_masks=None):
    """Check every represented word without narrowing, clipping or normalization.

    bounds maps a field to its frozen (lower,upper) arrays, not a universal
    epsilon. write_masks maps floating fields to frozen uint8 0/1 masks. A
    zero mask denotes a held value: numerical zero equivalence is lawful, but
    a held NaN/inf must preserve the original payload. Actual write events are
    a separate integer assertion, never inferred from equal output values.
    """
    failures = []
    bounds = {} if bounds is None else bounds
    write_masks = {} if write_masks is None else write_masks
    if set(bounds) - set(expected) or set(write_masks) - set(expected):
        raise ValueError("numeric policy references an absent field")
    if set(expected) != set(actual):
        failures.append({"reason": "field_set",
                         "missing": sorted(set(expected)-set(actual)),
                         "extra": sorted(set(actual)-set(expected))})
    for name, e in sorted(expected.items()):
        if name not in actual:
            continue
        g = actual[name]
        if not isinstance(e, np.ndarray) or not isinstance(g, np.ndarray):
            raise TypeError("typed arrays required")
        if e.shape != g.shape or e.dtype.str != g.dtype.str:
            failures.append({"field": name, "reason": "schema",
                             "expected_dtype": e.dtype.str, "actual_dtype": g.dtype.str,
                             "expected_shape": list(e.shape), "actual_shape": list(g.shape)})
            continue
        if e.dtype.kind == "f":
            mask = write_masks.get(name)
            if mask is None:
                mask = np.full(e.shape, 0 if name.startswith("held_") else 1, np.uint8)
            if mask.dtype != np.uint8 or mask.shape != e.shape or np.any(mask > 1):
                raise ValueError("frozen write-mask schema: "+name)
            written = mask.astype(bool)
            finite_e, finite_g = np.isfinite(e), np.isfinite(g)
            ok = finite_e & finite_g & (e == g)  # -0/+0: exact absolute error zero.
            if name in bounds:
                lo, hi = bounds[name]
                if (lo.shape != e.shape or hi.shape != e.shape or
                    lo.dtype.str != e.dtype.str or hi.dtype.str != e.dtype.str or
                    not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi)):
                    raise ValueError("source interval schema: "+name)
                if np.any(written & (~finite_e | (e < lo) | (e > hi))):
                    raise ValueError("native frozen truth outside source interval: "+name)
                ok |= written & finite_g & (g >= lo) & (g <= hi)
            # Readonly held nonfinite payloads remain byte exact.
            eb = np.ascontiguousarray(e).view(np.uint8).reshape(e.size, e.dtype.itemsize)
            gb = np.ascontiguousarray(g).view(np.uint8).reshape(g.size, g.dtype.itemsize)
            same_payload = np.all(eb == gb, axis=1).reshape(e.shape)
            ok |= (~written) & (~finite_e) & same_payload
            ok &= ~(written & ~finite_g)
            bad = ~ok
            reason = "numerical_or_held_payload"
        else:
            eb = np.ascontiguousarray(e).view(np.uint8).reshape(e.size, e.dtype.itemsize)
            gb = np.ascontiguousarray(g).view(np.uint8).reshape(g.size, g.dtype.itemsize)
            bad = (~np.all(eb == gb, axis=1)).reshape(e.shape)
            reason = "integer_or_structural_bytes"
        if np.any(bad):
            flat = int(np.flatnonzero(bad.reshape(-1))[0])
            idx = list(np.unravel_index(flat, e.shape)) if e.shape else []
            item = {"field": name, "reason": reason, "index": [int(v) for v in idx],
                    "bad_words": int(np.count_nonzero(bad)),
                    "expected_bytes": eb[flat].tobytes().hex(),
                    "actual_bytes": gb[flat].tobytes().hex()}
            if e.dtype.kind == "f" and name in bounds:
                item["frozen_lower"] = float(bounds[name][0].reshape(-1)[flat]).hex()
                item["frozen_upper"] = float(bounds[name][1].reshape(-1)[flat]).hex()
            failures.append(item)
    return {"ok": not failures, "fields": len(expected), "failures": failures}


def typed_value_equal(left, right):
    """Source copy/hold relation: exact represented numeric values, raw NaN payloads.

    This does not infer whether an actual write was authorized. Those event masks
    and integer routes are checked independently.
    """
    return compare_arrays({"held_value":np.asarray(left)},
                          {"held_value":np.asarray(right)})["ok"]


def verify_index(npz, index):
    arrays = load(npz)
    if set(arrays) != set(index["fields"]):
        raise ValueError("typed NPZ field set differs from index")
    if pathlib.Path(npz).stat().st_size != index["artifact"]["bytes"]:
        raise ValueError("typed NPZ artifact length mismatch")
    if hash_file(npz) != index["artifact"]["sha256"]:
        raise ValueError("typed NPZ byte hash mismatch")
    for name, a in arrays.items():
        d = index["fields"][name]
        if (a.dtype.str != d["dtype"] or list(a.shape) != d["shape"] or
            a.nbytes != d["bytes"] or hashlib.sha256(a.tobytes()).hexdigest() != d["sha256"]):
            raise ValueError("typed field binding mismatch: "+name)
    return arrays
