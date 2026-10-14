"""Exact immutable CPU model bytes; no mutable owners or execution permission."""
from collections import OrderedDict
import json
from threading import RLock
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
import numpy as np

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

class ArrayBytes(tuple):
    """Immutable value tuple; object.__setattr__ cannot replace payload fields."""
    __slots__ = ()

    def __new__(cls, name, dtype, shape, payload):
        if not isinstance(name, str) or not isinstance(dtype, str) or not isinstance(payload, bytes):
            raise TypeError("array identity requires a name and immutable bytes")
        actual_dtype = np.dtype(dtype)
        if actual_dtype.hasobject or not isinstance(shape, tuple) or any(
                not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("stable dtype and nonnegative tuple shape required")
        count = 1
        for dimension in shape:
            count *= dimension
        if count * actual_dtype.itemsize != len(payload):
            raise ValueError("array payload size differs from dtype/shape")
        return tuple.__new__(cls, (name, dtype, shape, payload))

    name = property(lambda self: self[0])
    dtype = property(lambda self: self[1])
    shape = property(lambda self: self[2])
    payload = property(lambda self: self[3])

    @classmethod
    def capture(cls, name, value):
        original = np.asarray(value)
        a = np.ascontiguousarray(original)
        if a.dtype.hasobject:
            raise ValueError("Object arrays cannot define a stable model identity: " + name)
        return cls(name, original.dtype.str, original.shape, a.tobytes())

    @property
    def header(self):
        # np.ascontiguousarray promotes a zero-dimensional array to shape (1,).
        # The historical digest uses that shape; the public view keeps shape ().
        encoded_shape = self.shape if self.shape else (1,)
        return (self.name.encode("utf-8") + b"\0" + self.dtype.encode("ascii") +
                b"\0" + str(encoded_shape).encode("ascii") + b"\0")

    @property
    def encoded_digest(self):
        h = sha256(self.header); h.update(self.payload)
        return h.hexdigest()

    def array(self):
        # The ultimate owner is immutable bytes; setflags(write=True) fails.
        return np.frombuffer(self.payload, dtype=np.dtype(self.dtype)).reshape(self.shape)

    def equals(self, value):
        a = np.asarray(value)
        return (a.dtype.str == self.dtype and a.shape == self.shape and
                np.ascontiguousarray(a).tobytes() == self.payload)

@dataclass(frozen=True)
class ReadOnlyModelSnapshot:
    fields: object
    model_blob: bytes = None

    def __post_init__(self):
        if self.model_blob is not None and not isinstance(self.model_blob, bytes):
            raise TypeError("model_blob must be exact bytes")
        fields = dict(self.fields)
        if any(not isinstance(field, ArrayBytes) or name != field.name
               for name, field in fields.items()):
            raise TypeError("snapshot fields must be immutable named ArrayBytes")
        # A frozen dataclass alone would still accept a caller-owned mutable dict.
        object.__setattr__(self, "fields", MappingProxyType(dict(sorted(fields.items()))))

    @classmethod
    def capture(cls, arrays, *, model_blob=None):
        if model_blob is not None and not isinstance(model_blob, bytes):
            raise TypeError("model_blob must be exact bytes")
        fields = {name: ArrayBytes.capture(name, value) for name, value in sorted(arrays.items())}
        return cls(MappingProxyType(fields), model_blob)

    def arrays(self, names=None):
        names = tuple(self.fields) if names is None else tuple(names)
        return MappingProxyType({name: self.fields[name].array() for name in names})

    def readset_digest(self, names=None):
        h = sha256()
        for name in sorted(self.fields if names is None else names):
            field = self.fields[name]; h.update(field.header); h.update(field.payload)
        return h.hexdigest()

    @property
    def array_digests(self):
        return {name: field.encoded_digest for name, field in self.fields.items()}

    @property
    def byte_size(self):
        return sum(len(field.payload) for field in self.fields.values()) + len(self.model_blob or b"")

    def validate_model(self, native, names=None):
        # A snapshot is not proof that a publicly reachable native owner stayed
        # unchanged. The caller still compares actual current bytes.
        return all(self.fields[name].equals(getattr(native, name))
                   for name in (self.fields if names is None else names))

    def reuse_unchanged(self, previous):
        """Share only exact equal immutable payloads; retain the new MJB identity."""
        fields = {name: (previous.fields[name] if previous.fields.get(name) == field else field)
                  for name, field in self.fields.items()}
        return ReadOnlyModelSnapshot(MappingProxyType(fields), self.model_blob)

    def materialize_blob(self, directory, filename="ACTUAL_MODEL.mjb"):
        if not self.model_blob:
            raise ValueError("actual compiled model bytes were not captured")
        if Path(filename).name != filename or filename in ("", ".", ".."):
            raise ValueError("single file component required")
        root = Path(directory); root.mkdir(parents=True, exist_ok=True)
        path = root / filename
        # Do not overwrite a different retained owner's evidence.
        with path.open("xb") as stream:
            stream.write(self.model_blob)
        return {"path": str(path.resolve()), "sha256": sha256(self.model_blob).hexdigest()}

class SnapshotPool:
    """Bounded exact interning of immutable bytes, never mutable native objects."""
    def __init__(self, max_entries=16, max_bytes=128 << 20):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("nonnegative cache bounds required")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._items = OrderedDict(); self._bytes = 0; self._lock = RLock()
        self.hits = self.misses = 0

    def intern(self, snapshot, *, decoder_identity):
        # Caller supplies exact decoder/version/schema identity. This cache does
        # not grant source/runtime or numerical permission.
        identity = _canonical(decoder_identity)
        # An empty cache cannot have a hit. Reject a known uncacheable
        # exact immutable snapshot before hashing all its fields/MJB again.
        # Preserve decoder JSON errors above this branch, and retain original
        # hash-first semantics for existing entries, subclasses or overrides.
        exact = (type(snapshot) is ReadOnlyModelSnapshot and
                 set(vars(snapshot)) == {"fields", "model_blob"} and
                 type(snapshot.fields) is MappingProxyType and
                 (snapshot.model_blob is None or type(snapshot.model_blob) is bytes) and
                 all(type(name) is str and type(field) is ArrayBytes and
                     type(field.name) is str and type(field.dtype) is str and
                     type(field.payload) is bytes and type(field.shape) is tuple and
                     all(type(n) is int for n in field.shape)
                     for name, field in snapshot.fields.items()))
        if exact:
            with self._lock:
                if not self._items and (not self.max_entries or snapshot.byte_size > self.max_bytes):
                    # Header encoding can fail even for byte-backed fields (for
                    # example an unpaired surrogate name). Keep those original
                    # sorted readset errors before changing the miss counter.
                    for name in sorted(snapshot.fields):
                        snapshot.fields[name].header
                    self.misses += 1
                    return snapshot
        key = (identity, snapshot.readset_digest(),
               sha256(snapshot.model_blob).digest() if snapshot.model_blob is not None else None)
        with self._lock:
            old = self._items.get(key)
            if old is not None:
                # Hashes only index candidates; exact payload equality proves hit.
                if old.fields == snapshot.fields and old.model_blob == snapshot.model_blob:
                    self._items.move_to_end(key); self.hits += 1
                    return old
            self.misses += 1
            if not self.max_entries or snapshot.byte_size > self.max_bytes:
                return snapshot
            previous = self._items.pop(key, None)
            if previous is not None: self._bytes -= previous.byte_size
            self._items[key] = snapshot; self._bytes += snapshot.byte_size
            while len(self._items) > self.max_entries or self._bytes > self.max_bytes:
                _, evicted = self._items.popitem(last=False); self._bytes -= evicted.byte_size
            return snapshot

    def ledger(self):
        with self._lock:
            return {"entries": len(self._items), "bytes": self._bytes,
                    "hits": self.hits, "misses": self.misses,
                    "max_entries": self.max_entries, "max_bytes": self.max_bytes}
