"""Owned bytes-backed NumPy readsets; ordinary readonly arrays are insufficient."""
import numpy as np

def immutable_array(value):
    copied=np.asarray(value,order='C')
    if copied.dtype.hasobject: raise ValueError('object readsets cannot be frozen as bytes')
    return np.frombuffer(copied.tobytes(order='C'),dtype=copied.dtype).reshape(copied.shape)

def _token(array):
    return (id(array),array.dtype.str,array.dtype.descr if array.dtype.fields else None,
            tuple(array.shape),tuple(array.strides),bool(array.flags.c_contiguous),
            bool(array.flags.writeable),int(array.__array_interface__['data'][0]))

def _bytes_owner(array):
    seen=set();value=array
    while isinstance(value,np.ndarray):
        if id(value) in seen: return None
        seen.add(id(value));value=value.base
    return value if type(value) is bytes else None

class StaticReadsetSeal:
    """Digest encoding comes from the original consumer, never a new schema.

    Replacing arrays, mutating shape/dtype/strides, or ordinary mutable storage
    uses the original strict hash. Only retained bytes-backed arrays reuse it.
    This seals host static data, never GPU in-place content or upload readiness.
    """
    def __init__(self, arrays, identity, digest):
        self.identity=identity;self.digest=digest
        self.rows=tuple((k,a,_token(a),_bytes_owner(a)) for k,a in arrays.items())
        self.enabled=all(owner is not None and not a.flags.writeable for _,a,_,owner in self.rows)
    def valid(self, arrays, identity):
        return (self.enabled and identity==self.identity and
                tuple(arrays)==tuple(k for k,_,_,_ in self.rows) and
                all(arrays[k] is a and _token(a)==token and _bytes_owner(a) is owner
                    for k,a,token,owner in self.rows))
    def current_digest(self, arrays, identity, strict):
        return self.digest if self.valid(arrays,identity) else strict()
