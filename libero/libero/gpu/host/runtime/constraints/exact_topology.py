"""Bounded exact integer-topology cache prototype, without numeric reuse.

Not enabled by the production schedule. A GPU readset comparison and a sealed
row-descriptor publisher are required before any device parser may be skipped.
"""
import numpy as np
import struct


class ExactTopologyCache:
    def __init__(self, byte_budget):
        if isinstance(byte_budget, bool) or not isinstance(byte_budget, int) or byte_budget < 0:
            raise ValueError('nonnegative topology cache budget required')
        self.byte_budget = byte_budget
        self._key = self._payload = self._domain = None

    @staticmethod
    def _freeze(arrays):
        result = {}
        for key, value in arrays.items():
            a = np.asarray(value)
            if a.dtype != np.int32 or not a.flags.c_contiguous:
                raise ValueError('topology keys/payloads require contiguous int32: ' + key)
            result[key] = (a.shape, a.tobytes(order='C'))
        return result

    def invalidate(self):
        self._key = self._payload = self._domain = None

    @staticmethod
    def _freeze_domain(value):
        if type(value) in (str, bytes, int, bool, type(None)):
            return (type(value).__name__, value)
        if type(value) is float:
            return ('float', struct.pack('!d', value))
        if type(value) in (tuple, list):
            return (type(value).__name__, tuple(ExactTopologyCache._freeze_domain(x) for x in value))
        raise ValueError('topology domain requires exact primitive facts')

    def publish(self, keys, payload, *, domain, valid):
        """Publish only a fully validated producer result, all or nothing.

        Required key leaves are the complete *included and excluded* contact
        sequence and original limit-side map, plus scalar count/capacity. IDs
        are bodies after the separately validated geom->body lookup.
        Numeric fields are deliberately not payloads and must be recomputed.
        """
        required = {'body1', 'body2', 'dim', 'exclude', 'limit_sides', 'counts_capacity'}
        allowed = {'rowstart', 'contact_queue', 'dim_offsets', 'compact_heads',
                   'head_type', 'head_id', 'head_dim', 'mu_owner'}
        self.invalidate()
        if valid is not True: return False
        if set(keys) != required or not set(payload).issubset(allowed):
            raise ValueError('exact topology readset/payload schema')
        k, p = self._freeze(keys), self._freeze(payload)
        if sum(len(raw) for _, raw in [*k.values(), *p.values()]) > self.byte_budget:
            return False
        self._key, self._payload, self._domain = k, p, self._freeze_domain(domain)
        return True

    def lookup(self, keys, *, domain, valid):
        if valid is not True or self._key is None or self._freeze_domain(domain) != self._domain:
            return None
        if self._freeze(keys) != self._key:
            return None
        # New CPU owner on every diagnostic lookup; no mutable alias to cache.
        return {name: np.frombuffer(raw, np.int32).reshape(shape).copy()
                for name, (shape, raw) in self._payload.items()}
