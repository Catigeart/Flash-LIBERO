"""Cold, bounded cache of ordered contact-pair structural Jacobian support.

Only integer topology is cached. Contact positions, frames, cdof and subtree
centers remain runtime inputs. The union is a conservative support: a column
in it may cancel numerically, and cannot alone prove solver independence.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np


def _readonly(value):
    result = np.ascontiguousarray(value, dtype=np.int32)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ConstraintPairSupport:
    ancestor_mask: np.ndarray
    pair_lookup: np.ndarray
    pair_offsets: np.ndarray
    pair_columns: np.ndarray
    pair_members: np.ndarray
    source_sha256: str
    byte_budget: int
    cached_body_pairs: int

    @property
    def nbody(self):
        return self.ancestor_mask.shape[0]

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.pair_lookup, self.pair_offsets,
                   self.pair_columns, self.pair_members))

    def pair_support(self, body1, body2):
        """Return columns/membership, including the qualified CSR fallback."""
        if not (0 <= body1 < self.nbody and 0 <= body2 < self.nbody):
            raise ValueError('S pair support body range')
        key = int(self.pair_lookup[body1 * self.nbody + body2]) if self.pair_lookup.size else -1
        if key >= 0:
            first, last = self.pair_offsets[key:key + 2]
            return self.pair_columns[first:last], self.pair_members[first:last]
        first, second = self.ancestor_mask[body1], self.ancestor_mask[body2]
        columns = np.flatnonzero(first | second).astype(np.int32)
        members = (first[columns] | (second[columns] << 1)).astype(np.int32)
        return columns, members

    def validate_source(self, ancestor_mask):
        source = np.asarray(ancestor_mask)
        if source.shape != self.ancestor_mask.shape or not np.array_equal(source, self.ancestor_mask):
            raise ValueError('S pair support source changed; rebuild constraints')


def build_constraint_pair_support(ancestor_mask, *, byte_budget=8 * 1024 * 1024):
    """Cache each distinct ordered ancestor-pair pattern once, within budget.

    A dense body-pair lookup makes the GPU lookup constant time. If that lookup
    itself exceeds budget, caching is disabled. Individual patterns exceeding
    remaining budget retain -1 and use the existing ascending CSR merge.
    Construction neither invents ancestry nor drops possible Jacobian columns.
    """
    original = np.asarray(ancestor_mask)
    if original.ndim != 2 or original.shape[0] < 1 or original.dtype != np.int32:
        raise ValueError('S pair support requires int32 body/DOF mask')
    if np.any((original != 0) & (original != 1)):
        raise ValueError('S pair support ancestor mask must be binary')
    if not isinstance(byte_budget, (int, np.integer)) or byte_budget < 0:
        raise ValueError('S pair support byte budget must be nonnegative')
    source = _readonly(original.copy())
    digest = sha256(source.tobytes(order='C')).hexdigest()
    nbody = source.shape[0]
    if 4 * nbody * nbody + 4 > byte_budget or nbody * nbody > np.iinfo(np.int32).max:
        return ConstraintPairSupport(source, _readonly([]), _readonly([0]),
            _readonly([]), _readonly([]), digest, int(byte_budget), 0)
    # Bodies with the same ancestor set share all pair-pattern payloads.
    signatures, ids, body_keys = [], {}, []
    for row in source:
        key = row.tobytes()
        if key not in ids:
            ids[key] = len(signatures)
            signatures.append(row)
        body_keys.append(ids[key])
    patterns, offsets, columns, members = {}, [0], [], []
    lookup = np.full(nbody * nbody, -1, dtype=np.int32)
    used_bytes = lookup.nbytes + 4
    # Ordered patterns intentionally keep the original body1/body2 member bits.
    for body1 in range(nbody):
        first = source[body1]
        for body2 in range(nbody):
            pattern = (body_keys[body1], body_keys[body2])
            if pattern not in patterns:
                second = source[body2]
                support = np.flatnonzero(first | second)
                added_bytes = 4 + 8 * len(support)
                if used_bytes + added_bytes <= byte_budget:
                    index = len(offsets) - 1
                    columns.extend(support.tolist())
                    members.extend((first[support] | (second[support] << 1)).tolist())
                    offsets.append(len(columns))
                    patterns[pattern] = index
                    used_bytes += added_bytes
                else:
                    patterns[pattern] = -1
            lookup[body1 * nbody + body2] = patterns[pattern]
    result = ConstraintPairSupport(source, _readonly(lookup), _readonly(offsets),
        _readonly(columns), _readonly(members), digest, int(byte_budget),
        int(np.count_nonzero(lookup >= 0)))
    if result.nbytes > max(4, byte_budget):
        raise AssertionError('S pair cache byte accounting')
    return result
