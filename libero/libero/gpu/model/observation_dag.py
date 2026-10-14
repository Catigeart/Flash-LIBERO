"""Cold, conservative schedules for the handwritten low-dimensional sampler.

Edges preserve the original sensor order for RAW, WAR, WAW and presence hazards.
A dependency on a later sensor therefore still reads its retained prior sample.
The CUDA entry checks the exact integer schema before using this schedule.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObservationDAG:
    layer_offsets: tuple
    sensor_order: tuple
    schema: tuple
    nindex: int
    reason: str = ""

    @property
    def eligible(self):
        return not self.reason

    @property
    def nlayers(self):
        return len(self.layer_offsets) - 1 if self.eligible else 0

    def matches(self, meta):
        """Cold CPU check. Runtime CUDA uses exact comparisons, without a hash."""
        return self.schema == tuple(int(x) for a in meta[:9] for x in np.asarray(a).reshape(-1))

    def upload(self, device):
        """Cold upload only; descriptors are private owners, not mutable metadata."""
        if not self.eligible:
            return None
        import warp as wp
        def ai(values):
            return wp.array(values, dtype=wp.int32, device=device)
        return (self.nlayers, self.nindex, ai(self.layer_offsets),
                ai(self.sensor_order), ai(self.schema))


def build_observation_dag(meta, width, source_stride):
    """Return a validated immutable schedule or an explicit ordered fallback.

    ``meta`` is the existing eleven-entry sampler metadata list, supplied as
    host arrays. Clocks, enabled flags and periods may change independently;
    all structural integer arrays (the first nine entries) are frozen/checked.
    """
    def fallback(reason):
        return ObservationDAG((), (), (), 0, reason)
    try:
        arrays = [np.asarray(a) for a in meta[:9]]
        if len(arrays) != 9 or any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrays):
            return fallback("integer vector schema required")
        if any(np.any(a < np.iinfo(np.int32).min) or np.any(a > np.iinfo(np.int32).max) for a in arrays):
            return fallback("schema exceeds int32")
        op, offsets, sizes, ixoffs, indices, d0, d1, d2, presence = arrays
        ns = len(op)
        if ns < 2 or any(len(arrays[i]) != ns for i in (1, 2, 3, 5, 6, 7)) or len(presence) != 3*ns:
            return fallback("no parallel sensors or invalid schema dimensions")
        if width < 0 or source_stride < 0 or np.any(op < 0) or np.any(op > 5):
            return fallback("unsupported operation or dimensions")
        if np.any(offsets < 0) or np.any(sizes < 0) or np.any(offsets.astype(np.int64)+sizes > width):
            return fallback("sensor output exceeds width")
        reads, writes = [], []
        def interval(space, at, n):
            at, n = int(at), int(n)
            if at < 0 or n < 0 or at+n > (ns if space == "present" else width):
                raise ValueError("dependency range exceeds storage")
            return (space, at, at+n)
        for s in range(ns):
            kind, n, at = int(op[s]), int(sizes[s]), int(offsets[s])
            if (kind == 3 and n != 16) or (kind == 4 and n != 3) or (kind == 5 and n != 4):
                return fallback("pose operation width mismatch")
            r = []
            w = [interval("cache", at, n), interval("observed", at, n), interval("present", s, 1)]
            if kind == 0:
                begin = int(ixoffs[s])
                if begin < 0 or begin+n > len(indices) or np.any(indices[begin:begin+n] < 0) or np.any(indices[begin:begin+n] >= source_stride):
                    return fallback("raw source index exceeds storage")
            else:
                count = 2 if kind == 3 else 3 if kind == 4 else 1
                r += [interval("present", presence[3*s+k], 1) for k in range(count)]
                if kind in (1, 2):
                    r += [interval("cache", d0[s], n)]
                elif kind == 3:
                    r += [interval("cache", d0[s], 3), interval("cache", d1[s], 4)]
                elif kind == 4:
                    r += [interval("cache", d0[s], 3), interval("cache", d1[s], 4), interval("cache", d2[s], 16)]
                    w += [interval("side", at, 4)]
                else:
                    r += [interval("side", d0[s], 4)]
            reads.append(r); writes.append(w)
        def intersects(a, b):
            return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]
        def hazard(left, right):
            return any(intersects(a, b) for a in left for b in right)
        levels = [0]*ns
        edges = []
        for i in range(ns):
            for j in range(i+1, ns):
                if hazard(writes[i], reads[j]) or hazard(reads[i], writes[j]) or hazard(writes[i], writes[j]):
                    edges.append((i, j))
                    levels[j] = max(levels[j], levels[i]+1)
        buckets = [[s for s in range(ns) if levels[s] == level] for level in range(max(levels)+1)]
        if max(map(len, buckets)) < 2:
            return fallback("ordered chain has no parallel layer")
        order = tuple(s for bucket in buckets for s in bucket)
        starts = [0]
        for bucket in buckets:
            starts.append(starts[-1]+len(bucket))
        # Final immutable proof checks avoid silently producing malformed layers.
        assert sorted(order) == list(range(ns))
        assert all(levels[i] < levels[j] for i, j in edges)
        assert all(not (hazard(writes[i], reads[j]) or hazard(reads[i], writes[j]) or hazard(writes[i], writes[j]))
                   for bucket in buckets for i in bucket for j in bucket if i < j)
        schema = tuple(int(x) for a in arrays for x in a)
        return ObservationDAG(tuple(starts), order, schema, len(indices))
    except (ValueError, TypeError, OverflowError, IndexError) as exc:
        return fallback(str(exc))


def prepare_observation_dag(low, width, source_stride):
    """Construction-time metadata download/upload, never called in hot sample."""
    desc = build_observation_dag([a.numpy() for a in low.meta[:9]], width, source_stride)
    return desc, desc.upload(low.device)
