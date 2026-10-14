"""CPU planning/oracle for the handwritten, dynamic Hessian-island CUDA path.

This is cold analysis metadata, never the production numerical solver.  The
complete Hessian lower triangle is mandatory and authoritative.  Optional
mass/J edges can conservatively merge components, including complete cone
support, but cannot delete any numerical matrix coupling.
"""
from dataclasses import dataclass
import numpy as np


def island_resource_requirements(nv):
    """Conservative cold ledger additions; linked CUDA resources still decide.

    Existing global Dx workspace is reused. POTRF stages at most NV^2 cells;
    TRSM descriptors account for aligned matrix AND RHS shared operands.
    Static reserve covers both live function partitions and private vectors.
    """
    nv = int(nv)
    if nv < 1:
        raise ValueError("positive NV capacity required")
    align2 = lambda n: (n+1) & ~1
    return {"dx_workspace_min_bytes": (align2(nv*nv)+align2(nv))*8,
            "static_reserve_extra_bytes": 64*nv+64,
            "factor_staging_capacity_cells": nv*nv,
            "capacity_buckets": tuple(c for c in (1,2,4,8,16,32) if c <= nv)+(nv,),
            "qualification": "estimate; linked resource certificate is authoritative"}


@dataclass(frozen=True)
class HessianIslands:
    components: tuple
    permutation: np.ndarray
    sizes: tuple
    capacities: tuple
    matrix_offsets: tuple
    padded_cells: int
    use_islands: bool
    fallback_reason: str | None

    @property
    def factor_cubic_work(self):
        return sum(c ** 3 for c in self.capacities)


def padded_capacity(size, nv):
    """Compile-time Dx buckets admitted by cuda/newton/islands_dx_bridge.cu."""
    if size < 1 or size > nv:
        raise ValueError("island size outside NV capacity")
    for capacity in (1, 2, 4, 8, 16, 32):
        if size <= capacity <= nv:
            return capacity
    return nv


def hessian_islands(lower, *, mass=None, jacobian=None, state=None,
                   diagonal=None, contact_id=None, contact_dim=None):
    """Exact current-H graph, optionally unioned with admitted source support.

    `mass` is symmetric dense cold analysis input.  Production reads packed
    mass only if its caller explicitly supplies already-admitted metadata.
    No relative/absolute zero tolerance is permitted in partition discovery.
    D==0 and inactive rows never read their possibly dirty Jacobian values.
    """
    h = np.asarray(lower, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0:
        raise ValueError("H must be a nonempty square matrix")
    nv = h.shape[0]
    parent = list(range(nv))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    def add_matrix(matrix):
        for i in range(nv):
            for j in range(i + 1):
                value = matrix[i, j]
                if not np.isfinite(value):
                    raise ValueError("nonfinite admitted lower matrix")
                if i != j and value != 0.0:
                    join(i, j)

    add_matrix(h)
    if mass is not None:
        m = np.asarray(mass, dtype=np.float64)
        if m.shape != h.shape:
            raise ValueError("mass capacity mismatch")
        add_matrix(m)
    if jacobian is not None:
        j = np.asarray(jacobian, dtype=np.float64)
        s = np.asarray(state)
        if j.ndim != 2 or j.shape[1] != nv or s.shape != (len(j),):
            raise ValueError("J/state capacity mismatch")
        d = None if diagonal is None else np.asarray(diagonal)
        ids = None if contact_id is None else np.asarray(contact_id)
        dims = None if contact_dim is None else np.asarray(contact_dim)
        if d is not None and d.shape != s.shape:
            raise ValueError("diagonal capacity mismatch")
        row = 0
        while row < len(j):
            count = 1
            if s[row] == 4:
                if ids is None or dims is None or ids.shape != s.shape:
                    raise ValueError("cone schema required")
                contact = int(ids[row])
                if contact < 0 or contact >= len(dims):
                    raise ValueError("cone contact id")
                count = int(dims[contact])
                if count not in (1, 3, 4, 6) or row + count > len(j):
                    raise ValueError("cone dimension")
                if np.any(s[row:row+count] != 4) or np.any(ids[row:row+count] != contact):
                    raise ValueError("cone group is not complete")
            elif s[row] != 1 or d is None or d[row] == 0.0:
                row += 1
                continue
            elif not np.isfinite(d[row]):
                raise ValueError("nonfinite admitted weight")
            values = j[row:row+count]
            if not np.isfinite(values).all():
                raise ValueError("nonfinite admitted J support")
            support = np.flatnonzero(np.any(values != 0.0, axis=0))
            if len(support):
                for column in support[1:]:
                    join(int(support[0]), int(column))
            row += count
    labels = [root(i) for i in range(nv)]
    components = tuple(tuple(i for i in range(nv) if labels[i] == label)
                       for label in sorted(set(labels)))
    # Strict postcondition, including subnormal and signed-zero handling.
    for i in range(nv):
        for k in range(i):
            if labels[i] != labels[k] and h[i, k] != 0.0:
                raise AssertionError("off-block Hessian is nonzero")
    sizes = tuple(len(c) for c in components)
    capacities = tuple(padded_capacity(n, nv) for n in sizes)
    offsets = []
    cells = 0
    for cap in capacities:
        cells = (cells + 1) & ~1
        offsets.append(cells)
        cells += cap * cap
    reason = "single_component" if len(components) == 1 else (
        "padded_workspace_capacity" if cells > nv * nv else None)
    return HessianIslands(components, np.array(sum(components, ()), dtype=np.int32),
                         sizes, capacities, tuple(offsets), cells, reason is None, reason)


def lower_symmetric(lower):
    """Build the effective full matrix without reading caller-owned upper cells."""
    a = np.tril(np.asarray(lower, dtype=np.float64))
    return a + np.tril(a, -1).T


def direction_error_bound(lower, rhs, solution):
    """Function-level backward-error gate; no full-chain equivalence claim."""
    a = lower_symmetric(lower)
    b = np.asarray(rhs, dtype=np.float64)
    x = np.asarray(solution, dtype=np.float64)
    residual = np.linalg.norm(a @ x - b, ord=np.inf)
    scale = np.linalg.norm(a, ord=np.inf)*np.linalg.norm(x, ord=np.inf)+np.linalg.norm(b, ord=np.inf)
    nv = len(b)
    # FP64 Cholesky+two triangular solves: a conservative O(NV*eps) bound.
    bound = 128.0*nv*np.finfo(np.float64).eps*max(scale, np.finfo(np.float64).tiny)
    return {"residual_inf": float(residual), "scale": float(scale),
            "bound": float(bound), "passed": bool(np.isfinite(x).all() and residual <= bound)}
