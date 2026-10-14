"""Cold conservative body-pair graph for the temporal collision front end.

This module contains no simulation math. Geometry bounds and motion admission
are computed by handwritten CUDA; this graph removes only model-static pairs
which the existing broadphase can never emit. Explicit geom pairs are handled
by the unchanged stage builder, independently of this graph.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


INTEGER_FIELDS = (
    'body_geomadr', 'body_geomnum', 'geom_bodyid', 'body_weldid',
    'body_parentid', 'can', 'plane', 'geom_type', 'geom_contype',
    'geom_conaffinity', 'gates', 'collision_mode', 'filterparent_disabled',
)
FLOAT_FIELDS = ('geom_rbound', 'geom_margin')


@dataclass(frozen=True)
class CollisionCandidateGraph:
    pairs: np.ndarray
    global_pairs: np.ndarray
    integer_shadow: np.ndarray
    float_shadow: np.ndarray
    max_geom_requests: int
    source_sha256: str


def build_collision_candidates(source, nbody, ngeom, *, max_pair_scan=1_000_000):
    """Return packed, sorted, unique conservative body-pair candidates.

    Invalid mutable model data is not repaired here. Reject it before GPU
    allocation. A source value mutation after construction chooses the old
    device path, and replacing any owner requires rebuilding the native plan.
    """
    if not (1 <= nbody <= 32768 and 0 <= ngeom <= 2147483647):
        raise ValueError('temporal collision body/geom domain')
    if not isinstance(max_pair_scan, int) or max_pair_scan < 0:
        raise ValueError('temporal collision cold pair scan ceiling')
    if nbody * (nbody - 1) // 2 > max_pair_scan:
        raise ValueError('temporal collision cold pair scan capacity fallback')
    a = {name: np.asarray(source[name]).reshape(-1) for name in INTEGER_FIELDS + FLOAT_FIELDS}
    lengths = {
        **{name: nbody for name in ('body_geomadr', 'body_geomnum', 'body_weldid', 'body_parentid', 'can', 'plane')},
        **{name: ngeom for name in ('geom_bodyid', 'geom_type', 'geom_contype', 'geom_conaffinity', 'geom_rbound', 'geom_margin')},
        'gates': 3, 'collision_mode': 1, 'filterparent_disabled': 1,
    }
    if any(a[name].size != width for name, width in lengths.items()):
        raise ValueError('temporal collision static shape')
    for b in range(nbody):
        adr, num = int(a['body_geomadr'][b]), int(a['body_geomnum'][b])
        if num < 0 or (num and (adr < 0 or adr + num > ngeom)):
            raise ValueError('temporal collision body geom range')
        if num and np.any(a['geom_bodyid'][adr:adr+num] != b):
            raise ValueError('temporal collision contiguous geom owner mapping')
        weld, parent = int(a['body_weldid'][b]), int(a['body_parentid'][b])
        if not (0 <= weld < nbody and 0 <= parent < nbody):
            raise ValueError('temporal collision body topology')
    if np.any((a['geom_bodyid'] < 0) | (a['geom_bodyid'] >= nbody)):
        raise ValueError('temporal collision geom owner')
    if not np.any(a['geom_bodyid'] != 0):
        raise ValueError('empty moving-geom cloud uses original collision front')
    radius = a['geom_rbound'].astype(np.float64) + a['geom_margin'].astype(np.float64)
    if not (np.all(np.isfinite(radius)) and np.all(a['geom_rbound'] >= 0) and
            np.all(a['geom_margin'] >= 0)):
        raise ValueError('temporal collision finite nonnegative radius and margin')
    # A moving infinite plane needs a different spatial certificate. Retain
    # the original front end for that supported input domain.
    for g in range(ngeom):
        b = int(a['geom_bodyid'][g])
        if int(a['geom_type'][g]) in (0, 1) and int(a['body_weldid'][b]) != 0:
            raise ValueError('moving plane/heightfield uses original collision front')

    def global_body(b):
        return bool(a['can'][b] and ((b == 0 and a['body_geomnum'][b] > 0) or
                    (a['body_weldid'][b] == 0 and a['plane'][b])))

    pairs, global_flags, request_ceiling = [], [], 0
    for b1 in range(nbody):
        for b2 in range(b1 + 1, nbody):
            w1, w2 = int(a['body_weldid'][b1]), int(a['body_weldid'][b2])
            is_global = (global_body(b1) and w2 != 0) or (global_body(b2) and w1 != 0)
            possible = bool(is_global)
            if not possible and b1 > 0 and a['can'][b1] and a['can'][b2]:
                p1 = int(a['body_weldid'][int(a['body_parentid'][w1])])
                p2 = int(a['body_weldid'][int(a['body_parentid'][w2])])
                parent_rejected = not a['filterparent_disabled'][0] and w1 and w2 and (w1 == p2 or w2 == p1)
                possible = w1 != w2 and not parent_rejected
                if possible and a['body_geomnum'][b1] == 1 and a['body_geomnum'][b2] == 1:
                    g1, g2 = int(a['body_geomadr'][b1]), int(a['body_geomadr'][b2])
                    possible = bool((int(a['geom_contype'][g1]) & int(a['geom_conaffinity'][g2])) or
                                    (int(a['geom_contype'][g2]) & int(a['geom_conaffinity'][g1])))
            if possible:
                pairs.append((b1 << 16) + b2)
                global_flags.append(int(is_global))
                request_ceiling += int(a['body_geomnum'][b1]) * int(a['body_geomnum'][b2])
    integers = np.concatenate([a[name].astype(np.int32) for name in INTEGER_FIELDS])
    floats = np.concatenate([a[name].astype(np.float64) for name in FLOAT_FIELDS])
    digest = hashlib.sha256(integers.tobytes() + floats.tobytes()).hexdigest()
    return CollisionCandidateGraph(np.asarray(pairs, dtype=np.int32),
                                   np.asarray(global_flags, dtype=np.int32),
                                   integers, floats, request_ceiling, digest)
