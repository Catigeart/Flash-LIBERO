"""Bounded cold collision metadata with exact device snapshot guards.

CUDA compares the original static fields with these retained int32 snapshots
once per call. In-place static changes choose the original classification and
suppression scan on device. Cold downloads/uploads occur before compilation;
the simulation interval performs no host reads or allocations.
"""
from collections import UserDict
import numpy as np
from libero.gpu.model.topology import (
    build_geom_pair_routes, build_pair_suppression, upload_topology,
)

_ROUTE_FIELDS = (
    'geom_type', 'geom_contype', 'geom_conaffinity',
    'geom_priority', 'geom_condim', 'dispatch_present',
)


def _source_owners(ws):
    static = ws.dispatch.static
    result = {name: static[name] for name in _ROUTE_FIELDS}
    result['tree_pair1'] = ws.tree.pair1
    result['tree_pair2'] = ws.tree.pair2
    return result


def _owner_signature(owners):
    return tuple((name, id(value), int(value.ptr), int(value.size),
                  str(value.dtype), str(value.device))
                 for name, value in sorted(owners.items()))


def prepare_collision_metadata(ws, *, max_table_bytes=8 * 1024**2):
    """Return uploaded optional arrays with exactly the CUDA dense-key layout.

    The general topology builder's sparse ``pair_keys`` descriptor must never
    be passed to ``cf_tree``'s dense offsets. No immutable-owner flag is needed:
    the CUDA entry admits each cache using its complete source snapshot.

    Result keys are ``geom_pair_route`` (uint32), ``pair_key_offsets`` and
    ``pair_key_indices`` (int32), ``route_shadow`` and ``pair_shadow`` (int32),
    plus retained host/device owners and proof.
    Missing keys mean that cache is ineligible; omit its optional kernel args.
    The byte cap covers both dense tables and occurrence storage combined.
    """
    if not isinstance(max_table_bytes, int) or max_table_bytes < 0:
        raise ValueError('collision metadata byte cap must be nonnegative int')
    owners = _source_owners(ws)
    signature = (int(ws.ngeom), int(ws.tree.npair), _owner_signature(owners))
    cached = getattr(ws, '_cuda_collision_metadata', None)
    if cached is not None:
        if cached['source_owner_signature'] != signature:
            raise ValueError('collision static owner replaced; rebuild workspace/native plan')
        if cached['max_table_bytes'] != max_table_bytes:
            raise ValueError('collision metadata capacity changed; rebuild native plan')
        return cached
    ng = int(ws.ngeom)
    cells = ng * ng
    # Check the dense ceiling before any device download or Python rows table.
    if ng < 0 or cells > 2147483647 or 4 * cells > max_table_bytes:
        ws._cuda_collision_metadata_reason = 'dense metadata capacity fallback'
        return UserDict()
    source = {name: np.asarray(value.numpy()).reshape(-1)
              for name, value in owners.items()}
    route = None
    route_bytes = 4 * (cells + 5 * ng + 64)
    if (route_bytes <= max_table_bytes and
            all(source[name].size == ng for name in _ROUTE_FIELDS[:-1]) and
            source['dispatch_present'].size == 64):
        route = build_geom_pair_routes(
            source['geom_type'], source['geom_contype'], source['geom_conaffinity'],
            source['geom_priority'], source['geom_condim'], source['dispatch_present'],
            max_cells=cells)
    npair = int(ws.tree.npair)
    pairs = None
    # Two directed occurrences per pair plus the two source snapshots are a
    # safe upper bound; decline before allocating a large Python CSR row list.
    pair_bytes_upper = 4 * (cells + 1) + 16 * max(npair, 0)
    if (npair >= 0 and pair_bytes_upper <= max_table_bytes and
            source['tree_pair1'].size >= npair and source['tree_pair2'].size >= npair):
        pairs = build_pair_suppression(
            source['tree_pair1'][:npair], source['tree_pair2'][:npair], ng,
            max_cells=cells)
    result = UserDict(source_owner_signature=signature,
                      max_table_bytes=max_table_bytes,
                      retained_source_owners=UserDict(owners),
                      host_owners=UserDict(), device_owners=UserDict())
    used = 0
    for kind, descriptor in (('route', route), ('suppression', pairs)):
        if descriptor is None:
            continue
        if kind == 'route':
            shadow = np.concatenate([source[name] for name in _ROUTE_FIELDS]).astype(np.int32)
        else:
            shadow = np.concatenate([source['tree_pair1'][:npair],
                                     source['tree_pair2'][:npair]]).astype(np.int32)
        size = sum(value.nbytes for value in descriptor.arrays.values()) + shadow.nbytes
        if used + size > max_table_bytes:
            continue
        uploaded = upload_topology(descriptor, ws.device)
        import warp as wp
        result['route_shadow' if kind == 'route' else 'pair_shadow'] = wp.array(
            shadow if shadow.size else np.zeros(1, np.int32), dtype=wp.int32, device=ws.device)
        result['host_owners'][kind] = descriptor
        result['device_owners'][kind] = uploaded
        result.update(uploaded.arrays)
        used += size
    result['proof'] = {
        'admission': 'each device invocation compares the full source against its shadow',
        'dense_key': 'g1 * ngeom + g2',
        'duplicates': 'all original explicit pair ordinals retained',
        'source_sha256': {kind: value.source_sha256
                          for kind, value in result['host_owners'].items()},
        'canonical_bytes': used,
        'lifetime': 'retained workspace; changed source values take the scalar device fallback',
    }
    ws._cuda_collision_metadata = result
    ws._cuda_collision_metadata_reason = 'cold descriptors guarded by device snapshots'
    return result
