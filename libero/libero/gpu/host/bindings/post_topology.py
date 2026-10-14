"""Pure cold Post descriptors. No device import and no change to model topology."""
import numpy as np


def _int_table(values,name):
    a=np.asarray(values)
    if a.dtype.kind not in ('i','u') or np.any(a<0) or np.any(a>2147483647):
        raise ValueError(name+' requires exact nonnegative int32-domain integers')
    return a.astype(np.int32,copy=False).reshape(-1)


def post_chains(parent):
    parent = _int_table(parent,'Post parent')
    b = len(parent)
    if b < 1 or parent[0] != 0 or any(not 0 <= int(parent[j]) < j for j in range(1, b)):
        raise ValueError('Post chain requires parent-before-child rooted tree')
    children = [[] for _ in range(b)]
    for j in range(b-1, 0, -1):
        children[int(parent[j])].append(j)
    chains, owner = [], np.full(b, -1, np.int32)
    for j in range(1, b):
        if parent[j] != 0 and len(children[int(parent[j])]) == 1:
            continue
        row, k = [], j
        while True:
            owner[k] = len(chains)
            row.append(k)
            if len(children[k]) != 1:
                break
            k = children[k][0]
        chains.append(row)
    assert sorted(j for row in chains for j in row) == list(range(1, b))
    depths = []
    for row in chains:
        p = int(parent[row[0]])
        depths.append(0 if p == 0 else depths[int(owner[p])]+1)
    levels = [[i for i, d in enumerate(depths) if d == level]
              for level in range(max(depths, default=-1)+1)]
    def pack(rows):
        offsets = np.array([0]+list(np.cumsum([len(row) for row in rows])), np.int32)
        return offsets, np.array([j for row in rows for j in row], np.int32)
    co, cb = pack(chains)
    lo, li = pack(levels)
    return dict(chain_offsets=co, chain_bodies=cb, chain_level_offsets=lo,
                chain_level_indices=li, nchain_levels=len(levels),
                proof=dict(body_coverage=b-1, nchains=len(chains),
                           forward_order='parent chain before child; within chain ascending ancestry',
                           reverse_order='child chains before parent; direct children descending body ID'))


def sensor_closure(parent, site_body, sensor_site):
    """Explicit SENSOR-ONLY ABI: descendants for force, ancestors for acc.

    This descriptor cannot certify the legacy full cfrc_ext/cfrc_int domain.
    World body 0 is retained when a sensor consumes its summed wrench.
    """
    parent=_int_table(parent,'Post parent')
    post_chains(parent)  # same cold tree proof
    sb=_int_table(site_body,'Post site body')
    sites=_int_table(sensor_site,'Post sensor site')
    if np.any(sites<0) or np.any(sites>=len(sb)):
        raise ValueError('sensor site domain')
    roots=sb[sites]
    if np.any(roots<0) or np.any(roots>=len(parent)):
        raise ValueError('sensor body domain')
    force=np.zeros(len(parent), np.int32)
    for j in range(len(parent)):
        k=j
        while True:
            if k in roots:
                force[j]=1
                break
            if k==0:
                break
            k=int(parent[k])
    acc=force.copy()
    for j in np.flatnonzero(force):
        k=int(j)
        while k:
            k=int(parent[k]);acc[k]=1
    return dict(force_mask=force, acceleration_mask=acc, sensor_bodies=roots,
                abi='post_sensor_only_v1', public_fields=('sensor',),
                unowned_fields=('cfrc_ext','cfrc_int','cacc','contact_force'),
                proof='union of sensor-body descendants plus all acceleration ancestors')
