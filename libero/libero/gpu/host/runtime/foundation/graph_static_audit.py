"""Pure static order proof. Driver facts are freshly read on EVERY replay.

Opaque source/private/exec handles are exposed by the current API, so this
memo caches only the exact immutable topology calculation. It never licenses
skipping node/parameter/owner audits, executable rebind/enable or device gates.
"""
from functools import lru_cache

@lru_cache(maxsize=256)
def single_stream_order(ids,dependencies):
    if len(set(ids))!=len(ids) or len(dependencies)!=len(ids):
        raise ValueError('invalid Graph node inventory')
    known=set(ids);children={n:[] for n in ids};remaining={}
    for node,parents in zip(ids,dependencies):
        parents=set(parents)
        if not parents<=known: raise ValueError('Graph dependency outside current node inventory')
        remaining[node]=len(parents)
        for parent in parents: children[parent].append(node)
    ready=[node for node in ids if not remaining[node]];ordered=[]
    while len(ordered)<len(ids):
        if len(ready)!=1: raise ValueError('Graph is not the frozen single-stream source kernel order')
        node=ready.pop();ordered.append(node)
        for child in children[node]:
            remaining[child]-=1
            if remaining[child]==0:ready.append(child)
    return tuple(ordered)
