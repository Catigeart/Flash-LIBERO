"""Lossless immutable parsed JSON facts; public projections remain independent."""
from types import MappingProxyType

def freeze_json(value):
    if isinstance(value,dict):
        return MappingProxyType({k:freeze_json(v) for k,v in value.items()})
    if isinstance(value,list):
        return tuple(freeze_json(v) for v in value)
    return value

def thaw_json(value):
    if hasattr(value,"items"):
        return {k:thaw_json(v) for k,v in value.items()}
    if isinstance(value,tuple):
        return [thaw_json(v) for v in value]
    return value
