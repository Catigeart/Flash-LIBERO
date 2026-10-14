"""Explicit integer segmentation lookup leaf; no renderer/task installation."""
import numpy as np


class SegmentationLookup:
    def __init__(self, mapping, ngeom):
        if ngeom < 0 or any(not isinstance(k, (int, np.integer)) or not 0 <= k < ngeom for k in mapping):
            raise ValueError('segmentation mapping outside geom range')
        self.lut = np.zeros(ngeom, dtype=np.int32)
        for key, value in mapping.items():
            if not isinstance(value, (int, np.integer)) or not -1 <= value < np.iinfo(np.int32).max:
                raise ValueError('segmentation group ID outside int32 range')
            self.lut[key] = value + 1

    def convert(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in 'iu':
            raise TypeError('integer element segmentation IDs required')
        valid = (ids >= 0) & (ids < len(self.lut))
        # Negative and absent IDs match mapping.get(id, -1)+1 exactly.
        out = np.zeros(ids.shape, dtype=np.int32)
        out[valid] = self.lut[ids[valid]]
        return out
