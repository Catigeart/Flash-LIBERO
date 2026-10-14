"""Chain-local error routing: sticky first_error and stage codes.

Local codes stay independent of the shared contract enums. The new formal
cross-module error binding remains a separate integration dependency. A chain first_error is (stage_id, local_code, item) recorded on
device and never overwritten until the single explicit reset at a batch
boundary. An upstream error must not be masked by a clean zero-contact
result: any nonzero stage error forces the world's first_error and blocks
the finalize commit for that world.
"""
from __future__ import annotations

import numpy as np

# Stage ids for first_error.stage_id (chain order).
STAGE_BOUNDS = 1
STAGE_SAP = 2
STAGE_STAGEBUILD = 3
STAGE_TREE = 4
STAGE_DISPATCH = 5
STAGE_DETECT = 6
STAGE_FINALIZE = 7

STAGE_NAMES = {
    STAGE_BOUNDS: 'bounds',
    STAGE_SAP: 'sap',
    STAGE_STAGEBUILD: 'bodypair_stagebuild',
    STAGE_TREE: 'bvh_invoke',
    STAGE_DISPATCH: 'dispatch',
    STAGE_DETECT: 'detect',
    STAGE_FINALIZE: 'finalize',
}

# Chain-local codes (kept separate from historical module-local codes).
OK = 0
ERR_BAD_ACTIVE = 1
ERR_BAD_COUNT = 2
ERR_CAPACITY = 3
ERR_UNSUPPORTED_ROUTE = 4
ERR_NONFINITE = 5
ERR_NUMERIC_DOMAIN = 6
ERR_BINDING = 7
ERR_INTERNAL = 8
ERR_UPSTREAM_MODULE = 9

CODE_NAMES = {
    ERR_BAD_ACTIVE: 'BAD_ACTIVE',
    ERR_BAD_COUNT: 'BAD_COUNT',
    ERR_CAPACITY: 'CAPACITY',
    ERR_UNSUPPORTED_ROUTE: 'UNSUPPORTED_ROUTE',
    ERR_NONFINITE: 'NONFINITE',
    ERR_NUMERIC_DOMAIN: 'NUMERIC_DOMAIN',
    ERR_BINDING: 'BINDING',
    ERR_INTERNAL: 'INTERNAL',
    ERR_UPSTREAM_MODULE: 'UPSTREAM_MODULE',
}


class FirstError:
    """Host-side snapshot of one world's sticky first error."""

    __slots__ = ('stage', 'code', 'item')

    def __init__(self, stage: int, code: int, item: int):
        self.stage = int(stage)
        self.code = int(code)
        self.item = int(item)

    @property
    def ok(self) -> bool:
        return self.code == OK

    def as_dict(self):
        return {
            'stage': self.stage,
            'stage_name': STAGE_NAMES.get(self.stage, 'unknown'),
            'code': self.code,
            'code_name': CODE_NAMES.get(self.code, 'unknown'),
            'item': self.item,
        }

    def __repr__(self):
        return f"FirstError(stage={self.stage}, code={self.code}, item={self.item})"


def decode(stages: np.ndarray, codes: np.ndarray, items: np.ndarray) -> list[FirstError]:
    """Decode device first_error arrays into per-world snapshots."""
    stages = np.asarray(stages).ravel()
    codes = np.asarray(codes).ravel()
    items = np.asarray(items).ravel()
    if not (len(stages) == len(codes) == len(items)):
        raise ValueError('first_error arrays must have equal length')
    return [FirstError(s, c, i) for s, c, i in zip(stages, codes, items)]
