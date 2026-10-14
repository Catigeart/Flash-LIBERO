"""Local row/ABI schema, error protocol, Newton input descriptor, hold domains.

This is the P06-S local schema under the exclusive ``constraint runtime``
namespace.  The cross-task shared-v1 ABI (P06-F) is NOT supplied by the REF/F
user decision; the fields below are the local contract this task publishes and
the shared binding stays a recorded blocker (see doc/phase_06/tasks/S/
BLOCKERS.md).  Nothing here rewrites production standards, existing pins or
tolerances.
"""
from __future__ import annotations

import numpy as np

from .identity import NV, NF_CAP, NL_CAP

# Row type enum, literal values from mjmodel.h mjCNSTR enum.
TYPE_EQUALITY = 0
TYPE_FRICTION_DOF = 1
TYPE_FRICTION_TENDON = 2
TYPE_LIMIT_JOINT = 3
TYPE_LIMIT_TENDON = 4
TYPE_CONTACT_FRICTIONLESS = 5
TYPE_CONTACT_PYRAMIDAL = 6
TYPE_CONTACT_ELLIPTIC = 7

SUPPORTED_ROW_TYPES = (TYPE_FRICTION_DOF, TYPE_LIMIT_JOINT,
                       TYPE_CONTACT_FRICTIONLESS, TYPE_CONTACT_ELLIPTIC)

# ---------------------------------------------------------------------------
# Error protocol.  Local codes are new for this package (shared-v1 enum is not
# frozen); codes are stable within phase06-constraints-v1.  status is sticky
# per world: the first error code is written once and survives later launches
# until the explicit reset_status() boundary call.  On any error world, no
# public output is written (all-or-nothing commit at the assembler).
# ---------------------------------------------------------------------------
STATUS_OK = 0
ERR_BAD_ACTIVE = 1       # active not in {0, 1}
ERR_BAD_COUNT = 2        # ncon/counts out of range
ERR_CAPACITY = 3         # row/contact capacity insufficient
ERR_NONFINITE = 4        # non-finite or >1e100 magnitude in a numeric input
ERR_BAD_CONTACT = 5      # invalid dim/geom/body on a contact record
ERR_BAD_STATIC = 6       # fixed-static gate failure (prepare-side)
ERR_BAD_SCHEMA = 7       # host-side dtype/shape/schema failure (prepare-side)
ERR_UNSUPPORTED = 8      # route/power/model outside the supported domain
ERR_STALE_EPOCH = 9      # reserved for connection witnesses; local tests use it

ERROR_NAMES = {
    STATUS_OK: 'OK',
    ERR_BAD_ACTIVE: 'BAD_ACTIVE',
    ERR_BAD_COUNT: 'BAD_COUNT',
    ERR_CAPACITY: 'CAPACITY',
    ERR_NONFINITE: 'NONFINITE',
    ERR_BAD_CONTACT: 'BAD_CONTACT',
    ERR_BAD_STATIC: 'BAD_STATIC',
    ERR_BAD_SCHEMA: 'BAD_SCHEMA',
    ERR_UNSUPPORTED: 'UNSUPPORTED',
    ERR_STALE_EPOCH: 'STALE_EPOCH',
}

SAFE_MAGNITUDE = 1.0e100

# Dense-J row schema: every row is a full nv=43 column vector.  All-zero rows
# are legal and are never deleted; the dense43 schema is exact (a 44th column
# or any NaN fails the comparator at the schema layer).
DENSE_COLS = NV

# Segment capacities inherited from the accepted S02 producer.
SEG_FRICTION_ROWS = NF_CAP
SEG_LIMIT_ROWS = NL_CAP


def check_host_array(value, shape, dtype, label):
    """Strict host schema check used by prepare-side upload paths."""
    arr = np.asarray(value)
    if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype):
        raise ValueError(
            f'{label}: expected {tuple(shape)}/{np.dtype(dtype)}, got {arr.shape}/{arr.dtype}')
    return np.ascontiguousarray(arr)


ROW_FIELDS_FLOAT = ('pos', 'margin', 'frictionloss')
ROW_FIELDS_INT = ('type', 'id')


def newton_input_descriptor(row_capacity: int, contact_capacity: int,
                            nworld: int = 1) -> dict:
    """Describe exactly what a consumer (P06-N) reads after launch_position.

    Writer stage, epoch and hold domain are explicit for every field.  The
    descriptor is the connection contract; verification against an actual N
    consumer requires the missing shared-v1 ABI and is recorded as blocked.
    """
    R, C = int(row_capacity), int(contact_capacity)
    N = int(nworld)
    return {
        'schema': 'phase06-constraints-newton-input-v1',
        'nworld': N,
        'row_capacity': R,
        'contact_capacity': C,
        'dense_cols': NV,
        'arrays': {
            'efc_J': {
                'dtype': 'float64', 'shape': [N, R, NV], 'writer': 'S.assemble',
                'valid_prefix': 'rows [0, nefc) per world; tail preserved',
                'notes': 'dense full columns; all-zero rows retained',
            },
            'efc_type': {
                'dtype': 'int32', 'shape': [N, R], 'writer': 'S.assemble',
                'valid_prefix': 'rows [0, nefc)',
            },
            'efc_id': {
                'dtype': 'int32', 'shape': [N, R], 'writer': 'S.assemble',
                'valid_prefix': 'rows [0, nefc)',
            },
            'efc_pos': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.assemble',
                'valid_prefix': 'rows [0, nefc)',
            },
            'efc_margin': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.assemble',
                'valid_prefix': 'rows [0, nefc)',
            },
            'efc_frictionloss': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.assemble',
                'valid_prefix': 'rows [0, nefc)',
            },
            'efc_diagApprox': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.impedance (postdiag overwrite)',
                'valid_prefix': 'rows [0, nefc) when nefc>0; whole array preserved when nefc==0',
                'notes': (
                    'source semantics: mj_diagApprox writes the pre value and '
                    'mj_makeImpedance overwrites efc_diagApprox with '
                    'R*imp/(1-imp) (postdiag). N consumes the POST value.'
                ),
            },
            'efc_R': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.impedance',
                'valid_prefix': 'rows [0, nefc) when nefc>0; preserved when nefc==0',
            },
            'efc_D': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.impedance',
                'valid_prefix': 'rows [0, nefc) when nefc>0; preserved when nefc==0',
            },
            'efc_KBIP': {
                'dtype': 'float64', 'shape': [N, R, 4], 'writer': 'S.impedance',
                'components': ['K', 'B', 'I(imp)', "P(imp')"],
                'valid_prefix': 'rows [0, nefc) when nefc>0; preserved when nefc==0',
            },
            'efc_vel': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.reference',
                'valid_prefix': 'rows [0, nefc) when nefc>0; preserved when nefc==0',
                'epoch': 'bound to the qvel epoch passed to launch_reference',
            },
            'efc_aref': {
                'dtype': 'float64', 'shape': [N, R], 'writer': 'S.reference',
                'valid_prefix': 'rows [0, nefc) when nefc>0; preserved when nefc==0',
                'epoch': 'bound to the qvel epoch passed to launch_reference',
                'formula': 'aref = -B*vel - K*I*(pos-margin)  (mj_referenceConstraint)',
            },
            'counts': {
                'dtype': 'int32', 'shape': [N, 5], 'writer': 'S.assemble',
                'fields': ['ne', 'nf', 'nlimit', 'nefc', 'nnzJ'],
                'notes': 'ne is the static-zero equality count; nnzJ = nefc*43 (dense)',
            },
            'contact_efc_address': {
                'dtype': 'int32', 'shape': [N, C], 'writer': 'S.contact segment',
                'notes': 'first global row of the contact block; -1 when excluded/no-dof',
            },
            'contact_exclude': {
                'dtype': 'int32', 'shape': [N, C], 'writer': 'S.contact segment',
                'notes': 'source exclude flags preserved for excluded contacts; 3 for no-dof',
            },
            'contact_mu': {
                'dtype': 'float64', 'shape': [N, C], 'writer': 'S.impedance (partial write set)',
                'notes': (
                    'written ONLY for frictional elliptic contacts '
                    '(type 7 with dim>=2) in the source pass-2 order; all other '
                    'slots preserve their previous bytes'
                ),
            },
            'status': {
                'dtype': 'int32', 'shape': [N], 'writer': 'any stage (sticky first error)',
                'notes': '0 on success (byte preserved); first nonzero code sticks until reset_status()',
            },
            'ref_epoch': {
                'dtype': 'int64', 'shape': [N], 'writer': 'S.reference',
                'notes': (
                    'copies the qvel_epoch operand of the last launch_reference; '
                    'stamped by the row-0 thread for committed worlds with nefc>0; '
                    'nefc==0 and inactive worlds hold their previous bytes (no '
                    'stamp), matching the source early-exit of mj_mulJacVec'
                ),
            },
        },
        'hold_domains': {
            'contact_H': (
                'NOT written by S; retains the collision-initialization value '
                'until the N solver write path (source: collision_driver sets '
                'H=+0; only the solver writes H on its own paths)'
            ),
            'efc_AR': (
                'NOT written by S and NOT written by mj_projectConstraint in '
                'the primal configuration (see primal_noop_proof)'
            ),
            'row_tail': 'rows [nefc, R) keep their previous bytes on every launch',
            'contact_tail': 'contact slots [ncon, C) keep their previous bytes',
            'inactive_world': 'every public byte of an inactive world is preserved',
            'zero_nefc_world': (
                'source `if (!d->nefc) return` skips diagApprox and impedance '
                'entirely: when nefc==0 the previous efc_R/D/KBIP/diagApprox/'
                'mu bytes are preserved, exactly like the CPU arena path'
            ),
        },
        'not_written_by_S': ['efc_AR', 'contact_H', 'efc_force', 'efc_state',
                             'qacc', 'qacc_warmstart', 'qfrc_constraint'],
    }


def coverage_row(name: str, status: str, note: str = '') -> dict:
    return {'item': name, 'status': status, 'note': note}
