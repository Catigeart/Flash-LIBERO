"""Fixed-model identity, frozen static read set, equality/tendon zero proof.

The pinned model is the original task MJB (MuJoCo 2.3.7 binary, SHA256
3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42):
nq=48 nv=43 nu=9 nM=153 nbody=38 ngeom=270 nsite=31 njnt=18
neq=0 ntendon=0 nplugin=0 nmocap=0 na=0 npair=0 nconmax=5000,
opt: timestep=0.002 impratio=20 solver=Newton(2) cone=elliptic(1)
jacobian=AUTO(2 -> dense because nv<60) disableflags=0 enableflags=0
noslip_iterations=0.

Equality/tendon static proof
----------------------------
``neq == 0`` and ``ntendon == 0`` are read from the frozen MJB itself and
sealed here as booleans with the model digest.  Under those values the source
loops ``mj_instantiateEquality`` (over ``m->neq``), tendon friction
(``m->ntendon``) and tendon limit (``m->ntendon``) iterate zero times, so
their row contributions are structurally zero -- not ignored.  Any model whose
static identity differs is rejected by :func:`validate_fixed_static` with
``unsupported_static``; a nonzero equality/tendon model is never processed by
this package.
"""
from __future__ import annotations

import hashlib

import numpy as np

MJB_SHA256 = '3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42'
ENGINE_CORE_CONSTRAINT_SHA256 = '1717235915250442ba48b10f22d20771aecd28c66f412bb41237c6a585dbff03'

# Scalar dimensions of the frozen model.
NQ, NV, NU, NM, NBODY, NGEOM, NSITE, NJNT = 48, 43, 9, 153, 38, 270, 31, 18
NEQ, NTENDON, NPLUGIN, NMOCAP, NA, NPAIR = 0, 0, 0, 0, 0, 0
NCONMAX = 5000
TIMESTEP, IMPRATIO = 0.002, 20.0
SOLVER, CONE, JACOBIAN, DISABLEFLAGS, ENABLEFLAGS, NOSLIP = 2, 1, 2, 0, 0, 0

# Counts proven from the frozen MJB read set.
NF_FRICTION_DOF = 9        # dofs with frictionloss > 0
NLIMIT_JOINTS = 13         # limited slide/hinge joints
NL_LIMIT_MAX = 26          # both sides of each limited joint can fire
NF_CAP, NL_CAP = 9, 26     # segment capacities fixed by S02's accepted schema

# Fixed static digests already pinned by the accepted phase-05 producers.
FIXED_STATIC_SHA = {
    'dof_frictionloss': '4d0f6b2a64c9435b80fe32a0beff4cee9931b12207138a9d5295805afe6a9a58',
    'jnt_type': '4c0e0d380d48f2a6a7f66baa70f16e9bef2974dcffe93545a1da175cd5c9f2ce',
    'jnt_limited': '9fa4b56dad88df916bdb60474e25f570af8f4620424a9648f058b373cfbdc38f',
    'jnt_qposadr': '38ae4abac73ad2b845e1908b24e91cd00b1d3df40b5c6f885075fe46adb9ab35',
    'jnt_dofadr': '13c9448fa632713f92cc44c1c525c473e12ce7105bc7b23c30d8fcdf8b90a02b',
    'jnt_range': '043664b7a7266553f2d7d50e22bb94502e3dbca01c993b91831ea12dd98509fe',
    'jnt_margin': '81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f',
    'dof_solref': '3a51b5cfd4ed59fc256e10ff95668f53808d6c3d6285e69b85f45477b6668d50',
    'dof_solimp': '3d5ff332a4210168e5ee911f9d443f5019371006905232fa018f1184fff56687',
    'jnt_solref': 'fd23518305cd031f74079325f4349194a1cec0fda80de3594c9c303455377a9c',
    'jnt_solimp': 'e27ba3e0e41de43a80a5399be5db810b36b8062dc87e6ca71f05cb146f3f91be',
    'geom_bodyid': '070eb94ada17eacb0594540416f6bd8f58daa167c0158727e3b7bfbfbf206189',
    'body_invweight0': 'acdcd90bdc312dbb12207478253c863d9e3eb8f94f7a3a8c8361c2409d5290f6',
    'dof_invweight0': '948ba8f2b7f47fcb111d8fbe72a24c7a6dce8a63e9adadbce1154d0451b21568',
    'body_rootid': 'df271de92a2e6e9782eaed57ef00820880168b5761ab153bfa2b28cf9af7e0dd',
    'body_parentid': '8d18b41c7e02b2def07e3c35852487ddd6e10d4f7e4d6ecfcbedd7c1103a4690',
    'body_dofnum': '71beb793d979c86c1de61266643a4a4df400230df830cf95aed21129d4328e76',
    'body_dofadr': '0c4901bfb9cac58bf2e64a19b80dd06c0ec5da06d565a9a7f771197d48fcbd33',
    'dof_parentid': 'a57334b5f8e08a96647a0de0fc27eaac9e1fbfa1c76f007f5308360f2f4fd76a',
}

# Additional topology arrays consumed by the integrated chain (S01 contact
# static).  Digests were extracted from the frozen MJB (sha256 above) by
# scripts/gpu_phase06_constraints/casegen.extract_static and pinned here so
# validate_fixed_static rejects any topology drift, shape-only checks
# notwithstanding.
TOPOLOGY_FIELDS = (
    'body_rootid', 'body_parentid', 'body_dofnum', 'body_dofadr', 'dof_parentid',
)

# Model-level facts that live outside the array read set.  A caller must pass
# them via validate_model_facts (the workspace gate does); a model that would
# need equality/tendon rows or the dual solver cannot present matching arrays
# and slip through.
MODEL_FACTS = {
    'neq': 0, 'ntendon': 0, 'nplugin': 0, 'nmocap': 0, 'na': 0,
    'nq': NQ, 'nv': NV, 'nu': NU, 'nM': NM,
    'nbody': NBODY, 'ngeom': NGEOM, 'nsite': NSITE, 'njnt': NJNT,
    'solver': SOLVER, 'cone': CONE, 'jacobian': JACOBIAN,
    'timestep': TIMESTEP, 'impratio': IMPRATIO,
    'disableflags': DISABLEFLAGS, 'enableflags': ENABLEFLAGS,
    'noslip_iterations': NOSLIP,
}

STATIC_SCHEMA = {
    'dof_frictionloss': ((NV,), np.float64),
    'jnt_type': ((NJNT,), np.int32),
    'jnt_limited': ((NJNT,), np.uint8),
    'jnt_qposadr': ((NJNT,), np.int32),
    'jnt_dofadr': ((NJNT,), np.int32),
    'jnt_range': ((NJNT, 2), np.float64),
    'jnt_margin': ((NJNT,), np.float64),
    'dof_solref': ((NV, 2), np.float64),
    'dof_solimp': ((NV, 5), np.float64),
    'jnt_solref': ((NJNT, 2), np.float64),
    'jnt_solimp': ((NJNT, 5), np.float64),
    'geom_bodyid': ((NGEOM,), np.int32),
    'body_invweight0': ((NBODY, 2), np.float64),
    'dof_invweight0': ((NV,), np.float64),
    'body_rootid': ((NBODY,), np.int32),
    'body_parentid': ((NBODY,), np.int32),
    'body_dofnum': ((NBODY,), np.int32),
    'body_dofadr': ((NBODY,), np.int32),
    'dof_parentid': ((NV,), np.int32),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


class UnsupportedStatic(ValueError):
    """Static identity differs from the frozen fixed model."""


def validate_fixed_static(static: dict) -> dict:
    """Validate and freeze the complete static read set.

    Returns the checked arrays (original dtypes preserved).  Any mismatch in
    field set, shape, dtype, digest or the structural counts is rejected; in
    particular a model with ``neq``/``ntendon`` materialized anywhere other
    than the sealed zero proof cannot enter the chain.
    """
    if set(static) != set(STATIC_SCHEMA):
        raise UnsupportedStatic('static field set')
    checked = {}
    for key, (shape, dtype) in STATIC_SCHEMA.items():
        raw = static[key]
        arr = np.asarray(raw)
        if arr.shape != shape or arr.dtype != np.dtype(dtype):
            raise UnsupportedStatic(f'{key} schema {arr.shape}/{arr.dtype} != {shape}/{np.dtype(dtype)}')
        arr = np.ascontiguousarray(arr)
        expected = FIXED_STATIC_SHA.get(key)
        if expected is not None and array_sha256(arr) != expected:
            raise UnsupportedStatic(f'{key} digest')
        checked[key] = arr
    # Structural proof of the fixed model: 9 positive friction dofs, 13
    # limited slide/hinge joints, no ball-limited joints.
    if int(np.count_nonzero(checked['dof_frictionloss'] > 0)) != NF_FRICTION_DOF:
        raise UnsupportedStatic('friction dof count')
    if int(np.count_nonzero(checked['jnt_limited'])) != NLIMIT_JOINTS:
        raise UnsupportedStatic('limited joint count')
    limited_types = checked['jnt_type'][checked['jnt_limited'] != 0]
    if np.any(~np.isin(limited_types, [2, 3])):
        raise UnsupportedStatic('limited joint type')
    for key in ('dof_frictionloss', 'jnt_range', 'jnt_margin',
                'dof_solref', 'dof_solimp', 'jnt_solref', 'jnt_solimp',
                'body_invweight0', 'dof_invweight0'):
        if np.any(~np.isfinite(checked[key])):
            raise UnsupportedStatic(f'{key} nonfinite')
    if (np.any(checked['jnt_qposadr'] < 0) or np.any(checked['jnt_qposadr'] >= NQ) or
            np.any(checked['jnt_dofadr'] < 0) or np.any(checked['jnt_dofadr'] >= NV)):
        raise UnsupportedStatic('joint address')
    return checked


def validate_model_facts(facts: dict) -> dict:
    """Gate model-level facts that are not carried by the static arrays.

    Rejects any drift in neq/ntendon (equality/tendon must be structurally
    zero here, never silently dropped) and in the option domain the kernels
    assume (primal Newton, elliptic cone, dense-AUTO jacobian, fixed
    timestep/impratio/flags).
    """
    if set(facts) != set(MODEL_FACTS):
        raise UnsupportedStatic(f'model facts field set {sorted(facts)}')
    for key, want in MODEL_FACTS.items():
        got = facts[key]
        if isinstance(want, float):
            if float(got) != want:
                raise UnsupportedStatic(f'model fact {key}: {got} != {want}')
        elif int(got) != int(want):
            raise UnsupportedStatic(f'model fact {key}: {got} != {want}')
    return dict(facts)


EQUALITY_TENDON_STATIC_PROOF = {
    'model_sha256': MJB_SHA256,
    'neq': NEQ,
    'ntendon': NTENDON,
    'statement': (
        'neq=0 and ntendon=0 are read from the frozen MJB. Source '
        'mj_instantiateEquality iterates m->neq (0 iterations) and the tendon '
        'friction/limit loops iterate m->ntendon (0 iterations), so equality '
        'and tendon row contributions are structurally zero. The fixed static '
        'gate rejects any other model; nonzero equality/tendon is never '
        'silently ignored.'
    ),
    'sources': {
        'engine_core_constraint.c': ENGINE_CORE_CONSTRAINT_SHA256,
        'functions': [
            'mj_makeConstraint', 'mj_instantiateEquality',
            'mj_instantiateFriction', 'mj_instantiateLimit',
        ],
    },
}


def primal_noop_proof() -> dict:
    """Static proof that mj_projectConstraint is a primal no-op here.

    ``mj_isDual`` is true only for PGS or noslip_iterations>0.  The frozen
    model has solver=Newton(2) and noslip_iterations=0, so ``mj_isDual`` is
    false and ``mj_projectConstraint`` returns before the arena allocation and
    before any ``efc_AR`` write.
    """
    return {
        'solver': SOLVER,
        'solver_name': 'Newton',
        'noslip_iterations': NOSLIP,
        'mj_isDual': False,
        'conclusion': (
            'mj_projectConstraint returns at `if (nefc == 0 || !mj_isDual(m))` '
            'with no efc_AR write; efc_AR stays on its previous bytes. '
            'This package never allocates or writes an efc_AR buffer.'
        ),
        'source_sha256': ENGINE_CORE_CONSTRAINT_SHA256,
        'source_functions': ['mj_isDual', 'mj_projectConstraint'],
    }
