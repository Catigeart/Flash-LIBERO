"""Fixed-model staged rules for the P06-D dynamics entries.

The rules record, per MuJoCo 2.3.7 forward stage, which original segments run
for the frozen Panda model and which are structural no-ops.  They are proven
from the original MJB topology at prepare time, not assumed.  Source order is
engine_forward.c d8957dc8c2b2aa7ac690ab56e7425deb7632e5df40c609cf9914ab44553c51d5
(Apache-2.0 MuJoCo 2.3.7).

Stage-to-entry mapping (source order preserved across entries):

  launch_position       kinematics, comPos, camlight(reg), tendon(empty),
                        crb/factorM, [collision -> P06-C], [makeConstraint ->
                        P06-S], transmission(length/moment),
                        [projectConstraint -> P06-N]
  launch_velocity_prefix  tendonVel(empty), actuator_velocity, comVel,
                        passive          (S.reference inserts after this entry)
  launch_bias           rneBias
  launch_control        controller OSC update (post-forward boundary)
  launch_actuation      fwdActuation (force/qfrc_actuator)
  launch_acceleration   fwdAcceleration: three-RN smooth, xfrc accumulate,
                        solveM
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np

# Original frozen MJB identity (K corpus task_model.mjb).
FROZEN_MJB_SHA256 = '3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42'
ENGINE_FORWARD_SHA256 = 'd8957dc8c2b2aa7ac690ab56e7425deb7632e5df40c609cf9914ab44553c51d5'

# Fixed topology recorded in the design contracts (CPU_FLOW, SHARED_CONTRACT).
FIXED_DIMS = {
    'nq': 48, 'nv': 43, 'nu': 9, 'na': 0, 'nM': 153,
    'nbody': 38, 'ngeom': 270, 'nsite': 31, 'njnt': 18,
    'ntendon': 0, 'neq': 0, 'nplugin': 0, 'nmocap': 0,
}

# The nine fixed Panda actuators are scalar-joint transmissions (trntype=0),
# constant gear/gain, affine bias on the two grippers only (accepted actuation
# kernel contract).  Joint types: 7 hinges (3) + 2 slides (2) for actuators 0-8.
ACTUATOR_JOINT_TYPE = (3, 3, 3, 3, 3, 3, 3, 2, 2)

# Source-stage map: ordered segments, entry owner, and fixed-model verdict.
# verdict: 'produces' (entry computes), 'structural_zero' (proven empty),
# 'registered_foreign' (owned by C/S/N/I, boundaries documented),
# 'deferred' (outside this task, explicit not_run).
STAGE_MAP = (
    # (segment, source, entry, verdict, note)
    ('mj_kinematics', 'engine_forward.c:100', 'launch_position', 'produces',
     'xpos/xquat/xmat/xanchor/xaxis via accepted body_kinematics'),
    ('mj_comPos', 'engine_forward.c:101', 'launch_position', 'produces',
     'subtree_com/cinert/cdof via accepted compos'),
    ('mj_camlight', 'engine_forward.c:102', 'launch_position', 'registered_foreign',
     'D01 module owns camera/light; no dynamics outputs'),
    ('mj_tendon', 'engine_forward.c:103', 'launch_position', 'structural_zero',
     'ntendon=0: ten_* arrays zero length, no writes'),
    ('mj_crb', 'engine_forward.c:107', 'launch_position', 'produces',
     'cinert combine via accepted mass_chain crb stage'),
    ('mj_factorM', 'engine_forward.c:108', 'launch_position', 'produces',
     'qM->qLD/qLDiagInv packed parent/Madr order via accepted mass_chain'),
    ('mj_collision', 'engine_forward.c:112', 'launch_position', 'registered_foreign',
     'P06-C owns collision; interleaves before makeConstraint'),
    ('mj_makeConstraint', 'engine_forward.c:116', 'launch_position', 'registered_foreign',
     'P06-S owns constraint counts/rows'),
    ('mj_transmission', 'engine_forward.c:117', 'launch_position', 'produces',
     'actuator_length/moment: 9 scalar-joint transmissions'),
    ('mj_projectConstraint', 'engine_forward.c:121', 'launch_position', 'registered_foreign',
     'primal Newton: original no-op, old AR domain preserved (P06-N)'),
    ('mj_tendonVel', 'engine_forward.c:131-140', 'launch_velocity_prefix', 'structural_zero',
     'ntendon=0: ten_velocity zero length'),
    ('actuator_velocity', 'engine_forward.c:142', 'launch_velocity_prefix', 'produces',
     'actuator_velocity = actuator_moment @ qvel (mju_mulMatVec)'),
    ('mj_comVel', 'engine_forward.c:145', 'launch_velocity_prefix', 'produces',
     'cvel/cdof_dot via accepted comvel'),
    ('mj_passive', 'engine_forward.c:146', 'launch_velocity_prefix', 'produces',
     'joint/DOF seed + fluid suffix via accepted passive_chain'),
    ('mj_referenceConstraint', 'engine_forward.c:147', 'launch_bias', 'registered_foreign',
     'S.reference at the correct velocity position: P06-S owns efc_vel/aref'),
    ('mj_rne', 'engine_forward.c:150', 'launch_bias', 'produces',
     'qfrc_bias via accepted rne_bias'),
    ('mj_fwdActuation', 'engine_forward.c:158-358', 'launch_actuation', 'produces',
     'actuator_force/qfrc_actuator with ctrl clamp/bad-ctrl warning'),
    ('mj_fwdAcceleration.three_rn', 'engine_forward.c:370-372', 'launch_acceleration',
     'produces', 'mju_sub then two mju_addTo: three distinct rounding nodes'),
    ('mj_fwdAcceleration.xfrc', 'engine_support.c:1174-1216', 'launch_acceleration',
     'produces', 'D02 seeded per-body force-then-torque accumulation'),
    ('mj_fwdAcceleration.solveM', 'engine_forward.c:376', 'launch_acceleration',
     'produces', 'accepted N06 factor/solve -> qacc_smooth'),
    ('sensorPos/sensorVel/sensorAcc', 'engine_forward.c:799,810,826;engine_sensor.c', '-', 'deferred',
     'D03 post-constraint sensors depend on P06-N; full sensors in P06-I'),
    ('fwdConstraint/Newton', 'engine_solver.c', '-', 'deferred', 'P06-N owns Newton'),
    ('Euler/integration', 'engine_forward.c:557-609', '-', 'deferred', 'P06-I owns integrator'),
)

FIELD_GROUPS = {
    'position': (
        'xpos', 'xquat', 'xmat', 'xanchor', 'xaxis', 'xipos', 'ximat',
        'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat',
        'subtree_com', 'cinert', 'cdof', 'qm', 'ld', 'inv', 'sqrt_inv',
        'actuator_length', 'actuator_moment',
    ),
    'velocity_prefix': ('actuator_velocity', 'cvel', 'cdof_dot', 'qfrc_passive'),
    'bias': ('qfrc_bias',),
    'control': ('jacp', 'jacr', 'cache_M', 'cache_J', 'cache_pose', 'cache_vel',
                'live_bias', 'goal_pos', 'goal_ori', 'gripper', 'ctrl'),
    'actuation': ('actuator_force', 'qfrc_actuator'),
    'acceleration': ('qfrc_smooth', 'qacc_smooth'),
}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class FixedModelRules:
    """Topology-derived rules for one frozen model; produced by load_rules."""

    SCHEMA = 'phase06-dynamics-fixed-model-rules-v1'

    def __init__(self, dims: dict, actuator_joint_type, mjb_sha256: str,
                 stage_map=STAGE_MAP):
        self.dims = {k: int(v) for k, v in dims.items()}
        self.actuator_joint_type = tuple(int(v) for v in actuator_joint_type)
        self.mjb_sha256 = str(mjb_sha256)
        self.stage_map = tuple(stage_map)
        self._validate()

    def _validate(self):
        for key, expected in FIXED_DIMS.items():
            if self.dims.get(key) != expected:
                raise ValueError(f'fixed model rule violation: {key}='
                                 f'{self.dims.get(key)} expected {expected}')
        if self.mjb_sha256 != FROZEN_MJB_SHA256:
            raise ValueError('fixed model rule violation: MJB SHA')
        if self.actuator_joint_type != ACTUATOR_JOINT_TYPE:
            raise ValueError('fixed model rule violation: actuator joint types')
        if len({row[0] for row in self.stage_map}) != len(self.stage_map):
            raise ValueError('duplicate stage-map segment')

    @property
    def empty_tendon(self):
        return self.dims['ntendon'] == 0

    @property
    def empty_equality(self):
        return self.dims['neq'] == 0

    @property
    def no_mocap(self):
        return self.dims['nmocap'] == 0

    def segments_for(self, entry):
        return tuple(row for row in self.stage_map if row[2] == entry)

    def to_json(self):
        return json.dumps({
            'schema': self.SCHEMA,
            'dims': self.dims,
            'actuator_joint_type': self.actuator_joint_type,
            'mjb_sha256': self.mjb_sha256,
            'engine_forward_sha256': ENGINE_FORWARD_SHA256,
            'stage_map': [list(row) for row in self.stage_map],
            'field_groups': {k: list(v) for k, v in FIELD_GROUPS.items()},
        }, indent=2, sort_keys=True)


def load_rules(mjb_path, model=None) -> FixedModelRules:
    """Build rules from the actual frozen model, checking every fixed value.

    `model` may be a pre-loaded mujoco.MjModel; otherwise the MJB is loaded
    from `mjb_path` and its file SHA is bound.
    """
    mjb_sha = _sha256_file(mjb_path)
    if model is None:
        import mujoco
        model = mujoco.MjModel.from_binary_path(str(mjb_path))
    dims = {
        'nq': int(model.nq), 'nv': int(model.nv), 'nu': int(model.nu),
        'na': int(model.na), 'nM': int(model.nM), 'nbody': int(model.nbody),
        'ngeom': int(model.ngeom), 'nsite': int(model.nsite),
        'njnt': int(model.njnt), 'ntendon': int(model.ntendon),
        'neq': int(model.neq), 'nplugin': int(model.nplugin),
        'nmocap': int(model.nmocap),
        'nsensordata': int(model.nsensordata),
    }
    joint_types = tuple(int(model.jnt_type[int(j)]) for j in
                        np.asarray(model.actuator_trnid[:, 0]))
    return FixedModelRules(dims, joint_types, mjb_sha)
