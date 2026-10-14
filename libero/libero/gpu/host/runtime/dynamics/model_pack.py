"""SHA-bound model pack for the P06-D staged workspace.

Loads the frozen Panda MJB plus the accepted static exports, verifying every
original SHA before any mathematics uses them.  Paths to the migrated cold
records are runtime bindings (this machine), while the SHA identities are the
original frozen values and must not change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FROZEN_MJB_SHA256 = '3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42'
FROZEN_SPEC_SHA256 = '21c54ef50cf43e6c7e986e5b13163bd6b46fc7038f5eaf54771f625083329a9c'
FROZEN_FIELD_SHA256 = 'cb64df28b2a92bf5e2771cae5f8077118b6a8ddc687c0b7bf699f326abbaef10'
FROZEN_READSET_SHA256 = '875b705684cc92976c92b3ab49810fd7048e6ecbb70d78ca2f384248966cb483'
# tests/gpu_velocity_bias_chain/model_real.npz (tracked in the original repo)
VELOCITY_BIAS_MODEL_SHA256 = '745a77cae16c461e0cab5576249231f8d54e310de721d4d7fa423bcb6186f48c'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class ModelPack:
    """One frozen model's CPU-side arrays, each bound to an original SHA."""

    SCHEMA = 'phase06-dynamics-model-pack-v1'

    def __init__(self, *, mjb_path, frozen_spec_path, frozen_manifest_path,
                 velocity_bias_path, mjb_model=None):
        self.mjb_path = Path(mjb_path)
        self.frozen_spec_path = Path(frozen_spec_path)
        self.frozen_manifest_path = Path(frozen_manifest_path)
        self.velocity_bias_path = Path(velocity_bias_path)
        self.mjb_sha256 = sha256_file(self.mjb_path)
        self.frozen_spec_sha256 = sha256_file(self.frozen_spec_path)
        self.velocity_bias_sha256 = sha256_file(self.velocity_bias_path)
        if self.mjb_sha256 != FROZEN_MJB_SHA256:
            raise ValueError('MJB SHA mismatch: ' + self.mjb_sha256)
        if self.frozen_spec_sha256 != FROZEN_SPEC_SHA256:
            raise ValueError('frozen kinematics spec SHA mismatch: '
                             + self.frozen_spec_sha256)
        if self.velocity_bias_sha256 != VELOCITY_BIAS_MODEL_SHA256:
            raise ValueError('velocity/bias model SHA mismatch: '
                             + self.velocity_bias_sha256)
        self.frozen_manifest = json.loads(self.frozen_manifest_path.read_text())
        real = self.frozen_manifest['real']
        for key, expected in (('file_sha256', FROZEN_MJB_SHA256),
                              ('field_sha256', FROZEN_FIELD_SHA256),
                              ('spec_sha256', FROZEN_SPEC_SHA256)):
            if real[key] != expected:
                raise ValueError('frozen manifest ' + key + ' mismatch')
        if mjb_model is None:
            import mujoco
            mjb_model = mujoco.MjModel.from_binary_path(str(self.mjb_path))
        self.model = mjb_model
        with np.load(self.frozen_spec_path, allow_pickle=False) as z:
            self.frozen_spec = {name: np.array(z[name], copy=True) for name in z.files}
        with np.load(self.velocity_bias_path, allow_pickle=False) as z:
            self.velocity_bias_fields = {name: np.array(z[name], copy=True)
                                         for name in z.files}
        self._check_topology()

    def _check_topology(self):
        m = self.model
        dims = (int(m.nq), int(m.nv), int(m.nu), int(m.nM), int(m.nbody),
                int(m.ngeom), int(m.nsite), int(m.njnt), int(m.ntendon),
                int(m.neq), int(m.na), int(m.nmocap), int(m.nplugin))
        expected = (48, 43, 9, 153, 38, 270, 31, 18, 0, 0, 0, 0, 0)
        if dims != expected:
            raise ValueError(f'fixed topology mismatch: {dims} != {expected}')

    def frozen(self):
        from libero.gpu.reference.position.body_kinematics.workspace import FrozenModel
        return FrozenModel(self.frozen_spec_path, self.frozen_manifest, 'real')

    def flat_model(self):
        """Flat-attribute adapter over MjModel for the accepted physics chains.

        The accepted actuation/passive validators read original flat names
        (opt_disableflags, opt_wind, ...).  This view maps the nested MjModel
        option struct without copying any array twice; every array attribute
        is a fresh contiguous copy for device upload.
        """
        m = self.model

        class _Flat:
            pass

        f = _Flat()
        for name in dir(m):
            if name.startswith('_'):
                continue
            try:
                value = getattr(m, name)
            except Exception:
                continue
            if isinstance(value, np.ndarray):
                setattr(f, name, np.array(value, copy=True))
        for name in ('nq', 'nv', 'nu', 'na', 'nM', 'nbody', 'ngeom', 'nsite',
                     'njnt', 'ntendon', 'neq', 'nplugin', 'nmocap'):
            setattr(f, name, int(getattr(m, name)))
        f.opt_disableflags = int(m.opt.disableflags)
        f.opt_viscosity = np.float64(m.opt.viscosity)
        f.opt_density = np.float64(m.opt.density)
        f.opt_wind = np.array(m.opt.wind, copy=True)
        f.opt_gravity = np.array(m.opt.gravity, copy=True)
        return f

    def mass_model(self):
        """SimpleNamespace for MassChainWorkspace (accepted MASS_FIELDS)."""
        m = self.model
        fields = {}
        for name in ('body_mass', 'body_subtreemass', 'body_inertia',
                     'body_parentid', 'body_rootid', 'jnt_type', 'jnt_bodyid',
                     'jnt_dofadr', 'dof_bodyid', 'dof_parentid', 'dof_Madr',
                     'dof_simplenum', 'dof_M0', 'dof_armature'):
            fields[name] = np.array(getattr(m, name), copy=True)
        fields.update({name: np.array(getattr(m, name), dtype=np.int32) for name in
                       ('nbody', 'njnt', 'nq', 'nv', 'nM', 'nmocap')})
        return SimpleNamespace(**fields)

    def frame_model(self):
        """Exact MODEL_FIELDS dict for the accepted FrameWorkspace."""
        m = self.model
        return {
            'body_ipos': np.ascontiguousarray(m.body_ipos, dtype=np.float64),
            'body_iquat': np.ascontiguousarray(m.body_iquat, dtype=np.float64),
            'body_sameframe': np.ascontiguousarray(m.body_sameframe, dtype=np.int32),
            'geom_bodyid': np.ascontiguousarray(m.geom_bodyid, dtype=np.int32),
            'geom_pos': np.ascontiguousarray(m.geom_pos, dtype=np.float64),
            'geom_quat': np.ascontiguousarray(m.geom_quat, dtype=np.float64),
            'geom_sameframe': np.ascontiguousarray(m.geom_sameframe, dtype=np.int32),
            'site_bodyid': np.ascontiguousarray(m.site_bodyid, dtype=np.int32),
            'site_pos': np.ascontiguousarray(m.site_pos, dtype=np.float64),
            'site_quat': np.ascontiguousarray(m.site_quat, dtype=np.float64),
            'site_sameframe': np.ascontiguousarray(m.site_sameframe, dtype=np.int32),
        }

    def solve_model(self):
        """Namespace for accepted MassSolveWorkspace."""
        m = self.model
        return SimpleNamespace(
            nq=int(m.nq), nv=int(m.nv), nM=int(m.nM), nmocap=int(m.nmocap),
            dof_Madr=np.array(m.dof_Madr, copy=True),
            dof_parentid=np.array(m.dof_parentid, copy=True),
            dof_simplenum=np.array(m.dof_simplenum, copy=True),
        )

    def osc_fields(self):
        """Fields for the accepted D07 ChainWorkspace (StaticModel + OSC)."""
        m = self.model
        scalar = {name: np.array(value, dtype=np.int32) for name, value in
                  (('nq', 48), ('nv', 43), ('nM', 153), ('nsite', 31),
                   ('eef_site_id', 13))}
        arrays = {
            'qpos_index': np.arange(7, dtype=np.int32),
            'qvel_index': np.arange(7, dtype=np.int32),
            'dof_parentid': np.array(m.dof_parentid, dtype=np.int32, copy=True),
            'dof_Madr': np.array(m.dof_Madr, dtype=np.int32, copy=True),
            'nbody': np.array(int(m.nbody), dtype=np.int32),
            'body_rootid': np.array(m.body_rootid, dtype=np.int32, copy=True),
            'body_parentid': np.array(m.body_parentid, dtype=np.int32, copy=True),
            'body_dofnum': np.array(m.body_dofnum, dtype=np.int32, copy=True),
            'body_dofadr': np.array(m.body_dofadr, dtype=np.int32, copy=True),
            'site_bodyid': np.array(m.site_bodyid, dtype=np.int32, copy=True),
        }
        return {**scalar, **arrays}

    def identity(self):
        return {
            'schema': self.SCHEMA,
            'mjb_path': str(self.mjb_path),
            'mjb_sha256': self.mjb_sha256,
            'frozen_spec_path': str(self.frozen_spec_path),
            'frozen_spec_sha256': self.frozen_spec_sha256,
            'frozen_manifest_path': str(self.frozen_manifest_path),
            'velocity_bias_path': str(self.velocity_bias_path),
            'velocity_bias_sha256': self.velocity_bias_sha256,
        }
