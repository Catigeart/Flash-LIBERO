"""P06-D author self-test battery (developer_self_test_only only).

R02 comparator repairs (findings F05/F06): every decisive assertion runs
unconditionally -- a non-success status FAILS the section instead of silently
skipping its bitwise check; literal-True checks are gone; stamp checks run at
state_epoch > 0; skip checks cover all seven cache fields under a
discriminating condition; sticky and repair(25) routes are exercised; the
contamination counterexample really poisons an existing device buffer and the
input witness plus comparator catch it.

Usage (from the worktree root, after sourcing RUNTIME_ENV_R02.sh):

  COMPUTE_ONCE.sh <python> scripts/gpu_phase06_dynamics/run_author.py \
      --output doc/results/phase_06/D/<run> --capacities 1 4 20 128
"""
from __future__ import annotations
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = (REPO_ROOT / '.')
sys.path.insert(0, str((REPO_ROOT / 'libero')))
sys.path.insert(0, str(ROOT))

COLD = Path('/srv/libero-split-receive-20261004-r01/received-r01/evidence'
            '/records/by-origin')
MJB_PATH = (COLD / 'development-runs/kinematics/doc/results/phase_02k0'
            '/kinematics_20260929_03/task_model.mjb')
FROZEN_SPEC = (COLD / 'development-runs/phase04-body-kinematics/doc/results'
               '/phase_04_body_kinematics/preflight_20260930/model_real.npz')
FROZEN_MANIFEST = FROZEN_SPEC.parent / 'model_manifest.json'
VB_MODEL = (REPO_ROOT / 'tests/gpu_velocity_bias_chain/model_real.npz')
ENGINE_FORWARD_RECORD = (COLD / 'temporary-runs/libero-mujoco-2.3.7-source'
                         '/src/engine/engine_forward.c')

FROZEN_SHA = {
    'mjb': '3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42',
    'frozen_spec': '21c54ef50cf43e6c7e986e5b13163bd6b46fc7038f5eaf54771f625083329a9c',
    'vb_model': '745a77cae16c461e0cab5576249231f8d54e310de721d4d7fa423bcb6186f48c',
    'engine_forward': 'd8957dc8c2b2aa7ac690ab56e7425deb7632e5df40c609cf9914ab44553c51d5',
}


def bits(x):
    return struct.unpack('>Q', struct.pack('>d', float(x)))[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Report:
    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.sections = []
        self.t0 = time.time()

    def section(self, name):
        return Section(self, name)


class Section:
    def __init__(self, report: Report, name: str):
        self.report = report
        self.name = name
        self.checks = []
        self.first_diff = None
        self.error = None
        self.status = 'running'
        self.raw = {}

    def check(self, label, ok, detail=''):
        self.checks.append({'label': label, 'ok': bool(ok), 'detail': str(detail)})
        if not ok and self.first_diff is None:
            self.first_diff = {'label': label, 'detail': str(detail)}
        return ok

    def fail(self, exc):
        self.error = ''.join(traceback.format_exception(exc))
        self.status = 'error'

    def done(self):
        if self.status == 'running':
            self.status = 'passed' if all(c['ok'] for c in self.checks) else 'failed'
        entry = {'name': self.name, 'status': self.status,
                 'checks': self.checks, 'first_diff': self.first_diff,
                 'error': self.error, 'raw': self.raw}
        self.report.sections.append(entry)
        print(f"[{self.status}] {self.name}: "
              f"{sum(c['ok'] for c in self.checks)}/{len(self.checks)} checks"
              + (f" first_diff={self.first_diff}" if self.first_diff else ''))
        return entry


def load_pack():
    from libero.gpu.host.runtime.dynamics.model_pack import ModelPack
    return ModelPack(mjb_path=MJB_PATH, frozen_spec_path=FROZEN_SPEC,
                     frozen_manifest_path=FROZEN_MANIFEST,
                     velocity_bias_path=VB_MODEL)


def cpu_three_rn(passive, bias, applied, actuator):
    t0 = passive - bias
    t1 = t0 + applied
    t2 = t1 + actuator
    return t2


def make_qpos(rng, model):
    qpos = np.array(model.qpos0, copy=True)
    qpos[:7] += rng.normal(scale=0.25, size=7)
    qpos[7:9] = rng.uniform(0.0, 0.04, size=2)
    return qpos


def bitwise_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _historical_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--capacities', type=int, nargs='+', default=[1, 4, 20, 128])
    ap.add_argument('--seed', type=int, default=4401)
    args = ap.parse_args()
    report = Report(args.output)

    import mujoco
    import warp as wp

    # ------------------------------------------------------------------
    s = report.section('environment_and_bindings')
    try:
        # Independent file hashes first; module constants cannot self-certify.
        for label, path, expected in (
                ('mjb', MJB_PATH, FROZEN_SHA['mjb']),
                ('frozen_spec', FROZEN_SPEC, FROZEN_SHA['frozen_spec']),
                ('vb_model', VB_MODEL, FROZEN_SHA['vb_model']),
                ('engine_forward_record', ENGINE_FORWARD_RECORD,
                 FROZEN_SHA['engine_forward'])):
            got = sha256_file(path)
            s.check(f'{label} file sha256', got == expected,
                    f'{got} vs {expected}')
        pack = load_pack()
        s.raw['model_pack'] = pack.identity()
        m = pack.model
        dims = (int(m.nq), int(m.nv), int(m.nu), int(m.nM), int(m.nbody),
                int(m.ngeom), int(m.nsite), int(m.ntendon), int(m.neq),
                int(m.na), int(m.nmocap))
        s.check('model dims from loaded MJB', dims ==
                (48, 43, 9, 153, 38, 270, 31, 0, 0, 0, 0), dims)
        from libero.gpu.host.runtime.dynamics.rules import load_rules
        rules = load_rules(MJB_PATH, model=pack.model)
        s.raw['rules'] = json.loads(rules.to_json())
        s.check('rules bind frozen dims and zero tendon',
                rules.dims['ntendon'] == 0 and rules.dims['nv'] == 43)
        s.raw['runtime'] = {
            'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'mujoco': mujoco.__version__,
            'warp': getattr(wp.config, 'version', 'unknown'),
            'device': str(wp.get_device('cuda:0')),
        }
    except Exception as e:
        s.fail(e)
    s.done()

    pack = load_pack()
    model = pack.model

    # ------------------------------------------------------------------
    s = report.section('three_rn_gpu_vs_cpu')
    try:
        from libero.gpu.host.runtime.dynamics.acceleration import SmoothXfrcSolveWorkspace, StaticTopology
        rng = np.random.default_rng(args.seed)
        topology = StaticTopology({
            name: np.asarray(getattr(model, name), dtype=np.int32)
            for name in ('body_parentid', 'body_rootid', 'body_dofadr',
                         'body_dofnum', 'dof_parentid')})
        acc = SmoothXfrcSolveWorkspace(topology, pack.solve_model(), 4,
                                       device='cuda:0')
        cases = []
        for i in range(4):
            passive = rng.normal(size=(1, 43)) * 10.0 ** rng.integers(-3, 6)
            bias = rng.normal(size=(1, 43)) * 10.0 ** rng.integers(-3, 6)
            applied = rng.normal(size=(1, 43)) * 10.0 ** rng.integers(-3, 6)
            actuator = rng.normal(size=(1, 43)) * 10.0 ** rng.integers(-3, 6)
            cases.append((passive, bias, applied, actuator))
        cases[0][0][0, 0] = 1.0 + 2.0 ** 52
        cases[0][1][0, 0] = 2.0 ** 52
        cases[0][2][0, 0] = -1.0
        cases[0][3][0, 0] = 0.5
        zeros = np.zeros((4, 38, 6))
        xipos = np.tile(np.arange(38 * 3, dtype=np.float64).reshape(38, 3),
                        (4, 1, 1)) * 1e-3
        sub = np.zeros((4, 38, 3))
        cdof = rng.normal(size=(4, 43, 6))
        qm = np.zeros((4, 153))
        for w in range(4):
            for k in range(43):
                qm[w, model.dof_Madr[k]] = 2.0 + 0.01 * k
        acc.upload(active=np.ones(4, np.int32),
                   passive=np.concatenate([c[0] for c in cases]),
                   bias=np.concatenate([c[1] for c in cases]),
                   applied=np.concatenate([c[2] for c in cases]),
                   actuator=np.concatenate([c[3] for c in cases]),
                   xfrc=zeros, xipos=xipos, subtree_com=sub, cdof=cdof, qm=qm)
        acc.launch()
        out = acc.download()
        s.raw['status'] = out['status'].tolist()
        # Decisive: the status gate runs the bitwise check on success and
        # FAILS the section on any other status (F05-1).
        s.check('status success', bool(np.all(out['status'] == 0)),
                out['status'].tolist())
        ok_all = True
        for w in range(4):
            expect = cpu_three_rn(cases[w][0][0], cases[w][1][0],
                                  cases[w][2][0], cases[w][3][0])
            got = out['smooth'][w]
            for j in range(43):
                if bits(expect[j]) != bits(got[j]):
                    ok_all = s.check(f'w{w}d{j} bitwise', False,
                                     f'cpu={expect[j]!r} gpu={got[j]!r}') and ok_all
                    break
            else:
                continue
            break
        s.check('all worlds bitwise equal', ok_all)
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('acceleration_nonzero_xfrc_source_order')
    try:
        rng = np.random.default_rng(args.seed + 1)
        n = 1
        topology = StaticTopology({
            name: np.asarray(getattr(model, name), dtype=np.int32)
            for name in ('body_parentid', 'body_rootid', 'body_dofadr',
                         'body_dofnum', 'dof_parentid')})
        acc = SmoothXfrcSolveWorkspace(topology, pack.solve_model(), n,
                                       device='cuda:0')
        passive = rng.normal(size=(n, 43))
        bias = rng.normal(size=(n, 43))
        applied = rng.normal(size=(n, 43))
        actuator = rng.normal(size=(n, 43))
        xfrc = np.zeros((n, 38, 6))
        xfrc[0, 1, :] = rng.normal(size=6) * 1e3
        xfrc[0, 7, :] = rng.normal(size=6) * 1e3
        xfrc[0, 20, :] = rng.normal(size=6) * 1e3
        xipos = rng.normal(size=(n, 38, 3)) * 0.1
        sub = rng.normal(size=(n, 38, 3)) * 0.01
        cdof = rng.normal(size=(n, 43, 6))
        qm = np.zeros((n, 153))
        for k in range(43):
            qm[0, model.dof_Madr[k]] = 2.0 + 0.01 * k
        acc.upload(active=np.ones(n, np.int32), passive=passive, bias=bias,
                   applied=applied, actuator=actuator, xfrc=xfrc, xipos=xipos,
                   subtree_com=sub, cdof=cdof, qm=qm)
        acc.launch()
        out = acc.download()
        expect_smooth = cpu_three_rn(passive[0], bias[0], applied[0], actuator[0])
        nbody, nv = 38, 43
        ancestor = np.zeros((nbody, nv), np.int32)
        for body in range(1, nbody):
            cursor = body
            while cursor and model.body_dofnum[cursor] == 0:
                cursor = model.body_parentid[cursor]
            if not cursor:
                continue
            dof = int(model.body_dofadr[cursor] + model.body_dofnum[cursor] - 1)
            while dof >= 0:
                ancestor[body, dof] = 1
                dof = int(model.dof_parentid[dof])

        def accumulate(order):
            expect = expect_smooth.copy()
            for body in order:
                fx = xfrc[0, body]
                if not np.any(fx != 0.0):
                    continue
                root = int(model.body_rootid[body])
                ox = xipos[0, body, 0] - sub[0, root, 0]
                oy = xipos[0, body, 1] - sub[0, root, 1]
                oz = xipos[0, body, 2] - sub[0, root, 2]
                for j in range(nv):
                    jp0 = jp1 = jp2 = jr0 = jr1 = jr2 = 0.0
                    if ancestor[body, j]:
                        jr0, jr1, jr2 = (cdof[0, j, 0], cdof[0, j, 1],
                                         cdof[0, j, 2])
                        jp0 = cdof[0, j, 3] + (jr1 * oz - jr2 * oy)
                        jp1 = cdof[0, j, 4] + (jr2 * ox - jr0 * oz)
                        jp2 = cdof[0, j, 5] + (jr0 * oy - jr1 * ox)
                    force = 0.0
                    if fx[0] != 0.0:
                        force = force + jp0 * fx[0]
                    if fx[1] != 0.0:
                        force = force + jp1 * fx[1]
                    if fx[2] != 0.0:
                        force = force + jp2 * fx[2]
                    expect[j] = expect[j] + force
                    torque = 0.0
                    if fx[3] != 0.0:
                        torque = torque + jr0 * fx[3]
                    if fx[4] != 0.0:
                        torque = torque + jr1 * fx[4]
                    if fx[5] != 0.0:
                        torque = torque + jr2 * fx[5]
                    expect[j] = expect[j] + torque
            return expect

        expect_qfrc = accumulate(range(1, nbody))
        ok = True
        for j in range(nv):
            if bits(expect_qfrc[j]) != bits(out['qfrc_smooth'][0, j]):
                ok = s.check(f'qfrc_smooth d{j} bitwise', False,
                             f'cpu={expect_qfrc[j]!r} gpu={out["qfrc_smooth"][0, j]!r}') and ok
                break
        s.check('accumulated force bitwise (source order)', ok)
        s.check('status success', out['status'][0] == 0, int(out['status'][0]))
        s.check('smooth matches three-rn',
                all(bits(expect_smooth[j]) == bits(out['smooth'][0, j])
                    for j in range(nv)))
        # R02-F05-5: real order-sensitivity witness on a crafted association
        # case: seed 1.0 on dof 0, bodies 1/2/3 contributing 2^53, -2^53, 1
        # through unit jacobian columns.  Source order
        # (((1+2^53)-2^53)+1) = 1; reversed = 0; one-shot = 2.
        craft = dict(passive=np.zeros((1, 43)), bias=np.zeros((1, 43)),
                     applied=np.zeros((1, 43)), actuator=np.zeros((1, 43)),
                     xfrc=np.zeros((1, 38, 6)),
                     xipos=np.zeros((1, 38, 3)), subtree_com=np.zeros((1, 38, 3)),
                     cdof=np.zeros((1, 43, 6)), qm=np.zeros((1, 153)))
        craft['passive'][0, 0] = 1.0
        craft['cdof'][0, 0, 3] = 1.0
        for k in range(43):
            craft['qm'][0, model.dof_Madr[k]] = 2.0
        for slot, body in enumerate((1, 2, 3)):
            craft['xfrc'][0, body, 0] = (2.0 ** 53, -2.0 ** 53, 1.0)[slot]
        acc2 = SmoothXfrcSolveWorkspace(topology, pack.solve_model(), 1,
                                        device='cuda:0')
        acc2.upload(active=np.ones(1, np.int32), **craft)
        acc2.launch()
        craft_out = acc2.download()
        source_assoc = 1.0
        for value in (2.0 ** 53, -2.0 ** 53, 1.0):
            source_assoc = source_assoc + value
        reversed_assoc = 1.0
        for value in (1.0, -2.0 ** 53, 2.0 ** 53):
            reversed_assoc = reversed_assoc + value
        one_shot = 1.0 + ((2.0 ** 53 - 2.0 ** 53) + 1.0)
        s.check('crafted source-order matches GPU',
                bits(craft_out['qfrc_smooth'][0, 0]) == bits(source_assoc),
                (float(craft_out['qfrc_smooth'][0, 0]), source_assoc))
        s.check('crafted status success',
                craft_out['status'][0] == 0, int(craft_out['status'][0]))
        s.check('reversed body order differs (witness)',
                bits(reversed_assoc) != bits(source_assoc))
        s.check('one-shot sum differs (witness)',
                bits(one_shot) != bits(source_assoc))
        s.raw['status'] = out['status'].tolist()
        s.raw['crafted_status'] = craft_out['status'].tolist()
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('acceleration_cpu_mujoco_reference')
    try:
        m = model
        d = mujoco.MjData(m)
        rng = np.random.default_rng(args.seed + 2)
        d.qpos[:] = make_qpos(rng, m)
        d.qvel[:] = rng.normal(scale=0.5, size=43)
        d.qfrc_applied[:] = rng.normal(scale=1.0, size=43)
        d.xfrc_applied[1, :] = rng.normal(scale=1.0, size=6)
        d.xfrc_applied[5, :] = rng.normal(scale=1.0, size=6)
        d.xfrc_applied[23, :] = rng.normal(scale=1.0, size=6)
        mujoco.mj_forward(m, d)
        cpu_smooth = np.array(d.qfrc_smooth, copy=True)
        cpu_acc = np.array(d.qacc_smooth, copy=True)
        topology = StaticTopology({
            name: np.asarray(getattr(m, name), dtype=np.int32)
            for name in ('body_parentid', 'body_rootid', 'body_dofadr',
                         'body_dofnum', 'dof_parentid')})
        acc = SmoothXfrcSolveWorkspace(topology, pack.solve_model(), 1,
                                       device='cuda:0')
        qm = np.zeros((1, 153))
        qm[0] = np.array(d.qM, copy=True)
        acc.upload(active=np.ones(1, np.int32),
                   passive=np.array(d.qfrc_passive, copy=True)[None],
                   bias=np.array(d.qfrc_bias, copy=True)[None],
                   applied=np.array(d.qfrc_applied, copy=True)[None],
                   actuator=np.array(d.qfrc_actuator, copy=True)[None],
                   xfrc=np.array(d.xfrc_applied, copy=True)[None],
                   xipos=np.array(d.xipos, copy=True)[None],
                   subtree_com=np.array(d.subtree_com, copy=True)[None],
                   cdof=np.array(d.cdof, copy=True)[None], qm=qm)
        acc.launch()
        out = acc.download()
        status = int(out['status'][0])
        s.raw['status'] = status
        s.raw['cpu_smooth_sha'] = hashlib.sha256(cpu_smooth.tobytes()).hexdigest()
        s.raw['cpu_acc_sha'] = hashlib.sha256(cpu_acc.tobytes()).hexdigest()
        # Decisive and unconditional (F05-1): status must be 0 for the bitwise
        # claims to hold; anything else fails the section outright.
        s.check('status is success (0)', status == 0, status)
        ok_smooth = all(bits(out['qfrc_smooth'][0, j]) == bits(cpu_smooth[j])
                        for j in range(43))
        s.check('qfrc_smooth bitwise vs CPU MuJoCo', ok_smooth)
        ok_acc = all(bits(out['qacc_smooth'][0, j]) == bits(cpu_acc[j])
                     for j in range(43))
        s.check('qacc_smooth bitwise vs CPU MuJoCo', ok_acc)
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('staged_acceleration_vs_cpu')
    try:
        # R02-F09 acceptance: the STAGED pipeline (not a hand-upload of the
        # combinator) must produce qfrc_smooth/qacc_smooth consistent with
        # fresh CPU MuJoCo on the same fixed inputs.  Computed GPU producers
        # may differ from CPU by ULP-level rounding (review CE-12), so the
        # criterion here is bitwise where inputs isolate the force chain and a
        # recorded max-abs residual otherwise -- stated per field below.
        from libero.gpu.host.runtime.dynamics.stages import DynamicsStages
        ds = DynamicsStages(pack, 1, device='cuda:0')
        m = model
        d = mujoco.MjData(m)
        rng = np.random.default_rng(args.seed + 5)
        qpos = make_qpos(rng, m)[None]
        qvel = rng.normal(scale=0.3, size=(1, 43))
        qfrc_ap = rng.normal(size=(1, 43)) * 0.2
        xfrc_ap = np.zeros((1, 38, 6))
        xfrc_ap[0, 2] = rng.normal(size=6) * 0.5
        xfrc_ap[0, 11] = rng.normal(size=6) * 0.5
        ds.upload_state(active=np.ones(1, np.int32), qpos=qpos, qvel=qvel,
                        mocap_pos=np.zeros((1, 0, 3)),
                        mocap_quat=np.zeros((1, 0, 4)),
                        rhs=np.zeros((1, 4, 43)),
                        count=np.ones(1, np.int32),
                        qfrc_applied=qfrc_ap, xfrc_applied=xfrc_ap)
        ds.begin_forward()
        ds.launch_position()
        ds.launch_velocity_prefix()
        ds.launch_bias()
        ds.launch_control(force_refresh=True, policy_step=True)
        ds.launch_actuation()
        ds.launch_acceleration()
        snap = ds.witness.host_snapshot()
        acc_status = int(np.array(ds.acceleration.status.numpy())[0])
        acc_valid = int(np.array(ds.witness.valid['acceleration'].numpy())[0])
        s.raw['acc_status'] = acc_status
        s.raw['acc_valid'] = acc_valid
        s.check('acceleration status 0 (or documented 25 repair)',
                acc_status in (0, 25), acc_status)
        s.check('acceleration valid == 1', acc_valid == 1, acc_valid)
        gpu_smooth = np.array(ds.acceleration.qfrc.numpy())[0]
        gpu_acc = np.array(ds.acceleration.qacc_smooth.numpy())[0]
        # Bitwise combinator edge on the pipeline's OWN stage outputs: the
        # three-RN + source-order accumulation recomputed independently.
        g_passive = np.array(ds.qfrc_passive.numpy())[0]
        g_bias = np.array(ds.qfrc_bias.numpy())[0]
        g_applied = np.array(ds.qfrc_applied.numpy())[0]
        g_actuator = np.array(ds.actuation.qfrc_actuator.numpy())[0]
        g_xipos = np.array(ds.acceleration.d02.xipos.numpy())[0]
        g_com = np.array(ds.acceleration.d02.subtree_com.numpy())[0]
        g_cdof = np.array(ds.acceleration.d02.cdof.numpy())[0]
        expect = cpu_three_rn(g_passive, g_bias, g_applied, g_actuator)
        ancestor = np.zeros((38, 43), np.int32)
        for body in range(1, 38):
            cursor = body
            while cursor and model.body_dofnum[cursor] == 0:
                cursor = model.body_parentid[cursor]
            if not cursor:
                continue
            dof = int(model.body_dofadr[cursor] + model.body_dofnum[cursor] - 1)
            while dof >= 0:
                ancestor[body, dof] = 1
                dof = int(model.dof_parentid[dof])
        for body in range(1, 38):
            fx = xfrc_ap[0, body]
            if not np.any(fx != 0.0):
                continue
            root = int(model.body_rootid[body])
            ox = g_xipos[body, 0] - g_com[root, 0]
            oy = g_xipos[body, 1] - g_com[root, 1]
            oz = g_xipos[body, 2] - g_com[root, 2]
            for j in range(43):
                jp0 = jp1 = jp2 = jr0 = jr1 = jr2 = 0.0
                if ancestor[body, j]:
                    jr0, jr1, jr2 = g_cdof[j, 0], g_cdof[j, 1], g_cdof[j, 2]
                    jp0 = g_cdof[j, 3] + (jr1 * oz - jr2 * oy)
                    jp1 = g_cdof[j, 4] + (jr2 * ox - jr0 * oz)
                    jp2 = g_cdof[j, 5] + (jr0 * oy - jr1 * ox)
                force = 0.0
                if fx[0] != 0.0:
                    force = force + jp0 * fx[0]
                if fx[1] != 0.0:
                    force = force + jp1 * fx[1]
                if fx[2] != 0.0:
                    force = force + jp2 * fx[2]
                expect[j] = expect[j] + force
                torque = 0.0
                if fx[3] != 0.0:
                    torque = torque + jr0 * fx[3]
                if fx[4] != 0.0:
                    torque = torque + jr1 * fx[4]
                if fx[5] != 0.0:
                    torque = torque + jr2 * fx[5]
                expect[j] = expect[j] + torque
        ok_edge = all(bits(expect[j]) == bits(gpu_smooth[j]) for j in range(43))
        s.check('qfrc_smooth bitwise vs independent transcription on GPU outputs',
                ok_edge)
        # Bitwise solve edge: independent CPU dense solve of the GPU packed qM.
        packed_qm = np.array(ds.position.mass.mass.qm.numpy())[0]
        M = np.zeros((43, 43))
        mujoco.mj_fullM(m, M, packed_qm)
        dense_acc = np.linalg.solve(M, expect)
        ok_solve = all(bits(dense_acc[j]) == bits(gpu_acc[j]) for j in range(43))
        s.check('qacc_smooth bitwise vs independent dense solve (or ULP band)',
                ok_solve or float(np.max(np.abs(dense_acc - gpu_acc))) <= 1e-9,
                f'bitwise={ok_solve}')
        # Same-state, same-ctrl CPU MuJoCo reference: the fixed-input
        # comparison.  Computed GPU producers may differ from CPU by ULP-level
        # rounding (review CE-12), so the criterion is a recorded ULP band.
        ctrl_used = np.array(ds.control.ctrl.numpy(), copy=True)
        d.qpos[:] = qpos[0]
        d.qvel[:] = qvel[0]
        d.qfrc_applied[:] = qfrc_ap[0]
        d.xfrc_applied[:] = xfrc_ap[0]
        d.ctrl[:] = ctrl_used[0]
        mujoco.mj_forward(m, d)
        cpu_smooth = np.array(d.qfrc_smooth, copy=True)
        cpu_acc = np.array(d.qacc_smooth, copy=True)
        diff_smooth = float(np.max(np.abs(gpu_smooth - cpu_smooth)))
        diff_acc = float(np.max(np.abs(gpu_acc - cpu_acc)))
        s.raw['diff_smooth_maxabs'] = diff_smooth
        s.raw['diff_acc_maxabs'] = diff_acc
        s.check('qfrc_smooth within ULP band of same-ctrl CPU MuJoCo',
                diff_smooth <= 1e-9, diff_smooth)
        s.check('qacc_smooth within ULP band of same-ctrl CPU MuJoCo',
                diff_acc <= 1e-6, diff_acc)
        if acc_status == 25:
            s.check('repair exposes warning_count', int(
                np.array(ds.acceleration.solver.warning_count.numpy())[0]) >= 1)
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('transmission_split_vs_fused')
    try:
        from libero.gpu.host.runtime.dynamics.transmission import TransmissionWorkspace
        from libero.gpu.host.runtime.dynamics.actuation_split import ActuationForceWorkspace
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'p06d_fused_actuation', (GPU_ROOT / 'reference/dynamics/actuation/kernel.py'))
        fused_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fused_mod)
        rng = np.random.default_rng(args.seed + 3)
        n = 4
        qpos = np.stack([make_qpos(rng, model) for _ in range(n)])
        qvel = rng.normal(size=(n, 43))
        ctrl = rng.uniform(-1.0, 1.0, size=(n, 9)) * np.array(
            [80] * 5 + [12] * 2 + [.04] * 2)
        flat = pack.flat_model()
        tr = TransmissionWorkspace(flat, n, device='cuda:0')
        af = ActuationForceWorkspace(flat, n, device='cuda:0')
        tr.qpos.assign(qpos)
        tr.qvel.assign(qvel)
        tr.position_active.assign(np.ones(n, np.int32))
        tr.velocity_active.assign(np.ones(n, np.int32))
        af.active.assign(np.ones(n, np.int32))
        af.ctrl.assign(np.ascontiguousarray(ctrl))
        tr.launch_position()
        tr.launch_velocity()
        af.launch(tr.length, tr.moment, tr.velocity, tr.velocity_status)
        split = tr.download()
        force_out = {'force': np.array(af.force.numpy(), copy=True),
                     'qfrc': np.array(af.qfrc_actuator.numpy(), copy=True),
                     'status': np.array(af.status.numpy(), copy=True)}
        fused = fused_mod.ActuationWorkspace(flat, n, device='cuda:0')
        fused.qpos.assign(qpos)
        fused.qvel.assign(qvel)
        fused.ctrl.assign(np.ascontiguousarray(ctrl))
        fused.active.assign(np.ones(n, np.int32))
        fused.launch()
        ref = {name: np.array(getattr(fused, name).numpy(), copy=True)
               for name in ('length', 'moment', 'velocity', 'force', 'qfrc',
                            'status')}
        ok = True
        for name in ('length', 'moment', 'velocity'):
            ok = s.check(f'{name} bitwise vs fused',
                         bitwise_equal(split[name], ref[name])) and ok
        ok = s.check('force bitwise vs fused',
                     bitwise_equal(force_out['force'], ref['force'])) and ok
        ok = s.check('qfrc bitwise vs fused',
                     bitwise_equal(force_out['qfrc'], ref['qfrc'])) and ok
        ok = s.check('status equal',
                     bitwise_equal(force_out['status'], ref['status'])) and ok
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('control_stage_and_clock')
    try:
        # R02-F01/F04/F10 acceptance battery for the control entry.
        from libero.gpu.host.runtime.dynamics.stages import DynamicsStages
        from libero.gpu.reference.control.osc_actuation_chain.chain import OSCActuationChain
        n = 4
        ds = DynamicsStages(pack, n, device='cuda:0')
        rng = np.random.default_rng(args.seed + 6)
        qpos = np.stack([make_qpos(rng, model) for _ in range(n)])
        qvel = rng.normal(scale=0.3, size=(n, 43))
        qfrc_ap = rng.normal(size=(n, 43)) * 0.1
        xfrc_ap = np.zeros((n, 38, 6))
        action = rng.uniform(-1.0, 1.0, size=(n, 7))
        q0 = np.array(model.qpos0[:7], copy=True)[None].repeat(n, 0)
        ds.upload_state(active=np.ones(n, np.int32), qpos=qpos, qvel=qvel,
                        mocap_pos=np.zeros((n, 0, 3)),
                        mocap_quat=np.zeros((n, 0, 4)),
                        rhs=np.zeros((n, 4, 43)),
                        count=np.ones(n, np.int32),
                        qfrc_applied=qfrc_ap, xfrc_applied=xfrc_ap)
        ds.set_initial_joints(q0)
        ds.stage_action(action)
        ds.stage_gripper(np.zeros((n, 2)))
        ds.begin_forward()
        ds.launch_position()
        ds.launch_velocity_prefix()
        ds.launch_bias()
        ds.launch_control(force_refresh=True, policy_step=True)
        control_error = np.array(ds.control.error.numpy(), copy=True)
        status_control = np.array(ds.status_control.numpy(), copy=True)
        ctrl = np.array(ds.control.ctrl.numpy(), copy=True)
        ranks = np.array(ds.control.ranks.numpy(), copy=True)
        d07_M = np.array(ds.d07.osc.M.numpy(), copy=True)
        s.raw['control_error'] = control_error.tolist()
        s.raw['status_control'] = status_control.tolist()
        s.raw['ctrl_absmax'] = float(np.abs(ctrl).max())
        s.raw['ranks'] = ranks.tolist()
        s.raw['d07_M_absmax'] = float(np.abs(d07_M).max())
        s.check('control.error == 0 every world', bool(np.all(control_error == 0)),
                control_error.tolist())
        s.check('status_control == 0 every world',
                bool(np.all(status_control == 0)), status_control.tolist())
        s.check('ctrl nonzero produced', float(np.abs(ctrl).max()) > 0.0)
        s.check('d07 M populated', float(np.abs(d07_M).max()) > 0.0)
        # Accepted D08 reference path: same qpos/qvel and the same upstream
        # inputs through OSCActuationChain's physical adapter + controller.
        flat = pack.flat_model()
        ref_model = SimpleNamespace(**{**vars(flat), **pack.osc_fields()})
        ref = OSCActuationChain(ref_model, n, device='cuda:0')
        site_pos = np.array(ds.d07.osc.site_pos.numpy(), copy=True)
        site_mat = np.array(ds.d07.osc.site_mat.numpy(), copy=True)
        qm = np.array(ds.d07.osc.qM.numpy(), copy=True)
        qfrc_bias = np.array(ds.qfrc_bias.numpy(), copy=True)
        ref.physical.qpos.assign(np.ascontiguousarray(qpos))
        ref.physical.qvel.assign(np.ascontiguousarray(qvel))
        ref.physical.site_pos.assign(np.ascontiguousarray(site_pos))
        ref.physical.site_mat.assign(np.ascontiguousarray(site_mat))
        ref.physical.qM.assign(np.ascontiguousarray(qm))
        ref.physical.qfrc_bias.assign(np.ascontiguousarray(qfrc_bias))
        ref.physical.jacp.assign(
            np.array(ds.d07.osc.jacp.numpy(), copy=True))
        ref.physical.jacr.assign(
            np.array(ds.d07.osc.jacr.numpy(), copy=True))
        ref.physical.active.assign(np.ones(n, np.int32))
        ref.control.gains.assign(np.array(ds.control.gains.numpy(), copy=True))
        ref.control.ctrl_range.assign(
            np.array(ds.control.ctrl_range.numpy(), copy=True))
        ref.control.q0.assign(np.ascontiguousarray(q0))
        ref.control.action.assign(np.ascontiguousarray(action))
        ref.control.policy_step.assign(np.ones(n, np.int32))
        ref.force_refresh.assign(np.ones(n, np.int32))
        ref.launch()
        ref_ctrl = np.array(ref.control.ctrl.numpy(), copy=True)
        ref_error = np.array(ref.control.error.numpy(), copy=True)
        s.raw['ref_ctrl_absmax'] = float(np.abs(ref_ctrl).max())
        s.check('reference control.error == 0',
                bool(np.all(ref_error == 0)), ref_error.tolist())
        ctrl_bitwise = bitwise_equal(ctrl, ref_ctrl)
        # R03-N05: this reference path SHARES jacp/jacr/gains/ctrl_range/q0/
        # action/policy_step with the object under test; the bitwise result
        # proves only controller-arithmetic equivalence under identical inputs
        # and configuration -- not the inputs themselves.
        s.check('controller arithmetic bitwise vs accepted D08 under identical '
                'inputs and config (shared jacp/jacr/gains/ctrl_range/q0/'
                'action/policy_step; inputs themselves covered by the '
                'registered independent CPU comparison)', ctrl_bitwise)
        ref_ranks = np.array(ref.control.ranks.numpy(), copy=True)
        s.check('ranks equal reference (same shared inputs)', bitwise_equal(ranks, ref_ranks))
        s.raw['independent_check_osc_inputs_vs_cpu_mujoco'] = {
            'kind': 'independent_reviewer_evidence_registration',
            'reviewer': 'P06-D independent reviewer (user-forward-r01/p06-d-review), round r03, MSG 009',
            'review_message_filename': '009_REVIEW_r03_supplement_c12e0c25.md',
            'review_message_sha256':
                '4affa7bc6efff390f854657112c4d7035c081ea19f350689679473943a018399',
            'evidence': [
                {'file': 'doc/phase_06/review/P06-D/r03/reverify_r03d.json',
                 'sha256': '2f8669b5b2db61c50fbba3f2a89599ea59d3c6df975b180391779f5ce7e6e9c5'},
                {'file': 'doc/phase_06/review/P06-D/r03/reverify_r03d.py',
                 'sha256': 'f15abb9f5941c9d3f931f11fa8662e118d7ba88fd8482304caf697f134ef74fd'},
            ],
            'method': 'fresh CPU MuJoCo: mj_forward + mj_jacSite(model, d, jacp, jacr, site_id=13)',
            'per_field_verdicts': {
                'jacr': 'bitwise_equal (n=129, n_diff=0, maxabs=0.0)',
                'joint_pos': 'bitwise_equal (n=7, n_diff=0, maxabs=0.0)',
                'joint_vel': 'bitwise_equal (n=7, n_diff=0, maxabs=0.0)',
                'ee_mat': 'bitwise_equal (n=9, n_diff=0, maxabs=0.0)',
                'qM': 'bitwise_equal (n=153; r03 S7/S2)',
                'qfrc_bias': 'bitwise_equal (n=43; r03 S7/S2)',
                'site_pos': 'bitwise_equal (n=3; r03 S7/S2)',
                'site_mat': 'bitwise_equal (n=9; r03 S7/S2)',
                'jacp': 'not bitwise: n=129, n_diff=5, maxabs=1.39e-17 (<=1 ULP scale, in computed-producer band)',
                'selectedJ': 'not bitwise: n=42, n_diff=5, maxabs=1.39e-17 (<=1 ULP scale, in band)',
                'ee_pos': 'not bitwise: n=3, n_diff=1, maxabs=6.94e-18 (<=1 ULP scale, in band)',
                'ee_velocity': 'not bitwise: n=6, n_diff=3, maxabs=8.33e-17; layout [velp,velr]=[lin,ang] '
                               'confirmed ([ang,lin] disproven: n_diff=6, maxabs=0.465)',
            },
            'scope_note': 'independent check of OSC INPUT correctness -- the part the '
                          'in-battery shared-input D08 reference comparison cannot cover; '
                          'registered as external reviewer evidence, not re-run inside the battery',
            'related_config_provenance': 'MSG 009 sec.1: gains=[150]*6+[2*sqrt(150)]*6 and '
                                         'ctrl_range=actuator_ctrlrange traced to robosuite 1.4.0 '
                                         'osc.py:118,157-158 and base_controller.py:151-156,237',
        }
        # Clock semantics (F04).  The bias generation is read at an explicit
        # boundary (R03-F11), never inside the launch.
        witness_ctrl = ds.capture_control_witness()
        s.raw['clock'] = ds.clock.as_dict()
        s.check('goal clock advanced (policy_step)', ds.clock.goal_writes == 1,
                ds.clock.goal_writes)
        device_clock = np.asarray(witness_ctrl['actual_device_clock'], dtype=np.int64)
        s.raw['actual_control_witness'] = witness_ctrl
        s.check('successful device gripper clock advanced',
                bool(np.all(device_clock[:,6] == 1)), device_clock[:,6].tolist())
        s.check('biaslive reads recorded', ds.clock.bias_live_reads == 1,
                ds.clock.bias_live_reads)
        s.check('bias cache version bound to bias stamp',
                bool(np.array_equal(np.asarray(witness_ctrl['actual_bias_writer_versions'], dtype=np.int64),
                                    np.array(ds.witness.versions['bias'].numpy(), copy=True))))
        # F10: injected OSC failure must surface in status_control.  Poison
        # the upstream qM to all-zero so the D07 adapter rebuilds a singular
        # arm mass: _inverse7 then fails exactly the way the R02-F01 unwired
        # path did (error=2).  Poisoning phys_mass alone would be overwritten
        # by the adapter recompute inside launch_control.
        ds.d07.osc.qM.assign(np.zeros((n, 153)))
        ds.launch_control(force_refresh=True, policy_step=False)
        err_after = np.array(ds.control.error.numpy(), copy=True)
        st_after = np.array(ds.status_control.numpy(), copy=True)
        s.raw['poisoned_qm_error'] = err_after.tolist()
        s.raw['poisoned_qm_status_control'] = st_after.tolist()
        poisoned = np.any(err_after != 0)
        s.check('injected control failure observed', poisoned, err_after.tolist())
        if poisoned:
            s.check('control failure visible in status_control',
                    bool(np.all(st_after != 0)), st_after.tolist())
            snap = ds.witness.host_snapshot()
            s.check('failed control not stamped',
                    bool(np.all(snap['valid']['control'] == 0)),
                    snap['valid']['control'].tolist())
        # F03: failed control must not publish actuator force.
        ds.launch_actuation()
        aa = np.array(ds.actuation_active.numpy(), copy=True)
        qfrc_act = np.array(ds.actuation.qfrc_actuator.numpy(), copy=True)
        s.raw['actuation_active_after_fail'] = aa.tolist()
        s.check('actuation_active zero when control failed',
                bool(np.all(aa == 0)), aa.tolist())
        s.check('no published qfrc_actuator when control failed',
                float(np.abs(qfrc_act).max()) == 0.0)
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('control_graph_capture')
    try:
        # R03-F11 acceptance: the control entry must be CUDA-Graph capturable
        # (no D2H/H2D inside its launch path), replay must be byte-identical
        # to eager, and a changed input device buffer must change the output
        # through the same graph.
        from libero.gpu.host.runtime.dynamics.stages import DynamicsStages
        n = 4
        ds = DynamicsStages(pack, n, device='cuda:0')
        rng = np.random.default_rng(args.seed + 8)
        qpos = np.stack([make_qpos(rng, model) for _ in range(n)])
        qvel = rng.normal(scale=0.3, size=(n, 43))
        q0 = np.array(model.qpos0[:7], copy=True)[None].repeat(n, 0)
        ds.upload_state(active=np.ones(n, np.int32), qpos=qpos, qvel=qvel,
                        mocap_pos=np.zeros((n, 0, 3)),
                        mocap_quat=np.zeros((n, 0, 4)),
                        rhs=np.zeros((n, 4, 43)),
                        count=np.ones(n, np.int32),
                        qfrc_applied=np.zeros((n, 43)),
                        xfrc_applied=np.zeros((n, 38, 6)))
        ds.set_initial_joints(q0)
        # prewarm every kernel module eagerly (module load must not happen
        # inside the capture region)
        ds.begin_forward()
        ds.launch_position()
        ds.launch_velocity_prefix()
        ds.launch_bias()
        ds.launch_control(force_refresh=True)
        ds.launch_actuation()
        ds.launch_acceleration()
        ds.begin_forward()
        ds.launch_position()
        ds.launch_velocity_prefix()
        ds.launch_bias()
        ds.launch_control(force_refresh=True)
        eager_ctrl = np.array(ds.control.ctrl.numpy(), copy=True)
        eager_status = np.array(ds.status_control.numpy(), copy=True)
        eager_cache = np.array(ds.control.mass.numpy(), copy=True)
        # capture the control entry alone (the pattern the reviewer exercised)
        capture_error = None
        graph = None
        wp.capture_begin()
        try:
            ds.launch_control(force_refresh=True)
            graph = wp.capture_end()
        except Exception as exc:  # noqa: BLE001 - recorded, not re-raised
            capture_error = f'{type(exc).__name__}: {exc}'
            try:
                wp.capture_end()
            except Exception:
                pass
        s.check('launch_control graph capture succeeds', capture_error is None,
                capture_error or 'ok')
        if graph is not None:
            wp.capture_launch(graph)
            replay_ctrl = np.array(ds.control.ctrl.numpy(), copy=True)
            replay_status = np.array(ds.status_control.numpy(), copy=True)
            replay_cache = np.array(ds.control.mass.numpy(), copy=True)
            s.check('replay bitwise == eager (ctrl/status/cache)',
                    bitwise_equal(replay_ctrl, eager_ctrl)
                    and bitwise_equal(replay_status, eager_status)
                    and bitwise_equal(replay_cache, eager_cache))
            ds.qvel.assign(np.ascontiguousarray(qvel + 0.11))
            wp.capture_launch(graph)
            changed_ctrl = np.array(ds.control.ctrl.numpy(), copy=True)
            s.check('same graph, changed device input -> changed ctrl',
                    changed_ctrl.tobytes() != eager_ctrl.tobytes())
        # full six-entry chain capture (position..acceleration)
        chain_error = None
        chain_graph = None
        ds.begin_forward()
        wp.capture_begin()
        try:
            ds.launch_position()
            ds.launch_velocity_prefix()
            ds.launch_bias()
            ds.launch_control(force_refresh=True)
            ds.launch_actuation()
            ds.launch_acceleration()
            chain_graph = wp.capture_end()
        except Exception as exc:  # noqa: BLE001
            chain_error = f'{type(exc).__name__}: {exc}'
            try:
                wp.capture_end()
            except Exception:
                pass
        s.check('full six-entry chain graph capture succeeds',
                chain_error is None, chain_error or 'ok')
        if chain_graph is not None:
            wp.capture_launch(chain_graph)
            first_acc = np.array(ds.acceleration.qacc_smooth.numpy(), copy=True)
            wp.capture_launch(chain_graph)
            second_acc = np.array(ds.acceleration.qacc_smooth.numpy(), copy=True)
            s.check('chain replay deterministic on unchanged inputs',
                    first_acc.tobytes() == second_acc.tobytes())
        s.raw['capture_error'] = capture_error
        s.raw['chain_error'] = chain_error
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('staged_epochs_witness_and_stale_consumer')
    try:
        from libero.gpu.host.runtime.dynamics.stages import DynamicsStages
        n = 4
        ds = DynamicsStages(pack, n, device='cuda:0')
        rng = np.random.default_rng(args.seed + 4)
        qpos = np.stack([make_qpos(rng, model) for _ in range(n)])
        qvel = rng.normal(scale=0.3, size=(n, 43))
        ds.upload_state(active=np.ones(n, np.int32), qpos=qpos, qvel=qvel,
                        mocap_pos=np.zeros((n, 0, 3)),
                        mocap_quat=np.zeros((n, 0, 4)),
                        rhs=np.zeros((n, 4, 43)),
                        count=np.ones(n, np.int32),
                        qfrc_applied=np.zeros((n, 43)),
                        xfrc_applied=np.zeros((n, 38, 6)))
        # F05-2: run at state_epoch > 0 so stamps are not 0==0.
        ds.bump_state_epoch()
        ds.bump_state_epoch()
        ds.begin_forward()
        snap = ds.witness.host_snapshot()
        s.check('state_epoch > 0 for stamp checks',
                bool(np.all(snap['state_epoch'] > 0)),
                snap['state_epoch'].tolist())
        ds.launch_position()
        ds.launch_velocity_prefix()
        ds.launch_bias()
        ds.launch_control(force_refresh=True)
        ds.launch_actuation()
        ds.launch_acceleration()
        snap = ds.witness.host_snapshot()
        nonzero_epoch = snap['scratch_epoch']
        for stage in ('position', 'velocity_prefix', 'bias', 'control',
                      'actuation', 'acceleration'):
            stamped = bool(np.all(snap['stamps'][stage] == nonzero_epoch))
            valid = bool(np.all(snap['valid'][stage] == 1))
            s.check(f'{stage} stamped at scratch epoch', stamped,
                    snap['stamps'][stage].tolist())
            s.check(f'{stage} valid == 1', valid,
                    snap['valid'][stage].tolist())
        # F02 decisive counterexample: run a consumer with a stale producer.
        ds.bump_state_epoch()
        ds.begin_forward()  # new scratch generation; producers NOT re-run
        before_vel = np.array(ds.cvel.numpy(), copy=True)
        ds.launch_velocity_prefix()  # gate must refuse to compute/stamp
        gates = ds.gate_arrays()
        snap = ds.witness.host_snapshot()
        s.check('stale gate_velocity == 0', bool(np.all(gates['gate_velocity'] == 0)),
                gates['gate_velocity'].tolist())
        s.check('stale consumer not stamped',
                bool(np.all(snap['valid']['velocity_prefix'] == 0)),
                snap['valid']['velocity_prefix'].tolist())
        after_vel = np.array(ds.cvel.numpy(), copy=True)
        unchanged = before_vel.tobytes() == after_vel.tobytes()
        clean_zero = bool(np.all(after_vel == 0.0))
        s.check('stale consumer not committed (old bytes or clean zero)',
                unchanged or clean_zero,
                f'unchanged={unchanged} clean_zero={clean_zero}')
        s.raw['stale_output_semantics'] = (
            'gate=0 runs the accepted inactive path: outputs are clean zeros '
            'or old-generation bytes, never a fresh-generation computation')
        # F02/F03 follow-through: downstream gates also refuse.
        ds.launch_bias()
        ds.launch_actuation()
        ds.launch_acceleration()
        gates = ds.gate_arrays()
        snap = ds.witness.host_snapshot()
        s.check('gate_bias == 0 on stale chain',
                bool(np.all(gates['gate_bias'] == 0)), gates['gate_bias'].tolist())
        s.check('gate_actuation == 0 on stale chain',
                bool(np.all(gates['gate_actuation'] == 0)))
        s.check('gate_acceleration == 0 on stale chain',
                bool(np.all(gates['gate_acceleration'] == 0)))
        s.check('bias not stamped on stale chain',
                bool(np.all(snap['valid']['bias'] == 0)))
        s.check('acceleration not stamped on stale chain',
                bool(np.all(snap['valid']['acceleration'] == 0)))
        # Full-pipeline skip/refresh cache check under a discriminating
        # condition (F05-3): all seven cache fields, with a deliberate
        # in-place perturbation between refresh and skip that must survive.
        ds.begin_forward()
        ds.launch_position()
        ds.launch_velocity_prefix()
        ds.launch_bias()
        ds.launch_control(force_refresh=True)
        cache_fields = ('mass', 'jacobian', 'q', 'qvel', 'ee_pos', 'ee_rot',
                        'ee_velocity')
        before = {f: np.array(getattr(ds.control, f).numpy(), copy=True)
                  for f in cache_fields}
        s.check('refresh produced nonzero cache mass',
                float(np.abs(before['mass']).max()) > 0.0)
        # deliberate perturbation: skip must NOT silently recompute
        poisoned = before['mass'].copy()
        poisoned[0, 0] += 123.0
        ds.control.mass.assign(poisoned)
        ds.launch_control(force_refresh=False)
        after = {f: np.array(getattr(ds.control, f).numpy(), copy=True)
                 for f in cache_fields}
        s.check('skip keeps poisoned mass (no silent refresh)',
                after['mass'].tobytes() == poisoned.tobytes())
        s.check('skip keeps all other cache fields bitwise',
                all(after[f].tobytes() == before[f].tobytes()
                    for f in cache_fields if f != 'mass'))
        s.check('skip counted once, refresh twice total',
                ds.clock.skip_count == 1 and ds.clock.refresh_count == 2,
                (ds.clock.skip_count, ds.clock.refresh_count))
        v0 = ds.clock.ctrl_version
        ds.launch_actuation()
        s.check('ctrl version bump on actuation',
                ds.clock.ctrl_version == v0 + 1)
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('error_sticky_inactive_repair_routing')
    try:
        from libero.gpu.host.runtime.dynamics.acceleration import SmoothXfrcSolveWorkspace, StaticTopology
        topology = StaticTopology({
            name: np.asarray(getattr(model, name), dtype=np.int32)
            for name in ('body_parentid', 'body_rootid', 'body_dofadr',
                         'body_dofnum', 'dof_parentid')})
        n = 5
        acc = SmoothXfrcSolveWorkspace(topology, pack.solve_model(), n,
                                       device='cuda:0')
        active = np.ones(n, np.int32)
        active[1] = 0
        active[2] = 7
        passive = np.zeros((n, 43))
        bias = np.zeros((n, 43))
        applied = np.zeros((n, 43))
        actuator = np.zeros((n, 43))
        passive[0, 0] = 1.0
        passive[3, 0] = np.nan
        # world 4: near-singular qM forces the native pivot repair (25).
        qm = np.zeros((n, 153))
        for w in range(n):
            for k in range(43):
                qm[w, model.dof_Madr[k]] = 2.0
        qm[4, model.dof_Madr[5]] = 1e-18
        xfrc = np.zeros((n, 38, 6))
        xipos = np.zeros((n, 38, 3))
        sub = np.zeros((n, 38, 3))
        cdof = np.zeros((n, 43, 6))
        acc.upload(active=active, passive=passive, bias=bias, applied=applied,
                   actuator=actuator, xfrc=xfrc, xipos=xipos, subtree_com=sub,
                   cdof=cdof, qm=qm)
        acc.launch()
        out = acc.download()
        s.raw['status'] = out['status'].tolist()
        s.raw['entry_status'] = out['entry_status'].tolist()
        s.raw['solve_status'] = out['solve_status'].tolist()
        s.check('inactive world clean zero', out['status'][1] == 0 and
                bool(np.all(out['qacc_smooth'][1] == 0.0)))
        s.check('invalid active routes 1', out['status'][2] == 1)
        s.check('nonfinite input routes 3', out['status'][3] == 3)
        s.check('ok world succeeds', out['status'][0] == 0)
        s.check('repair world routes 25', out['status'][4] == 25,
                int(out['status'][4]))
        s.check('repair visible in warning_count',
                int(out['warning_count'][4]) >= 1,
                out['warning_count'].tolist())
        # R02-F07: failed worlds publish zero smooth as well.
        s.check('failed world smooth zero (F07)',
                bool(np.all(out['smooth'][3] == 0.0)))
        # Sticky semantics: same input relaunch reproduces status bits;
        # healthy re-input clears (per accepted launch-scoped re-derivation).
        acc.launch()
        out2 = acc.download()
        s.check('relaunch retains status bits',
                bitwise_equal(out['status'], out2['status']))
        healthy = dict(active=np.ones(n, np.int32), passive=np.zeros((n, 43)),
                       bias=np.zeros((n, 43)), applied=np.zeros((n, 43)),
                       actuator=np.zeros((n, 43)), xfrc=np.zeros((n, 38, 6)),
                       xipos=np.zeros((n, 38, 3)), subtree_com=np.zeros((n, 38, 3)),
                       cdof=np.zeros((n, 43, 6)))
        qm2 = np.zeros((n, 153))
        for w in range(n):
            for k in range(43):
                qm2[w, model.dof_Madr[k]] = 2.0
        healthy['qm'] = qm2
        acc.upload(**healthy)
        acc.launch()
        out3 = acc.download()
        s.check('healthy re-input clears to 0',
                bool(np.all(out3['status'] == 0)), out3['status'].tolist())
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('contamination_whitelist_vs_inplace')
    try:
        from libero.gpu.host.runtime.dynamics.stages import DynamicsStages
        n = 4
        ds = DynamicsStages(pack, n, device='cuda:0')
        rng = np.random.default_rng(args.seed + 7)
        qpos = np.stack([make_qpos(rng, model) for _ in range(n)])
        qvel = rng.normal(scale=0.2, size=(n, 43))
        ds.upload_state(active=np.ones(n, np.int32), qpos=qpos, qvel=qvel,
                        mocap_pos=np.zeros((n, 0, 3)),
                        mocap_quat=np.zeros((n, 0, 4)),
                        rhs=np.zeros((n, 4, 43)),
                        count=np.ones(n, np.int32),
                        qfrc_applied=np.zeros((n, 43)),
                        xfrc_applied=np.zeros((n, 38, 6)))
        ds.begin_forward()
        ds.launch_position()
        ds.launch_velocity_prefix()
        ds.launch_bias()
        # (1) whitelist rejection: named as such, NOT a numeric boundary.
        whitelist_rejected = False
        try:
            ds.d07.upload(jacp=np.zeros((n, 3, 43)))
        except KeyError:
            whitelist_rejected = True
        s.check('whitelist rejects unknown keyword jacp (name check only)',
                whitelist_rejected)
        # (2) in-place pollution of a frozen device buffer: pointer validation
        # cannot see it, the input witness and the comparator must.
        ds.launch_control(force_refresh=True)
        witness_clean = ds.capture_input_witness()
        ctrl_clean = np.array(ds.control.ctrl.numpy(), copy=True)
        ranks_clean = np.array(ds.control.ranks.numpy(), copy=True)
        cache_mass_clean = np.array(ds.control.mass.numpy(), copy=True)
        bindings_ok = True
        try:
            ds.validate_bindings()
        except ValueError:
            bindings_ok = False
        s.check('pointer/shape validation cannot see in-place write',
                bindings_ok)
        poison = np.array(ds.d07.osc.qM.numpy(), copy=True)
        poison[:, model.dof_Madr[5]] += 7.0
        ds.d07.osc.qM.assign(poison)
        ds.launch_control(force_refresh=True)
        witness_dirty = ds.capture_input_witness()
        from typed_compare import compare_arrays
        flatten = lambda witness: {stage+'.'+field: value
            for stage, fields in witness.items() for field,value in fields.items()}
        witness_verdict = compare_arrays(flatten(witness_clean), flatten(witness_dirty))
        s.raw['actual_typed_input_witness_pollution_verdict'] = witness_verdict
        s.check('actual consumer input witness detects in-place pollution',
                not witness_verdict['ok'])
        ctrl_dirty = np.array(ds.control.ctrl.numpy(), copy=True)
        ranks_dirty = np.array(ds.control.ranks.numpy(), copy=True)
        cache_mass_dirty = np.array(ds.control.mass.numpy(), copy=True)
        s.check('comparator catches pollution (ctrl/ranks/cache differ from clean run)',
                ctrl_dirty.tobytes() != ctrl_clean.tobytes()
                or ranks_dirty.tobytes() != ranks_clean.tobytes()
                or not bitwise_equal(cache_mass_dirty, cache_mass_clean))
        s.raw['witness_clean'] = {stage:{name:value.tolist() for name,value in fields.items()} for stage,fields in witness_clean.items()}
        s.raw['witness_dirty'] = {stage:{name:value.tolist() for name,value in fields.items()} for stage,fields in witness_dirty.items()}
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('capacities_eager_and_graph')
    try:
        from libero.gpu.host.runtime.dynamics.acceleration import SmoothXfrcSolveWorkspace, StaticTopology
        topology = StaticTopology({
            name: np.asarray(getattr(model, name), dtype=np.int32)
            for name in ('body_parentid', 'body_rootid', 'body_dofadr',
                         'body_dofnum', 'dof_parentid')})
        graph_results = {}
        for n in args.capacities:
            acc = SmoothXfrcSolveWorkspace(topology, pack.solve_model(), n,
                                           device='cuda:0')
            rng = np.random.default_rng(args.seed + 100 + n)
            active = np.ones(n, np.int32)
            if n > 1:
                active[-1] = 0
            payload = dict(
                active=active,
                passive=rng.normal(size=(n, 43)),
                bias=rng.normal(size=(n, 43)),
                applied=rng.normal(size=(n, 43)),
                actuator=rng.normal(size=(n, 43)),
                xfrc=np.zeros((n, 38, 6)),
                xipos=rng.normal(size=(n, 38, 3)) * 0.01,
                subtree_com=rng.normal(size=(n, 38, 3)) * 0.001,
                cdof=rng.normal(size=(n, 43, 6)),
                qm=np.zeros((n, 153)))
            for w in range(n):
                for k in range(43):
                    payload['qm'][w, model.dof_Madr[k]] = 3.0 + 0.01 * k
            acc.upload(**payload)
            acc.launch()
            eager = acc.download()
            # Decisive graph test (F05-4): poison the output buffers, then
            # replay -- the graph must recompute them from the same inputs.
            acc.qacc_smooth.assign(np.full((n, 43), 42.0))
            wp.capture_begin()
            acc.launch()
            graph = wp.capture_end()
            wp.capture_launch(graph)
            replay = acc.download()
            recomputed = bitwise_equal(replay['qacc_smooth'],
                                       eager['qacc_smooth'])
            s.check(f'N={n} graph replay recomputes outputs',
                    recomputed)
            changed = False
            payload['passive'] = payload['passive'] + 1.0
            acc.upload(passive=payload['passive'])
            wp.capture_launch(graph)
            dynamic = acc.download()
            changed = dynamic['qacc_smooth'].tobytes() != eager['qacc_smooth'].tobytes()
            s.check(f'N={n} dynamic input effect through graph', changed)
            # statuses: 0 ok; inactive worlds 0; any 25 is recorded as repair,
            # never silently counted as plain success (F05-4).
            st = eager['status'].tolist()
            s.check(f'N={n} statuses 0 (or 25 repair only)',
                    all(x in (0, 25) for x in st), st)
            graph_results[n] = {'status': st,
                                'recomputed': bool(recomputed),
                                'dynamic_input_changed_output': bool(changed)}
        s.raw['graph_results'] = graph_results
    except Exception as e:
        s.fail(e)
    s.done()

    # ------------------------------------------------------------------
    s = report.section('not_run_scopes')
    s.check('disclosed_paired_corpus_comparison', True,
            'not_run: P06-REF paired corpus not supplied on this host')
    s.check('disclosed_producer_interval_certificates', True,
            'not_run: shared-v1 ABI (P06-F) and independent producer interval '
            'bounds not supplied (both covered by this disclosure)')
    s.check('disclosed_full_refresh_four_chain', True,
            'not_run: scheduled by P06-I with C/S/N dependencies')
    s.check('disclosed_d03_post_constraint_sensors', True,
            'not_run: depends on P06-N results')
    s.check('disclosed_capacity_performance_residency', True,
            'not_run: thresholds not frozen (L03)')
    s.check('disclosed_complete_substep_action_trajectory', True,
            'not_run: not claimed by this task')
    s.check('disclosed_native_nvcc_certificate', True,
            'not_run: nvcc absent on this host (gcc 15.2.0 present); device '
            'kernels compiled by the Warp toolchain, not an nvcc certificate')
    s.check('disclosed_repair_neighbourhood_numerics', True,
            'unverified: pivot-repair neighbourhood numerical equivalence not '
            'established (routing/visibility only; independent review measured '
            'device 9.999e14 vs CPU dense true solution 1e18 when the repair '
            'direction is excited)')
    s.check('disclosed_cpu500', True,
            'not_run: CPU500 not run; environment-callability prerequisite '
            'waived by user (kept not_run). CPU500 is environment scope, not a '
            'P06-D task verification item; this task\'s CPU-side reference is '
            'fresh MuJoCo 2.3.7 oracle runs')
    s.status = 'disclosed'
    s.done()

    report.t1 = time.time()
    summary = {
        'schema': 'phase06-dynamics-author-battery-v2',
        'claim': 'developer_self_test_only; not accepted',
        'seed': args.seed,
        'capacities': args.capacities,
        'elapsed_s': report.t1 - report.t0,
        'sections': [{k: e[k] for k in ('name', 'status', 'first_diff')}
                     for e in report.sections],
        'passed': sum(e['status'] == 'passed' for e in report.sections),
        'failed': sum(e['status'] == 'failed' for e in report.sections),
        'errored': sum(e['status'] == 'error' for e in report.sections),
    }
    (report.out / 'report.json').write_text(
        json.dumps({'summary': summary, 'sections': report.sections},
                   indent=2, sort_keys=True, default=str))
    (report.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary['failed'] == 0 and summary['errored'] == 0 else 1


def main():
    # Original engineering/post-output checks are retained as historical
    # diagnostics only. They cannot enter the source-typed scientific seal.
    if "--mode" in sys.argv:
        from scientific_controls import main as scientific_main
        return scientific_main()
    if "--historical-diagnostic" in sys.argv:
        sys.argv.remove("--historical-diagnostic")
        code = _historical_main()
        print(json.dumps({"historical_diagnostic_exit":code,
                          "scientific_acceptance":False,
                          "reason":"post-output truth/engineering tolerances are historical only"}))
        return code
    print(json.dumps({"ok":False,"status":"not_frozen",
                      "reason":"Use the pre-output source-typed --mode cpu/gpu/readonly unit. The original full thirteen-section/paired-producer package is still required; historical diagnostics require explicit --historical-diagnostic and supply no scientific acceptance."}))
    return 2


if __name__ == '__main__':
    sys.exit(main())
