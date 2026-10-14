"""D09 frozen CPU boundary. All numerical reference calls use MuJoCo 2.3.7."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path('/rjs2/lhw/LIBERO')
STATE = ROOT / 'doc/results/phase_03_sources/states_20260930_01'
MODEL = Path('/rjs2/lhw/LIBERO-wt/kinematics/doc/results/phase_02k0/kinematics_20260929_03/task_model.mjb')
SOURCE = Path('/rjs2/lhw/tmp/libero-mujoco-2.3.7-source/src/engine')
LIBRARY = Path(mujoco.__file__).parent / 'libmujoco.so.2.3.7'
LIB_SHA = 'a82e1280ec6a72a309135202aff5f7a2ee71fc1eb69c719ac9bbf30ce7b9ef0e'
MODEL_SHA = '3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42'
FIELDS = ('passive', 'bias', 'applied', 'actuator', 'qm', 'xfrc', 'smooth', 'acc', 'ld', 'inv', 'sqrt_inv')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(a, b):
    return np.array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


def capture(m, d):
    return {
        'passive': np.array(d.qfrc_passive, copy=True),
        'bias': np.array(d.qfrc_bias, copy=True),
        'applied': np.array(d.qfrc_applied, copy=True),
        'actuator': np.array(d.qfrc_actuator, copy=True),
        'qm': np.array(d.qM, copy=True),
        'xfrc': np.array(d.xfrc_applied, copy=True),
        'smooth': np.array(d.qfrc_smooth, copy=True),
        'acc': np.array(d.qacc_smooth, copy=True),
        'ld': np.array(d.qLD, copy=True),
        'inv': np.array(d.qLDiagInv, copy=True),
        'sqrt_inv': np.array(d.qLDiagSqrtInv, copy=True),
    }


def replay_acc(m, d, packet, fill):
    d.qfrc_passive[:] = packet['passive']
    d.qfrc_bias[:] = packet['bias']
    d.qfrc_applied[:] = packet['applied']
    d.qfrc_actuator[:] = packet['actuator']
    d.qM[:] = packet['qm']
    d.xfrc_applied[:] = packet['xfrc']
    d.qfrc_smooth[:] = fill
    d.qacc_smooth[:] = -fill
    d.qLD[:] = fill
    d.qLDiagInv[:] = fill
    d.qLDiagSqrtInv[:] = fill
    before = int(d.warning[mujoco.mjtWarning.mjWARN_INERTIA].number)
    mujoco.mj_factorM(m, d)
    mujoco.mj_fwdAcceleration(m, d)
    out = capture(m, d)
    warning = d.warning[mujoco.mjtWarning.mjWARN_INERTIA]
    out['warning_count'] = np.int32(int(warning.number) - before)
    out['warning_last'] = np.int32(int(warning.lastinfo) if out['warning_count'] else -1)
    return out


def run(target):
    if target.exists():
        raise FileExistsError(target)
    if sha(LIBRARY) != LIB_SHA or sha(MODEL) != MODEL_SHA:
        raise ValueError('frozen library or model SHA drift')
    manifest = json.loads((STATE / 'manifest.json').read_text())
    if manifest['total_records'] != 1498:
        raise ValueError('STATE count')
    source_files = [MODEL, LIBRARY, STATE / 'manifest.json']
    source_files += [STATE / (entry + ext) for entry in ('reset', 'set_init_state') for ext in ('.npz', '.json')]
    source_files += [SOURCE / name for name in ('engine_forward.c', 'engine_core_smooth.c', 'engine_support.c')]
    before_sha = {str(p): sha(p) for p in source_files}
    m = mujoco.MjModel.from_binary_path(str(MODEL))
    if (m.nq, m.nv, m.nM, m.nbody) != (48, 43, 153, 38):
        raise ValueError('model shape')
    d, dirty = mujoco.MjData(m), mujoco.MjData(m)
    real = {k: [] for k in FIELDS}
    identities = []
    counts = {}
    first_packet = None
    for entry in ('reset', 'set_init_state'):
        raw = STATE / (entry + '.npz')
        with np.load(raw, allow_pickle=False) as z:
            state = {k: np.array(z[k], copy=True) for k in ('qpos','qvel','qacc','qacc_warmstart','ctrl','time')}
        expected = {'qpos':(749,48), 'qvel':(749,43), 'qacc':(749,43),
                    'qacc_warmstart':(749,43), 'ctrl':(749,9), 'time':(749,)}
        for k, shape in expected.items():
            if state[k].shape != shape or state[k].dtype != np.float64:
                raise TypeError(('raw state schema',entry,k))
        events = json.loads((STATE / (entry+'.json')).read_text())
        if len(events) != 749:
            raise ValueError('raw event count')
        counts[entry] = {'total':len(events), 'context':{c:sum(x['context']==c for x in events) for c in 'ABC'}}
        if counts[entry]['context'] != {'A':250,'B':249,'C':250}:
            raise ValueError('raw event census')
        for j, event in enumerate(events):
            if event['entry'] != entry or event['state_boundary'] != 'outer_call_exit':
                raise ValueError('raw identity')
            mujoco.mj_resetData(m, d)
            d.qpos[:] = state['qpos'][j]
            d.qvel[:] = state['qvel'][j]
            d.qacc[:] = state['qacc'][j]
            d.qacc_warmstart[:] = state['qacc_warmstart'][j]
            d.ctrl[:] = state['ctrl'][j]
            d.time = float(state['time'][j])
            d.qfrc_applied[:] = 0
            d.xfrc_applied[:] = 0
            mujoco.mj_forward(m, d)
            native = capture(m, d)
            if not np.all(native['xfrc'] == 0) or not np.all(native['applied'] == 0):
                raise AssertionError('nonzero initialized external force')
            # Independent public entrypoint replays with two dirty write sets.
            for data, fill in ((d, 17.25), (dirty, -29.5)):
                replay = replay_acc(m, data, native, fill)
                for k in FIELDS:
                    if not bits_equal(replay[k], native[k]):
                        raise AssertionError(('native replay/prefill transparency',entry,j,k))
                if replay['warning_count'] != 0:
                    raise AssertionError('unexpected real inertia warning')
            for k in FIELDS:
                real[k].append(native[k])
            identity = dict(event)
            identity.update(record_id=f'{entry}:{j}', source_file_sha256=before_sha[str(raw)],
                            source_event_file_sha256=before_sha[str(STATE/(entry+'.json'))],
                            model_sha256=MODEL_SHA,
                            offline_replay_id=f'D09:STATE1498:{entry}:{j}:scratch_mj_forward',
                            stage='smooth_acceleration_chain', invocation='offline_scratch_fwdAcceleration',
                            iteration=None, call_seq=None, local_row_id=None,
                            original_boundary='outer_call_exit',
                            derived_boundary='mj_forward_scratch_same_state',
                            external_force_init='qfrc_applied=0,xfrc_applied=0')
            identities.append(identity)
            if first_packet is None:
                first_packet = native
    real = {k:np.stack(v) for k,v in real.items()}
    if len({json.dumps(x, sort_keys=True) for x in identities}) != 1498:
        raise AssertionError('duplicate identity')
    # Seven deterministic local original-C diagnostics, never counted as source events.
    local_specs = ('applied_positive', 'cancellation', 'nontrivial_qm',
                   'pivot_below', 'pivot_equal', 'pivot_above', 'association')
    local = {k:[] for k in FIELDS}
    local_warning = []
    for j, name in enumerate(local_specs):
        packet = {k:v.copy() for k,v in first_packet.items()}
        if name == 'applied_positive':
            packet['applied'][3] = 0.125
            packet['applied'][17] = -0.75
        elif name == 'cancellation':
            packet['passive'][:] = packet['bias']
            packet['applied'][:] = 0
            packet['actuator'][:] = 0
            packet['applied'][4] = 1.0
            packet['actuator'][4] = -1.0
        elif name == 'nontrivial_qm':
            packet['qm'][np.asarray(m.dof_Madr)] *= 1.125
            packet['applied'][2] = 0.5
        elif name == 'association':
            packet['passive'][:] = 0
            packet['bias'][:] = 0
            packet['applied'][:] = 0
            packet['actuator'][:] = 0
            packet['passive'][5] = 1e16
            packet['bias'][5] = -1.0
            packet['applied'][5] = -1e16
            packet['actuator'][5] = 1.0
        else:
            packet['qm'][:] = 0
            packet['qm'][np.asarray(m.dof_Madr)] = 1.0
            packet['qm'][int(m.dof_Madr[0])] = {
                'pivot_below':np.nextafter(1e-15,0),
                'pivot_equal':1e-15,
                'pivot_above':np.nextafter(1e-15,np.inf)}[name]
            packet['applied'][0] = 1.0
        a = replay_acc(m, d, packet, 11.25)
        b = replay_acc(m, dirty, packet, -13.5)
        for k in FIELDS:
            if not bits_equal(a[k],b[k]):
                raise AssertionError(('local prefill',name,k))
            local[k].append(a[k])
        if a['warning_count'] != b['warning_count'] or a['warning_last'] != b['warning_last']:
            raise AssertionError('local warning route')
        local_warning.append((int(a['warning_count']),int(a['warning_last'])))
    local = {k:np.stack(v) for k,v in local.items()}
    target.mkdir(parents=True)
    np.savez_compressed(target/'real.npz',**real)
    np.savez_compressed(target/'local.npz',**local)
    np.savez(target/'topology.npz',dof_Madr=np.asarray(m.dof_Madr,np.int32),
             dof_parentid=np.asarray(m.dof_parentid,np.int32),
             dof_simplenum=np.asarray(m.dof_simplenum,np.int32))
    (target/'identities.json').write_text(json.dumps(identities,indent=2)+'\n')
    (target/'local_specs.json').write_text(json.dumps(dict(zip(local_specs,local_warning)),indent=2)+'\n')
    after_sha = {str(p):sha(p) for p in source_files}
    if before_sha != after_sha:
        raise AssertionError('source drift during CPU run')
    result = {'task':'D09','stage':'cpu','real_count':len(identities),'local_count':len(local_specs),
              'census':counts,'native_forward_vs_replay':'bitwise_equal_all_fields',
              'double_prefill':'bitwise_equal_all_fields','local_warning':dict(zip(local_specs,local_warning)),
              'sources_before':before_sha,'sources_after':after_sha,
              'python_executable':__import__('sys').executable,'mujoco_import':mujoco.__file__,
              'files':{p.name:sha(p) for p in target.iterdir() if p.is_file()}}
    (target/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path)
    print(json.dumps(run(ap.parse_args().output),indent=2))
