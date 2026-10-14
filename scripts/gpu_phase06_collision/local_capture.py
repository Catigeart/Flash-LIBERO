"""Private local same-call capture of the pinned library's collision table.

The callback invokes the retained original detector pointer, records its actual
arguments and returned raw prefix, then returns that exact count to the original
driver. It is never a detector used by the GPU. The plain-versus-hooked collision
check must be bitwise identical. This local capture does not certify REF's full
request-stage corpus or the producer contract.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np

_LOCK = threading.Lock()


class Contact(ct.Structure):
    _fields_ = [(n, ct.c_double * k if k > 1 else ct.c_double) for n, k in (
        ('dist', 1), ('pos', 3), ('frame', 9), ('includemargin', 1),
        ('friction', 5), ('solref', 2), ('solreffriction', 2), ('solimp', 5),
        ('mu', 1), ('H', 36))] + [(n, ct.c_int) for n in (
        'dim', 'geom1', 'geom2', 'exclude', 'efc_address')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _abi(mujoco):
    include = Path(mujoco.__path__[0]) / 'include'
    source = Path(__file__).with_name('local_capture_abi.c')
    headers = {str(p.relative_to(include)): sha(p)
               for p in sorted(include.rglob('*.h'))}
    cc = shutil.which('gcc')
    if not cc:
        raise RuntimeError('local capture ABI compiler unavailable')
    key = hashlib.sha256(json.dumps(
        {'source': sha(source), 'headers': headers, 'cc': sha(cc)},
        sort_keys=True).encode()).hexdigest()
    root = Path(os.environ['P06C_NATIVE_BUILD_ROOT']) / ('capture-abi-' + key)
    artifact = root / 'abi.so'
    if not root.exists():
        root.mkdir()
        argv = [cc, '-std=c11', '-shared', '-fPIC', '-O0',
                '-I', str(include), str(source), '-o', str(artifact)]
        from scripts.gpu_phase06_collision.local_native_domain import child_env
        env = child_env(root)
        env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib'
        result = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True)
        (root / 'stdout').open('x').write(result.stdout)
        (root / 'stderr').open('x').write(result.stderr)
        (root / 'BUILD.json').open('x').write(json.dumps({
            'argv': argv, 'cwd': str(root), 'PWD': env['PWD'],
            'env': {k:v for k,v in env.items() if k.startswith('XDG_') or
                    k in ('LIBERO_CONFIG_PATH','TMPDIR','TMP','TEMP','CUDA_CACHE_PATH','WP_CACHE_DIR')},
            'exit_code': result.returncode, 'key': key,
            'source_sha256': sha(source), 'header_sha256': headers,
            'compiler_path': cc, 'compiler_sha256': sha(cc),
            'artifact_sha256': sha(artifact) if artifact.is_file() else None,
        }, indent=2))
    record = json.loads((root / 'BUILD.json').read_text())
    if record['exit_code'] != 0 or record['artifact_sha256'] != sha(artifact):
        raise RuntimeError('local capture ABI build failed or drifted')
    lib = ct.CDLL(str(artifact))
    lib.p06c_local_contact_abi.argtypes = [ct.POINTER(ct.c_size_t)]
    lib.p06c_local_contact_abi.restype = None
    values = (ct.c_size_t * (len(Contact._fields_) + 4))()
    lib.p06c_local_contact_abi(values)
    expected = [ct.sizeof(ct.c_double), ct.sizeof(ct.c_int),
                ct.sizeof(Contact), ct.alignment(Contact)] + [
                    getattr(Contact, n).offset for n, _ in Contact._fields_]
    if list(values) != expected:
        raise RuntimeError(f'actual header/contact ABI mismatch: {list(values)} != {expected}')
    lib.p06c_local_mxcsr.argtypes = []
    lib.p06c_local_mxcsr.restype = ct.c_uint
    mxcsr=int(lib.p06c_local_mxcsr())
    if mxcsr & ((3<<13)|(1<<15)|(1<<6)):
        raise RuntimeError('passive original CPU MXCSR rounding/FTZ/DAZ unsupported')
    return {'passive_mxcsr':mxcsr,'native_values': list(values), 'ctypes_values': expected,
            'fields': [n for n, _ in Contact._fields_], 'build': record,
            'build_path': str(root), 'scope': 'local original-library capture ABI'}


def _record(m, d):
    contacts = []
    for c in d.contact[:int(d.ncon)]:
        contacts.append({n: int(getattr(c, n)) for n in (
            'dim', 'geom1', 'geom2', 'exclude', 'efc_address')} | {
            n: float(getattr(c, n)) for n in ('dist', 'includemargin', 'mu')} | {
            n: [float(v) for v in getattr(c, n)] for n in (
                'pos', 'frame', 'friction', 'solref', 'solreffriction', 'solimp', 'H')})
    return {'ncon': int(d.ncon), 'contacts': contacts,
            'nbodypair_broad': int(d.nbodypair_broad),
            'nbodypair_narrow': int(d.nbodypair_narrow),
            'ngeompair_mid': int(d.ngeompair_mid),
            'ngeompair_narrow': int(d.ngeompair_narrow),
            'bvh_active': np.asarray(d.bvh_active[:int(np.sum(m.body_bvhnum))],
                                     dtype=np.uint8).tolist(),
            'bvh_active_sum': int(np.sum(d.bvh_active[:int(np.sum(m.body_bvhnum))]))}


def _contact_bytes(record):
    floats, ints = [], []
    for c in record['contacts']:
        floats.extend([c['dist'], *c['pos'], *c['frame'], c['includemargin'],
                       *c['friction'], *c['solref'], *c['solreffriction'],
                       *c['solimp'], c['mu'], *c['H']])
        ints.extend([c[n] for n in ('dim', 'geom1', 'geom2', 'exclude', 'efc_address')])
    return np.asarray(floats, np.float64).tobytes() + np.asarray(ints, np.int32).tobytes()


def cpu_capture(m, d):
    import mujoco
    # Processes which capture scientific inputs use the explicit original loader.
    # Record actual maps, rather than assuming a subprocess inherits it.
    abi = _abi(mujoco)
    lib_path = Path(mujoco.__path__[0]) / 'libmujoco.so.2.3.7'
    lib = ct.CDLL(str(lib_path))
    table = (ct.c_void_p * 64).in_dll(lib, 'mjCOLLISIONFUNC')
    signature = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p,
                             ct.POINTER(Contact), ct.c_int, ct.c_int, ct.c_double)
    before_pos = np.array(d.geom_xpos, copy=True)
    before_mat = np.array(d.geom_xmat, copy=True)
    original = list(table)
    from scripts.gpu_phase06_collision.local_native_domain import SourceDomain
    domain = SourceDomain(m)
    calls, callbacks, callback_errors = [], [], []
    original_fenv = ct.CDLL(None).fegetround
    original_fenv.argtypes = []
    original_fenv.restype = ct.c_int
    fenv_before = int(original_fenv())
    if fenv_before != 0:
        raise RuntimeError('local capture requires original passive FE_TONEAREST')
    with _LOCK:
        mujoco.mj_collision(m, d)
        plain = _record(m, d)
        try:
            for slot, ptr in enumerate(original):
                if not ptr:
                    continue
                fn = signature(ptr)
                def callback(mp, dp, con, g1, g2, margin, fn=fn, slot=slot):
                    native_route = 5 <= slot // 8 <= slot % 8 <= 7
                    # Original driver allocates mjContact con[mjMAXCONPAIR=50].
                    # Keep all original bytes typed; padding/held bytes are not
                    # numerical source expected values.
                    if native_route:
                        before_bytes = np.frombuffer(ct.string_at(con,50*ct.sizeof(Contact)),
                            dtype=np.uint8).reshape(50,ct.sizeof(Contact)).copy()
                        before_bits = before_bytes.view(np.uint64).reshape(50,-1)[:,:10].copy()
                    count = fn(mp, dp, con, g1, g2, margin)
                    try:
                        if count < 0 or count > 50:
                            raise RuntimeError(f'original raw count outside 0..50: {count}')
                        raw = [[float(c.dist), *map(float, c.pos),
                                *map(float, c.frame[:6])] for c in con[:count]]
                        call = {'g1': g1, 'g2': g2, 'type1': slot // 8,
                                'type2': slot % 8, 'margin': float(margin),
                                'raw_count': count, 'raw': raw,
                                'original_function_address': int(ct.cast(fn,ct.c_void_p).value)}
                        if native_route:
                            after_bytes = np.frombuffer(ct.string_at(con,50*ct.sizeof(Contact)),
                                dtype=np.uint8).reshape(50,ct.sizeof(Contact)).copy()
                            after_bits = after_bytes.view(np.uint64).reshape(50,-1)[:,:10].copy()
                            witness = domain.observe(d,g1,g2,margin,before_bits,after_bits,count)
                            call['_native_capture'] = witness | {
                                'before_bytes':before_bytes,'after_bytes':after_bytes,
                                'before_bits':before_bits,'after_bits':after_bits}
                            call['source_domain'] = {
                                'schema':'p06c-original-native-call-domain-v1',
                                'before_after_same_original_call':True,
                                'source_witness_exact_written_and_held':True,
                                'native_adapter_copies_returned_prefix_only':True,
                                'original_scratch_contact_bytes':ct.sizeof(Contact)}
                        calls.append(call)
                    except BaseException as exc:
                        callback_errors.append(f'{type(exc).__name__}: {exc}')
                    return count
                cb = signature(callback)
                callbacks.append(cb)
                table[slot] = ct.cast(cb, ct.c_void_p).value
            mujoco.mj_collision(m, d)
            captured = _record(m, d)
        finally:
            for slot, ptr in enumerate(original):
                table[slot] = ptr
    if callback_errors:
        raise RuntimeError('local raw capture failed: ' + repr(callback_errors))
    counters = ('ncon', 'nbodypair_broad', 'nbodypair_narrow',
                'ngeompair_mid', 'ngeompair_narrow', 'bvh_active')
    if any(plain[k] != captured[k] for k in counters) or \
            _contact_bytes(plain) != _contact_bytes(captured):
        raise RuntimeError('local original callback changed collision result')
    if before_pos.tobytes() != np.asarray(d.geom_xpos).tobytes() or \
            before_mat.tobytes() != np.asarray(d.geom_xmat).tobytes():
        raise RuntimeError('original collision modified input geom frames')
    if len(calls) != captured['ngeompair_mid'] or \
            sum(c['raw_count'] > 0 for c in calls) != captured['ngeompair_narrow']:
        raise RuntimeError('local callback count/counter disagreement')
    if int(original_fenv()) != fenv_before:
        raise RuntimeError('original detector/source witness changed floating environment')
    passive_lib=ct.CDLL(str(Path(abi['build_path'])/'abi.so'))
    passive_lib.p06c_local_mxcsr.argtypes=[]
    passive_lib.p06c_local_mxcsr.restype=ct.c_uint
    mxcsr_after=int(passive_lib.p06c_local_mxcsr())
    if mxcsr_after & ((3<<13)|(1<<15)|(1<<6)):
        raise RuntimeError('original CPU/source witness changed MXCSR control mode')
    domain.identity['actual_loaded_libraries'] = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split()
        if len(fields)>5 and fields[-1].startswith('/') and Path(fields[-1]).is_file() and (
                '.so' in fields[-1] or 'ld-linux' in fields[-1]):
            domain.identity['actual_loaded_libraries'][fields[-1]] = sha(fields[-1])
    captured.update({'calls': calls, 'cpu_raw_counts': [c['raw_count'] for c in calls],
                     'capture_identity': {'library': str(lib_path),
                         'library_sha256': sha(lib_path), 'abi': abi,
                         'source_domain_witness': domain.identity,
                         'passive_fegetround_before':fenv_before,
                         'passive_fegetround_after':int(original_fenv()),
                         'passive_mxcsr_before':abi['passive_mxcsr'],'passive_mxcsr_after':mxcsr_after,
                         'actual_cwd':str(Path.cwd()),'actual_PWD':os.environ.get('PWD'),
                         'plain_hooked_bitwise_equal': True,
                         'geom_frames_unchanged': True,
                         'scope': 'local same-driver-execution ordered calls/raw/final/counters',
                         'full_REF_request_capture': False}})
    return captured
