"""D09 Astra rework regressions using this candidate's saved actual CUDA arrays.

The create process constructs packets without launching GPU work.  A separate
replay process reads only those packets and the sealed CPU reference.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import budget
import run_all as r
import verify_negative as vn


def clone(got):
    return {k:v.copy() for k,v in got.items()}


def reject(fn, marker):
    try:
        fn()
    except (AssertionError,TypeError,ValueError) as exc:
        reason=exc.args[0] if len(exc.args)==1 else str(exc)
        if isinstance(marker,tuple):
            if not isinstance(reason,tuple) or reason[:len(marker)]!=marker:
                raise AssertionError(('wrong first difference',marker,reason)) from exc
        elif marker not in str(exc):
            raise AssertionError(('wrong rejection',marker,reason)) from exc
        return repr(reason)
    raise AssertionError(('negative accepted',marker))


def modules():
    return {name:{'file':mod.__file__,'sha256':r.sha(mod.__file__)}
            for name,mod in (('run_all',r),('budget',budget),('verify_negative',vn))}


def packet(base,clean,corrupt):
    return dict(ids=base['ids'].copy(),active=base['active'].copy(),
                input_schema_version=np.int32(2),
                input_reference_sha256=np.bytes_(base['manifest_sha']),
                input_sha256=np.bytes_(vn.bound_digest(base['input'],base['ids'],base['manifest_sha'])),
                **{'input_'+k:v.copy() for k,v in base['input'].items()},
                **{'clean_'+k:v.copy() for k,v in clean.items()},
                **{'corrupt_'+k:v.copy() for k,v in corrupt.items()})


def write(path,values):
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**values)
    return r.sha(path)


def create(reference,gpu_dir,target):
    if target.exists():raise FileExistsError(target)
    target.mkdir(parents=True)
    rows,topology,_,_=r.load_reference(reference)
    cert=r.interval_for(rows['local'],topology)  # frozen before GPU output read
    source=gpu_dir/'gpu_local.npz'
    summary=json.loads((gpu_dir/'summary.json').read_text())
    if r.sha(source)!=summary['files']['gpu_local.npz']:
        raise AssertionError('actual GPU source SHA')
    with np.load(source,allow_pickle=False) as z:
        all_local={k:z[k].copy() for k in z.files}
    ids=np.array([6],np.int32)
    p=r.payload(rows['local'],ids,1)
    clean={k:v[6:7].copy() for k,v in all_local.items()}
    r.check(clean,rows['local'],cert,ids,p['active'],'local')
    r.residual_check(rows['local'],cert,clean,ids,topology)
    bound5=[float(x[6,0,5]) for x in cert['factor']['solution']]
    if cert['force'][0][6,5]!=1.0 or cert['force'][1][6,5]!=1.0 or bound5[0]<=0:
        raise AssertionError(('association certificate',bound5))
    cases={}
    checker=lambda g:r.check(g,rows['local'],cert,ids,p['active'],'local')
    zero=clone(clean);zero['acc'][:]=0.0;zero['solution'][:]=0.0
    cases['F01_actual_GPU_all_zero_check']=reject(lambda:checker(zero),
            ('GPU outside source interval','solve','local','solution',6,[0,0]))
    cases['F01_actual_GPU_all_zero_residual']=reject(
            lambda:r.residual_check(rows['local'],cert,zero,ids,topology),
            ('residual exceeds derived bound',6))
    cancellation={k:v[1:2].copy() for k,v in all_local.items()}
    cid=np.array([1],np.int32);cp=r.payload(rows['local'],cid,1)
    r.check(cancellation,rows['local'],cert,cid,cp['active'],'local')
    r.residual_check(rows['local'],cert,cancellation,cid,topology)
    if np.any(cancellation['smooth']) or np.any(cancellation['acc']):
        raise AssertionError('legal cancellation is not zero')
    cases['legal_cancellation_zero']='passed'
    stale=clone(clean)
    stale['acc'][:]=cancellation['acc'];stale['solution'][:]=cancellation['solution']
    cases['F01_stale_real_zero_solve']=reject(lambda:checker(stale),
            ('GPU outside source interval','solve','local','solution',6,[0,0]))
    one_ulp=clone(clean)
    beyond=np.nextafter(cert['factor']['solution'][1][6,0,5],np.inf)
    one_ulp['solution'][0,0,5]=beyond;one_ulp['acc'][0,5]=beyond
    cases['F01_one_ULP_beyond']=reject(lambda:checker(one_ulp),
            ('GPU outside source interval','solve','local','solution',6,[0,5]))
    wired=clone(clean);wired['solution'][0,0,5]=100.0;wired['acc'][0,5]=100.0
    cases['F01_consistent_wiring_wrong_solve']=reject(lambda:checker(wired),
            ('GPU outside source interval','solve','local','solution',6,[0,5]))
    repair={k:v[3:4].copy() for k,v in all_local.items()}
    pid=np.array([3],np.int32);pp=r.payload(rows['local'],pid,1)
    r.check(repair,rows['local'],cert,pid,pp['active'],'local')
    repaired=r.residual_check(rows['local'],cert,repair,pid,topology)
    if repaired['repaired_modified_M']!=1:raise AssertionError('status 5 missing')
    cases['status5_modified_M']='passed'
    factor=clone(clean);factor['ld'][:]=1e9
    cases['F03_factor_only']=reject(lambda:checker(factor),
            ('GPU outside source interval','factor','local','ld',6,[0,0]))
    factor['acc'][:]=1e9;factor['solution'][:,0,:]=1e9
    cases['F03_factor_and_solve']=reject(lambda:checker(factor),
            ('GPU outside source interval','factor','local','ld',6,[0,0]))
    factor['smooth'][0,5]=0.0
    cases['F03_force_factor_solve']=reject(lambda:checker(factor),
            ('source-order force bit difference','force','local',6,[0,5]))
    structural=clone(clean);structural['solution'][0,1,0]=-0.0
    cases['unused_RHS_negative_zero']=reject(lambda:checker(structural),'unused RHS structural +0')
    base={'ids':ids,'active':p['active'],'input':p,'manifest_sha':r.sha(reference/'manifest.json')}
    corrupt=clone(clean);corrupt['acc'][0,5]=beyond
    original=packet(base,clean,corrupt)
    files={}
    files['valid_failure.npz']=write(target/'valid_failure.npz',original)
    bad=clone(original);bad['input_passive']=bad['input_passive'].view(np.float32)
    if bad['input_passive'].tobytes()!=original['input_passive'].tobytes():
        raise AssertionError('schema same-byte control')
    files['wrong_schema_same_bytes.npz']=write(target/'wrong_schema_same_bytes.npz',bad)
    bad=clone(original);bad['input_applied'][0,5]=123.0
    bad['input_sha256']=np.bytes_(vn.bound_digest({k:bad['input_'+k] for k in vn.INPUTS},ids,base['manifest_sha']))
    files['wrong_payload_self_hash.npz']=write(target/'wrong_payload_self_hash.npz',bad)
    for name,change in (
        ('wrong_ids',lambda x:x.update(ids=np.array([5],np.int32))),
        ('wrong_active',lambda x:x['active'].__setitem__(0,0)),
        ('wrong_input_active',lambda x:x['input_active'].__setitem__(0,0)),
        ('deleted_field',lambda x:x.pop('input_qm')),
        ('deleted_row',lambda x:x.update(clean_acc=x['clean_acc'][:0])),
        ('empty_ids',lambda x:x.update(ids=np.array([],np.int32)))):
        bad=clone(original);change(bad)
        files[name+'.npz']=write(target/(name+'.npz'),bad)
    meta={'candidate_modules':modules(),'actual_gpu_source':str(source),
          'actual_gpu_sha256':r.sha(source),'budget_before_gpu_read':True,
          'force5':1.0,'solution5_certificate':bound5,'one_ULP_value':float(beyond),
          'cases':cases,'packets':files}
    (target/'create.json').write_text(json.dumps(meta,indent=2)+'\n')
    child=subprocess.run([str(r.CPU_PYTHON),str(Path(__file__)),'--mode','replay',
                          '--reference',str(reference),'--target',str(target)],
                         capture_output=True,text=True)
    (target/'replay_stdout.txt').write_text(child.stdout)
    (target/'replay_stderr.txt').write_text(child.stderr)
    if child.returncode:raise AssertionError(('new process packet replay',child.returncode,child.stderr))
    print(json.dumps({'create':meta,'new_process_replay':json.loads(child.stdout)},indent=2))


def replay(reference,target):
    expected={'wrong_schema_same_bytes':'saved input schema',
              'wrong_payload_self_hash':'saved input provenance',
              'wrong_ids':'saved input provenance',
              'wrong_active':'saved active routing',
              'wrong_input_active':'saved active routing',
              'deleted_field':'saved packet fields',
              'deleted_row':'saved output schema',
              'empty_ids':'saved identity selection'}
    result={'candidate_modules':modules(),'gpu_recomputed':False}
    good=vn.replay(reference,target/'valid_failure.npz')
    if good['clean']!='passed' or good['corrupt_actual_gpu_output']!='rejected' or \
       'GPU outside source interval' not in good['corrupt_reason']:
        raise AssertionError(('valid saved output route',good))
    result['valid_failure']=good
    for name,marker in expected.items():
        result[name]=reject(lambda n=name:vn.replay(reference,target/(n+'.npz')),marker)
    # Astra's original small packets are read-only controls.  They exercise
    # the same legacy ABI without importing any Astra implementation.
    frozen=Path('/rjs2/lhw/LIBERO-review/phase05-d09/tests/gpu_phase05_reviews/D09/frozen/probes')
    if frozen.is_dir():
        result['astra_frozen_valid']=vn.replay(reference,frozen/'valid_failure.npz')
        result['astra_frozen_schema']=reject(lambda:vn.replay(reference,frozen/'wrong_schema_same_input_sha.npz'),'saved input schema')
        result['astra_frozen_provenance']=reject(lambda:vn.replay(reference,frozen/'wrong_payload_self_hash.npz'),'saved input provenance')
    (target/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('create','replay'),required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--gpu-dir',type=Path)
    ap.add_argument('--target',type=Path,required=True)
    a=ap.parse_args()
    if a.mode=='create':create(a.reference,a.gpu_dir,a.target)
    else:replay(a.reference,a.target)
